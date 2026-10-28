"""MedIntel MCP server (stdio). Cross-call state lives in SQLite and moves between calls as explicit
handles (batch_id, case_id, proposal_id), as the stateless MCP 2026-07-28 revision expects."""

import json
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import anyio
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from medintel import __version__
from medintel.agents.rule_agent import default_rulebase_path
from medintel.config import ROLE_LABELS
from medintel.identity import Identity, UserStore
from medintel.models import CaseAnalysis
from medintel.service import (
    NOTICE,
    AuditTrail,
    BatchResult,
    BatchStatus,
    BatchSummary,
    CaseResult,
    DecisionExplanation,
    EvidenceSearch,
    JobStatus,
    KnowledgeDecision,
    KnowledgeUpdates,
    MedIntelApp,
    MedIntelError,
    PatientChanges,
    PatientRecordInput,
    ReviewResult,
    WeightChanges,
    WhatIfResult,
    patient_fields,
)

__all__ = ["MedIntelApp", "create_server", "main"]

T = TypeVar("T")

INSTRUCTIONS = f"""MedIntel analyses patient records with four agents (a trained risk model, versioned rules,
knowledge retrieval and a clinical context writer) and fuses them with Equation (4.2):
C = alpha*M + beta*R + gamma*K - lambda*D.
{NOTICE}
Flow: validate_records returns a batch_id; analyze_batch (or analyze_case for one patient) returns DecisionReports
with case_ids in triage order; explain_decision, what_if and search_evidence help a clinician understand a case.
The risk model uses a pre-test submodel when no exercise test is on record and a full submodel otherwise; inputs
that were not recorded are filled in and listed by explain_decision. Summaries may be written by Claude: every
sentence cites its sources, and sentences the sources do not support are removed before the report is saved.
Only a clinician's answer may be passed to record_review; an accepted high- or low-risk outcome becomes a proposal
that only the Knowledge Curator may approve with approve_knowledge_update. Never decide a review or an approval on
the user's behalf. Report findings, rule IDs, evidence source IDs and confidence exactly as returned, and do not
add clinical facts the results do not contain."""

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
WRITES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


def guarded(call: Callable[[], T]) -> T:
    try:
        return call()
    except MedIntelError as exc:
        raise ToolError(str(exc)) from exc


class UserTokenVerifier:
    """Bearer tokens from the MedIntel user store (medintel users add); the user's roles become the token scopes."""

    def __init__(self, users: UserStore) -> None:
        self.users = users

    async def verify_token(self, token: str) -> AccessToken | None:
        who = await anyio.to_thread.run_sync(self.users.verify, token)
        if who is None:
            return None
        return AccessToken(token=token, client_id=who.name, scopes=list(who.roles), subject=who.name)


def auth_settings(public_url: str) -> AuthSettings:
    """This server mints its own tokens, so it names itself as the issuer; tokens are sent as a bearer header."""
    base = public_url.rstrip("/")
    return AuthSettings(issuer_url=base, resource_server_url=f"{base}/mcp", validate_token_resource=False)


def create_server(app: MedIntelApp | None = None, public_url: str | None = None) -> MCPServer:
    """The MedIntel MCP server. With `public_url`, every request must carry a user's bearer token (HTTP)."""
    app = app or MedIntelApp.from_env()
    if public_url:
        server = MCPServer(
            "medintel",
            instructions=INSTRUCTIONS,
            version=__version__,
            token_verifier=UserTokenVerifier(app.users),
            auth=auth_settings(public_url),
        )
    else:
        server = MCPServer("medintel", instructions=INSTRUCTIONS, version=__version__)

    def caller() -> Identity:
        token = get_access_token()
        if token is None:
            return app.local_identity
        roles = [r for r in token.scopes if r in ROLE_LABELS]
        return Identity(name=token.subject or token.client_id, roles=roles, authenticated=True)  # type: ignore[arg-type]

    @server.tool(title="Validate patient records", annotations=WRITES)
    def validate_records(
        csv_text: str | None = None,
        sample_name: str | None = None,
        events_csv_text: str | None = None,
        source_name: str = "upload.csv",
    ) -> BatchSummary:
        """Check a CSV batch of patient records against the schema before any agent runs (FR1). Data Administrator.

        Pass csv_text, or sample_name for a bundled synthetic batch: morning_batch.csv (routine records),
        boundary_tests.csv (rule thresholds) or chest_pain_clinic.csv (records with exercise-test results).
        Cardiology columns (chest_pain_type, resting_ecg, max_heart_rate, exercise_angina, st_depression,
        st_slope, major_vessels, thal) are optional. Rejected rows come back with the reason and the test case
        they cover (TC-01 missing ID, TC-02 Fahrenheit temperature); nothing is converted silently. Returns a
        batch_id for analyze_batch.
        """
        who = caller()
        return guarded(lambda: app.validate(csv_text, sample_name, events_csv_text, source_name, who))

    @server.tool(title="Analyze a batch", annotations=WRITES)
    async def analyze_batch(batch_id: str, ctx: Context, max_workers: int = 4) -> BatchResult:
        """Analyze every valid patient in a batch in parallel, each in its own isolated context (FR2, NFR2).

        Reports progress as each case finishes and returns the triage queue: priority first (Escalate, Review,
        Priority, Assess manually, Routine), then probability of high risk. Every report is audited.
        """
        who = caller()
        cases = guarded(lambda: app.batch_cases(batch_id, who))
        limiter = anyio.CapacityLimiter(max(1, min(max_workers, 8)))
        analyses: list[CaseAnalysis] = []

        async def run(case_index: int) -> None:
            case = cases[case_index]
            analysis = await anyio.to_thread.run_sync(app.run_and_store, case, batch_id, limiter=limiter)
            analyses.append(analysis)
            await ctx.report_progress(len(analyses), len(cases), f"{case.case_id} analyzed")

        async with anyio.create_task_group() as group:
            for index in range(len(cases)):
                group.start_soon(run, index)
        return guarded(lambda: app.batch_result(batch_id, analyses))

    @server.tool(title="Start a background batch analysis", annotations=WRITES)
    def start_batch_analysis(batch_id: str) -> JobStatus:
        """Start analysing a validated batch in the background and return a job_id at once. Use it for long batches
        or when the client may disconnect; poll get_job_status for progress and the triage queue."""
        who = caller()
        return guarded(lambda: app.start_job(batch_id, who))

    @server.tool(title="Job status", annotations=READ_ONLY)
    def get_job_status(job_id: str) -> JobStatus:
        """Progress of a background batch job, and the triage queue once it has finished."""
        return guarded(lambda: app.job_status(job_id))

    @server.tool(title="Cancel a job", annotations=WRITES)
    def cancel_job(job_id: str) -> JobStatus:
        """Stop a background batch job between cases. Cases already analysed keep their reports."""
        who = caller()
        return guarded(lambda: app.cancel_job(job_id, who))

    @server.tool(title="Batch status", annotations=READ_ONLY)
    def get_batch_status(batch_id: str) -> BatchStatus:
        """How far a batch has got: valid records, analyzed cases, reviews still pending, and the triage queue."""
        return guarded(lambda: app.batch_status(batch_id))

    @server.tool(title="Analyze a case", annotations=WRITES)
    def analyze_case(
        batch_id: str | None = None,
        patient_id: str | None = None,
        record: PatientRecordInput | None = None,
    ) -> CaseResult:
        """Run the full MedIntel pipeline on one patient and return the DecisionReport (Section 4.4.2).

        Pass batch_id and patient_id from validate_records, or one record. The report carries the decision,
        confidence C, triggered rules, evidence IDs, conflicts and a recommendation, and is written to the
        append-only audit trail with status pending_review. summary_written_by names the writer (Claude or the
        template, with the reason for a fallback) and summary_cost_usd its estimated cost.
        """
        who = caller()
        return guarded(lambda: app.analyze(batch_id, patient_id, record, who))

    @server.tool(title="Explain a decision", annotations=READ_ONLY)
    def explain_decision(case_id: str) -> DecisionExplanation:
        """Show how a case's decision was reached: M, R, K, D, the weights, the threshold, rule precedence,
        the evidence used, the conflicts found, any statements removed for lacking support, which risk submodel
        scored the case with its largest contributions (exact SHAP values) and filled-in inputs, and who wrote
        the summary."""
        return guarded(lambda: app.explain(case_id))

    @server.tool(title="What if", annotations=READ_ONLY)
    def what_if(
        case_id: str,
        changes: PatientChanges | None = None,
        weights: WeightChanges | None = None,
    ) -> WhatIfResult:
        """Recompute a case with changed patient values or fusion weights and list what changes, for example
        whether a rule starts or stops firing. alpha, beta and gamma are rescaled to sum to 1. Nothing is saved
        or audited."""
        return guarded(lambda: app.what_if(case_id, changes, weights))

    @server.tool(title="Search evidence", annotations=READ_ONLY)
    def search_evidence(query: str | None = None, case_id: str | None = None, k: int = 5) -> EvidenceSearch:
        """Search the knowledge base: pass a free-text query, or a case_id to see the evidence that case matches.
        Results carry source IDs, similarity scores and the finding each passage matched. The method field says
        how the search was done (embedding similarity, or the demo knowledge base)."""
        return guarded(lambda: app.search_evidence(query, case_id, k))

    @server.tool(title="Record a clinician review", annotations=WRITES)
    def record_review(
        case_id: str,
        decision: Annotated[
            Literal["accept", "annotate", "reject"],
            Field(
                description="accept: the clinician agrees with the report (a note is optional and does not change "
                "this). annotate: the clinician adds a note without accepting or rejecting. reject: the clinician "
                "disagrees with the report."
            ),
        ],
        notes: Annotated[
            str | None,
            Field(description="The clinician's own words. Required for annotate and reject, optional for accept."),
        ] = None,
        reviewer: Annotated[
            str | None,
            Field(description="The clinician's name as they gave it (ignored when signed in: the identity is used)."),
        ] = None,
    ) -> ReviewResult:
        """Record a clinician's decision on a report. Use exactly the decision the clinician stated: "accept with a
        note" is accept, not annotate. The review is appended to the audit trail (entries cannot be changed later);
        an accepted high- or low-risk outcome is proposed to the Knowledge Curator, and nothing is re-indexed until
        approved."""
        who = caller()
        return guarded(lambda: app.review(case_id, decision, notes, reviewer, who))

    @server.tool(title="List knowledge updates", annotations=READ_ONLY)
    def list_knowledge_updates(
        status: Literal["pending", "approved", "declined", "all"] = "pending",
    ) -> KnowledgeUpdates:
        """Knowledge update proposals, the current index version and its history of approved changes."""
        return guarded(lambda: app.knowledge_updates(status))

    @server.tool(title="Approve or decline a knowledge update", annotations=WRITES)
    def approve_knowledge_update(
        proposal_id: str,
        approve: Annotated[bool, Field(description="true to approve (new index version), false to decline.")],
        notes: Annotated[str | None, Field(description="The curator's own words, optional.")] = None,
        curator: Annotated[
            str | None,
            Field(description="The curator's name as they gave it (ignored when signed in: the identity is used)."),
        ] = None,
    ) -> KnowledgeDecision:
        """Knowledge Curator only: approve (the index moves to a new version) or decline (unchanged) a validated
        outcome. Use only the curator's own answer."""
        who = caller()
        return guarded(lambda: app.decide(proposal_id, approve, notes, curator, who))

    @server.tool(title="Read the audit trail", annotations=READ_ONLY)
    def get_audit_trail(case_id: str | None = None, limit: int = 20) -> AuditTrail:
        """Return audit records, newest first: case ID, versions, retrieval IDs, status, timestamp, event and
        actor, so any report can be reproduced later (NFR6)."""
        return guarded(lambda: app.audit_trail(case_id, limit))

    @server.resource(
        "medintel://rulebase",
        name="rulebase",
        mime_type="application/json",
        description="The versioned rulebase the Rule Validation Agent applies.",
    )
    def rulebase() -> str:
        return default_rulebase_path().read_text(encoding="utf-8")

    @server.resource(
        "medintel://schema/decision-report",
        name="decision-report-schema",
        mime_type="application/json",
        description="JSON Schema of the DecisionReport (Section 4.4.2).",
    )
    def decision_report_schema() -> str:
        return (Path(str(files("medintel") / "resources")) / "decision_report.schema.json").read_text(encoding="utf-8")

    @server.resource(
        "medintel://cases/{case_id}/report",
        name="case-report",
        mime_type="application/json",
        description="The latest DecisionReport for a case, including its review status.",
    )
    def case_report(case_id: str) -> str:
        analysis = app.store.get_analysis(case_id)
        if analysis is None:
            raise ResourceError(f"Unknown case_id {case_id}")
        return json.dumps(analysis.report.model_dump(), indent=2)

    @server.resource(
        "medintel://model-card",
        name="model-card",
        mime_type="text/markdown",
        description="Model card of the trained risk model: data, submodels, results, limitations.",
    )
    def model_card() -> str:
        return app.model_card()

    @server.resource(
        "medintel://knowledge-base",
        name="knowledge-base",
        mime_type="application/json",
        description="The knowledge base passages behind evidence IDs, with their sources and review status.",
    )
    def knowledge_base() -> str:
        return json.dumps(app.knowledge_listing(), indent=2)

    @server.resource(
        "medintel://schema/patient-record",
        name="patient-record-schema",
        mime_type="application/json",
        description="Fields of a patient record: type, unit, allowed range or values, and whether required.",
    )
    def patient_record_schema() -> str:
        return json.dumps(patient_fields(), indent=2)

    @server.prompt(title="Triage a batch")
    def triage_batch(batch_id: str) -> str:
        return (
            f"Use the MedIntel tools to triage batch {batch_id}.\n"
            f'1. Call analyze_batch with batch_id "{batch_id}".\n'
            "2. Summarize the queue in priority order: Escalate (same-day review), Review (uncertain), Priority, "
            "Assess manually, Routine. For each case give the case ID, decision, confidence C against the "
            "threshold, triggered rule IDs and the recommendation, exactly as returned.\n"
            "3. Do not add clinical facts the results do not contain. End by noting that a clinician makes "
            "every final decision."
        )

    @server.prompt(title="Review a case with a clinician")
    def review_case(case_id: str) -> str:
        return (
            f"Walk a clinician through case {case_id}.\n"
            f'1. Call explain_decision with case_id "{case_id}".\n'
            "2. Present the decision, confidence C against the threshold, what each agent contributed, the "
            "conflicts and any statements removed for lacking support. Offer what_if if they want to test a "
            "value.\n"
            "3. Ask the clinician whether to accept, annotate or reject the report, and for their notes and name.\n"
            "4. Only after they answer, call record_review with exactly their decision, notes and name."
        )

    @server.prompt(title="Curator review of knowledge updates")
    def curator_review() -> str:
        return (
            "Help the Knowledge Curator review pending knowledge updates.\n"
            '1. Call list_knowledge_updates with status "pending".\n'
            "2. For each proposal show the case, the confirmed decision and the clinician's notes; open the case "
            "with explain_decision if the curator asks.\n"
            "3. Ask the curator to approve or decline each one, and for their name.\n"
            "4. Only after they answer, call approve_knowledge_update with exactly their choice."
        )

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
