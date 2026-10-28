"""MedIntel application service: everything the MCP tools do, independent of the transport."""

from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from medintel.agents.claude_writer import create_writer
from medintel.agents.context_agent import DraftWriter, TemplateWriter, shown_value
from medintel.agents.embedding_store import EmbeddingKnowledgeStore, KnowledgeBase, SentenceTransformerEncoder
from medintel.agents.retrieval_agent import DemoKnowledgeStore, KnowledgeStore
from medintel.audit import AuditLogger, CaseStore
from medintel.config import (
    ROLE_LABELS,
    Role,
    Settings,
    load_env_file,
    load_settings,
    medintel_home,
    project_env_file,
)
from medintel.fusion import FusionSettings, decision_steps
from medintel.identity import Identity, UserStore
from medintel.ingestion import DataIngestionService, IngestionResult, SchemaValidator, as_text, parse_events
from medintel.jobs import JobInfo, JobRunner
from medintel.knowledge import GovernedKnowledgeStore, KnowledgeRegistry, KnowledgeVersion, Proposal, version_label
from medintel.models import (
    AuditRecord,
    CaseAnalysis,
    DecisionReport,
    EvidenceItem,
    LLMUsage,
    MLResult,
    PatientCase,
    Priority,
    RejectedRecord,
    Versions,
)
from medintel.numeric import fixed2
from medintel.orchestrator import create_orchestrator
from medintel.schema import PATIENT_FIELDS, STRUCTURED_KEYS, Value

NOTICE = (
    "Research prototype on synthetic or de-identified data. Not a medical device; a clinician makes the final decision."
)
SAMPLES = Path(str(files("medintel") / "samples"))
SAMPLE_BATCHES = {"morning_batch.csv": "morning_events.csv", "boundary_tests.csv": None, "chest_pain_clinic.csv": None}
DECISION_TITLE = {"high_risk": "High risk", "low_risk": "Low risk", "uncertain": "Uncertain", "withheld": "Withheld"}
REVIEW_STATUS = {"accept": "accepted", "annotate": "annotated", "reject": "rejected"}


class MedIntelError(Exception):
    """A request the caller can correct; the MCP layer reports it as a tool error, the REST API as 400."""


class PermissionDenied(MedIntelError):
    """The caller's identity lacks the role an action needs (REST: 403)."""


class NotFound(MedIntelError):
    """An ID that does not exist (REST: 404)."""


class PatientRecordInput(BaseModel):
    patient_id: str
    age: int = Field(description="Years.")
    sex: Literal["F", "M"]
    systolic_bp: float = Field(description="mmHg.")
    diastolic_bp: float = Field(description="mmHg.")
    heart_rate: float = Field(description="Beats per minute.")
    temperature_c: float = Field(description="Degrees Celsius (Fahrenheit values are rejected, not converted).")
    total_cholesterol: float = Field(description="mg/dL.")
    fasting_glucose: float = Field(description="mg/dL.")
    creatinine: float | None = Field(default=None, description="mg/dL; leave empty when not measured.")
    bmi: float = Field(description="kg/m².")
    smoker: bool
    chest_pain_type: Literal["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"] | None = None
    resting_ecg: Literal["normal", "st_t_abnormality", "lv_hypertrophy"] | None = None
    max_heart_rate: float | None = Field(default=None, description="Peak heart rate on an exercise test, bpm.")
    exercise_angina: bool | None = Field(default=None, description="Angina brought on by the exercise test.")
    st_depression: float | None = Field(default=None, description="ST depression on the exercise test, mm.")
    st_slope: Literal["upsloping", "flat", "downsloping"] | None = None
    major_vessels: int | None = Field(default=None, ge=0, le=3, description="Major vessels seen on fluoroscopy.")
    thal: Literal["normal", "fixed_defect", "reversible_defect"] | None = Field(
        default=None, description="Thallium stress test result."
    )
    notes: str = ""


class PatientChanges(BaseModel):
    age: int | None = None
    sex: Literal["F", "M"] | None = None
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    heart_rate: float | None = None
    temperature_c: float | None = None
    total_cholesterol: float | None = None
    fasting_glucose: float | None = None
    creatinine: float | None = None
    creatinine_missing: bool = Field(default=False, description="Treat creatinine as not measured.")
    bmi: float | None = None
    smoker: bool | None = None
    chest_pain_type: Literal["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"] | None = None
    resting_ecg: Literal["normal", "st_t_abnormality", "lv_hypertrophy"] | None = None
    max_heart_rate: float | None = Field(default=None, description="Peak heart rate on an exercise test, bpm.")
    exercise_angina: bool | None = Field(default=None, description="Angina brought on by the exercise test.")
    st_depression: float | None = Field(default=None, description="ST depression on the exercise test, mm.")
    st_slope: Literal["upsloping", "flat", "downsloping"] | None = None
    major_vessels: int | None = Field(default=None, ge=0, le=3, description="Major vessels seen on fluoroscopy.")
    thal: Literal["normal", "fixed_defect", "reversible_defect"] | None = Field(
        default=None, description="Thallium stress test result."
    )


class WeightChanges(BaseModel):
    alpha: float | None = Field(default=None, ge=0, le=1, description="Weight of ML support M.")
    beta: float | None = Field(default=None, ge=0, le=1, description="Weight of rule consistency R.")
    gamma: float | None = Field(default=None, ge=0, le=1, description="Weight of retrieval support K.")
    lam: float | None = Field(default=None, ge=0, le=1, description="Penalty on disagreement D.")
    threshold: float | None = Field(default=None, ge=0, le=1, description="Decision threshold for C.")


class BatchSummary(BaseModel):
    batch_id: str
    source_name: str
    records_read: int
    valid: int
    valid_patient_ids: list[str]
    missing_columns: list[str]
    rejected: list[RejectedRecord]


class CaseResult(BaseModel):
    report: DecisionReport
    priority: Priority
    summary_written_by: str
    summary_cost_usd: float | None = Field(default=None, description="Estimated Claude cost for this summary.")
    record: dict[str, Value] = Field(
        default_factory=dict, description="The structured values the case was analysed with."
    )
    clinical_notes: str = ""
    notice: str = NOTICE


def case_result(analysis: CaseAnalysis) -> CaseResult:
    usage = analysis.context.usage
    return CaseResult(
        report=analysis.report,
        priority=analysis.priority,
        summary_written_by=analysis.context.generated_by,
        summary_cost_usd=usage.cost_usd if usage else None,
        record=analysis.case.x_s,
        clinical_notes=analysis.case.x_t,
    )


class QueueItem(BaseModel):
    case_id: str
    patient_id: str
    priority: Priority
    decision: str
    confidence: float
    threshold: float
    probability_high_risk: float
    triggered_rules: list[str]
    evidence_count: int
    conflict_count: int
    status: str
    recommendation: str


class BatchResult(BaseModel):
    batch_id: str
    analyzed: int
    rejected_at_validation: int
    counts: dict[str, int] = Field(description="Cases per priority: escalate, review, priority, manual, routine.")
    queue: list[QueueItem] = Field(description="Triage order: priority first, then probability of high risk.")
    summary_cost_usd: float | None = Field(default=None, description="Estimated Claude cost for the summaries.")
    notice: str = NOTICE


class BatchListing(BaseModel):
    batch_id: str
    source_name: str
    created_at: str
    valid: int
    rejected: int
    analyzed: int


class BatchStatus(BaseModel):
    batch_id: str
    source_name: str
    valid: int
    analyzed: int
    not_analyzed: list[str]
    pending_review: int
    reviewed: int
    queue: list[QueueItem]


class EvidenceRef(BaseModel):
    source_id: str
    title: str
    score: float
    direction: str
    matched_on: str


class ModelInput(BaseModel):
    input: str
    value: str
    contribution: float = Field(description="Exact SHAP value on the calibrated log-odds scale.")


class ModelExplanation(BaseModel):
    version: str
    submodel: str
    probability_high_risk: float
    inputs_used: int
    inputs_filled_in: list[str]
    largest_contributions: list[ModelInput]
    note: str


def explain_model(ml: MLResult) -> ModelExplanation:
    submodel = {
        "pre_test": "pre-test model (no exercise test on record)",
        "full": "full model (exercise test on record)",
    }.get(ml.variant, "prototype demo weights (not trained)")
    note = (
        "Each contribution is the input's exact SHAP value; inputs that were not recorded are filled in with "
        "training averages and contribute 0. See the medintel://model-card resource."
        if ml.variant
        else "Hand-set weights from the Stage I concept prototype; run `uv run medintel train` for a trained model."
    )
    return ModelExplanation(
        version=ml.model_version,
        submodel=submodel,
        probability_high_risk=ml.probabilities["high_risk"],
        inputs_used=sum(not c.imputed for c in ml.contributions),
        inputs_filled_in=[c.label for c in ml.contributions if c.imputed],
        largest_contributions=[
            ModelInput(input=c.label, value=shown_value(c), contribution=c.contribution)
            for c in ml.importance
            if c.contribution and not c.imputed
        ][:5],
        note=note,
    )


class DecisionExplanation(BaseModel):
    case_id: str
    decision: str
    equation: str = "C = alpha*M + beta*R + gamma*K - lambda*D, clipped at 0"
    weights: dict[str, float]
    component_scores: dict[str, float]
    terms: dict[str, float]
    raw_confidence: float
    confidence: float
    threshold: float
    rule_override: str | None
    below_threshold: bool
    steps: list[str]
    rules_fired: list[str]
    evidence: list[EvidenceRef]
    conflicts: list[str]
    removed_statements: list[str]
    model: ModelExplanation
    summary_written_by: str
    summary_usage: LLMUsage | None
    timings: dict[str, float] = Field(default_factory=dict, description="Seconds per pipeline stage.")
    versions: Versions


class AuditTrail(BaseModel):
    case_id: str | None
    records: list[AuditRecord]


class Snapshot(BaseModel):
    decision: str
    confidence: float
    priority: str
    triggered_rules: list[str]
    evidence_refs: list[str]
    conflicts: list[str]
    recommendation: str


class WhatIfResult(BaseModel):
    case_id: str
    changes_applied: dict[str, Value]
    weights_used: dict[str, float]
    threshold: float
    before: Snapshot
    after: Snapshot
    differences: list[str]
    saved: bool = False
    note: str = "Exploration only: nothing is saved to the case or written to the audit trail."


class EvidenceHit(BaseModel):
    source_id: str
    title: str
    topic: str
    direction: str
    score: float
    passage: str
    matched_on: str


DEMO_KB_NOTE = "Illustrative guideline summaries for the prototype, not clinical guidance."
STARTER_KB_NOTE = (
    "Short summaries of public CDC, NIH and MedlinePlus pages, each with its source; drafts pending review by the "
    "team and their guide. Not clinical guidance."
)


class EvidenceSearch(BaseModel):
    query: str | None
    case_id: str | None
    method: str
    index_version: str
    results: list[EvidenceHit]
    note: str = DEMO_KB_NOTE


class ReviewResult(BaseModel):
    case_id: str
    status: str
    reviewer: str
    notes: str
    audit_record: AuditRecord
    proposal: Proposal | None
    message: str


class KnowledgeUpdates(BaseModel):
    index_version: str
    proposals: list[Proposal]
    history: list[KnowledgeVersion]
    validated_cases: list[str]


class JobStatus(BaseModel):
    job: JobInfo
    progress: float = Field(description="Share of cases finished, 0 to 1.")
    result: BatchResult | None = Field(default=None, description="The triage queue once the job has finished.")
    note: str = "Poll get_job_status until the job has finished; cancel_job stops it between cases."


class KnowledgeDecision(BaseModel):
    proposal: Proposal
    index_version_before: str
    index_version_after: str
    message: str


def queue_item(a: CaseAnalysis) -> QueueItem:
    return QueueItem(
        case_id=a.case.case_id,
        patient_id=a.case.patient_id,
        priority=a.priority,
        decision=a.fusion.decision,
        confidence=a.fusion.confidence,
        threshold=a.fusion.threshold,
        probability_high_risk=a.ml.probabilities["high_risk"],
        triggered_rules=a.rules.triggered,
        evidence_count=len(a.evidence),
        conflict_count=len(a.context.conflicts),
        status=a.report.status,
        recommendation=a.recommendation,
    )


def summary_cost(analyses: list[CaseAnalysis]) -> float | None:
    costs = [a.context.usage.cost_usd for a in analyses if a.context.usage]
    return round(sum(costs), 4) if costs else None


def triage_order(analyses: list[CaseAnalysis]) -> list[CaseAnalysis]:
    return sorted(analyses, key=lambda a: (a.priority.level, -a.ml.probabilities["high_risk"]))


def snapshot(a: CaseAnalysis) -> Snapshot:
    return Snapshot(
        decision=a.fusion.decision,
        confidence=a.fusion.confidence,
        priority=a.priority.label,
        triggered_rules=a.rules.triggered,
        evidence_refs=[e.source_id for e in a.evidence],
        conflicts=a.context.conflicts,
        recommendation=a.recommendation,
    )


def describe_changes(before: CaseAnalysis, after: CaseAnalysis) -> list[str]:
    out: list[str] = []
    messages = {c.rule.rule_id: c.rule.message for c in after.rules.checks}
    out += [f"Rule {r} fired: {messages[r]}" for r in after.rules.triggered if r not in before.rules.triggered]
    out += [f"Rule {r} no longer fires." for r in before.rules.triggered if r not in after.rules.triggered]
    before_ids = [e.source_id for e in before.evidence]
    after_ids = [e.source_id for e in after.evidence]
    if before_ids != after_ids:
        out.append(
            f"Evidence retrieved: {', '.join(after_ids)}."
            if after_ids
            else "No sufficiently relevant evidence: retrieval support K drops to 0."
        )
    n = len(after.context.conflicts)
    if n != len(before.context.conflicts):
        out.append(
            f"{n} conflict{'s' if n != 1 else ''} reported; D is now {fixed2(after.fusion.component_scores['D'])}."
        )
    b, a = before.fusion, after.fusion
    if b.decision != a.decision:
        out.append(f"Decision changed: {DECISION_TITLE[b.decision]} to {DECISION_TITLE[a.decision]}.")
    elif b.below_threshold != a.below_threshold:
        out.append(f"C crossed the threshold ({fixed2(a.confidence)} against {fixed2(a.threshold)}).")
    if fixed2(b.confidence) != fixed2(a.confidence):
        out.append(f"Confidence C moved from {fixed2(b.confidence)} to {fixed2(a.confidence)}.")
    if before.priority.key != after.priority.key:
        out.append(f"Priority changed: {before.priority.label} to {after.priority.label}.")
    return out or ["No change in rules, evidence, conflicts, decision or confidence."]


def merged_fusion(base: FusionSettings, changes: WeightChanges | None) -> FusionSettings:
    data = base.model_dump()
    if changes:
        data.update(changes.model_dump(exclude_none=True))
    total = data["alpha"] + data["beta"] + data["gamma"]
    if total <= 0:
        raise MedIntelError("alpha, beta and gamma cannot all be 0.")
    data.update(alpha=data["alpha"] / total, beta=data["beta"] / total, gamma=data["gamma"] / total)
    return FusionSettings(**data)


def patient_fields() -> list[dict[str, object]]:
    return [
        {
            "key": f.key,
            "label": f.label,
            "type": f.kind,
            "required": f.required,
            **({"unit": f.unit} if f.unit else {}),
            **({"min": f.min, "max": f.max} if f.min is not None else {}),
            **({"values": list(f.values)} if f.values else {}),
        }
        for f in PATIENT_FIELDS
    ]


def build_knowledge_store(settings: Settings) -> tuple[KnowledgeStore, str, str]:
    """The base store for the configured backend, a description of its content, and a fingerprint of it."""
    r = settings.retrieval
    if r.backend == "embedding":
        kb = KnowledgeBase.load(r.knowledge_base or None)
        store = EmbeddingKnowledgeStore(kb, SentenceTransformerEncoder(r.embedding_model), r.index)
        return store, f"{len(kb.passages)} passages from {kb.label}", kb.fingerprint
    return DemoKnowledgeStore(), "10 sample guideline summaries", "demo-kb"


class MedIntelApp:
    def __init__(self, home: Path, settings: Settings | None = None, writer: DraftWriter | None = None) -> None:
        self.settings = settings or Settings()
        db_path = home / "medintel.db"
        self.local_identity = Identity(name="local user", roles=list(self.settings.access.roles))
        self.users = UserStore(db_path)
        self.audit = AuditLogger(db_path)
        self.store = CaseStore(db_path)
        base, description, fingerprint = build_knowledge_store(self.settings)
        self.registry = KnowledgeRegistry(db_path, base_description=f"Initial index: {description}")
        self.registry.ensure_base(fingerprint, description)
        self.knowledge = GovernedKnowledgeStore(base, self.registry)
        self.writer = writer or create_writer(self.settings.context)
        self.orchestrator = create_orchestrator(self.settings, self.writer, self.audit, self.knowledge)
        self.ingestion = DataIngestionService()
        self.jobs = JobRunner(db_path, self.run_and_store)

    @classmethod
    def from_env(cls) -> "MedIntelApp":
        load_env_file(project_env_file())
        return cls(medintel_home(), load_settings())

    def model_card(self) -> str:
        path = Path(self.settings.ml.model).parent / "MODEL_CARD.md" if self.settings.ml.model else None
        if path and path.is_file():
            return path.read_text(encoding="utf-8")
        return (
            "No trained model is configured, so the risk model runs the Stage I prototype's demo weights. "
            "Train one with: uv run medintel fetch-data && uv run medintel train"
        )

    def knowledge_listing(self) -> dict[str, object]:
        base = self.knowledge.base
        kb = getattr(base, "kb", None)
        if kb is None:
            return {
                "name": "demo knowledge base",
                "index_version": self.knowledge.version,
                "note": DEMO_KB_NOTE,
                "passages": [
                    {"source_id": p.source_id, "title": p.title, "topic": p.topic, "passage": p.passage}
                    for p in getattr(base, "passages", ())
                ],
            }
        self.knowledge.sync()
        return {
            "name": kb.name,
            "version": kb.version,
            "fingerprint": kb.fingerprint,
            "index_version": self.knowledge.version,
            "review": kb.review,
            "passages": [p.model_dump(exclude_none=True) for p in base.passages],
        }

    def require(self, roles: tuple[Role, ...], action: str, who: Identity | None = None) -> Identity:
        """The caller (the local identity when none is given) if it holds one of `roles`."""
        who = who or self.local_identity
        if not who.has_any(roles):
            names = " or ".join(ROLE_LABELS[r] for r in roles)
            if who.authenticated:
                raise PermissionDenied(f"{action} needs the {names} role, which {who.name} does not have.")
            raise PermissionDenied(
                f"{action} needs the {names} role, which this MedIntel server is not granted (MEDINTEL_ROLES)."
            )
        return who

    def _analysis(self, case_id: str) -> CaseAnalysis:
        analysis = self.store.get_analysis(case_id)
        if analysis is None:
            raise NotFound(f"Unknown case_id {case_id}. Run analyze_case or analyze_batch first.")
        return analysis

    def validate(
        self,
        csv_text: str | None,
        sample_name: str | None,
        events_csv_text: str | None,
        source_name: str,
        who: Identity | None = None,
    ) -> BatchSummary:
        self.require(("data_admin",), "Uploading records", who)
        if (csv_text is None) == (sample_name is None):
            raise MedIntelError("Pass exactly one of csv_text or sample_name.")
        if sample_name is not None:
            if sample_name not in SAMPLE_BATCHES:
                raise MedIntelError(f"Unknown sample. Available: {', '.join(SAMPLE_BATCHES)}.")
            csv_text = (SAMPLES / sample_name).read_text(encoding="utf-8")
            events_file = SAMPLE_BATCHES[sample_name]
            if events_file and events_csv_text is None:
                events_csv_text = (SAMPLES / events_file).read_text(encoding="utf-8")
            source_name = sample_name
        events = parse_events(events_csv_text) if events_csv_text else None
        batch = self.ingestion.load_text(csv_text or "", source_name, events)
        return BatchSummary(
            batch_id=self.store.save_batch(batch),
            source_name=batch.source_name,
            records_read=batch.records_read,
            valid=len(batch.cases),
            valid_patient_ids=[c.patient_id for c in batch.cases],
            missing_columns=batch.missing_columns,
            rejected=batch.rejected,
        )

    def _batch(self, batch_id: str) -> IngestionResult:
        batch = self.store.get_batch(batch_id)
        if batch is None:
            raise NotFound(f"Unknown batch_id {batch_id}. Call validate_records first.")
        return batch

    def run_and_store(self, case: PatientCase, batch_id: str | None = None) -> CaseAnalysis:
        analysis = self.orchestrator.run_case(case)
        self.store.save_analysis(analysis, batch_id)
        return analysis

    def analyze(
        self,
        batch_id: str | None,
        patient_id: str | None,
        record: PatientRecordInput | None,
        who: Identity | None = None,
    ) -> CaseResult:
        self.require(("clinician", "data_admin"), "Analyzing a case", who)
        if record is not None:
            if batch_id or patient_id:
                raise MedIntelError("Pass either record, or batch_id with patient_id, not both.")
            batch = self.ingestion.load_record(record.model_dump(), source_name="direct-input")
            if batch.rejected:
                problems = "; ".join(f"{p.field}: {p.message}" for p in batch.rejected[0].problems)
                raise MedIntelError(f"Record failed validation: {problems}")
            analysis = self.run_and_store(batch.cases[0])
        else:
            if not (batch_id and patient_id):
                raise MedIntelError("Pass batch_id and patient_id from validate_records, or a single record.")
            case = next((c for c in self._batch(batch_id).cases if c.patient_id == patient_id), None)
            if case is None:
                raise NotFound(f"{patient_id} is not a valid record in {batch_id}.")
            analysis = self.run_and_store(case, batch_id)
        return case_result(analysis)

    def batch_cases(self, batch_id: str, who: Identity | None = None) -> list[PatientCase]:
        self.require(("clinician", "data_admin"), "Analyzing a batch", who)
        cases = self._batch(batch_id).cases
        if not cases:
            raise MedIntelError(f"{batch_id} has no valid records to analyze.")
        return cases

    def start_job(self, batch_id: str, who: Identity | None = None) -> JobStatus:
        caller = self.require(("clinician", "data_admin"), "Analyzing a batch", who)
        job = self.jobs.start(batch_id, self.batch_cases(batch_id, caller), caller.name)
        return self.job_status(job.job_id)

    def job_status(self, job_id: str) -> JobStatus:
        job = self.jobs.get(job_id)
        if job is None:
            raise NotFound(f"Unknown job_id {job_id}. Start one with start_batch_analysis.")
        finished = job.done + job.failed
        result = None
        if job.finished and job.done:
            result = self.batch_result(job.batch_id, self.store.batch_analyses(job.batch_id))
        return JobStatus(job=job, progress=round(finished / job.total, 3) if job.total else 1.0, result=result)

    def cancel_job(self, job_id: str, who: Identity | None = None) -> JobStatus:
        self.require(("clinician", "data_admin"), "Cancelling a batch job", who)
        if self.jobs.cancel(job_id) is None:
            raise NotFound(f"Unknown job_id {job_id}.")
        return self.job_status(job_id)

    def batch_result(self, batch_id: str, analyses: list[CaseAnalysis]) -> BatchResult:
        ordered = triage_order(analyses)
        counts: dict[str, int] = {}
        for a in ordered:
            counts[a.priority.key] = counts.get(a.priority.key, 0) + 1
        return BatchResult(
            batch_id=batch_id,
            analyzed=len(ordered),
            rejected_at_validation=len(self._batch(batch_id).rejected),
            counts=counts,
            queue=[queue_item(a) for a in ordered],
            summary_cost_usd=summary_cost(ordered),
        )

    def recent_batches(self, limit: int = 20) -> list[BatchListing]:
        return [
            BatchListing(
                batch_id=batch_id,
                source_name=batch.source_name,
                created_at=created_at,
                valid=len(batch.cases),
                rejected=len(batch.rejected),
                analyzed=self.store.analyzed_count(batch_id),
            )
            for batch_id, created_at, batch in self.store.list_batches(max(1, min(limit, 100)))
        ]

    def case(self, case_id: str) -> CaseResult:
        return case_result(self._analysis(case_id))

    def batch_status(self, batch_id: str) -> BatchStatus:
        batch = self._batch(batch_id)
        analyses = triage_order(self.store.batch_analyses(batch_id))
        done = {a.case.patient_id for a in analyses}
        reviewed = sum(1 for a in analyses if a.report.status in ("accepted", "annotated", "rejected"))
        return BatchStatus(
            batch_id=batch_id,
            source_name=batch.source_name,
            valid=len(batch.cases),
            analyzed=len(analyses),
            not_analyzed=[c.patient_id for c in batch.cases if c.patient_id not in done],
            pending_review=sum(1 for a in analyses if a.report.status == "pending_review"),
            reviewed=reviewed,
            queue=[queue_item(a) for a in analyses],
        )

    def explain(self, case_id: str) -> DecisionExplanation:
        analysis = self._analysis(case_id)
        f = analysis.fusion
        return DecisionExplanation(
            case_id=case_id,
            decision=f.decision,
            weights=f.weights,
            component_scores=f.component_scores,
            terms=f.terms,
            raw_confidence=f.raw,
            confidence=f.confidence,
            threshold=f.threshold,
            rule_override=f.override,
            below_threshold=f.below_threshold,
            steps=decision_steps(f, analysis.ml, analysis.rules, analysis.context),
            rules_fired=analysis.rules.triggered,
            evidence=[
                EvidenceRef(**e.model_dump(include={"source_id", "title", "score", "direction", "matched_on"}))
                for e in analysis.evidence
            ],
            conflicts=analysis.context.conflicts,
            removed_statements=analysis.context.unsupported_statements,
            model=explain_model(analysis.ml),
            summary_written_by=analysis.context.generated_by,
            summary_usage=analysis.context.usage,
            timings=analysis.timings,
            versions=analysis.report.versions,
        )

    def audit_trail(self, case_id: str | None, limit: int) -> AuditTrail:
        return AuditTrail(case_id=case_id, records=self.audit.records(case_id, max(1, min(limit, 200))))

    def what_if(self, case_id: str, changes: PatientChanges | None, weights: WeightChanges | None) -> WhatIfResult:
        base = self._analysis(case_id)
        values: dict[str, Value] = dict(base.case.x_s)
        applied: dict[str, Value] = {}
        if changes:
            for key, value in changes.model_dump(exclude_none=True, exclude={"creatinine_missing"}).items():
                values[key] = applied[key] = value
            if changes.creatinine_missing:
                values["creatinine"] = applied["creatinine"] = None
        raw = {key: as_text(value) for key, value in {"patient_id": base.case.patient_id, **values}.items()}
        normalized, problems = SchemaValidator().check(raw, set())
        if problems:
            details = "; ".join(f"{p.field}: {p.message}" for p in problems)
            raise MedIntelError(f"Changed values failed validation: {details}")
        case = base.case.model_copy(update={"x_s": {key: normalized[key] for key in STRUCTURED_KEYS}})
        fusion = merged_fusion(self.settings.fusion, weights)
        explorer = create_orchestrator(
            self.settings.model_copy(update={"fusion": fusion}), TemplateWriter(), None, self.knowledge
        )
        after = explorer.run_case(case)
        return WhatIfResult(
            case_id=case_id,
            changes_applied=applied,
            weights_used={"alpha": fusion.alpha, "beta": fusion.beta, "gamma": fusion.gamma, "lam": fusion.lam},
            threshold=fusion.threshold,
            before=snapshot(base),
            after=snapshot(after),
            differences=describe_changes(base, after),
        )

    def search_evidence(self, query: str | None, case_id: str | None, k: int) -> EvidenceSearch:
        if (query is None) == (case_id is None):
            raise MedIntelError("Pass exactly one of query or case_id.")
        k = max(1, min(k, 10))
        items: list[EvidenceItem]
        embedding = self.settings.retrieval.backend == "embedding"
        if case_id is not None:
            items = self.orchestrator.retriever.retrieve(self._analysis(case_id).case, k)
            method = self.knowledge.method
        else:
            items = self.knowledge.search_text(query or "", k)
            method = self.knowledge.method if embedding else "keyword overlap (demo knowledge base)"
        return EvidenceSearch(
            query=query,
            case_id=case_id,
            method=method,
            index_version=self.knowledge.version,
            results=[EvidenceHit(**e.model_dump()) for e in items],
            note=STARTER_KB_NOTE if embedding else DEMO_KB_NOTE,
        )

    def review(
        self,
        case_id: str,
        decision: Literal["accept", "annotate", "reject"],
        notes: str | None,
        reviewer: str | None,
        who: Identity | None = None,
    ) -> ReviewResult:
        caller = self.require(("clinician",), "Recording a review", who)
        analysis = self._analysis(case_id)
        if analysis.report.status == "failed":
            raise MedIntelError("This case failed analysis; run analyze_case again before reviewing it.")
        text = (notes or "").strip()
        if decision in ("annotate", "reject") and not text:
            raise MedIntelError(f"Add notes to {decision} a case.")
        status = REVIEW_STATUS[decision]
        actor = caller.name if caller.authenticated else (reviewer or "").strip() or ROLE_LABELS["clinician"]
        report = analysis.report.model_copy(update={"status": status, "notes": text or analysis.report.notes})
        updated = analysis.model_copy(update={"report": report})
        self.store.update_report(updated)
        record = self.audit.log(report, event=f"review_{status}", actor=actor)
        proposal = self.registry.propose(updated, actor, text) if status == "accepted" else None
        if proposal:
            message = (
                f"Review recorded. The outcome went to the Knowledge Curator as {proposal.proposal_id}; "
                "nothing is re-indexed until it is approved."
            )
        elif status == "accepted":
            message = "Review recorded. Uncertain and withheld decisions are not proposed as knowledge updates."
        else:
            message = "Review recorded in the audit trail."
        return ReviewResult(
            case_id=case_id,
            status=status,
            reviewer=actor,
            notes=report.notes,
            audit_record=record,
            proposal=proposal,
            message=message,
        )

    def knowledge_updates(self, status: Literal["pending", "approved", "declined", "all"]) -> KnowledgeUpdates:
        return KnowledgeUpdates(
            index_version=self.knowledge.version,
            proposals=self.registry.proposals(None if status == "all" else status),
            history=self.registry.history(),
            validated_cases=self.registry.validated_case_ids(),
        )

    def decide(
        self, proposal_id: str, approve: bool, notes: str | None, curator: str | None, who: Identity | None = None
    ) -> KnowledgeDecision:
        caller = self.require(("curator",), "Approving knowledge updates", who)
        actor = caller.name if caller.authenticated else (curator or "").strip() or ROLE_LABELS["curator"]
        try:
            proposal, before, after = self.registry.decide(proposal_id, approve, actor, (notes or "").strip())
        except KeyError as exc:
            raise NotFound(f"Unknown proposal {proposal_id}. Call list_knowledge_updates to see pending ones.") from exc
        except ValueError as exc:
            raise MedIntelError(str(exc)) from exc
        message = (
            f"Approved. The knowledge index moved from {version_label(before)} to {version_label(after)}."
            if approve
            else "Declined. The knowledge index is unchanged."
        )
        return KnowledgeDecision(
            proposal=proposal,
            index_version_before=version_label(before),
            index_version_after=version_label(after),
            message=message,
        )
