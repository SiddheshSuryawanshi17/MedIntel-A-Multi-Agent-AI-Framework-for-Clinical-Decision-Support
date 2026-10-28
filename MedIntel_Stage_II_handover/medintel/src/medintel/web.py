"""One ASGI app for Phase 3: MCP over Streamable HTTP at /mcp, the REST API at /v1 (OpenAPI docs at /docs), the
clinician dashboard at /dashboard and a health check at /healthz. All of them run on one MedIntelApp, so the front
doors share one core, one database and one audit trail.

With authentication on (the default), every /mcp and /v1 request carries a user's bearer token from
`medintel users add`; the user's roles decide what they may do, and the audit trail records their name."""

import os
import re
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from medintel import __version__
from medintel.config import ROLE_LABELS, Role
from medintel.identity import Identity
from medintel.mcp_server import create_server
from medintel.service import (
    NOTICE,
    AuditTrail,
    BatchListing,
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
    NotFound,
    PatientChanges,
    PatientRecordInput,
    PermissionDenied,
    ReviewResult,
    WeightChanges,
    WhatIfResult,
    patient_fields,
)

DASHBOARD = Path(str(files("medintel") / "dashboard"))
MAX_UPLOAD_BYTES = 2_000_000
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


class BatchRequest(BaseModel):
    sample_name: str | None = Field(default=None, description="A bundled sample, e.g. chest_pain_clinic.csv.")
    csv_text: str | None = Field(default=None, max_length=MAX_UPLOAD_BYTES)
    events_csv_text: str | None = Field(default=None, max_length=MAX_UPLOAD_BYTES)
    source_name: str = "upload.csv"


class CaseRequest(BaseModel):
    batch_id: str | None = None
    patient_id: str | None = None
    record: PatientRecordInput | None = None


class WhatIfRequest(BaseModel):
    changes: PatientChanges | None = None
    weights: WeightChanges | None = None


class ReviewRequest(BaseModel):
    decision: Literal["accept", "annotate", "reject"]
    notes: str | None = None
    reviewer: str | None = Field(default=None, description="Only used when the server runs without sign-in.")


class DecisionRequest(BaseModel):
    approve: bool
    notes: str | None = None
    curator: str | None = Field(default=None, description="Only used when the server runs without sign-in.")


class ServerInfo(BaseModel):
    version: str
    auth: bool
    model: str
    index: str
    summary_writer: str
    retrieval: str
    notice: str = NOTICE


class Me(BaseModel):
    identity: Identity
    role_labels: dict[Role, str]
    server: ServerInfo


def transport_security(public_url: str) -> TransportSecuritySettings:
    """DNS-rebinding protection for /mcp: accept only this server's own host names and origins."""
    url = urlparse(public_url)
    host = url.hostname or "localhost"
    hosts = {"127.0.0.1:*", "localhost:*", "[::1]:*", f"{host}:*", url.netloc}
    origins = {"http://127.0.0.1:*", "http://localhost:*", f"{url.scheme}://{url.netloc}"}
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=sorted(hosts), allowed_origins=sorted(origins)
    )


def server_info(app: MedIntelApp, auth: bool) -> ServerInfo:
    writer = app.writer
    return ServerInfo(
        version=__version__,
        auth=auth,
        model=app.orchestrator.versions().model,
        index=app.knowledge.version,
        summary_writer=str(getattr(writer, "name", "template")) if getattr(writer, "client", None) else "template",
        retrieval=app.knowledge.method,
    )


def create_app(app: MedIntelApp, public_url: str = "http://127.0.0.1:8000", auth: bool = True) -> FastAPI:
    mcp_server = create_server(app, public_url if auth else None)
    mcp_http = mcp_server.streamable_http_app(
        stateless_http=True,
        transport_security=transport_security(public_url),
        host=urlparse(public_url).hostname or "127.0.0.1",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        async with mcp_server.session_manager.run():
            yield
        app.jobs.shutdown()

    api = FastAPI(
        title="MedIntel",
        version=__version__,
        summary="Multi-agent clinical decision support (research prototype; synthetic or de-identified data only).",
        lifespan=lifespan,
    )
    bearer = HTTPBearer(auto_error=False)

    def identity(creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> Identity:
        if not auth:
            return app.local_identity
        who = app.users.verify(creds.credentials) if creds else None
        if who is None:
            raise HTTPException(401, "Sign in with a MedIntel bearer token.", headers={"WWW-Authenticate": "Bearer"})
        return who

    Who = Annotated[Identity, Depends(identity)]

    @api.exception_handler(MedIntelError)
    async def medintel_error(_: Request, exc: MedIntelError) -> JSONResponse:
        status = 403 if isinstance(exc, PermissionDenied) else 404 if isinstance(exc, NotFound) else 400
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @api.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/dashboard"):
            response.headers["Content-Security-Policy"] = CSP
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Cache-Control"] = "no-cache"
        return response

    @api.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/dashboard/")

    @api.get("/healthz", tags=["server"])
    def healthz() -> dict[str, object]:
        return {"status": "ok", "version": __version__, "auth": auth}

    @api.get("/v1/me", tags=["server"])
    def me(who: Who) -> Me:
        return Me(identity=who, role_labels=ROLE_LABELS, server=server_info(app, auth))

    @api.get("/v1/batches", tags=["batches"])
    def list_batches(who: Who, limit: int = Query(20, ge=1, le=100)) -> list[BatchListing]:
        return app.recent_batches(limit)

    @api.post("/v1/batches", tags=["batches"])
    def create_batch(body: BatchRequest, who: Who) -> BatchSummary:
        return app.validate(body.csv_text, body.sample_name, body.events_csv_text, body.source_name, who)

    @api.post("/v1/batches/upload", tags=["batches"])
    async def upload_batch(
        who: Who,
        file: Annotated[UploadFile, File(description="Patient records, CSV")],
        events: Annotated[UploadFile | None, File(description="Event log, CSV (optional)")] = None,
    ) -> BatchSummary:
        async def text(upload: UploadFile) -> str:
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{upload.filename} is larger than {MAX_UPLOAD_BYTES // 1_000_000} MB.")
            try:
                return data.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise HTTPException(400, f"{upload.filename} is not UTF-8 text.") from exc

        csv_text = await text(file)
        events_text = await text(events) if events and events.filename else None
        return app.validate(csv_text, None, events_text, file.filename or "upload.csv", who)

    @api.get("/v1/batches/{batch_id}", tags=["batches"])
    def batch_status(batch_id: str, who: Who) -> BatchStatus:
        return app.batch_status(batch_id)

    @api.post("/v1/batches/{batch_id}/jobs", status_code=202, tags=["jobs"])
    def start_job(batch_id: str, who: Who) -> JobStatus:
        return app.start_job(batch_id, who)

    @api.get("/v1/jobs/{job_id}", tags=["jobs"])
    def job_status(job_id: str, who: Who) -> JobStatus:
        return app.job_status(job_id)

    @api.post("/v1/jobs/{job_id}/cancel", tags=["jobs"])
    def cancel_job(job_id: str, who: Who) -> JobStatus:
        return app.cancel_job(job_id, who)

    @api.post("/v1/cases", tags=["cases"])
    def analyze_case(body: CaseRequest, who: Who) -> CaseResult:
        return app.analyze(body.batch_id, body.patient_id, body.record, who)

    @api.get("/v1/cases/{case_id}", tags=["cases"])
    def get_case(case_id: str, who: Who) -> CaseResult:
        return app.case(case_id)

    @api.get("/v1/cases/{case_id}/explanation", tags=["cases"])
    def explain(case_id: str, who: Who) -> DecisionExplanation:
        return app.explain(case_id)

    @api.post("/v1/cases/{case_id}/what-if", tags=["cases"])
    def what_if(case_id: str, body: WhatIfRequest, who: Who) -> WhatIfResult:
        return app.what_if(case_id, body.changes, body.weights)

    @api.post("/v1/cases/{case_id}/reviews", tags=["review"])
    def review(case_id: str, body: ReviewRequest, who: Who) -> ReviewResult:
        return app.review(case_id, body.decision, body.notes, body.reviewer, who)

    @api.get("/v1/evidence", tags=["evidence"])
    def search_evidence(
        who: Who, query: str | None = None, case_id: str | None = None, k: int = Query(5, ge=1, le=10)
    ) -> EvidenceSearch:
        return app.search_evidence(query, case_id, k)

    @api.get("/v1/knowledge/updates", tags=["knowledge"])
    def knowledge_updates(
        who: Who, status: Literal["pending", "approved", "declined", "all"] = "pending"
    ) -> KnowledgeUpdates:
        return app.knowledge_updates(status)

    @api.post("/v1/knowledge/updates/{proposal_id}/decision", tags=["knowledge"])
    def decide(proposal_id: str, body: DecisionRequest, who: Who) -> KnowledgeDecision:
        return app.decide(proposal_id, body.approve, body.notes, body.curator, who)

    @api.get("/v1/knowledge/base", tags=["knowledge"])
    def knowledge_base(who: Who) -> dict[str, object]:
        return app.knowledge_listing()

    @api.get("/v1/audit", tags=["audit"])
    def audit(who: Who, case_id: str | None = None, limit: int = Query(50, ge=1, le=200)) -> AuditTrail:
        return app.audit_trail(case_id, limit)

    @api.get("/v1/model-card", tags=["server"], response_class=PlainTextResponse)
    def model_card(who: Who) -> str:
        return app.model_card()

    def evaluation_dir() -> Path:
        return Path(os.environ.get("MEDINTEL_REPORTS", "reports")) / "evaluation"

    @api.get("/v1/evaluation", tags=["evaluation"], response_class=PlainTextResponse)
    def evaluation(who: Who) -> str:
        path = evaluation_dir() / "EVALUATION.md"
        if not path.is_file():
            raise HTTPException(404, "No evaluation yet. Run: uv run medintel evaluate")
        return path.read_text(encoding="utf-8")

    @api.get("/v1/evaluation/figures/{name}", tags=["evaluation"])
    def evaluation_figure(name: str, who: Who) -> Response:
        if not re.fullmatch(r"[a-z_]+\.svg", name):
            raise HTTPException(404, "Unknown figure.")
        path = evaluation_dir() / "figures" / name
        if not path.is_file():
            raise HTTPException(404, "Unknown figure.")
        return Response(path.read_text(encoding="utf-8"), media_type="image/svg+xml")

    @api.get("/v1/schema/patient-record", tags=["server"])
    def record_schema(who: Who) -> list[dict[str, object]]:
        return patient_fields()

    api.mount("/dashboard", StaticFiles(directory=DASHBOARD, html=True), name="dashboard")
    api.mount("/", mcp_http)  # last: serves /mcp and its OAuth protected-resource metadata
    return api
