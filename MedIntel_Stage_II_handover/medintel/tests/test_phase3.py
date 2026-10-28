"""Phase 3: identities and tokens, background jobs, the REST API, MCP over Streamable HTTP with bearer auth, and the
dashboard's headers. Everything runs in process; no network."""

import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from medintel.config import Settings
from medintel.identity import UserError, UserStore, token_hash
from medintel.jobs import JobRunner, JobStore
from medintel.models import CaseAnalysis, PatientCase
from medintel.service import MedIntelApp
from medintel.web import create_app

BASE = "http://127.0.0.1:8000"
MORNING_CSV = (Path(__file__).resolve().parents[1] / "src" / "medintel" / "samples" / "morning_batch.csv").read_text()


@pytest.fixture
def app(tmp_path: Path) -> MedIntelApp:
    return MedIntelApp(tmp_path, Settings())


@pytest.fixture
def tokens(app: MedIntelApp) -> dict[str, str]:
    return {
        "clinician": app.users.add("Dr. Rao", ["clinician", "data_admin"]),
        "curator": app.users.add("Ms. Iyer", ["curator"]),
    }


@pytest.fixture
def client(app: MedIntelApp, tokens: dict[str, str]) -> Iterator[TestClient]:
    with TestClient(create_app(app, BASE), base_url=BASE) as c:
        yield c


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_tokens_are_shown_once_and_stored_hashed(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "u.db")
    token = store.add("Dr. Rao", ["clinician"])
    assert token.startswith("mi_") and len(token) > 40
    who = store.verify(token)
    assert who is not None and who.name == "Dr. Rao" and who.roles == ["clinician"] and who.authenticated
    assert store.verify("mi_wrong") is None and store.verify("") is None
    rows = sqlite3.connect(tmp_path / "u.db").execute("SELECT token_hash FROM users").fetchall()
    assert rows == [(token_hash(token),)], "only the hash is stored"
    new = store.rotate("Dr. Rao")
    assert store.verify(token) is None and store.verify(new) is not None
    store.set_disabled("Dr. Rao", True)
    assert store.verify(new) is None
    with pytest.raises(UserError, match="already exists"):
        store.add("Dr. Rao", ["clinician"])
    with pytest.raises(UserError, match="Unknown role"):
        store.add("Nurse Joy", ["nurse"])
    with pytest.raises(UserError, match="at least one role"):
        store.add("Nobody", [])


def test_background_job_runs_the_whole_batch(app: MedIntelApp) -> None:
    batch = app.validate(None, "boundary_tests.csv", None, "x")
    status = app.start_job(batch.batch_id)
    job = app.jobs.wait(status.job.job_id, 30)
    assert job is not None and job.status == "completed" and job.done == job.total == 6
    final = app.job_status(job.job_id)
    assert final.progress == 1.0 and final.result is not None and final.result.analyzed == 6
    assert job.created_by == "local user"


def test_jobs_can_be_cancelled_and_survive_a_restart(tmp_path: Path, boundary: Any) -> None:
    gate = threading.Event()

    def slow(case: PatientCase, batch_id: str) -> CaseAnalysis:
        gate.wait(5)
        raise RuntimeError("not needed")

    runner = JobRunner(tmp_path / "jobs.db", slow, per_job=1)
    job = runner.start("B-1", list(boundary.cases), "tester")
    time.sleep(0.1)
    runner.cancel(job.job_id)
    gate.set()
    final = runner.wait(job.job_id, 10)
    assert final is not None and final.status == "cancelled" and final.done + final.failed < final.total
    stuck = JobStore(tmp_path / "jobs.db").create("B-2", 3, "tester")
    JobRunner(tmp_path / "jobs.db", slow)  # a fresh server start
    assert JobStore(tmp_path / "jobs.db").get(stuck.job_id).status == "interrupted"  # type: ignore[union-attr]


def test_rest_requires_a_token(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/v1/me").status_code == 401
    assert client.get("/v1/me", headers=bearer("mi_nope")).status_code == 401


def test_rest_end_to_end_with_roles(client: TestClient, tokens: dict[str, str]) -> None:
    clinician, curator = bearer(tokens["clinician"]), bearer(tokens["curator"])
    me = client.get("/v1/me", headers=clinician).json()
    assert me["identity"]["name"] == "Dr. Rao" and me["server"]["auth"] is True
    batch = client.post("/v1/batches", json={"sample_name": "morning_batch.csv"}, headers=clinician).json()
    assert batch["valid"] == 8
    assert client.post("/v1/batches", json={"sample_name": "morning_batch.csv"}, headers=curator).status_code == 403
    started = client.post(f"/v1/batches/{batch['batch_id']}/jobs", headers=clinician)
    assert started.status_code == 202
    job_id = started.json()["job"]["job_id"]
    for _ in range(200):
        status = client.get(f"/v1/jobs/{job_id}", headers=clinician).json()
        if status["job"]["status"] == "completed":
            break
        time.sleep(0.05)
    queue = status["result"]["queue"]
    assert [q["case_id"] for q in queue][:1] == ["C-1042"]
    assert client.get(f"/v1/cases/{queue[0]['case_id']}", headers=curator).json()["report"]["decision"] == "high_risk"
    explanation = client.get("/v1/cases/C-1042/explanation", headers=clinician).json()
    assert explanation["rules_fired"] == ["R-001"]
    what_if = client.post("/v1/cases/C-1042/what-if", json={"changes": {"systolic_bp": 179}}, headers=clinician)
    assert "Rule R-001 no longer fires." in what_if.json()["differences"]
    review = client.post(
        "/v1/cases/C-1043/reviews", json={"decision": "accept", "reviewer": "Someone Else"}, headers=clinician
    ).json()
    assert review["reviewer"] == "Dr. Rao", "the signed-in identity is the actor"
    proposal = review["proposal"]["proposal_id"]
    denied = client.post(f"/v1/knowledge/updates/{proposal}/decision", json={"approve": True}, headers=clinician)
    assert denied.status_code == 403 and "Dr. Rao does not have" in denied.json()["detail"]
    approved = client.post(f"/v1/knowledge/updates/{proposal}/decision", json={"approve": True}, headers=curator)
    assert approved.json()["index_version_after"] == "kb-v4"
    trail = client.get("/v1/audit", params={"case_id": "C-1043"}, headers=curator).json()["records"]
    assert (trail[0]["event"], trail[0]["actor"]) == ("review_accepted", "Dr. Rao")
    assert client.get("/v1/cases/C-9999", headers=clinician).status_code == 404
    listing = client.get("/v1/batches", headers=clinician).json()
    assert listing[0]["batch_id"] == batch["batch_id"] and listing[0]["analyzed"] == 8


def test_csv_upload_and_its_size_limit(client: TestClient, tokens: dict[str, str]) -> None:
    files = {"file": ("morning.csv", MORNING_CSV.encode(), "text/csv")}
    summary = client.post("/v1/batches/upload", files=files, headers=bearer(tokens["clinician"])).json()
    assert summary["valid"] == 8 and summary["source_name"] == "morning.csv"
    huge = {"file": ("huge.csv", b"x" * 2_000_001, "text/csv")}
    assert client.post("/v1/batches/upload", files=huge, headers=bearer(tokens["clinician"])).status_code == 413


def test_docs_dashboard_and_headers(client: TestClient) -> None:
    assert "/v1/cases/{case_id}/explanation" in client.get("/openapi.json").json()["paths"]
    page = client.get("/dashboard/")
    assert page.status_code == 200 and "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert page.headers["x-content-type-options"] == "nosniff"
    assert client.get("/", follow_redirects=False).headers["location"] == "/dashboard/"


async def mcp_session(app: MedIntelApp, token: str | None, auth: bool = True) -> Any:
    api = create_app(app, BASE, auth=auth)
    headers = bearer(token) if token else {}
    http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=api), base_url=BASE, headers=headers, timeout=60)
    return api, Client(streamable_http_client(f"{BASE}/mcp", http_client=http))


@pytest.mark.anyio
async def test_mcp_over_http_uses_the_signed_in_identity(app: MedIntelApp, tokens: dict[str, str]) -> None:
    api, session = await mcp_session(app, tokens["clinician"])
    async with api.router.lifespan_context(api), session as client:
        batch = (await client.call_tool("validate_records", {"sample_name": "morning_batch.csv"})).structured_content
        assert batch is not None
        await client.call_tool("analyze_case", {"batch_id": batch["batch_id"], "patient_id": "P-1043"})
        review = await client.call_tool("record_review", {"case_id": "C-1043", "decision": "accept", "reviewer": "X"})
        assert review.structured_content["reviewer"] == "Dr. Rao"  # type: ignore[index]
        denied = await client.call_tool("approve_knowledge_update", {"proposal_id": "KU-001", "approve": True})
        assert denied.is_error and "Knowledge Curator" in denied.content[0].text  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_mcp_over_http_rejects_bad_tokens_and_foreign_hosts(app: MedIntelApp, tokens: dict[str, str]) -> None:
    api, session = await mcp_session(app, "mi_wrong")
    async with api.router.lifespan_context(api):
        with pytest.raises(Exception):  # noqa: B017  the client surfaces the 401 as an exception group
            async with session as client:
                await client.list_tools()
        transport = httpx2.ASGITransport(app=api)
        async with httpx2.AsyncClient(transport=transport, base_url="http://evil.example") as raw:
            response = await raw.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={**bearer(tokens["clinician"]), "Accept": "application/json, text/event-stream"},
            )
        assert response.status_code in (400, 403, 421), "DNS-rebinding protection refuses unknown hosts"


@pytest.mark.anyio
async def test_local_no_auth_mode(app: MedIntelApp) -> None:
    api, session = await mcp_session(app, None, auth=False)
    async with api.router.lifespan_context(api), session as client:
        tools = (await client.list_tools()).tools
    assert len(tools) == 14


def test_anyio_backend_is_asyncio() -> None:
    assert anyio.get_all_backends()[0] == "asyncio"


def test_users_cli_saves_tokens_privately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from medintel.cli import main

    monkeypatch.setenv("MEDINTEL_HOME", str(tmp_path))
    monkeypatch.setenv("MEDINTEL_SETTINGS", str(tmp_path / "none.toml"))
    token_file = tmp_path / "tokens" / "rao"
    main(["users", "add", "Dr. Rao", "--role", "clinician", "--save-token", str(token_file)])
    assert oct(token_file.stat().st_mode)[-3:] == "600"
    assert UserStore(tmp_path / "medintel.db").verify(token_file.read_text().strip()) is not None
    with pytest.raises(SystemExit):
        main(["users", "add", "Nurse", "--role", "nurse"])


def test_serve_refuses_unsafe_setups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from medintel.cli import main

    monkeypatch.setenv("MEDINTEL_HOME", str(tmp_path))
    monkeypatch.setenv("MEDINTEL_SETTINGS", str(tmp_path / "none.toml"))
    with pytest.raises(SystemExit):
        main(["serve", "--host", "0.0.0.0", "--no-auth"])
    with pytest.raises(SystemExit):
        main(["serve"])  # no users yet, and sign-in is on
