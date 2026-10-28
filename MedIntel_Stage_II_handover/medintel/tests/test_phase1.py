import sqlite3
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from pydantic import ValidationError

from medintel.audit import AuditLogger, CaseStore
from medintel.config import AccessSettings, Role, Settings, load_settings
from medintel.knowledge import KnowledgeRegistry
from medintel.mcp_server import create_server
from medintel.service import MedIntelApp

pytestmark = pytest.mark.anyio


def make_server(home: Path, roles: list[Role] | None = None) -> MCPServer:
    settings = Settings() if roles is None else Settings(access=AccessSettings(roles=roles))
    return create_server(MedIntelApp(home, settings))


async def call(client: Client, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


async def error(client: Client, tool: str, arguments: dict[str, Any]) -> str:
    result = await client.call_tool(tool, arguments)
    assert result.is_error
    return result.content[0].text


async def analyzed_morning_batch(client: Client) -> tuple[str, dict[str, Any]]:
    batch = await call(client, "validate_records", {"sample_name": "morning_batch.csv"})
    return batch["batch_id"], await call(client, "analyze_batch", {"batch_id": batch["batch_id"]})


async def test_analyze_batch_returns_the_triage_queue_with_progress(tmp_path: Path) -> None:
    events: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        events.append((progress, total, message))

    async with Client(make_server(tmp_path)) as client:
        batch = await call(client, "validate_records", {"sample_name": "morning_batch.csv"})
        result = await client.call_tool("analyze_batch", {"batch_id": batch["batch_id"]}, progress_callback=on_progress)
    queue = result.structured_content["queue"]
    assert result.structured_content["analyzed"] == 8 and result.structured_content["rejected_at_validation"] == 2
    assert result.structured_content["counts"] == {"escalate": 1, "review": 1, "priority": 3, "manual": 1, "routine": 2}
    assert [q["case_id"] for q in queue[:2]] == ["C-1042", "C-1044"]
    keys = [(q["priority"]["level"], -q["probability_high_risk"]) for q in queue]
    assert keys == sorted(keys)
    assert sorted(e[0] for e in events) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert all(e[1] == 8 for e in events)


async def test_batch_status_follows_analysis_and_review(tmp_path: Path) -> None:
    async with Client(make_server(tmp_path)) as client:
        batch = await call(client, "validate_records", {"sample_name": "morning_batch.csv"})
        before = await call(client, "get_batch_status", {"batch_id": batch["batch_id"]})
        await call(client, "analyze_batch", {"batch_id": batch["batch_id"]})
        await call(client, "record_review", {"case_id": "C-1048", "decision": "accept"})
        after = await call(client, "get_batch_status", {"batch_id": batch["batch_id"]})
    assert (before["analyzed"], len(before["not_analyzed"])) == (0, 8)
    assert (after["analyzed"], after["pending_review"], after["reviewed"]) == (8, 7, 1)
    assert next(q for q in after["queue"] if q["case_id"] == "C-1048")["status"] == "accepted"


async def test_review_then_curator_approval_versions_the_index(tmp_path: Path) -> None:
    async with Client(make_server(tmp_path)) as client:
        batch_id, _ = await analyzed_morning_batch(client)
        review = await call(
            client,
            "record_review",
            {"case_id": "C-1042", "decision": "accept", "notes": "Repeat BP 182/110 mmHg.", "reviewer": "Dr. Rao"},
        )
        assert review["status"] == "accepted" and review["proposal"]["proposal_id"] == "KU-001"
        assert review["audit_record"]["event"] == "review_accepted" and review["audit_record"]["actor"] == "Dr. Rao"
        again = await call(client, "record_review", {"case_id": "C-1042", "decision": "accept"})
        assert again["proposal"]["proposal_id"] == "KU-001"

        pending = await call(client, "list_knowledge_updates", {})
        assert pending["index_version"] == "kb-v3" and [p["proposal_id"] for p in pending["proposals"]] == ["KU-001"]

        decision = await call(
            client, "approve_knowledge_update", {"proposal_id": "KU-001", "approve": True, "curator": "Ms. Iyer"}
        )
        assert (decision["index_version_before"], decision["index_version_after"]) == ("kb-v3", "kb-v4")
        assert decision["proposal"]["status"] == "approved" and decision["proposal"]["decided_by"] == "Ms. Iyer"

        overview = await call(client, "list_knowledge_updates", {"status": "all"})
        assert overview["index_version"] == "kb-v4" and overview["validated_cases"] == ["C-1042"]
        assert overview["history"][0]["description"] == (
            "Added validated outcome from C-1042 (high risk confirmed by a clinician)"
        )
        assert "already approved" in await error(
            client, "approve_knowledge_update", {"proposal_id": "KU-001", "approve": True}
        )

        fresh = await call(client, "analyze_case", {"batch_id": batch_id, "patient_id": "P-1041"})
        assert fresh["report"]["versions"]["index"] == "kb-v4"
        trail = await call(client, "get_audit_trail", {"case_id": "C-1042"})
        assert [r["event"] for r in trail["records"]] == ["review_accepted", "review_accepted", "report_created"]
        resource = await client.read_resource("medintel://cases/C-1042/report")
        assert '"status": "accepted"' in resource.contents[0].text


async def test_decline_keeps_the_index_and_uncertain_is_not_proposed(tmp_path: Path) -> None:
    async with Client(make_server(tmp_path)) as client:
        await analyzed_morning_batch(client)
        low = await call(client, "record_review", {"case_id": "C-1041", "decision": "accept"})
        declined = await call(
            client,
            "approve_knowledge_update",
            {"proposal_id": low["proposal"]["proposal_id"], "approve": False, "notes": "Too early."},
        )
        assert declined["index_version_after"] == "kb-v3" and declined["proposal"]["status"] == "declined"
        uncertain = await call(client, "record_review", {"case_id": "C-1044", "decision": "accept"})
        assert uncertain["proposal"] is None and "not proposed" in uncertain["message"]
        assert "Unknown proposal" in await error(
            client, "approve_knowledge_update", {"proposal_id": "KU-9", "approve": True}
        )


async def test_annotate_and_reject_need_notes(tmp_path: Path) -> None:
    async with Client(make_server(tmp_path)) as client:
        await analyzed_morning_batch(client)
        assert "Add notes to annotate" in await error(
            client, "record_review", {"case_id": "C-1043", "decision": "annotate"}
        )
        assert "Add notes to reject" in await error(
            client, "record_review", {"case_id": "C-1043", "decision": "reject"}
        )
        rejected = await call(
            client,
            "record_review",
            {"case_id": "C-1043", "decision": "reject", "notes": "ECG normal; refer for lipid review only."},
        )
    assert rejected["status"] == "rejected" and rejected["proposal"] is None


async def test_what_if_crosses_the_r001_boundary_without_saving(tmp_path: Path) -> None:
    async with Client(make_server(tmp_path)) as client:
        await analyzed_morning_batch(client)
        audit_before = len((await call(client, "get_audit_trail", {"limit": 200}))["records"])
        below = await call(client, "what_if", {"case_id": "C-1042", "changes": {"systolic_bp": 179}})
        at = await call(client, "what_if", {"case_id": "C-1042", "changes": {"systolic_bp": 180}})
        stored = await call(client, "explain_decision", {"case_id": "C-1042"})
        audit_after = len((await call(client, "get_audit_trail", {"limit": 200}))["records"])
    assert below["after"]["triggered_rules"] == [] and below["after"]["decision"] == "low_risk"
    assert "Rule R-001 no longer fires." in below["differences"]
    assert "Decision changed: High risk to Low risk." in below["differences"]
    assert "Priority changed: Escalate to Routine." in below["differences"]
    assert at["after"]["triggered_rules"] == ["R-001"] and at["after"]["decision"] == "high_risk"
    assert below["saved"] is False and stored["decision"] == "high_risk" and audit_before == audit_after


async def test_what_if_weights_missing_values_and_validation(tmp_path: Path) -> None:
    async with Client(make_server(tmp_path)) as client:
        await analyzed_morning_batch(client)
        weighted = await call(client, "what_if", {"case_id": "C-1044", "weights": {"alpha": 1.0, "threshold": 0.2}})
        missing = await call(client, "what_if", {"case_id": "C-1041", "changes": {"creatinine_missing": True}})
        invalid = await error(client, "what_if", {"case_id": "C-1041", "changes": {"systolic_bp": 400}})
    weights = weighted["weights_used"]
    assert weights["alpha"] + weights["beta"] + weights["gamma"] == pytest.approx(1.0)
    assert weights["alpha"] == pytest.approx(1 / 1.5)
    assert weighted["before"]["decision"] == "uncertain" and weighted["after"]["decision"] == "high_risk"
    assert missing["changes_applied"] == {"creatinine": None}
    assert any(d.startswith("Rule R-002 fired") for d in missing["differences"])
    assert "400 is outside 60-260 mmHg" in invalid


async def test_search_evidence_by_query_and_by_case(tmp_path: Path) -> None:
    async with Client(make_server(tmp_path)) as client:
        await analyzed_morning_batch(client)
        by_query = await call(client, "search_evidence", {"query": "kidney creatinine"})
        by_case = await call(client, "search_evidence", {"case_id": "C-1042"})
        both = await error(client, "search_evidence", {"query": "x", "case_id": "C-1042"})
    assert [r["source_id"] for r in by_query["results"]] == ["KB-CKD-004", "KB-CKD-009"]
    assert by_query["results"][0]["score"] == 1.0 and by_query["method"].startswith("keyword")
    assert [r["source_id"] for r in by_case["results"]] == ["KB-HTN-001", "KB-HTN-004"]
    assert by_case["index_version"] == "kb-v3"
    assert "exactly one" in both


async def test_roles_gate_the_write_tools(tmp_path: Path) -> None:
    async with Client(make_server(tmp_path)) as client:
        await analyzed_morning_batch(client)
        await call(client, "record_review", {"case_id": "C-1042", "decision": "accept"})
    async with Client(make_server(tmp_path, ["clinician"])) as clinician:
        assert "Data Administrator" in await error(clinician, "validate_records", {"sample_name": "morning_batch.csv"})
        assert "Knowledge Curator" in await error(
            clinician, "approve_knowledge_update", {"proposal_id": "KU-001", "approve": True}
        )
        await call(clinician, "record_review", {"case_id": "C-1048", "decision": "accept"})
    async with Client(make_server(tmp_path, ["curator"])) as curator:
        assert "Clinician" in await error(curator, "record_review", {"case_id": "C-1041", "decision": "accept"})
        await call(curator, "approve_knowledge_update", {"proposal_id": "KU-001", "approve": True})
        await call(curator, "list_knowledge_updates", {})


def test_roles_come_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEDINTEL_ROLES", "clinician, curator")
    assert load_settings(tmp_path / "missing.toml").access.roles == ["clinician", "curator"]
    monkeypatch.setenv("MEDINTEL_ROLES", "doctor")
    with pytest.raises(ValidationError):
        load_settings(tmp_path / "missing.toml")


async def test_prompts(tmp_path: Path) -> None:
    async with Client(make_server(tmp_path)) as client:
        names = {p.name for p in (await client.list_prompts()).prompts}
        prompt = await client.get_prompt("review_case", {"case_id": "C-1042"})
    assert names == {"triage_batch", "review_case", "curator_review"}
    text = prompt.messages[0].content.text
    assert "explain_decision" in text and "C-1042" in text and "Only after they answer" in text


def test_databases_from_phase_0_are_migrated(tmp_path: Path) -> None:
    db_path = tmp_path / "medintel.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE audit (case_id TEXT NOT NULL, model_version TEXT NOT NULL, rulebase_version TEXT NOT NULL, "
            "retrieval_ids TEXT NOT NULL, status TEXT NOT NULL, timestamp TEXT NOT NULL)"
        )
        db.execute("INSERT INTO audit VALUES ('C-1042', 'm', '1.0.0', '', 'pending_review', '2026-10-17T00:00:00')")
        db.execute("CREATE TABLE analyses (case_id TEXT PRIMARY KEY, analyzed_at TEXT NOT NULL, payload TEXT NOT NULL)")
    record = AuditLogger(db_path).records()[0]
    assert (record.event, record.actor) == ("report_created", "system")
    assert CaseStore(db_path).batch_analyses("B-none") == []


def test_registry_versions_start_at_the_base_index(tmp_path: Path) -> None:
    registry = KnowledgeRegistry(tmp_path / "kb.db")
    assert registry.label() == "kb-v3"
    assert [v.label for v in registry.history()] == ["kb-v3"]
    with pytest.raises(KeyError):
        registry.decide("KU-404", True, "Curator")
    assert KnowledgeRegistry(tmp_path / "kb.db").current_version() == 3
