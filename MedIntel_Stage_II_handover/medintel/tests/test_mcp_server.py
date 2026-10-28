import json
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer

from medintel.mcp_server import MedIntelApp, create_server

pytestmark = pytest.mark.anyio

RECORD = {
    "patient_id": "P-3001",
    "age": 52,
    "sex": "F",
    "systolic_bp": 132,
    "diastolic_bp": 84,
    "heart_rate": 124,
    "temperature_c": 37.0,
    "total_cholesterol": 205,
    "fasting_glucose": 99,
    "creatinine": 0.9,
    "bmi": 27.5,
    "smoker": False,
}


@pytest.fixture
def server(tmp_path: Path) -> MCPServer:
    return create_server(MedIntelApp(tmp_path))


async def call(client: Client, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


async def test_tools_and_annotations(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    read_only = {
        "get_batch_status",
        "explain_decision",
        "what_if",
        "search_evidence",
        "list_knowledge_updates",
        "get_audit_trail",
        "get_job_status",
    }
    writes = {
        "validate_records",
        "analyze_batch",
        "analyze_case",
        "record_review",
        "approve_knowledge_update",
        "start_batch_analysis",
        "cancel_job",
    }
    assert set(tools) == read_only | writes
    for name, tool in tools.items():
        hints = tool.annotations.model_dump(by_alias=True)
        assert hints["readOnlyHint"] is (name in read_only), name
        assert hints["openWorldHint"] is False
        assert tool.model_dump(by_alias=True)["outputSchema"]["type"] == "object", name
        assert tool.title, name
    assert tools["record_review"].annotations.model_dump(by_alias=True)["destructiveHint"] is False
    review_inputs = tools["record_review"].model_dump(by_alias=True)["inputSchema"]["properties"]
    assert "a note is optional" in review_inputs["decision"]["description"]


async def test_validate_analyze_explain_and_audit(server: MCPServer) -> None:
    async with Client(server) as client:
        batch = await call(client, "validate_records", {"sample_name": "morning_batch.csv"})
        assert batch["valid"] == 8 and batch["records_read"] == 10
        assert [(r["row"], r["problems"][0]["test_case"]) for r in batch["rejected"]] == [(10, "TC-01"), (11, "TC-02")]

        case = await call(client, "analyze_case", {"batch_id": batch["batch_id"], "patient_id": "P-1042"})
        report = case["report"]
        assert (report["case_id"], report["decision"], report["confidence"]) == ("C-1042", "high_risk", 0.253)
        assert case["priority"]["key"] == "escalate"
        assert report["triggered_rules"] == ["R-001"] and len(report["conflicts"]) == 2
        assert case["summary_written_by"] == "template"

        explained = await call(client, "explain_decision", {"case_id": "C-1042"})
        assert explained["component_scores"]["R"] == 0
        assert explained["steps"][0].startswith("Rule R-001 fired with action elevate")
        assert [e["source_id"] for e in explained["evidence"]] == ["KB-HTN-001", "KB-HTN-004"]

        trail = await call(client, "get_audit_trail", {"case_id": "C-1042"})
        assert [r["status"] for r in trail["records"]] == ["pending_review"]
        assert trail["records"][0]["retrieval_ids"] == "KB-HTN-001,KB-HTN-004"

        resource = await client.read_resource("medintel://cases/C-1042/report")
        assert json.loads(resource.contents[0].text)["recommendation"].startswith("Escalate")
        rulebase = await client.read_resource("medintel://rulebase")
        assert json.loads(rulebase.contents[0].text)["version"] == "1.0.0"
        schema = await client.read_resource("medintel://schema/decision-report")
        assert json.loads(schema.contents[0].text)["title"] == "DecisionReport"


async def test_single_record_input(server: MCPServer) -> None:
    async with Client(server) as client:
        case = await call(client, "analyze_case", {"record": RECORD})
        assert case["report"]["triggered_rules"] == ["R-004"]
        assert case["report"]["decision"] == "low_risk"
        rejected = await client.call_tool("analyze_case", {"record": {**RECORD, "temperature_c": 101.2}})
    assert rejected.is_error
    assert "TC-02" not in rejected.content[0].text
    assert "looks like °F" in rejected.content[0].text


async def test_errors_are_readable(server: MCPServer) -> None:
    async with Client(server) as client:
        unknown = await client.call_tool("analyze_case", {"batch_id": "B-nope", "patient_id": "P-1"})
        both = await client.call_tool("validate_records", {"csv_text": "x", "sample_name": "morning_batch.csv"})
        missing = await client.call_tool("explain_decision", {"case_id": "C-9999"})
        sample = await client.call_tool("validate_records", {"sample_name": "../secrets.csv"})
    for result in (unknown, both, missing, sample):
        assert result.is_error
    assert "validate_records first" in unknown.content[0].text
    assert "Unknown sample" in sample.content[0].text


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_runs_over_stdio(tmp_path: Path, mode: str) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "medintel.mcp_server"],
        env={"MEDINTEL_HOME": str(tmp_path), "MEDINTEL_SETTINGS": str(tmp_path / "defaults.toml")},
    )
    async with Client(params, mode=mode) as client:
        batch = await call(client, "validate_records", {"sample_name": "boundary_tests.csv"})
        assert batch["valid"] == 6
        case = await call(client, "analyze_case", {"batch_id": batch["batch_id"], "patient_id": "P-2002"})
        assert case["report"]["triggered_rules"] == ["R-001"]
        assert case["priority"]["key"] == "escalate"
    assert (tmp_path / "medintel.db").exists()


async def test_phase2_resources_and_cardiology_records(server: MCPServer) -> None:
    async with Client(server) as client:
        uris = {str(r.uri) for r in (await client.list_resources()).resources}
        assert {"medintel://model-card", "medintel://knowledge-base", "medintel://schema/patient-record"} <= uris
        card = await client.read_resource("medintel://model-card")
        assert "demo weights" in card.contents[0].text
        fields = json.loads((await client.read_resource("medintel://schema/patient-record")).contents[0].text)
        thal = next(f for f in fields if f["key"] == "thal")
        assert thal["required"] is False and "reversible_defect" in thal["values"]
        kb = json.loads((await client.read_resource("medintel://knowledge-base")).contents[0].text)
        assert kb["index_version"] == "kb-v3" and kb["passages"]
        batch = await call(client, "validate_records", {"sample_name": "chest_pain_clinic.csv"})
        assert batch["valid"] == 10
        record = {**RECORD, "chest_pain_type": "typical_angina", "st_depression": 2.1, "exercise_angina": True}
        case = await call(client, "analyze_case", {"record": record})
        explanation = await call(client, "explain_decision", {"case_id": case["report"]["case_id"]})
    assert explanation["model"]["submodel"] == "prototype demo weights (not trained)"
    assert explanation["summary_written_by"] == "template"
