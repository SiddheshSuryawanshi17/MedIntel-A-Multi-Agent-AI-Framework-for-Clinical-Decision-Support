"""Claude writer with a stand-in client: structured replies, the TC-12 check on Claude's sentences, fallbacks and
cost accounting. No network calls and no API key are needed."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from medintel.agents.claude_writer import (
    OUTPUT_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    ClaudeWriter,
    case_facts,
    create_writer,
    estimate_cost,
)
from medintel.agents.context_agent import ClinicalContextAgent, TemplateWriter, find_conflicts
from medintel.config import ContextSettings, Settings
from medintel.models import CaseAnalysis
from medintel.orchestrator import Orchestrator
from medintel.service import MedIntelApp, PatientChanges


class FakeMessages:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeClient:
    def __init__(self, *replies: Any) -> None:
        self.messages = FakeMessages(list(replies))


def reply(sentences: list[dict[str, Any]] | str, stop_reason: str = "end_turn") -> Any:
    text = sentences if isinstance(sentences, str) else json.dumps({"sentences": sentences})
    usage = SimpleNamespace(
        input_tokens=2000, output_tokens=500, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=usage,
        _request_id="req_test",
    )


class Boom(anthropic.AnthropicError):
    pass


@pytest.fixture
def c1042(morning: Any, orchestrator: Orchestrator) -> CaseAnalysis:
    case = next(c for c in morning.cases if c.patient_id == "P-1042")
    return orchestrator.run_case(case)


def synthesize(writer: ClaudeWriter, a: CaseAnalysis) -> Any:
    return ClinicalContextAgent(writer).synthesize(a.case, a.ml, a.rules, a.evidence)


def test_structured_reply_becomes_the_summary(c1042: CaseAnalysis) -> None:
    client = FakeClient(
        reply(
            [
                {"text": "Blood pressure is 186/112 mmHg.", "sources": ["record:systolic_bp", "record:diastolic_bp"]},
                {"text": "Rule R-001 elevates the case.", "sources": ["rule:R-001"]},
            ]
        )
    )
    context = synthesize(ClaudeWriter(client=client), c1042)
    assert context.summary == "Blood pressure is 186/112 mmHg. Rule R-001 elevates the case."
    assert context.generated_by == f"claude-opus-5-5 (effort low, prompt {PROMPT_VERSION})"
    assert context.usage is not None and context.usage.cost_usd == pytest.approx(0.018)
    assert context.usage.request_id == "req_test" and not context.unsupported_statements


def test_tc12_removes_unsupported_claude_sentences(c1042: CaseAnalysis) -> None:
    client = FakeClient(
        reply(
            [
                {"text": "Blood pressure is 186/112 mmHg.", "sources": ["record:systolic_bp"]},
                {"text": "The patient has a history of stroke.", "sources": ["record:history"]},
                {"text": "No rule fired.", "sources": []},
                {"text": "Rule R-001 elevates the case.", "sources": ["rule:R-001"]},
            ]
        )
    )
    context = synthesize(ClaudeWriter(client=client), c1042)
    assert context.summary == "Rule R-001 elevates the case."
    reasons = {s.text: s.reason for s in context.draft if not s.supported}
    assert "112 does not appear" in reasons["Blood pressure is 186/112 mmHg."]
    assert "record:history" in reasons["The patient has a history of stroke."]
    assert "cites no source" in reasons["No rule fired."]


@pytest.mark.parametrize(
    ("replies", "reason"),
    [
        ([Boom("down")], "template (Claude unavailable: Boom)"),
        ([reply([], "refusal")], "template (Claude reply unusable: stop reason refusal)"),
        ([reply("not json")], "template (Claude reply unusable: reply did not match the schema)"),
        ([reply([])], "template (Claude reply unusable: empty reply)"),
    ],
)
def test_failures_fall_back_to_the_template(c1042: CaseAnalysis, replies: list[Any], reason: str) -> None:
    context = synthesize(ClaudeWriter(client=FakeClient(*replies)), c1042)
    assert context.generated_by == reason
    template = ClinicalContextAgent(TemplateWriter()).synthesize(c1042.case, c1042.ml, c1042.rules, c1042.evidence)
    assert context.summary == template.summary


def test_spent_tokens_are_kept_when_the_reply_is_unusable(c1042: CaseAnalysis) -> None:
    context = synthesize(ClaudeWriter(client=FakeClient(reply([], "refusal"))), c1042)
    assert context.usage is not None and context.usage.output_tokens == 500


def test_missing_key_uses_the_template(c1042: CaseAnalysis, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    context = synthesize(ClaudeWriter(), c1042)
    assert context.generated_by == "template (no ANTHROPIC_API_KEY)" and context.usage is None


def test_request_uses_structured_output_and_cached_system_prompt(c1042: CaseAnalysis) -> None:
    client = FakeClient(reply([{"text": "Rule R-001 elevates the case.", "sources": ["rule:R-001"]}]))
    synthesize(ClaudeWriter(client=client, effort="medium"), c1042)
    params = client.messages.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}
    assert params["system"][0]["text"] == SYSTEM_PROMPT
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    facts = json.loads(params["messages"][0]["content"])
    assert "rule:R-001" in facts["allowed_sources"] and "notes" in facts["allowed_sources"]
    assert facts["record"]["systolic_bp"] == {"label": "Systolic BP", "value": 186.0, "unit": "mmHg"}
    assert "thinking" not in params, "Opus 5.5 always thinks; depth is set by effort"


def test_facts_name_missing_values(boundary: Any, orchestrator: Orchestrator) -> None:
    a = orchestrator.run_case(next(c for c in boundary.cases if c.patient_id == "P-2006"))
    facts = case_facts(a.case, a.ml, a.rules, a.evidence, find_conflicts(a.case, a.ml, a.rules, a.evidence))
    assert "Creatinine" in facts["not_recorded"]
    assert [r["id"] for r in facts["rules"]] == ["R-002"]


def test_cost_estimate_uses_list_prices() -> None:
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=1000, cache_creation_input_tokens=2000, cache_read_input_tokens=0
    )
    assert estimate_cost("claude-opus-5-5", usage) == pytest.approx(0.004 + 0.02 + 0.01)
    assert estimate_cost("claude-opus-5-5", usage, batch=True) == pytest.approx(0.017)


def test_writer_follows_settings() -> None:
    assert isinstance(create_writer(ContextSettings()), TemplateWriter)
    writer = create_writer(ContextSettings(writer="claude", effort="medium"))
    assert isinstance(writer, ClaudeWriter) and writer.effort == "medium"


def test_app_uses_claude_for_reports_but_not_for_what_if(tmp_path: Path) -> None:
    client = FakeClient(reply([{"text": "Rule R-001 elevates the case.", "sources": ["rule:R-001"]}]))
    app = MedIntelApp(tmp_path, Settings(), writer=ClaudeWriter(client=client))
    batch = app.validate(None, "morning_batch.csv", None, "x")
    result = app.analyze(batch.batch_id, "P-1042", None)
    assert result.summary_written_by.startswith("claude-opus-5-5")
    app.what_if(result.report.case_id, PatientChanges(systolic_bp=150), None)
    assert len(client.messages.calls) == 1, "what-if explorations never make paid calls"
