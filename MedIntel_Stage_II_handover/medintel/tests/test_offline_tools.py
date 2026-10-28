"""Offline tools: fusion tuning, the writer evaluation (with a stand-in Batch API) and the CLI's spending guard."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from medintel.agents.claude_writer import ClaudeWriter
from medintel.cli import main
from medintel.fusion import FusionSettings
from medintel.orchestrator import Orchestrator
from medintel.training import fusion_tuning as ft
from medintel.training import writer_eval as we


def row(m: float, correct: int, r: float = 1.0, k: float = 0.5, d: float = 0.0, rule_conflict: int = 0) -> dict:
    return {"M": m, "R": r, "K": k, "D": d, "correct": correct, "rule_conflict": rule_conflict}


ROWS = [row(0.95, 1), row(0.9, 1), row(0.85, 1), row(0.8, 1), row(0.6, 0), row(0.55, 0), row(0.7, 1), row(0.58, 1)]


def test_untestable_weights_are_held() -> None:
    held = ft.held_weights(ROWS, FusionSettings())
    assert set(held) == {"beta", "lam"}
    tuned, _ = ft.search_weights(ROWS, FusionSettings(), held)
    assert (tuned.beta, tuned.lam) == (0.3, 0.5)
    assert tuned.alpha + tuned.beta + tuned.gamma == pytest.approx(1.0)


def test_free_search_puts_weight_on_the_informative_component() -> None:
    free, score = ft.search_weights(ROWS, FusionSettings())
    assert free.alpha >= 0.8 and score < ft.brier(ft.confidences(ROWS, FusionSettings()), ft.scored(ROWS)[1])


def test_threshold_reaches_the_target() -> None:
    f = FusionSettings(alpha=1.0, beta=0.0, gamma=0.0, lam=0.0)
    t = ft.choose_threshold(ROWS, f, target=1.0)
    kept, y = ft.scored(ROWS)
    assert y[ft.confidences(kept, f) >= t].mean() == 1.0
    table = ft.threshold_table(ROWS, f)
    assert [r["coverage"] for r in table] == sorted((r["coverage"] for r in table), reverse=True)


def reply(text: str) -> Any:
    usage = SimpleNamespace(
        input_tokens=1500, output_tokens=400, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)], usage=usage)


class FakeBatches:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, requests: list[dict]) -> Any:
        self.requests = requests
        return SimpleNamespace(id="msgbatch_1", processing_status="in_progress")

    def retrieve(self, batch_id: str) -> Any:
        return SimpleNamespace(
            id=batch_id, processing_status="ended", request_counts=SimpleNamespace(succeeded=2, processing=0)
        )

    def results(self, batch_id: str) -> list[Any]:
        good = json.dumps({"sentences": [{"text": "Rule R-001 elevates the case.", "sources": ["rule:R-001"]}]})
        bad = json.dumps({"sentences": [{"text": "Blood pressure is 999 mmHg.", "sources": ["record:systolic_bp"]}]})
        out = []
        for i, request in enumerate(self.requests):
            message = reply(good if i == 0 else bad)
            out.append(
                SimpleNamespace(
                    custom_id=request["custom_id"], result=SimpleNamespace(type="succeeded", message=message)
                )
            )
        return out


def test_batch_results_go_through_tc12(boundary: Any, orchestrator: Orchestrator) -> None:
    analyses = [orchestrator.run_case(c) for c in boundary.cases if c.patient_id in ("P-2002", "P-2003")]
    client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches()))
    writer = ClaudeWriter(client=client)
    rows = we.run_batch(client, [writer], analyses, poll_s=0, log=lambda _: None)
    assert len(client.messages.batches.requests) == 2
    assert rows[0]["removed"] == 0 and rows[1]["removed"] == 1
    summary = we.summarize(rows)[writer.name]
    assert summary["sentences_removed_by_tc12"] == 1 and summary["cases_with_a_removal"] == 1
    assert summary["cost_usd_total"] == pytest.approx(2 * (1500 * 4 + 400 * 20) / 1e6 * 0.5)


def test_eval_writer_spends_nothing_without_yes(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEDINTEL_SETTINGS", "/nonexistent/settings.toml")
    main(["eval-writer", "--cases", "samples", "--limit", "2"])
    out = capsys.readouterr().out
    assert "2 requests" in out and "Nothing sent" in out
