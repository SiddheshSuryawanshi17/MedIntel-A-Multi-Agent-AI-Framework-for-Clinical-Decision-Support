"""Evaluate the Claude writer (FR6, Chapter 7: per-case cost and explanation quality).

Each case is analysed once with the template, then its facts go to Claude under every model and effort setting
being compared. Replies pass the same TC-12 check the server applies. Batch mode uses the Message Batches API at
half price; live mode calls the Messages API to time each case. The summaries are saved for the team's explanation
rubric (1-5)."""

import json
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path
from typing import Any

import anthropic

from medintel.agents.claude_writer import PRICES, ClaudeWriter, case_facts, parse_reply, usage_record
from medintel.agents.context_agent import TemplateWriter, find_conflicts, verify_sentence
from medintel.config import Settings
from medintel.ingestion import DataIngestionService, parse_events
from medintel.models import CaseAnalysis, DraftSentence, PatientCase
from medintel.orchestrator import create_orchestrator
from medintel.training import uci
from medintel.training.fusion_tuning import uci_case
from medintel.training.train import split_cleveland

SAMPLES = Path(str(files("medintel") / "samples"))
SAMPLE_FILES = {"morning_batch.csv": "morning_events.csv", "boundary_tests.csv": None, "chest_pain_clinic.csv": None}


def sample_cases() -> list[PatientCase]:
    cases: list[PatientCase] = []
    for name, events in SAMPLE_FILES.items():
        joined = parse_events((SAMPLES / events).read_text(encoding="utf-8")) if events else None
        cases += DataIngestionService().load_text((SAMPLES / name).read_text(encoding="utf-8"), name, joined).cases
    return cases


def uci_test_cases(data_dir: Path) -> list[PatientCase]:
    return [uci_case(r) for r in split_cleveland(uci.load(data_dir)["cleveland"]).test]


def analyse(settings: Settings, cases: list[PatientCase], knowledge: Any) -> list[CaseAnalysis]:
    orchestrator = create_orchestrator(settings, TemplateWriter(), None, knowledge)
    return [orchestrator.run_case(case) for case in cases]


def check(sentences: list[DraftSentence], a: CaseAnalysis) -> list[DraftSentence]:
    conflicts = find_conflicts(a.case, a.ml, a.rules, a.evidence)
    return [verify_sentence(s, a.case, a.ml, a.rules, a.evidence, conflicts) for s in sentences]


def facts_for(a: CaseAnalysis) -> dict[str, Any]:
    return case_facts(a.case, a.ml, a.rules, a.evidence, find_conflicts(a.case, a.ml, a.rules, a.evidence))


def estimate_tokens(text: str) -> int:
    return len(text) // 3 + 1


def plan(writers: list[ClaudeWriter], analyses: list[CaseAnalysis]) -> dict[str, Any]:
    """Upper-bound cost before anything is sent: input from text length, output at 1,500 tokens per reply."""
    total = 0.0
    for w in writers:
        price_in, price_out, _, _ = PRICES[w.model]
        for a in analyses:
            params = w.request(facts_for(a))
            tokens_in = estimate_tokens(json.dumps(params["system"]) + params["messages"][0]["content"])
            total += (tokens_in * price_in + 1500 * price_out) / 1e6
    return {"requests": len(writers) * len(analyses), "max_cost_usd_standard": round(total, 2)}


def record(writer: ClaudeWriter, a: CaseAnalysis, message: Any, latency: float | None, batch: bool) -> dict[str, Any]:
    usage = usage_record(writer.model, message.usage, latency, getattr(message, "_request_id", None), batch)
    sentences, problem = parse_reply(message)
    checked = check(sentences, a) if sentences else []
    return {
        "writer": writer.name,
        "case_id": a.case.case_id,
        "source": a.case.source,
        "usable": sentences is not None,
        "problem": problem,
        "sentences": [s.model_dump() for s in checked],
        "removed": sum(not s.supported for s in checked),
        "summary": " ".join(s.text for s in checked if s.supported),
        "usage": usage.model_dump(),
    }


def failed(writer: ClaudeWriter, a: CaseAnalysis, problem: str) -> dict[str, Any]:
    return {
        "writer": writer.name,
        "case_id": a.case.case_id,
        "source": a.case.source,
        "usable": False,
        "problem": problem,
        "sentences": [],
        "removed": 0,
        "summary": "",
        "usage": None,
    }


def run_live(writers: list[ClaudeWriter], analyses: list[CaseAnalysis], workers: int = 4) -> list[dict[str, Any]]:
    def one(job: tuple[ClaudeWriter, CaseAnalysis]) -> dict[str, Any]:
        writer, a = job
        started = time.monotonic()
        try:
            message = writer.client.messages.create(**writer.request(facts_for(a)))
        except anthropic.AnthropicError as exc:
            return failed(writer, a, type(exc).__name__)
        return record(writer, a, message, time.monotonic() - started, False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, [(w, a) for w in writers for a in analyses]))


def run_batch(
    client: Any,
    writers: list[ClaudeWriter],
    analyses: list[CaseAnalysis],
    poll_s: float = 30.0,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    jobs = {f"w{i}-{a.case.case_id}": (w, a) for i, w in enumerate(writers) for a in analyses}
    batch = client.messages.batches.create(
        requests=[{"custom_id": key, "params": w.request(facts_for(a))} for key, (w, a) in jobs.items()]
    )
    log(f"Batch {batch.id} submitted with {len(jobs)} requests")
    while batch.processing_status != "ended":
        time.sleep(poll_s)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        log(f"  {batch.processing_status}: {counts.succeeded} succeeded, {counts.processing} processing")
    done: dict[str, dict[str, Any]] = {}
    for result in client.messages.batches.results(batch.id):
        writer, a = jobs[result.custom_id]
        if result.result.type == "succeeded":
            done[result.custom_id] = record(writer, a, result.result.message, None, True)
        else:
            done[result.custom_id] = failed(writer, a, result.result.type)
    return [done.get(key) or failed(w, a, "missing from the batch results") for key, (w, a) in jobs.items()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_writer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_writer.setdefault(row["writer"], []).append(row)
    out: dict[str, Any] = {}
    for name, group in by_writer.items():
        usable = [r for r in group if r["usable"]]
        sentences = sum(len(r["sentences"]) for r in usable)
        removed = sum(r["removed"] for r in usable)
        usages = [r["usage"] for r in group if r["usage"]]
        latencies = [u["latency_s"] for u in usages if u.get("latency_s") is not None]
        out[name] = {
            "cases": len(group),
            "usable_replies": len(usable),
            "fallbacks": len(group) - len(usable),
            "sentences_per_case": round(sentences / len(usable), 2) if usable else None,
            "sentences_removed_by_tc12": removed,
            "removed_share": round(removed / sentences, 3) if sentences else None,
            "cases_with_a_removal": sum(1 for r in usable if r["removed"]),
            "mean_input_tokens": round(statistics.mean(u["input_tokens"] for u in usages)) if usages else None,
            "mean_output_tokens": round(statistics.mean(u["output_tokens"] for u in usages)) if usages else None,
            "cost_usd_total": round(sum(u["cost_usd"] for u in usages), 4),
            "cost_usd_per_case": round(sum(u["cost_usd"] for u in usages) / len(usages), 5) if usages else None,
            "latency_s_mean": round(statistics.mean(latencies), 2) if latencies else None,
            "latency_s_max": round(max(latencies), 2) if latencies else None,
        }
    return out
