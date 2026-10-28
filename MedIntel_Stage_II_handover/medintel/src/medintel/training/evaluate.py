"""Stage II evaluation: the metrics of Chapters 6 and 7 of the Stage I report, computed with the configured agents on
the held-out UCI test split (the validation split where the plan says so) and the bundled sample batches, and written
to reports/evaluation/ as JSON, Markdown and SVG figures.

Four conditions of increasing richness are compared on the same test split (Table 7.1): ML alone; ML with the rule
engine (overrides and the R term); ML with rules and retrieval (adds the K term); and the full pipeline, which adds
the context agent's conflict penalty (lambda * D) and the confidence threshold that sends uncertain cases to a
clinician. The ablation removes one component at a time from the full pipeline. Classification metrics count the
cases the pipeline decides; the share it decides is reported as coverage."""

import csv
import json
import re
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from medintel.agents.context_agent import TemplateWriter
from medintel.config import Settings
from medintel.fusion import FusionSettings
from medintel.models import CaseAnalysis, PatientCase
from medintel.orchestrator import CaseScheduler, Orchestrator, create_orchestrator
from medintel.training import uci
from medintel.training.figures import bars, reliability
from medintel.training.fusion_tuning import choose_threshold, held_weights, search_weights, uci_case
from medintel.training.metrics import ece
from medintel.training.metrics import reliability as reliability_bins
from medintel.training.train import SEED, split_cleveland
from medintel.training.writer_eval import sample_cases

CONDITIONS: dict[str, dict[str, bool]] = {
    "ML only": {"rules": False, "retrieval": False, "conflicts": False, "threshold": False},
    "ML + rules": {"rules": True, "retrieval": False, "conflicts": False, "threshold": False},
    "ML + rules + retrieval": {"rules": True, "retrieval": True, "conflicts": False, "threshold": False},
    "Full pipeline": {"rules": True, "retrieval": True, "conflicts": True, "threshold": True},
}
ABLATIONS: dict[str, dict[str, bool]] = {
    "Full pipeline": {},
    "without ML": {"ml": False},
    "without rules": {"rules": False},
    "without retrieval": {"retrieval": False},
    "without conflict detection": {"conflicts": False},
}
SCHEMA = Path(str(files("medintel") / "resources")) / "decision_report.schema.json"


def case_row(a: CaseAnalysis, label: int | None) -> dict[str, Any]:
    kinds = [c.type for c in a.context.conflict_details]
    return {
        "case_id": a.case.case_id,
        "label": label,
        "p": a.ml.probabilities["high_risk"],
        "prediction": a.ml.prediction,
        "M": a.fusion.component_scores["M"],
        "R": a.fusion.component_scores["R"],
        "K": a.fusion.component_scores["K"],
        "override": a.fusion.override,
        "rule_conflicts": kinds.count("rule_vs_ml"),
        "evidence_conflicts": kinds.count("evidence_vs_ml"),
        "rules": list(a.rules.triggered),
    }


def decide(
    r: dict[str, Any],
    f: FusionSettings,
    ml: bool = True,
    rules: bool = True,
    retrieval: bool = True,
    conflicts: bool = True,
    threshold: bool = True,
) -> tuple[str, float]:
    """(decision, confidence) for one case under a reduced pipeline: Equation (4.2) with the removed terms dropped and
    the remaining weights rescaled to sum to 1. The decision is high_risk, low_risk or abstain."""
    weights = {"M": f.alpha if ml else 0.0, "R": f.beta if rules else 0.0, "K": f.gamma if retrieval else 0.0}
    total = sum(weights.values())
    conf = sum(w * r[k] for k, w in weights.items()) / total if total else 0.0
    if conflicts:
        count = (r["rule_conflicts"] if rules else 0) + (r["evidence_conflicts"] if retrieval else 0)
        conf = max(0.0, conf - f.lam * min(1.0, count / 3))
    override = r["override"] if rules else None
    if override == "withheld":
        return "abstain", conf
    if override == "high_risk":
        return "high_risk", conf
    if not ml or (threshold and conf < f.threshold):
        return "abstain", conf
    return r["prediction"], conf


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based ROC-AUC (Mann-Whitney U); ties count half."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return round(float(wins / (len(pos) * len(neg))), 3)


def brier(conf: np.ndarray, correct: np.ndarray) -> float | None:
    return round(float(np.mean((conf - correct) ** 2)), 3) if len(conf) else None


def score_condition(rows: list[dict[str, Any]], f: FusionSettings, switches: dict[str, bool]) -> dict[str, Any]:
    use_rules, use_retrieval = switches.get("rules", True), switches.get("retrieval", True)
    results = [decide(r, f, **switches) for r in rows]
    labels = np.array([r["label"] for r in rows])
    decided = [(d, r["label"]) for (d, _), r in zip(results, rows, strict=True) if d != "abstain"]
    tp = sum(1 for d, y in decided if d == "high_risk" and y == 1)
    fp = sum(1 for d, y in decided if d == "high_risk" and y == 0)
    tn = sum(1 for d, y in decided if d == "low_risk" and y == 0)
    fn = sum(1 for d, y in decided if d == "low_risk" and y == 1)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    elevated = [use_rules and r["override"] == "high_risk" for r in rows]
    scores = np.array([1.0 if up else r["p"] for up, r in zip(elevated, rows, strict=True)])
    directed = [
        ("high_risk" if up else r["prediction"], conf, r["label"])
        for up, (_, conf), r in zip(elevated, results, rows, strict=True)
        if not (use_rules and r["override"] == "withheld")
    ]
    conf = np.array([c for _, c, _ in directed])
    correct = np.array([float((d == "high_risk") == bool(y)) for d, _, y in directed])
    disagree = [r for r in rows if (use_rules and r["rule_conflicts"]) or (use_retrieval and r["evidence_conflicts"])]
    return {
        "n": len(rows),
        "coverage": round(len(decided) / len(rows), 3),
        "accuracy": round((tp + tn) / len(decided), 3) if decided else None,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "roc_auc": auroc(scores, labels),
        "brier_confidence": brier(conf, correct),
        "ece_confidence": round(ece(np.clip(conf, 0, 1), correct), 3) if len(conf) else None,
        "disagreement_rate": round(len(disagree) / len(rows), 3),
    }


def calibration(rows: list[dict[str, Any]], design: FusionSettings, tuned: FusionSettings) -> dict[str, Any]:
    p = np.array([r["p"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    kept = [r for r in rows if r["override"] != "withheld"]
    direction = ["high_risk" if r["override"] == "high_risk" else r["prediction"] for r in kept]
    correct = np.array([float((d == "high_risk") == bool(r["label"])) for d, r in zip(direction, kept, strict=True)])
    m = np.array([r["M"] for r in kept])
    c_design = np.clip(np.array([decide(r, design)[1] for r in kept]), 0, 1)
    c_tuned = np.clip(np.array([decide(r, tuned)[1] for r in kept]), 0, 1)
    return {
        "ml_probability": {"ece": round(ece(p, labels), 3), "reliability": reliability_bins(p, labels)},
        "ml_confidence": {"ece": round(ece(m, correct), 3), "reliability": reliability_bins(m, correct)},
        "fused_C_design": {"ece": round(ece(c_design, correct), 3), "reliability": reliability_bins(c_design, correct)},
        "fused_C_tuned": {"ece": round(ece(c_tuned, correct), 3), "reliability": reliability_bins(c_tuned, correct)},
    }


def rule_alerts(rows: list[dict[str, Any]], actions: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule_id, action in actions.items():
        fired = [r for r in rows if rule_id in r["rules"]]
        entry: dict[str, Any] = {"action": action, "fired": len(fired)}
        if action == "elevate":
            false = sum(1 for r in fired if r["label"] == 0)
            entry["false_alerts"] = false
            entry["false_alert_rate"] = round(false / len(fired), 3) if fired else None
        else:
            entry["note"] = "data-completeness or scope flag, not a risk alert" if action == "flag" else "scope rule"
        out[rule_id] = entry
    return out


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "median_ms": round(statistics.median(ordered) * 1000, 2),
        "p95_ms": round(p95 * 1000, 2),
        "max_ms": round(ordered[-1] * 1000, 2),
    }


def latency(
    settings: Settings, knowledge_factory: Any, analyses: list[CaseAnalysis], batch: list[PatientCase]
) -> dict[str, Any]:
    stages = ("ml", "rules", "retrieval", "context", "fusion", "total")
    per_stage = {s: percentiles([a.timings[s] for a in analyses if s in a.timings]) for s in stages}
    throughput: dict[str, float] = {}
    for workers in (1, 2, 4, 8):
        orchestrator = create_orchestrator(settings, TemplateWriter(), None, knowledge_factory())
        orchestrator.run_case(batch[0])  # load the embedding model first, so only steady-state work is timed
        scheduler = CaseScheduler(orchestrator)
        scheduler.submit(batch)
        started = time.perf_counter()
        scheduler.run(max_workers=workers)
        throughput[str(workers)] = round(len(batch) / ((time.perf_counter() - started) / 60), 1)
    return {"stages": per_stage, "throughput_cases_per_minute": throughput}


def cold_start(settings: Settings, knowledge_factory: Any, case: PatientCase) -> dict[str, float]:
    started = time.perf_counter()
    orchestrator = create_orchestrator(settings, TemplateWriter(), None, knowledge_factory())
    built = time.perf_counter()
    analysis = orchestrator.run_case(case)
    return {
        "build_s": round(built - started, 3),
        "first_case_s": round(time.perf_counter() - built, 3),
        "first_case_retrieval_s": analysis.timings["retrieval"],
    }


def completeness(analyses: list[CaseAnalysis]) -> dict[str, Any]:
    import jsonschema

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    passed = 0
    for a in analyses:
        try:
            jsonschema.validate(a.report.model_dump(), schema)
            passed += 1
        except jsonschema.ValidationError:
            pass
    sentences = [s for a in analyses for s in a.context.draft]
    kept = [s for s in sentences if s.supported]
    return {
        "reports": len(analyses),
        "schema_pass_rate": round(passed / len(analyses), 3) if analyses else None,
        "summary_sentences": len(sentences),
        "removed_by_tc12": len(sentences) - len(kept),
        "kept_with_a_source": round(sum(1 for s in kept if s.sources) / len(kept), 3) if kept else None,
        "writers": sorted({a.context.generated_by for a in analyses}),
    }


def evidence_check(analyses: list[CaseAnalysis], known_ids: set[str]) -> dict[str, Any]:
    cited = [e.source_id for a in analyses for e in a.evidence]
    return {
        "cited": len(cited),
        "valid_source_ids": round(sum(1 for c in cited if c in known_ids) / len(cited), 3) if cited else None,
        "cases_without_evidence": sum(1 for a in analyses if not a.evidence),
    }


def precision_at_5(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    per_case: dict[str, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            value = (row.get("relevant") or "").strip()
            if value in ("0", "1") and int(row.get("rank") or 99) <= 5:
                per_case.setdefault(row["case_id"], []).append(int(value))
    if not per_case:
        return {"status": "labels pending", "file": str(path)}
    scores = [sum(v) / len(v) for v in per_case.values()]
    return {
        "status": "labelled",
        "cases": len(per_case),
        "labelled_passages": sum(len(v) for v in per_case.values()),
        "precision_at_5": round(statistics.mean(scores), 3),
    }


def rubric(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    criteria = ("completeness", "factual_consistency", "usefulness")
    scores: dict[str, list[float]] = {c: [] for c in criteria}
    seconds: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for c in criteria:
                if (row.get(c) or "").strip():
                    scores[c].append(float(row[c]))
            if (row.get("reading_seconds") or "").strip():
                seconds.append(float(row["reading_seconds"]))
    if not any(scores.values()):
        return {"status": "scores pending", "file": str(path)}
    return {
        "status": "scored",
        "means": {c: round(statistics.mean(v), 2) for c, v in scores.items() if v},
        "reports_scored": max(len(v) for v in scores.values()),
        "median_reading_seconds": round(statistics.median(seconds), 1) if seconds else None,
    }


def planned_cases(project: Path, out_dir: Path) -> dict[str, str] | None:
    tests = project / "tests" / "test_planned_cases.py"
    if not tests.is_file():
        return None
    report = out_dir / "planned_cases.xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", str(tests), "-q", f"--junitxml={report}", "-p", "no:cacheprovider"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    out: dict[str, str] = {}
    for tc in ET.parse(report).getroot().iter("testcase"):
        name = tc.get("name", "")
        if name.startswith("test_tc"):
            failed = tc.find("failure") is not None or tc.find("error") is not None
            out[f"TC-{name[7:9]}"] = "fail" if failed else "pass"
    return dict(sorted(out.items()))


def tuned_settings(rows: list[dict[str, Any]], design: FusionSettings, target: float = 0.85) -> FusionSettings:
    """The same tuning as `medintel tune-fusion`, on the validation rows."""
    component_rows = [
        {
            "M": r["M"],
            "R": r["R"],
            "K": r["K"],
            "D": min(1.0, (r["rule_conflicts"] + r["evidence_conflicts"]) / 3),
            "rule_conflict": int(bool(r["rule_conflicts"])),
            "correct": None
            if r["override"] == "withheld"
            else int(((r["override"] or r["prediction"]) == "high_risk") == bool(r["label"])),
        }
        for r in rows
    ]
    weights, _ = search_weights(component_rows, design, held_weights(component_rows, design))
    return weights.model_copy(update={"threshold": choose_threshold(component_rows, weights, target)})


def run(orchestrator: Orchestrator, cases: list[PatientCase]) -> list[CaseAnalysis]:
    return [orchestrator.run_case(c) for c in cases]


def known_source_ids(store: Any) -> set[str]:
    return {p.source_id for p in getattr(store, "passages", ())}


def read_report(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def live_report(writer_report: Path) -> Path:
    return writer_report.with_name(f"{writer_report.stem}_live.json")


def evaluate(
    settings: Settings,
    data_dir: Path,
    knowledge_factory: Any,
    out_dir: Path,
    project: Path,
    labels_dir: Path,
    writer_report: Path,
) -> dict[str, Any]:
    if not settings.ml.model or not Path(settings.ml.model).is_file():
        raise ValueError(
            "no trained risk model is configured. Run: uv run medintel fetch-data && uv run medintel train"
        )
    samples = sample_cases()
    cold = cold_start(settings, knowledge_factory, samples[0])
    split = split_cleveland(uci.load(data_dir)["cleveland"], SEED)
    store = knowledge_factory()
    orchestrator = create_orchestrator(settings, TemplateWriter(), None, store)
    test_cases = [uci_case(r) for r in split.test]
    val = run(orchestrator, [uci_case(r) for r in split.validation])
    test = run(orchestrator, test_cases)
    sample = run(orchestrator, samples)
    val_rows = [case_row(a, r.label) for a, r in zip(val, split.validation, strict=True)]
    test_rows = [case_row(a, r.label) for a, r in zip(test, split.test, strict=True)]
    design = settings.fusion
    tuned = tuned_settings(val_rows, design)
    actions = {c.rule.rule_id: c.rule.action for c in test[0].rules.checks}
    versions = orchestrator.versions()
    result: dict[str, Any] = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "configuration": {
            "model": versions.model,
            "rulebase": versions.rulebase,
            "index": getattr(store, "version", "demo"),
            "retrieval": getattr(store, "method", "demo knowledge base"),
            "summary_writer": "template (deterministic; Claude summaries are evaluated by eval-writer)",
            "design_fusion": design.model_dump(),
            "tuned_fusion": tuned.model_dump(),
            "split": f"UCI Cleveland, stratified 60/20/20, seed {SEED}: {len(val)} validation, {len(test)} test",
        },
        "conditions": {
            name: {"design": score_condition(test_rows, design, s), "tuned": score_condition(test_rows, tuned, s)}
            for name, s in CONDITIONS.items()
        },
        "ablation": {name: score_condition(test_rows, design, s) for name, s in ABLATIONS.items()},
        "calibration": calibration(test_rows, design, tuned),
        "rule_alerts_validation": rule_alerts(val_rows, actions),
        "latency": {"cold_start": cold, **latency(settings, knowledge_factory, val + test + sample, test_cases)},
        "completeness": completeness(val + test + sample),
        "evidence": evidence_check(val + test + sample, known_source_ids(store)),
        "precision_at_5": precision_at_5(labels_dir / "evidence_relevance.csv"),
        "explanation_rubric": rubric(labels_dir / "explanation_rubric.csv"),
        "summary_writer": read_report(writer_report),
        "summary_writer_live": read_report(live_report(writer_report)),
        "planned_cases": planned_cases(project, out_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_figures(result, out_dir / "figures")
    (out_dir / "EVALUATION.md").write_text(markdown(result), encoding="utf-8")
    return result


def write_figures(r: dict[str, Any], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    cal = r["calibration"]
    (folder / "reliability_ml.svg").write_text(
        reliability(
            "Risk model calibration (test split)",
            [("ML probability of high risk", cal["ml_probability"]["reliability"])],
            "Mean predicted probability",
            "Observed rate of disease",
        ),
        encoding="utf-8",
    )
    (folder / "reliability_confidence.svg").write_text(
        reliability(
            "Does the confidence mean what it says? (test split)",
            [
                ("ML confidence M", cal["ml_confidence"]["reliability"]),
                ("Fused C, design weights", cal["fused_C_design"]["reliability"]),
                ("Fused C, tuned weights", cal["fused_C_tuned"]["reliability"]),
            ],
            "Mean confidence (hollow: bins with fewer than 3 cases)",
            "Share of decisions that were right",
            x_min=0.4,
        ),
        encoding="utf-8",
    )
    names = list(r["conditions"])
    (folder / "conditions.svg").write_text(
        bars(
            "Four conditions on the test split (weights tuned on validation)",
            ["ML only", "+ rules", "+ retrieval", "Full"],
            [
                ("Accuracy of decided cases", [r["conditions"][n]["tuned"]["accuracy"] for n in names]),
                ("Coverage (share decided)", [r["conditions"][n]["tuned"]["coverage"] for n in names]),
                ("F1", [r["conditions"][n]["tuned"]["f1"] for n in names]),
            ],
            "Score (0 to 1)",
        ),
        encoding="utf-8",
    )
    ablations = list(r["ablation"])
    (folder / "ablation.svg").write_text(
        bars(
            "Removing one component at a time (test split)",
            ["Full", "no ML", "no rules", "no retrieval", "no conflicts"],
            [
                ("Accuracy of decided cases", [r["ablation"][n]["accuracy"] for n in ablations]),
                ("Coverage (share decided)", [r["ablation"][n]["coverage"] for n in ablations]),
            ],
            "Score (0 to 1)",
        ),
        encoding="utf-8",
    )


def _v(value: Any, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.0%}" if pct else f"{value:.3f}"
    return str(value)


def _met(ok: bool | None) -> str:
    return "pending" if ok is None else ("yes" if ok else "no")


def _weights(f: dict[str, Any]) -> str:
    return f"alpha {f['alpha']}, beta {f['beta']}, gamma {f['gamma']}, lambda {f['lam']}, threshold {f['threshold']}"


def _alerts(alerts: dict[str, Any]) -> str:
    parts = []
    for rule_id, a in alerts.items():
        if "false_alerts" in a:
            parts.append(f"{rule_id} {a['false_alerts']} of {a['fired']}")
        else:
            parts.append(f"{rule_id} fired {a['fired']} ({'flag' if a['action'] == 'flag' else 'scope rule'})")
    return "; ".join(parts)


def _p5(p5: dict[str, Any] | None) -> str:
    if not p5 or p5.get("status") != "labelled":
        return "pending the team's labels"
    return f"{p5['precision_at_5']:.2f} over {p5['cases']} cases"


def _mode(report: dict[str, Any]) -> str:
    return "Batch API" if report.get("mode") == "batch" else "live"


WRITER_NAME = re.compile(r"(\S+) \(effort (\w+), prompt (\w+)\)")


def _writer_label(name: str) -> str:
    m = WRITER_NAME.fullmatch(name)
    return f"{m[1]}, effort {m[2]}" if m else name


def _cost(writer: dict[str, Any] | None, live: dict[str, Any] | None) -> str:
    parts, labels = [], []
    for report in (live, writer):
        if report:
            name, s = next(iter(report["results"].items()))
            if s["cost_usd_per_case"] is not None:
                how = "with the Batch API" if report.get("mode") == "batch" else "live"
                parts.append(f"${s['cost_usd_per_case']:.3f} per case {how}")
                labels.append(_writer_label(name))
    if not parts:
        return "template: no API cost; Claude pending an API key"
    return f"about {' and '.join(parts)} ({'; '.join(dict.fromkeys(labels))})"


def _live(live: dict[str, Any] | None) -> dict[str, Any] | None:
    if not live:
        return None
    s = next(iter(live["results"].values()))
    return s if s.get("latency_s_max") is not None else None


def _rubric(rub: dict[str, Any] | None) -> str:
    if not rub or rub.get("status") != "scored":
        return "pending reviewer scores"
    return ", ".join(f"{k} {v}" for k, v in rub["means"].items())


def summary_rows(r: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    c = r["conditions"]
    full, ml = c["Full pipeline"]["tuned"], c["ML only"]["tuned"]
    full_d = c["Full pipeline"]["design"]
    cal, lat, comp = r["calibration"], r["latency"], r["completeness"]
    p5, rub, writer = r["precision_at_5"], r["explanation_rubric"], r["summary_writer"]
    live = r.get("summary_writer_live")
    claude = _live(live)
    api_max = claude["latency_s_max"] if claude else 0.0
    claude_latency = (
        f"; the Claude writer adds {claude['latency_s_mean']} s on average, {api_max} s at most "
        f"({claude['cases']} live cases)"
        if claude
        else ""
    )
    tcs = r["planned_cases"] or {}
    worst = lat["stages"]["total"]["max_ms"] / 1000
    ml_ms, rules_ms = lat["stages"]["ml"]["max_ms"], lat["stages"]["rules"]["max_ms"]
    ece_c, ece_m = cal["fused_C_tuned"]["ece"], cal["ml_confidence"]["ece"]
    labelled = bool(p5 and p5.get("status") == "labelled")
    valid_ids = r["evidence"]["valid_source_ids"]
    rubric_ok = all(v >= 4 for v in rub["means"].values()) if rub and rub.get("status") == "scored" else None
    passed = sum(v == "pass" for v in tcs.values())
    f1_ok = full["f1"] is not None and ml["f1"] is not None and full["f1"] > ml["f1"]
    auc_ok = full["roc_auc"] is not None and ml["roc_auc"] is not None and full["roc_auc"] >= ml["roc_auc"]
    disagreement = ", ".join(f"{k}: {_v(v['design']['disagreement_rate'], True)}" for k, v in c.items())
    return [
        (
            "F1",
            "Full pipeline above ML only",
            f"{_v(full['f1'])} vs {_v(ml['f1'])}; the full pipeline decides {_v(full['coverage'], True)} "
            f"(design weights: {_v(full_d['f1'])}, {_v(full_d['coverage'], True)})",
            _met(f1_ok),
        ),
        ("ROC-AUC", "Full pipeline not below ML only", f"{_v(full['roc_auc'])} vs {_v(ml['roc_auc'])}", _met(auc_ok)),
        (
            "Calibration error",
            "Fused C no worse than the ML confidence",
            f"ECE {_v(ece_c)} vs {_v(ece_m)} (design weights: {_v(cal['fused_C_design']['ece'])})",
            _met(ece_c <= ece_m),
        ),
        (
            "Evidence precision at 5",
            "Reported; every cited passage has a valid source_id",
            f"{_p5(p5)}; valid IDs {_v(valid_ids, True)}",
            _met(valid_ids == 1.0 if labelled else None),
        ),
        ("Disagreement rate", "Reported for each condition", disagreement, "yes"),
        ("Rule false-alert rate", "Reported per rule (validation split)", _alerts(r["rule_alerts_validation"]), "yes"),
        (
            "Report completeness",
            "100% pass the schema",
            f"{_v(comp['schema_pass_rate'], True)} of {comp['reports']} reports",
            _met(comp["schema_pass_rate"] == 1.0),
        ),
        (
            "End-to-end latency",
            "30 s per case; ML and rules under 1 s",
            f"worst case {worst:.2f} s with the template writer{claude_latency}; "
            f"ML max {ml_ms:.1f} ms, rules max {rules_ms:.1f} ms",
            _met(worst + api_max < 30 and ml_ms < 1000 and rules_ms < 1000),
        ),
        ("Per-case cost", "Reported", _cost(writer, live), "yes" if writer or live else "pending"),
        ("Explanation rubric", "Mean of at least 4 per criterion", _rubric(rub), _met(rubric_ok)),
        (
            "Planned test cases",
            "All twelve pass",
            f"{passed} of {len(tcs) or 12} pass",
            _met(len(tcs) == 12 and passed == 12 if tcs else None),
        ),
    ]


def markdown(r: dict[str, Any]) -> str:
    cfg, cal, lat, comp = r["configuration"], r["calibration"], r["latency"], r["completeness"]
    lines = [
        "# MedIntel Stage II evaluation",
        "",
        f"Generated by `uv run medintel evaluate` on {r['created'][:10]}. Research prototype on public de-identified "
        "data (UCI Heart Disease) and synthetic sample batches; not a clinical study.",
        "",
        "## Configuration",
        "",
        f"- Risk model: {cfg['model']}",
        f"- Rulebase {cfg['rulebase']}; knowledge index: {cfg['index']}",
        f"- Retrieval: {cfg['retrieval']}",
        f"- Summaries: {cfg['summary_writer']}",
        f"- Data: {cfg['split']}; plus the 24 bundled sample cases for latency and completeness",
        f"- Fusion, design weights: {_weights(cfg['design_fusion'])}",
        f"- Fusion, tuned on the validation split: {_weights(cfg['tuned_fusion'])}",
        "",
        "## Results against the targets of Table 7.1",
        "",
        "As the plan requires, the fusion weights and threshold were tuned on the validation split only and are judged "
        "here on the test split; results with the illustrative design weights are given in brackets.",
        "",
        "| Metric | Target | Result | Met |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {d} |" for a, b, c, d in summary_rows(r)],
        "",
        "## 1. Planned test cases (Table 6.4)",
        "",
        "| Test case | Result |",
        "|---|---|",
        *([f"| {k} | {v} |" for k, v in (r["planned_cases"] or {}).items()] or ["| (tests folder not found) | n/a |"]),
        "",
        "Each is a named test in `tests/test_planned_cases.py`; the docstrings quote the plan.",
        "",
        "## 2. Four conditions on the test split (Table 7.1)",
        "",
        "Classification metrics count the cases a condition decides; coverage is the share it decides. Only the full "
        "pipeline abstains (uncertain cases go to a clinician), so its accuracy and F1 cover fewer cases.",
        "",
        "| Condition | Weights | Coverage | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC "
        "| Brier (C) | ECE (C) | Disagreement |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, both in r["conditions"].items():
        for which in ("design", "tuned"):
            s = both[which]
            lines.append(
                f"| {name} | {which} | {_v(s['coverage'], True)} | {_v(s['accuracy'])} | {_v(s['precision'])} | "
                f"{_v(s['recall'])} | {_v(s['specificity'])} | {_v(s['f1'])} | {_v(s['roc_auc'])} | "
                f"{_v(s['brier_confidence'])} | {_v(s['ece_confidence'])} | {_v(s['disagreement_rate'], True)} |"
            )
    lines += [
        "",
        "![Four conditions](figures/conditions.svg)",
        "",
        "## 3. Component ablation (design weights, test split)",
        "",
        "| Pipeline | Coverage | Accuracy | F1 | Brier (C) | ECE (C) |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in r["ablation"].items():
        lines.append(
            f"| {name} | {_v(s['coverage'], True)} | {_v(s['accuracy'])} | {_v(s['f1'])} | "
            f"{_v(s['brier_confidence'])} | {_v(s['ece_confidence'])} |"
        )
    bins = cal["ml_probability"]["reliability"]
    lines += [
        "",
        "![Ablation](figures/ablation.svg)",
        "",
        "## 4. Calibration (test split)",
        "",
        f"- ML probability of high risk against the outcome: ECE {_v(cal['ml_probability']['ece'])}.",
        "- Confidence in the decision against whether the decision was right: ML confidence M "
        f"{_v(cal['ml_confidence']['ece'])}, fused C with design weights {_v(cal['fused_C_design']['ece'])}, "
        f"with tuned weights {_v(cal['fused_C_tuned']['ece'])}.",
        "",
        "![ML calibration](figures/reliability_ml.svg)",
        "",
        "![Confidence calibration](figures/reliability_confidence.svg)",
        "",
        "| Bin | n | Mean predicted | Observed |",
        "|---|---|---|---|",
        *[f"| {b['bin']} | {b['n']} | {b['mean_predicted']} | {b['observed']} |" for b in bins],
        "",
        "## 5. Rule alerts (validation split)",
        "",
        "| Rule | Action | Fired | False-alert rate | Note |",
        "|---|---|---|---|---|",
    ]
    for rule_id, a in r["rule_alerts_validation"].items():
        rate = _v(a.get("false_alert_rate"), True) if "false_alerts" in a else "n/a"
        lines.append(f"| {rule_id} | {a['action']} | {a['fired']} | {rate} | {a.get('note', '')} |")
    lines += [
        "",
        "UCI records carry no creatinine, resting heart rate or patients under 18, so R-002 flags every case and R-003 "
        "and R-004 never fire here; the planned test cases and `boundary_tests.csv` check every rule at its boundary.",
        "",
        "## 6. Latency and throughput (Section 6.5.1, TC-11)",
        "",
        f"Cold start: building the pipeline took {lat['cold_start']['build_s']} s and the first case "
        f"{lat['cold_start']['first_case_s']} s (loading the embedding model and building the index).",
        "",
        "| Stage | Median ms | p95 ms | Max ms |",
        "|---|---|---|---|",
        *[f"| {s} | {v['median_ms']} | {v['p95_ms']} | {v['max_ms']} |" for s, v in lat["stages"].items()],
        "",
        "| Parallel workers | Cases per minute |",
        "|---|---|",
        *[f"| {w} | {v} |" for w, v in lat["throughput_cases_per_minute"].items()],
        "",
        latency_note(r.get("summary_writer_live")),
        "",
        "## 7. Report completeness and statement tracing (Section 6.5.3)",
        "",
        f"- {comp['reports']} reports checked against the DecisionReport schema: "
        f"{_v(comp['schema_pass_rate'], True)} pass.",
        f"- {comp['summary_sentences']} summary sentences; {comp['removed_by_tc12']} removed by the TC-12 check; "
        f"{_v(comp['kept_with_a_source'], True)} of the kept sentences cite a source. "
        f"Writers: {', '.join(comp['writers'])}.",
        f"- {r['evidence']['cited']} evidence citations, {_v(r['evidence']['valid_source_ids'], True)} with a valid "
        f"source_id; {r['evidence']['cases_without_evidence']} cases had no evidence above the threshold (K = 0).",
        "",
        "## 8. Evidence precision at 5",
        "",
        precision_text(r["precision_at_5"]),
        "",
        "## 9. Summary writer: cost, latency and removed statements",
        "",
        writer_text(r["summary_writer"], r.get("summary_writer_live")),
        "",
        "## 10. Explanation rubric (Section 7.2)",
        "",
        rubric_text(r["explanation_rubric"]),
        "",
        "## Reading these results",
        "",
        "- The test split has 61 patients, so differences of a few points are within noise.",
        "- UCI patients rarely trigger a rule override or a conflict, so the rule engine's value shows in the safety "
        "and boundary tests rather than in these averages; the fusion weights for R and D could not be tuned on this "
        "data (see reports/fusion_tuning.json).",
        "- Coverage and accuracy trade off through the threshold; the team chooses the operating point.",
        "",
    ]
    return "\n".join(lines)


def precision_text(p5: dict[str, Any] | None) -> str:
    if not p5 or p5.get("status") != "labelled":
        return (
            "Pending: run `uv run medintel export-labels`, mark each retrieved passage in "
            "`reports/labeling/evidence_relevance.csv` as relevant (1) or not (0), and run `medintel evaluate` again."
        )
    return (
        f"Precision at 5: {p5['precision_at_5']:.2f}, from {p5['labelled_passages']} labelled passages over "
        f"{p5['cases']} cases."
    )


def latency_note(live: dict[str, Any] | None) -> str:
    claude = _live(live)
    if not claude:
        return (
            "Times are for the template summary writer on this machine; a Claude-written summary adds the API call "
            "(section 9)."
        )
    return (
        "Times are for the template summary writer on this machine. A Claude-written summary adds the API call: "
        f"{claude['latency_s_mean']} s on average and {claude['latency_s_max']} s at most over {claude['cases']} live "
        "cases (section 9)."
    )


def writer_text(writer: dict[str, Any] | None, live: dict[str, Any] | None = None) -> str:
    if not writer and not live:
        return (
            "Pending: with ANTHROPIC_API_KEY in .env, run `uv run medintel eval-writer --cases all "
            "--efforts low,medium --yes` (Batch API, half price) and `uv run medintel eval-writer --mode live --yes` "
            "(latency), then `medintel evaluate` again."
        )
    results = [(report, name, s) for report in (writer, live) if report for name, s in report["results"].items()]
    parsed = [WRITER_NAME.fullmatch(name) for _, name, _ in results]
    shared = all(parsed) and len({(m[1], m[3]) for m in parsed if m}) == 1
    rows = [f"All rows use {parsed[0][1]} with prompt version {parsed[0][3]}.", ""] if shared and parsed[0] else []
    rows += [
        f"| {'Effort' if shared else 'Writer'} | Mode | Cases | Fallbacks | Removed sentences | Cost per case | "
        "Mean latency s | Max latency s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (report, name, s), m in zip(results, parsed, strict=True):
        cost = f"${s['cost_usd_per_case']}" if s["cost_usd_per_case"] is not None else "n/a"
        mean, worst = s["latency_s_mean"] or "n/a", s["latency_s_max"] or "n/a"
        rows.append(
            f"| {m[2] if shared and m else name} | {_mode(report)} | {s['cases']} | {s['fallbacks']} | "
            f"{s['sentences_removed_by_tc12']} | {cost} | {mean} | {worst} |"
        )
    rows += [
        "",
        "Batch API rows are billed at half the standard price and carry no per-case latency; live rows call the "
        "Messages API as the server does, four cases at a time. Removed sentences are those the TC-12 check took out "
        "before the report was written.",
    ]
    if not live:
        rows.append("Live latency pending: run `uv run medintel eval-writer --mode live --yes`.")
    if not writer:
        rows.append(
            "Effort comparison pending: run `uv run medintel eval-writer --cases all --efforts low,medium --yes`."
        )
    return "\n".join(rows)


def rubric_text(rub: dict[str, Any] | None) -> str:
    if not rub or rub.get("status") != "scored":
        return (
            "Pending: reviewers score each report in `reports/labeling/explanation_rubric.csv` from 1 to 5 for "
            "completeness, factual consistency and usefulness, and time their reading (target: two minutes, NFR7)."
        )
    means = ", ".join(f"{k} {v}" for k, v in rub["means"].items())
    reading = rub["median_reading_seconds"]
    return f"{rub['reports_scored']} reports scored. Means: {means}. Median reading time: {reading} s."
