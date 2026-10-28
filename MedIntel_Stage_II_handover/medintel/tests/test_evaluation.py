"""The evaluation harness (Chapters 6 and 7): decision rules for each condition, metrics, label sheets, and a full run
on a synthetic UCI-format dataset so the report and figures are produced without the real download."""

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from medintel.agents.retrieval_agent import DemoKnowledgeStore
from medintel.agents.risk_model import MODEL_FEATURES, PRE_TEST_FEATURES, RiskModelSet
from medintel.config import MLSettings, Settings
from medintel.fusion import FusionSettings
from medintel.training import uci
from medintel.training.evaluate import (
    auroc,
    decide,
    evaluate,
    precision_at_5,
    rubric,
    score_condition,
    summary_rows,
    writer_text,
)
from medintel.training.labels import rubric_sample, write_sheet
from medintel.training.train import fit_submodel

F = FusionSettings()


def row(
    p: float, label: int, override: str | None = None, rule_conflicts: int = 0, evidence_conflicts: int = 0
) -> dict:
    prediction = "high_risk" if p >= 0.5 else "low_risk"
    r = 0.0 if override == "high_risk" and prediction != "high_risk" else 1.0
    return {
        "case_id": "C-1",
        "label": label,
        "p": p,
        "prediction": prediction,
        "M": max(p, 1 - p),
        "R": r,
        "K": 0.6,
        "override": override,
        "rule_conflicts": rule_conflicts,
        "evidence_conflicts": evidence_conflicts,
        "rules": ["R-001"] if override == "high_risk" else [],
    }


def test_decisions_follow_the_condition() -> None:
    sure_low = row(0.05, 0)
    assert decide(sure_low, F) == ("low_risk", pytest.approx(0.5 * 0.95 + 0.3 + 0.2 * 0.6))
    elevated = row(0.2, 1, override="high_risk", rule_conflicts=1)
    assert decide(elevated, F)[0] == "high_risk", "a rule override always decides"
    assert decide(elevated, F, rules=False)[0] == "low_risk"
    assert decide(row(0.1, 0, override="withheld"), F)[0] == "abstain"
    borderline = row(0.52, 1, evidence_conflicts=2)
    assert decide(borderline, F)[0] == "abstain", "low confidence goes to a clinician"
    assert decide(borderline, F, threshold=False)[0] == "high_risk"
    assert decide(sure_low, F, ml=False)[0] == "abstain", "without ML only overrides decide"


def test_condition_metrics() -> None:
    rows = [row(0.9, 1), row(0.8, 1), row(0.3, 0), row(0.2, 0), row(0.6, 0), row(0.45, 1, override="high_risk")]
    s = score_condition(rows, F, {"rules": True, "retrieval": True, "conflicts": True, "threshold": False})
    assert s["coverage"] == 1.0 and s["n"] == 6
    assert s["accuracy"] == pytest.approx(5 / 6, abs=0.001)
    assert s["recall"] == 1.0 and s["precision"] == pytest.approx(0.75)
    assert s["disagreement_rate"] == 0.0


def test_auroc_matches_sklearn() -> None:
    rng = np.random.default_rng(3)
    scores, labels = rng.random(80), rng.integers(0, 2, 80)
    assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=0.001)
    assert auroc(np.array([0.1, 0.2]), np.array([1, 1])) is None


def test_label_sheets_round_trip(tmp_path: Path) -> None:
    rel = tmp_path / "evidence_relevance.csv"
    assert precision_at_5(rel) is None
    with rel.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "rank", "relevant"])
        w.writeheader()
        w.writerows(
            [
                {"case_id": "C-1", "rank": 1, "relevant": "1"},
                {"case_id": "C-1", "rank": 2, "relevant": "0"},
                {"case_id": "C-2", "rank": 1, "relevant": "1"},
                {"case_id": "C-2", "rank": 6, "relevant": "0"},
            ]
        )
    assert precision_at_5(rel) == {"status": "labelled", "cases": 2, "labelled_passages": 3, "precision_at_5": 0.75}
    sheet = tmp_path / "explanation_rubric.csv"
    cols = ("case_id", "completeness", "factual_consistency", "usefulness", "reading_seconds")
    assert write_sheet(sheet, cols, [{"case_id": "C-1"}], force=False)
    assert rubric(sheet) == {"status": "scores pending", "file": str(sheet)}
    assert not write_sheet(sheet, cols, [], force=False), "filled sheets are never overwritten silently"
    write_sheet(
        sheet,
        cols,
        [{"case_id": "C-1", "completeness": 5, "factual_consistency": 4, "usefulness": 4, "reading_seconds": 75}],
        force=True,
    )
    assert rubric(sheet)["means"] == {"completeness": 5.0, "factual_consistency": 4.0, "usefulness": 4.0}


def write_fake_uci(folder: Path, n: int = 100) -> None:
    rng = random.Random(11)
    folder.mkdir(parents=True)
    for site, filename in uci.SITES.items():
        lines = []
        for _ in range(n if site == "cleveland" else 30):
            age = rng.randint(30, 75)
            sex = rng.choice([0, 1])
            cp = rng.choice([1, 2, 3, 4])
            st = round(rng.uniform(0, 4), 1)
            ca = rng.choice([0, 0, 1, 2, 3])
            thal = rng.choice([3, 3, 6, 7])
            disease = (0.04 * (age - 55) + 0.8 * sex + 0.6 * st + 0.5 * ca + (cp == 4) + rng.gauss(0, 1)) > 1.3
            values = [
                age,
                sex,
                cp,
                rng.randint(100, 185),
                rng.randint(150, 320),
                rng.choice([0, 1]),
                rng.choice([0, 1, 2]),
                rng.randint(90, 195),
                rng.choice([0, 1]),
                st,
                rng.choice([1, 2, 3]),
                ca,
                thal,
                int(disease) * 2,
            ]
            lines.append(",".join(str(v) for v in values))
        (folder / filename).write_text("\n".join(lines) + "\n")


def trained_settings(data: Path, folder: Path) -> Settings:
    records = uci.load(data)["cleveland"]
    model = RiskModelSet(
        {
            "pre_test": fit_submodel(records, PRE_TEST_FEATURES, ("logistic_regression",), 1)[0],
            "full": fit_submodel(records, MODEL_FEATURES, ("logistic_regression",), 1)[0],
        },
        "risk-lr-synthetic",
        "0.0.1",
    )
    path = folder / "model.json"
    path.write_text(json.dumps(model.to_dict()))
    return Settings(ml=MLSettings(model=str(path)))


def test_evaluate_needs_a_trained_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="medintel train"):
        evaluate(Settings(), tmp_path, DemoKnowledgeStore, tmp_path, tmp_path, tmp_path, tmp_path / "x.json")


def test_full_evaluation_run_on_synthetic_data(tmp_path: Path) -> None:
    data = tmp_path / "uci"
    write_fake_uci(data)
    out = tmp_path / "evaluation"
    result: dict[str, Any] = evaluate(
        trained_settings(data, tmp_path), data, DemoKnowledgeStore, out, tmp_path, tmp_path / "labels", tmp_path / "x"
    )
    assert (out / "EVALUATION.md").is_file() and (out / "evaluation.json").is_file()
    assert {p.name for p in (out / "figures").iterdir()} == {
        "reliability_ml.svg",
        "reliability_confidence.svg",
        "conditions.svg",
        "ablation.svg",
    }
    assert set(result["conditions"]) == {"ML only", "ML + rules", "ML + rules + retrieval", "Full pipeline"}
    assert result["completeness"]["schema_pass_rate"] == 1.0
    assert result["planned_cases"] is None and result["precision_at_5"] is None
    assert set(result["latency"]["throughput_cases_per_minute"]) == {"1", "2", "4", "8"}
    markdown = (out / "EVALUATION.md").read_text()
    for heading in ("## Results against the targets of Table 7.1", "## 3. Component ablation", "## 6. Latency"):
        assert heading in markdown
    assert "—" not in markdown and "–" not in markdown
    assert [m for m, *_ in summary_rows(result)][:3] == ["F1", "ROC-AUC", "Calibration error"]
    json.loads((out / "evaluation.json").read_text())

    result["summary_writer_live"] = {"mode": "live", "results": {"opus (low)": writer_stats(9.5, 31.0)}}
    latency_row = next(row for row in summary_rows(result) if row[0] == "End-to-end latency")
    assert "31.0 s at most (4 live cases)" in latency_row[2]
    assert latency_row[3] == "no", "the 30-second target counts the API call"
    result["summary_writer"] = {
        "mode": "batch",
        "results": {"claude-opus-5-5 (effort low, prompt 1)": writer_stats(None, None)},
    }
    cost_row = next(row for row in summary_rows(result) if row[0] == "Per-case cost")
    assert (
        cost_row[2]
        == "about $0.020 per case live and $0.020 per case with the Batch API (opus (low); claude-opus-5-5, effort low)"
    )


def writer_stats(mean: float | None, worst: float | None) -> dict[str, Any]:
    return {
        "cases": 4,
        "fallbacks": 0,
        "sentences_removed_by_tc12": 1,
        "cost_usd_per_case": 0.02,
        "latency_s_mean": mean,
        "latency_s_max": worst,
    }


def test_writer_section_shows_batch_and_live_runs() -> None:
    batch = {"mode": "batch", "results": {"opus (low)": writer_stats(None, None)}}
    live = {"mode": "live", "results": {"opus (low)": writer_stats(9.5, 14.2)}}
    both = writer_text(batch, live)
    assert "| opus (low) | Batch API | 4 | 0 | 1 | $0.02 | n/a | n/a |" in both
    assert "| opus (low) | live | 4 | 0 | 1 | $0.02 | 9.5 | 14.2 |" in both
    assert "pending" not in both
    assert "Live latency pending" in writer_text(batch, None)
    assert writer_text(None, None).startswith("Pending")

    same = {"mode": "batch", "results": {"claude-opus-5-5 (effort low, prompt ab12)": writer_stats(None, None)}}
    shared = writer_text(
        same, {"mode": "live", "results": {"claude-opus-5-5 (effort low, prompt ab12)": live["results"]["opus (low)"]}}
    )
    assert shared.startswith("All rows use claude-opus-5-5 with prompt version ab12.")
    assert "| Effort | Mode |" in shared and "| low | live | 4 | 0 | 1 | $0.02 | 9.5 | 14.2 |" in shared


def test_rubric_sample_mixes_case_types(morning: Any, orchestrator: Any) -> None:
    analyses = [orchestrator.run_case(c) for c in morning.cases]
    chosen = rubric_sample(analyses, size=6)
    assert len(chosen) == 6
    assert any(a.context.conflicts for a in chosen) or not any(a.context.conflicts for a in analyses)
