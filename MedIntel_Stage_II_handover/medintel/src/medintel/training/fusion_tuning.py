"""Tune the fusion settings of Equation (4.2) on the validation split (Sections 4.4 and 6.5.3).

C is meant to read like a probability that the reported decision is right, so the weights are chosen to minimize
the Brier score of C against "the decision before the threshold was correct" on the validation split, with
alpha + beta + gamma = 1. The threshold is then the lowest value at which the cases with C at or above it reach the
target accuracy. The test split is scored once, with the tuned settings and with the current ones. Nothing is
written to config/settings.toml: the team decides whether to adopt the result.

A weight can only be tuned when the validation data exercise it. UCI patients rarely trigger a rule override or a
conflict, so beta (rule agreement R) and lambda (disagreement D) are held at their design values when R never drops
or D is almost always 0; the unconstrained optimum is reported alongside for reference."""

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from medintel.agents.context_agent import TemplateWriter
from medintel.agents.risk_model import load_risk_model
from medintel.config import Settings
from medintel.fusion import FusionSettings
from medintel.models import PatientCase
from medintel.orchestrator import create_orchestrator
from medintel.training import uci
from medintel.training.metrics import ece
from medintel.training.train import SEED, split_cleveland

GRID = [round(i / 20, 2) for i in range(21)]
LAMBDAS = [round(i / 10, 1) for i in range(11)]


def uci_case(record: uci.UCIRecord) -> PatientCase:
    return PatientCase(
        case_id=f"U-{record.site[:2].upper()}-{record.row:03d}",
        patient_id=f"UCI-{record.site}-{record.row}",
        x_s=dict(record.x_s),
        source=f"uci:{record.site}:{record.row}",
    )


def component_rows(settings: Settings, records: list[uci.UCIRecord], knowledge: Any) -> list[dict[str, Any]]:
    orchestrator = create_orchestrator(settings, TemplateWriter(), None, knowledge)
    rows: list[dict[str, Any]] = []
    for record in records:
        a = orchestrator.run_case(uci_case(record))
        decision = a.fusion.override or a.ml.prediction
        rows.append(
            {
                "case_id": a.case.case_id,
                "label": record.label,
                "decision": decision,
                "correct": None if decision == "withheld" else int((decision == "high_risk") == bool(record.label)),
                "p_high": a.ml.probabilities["high_risk"],
                "rule_conflict": int(any(c.type == "rule_vs_ml" for c in a.context.conflict_details)),
                **{k: float(a.fusion.component_scores[k]) for k in ("M", "R", "K", "D")},
            }
        )
    return rows


def confidences(rows: list[dict[str, Any]], f: FusionSettings) -> np.ndarray:
    return np.array(
        [max(0.0, f.alpha * r["M"] + f.beta * r["R"] + f.gamma * r["K"] - f.lam * r["D"]) for r in rows], dtype=float
    )


def scored(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    kept = [r for r in rows if r["correct"] is not None]
    return kept, np.array([r["correct"] for r in kept], dtype=float)


def brier(c: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.clip(c, 0, 1) - y) ** 2))


def held_weights(rows: list[dict[str, Any]], current: FusionSettings) -> dict[str, str]:
    held: dict[str, str] = {}
    if all(r["R"] == 1 for r in rows):
        held["beta"] = "no validation case has a rule override that disagrees with the model (R is always 1)"
    if sum(r["rule_conflict"] for r in rows) < 3:
        held["lam"] = "rule-versus-model conflicts are absent or too rare in the validation split to tune their penalty"
    elif sum(r["D"] > 0 for r in rows) < max(3, 0.05 * len(rows)):
        held["lam"] = "conflicts (D > 0) are too rare in the validation split to tune their penalty"
    return held


def search_weights(
    rows: list[dict[str, Any]], current: FusionSettings, held: dict[str, str] | None = None
) -> tuple[FusionSettings, float]:
    kept, y = scored(rows)
    held = held or {}
    betas = [current.beta] if "beta" in held else GRID
    lambdas = [current.lam] if "lam" in held else LAMBDAS
    best: tuple[float, FusionSettings] | None = None
    for alpha, beta, lam in product(GRID, betas, lambdas):
        gamma = round(1 - alpha - beta, 2)
        if gamma < 0:
            continue
        f = FusionSettings(alpha=alpha, beta=beta, gamma=gamma, lam=lam, threshold=current.threshold)
        score = brier(confidences(kept, f), y)
        if best is None or score < best[0] - 1e-12:
            best = (score, f)
    assert best is not None
    return best[1], best[0]


def choose_threshold(rows: list[dict[str, Any]], f: FusionSettings, target: float) -> float:
    kept, y = scored(rows)
    c = confidences(kept, f)
    for t in sorted(set(np.round(c, 3).tolist())):
        mask = c >= t
        if mask.any() and y[mask].mean() >= target:
            return float(t)
    return float(round(c.max(), 3)) if len(c) else f.threshold


def summarize(rows: list[dict[str, Any]], f: FusionSettings) -> dict[str, Any]:
    kept, y = scored(rows)
    c = confidences(kept, f)
    confident = c >= f.threshold
    return {
        "n": len(kept),
        "accuracy_before_threshold": round(float(y.mean()), 3),
        "brier_C": round(brier(c, y), 3),
        "ece_C": round(ece(np.clip(c, 0, 1), y), 3),
        "coverage": round(float(confident.mean()), 3),
        "accuracy_when_confident": round(float(y[confident].mean()), 3) if confident.any() else None,
        "accuracy_when_uncertain": round(float(y[~confident].mean()), 3) if (~confident).any() else None,
        "mean_C": round(float(c.mean()), 3),
    }


def threshold_table(rows: list[dict[str, Any]], f: FusionSettings) -> list[dict[str, Any]]:
    """Coverage and accuracy of the confident cases at a range of thresholds: the trade-off the team chooses."""
    kept, y = scored(rows)
    c = confidences(kept, f)
    table = []
    for t in [round(0.5 + 0.05 * i, 2) for i in range(10)]:
        mask = c >= t
        table.append(
            {
                "threshold": t,
                "coverage": round(float(mask.mean()), 3),
                "accuracy_when_confident": round(float(y[mask].mean()), 3) if mask.any() else None,
            }
        )
    return table


def tune(settings: Settings, data_dir: Path, knowledge: Any, target: float = 0.85, seed: int = SEED) -> dict[str, Any]:
    split = split_cleveland(uci.load(data_dir)["cleveland"], seed)
    validation = component_rows(settings, split.validation, knowledge)
    test = component_rows(settings, split.test, knowledge)
    current = settings.fusion
    held = held_weights(validation, current)
    weights, val_brier = search_weights(validation, current, held)
    tuned = weights.model_copy(update={"threshold": choose_threshold(validation, weights, target)})
    free, free_brier = search_weights(validation, current)
    unconstrained = free.model_copy(update={"threshold": choose_threshold(validation, free, target)})
    return {
        "objective": "Brier score of C against a correct pre-threshold decision, validation split",
        "target_accuracy_when_confident": target,
        "split_seed": seed,
        "retrieval_backend": settings.retrieval.backend,
        "risk_model": load_risk_model(settings.ml.model).version,
        "current": current.model_dump(),
        "tuned": tuned.model_dump(),
        "held_at_design_values": held,
        "unconstrained_for_reference": unconstrained.model_dump(),
        "validation": {
            "current": summarize(validation, current),
            "tuned": summarize(validation, tuned),
            "unconstrained": summarize(validation, unconstrained),
        },
        "test": {
            "current": summarize(test, current),
            "tuned": summarize(test, tuned),
            "unconstrained": summarize(test, unconstrained),
        },
        "threshold_table_validation": threshold_table(validation, tuned),
        "component_means_validation": {
            k: round(float(np.mean([r[k] for r in validation])), 3) for k in ("M", "R", "K", "D")
        },
        "validation_brier": {"tuned": round(val_brier, 4), "unconstrained": round(free_brier, 4)},
    }


def settings_block(f: dict[str, Any]) -> str:
    return "\n".join(["[fusion]"] + [f"{key} = {f[key]}" for key in ("alpha", "beta", "gamma", "lam", "threshold")])
