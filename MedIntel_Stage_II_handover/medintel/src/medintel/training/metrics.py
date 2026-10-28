"""Classification and calibration metrics used in the model card and the evaluation (Sections 6.5.3 and 7)."""

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on the predicted probability of the positive class."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str, float | int | str]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows: list[dict[str, float | int | str]] = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            rows.append(
                {
                    "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                    "n": int(mask.sum()),
                    "mean_predicted": round(float(p[mask].mean()), 3),
                    "observed": round(float(y[mask].mean()), 3),
                }
            )
    return rows


def classification(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict[str, float | int | None]:
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    both = len(set(y.tolist())) == 2
    return {
        "n": int(len(y)),
        "prevalence": round(float(y.mean()), 3),
        "auroc": round(float(roc_auc_score(y, p)), 3) if both else None,
        "brier": round(float(brier_score_loss(y, p)), 3),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), 3),
        "ece": round(ece(p, y), 3),
        "mean_predicted": round(float(p.mean()), 3),
        "accuracy": round((tp + tn) / len(y), 3),
        "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
    }


def bootstrap_ci(
    p: np.ndarray, y: np.ndarray, metric: str, n: int = 2000, seed: int = 2026
) -> tuple[float, float] | None:
    """Percentile 95% interval; resamples that contain one class only are skipped."""
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        ys, ps = y[idx], p[idx]
        if metric == "auroc":
            if len(set(ys.tolist())) < 2:
                continue
            values.append(float(roc_auc_score(ys, ps)))
        elif metric == "brier":
            values.append(float(brier_score_loss(ys, ps)))
        elif metric == "ece":
            values.append(ece(ps, ys))
    if not values:
        return None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)
