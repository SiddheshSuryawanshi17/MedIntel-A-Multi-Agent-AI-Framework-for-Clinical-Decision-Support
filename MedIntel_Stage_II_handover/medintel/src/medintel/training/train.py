"""Train, calibrate and document the MedIntel risk model on UCI Heart Disease (Phase 2).

Protocol: the Cleveland data (the subset used in published work) is split 60/20/20 into train, validation and test,
stratified by outcome with a fixed seed. Settings and model families are compared by repeated cross-validation on
the training split only. Each deployed submodel is a logistic regression, calibrated with Platt scaling fitted on
out-of-fold scores, so the validation split stays free for fusion tuning and the test split is used once, for the
final numbers. The Hungarian, Swiss and VA data, where many work-up fields are missing, serve as an external check.

Two submodels are trained (see agents/risk_model.py): the pre-test model on the inputs known before an exercise
test, and the full model on all 13 inputs."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from medintel.agents.risk_model import (
    MODEL_FEATURES,
    PRE_TEST_FEATURES,
    FeatureSpec,
    FittedFeature,
    LinearRiskModel,
    RiskModelSet,
    column_names,
    design_row,
    fit_features,
    has_exercise_test,
)
from medintel.training import uci
from medintel.training.metrics import bootstrap_ci, classification, reliability

NAME = "risk-lr-uci"
VERSION = "1.0.0"
SEED = 2026
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass(frozen=True)
class Split:
    train: list[uci.UCIRecord]
    validation: list[uci.UCIRecord]
    test: list[uci.UCIRecord]


def split_cleveland(records: list[uci.UCIRecord], seed: int = SEED) -> Split:
    idx = np.arange(len(records))
    y = np.array([r.label for r in records])
    train_idx, rest_idx = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=y[rest_idx], random_state=seed)
    return Split(*([records[i] for i in sorted(ids)] for ids in (train_idx, val_idx, test_idx)))


def matrix(features: list[FittedFeature], records: list[uci.UCIRecord]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([design_row(features, r.x_s) for r in records], dtype=float)
    y = np.array([r.label for r in records], dtype=int)
    return x, y


def candidates(seed: int, families: tuple[str, ...]) -> list[tuple[str, dict[str, Any], Any]]:
    out: list[tuple[str, dict[str, Any], Any]] = []
    if "logistic_regression" in families:
        out += [("logistic_regression", {"C": c}, LogisticRegression(C=c, max_iter=5000)) for c in C_GRID]
    if "random_forest" in families:
        for leaf in (1, 5):
            forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, random_state=seed, n_jobs=-1)
            out.append(("random_forest", {"n_estimators": 500, "min_samples_leaf": leaf}, forest))
    if "gradient_boosting" in families:
        for rate in (0.05, 0.1):
            for depth in (2, 3):
                boosting = HistGradientBoostingClassifier(
                    learning_rate=rate, max_depth=depth, max_iter=200, early_stopping=False, random_state=seed
                )
                out.append(
                    ("gradient_boosting", {"learning_rate": rate, "max_depth": depth, "max_iter": 200}, boosting)
                )
    return out


def compare(x: np.ndarray, y: np.ndarray, families: tuple[str, ...], seed: int = SEED) -> list[dict[str, Any]]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    scoring = {"auroc": "roc_auc", "log_loss": "neg_log_loss", "brier": "neg_brier_score"}
    rows: list[dict[str, Any]] = []
    for family, params, estimator in candidates(seed, families):
        scores = cross_validate(estimator, x, y, cv=cv, scoring=scoring)
        rows.append(
            {
                "family": family,
                "params": params,
                "auroc": round(float(scores["test_auroc"].mean()), 4),
                "auroc_sd": round(float(scores["test_auroc"].std()), 4),
                "log_loss": round(float(-scores["test_log_loss"].mean()), 4),
                "log_loss_sd": round(float(scores["test_log_loss"].std()), 4),
                "brier": round(float(-scores["test_brier"].mean()), 4),
            }
        )
    return rows


def best_per_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["family"] not in best or row["log_loss"] < best[row["family"]]["log_loss"]:
            best[row["family"]] = row
    return best


def platt(x: np.ndarray, y: np.ndarray, c: float, seed: int = SEED) -> tuple[float, float]:
    """Platt scaling fitted on out-of-fold scores of the training split."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_predict(LogisticRegression(C=c, max_iter=5000), x, y, cv=folds, method="decision_function")
    calibrator = LogisticRegression(C=1e6, max_iter=5000).fit(scores.reshape(-1, 1), y)
    return float(calibrator.coef_[0][0]), float(calibrator.intercept_[0])


def fit_submodel(
    records: list[uci.UCIRecord], specs: tuple[FeatureSpec, ...], families: tuple[str, ...], seed: int
) -> tuple[LinearRiskModel, dict[str, Any]]:
    features = fit_features([r.x_s for r in records], specs)
    x, y = matrix(features, records)
    comparison = compare(x, y, families, seed)
    best = best_per_family(comparison)
    chosen = best["logistic_regression"]
    c = float(chosen["params"]["C"])
    lr = LogisticRegression(C=c, max_iter=5000).fit(x, y)
    coefs = lr.coef_[0].tolist()
    fitted: list[FittedFeature] = []
    start = 0
    for f in features:
        fitted.append(FittedFeature(f.spec, f.center, f.scale, tuple(coefs[start : start + f.width])))
        start += f.width
    model = LinearRiskModel(fitted, float(lr.intercept_[0]), platt(x, y, c, seed))
    leader = min(best.values(), key=lambda row: row["log_loss"])
    if len(best) == 1:
        reason = f"Logistic regression with C = {c} (lowest cross-validated log loss over the C grid)."
    elif leader["family"] == "logistic_regression":
        reason = "Lowest cross-validated log loss of the three families, and its SHAP values are exact."
    else:
        reason = (
            f"Deployed for exact SHAP explanations; {leader['family'].replace('_', ' ')} had a lower "
            f"cross-validated log loss ({leader['log_loss']:.3f} against {chosen['log_loss']:.3f})."
        )
    selection = {
        "cv": "5-fold, repeated 5 times, training split only",
        "criterion": "log loss",
        "reason": reason,
        "best_per_family": best,
        "all": comparison,
        "coefficients": dict(zip(column_names(fitted), [round(w, 4) for w in coefs], strict=True)),
    }
    return model, selection


def probabilities(model: LinearRiskModel | RiskModelSet, records: list[uci.UCIRecord]) -> np.ndarray:
    logits = np.array([model.logit(r.x_s)[0] for r in records])
    return 1 / (1 + np.exp(-logits))


def evaluate(model: LinearRiskModel | RiskModelSet, records: list[uci.UCIRecord], ci: bool = False) -> dict[str, Any]:
    p = probabilities(model, records)
    y = np.array([r.label for r in records])
    out: dict[str, Any] = classification(p, y)
    if ci:
        out["auroc_ci95"] = bootstrap_ci(p, y, "auroc")
        out["brier_ci95"] = bootstrap_ci(p, y, "brier")
        out["ece_ci95"] = bootstrap_ci(p, y, "ece")
    out["reliability"] = reliability(p, y)
    return out


def inputs_not_recorded(records: list[uci.UCIRecord]) -> float:
    """Share of the 13 model inputs missing (the UCI glucose flag counts as recorded)."""
    keys = [spec.key for spec in MODEL_FEATURES]
    missing = 0
    for r in records:
        for key in keys:
            if key == "fasting_glucose":
                missing += r.x_s.get("fasting_glucose_over_120") is None
            else:
                missing += r.x_s.get(key) is None
    return round(missing / (len(records) * len(keys)), 3)


def train(data_dir: Path, out_dir: Path, seed: int = SEED) -> dict[str, Any]:
    sites = uci.load(data_dir)
    split = split_cleveland(sites["cleveland"], seed)
    full, full_selection = fit_submodel(
        split.train, MODEL_FEATURES, ("logistic_regression", "random_forest", "gradient_boosting"), seed
    )
    pre_test, pre_selection = fit_submodel(split.train, PRE_TEST_FEATURES, ("logistic_regression",), seed)
    model = RiskModelSet({"pre_test": pre_test, "full": full}, NAME, VERSION)

    external: dict[str, Any] = {}
    for site in ("hungarian", "switzerland", "va"):
        records = sites[site]
        external[site] = {
            **evaluate(model, records, ci=True),
            "inputs_not_recorded": inputs_not_recorded(records),
            "scored_by_pre_test": round(sum(not has_exercise_test(r.x_s) for r in records) / len(records), 3),
        }
    model.metadata = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": "Angiographic coronary artery disease (over 50% diameter narrowing) in adults referred for evaluation",
        "dataset": {
            "name": "UCI Heart Disease, Cleveland subset",
            "citation": uci.CITATION,
            "license": uci.LICENSE,
            "sha256": uci.file_hashes(data_dir),
        },
        "split": {
            "seed": seed,
            "method": "stratified 60/20/20 on the Cleveland data",
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "model_selection": {"full": full_selection, "pre_test": pre_selection},
        "evaluation": {
            "full": {
                "validation": evaluate(full, split.validation),
                "test": evaluate(full, split.test, ci=True),
                "test_uncalibrated": evaluate(LinearRiskModel(full.features, full.intercept), split.test),
            },
            "pre_test": {
                "validation": evaluate(pre_test, split.validation),
                "test": evaluate(pre_test, split.test, ci=True),
            },
            "external": external,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{NAME}.json"
    path.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    deployed = RiskModelSet.load(path)
    (out_dir / "MODEL_CARD.md").write_text(model_card(deployed), encoding="utf-8")
    return {"path": str(path), "version": deployed.version, "metadata": deployed.metadata}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{value[0]:.3f} to {value[1]:.3f}"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _results(name: str, ev: dict[str, Any], cv: dict[str, Any]) -> list[str]:
    test = ev["test"]
    lines = [
        f"### {name}",
        "",
        f"Cross-validation on the training split: AUROC {cv['auroc']:.3f} (SD {cv['auroc_sd']:.3f} across folds), "
        f"the better guide to expected performance. Validation split: AUROC {_fmt(ev['validation']['auroc'])}.",
        "",
        f"- Test AUROC {_fmt(test['auroc'])} (95% CI {_fmt(test['auroc_ci95'])})",
        f"- Brier score {_fmt(test['brier'])} (95% CI {_fmt(test['brier_ci95'])})",
        f"- Expected calibration error {_fmt(test['ece'])} (95% CI {_fmt(test['ece_ci95'])})"
        + (f"; before Platt scaling {_fmt(ev['test_uncalibrated']['ece'])}" if "test_uncalibrated" in ev else ""),
        f"- At probability 0.5: accuracy {_fmt(test['accuracy'])}, sensitivity {_fmt(test['sensitivity'])}, "
        f"specificity {_fmt(test['specificity'])}",
        "",
        "| Predicted | n | Mean predicted | Observed |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['bin']} | {r['n']} | {r['mean_predicted']} | {r['observed']} |" for r in test["reliability"]]
    return lines + [""]


def _coefficients(name: str, selection: dict[str, Any], model: LinearRiskModel) -> list[str]:
    a, c = model.calibration
    lines = [f"### {name}", "", "| Input | Coefficient |", "|---|---|"]
    lines += [f"| {k} | {v:+.3f} |" for k, v in selection["coefficients"].items()]
    return lines + ["", f"Platt scaling: calibrated logit = {a:.3f} x score {c:+.3f}.", ""]


def model_card(model: RiskModelSet) -> str:
    m = model.metadata
    ev = m["evaluation"]
    sel = m["model_selection"]
    n_test = ev["full"]["test"]["n"]
    lines = [
        f"# Model card: {model.version}",
        "",
        "Generated by `uv run medintel train`. Research prototype on public de-identified data; not a medical device.",
        "",
        "## Intended use",
        "",
        f"Estimates the probability of {m['task'].lower()}, as one of four inputs to MedIntel's fusion step "
        "(Equation 4.2). The rules, the evidence and a clinician's review always come on top of it. The training "
        "population was referred for coronary angiography, so its disease rate (46%) is far higher than in a general "
        "clinic; probabilities will run high for routine check-ups.",
        "",
        "## Data",
        "",
        f"- {m['dataset']['citation']} ({m['dataset']['license']}).",
        f"- Split: {m['split']['method']}, seed {m['split']['seed']}: {m['split']['train']} train, "
        f"{m['split']['validation']} validation, {m['split']['test']} test.",
        "- A value that was not recorded is set to the training average, contributes nothing to the score, and is "
        "listed as filled in on the report.",
        "",
        "## Two submodels",
        "",
        "- **Pre-test model** (7 inputs): age, sex, chest pain type, resting systolic BP, total cholesterol, fasting "
        "glucose above 120 mg/dL, resting ECG. Used when no exercise test is on record, which covers routine "
        "check-up records.",
        "- **Full model** (13 inputs): the above plus peak exercise heart rate, exercise-induced angina, exercise ST "
        "depression, peak exercise ST slope, major vessels on fluoroscopy and the thallium stress test. Used when "
        "any of peak heart rate, exercise angina or ST depression is recorded.",
        "",
        "Why two: filling in many correlated inputs with averages distorts the rest. In the full model, age carries "
        "a small negative weight once peak heart rate is known; with peak heart rate filled in, older patients would "
        "look safer. A separate pre-test model learns the plain effect of each input instead.",
        "",
        "Both are logistic regressions (L2) on centered inputs (numeric inputs also scaled), with Platt scaling "
        "fitted on out-of-fold scores of the training split. Each feature's term is its exact SHAP value.",
        "",
        "## Model selection (full model)",
        "",
        "Model families compared by repeated cross-validation on the training split (best setting of each):",
        "",
        "| Family | Setting | AUROC | Log loss | Brier |",
        "|---|---|---|---|---|",
    ]
    for family, row in sel["full"]["best_per_family"].items():
        setting = ", ".join(f"{k}={v}" for k, v in row["params"].items())
        lines.append(
            f"| {family.replace('_', ' ')} | {setting} | {row['auroc']:.3f} | {row['log_loss']:.3f} | "
            f"{row['brier']:.3f} |"
        )
    lines += [
        "",
        f"Deployed: logistic regression. {sel['full']['reason']} Pre-test model: {sel['pre_test']['reason']}",
        "",
        f"## Results on the held-out Cleveland test split (n = {n_test})",
        "",
        "The test split is small, so its numbers can land well above or below the cross-validation estimate.",
        "",
    ]
    lines += _results("Full model", ev["full"], sel["full"]["best_per_family"]["logistic_regression"])
    lines += _results(
        "Pre-test model (same patients, work-up inputs ignored)",
        ev["pre_test"],
        sel["pre_test"]["best_per_family"]["logistic_regression"],
    )
    lines += [
        "## External check (other UCI sites, routed as in production)",
        "",
        "| Site | n | Disease rate | Inputs not recorded | Scored by pre-test | AUROC | Brier | ECE | Mean predicted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for site, r in ev["external"].items():
        lines.append(
            f"| {site} | {r['n']} | {r['prevalence']} | {r['inputs_not_recorded']:.0%} | "
            f"{r['scored_by_pre_test']:.0%} | {_fmt(r['auroc'])} | {_fmt(r['brier'])} | {_fmt(r['ece'])} | "
            f"{_fmt(r['mean_predicted'])} |"
        )
    lines += [
        "",
        "These sites lack many work-up fields and have different disease rates, so calibration is expected to drift; "
        "the table shows how much.",
        "",
        "## Coefficients",
        "",
        "Log-odds before Platt scaling: numeric inputs per standard deviation, yes/no inputs (male sex, glucose above "
        "120 mg/dL, exercise angina) per unit, categories against the training mix.",
        "",
    ]
    lines += _coefficients("Full model", sel["full"], model.submodels["full"])
    lines += _coefficients("Pre-test model", sel["pre_test"], model.submodels["pre_test"])
    lines += [
        "## Limitations",
        "",
        "- Small single-site development data (303 Cleveland Clinic patients, donated to UCI in 1988); the test "
        f"split has {n_test} patients.",
        "- The work-up inputs (fluoroscopy, thallium test) are only available after specialist testing.",
        "- Not validated for patients under 18, or outside a referral population.",
        "- In this referral data, patients without chest pain (asymptomatic) had the highest disease rate and typical "
        "angina a lower one, so the model gives typical angina a negative contribution. That reflects who was referred "
        "for angiography, not what the symptom means clinically; read the contributions with that in mind.",
        "- Sex is recorded as male or female only, as in the source data.",
        "",
    ]
    return "\n".join(lines)
