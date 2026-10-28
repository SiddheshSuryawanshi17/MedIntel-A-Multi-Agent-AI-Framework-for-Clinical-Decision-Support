"""Trained risk model (FR3): calibrated logistic regressions stored as JSON and explained with exact SHAP values.

Training (medintel.training.train) and inference share the encoder below, so a record becomes model inputs the
same way in both. Every input column is centered on the training data. That makes the intercept the score of the
average training patient, and each feature's term w * (x - mean) its exact SHAP value for a linear model. A value
that was not recorded is set to the training average: it contributes nothing and is reported as filled in.

Filling in many correlated inputs distorts the rest (without peak heart rate, the full model's adjusted age term
points the wrong way), so the file holds two pattern submodels: a pre-test model on the inputs known before any
exercise test, and the full model, used once an exercise test is on record."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from medintel.agents.ml_agent import DemoRiskModel, RiskModel
from medintel.models import FeatureContribution
from medintel.numeric import round3
from medintel.schema import Value

FORMAT = "medintel-risk-model/1"
FASTING_GLUCOSE_FLAG = "fasting_glucose_over_120"
EXERCISE_TEST_KEYS = ("max_heart_rate", "exercise_angina", "st_depression")
FeatureKind = Literal["numeric", "binary", "categorical"]
log = logging.getLogger("medintel")


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    kind: FeatureKind
    unit: str = ""
    levels: tuple[str, ...] = ()


MODEL_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("age", "Age", "numeric", "years"),
    FeatureSpec("sex", "Sex", "binary"),
    FeatureSpec(
        "chest_pain_type",
        "Chest pain type",
        "categorical",
        levels=("typical_angina", "atypical_angina", "non_anginal", "asymptomatic"),
    ),
    FeatureSpec("systolic_bp", "Systolic BP", "numeric", "mmHg"),
    FeatureSpec("total_cholesterol", "Total cholesterol", "numeric", "mg/dL"),
    FeatureSpec("fasting_glucose", "Fasting glucose above 120 mg/dL", "binary", "mg/dL"),
    FeatureSpec("resting_ecg", "Resting ECG", "categorical", levels=("normal", "st_t_abnormality", "lv_hypertrophy")),
    FeatureSpec("max_heart_rate", "Peak exercise heart rate", "numeric", "bpm"),
    FeatureSpec("exercise_angina", "Exercise-induced angina", "binary"),
    FeatureSpec("st_depression", "Exercise ST depression", "numeric", "mm"),
    FeatureSpec("st_slope", "Peak exercise ST slope", "categorical", levels=("upsloping", "flat", "downsloping")),
    FeatureSpec("major_vessels", "Major vessels on fluoroscopy", "numeric"),
    FeatureSpec("thal", "Thallium stress test", "categorical", levels=("normal", "fixed_defect", "reversible_defect")),
)
PRE_TEST_KEYS = ("age", "sex", "chest_pain_type", "systolic_bp", "total_cholesterol", "fasting_glucose", "resting_ecg")
PRE_TEST_FEATURES = tuple(f for f in MODEL_FEATURES if f.key in PRE_TEST_KEYS)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_feature(spec: FeatureSpec, x_s: dict[str, Value]) -> tuple[float | str | None, float | str | None]:
    """Return (value shown in the report, value the model uses). The model value is None when not recorded."""
    if spec.key == "sex":
        sex = x_s.get("sex")
        return (sex, 1.0 if sex == "M" else 0.0) if sex in ("M", "F") else (None, None)
    if spec.key == "fasting_glucose":
        glucose = x_s.get("fasting_glucose")
        if _is_number(glucose):
            return float(glucose), 1.0 if float(glucose) > 120 else 0.0  # type: ignore[arg-type]
        flag = x_s.get(FASTING_GLUCOSE_FLAG)  # the UCI data records only this flag
        if isinstance(flag, bool):
            return ("yes" if flag else "no"), 1.0 if flag else 0.0
        return None, None
    value = x_s.get(spec.key)
    if value is None:
        return None, None
    if spec.kind == "binary":
        return ("yes" if value else "no"), 1.0 if value else 0.0
    if spec.kind == "categorical":
        return (value, value) if value in spec.levels else (None, None)
    if not _is_number(value):
        return None, None
    return float(value), float(value)  # type: ignore[arg-type]


def has_exercise_test(x_s: dict[str, Value]) -> bool:
    return any(x_s.get(key) is not None for key in EXERCISE_TEST_KEYS)


@dataclass(frozen=True)
class FittedFeature:
    spec: FeatureSpec
    center: tuple[float, ...]
    scale: float = 1.0
    coefs: tuple[float, ...] = ()

    @property
    def width(self) -> int:
        return len(self.spec.levels) if self.spec.kind == "categorical" else 1

    def columns(self, model_value: float | str | None) -> list[float]:
        if model_value is None:
            return [0.0] * self.width
        if self.spec.kind == "categorical":
            return [
                (1.0 if model_value == lv else 0.0) - f for lv, f in zip(self.spec.levels, self.center, strict=True)
            ]
        return [(float(model_value) - self.center[0]) / self.scale]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.spec.key,
            "label": self.spec.label,
            "kind": self.spec.kind,
            "unit": self.spec.unit,
            "levels": list(self.spec.levels),
            "center": list(self.center),
            "scale": self.scale,
            "coefs": list(self.coefs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FittedFeature":
        spec = FeatureSpec(
            data["key"], data["label"], data["kind"], data.get("unit", ""), tuple(data.get("levels", ()))
        )
        return cls(spec, tuple(data["center"]), float(data.get("scale", 1.0)), tuple(data["coefs"]))


def fit_features(
    records: list[dict[str, Value]], specs: tuple[FeatureSpec, ...] = MODEL_FEATURES
) -> list[FittedFeature]:
    """Training-set centers (and scales for numeric inputs), computed over the values actually recorded."""
    fitted: list[FittedFeature] = []
    for spec in specs:
        values = [v for v in (read_feature(spec, x)[1] for x in records) if v is not None]
        if not values:
            raise ValueError(f"No recorded values for {spec.key}")
        if spec.kind == "categorical":
            fitted.append(FittedFeature(spec, tuple(values.count(level) / len(values) for level in spec.levels)))
            continue
        numbers = [float(v) for v in values]
        mean = sum(numbers) / len(numbers)
        if spec.kind == "binary":
            fitted.append(FittedFeature(spec, (mean,)))
            continue
        variance = sum((v - mean) ** 2 for v in numbers) / max(1, len(numbers) - 1)
        fitted.append(FittedFeature(spec, (mean,), variance**0.5 or 1.0))
    return fitted


def design_row(features: list[FittedFeature], x_s: dict[str, Value]) -> list[float]:
    return [z for f in features for z in f.columns(read_feature(f.spec, x_s)[1])]


def column_names(features: list[FittedFeature]) -> list[str]:
    names: list[str] = []
    for f in features:
        names += [f"{f.spec.key}={level}" for level in f.spec.levels] if f.spec.kind == "categorical" else [f.spec.key]
    return names


class LinearRiskModel:
    """One calibrated logistic regression over centered inputs."""

    def __init__(self, features: list[FittedFeature], intercept: float, calibration: tuple[float, float] = (1.0, 0.0)):
        self.features = features
        self.intercept = intercept
        self.calibration = calibration

    @property
    def base_logit(self) -> float:
        a, c = self.calibration
        return a * self.intercept + c

    def logit(self, x_s: dict[str, Value]) -> tuple[float, list[FeatureContribution]]:
        a, _ = self.calibration
        total = self.base_logit
        contributions: list[FeatureContribution] = []
        for f in self.features:
            shown, used = read_feature(f.spec, x_s)
            phi = a * sum(w * z for w, z in zip(f.coefs, f.columns(used), strict=True))
            total += phi
            contributions.append(
                FeatureContribution(
                    key=f.spec.key,
                    label=f.spec.label,
                    unit=f.spec.unit,
                    value=shown,
                    imputed=used is None,
                    contribution=round3(phi),
                )
            )
        return total, contributions

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": self.intercept,
            "calibration": {"method": "platt", "a": self.calibration[0], "c": self.calibration[1]},
            "features": [f.to_dict() for f in self.features],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearRiskModel":
        return cls(
            [FittedFeature.from_dict(f) for f in data["features"]],
            float(data["intercept"]),
            (float(data["calibration"]["a"]), float(data["calibration"]["c"])),
        )


VARIANT_LABEL = {"pre_test": "pre-test model", "full": "full model"}


class RiskModelSet:
    """The trained model file: a pre-test and a full submodel, chosen per record by whether an exercise test is
    on record. Implements the RiskModel protocol."""

    def __init__(
        self,
        submodels: dict[str, LinearRiskModel],
        name: str,
        release: str,
        digest: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.submodels = submodels
        self.name = name
        self.release = release
        self.version = f"{name} {release}" + (f" ({digest[:8]})" if digest else "")
        self.metadata = metadata or {}

    def variant_for(self, x_s: dict[str, Value]) -> str:
        return "full" if has_exercise_test(x_s) else "pre_test"

    def logit(self, x_s: dict[str, Value]) -> tuple[float, list[FeatureContribution]]:
        return self.submodels[self.variant_for(x_s)].logit(x_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "name": self.name,
            "version": self.release,
            "routing": {"full_when_any_recorded": list(EXERCISE_TEST_KEYS), "otherwise": "pre_test"},
            "submodels": {name: model.to_dict() for name, model in self.submodels.items()},
            **self.metadata,
        }

    @classmethod
    def load(cls, path: str | Path) -> "RiskModelSet":
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if data.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        known = {"format", "name", "version", "routing", "submodels"}
        return cls(
            {name: LinearRiskModel.from_dict(sub) for name, sub in data["submodels"].items()},
            data["name"],
            data["version"],
            hashlib.sha256(raw).hexdigest(),
            {k: v for k, v in data.items() if k not in known},
        )


def load_risk_model(path: str | Path | None) -> RiskModel:
    """The trained model at `path`, or the prototype's demo-weight model when no trained model is configured."""
    if not path:
        return DemoRiskModel()
    candidate = Path(path)
    if not candidate.is_file():
        log.warning("Risk model %s not found; using the demo-weight model. Run: uv run medintel train", candidate)
        return DemoRiskModel()
    return RiskModelSet.load(candidate)
