"""MLAnalysisAgent (FR3). Runs the trained model (agents/risk_model.py) or, without one, the prototype's demo
weights."""

import math
from dataclasses import dataclass
from typing import Protocol

from medintel.models import FeatureContribution, MLResult, PatientCase, Prediction
from medintel.numeric import round3
from medintel.schema import Value


@dataclass(frozen=True)
class LinearFeature:
    key: str
    label: str
    unit: str
    coef: float
    ref: float
    impute: float | None = None


class RiskModel(Protocol):
    version: str

    def logit(self, x_s: dict[str, Value]) -> tuple[float, list[FeatureContribution]]: ...

    def variant_for(self, x_s: dict[str, Value]) -> str: ...


def feature_value(x_s: dict[str, Value], key: str) -> float | None:
    if key == "sex_m":
        return 1.0 if x_s.get("sex") == "M" else 0.0
    if key == "smoker":
        return 1.0 if x_s.get("smoker") else 0.0
    value = x_s.get(key)
    if value is None or isinstance(value, str):
        return None
    return float(value)


class DemoRiskModel:
    """Logistic model with hand-set weights carried over from the Stage I concept prototype."""

    version = "risk-lr 0.3 (demo weights)"
    base_logit = -2.2
    features: tuple[LinearFeature, ...] = (
        LinearFeature("age", "Age", "years", 0.06, 50),
        LinearFeature("sex_m", "Male sex", "", 0.45, 0),
        LinearFeature("systolic_bp", "Systolic BP", "mmHg", 0.018, 120),
        LinearFeature("total_cholesterol", "Total cholesterol", "mg/dL", 0.009, 190),
        LinearFeature("fasting_glucose", "Fasting glucose", "mg/dL", 0.01, 90),
        LinearFeature("smoker", "Current smoker", "", 0.85, 0),
        LinearFeature("bmi", "BMI", "kg/m²", 0.04, 24),
        LinearFeature("creatinine", "Creatinine", "mg/dL", 0.6, 0.9, impute=0.9),
    )

    def variant_for(self, x_s: dict[str, Value]) -> str:
        return ""

    def logit(self, x_s: dict[str, Value]) -> tuple[float, list[FeatureContribution]]:
        total = self.base_logit
        contributions: list[FeatureContribution] = []
        for feature in self.features:
            value = feature_value(x_s, feature.key)
            imputed = value is None
            if value is None:
                if feature.impute is None:
                    raise ValueError(f"{feature.key} is required by {self.version}")
                value = feature.impute
            contribution = feature.coef * (value - feature.ref)
            total += contribution
            contributions.append(
                FeatureContribution(
                    key=feature.key,
                    label=feature.label,
                    unit=feature.unit,
                    value=value,
                    imputed=imputed,
                    contribution=round3(contribution),
                )
            )
        return total, contributions


class MLAnalysisAgent:
    def __init__(self, model: RiskModel | None = None) -> None:
        self.model = model or DemoRiskModel()

    def analyze(self, case: PatientCase) -> MLResult:
        logit, contributions = self.model.logit(case.x_s)
        p_high = 1 / (1 + math.exp(-logit))
        prediction: Prediction = "high_risk" if p_high >= 0.5 else "low_risk"
        return MLResult(
            prediction=prediction,
            probabilities={"high_risk": round3(p_high), "low_risk": round3(1 - p_high)},
            logit=round3(logit),
            contributions=contributions,
            importance=sorted(contributions, key=lambda c: -abs(c.contribution)),
            model_version=self.model.version,
            variant=self.model.variant_for(case.x_s),
        )
