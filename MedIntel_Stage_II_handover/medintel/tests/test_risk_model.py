"""Trained risk model: encoder, exact SHAP values, pattern submodels, the model file and the UCI parser."""

import json
import logging
import random
from pathlib import Path

import pytest

from medintel.agents.ml_agent import DemoRiskModel, MLAnalysisAgent
from medintel.agents.risk_model import (
    MODEL_FEATURES,
    PRE_TEST_FEATURES,
    FittedFeature,
    LinearRiskModel,
    RiskModelSet,
    fit_features,
    load_risk_model,
    read_feature,
)
from medintel.models import PatientCase
from medintel.training import uci
from medintel.training.train import fit_submodel, split_cleveland

UCI_ROWS = """63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,?,3.0,0
"""
SWISS_ROW = "32,1,1,95,0,?,0,127,0,.7,1,?,?,1\n"
WORKUP = {
    "chest_pain_type": "asymptomatic",
    "resting_ecg": "lv_hypertrophy",
    "max_heart_rate": 108.0,
    "exercise_angina": True,
    "st_depression": 1.5,
    "st_slope": "flat",
    "major_vessels": 3,
    "thal": "normal",
}
BASE = {"age": 67, "sex": "M", "systolic_bp": 160.0, "total_cholesterol": 286.0, "fasting_glucose": 99.0}


def synthetic_records(n: int = 160, seed: int = 7) -> list[uci.UCIRecord]:
    rng = random.Random(seed)
    records = []
    for i in range(n):
        age = rng.randint(30, 77)
        male = rng.random() < 0.65
        st = round(rng.uniform(0, 4), 1)
        x = {
            "age": age,
            "sex": "M" if male else "F",
            "chest_pain_type": rng.choice(["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"]),
            "systolic_bp": float(rng.randint(100, 180)),
            "total_cholesterol": float(rng.randint(150, 330)),
            "fasting_glucose_over_120": rng.random() < 0.15,
            "resting_ecg": rng.choice(["normal", "st_t_abnormality", "lv_hypertrophy"]),
            "max_heart_rate": float(rng.randint(90, 195)),
            "exercise_angina": rng.random() < 0.3,
            "st_depression": st,
            "st_slope": rng.choice(["upsloping", "flat", "downsloping"]),
            "major_vessels": rng.randint(0, 3),
            "thal": rng.choice(["normal", "fixed_defect", "reversible_defect"]),
        }
        score = 0.05 * (age - 55) + (0.8 if male else -0.8) + 0.9 * (st - 1.0) + rng.gauss(0, 0.8)
        records.append(uci.UCIRecord("cleveland", i + 1, x, int(score > 0)))
    return records


@pytest.fixture(scope="module")
def trained() -> RiskModelSet:
    records = synthetic_records()
    full, _ = fit_submodel(records, MODEL_FEATURES, ("logistic_regression",), seed=1)
    pre, _ = fit_submodel(records, PRE_TEST_FEATURES, ("logistic_regression",), seed=1)
    return RiskModelSet({"pre_test": pre, "full": full}, "risk-lr-test", "0.0.1")


def test_uci_rows_map_to_schema_fields() -> None:
    first, second, third = uci.parse_site(UCI_ROWS, "cleveland")
    assert first.x_s["chest_pain_type"] == "typical_angina"
    assert first.x_s["thal"] == "fixed_defect" and first.x_s["st_slope"] == "downsloping"
    assert first.x_s["fasting_glucose_over_120"] is True and first.label == 0
    assert second.label == 1, "num 1-4 means disease"
    assert third.x_s["major_vessels"] is None, "? marks a missing value"
    (swiss,) = uci.parse_site(SWISS_ROW, "switzerland")
    assert swiss.x_s["total_cholesterol"] is None, "0 cholesterol means not measured"
    assert swiss.x_s["fasting_glucose_over_120"] is None and swiss.x_s["thal"] is None


def test_encoder_centers_on_training_data() -> None:
    records = [r.x_s for r in synthetic_records(40)]
    features = {f.spec.key: f for f in fit_features(records)}
    assert sum(features["chest_pain_type"].center) == pytest.approx(1.0)
    ages = [x["age"] for x in records]
    assert features["age"].center[0] == pytest.approx(sum(ages) / len(ages))
    assert read_feature(features["fasting_glucose"].spec, {"fasting_glucose_over_120": True}) == ("yes", 1.0)
    assert read_feature(features["fasting_glucose"].spec, {"fasting_glucose": 130.0}) == (130.0, 1.0)


def test_contributions_are_exact_shap_values(trained: RiskModelSet) -> None:
    x = {**BASE, **WORKUP}
    model = trained.submodels["full"]
    logit, contributions = model.logit(x)
    assert model.base_logit + sum(c.contribution for c in contributions) == pytest.approx(logit, abs=0.0005 * 13)
    assert not any(c.imputed for c in contributions)


def test_missing_inputs_contribute_nothing(trained: RiskModelSet) -> None:
    x = {**BASE, **WORKUP, "thal": None, "major_vessels": None}
    _, contributions = trained.submodels["full"].logit(x)
    filled = {c.key: c for c in contributions if c.imputed}
    assert set(filled) == {"thal", "major_vessels"}
    assert all(c.contribution == 0 and c.value is None for c in filled.values())


def test_pre_test_model_scores_records_without_an_exercise_test(trained: RiskModelSet) -> None:
    agent = MLAnalysisAgent(trained)
    routine = PatientCase(case_id="C-0001", patient_id="P-0001", x_s=BASE, source="test")
    workup = PatientCase(case_id="C-0002", patient_id="P-0002", x_s={**BASE, **WORKUP}, source="test")
    assert agent.analyze(routine).variant == "pre_test"
    assert len(agent.analyze(routine).contributions) == len(PRE_TEST_FEATURES)
    assert agent.analyze(workup).variant == "full"
    assert {c.key for c in agent.analyze(routine).contributions if c.imputed} == {"chest_pain_type", "resting_ecg"}


def test_learned_signal_has_the_right_direction(trained: RiskModelSet) -> None:
    coefs = {f.spec.key: f.coefs for f in trained.submodels["full"].features}
    assert coefs["st_depression"][0] > 0 and coefs["sex"][0] > 0


def test_model_file_round_trip(trained: RiskModelSet, tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(trained.to_dict()), encoding="utf-8")
    loaded = load_risk_model(path)
    assert isinstance(loaded, RiskModelSet)
    assert loaded.version.startswith("risk-lr-test 0.0.1 (") and len(loaded.version.split("(")[1]) == 9
    x = {**BASE, **WORKUP}
    assert loaded.logit(x)[0] == pytest.approx(trained.logit(x)[0])
    with pytest.raises(ValueError, match="medintel-risk-model"):
        path.write_text(json.dumps({"format": "other"}), encoding="utf-8")
        RiskModelSet.load(path)


def test_demo_model_is_the_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert isinstance(load_risk_model(""), DemoRiskModel)
    with caplog.at_level(logging.WARNING, logger="medintel"):
        assert isinstance(load_risk_model(tmp_path / "missing.json"), DemoRiskModel)
    assert "medintel train" in caplog.text


def test_split_is_stratified_and_disjoint() -> None:
    records = synthetic_records(100)
    split = split_cleveland(records, seed=3)
    assert (len(split.train), len(split.validation), len(split.test)) == (60, 20, 20)
    ids = [r.row for part in (split.train, split.validation, split.test) for r in part]
    assert len(set(ids)) == 100


def test_fitted_feature_survives_json() -> None:
    feature = FittedFeature(MODEL_FEATURES[2], (0.1, 0.2, 0.3, 0.4), 1.0, (0.5, -0.5, 0.0, 1.0))
    assert FittedFeature.from_dict(json.loads(json.dumps(feature.to_dict()))) == feature


def test_linear_model_dict_round_trip(trained: RiskModelSet) -> None:
    model = trained.submodels["pre_test"]
    again = LinearRiskModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert again.logit(BASE)[0] == pytest.approx(model.logit(BASE)[0])


TRAINED_FILE = Path(__file__).resolve().parents[1] / "models" / "risk-lr-uci.json"


@pytest.mark.skipif(not TRAINED_FILE.is_file(), reason="run `uv run medintel train` first")
def test_trained_uci_model_scores_the_sample_batch(morning: object) -> None:
    model = load_risk_model(TRAINED_FILE)
    agent = MLAnalysisAgent(model)
    results = [agent.analyze(case) for case in morning.cases]  # type: ignore[attr-defined]
    assert all(r.variant == "pre_test" for r in results)
    assert all(0 < r.probabilities["high_risk"] < 1 for r in results)
    assert model.version.startswith("risk-lr-uci 1.0.0")
