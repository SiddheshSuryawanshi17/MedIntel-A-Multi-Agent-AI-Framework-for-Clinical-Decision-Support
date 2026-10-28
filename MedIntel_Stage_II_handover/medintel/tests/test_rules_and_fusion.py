import pytest

from medintel.agents.rule_agent import RuleBase, RuleValidationAgent, fires
from medintel.fusion import DecisionFusionEngine, FusionSettings, priority, recommend
from medintel.ingestion import IngestionResult
from medintel.models import (
    ContextSummary,
    EvidenceItem,
    MLResult,
    PatientCase,
    RuleResult,
)


def with_values(case: PatientCase, **values: object) -> PatientCase:
    return case.model_copy(update={"x_s": {**case.x_s, **values}})


@pytest.fixture(scope="module")
def agent() -> RuleValidationAgent:
    return RuleValidationAgent()


@pytest.mark.parametrize(("sbp", "fired"), [(179, False), (180, True), (181, True)])
def test_r001_boundary(agent: RuleValidationAgent, morning: IngestionResult, sbp: int, fired: bool) -> None:
    result = agent.evaluate(with_values(morning.cases[1], systolic_bp=sbp))
    assert ("R-001" in result.triggered) is fired


def test_suppress_outranks_elevate_at_equal_severity(agent: RuleValidationAgent, morning: IngestionResult) -> None:
    result = agent.evaluate(with_values(morning.cases[1], age=16, systolic_bp=185))
    assert result.triggered == ["R-001", "R-003"]
    assert result.action == "suppress"


def test_missing_creatinine_and_heart_rate_flags(agent: RuleValidationAgent, morning: IngestionResult) -> None:
    case = morning.cases[0]
    assert agent.evaluate(with_values(case, creatinine=None)).triggered == ["R-002"]
    assert agent.evaluate(with_values(case, heart_rate=119)).triggered == []
    assert agent.evaluate(with_values(case, heart_rate=120)).action == "flag"


def test_rulebase_is_versioned() -> None:
    rulebase = RuleBase.load()
    assert rulebase.version == "1.0.0"
    assert [r.rule_id for r in rulebase.rules] == ["R-001", "R-002", "R-003", "R-004"]
    assert not fires(rulebase.rules[0], None)


def ml_result(prediction: str, p_high: float) -> MLResult:
    return MLResult(
        prediction=prediction,
        probabilities={"high_risk": p_high, "low_risk": round(1 - p_high, 3)},
        logit=0.0,
        contributions=[],
        importance=[],
        model_version="test",
    )


def rule_result(action: str | None) -> RuleResult:
    return RuleResult(triggered=[], flags=[], action=action, reason="", top_rule=None, checks=[], rulebase_version="t")


def context(conflicts: int) -> ContextSummary:
    return ContextSummary(
        summary="",
        conflicts=["c"] * conflicts,
        conflict_details=[],
        unsupported_statements=[],
        draft=[],
        generated_by="test",
    )


def evidence(*scores: float) -> list[EvidenceItem]:
    return [
        EvidenceItem(source_id=f"KB-T-{i}", title="", topic="", passage="", direction="raises", score=s, matched_on="")
        for i, s in enumerate(scores)
    ]


def test_equation_4_2() -> None:
    engine = DecisionFusionEngine(FusionSettings(alpha=0.5, beta=0.3, gamma=0.2, lam=0.5, threshold=0.6))
    fusion = engine.fuse(ml_result("high_risk", 0.8), rule_result(None), evidence(0.9, 0.7), context(1))
    expected = 0.5 * 0.8 + 0.3 * 1 + 0.2 * 0.8 - 0.5 * (1 / 3)
    assert fusion.confidence == pytest.approx(round(expected, 3))
    assert fusion.component_scores == {"M": 0.8, "R": 1.0, "K": 0.8, "D": 0.333}
    assert fusion.decision == "high_risk" and not fusion.below_threshold


def test_rule_override_sets_r_to_zero_and_wins() -> None:
    engine = DecisionFusionEngine()
    fusion = engine.fuse(ml_result("low_risk", 0.2), rule_result("elevate"), evidence(), context(0))
    assert fusion.component_scores["R"] == 0.0
    assert fusion.decision == "high_risk" and fusion.override == "high_risk"
    agreeing = engine.fuse(ml_result("high_risk", 0.9), rule_result("elevate"), evidence(), context(0))
    assert agreeing.component_scores["R"] == 1.0


def test_disagreement_is_capped_and_confidence_clipped_at_zero() -> None:
    fusion = DecisionFusionEngine().fuse(ml_result("low_risk", 0.4), rule_result("elevate"), evidence(), context(4))
    assert fusion.component_scores["D"] == 1.0
    assert fusion.raw < 0 and fusion.confidence == 0.0


def test_low_confidence_without_override_is_uncertain() -> None:
    fusion = DecisionFusionEngine().fuse(ml_result("high_risk", 0.55), rule_result(None), evidence(), context(0))
    assert fusion.decision == "uncertain"
    assert priority(fusion).key == "review"
    assert recommend(fusion, rule_result(None)).startswith("Clinician review advised")


def test_suppress_withholds() -> None:
    fusion = DecisionFusionEngine().fuse(ml_result("low_risk", 0.01), rule_result("suppress"), evidence(), context(0))
    assert fusion.decision == "withheld"
    assert priority(fusion).key == "manual"
