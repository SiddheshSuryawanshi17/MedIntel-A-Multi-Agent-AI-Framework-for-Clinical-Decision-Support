"""DecisionFusionEngine (FR7) and ExplanationGenerator (FR8): Equation (4.2), rule precedence, the report."""

from pydantic import BaseModel

from medintel.models import (
    ContextSummary,
    DecisionReport,
    EvidenceItem,
    FusionResult,
    MLResult,
    PatientCase,
    Priority,
    RuleResult,
    Status,
    Versions,
)
from medintel.numeric import fixed2, mean, number_text, round3

OVERRIDES = {"elevate": "high_risk", "suppress": "withheld"}
RULE_HINTS = {
    "R-001": "Confirm the blood pressure with a repeat reading.",
    "R-004": "Check the rhythm and look for an acute cause.",
}
LABEL = {"high_risk": "high risk", "low_risk": "low risk", "withheld": "withheld", "uncertain": "uncertain"}


class FusionSettings(BaseModel):
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    lam: float = 0.5
    threshold: float = 0.6


class DecisionFusionEngine:
    def __init__(self, settings: FusionSettings | None = None) -> None:
        self.settings = settings or FusionSettings()

    def fuse(
        self, ml: MLResult, rules: RuleResult, evidence: list[EvidenceItem], context: ContextSummary
    ) -> FusionResult:
        s = self.settings
        override = OVERRIDES.get(rules.action) if rules.action else None
        m = ml.probabilities[ml.prediction]
        r = 0.0 if override and override != ml.prediction else 1.0
        k = mean([e.score for e in evidence])
        d = min(1.0, len(context.conflicts) / 3)
        terms = {"aM": s.alpha * m, "bR": s.beta * r, "gK": s.gamma * k, "lD": s.lam * d}
        raw = terms["aM"] + terms["bR"] + terms["gK"] - terms["lD"]
        confidence = max(0.0, raw)
        below = confidence < s.threshold
        return FusionResult(
            decision=override or ("uncertain" if below else ml.prediction),
            confidence=round3(confidence),
            raw=round3(raw),
            disagreement=round3(d),
            component_scores={"M": round3(m), "R": r, "K": round3(k), "D": round3(d)},
            terms={name: round3(value) for name, value in terms.items()},
            threshold=s.threshold,
            weights={"alpha": s.alpha, "beta": s.beta, "gamma": s.gamma, "lam": s.lam},
            override=override,
            below_threshold=below,
        )


def recommend(fusion: FusionResult, rules: RuleResult) -> str:
    top = rules.top_rule
    hint = f" {RULE_HINTS[top.rule_id]}" if top and top.rule_id in RULE_HINTS else ""
    if fusion.decision == "withheld":
        return (
            "Recommendation withheld: the risk model is intended for adults. Assess against pediatric reference ranges."
        )
    if fusion.decision == "high_risk" and fusion.override and fusion.below_threshold and top:
        return f"Escalate for same-day clinician review; rule {top.rule_id} overrides the model.{hint}"
    if fusion.decision == "high_risk":
        return f"Prioritize clinician review today.{hint}"
    if fusion.decision == "uncertain":
        return "Clinician review advised: the evidence is not strong enough for a confident recommendation."
    return f"Routine follow-up once the flagged item is resolved.{hint}" if rules.flags else "Routine follow-up."


def priority(fusion: FusionResult) -> Priority:
    if fusion.decision == "high_risk" and fusion.below_threshold:
        return Priority(level=1, key="escalate", label="Escalate")
    if fusion.decision == "uncertain":
        return Priority(level=2, key="review", label="Review")
    if fusion.decision == "high_risk":
        return Priority(level=3, key="priority", label="Priority")
    if fusion.decision == "withheld":
        return Priority(level=4, key="manual", label="Assess manually")
    return Priority(level=5, key="routine", label="Routine")


def decision_steps(fusion: FusionResult, ml: MLResult, rules: RuleResult, context: ContextSummary) -> list[str]:
    top = rules.top_rule
    c, t = fixed2(fusion.confidence), fixed2(fusion.threshold)
    steps: list[str] = []
    if fusion.override == "high_risk" and top:
        steps.append(
            f"Rule {top.rule_id} fired with action elevate, so the decision is high risk. "
            "A triggered rule always takes precedence over the learned prediction."
        )
    elif fusion.override == "withheld" and top:
        steps.append(f"Rule {top.rule_id} fired with action suppress, so the recommendation is withheld.")
    else:
        steps.append(
            f"No rule overrides the model, so the decision follows the ML prediction ({LABEL[ml.prediction]}) "
            "when C reaches the threshold."
        )
    if fusion.override == "high_risk" and fusion.below_threshold:
        steps.append(
            f"C = {c} is below the threshold {t}, so the case is marked low confidence and escalated "
            "for same-day review."
        )
    elif fusion.override:
        steps.append(f"C = {c} is reported with the decision for the clinician's information.")
    elif fusion.below_threshold:
        steps.append(f"C = {c} is below the threshold {t}, so the decision is reported as uncertain.")
    else:
        steps.append(f"C = {c} is at or above the threshold {t}, so the prediction is reported with normal confidence.")
    n = len(context.conflicts)
    if fusion.component_scores["D"] > 0:
        steps.append(
            f"{n} reported conflict{'s' if n != 1 else ''} set the disagreement D to "
            f"{fixed2(fusion.component_scores['D'])}, which subtracts {fixed2(fusion.terms['lD'])} from C."
        )
    if fusion.component_scores["K"] == 0:
        steps.append("No sufficiently relevant evidence was retrieved, so retrieval support K is 0.")
    return steps


def findings(case: PatientCase, ml: MLResult, rules: RuleResult) -> list[str]:
    """The three inputs that moved the model most (filled-in inputs move it by nothing), then the rule flags."""
    out: list[str] = []
    for feature in [f for f in ml.importance if not f.imputed or f.contribution][:3]:
        key = "sex" if feature.key == "sex_m" else feature.key
        raw = case.x_s.get(key)
        if feature.key == "sex_m":
            value = str(case.x_s.get("sex"))
        elif isinstance(raw, bool) or feature.key == "smoker":
            value = "yes" if raw else "no"
        else:
            value = "missing" if raw is None else number_text(raw) if not isinstance(raw, str) else raw
        out.append(f"{key} = {value}")
    return out + rules.flags


class ExplanationGenerator:
    def build_report(
        self,
        case: PatientCase,
        ml: MLResult,
        rules: RuleResult,
        evidence: list[EvidenceItem],
        context: ContextSummary,
        fusion: FusionResult,
        recommendation: str,
        versions: Versions,
        status: Status = "pending_review",
        notes: str = "",
    ) -> DecisionReport:
        return DecisionReport(
            case_id=case.case_id,
            patient_id=case.patient_id,
            findings=findings(case, ml, rules),
            prediction=ml.prediction,
            decision=fusion.decision,
            triggered_rules=rules.triggered,
            evidence_refs=[e.source_id for e in evidence],
            conflicts=context.conflicts,
            confidence=fusion.confidence,
            component_scores=fusion.component_scores,
            summary=context.summary,
            recommendation=recommendation,
            status=status,
            notes=notes,
            versions=versions,
        )

    def failed_report(self, case: PatientCase, category: str, versions: Versions) -> DecisionReport:
        return DecisionReport(
            case_id=case.case_id,
            patient_id=case.patient_id,
            findings=[],
            prediction="unavailable",
            decision="failed",
            triggered_rules=[],
            evidence_refs=[],
            conflicts=[],
            confidence=0.0,
            component_scores={},
            summary="",
            recommendation=f"Analysis failed ({category}); assess this patient manually.",
            status="failed",
            versions=versions,
        )
