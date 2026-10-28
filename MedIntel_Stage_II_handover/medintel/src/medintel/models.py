"""Typed output contracts of Table 4.2.1 and the DecisionReport of Section 4.4.2."""

from typing import Literal

from pydantic import BaseModel, Field

from medintel.schema import Value

Prediction = Literal["high_risk", "low_risk"]
Decision = Literal["high_risk", "low_risk", "uncertain", "withheld"]
Override = Literal["high_risk", "withheld"]
Action = Literal["elevate", "flag", "suppress"]
Status = Literal["pending_review", "accepted", "rejected", "annotated", "failed"]
Direction = Literal["raises", "lowers", "neutral"]


class ValidationProblem(BaseModel):
    field: str
    message: str
    test_case: str | None = Field(default=None, description="Planned test case this check covers, e.g. TC-01.")


class RejectedRecord(BaseModel):
    row: int
    patient_id: str
    problems: list[ValidationProblem]


class Event(BaseModel):
    time: str
    text: str


class PatientCase(BaseModel):
    case_id: str
    patient_id: str
    x_s: dict[str, Value] = Field(description="Structured features.")
    x_t: str = Field(default="", description="Free-text clinical notes.")
    x_l: list[Event] = Field(default_factory=list, description="Event log joined by patient identifier.")
    source: str


class FeatureContribution(BaseModel):
    key: str
    label: str
    unit: str
    value: float | str | None = Field(description="Recorded value; None when not recorded and filled in.")
    imputed: bool
    contribution: float = Field(description="Change in the risk logit against the model's reference profile.")


class MLResult(BaseModel):
    prediction: Prediction
    probabilities: dict[Prediction, float]
    logit: float
    contributions: list[FeatureContribution]
    importance: list[FeatureContribution] = Field(description="Contributions sorted by absolute size.")
    model_version: str
    variant: str = Field(default="", description="Submodel that scored the case: pre_test or full.")


class Rule(BaseModel):
    rule_id: str
    field: str
    operator: Literal[">", ">=", "<", "<=", "missing"]
    threshold: float | None
    action: Action
    severity: Literal["low", "medium", "high"]
    message: str


class RuleCheck(BaseModel):
    rule: Rule
    value: Value
    fired: bool


class RuleResult(BaseModel):
    triggered: list[str]
    flags: list[str]
    action: Action | None
    reason: str
    top_rule: Rule | None
    checks: list[RuleCheck]
    rulebase_version: str


class EvidenceItem(BaseModel):
    source_id: str
    title: str
    topic: str
    passage: str
    direction: Direction
    score: float
    matched_on: str
    marked: bool | None = Field(
        default=None,
        description="Matched by a finding past a threshold stated in the knowledge base or rulebase; "
        "None when the store does not say.",
    )


class DraftSentence(BaseModel):
    text: str
    sources: list[str] = Field(
        description='What the sentence relies on: "record", "record:<field>", "ml", "rules", "rule:<id>", '
        '"evidence", "evidence:<source_id>", "conflicts", "notes" or "events".'
    )
    supported: bool = True
    reason: str | None = None


class Conflict(BaseModel):
    type: Literal["rule_vs_ml", "evidence_vs_ml"]
    text: str


class LLMUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int = Field(description="Includes thinking tokens, which are billed as output.")
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = Field(description="Estimate from token counts and list prices.")
    latency_s: float | None = None
    request_id: str | None = None
    batch: bool = False


class ContextSummary(BaseModel):
    summary: str
    conflicts: list[str]
    conflict_details: list[Conflict]
    unsupported_statements: list[str]
    draft: list[DraftSentence]
    generated_by: str
    usage: LLMUsage | None = None


class FusionResult(BaseModel):
    decision: Decision
    confidence: float
    raw: float
    disagreement: float
    component_scores: dict[str, float]
    terms: dict[str, float]
    threshold: float
    weights: dict[str, float]
    override: Override | None
    below_threshold: bool


class Priority(BaseModel):
    level: int
    key: Literal["escalate", "review", "priority", "manual", "routine", "failed"]
    label: str


class Versions(BaseModel):
    model: str
    rulebase: str
    index: str


class DecisionReport(BaseModel):
    case_id: str
    patient_id: str
    findings: list[str]
    prediction: str
    decision: str
    triggered_rules: list[str]
    evidence_refs: list[str]
    conflicts: list[str]
    confidence: float
    component_scores: dict[str, float]
    summary: str
    recommendation: str
    status: Status
    notes: str = ""
    versions: Versions


class AuditRecord(BaseModel):
    case_id: str
    model_version: str
    rulebase_version: str
    retrieval_ids: str
    status: Status
    timestamp: str
    event: str = Field(default="report_created", description="What happened, e.g. report_created or review_accepted.")
    actor: str = Field(default="system", description="Who did it: system, or the reviewer's name or role.")


class CaseAnalysis(BaseModel):
    case: PatientCase
    ml: MLResult
    rules: RuleResult
    evidence: list[EvidenceItem]
    context: ContextSummary
    fusion: FusionResult
    recommendation: str
    priority: Priority
    report: DecisionReport
    timings: dict[str, float] = Field(
        default_factory=dict, description="Seconds per stage: ml, rules, retrieval, context, fusion, total (6.5.1)."
    )
