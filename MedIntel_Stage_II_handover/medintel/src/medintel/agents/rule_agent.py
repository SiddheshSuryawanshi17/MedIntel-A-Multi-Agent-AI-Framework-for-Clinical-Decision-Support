"""RuleValidationAgent (FR4): deterministic checks against the versioned rulebase of Section 4.4.1."""

import json
import operator
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel

from medintel.models import PatientCase, Rule, RuleCheck, RuleResult
from medintel.schema import Value

OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}
SEVERITY = {"low": 1, "medium": 2, "high": 3}
PRECEDENCE = {"flag": 0, "elevate": 1, "suppress": 2}


def default_rulebase_path() -> Path:
    return Path(str(files("medintel") / "resources" / "rulebase.json"))


class RuleBase(BaseModel):
    version: str
    approved_by: str = ""
    rules: list[Rule]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RuleBase":
        source = Path(path) if path else default_rulebase_path()
        return cls.model_validate(json.loads(source.read_text(encoding="utf-8")))


def fires(rule: Rule, value: Value) -> bool:
    if rule.operator == "missing":
        return value is None
    if value is None or isinstance(value, (bool, str)) or rule.threshold is None:
        return False
    return OPERATORS[rule.operator](value, rule.threshold)


class RuleValidationAgent:
    def __init__(self, rulebase: RuleBase | None = None) -> None:
        self.rulebase = rulebase or RuleBase.load()

    def evaluate(self, case: PatientCase) -> RuleResult:
        checks = [
            RuleCheck(rule=rule, value=case.x_s.get(rule.field), fired=fires(rule, case.x_s.get(rule.field)))
            for rule in self.rulebase.rules
        ]
        hits = [check.rule for check in checks if check.fired]
        top = max(hits, key=lambda r: (SEVERITY[r.severity], PRECEDENCE[r.action]), default=None)
        return RuleResult(
            triggered=[r.rule_id for r in hits],
            flags=[r.message for r in hits],
            action=top.action if top else None,
            reason=top.message if top else "no rule triggered",
            top_rule=top,
            checks=checks,
            rulebase_version=self.rulebase.version,
        )
