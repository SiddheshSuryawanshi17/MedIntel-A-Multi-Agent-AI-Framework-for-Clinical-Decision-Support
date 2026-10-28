"""Claude writer for the Clinical Context Agent (FR6).

Claude receives only the structured outputs of the other agents and must name a source for every sentence. The
TC-12 check in context_agent.py then removes any sentence its sources do not support, so Claude's word is never
taken on its own. When Claude cannot be reached or its reply is unusable, the template writes the summary and the
report says why (NFR5). Case content leaves the machine: use synthetic or de-identified records only."""

import hashlib
import json
import os
import time
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from medintel.agents.context_agent import LABEL, Draft, DraftWriter, TemplateWriter, shown_value
from medintel.config import ContextSettings
from medintel.models import Conflict, DraftSentence, EvidenceItem, LLMUsage, MLResult, PatientCase, RuleResult
from medintel.schema import FIELDS

# USD per million tokens: input, output, cache write (5-minute), cache read. List prices; the Batch API halves them.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}

SYSTEM_PROMPT = """You write the clinical context summary for MedIntel, a research prototype for cardiovascular \
decision support. A clinician reviews every report; MedIntel is not a medical device.

The user message is a JSON object with the facts of one case, grouped by source key:
- "record": the patient's recorded values. Cite "record:<field>" for a value, or "record" for the record as a whole. \
"not_recorded" lists fields with no value.
- "notes" and "events": free text from the record. Cite "notes" or "events".
- "ml": the risk model's estimate, the inputs that moved it most and any inputs it had to fill in. Cite "ml".
- "rules": rules that fired. Cite "rule:<id>", or "rules" when none fired.
- "evidence": knowledge-base passages that passed the relevance threshold. Cite "evidence:<id>", or "evidence" \
when none did.
- "conflicts": disagreements between components that MedIntel has already detected. Cite "conflicts".
"allowed_sources" lists every source key you may cite for this case.

Write 3 to 6 sentences in plain clinical English for a busy clinician, covering in this order:
1. The notable recorded findings: abnormal values, symptoms in the notes, work-up results.
2. The model's estimate and what drove it, and which inputs were not recorded and were filled in, if any.
3. Each rule that fired and what it does to the case.
4. What the retrieved evidence says about this patient, naming passages by their ID, or that none was relevant \
enough.
5. Each conflict in plain words, if there is one.

Rules:
- Every sentence lists in "sources" the keys its content comes from, using only keys in allowed_sources. Prefer the \
specific key ("record:systolic_bp", "rule:R-001", "evidence:KB-HTN-001") over a general one.
- Every number you write must appear, with the same value, in the facts of a source the sentence cites. You may \
round a probability to two decimals or give it as a percentage. Do not compute new numbers: no differences, \
ratios or counts that are not given.
- Use only the facts given. Do not add symptoms, history, diagnoses, causes, medicines, treatment or follow-up \
advice, and do not state the final decision or priority; MedIntel's fusion step decides those.
- If a value is not recorded, say so; never guess it.
- Write rule and evidence IDs exactly as given, for example R-001 and KB-HTN-001."""

PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "sources"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sentences"],
    "additionalProperties": False,
}


class OutputSentence(BaseModel):
    text: str
    sources: list[str]


class DraftOutput(BaseModel):
    sentences: list[OutputSentence]


def allowed_sources(
    case: PatientCase, rules: RuleResult, evidence: list[EvidenceItem], conflicts: list[Conflict]
) -> list[str]:
    out = ["record"] + [f"record:{k}" for k, v in case.x_s.items() if v is not None and k in FIELDS] + ["ml"]
    fired = [c.rule.rule_id for c in rules.checks if c.fired]
    out += [f"rule:{r}" for r in fired] or ["rules"]
    out += [f"evidence:{e.source_id}" for e in evidence] or ["evidence"]
    if conflicts:
        out.append("conflicts")
    if case.x_t.strip():
        out.append("notes")
    if case.x_l:
        out.append("events")
    return out


def case_facts(
    case: PatientCase, ml: MLResult, rules: RuleResult, evidence: list[EvidenceItem], conflicts: list[Conflict]
) -> dict[str, Any]:
    record = {
        k: {"label": FIELDS[k].label, "value": v, **({"unit": FIELDS[k].unit} if FIELDS[k].unit else {})}
        for k, v in case.x_s.items()
        if v is not None and k in FIELDS
    }
    top = [c for c in ml.importance if c.contribution and not c.imputed][:5]
    return {
        "case_id": case.case_id,
        "record": record,
        "not_recorded": [FIELDS[k].label for k, v in case.x_s.items() if v is None and k in FIELDS],
        "notes": case.x_t.strip(),
        "events": [f"{e.time} {e.text}".strip() for e in case.x_l],
        "ml": {
            "submodel": {"pre_test": "pre-test model (no exercise test on record)", "full": "full model"}.get(
                ml.variant, "demo-weight model"
            ),
            "probability_high_risk": ml.probabilities["high_risk"],
            "prediction": LABEL[ml.prediction],
            "largest_contributions": [
                {
                    "input": c.label,
                    "value": shown_value(c),
                    "effect": "raises the estimate" if c.contribution > 0 else "lowers the estimate",
                    "logit_contribution": c.contribution,
                }
                for c in top
            ],
            "inputs_used": sum(not c.imputed for c in ml.contributions),
            "inputs_filled_in": [c.label for c in ml.contributions if c.imputed],
        },
        "rules": [
            {"id": c.rule.rule_id, "action": c.rule.action, "severity": c.rule.severity, "message": c.rule.message}
            for c in rules.checks
            if c.fired
        ],
        "evidence": [
            {
                "id": e.source_id,
                "title": e.title,
                "direction": e.direction,
                "relevance_score": e.score,
                "matched_on": e.matched_on,
                "passage": e.passage,
            }
            for e in evidence
        ],
        "conflicts": [c.text for c in conflicts],
        "allowed_sources": allowed_sources(case, rules, evidence, conflicts),
    }


def estimate_cost(model: str, usage: Any, batch: bool = False) -> float:
    price_in, price_out, price_write, price_read = PRICES.get(model, PRICES["claude-opus-5-5"])
    cost = (
        (usage.input_tokens or 0) * price_in
        + (usage.output_tokens or 0) * price_out
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * price_write
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * price_read
    ) / 1e6
    return round(cost * (0.5 if batch else 1.0), 6)


def usage_record(model: str, usage: Any, latency: float | None, request_id: str | None, batch: bool) -> LLMUsage:
    return LLMUsage(
        model=model,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cost_usd=estimate_cost(model, usage, batch),
        latency_s=None if latency is None else round(latency, 2),
        request_id=request_id,
        batch=batch,
    )


def parse_reply(message: Any) -> tuple[list[DraftSentence] | None, str]:
    """Sentences from a structured reply, or None with the reason it cannot be used."""
    if message.stop_reason in ("refusal", "max_tokens"):
        return None, f"stop reason {message.stop_reason}"
    text = next((b.text for b in message.content if getattr(b, "type", "") == "text"), None)
    if text is None:
        return None, "no text in the reply"
    try:
        parsed = DraftOutput.model_validate_json(text)
    except ValidationError:
        return None, "reply did not match the schema"
    sentences = [DraftSentence(text=s.text.strip(), sources=s.sources) for s in parsed.sentences if s.text.strip()]
    return (sentences, "") if sentences else (None, "empty reply")


class ClaudeWriter:
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "low",
        max_tokens: int = 8000,
        timeout_s: float = 90.0,
        client: Any = None,
        fallback: DraftWriter | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallback = fallback or TemplateWriter()
        if client is None and os.environ.get("ANTHROPIC_API_KEY"):
            client = anthropic.Anthropic(timeout=timeout_s, max_retries=2)
        self.client = client

    @property
    def name(self) -> str:
        return f"{self.model} (effort {self.effort}, prompt {PROMPT_VERSION})"

    def request(self, facts: dict[str, Any]) -> dict[str, Any]:
        """Messages API parameters for one case; also used for Batch API requests."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": json.dumps(facts, ensure_ascii=False)}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        }

    def _fallback(self, reason: str, usage: LLMUsage | None, *args: Any) -> Draft:
        draft = self.fallback.write(*args)
        return Draft(draft.sentences, f"template ({reason})", usage)

    def write(
        self,
        case: PatientCase,
        ml: MLResult,
        rules: RuleResult,
        evidence: list[EvidenceItem],
        conflicts: list[Conflict],
    ) -> Draft:
        args = (case, ml, rules, evidence, conflicts)
        if self.client is None:
            return self._fallback("no ANTHROPIC_API_KEY", None, *args)
        started = time.monotonic()
        try:
            message = self.client.messages.create(**self.request(case_facts(*args)))
        except anthropic.AnthropicError as exc:
            return self._fallback(f"Claude unavailable: {type(exc).__name__}", None, *args)
        usage = usage_record(
            self.model, message.usage, time.monotonic() - started, getattr(message, "_request_id", None), False
        )
        sentences, problem = parse_reply(message)
        if sentences is None:
            return self._fallback(f"Claude reply unusable: {problem}", usage, *args)
        return Draft(sentences, self.name, usage)


def create_writer(settings: ContextSettings) -> DraftWriter:
    if settings.writer == "claude":
        return ClaudeWriter(settings.model, settings.effort, settings.max_tokens, settings.timeout_s)
    return TemplateWriter()
