"""ClinicalContextAgent (FR6): reconciles the three analysis results, names conflicts and drops any
sentence its cited sources do not support (TC-12). The draft comes from a writer: the template below, or the
Claude writer in agents/claude_writer.py, whose output goes through the same check."""

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from medintel.models import (
    Conflict,
    ContextSummary,
    DraftSentence,
    EvidenceItem,
    FeatureContribution,
    LLMUsage,
    MLResult,
    PatientCase,
    RuleResult,
)
from medintel.numeric import fixed2, number_text
from medintel.schema import CARDIOLOGY_KEYS, FIELDS, VALUE_LABELS, abnormal, format_value

log = logging.getLogger("medintel")
OVERRIDES = {"elevate": "high_risk", "suppress": "withheld"}
LABEL = {"high_risk": "high risk", "low_risk": "low risk", "withheld": "withheld", "uncertain": "uncertain"}
ACTION_VERB = {"flag": "flags", "elevate": "elevates", "suppress": "suppresses"}
RAISED_KEYS = ("systolic_bp", "total_cholesterol", "fasting_glucose", "creatinine", "heart_rate")
STRONG_EVIDENCE = 0.75

ID_PATTERN = re.compile(r"\b(?:KB-[A-Z]+-\d+|[RCPB]-\d+|TC-\d+)\b")
# Symptoms a summary may name only when a cited source mentions them (TC-12: "a symptom absent from the case, agent
# outputs and evidence"). Each entry lists the ways the same symptom is written; the team can extend it.
SYMPTOMS: dict[str, tuple[str, ...]] = {
    "chest pain": (r"chest (?:pain|tightness|pressure|discomfort)", r"angina"),
    "headache": (r"headache",),
    "palpitations": (r"palpitation", r"racing heart", r"heart (?:is )?racing"),
    "fainting": (r"faint", r"syncop", r"passed out", r"loss of consciousness", r"black(?:ed)? ?out"),
    "shortness of breath": (r"short(?:ness)? of breath", r"breathless", r"dyspn", r"trouble breathing"),
    "dizziness": (r"dizz", r"light-?headed"),
    "nausea or vomiting": (r"nause", r"vomit"),
    "fatigue": (r"fatigue", r"tired"),
    "swelling": (r"swelling", r"swollen", r"o?edema"),
    "fever": (r"fever", r"febrile"),
    "sweating": (r"sweat", r"diaphores"),
    "vision changes": (r"vision\b", r"blurred"),
    "numbness or weakness": (r"numb", r"weakness"),
    "cough": (r"cough",),
    "radiating pain": (
        r"radiat",
        r"(?:arm|jaw|neck|back) pain",
        r"(?:pain|discomfort)(?: or discomfort)? in the (?:arms?|jaw|neck|back|shoulders?)",
    ),
}
SYMPTOM_PATTERNS = {name: [re.compile(rf"\b{p}", re.IGNORECASE) for p in forms] for name, forms in SYMPTOMS.items()}
FIELD_ALIASES = {"exercise_angina": ("exercise angina", "exercise induced angina")}
NUMBER_PATTERN = re.compile(r"(?<![\w.:])\d+(?:\.\d+)?(?![\w:])")
CLAUSE_PATTERN = re.compile(r"[;:]|\b(?:but|while|whereas|although|though)\b", re.IGNORECASE)
NEGATION_PATTERN = re.compile(r"\b(?:not|no|none|nor|neither|without)\b", re.IGNORECASE)
ABSENCE_PATTERN = re.compile(r"\b(?:recorded|on record|documented|available)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Draft:
    sentences: list[DraftSentence]
    generated_by: str
    usage: LLMUsage | None = None


class DraftWriter(Protocol):
    def write(
        self,
        case: PatientCase,
        ml: MLResult,
        rules: RuleResult,
        evidence: list[EvidenceItem],
        conflicts: list[Conflict],
    ) -> Draft: ...


def find_conflicts(
    case: PatientCase,
    ml: MLResult,
    rules: RuleResult,
    evidence: list[EvidenceItem],
    strong_score: float = STRONG_EVIDENCE,
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    override = OVERRIDES.get(rules.action) if rules.action else None
    if override == "high_risk" and ml.prediction != "high_risk" and rules.top_rule:
        rule = rules.top_rule
        observed = f"{FIELDS[rule.field].label} {format_value(rule.field, case.x_s.get(rule.field))}"
        conflicts.append(
            Conflict(
                type="rule_vs_ml",
                text=f"ML predicts low risk, but rule {rule.rule_id} elevates the case ({observed}).",
            )
        )
    strong = [e for e in evidence if e.score >= strong_score and e.marked is not False]
    raising = [e.source_id for e in strong if e.direction == "raises"]
    lowering = [e.source_id for e in strong if e.direction == "lowers"]
    if ml.prediction == "low_risk" and raising:
        conflicts.append(
            Conflict(
                type="evidence_vs_ml",
                text=f"Retrieved evidence ({', '.join(raising)}) points to raised risk, while ML predicts low risk.",
            )
        )
    if ml.prediction == "high_risk" and lowering:
        conflicts.append(
            Conflict(
                type="evidence_vs_ml",
                text=f"Retrieved evidence ({', '.join(lowering)}) describes a low-risk profile, "
                "while ML predicts high risk.",
            )
        )
    return conflicts


def _values_sentence(case: PatientCase) -> DraftSentence:
    raised = [k for k in RAISED_KEYS if abnormal(k, case.x_s.get(k)) == "high"]
    if not raised:
        return DraftSentence(text="The recorded values are within the reference ranges.", sources=["record"])
    parts = [
        f"{FIELDS[k].label.lower().replace('bp', 'blood pressure', 1)} ({format_value(k, case.x_s.get(k))})"
        for k in raised
    ]
    if len(parts) == 1:
        text = f"{parts[0][0].upper()}{parts[0][1:]} is above the reference range."
    else:
        text = f"Raised values: {', '.join(parts)}."
    return DraftSentence(text=text, sources=[f"record:{k}" for k in raised])


def _rule_sentence(rules: RuleResult) -> DraftSentence:
    if rules.top_rule:
        rule = rules.top_rule
        return DraftSentence(
            text=f"Rule {rule.rule_id} {ACTION_VERB[rule.action]} the case: {rule.message}",
            sources=[f"rule:{rule.rule_id}"],
        )
    return DraftSentence(text="No rule was triggered.", sources=["rules"])


def _evidence_sentence(evidence: list[EvidenceItem]) -> DraftSentence:
    if evidence:
        return DraftSentence(
            text=f"Retrieved evidence: {', '.join(e.source_id for e in evidence)}.",
            sources=[f"evidence:{e.source_id}" for e in evidence],
        )
    return DraftSentence(
        text="No sufficiently relevant evidence was found in the knowledge base.", sources=["evidence"]
    )


def _conflict_sentence(conflicts: list[Conflict]) -> DraftSentence | None:
    if not conflicts:
        return None
    text = (
        "One conflict between the components is reported for review."
        if len(conflicts) == 1
        else f"{len(conflicts)} conflicts between the components are reported for review."
    )
    return DraftSentence(text=text, sources=["conflicts"])


class PrototypeTemplateWriter:
    """The Stage I concept prototype's template, kept unchanged for the golden parity tests."""

    def write(
        self,
        case: PatientCase,
        ml: MLResult,
        rules: RuleResult,
        evidence: list[EvidenceItem],
        conflicts: list[Conflict],
    ) -> Draft:
        out = [_values_sentence(case)]
        if case.x_s.get("smoker"):
            out.append(DraftSentence(text="The patient smokes.", sources=["record:smoker"]))
        out.append(
            DraftSentence(
                text=f"The ML model estimates a {LABEL[ml.prediction]} "
                f"(probability of high risk {fixed2(ml.probabilities['high_risk'])}).",
                sources=["ml"],
            )
        )
        out += [_rule_sentence(rules), _evidence_sentence(evidence)]
        conflict = _conflict_sentence(conflicts)
        return Draft(out + ([conflict] if conflict else []), "template")


VARIANT_TEXT = {"pre_test": "pre-test model (no exercise test on record)", "full": "full model"}


def lower_first(label: str) -> str:
    """Lowercase a label for use mid-sentence without breaking acronyms: "Resting ECG" becomes "resting ECG"."""
    return label[:1].lower() + label[1:]


def shown_value(c: FeatureContribution) -> str:
    if c.value is None:
        return "not recorded"
    if c.key == "sex" and c.value in ("M", "F"):
        return "male" if c.value == "M" else "female"
    if isinstance(c.value, str):
        return VALUE_LABELS.get(c.value, c.value)
    return f"{number_text(c.value)} {c.unit}".strip()


def _workup_sentence(case: PatientCase) -> DraftSentence | None:
    keys = [k for k in CARDIOLOGY_KEYS if case.x_s.get(k) is not None]
    if not keys:
        return None
    parts = [f"{lower_first(FIELDS[k].label)} {format_value(k, case.x_s[k])}" for k in keys]
    return DraftSentence(text=f"Work-up on record: {'; '.join(parts)}.", sources=[f"record:{k}" for k in keys])


def _model_sentences(ml: MLResult) -> list[DraftSentence]:
    p = fixed2(ml.probabilities["high_risk"])
    if not ml.variant:
        return [
            DraftSentence(
                text=f"The ML model estimates a {LABEL[ml.prediction]} (probability of high risk {p}).", sources=["ml"]
            )
        ]
    out = [
        DraftSentence(
            text=f"The {VARIANT_TEXT.get(ml.variant, ml.variant)} estimates a {LABEL[ml.prediction]} "
            f"(probability of high risk {p}).",
            sources=["ml"],
        )
    ]
    top = [c for c in ml.importance if c.contribution and not c.imputed][:3]
    if top:
        parts = [
            f"{lower_first(c.label)} ({shown_value(c)}, {'raises' if c.contribution > 0 else 'lowers'} the estimate)"
            for c in top
        ]
        out.append(DraftSentence(text=f"Largest model contributions: {'; '.join(parts)}.", sources=["ml"]))
    filled = [lower_first(c.label) for c in ml.contributions if c.imputed]
    if filled:
        out.append(
            DraftSentence(
                text=f"{len(filled)} of the model's {len(ml.contributions)} inputs were not recorded and were filled "
                f"in with training averages: {', '.join(filled)}.",
                sources=["ml"],
            )
        )
    return out


class TemplateWriter:
    """Deterministic summary used without an LLM, and as the fallback when the LLM cannot be reached."""

    def write(
        self,
        case: PatientCase,
        ml: MLResult,
        rules: RuleResult,
        evidence: list[EvidenceItem],
        conflicts: list[Conflict],
    ) -> Draft:
        out = [_values_sentence(case)]
        workup = _workup_sentence(case)
        if workup:
            out.append(workup)
        if case.x_s.get("smoker"):
            out.append(DraftSentence(text="The patient smokes.", sources=["record:smoker"]))
        out += _model_sentences(ml)
        out += [_rule_sentence(rules), _evidence_sentence(evidence)]
        conflict = _conflict_sentence(conflicts)
        return Draft(out + ([conflict] if conflict else []), "template")


class FixtureWriter:
    """Replays fixed drafts (the prototype's pre-written LLM text) where one exists; template otherwise."""

    def __init__(self, drafts: dict[str, list[DraftSentence]], fallback: DraftWriter | None = None) -> None:
        self.drafts = drafts
        self.fallback = fallback or PrototypeTemplateWriter()

    def write(
        self,
        case: PatientCase,
        ml: MLResult,
        rules: RuleResult,
        evidence: list[EvidenceItem],
        conflicts: list[Conflict],
    ) -> Draft:
        if case.patient_id in self.drafts:
            return Draft([s.model_copy() for s in self.drafts[case.patient_id]], "scripted")
        return self.fallback.write(case, ml, rules, evidence, conflicts)


def _numbers(text: str) -> list[float]:
    return [float(n) for n in NUMBER_PATTERN.findall(ID_PATTERN.sub(" ", text))]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def source_numbers(
    source: str,
    case: PatientCase,
    ml: MLResult,
    rules: RuleResult,
    evidence: list[EvidenceItem],
    conflicts: list[Conflict],
) -> list[float] | None:
    """Numbers a sentence may state when it cites `source`; None when the source does not exist."""
    kind, _, ref = source.partition(":")
    if kind == "record":
        if not ref:
            return [float(v) for v in case.x_s.values() if _is_number(v)]
        if ref not in case.x_s:
            return None
        value = case.x_s[ref]
        return [float(value)] if _is_number(value) else []
    if kind == "ml" and not ref:
        probabilities = list(ml.probabilities.values())
        inputs = ml.contributions
        counts = [float(len(inputs)), float(sum(c.imputed for c in inputs)), float(sum(not c.imputed for c in inputs))]
        values = [float(c.value) for c in inputs if _is_number(c.value)]
        label_numbers = [n for c in inputs for n in _numbers(c.label)]
        contributions = [c.contribution for c in inputs]
        return probabilities + [p * 100 for p in probabilities] + counts + values + label_numbers + contributions
    if kind == "rules" and not ref:
        return [float(len(rules.triggered))]
    if kind == "rule":
        fired = next((c.rule for c in rules.checks if c.fired and c.rule.rule_id == ref), None)
        if fired is None:
            return None
        return ([fired.threshold] if fired.threshold is not None else []) + _numbers(fired.message)
    if kind == "evidence":
        if not ref:
            return [float(len(evidence))]
        item = next((e for e in evidence if e.source_id == ref), None)
        if item is None:
            return None
        return [item.score, *_numbers(item.passage), *_numbers(item.matched_on)]
    if kind == "conflicts" and not ref:
        return [float(len(conflicts)), *(n for c in conflicts for n in _numbers(c.text))]
    if kind == "notes" and not ref:
        return _numbers(case.x_t)
    if kind == "events" and not ref:
        return [n for e in case.x_l for n in _numbers(e.text)]
    return None


def source_text(
    source: str,
    case: PatientCase,
    ml: MLResult,
    rules: RuleResult,
    evidence: list[EvidenceItem],
    conflicts: list[Conflict],
) -> str:
    """The words a cited source contains, for the symptom check."""
    kind, _, ref = source.partition(":")
    present = {k: v for k, v in case.x_s.items() if v is not None and k in FIELDS}
    if kind == "record":
        keys = [ref] if ref else list(present)
        return " ".join(f"{FIELDS[k].label} {format_value(k, present[k])}" for k in keys if k in present)
    if kind == "ml":
        return " ".join(f"{c.label} {shown_value(c)}" for c in ml.contributions)
    if kind == "rules":
        return " ".join(c.rule.message for c in rules.checks if c.fired)
    if kind == "rule":
        return " ".join(c.rule.message for c in rules.checks if c.fired and c.rule.rule_id == ref)
    if kind == "evidence":
        return " ".join(f"{e.title} {e.passage} {e.matched_on}" for e in evidence if not ref or e.source_id == ref)
    if kind == "conflicts":
        return " ".join(c.text for c in conflicts)
    if kind == "notes":
        return case.x_t
    if kind == "events":
        return " ".join(e.text for e in case.x_l)
    return ""


def unsupported_symptoms(text: str, cited: str) -> list[str]:
    named = [n for n, patterns in SYMPTOM_PATTERNS.items() if any(p.search(text) for p in patterns)]
    return [n for n in named if not any(p.search(cited) for p in SYMPTOM_PATTERNS[n])]


def without_missing_fields(text: str, case: PatientCase) -> str:
    """Drop the names of unrecorded fields from clauses saying they are missing ("chest pain type was not recorded"),
    so the symptom check does not read a field's name as a symptom the patient has."""
    missing = [
        re.compile(re.escape(name), re.IGNORECASE)
        for k, f in FIELDS.items()
        if case.x_s.get(k) is None
        for name in (f.label, *FIELD_ALIASES.get(k, ()))
    ]
    clauses = CLAUSE_PATTERN.split(text)
    for i, clause in enumerate(clauses):
        if NEGATION_PATTERN.search(clause) and ABSENCE_PATTERN.search(clause):
            for label in missing:
                clause = label.sub(" ", clause)
            clauses[i] = clause
    return " ".join(clauses)


def verify_sentence(
    sentence: DraftSentence,
    case: PatientCase,
    ml: MLResult,
    rules: RuleResult,
    evidence: list[EvidenceItem],
    conflicts: list[Conflict],
) -> DraftSentence:
    problems: list[str] = []
    if not sentence.sources:
        problems.append("it cites no source")
    allowed: list[float] = []
    for source in sentence.sources:
        values = source_numbers(source, case, ml, rules, evidence, conflicts)
        if values is None:
            problems.append(f"it cites {source}, which this case does not contain")
        else:
            allowed.extend(values)
    for token in NUMBER_PATTERN.findall(ID_PATTERN.sub(" ", sentence.text)):
        decimals = len(token.partition(".")[2])
        tolerance = 0.5 * 10**-decimals + 1e-9
        if not any(abs(float(token) - value) <= tolerance for value in allowed):
            problems.append(f"the number {token} does not appear in the cited sources")
    cited = " ".join(source_text(s, case, ml, rules, evidence, conflicts) for s in sentence.sources)
    for symptom in unsupported_symptoms(without_missing_fields(sentence.text, case), cited):
        problems.append(f"it names {symptom}, which its cited sources do not mention")
    if problems:
        return sentence.model_copy(update={"supported": False, "reason": "Not supported: " + "; ".join(problems) + "."})
    return sentence.model_copy(update={"supported": True, "reason": None})


class ClinicalContextAgent:
    def __init__(self, writer: DraftWriter | None = None, strong_score: float = STRONG_EVIDENCE) -> None:
        self.writer = writer or TemplateWriter()
        self.strong_score = strong_score

    def synthesize(
        self, case: PatientCase, ml: MLResult, rules: RuleResult, evidence: list[EvidenceItem]
    ) -> ContextSummary:
        conflicts = find_conflicts(case, ml, rules, evidence, self.strong_score)
        draft = self.writer.write(case, ml, rules, evidence, conflicts)
        checked = [verify_sentence(s, case, ml, rules, evidence, conflicts) for s in draft.sentences]
        removed = [s for s in checked if not s.supported]
        if removed:
            log.warning("TC-12: removed %d unsupported statement(s) from %s", len(removed), case.case_id)
        return ContextSummary(
            summary=" ".join(s.text for s in checked if s.supported),
            conflicts=[c.text for c in conflicts],
            conflict_details=conflicts,
            unsupported_statements=[s.text for s in checked if not s.supported],
            draft=checked,
            generated_by=draft.generated_by,
            usage=draft.usage,
        )
