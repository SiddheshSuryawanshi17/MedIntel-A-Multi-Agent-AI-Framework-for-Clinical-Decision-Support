"""KnowledgeRetrievalAgent (FR5). Phase 0 matches a demo knowledge base; Phase 2 uses embeddings and FAISS."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from medintel.models import Direction, EvidenceItem, PatientCase
from medintel.numeric import number_text, round3
from medintel.schema import Value

Features = dict[str, Value]


def _num(x: Features, key: str) -> float | None:
    value = x.get(key)
    if value is None or isinstance(value, (bool, str)):
        return None
    return float(value)


def _at_least(x: Features, key: str, bound: float) -> bool:
    value = _num(x, key)
    return value is not None and value >= bound


def _below(x: Features, key: str, bound: float) -> bool:
    value = _num(x, key)
    return value is not None and value < bound


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _text(x: Features, key: str) -> str:
    value = x.get(key)
    return "missing" if value is None else number_text(value) if not isinstance(value, str) else value


@dataclass(frozen=True)
class DemoPassage:
    source_id: str
    title: str
    topic: str
    direction: Direction
    passage: str
    match: Callable[[Features], bool]
    score: Callable[[Features], float]
    matched_on: Callable[[Features], str]


DEMO_PASSAGES: tuple[DemoPassage, ...] = (
    DemoPassage(
        "KB-HTN-001",
        "Severely raised blood pressure",
        "Blood pressure",
        "raises",
        "A systolic reading of 180 mmHg or higher needs prompt clinical review and a repeat measurement "
        "to confirm the value before any decision is made.",
        lambda x: _at_least(x, "systolic_bp", 180) or _at_least(x, "diastolic_bp", 120),
        lambda x: _clamp(0.86 + 0.002 * ((_num(x, "systolic_bp") or 0) - 180), 0.86, 0.95),
        lambda x: f"systolic BP {_text(x, 'systolic_bp')} mmHg",
    ),
    DemoPassage(
        "KB-HTN-004",
        "Raised blood pressure and cardiovascular risk",
        "Blood pressure",
        "raises",
        "Blood pressure that stays at 140/90 mmHg or above increases long-term cardiovascular risk and is "
        "usually managed with lifestyle changes and, where indicated, medication.",
        lambda x: _at_least(x, "systolic_bp", 140) or _at_least(x, "diastolic_bp", 90),
        lambda x: _clamp(0.70 + 0.004 * (max(_num(x, "systolic_bp") or 0, 140) - 140), 0.70, 0.84),
        lambda x: f"BP {_text(x, 'systolic_bp')}/{_text(x, 'diastolic_bp')} mmHg",
    ),
    DemoPassage(
        "KB-LIP-007",
        "High total cholesterol",
        "Lipids",
        "raises",
        "Total cholesterol of 240 mg/dL or more is considered high and adds to cardiovascular risk alongside "
        "blood pressure, smoking and diabetes.",
        lambda x: _at_least(x, "total_cholesterol", 240),
        lambda x: _clamp(0.72 + 0.002 * ((_num(x, "total_cholesterol") or 0) - 240), 0.72, 0.82),
        lambda x: f"total cholesterol {_text(x, 'total_cholesterol')} mg/dL",
    ),
    DemoPassage(
        "KB-SMK-003",
        "Smoking and cardiovascular risk",
        "Smoking",
        "raises",
        "Current smoking substantially raises cardiovascular risk. Risk falls steadily after stopping, so "
        "smoking status should be recorded at every visit.",
        lambda x: x.get("smoker") is True,
        lambda x: 0.79,
        lambda x: "current smoker",
    ),
    DemoPassage(
        "KB-DM-010",
        "Raised fasting glucose",
        "Glucose",
        "raises",
        "A fasting glucose of 126 mg/dL or higher, confirmed on repeat testing, is consistent with diabetes, "
        "which increases cardiovascular risk.",
        lambda x: _at_least(x, "fasting_glucose", 126),
        lambda x: _clamp(0.74 + 0.002 * ((_num(x, "fasting_glucose") or 0) - 126), 0.74, 0.82),
        lambda x: f"fasting glucose {_text(x, 'fasting_glucose')} mg/dL",
    ),
    DemoPassage(
        "KB-CKD-004",
        "Kidney function and cardiovascular risk",
        "Kidney",
        "raises",
        "Reduced kidney function, often seen first as a raised serum creatinine, is linked to higher "
        "cardiovascular risk and changes how some medicines are dosed.",
        lambda x: _at_least(x, "creatinine", 1.5),
        lambda x: 0.71,
        lambda x: f"creatinine {_text(x, 'creatinine')} mg/dL",
    ),
    DemoPassage(
        "KB-CKD-009",
        "Missing kidney function results",
        "Kidney",
        "neutral",
        "When serum creatinine is not available, kidney-related risk checks cannot be completed and the "
        "result should be requested before medication decisions.",
        lambda x: x.get("creatinine") is None,
        lambda x: 0.62,
        lambda x: "creatinine missing",
    ),
    DemoPassage(
        "KB-HR-005",
        "Raised resting heart rate",
        "Heart rate",
        "neutral",
        "A resting heart rate that stays above 100 beats per minute should be evaluated for an underlying "
        "cause, such as an arrhythmia, infection or dehydration.",
        lambda x: (_num(x, "heart_rate") or 0) > 100,
        lambda x: _clamp(0.68 + 0.003 * ((_num(x, "heart_rate") or 0) - 100), 0.68, 0.80),
        lambda x: f"heart rate {_text(x, 'heart_rate')} bpm",
    ),
    DemoPassage(
        "KB-PED-002",
        "Adult risk tools in patients under 18",
        "Pediatrics",
        "neutral",
        "Risk tools developed for adults should not be applied to patients under 18. Pediatric assessment "
        "uses age-specific reference ranges.",
        lambda x: _below(x, "age", 18),
        lambda x: 0.88,
        lambda x: f"age {_text(x, 'age')}",
    ),
    DemoPassage(
        "KB-GEN-001",
        "Low-risk cardiovascular profile",
        "General",
        "lowers",
        "Normal blood pressure, cholesterol and fasting glucose without smoking are associated with a low "
        "short-term cardiovascular risk in adults.",
        lambda x: (
            _at_least(x, "age", 18)
            and _below(x, "systolic_bp", 130)
            and _below(x, "diastolic_bp", 85)
            and _below(x, "total_cholesterol", 200)
            and _below(x, "fasting_glucose", 100)
            and x.get("smoker") is False
        ),
        lambda x: 0.78,
        lambda x: "BP, cholesterol and glucose in range; non-smoker",
    ),
)


STOPWORDS = frozenset(
    "the and for with that this are was were from into have has had not but any can should when which who "
    "what how why about over under than then there their them they its also such per".split()
)


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


class KnowledgeStore(Protocol):
    version: str

    def search(self, case: PatientCase, k: int, min_score: float) -> list[EvidenceItem]: ...


class DemoKnowledgeStore:
    """Ten illustrative guideline summaries matched by rule. Not clinical guidance."""

    version = "kb-v3"

    def __init__(self, passages: tuple[DemoPassage, ...] = DEMO_PASSAGES) -> None:
        self.passages = passages

    def search(self, case: PatientCase, k: int, min_score: float) -> list[EvidenceItem]:
        x = case.x_s
        hits = [
            EvidenceItem(
                source_id=p.source_id,
                title=p.title,
                topic=p.topic,
                passage=p.passage,
                direction=p.direction,
                score=round3(p.score(x)),
                matched_on=p.matched_on(x),
            )
            for p in self.passages
            if p.match(x)
        ]
        relevant = sorted((e for e in hits if e.score >= min_score), key=lambda e: -e.score)
        return relevant[:k]

    def search_text(self, query: str, k: int) -> list[EvidenceItem]:
        """Keyword overlap between the query and each passage; Phase 2 replaces it with embedding search."""
        wanted = _terms(query)
        if not wanted:
            return []
        hits: list[EvidenceItem] = []
        for p in self.passages:
            shared = wanted & _terms(f"{p.title} {p.topic} {p.passage}")
            if shared:
                hits.append(
                    EvidenceItem(
                        source_id=p.source_id,
                        title=p.title,
                        topic=p.topic,
                        passage=p.passage,
                        direction=p.direction,
                        score=round3(len(shared) / len(wanted)),
                        matched_on=f"keywords: {', '.join(sorted(shared))}",
                    )
                )
        return sorted(hits, key=lambda e: -e.score)[:k]


class KnowledgeRetrievalAgent:
    def __init__(self, store: KnowledgeStore | None = None, k: int = 5, min_score: float = 0.55) -> None:
        self.store = store or DemoKnowledgeStore()
        self.k = k
        self.min_score = min_score

    @property
    def version(self) -> str:
        return self.store.version

    def retrieve(self, case: PatientCase, k: int | None = None) -> list[EvidenceItem]:
        return self.store.search(case, k or self.k, self.min_score)
