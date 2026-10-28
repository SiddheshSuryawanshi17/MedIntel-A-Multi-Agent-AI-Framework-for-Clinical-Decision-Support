"""Embedding retrieval (FR5, Section 5.4.1): sentence-transformers embeddings in a local FAISS inner-product index.

Embeddings are normalized, so every score is a cosine similarity. A case is searched with one query per notable
finding, plus its notes; each passage keeps its best score and the finding that matched it. Clinician-validated
outcomes join the index only after a Knowledge Curator approves them (knowledge.py)."""

import hashlib
import json
import logging
import threading
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel

from medintel.models import CaseAnalysis, Direction, EvidenceItem, PatientCase
from medintel.numeric import number_text, round3
from medintel.schema import Value, abnormal

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class KBPassage(BaseModel):
    source_id: str
    title: str
    topic: str
    direction: Direction
    passage: str
    source: str = ""
    source_title: str = ""
    url: str = ""
    retrieved: str = ""
    license: str = ""
    status: str = ""
    case_id: str | None = None
    approved_by: str | None = None

    def text(self) -> str:
        return f"{self.title}. {self.passage}"


class KnowledgeBase(BaseModel):
    name: str
    version: str
    description: str = ""
    review: dict[str, Any] = {}
    passages: list[KBPassage]
    fingerprint: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} {self.version} ({self.fingerprint[:8]})"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "KnowledgeBase":
        source = Path(path) if path else Path(str(files("medintel") / "resources" / "knowledge_base.json"))
        raw = source.read_bytes()
        kb = cls.model_validate(json.loads(raw))
        ids = [p.source_id for p in kb.passages]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate source_id in {source}: {', '.join(duplicates)}")
        return kb.model_copy(update={"fingerprint": hashlib.sha256(raw).hexdigest()})


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Loads the model on first use, from the local cache when it is there (no network call)."""

    def __init__(self, name: str = DEFAULT_MODEL) -> None:
        self.name = name
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        from sentence_transformers import SentenceTransformer

        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

        try:
            return SentenceTransformer(self.name, device="cpu", local_files_only=True)
        except OSError:
            return SentenceTransformer(self.name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            vectors = self._model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
        return np.ascontiguousarray(vectors, dtype=np.float32)


def _num(x: dict[str, Value], key: str) -> float | None:
    value = x.get(key)
    return None if value is None or isinstance(value, (bool, str)) else float(value)


def _n(value: float) -> str:
    return number_text(value)


Query = tuple[str, str, Direction | None]


def case_queries(case: PatientCase) -> list[Query]:
    """(query, finding shown as matched_on, marks) for every notable finding of the case, then its notes.

    Queries name the clinical category a value falls in, because embeddings do not compare numbers. A risk finding
    that crosses a threshold stated in the knowledge base or the rulebase marks "raises" passages, and a fully
    normal profile marks "lowers" passages; only marked evidence can raise an evidence-versus-model conflict."""
    x = case.x_s
    out: list[Query] = []
    sbp, dbp = _num(x, "systolic_bp"), _num(x, "diastolic_bp")
    if abnormal("systolic_bp", sbp) == "high" or abnormal("diastolic_bp", dbp) == "high":
        bp = f"{_n(sbp)}/{_n(dbp)}" if sbp is not None and dbp is not None else _n(sbp or dbp or 0)
        severe = (sbp or 0) >= 180 or (dbp or 0) >= 120
        query = (
            "severely high blood pressure above 180/120 mm Hg that needs prompt medical attention"
            if severe
            else "high blood pressure (hypertension) at or above 130/80 mm Hg, raising heart disease and stroke risk"
        )
        out.append((query, f"BP {bp} mmHg", "raises" if severe else None))
    chol = _num(x, "total_cholesterol")
    if chol is not None and abnormal("total_cholesterol", chol) == "high":
        out.append(
            (
                "total cholesterol above the healthy level of 200 mg/dL, which can narrow the arteries",
                f"total cholesterol {_n(chol)} mg/dL",
                None,
            )
        )
    glucose = _num(x, "fasting_glucose")
    if glucose is not None and glucose >= 126:
        query = "fasting blood sugar of 126 mg/dL or above, in the diabetes range"
        out.append((query, f"fasting glucose {_n(glucose)} mg/dL", "raises"))
    elif glucose is not None and glucose >= 100:
        query = "fasting blood sugar of 100 to 125 mg/dL, in the prediabetes range"
        out.append((query, f"fasting glucose {_n(glucose)} mg/dL", None))
    hr = _num(x, "heart_rate")
    if hr is not None and hr > 100:
        query = "fast resting heart rate above 100 beats per minute (tachycardia)"
        out.append((query, f"heart rate {_n(hr)} bpm", "raises"))
    bmi = _num(x, "bmi")
    if bmi is not None and bmi >= 25:
        query = "obesity with a body mass index of 30 or above" if bmi >= 30 else "overweight with a BMI of 25 to 29.9"
        out.append((query, f"BMI {_n(bmi)}", None))
    creatinine = _num(x, "creatinine")
    if "creatinine" in x and creatinine is None:
        out.append(("kidney function blood test (creatinine and GFR) not available", "creatinine missing", None))
    elif abnormal("creatinine", creatinine) == "high":
        query = "raised serum creatinine suggesting reduced kidney function (GFR)"
        out.append((query, f"creatinine {_n(creatinine)} mg/dL", None))  # type: ignore[arg-type]
    if x.get("smoker") is True:
        out.append(
            (
                "current cigarette smoker; smoking raises coronary heart disease and stroke risk",
                "current smoker",
                "raises",
            )
        )
    age = _num(x, "age")
    if age is not None and age < 18:
        query = "child or teenager whose blood pressure is judged against children of the same age, sex and height"
        out.append((query, f"age {_n(age)}", None))
    out += _workup_queries(x)
    if not out:
        query = "healthy habits with normal blood pressure, cholesterol and blood sugar and no smoking"
        out.append((query, "values within reference ranges", "lowers"))
    if case.x_t.strip():
        out.append((case.x_t.strip(), "clinical notes", None))
    return out


WORKUP_QUERIES: dict[tuple[str, object], tuple[str, str, Direction | None]] = {
    ("chest_pain_type", "typical_angina"): (
        "typical angina: chest pain or pressure with physical activity that eases with rest",
        "chest pain type: typical angina",
        "raises",
    ),
    ("chest_pain_type", "atypical_angina"): (
        "atypical chest pain or discomfort that may be angina",
        "chest pain type: atypical angina",
        None,
    ),
    ("chest_pain_type", "non_anginal"): (
        "chest pain that is probably not angina (non-anginal pain)",
        "chest pain type: non-anginal pain",
        None,
    ),
    ("exercise_angina", True): (
        "chest pain brought on by an exercise stress test (exercise-induced angina)",
        "exercise-induced angina",
        "raises",
    ),
    ("st_slope", "flat"): ("flat ST segment on an exercise stress test ECG", "flat ST slope", None),
    ("st_slope", "downsloping"): (
        "downsloping ST segment on an exercise stress test ECG",
        "downsloping ST slope",
        None,
    ),
    ("thal", "reversible_defect"): (
        "nuclear stress scan showing parts of the heart without enough blood flow during stress",
        "thallium test: reversible defect",
        "raises",
    ),
    ("thal", "fixed_defect"): (
        "nuclear heart scan showing damaged heart muscle from a past heart attack",
        "thallium test: fixed defect",
        None,
    ),
    ("resting_ecg", "lv_hypertrophy"): (
        "thickened left ventricle heart muscle (left ventricular hypertrophy) from high blood pressure",
        "resting ECG: left ventricular hypertrophy",
        None,
    ),
    ("resting_ecg", "st_t_abnormality"): (
        "ST-T wave abnormality on a resting ECG",
        "resting ECG: ST-T wave abnormality",
        None,
    ),
}


def _workup_queries(x: dict[str, Value]) -> list[Query]:
    out: list[Query] = [
        (query, finding, marks)
        for (key, value), (query, finding, marks) in WORKUP_QUERIES.items()
        if x.get(key) == value
    ]
    st = _num(x, "st_depression")
    if st is not None and st >= 1:
        query = "abnormal exercise stress test ECG with ST depression, suggesting reduced blood flow to the heart"
        out.append((query, f"ST depression {_n(st)} mm", None))
    vessels = _num(x, "major_vessels")
    if vessels:
        query = "coronary angiography showing narrowed or blocked coronary arteries"
        out.append((query, f"{_n(vessels)} major vessels on fluoroscopy", "raises"))
    return out


def validated_passage(case_id: str, decision: str, payload: str, approved_by: str) -> KBPassage:
    analysis = CaseAnalysis.model_validate_json(payload)
    x = analysis.case.x_s
    label = "high risk" if decision == "high_risk" else "low risk"
    sex = {"M": "man", "F": "woman"}.get(str(x.get("sex")), "patient")
    findings = [finding for _, finding, _ in case_queries(analysis.case) if finding != "clinical notes"]
    notes = analysis.report.notes.strip()
    text = (
        f"Clinician-confirmed {label} outcome for case {case_id}: {x.get('age')}-year-old {sex}; {'; '.join(findings)}."
    )
    if analysis.case.x_t.strip():
        text += f" Case notes: {analysis.case.x_t.strip()}"
    if notes:
        text += f" Reviewer notes: {notes}"
    digits = "".join(ch for ch in case_id if ch.isdigit()) or "0"
    return KBPassage(
        source_id=f"KB-CASE-{digits}",
        title=f"Validated outcome: {case_id} ({label})",
        topic="Validated cases",
        direction="raises" if decision == "high_risk" else "lowers",
        passage=text,
        source="MedIntel clinician review",
        status="validated",
        case_id=case_id,
        approved_by=approved_by,
    )


class VectorIndex(Protocol):
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class NumpyIndex:
    """Exact inner-product search over normalized vectors: the computation FAISS's IndexFlatIP performs."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = queries @ self.vectors.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, : min(k, len(self.vectors))]
        return np.take_along_axis(scores, order, axis=1), order


class FaissIndex:
    """FAISS IndexFlatIP, for larger knowledge bases. Use it only where FAISS and PyTorch share one OpenMP runtime
    (the Linux wheels). The macOS and Windows wheels each bundle their own, and a process that loads both can
    abort with "OMP: Error #15", so the exact numpy search is the default."""

    def __init__(self, vectors: np.ndarray) -> None:
        import faiss

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(queries, k)


INDEXES: dict[str, type[NumpyIndex] | type[FaissIndex]] = {"numpy": NumpyIndex, "faiss": FaissIndex}


class EmbeddingKnowledgeStore:
    """A KnowledgeStore over an inner-product index of normalized embeddings."""

    def __init__(self, kb: KnowledgeBase, encoder: Encoder | None = None, index: str = "numpy") -> None:
        self.kb = kb
        self.encoder = encoder or SentenceTransformerEncoder()
        self.index_kind = index
        self.passages: list[KBPassage] = list(kb.passages)
        self._index: Any = None
        self._lock = threading.RLock()

    @property
    def version(self) -> str:
        return self.kb.label

    @property
    def method(self) -> str:
        return f"embedding similarity ({self.encoder.name}, cosine, {self.index_kind} inner-product index)"

    def _ensure_index(self) -> VectorIndex:
        with self._lock:
            if self._index is None:
                vectors = self.encoder.encode([p.text() for p in self.passages])
                self._index = INDEXES[self.index_kind](vectors)
            return self._index

    def add_validated(self, passage: KBPassage) -> None:
        with self._lock:
            self.passages = [p for p in self.passages if p.source_id != passage.source_id]
            self.passages.append(passage)
            self._index = None

    def known_ids(self) -> set[str]:
        with self._lock:
            return {p.source_id for p in self.passages}

    def search_queries(
        self, queries: list[Query], k: int, min_score: float, exclude_case: str | None = None
    ) -> list[EvidenceItem]:
        if not queries:
            return []
        with self._lock:
            index = self._ensure_index()
            passages = list(self.passages)
            vectors = self.encoder.encode([q for q, _, _ in queries])
            scores, ids = index.search(vectors, min(len(passages), max(10, 3 * k)))
        best: dict[int, tuple[float, str, Direction | None]] = {}
        for row, (_, finding, marks) in enumerate(queries):
            for score, idx in zip(scores[row], ids[row], strict=True):
                if idx < 0 or (exclude_case is not None and passages[idx].case_id == exclude_case):
                    continue
                if idx not in best or score > best[idx][0]:
                    best[idx] = (float(score), finding, marks)
        hits = [
            EvidenceItem(
                source_id=passages[i].source_id,
                title=passages[i].title,
                topic=passages[i].topic,
                passage=passages[i].passage,
                direction=passages[i].direction,
                score=round3(score),
                matched_on=finding,
                marked=marks is not None and marks == passages[i].direction,
            )
            for i, (score, finding, marks) in best.items()
            if score >= min_score
        ]
        return sorted(hits, key=lambda e: (-e.score, e.source_id))[:k]

    def search(self, case: PatientCase, k: int, min_score: float) -> list[EvidenceItem]:
        return self.search_queries(case_queries(case), k, min_score, exclude_case=case.case_id)

    def search_text(self, query: str, k: int) -> list[EvidenceItem]:
        return self.search_queries([(query, f"query: {query}", None)], k, 0.0)
