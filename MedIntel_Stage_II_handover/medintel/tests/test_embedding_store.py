"""Embedding retrieval with a deterministic stand-in encoder (no model download), the packaged knowledge base, and
governed index versions."""

import re
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest
from conftest import HashEncoder

from medintel.agents.embedding_store import (
    EmbeddingKnowledgeStore,
    KBPassage,
    KnowledgeBase,
    case_queries,
    validated_passage,
)
from medintel.ingestion import DataIngestionService
from medintel.knowledge import GovernedKnowledgeStore, KnowledgeRegistry
from medintel.models import PatientCase
from medintel.orchestrator import create_orchestrator

KB_FILE = Path(str(files("medintel") / "resources" / "knowledge_base.json"))


def passage(source_id: str, title: str, text: str, direction: str = "raises") -> KBPassage:
    return KBPassage(source_id=source_id, title=title, topic="Test", direction=direction, passage=text)  # type: ignore[arg-type]


TINY = KnowledgeBase(
    name="tiny",
    version="1",
    fingerprint="f" * 64,
    passages=[
        passage("KB-HTN-001", "Very high blood pressure", "A blood pressure reading this high needs prompt care."),
        passage("KB-LIP-001", "High cholesterol", "High total cholesterol adds to heart disease risk."),
        passage(
            "KB-GEN-001",
            "Normal levels",
            "Normal blood pressure, cholesterol and blood sugar in a non-smoking adult lower risk.",
            "lowers",
        ),
    ],
)


def make_case(**values: object) -> PatientCase:
    base: dict[str, object] = {
        "age": 50,
        "sex": "M",
        "systolic_bp": 118.0,
        "diastolic_bp": 76.0,
        "heart_rate": 70.0,
        "total_cholesterol": 180.0,
        "fasting_glucose": 90.0,
        "creatinine": 0.9,
        "bmi": 23.0,
        "smoker": False,
    }
    base.update(values)
    return PatientCase(case_id="C-0001", patient_id="P-0001", x_s=base, source="test")  # type: ignore[arg-type]


def test_queries_follow_the_findings() -> None:
    assert case_queries(make_case())[0][1] == "values within reference ranges"
    queries = case_queries(make_case(systolic_bp=186.0, diastolic_bp=112.0, creatinine=None))
    findings = [f for _, f, _ in queries]
    assert findings == ["BP 186/112 mmHg", "creatinine missing"]
    assert [m for _, _, m in queries] == ["raises", None], "BP at or above 180/120 is a marked finding"
    assert case_queries(make_case(systolic_bp=150.0))[0][2] is None
    workup = make_case(st_depression=2.3, thal="reversible_defect", exercise_angina=True, major_vessels=2)
    assert {"ST depression 2.3 mm", "thallium test: reversible defect", "2 major vessels on fluoroscopy"} <= {
        f for _, f, _ in case_queries(workup)
    }
    with_notes = make_case().model_copy(update={"x_t": "Chest tightness on stairs."})
    assert case_queries(with_notes)[-1] == ("Chest tightness on stairs.", "clinical notes", None)


def test_search_scores_and_filters() -> None:
    store = EmbeddingKnowledgeStore(TINY, HashEncoder())
    hits = store.search(make_case(systolic_bp=186.0, diastolic_bp=112.0), k=5, min_score=0.3)
    assert hits[0].source_id == "KB-HTN-001"
    assert hits[0].matched_on == "BP 186/112 mmHg"
    assert all(h.score >= 0.3 for h in hits)
    assert store.search(make_case(systolic_bp=186.0), k=5, min_score=0.99) == [], "TC-06: nothing relevant enough"
    assert len(store.search_text("cholesterol and blood pressure", k=2)) == 2
    normal = store.search(make_case(), k=3, min_score=0.0)
    assert normal[0].source_id == "KB-GEN-001" and normal[0].marked is True
    assert all(h.marked is False for h in normal if h.direction == "raises"), "a normal profile marks no risk passage"
    severe = store.search(make_case(systolic_bp=186.0, diastolic_bp=112.0), k=1, min_score=0.0)[0]
    mild = store.search(make_case(systolic_bp=150.0), k=1, min_score=0.0)[0]
    assert (severe.source_id, severe.marked) == ("KB-HTN-001", True)
    assert mild.marked is False, "only findings past a stated threshold can raise a conflict"


def test_validated_outcomes_join_the_index_after_approval(tmp_path: Path, morning: object) -> None:
    registry = KnowledgeRegistry(tmp_path / "kb.db", base_description="Initial index: tiny")
    registry.ensure_base(TINY.fingerprint, "tiny")
    store = EmbeddingKnowledgeStore(TINY, HashEncoder())
    governed = GovernedKnowledgeStore(store, registry)
    case = next(c for c in morning.cases if c.patient_id == "P-1043")  # type: ignore[attr-defined]
    analysis = create_orchestrator().run_case(case)
    assert analysis.report.decision == "high_risk"
    proposal = registry.propose(analysis, "Dr. Rao", "Agree, refer to cardiology.")
    assert proposal is not None
    governed.sync()
    assert "KB-CASE-1043" not in store.known_ids(), "nothing joins the index before approval"
    registry.decide(proposal.proposal_id, True, "Ms. Iyer")
    governed.sync()
    assert "KB-CASE-1043" in store.known_ids()
    similar = make_case(systolic_bp=164.0, diastolic_bp=96.0, total_cholesterol=258.0, smoker=True)
    hits = governed.search(similar, k=5, min_score=0.1)
    assert "KB-CASE-1043" in [h.source_id for h in hits]
    own = case.model_copy()
    assert "KB-CASE-1043" not in [h.source_id for h in governed.search(own, k=5, min_score=0.0)], (
        "a case never retrieves its own validated outcome"
    )


def test_validated_passage_text(morning: object) -> None:
    case = next(c for c in morning.cases if c.patient_id == "P-1042")  # type: ignore[attr-defined]
    analysis = create_orchestrator().run_case(case)
    item = validated_passage("C-1042", "high_risk", analysis.model_dump_json(), "Ms. Iyer")
    assert item.source_id == "KB-CASE-1042" and item.direction == "raises"
    assert "34-year-old man" in item.passage and "BP 186/112 mmHg" in item.passage


def test_index_version_tracks_the_knowledge_base_file(tmp_path: Path) -> None:
    registry = KnowledgeRegistry(tmp_path / "kb.db", base_description="Initial index: A")
    assert registry.ensure_base("aaa", "A") == 3
    assert registry.ensure_base("aaa", "A") == 3, "same file, same version"
    assert registry.ensure_base("bbb", "B") == 4
    assert registry.history()[0].description == "Knowledge base replaced: B"


def test_duplicate_source_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "kb.json"
    path.write_text(
        TINY.model_copy(update={"passages": TINY.passages + TINY.passages[:1]}).model_dump_json(), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate source_id"):
        KnowledgeBase.load(path)


@pytest.mark.skipif(not KB_FILE.is_file(), reason="packaged knowledge base not written yet")
def test_packaged_knowledge_base_is_sourced() -> None:
    kb = KnowledgeBase.load()
    assert len(kb.passages) >= 10
    for p in kb.passages:
        assert re.fullmatch(r"KB-[A-Z]+-\d{3}", p.source_id), p.source_id
        assert p.url.startswith("https://") and p.source_title and p.retrieved, p.source_id
        assert re.match(r"https://([a-z]+\.)*(cdc\.gov|nih\.gov|medlineplus\.gov)/", p.url), p.url
        assert "—" not in p.passage and "–" not in p.passage, p.source_id
    assert len(kb.fingerprint) == 64


def test_ingested_cases_produce_queries() -> None:
    text = "patient_id,age,sex,systolic_bp,diastolic_bp,heart_rate,temperature_c,total_cholesterol,fasting_glucose,"
    text += "creatinine,bmi,smoker,notes\nP-1,45,M,150,95,72,36.8,250,130,,31,yes,Breathless on exertion.\n"
    case = DataIngestionService().load_text(text, "t.csv").cases[0]
    findings = [f for _, f, _ in case_queries(case)]
    assert findings[:4] == ["BP 150/95 mmHg", "total cholesterol 250 mg/dL", "fasting glucose 130 mg/dL", "BMI 31"]
    assert "creatinine missing" in findings and "current smoker" in findings and findings[-1] == "clinical notes"


FAISS_CHECK = """
import numpy as np
from medintel.agents.embedding_store import FaissIndex, NumpyIndex
rng = np.random.default_rng(0)
v = rng.random((50, 32)).astype("float32")
v /= np.linalg.norm(v, axis=1, keepdims=True)
q = v[:5] + 0.01
q /= np.linalg.norm(q, axis=1, keepdims=True)
s1, i1 = NumpyIndex(v).search(q, 5)
s2, i2 = FaissIndex(v).search(q, 5)
assert (i1 == i2).all() and np.allclose(s1, s2, atol=1e-5)
print("same results")
"""


def test_faiss_index_matches_the_numpy_index() -> None:
    """Run in a fresh process without PyTorch or scikit-learn: on macOS their OpenMP runtimes clash with FAISS's."""
    result = subprocess.run([sys.executable, "-c", FAISS_CHECK], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0 and "same results" in result.stdout, result.stderr[-2000:]
