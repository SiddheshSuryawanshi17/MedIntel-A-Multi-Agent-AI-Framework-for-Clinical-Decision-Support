"""The twelve planned test cases of the Stage I report (Table 6.4), one test each. Each docstring quotes the plan:
what is tested, with which data, and the expected result."""

import json
import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from conftest import HashEncoder
from fastapi.testclient import TestClient

from medintel.agents.context_agent import Draft
from medintel.agents.embedding_store import EmbeddingKnowledgeStore, KBPassage, KnowledgeBase
from medintel.agents.ml_agent import DemoRiskModel, MLAnalysisAgent
from medintel.agents.risk_model import MODEL_FEATURES, PRE_TEST_FEATURES, RiskModelSet
from medintel.agents.rule_agent import RuleValidationAgent
from medintel.config import Settings
from medintel.ingestion import DataIngestionService, SchemaValidator
from medintel.models import Conflict, DraftSentence, EvidenceItem, MLResult, PatientCase, RuleResult
from medintel.orchestrator import CaseScheduler, create_orchestrator
from medintel.service import MedIntelApp
from medintel.training.train import fit_submodel
from medintel.web import create_app

HEADER = "patient_id,age,sex,systolic_bp,diastolic_bp,heart_rate,temperature_c,total_cholesterol,fasting_glucose,"
HEADER += "creatinine,bmi,smoker,notes\n"
SCHEMA = json.loads((Path(str(files("medintel") / "resources")) / "decision_report.schema.json").read_text())
BASE_RECORD: dict[str, Any] = {
    "age": 34,
    "sex": "M",
    "systolic_bp": 120.0,
    "diastolic_bp": 78.0,
    "heart_rate": 72.0,
    "temperature_c": 36.7,
    "total_cholesterol": 180.0,
    "fasting_glucose": 90.0,
    "creatinine": 0.9,
    "bmi": 23.0,
    "smoker": False,
}


def case(patient: str = "P-9001", notes: str = "", **values: Any) -> PatientCase:
    digits = "".join(ch for ch in patient if ch.isdigit())
    return PatientCase(
        case_id=f"C-{digits}", patient_id=patient, x_s={**BASE_RECORD, **values}, x_t=notes, source="planned-test"
    )


def kb(*passages: tuple[str, str, str]) -> KnowledgeBase:
    items = [
        KBPassage(source_id=i, title=title, topic="Test", direction="raises", passage=text)
        for i, title, text in passages
    ]
    return KnowledgeBase(name="test", version="1", fingerprint="0" * 64, passages=items)


def test_tc01_record_without_patient_id_is_rejected() -> None:
    """TC-01, unit: SchemaValidator rejects a record with no patient identifier (FR1). Data: a CSV row with an empty
    patient_id field. Expected: validate() reports the missing patient_id; no PatientCase is built; the
    data-validation error is logged."""
    row = ",45,M,120,80,70,36.7,190,95,0.9,24,no,\n"
    batch = DataIngestionService().load_text(HEADER + row, "tc01.csv")
    assert batch.cases == []
    problem = batch.rejected[0].problems[0]
    assert (problem.field, problem.message, problem.test_case) == ("patient_id", "field required", "TC-01")
    raw = dict(zip(HEADER.strip().split(","), row.strip().split(","), strict=False))
    assert [p.field for p in SchemaValidator().validate(raw)] == ["patient_id"]


def test_tc02_fahrenheit_temperature_is_held_not_converted() -> None:
    """TC-02, unit: unit consistency check (FR1). Data: temperature 101.3 recorded in °F where the schema expects °C.
    Expected: unit mismatch reported; value not converted silently; record held for the Data Administrator."""
    batch = DataIngestionService().load_text(HEADER + "P-9002,45,M,120,80,70,101.3,190,95,0.9,24,no,\n", "tc02.csv")
    assert batch.cases == [], "the record is held, not analysed"
    problem = batch.rejected[0].problems[0]
    assert problem.test_case == "TC-02" and "looks like °F" in problem.message and "Not converted" in problem.message


def test_tc03_r001_fires_on_high_systolic_bp() -> None:
    """TC-03, unit: threshold rule fires on high systolic blood pressure (FR4). Data: systolic BP of 182 mmHg against
    rule R-001 (systolic BP at or above 180 mmHg). Expected: RuleResult lists R-001 with action elevate, its reason
    and the rulebase version."""
    result = RuleValidationAgent().evaluate(case(systolic_bp=182.0))
    assert result.triggered == ["R-001"] and result.action == "elevate"
    assert result.reason.startswith("Systolic blood pressure at or above 180 mmHg")
    assert result.rulebase_version == "1.0.0"


def test_tc04_ml_output_contract() -> None:
    """TC-04, unit: MLAnalysisAgent output contract (FR3). Data: a valid PatientCase with all structured features
    present. Expected: class probabilities sum to 1; prediction, feature importance and model_version present."""
    from test_risk_model import synthetic_records

    records = synthetic_records()
    trained = RiskModelSet(
        {
            "pre_test": fit_submodel(records, PRE_TEST_FEATURES, ("logistic_regression",), 1)[0],
            "full": fit_submodel(records, MODEL_FEATURES, ("logistic_regression",), 1)[0],
        },
        "risk-lr-test",
        "0.0.1",
    )
    full_case = case(
        chest_pain_type="typical_angina",
        resting_ecg="normal",
        max_heart_rate=140.0,
        exercise_angina=True,
        st_depression=1.8,
        st_slope="flat",
        major_vessels=1,
        thal="reversible_defect",
    )
    for model in (trained, DemoRiskModel()):
        result: MLResult = MLAnalysisAgent(model).analyze(full_case)
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=0.0015)
        assert result.prediction in ("high_risk", "low_risk") and result.model_version
        sizes = [abs(c.contribution) for c in result.importance]
        assert result.importance and sizes == sorted(sizes, reverse=True)


def test_tc05_retrieval_returns_top_k_in_order() -> None:
    """TC-05, unit: retrieval returns the top k passages (FR5). Data: a case query with k = 5 on a test store holding
    at least five relevant passages. Expected: five EvidenceItem objects in descending order of similarity, each with
    source_id, passage and score."""
    store = EmbeddingKnowledgeStore(
        kb(
            ("KB-BP-001", "High blood pressure", "Very high blood pressure needs prompt review."),
            ("KB-BP-002", "Blood pressure categories", "Blood pressure above normal raises heart risk."),
            ("KB-BP-003", "Repeat blood pressure", "Confirm a high blood pressure reading by repeating it."),
            ("KB-BP-004", "Blood pressure and stroke", "High blood pressure raises stroke risk."),
            ("KB-BP-005", "Blood pressure at home", "Home blood pressure readings help confirm hypertension."),
            ("KB-BP-006", "Severe blood pressure", "Severely high blood pressure needs attention."),
        ),
        HashEncoder(),
    )
    hits: list[EvidenceItem] = store.search(case(systolic_bp=186.0, diastolic_bp=112.0), k=5, min_score=0.0)
    assert len(hits) == 5
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(h.source_id.startswith("KB-BP-") and h.passage and h.score > 0 for h in hits)


def test_tc06_empty_retrieval_is_stated() -> None:
    """TC-06, integration: empty retrieval is stated explicitly (FR5, FR6). Data: a query on a topic absent from the
    test knowledge store. Expected: empty evidence list; the report says "no sufficiently relevant evidence found";
    retrieval support K = 0; no source cited."""
    unrelated = EmbeddingKnowledgeStore(
        kb(("KB-DEN-001", "Dental floss", "Floss daily to keep gums clean."), ("KB-EYE-001", "Eyeglasses", "Lenses.")),
        HashEncoder(),
    )
    analysis = create_orchestrator(Settings(), knowledge_store=unrelated).run_case(case(systolic_bp=186.0))
    assert analysis.evidence == [] and analysis.report.evidence_refs == []
    assert analysis.fusion.component_scores["K"] == 0
    assert "No sufficiently relevant evidence was found in the knowledge base." in analysis.report.summary
    assert "KB-" not in analysis.report.summary


def test_tc07_ml_rule_disagreement_lowers_confidence() -> None:
    """TC-07, integration: ML and rule disagreement (FR6, FR7). Data: ML predicts low risk while a rule elevates the
    same case. Expected: D rises and C falls compared with a matching agreeing case; the conflict is listed in the
    report."""
    orchestrator = create_orchestrator(Settings())
    agreeing = orchestrator.run_case(case("P-9071", systolic_bp=179.0))
    conflicting = orchestrator.run_case(case("P-9072", systolic_bp=182.0))
    assert agreeing.ml.prediction == conflicting.ml.prediction == "low_risk"
    assert conflicting.rules.action == "elevate" and agreeing.rules.action is None
    assert conflicting.fusion.disagreement > agreeing.fusion.disagreement
    assert conflicting.fusion.confidence < agreeing.fusion.confidence
    assert any("rule R-001 elevates" in c for c in conflicting.report.conflicts)


def test_tc08_full_agreement() -> None:
    """TC-08, integration: full agreement among components (FR7). Data: ML prediction, triggered rules and retrieved
    evidence all support the same decision. Expected: C above the configured decision threshold; D = 0; the
    conflicts list is empty."""
    analysis = create_orchestrator(Settings()).run_case(
        case("P-9080", age=67, systolic_bp=186.0, total_cholesterol=258.0, fasting_glucose=142.0, smoker=True)
    )
    assert analysis.ml.prediction == "high_risk" and analysis.rules.action == "elevate"
    assert all(e.direction != "lowers" for e in analysis.evidence) and analysis.evidence
    assert analysis.fusion.disagreement == 0 and analysis.report.conflicts == []
    assert analysis.fusion.confidence >= analysis.fusion.threshold


def test_tc09_parallel_runs_keep_patients_isolated() -> None:
    """TC-09, system: parallel processing with patient isolation (FR2, NFR2). Data: a batch of four cases with unique
    marker values, run with max_workers = 4. Expected: four reports whose rule results and evidence IDs match
    single-case runs; no case data or evidence leaks between cases."""
    orchestrator = create_orchestrator(Settings())
    markers = ["MARKER-ALPHA", "MARKER-BRAVO", "MARKER-CHARLIE", "MARKER-DELTA"]
    cases = [
        case(f"P-910{i}", notes=f"{marker} note", systolic_bp=bp, heart_rate=hr, creatinine=cr)
        for i, (marker, bp, hr, cr) in enumerate(
            zip(markers, (186.0, 150.0, 120.0, 132.0), (70.0, 125.0, 80.0, 72.0), (0.9, 0.9, None, 1.8), strict=True)
        )
    ]
    scheduler = CaseScheduler(orchestrator)
    scheduler.submit(cases)
    parallel = scheduler.run(max_workers=4)
    single = [orchestrator.analyze(c) for c in cases]
    for p, s in zip(parallel, single, strict=True):
        assert (p.case_id, p.triggered_rules, p.evidence_refs, p.findings) == (
            s.case_id,
            s.triggered_rules,
            s.evidence_refs,
            s.findings,
        )
    for i, report in enumerate(parallel):
        text = report.model_dump_json()
        assert all(m not in text for j, m in enumerate(markers) if j != i)


def test_tc10_end_to_end_report_review_and_audit(tmp_path: Path) -> None:
    """TC-10, system: end-to-end report, review and audit record (FR8, FR9, NFR1). Data: a de-identified case uploaded,
    analysed and reviewed through the REST API (planned as POST /cases/upload, /analyze and /review; built as
    POST /v1/batches/upload, POST /v1/cases and POST /v1/cases/{case_id}/reviews). Expected: the report holds every
    mandatory field; the AuditRecord stores model version, timestamp, rule version, retrieval IDs and status; the
    review adds a second AuditRecord and leaves the KnowledgeStore unchanged."""
    app = MedIntelApp(tmp_path, Settings())
    token = app.users.add("Dr. Rao", ["clinician", "data_admin"])
    headers = {"Authorization": f"Bearer {token}"}
    csv = HEADER + "P-9100,61,M,186,104,78,36.6,228,118,1.1,28.9,no,Chest tightness on stairs.\n"
    with TestClient(create_app(app, "http://127.0.0.1:8000"), base_url="http://127.0.0.1:8000") as client:
        batch = client.post("/v1/batches/upload", files={"file": ("tc10.csv", csv.encode())}, headers=headers).json()
        report = client.post(
            "/v1/cases", json={"batch_id": batch["batch_id"], "patient_id": "P-9100"}, headers=headers
        ).json()["report"]
        jsonschema.validate(report, SCHEMA)
        first = client.get("/v1/audit", params={"case_id": "C-9100"}, headers=headers).json()["records"]
        assert len(first) == 1
        record = first[0]
        assert record["model_version"] and record["rulebase_version"] == "1.0.0" and record["timestamp"]
        assert record["retrieval_ids"] == ",".join(report["evidence_refs"]) and record["status"] == "pending_review"
        index_before = app.knowledge.version
        client.post("/v1/cases/C-9100/reviews", json={"decision": "accept", "notes": "Agree."}, headers=headers)
        after = client.get("/v1/audit", params={"case_id": "C-9100"}, headers=headers).json()["records"]
    assert len(after) == 2 and after[0]["event"] == "review_accepted" and after[0]["actor"] == "Dr. Rao"
    assert app.knowledge.version == index_before, "nothing reaches the index before a curator approves it"


def test_tc11_per_case_latency_targets() -> None:
    """TC-11, performance: per-case latency (NFR4). Data: one valid case. Expected: end to end within 30 s; ML and
    rule stages each under 1 s. (Measured here with the default components; `medintel evaluate` times the configured
    ones, including the trained model, embedding retrieval and the summary writer.)"""
    analysis = create_orchestrator(Settings()).run_case(case(systolic_bp=150.0))
    assert set(analysis.timings) == {"ml", "rules", "retrieval", "context", "fusion", "total"}
    assert analysis.timings["total"] < 30 and analysis.timings["ml"] < 1 and analysis.timings["rules"] < 1


class StubWriter:
    """Stands in for the LLM Service with a reply that names a symptom the case never mentions."""

    def write(
        self,
        case: PatientCase,
        ml: MLResult,
        rules: RuleResult,
        evidence: list[EvidenceItem],
        conflicts: list[Conflict],
    ) -> Draft:
        return Draft(
            [
                DraftSentence(text="The patient reports chest pain radiating to the left arm.", sources=["notes"]),
                DraftSentence(text="Systolic blood pressure is 150 mmHg.", sources=["record:systolic_bp"]),
            ],
            "stub",
        )


def test_tc12_unsupported_statement_is_removed(caplog: pytest.LogCaptureFixture) -> None:
    """TC-12, safety: synthesis flags an unsupported statement (FR6). Data: a stubbed LLM reply naming a symptom absent
    from the case, agent outputs and evidence. Expected: the statement is listed in unsupported_statements and absent
    from the final DecisionReport; the synthesis error is logged."""
    orchestrator = create_orchestrator(Settings(), writer=StubWriter())
    with caplog.at_level(logging.WARNING, logger="medintel"):
        analysis = orchestrator.run_case(case("P-9120", notes="Annual check-up. No complaints.", systolic_bp=150.0))
    claim = "The patient reports chest pain radiating to the left arm."
    assert analysis.context.unsupported_statements == [claim]
    assert claim not in analysis.report.summary and analysis.report.summary == "Systolic blood pressure is 150 mmHg."
    reason = next(s.reason for s in analysis.context.draft if not s.supported)
    assert "names chest pain" in (reason or "")
    assert "TC-12: removed 1 unsupported statement(s) from C-9120" in caplog.text
