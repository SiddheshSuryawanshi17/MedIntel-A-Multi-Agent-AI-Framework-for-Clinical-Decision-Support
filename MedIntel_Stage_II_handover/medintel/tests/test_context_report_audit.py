import json
import sqlite3
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from medintel.agents.context_agent import TemplateWriter, verify_sentence
from medintel.audit import AuditLogger, CaseStore, export_report
from medintel.fusion import decision_steps
from medintel.ingestion import IngestionResult
from medintel.models import DraftSentence
from medintel.orchestrator import CaseScheduler, Orchestrator, create_orchestrator


def test_unsupported_statement_is_removed_tc12(morning: IngestionResult, prototype_orchestrator: Orchestrator) -> None:
    analysis = prototype_orchestrator.run_case(next(c for c in morning.cases if c.case_id == "C-1047"))
    claim = "She has a known history of type 2 diabetes."
    assert analysis.context.unsupported_statements == [claim]
    assert claim not in analysis.context.summary and claim not in analysis.report.summary
    rejected = next(s for s in analysis.context.draft if not s.supported)
    assert "record:history" in (rejected.reason or "")


def test_no_relevant_evidence_gives_k_zero_tc06(morning: IngestionResult, orchestrator: Orchestrator) -> None:
    analysis = orchestrator.run_case(next(c for c in morning.cases if c.case_id == "C-1044"))
    assert analysis.evidence == []
    assert analysis.fusion.component_scores["K"] == 0
    assert analysis.fusion.decision == "uncertain"
    assert "No sufficiently relevant evidence was found in the knowledge base." in analysis.context.summary
    assert decision_steps(analysis.fusion, analysis.ml, analysis.rules, analysis.context)[-1].startswith(
        "No sufficiently"
    )


def test_template_sentences_all_pass_the_source_check(
    morning: IngestionResult, boundary: IngestionResult, orchestrator: Orchestrator
) -> None:
    for case in morning.cases + boundary.cases:
        context = orchestrator.run_case(case).context
        assert context.generated_by == "template"
        assert context.unsupported_statements == [], case.case_id


def test_source_check_rejects_invented_numbers_and_missing_sources(
    morning: IngestionResult, orchestrator: Orchestrator
) -> None:
    a = orchestrator.run_case(morning.cases[1])
    args = (a.case, a.ml, a.rules, a.evidence, a.context.conflict_details)
    ok = verify_sentence(DraftSentence(text="Systolic BP is 186 mmHg.", sources=["record:systolic_bp"]), *args)
    invented = verify_sentence(DraftSentence(text="Systolic BP is 200 mmHg.", sources=["record:systolic_bp"]), *args)
    uncited = verify_sentence(DraftSentence(text="The patient is stable.", sources=[]), *args)
    unknown = verify_sentence(DraftSentence(text="See KB-LIP-007.", sources=["evidence:KB-LIP-007"]), *args)
    assert ok.supported
    assert not invented.supported and "200" in (invented.reason or "")
    assert not uncited.supported
    assert not unknown.supported
    assert (
        TemplateWriter().write(a.case, a.ml, a.rules, a.evidence, a.context.conflict_details).generated_by == "template"
    )


def test_missing_fields_are_not_read_as_symptoms_tc12(morning: IngestionResult, orchestrator: Orchestrator) -> None:
    a = orchestrator.run_case(morning.cases[1])
    args = (a.case, a.ml, a.rules, a.evidence, a.context.conflict_details)

    def supported(text: str) -> bool:
        return verify_sentence(DraftSentence(text=text, sources=["record"]), *args).supported

    assert supported("Chest pain type, resting ECG and exercise-induced angina were not recorded.")
    assert supported("Systolic BP is 186 mmHg; no exercise test, resting ECG or chest pain type is recorded.")
    assert supported("No work-up is on record: chest pain type, exercise angina and ST slope were not recorded.")
    assert not supported("He has exercise-induced angina, but no resting ECG is recorded.")
    assert not supported("No resting ECG is recorded and he reports chest pain.")


def test_reports_match_the_section_4_4_2_schema(morning: IngestionResult, orchestrator: Orchestrator) -> None:
    schema = json.loads((Path(str(files("medintel") / "resources")) / "decision_report.schema.json").read_text())
    validator = Draft202012Validator(schema)
    for case in morning.cases:
        report = orchestrator.run_case(case).report.model_dump()
        assert list(validator.iter_errors(report)) == [], case.case_id


def test_audit_log_is_append_only(tmp_path: Path, morning: IngestionResult) -> None:
    logger = AuditLogger(tmp_path / "audit.db")
    orchestrator = create_orchestrator(audit_logger=logger)
    for case in morning.cases[:3]:
        orchestrator.analyze(case)
    records = logger.records()
    assert [r.case_id for r in records] == ["C-1043", "C-1042", "C-1041"]
    assert records[1].retrieval_ids == "KB-HTN-001,KB-HTN-004"
    assert records[1].model_version == "risk-lr 0.3 (demo weights)" and records[1].rulebase_version == "1.0.0"
    with pytest.raises(sqlite3.DatabaseError, match="append-only"), logger.db:
        logger.db.execute("UPDATE audit SET status = 'accepted'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"), logger.db:
        logger.db.execute("DELETE FROM audit")
    assert len(logger.records(case_id="C-1042")) == 1


def test_failing_agent_produces_an_audited_failure_report(tmp_path: Path, morning: IngestionResult) -> None:
    logger = AuditLogger(tmp_path / "audit.db")
    orchestrator = create_orchestrator(audit_logger=logger)

    def broken(*_: object) -> None:
        raise RuntimeError("model file missing")

    orchestrator.ml_agent.analyze = broken  # type: ignore[method-assign]
    report = orchestrator.analyze(morning.cases[0])
    assert report.status == "failed" and report.decision == "failed"
    assert "RuntimeError" in report.recommendation
    assert logger.records()[0].status == "failed"


def test_scheduler_runs_isolated_copies_in_parallel(morning: IngestionResult, orchestrator: Orchestrator) -> None:
    scheduler = CaseScheduler(orchestrator)
    scheduler.submit(morning.cases)
    morning.cases[0].x_s["systolic_bp"] = 999
    try:
        reports = scheduler.run(max_workers=4)
    finally:
        morning.cases[0].x_s["systolic_bp"] = 118
    assert [r.case_id for r in reports] == [c.case_id for c in morning.cases]
    assert reports[0].decision == "low_risk"
    assert scheduler.queue == []


def test_case_store_round_trip_and_export(tmp_path: Path, morning: IngestionResult, orchestrator: Orchestrator) -> None:
    store = CaseStore(tmp_path / "store.db")
    batch_id = store.save_batch(morning)
    assert batch_id.startswith("B-") and store.get_batch(batch_id) == morning
    analysis = orchestrator.run_case(morning.cases[1])
    store.save_analysis(analysis)
    assert store.get_analysis("C-1042") == analysis
    assert store.get_batch("B-missing") is None
    path = export_report(analysis.report, tmp_path)
    assert json.loads(path.read_text())["case_id"] == "C-1042"
