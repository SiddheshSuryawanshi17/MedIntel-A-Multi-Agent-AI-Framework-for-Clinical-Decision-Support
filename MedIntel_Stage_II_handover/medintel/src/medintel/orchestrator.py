"""Orchestrator and CaseScheduler (FR2, NFR2, NFR5): one isolated pipeline run per case, with a timestamp at each
stage boundary (Section 6.5.1)."""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from medintel.agents.context_agent import ClinicalContextAgent, DraftWriter
from medintel.agents.ml_agent import MLAnalysisAgent
from medintel.agents.retrieval_agent import KnowledgeRetrievalAgent, KnowledgeStore
from medintel.agents.risk_model import load_risk_model
from medintel.agents.rule_agent import RuleValidationAgent
from medintel.audit import AuditLogger
from medintel.config import Settings
from medintel.fusion import DecisionFusionEngine, ExplanationGenerator, priority, recommend
from medintel.models import CaseAnalysis, DecisionReport, PatientCase, Versions

T = TypeVar("T")


def timed(fn: Callable[..., T], *args: Any) -> tuple[T, float]:
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


class Orchestrator:
    def __init__(
        self,
        ml_agent: MLAnalysisAgent,
        rule_agent: RuleValidationAgent,
        retriever: KnowledgeRetrievalAgent,
        context_agent: ClinicalContextAgent,
        fusion_engine: DecisionFusionEngine,
        explainer: ExplanationGenerator,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.ml_agent = ml_agent
        self.rule_agent = rule_agent
        self.retriever = retriever
        self.context_agent = context_agent
        self.fusion_engine = fusion_engine
        self.explainer = explainer
        self.audit_logger = audit_logger

    def versions(self) -> Versions:
        return Versions(
            model=self.ml_agent.model.version,
            rulebase=self.rule_agent.rulebase.version,
            index=self.retriever.version,
        )

    def run_case(self, case: PatientCase) -> CaseAnalysis:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as pool:
            ml_future = pool.submit(timed, self.ml_agent.analyze, case)
            rules_future = pool.submit(timed, self.rule_agent.evaluate, case)
            evidence_future = pool.submit(timed, self.retriever.retrieve, case)
            (ml, t_ml), (rules, t_rules), (evidence, t_retrieval) = (
                ml_future.result(),
                rules_future.result(),
                evidence_future.result(),
            )
        context, t_context = timed(self.context_agent.synthesize, case, ml, rules, evidence)
        fusion_started = time.perf_counter()
        fusion = self.fusion_engine.fuse(ml, rules, evidence, context)
        recommendation = recommend(fusion, rules)
        report = self.explainer.build_report(
            case, ml, rules, evidence, context, fusion, recommendation, self.versions()
        )
        if self.audit_logger:
            self.audit_logger.log(report)
        timings = {
            "ml": t_ml,
            "rules": t_rules,
            "retrieval": t_retrieval,
            "context": t_context,
            "fusion": time.perf_counter() - fusion_started,
            "total": time.perf_counter() - started,
        }
        return CaseAnalysis(
            case=case,
            ml=ml,
            rules=rules,
            evidence=evidence,
            context=context,
            fusion=fusion,
            recommendation=recommendation,
            priority=priority(fusion),
            report=report,
            timings={stage: round(seconds, 4) for stage, seconds in timings.items()},
        )

    def analyze(self, case: PatientCase) -> DecisionReport:
        try:
            return self.run_case(case).report
        except Exception as exc:  # NFR5: a failing case becomes a failure report, never a crashed batch
            report = self.explainer.failed_report(case, type(exc).__name__, self.versions())
            if self.audit_logger:
                self.audit_logger.log(report)
            return report


class CaseScheduler:
    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.queue: list[PatientCase] = []

    def submit(self, cases: list[PatientCase]) -> None:
        self.queue.extend(case.model_copy(deep=True) for case in cases)

    def run(self, max_workers: int = 4) -> list[DecisionReport]:
        batch, self.queue = self.queue, []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(self.orchestrator.analyze, batch))


def create_orchestrator(
    settings: Settings | None = None,
    writer: DraftWriter | None = None,
    audit_logger: AuditLogger | None = None,
    knowledge_store: KnowledgeStore | None = None,
) -> Orchestrator:
    settings = settings or Settings()
    return Orchestrator(
        ml_agent=MLAnalysisAgent(load_risk_model(settings.ml.model)),
        rule_agent=RuleValidationAgent(),
        retriever=KnowledgeRetrievalAgent(
            knowledge_store, k=settings.retrieval.k, min_score=settings.retrieval.min_score
        ),
        context_agent=ClinicalContextAgent(writer, settings.retrieval.strong_score),
        fusion_engine=DecisionFusionEngine(settings.fusion),
        explainer=ExplanationGenerator(),
        audit_logger=audit_logger,
    )
