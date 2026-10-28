"""Labelling sheets for the two human judgements in the evaluation plan: evidence relevance (precision at 5,
Section 6.5.3) and the explanation rubric (Section 7.2). The team fills them in; `medintel evaluate` reads them back."""

import csv
from pathlib import Path
from typing import Any

from medintel.models import CaseAnalysis
from medintel.schema import FIELDS, abnormal, format_value

RELEVANCE_COLUMNS = (
    "case_id",
    "rank",
    "source_id",
    "title",
    "matched_on",
    "score",
    "case_context",
    "passage",
    "relevant",
    "labeller",
    "comment",
)
RUBRIC_COLUMNS = (
    "case_id",
    "priority",
    "decision",
    "confidence",
    "conflicts",
    "summary",
    "recommendation",
    "written_by",
    "completeness",
    "factual_consistency",
    "usefulness",
    "reading_seconds",
    "reviewer",
    "comments",
)


def case_context(a: CaseAnalysis) -> str:
    x = a.case.x_s
    parts = [f"{x.get('age')}-year-old {'man' if x.get('sex') == 'M' else 'woman'}"]
    parts += [f"{FIELDS[k].label} {format_value(k, v)}" for k, v in x.items() if abnormal(k, v) == "high"]
    if x.get("smoker"):
        parts.append("smoker")
    if a.case.x_t:
        parts.append(f"notes: {a.case.x_t}")
    return "; ".join(parts)


def relevance_rows(analyses: list[CaseAnalysis]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in analyses:
        context = case_context(a)
        for rank, e in enumerate(a.evidence[:5], start=1):
            rows.append(
                {
                    "case_id": a.case.case_id,
                    "rank": rank,
                    "source_id": e.source_id,
                    "title": e.title,
                    "matched_on": e.matched_on,
                    "score": e.score,
                    "case_context": context,
                    "passage": e.passage,
                    "relevant": "",
                    "labeller": "",
                    "comment": "",
                }
            )
    return rows


def rubric_sample(analyses: list[CaseAnalysis], size: int = 12) -> list[CaseAnalysis]:
    """A fixed mix of conflicting, low-confidence and clear cases (Section 7.2)."""
    conflicting = [a for a in analyses if a.context.conflicts]
    uncertain = [a for a in analyses if a.fusion.below_threshold and not a.context.conflicts]
    clear = [a for a in analyses if not a.fusion.below_threshold and not a.context.conflicts]
    chosen: list[CaseAnalysis] = []
    for group in (conflicting, uncertain, clear):
        for a in group:
            if len(chosen) < size and a not in chosen and len([c for c in chosen if c in group]) < size // 3 + 1:
                chosen.append(a)
    for a in analyses:
        if len(chosen) >= size:
            break
        if a not in chosen:
            chosen.append(a)
    return chosen


def rubric_rows(analyses: list[CaseAnalysis]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": a.case.case_id,
            "priority": a.priority.label,
            "decision": a.fusion.decision,
            "confidence": a.fusion.confidence,
            "conflicts": " | ".join(a.context.conflicts),
            "summary": a.report.summary,
            "recommendation": a.recommendation,
            "written_by": a.context.generated_by,
            "completeness": "",
            "factual_consistency": "",
            "usefulness": "",
            "reading_seconds": "",
            "reviewer": "",
            "comments": "",
        }
        for a in rubric_sample(analyses)
    ]


def write_sheet(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]], force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return True
