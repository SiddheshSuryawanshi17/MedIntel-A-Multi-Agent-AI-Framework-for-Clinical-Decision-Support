"""DataIngestionService and SchemaValidator (FR1, FR2): every record is checked before any agent runs."""

import csv
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from medintel.models import Event, PatientCase, RejectedRecord, ValidationProblem
from medintel.numeric import number_text
from medintel.schema import PATIENT_FIELDS, REQUIRED_COLUMNS, STRUCTURED_KEYS, Value, canonical_choice

TRUE_WORDS = {"yes", "true", "1", "y"}
FALSE_WORDS = {"no", "false", "0", "n"}


@dataclass(frozen=True)
class RawRecord:
    row: int
    values: dict[str, str]


@dataclass(frozen=True)
class ParsedCSV:
    header: list[str]
    records: list[RawRecord]
    missing_columns: list[str]


class IngestionResult(BaseModel):
    source_name: str
    records_read: int
    missing_columns: list[str]
    cases: list[PatientCase]
    rejected: list[RejectedRecord]


def _rows(text: str) -> list[list[str]]:
    rows = csv.reader(io.StringIO(text.removeprefix("﻿"), newline=""))
    return [row for row in rows if any(cell.strip() for cell in row)]


def parse_csv(text: str) -> ParsedCSV:
    rows = _rows(text)
    if not rows:
        return ParsedCSV([], [], list(REQUIRED_COLUMNS))
    header = [h.strip().lower() for h in rows[0]]
    records = [
        RawRecord(i + 2, {h: (cells[j] if j < len(cells) else "").strip() for j, h in enumerate(header)})
        for i, cells in enumerate(rows[1:])
    ]
    missing = [key for key in REQUIRED_COLUMNS if key not in header]
    return ParsedCSV(header, records, missing)


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


class SchemaValidator:
    def check(self, raw: dict[str, str], seen: set[str]) -> tuple[dict[str, Value], list[ValidationProblem]]:
        record: dict[str, Value] = {}
        problems: list[ValidationProblem] = []

        def problem(key: str, message: str, test_case: str | None = None) -> None:
            problems.append(ValidationProblem(field=key, message=message, test_case=test_case))

        for spec in PATIENT_FIELDS:
            text = raw.get(spec.key, "").strip()
            if spec.kind == "id":
                if not text:
                    problem(spec.key, "field required", "TC-01")
                elif text in seen:
                    problem(spec.key, "repeated patient_id")
                record[spec.key] = text
            elif spec.kind == "text":
                record[spec.key] = text
            elif spec.kind == "enum":
                choice = canonical_choice(spec, text) if text else None
                if not text:
                    if spec.required:
                        problem(spec.key, "field required")
                elif choice is None:
                    problem(spec.key, f"expected {' or '.join(spec.values)}")
                if choice is not None or not spec.required:
                    record[spec.key] = choice
                else:
                    record[spec.key] = text.upper()
            elif spec.kind == "bool":
                word = text.lower()
                if word in TRUE_WORDS:
                    record[spec.key] = True
                elif word in FALSE_WORDS:
                    record[spec.key] = False
                else:
                    if text or spec.required:
                        problem(spec.key, "expected yes or no" if text else "field required")
                    record[spec.key] = None
            else:
                if text == "":
                    if spec.required:
                        problem(spec.key, "field required")
                    record[spec.key] = None
                    continue
                value = _number(text)
                if value is None or (spec.kind == "int" and not value.is_integer()):
                    kind = "whole number" if spec.kind == "int" else "number"
                    problem(spec.key, f'"{text}" is not a {kind}')
                    record[spec.key] = None
                    continue
                bounds = f"{number_text(spec.min)}-{number_text(spec.max)}"
                if spec.fahrenheit_check and 93 <= value <= 110:
                    problem(
                        spec.key,
                        f"{number_text(value)} looks like °F; expected °C ({bounds}). Not converted.",
                        "TC-02",
                    )
                elif value < spec.min or value > spec.max:
                    problem(spec.key, f"{number_text(value)} is outside {bounds} {spec.unit}")
                record[spec.key] = int(value) if spec.kind == "int" else value
        return record, problems

    def validate(self, raw: dict[str, str]) -> list[ValidationProblem]:
        return self.check(raw, set())[1]


def case_id_for(patient_id: str) -> str:
    digits = "".join(re.findall(r"\d+", patient_id)) or "0000"
    return f"C-{digits[-4:].rjust(4, '0')}"


def build_case(record: dict[str, Value], source: str, events: dict[str, list[Event]] | None = None) -> PatientCase:
    patient_id = str(record["patient_id"])
    return PatientCase(
        case_id=case_id_for(patient_id),
        patient_id=patient_id,
        x_s={key: record.get(key) for key in STRUCTURED_KEYS},
        x_t=str(record.get("notes") or ""),
        x_l=list((events or {}).get(patient_id, [])),
        source=source,
    )


def parse_events(text: str) -> dict[str, list[Event]]:
    rows = _rows(text)
    if not rows:
        return {}
    header = [h.strip().lower() for h in rows[0]]
    events: dict[str, list[Event]] = {}
    for cells in rows[1:]:
        row = dict(zip(header, (c.strip() for c in cells), strict=False))
        if row.get("patient_id"):
            events.setdefault(row["patient_id"], []).append(Event(time=row.get("time", ""), text=row.get("text", "")))
    return events


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return number_text(value)
    return str(value)


class DataIngestionService:
    def __init__(self, validator: SchemaValidator | None = None) -> None:
        self.validator = validator or SchemaValidator()

    def load_text(self, text: str, source_name: str, events: dict[str, list[Event]] | None = None) -> IngestionResult:
        parsed = parse_csv(text)
        cases: list[PatientCase] = []
        rejected: list[RejectedRecord] = []
        if not parsed.missing_columns:
            seen: set[str] = set()
            for raw in parsed.records:
                record, problems = self.validator.check(raw.values, seen)
                if record.get("patient_id"):
                    seen.add(str(record["patient_id"]))
                if problems:
                    rejected.append(
                        RejectedRecord(row=raw.row, patient_id=str(record.get("patient_id") or ""), problems=problems)
                    )
                else:
                    cases.append(build_case(record, f"{source_name}:{raw.row}", events))
        return IngestionResult(
            source_name=source_name,
            records_read=len(parsed.records),
            missing_columns=parsed.missing_columns,
            cases=cases,
            rejected=rejected,
        )

    def load(self, path: str | Path, events_path: str | Path | None = None) -> IngestionResult:
        csv_path = Path(path)
        events = parse_events(Path(events_path).read_text(encoding="utf-8")) if events_path else None
        return self.load_text(csv_path.read_text(encoding="utf-8"), csv_path.name, events)

    def load_record(self, record: dict[str, Any], source_name: str = "direct-input") -> IngestionResult:
        columns = [spec.key for spec in PATIENT_FIELDS]
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(columns)
        writer.writerow([as_text(record.get(key)) for key in columns])
        return self.load_text(buffer.getvalue(), source_name)
