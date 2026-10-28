"""AuditLogger (FR8, NFR1, NFR6): an append-only SQLite trail, plus the store behind the MCP handles."""

import json
import secrets
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from medintel.ingestion import IngestionResult
from medintel.models import AuditRecord, CaseAnalysis, DecisionReport

AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    case_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    rulebase_version TEXT NOT NULL,
    retrieval_ids TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event TEXT NOT NULL DEFAULT 'report_created',
    actor TEXT NOT NULL DEFAULT 'system'
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""

STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analyses (
    case_id TEXT PRIMARY KEY,
    analyzed_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    batch_id TEXT
);
"""

AUDIT_COLUMNS = (
    "case_id",
    "model_version",
    "rulebase_version",
    "retrieval_ids",
    "status",
    "timestamp",
    "event",
    "actor",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path), check_same_thread=False)


def add_missing_columns(db: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Bring a database created by an earlier version up to the current columns."""
    existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    with db:
        for name, declaration in columns.items():
            if name not in existing:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


class AuditLogger:
    def __init__(self, db_path: str | Path) -> None:
        self.lock = threading.Lock()
        self.db = connect(db_path)
        self.db.executescript(AUDIT_SCHEMA)
        add_missing_columns(
            self.db,
            "audit",
            {"event": "TEXT NOT NULL DEFAULT 'report_created'", "actor": "TEXT NOT NULL DEFAULT 'system'"},
        )

    def log(self, report: DecisionReport, event: str = "report_created", actor: str = "system") -> AuditRecord:
        record = AuditRecord(
            case_id=report.case_id,
            model_version=report.versions.model,
            rulebase_version=report.versions.rulebase,
            retrieval_ids=",".join(report.evidence_refs),
            status=report.status,
            timestamp=now(),
            event=event,
            actor=actor,
        )
        with self.lock, self.db:
            self.db.execute(
                f"INSERT INTO audit ({', '.join(AUDIT_COLUMNS)}) VALUES ({', '.join('?' * len(AUDIT_COLUMNS))})",
                tuple(getattr(record, column) for column in AUDIT_COLUMNS),
            )
        return record

    def records(self, case_id: str | None = None, limit: int = 100) -> list[AuditRecord]:
        query = f"SELECT {', '.join(AUDIT_COLUMNS)} FROM audit"
        params: tuple[object, ...] = ()
        if case_id:
            query += " WHERE case_id = ?"
            params = (case_id,)
        query += " ORDER BY rowid DESC LIMIT ?"
        with self.lock:
            rows = self.db.execute(query, (*params, limit)).fetchall()
        return [AuditRecord(**dict(zip(AUDIT_COLUMNS, row, strict=True))) for row in rows]

    def close(self) -> None:
        self.db.close()


class CaseStore:
    """Validated batches and analyses, addressed by server-minted batch_id and case_id handles."""

    def __init__(self, db_path: str | Path) -> None:
        self.lock = threading.Lock()
        self.db = connect(db_path)
        self.db.executescript(STORE_SCHEMA)
        add_missing_columns(self.db, "analyses", {"batch_id": "TEXT"})

    def save_batch(self, batch: IngestionResult) -> str:
        batch_id = f"B-{secrets.token_hex(4)}"
        with self.lock, self.db:
            self.db.execute("INSERT INTO batches VALUES (?, ?, ?)", (batch_id, now(), batch.model_dump_json()))
        return batch_id

    def list_batches(self, limit: int = 20) -> list[tuple[str, str, IngestionResult]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT batch_id, created_at, payload FROM batches ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [(b, c, IngestionResult.model_validate_json(p)) for b, c, p in rows]

    def analyzed_count(self, batch_id: str) -> int:
        with self.lock:
            return int(self.db.execute("SELECT COUNT(*) FROM analyses WHERE batch_id = ?", (batch_id,)).fetchone()[0])

    def get_batch(self, batch_id: str) -> IngestionResult | None:
        with self.lock:
            row = self.db.execute("SELECT payload FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        return IngestionResult.model_validate_json(row[0]) if row else None

    def save_analysis(self, analysis: CaseAnalysis, batch_id: str | None = None) -> None:
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO analyses (case_id, analyzed_at, payload, batch_id) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(case_id) DO UPDATE SET analyzed_at = excluded.analyzed_at, payload = excluded.payload, "
                "batch_id = COALESCE(excluded.batch_id, analyses.batch_id)",
                (analysis.case.case_id, now(), analysis.model_dump_json(), batch_id),
            )

    def update_report(self, analysis: CaseAnalysis) -> None:
        with self.lock, self.db:
            self.db.execute(
                "UPDATE analyses SET payload = ? WHERE case_id = ?",
                (analysis.model_dump_json(), analysis.case.case_id),
            )

    def get_analysis(self, case_id: str) -> CaseAnalysis | None:
        with self.lock:
            row = self.db.execute("SELECT payload FROM analyses WHERE case_id = ?", (case_id,)).fetchone()
        return CaseAnalysis.model_validate_json(row[0]) if row else None

    def batch_analyses(self, batch_id: str) -> list[CaseAnalysis]:
        with self.lock:
            rows = self.db.execute("SELECT payload FROM analyses WHERE batch_id = ?", (batch_id,)).fetchall()
        return [CaseAnalysis.model_validate_json(row[0]) for row in rows]

    def close(self) -> None:
        self.db.close()


def export_report(report: DecisionReport, out_dir: str | Path) -> Path:
    path = Path(out_dir) / f"{report.case_id}.json"
    path.write_text(json.dumps(report.model_dump(), indent=2) + "\n", encoding="utf-8")
    return path
