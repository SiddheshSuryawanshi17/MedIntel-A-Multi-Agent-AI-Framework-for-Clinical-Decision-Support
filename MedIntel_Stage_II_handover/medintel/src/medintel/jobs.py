"""Background analysis jobs (Phase 3): a long batch runs on worker threads while clients poll its progress, over
MCP (start_batch_analysis, get_job_status, cancel_job) or the REST API. Job rows live in SQLite next to the
analyses; a job that was running when the server stopped is marked interrupted on the next start."""

import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from medintel.audit import connect, now
from medintel.models import CaseAnalysis, PatientCase

JobState = Literal["queued", "running", "completed", "failed", "cancelled", "interrupted"]
FINISHED: tuple[JobState, ...] = ("completed", "failed", "cancelled", "interrupted")
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    status TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
"""
COLUMNS = (
    "job_id",
    "batch_id",
    "status",
    "done",
    "failed",
    "total",
    "created_by",
    "created_at",
    "started_at",
    "finished_at",
    "error",
    "cancel_requested",
)


class JobInfo(BaseModel):
    job_id: str
    batch_id: str
    status: JobState
    done: int
    failed: int
    total: int
    created_by: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    cancel_requested: bool

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


class JobStore:
    def __init__(self, db_path: str | Path) -> None:
        self.lock = threading.Lock()
        self.db = connect(db_path)
        self.db.executescript(SCHEMA)

    def _row(self, job_id: str) -> JobInfo | None:
        row = self.db.execute(f"SELECT {', '.join(COLUMNS)} FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return JobInfo(**dict(zip(COLUMNS, row, strict=True))) if row else None

    def create(self, batch_id: str, total: int, created_by: str) -> JobInfo:
        job_id = f"J-{secrets.token_hex(4)}"
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO jobs (job_id, batch_id, status, total, created_by, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, batch_id, "queued", total, created_by, now()),
            )
            job = self._row(job_id)
        assert job is not None
        return job

    def get(self, job_id: str) -> JobInfo | None:
        with self.lock:
            return self._row(job_id)

    def set(self, job_id: str, **fields: object) -> None:
        names = ", ".join(f"{k} = ?" for k in fields)
        with self.lock, self.db:
            self.db.execute(f"UPDATE jobs SET {names} WHERE job_id = ?", (*fields.values(), job_id))

    def count_case(self, job_id: str, ok: bool) -> None:
        column = "done" if ok else "failed"
        with self.lock, self.db:
            self.db.execute(f"UPDATE jobs SET {column} = {column} + 1 WHERE job_id = ?", (job_id,))

    def mark_interrupted(self) -> int:
        with self.lock, self.db:
            return self.db.execute(
                "UPDATE jobs SET status = 'interrupted', finished_at = ?, error = 'server stopped' "
                "WHERE status IN ('queued', 'running')",
                (now(),),
            ).rowcount

    def close(self) -> None:
        self.db.close()


class JobRunner:
    """Runs batch jobs on a small thread pool; each job analyses up to `per_job` cases at a time."""

    def __init__(
        self,
        db_path: str | Path,
        analyse: Callable[[PatientCase, str], CaseAnalysis],
        max_jobs: int = 2,
        per_job: int = 4,
    ) -> None:
        self.store = JobStore(db_path)
        self.store.mark_interrupted()
        self.analyse = analyse
        self.per_job = per_job
        self.pool = ThreadPoolExecutor(max_workers=max_jobs, thread_name_prefix="medintel-job")

    def start(self, batch_id: str, cases: list[PatientCase], created_by: str) -> JobInfo:
        job = self.store.create(batch_id, len(cases), created_by)
        self.pool.submit(self._run, job.job_id, batch_id, cases)
        return job

    def _run(self, job_id: str, batch_id: str, cases: list[PatientCase]) -> None:
        self.store.set(job_id, status="running", started_at=now())
        pending: set[Future[CaseAnalysis]] = set()
        queue = list(cases)
        try:
            with ThreadPoolExecutor(max_workers=self.per_job, thread_name_prefix="medintel-case") as inner:
                while queue or pending:
                    cancelled = bool(self.store.get(job_id).cancel_requested)  # type: ignore[union-attr]
                    while queue and len(pending) < self.per_job and not cancelled:
                        pending.add(inner.submit(self.analyse, queue.pop(0), batch_id))
                    if not pending:
                        break
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        self.store.count_case(job_id, future.exception() is None)
            job = self.store.get(job_id)
            status: JobState = "cancelled" if job and job.cancel_requested and queue else "completed"
            self.store.set(job_id, status=status, finished_at=now())
        except Exception as exc:  # a broken job must not take the server down (NFR5)
            self.store.set(job_id, status="failed", finished_at=now(), error=type(exc).__name__)

    def get(self, job_id: str) -> JobInfo | None:
        return self.store.get(job_id)

    def cancel(self, job_id: str) -> JobInfo | None:
        job = self.store.get(job_id)
        if job is None or job.finished:
            return job
        self.store.set(job_id, cancel_requested=1)
        return self.store.get(job_id)

    def wait(self, job_id: str, timeout: float = 60.0) -> JobInfo | None:
        """Block until the job finishes (tests and the CLI)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.store.get(job_id)
            if job is None or job.finished:
                return job
            time.sleep(0.05)
        return self.store.get(job_id)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
