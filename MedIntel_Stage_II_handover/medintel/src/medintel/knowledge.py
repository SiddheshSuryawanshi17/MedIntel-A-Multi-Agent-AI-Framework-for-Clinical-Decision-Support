"""Governed knowledge updates (FR9): a clinician-validated outcome reaches the index only after a Knowledge
Curator approves it, and every approval creates a new index version."""

import sqlite3
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from medintel.agents.embedding_store import validated_passage
from medintel.agents.retrieval_agent import KnowledgeStore
from medintel.audit import connect, now
from medintel.models import CaseAnalysis, EvidenceItem, PatientCase

SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    proposed_by TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    decided_by TEXT,
    decided_at TEXT,
    decision_notes TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_versions (
    version INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    proposal_id TEXT,
    description TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validated_cases (
    case_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    decision TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS base_files (
    fingerprint TEXT NOT NULL,
    version INTEGER NOT NULL,
    recorded_at TEXT NOT NULL
);
"""

DECISION_LABEL = {"high_risk": "high risk", "low_risk": "low risk"}
ProposalStatus = Literal["pending", "approved", "declined"]


class Proposal(BaseModel):
    proposal_id: str
    case_id: str
    decision: str
    proposed_by: str
    notes: str
    created_at: str
    status: ProposalStatus
    decided_by: str | None = None
    decided_at: str | None = None
    decision_notes: str | None = None


class KnowledgeVersion(BaseModel):
    version: int
    label: str
    created_at: str
    approved_by: str
    proposal_id: str | None
    description: str


PROPOSAL_COLUMNS = tuple(Proposal.model_fields)


def version_label(version: int) -> str:
    return f"kb-v{version}"


class KnowledgeRegistry:
    def __init__(
        self,
        db_path: str | Path,
        base_version: int = 3,
        base_description: str = "Initial index: 10 sample guideline summaries",
    ) -> None:
        self.lock = threading.Lock()
        self.db = connect(db_path)
        self.db.executescript(SCHEMA)
        with self.lock, self.db:
            if self.db.execute("SELECT COUNT(*) FROM knowledge_versions").fetchone()[0] == 0:
                self.db.execute(
                    "INSERT INTO knowledge_versions VALUES (?, ?, ?, ?, ?)",
                    (base_version, now(), "knowledge_curator", None, base_description),
                )

    def ensure_base(self, fingerprint: str, description: str) -> int:
        """Record which knowledge base file the index is built from. A different file than last time becomes a
        new index version, so every report names an index whose content can be identified."""
        with self.lock, self.db:
            last = self.db.execute("SELECT fingerprint FROM base_files ORDER BY rowid DESC LIMIT 1").fetchone()
            version = int(self.db.execute("SELECT MAX(version) FROM knowledge_versions").fetchone()[0])
            if last is None or last[0] != fingerprint:
                if last is not None:
                    version += 1
                    self.db.execute(
                        "INSERT INTO knowledge_versions VALUES (?, ?, ?, ?, ?)",
                        (version, now(), "knowledge base file", None, f"Knowledge base replaced: {description}"),
                    )
                self.db.execute("INSERT INTO base_files VALUES (?, ?, ?)", (fingerprint, version, now()))
        return version

    def validated(self) -> list[tuple[str, str, str, str]]:
        """(case_id, decision, payload, approved_by) for every approved outcome, oldest first."""
        with self.lock:
            rows = self.db.execute(
                "SELECT v.case_id, v.decision, v.payload, k.approved_by FROM validated_cases v "
                "JOIN knowledge_versions k ON k.version = v.version ORDER BY v.version"
            ).fetchall()
        return [(str(c), str(d), str(p), str(a)) for c, d, p, a in rows]

    def current_version(self) -> int:
        with self.lock:
            return int(self.db.execute("SELECT MAX(version) FROM knowledge_versions").fetchone()[0])

    def label(self) -> str:
        return version_label(self.current_version())

    def _proposal(self, row: sqlite3.Row | tuple[object, ...] | None) -> Proposal | None:
        return Proposal(**dict(zip(PROPOSAL_COLUMNS, row, strict=True))) if row else None

    def get(self, proposal_id: str) -> Proposal | None:
        with self.lock:
            row = self.db.execute(
                f"SELECT {', '.join(PROPOSAL_COLUMNS)} FROM proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        return self._proposal(row)

    def pending_for(self, case_id: str) -> Proposal | None:
        with self.lock:
            row = self.db.execute(
                f"SELECT {', '.join(PROPOSAL_COLUMNS)} FROM proposals WHERE case_id = ? AND status = 'pending'",
                (case_id,),
            ).fetchone()
        return self._proposal(row)

    def propose(self, analysis: CaseAnalysis, proposed_by: str, notes: str) -> Proposal | None:
        """Queue a validated outcome for the curator. Only confirmed high- or low-risk decisions qualify."""
        decision = analysis.report.decision
        if decision not in DECISION_LABEL:
            return None
        existing = self.pending_for(analysis.case.case_id)
        if existing:
            return existing
        with self.lock, self.db:
            count = self.db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
            proposal = Proposal(
                proposal_id=f"KU-{count + 1:03d}",
                case_id=analysis.case.case_id,
                decision=decision,
                proposed_by=proposed_by,
                notes=notes,
                created_at=now(),
                status="pending",
            )
            self.db.execute(
                f"INSERT INTO proposals ({', '.join(PROPOSAL_COLUMNS)}, payload) "
                f"VALUES ({', '.join('?' * (len(PROPOSAL_COLUMNS) + 1))})",
                (*(getattr(proposal, c) for c in PROPOSAL_COLUMNS), analysis.model_dump_json()),
            )
        return proposal

    def decide(self, proposal_id: str, approve: bool, curator: str, notes: str = "") -> tuple[Proposal, int, int]:
        """Approve or decline a pending proposal. Returns the proposal and the index version before and after."""
        proposal = self.get(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        if proposal.status != "pending":
            raise ValueError(f"{proposal_id} was already {proposal.status}")
        before = self.current_version()
        decided = proposal.model_copy(
            update={
                "status": "approved" if approve else "declined",
                "decided_by": curator,
                "decided_at": now(),
                "decision_notes": notes,
            }
        )
        with self.lock, self.db:
            self.db.execute(
                "UPDATE proposals SET status = ?, decided_by = ?, decided_at = ?, decision_notes = ? "
                "WHERE proposal_id = ?",
                (decided.status, decided.decided_by, decided.decided_at, decided.decision_notes, proposal_id),
            )
            if approve:
                payload = self.db.execute(
                    "SELECT payload FROM proposals WHERE proposal_id = ?", (proposal_id,)
                ).fetchone()[0]
                after = before + 1
                description = (
                    f"Added validated outcome from {proposal.case_id} "
                    f"({DECISION_LABEL[proposal.decision]} confirmed by a clinician)"
                )
                self.db.execute(
                    "INSERT INTO knowledge_versions VALUES (?, ?, ?, ?, ?)",
                    (after, decided.decided_at, curator, proposal_id, description),
                )
                self.db.execute(
                    "INSERT INTO validated_cases VALUES (?, ?, ?, ?)",
                    (proposal.case_id, after, proposal.decision, payload),
                )
        return decided, before, before + 1 if approve else before

    def proposals(self, status: ProposalStatus | None = None) -> list[Proposal]:
        query = f"SELECT {', '.join(PROPOSAL_COLUMNS)} FROM proposals"
        params: tuple[object, ...] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        with self.lock:
            rows = self.db.execute(query + " ORDER BY proposal_id", params).fetchall()
        return [p for p in (self._proposal(row) for row in rows) if p]

    def history(self) -> list[KnowledgeVersion]:
        with self.lock:
            rows = self.db.execute(
                "SELECT version, created_at, approved_by, proposal_id, description FROM knowledge_versions "
                "ORDER BY version DESC"
            ).fetchall()
        return [
            KnowledgeVersion(
                version=v, label=version_label(v), created_at=c, approved_by=a, proposal_id=p, description=d
            )
            for v, c, a, p, d in rows
        ]

    def validated_case_ids(self) -> list[str]:
        with self.lock:
            return [row[0] for row in self.db.execute("SELECT case_id FROM validated_cases ORDER BY version")]

    def close(self) -> None:
        self.db.close()


class GovernedKnowledgeStore:
    """A knowledge store whose version comes from the registry, so every report names the index it used. Approved
    outcomes are added to the base store before the next search when the base store can hold them."""

    def __init__(self, base: KnowledgeStore, registry: KnowledgeRegistry) -> None:
        self.base = base
        self.registry = registry
        self._synced = 0
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return self.registry.label()

    @property
    def method(self) -> str:
        return str(getattr(self.base, "method", "case match (demo knowledge base rules)"))

    def sync(self) -> None:
        add = getattr(self.base, "add_validated", None)
        if add is None:
            return
        with self._lock:
            rows = self.registry.validated()
            for case_id, decision, payload, approved_by in rows[self._synced :]:
                add(validated_passage(case_id, decision, payload, approved_by))
            self._synced = len(rows)

    def search(self, case: PatientCase, k: int, min_score: float) -> list[EvidenceItem]:
        self.sync()
        return self.base.search(case, k, min_score)

    def search_text(self, query: str, k: int) -> list[EvidenceItem]:
        self.sync()
        search_text = getattr(self.base, "search_text", None)
        return search_text(query, k) if search_text else []
