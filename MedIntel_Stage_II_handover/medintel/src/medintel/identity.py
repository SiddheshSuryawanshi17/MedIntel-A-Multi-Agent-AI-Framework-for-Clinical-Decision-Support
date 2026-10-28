"""Identities for the HTTP server (Phase 3): every person gets a bearer token tied to their roles, so roles come
from who is signed in rather than from the server's configuration. A token is a random 256-bit secret shown once;
only its SHA-256 is stored, and a disabled user's token stops working at once."""

import hashlib
import hmac
import re
import secrets
import threading
from pathlib import Path

from pydantic import BaseModel

from medintel.audit import connect, now
from medintel.config import ROLE_LABELS, Role

TOKEN_PREFIX = "mi_"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._'-]{0,63}$")
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY,
    roles TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT
);
"""


class Identity(BaseModel):
    name: str
    roles: list[Role]
    authenticated: bool = False

    def has_any(self, roles: tuple[Role, ...]) -> bool:
        return any(role in self.roles for role in roles)


class UserInfo(BaseModel):
    name: str
    roles: list[Role]
    created_at: str
    disabled: bool
    last_used_at: str | None


class UserError(ValueError):
    pass


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def check_roles(roles: list[str]) -> list[Role]:
    if not roles:
        raise UserError("Give at least one role.")
    unknown = [r for r in roles if r not in ROLE_LABELS]
    if unknown:
        raise UserError(f"Unknown role {', '.join(unknown)}. Roles: {', '.join(ROLE_LABELS)}.")
    return list(dict.fromkeys(roles))  # type: ignore[arg-type]


class UserStore:
    def __init__(self, db_path: str | Path) -> None:
        self.lock = threading.Lock()
        self.db = connect(db_path)
        self.db.executescript(SCHEMA)

    def count(self, active_only: bool = True) -> int:
        query = "SELECT COUNT(*) FROM users" + (" WHERE disabled = 0" if active_only else "")
        with self.lock:
            return int(self.db.execute(query).fetchone()[0])

    def add(self, name: str, roles: list[str]) -> str:
        """Create a user and return their token. The token is not stored and cannot be shown again."""
        name = name.strip()
        if not NAME_PATTERN.match(name):
            raise UserError("Names are 1 to 64 letters, digits, spaces, dots, hyphens or apostrophes.")
        granted = check_roles(roles)
        token = new_token()
        try:
            with self.lock, self.db:
                self.db.execute(
                    "INSERT INTO users (name, roles, token_hash, created_at) VALUES (?, ?, ?, ?)",
                    (name, ",".join(granted), token_hash(token), now()),
                )
        except Exception as exc:  # sqlite3.IntegrityError
            raise UserError(f"A user called {name} already exists.") from exc
        return token

    def rotate(self, name: str) -> str:
        token = new_token()
        with self.lock, self.db:
            changed = self.db.execute(
                "UPDATE users SET token_hash = ? WHERE name = ?", (token_hash(token), name)
            ).rowcount
        if not changed:
            raise UserError(f"No user called {name}.")
        return token

    def set_roles(self, name: str, roles: list[str]) -> None:
        granted = check_roles(roles)
        with self.lock, self.db:
            changed = self.db.execute("UPDATE users SET roles = ? WHERE name = ?", (",".join(granted), name)).rowcount
        if not changed:
            raise UserError(f"No user called {name}.")

    def set_disabled(self, name: str, disabled: bool) -> None:
        with self.lock, self.db:
            changed = self.db.execute("UPDATE users SET disabled = ? WHERE name = ?", (int(disabled), name)).rowcount
        if not changed:
            raise UserError(f"No user called {name}.")

    def list(self) -> list[UserInfo]:
        with self.lock:
            rows = self.db.execute(
                "SELECT name, roles, created_at, disabled, last_used_at FROM users ORDER BY name"
            ).fetchall()
        return [
            UserInfo(name=n, roles=r.split(","), created_at=c, disabled=bool(d), last_used_at=u)  # type: ignore[arg-type]
            for n, r, c, d, u in rows
        ]

    def verify(self, token: str) -> Identity | None:
        if not token or not token.startswith(TOKEN_PREFIX):
            return None
        digest = token_hash(token)
        with self.lock:
            row = self.db.execute(
                "SELECT name, roles, token_hash, disabled FROM users WHERE token_hash = ?", (digest,)
            ).fetchone()
            if row is None or not hmac.compare_digest(row[2], digest) or row[3]:
                return None
            with self.db:
                self.db.execute("UPDATE users SET last_used_at = ? WHERE name = ?", (now(), row[0]))
        return Identity(name=row[0], roles=row[1].split(","), authenticated=True)

    def close(self) -> None:
        self.db.close()
