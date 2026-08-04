"""User storage, owned by exactly one module. Previously app.py and
pages/prediction.py each opened their own sqlite3 connection to users.db
and re-declared the same schema; this is now the only place that happens."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol


@dataclass
class UserRecord:
    username: str
    password_hash: str
    failed_attempts: int
    locked_until: Optional[str]


class UserRepository(Protocol):
    def create_user(self, username: str, password_hash: str) -> bool: ...

    def get_user(self, username: str) -> Optional[UserRecord]: ...

    def update_password(self, username: str, password_hash: str) -> bool: ...

    def record_failed_login(self, username: str, locked_until: Optional[str]) -> None: ...

    def record_successful_login(self, username: str) -> None: ...

    def reset_login_attempts(self, username: str) -> None: ...


class SqliteUserRepository:
    def __init__(self, db_path: str = "users.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username       TEXT PRIMARY KEY,
                    password_hash  TEXT NOT NULL,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until   TEXT
                )
                """
            )
            self._conn.commit()

    def create_user(self, username: str, password_hash: str) -> bool:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, password_hash),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_user(self, username: str) -> Optional[UserRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT username, password_hash, failed_attempts, locked_until "
                "FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return UserRecord(*row) if row else None

    def update_password(self, username: str, password_hash: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (password_hash, username),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def record_failed_login(self, username: str, locked_until: Optional[str]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET failed_attempts = failed_attempts + 1, locked_until = ? "
                "WHERE username = ?",
                (locked_until, username),
            )
            self._conn.commit()

    def record_successful_login(self, username: str) -> None:
        self._clear_attempts(username)

    def reset_login_attempts(self, username: str) -> None:
        # Same effect as record_successful_login but called for a different
        # reason (an expired lockout, before the password has even been
        # checked) — kept as its own named method so callers read clearly
        # rather than implying a login already succeeded.
        self._clear_attempts(username)

    def _clear_attempts(self, username: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE username = ?",
                (username,),
            )
            self._conn.commit()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def minutes_from_now_iso(minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
