"""User storage, owned by exactly one module. Previously app.py and
pages/prediction.py each opened their own sqlite3 connection to users.db
and re-declared the same schema; this is now the only place that happens.

Identity is an email address (not an arbitrary username) — see
auth/service.py for why, and otp_repository.py for the separate table
that backs email verification and password-reset codes.

`language` is the user's sidebar language choice at the moment they
registered, stored so OTP emails triggered later (a resend, a reset
requested from a different device/session) can still be sent in a
language the recipient actually chose — there's no "current session" to
read it from at that point, only what was saved here."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from infra.db import connect as _connect_hardened


@dataclass
class UserRecord:
    email: str
    password_hash: str
    failed_attempts: int
    locked_until: Optional[str]
    email_verified: bool
    language: str


class UserRepository(Protocol):
    def create_user(self, email: str, password_hash: str, language: str = "en") -> bool: ...

    def get_user(self, email: str) -> Optional[UserRecord]: ...

    def update_password(self, email: str, password_hash: str) -> bool: ...

    def mark_email_verified(self, email: str) -> None: ...

    def record_failed_login(self, email: str, locked_until: Optional[str]) -> None: ...

    def record_successful_login(self, email: str) -> None: ...

    def reset_login_attempts(self, email: str) -> None: ...


class SqliteUserRepository:
    def __init__(self, db_path: str = "users.db"):
        self._conn = _connect_hardened(db_path)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    email          TEXT PRIMARY KEY,
                    password_hash  TEXT NOT NULL,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until   TEXT,
                    email_verified INTEGER NOT NULL DEFAULT 0,
                    language       TEXT NOT NULL DEFAULT 'en'
                )
                """
            )
            self._conn.commit()

    def create_user(self, email: str, password_hash: str, language: str = "en") -> bool:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO users (email, password_hash, language) VALUES (?, ?, ?)",
                    (email, password_hash, language),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_user(self, email: str) -> Optional[UserRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT email, password_hash, failed_attempts, locked_until, email_verified, language "
                "FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row is None:
            return None
        return UserRecord(row[0], row[1], row[2], row[3], bool(row[4]), row[5])

    def update_password(self, email: str, password_hash: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?",
                (password_hash, email),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def mark_email_verified(self, email: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET email_verified = 1 WHERE email = ?", (email,)
            )
            self._conn.commit()

    def record_failed_login(self, email: str, locked_until: Optional[str]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET failed_attempts = failed_attempts + 1, locked_until = ? "
                "WHERE email = ?",
                (locked_until, email),
            )
            self._conn.commit()

    def record_successful_login(self, email: str) -> None:
        self._clear_attempts(email)

    def reset_login_attempts(self, email: str) -> None:
        # Same effect as record_successful_login but called for a different
        # reason (an expired lockout, before the password has even been
        # checked) — kept as its own named method so callers read clearly
        # rather than implying a login already succeeded.
        self._clear_attempts(email)

    def _clear_attempts(self, email: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE email = ?",
                (email,),
            )
            self._conn.commit()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def minutes_from_now_iso(minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
