"""Storage for OTP codes — its own table, separate from `users`, for the
same reason track_record.db and monitoring.db are separate files from
users.db: this data has a different lifecycle (short-lived, ten minutes)
and a different owner (AuthService's verification flow) than account
records do.

One active code per (email, purpose): issuing a new code overwrites the
previous one rather than accumulating rows — same upsert-on-natural-key
idempotency every other repository in this app follows.

`issued_at` exists so AuthService can enforce a cooldown between two
emails for the same (email, purpose) — the repository just records when,
it doesn't decide whether that counts as "too soon"; that policy lives in
the service, same split as every other repository/service pair here."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Protocol

from infra.db import connect as _connect_hardened


@dataclass
class OtpRecord:
    code_hash: str
    expires_at: str
    attempts: int
    used_at: Optional[str]
    issued_at: str


class OtpRepository(Protocol):
    def issue(self, email: str, purpose: str, code_hash: str, expires_at: str, issued_at: str) -> None: ...

    def get(self, email: str, purpose: str) -> Optional[OtpRecord]: ...

    def record_failed_attempt(self, email: str, purpose: str) -> None: ...

    def mark_used(self, email: str, purpose: str) -> None: ...


class SqliteOtpRepository:
    def __init__(self, db_path: str = "users.db"):
        self._conn = _connect_hardened(db_path)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS otp_codes (
                    email      TEXT NOT NULL,
                    purpose    TEXT NOT NULL,
                    code_hash  TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    attempts   INTEGER NOT NULL DEFAULT 0,
                    used_at    TEXT,
                    issued_at  TEXT NOT NULL DEFAULT '1970-01-01T00:00:00+00:00',
                    PRIMARY KEY (email, purpose)
                )
                """
            )
            self._conn.commit()

    def issue(self, email: str, purpose: str, code_hash: str, expires_at: str, issued_at: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO otp_codes (email, purpose, code_hash, expires_at, attempts, used_at, issued_at)
                VALUES (?, ?, ?, ?, 0, NULL, ?)
                ON CONFLICT(email, purpose) DO UPDATE SET
                    code_hash=excluded.code_hash,
                    expires_at=excluded.expires_at,
                    attempts=0,
                    used_at=NULL,
                    issued_at=excluded.issued_at
                """,
                (email, purpose, code_hash, expires_at, issued_at),
            )
            self._conn.commit()

    def get(self, email: str, purpose: str) -> Optional[OtpRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT code_hash, expires_at, attempts, used_at, issued_at FROM otp_codes "
                "WHERE email = ? AND purpose = ?",
                (email, purpose),
            ).fetchone()
        return OtpRecord(*row) if row else None

    def record_failed_attempt(self, email: str, purpose: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE otp_codes SET attempts = attempts + 1 WHERE email = ? AND purpose = ?",
                (email, purpose),
            )
            self._conn.commit()

    def mark_used(self, email: str, purpose: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE otp_codes SET used_at = datetime('now') WHERE email = ? AND purpose = ?",
                (email, purpose),
            )
            self._conn.commit()
