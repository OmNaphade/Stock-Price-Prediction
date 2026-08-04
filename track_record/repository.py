"""Storage for prediction records — its own SQLite file, separate from
`users.db`, so auth and track-record data can be backed by different
stores later without one dragging the other along.

Every record belongs to exactly one user (`username` is part of the
primary key) — this is a personal accuracy history, not a shared one:
two users predicting the same ticker/model on the same day get two
separate rows, never one overwriting the other.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Optional, Protocol

from .models import PredictionRecord


class PredictionRecordRepository(Protocol):
    def save(self, record: PredictionRecord) -> None: ...

    def resolve(
        self, username: str, ticker: str, model_name: str, target_date: str, actual_close: float
    ) -> None: ...

    def get_unresolved_before(self, cutoff_date: str) -> list[PredictionRecord]: ...

    def get_history(
        self,
        username: str,
        ticker: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 200,
    ) -> list[PredictionRecord]: ...


_COLUMNS = (
    "username", "ticker", "model_name", "made_at", "target_date",
    "last_close", "predicted_close", "predicted_log_return",
    "actual_close", "resolved_at",
)


def _row_to_record(row: tuple) -> PredictionRecord:
    return PredictionRecord(**dict(zip(_COLUMNS, row)))


class SqlitePredictionRecordRepository:
    def __init__(self, db_path: str = "track_record.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_records (
                    username             TEXT NOT NULL,
                    ticker               TEXT NOT NULL,
                    model_name           TEXT NOT NULL,
                    made_at              TEXT NOT NULL,
                    target_date          TEXT NOT NULL,
                    last_close           REAL NOT NULL,
                    predicted_close      REAL NOT NULL,
                    predicted_log_return REAL NOT NULL,
                    actual_close         REAL,
                    resolved_at          TEXT,
                    PRIMARY KEY (username, ticker, model_name, target_date)
                )
                """
            )
            self._conn.commit()

    def save(self, record: PredictionRecord) -> None:
        # Upsert: re-analyzing the same ticker/model before its target date
        # resolves just updates that user's pending prediction rather than
        # creating a duplicate row. Deliberately does not touch
        # actual_close / resolved_at — a record can only move from pending
        # to resolved via resolve(), never backward.
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO prediction_records
                    (username, ticker, model_name, made_at, target_date, last_close,
                     predicted_close, predicted_log_return)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username, ticker, model_name, target_date) DO UPDATE SET
                    made_at=excluded.made_at,
                    last_close=excluded.last_close,
                    predicted_close=excluded.predicted_close,
                    predicted_log_return=excluded.predicted_log_return
                """,
                (
                    record.username, record.ticker, record.model_name, record.made_at,
                    record.target_date, record.last_close, record.predicted_close,
                    record.predicted_log_return,
                ),
            )
            self._conn.commit()

    def resolve(
        self, username: str, ticker: str, model_name: str, target_date: str, actual_close: float
    ) -> None:
        # `AND actual_close IS NULL` makes this idempotent in the strong
        # sense: calling it again for an already-resolved record is a
        # deliberate no-op, not just "harmless if callers behave." Once
        # written, a real historical outcome can't be silently overwritten
        # by a second call.
        with self._lock:
            self._conn.execute(
                """
                UPDATE prediction_records
                SET actual_close = ?, resolved_at = ?
                WHERE username = ? AND ticker = ? AND model_name = ? AND target_date = ?
                  AND actual_close IS NULL
                """,
                (actual_close, _utcnow_iso(), username, ticker, model_name, target_date),
            )
            self._conn.commit()

    def get_unresolved_before(self, cutoff_date: str) -> list[PredictionRecord]:
        # Deliberately not scoped to one user — resolving predictions
        # against real market data is the same operation for everyone;
        # each returned record already carries its own username so the
        # caller can resolve() the right row per user.
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM prediction_records "
                "WHERE actual_close IS NULL AND target_date < ?",
                (cutoff_date,),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def get_history(
        self,
        username: str,
        ticker: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 200,
    ) -> list[PredictionRecord]:
        clauses, params = ["username = ?"], [username]
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        where = f"WHERE {' AND '.join(clauses)}"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM prediction_records "
                f"{where} ORDER BY target_date DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_row_to_record(row) for row in rows]


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
