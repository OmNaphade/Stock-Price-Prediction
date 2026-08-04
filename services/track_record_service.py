"""Turns a stream of predictions into an honest, checkable history: record
what was predicted when it was made, resolve it against the real close
once the target date has passed, and summarize how often that's actually
been right — the whole point being that this can't be gamed after the
fact, since every record is written before its outcome is known.

Every record belongs to one user — this is a personal accuracy history,
not a shared dashboard, so `record_prediction` and `get_track_record` both
require a username. `resolve_pending`, on the other hand, is intentionally
NOT scoped to one user: resolving a prediction against the real market
close is the same fact-check for everyone, so it walks every user's
pending predictions in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import log
from data_access.sources import MarketDataSource
from services.prediction_service import PredictionReport
from track_record.models import PredictionRecord
from track_record.repository import PredictionRecordRepository


@dataclass
class TrackRecordSummary:
    records: list[PredictionRecord] = field(default_factory=list)  # most recent first
    resolved_count: int = 0
    pending_count: int = 0
    directional_accuracy: Optional[float] = None
    mean_abs_pct_error: Optional[float] = None


class TrackRecordService:
    def __init__(self, repository: PredictionRecordRepository, data_source: MarketDataSource):
        self._repo = repository
        self._data_source = data_source

    def record_prediction(self, username: str, report: PredictionReport) -> None:
        record = PredictionRecord(
            username=username,
            ticker=report.ticker,
            model_name=report.model_name,
            made_at=datetime.now(timezone.utc).isoformat(),
            target_date=pd.Timestamp(report.target_date).date().isoformat(),
            last_close=report.last_close,
            predicted_close=report.predicted_next_close,
            predicted_log_return=report.predicted_log_return,
        )
        self._repo.save(record)

    def resolve_pending(self) -> int:
        """Finds every prediction whose target date has fully passed (not
        just arrived — 'today' might not have closed yet), across every
        user, and fills in what actually happened. Safe to call
        repeatedly; already-resolved records are never touched again."""
        pending = self._repo.get_unresolved_before(date.today().isoformat())
        # Two users predicting the same ticker on the same target date are
        # two separate records but the same real-world fact — one fetch
        # serves both instead of hitting the data source once per record.
        actual_close_cache: dict[tuple[str, str], Optional[float]] = {}
        resolved_count = 0
        for record in pending:
            cache_key = (record.ticker, record.target_date)
            if cache_key not in actual_close_cache:
                actual_close_cache[cache_key] = self._fetch_actual_close(
                    record.ticker, record.target_date
                )
            actual_close = actual_close_cache[cache_key]
            if actual_close is not None:
                self._repo.resolve(
                    record.username, record.ticker, record.model_name, record.target_date,
                    actual_close,
                )
                resolved_count += 1
        return resolved_count

    def _fetch_actual_close(self, ticker: str, target_date: str) -> Optional[float]:
        target = date.fromisoformat(target_date)
        # A week-wide window past the target date absorbs weekends/holidays
        # that shifted the *actual* next trading day later than the naive
        # business-day estimate used when the prediction was made.
        window_end = (target + timedelta(days=7)).isoformat()
        try:
            history = self._data_source.get_history(ticker, target_date, window_end)
        except Exception:
            log.warning("Resolving prediction failed for %s/%s", ticker, target_date, exc_info=True)
            return None
        if history.empty:
            return None
        on_or_after = history[history.index >= pd.Timestamp(target)]
        if on_or_after.empty:
            return None
        return float(on_or_after["Close"].iloc[0])

    def get_track_record(
        self,
        username: str,
        ticker: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 200,
    ) -> TrackRecordSummary:
        records = self._repo.get_history(username, ticker, model_name, limit)
        resolved = [r for r in records if r.is_resolved]
        pending = [r for r in records if not r.is_resolved]

        directional_accuracy = None
        mean_abs_pct_error = None
        if resolved:
            directional_accuracy = float(np.mean([r.direction_correct for r in resolved]))
            mean_abs_pct_error = float(np.mean([r.abs_pct_error for r in resolved]))

        return TrackRecordSummary(
            records=records,
            resolved_count=len(resolved),
            pending_count=len(pending),
            directional_accuracy=directional_accuracy,
            mean_abs_pct_error=mean_abs_pct_error,
        )
