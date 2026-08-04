"""Turns raw logged metric rows into a ready-to-render summary — the same
shape of responsibility split as `TrackRecordService`: the service computes
what something means (latest accuracy, how many days drifted), the page
only renders it. Without this, `pages/monitoring.py` would be the only
page in the app doing its own aggregation instead of calling a service for
it, which is exactly the kind of inconsistency a SOLID/SoC pass exists to
catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from monitoring.models import ModelMetricRecord
from monitoring.sqlite_tracker import ModelMetricsReader


@dataclass
class ModelMonitoringSummary:
    records: list[ModelMetricRecord] = field(default_factory=list)  # chronological, oldest first
    logged_days: int = 0
    drift_days: int = 0
    latest_directional_accuracy: Optional[float] = None


class ModelMonitoringService:
    def __init__(self, reader: ModelMetricsReader):
        self._reader = reader

    def get_known_tickers(self) -> list[str]:
        return self._reader.get_tickers()

    def get_summary(
        self, ticker: Optional[str] = None, model_name: Optional[str] = None, limit: int = 500
    ) -> ModelMonitoringSummary:
        records = self._reader.get_recent(ticker=ticker, model_name=model_name, limit=limit)
        chronological = sorted(records, key=lambda r: r.log_date)
        drift_days = sum(1 for r in records if r.has_drift)
        latest_accuracy = (
            chronological[-1].model_directional_accuracy if chronological else None
        )
        return ModelMonitoringSummary(
            records=chronological,
            logged_days=len(records),
            drift_days=drift_days,
            latest_directional_accuracy=latest_accuracy,
        )
