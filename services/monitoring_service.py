"""Turns raw logged metric rows into a ready-to-render summary — the same
shape of responsibility split as `TrackRecordService`: the service computes
what something means (latest accuracy, how many days drifted), the page
only renders it. Without this, `pages/monitoring.py` would be the only
page in the app doing its own aggregation instead of calling a service for
it, which is exactly the kind of inconsistency a SOLID/SoC pass exists to
catch.

Also owns turning a `PredictionReport` into a logged metric row
(`log_from_report`) — moved here from `PredictionService`, which used to
log to monitoring as a side effect of `analyze()`. That made
PredictionService implicitly aware of "who's asking" once monitoring
became per-user; pulling logging out into an explicit call the caller
makes (mirroring `TrackRecordService.record_prediction`) keeps
PredictionService a pure ticker-in, report-out function with no notion of
users at all.

Depends on two narrow interfaces, not one wide one, even though the same
concrete `SqliteExperimentTracker` instance satisfies both in practice:
`ExperimentTracker` for writing, `ModelMetricsReader` for reading —
Interface Segregation, not just Dependency Inversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from monitoring.experiment_tracking import ExperimentTracker
from monitoring.models import ModelMetricRecord
from monitoring.sqlite_tracker import ModelMetricsReader
from services.prediction_service import PredictionReport


@dataclass
class ModelMonitoringSummary:
    records: list[ModelMetricRecord] = field(default_factory=list)  # chronological, oldest first
    logged_days: int = 0
    drift_days: int = 0
    latest_directional_accuracy: Optional[float] = None


class ModelMonitoringService:
    def __init__(self, tracker: ExperimentTracker, reader: ModelMetricsReader):
        self._tracker = tracker
        self._reader = reader

    def log_from_report(self, username: str, report: PredictionReport) -> None:
        try:
            self._tracker.log_backtest(
                username=username,
                ticker=report.ticker,
                model_name=report.model_name,
                params={
                    "n_folds": len(report.model_backtest.folds),
                    "n_features": len(report.feature_columns),
                },
                metrics={
                    "model_directional_accuracy": report.model_backtest.mean_directional_accuracy,
                    "baseline_directional_accuracy": report.baseline_backtest.mean_directional_accuracy,
                    "model_rmse_price": report.model_backtest.mean_rmse_price,
                    "baseline_rmse_price": report.baseline_backtest.mean_rmse_price,
                    "has_drift": report.drift_report.has_drift if report.drift_report else None,
                    "drifted_feature_count": (
                        len(report.drift_report.drifted_features) if report.drift_report else None
                    ),
                },
            )
        except Exception:
            from config import log

            log.warning(
                "Monitoring log failed for %s/%s/%s", username, report.ticker, report.model_name,
                exc_info=True,
            )

    def get_known_tickers(self, username: str) -> list[str]:
        return self._reader.get_tickers(username)

    def get_summary(
        self,
        username: str,
        ticker: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 500,
    ) -> ModelMonitoringSummary:
        records = self._reader.get_recent(username, ticker=ticker, model_name=model_name, limit=limit)
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
