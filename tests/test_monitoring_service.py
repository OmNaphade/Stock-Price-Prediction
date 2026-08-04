from __future__ import annotations

from monitoring.models import ModelMetricRecord
from monitoring.sqlite_tracker import SqliteExperimentTracker
from services.monitoring_service import ModelMonitoringService


def _tracker(tmp_path) -> SqliteExperimentTracker:
    return SqliteExperimentTracker(str(tmp_path / "test_monitoring_service.db"))


def _log(tracker, ticker, model_name, **metric_overrides):
    metrics = dict(
        model_directional_accuracy=0.5,
        baseline_directional_accuracy=0.02,
        model_rmse_price=3.0,
        baseline_rmse_price=3.5,
        has_drift=False,
        drifted_feature_count=0,
    )
    metrics.update(metric_overrides)
    tracker.log_backtest(ticker, model_name, {}, metrics)


def test_get_known_tickers_reflects_logged_data(tmp_path):
    tracker = _tracker(tmp_path)
    _log(tracker, "AAPL", "Ridge (linear)")
    _log(tracker, "MSFT", "Ridge (linear)")

    service = ModelMonitoringService(tracker)
    assert service.get_known_tickers() == ["AAPL", "MSFT"]


def test_get_summary_computes_drift_days_and_latest_accuracy(tmp_path):
    tracker = _tracker(tmp_path)
    _log(tracker, "AAPL", "Ridge (linear)", has_drift=True, model_directional_accuracy=0.6)

    service = ModelMonitoringService(tracker)
    summary = service.get_summary(ticker="AAPL")
    assert summary.logged_days == 1
    assert summary.drift_days == 1
    assert summary.latest_directional_accuracy == 0.6


class _FakeReader:
    """A ModelMetricsReader stand-in that deliberately returns records out
    of order, so the test proves the service does the sorting — not that
    whatever storage happens to already return them sorted."""

    def __init__(self, records):
        self._records = records

    def get_recent(self, ticker=None, model_name=None, limit=500):
        return self._records

    def get_tickers(self):
        return sorted({r.ticker for r in self._records})


def _fake_record(log_date: str) -> ModelMetricRecord:
    return ModelMetricRecord(
        ticker="AAPL", model_name="Ridge (linear)", log_date=log_date, logged_at=log_date,
        n_folds=5, n_features=10, model_directional_accuracy=0.5,
        baseline_directional_accuracy=0.02, model_rmse_price=3.0, baseline_rmse_price=3.5,
        has_drift=False, drifted_feature_count=0,
    )


def test_get_summary_records_are_chronological_oldest_first():
    """The page charts these directly — they must already be in the right
    order, not something the page has to sort itself."""
    out_of_order = [_fake_record("2026-08-03"), _fake_record("2026-08-01"), _fake_record("2026-08-02")]
    service = ModelMonitoringService(_FakeReader(out_of_order))

    summary = service.get_summary(ticker="AAPL")
    dates = [r.log_date for r in summary.records]
    assert dates == ["2026-08-01", "2026-08-02", "2026-08-03"]


def test_get_summary_empty_when_nothing_logged(tmp_path):
    service = ModelMonitoringService(_tracker(tmp_path))
    summary = service.get_summary(ticker="AAPL")
    assert summary.records == []
    assert summary.logged_days == 0
    assert summary.latest_directional_accuracy is None
