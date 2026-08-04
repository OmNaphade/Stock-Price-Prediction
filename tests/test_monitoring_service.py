from __future__ import annotations

from types import SimpleNamespace

from monitoring.models import ModelMetricRecord
from monitoring.sqlite_tracker import SqliteExperimentTracker
from services.monitoring_service import ModelMonitoringService


def _tracker(tmp_path) -> SqliteExperimentTracker:
    return SqliteExperimentTracker(str(tmp_path / "test_monitoring_service.db"))


def _log(tracker, username, ticker, model_name, **metric_overrides):
    metrics = dict(
        model_directional_accuracy=0.5,
        baseline_directional_accuracy=0.02,
        model_rmse_price=3.0,
        baseline_rmse_price=3.5,
        has_drift=False,
        drifted_feature_count=0,
    )
    metrics.update(metric_overrides)
    tracker.log_backtest(username, ticker, model_name, {}, metrics)


def test_get_known_tickers_reflects_logged_data_for_that_user(tmp_path):
    tracker = _tracker(tmp_path)
    _log(tracker, "alice", "AAPL", "Ridge (linear)")
    _log(tracker, "alice", "MSFT", "Ridge (linear)")
    _log(tracker, "bob", "GOOGL", "Ridge (linear)")

    service = ModelMonitoringService(tracker, tracker)
    assert service.get_known_tickers("alice") == ["AAPL", "MSFT"]
    assert service.get_known_tickers("bob") == ["GOOGL"]


def test_get_summary_computes_drift_days_and_latest_accuracy(tmp_path):
    tracker = _tracker(tmp_path)
    _log(tracker, "alice", "AAPL", "Ridge (linear)", has_drift=True, model_directional_accuracy=0.6)

    service = ModelMonitoringService(tracker, tracker)
    summary = service.get_summary("alice", ticker="AAPL")
    assert summary.logged_days == 1
    assert summary.drift_days == 1
    assert summary.latest_directional_accuracy == 0.6


class _FakeReader:
    """A ModelMetricsReader stand-in that deliberately returns records out
    of order, so the test proves the service does the sorting — not that
    whatever storage happens to already return them sorted."""

    def __init__(self, records):
        self._records = records

    def get_recent(self, username, ticker=None, model_name=None, limit=500):
        return self._records

    def get_tickers(self, username):
        return sorted({r.ticker for r in self._records})


def _fake_record(log_date: str) -> ModelMetricRecord:
    return ModelMetricRecord(
        username="alice", ticker="AAPL", model_name="Ridge (linear)", log_date=log_date,
        logged_at=log_date, n_folds=5, n_features=10, model_directional_accuracy=0.5,
        baseline_directional_accuracy=0.02, model_rmse_price=3.0, baseline_rmse_price=3.5,
        has_drift=False, drifted_feature_count=0,
    )


def test_get_summary_records_are_chronological_oldest_first():
    """The page charts these directly — they must already be in the right
    order, not something the page has to sort itself."""
    out_of_order = [_fake_record("2026-08-03"), _fake_record("2026-08-01"), _fake_record("2026-08-02")]
    reader = _FakeReader(out_of_order)
    service = ModelMonitoringService(reader, reader)

    summary = service.get_summary("alice", ticker="AAPL")
    dates = [r.log_date for r in summary.records]
    assert dates == ["2026-08-01", "2026-08-02", "2026-08-03"]


def test_get_summary_empty_when_nothing_logged(tmp_path):
    tracker = _tracker(tmp_path)
    service = ModelMonitoringService(tracker, tracker)
    summary = service.get_summary("alice", ticker="AAPL")
    assert summary.records == []
    assert summary.logged_days == 0
    assert summary.latest_directional_accuracy is None


class _RecordingTracker:
    def __init__(self):
        self.calls = []

    def log_backtest(self, username, ticker, model_name, params, metrics):
        self.calls.append((username, ticker, model_name, params, metrics))


def _fake_report(drift_report=None):
    backtest = SimpleNamespace(
        folds=[object(), object()],
        mean_directional_accuracy=0.55,
        mean_rmse_price=3.1,
    )
    baseline = SimpleNamespace(mean_directional_accuracy=0.02, mean_rmse_price=3.6)
    return SimpleNamespace(
        ticker="AAPL", model_name="Ridge (linear)", feature_columns=["a", "b", "c"],
        model_backtest=backtest, baseline_backtest=baseline, drift_report=drift_report,
    )


def test_log_from_report_extracts_username_and_metrics_correctly():
    tracker = _RecordingTracker()
    service = ModelMonitoringService(tracker, tracker)

    service.log_from_report("alice", _fake_report())

    assert len(tracker.calls) == 1
    username, ticker, model_name, params, metrics = tracker.calls[0]
    assert username == "alice"
    assert ticker == "AAPL"
    assert model_name == "Ridge (linear)"
    assert params == {"n_folds": 2, "n_features": 3}
    assert metrics["model_directional_accuracy"] == 0.55
    assert metrics["baseline_rmse_price"] == 3.6


def test_log_from_report_handles_missing_drift_report_gracefully():
    tracker = _RecordingTracker()
    service = ModelMonitoringService(tracker, tracker)

    service.log_from_report("alice", _fake_report(drift_report=None))  # must not raise

    _, _, _, _, metrics = tracker.calls[0]
    assert metrics["has_drift"] is None
    assert metrics["drifted_feature_count"] is None


def test_log_from_report_swallows_tracker_failures():
    class _BoomTracker:
        def log_backtest(self, *a, **k):
            raise RuntimeError("boom")

    service = ModelMonitoringService(_BoomTracker(), _BoomTracker())
    service.log_from_report("alice", _fake_report())  # must not raise
