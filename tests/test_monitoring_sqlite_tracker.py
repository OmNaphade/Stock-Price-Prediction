from __future__ import annotations

from monitoring.experiment_tracking import CompositeExperimentTracker, NullExperimentTracker
from monitoring.sqlite_tracker import SqliteExperimentTracker


def _tracker(tmp_path) -> SqliteExperimentTracker:
    return SqliteExperimentTracker(str(tmp_path / "test_monitoring.db"))


def _metrics(**overrides) -> dict:
    defaults = dict(
        model_directional_accuracy=0.55,
        baseline_directional_accuracy=0.02,
        model_rmse_price=3.2,
        baseline_rmse_price=3.5,
        has_drift=False,
        drifted_feature_count=0,
    )
    defaults.update(overrides)
    return defaults


def test_log_backtest_then_get_recent_roundtrips(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.log_backtest("AAPL", "Ridge (linear)", {"n_folds": 5, "n_features": 10}, _metrics())

    records = tracker.get_recent(ticker="AAPL")
    assert len(records) == 1
    assert records[0].model_directional_accuracy == 0.55
    assert records[0].has_drift is False


def test_log_backtest_same_day_upserts_not_duplicates(tmp_path):
    """Idempotency: seeded models (Ridge/GBM/LSTM all fix random_state) give
    the same backtest result on a same-day rerun, so re-logging that day
    should update the one row, not accumulate duplicates."""
    tracker = _tracker(tmp_path)
    tracker.log_backtest("AAPL", "Ridge (linear)", {"n_folds": 5}, _metrics(model_rmse_price=3.2))
    tracker.log_backtest("AAPL", "Ridge (linear)", {"n_folds": 5}, _metrics(model_rmse_price=3.9))

    records = tracker.get_recent(ticker="AAPL")
    assert len(records) == 1
    assert records[0].model_rmse_price == 3.9  # the later value wins


def test_different_tickers_and_models_stay_separate_rows(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.log_backtest("AAPL", "Ridge (linear)", {}, _metrics())
    tracker.log_backtest("AAPL", "Gradient Boosting", {}, _metrics())
    tracker.log_backtest("MSFT", "Ridge (linear)", {}, _metrics())

    assert len(tracker.get_recent(ticker="AAPL")) == 2
    assert len(tracker.get_recent(model_name="Ridge (linear)")) == 2
    assert len(tracker.get_recent()) == 3


def test_get_tickers_returns_distinct_sorted(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.log_backtest("MSFT", "Ridge (linear)", {}, _metrics())
    tracker.log_backtest("AAPL", "Ridge (linear)", {}, _metrics())
    tracker.log_backtest("AAPL", "Gradient Boosting", {}, _metrics())

    assert tracker.get_tickers() == ["AAPL", "MSFT"]


def test_drift_fields_survive_roundtrip(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.log_backtest(
        "AAPL", "Ridge (linear)", {}, _metrics(has_drift=True, drifted_feature_count=3)
    )
    record = tracker.get_recent(ticker="AAPL")[0]
    assert record.has_drift is True
    assert record.drifted_feature_count == 3


class _RecordingTracker:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def log_backtest(self, ticker, model_name, params, metrics):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((ticker, model_name))


def test_composite_tracker_logs_to_all():
    a, b = _RecordingTracker(), _RecordingTracker()
    composite = CompositeExperimentTracker([a, b])
    composite.log_backtest("AAPL", "Ridge (linear)", {}, {})
    assert a.calls == [("AAPL", "Ridge (linear)")]
    assert b.calls == [("AAPL", "Ridge (linear)")]


def test_composite_tracker_one_failure_does_not_block_others():
    failing, working = _RecordingTracker(fail=True), _RecordingTracker()
    composite = CompositeExperimentTracker([failing, working])
    composite.log_backtest("AAPL", "Ridge (linear)", {}, {})  # must not raise
    assert working.calls == [("AAPL", "Ridge (linear)")]


def test_null_tracker_still_a_no_op():
    NullExperimentTracker().log_backtest("AAPL", "Ridge (linear)", {}, {})  # must not raise
