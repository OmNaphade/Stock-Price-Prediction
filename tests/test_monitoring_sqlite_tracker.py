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
    tracker.log_backtest("alice", "AAPL", "Ridge (linear)", {"n_folds": 5, "n_features": 10}, _metrics())

    records = tracker.get_recent("alice", ticker="AAPL")
    assert len(records) == 1
    assert records[0].model_directional_accuracy == 0.55
    assert records[0].has_drift is False


def test_log_backtest_same_day_upserts_not_duplicates(tmp_path):
    """Idempotency: seeded models (Ridge/GBM/LSTM all fix random_state) give
    the same backtest result on a same-day rerun, so re-logging that day
    should update the one row, not accumulate duplicates."""
    tracker = _tracker(tmp_path)
    tracker.log_backtest("alice", "AAPL", "Ridge (linear)", {"n_folds": 5}, _metrics(model_rmse_price=3.2))
    tracker.log_backtest("alice", "AAPL", "Ridge (linear)", {"n_folds": 5}, _metrics(model_rmse_price=3.9))

    records = tracker.get_recent("alice", ticker="AAPL")
    assert len(records) == 1
    assert records[0].model_rmse_price == 3.9  # the later value wins


def test_different_users_get_separate_rows(tmp_path):
    """Two users logging the same ticker/model on the same day must not
    collide into one shared row — this is a personal signal, not a shared
    dashboard."""
    tracker = _tracker(tmp_path)
    tracker.log_backtest("alice", "AAPL", "Ridge (linear)", {}, _metrics(model_rmse_price=3.2))
    tracker.log_backtest("bob", "AAPL", "Ridge (linear)", {}, _metrics(model_rmse_price=9.9))

    assert tracker.get_recent("alice", ticker="AAPL")[0].model_rmse_price == 3.2
    assert tracker.get_recent("bob", ticker="AAPL")[0].model_rmse_price == 9.9


def test_different_tickers_and_models_stay_separate_rows(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.log_backtest("alice", "AAPL", "Ridge (linear)", {}, _metrics())
    tracker.log_backtest("alice", "AAPL", "Gradient Boosting", {}, _metrics())
    tracker.log_backtest("alice", "MSFT", "Ridge (linear)", {}, _metrics())

    assert len(tracker.get_recent("alice", ticker="AAPL")) == 2
    assert len(tracker.get_recent("alice", model_name="Ridge (linear)")) == 2
    assert len(tracker.get_recent("alice")) == 3


def test_get_tickers_returns_distinct_sorted_for_that_user_only(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.log_backtest("alice", "MSFT", "Ridge (linear)", {}, _metrics())
    tracker.log_backtest("alice", "AAPL", "Ridge (linear)", {}, _metrics())
    tracker.log_backtest("alice", "AAPL", "Gradient Boosting", {}, _metrics())
    tracker.log_backtest("bob", "GOOGL", "Ridge (linear)", {}, _metrics())

    assert tracker.get_tickers("alice") == ["AAPL", "MSFT"]
    assert tracker.get_tickers("bob") == ["GOOGL"]


def test_drift_fields_survive_roundtrip(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.log_backtest(
        "alice", "AAPL", "Ridge (linear)", {}, _metrics(has_drift=True, drifted_feature_count=3)
    )
    record = tracker.get_recent("alice", ticker="AAPL")[0]
    assert record.has_drift is True
    assert record.drifted_feature_count == 3


class _RecordingTracker:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def log_backtest(self, username, ticker, model_name, params, metrics):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((username, ticker, model_name))


def test_composite_tracker_logs_to_all():
    a, b = _RecordingTracker(), _RecordingTracker()
    composite = CompositeExperimentTracker([a, b])
    composite.log_backtest("alice", "AAPL", "Ridge (linear)", {}, {})
    assert a.calls == [("alice", "AAPL", "Ridge (linear)")]
    assert b.calls == [("alice", "AAPL", "Ridge (linear)")]


def test_composite_tracker_one_failure_does_not_block_others():
    failing, working = _RecordingTracker(fail=True), _RecordingTracker()
    composite = CompositeExperimentTracker([failing, working])
    composite.log_backtest("alice", "AAPL", "Ridge (linear)", {}, {})  # must not raise
    assert working.calls == [("alice", "AAPL", "Ridge (linear)")]


def test_null_tracker_still_a_no_op():
    NullExperimentTracker().log_backtest("alice", "AAPL", "Ridge (linear)", {}, {})  # must not raise
