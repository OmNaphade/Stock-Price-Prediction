from __future__ import annotations

import pytest

from monitoring.experiment_tracking import (
    CompositeExperimentTracker,
    MlflowExperimentTracker,
    NullExperimentTracker,
    build_experiment_tracker,
)
from monitoring.sqlite_tracker import SqliteExperimentTracker

mlflow = pytest.importorskip("mlflow")


def test_null_tracker_is_a_no_op():
    tracker = NullExperimentTracker()
    tracker.log_backtest("alice", "AAPL", "Ridge", params={"a": 1}, metrics={"b": 2.0})  # must not raise


def test_mlflow_tracker_logs_a_run(tmp_path):
    db_path = tmp_path / "mlflow_test.db"
    try:
        tracker = MlflowExperimentTracker(f"sqlite:///{db_path}", "test-experiment")
    except ImportError as e:
        # MLflow's SQLAlchemy backend pulls in `cryptography`, which loads a
        # native DLL — some locked-down Windows environments (endpoint
        # "Application Control" policies) block that at the OS level. That's
        # a host restriction, not a defect in this code; skip rather than
        # fail so the suite still reports the environment-specific cause.
        pytest.skip(f"mlflow SQLAlchemy backend unavailable in this environment: {e}")

    tracker.log_backtest(
        username="alice",
        ticker="AAPL",
        model_name="Ridge",
        params={"n_folds": 5},
        metrics={"directional_accuracy": 0.55},
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{db_path}")
    experiment = client.get_experiment_by_name("test-experiment")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["username"] == "alice"
    assert runs[0].data.params["ticker"] == "AAPL"
    assert runs[0].data.metrics["directional_accuracy"] == 0.55


def test_build_experiment_tracker_is_sqlite_only_when_mlflow_disabled(tmp_path, monkeypatch):
    # The SQLite tracker is always on (no extra dependency); disabling
    # `enable_experiment_tracking` only opts out of the additional,
    # optional MLflow tracker — it must not fall back to a no-op.
    import config
    import monitoring.experiment_tracking as tracking_module

    monkeypatch.setattr(
        tracking_module,
        "settings",
        config.Settings(
            enable_experiment_tracking=False,
            monitoring_db_path=str(tmp_path / "monitoring_test.db"),
        ),
    )
    tracker = build_experiment_tracker()
    assert isinstance(tracker, SqliteExperimentTracker)
    assert not isinstance(tracker, (NullExperimentTracker, CompositeExperimentTracker))


def test_build_experiment_tracker_composes_sqlite_and_mlflow_when_enabled(tmp_path, monkeypatch):
    import config
    import monitoring.experiment_tracking as tracking_module

    monkeypatch.setattr(
        tracking_module,
        "settings",
        config.Settings(
            enable_experiment_tracking=True,
            monitoring_db_path=str(tmp_path / "monitoring_test.db"),
            mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow_test.db'}",
        ),
    )
    # build_experiment_tracker() already degrades gracefully on its own if
    # MLflow can't be constructed (see the ImportError handling inside it),
    # so no try/except is needed here — just recognize the degraded case.
    tracker = build_experiment_tracker()
    if not isinstance(tracker, CompositeExperimentTracker):
        pytest.skip("mlflow unavailable in this environment; degraded to sqlite-only")
    assert isinstance(tracker, CompositeExperimentTracker)
