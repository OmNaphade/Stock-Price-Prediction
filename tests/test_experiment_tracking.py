from __future__ import annotations

import pytest

from monitoring.experiment_tracking import (
    MlflowExperimentTracker,
    NullExperimentTracker,
    build_experiment_tracker,
)

mlflow = pytest.importorskip("mlflow")


def test_null_tracker_is_a_no_op():
    tracker = NullExperimentTracker()
    tracker.log_backtest("AAPL", "Ridge", params={"a": 1}, metrics={"b": 2.0})  # must not raise


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
    assert runs[0].data.params["ticker"] == "AAPL"
    assert runs[0].data.metrics["directional_accuracy"] == 0.55


def test_build_experiment_tracker_respects_disabled_setting(monkeypatch):
    import config
    import monitoring.experiment_tracking as tracking_module

    monkeypatch.setattr(
        tracking_module, "settings", config.Settings(enable_experiment_tracking=False)
    )
    assert isinstance(build_experiment_tracker(), NullExperimentTracker)
