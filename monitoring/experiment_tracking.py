"""Every backtest run — which ticker, which model, what it scored — gets
logged so model quality over time is visible instead of disappearing the
moment the next click overwrites it. Behind a Protocol with a Null Object
default so PredictionService always has a tracker to call, whether or not
MLflow is installed or enabled (Open/Closed + Dependency Inversion, same
pattern as MarketDataSource and MacroFeatureSource elsewhere)."""

from __future__ import annotations

from typing import Protocol

from config import log, settings


class ExperimentTracker(Protocol):
    def log_backtest(self, ticker: str, model_name: str, params: dict, metrics: dict) -> None: ...


class NullExperimentTracker:
    def log_backtest(self, ticker: str, model_name: str, params: dict, metrics: dict) -> None:
        return None


class MlflowExperimentTracker:
    """Logs to a local SQLite store (`sqlite:///mlflow.db`) by default — no
    server needed for logging to work; run `mlflow ui --backend-store-uri
    sqlite:///mlflow.db` in the project directory to browse runs."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_backtest(self, ticker: str, model_name: str, params: dict, metrics: dict) -> None:
        with self._mlflow.start_run(run_name=f"{ticker}-{model_name}"):
            self._mlflow.log_param("ticker", ticker)
            self._mlflow.log_param("model", model_name)
            for key, value in params.items():
                self._mlflow.log_param(key, value)
            for key, value in metrics.items():
                self._mlflow.log_metric(key, value)


def build_experiment_tracker() -> ExperimentTracker:
    if not settings.enable_experiment_tracking:
        return NullExperimentTracker()
    try:
        return MlflowExperimentTracker(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)
    except ImportError as e:
        # Either mlflow itself isn't installed, or one of its transitive
        # imports failed (e.g. `cryptography`'s native extension blocked by
        # an OS-level policy) — either way, degrade gracefully.
        log.warning("MLflow experiment tracking unavailable (%s); disabling it.", e)
        return NullExperimentTracker()
