"""Every backtest run — which user, which ticker, which model, what it
scored — gets logged so model quality over time is visible instead of
disappearing the moment the next click overwrites it. Behind a Protocol so
callers never depend on a concrete tracker (Dependency Inversion, same
pattern as MarketDataSource and MacroFeatureSource elsewhere).

Two trackers compose here, not one:
- `SqliteExperimentTracker` (sqlite_tracker.py) — always on, no extra
  dependency, upserts one row per (user, ticker, model, day), browsable
  from the app's own Monitoring page.
- `MlflowExperimentTracker` — optional, heavier, append-only by MLflow's
  own design (every run is its own permanent record, not upserted), for
  anyone who wants the fuller MLflow UI/ecosystem.

`build_experiment_tracker()` always includes the former and adds the
latter only when available and enabled.
"""

from __future__ import annotations

from typing import Protocol

from config import log, settings

from .sqlite_tracker import SqliteExperimentTracker


class ExperimentTracker(Protocol):
    def log_backtest(
        self, username: str, ticker: str, model_name: str, params: dict, metrics: dict
    ) -> None: ...


class NullExperimentTracker:
    def log_backtest(
        self, username: str, ticker: str, model_name: str, params: dict, metrics: dict
    ) -> None:
        return None


class CompositeExperimentTracker:
    """Logs to every configured tracker; one tracker failing (e.g. MLflow
    hitting an environment-specific import error) never blocks another."""

    def __init__(self, trackers: list[ExperimentTracker]):
        self._trackers = trackers

    def log_backtest(
        self, username: str, ticker: str, model_name: str, params: dict, metrics: dict
    ) -> None:
        for tracker in self._trackers:
            try:
                tracker.log_backtest(username, ticker, model_name, params, metrics)
            except Exception:
                log.warning(
                    "%s failed to log %s/%s/%s", type(tracker).__name__, username, ticker, model_name,
                    exc_info=True,
                )


class MlflowExperimentTracker:
    """Logs to a local SQLite store (`sqlite:///mlflow.db`) by default — no
    server needed for logging to work; run `mlflow ui --backend-store-uri
    sqlite:///mlflow.db` in the project directory to browse runs. Every
    call creates a new run — that's MLflow's own append-only data model,
    not something this wrapper controls."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_backtest(
        self, username: str, ticker: str, model_name: str, params: dict, metrics: dict
    ) -> None:
        with self._mlflow.start_run(run_name=f"{username}-{ticker}-{model_name}"):
            self._mlflow.log_param("username", username)
            self._mlflow.log_param("ticker", ticker)
            self._mlflow.log_param("model", model_name)
            for key, value in params.items():
                self._mlflow.log_param(key, value)
            for key, value in metrics.items():
                self._mlflow.log_metric(key, value)


def build_experiment_tracker() -> ExperimentTracker:
    trackers: list[ExperimentTracker] = [SqliteExperimentTracker(settings.monitoring_db_path)]

    if settings.enable_experiment_tracking:
        try:
            trackers.append(
                MlflowExperimentTracker(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)
            )
        except ImportError as e:
            # Either mlflow itself isn't installed, or one of its transitive
            # imports failed (e.g. `cryptography`'s native extension blocked
            # by an OS-level policy) — either way, the always-on SQLite
            # tracker above still works, so this only disables the optional
            # extra, not monitoring as a whole.
            log.warning("MLflow experiment tracking unavailable (%s); continuing without it.", e)

    return trackers[0] if len(trackers) == 1 else CompositeExperimentTracker(trackers)
