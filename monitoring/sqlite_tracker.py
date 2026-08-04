"""Always-on, dependency-free model monitoring: one row per
(ticker, model, day), upserted — re-running the same ticker/model more
than once in a day updates that day's row instead of accumulating
duplicates. That's a deliberate idempotency choice, not just tidiness:
Ridge/GradientBoosting/LSTM are all seeded (`random_state=42` /
`torch.manual_seed(42)`), so a same-day rerun on the same data produces
the same backtest result anyway — logging it again as a new row would
just be noise, and would make "how many times was this run" look like
"how much history exists," which isn't what the monitoring page is for.

This is the lightweight complement to the optional, heavier MLflow
tracker in `experiment_tracking.py` — no server, no extra dependency,
always available, and browsable from the app's own Monitoring page
instead of a separate `mlflow ui` process.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime, timezone
from typing import Optional, Protocol

from .models import ModelMetricRecord

_COLUMNS = (
    "ticker", "model_name", "log_date", "logged_at", "n_folds", "n_features",
    "model_directional_accuracy", "baseline_directional_accuracy",
    "model_rmse_price", "baseline_rmse_price", "has_drift", "drifted_feature_count",
)


class ModelMetricsReader(Protocol):
    """The read side, deliberately separate from ExperimentTracker's write
    side (`log_backtest`) — PredictionService only ever writes; only the
    monitoring page reads, and it shouldn't have to depend on a Protocol
    shaped for the writer's job (Interface Segregation)."""

    def get_recent(
        self, ticker: Optional[str] = None, model_name: Optional[str] = None, limit: int = 500
    ) -> list[ModelMetricRecord]: ...

    def get_tickers(self) -> list[str]: ...


def _row_to_record(row: tuple) -> ModelMetricRecord:
    data = dict(zip(_COLUMNS, row))
    data["has_drift"] = bool(data["has_drift"]) if data["has_drift"] is not None else None
    return ModelMetricRecord(**data)


class SqliteExperimentTracker:
    def __init__(self, db_path: str = "monitoring.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_metrics (
                    ticker                        TEXT NOT NULL,
                    model_name                    TEXT NOT NULL,
                    log_date                      TEXT NOT NULL,
                    logged_at                     TEXT NOT NULL,
                    n_folds                       INTEGER,
                    n_features                    INTEGER,
                    model_directional_accuracy    REAL,
                    baseline_directional_accuracy REAL,
                    model_rmse_price              REAL,
                    baseline_rmse_price           REAL,
                    has_drift                     INTEGER,
                    drifted_feature_count         INTEGER,
                    PRIMARY KEY (ticker, model_name, log_date)
                )
                """
            )
            self._conn.commit()

    def log_backtest(self, ticker: str, model_name: str, params: dict, metrics: dict) -> None:
        log_date = date.today().isoformat()
        logged_at = datetime.now(timezone.utc).isoformat()
        has_drift = metrics.get("has_drift")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO model_metrics
                    (ticker, model_name, log_date, logged_at, n_folds, n_features,
                     model_directional_accuracy, baseline_directional_accuracy,
                     model_rmse_price, baseline_rmse_price, has_drift, drifted_feature_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, model_name, log_date) DO UPDATE SET
                    logged_at=excluded.logged_at,
                    n_folds=excluded.n_folds,
                    n_features=excluded.n_features,
                    model_directional_accuracy=excluded.model_directional_accuracy,
                    baseline_directional_accuracy=excluded.baseline_directional_accuracy,
                    model_rmse_price=excluded.model_rmse_price,
                    baseline_rmse_price=excluded.baseline_rmse_price,
                    has_drift=excluded.has_drift,
                    drifted_feature_count=excluded.drifted_feature_count
                """,
                (
                    ticker, model_name, log_date, logged_at,
                    params.get("n_folds"), params.get("n_features"),
                    metrics.get("model_directional_accuracy"),
                    metrics.get("baseline_directional_accuracy"),
                    metrics.get("model_rmse_price"),
                    metrics.get("baseline_rmse_price"),
                    int(has_drift) if has_drift is not None else None,
                    metrics.get("drifted_feature_count"),
                ),
            )
            self._conn.commit()

    def get_recent(
        self, ticker: Optional[str] = None, model_name: Optional[str] = None, limit: int = 500
    ) -> list[ModelMetricRecord]:
        clauses, params_list = [], []
        if ticker:
            clauses.append("ticker = ?")
            params_list.append(ticker)
        if model_name:
            clauses.append("model_name = ?")
            params_list.append(model_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM model_metrics "
                f"{where} ORDER BY log_date DESC LIMIT ?",
                (*params_list, limit),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def get_tickers(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT ticker FROM model_metrics ORDER BY ticker"
            ).fetchall()
        return [row[0] for row in rows]
