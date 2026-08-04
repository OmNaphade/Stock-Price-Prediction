"""Single source of truth for settings. Nothing else in the app should read
os.environ or st.secrets directly — import Settings from here instead."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field


def _get_secret(key: str, default: str = "") -> str:
    """Read a value from Streamlit secrets if available, else the environment."""
    try:
        import streamlit as st

        value = st.secrets.get(key)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    # Data sources
    av_api_key: str = field(default_factory=lambda: _get_secret("AV_API_KEY"))
    fred_api_key: str = field(default_factory=lambda: _get_secret("FRED_API_KEY"))
    news_api_key: str = field(default_factory=lambda: _get_secret("NEWS_API_KEY"))
    history_start: str = "2016-01-01"
    # No history_end setting on purpose: "today" is a call-time fact, not a
    # boot-time one. A frozen Settings singleton (or a function default
    # evaluated once at import) would capture whatever date the process
    # happened to start on and silently go stale over the life of a
    # long-running server — see PredictionService.analyze()'s own default.

    # Auth / storage
    db_path: str = "users.db"
    max_login_attempts: int = 5
    lockout_minutes: int = 15

    # Modeling
    min_training_rows: int = 60
    walk_forward_folds: int = 5
    test_fold_size: int = 30
    autoreg_max_lags: int = 60
    autoreg_forecast_days: int = 30

    # Experiment tracking (MLflow) — a local SQLite file by default (the
    # plain filesystem store is deprecated as of MLflow 2.16+), no server
    # needed for logging to work; `mlflow ui --backend-store-uri
    # sqlite:///mlflow.db` reads the same file.
    enable_experiment_tracking: bool = True
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow_experiment_name: str = "stock-prediction"

    # Logging
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))


settings = Settings()


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("stock_prediction")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
        logger.propagate = False
    return logger


log = configure_logging()
