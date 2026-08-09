"""Single source of truth for settings. Nothing else in the app should read
os.environ or st.secrets directly — import Settings from here instead."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Loads .env into the real process environment if one exists in the
# working directory (see .env.example for every variable this app reads)
# — a no-op, not an error, when there isn't one, so this is safe in every
# environment: local dev with a .env, Docker/Streamlit Cloud with real
# env vars already set, or `st.secrets` (checked first, below) on
# Streamlit Community Cloud. Real environment variables always win over
# .env — this only fills in what isn't already set.
load_dotenv()


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

    # OpenAlgo — a self-hosted gateway that normalizes many Indian brokers
    # (Angel One, Zerodha, Upstox, Fyers, ...) behind one REST API keyed by
    # a single static API key (no TOTP/session handling on our side at
    # all — OpenAlgo owns that against whichever broker it's linked to).
    # Both unset by default; covers only NSE/BSE-style exchanges
    # (`.NS`/`.BO` tickers), same as AlphaVantageSource does when
    # AV_API_KEY is unset. openalgo_base_url is the root of your own
    # OpenAlgo instance (self-hosted locally, or tunneled — e.g. ngrok),
    # not a shared public endpoint.
    openalgo_base_url: str = field(default_factory=lambda: _get_secret("OPENALGO_BASE_URL"))
    openalgo_api_key: str = field(default_factory=lambda: _get_secret("OPENALGO_API_KEY"))
    history_start: str = "2016-01-01"
    # No history_end setting on purpose: "today" is a call-time fact, not a
    # boot-time one. A frozen Settings singleton (or a function default
    # evaluated once at import) would capture whatever date the process
    # happened to start on and silently go stale over the life of a
    # long-running server — see PredictionService.analyze()'s own default.

    # Auth / storage — identity is an email address (see auth/service.py).
    db_path: str = "users.db"
    max_login_attempts: int = 5
    lockout_minutes: int = 15

    # Admin account, auto-provisioned on first run if both are set (see
    # auth/bootstrap.py) — gates access to the Monitoring page (see
    # auth_ui.require_admin_user). Both optional; Monitoring is simply
    # unavailable to everyone until they're configured. Bootstrap only
    # ever *creates* the account if it doesn't exist yet — it never
    # overwrites, so changing the password later via the normal
    # forgot-password flow survives a redeploy instead of being silently
    # reset back to whatever these env vars still say.
    admin_email: str = field(default_factory=lambda: _get_secret("ADMIN_EMAIL"))
    admin_password: str = field(default_factory=lambda: _get_secret("ADMIN_PASSWORD"))

    # Email delivery for OTP codes (registration verification + password
    # reset). Unset by default: EmailSender falls back to logging the code
    # server-side instead of emailing it, so local dev/tests never need a
    # real SMTP account — see auth/email_sender.py.
    smtp_host: str = field(default_factory=lambda: _get_secret("SMTP_HOST"))
    smtp_port: int = field(default_factory=lambda: int(_get_secret("SMTP_PORT", "587") or "587"))
    smtp_username: str = field(default_factory=lambda: _get_secret("SMTP_USERNAME"))
    smtp_password: str = field(default_factory=lambda: _get_secret("SMTP_PASSWORD"))
    smtp_from_address: str = field(
        default_factory=lambda: _get_secret("SMTP_FROM_ADDRESS") or _get_secret("SMTP_USERNAME")
    )
    smtp_use_tls: bool = field(default_factory=lambda: _get_secret("SMTP_USE_TLS", "true").lower() != "false")

    otp_code_length: int = 6
    otp_expiry_minutes: int = 10
    otp_max_attempts: int = 5
    # Minimum time between two OTP emails for the same (email, purpose) —
    # caps how fast "forgot password"/"resend code" can be hit for a
    # single account. Doesn't limit how many *distinct* accounts can be
    # registered (that needs IP-level/CAPTCHA throttling this app has no
    # request context for) — see auth/service.py's docstring.
    otp_resend_cooldown_seconds: int = 60

    # Password policy: length only (no forced complexity rules — composition
    # requirements are out of favor per NIST 800-63B; length is what
    # actually matters). Max is bcrypt's real limit: bcrypt silently
    # truncates at 72 *bytes* rather than erroring, so two different
    # 100-character passwords sharing the same first 72 bytes would
    # otherwise be treated as the same credential without either password's
    # owner ever being told.
    min_password_length: int = 8
    max_password_length_bytes: int = 72

    max_email_length: int = 254  # RFC 5321 practical limit
    max_ticker_length: int = 20  # real tickers + exchange suffix never approach this

    # Prediction track record — its own SQLite file, separate from users.db
    track_record_db_path: str = "track_record.db"

    # Modeling
    min_training_rows: int = 60
    walk_forward_folds: int = 5
    test_fold_size: int = 30
    autoreg_max_lags: int = 60
    autoreg_forecast_days: int = 30

    # Model monitoring — always-on, SQLite-backed, no extra dependency;
    # one row per (ticker, model, day). See monitoring/sqlite_tracker.py.
    monitoring_db_path: str = "monitoring.db"

    # Experiment tracking (MLflow) — optional and additive to the above.
    # Local SQLite file by default (the plain filesystem store is
    # deprecated as of MLflow 2.16+), no server needed for logging to
    # work; `mlflow ui --backend-store-uri sqlite:///mlflow.db` reads the
    # same file.
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
