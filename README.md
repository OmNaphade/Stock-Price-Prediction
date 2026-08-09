# Algorithmic Stock Price Prediction 📈
> Streamlit-based stock analysis and forecasting app with authenticated access, live market data fetches, moving-average charts, and lightweight prediction models.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under [MIT](LICENSE): use, modify, and redistribute freely.

> ⚠️ **Data note:** The app relies on public market data sources. Yahoo Finance can occasionally throttle requests, so the app also supports an optional Alpha Vantage fallback for more reliable cloud runs.

> 🚧 **Work in progress:** This project is still evolving. Data sources, models, and UI details may change as the app matures.

---

## What It Does

This app helps you explore stock history and generate next-price forecasts from the browser, with every forecast benchmarked against a naive baseline so you can tell whether it's actually predictive.

- **Email-based authenticated access** — register/login with an email + password, email ownership verified via a one-time code before the account can log in, lockout after repeated failed login attempts, and a secure OTP-based password reset (see [Authentication](#authentication))
- **Ticker-based lookup** for equities such as `AAPL`, `TSLA`, `RELIANCE.NS`, and `TCS.NS`
- **Historical analysis** with summaries, price charts, and moving averages
- **Next-close prediction** with a choice of models (naive baseline, regularized linear, gradient boosting, and an optional LSTM), each evaluated with walk-forward backtesting and directional accuracy — not just a single train/test split
- **Prediction intervals** derived from real out-of-sample backtest error, not a guessed margin
- **Feature drift detection** flags when a ticker's recent behavior statistically diverges from the window the backtest was computed on
- **News sentiment panel** (free, no key required) as descriptive context — deliberately not fed into the model; see [Additional Features](#additional-features) for why
- **Optional macro features** (interest rates, CPI via FRED)
- **Model monitoring** — every backtest run is logged (always on, no extra dependency), so directional accuracy, RMSE, and drift status are all visible as trends over time, not just in the moment; MLflow is available as an optional, heavier addition on top. The in-app Monitoring page is **admin-only** and shows system-wide health across every user (see [Authentication](#authentication))
- **Track record** — every prediction is recorded before its outcome is known, then checked against the real close once its target date passes, building an honest, unfalsifiable accuracy history over time
- **Multi-user by design** — the track record and monitoring history are both scoped per account: what you've analyzed and how it's turned out is yours, not merged with anyone else logged into the same app
- **Watchlist page** comparing predictions across several tickers at once
- **Alternative exploration page** under `pages/prediction.py` with candlestick charts and AutoReg forecasting
- **Live market context** via `yfinance`, with optional Alpha Vantage support when an API key is available, and optional OpenAlgo support for exchange-native NSE/BSE data via your own self-hosted OpenAlgo instance — including live market depth (top bid/ask levels), a market-open/closed status indicator with the next trading holiday, and live NSE/BSE symbol search, when configured
- **Multi-language UI** — switch between English, 中文, 한국어, 日本語, Türkçe, and Русский from the sidebar on any page
- **Production-hardened SQLite** — WAL mode, busy-timeout, and integrity checks by default, with optional continuous Litestream replication so data survives a redeploy
- **Docker-ready deployment** for consistent local and cloud execution

### Architecture

The app is organized in layers so the UI never touches a data provider, the database, or a model class directly:

```
config.py            single source of settings (env vars / secrets)
data_access/          MarketDataSource, MacroFeatureSource, NewsSource interfaces
                       — yfinance, Alpha Vantage, OpenAlgo, FRED, NewsAPI, fallback chains
infra/                 shared SQLite hardening (WAL, busy-timeout, backup, integrity check)
i18n/                   translator + language switcher; JSON catalogs per language
auth/                 UserRepository + OtpRepository + EmailSender + AuthService
                       (email identity, verification, lockout, OTP password reset)
                       + password_hashing (shared bcrypt helper) + bootstrap
                       (auto-provisions the admin account from ADMIN_EMAIL/PASSWORD)
features/             FeaturePipeline: log-return target, technical indicators,
                       optional macro features; sentiment scoring (VADER)
models/               Predictor interface: naive baseline, Ridge, gradient
                       boosting, optional LSTM, AutoReg
evaluation/           walk-forward backtesting, prediction intervals, drift detection
monitoring/            ExperimentTracker interface: always-on SQLite tracker,
                       optional MLflow, composed together
track_record/           PredictionRecord + its own SQLite-backed repository
services/             PredictionService / SentimentService / TrackRecordService /
                       ModelMonitoringService — each answers one question
web_context.py        shared, cached service instances for all pages
app.py, pages/         thin Streamlit views; no business logic lives here
tests/                 pytest suite covering every layer above
```

Adding a data provider or a model means adding one class that satisfies the relevant interface — no existing file needs to change.

---

## Design Principles

This isn't an abstract goal bolted on afterward — every layer above exists because of a specific principle, and it's worth being concrete about which one and where, so the architecture can be checked against reality rather than taken on faith.

**Separation of concerns.** Five questions, five layers, none of which know how to answer each other's question: `data_access/` gets raw prices, `features/` turns prices into model inputs, `models/` turns inputs into a return prediction, `evaluation/` decides whether that prediction is any good, `track_record/` + `monitoring/` decide whether it's *still* any good over time. A page never answers any of these itself — it calls a `services/` object and renders what comes back.

**Single Responsibility.** `AuthService` (hashing, lockout policy) and `SqliteUserRepository` (storage) are two classes, not one, so a password-policy change and a schema change are unrelated edits. The same split repeats deliberately: `TrackRecordService` answers "was this specific prediction right" while `ModelMonitoringService` answers "how has this model's backtest performance looked over time" — two different questions about the same predictions, kept in two different services rather than one growing class that answers both badly. (`pages/monitoring.py` used to compute its own summary stats inline, the one place this split had been skipped — moved into `ModelMonitoringService` so every page follows the same rule: services compute what something *means*, pages only render it.) `auth/password_hashing.py` is the same idea one level down: bcrypt hashing used to be two private functions inside `AuthService`; `auth/bootstrap.py` needed the exact same operation to seed the admin account and had no business reaching into another class's internals to get it, so it's now its own two-function module both depend on.

**Open/Closed.** `AVAILABLE_MODELS` is a dict from name to `Predictor` factory (`services/prediction_service.py`) — the LSTM entry is *added* conditionally when PyTorch is importable, nothing else in the file changes to support it. `CompositeMarketDataSource` and `CompositeExperimentTracker` both take a list of providers and try/log each in turn — a new data source or a new tracker is an item appended to a list at the composition root (`web_context.py`), never a new `if` branch in existing logic. `FeaturePipeline(macro_source=...)` defaults to `NullMacroSource()` — turning macro features on is a constructor argument, not a code path threaded through the pipeline.

**Liskov Substitution.** `NaivePredictor`, `RidgeReturnPredictor`, `GradientBoostingReturnPredictor`, and `LSTMReturnPredictor` are fully interchangeable behind `fit(X, y)` / `predict(X)` — `walk_forward_backtest()` never branches on which one it was handed, and `test_all_predictors_share_the_same_interface` in `tests/test_models.py` pins exactly that.

**Interface Segregation.** `ExperimentTracker` (write: `log_backtest`) and `ModelMetricsReader` (read: `get_recent`, `get_tickers`) are two separate Protocols in `monitoring/`, even though one concrete class (`SqliteExperimentTracker`) satisfies both — `PredictionService` only ever needs to write, the Monitoring page only ever needs to read, and neither should have to depend on the half it doesn't use. `MarketDataSource` stays deliberately narrow (history + quote) rather than bundling in the 30-field fundamentals blob (`data_access/fundamentals.py`), Yahoo-specific period/interval history (`YFinanceSource.get_history_by_period`), or OpenAlgo-specific depth/market-hours data (`OpenAlgoSource.get_depth`, `OpenAlgoMarketCalendar`) that most callers never need — those live as extra capabilities on the concrete classes, not on the shared Protocol every provider has to implement.

**Dependency Inversion.** `PredictionService.__init__(data_source: MarketDataSource, ...)` depends on an interface, never on `yfinance` or `sqlite3` directly — the concrete wiring happens exactly once, in `web_context.py`, the app's single composition root. No page imports `yfinance`, `torch`, or `sqlite3` itself; every page only ever imports from `services` and `web_context`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Localization | Custom `i18n/` translator (JSON catalogs) — English, 中文, 한국어, 日本語, Türkçe, Русский |
| Data | yfinance, Alpha Vantage (optional), OpenAlgo (optional, self-hosted, NSE/BSE only) |
| Analysis | NumPy, Pandas, Matplotlib, Plotly |
| Forecasting | scikit-learn (Ridge, GradientBoosting), statsmodels AutoReg, optional PyTorch LSTM |
| Backtesting | Walk-forward validation (`sklearn.model_selection.TimeSeriesSplit`) |
| Macro data | FRED (`fredapi`), optional |
| Sentiment | VADER (`vaderSentiment`) over yfinance/NewsAPI headlines |
| Model monitoring | SQLite (always on) + MLflow (optional, additive) |
| Testing | pytest |
| Authentication | Email + password (SQLite, WAL, hardened) + bcrypt, with email-verified registration and OTP-based password reset |
| Authorization | Single admin account, auto-provisioned from `ADMIN_EMAIL`/`ADMIN_PASSWORD` — gates the Monitoring page only |
| Email delivery | `smtplib` (stdlib) — any SMTP provider; logs the code server-side instead when unconfigured |
| License | MIT |
| Durability | Litestream (optional, continuous replication to S3-compatible storage) |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## How It Works

```
User enters a ticker + picks a model
      │
      ├── Auth check (AuthService → SqliteUserRepository / SqliteOtpRepository / EmailSender)
      │
      ├── PredictionService.analyze()
      │     ├── MarketDataSource: Alpha Vantage if AV_API_KEY is set,
      │     │   else yfinance (curl_cffi / requests fallback)
      │     ├── FeaturePipeline: log-return target + technical indicators
      │     ├── walk-forward backtest of the chosen model AND the naive
      │     │   baseline, so results are always shown side by side
      │     └── final fit on all data → next-close prediction
      │
      └── app.py renders charts, backtest metrics, and the forecast
```

The main app in [app.py](app.py) predicts the next close via `services.PredictionService`, with a model dropdown (naive baseline / Ridge / gradient boosting) and every result benchmarked against the baseline. The exploration page in [pages/prediction.py](pages/prediction.py) focuses on stock selection, candlesticks, and multi-day AutoReg forecasting.

---

## Features

| Feature | Details |
|---|---|
| Login / Register | Email + password, backed by SQLite and bcrypt; registration requires verifying the email via an emailed code before the account can log in, and lockout after repeated failed logins |
| Password reset | Request a code emailed to the account → enter the code + a new password. Replaces an earlier flow that reset a password with no identity check at all — see [Authentication](#authentication) |
| Historical charts | Price history plus 100-day and 200-day moving averages |
| Model selection | Naive baseline, Ridge (regularized linear), gradient boosting, optional LSTM — same interface, swap freely |
| Backtested predictions | Walk-forward validation across multiple folds, with directional accuracy and price RMSE shown against the naive baseline |
| Prediction intervals | A confidence band built from real backtest error quantiles, not a fixed guessed margin |
| Drift detection | KS-test comparison of recent vs. reference feature windows, surfaced as a warning when they diverge |
| News sentiment | Free (yfinance) or NewsAPI headlines scored with VADER, shown as live context — not a training feature |
| Macro features (optional) | 10-year Treasury yield and CPI from FRED, added to the feature set only when `FRED_API_KEY` is set |
| Model monitoring | Every backtest run logged (always on) — directional accuracy, RMSE, and drift status charted over time per ticker/model, **admin-only, system-wide across every user** (not scoped to one account); MLflow logs the same runs too when installed |
| Held-out fold chart | Actual vs. predicted price on the most recent backtest fold, not a cherry-picked window |
| Watchlist | Compare predictions across several tickers at once, ranked by predicted % change |
| Track record | Every prediction you make recorded pre-outcome, resolved against the real close once due, with a predicted-vs-actual chart and full history table — your own, not shared |
| Multi-page Streamlit UI | Exploration page (`pages/prediction.py`), watchlist, track record, and monitoring, alongside the main app |
| Cloud-friendly fetches | curl_cffi session fallback for yfinance reliability |
| OpenAlgo data source (optional) | Your own self-hosted OpenAlgo instance as an exchange-native source for NSE/BSE (`.NS`/`.BO`) tickers, ahead of Alpha Vantage/yfinance in the provider chain when configured |
| Market depth (optional) | Top bid/ask levels for Indian tickers, shown alongside the live quote — descriptive context only, same as news sentiment, never fed into a model. Requires OpenAlgo |
| Market status (optional) | "NSE is open, closes at 15:30 IST" / "closed, opens at 09:15 IST" — real session timings, not a guessed schedule. When closed, also shows the next upcoming trading holiday for that exchange. Requires OpenAlgo |
| Live symbol search (optional) | Sidebar search box on the exploration page (`pages/prediction.py`) that queries OpenAlgo's live NSE/BSE symbol list by name or code — purely additive alongside the static CSV picker, which stays as the default/fallback. Requires OpenAlgo |
| Admin-gated Monitoring (optional) | The Monitoring page only opens for the account matching `ADMIN_EMAIL`, auto-provisioned from `.env` on first run (see [Authentication](#authentication)) |
| Multi-language UI | Sidebar language switcher — English, 中文, 한국어, 日本語, Türkçe, Русский — persisted for the session |
| Hardened SQLite | WAL journaling, busy-timeout, `synchronous=NORMAL`, and integrity checks on every store by default (see [Production Hardening](#production-hardening-sqlite--durability)) |
| Theme-aware styling | Shared CSS layer (`theme_ui.py`) applied on every page — polished buttons, metric cards, headings, alerts, and tables, built entirely on Streamlit's own theme CSS variables so it automatically matches light/dark/custom themes, no hardcoded colors |

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip`
- Optional: Docker Desktop
- Optional: Alpha Vantage API key for better cloud reliability
- Optional: a self-hosted [OpenAlgo](https://openalgo.in) instance + API key for exchange-native NSE/BSE data (see [OpenAlgo Integration](#openalgo-integration))

### Local Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Algorithmic-Stock-Price-Prediction

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: copy the env template and fill in whichever keys you want.
# Everything in it is optional — the app runs fully without any of them.
cp .env.example .env

# Run the app
streamlit run app.py
```

Open the app in your browser at the URL Streamlit prints in the terminal, usually `http://localhost:8501`.

`.env` is loaded automatically (`config.py` calls `load_dotenv()` at import time via `python-dotenv`) — it's a local-dev convenience so you don't have to `export` every variable in your shell each session. It's gitignored; never commit your real one. Real environment variables always take precedence over `.env` if both are set. This is separate from Streamlit Community Cloud's own secrets mechanism (`st.secrets`, backed by `.streamlit/secrets.toml` or the app's settings UI there) — `.env` only applies to local/Docker runs where `config.py` actually gets a chance to load it before the process's environment is read.

### Optional Extras (LSTM model + MLflow tracking)

`requirements.txt` stays lightweight on purpose, so free-tier hosts (Render free tier, Streamlit Community Cloud) aren't forced into a large PyTorch/MLflow install. Every feature built on them degrades gracefully without them — the LSTM option just doesn't appear in the model dropdown, and experiment tracking silently no-ops. To enable both:

```bash
pip install -r requirements-ml-extra.txt
```

### Running Tests

```bash
pytest tests/
```

The suite runs entirely offline — models and the backtester are tested against synthetic OHLCV data, and the service layer is tested against fake data/news sources, so no network access or API key is required. Tests for the optional extras (`test_lstm.py`, parts of `test_experiment_tracking.py`) skip automatically if PyTorch/MLflow aren't installed.

### Optional API Keys

None of these are required — the app works with none of them set, using free/no-key sources throughout. See `.env.example` for a ready-to-fill template, and the table below for what each one does.

---

## Environment Variables

| Variable | Description |
|---|---|
| `ADMIN_EMAIL` | Email for the auto-provisioned admin account — gates the Monitoring page when set alongside `ADMIN_PASSWORD` |
| `ADMIN_PASSWORD` | Password for that account, used only on first run (see [Authentication](#authentication) for why it's never overwritten after) |
| `SMTP_HOST` | SMTP server hostname — enables real email delivery for registration/reset OTP codes when set alongside the three below |
| `SMTP_PORT` | SMTP port (default `587`, i.e. STARTTLS) |
| `SMTP_USERNAME` | SMTP auth username (usually the sending mailbox address) |
| `SMTP_PASSWORD` | SMTP auth password (for Gmail, an [App Password](https://myaccount.google.com/apppasswords), not your account password) |
| `SMTP_FROM_ADDRESS` | `From:` address on outgoing mail (defaults to `SMTP_USERNAME` if unset) |
| `SMTP_USE_TLS` | `"false"` to disable STARTTLS (default `true`) — only for providers that don't support it |
| `AV_API_KEY` | Alpha Vantage key, used as the primary cloud data source when set |
| `OPENALGO_BASE_URL` | Root URL of your own self-hosted OpenAlgo instance (e.g. `http://127.0.0.1:5000`, or a tunnel URL like ngrok while developing) — enables the OpenAlgo data source for NSE/BSE tickers when set alongside the key below |
| `OPENALGO_API_KEY` | API key from your OpenAlgo instance's own settings page |
| `FRED_API_KEY` | Free FRED key ([fred.stlouisfed.org](https://fred.stlouisfed.org)) — enables macro features (Treasury yield, CPI) in the feature set |
| `NEWS_API_KEY` | Free NewsAPI.org dev key — broader headline coverage for the sentiment panel; falls back to yfinance's free news feed if unset |
| `MLFLOW_TRACKING_URI` | Overrides the default local `sqlite:///mlflow.db` store for experiment tracking |
| `LITESTREAM_BUCKET_URL` | S3-compatible replica URL root (e.g. `s3://my-bucket/stock-app`) — enables continuous SQLite replication in Docker; see [Production Hardening](#production-hardening-sqlite--durability) |
| `LOG_LEVEL` | Python logging level (default `INFO`) |

If `AV_API_KEY` is not set, the app falls back to Yahoo Finance. All other keys above are purely additive — unset means that feature is either off or uses its free fallback. `OPENALGO_BASE_URL`/`OPENALGO_API_KEY` are both-or-neither: OpenAlgoSource only activates once both are set, and points at an instance you run yourself — it is not a shared/public data API. If `SMTP_HOST`/`SMTP_USERNAME`/`SMTP_PASSWORD` aren't all set, OTP codes are logged server-side instead of emailed — see [Authentication](#authentication). `ADMIN_EMAIL`/`ADMIN_PASSWORD` are also both-or-neither: the Monitoring page stays unavailable to everyone (not open-to-all) until both are set.

**None of these belong in source control.** `.env` is gitignored; `.env.example` is the tracked template with every value left blank. If you're setting this up for your own deployment, copy it and fill in your own — see [CONTRIBUTING.md](CONTRIBUTING.md) for what "your own" means for each one.

---

## Project Structure

```
Algorithmic-Stock-Price-Prediction/
├── app.py                 # Thin Streamlit view: auth + prediction dashboard
├── web_context.py         # Cached service singletons shared across all pages
├── config.py              # Single source of settings (env vars / st.secrets)
├── theme_ui.py             # Shared CSS layer, applied once per page (theme-variable based)
├── pages/
│   ├── prediction.py      # Exploration page: candlesticks + AutoReg forecast
│   ├── watchlist.py       # Compare predictions across several tickers
│   ├── track_record.py    # Predicted-vs-actual accuracy history
│   └── monitoring.py      # Model health over time: accuracy, RMSE, drift
├── data_access/           # MarketDataSource, MacroFeatureSource, NewsSource
│                            interfaces + yfinance/Alpha Vantage/OpenAlgo/FRED/NewsAPI
│                            + OpenAlgoMarketCalendar (session status), depth snapshots
├── infra/                  # infra.db: shared SQLite hardening (WAL, busy-timeout,
│                            backup, integrity check) used by every repository below
├── i18n/                   # Translator + sidebar language switcher; translations/*.json
├── auth/                  # UserRepository, OtpRepository, EmailSender, AuthService
│                            (email identity + verification + OTP password reset)
│                            + password_hashing (shared bcrypt) + bootstrap (admin seeding)
├── features/              # FeaturePipeline + technical indicators + sentiment scoring
├── models/                # Predictor interface: naive, Ridge, gradient boosting, LSTM, AutoReg
├── evaluation/            # Walk-forward backtesting, prediction intervals, drift detection
├── monitoring/             # ExperimentTracker: always-on SqliteExperimentTracker
│                            + optional MlflowExperimentTracker, composed
├── track_record/            # PredictionRecord + its own SQLite-backed repository
├── services/                # PredictionService / SentimentService / TrackRecordService
├── scripts/
│   └── backup_sqlite.py   # Manual/scheduled hot backup + integrity check of all 3 DBs
├── tests/                   # pytest suite
├── data/
│   └── equity_issuers.csv
├── requirements.txt
├── requirements-ml-extra.txt  # Optional: PyTorch (LSTM) + MLflow
├── .env.example            # Template for every optional env var this app reads
├── dockerfile
├── entrypoint.sh           # Restores from Litestream (if configured) before starting
├── litestream.yml          # Continuous SQLite replication config (inactive by default)
├── .dockerignore
├── LICENSE                 # MIT
├── CONTRIBUTING.md         # Setup with your own secrets; what "open-source" means here
└── .github/workflows/      # Test suite + Docker build/push pipeline
```

---

## Deployment

The repository includes a Docker build that runs Streamlit on port `8501`, plus a GitHub Actions workflow that builds the image and pushes it to GHCR.

To run the container locally:

```bash
docker build -t stock-prediction-app .
docker run --rm -p 8501:8501 stock-prediction-app
```

To run it with continuous SQLite replication (see [Production Hardening](#production-hardening-sqlite--durability) below):

```bash
docker run --rm -p 8501:8501 \
  -e LITESTREAM_BUCKET_URL="s3://my-bucket/stock-app" \
  -e LITESTREAM_ACCESS_KEY_ID="..." \
  -e LITESTREAM_SECRET_ACCESS_KEY="..." \
  stock-prediction-app
```

**This only applies to Docker-based hosts** (Render, Fly.io, Railway, GHCR + your own compute, `docker run` locally). **Streamlit Community Cloud does not use this repository's Dockerfile at all** — it clones the repo and runs `streamlit run app.py` directly with its own managed runtime, so `entrypoint.sh`/Litestream never execute there; its filesystem is ephemeral regardless of anything in this repo. If you need the three SQLite stores to survive a redeploy on Streamlit Community Cloud specifically, either move to a Docker-based host above, or swap the SQLite repositories for a hosted Postgres (Supabase/Neon free tier) behind the same `UserRepository` / `PredictionRecordRepository` / `SqliteExperimentTracker` interfaces — see [Notes](#notes).

---

## Authentication

Identity is an email address, not an arbitrary username — `auth/service.py`'s `AuthService` composes three narrow, independently-testable pieces (constructor-injected, same Dependency Inversion pattern as everywhere else in this app):

- **`UserRepository`** (`auth/repository.py`) — accounts: email, password hash, lockout state, `email_verified` flag.
- **`OtpRepository`** (`auth/otp_repository.py`) — one active OTP code per `(email, purpose)`, hashed (never stored in plaintext), with an expiry and a bounded attempt counter. Its own table in `users.db`, same file, separate concern.
- **`EmailSender`** (`auth/email_sender.py`) — `SmtpEmailSender` for real delivery via any SMTP provider (stdlib `smtplib`, no extra dependency), or `NullEmailSender` when SMTP isn't configured, which logs the code server-side instead of emailing it. Local dev and the test suite never need a real mailbox — same degrade-gracefully contract every other optional integration in this app follows.

**The flow:**

1. **Register** (email + password) → account created unverified → a 6-digit code is emailed, valid for `otp_expiry_minutes` (default 10).
2. **Verify** → entering the correct code marks the account verified. Login is refused with a clear message until this happens ("resend code" is one click away, from the same screen the failed login lands on).
3. **Login** → email + password, same lockout policy as before (`max_login_attempts`, `lockout_minutes`).
4. **Forgot password** → request a code → enter the code + a new password. **This replaces an earlier flow that had no identity check at all** — typing any username and a new password reset that account's password outright. The OTP flow closes that gap. `request_password_reset()` also always returns the same response whether or not the email is registered, so the response itself can't be used to enumerate which emails have accounts.

OTP codes are short-lived, single-use (`used_at` is set on success and never reused — replaying a spent code fails the same as a wrong one), and rate-limited per code (`otp_max_attempts`, default 5) — a wrong guess is recorded and a new code must be requested after too many.

**Requesting new codes is rate-limited too** (`otp_resend_cooldown_seconds`, default 60) — verified live that without this, nothing stopped firing dozens of "resend code"/"forgot password" requests for the same account in under a second, each one a real email. The cooldown is per `(email, purpose)`, enforced silently: `request_password_reset()` always returns the same response regardless of whether a new email actually went out, so the throttling itself can't become a second enumeration signal. This limits abuse *targeted at one account* — it does not limit how many *distinct* accounts can be registered in bulk, which needs IP-level/CAPTCHA throttling this app has no request context for (better handled by a reverse proxy or the hosting platform).

**Passwords must be 8–72 bytes** (`min_password_length` / `max_password_length_bytes`) — length only, no forced complexity rules (composition requirements are out of favor per NIST 800-63B). The upper bound isn't arbitrary: this bcrypt build doesn't error past 72 bytes, it silently truncates, so without an explicit cap two different 100-character passwords sharing the same first 72 bytes would be accepted as if they were genuinely different credentials.

**Setting up real email delivery:** set `SMTP_HOST`/`SMTP_PORT`/`SMTP_USERNAME`/`SMTP_PASSWORD` (see [Environment Variables](#environment-variables)). Any SMTP provider works — a Gmail account with an [App Password](https://myaccount.google.com/apppasswords) is the fastest way to get this running for free in development; for production, a transactional provider (Brevo, Resend, Mailgun, SES, …) will have better deliverability than a personal Gmail account. Without any of these set, codes are written to the server log instead — check the console output to complete registration/reset locally.

**Migration note:** this replaces the previous username-based schema outright — `users.db`'s `users` table now has an `email` primary key, an `email_verified` column, and a `language` column (the account's language at registration, used for OTP emails — see [Multi-Language Support](#multi-language-support)) instead of `username`; `otp_codes` also gained an `issued_at` column (backs the resend cooldown above). There's no migration path from the old schema; a pre-existing local `users.db` needs to be deleted and accounts re-registered (it's gitignored local state, not shipped data). `track_record.db`/`monitoring.db` are unaffected — they still key off `st.session_state.username`, which just holds an email address now.

### Admin account (gates the Monitoring page)

The Monitoring page (`pages/monitoring.py`) is **admin-only**, and shows model health across *every* user of the app, not just the viewer's own — that's what makes an admin gate meaningful here rather than just a locked door around a page that would otherwise show the admin their own data.

- Set `ADMIN_EMAIL` and `ADMIN_PASSWORD` (see [Environment Variables](#environment-variables)). `auth/bootstrap.py`'s `ensure_admin_account()` runs once at startup (from `web_context.get_auth_service()`) and creates that account **already verified** — no OTP step, since you set the password yourself via a trusted channel (your own `.env`), not an email you don't yet control.
- **It only ever creates, never overwrites.** If the account already exists — including because you changed the password later via the normal forgot-password flow — bootstrap is a silent no-op. This is the one property the whole design hinges on: without it, every redeploy would quietly reset the admin password back to whatever `.env` still says, undoing any rotation.
- **Rotate the password after first login.** If `ADMIN_PASSWORD` was ever typed somewhere outside your own terminal (chat, a screen share, a support ticket), treat it as compromised — log in once, then use the normal "Forgot Password?" flow to set a new one you haven't shared anywhere. Bootstrap won't touch it again afterward.
- Both unset (the default) means the Monitoring page is unavailable to *everyone*, not open-to-all — there's no meaningful "admin" to gate it against otherwise.
- `pages/monitoring.py` calls `ModelMonitoringService.get_summary_all_users()`/`get_known_tickers_all_users()` instead of the per-user variants, and the history table gains a "User" column (`ModelMetricRecord.username`, already present on every row — no schema change needed) so the admin can see whose analysis is whose.

---

## Production Hardening (SQLite & Durability)

All three SQLite stores (`users.db`, `track_record.db`, `monitoring.db`) go through `infra/db.py`'s `connect()` instead of a bare `sqlite3.connect()`, which applies:

- **WAL journaling** (`PRAGMA journal_mode=WAL`) — readers are never blocked behind a writer.
- **`synchronous=NORMAL`** — the documented safe pairing with WAL; durable against application crashes, meaningfully cheaper than `FULL` on every write.
- **`busy_timeout=5000`** — momentary write contention retries internally instead of raising `database is locked` straight to the caller.
- **`foreign_keys=ON`** for correctness, even though none of the three schemas currently use cross-table foreign keys.

`infra/db.py` also exposes `integrity_check()` (SQLite's own `PRAGMA integrity_check`) and `backup_to()` (SQLite's *online* backup API — safe to call on a live connection, unlike copying the `.db` file on disk).

**Two independent layers of durability, both optional and off by default:**

1. **`scripts/backup_sqlite.py`** — a dependency-free local/scheduled backup: integrity-checks each store, then hot-backs it up to a timestamped file under `--out-dir` (default `backups/`). Run it from cron, a CI job, or by hand:
   ```bash
   python scripts/backup_sqlite.py --out-dir backups/
   ```
2. **Litestream (`litestream.yml`, `entrypoint.sh`)** — continuous replication of the WAL to S3-compatible object storage (AWS S3, Cloudflare R2, Backblaze B2, MinIO, …), for Docker-based deployments. Inactive unless `LITESTREAM_BUCKET_URL` is set; when it is, `entrypoint.sh` restores each store from its replica on container start (so a fresh container after a redeploy comes back with real data) and then replicates continuously while the app runs. See the `docker run` example above and `litestream.yml`'s comments for the credential env vars your storage provider expects.

**What this does not do:** no PRAGMA or backup tool changes SQLite's single-writer-at-a-time model — for this app's realistic traffic (one write per "Analyze" click, from a handful of users) that's not a real constraint, but it's the honest ceiling if you ever need Postgres-style concurrent writers instead.

---

## Multi-Language Support

Every page has a 🌐 language selector in the sidebar — English, 中文, 한국어, 日本語, Türkçe, and Русский — backed by `i18n/`:

- `i18n/translator.py` — `t(key, **kwargs)` looks up `key` in the active language's JSON catalog (`i18n/translations/{code}.json`), falling back to English and then to the raw key itself, so a missing translation degrades to readable text instead of a crash.
- `i18n/widget.py` — the sidebar selector, storing the choice in `st.session_state["lang"]` for the *session* (this part still resets on a fresh session/browser — the live UI language isn't restored on your next login).
- Every page calls `render_language_selector()` *before* the auth gate, so the login/register screen itself is shown in the chosen language, not just the app after logging in.
- `AuthService` (in `auth/service.py`) returns a `message_key` + params rather than a rendered string — the service computes *what happened*, `auth_ui.py` is the only place that turns it into displayed text via `t()`. This keeps translation out of the business-logic layer, the same split this app draws everywhere else between a service computing meaning and a page rendering it.
- **Outbound email content is localized too.** `t()` takes an optional `lang=` override (`i18n/translator.py`) precisely so `AuthService` — which never imports Streamlit — can render OTP email subject/body in a specific language rather than "whatever the current session happens to be in." The language a user registered under is stored on their account (`UserRecord.language`) and reused for every later email to that address (a resend, a password-reset code), since there's no browsing session to read a live language choice from at that point. This is one language snapshot per account (taken at registration), not a live preference — it doesn't change if they switch the sidebar selector afterward.

**Known scope limit:** table column headers and chart labels across all pages are translated; a few dynamically-generated strings that pass through third-party formatting (e.g. `sentiment.label.capitalize()`'s "Positive"/"Neutral"/"Negative") are not yet wired through `t()`. The translations themselves (zh/ko/ja/tr/ru) were drafted for this PR and have not had a native-speaker review — treat them as a solid first pass, not a certified translation, especially for the more technical phrasing (RMSE, drift, prediction intervals).

Adding a new language: drop a new `i18n/translations/{code}.json` with the same keys as `en.json`, add `{code}: "Display Name"}` to `LANGUAGES` in `i18n/translator.py`. `tests/test_i18n.py` enforces that every catalog has exactly the same key set as English, so a partial translation fails CI instead of shipping silently broken.

---

## OpenAlgo Integration

[OpenAlgo](https://openalgo.in) is a self-hosted, open-source gateway that normalizes many Indian brokers (Angel One, Zerodha, Upstox, Fyers, and others) behind one REST API. `data_access/openalgo_source.py` (`OpenAlgoSource`) is an additional `MarketDataSource`, composed ahead of Alpha Vantage/yfinance in `build_default_source()` for **NSE/BSE tickers only** (`.NS`/`.BO` suffixes — it never touches `AAPL`-style tickers).

This replaced an earlier direct Angel One SmartAPI integration. The difference matters: SmartAPI required this app to manage broker login itself (client code + PIN + TOTP → JWT, token refresh, resolving each ticker to Angel One's own `symboltoken` before every fetch). OpenAlgo does all of that on its own side, against whichever broker your instance is linked to — this app only ever sends a **single static API key** with a plain `symbol`/`exchange` pair:

- `POST {base_url}/api/v1/history` — `{apikey, symbol, exchange, interval: "D", start_date, end_date}` → daily OHLCV candles.
- `POST {base_url}/api/v1/quotes` — `{apikey, symbol, exchange}` → real-time quote, including `ltp` (last traded price).

No login step, no token/session state to hold, no separate symbol-resolution call — `OpenAlgoSource` is a single class with no auth counterpart, unlike the SmartAPI integration it replaced.

**More OpenAlgo capabilities, each its own class/method (Interface Segregation — see [Design Principles](#design-principles)):**

- **Market depth** (`OpenAlgoSource.get_depth`) — `POST /api/v1/depth` → top bid/ask levels for a ticker. Shown on the main app page alongside the live quote, purely descriptive (same "shown, never fed into a model" rule the news-sentiment panel follows).
- **Market status** (`data_access/openalgo_calendar.py`'s `OpenAlgoMarketCalendar.get_session`) — `POST /api/v1/market/timings` → whether NSE/BSE is in session right now, and when it opens/closes today. Kept in its own file/class, not on `OpenAlgoSource`: "give me data for this ticker" and "is the exchange open at all" are different questions, one ticker-scoped and one not, the same reasoning that kept `SmartAPISession`/`SmartAPISource` apart before OpenAlgo replaced that integration.
- **Next trading holiday** (`OpenAlgoMarketCalendar.get_next_holiday`) — `POST /api/v1/market/holidays` → the nearest upcoming holiday for an exchange this calendar year. Shown as extra context alongside the "market closed" status, not as its own widget — it only looks ahead within the current year, so it's "next holiday we know about," not an absolute guarantee.
- **Live symbol search** (`OpenAlgoSource.search_symbols`) — `POST /api/v1/search` → live NSE/BSE symbol/name lookup, filtered to equities only (futures/options contracts for the same underlying are excluded). Powers an optional search box on `pages/prediction.py`, purely additive alongside the existing static `data/equity_issuers.csv` picker, which remains the default.

**This points at an instance you run yourself, not a shared public API.** `OPENALGO_BASE_URL` is the root of your own OpenAlgo deployment — running locally (`http://127.0.0.1:5000` by default), tunneled for development (ngrok or similar), or hosted somewhere stable for production. `OPENALGO_API_KEY` comes from that instance's own settings page. Both unset means every method on `OpenAlgoSource`/`OpenAlgoMarketCalendar` no-ops without making a network request, exactly like `AlphaVantageSource` when `AV_API_KEY` is unset — see [Environment Variables](#environment-variables). That also means:

- For a single-owner deployment, this becomes a shared, higher-quality Indian-equity data source for everyone using the app — you link your broker to OpenAlgo once, and this app never sees your broker credentials at all, only OpenAlgo's own API key.
- A tunnel URL (ngrok's free tier, in particular) is ephemeral — it changes every time the tunnel restarts. Treat `OPENALGO_BASE_URL` as a value you'll need to update when that happens; for anything beyond local development, point it at a stable, persistently-running OpenAlgo deployment instead.

**Only read-only, non-account market data is used (history, quotes, depth, session timings) — and that's a permanent boundary, not a todo list.** OpenAlgo itself exposes a much larger surface: order placement, GTT orders, and — critically — **`/funds`, `/holdings`, `/positionbook`, `/orderbook`, `/tradebook`**, which return *your own real brokerage account's* money, positions, and trade history. This app's OpenAlgo connection is the *operator's* single instance, shared read-only across every visitor (see above) — it is not each end user's own account. That split is exactly why the account-specific endpoints must never be wired in here: doing so would leak the operator's real personal financial data to every visitor of a public deployment, not just to the operator. `OpenAlgoSource`/`OpenAlgoMarketCalendar` only ever call ticker- or exchange-scoped endpoints that return public market data, never anything scoped to the linked account. If you're contributing and considering adding one of the account endpoints above: don't — open an issue to discuss it first, since it changes this app's trust model, not just its feature list. This is a prediction tool, not a trading tool, and that's deliberate.

---

## Additional Features

A few design decisions behind the newer features, since they're not obvious from the UI alone:

- **Sentiment is descriptive, not a training feature.** Free news sources (yfinance, NewsAPI's free tier) only return *recent* headlines, not a historical archive. There's no honest way to backfill a per-day sentiment value across years of training history without fabricating it — so the sentiment panel is shown as live context next to a prediction, never fed into the model that prediction came from.
- **Prediction intervals come from real backtest error**, not an assumed distribution — the band is the empirical quantile spread of (actual − predicted) return across every walk-forward fold, so it only exists when there's enough backtest history to compute it honestly (it's omitted, not faked, otherwise).
- **Drift detection uses a KS-test**, not a full Evidently integration, to avoid taking on a heavier dependency whose exact API shape wasn't worth pinning against sight-unseen. `evaluation/drift.py`'s `DriftReport` is shaped so an Evidently-backed version could satisfy the same interface later without any caller changing.
- **The LSTM reuses the tabular `Predictor` interface** (`fit(X, y)` / `predict(X)`) rather than needing a separate code path — it carries the tail of its training window forward as sequence context for the start of the next prediction call, which is legitimate history, not a peek at the future, since walk-forward folds are always temporally adjacent.
- **Heavy optional deps (PyTorch, MLflow) live in `requirements-ml-extra.txt`**, not the base install, so the free-tier deployment story in this README stays true — the app is fully functional without them.
- **"Predicted next close" targets a genuinely future, unresolved date.** Earlier, the final live prediction was built from the same feature table used for training — which correctly drops the most recent trading day (its target return is unknown, nothing to compute it from yet). That meant the "prediction" was actually being made from *yesterday's* features, targeting a close that was already sitting in the fetched data. `FeaturePipeline.build_live_features()` now computes that one row separately, without requiring a target, so `predicted_next_close` is always a real forecast of a date not yet known — which is what makes the track record below meaningful rather than trivially resolvable.
- **Every prediction is recorded before its outcome exists**, keyed on `(username, ticker, model, target_date)` — re-analyzing the same ticker/model before that date arrives updates your pending record rather than duplicating it, but a record can never move backward from resolved to pending. That ordering (write first, resolve later, never edit after) is what makes the track record honest rather than a number that could be curated after the fact.
- **Model monitoring is always on and separate from MLflow on purpose.** `SqliteExperimentTracker` needs no extra dependency and upserts one row per `(username, ticker, model, day)` — clicking Analyze five times on the same ticker/model/day updates that day's row rather than piling up five identical entries (the models are all seeded, so a same-day rerun on the same data gives the same backtest result anyway). MLflow, when installed, is composed alongside it via `CompositeExperimentTracker` and keeps its own append-only history — that's MLflow's own data model, not something this app overrides, so the two intentionally behave differently: the in-app Monitoring page shows current state per day, MLflow shows every run ever.
- **Track record and monitoring are scoped per user *in storage*, auth is the only shared table.** Both `prediction_records` and `model_metrics` carry `username` as part of their primary key — two users predicting the same ticker with the same model on the same day get two separate rows, not one overwriting the other. That's independent of who's allowed to *read* the data: the Track Record page still only ever queries its own viewer's rows, while the admin-only Monitoring page (see [Authentication](#authentication)) deliberately drops the username filter to show every row — same storage, two different read policies layered on top, not two different schemas. `resolve_pending()` is the one deliberate exception on the write side: it isn't scoped to a single user, since checking a prediction against the real market close is the same fact-check for everyone, and it caches that lookup across users predicting the same ticker/date in one pass rather than re-fetching per user. This also drove a cleanup: `PredictionService` used to log to monitoring internally as a side effect of `analyze()`, which meant it had to know who was asking once monitoring became per-user. Logging is now an explicit call the page makes with the report afterward (`ModelMonitoringService.log_from_report`), symmetric with how `TrackRecordService.record_prediction` already worked — `PredictionService` stays a pure ticker-in, report-out function with no notion of users at all.
- **Every write path in this app is idempotent by design, and it's enforced at the repository, not just by caller discipline.** `save()` for both predictions and auth users upserts on their natural key; `resolve()` only updates a prediction `WHERE actual_close IS NULL`, so a second call — from any caller, not just the one that's careful — can't silently overwrite a real recorded outcome; `record_successful_login`/`reset_login_attempts` set fixed end states. The one deliberate exception is `record_failed_login`, which increments a counter — that's supposed to accumulate, since each failed attempt is a genuinely new event.
- **Styling is one shared CSS layer, not per-page inline styles.** `theme_ui.py`'s `apply_theme()` follows the exact same pattern as `i18n.render_language_selector()` — a single function every page calls once, near the top. The CSS itself only ever references Streamlit's own theme CSS variables (`--primary-color`, `--background-color`, `--secondary-background-color`, `--text-color`) rather than hardcoded hex colors, so it automatically matches whichever theme (light, dark, or a custom `.streamlit/config.toml`) the viewer has selected, with no separate dark-mode branch to maintain here.

---

## Notes

- The app creates local SQLite files for state: `users.db` (authentication), `track_record.db` (prediction history), and `monitoring.db` (model metrics over time). All three are listed in `.gitignore` and **not** tracked in git — don't commit any of them, and don't remove them from `.gitignore`.
- `track_record.db` and `monitoring.db` hold every user's data in one file, scoped internally by a `username` column — they're not one-file-per-user. Isolation is enforced in the queries (every read/write requires a username), not by the storage being physically separate.
- All three go through the hardened connection in `infra/db.py` (WAL, busy-timeout — see [Production Hardening](#production-hardening-sqlite--durability)), but hardening the SQLite engine doesn't change where the *file* lives. On local/Docker use with a real mounted volume, or Docker + Litestream, the data persists across restarts. **Streamlit Community Cloud's filesystem is ephemeral and doesn't run this repo's Dockerfile at all** — accounts, the track record, and the monitoring history all reset on every app restart/redeploy there, regardless of the hardening above. If you need persistence specifically on Streamlit Community Cloud, swap the SQLite repositories for a hosted Postgres (e.g. Supabase/Neon free tier) behind the same `UserRepository` / `PredictionRecordRepository` / `SqliteExperimentTracker` interface — that's exactly the seam those interfaces exist for.
- Every prediction is shown next to a naive "tomorrow = today" baseline. If a model doesn't beat it, the UI says so — that's a valid result, not a bug.
- Some tickers, especially on Yahoo Finance, may return incomplete history depending on market coverage and request limits.
- Indian stocks usually need `.NS` or `.BO` suffixes.
- If data fails to load in the browser but works locally, set `AV_API_KEY` and try again.

---

## Ownership

This project is maintained by its contributors.
