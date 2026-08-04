# Algorithmic Stock Price Prediction 📈
> Streamlit-based stock analysis and forecasting app with authenticated access, live market data fetches, moving-average charts, and lightweight prediction models.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)

> ⚠️ **Data note:** The app relies on public market data sources. Yahoo Finance can occasionally throttle requests, so the app also supports an optional Alpha Vantage fallback for more reliable cloud runs.

> 🚧 **Work in progress:** This project is still evolving. Data sources, models, and UI details may change as the app matures.

---

## What It Does

This app helps you explore stock history and generate next-price forecasts from the browser, with every forecast benchmarked against a naive baseline so you can tell whether it's actually predictive.

- **Authenticated access** with local username/password login, registration, and lockout after repeated failed attempts
- **Ticker-based lookup** for equities such as `AAPL`, `TSLA`, `RELIANCE.NS`, and `TCS.NS`
- **Historical analysis** with summaries, price charts, and moving averages
- **Next-close prediction** with a choice of models (naive baseline, regularized linear, gradient boosting, and an optional LSTM), each evaluated with walk-forward backtesting and directional accuracy — not just a single train/test split
- **Prediction intervals** derived from real out-of-sample backtest error, not a guessed margin
- **Feature drift detection** flags when a ticker's recent behavior statistically diverges from the window the backtest was computed on
- **News sentiment panel** (free, no key required) as descriptive context — deliberately not fed into the model; see [Additional Features](#additional-features) for why
- **Optional macro features** (interest rates, CPI via FRED)
- **Model monitoring** — every backtest run is logged (always on, no extra dependency), so directional accuracy, RMSE, and drift status are all visible as trends over time, not just in the moment; MLflow is available as an optional, heavier addition on top
- **Track record** — every prediction is recorded before its outcome is known, then checked against the real close once its target date passes, building an honest, unfalsifiable accuracy history over time
- **Watchlist page** comparing predictions across several tickers at once
- **Alternative exploration page** under `pages/prediction.py` with candlestick charts and AutoReg forecasting
- **Live market context** via `yfinance`, with optional Alpha Vantage support when an API key is available
- **Docker-ready deployment** for consistent local and cloud execution

### Architecture

The app is organized in layers so the UI never touches a data provider, the database, or a model class directly:

```
config.py            single source of settings (env vars / secrets)
data_access/          MarketDataSource, MacroFeatureSource, NewsSource interfaces
                       — yfinance, Alpha Vantage, FRED, NewsAPI, fallback chains
auth/                 UserRepository + AuthService (hashing, lockout)
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

**Single Responsibility.** `AuthService` (hashing, lockout policy) and `SqliteUserRepository` (storage) are two classes, not one, so a password-policy change and a schema change are unrelated edits. The same split repeats deliberately: `TrackRecordService` answers "was this specific prediction right" while `ModelMonitoringService` answers "how has this model's backtest performance looked over time" — two different questions about the same predictions, kept in two different services rather than one growing class that answers both badly. (`pages/monitoring.py` used to compute its own summary stats inline, the one place this split had been skipped — moved into `ModelMonitoringService` so every page follows the same rule: services compute what something *means*, pages only render it.)

**Open/Closed.** `AVAILABLE_MODELS` is a dict from name to `Predictor` factory (`services/prediction_service.py`) — the LSTM entry is *added* conditionally when PyTorch is importable, nothing else in the file changes to support it. `CompositeMarketDataSource` and `CompositeExperimentTracker` both take a list of providers and try/log each in turn — a new data source or a new tracker is an item appended to a list at the composition root (`web_context.py`), never a new `if` branch in existing logic. `FeaturePipeline(macro_source=...)` defaults to `NullMacroSource()` — turning macro features on is a constructor argument, not a code path threaded through the pipeline.

**Liskov Substitution.** `NaivePredictor`, `RidgeReturnPredictor`, `GradientBoostingReturnPredictor`, and `LSTMReturnPredictor` are fully interchangeable behind `fit(X, y)` / `predict(X)` — `walk_forward_backtest()` never branches on which one it was handed, and `test_all_predictors_share_the_same_interface` in `tests/test_models.py` pins exactly that.

**Interface Segregation.** `ExperimentTracker` (write: `log_backtest`) and `ModelMetricsReader` (read: `get_recent`, `get_tickers`) are two separate Protocols in `monitoring/`, even though one concrete class (`SqliteExperimentTracker`) satisfies both — `PredictionService` only ever needs to write, the Monitoring page only ever needs to read, and neither should have to depend on the half it doesn't use. `MarketDataSource` stays deliberately narrow (history + quote) rather than bundling in the 30-field fundamentals blob (`data_access/fundamentals.py`) or Yahoo-specific period/interval history (`YFinanceSource.get_history_by_period`) that most callers never need.

**Dependency Inversion.** `PredictionService.__init__(data_source: MarketDataSource, ...)` depends on an interface, never on `yfinance` or `sqlite3` directly — the concrete wiring happens exactly once, in `web_context.py`, the app's single composition root. No page imports `yfinance`, `torch`, or `sqlite3` itself; every page only ever imports from `services` and `web_context`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Data | yfinance, Alpha Vantage (optional) |
| Analysis | NumPy, Pandas, Matplotlib, Plotly |
| Forecasting | scikit-learn (Ridge, GradientBoosting), statsmodels AutoReg, optional PyTorch LSTM |
| Backtesting | Walk-forward validation (`sklearn.model_selection.TimeSeriesSplit`) |
| Macro data | FRED (`fredapi`), optional |
| Sentiment | VADER (`vaderSentiment`) over yfinance/NewsAPI headlines |
| Model monitoring | SQLite (always on) + MLflow (optional, additive) |
| Testing | pytest |
| Authentication | SQLite + bcrypt |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## How It Works

```
User enters a ticker + picks a model
      │
      ├── Auth check (AuthService → SqliteUserRepository)
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
| Login / Register | Local auth backed by SQLite and bcrypt, with lockout after repeated failed logins |
| Historical charts | Price history plus 100-day and 200-day moving averages |
| Model selection | Naive baseline, Ridge (regularized linear), gradient boosting, optional LSTM — same interface, swap freely |
| Backtested predictions | Walk-forward validation across multiple folds, with directional accuracy and price RMSE shown against the naive baseline |
| Prediction intervals | A confidence band built from real backtest error quantiles, not a fixed guessed margin |
| Drift detection | KS-test comparison of recent vs. reference feature windows, surfaced as a warning when they diverge |
| News sentiment | Free (yfinance) or NewsAPI headlines scored with VADER, shown as live context — not a training feature |
| Macro features (optional) | 10-year Treasury yield and CPI from FRED, added to the feature set only when `FRED_API_KEY` is set |
| Model monitoring | Every backtest run logged (always on) — directional accuracy, RMSE, and drift status charted over time per ticker/model; MLflow logs the same runs too when installed |
| Held-out fold chart | Actual vs. predicted price on the most recent backtest fold, not a cherry-picked window |
| Watchlist | Compare predictions across several tickers at once, ranked by predicted % change |
| Track record | Every prediction recorded pre-outcome, resolved against the real close once due, with a predicted-vs-actual chart and full history table |
| Multi-page Streamlit UI | Exploration page (`pages/prediction.py`), watchlist, track record, and monitoring, alongside the main app |
| Cloud-friendly fetches | curl_cffi session fallback for yfinance reliability |

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip`
- Optional: Docker Desktop
- Optional: Alpha Vantage API key for better cloud reliability

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

# Run the app
streamlit run app.py
```

Open the app in your browser at the URL Streamlit prints in the terminal, usually `http://localhost:8501`.

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

None of these are required — the app works with none of them set, using free/no-key sources throughout. See the table below.

---

## Environment Variables

| Variable | Description |
|---|---|
| `AV_API_KEY` | Alpha Vantage key, used as the primary cloud data source when set |
| `FRED_API_KEY` | Free FRED key ([fred.stlouisfed.org](https://fred.stlouisfed.org)) — enables macro features (Treasury yield, CPI) in the feature set |
| `NEWS_API_KEY` | Free NewsAPI.org dev key — broader headline coverage for the sentiment panel; falls back to yfinance's free news feed if unset |
| `MLFLOW_TRACKING_URI` | Overrides the default local `sqlite:///mlflow.db` store for experiment tracking |
| `LOG_LEVEL` | Python logging level (default `INFO`) |

If `AV_API_KEY` is not set, the app falls back to Yahoo Finance. All other keys above are purely additive — unset means that feature is either off or uses its free fallback.

---

## Project Structure

```
Algorithmic-Stock-Price-Prediction/
├── app.py                 # Thin Streamlit view: auth + prediction dashboard
├── web_context.py         # Cached service singletons shared across all pages
├── config.py              # Single source of settings (env vars / st.secrets)
├── pages/
│   ├── prediction.py      # Exploration page: candlesticks + AutoReg forecast
│   ├── watchlist.py       # Compare predictions across several tickers
│   ├── track_record.py    # Predicted-vs-actual accuracy history
│   └── monitoring.py      # Model health over time: accuracy, RMSE, drift
├── data_access/           # MarketDataSource, MacroFeatureSource, NewsSource
│                            interfaces + yfinance/Alpha Vantage/FRED/NewsAPI
├── auth/                  # UserRepository + AuthService (hashing, lockout)
├── features/              # FeaturePipeline + technical indicators + sentiment scoring
├── models/                # Predictor interface: naive, Ridge, gradient boosting, LSTM, AutoReg
├── evaluation/            # Walk-forward backtesting, prediction intervals, drift detection
├── monitoring/             # ExperimentTracker: always-on SqliteExperimentTracker
│                            + optional MlflowExperimentTracker, composed
├── track_record/            # PredictionRecord + its own SQLite-backed repository
├── services/                # PredictionService / SentimentService / TrackRecordService
├── tests/                   # pytest suite
├── data/
│   └── equity_issuers.csv
├── requirements.txt
├── requirements-ml-extra.txt  # Optional: PyTorch (LSTM) + MLflow
├── dockerfile
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

---

## Additional Features

A few design decisions behind the newer features, since they're not obvious from the UI alone:

- **Sentiment is descriptive, not a training feature.** Free news sources (yfinance, NewsAPI's free tier) only return *recent* headlines, not a historical archive. There's no honest way to backfill a per-day sentiment value across years of training history without fabricating it — so the sentiment panel is shown as live context next to a prediction, never fed into the model that prediction came from.
- **Prediction intervals come from real backtest error**, not an assumed distribution — the band is the empirical quantile spread of (actual − predicted) return across every walk-forward fold, so it only exists when there's enough backtest history to compute it honestly (it's omitted, not faked, otherwise).
- **Drift detection uses a KS-test**, not a full Evidently integration, to avoid taking on a heavier dependency whose exact API shape wasn't worth pinning against sight-unseen. `evaluation/drift.py`'s `DriftReport` is shaped so an Evidently-backed version could satisfy the same interface later without any caller changing.
- **The LSTM reuses the tabular `Predictor` interface** (`fit(X, y)` / `predict(X)`) rather than needing a separate code path — it carries the tail of its training window forward as sequence context for the start of the next prediction call, which is legitimate history, not a peek at the future, since walk-forward folds are always temporally adjacent.
- **Heavy optional deps (PyTorch, MLflow) live in `requirements-ml-extra.txt`**, not the base install, so the free-tier deployment story in this README stays true — the app is fully functional without them.
- **"Predicted next close" targets a genuinely future, unresolved date.** Earlier, the final live prediction was built from the same feature table used for training — which correctly drops the most recent trading day (its target return is unknown, nothing to compute it from yet). That meant the "prediction" was actually being made from *yesterday's* features, targeting a close that was already sitting in the fetched data. `FeaturePipeline.build_live_features()` now computes that one row separately, without requiring a target, so `predicted_next_close` is always a real forecast of a date not yet known — which is what makes the track record below meaningful rather than trivially resolvable.
- **Every prediction is recorded before its outcome exists**, keyed on `(ticker, model, target_date)` — re-analyzing the same ticker/model before that date arrives updates the pending record rather than duplicating it, but a record can never move backward from resolved to pending. That ordering (write first, resolve later, never edit after) is what makes the track record honest rather than a number that could be curated after the fact.
- **Model monitoring is always on and separate from MLflow on purpose.** `SqliteExperimentTracker` needs no extra dependency and upserts one row per `(ticker, model, day)` — clicking Analyze five times on the same ticker/model/day updates that day's row rather than piling up five identical entries (the models are all seeded, so a same-day rerun on the same data gives the same backtest result anyway). MLflow, when installed, is composed alongside it via `CompositeExperimentTracker` and keeps its own append-only history — that's MLflow's own data model, not something this app overrides, so the two intentionally behave differently: the in-app Monitoring page shows current state per day, MLflow shows every run ever.
- **Every write path in this app is idempotent by design, and it's enforced at the repository, not just by caller discipline.** `save()` for both predictions and auth users upserts on their natural key; `resolve()` only updates a prediction `WHERE actual_close IS NULL`, so a second call — from any caller, not just the one that's careful — can't silently overwrite a real recorded outcome; `record_successful_login`/`reset_login_attempts` set fixed end states. The one deliberate exception is `record_failed_login`, which increments a counter — that's supposed to accumulate, since each failed attempt is a genuinely new event.

---

## Notes

- The app creates local SQLite files for state: `users.db` (authentication), `track_record.db` (prediction history), and `monitoring.db` (model metrics over time). All three are listed in `.gitignore` and **not** tracked in git — don't commit any of them, and don't remove them from `.gitignore`.
- All three files live on local disk with no external backing store. That's fine for local/Docker use, but Streamlit Community Cloud's filesystem is ephemeral — accounts, the track record, and the monitoring history all reset on every app restart/redeploy there. If you want any of them to survive long-term on a free cloud host, swap the SQLite repository for a hosted Postgres (e.g. Supabase/Neon free tier) behind the same `UserRepository` / `PredictionRecordRepository` / `SqliteExperimentTracker` interface — that's exactly the seam those interfaces exist for.
- Every prediction is shown next to a naive "tomorrow = today" baseline. If a model doesn't beat it, the UI says so — that's a valid result, not a bug.
- Some tickers, especially on Yahoo Finance, may return incomplete history depending on market coverage and request limits.
- Indian stocks usually need `.NS` or `.BO` suffixes.
- If data fails to load in the browser but works locally, set `AV_API_KEY` and try again.

---

## Ownership

This project is maintained by its contributors.
