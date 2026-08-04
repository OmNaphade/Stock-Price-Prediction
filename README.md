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
- **Optional macro features** (interest rates, CPI via FRED) and **experiment tracking** (MLflow), both opt-in
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
monitoring/            ExperimentTracker interface: MLflow or a no-op
services/             PredictionService / SentimentService — the composition roots
web_context.py        shared, cached service instances for all pages
app.py, pages/         thin Streamlit views; no business logic lives here
tests/                 pytest suite covering every layer above
```

Adding a data provider or a model means adding one class that satisfies the relevant interface — no existing file needs to change.

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
| Experiment tracking | MLflow, optional |
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
| Experiment tracking (optional) | Every backtest run logged to MLflow (local SQLite store) when installed |
| Held-out fold chart | Actual vs. predicted price on the most recent backtest fold, not a cherry-picked window |
| Watchlist | Compare predictions across several tickers at once, ranked by predicted % change |
| Multi-page Streamlit UI | Exploration page (`pages/prediction.py`) with candlesticks and AutoReg forecasting, plus the watchlist |
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
│   └── watchlist.py       # Compare predictions across several tickers
├── data_access/           # MarketDataSource, MacroFeatureSource, NewsSource
│                            interfaces + yfinance/Alpha Vantage/FRED/NewsAPI
├── auth/                  # UserRepository + AuthService (hashing, lockout)
├── features/              # FeaturePipeline + technical indicators + sentiment scoring
├── models/                # Predictor interface: naive, Ridge, gradient boosting, LSTM, AutoReg
├── evaluation/            # Walk-forward backtesting, prediction intervals, drift detection
├── monitoring/             # ExperimentTracker interface: MLflow or a no-op
├── services/                # PredictionService / SentimentService — composition roots
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

---

## Notes

- The app creates a local `users.db` SQLite database for authentication. It's listed in `.gitignore` and is **not** tracked in git — don't remove it from `.gitignore`, and don't commit a database file that contains real credentials.
- Every prediction is shown next to a naive "tomorrow = today" baseline. If a model doesn't beat it, the UI says so — that's a valid result, not a bug.
- Some tickers, especially on Yahoo Finance, may return incomplete history depending on market coverage and request limits.
- Indian stocks usually need `.NS` or `.BO` suffixes.
- If data fails to load in the browser but works locally, set `AV_API_KEY` and try again.

---

## Ownership

This project is maintained by its contributors.
