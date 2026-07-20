# Algorithmic Stock Price Prediction 📈
> Streamlit-based stock analysis and forecasting app with authenticated access, live market data fetches, moving-average charts, and lightweight prediction models.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)

> ⚠️ **Data note:** The app relies on public market data sources. Yahoo Finance can occasionally throttle requests, so the app also supports an optional Alpha Vantage fallback for more reliable cloud runs.

> 🚧 **Work in progress:** This project is still evolving. Data sources, models, and UI details may change as the app matures.

---

## What It Does

This app helps you explore stock history and generate simple next-price forecasts from the browser. It combines market data ingestion, charting, and a minimal prediction workflow in a single Streamlit experience.

- **Authenticated access** with local username/password login and registration
- **Ticker-based lookup** for equities such as `AAPL`, `TSLA`, `RELIANCE.NS`, and `TCS.NS`
- **Historical analysis** with summaries, price charts, and moving averages
- **Next-close prediction** using engineered features and a simple regression model
- **Alternative prediction page** under `pages/prediction.py` with candlestick charts and AutoReg forecasting
- **Live market context** via `yfinance`, with optional Alpha Vantage support when an API key is available
- **Docker-ready deployment** for consistent local and cloud execution

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Data | yfinance, Alpha Vantage (optional) |
| Analysis | NumPy, Pandas, Matplotlib, Plotly |
| Forecasting | Custom linear regression, statsmodels AutoReg |
| Authentication | SQLite + bcrypt |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## How It Works

```
User enters a ticker
      │
      ├── Auth check (SQLite users.db)
      │
      ├── Fetch market data
      │     ├── Alpha Vantage if AV_API_KEY is set
      │     └── yfinance fallback with curl_cffi / requests
      │
      ├── Clean and transform OHLCV data
      │
      ├── Train a lightweight prediction model
      │
      └── Render charts, metrics, and forecast output
```

The main app in [app.py](app.py) uses engineered features such as open-close spread, high-low spread, and rolling averages to estimate the next close price. The alternative page in [pages/prediction.py](pages/prediction.py) focuses on stock selection, candlesticks, and AutoReg-based forecasting.

---

## Features

| Feature | Details |
|---|---|
| Login / Register | Local auth backed by SQLite and bcrypt |
| Historical charts | Price history plus 100-day and 200-day moving averages |
| Prediction output | Next-day close estimate with MAE and RMSE metrics |
| Test visualization | Actual vs predicted comparison on the test split |
| Multi-page Streamlit UI | Extra stock exploration page under `pages/` |
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

### Optional Alpha Vantage Setup

If you want more reliable data fetches when the app runs in the cloud, add an `AV_API_KEY` Streamlit secret or environment variable.

---

## Environment Variables

| Variable | Description |
|---|---|
| `AV_API_KEY` | Optional Alpha Vantage key used as the primary cloud data source |

If `AV_API_KEY` is not set, the app falls back to Yahoo Finance.

---

## Project Structure

```
Algorithmic-Stock-Price-Prediction/
├── app.py              # Main Streamlit app with auth, charts, and predictions
├── helper.py           # Shared fetch and forecasting helpers
├── pages/
│   └── prediction.py   # Additional Streamlit page for stock exploration
├── data/
│   └── equity_issuers.csv
├── requirements.txt
├── dockerfile
└── .github/workflows/  # Docker build and push pipeline
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

## Notes

- The app creates a local `users.db` SQLite database for authentication.
- Some tickers, especially on Yahoo Finance, may return incomplete history depending on market coverage and request limits.
- Indian stocks usually need `.NS` or `.BO` suffixes.
- If data fails to load in the browser but works locally, set `AV_API_KEY` and try again.

---

## Ownership

This project is maintained by its contributors.
