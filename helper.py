import datetime as dt
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg
import requests

# ── Session setup with fallback ───────────────────────────────────────────────
try:
    from curl_cffi import requests as curl_requests
    # Use chrome101 — supported in both 0.7.x and 0.15.x
    YF_SESSION = curl_requests.Session(impersonate="chrome101")
    USE_CURL = True
except Exception:
    YF_SESSION = requests.Session()
    YF_SESSION.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    USE_CURL = False


def _fetch_with_retry(ticker_str: str, **kwargs) -> pd.DataFrame:
    """
    Try yf.download first, fall back to Ticker.history.
    Retries 3 times with backoff. Works around datacenter IP blocks.
    """
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        needed = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        if 'Close' not in needed:
            return pd.DataFrame()
        df = df[needed].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['Close'])

    last_err = None
    for attempt in range(3):
        try:
            # Primary: yf.download
            raw = yf.download(
                ticker_str,
                progress=False,
                threads=False,
                session=YF_SESSION,
                **kwargs,
            )
            df = _clean(raw)
            if not df.empty:
                return df

            # Fallback: Ticker.history
            t = yf.Ticker(ticker_str, session=YF_SESSION)
            hist_kwargs = {k: v for k, v in kwargs.items()
                          if k in ('period', 'interval', 'start', 'end', 'auto_adjust')}
            df2 = _clean(t.history(**hist_kwargs))
            if not df2.empty:
                return df2

            return pd.DataFrame()

        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(4 * (attempt + 1))

    raise last_err if last_err else RuntimeError("Unknown fetch error")


# ── Simple Linear Regression (pure NumPy, multi-feature) ─────────────────────
class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        denom = np.sum((X - X_mean) ** 2, axis=0)
        denom = np.where(denom == 0, 1e-12, denom)
        self.coef_ = np.sum((X - X_mean) * (y - y_mean), axis=0) / denom
        self.intercept_ = y_mean - np.dot(self.coef_, X_mean)

    def predict(self, X):
        return np.dot(np.array(X, dtype=np.float64), self.coef_) + self.intercept_


def fetch_stocks() -> dict:
    candidates = [
        Path.cwd() / "equity_issuers.csv",
        Path.cwd() / "data" / "equity_issuers.csv",
        Path(__file__).parent / "equity_issuers.csv",
        Path(__file__).parent / "data" / "equity_issuers.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "equity_issuers.csv not found. Place it in the same folder as the app."
        )
    df = pd.read_csv(csv_path, usecols=["Security Code", "Issuer Name"])
    return dict(zip(df["Security Code"], df["Issuer Name"]))


def fetch_periods_intervals() -> dict:
    return {
        "1d":  ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d":  ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y":  ["1d", "5d", "1wk", "1mo"],
        "2y":  ["1d", "5d", "1wk", "1mo"],
        "5y":  ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }


def fetch_stock_history(stock_ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = _fetch_with_retry(
        stock_ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
    )
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
    return df[cols]


def fetch_stock_info(stock_ticker: str) -> dict:
    ticker = yf.Ticker(stock_ticker, session=YF_SESSION)
    info = ticker.info

    def safe_get(d, key):
        return d.get(key, "N/A")

    return {
        "Basic Information": {
            "symbol":   safe_get(info, "symbol"),
            "longName": safe_get(info, "longName"),
            "currency": safe_get(info, "currency"),
            "exchange": safe_get(info, "exchange"),
        },
        "Market Data": {
            "currentPrice":               safe_get(info, "currentPrice"),
            "previousClose":              safe_get(info, "previousClose"),
            "open":                       safe_get(info, "open"),
            "dayLow":                     safe_get(info, "dayLow"),
            "dayHigh":                    safe_get(info, "dayHigh"),
            "fiftyTwoWeekLow":            safe_get(info, "fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh":           safe_get(info, "fiftyTwoWeekHigh"),
            "fiftyDayAverage":            safe_get(info, "fiftyDayAverage"),
            "twoHundredDayAverage":       safe_get(info, "twoHundredDayAverage"),
        },
        "Volume and Shares": {
            "volume":             safe_get(info, "volume"),
            "averageVolume":      safe_get(info, "averageVolume"),
            "sharesOutstanding":  safe_get(info, "sharesOutstanding"),
        },
        "Dividends and Yield": {
            "dividendRate":  safe_get(info, "dividendRate"),
            "dividendYield": safe_get(info, "dividendYield"),
            "payoutRatio":   safe_get(info, "payoutRatio"),
        },
        "Valuation and Ratios": {
            "marketCap":      safe_get(info, "marketCap"),
            "enterpriseValue": safe_get(info, "enterpriseValue"),
            "priceToBook":    safe_get(info, "priceToBook"),
            "debtToEquity":   safe_get(info, "debtToEquity"),
            "grossMargins":   safe_get(info, "grossMargins"),
            "profitMargins":  safe_get(info, "profitMargins"),
        },
        "Financial Performance": {
            "totalRevenue":    safe_get(info, "totalRevenue"),
            "revenuePerShare": safe_get(info, "revenuePerShare"),
            "totalDebt":       safe_get(info, "totalDebt"),
            "earningsGrowth":  safe_get(info, "earningsGrowth"),
            "revenueGrowth":   safe_get(info, "revenueGrowth"),
            "returnOnAssets":  safe_get(info, "returnOnAssets"),
            "returnOnEquity":  safe_get(info, "returnOnEquity"),
        },
        "Cash Flow": {
            "freeCashflow":      safe_get(info, "freeCashflow"),
            "operatingCashflow": safe_get(info, "operatingCashflow"),
        },
        "Analyst Targets": {
            "targetHighPrice":   safe_get(info, "targetHighPrice"),
            "targetLowPrice":    safe_get(info, "targetLowPrice"),
            "targetMeanPrice":   safe_get(info, "targetMeanPrice"),
            "targetMedianPrice": safe_get(info, "targetMedianPrice"),
        },
    }


def generate_stock_prediction(stock_ticker: str):
    try:
        df = _fetch_with_retry(
            stock_ticker,
            period="2y",
            interval="1d",
            auto_adjust=True,
        )

        if df is None or df.empty or 'Close' not in df.columns:
            return None, None, None, None

        # Strip timezone before asfreq (pandas 2.x requirement)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        close = df[["Close"]].copy()
        close = close.asfreq("B").ffill()

        n = len(close)
        if n < 30:
            return None, None, None, None

        lags = min(250, n // 4)
        split = int(n * 0.9)
        train_df = close.iloc[: split + 1]
        test_df  = close.iloc[split:]

        model = AutoReg(train_df["Close"], lags=lags).fit(cov_type="HC0")

        predictions = model.predict(
            start=test_df.index[0],
            end=test_df.index[-1],
            dynamic=True,
        )

        forecast = model.predict(
            start=test_df.index[0],
            end=test_df.index[-1] + dt.timedelta(days=90),
            dynamic=True,
        )

        return train_df, test_df, forecast, predictions

    except Exception:
        return None, None, None, None
