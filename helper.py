import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg

# ── curl_cffi is in env (0.15.0) — always use it ─────────────────────────────
from curl_cffi import requests as curl_requests
YF_SESSION = curl_requests.Session(impersonate="chrome110")


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
    """
    Returns {company_name: ticker_symbol} from equity_issuers.csv.
    Searches for the CSV in several candidate locations so it works
    regardless of the Streamlit working directory.
    """
    candidates = [
        Path.cwd() / "equity_issuers.csv",
        Path.cwd() / "data" / "equity_issuers.csv",
        Path(__file__).parent / "equity_issuers.csv",
        Path(__file__).parent / "data" / "equity_issuers.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "equity_issuers.csv not found. Place it in the same folder as the app "
            "or in a 'data/' sub-folder."
        )
    df = pd.read_csv(csv_path, usecols=["Security Code", "Issuer Name"])
    # Security Code = company display name, Issuer Name = Yahoo Finance ticker symbol
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
    """Fetch OHLC history for a ticker given period and interval."""
    ticker = yf.Ticker(stock_ticker, session=YF_SESSION)
    hist = ticker.history(period=period, interval=interval)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist[["Open", "High", "Low", "Close"]]


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
            "regularMarketPreviousClose": safe_get(info, "regularMarketPreviousClose"),
            "regularMarketOpen":          safe_get(info, "regularMarketOpen"),
            "regularMarketDayLow":        safe_get(info, "regularMarketDayLow"),
            "regularMarketDayHigh":       safe_get(info, "regularMarketDayHigh"),
            "fiftyTwoWeekLow":            safe_get(info, "fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh":           safe_get(info, "fiftyTwoWeekHigh"),
            "fiftyDayAverage":            safe_get(info, "fiftyDayAverage"),
            "twoHundredDayAverage":       safe_get(info, "twoHundredDayAverage"),
        },
        "Volume and Shares": {
            "volume":                    safe_get(info, "volume"),
            "regularMarketVolume":       safe_get(info, "regularMarketVolume"),
            "averageVolume":             safe_get(info, "averageVolume"),
            "averageVolume10days":       safe_get(info, "averageVolume10days"),
            "averageDailyVolume10Day":   safe_get(info, "averageDailyVolume10Day"),
            "sharesOutstanding":         safe_get(info, "sharesOutstanding"),
            "impliedSharesOutstanding":  safe_get(info, "impliedSharesOutstanding"),
            "floatShares":               safe_get(info, "floatShares"),
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
            "totalCash":       safe_get(info, "totalCash"),
            "totalCashPerShare": safe_get(info, "totalCashPerShare"),
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
    """
    Fetch 2 years of daily close data, fit an AutoReg model,
    and return (train_df, test_df, forecast, predictions).

    FIX 1: strip timezone from index BEFORE calling asfreq() —
            pandas 2.3 raises TypeError on tz-aware DatetimeIndex.
    FIX 2: use ffill() instead of deprecated fillna(method='ffill').
    FIX 3: return (None, None, None, None) only on genuine failure;
            caller must NOT gate on forecast >= 0 (AutoReg residuals
            can be negative even for valid price predictions).
    """
    try:
        ticker = yf.Ticker(stock_ticker, session=YF_SESSION)
        hist = ticker.history(period="2y", interval="1d")

        if hist is None or hist.empty:
            return None, None, None, None

        # ── FIX 1: remove timezone BEFORE asfreq ─────────────────────────────
        hist.index = pd.to_datetime(hist.index).tz_localize(None)

        close = hist[["Close"]].copy()

        # ── FIX 2: pandas 2.x — use ffill() not fillna(method='ffill') ───────
        close = close.asfreq("B")   # business-day frequency fills weekend gaps
        close = close.ffill()

        n = len(close)
        if n < 30:                  # not enough data to fit AutoReg(lags=250)
            return None, None, None, None

        lags = min(250, n // 4)     # guard: lags must be < n/2

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
    