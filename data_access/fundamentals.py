"""Yahoo Finance fundamentals lookup. Deliberately not part of
MarketDataSource — that interface is about OHLCV history and quotes;
bundling 30 fundamentals fields into it would violate Interface
Segregation for callers that only need price history."""

from __future__ import annotations

from typing import Any

import yfinance as yf

from config import log

from .http_session import SESSION

_FIELD_GROUPS: dict[str, list[str]] = {
    "Basic Information": ["symbol", "longName", "currency", "exchange"],
    "Market Data": [
        "currentPrice",
        "previousClose",
        "open",
        "dayLow",
        "dayHigh",
        "fiftyTwoWeekLow",
        "fiftyTwoWeekHigh",
        "fiftyDayAverage",
        "twoHundredDayAverage",
    ],
    "Volume and Shares": ["volume", "averageVolume", "sharesOutstanding"],
    "Dividends and Yield": ["dividendRate", "dividendYield", "payoutRatio"],
    "Valuation and Ratios": [
        "marketCap",
        "enterpriseValue",
        "priceToBook",
        "debtToEquity",
        "grossMargins",
        "profitMargins",
    ],
    "Financial Performance": [
        "totalRevenue",
        "revenuePerShare",
        "totalDebt",
        "earningsGrowth",
        "revenueGrowth",
        "returnOnAssets",
        "returnOnEquity",
    ],
    "Cash Flow": ["freeCashflow", "operatingCashflow"],
    "Analyst Targets": [
        "targetHighPrice",
        "targetLowPrice",
        "targetMeanPrice",
        "targetMedianPrice",
    ],
}


def fetch_stock_info(ticker: str) -> dict[str, dict[str, Any]]:
    try:
        info = yf.Ticker(ticker, session=SESSION).info
    except Exception:
        log.warning("Fundamentals fetch failed for %s", ticker, exc_info=True)
        info = {}

    return {
        group: {field: info.get(field, "N/A") for field in fields}
        for group, fields in _FIELD_GROUPS.items()
    }
