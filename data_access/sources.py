"""Market data access, behind one narrow interface.

`MarketDataSource` is the seam the rest of the app depends on (Dependency
Inversion): the service layer and UI never import yfinance or call
Alpha Vantage's REST API directly. Adding a new provider means adding a new
class here that satisfies the Protocol — nothing else changes (Open/Closed).
"""

from __future__ import annotations

import time
from typing import Optional, Protocol, Sequence

import pandas as pd
import requests
import yfinance as yf

from config import log, settings

from .http_session import SESSION
from .ohlcv import clean_ohlcv as _clean_ohlcv
from .openalgo_source import OpenAlgoSource


class MarketDataSource(Protocol):
    """A provider of historical OHLCV data and a live quote for a ticker."""

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily OHLCV between start and end (inclusive-ish), or an empty
        DataFrame if the ticker/range can't be served by this source."""
        ...

    def get_quote(self, ticker: str) -> Optional[float]:
        """Latest known price, or None if unavailable from this source."""
        ...


class YFinanceSource:
    """Yahoo Finance via yfinance. Works well locally; can be IP-blocked on
    some cloud hosts, which is why it's usually paired with a fallback."""

    def __init__(self, session=SESSION, retries: int = 3, backoff_seconds: float = 4.0):
        self._session = session
        self._retries = retries
        self._backoff = backoff_seconds

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return self._fetch_with_retry(ticker, start=start, end=end, auto_adjust=True)

    def get_history_by_period(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Yahoo-specific period/interval history (e.g. intraday). Not part
        of the MarketDataSource contract since other providers don't share
        this exact vocabulary — callers that need it depend on YFinanceSource
        concretely, which is fine: it's a Yahoo-specific capability."""
        return self._fetch_with_retry(ticker, period=period, interval=interval, auto_adjust=True)

    def get_quote(self, ticker: str) -> Optional[float]:
        try:
            price = yf.Ticker(ticker, session=self._session).info.get("currentPrice")
            return float(price) if price is not None else None
        except Exception:
            log.warning("yfinance quote fetch failed for %s", ticker, exc_info=True)
            return None

    def _fetch_with_retry(self, ticker: str, **kwargs) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(self._retries):
            try:
                raw = yf.download(
                    ticker, progress=False, threads=False, session=self._session, **kwargs
                )
                df = _clean_ohlcv(raw)
                if not df.empty:
                    return df

                hist_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k in ("period", "interval", "start", "end", "auto_adjust")
                }
                df2 = _clean_ohlcv(
                    yf.Ticker(ticker, session=self._session).history(**hist_kwargs)
                )
                return df2
            except Exception as exc:
                last_err = exc
                if attempt < self._retries - 1:
                    time.sleep(self._backoff * (attempt + 1))
        if last_err is not None:
            log.warning("yfinance history fetch failed for %s: %s", ticker, last_err)
        return pd.DataFrame()


class AlphaVantageSource:
    """Alpha Vantage — reliable from cloud IPs that Yahoo blocks, but rate
    limited on the free tier and requires an API key."""

    def __init__(self, api_key: str, timeout_seconds: int = 15):
        self._api_key = api_key
        self._timeout = timeout_seconds

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if not self._api_key:
            return pd.DataFrame()
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": ticker,
                    "outputsize": "full",
                    "apikey": self._api_key,
                },
                timeout=self._timeout,
            )
            data = resp.json().get("Time Series (Daily)", {})
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data).T
            df.index = pd.to_datetime(df.index)
            df = df.rename(
                columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "5. adjusted close": "Close",
                    "6. volume": "Volume",
                }
            )
            df = _clean_ohlcv(df)
            return df.loc[start:end]
        except Exception:
            log.warning("Alpha Vantage history fetch failed for %s", ticker, exc_info=True)
            return pd.DataFrame()

    def get_quote(self, ticker: str) -> Optional[float]:
        if not self._api_key:
            return None
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": self._api_key},
                timeout=self._timeout,
            )
            price = resp.json().get("Global Quote", {}).get("05. price")
            return float(price) if price else None
        except Exception:
            log.warning("Alpha Vantage quote fetch failed for %s", ticker, exc_info=True)
            return None


class CompositeMarketDataSource:
    """Tries each source in order and returns the first usable result.

    Adding a provider is: construct it and pass it into this list — no
    branching logic to edit (this is what replaces the old
    if-Alpha-Vantage-else-yfinance chain)."""

    def __init__(self, sources: Sequence[MarketDataSource]):
        if not sources:
            raise ValueError("CompositeMarketDataSource needs at least one source")
        self._sources = list(sources)

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        for source in self._sources:
            df = source.get_history(ticker, start, end)
            if df is not None and not df.empty:
                return df
        return pd.DataFrame()

    def get_quote(self, ticker: str) -> Optional[float]:
        for source in self._sources:
            price = source.get_quote(ticker)
            if price is not None:
                return price
        return None


def build_default_source() -> CompositeMarketDataSource:
    """Composition root for the default provider chain: OpenAlgo first for
    NSE/BSE tickers (exchange-native, via your own self-hosted OpenAlgo
    instance — see config.py's openalgo_* settings), then Alpha Vantage
    (reliable from cloud IPs) when a key is configured, then yfinance.
    Each source no-ops for tickers/situations it can't serve, so this
    ordering only affects which provider answers first, never correctness."""
    return CompositeMarketDataSource(
        [
            OpenAlgoSource(settings.openalgo_base_url, settings.openalgo_api_key),
            AlphaVantageSource(settings.av_api_key),
            YFinanceSource(),
        ]
    )
