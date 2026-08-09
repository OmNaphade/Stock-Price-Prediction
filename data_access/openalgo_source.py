"""OpenAlgo — a self-hosted, open-source gateway that normalizes many
Indian brokers (Angel One, Zerodha, Upstox, Fyers, ...) behind one REST
API. This replaces an earlier direct Angel One SmartAPI integration:
OpenAlgo handles broker-specific auth (TOTP, session tokens, symbol-token
lookups) on its own side, so this class only ever needs a single static
API key sent with each request — no login step, no token refresh, no
separate symbol-resolution call before every fetch.

Like every other MarketDataSource here, this is account-bound — a
specific OpenAlgo instance you run yourself, pointed at your own linked
broker — not a free public API. Unconfigured (`openalgo_base_url`/
`openalgo_api_key` unset) means every method no-ops, same
degrade-gracefully contract AlphaVantageSource follows when AV_API_KEY
is unset.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from config import log

from .ohlcv import clean_ohlcv

_EXCHANGE_BY_SUFFIX = {"NS": "NSE", "BO": "BSE"}


def _split_ticker(ticker: str) -> Optional[tuple[str, str]]:
    """'RELIANCE.NS' -> ('RELIANCE', 'NSE'); None for anything OpenAlgo's
    NSE/BSE-style coverage doesn't serve (US tickers, a bare symbol with
    no exchange to infer, or a non-string/falsy value — reproduced live:
    a None ticker used to raise TypeError here instead of degrading
    gracefully like every other MarketDataSource does)."""
    if not ticker or "." not in ticker:
        return None
    symbol, _, suffix = ticker.rpartition(".")
    exchange = _EXCHANGE_BY_SUFFIX.get(suffix.upper())
    return (symbol, exchange) if exchange else None


class OpenAlgoSource:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int = 15):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds

    @property
    def _is_configured(self) -> bool:
        return bool(self._base_url and self._api_key)

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if not self._is_configured:
            return pd.DataFrame()
        parsed = _split_ticker(ticker)
        if parsed is None:
            return pd.DataFrame()
        symbol, exchange = parsed

        try:
            resp = requests.post(
                f"{self._base_url}/api/v1/history",
                json={
                    "apikey": self._api_key,
                    "symbol": symbol,
                    "exchange": exchange,
                    "interval": "D",
                    "start_date": start,
                    "end_date": end,
                },
                timeout=self._timeout,
            )
            payload = resp.json()
            if payload.get("status") != "success":
                return pd.DataFrame()
            candles = payload.get("data") or []
            if not candles:
                return pd.DataFrame()
            df = pd.DataFrame(candles).rename(
                columns={
                    "timestamp": "Date", "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "volume": "Volume",
                }
            )
            df["Date"] = pd.to_datetime(df["Date"])
            return clean_ohlcv(df.set_index("Date"))
        except Exception:
            log.warning("OpenAlgo history fetch failed for %s", ticker, exc_info=True)
            return pd.DataFrame()

    def get_quote(self, ticker: str) -> Optional[float]:
        if not self._is_configured:
            return None
        parsed = _split_ticker(ticker)
        if parsed is None:
            return None
        symbol, exchange = parsed

        try:
            resp = requests.post(
                f"{self._base_url}/api/v1/quotes",
                json={"apikey": self._api_key, "symbol": symbol, "exchange": exchange},
                timeout=self._timeout,
            )
            payload = resp.json()
            if payload.get("status") != "success":
                return None
            price = (payload.get("data") or {}).get("ltp")
            return float(price) if price is not None else None
        except Exception:
            log.warning("OpenAlgo quote fetch failed for %s", ticker, exc_info=True)
            return None
