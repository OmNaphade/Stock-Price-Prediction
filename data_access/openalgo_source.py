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

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

from config import log

from .ohlcv import clean_ohlcv

_EXCHANGE_BY_SUFFIX = {"NS": "NSE", "BO": "BSE"}


@dataclass
class DepthLevel:
    price: float
    quantity: int


@dataclass
class MarketDepth:
    bids: list[DepthLevel]  # best bid first
    asks: list[DepthLevel]  # best ask first
    total_buy_qty: int
    total_sell_qty: int


@dataclass
class SymbolMatch:
    symbol: str  # OpenAlgo/tradingsymbol, e.g. "RELIANCE-EQ"
    name: str  # underlying name, e.g. "Reliance Industries"
    exchange: str
    instrument_type: str  # "EQ", "FUT", "CE", "PE", ...

    @property
    def ticker(self) -> str:
        """This app's own ticker format ('RELIANCE.NS'), derived from the
        OpenAlgo tradingsymbol — strips broker-style suffixes like "-EQ"
        that this app's other tickers (yfinance-style) don't use."""
        suffix = "NS" if self.exchange == "NSE" else "BO"
        base = self.symbol.split("-")[0]
        return f"{base}.{suffix}"


def split_indian_ticker(ticker: str) -> Optional[tuple[str, str]]:
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
    def is_configured(self) -> bool:
        return bool(self._base_url and self._api_key)

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if not self.is_configured:
            return pd.DataFrame()
        parsed = split_indian_ticker(ticker)
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
        if not self.is_configured:
            return None
        parsed = split_indian_ticker(ticker)
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

    def get_depth(self, ticker: str) -> Optional[MarketDepth]:
        """Order-book snapshot (top bid/ask levels) for an NSE/BSE ticker
        — descriptive market context for the live-quote panel, same as
        the news-sentiment panel: shown alongside a prediction, never fed
        into it. Not part of the MarketDataSource contract (history/quote
        only) since no other provider here shares this capability."""
        if not self.is_configured:
            return None
        parsed = split_indian_ticker(ticker)
        if parsed is None:
            return None
        symbol, exchange = parsed

        try:
            resp = requests.post(
                f"{self._base_url}/api/v1/depth",
                json={"apikey": self._api_key, "symbol": symbol, "exchange": exchange},
                timeout=self._timeout,
            )
            payload = resp.json()
            if payload.get("status") != "success":
                return None
            data = payload.get("data") or {}
            bids = [DepthLevel(l["price"], l["quantity"]) for l in data.get("bids") or []]
            asks = [DepthLevel(l["price"], l["quantity"]) for l in data.get("asks") or []]
            if not bids and not asks:
                return None
            return MarketDepth(
                bids=bids,
                asks=asks,
                total_buy_qty=data.get("totalbuyqty", 0),
                total_sell_qty=data.get("totalsellqty", 0),
            )
        except Exception:
            log.warning("OpenAlgo depth fetch failed for %s", ticker, exc_info=True)
            return None

    def search_symbols(self, query: str, exchange: str = "NSE", limit: int = 20) -> list[SymbolMatch]:
        """Live symbol lookup — lets the UI offer a much broader, always-
        current stock picker than the static `data/equity_issuers.csv`
        list, for NSE/BSE. Equity results only (`instrumenttype == "EQ"`):
        `/search` also returns futures/options contracts for the same
        underlying, which aren't tickers this app's prediction pipeline
        can do anything with."""
        if not self.is_configured or not query:
            return []
        try:
            resp = requests.post(
                f"{self._base_url}/api/v1/search",
                json={"apikey": self._api_key, "query": query, "exchange": exchange},
                timeout=self._timeout,
            )
            payload = resp.json()
            if payload.get("status") != "success":
                return []
            results = payload.get("data") or []
            matches = [
                SymbolMatch(
                    symbol=r["symbol"],
                    name=r.get("name", r["symbol"]),
                    exchange=r.get("exchange", exchange),
                    instrument_type=r.get("instrumenttype", ""),
                )
                for r in results
                if r.get("instrumenttype") == "EQ"
            ]
            return matches[:limit]
        except Exception:
            log.warning("OpenAlgo symbol search failed for %r", query, exc_info=True)
            return []
