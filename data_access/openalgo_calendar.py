"""OpenAlgo's market-calendar endpoint (session timings) — kept separate
from OpenAlgoSource on purpose: that class answers "give me data for this
ticker" (the MarketDataSource contract); this one answers "is the market
open right now," a per-exchange calendar question with no ticker involved
at all. Same split SmartAPISession/SmartAPISource used to keep auth and
data-fetching apart: different concern, different class, even though both
end up talking to the same OpenAlgo instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from config import log

# India has no daylight saving time, so a fixed UTC+5:30 offset is exact —
# no need for the zoneinfo/tzdata dependency just for this one timezone.
IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class MarketSession:
    exchange: str
    is_open_now: bool
    start_time: datetime  # tz-aware, IST
    end_time: datetime  # tz-aware, IST


@dataclass
class Holiday:
    date: str  # ISO date (YYYY-MM-DD)
    description: str


class OpenAlgoMarketCalendar:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int = 15):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self._base_url and self._api_key)

    def get_session(self, exchange: str) -> Optional[MarketSession]:
        """Today's trading session for `exchange` ("NSE"/"BSE"), or None
        if unconfigured, unreachable, or there's no session today (OpenAlgo
        returns an empty list for weekends/holidays — indistinguishable
        here from "couldn't reach the exchange," which is the right call:
        callers shouldn't show a market-status widget at all rather than
        guess which case it was)."""
        if not self.is_configured:
            return None
        today = datetime.now(IST).strftime("%Y-%m-%d")
        try:
            resp = requests.post(
                f"{self._base_url}/api/v1/market/timings",
                json={"apikey": self._api_key, "date": today},
                timeout=self._timeout,
            )
            payload = resp.json()
            if payload.get("status") != "success":
                return None
            entries = payload.get("data") or []
            match = next((e for e in entries if e.get("exchange") == exchange), None)
            if match is None:
                return None
            start = datetime.fromtimestamp(match["start_time"] / 1000, tz=IST)
            end = datetime.fromtimestamp(match["end_time"] / 1000, tz=IST)
            now = datetime.now(IST)
            return MarketSession(exchange=exchange, is_open_now=start <= now <= end, start_time=start, end_time=end)
        except Exception:
            log.warning("OpenAlgo market timings fetch failed for %s", exchange, exc_info=True)
            return None

    def get_next_holiday(self, exchange: str) -> Optional[Holiday]:
        """The next upcoming trading holiday for `exchange` this calendar
        year, or None if unconfigured/unreachable, or there isn't one left
        this year — deliberately doesn't look ahead into next year (a
        December-31st edge case), so this is "next holiday we know about,"
        not an absolute guarantee one doesn't exist a few days out."""
        if not self.is_configured:
            return None
        today = datetime.now(IST).date()
        try:
            resp = requests.post(
                f"{self._base_url}/api/v1/market/holidays",
                json={"apikey": self._api_key, "year": today.year},
                timeout=self._timeout,
            )
            payload = resp.json()
            if payload.get("status") != "success":
                return None
            entries = payload.get("data") or []
            upcoming = sorted(
                (
                    h for h in entries
                    if exchange in (h.get("closed_exchanges") or [])
                    and datetime.strptime(h["date"], "%Y-%m-%d").date() >= today
                ),
                key=lambda h: h["date"],
            )
            if not upcoming:
                return None
            first = upcoming[0]
            return Holiday(date=first["date"], description=first.get("description", ""))
        except Exception:
            log.warning("OpenAlgo market holidays fetch failed for %s", exchange, exc_info=True)
            return None
