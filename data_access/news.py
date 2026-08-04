"""Recent headlines for a ticker, behind a Protocol so the sentiment
service doesn't care whether headlines came from yfinance (free, no key)
or NewsAPI — swapping the source is a constructor argument."""

from __future__ import annotations

from typing import Protocol

import requests
import yfinance as yf

from config import log

from .http_session import SESSION


class NewsSource(Protocol):
    def get_recent_headlines(self, ticker: str, limit: int = 10) -> list[str]: ...


class YFinanceNewsSource:
    """Free, no API key required — Yahoo Finance's own news feed."""

    def get_recent_headlines(self, ticker: str, limit: int = 10) -> list[str]:
        try:
            items = yf.Ticker(ticker, session=SESSION).news or []
        except Exception:
            log.warning("News fetch failed for %s", ticker, exc_info=True)
            return []

        headlines: list[str] = []
        for item in items[:limit]:
            title = item.get("title")
            if not title:
                title = (item.get("content") or {}).get("title")
            if title:
                headlines.append(title)
        return headlines


class NewsApiSource:
    """NewsAPI.org — broader coverage than a single exchange's feed, but
    needs a free API key and its free tier is dev-only (100 req/day)."""

    def __init__(self, api_key: str, timeout_seconds: int = 10):
        self._api_key = api_key
        self._timeout = timeout_seconds

    def get_recent_headlines(self, ticker: str, limit: int = 10) -> list[str]:
        if not self._api_key:
            return []
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": ticker,
                    "sortBy": "publishedAt",
                    "pageSize": limit,
                    "apiKey": self._api_key,
                },
                timeout=self._timeout,
            )
            articles = resp.json().get("articles", [])
            return [a["title"] for a in articles if a.get("title")]
        except Exception:
            log.warning("NewsAPI fetch failed for %s", ticker, exc_info=True)
            return []


def build_default_news_source(news_api_key: str = "") -> NewsSource:
    """NewsAPI when a key is configured (broader coverage), else the free
    yfinance feed — same fallback pattern as the market-data sources."""
    return NewsApiSource(news_api_key) if news_api_key else YFinanceNewsSource()
