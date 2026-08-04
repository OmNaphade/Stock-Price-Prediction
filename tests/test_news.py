from __future__ import annotations

from data_access.news import NewsApiSource, YFinanceNewsSource, build_default_news_source


def test_newsapi_source_returns_empty_without_key():
    source = NewsApiSource(api_key="")
    assert source.get_recent_headlines("AAPL") == []


def test_yfinance_news_source_handles_fetch_failure_gracefully(monkeypatch):
    class _BoomTicker:
        def __init__(self, *a, **k):
            raise RuntimeError("network down")

    monkeypatch.setattr("data_access.news.yf.Ticker", _BoomTicker)
    source = YFinanceNewsSource()
    assert source.get_recent_headlines("AAPL") == []


def test_build_default_news_source_picks_newsapi_when_key_present():
    assert isinstance(build_default_news_source("some-key"), NewsApiSource)


def test_build_default_news_source_falls_back_to_yfinance_without_key():
    assert isinstance(build_default_news_source(""), YFinanceNewsSource)
