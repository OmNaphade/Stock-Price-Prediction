"""Composes a news source with a sentiment scorer. Kept separate from
PredictionService — sentiment is descriptive context about a ticker right
now, not an input the prediction model was trained or validated on."""

from __future__ import annotations

from data_access.news import NewsSource
from features.sentiment import SentimentScorer, SentimentSnapshot


class SentimentService:
    def __init__(self, news_source: NewsSource, scorer: SentimentScorer):
        self._news_source = news_source
        self._scorer = scorer

    def snapshot(self, ticker: str, headline_limit: int = 10) -> SentimentSnapshot:
        headlines = self._news_source.get_recent_headlines(ticker, limit=headline_limit)
        return self._scorer.score(headlines)
