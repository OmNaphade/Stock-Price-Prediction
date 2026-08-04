"""Headline sentiment scoring — VADER, a free lexicon-based scorer well
suited to short text like headlines (no model download, pure Python).

Deliberately *not* fed into FeaturePipeline as a training feature: free
news sources only return recent headlines, not a historical archive, so
there's no honest way to backfill a per-day sentiment value across years
of training history without fabricating it. This is a live, descriptive
signal shown alongside a prediction — not a feature the model was ever
actually validated against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from config import log


@dataclass
class SentimentSnapshot:
    headline_count: int
    mean_compound: float
    label: str  # "positive" / "neutral" / "negative" / "unavailable"


class SentimentScorer(Protocol):
    def score(self, headlines: list[str]) -> SentimentSnapshot: ...


class VaderSentimentScorer:
    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            log.warning("vaderSentiment not installed; sentiment scoring disabled.")
            self._analyzer = None

    def score(self, headlines: list[str]) -> SentimentSnapshot:
        if self._analyzer is None or not headlines:
            return SentimentSnapshot(headline_count=0, mean_compound=0.0, label="unavailable")

        scores = [self._analyzer.polarity_scores(h)["compound"] for h in headlines]
        mean_compound = sum(scores) / len(scores)
        if mean_compound > 0.15:
            label = "positive"
        elif mean_compound < -0.15:
            label = "negative"
        else:
            label = "neutral"
        return SentimentSnapshot(
            headline_count=len(headlines), mean_compound=mean_compound, label=label
        )
