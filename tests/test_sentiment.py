from __future__ import annotations

from features.sentiment import VaderSentimentScorer


def test_positive_headlines_score_positive():
    scorer = VaderSentimentScorer()
    snapshot = scorer.score(
        ["Company beats earnings expectations, stock surges on record profit"]
    )
    assert snapshot.label == "positive"
    assert snapshot.mean_compound > 0
    assert snapshot.headline_count == 1


def test_negative_headlines_score_negative():
    scorer = VaderSentimentScorer()
    snapshot = scorer.score(
        ["Company misses earnings, shares plunge amid fraud investigation"]
    )
    assert snapshot.label == "negative"
    assert snapshot.mean_compound < 0


def test_no_headlines_is_unavailable_not_a_fake_neutral():
    scorer = VaderSentimentScorer()
    snapshot = scorer.score([])
    assert snapshot.label == "unavailable"
    assert snapshot.headline_count == 0
