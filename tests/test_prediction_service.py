from __future__ import annotations

import pandas as pd
import pytest

from evaluation.backtest import BacktestResult, FoldMetrics
from services.prediction_service import PredictionError, PredictionReport, PredictionService


class _FakeDataSource:
    """A MarketDataSource stand-in — proves the service depends only on the
    Protocol, not on yfinance/Alpha Vantage, and lets the test run offline."""

    def __init__(self, ohlcv, quote=None):
        self._ohlcv = ohlcv
        self._quote = quote

    def get_history(self, ticker, start, end):
        return self._ohlcv

    def get_quote(self, ticker):
        return self._quote


def _service(data_source) -> PredictionService:
    return PredictionService(data_source)


def test_analyze_end_to_end_with_fake_source(synthetic_ohlcv):
    service = _service(_FakeDataSource(synthetic_ohlcv, quote=123.45))
    report = service.analyze("FAKE", "2020-01-01", "2021-12-31", "Ridge (linear)")

    assert report.ticker == "FAKE"
    assert report.model_name == "Ridge (linear)"
    assert report.predicted_next_close > 0
    assert report.live_quote == 123.45
    assert report.model_backtest.folds
    assert report.baseline_backtest.folds
    assert report.interval_low is not None
    assert report.interval_low < report.interval_high  # a real, non-degenerate band
    assert report.drift_report is not None


def test_analyze_raises_prediction_error_on_empty_data():
    import pandas as pd

    service = _service(_FakeDataSource(pd.DataFrame()))
    with pytest.raises(PredictionError):
        service.analyze("EMPTY", "2020-01-01", "2021-12-31", "Ridge (linear)")


def test_analyze_raises_on_unknown_model(synthetic_ohlcv):
    service = _service(_FakeDataSource(synthetic_ohlcv))
    with pytest.raises(PredictionError):
        service.analyze("FAKE", "2020-01-01", "2021-12-31", "Not A Real Model")


def test_analyze_raises_on_too_little_history(make_ohlcv):
    tiny = make_ohlcv(n=30, seed=1)
    service = _service(_FakeDataSource(tiny))
    with pytest.raises(PredictionError):
        service.analyze("TINY", "2020-01-01", "2021-12-31", "Ridge (linear)")


def test_prediction_targets_a_genuinely_future_date_not_already_known_data(synthetic_ohlcv):
    """Regression test: the live prediction used to be built from
    features_df's last row, which build() drops the *actual* most recent
    trading day from (nothing to shift its target in from yet) — so
    'last_close' was silently one day stale, and 'predicted_next_close' was
    "predicting" a close that was already sitting in `ohlcv`. Both must now
    reflect the true most recent day and a date strictly after it."""
    service = _service(_FakeDataSource(synthetic_ohlcv))
    report = service.analyze("FAKE", "2020-01-01", "2021-12-31", "Ridge (linear)")

    assert report.last_close == synthetic_ohlcv["Close"].iloc[-1]
    assert report.target_date > synthetic_ohlcv.index[-1]


def _report_with_directional_accuracies(model_acc: float, baseline_acc: float) -> PredictionReport:
    model_backtest = BacktestResult(model_name="test-model")
    model_backtest.folds = [
        FoldMetrics(0, 10, 0.0, 0.0, model_acc, 1.0, 1.0),
    ]
    baseline_backtest = BacktestResult(model_name="naive_persistence")
    baseline_backtest.folds = [
        FoldMetrics(0, 10, 0.0, 0.0, baseline_acc, 1.0, 1.0),
    ]
    return PredictionReport(
        ticker="X", ohlcv=None, features_df=None, feature_columns=[], model_name="test-model",
        model_backtest=model_backtest, baseline_backtest=baseline_backtest,
        predicted_log_return=0.0, predicted_next_close=100.0, last_close=100.0,
        target_date=pd.Timestamp("2026-01-02"), live_quote=None, interval_low=None,
        interval_high=None, interval_confidence=None, drift_report=None,
    )


def test_beats_baseline_on_direction_compares_against_coin_flip_not_naive_accuracy():
    """Regression test: NaivePredictor always predicts exactly 0 return, so
    sign(0) can never equal sign(a nonzero actual return) — its own
    directional accuracy is ~0% by construction. Comparing a real model
    against that would make every model 'beat the baseline' trivially. The
    correct comparison is a 50% coin flip."""
    # A genuinely mediocre model (45% — worse than a coin flip) must not
    # register as "beating baseline" just because 45% > baseline's ~0%.
    mediocre = _report_with_directional_accuracies(model_acc=0.45, baseline_acc=0.02)
    assert not mediocre.beats_baseline_on_direction

    # A model that actually clears a coin flip should register as beating it.
    good = _report_with_directional_accuracies(model_acc=0.55, baseline_acc=0.02)
    assert good.beats_baseline_on_direction
