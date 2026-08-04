from __future__ import annotations

import numpy as np
import pandas as pd

from features.indicators import atr, bollinger_percent_b, macd, rsi


def test_rsi_is_100_on_a_strict_uptrend_not_50():
    """Regression test: avg_loss == 0 (no down days in the window) used to
    fall through a blanket fillna(50), flattening a maximally overbought
    signal into 'neutral.' A strict uptrend must read as RSI = 100."""
    close = pd.Series(np.linspace(100, 130, 30))  # strictly increasing
    result = rsi(close, window=14)
    assert result.iloc[-1] == 100.0


def test_rsi_is_0_on_a_strict_downtrend():
    close = pd.Series(np.linspace(130, 100, 30))  # strictly decreasing
    result = rsi(close, window=14)
    assert result.iloc[-1] == 0.0


def test_rsi_is_50_on_a_perfectly_flat_series():
    close = pd.Series([100.0] * 30)  # no movement at all — genuinely neutral
    result = rsi(close, window=14)
    assert result.iloc[-1] == 50.0


def test_rsi_stays_within_bounds_on_noisy_data(synthetic_ohlcv):
    result = rsi(synthetic_ohlcv["Close"], window=14)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd_histogram_is_difference_of_macd_and_signal(synthetic_ohlcv):
    macd_line, signal_line, histogram = macd(synthetic_ohlcv["Close"])
    pd.testing.assert_series_equal(histogram, macd_line - signal_line, check_names=False)


def test_bollinger_percent_b_is_bounded(synthetic_ohlcv):
    result = bollinger_percent_b(synthetic_ohlcv["Close"]).dropna()
    assert (result >= 0).all() and (result <= 1).all()


def test_atr_is_nonnegative(synthetic_ohlcv):
    result = atr(synthetic_ohlcv).dropna()
    assert (result >= 0).all()
