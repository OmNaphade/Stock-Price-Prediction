"""Technical indicators, implemented directly on pandas Series/DataFrames.

Hand-rolled rather than pulling in TA-Lib/pandas-ta: the formulas are a few
lines each, and it keeps the dependency list unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100 - (100 / (1 + rs))
    # avg_loss == 0 is where rs (and the formula above) blows up to NaN.
    # That's either a strict uptrend — textbook RSI = 100, maximally
    # overbought — or genuinely no movement at all (avg_gain also 0), which
    # is the only case that's actually neutral. A blanket fillna(50) here
    # would flatten real momentum into "neutral," which is wrong in exactly
    # the case (a strong trend) this indicator exists to catch.
    return result.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_percent_b(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_width = (upper - lower).replace(0.0, np.nan)
    return ((close - lower) / band_width).clip(0, 1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
