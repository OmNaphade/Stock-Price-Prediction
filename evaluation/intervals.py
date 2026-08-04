"""Prediction intervals derived from real walk-forward error, not an
in-sample residual spread. A point forecast with no uncertainty overstates
confidence; this gives a band a reader can actually trust, because it's
built from errors on data the model never trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import BacktestResult


@dataclass
class ReturnInterval:
    confidence: float
    lower_log_return: float
    upper_log_return: float


def empirical_return_interval(backtest_result: BacktestResult, confidence: float = 0.8) -> ReturnInterval:
    """Interval half-width comes from the empirical quantiles of every
    out-of-sample (actual - predicted) log-return the backtest produced.
    Centered on 0 error (i.e. added on top of the point prediction, not
    replacing it) since the point forecast is already bias-corrected by
    the model's own fit."""
    errors = np.asarray(backtest_result.all_return_errors, dtype=np.float64)
    if errors.size < 5:
        raise ValueError("Not enough backtest folds to estimate a prediction interval.")

    alpha = 1 - confidence
    lower = float(np.quantile(errors, alpha / 2))
    upper = float(np.quantile(errors, 1 - alpha / 2))
    return ReturnInterval(confidence=confidence, lower_log_return=lower, upper_log_return=upper)


def price_interval_from_return_interval(
    last_close: float, predicted_log_return: float, interval: ReturnInterval
) -> tuple[float, float]:
    low = last_close * float(np.exp(predicted_log_return + interval.lower_log_return))
    high = last_close * float(np.exp(predicted_log_return + interval.upper_log_return))
    return low, high
