from __future__ import annotations

import pytest

from evaluation.backtest import BacktestResult
from evaluation.intervals import empirical_return_interval, price_interval_from_return_interval


def _result_with_errors(errors: list[float]) -> BacktestResult:
    result = BacktestResult(model_name="test")
    result.all_return_errors = errors
    return result


def test_empirical_interval_widens_with_confidence():
    errors = [(-0.05 + 0.001 * i) for i in range(100)]  # spread from -0.05 to +0.05
    result = _result_with_errors(errors)

    narrow = empirical_return_interval(result, confidence=0.5)
    wide = empirical_return_interval(result, confidence=0.95)

    narrow_width = narrow.upper_log_return - narrow.lower_log_return
    wide_width = wide.upper_log_return - wide.lower_log_return
    assert wide_width > narrow_width


def test_empirical_interval_raises_with_too_few_folds():
    result = _result_with_errors([0.01, -0.01])
    with pytest.raises(ValueError):
        empirical_return_interval(result)


def test_price_interval_brackets_last_close_when_errors_are_symmetric():
    errors = [-0.02, -0.01, 0.0, 0.01, 0.02] * 10
    result = _result_with_errors(errors)
    interval = empirical_return_interval(result, confidence=0.8)

    low, high = price_interval_from_return_interval(
        last_close=100.0, predicted_log_return=0.0, interval=interval
    )
    assert low < 100.0 < high
