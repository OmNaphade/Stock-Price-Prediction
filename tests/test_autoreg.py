from __future__ import annotations

from models.autoreg import AutoRegForecaster


def test_autoreg_caps_lags_relative_to_sample_size(synthetic_ohlcv):
    forecaster = AutoRegForecaster(max_lags=250, forecast_days=30)
    close = synthetic_ohlcv[["Close"]]

    train_df, test_df, forecast, predictions = forecaster.fit_predict(close)

    assert train_df is not None
    # Regression guard for the original bug: with n=500 the old formula
    # (min(250, n // 4)) used up to 125 lags; the new cap keeps it far
    # smaller relative to sample size even when max_lags itself is large.
    assert forecaster.last_lags_used is not None
    assert forecaster.last_lags_used <= len(close) // 10
    assert len(predictions) > 0
    assert len(forecast) > len(predictions)  # forecast extends past the test window


def test_autoreg_returns_none_on_too_little_data(make_ohlcv):
    tiny = make_ohlcv(n=10, seed=1)
    forecaster = AutoRegForecaster()
    train_df, test_df, forecast, predictions = forecaster.fit_predict(tiny[["Close"]])
    assert train_df is None
    assert forecast is None
