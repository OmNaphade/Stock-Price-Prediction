"""Multi-step forecaster over the raw close-price series, used by the
exploration page's "N days ahead" chart. This is a different task from the
tabular Predictor models (forecasting the series' own future vs. predicting
next-day return from engineered features), so it isn't forced into the
Predictor Protocol.

Fixes relative to the original: lags were `min(250, n // 4)` — on ~500
daily observations that's up to 250 parameters, heavily overparameterized
and prone to unstable dynamic forecasts. Capped much lower here. The
forecast horizon is also capped: AutoReg's dynamic (multi-step) forecasts
compound one-step error at every step, so the further out you forecast, the
less the output means — treat a 90-day dynamic AR forecast as decorative,
not predictive.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from config import settings


class AutoRegForecaster:
    def __init__(
        self,
        max_lags: int = settings.autoreg_max_lags,
        forecast_days: int = settings.autoreg_forecast_days,
    ):
        self.max_lags = max_lags
        self.forecast_days = forecast_days
        self.last_lags_used: int | None = None

    def fit_predict(
        self, close: pd.DataFrame, train_fraction: float = 0.9
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | tuple[None, None, None, None]:
        """Returns (train_df, test_df, forecast, test_predictions)."""
        series = close.asfreq("B").ffill()
        n = len(series)
        if n < 30:
            return None, None, None, None

        lags = max(1, min(self.max_lags, n // 10))
        self.last_lags_used = lags
        split = int(n * train_fraction)
        train_df = series.iloc[: split + 1]
        test_df = series.iloc[split:]

        model = AutoReg(train_df["Close"], lags=lags).fit(cov_type="HC0")

        predictions = model.predict(start=test_df.index[0], end=test_df.index[-1], dynamic=True)
        forecast = model.predict(
            start=test_df.index[0],
            end=test_df.index[-1] + dt.timedelta(days=self.forecast_days),
            dynamic=True,
        )
        return train_df, test_df, forecast, predictions
