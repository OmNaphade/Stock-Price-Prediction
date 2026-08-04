"""Composition root for the prediction workflow. This is the only place
that wires a concrete MarketDataSource to a concrete model — Streamlit
pages call `PredictionService.analyze()` and never touch yfinance, sklearn,
or the backtester directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import log, settings
from data_access.sources import MarketDataSource
from evaluation.backtest import BacktestResult, walk_forward_backtest
from evaluation.drift import DriftReport, check_feature_drift
from evaluation.intervals import empirical_return_interval, price_interval_from_return_interval
from features.pipeline import TARGET_COLUMN, FeaturePipeline
from models.base import Predictor
from models.baseline import NaivePredictor
from models.gradient_boosting import GradientBoostingReturnPredictor
from models.linear import RidgeReturnPredictor
from monitoring.experiment_tracking import ExperimentTracker, build_experiment_tracker

AVAILABLE_MODELS: dict[str, Callable[[], Predictor]] = {
    "Naive (baseline)": NaivePredictor,
    "Ridge (linear)": RidgeReturnPredictor,
    "Gradient Boosting": GradientBoostingReturnPredictor,
}

# LSTM needs PyTorch, which is a heavy optional dependency — only offer it
# as a choice when it's actually importable, rather than crashing the
# dropdown or silently substituting a different model.
from models.lstm import TORCH_AVAILABLE, LSTMReturnPredictor

if TORCH_AVAILABLE:
    AVAILABLE_MODELS["LSTM (deep learning)"] = LSTMReturnPredictor


class PredictionError(Exception):
    """Raised for user-facing conditions (no data, not enough history) —
    callers show `str(error)` directly, no stack trace needed."""


@dataclass
class PredictionReport:
    ticker: str
    ohlcv: pd.DataFrame
    features_df: pd.DataFrame
    model_name: str
    model_backtest: BacktestResult
    baseline_backtest: BacktestResult
    predicted_log_return: float
    predicted_next_close: float
    last_close: float
    live_quote: Optional[float]
    interval_low: Optional[float]
    interval_high: Optional[float]
    interval_confidence: Optional[float]
    drift_report: Optional[DriftReport]

    @property
    def beats_baseline_on_direction(self) -> bool:
        # Deliberately NOT compared against the naive baseline's own
        # directional accuracy: NaivePredictor always predicts exactly 0
        # return, and sign(0) can never equal sign(a nonzero actual return)
        # — so its directional accuracy is mathematically ~0% by
        # construction, which would make every real model "beat" it
        # trivially. A coin flip (50%) is the honest floor for a binary
        # up/down call; the naive baseline stays the right comparison for
        # price-level error (below), just not for direction.
        return self.model_backtest.mean_directional_accuracy > 0.5

    @property
    def beats_baseline_on_price_error(self) -> bool:
        return self.model_backtest.mean_rmse_price < self.baseline_backtest.mean_rmse_price


class PredictionService:
    def __init__(
        self,
        data_source: MarketDataSource,
        feature_pipeline: Optional[FeaturePipeline] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        self._data_source = data_source
        self._feature_pipeline = feature_pipeline or FeaturePipeline()
        self._experiment_tracker = experiment_tracker or build_experiment_tracker()

    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return self._data_source.get_history(ticker, start, end)

    def get_live_quote(self, ticker: str) -> Optional[float]:
        try:
            return self._data_source.get_quote(ticker)
        except Exception:
            log.warning("Live quote fetch failed for %s", ticker, exc_info=True)
            return None

    def analyze(
        self,
        ticker: str,
        start: str = settings.history_start,
        end: Optional[str] = None,
        model_name: str = "Ridge (linear)",
    ) -> PredictionReport:
        if model_name not in AVAILABLE_MODELS:
            raise PredictionError(f"Unknown model '{model_name}'.")

        # Computed at call time, not baked in as a default — a Python
        # default argument (or a frozen Settings field) evaluated once at
        # import would freeze "today" at whatever date the process started
        # on, silently drifting stale over the life of a long-running
        # server, which is exactly the bug this replaced.
        end = end or date.today().isoformat()

        ohlcv = self.load_history(ticker, start, end)
        if ohlcv.empty:
            raise PredictionError(
                f"No data returned for '{ticker}'. Check the symbol, or try an "
                f"exchange suffix (e.g. .NS / .BO for Indian tickers)."
            )

        features_df = self._feature_pipeline.build(ohlcv)
        if len(features_df) < settings.min_training_rows:
            raise PredictionError(
                f"Only {len(features_df)} usable rows after feature warm-up — "
                f"need at least {settings.min_training_rows} to train and backtest."
            )

        feature_columns = self._feature_pipeline.feature_columns
        model_factory = AVAILABLE_MODELS[model_name]
        baseline_factory = AVAILABLE_MODELS["Naive (baseline)"]

        model_backtest = walk_forward_backtest(model_factory, features_df, feature_columns)
        baseline_backtest = walk_forward_backtest(baseline_factory, features_df, feature_columns)

        final_model = model_factory()
        X_all = features_df[feature_columns].to_numpy(dtype=np.float64)
        y_all = features_df[TARGET_COLUMN].to_numpy(dtype=np.float64)
        final_model.fit(X_all, y_all)

        last_row = features_df.iloc[[-1]]
        predicted_log_return = float(
            final_model.predict(last_row[feature_columns].to_numpy(dtype=np.float64))[0]
        )
        last_close = float(last_row["close"].iloc[0])
        predicted_next_close = last_close * float(np.exp(predicted_log_return))

        interval_low = interval_high = interval_confidence = None
        try:
            interval = empirical_return_interval(model_backtest)
            interval_low, interval_high = price_interval_from_return_interval(
                last_close, predicted_log_return, interval
            )
            interval_confidence = interval.confidence
        except ValueError:
            pass  # not enough backtest folds to estimate an interval — omit it, don't fake one

        drift_report = check_feature_drift(features_df, feature_columns)

        self._log_experiment(ticker, model_name, feature_columns, model_backtest, baseline_backtest)

        return PredictionReport(
            ticker=ticker,
            ohlcv=ohlcv,
            features_df=features_df,
            model_name=model_name,
            model_backtest=model_backtest,
            baseline_backtest=baseline_backtest,
            predicted_log_return=predicted_log_return,
            predicted_next_close=predicted_next_close,
            last_close=last_close,
            live_quote=self.get_live_quote(ticker),
            interval_low=interval_low,
            interval_high=interval_high,
            interval_confidence=interval_confidence,
            drift_report=drift_report,
        )

    def _log_experiment(
        self,
        ticker: str,
        model_name: str,
        feature_columns: list[str],
        model_backtest: BacktestResult,
        baseline_backtest: BacktestResult,
    ) -> None:
        try:
            self._experiment_tracker.log_backtest(
                ticker=ticker,
                model_name=model_name,
                params={"n_folds": len(model_backtest.folds), "n_features": len(feature_columns)},
                metrics={
                    "model_directional_accuracy": model_backtest.mean_directional_accuracy,
                    "baseline_directional_accuracy": baseline_backtest.mean_directional_accuracy,
                    "model_rmse_price": model_backtest.mean_rmse_price,
                    "baseline_rmse_price": baseline_backtest.mean_rmse_price,
                },
            )
        except Exception:
            log.warning("Experiment tracking failed for %s/%s", ticker, model_name, exc_info=True)
