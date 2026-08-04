"""Walk-forward validation. Replaces the original single 80/20 chronological
split — one arbitrary boundary tells you nothing about whether a model's
error is stable across market regimes. Every fold trains only on data that
precedes its test window, so nothing from the future leaks backward.

Every predictor is scored the same way, including `NaivePredictor`, so the
service layer can always show "does this beat doing nothing" next to the
model's own numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config import settings
from features.pipeline import TARGET_COLUMN, TARGET_DATE_COLUMN
from models.base import Predictor


@dataclass
class FoldMetrics:
    fold: int
    n_test: int
    mae_return: float
    rmse_return: float
    directional_accuracy: float
    mae_price: float
    rmse_price: float


@dataclass
class BacktestResult:
    model_name: str
    folds: list[FoldMetrics] = field(default_factory=list)
    # Real held-out predictions from the most recent fold, kept for charting
    # "actual vs. predicted" — not cherry-picked, just the latest window.
    # last_fold_index holds each prediction's *target* date (the date the
    # price actually refers to), not the date of the row that produced it —
    # those differ by one trading day, since the row at date D predicts D+1.
    last_fold_index: Optional[pd.Index] = None
    last_fold_actual_price: Optional[np.ndarray] = None
    last_fold_predicted_price: Optional[np.ndarray] = None
    # Every held-out (actual - predicted) log-return, pooled across all
    # folds — the honest source for a prediction interval: it's real
    # out-of-sample error, not an in-sample residual.
    all_return_errors: list[float] = field(default_factory=list)

    def _mean(self, attr: str) -> float:
        return float(np.mean([getattr(f, attr) for f in self.folds])) if self.folds else float("nan")

    @property
    def mean_directional_accuracy(self) -> float:
        return self._mean("directional_accuracy")

    @property
    def mean_mae_price(self) -> float:
        return self._mean("mae_price")

    @property
    def mean_rmse_price(self) -> float:
        return self._mean("rmse_price")

    @property
    def mean_mae_return(self) -> float:
        return self._mean("mae_return")


def _safe_splitter(n_samples: int, n_splits: int, test_size: int) -> TimeSeriesSplit:
    """Shrinks folds/test-size until the split actually fits the data,
    instead of letting sklearn raise on small tickers/date ranges."""
    splits = n_splits
    while splits >= 2:
        if n_samples > (splits + 1) * test_size:
            return TimeSeriesSplit(n_splits=splits, test_size=test_size)
        splits -= 1
    fallback_size = max(5, n_samples // 4)
    return TimeSeriesSplit(n_splits=2, test_size=fallback_size)


def walk_forward_backtest(
    predictor_factory: Callable[[], Predictor],
    features_df: pd.DataFrame,
    feature_columns: Sequence[str],
    n_splits: int = settings.walk_forward_folds,
    test_size: int = settings.test_fold_size,
) -> BacktestResult:
    """Fits a fresh predictor on each expanding training window and scores
    it on the held-out fold immediately after it."""
    if len(features_df) < 40:
        raise ValueError(
            f"Not enough rows ({len(features_df)}) to run a walk-forward backtest."
        )

    model_name = predictor_factory().name
    result = BacktestResult(model_name=model_name)

    X = features_df[list(feature_columns)].to_numpy(dtype=np.float64)
    y = features_df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    close = features_df["close"].to_numpy(dtype=np.float64)

    splitter = _safe_splitter(len(features_df), n_splits, test_size)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        predictor = predictor_factory()
        predictor.fit(X[train_idx], y[train_idx])
        pred_return = predictor.predict(X[test_idx])
        actual_return = y[test_idx]

        pred_price = close[test_idx] * np.exp(pred_return)
        actual_price = close[test_idx] * np.exp(actual_return)

        result.folds.append(
            FoldMetrics(
                fold=fold_idx,
                n_test=len(test_idx),
                mae_return=float(np.mean(np.abs(actual_return - pred_return))),
                rmse_return=float(np.sqrt(np.mean((actual_return - pred_return) ** 2))),
                directional_accuracy=float(
                    np.mean(np.sign(pred_return) == np.sign(actual_return))
                ),
                mae_price=float(np.mean(np.abs(actual_price - pred_price))),
                rmse_price=float(np.sqrt(np.mean((actual_price - pred_price) ** 2))),
            )
        )
        result.last_fold_index = pd.DatetimeIndex(
            features_df[TARGET_DATE_COLUMN].to_numpy()[test_idx]
        )
        result.last_fold_actual_price = actual_price
        result.last_fold_predicted_price = pred_price
        result.all_return_errors.extend((actual_return - pred_return).tolist())

    return result
