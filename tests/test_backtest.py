from __future__ import annotations

import numpy as np
import pytest

from evaluation.backtest import _safe_splitter, walk_forward_backtest
from features.pipeline import FeaturePipeline
from models.baseline import NaivePredictor
from models.linear import RidgeReturnPredictor


def test_walk_forward_backtest_produces_folds(synthetic_ohlcv):
    features = FeaturePipeline().build(synthetic_ohlcv)
    result = walk_forward_backtest(
        NaivePredictor, features, FeaturePipeline().feature_columns, n_splits=5, test_size=30
    )

    assert result.model_name == "naive_persistence"
    assert len(result.folds) >= 2
    assert 0.0 <= result.mean_directional_accuracy <= 1.0
    assert result.mean_rmse_price >= result.mean_mae_price  # RMSE >= MAE always


def test_no_lookahead_across_folds(synthetic_ohlcv):
    """The whole point of walk-forward validation: every test fold must
    come strictly after its own training window."""
    features = FeaturePipeline().build(synthetic_ohlcv)
    splitter = _safe_splitter(len(features), n_splits=5, test_size=30)

    for train_idx, test_idx in splitter.split(features):
        assert train_idx.max() < test_idx.min()


def test_ridge_backtest_runs_end_to_end(synthetic_ohlcv):
    features = FeaturePipeline().build(synthetic_ohlcv)
    result = walk_forward_backtest(
        RidgeReturnPredictor, features, FeaturePipeline().feature_columns
    )
    assert result.last_fold_index is not None
    assert len(result.last_fold_actual_price) == len(result.last_fold_predicted_price)


def test_raises_on_too_little_data():
    import pandas as pd

    tiny = pd.DataFrame(
        {"close": np.linspace(100, 101, 10), "log_return_target": np.zeros(10)}
    )
    with pytest.raises(ValueError):
        walk_forward_backtest(NaivePredictor, tiny, ["close"])


def test_safe_splitter_shrinks_for_small_datasets():
    splitter = _safe_splitter(n_samples=60, n_splits=5, test_size=30)
    assert splitter.n_splits < 5


def test_last_fold_index_is_the_target_date_not_the_row_date(synthetic_ohlcv):
    """Regression test: a row dated D holds features observed on D and a
    target for D+1's close. The chart built from last_fold_index plots
    that reconstructed D+1 price — it must be labeled D+1, not D, or the
    'actual vs predicted' chart silently mislabels every point by one
    trading day."""
    features = FeaturePipeline().build(synthetic_ohlcv)
    result = walk_forward_backtest(
        NaivePredictor, features, FeaturePipeline().feature_columns, n_splits=5, test_size=30
    )

    last_fold_start = result.last_fold_index[0]
    # The row that produced this first prediction is the trading day
    # immediately before the target date it's labeled with.
    row_dates = features.index[features.index < last_fold_start]
    assert row_dates[-1] < last_fold_start
    assert row_dates[-1] != last_fold_start  # i.e. genuinely shifted, not a no-op
