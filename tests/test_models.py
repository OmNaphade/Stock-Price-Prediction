from __future__ import annotations

import numpy as np

from features.pipeline import TARGET_COLUMN, FeaturePipeline
from models.baseline import NaivePredictor
from models.gradient_boosting import GradientBoostingReturnPredictor
from models.linear import RidgeReturnPredictor


def _split_features(synthetic_ohlcv):
    features = FeaturePipeline().build(synthetic_ohlcv)
    X = features[FeaturePipeline().feature_columns].to_numpy()
    y = features[TARGET_COLUMN].to_numpy()
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def test_naive_predictor_always_predicts_zero_return(synthetic_ohlcv):
    X_train, X_test, y_train, y_test = _split_features(synthetic_ohlcv)
    model = NaivePredictor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (len(X_test),)
    assert np.all(preds == 0.0)


def test_ridge_predictor_fits_and_predicts_right_shape(synthetic_ohlcv):
    X_train, X_test, y_train, y_test = _split_features(synthetic_ohlcv)
    model = RidgeReturnPredictor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (len(X_test),)
    assert np.all(np.isfinite(preds))


def test_gradient_boosting_predictor_fits_and_predicts_right_shape(synthetic_ohlcv):
    X_train, X_test, y_train, y_test = _split_features(synthetic_ohlcv)
    model = GradientBoostingReturnPredictor(n_estimators=20)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (len(X_test),)
    assert np.all(np.isfinite(preds))


def test_all_predictors_share_the_same_interface():
    for cls in (NaivePredictor, RidgeReturnPredictor, GradientBoostingReturnPredictor):
        instance = cls()
        assert hasattr(instance, "name")
        assert callable(instance.fit)
        assert callable(instance.predict)
