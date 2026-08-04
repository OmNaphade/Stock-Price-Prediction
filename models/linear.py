"""Regularized linear model, replacing the original hand-rolled
closed-form OLS. Ridge + StandardScaler fixes two issues at once: the
original model had no regularization (unstable on collinear moving-average
features) and no feature scaling (coefficients distorted by mismatched
feature magnitudes)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class RidgeReturnPredictor:
    name = "ridge_linear"

    def __init__(self, alpha: float = 1.0):
        self._pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=42))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)
