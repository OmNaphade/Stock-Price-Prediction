"""A second, non-linear model behind the same Predictor interface — this is
the concrete proof that the model layer is Open for extension: nothing in
the backtester, the service, or the UI changes to add it."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingReturnPredictor:
    name = "gradient_boosting"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
    ):
        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
