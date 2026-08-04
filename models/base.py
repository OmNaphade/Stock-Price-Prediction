"""The interface every tabular return-predictor implements. The backtester
and the service layer depend only on this — never on Ridge, GradientBoosting,
or any other concrete model class (Dependency Inversion). Adding a model
means adding a class that satisfies this Protocol (Open/Closed)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Predictor(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
