"""The naive baseline: predict zero log-return, i.e. "tomorrow's close
equals today's close." On daily-close data this is a genuinely hard
baseline to beat — every other model's metrics are only meaningful when
reported next to this one."""

from __future__ import annotations

import numpy as np


class NaivePredictor:
    name = "naive_persistence"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float64)
