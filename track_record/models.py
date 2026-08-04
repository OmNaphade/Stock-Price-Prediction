"""A prediction made at one point in time, checked against what actually
happened once its target date has passed. This is the data the whole
track-record feature exists to produce: not "the model claims X," but "the
model claimed X on this date, and here's what was actually true."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictionRecord:
    ticker: str
    model_name: str
    made_at: str  # ISO timestamp
    target_date: str  # ISO date (YYYY-MM-DD) — the date predicted_close is for
    last_close: float  # the known close the prediction was based on
    predicted_close: float
    predicted_log_return: float
    actual_close: Optional[float] = None
    resolved_at: Optional[str] = None

    @property
    def is_resolved(self) -> bool:
        return self.actual_close is not None

    @property
    def predicted_direction_up(self) -> bool:
        return self.predicted_close >= self.last_close

    @property
    def direction_correct(self) -> Optional[bool]:
        if self.actual_close is None:
            return None
        actual_direction_up = self.actual_close >= self.last_close
        return self.predicted_direction_up == actual_direction_up

    @property
    def abs_pct_error(self) -> Optional[float]:
        if self.actual_close is None or self.actual_close == 0:
            return None
        return abs(self.predicted_close - self.actual_close) / self.actual_close * 100
