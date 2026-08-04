"""One row per (ticker, model, day) — the model-health signal the
monitoring page reads. Separate from `track_record/models.py`: that
package answers "was this specific prediction right"; this one answers
"how has this model's backtest performance and drift status looked over
time," independent of any single prediction resolving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelMetricRecord:
    ticker: str
    model_name: str
    log_date: str  # ISO date (YYYY-MM-DD) — one row per ticker/model/day
    logged_at: str  # ISO timestamp of the most recent write to this row
    n_folds: Optional[int]
    n_features: Optional[int]
    model_directional_accuracy: Optional[float]
    baseline_directional_accuracy: Optional[float]
    model_rmse_price: Optional[float]
    baseline_rmse_price: Optional[float]
    has_drift: Optional[bool]
    drifted_feature_count: Optional[int]
