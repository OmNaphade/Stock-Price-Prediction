"""Feature drift detection: is the most recent window of a ticker's
features still drawn from the same distribution the model saw over the
rest of its history? If not, backtest metrics computed on older data don't
say much about how the model behaves right now.

Built on `scipy.stats.ks_2samp` (already a transitive dependency via
statsmodels/sklearn, so this needs nothing new) rather than a full
Evidently integration. `DriftReport` is intentionally shaped so a fuller
Evidently-backed implementation could satisfy the same interface later
without changing any caller — this module is the seam, not a life
sentence to the KS-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class FeatureDrift:
    feature: str
    p_value: float
    drifted: bool


@dataclass
class DriftReport:
    reference_window: int
    current_window: int
    features: list[FeatureDrift] = field(default_factory=list)

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]

    @property
    def has_drift(self) -> bool:
        return len(self.drifted_features) > 0


def check_feature_drift(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    current_window: int = 60,
    reference_window: int = 200,
    p_value_threshold: float = 0.05,
) -> DriftReport:
    """Compares the most recent `current_window` rows against the
    `reference_window` rows immediately before them, per feature, with a
    two-sample Kolmogorov-Smirnov test. A low p-value means the two
    windows likely come from different distributions."""
    n = len(features_df)
    current = features_df.iloc[-current_window:]
    reference = features_df.iloc[max(0, n - current_window - reference_window): n - current_window]

    report = DriftReport(reference_window=len(reference), current_window=len(current))
    if len(reference) < 30 or len(current) < 10:
        return report

    for column in feature_columns:
        stat_result = ks_2samp(reference[column].to_numpy(), current[column].to_numpy())
        p_value = float(stat_result.pvalue)
        report.features.append(
            FeatureDrift(feature=column, p_value=p_value, drifted=p_value < p_value_threshold)
        )

    return report
