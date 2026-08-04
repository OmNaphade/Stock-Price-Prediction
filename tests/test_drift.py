from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.drift import check_feature_drift


def test_no_drift_when_distribution_is_stable():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"feature_a": rng.normal(0, 1, size=300)})

    report = check_feature_drift(df, ["feature_a"], current_window=60, reference_window=200)

    assert not report.has_drift
    assert report.drifted_features == []


def test_detects_drift_on_a_shifted_window():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=200)
    current = rng.normal(8, 1, size=60)  # a very different mean
    df = pd.DataFrame({"feature_a": np.concatenate([reference, current])})

    report = check_feature_drift(df, ["feature_a"], current_window=60, reference_window=200)

    assert report.has_drift
    assert "feature_a" in report.drifted_features


def test_short_history_returns_empty_report_without_crashing():
    df = pd.DataFrame({"feature_a": np.arange(20)})
    report = check_feature_drift(df, ["feature_a"], current_window=60, reference_window=200)
    assert not report.has_drift
    assert report.features == []
