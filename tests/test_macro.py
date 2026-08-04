from __future__ import annotations

import pandas as pd

from data_access.macro import FredMacroSource, NullMacroSource
from features.pipeline import FeaturePipeline


def test_null_macro_source_returns_empty():
    source = NullMacroSource()
    assert source.get_series("2020-01-01", "2020-12-31").empty


def test_fred_source_returns_empty_without_api_key():
    source = FredMacroSource(api_key="")
    assert source.get_series("2020-01-01", "2020-12-31").empty


class _FakeMacroSource:
    """Satisfies MacroFeatureSource without touching the real FRED API —
    this is the seam FeaturePipeline actually depends on."""

    def get_series(self, start, end):
        dates = pd.bdate_range(start, end)
        return pd.DataFrame(
            {"treasury_10y": [4.0 + 0.001 * i for i in range(len(dates))],
             "cpi": [300.0 + 0.01 * i for i in range(len(dates))]},
            index=dates,
        )


def test_pipeline_includes_macro_columns_when_source_configured(synthetic_ohlcv):
    pipeline = FeaturePipeline(macro_source=_FakeMacroSource())
    features = pipeline.build(synthetic_ohlcv)

    assert "macro_rate_chg_5d" in pipeline.feature_columns
    assert "macro_cpi_yoy" in pipeline.feature_columns
    assert not features.empty
    assert not features[["macro_rate_chg_5d", "macro_cpi_yoy"]].isna().any().any()


def test_pipeline_omits_macro_columns_by_default(synthetic_ohlcv):
    pipeline = FeaturePipeline()
    assert "macro_rate_chg_5d" not in pipeline.feature_columns
    features = pipeline.build(synthetic_ohlcv)
    assert "macro_rate_chg_5d" not in features.columns
