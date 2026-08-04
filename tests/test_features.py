from __future__ import annotations

import numpy as np

from features.pipeline import TARGET_COLUMN, FeaturePipeline


def test_build_has_no_nans_and_expected_columns(synthetic_ohlcv):
    pipeline = FeaturePipeline()
    features = pipeline.build(synthetic_ohlcv)

    assert not features.empty
    assert not features.isna().any().any()
    for col in pipeline.feature_columns:
        assert col in features.columns
    assert "close" in features.columns
    assert TARGET_COLUMN in features.columns


def test_target_is_next_day_log_return(synthetic_ohlcv):
    pipeline = FeaturePipeline()
    features = pipeline.build(synthetic_ohlcv)

    close = synthetic_ohlcv["Close"]
    expected_next_return = np.log(close.shift(-1) / close)

    aligned_expected = expected_next_return.loc[features.index]
    np.testing.assert_allclose(
        features[TARGET_COLUMN].to_numpy(), aligned_expected.to_numpy(), rtol=1e-10
    )


def test_features_are_scale_free_not_raw_price(synthetic_ohlcv):
    """Regression guard: features must never smuggle in the raw price level
    (that's the bug the pipeline replaced — predicting price from price)."""
    pipeline = FeaturePipeline()
    features = pipeline.build(synthetic_ohlcv)

    last_close = features["close"].iloc[-1]
    for col in pipeline.feature_columns:
        # Every engineered feature should be a small ratio/oscillator, not
        # something on the same order of magnitude as the raw price.
        assert features[col].abs().median() < last_close / 2


def test_without_technical_indicators_has_fewer_columns(synthetic_ohlcv):
    full = FeaturePipeline(include_technical_indicators=True)
    minimal = FeaturePipeline(include_technical_indicators=False)

    assert len(minimal.feature_columns) < len(full.feature_columns)
    built = minimal.build(synthetic_ohlcv)
    assert set(minimal.feature_columns).issubset(built.columns)


def test_empty_input_returns_empty_output():
    import pandas as pd

    pipeline = FeaturePipeline()
    result = pipeline.build(pd.DataFrame())
    assert result.empty


def test_live_features_use_the_most_recent_row_that_build_drops(synthetic_ohlcv):
    """build() drops the most recent trading day (its target is unknown —
    nothing to shift(-1) in yet). build_live_features() must return
    exactly that row, not require a target for it."""
    pipeline = FeaturePipeline()
    trained_features = pipeline.build(synthetic_ohlcv)
    live = pipeline.build_live_features(synthetic_ohlcv)

    assert live is not None
    assert live.index[0] == synthetic_ohlcv.index[-1]
    assert live.index[0] > trained_features.index[-1]
    assert live["close"].iloc[0] == synthetic_ohlcv["Close"].iloc[-1]
    for col in pipeline.feature_columns:
        assert col in live.columns
    assert not live[pipeline.feature_columns].isna().any(axis=None)


def test_live_features_none_when_history_too_short_for_warmup(make_ohlcv):
    tiny = make_ohlcv(n=5, seed=1)
    pipeline = FeaturePipeline()
    assert pipeline.build_live_features(tiny) is None


def test_live_features_none_on_empty_input():
    import pandas as pd

    pipeline = FeaturePipeline()
    assert pipeline.build_live_features(pd.DataFrame()) is None
