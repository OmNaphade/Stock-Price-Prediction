from __future__ import annotations

import numpy as np
import pytest

from features.pipeline import TARGET_COLUMN, FeaturePipeline
from models.lstm import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def test_lstm_fits_and_predicts_right_shape(synthetic_ohlcv):
    from models.lstm import LSTMReturnPredictor

    features = FeaturePipeline().build(synthetic_ohlcv)
    X = features[FeaturePipeline().feature_columns].to_numpy()
    y = features[TARGET_COLUMN].to_numpy()
    split = int(len(X) * 0.85)

    model = LSTMReturnPredictor(lookback=10, hidden_size=8, epochs=5)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])

    assert preds.shape == (len(X) - split,)
    assert np.all(np.isfinite(preds))


def test_lstm_predict_uses_training_tail_as_context_not_zeros(synthetic_ohlcv):
    """Regression guard for the windowing trick: predicting right after fit
    should not silently pad with zeros for every row — only the model's own
    warm-up, bounded by lookback, should ever need padding."""
    from models.lstm import LSTMReturnPredictor

    features = FeaturePipeline().build(synthetic_ohlcv)
    X = features[FeaturePipeline().feature_columns].to_numpy()
    y = features[TARGET_COLUMN].to_numpy()
    split = int(len(X) * 0.85)

    model = LSTMReturnPredictor(lookback=10, hidden_size=8, epochs=5)
    model.fit(X[:split], y[:split])

    assert model._context is not None
    assert len(model._context) == 9  # lookback - 1


def test_lstm_raises_clear_error_when_torch_unavailable(monkeypatch):
    import models.lstm as lstm_module

    monkeypatch.setattr(lstm_module, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        lstm_module.LSTMReturnPredictor()
