from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int, seed: int, start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    log_returns = rng.normal(loc=0.0003, scale=0.015, size=n)
    close = start_price * np.exp(np.cumsum(log_returns))
    open_ = close * (1 + rng.normal(0, 0.003, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, size=n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """~2 years of plausible daily OHLCV data, deterministic (fixed seed)."""
    return _make_ohlcv(n=500, seed=42)


@pytest.fixture
def make_ohlcv():
    """Factory for tests that need a specific length or seed."""
    return _make_ohlcv
