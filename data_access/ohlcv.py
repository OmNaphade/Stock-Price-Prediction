"""OHLCV cleanup shared by every MarketDataSource, pulled out of sources.py
so a new provider (e.g. openalgo_source.py) can reuse it without importing
sources.py itself — sources.py is where they all get composed together,
so depending on it from a single provider would be a circular import."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in OHLCV_COLUMNS if c in df.columns]
    out = df[cols].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Close"])
