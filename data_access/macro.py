"""Macro/exogenous data — interest rates, inflation — as an optional input
to FeaturePipeline. Behind a Protocol so the pipeline never imports FRED
directly (Dependency Inversion), with a Null Object default so the
pipeline behaves identically whether or not macro data is configured
(Open/Closed: turning this on is a constructor argument, not a code path
that has to be threaded through the pipeline)."""

from __future__ import annotations

from typing import Optional, Protocol

import pandas as pd

from config import log


class MacroFeatureSource(Protocol):
    def get_series(self, start: str, end: str) -> pd.DataFrame:
        """A daily-ish DatetimeIndex DataFrame with columns 'treasury_10y'
        and 'cpi' (whatever subset is available), or empty if unavailable."""
        ...


class NullMacroSource:
    """Default source: no macro data. Exists so FeaturePipeline can always
    hold a MacroFeatureSource rather than an Optional it has to branch on."""

    def get_series(self, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()


class FredMacroSource:
    """Federal Reserve Economic Data (fred.stlouisfed.org) — free, no rate
    limit worth worrying about for this use case, but requires a free API
    key. DGS10 = 10-year Treasury yield, CPIAUCSL = CPI index."""

    def __init__(self, api_key: str, series_ids: Optional[dict[str, str]] = None):
        self._api_key = api_key
        self._series_ids = series_ids or {"treasury_10y": "DGS10", "cpi": "CPIAUCSL"}

    def get_series(self, start: str, end: str) -> pd.DataFrame:
        if not self._api_key:
            return pd.DataFrame()
        try:
            from fredapi import Fred
        except ImportError:
            log.warning("fredapi is not installed; skipping macro features.")
            return pd.DataFrame()

        try:
            client = Fred(api_key=self._api_key)
            columns = {
                name: client.get_series(series_id, observation_start=start, observation_end=end)
                for name, series_id in self._series_ids.items()
            }
            df = pd.DataFrame(columns)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        except Exception:
            log.warning("FRED fetch failed", exc_info=True)
            return pd.DataFrame()
