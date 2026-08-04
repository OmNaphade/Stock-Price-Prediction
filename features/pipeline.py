"""Turns raw OHLCV into a model-ready table.

Two correctness fixes relative to the original app live here:

1. The target is the next-day *log return*, not the next-day raw close.
   Price is trending and non-stationary; a model fit to reproduce it mostly
   just learns "tomorrow ~= today." Log-returns are close to stationary,
   which is what makes them learnable at all.
2. Every feature is scale-free (a ratio, a return, a bounded oscillator) —
   nothing carries the raw price level, so a single fitted model behaves
   consistently whether the ticker trades at $5 or $500, and no separate
   feature-scaling step is required for the linear model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data_access.macro import MacroFeatureSource, NullMacroSource

from .indicators import atr, bollinger_percent_b, macd, rsi

TARGET_COLUMN = "log_return_target"
# The calendar date the target actually refers to (tomorrow, relative to
# the feature row) — carried through so charts can label a prediction by
# the date it's *for*, not the date it was made *from*. Not a model input.
TARGET_DATE_COLUMN = "target_date"


class FeaturePipeline:
    def __init__(
        self,
        include_technical_indicators: bool = True,
        macro_source: Optional[MacroFeatureSource] = None,
    ):
        self.include_technical_indicators = include_technical_indicators
        self.macro_source = macro_source or NullMacroSource()
        self._macro_enabled = not isinstance(self.macro_source, NullMacroSource)

    @property
    def feature_columns(self) -> list[str]:
        cols = ["open_close_pct", "high_low_pct", "ma5_ratio", "ma10_ratio", "ma20_ratio",
                "volatility_10"]
        if self.include_technical_indicators:
            cols += ["rsi_14", "macd_hist", "bollinger_pct_b", "atr_pct"]
        if self._macro_enabled:
            cols += ["macro_rate_chg_5d", "macro_cpi_yoy"]
        return cols

    def build(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame indexed like `ohlcv`, with feature columns,
        a pass-through 'close' column, and TARGET_COLUMN — rows with any
        NaN (warm-up windows, and the final row with no next-day target)
        are dropped. If macro features are enabled but the fetch fails,
        those columns come back all-NaN and get dropped along with every
        row that needs them — an empty/short result surfaces as the
        existing "not enough rows" error upstream, rather than silently
        training on a feature that's actually missing."""
        if ohlcv.empty:
            return pd.DataFrame(
                columns=[*self.feature_columns, "close", TARGET_COLUMN, TARGET_DATE_COLUMN]
            )

        df = ohlcv.copy()
        close = df["Close"]
        log_close = np.log(close)

        out = pd.DataFrame(index=df.index)
        out["close"] = close
        out["open_close_pct"] = (close - df["Open"]) / df["Open"]
        out["high_low_pct"] = (df["High"] - df["Low"]) / close
        out["ma5_ratio"] = close / close.rolling(5).mean() - 1
        out["ma10_ratio"] = close / close.rolling(10).mean() - 1
        out["ma20_ratio"] = close / close.rolling(20).mean() - 1
        out["volatility_10"] = log_close.diff().rolling(10).std()

        if self.include_technical_indicators:
            out["rsi_14"] = rsi(close)
            _, _, hist = macd(close)
            out["macd_hist"] = hist / close
            out["bollinger_pct_b"] = bollinger_percent_b(close)
            if {"High", "Low"}.issubset(df.columns):
                out["atr_pct"] = atr(df) / close
            else:
                out["atr_pct"] = np.nan

        if self._macro_enabled:
            macro_df = self.macro_source.get_series(
                str(df.index.min().date()), str(df.index.max().date())
            )
            aligned = (
                macro_df.reindex(out.index, method="ffill")
                if not macro_df.empty
                else pd.DataFrame(index=out.index)
            )
            out["macro_rate_chg_5d"] = (
                aligned["treasury_10y"].diff(5) if "treasury_10y" in aligned else np.nan
            )
            out["macro_cpi_yoy"] = (
                aligned["cpi"].pct_change(252) if "cpi" in aligned else np.nan
            )

        out[TARGET_COLUMN] = log_close.shift(-1) - log_close
        # Computed from out's own (still-complete) index before dropna, so
        # the last surviving row still correctly gets the next calendar
        # date even though that date's own row gets dropped below (it has
        # no target of its own — nothing after it to shift in).
        out[TARGET_DATE_COLUMN] = out.index.to_series().shift(-1)

        ordered = [*self.feature_columns, "close", TARGET_COLUMN, TARGET_DATE_COLUMN]
        return out[ordered].dropna()
