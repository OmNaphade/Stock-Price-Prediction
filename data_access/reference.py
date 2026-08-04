"""Static reference data: the equity list and Yahoo's period/interval matrix.

Kept separate from `sources.py` on purpose — this isn't "get me price data,"
it's lookup tables the UI needs to build its selectors (Interface
Segregation: callers that only need the stock list shouldn't have to know
about MarketDataSource at all).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_equity_list() -> dict[str, str]:
    """Security Code -> Issuer Name, sourced from data/equity_issuers.csv."""
    candidates = [
        Path.cwd() / "equity_issuers.csv",
        Path.cwd() / "data" / "equity_issuers.csv",
        Path(__file__).parent.parent / "equity_issuers.csv",
        Path(__file__).parent.parent / "data" / "equity_issuers.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "equity_issuers.csv not found. Expected it under data/ alongside the app."
        )
    df = pd.read_csv(csv_path, usecols=["Security Code", "Issuer Name"])
    return dict(zip(df["Security Code"], df["Issuer Name"]))


def periods_and_intervals() -> dict[str, list[str]]:
    """Valid yfinance interval choices for each supported period."""
    return {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }
