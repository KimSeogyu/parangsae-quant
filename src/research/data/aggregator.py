"""Bar aggregation utilities for the research pipeline.

Aggregates 1-minute OHLCV bars into 5-minute (research) and 20-minute (decision) bars.
Also provides mid-price computation for label generation.

PRD Section 3: "5-min research bars and 20-min decision bars aggregated from 1m"
"""

from __future__ import annotations

import pandas as pd


def aggregate_ohlcv(df_1m: pd.DataFrame, freq_minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute OHLCV bars to a coarser frequency.

    Args:
        df_1m: DataFrame with OHLCV columns indexed by datetime.
        freq_minutes: Target bar frequency in minutes (e.g. 5, 20, 60).

    Returns:
        Aggregated DataFrame with same columns. Rows with no data are dropped.
    """
    rule = f"{freq_minutes}min"
    agg = df_1m.resample(rule, closed="left", label="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
        }
    )
    return agg.dropna(subset=["open"])


def compute_mid_price(df: pd.DataFrame) -> pd.Series:
    """Compute mid price from bid/ask or from OHLCV high/low.

    If 'best_bid' and 'best_ask' columns exist, uses (bid + ask) / 2.
    Otherwise falls back to (high + low) / 2 as an approximation.
    """
    if "best_bid" in df.columns and "best_ask" in df.columns:
        return (df["best_bid"] + df["best_ask"]) / 2
    return (df["high"] + df["low"]) / 2


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP = cumulative(quote_volume) / cumulative(volume).

    Works on any bar frequency. Returns NaN where volume is zero.
    """
    if "quote_volume" not in df.columns:
        qvol = df["close"] * df["volume"]
    else:
        qvol = df["quote_volume"]

    cum_qvol = qvol.cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_qvol / cum_vol.replace(0, float("nan"))


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns: ln(p_t / p_{t-1})."""
    import numpy as np

    return np.log(prices / prices.shift(1))
