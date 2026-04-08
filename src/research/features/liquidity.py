"""Block 3: Liquidity / Execution features (8 features).

PRD Section 5:
  qvol_z_1d, qvol_z_7d, amihud_1d, vwap_dev_1h,
  spread_bps, spread_z_1d, depth_top1_usd, ob_imbalance_1m

Purpose: Reflect execution feasibility and information content of volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_qvol_zscore(
    quote_volume: pd.Series,
    horizon_bars: int = 288,
) -> pd.Series:
    """Quote volume z-score over horizon. (qvol_z_1d/7d)

    z = (qvol - mean) / std over trailing horizon.
    """
    rolling_mean = quote_volume.rolling(window=horizon_bars, min_periods=horizon_bars).mean()
    rolling_std = quote_volume.rolling(window=horizon_bars, min_periods=horizon_bars).std(ddof=1)
    return (quote_volume - rolling_mean) / rolling_std.replace(0, np.nan)


def compute_amihud(
    closes: pd.Series,
    volume: pd.Series,
    horizon_bars: int = 288,
) -> pd.Series:
    """Amihud illiquidity ratio. (amihud_1d)

    amihud = mean(|return| / dollar_volume) over horizon.
    Higher = more illiquid.
    """
    abs_ret = np.log(closes / closes.shift(1)).abs()
    dollar_vol = closes * volume
    ratio = abs_ret / dollar_vol.replace(0, np.nan)
    return ratio.rolling(window=horizon_bars, min_periods=horizon_bars).mean()


def compute_vwap_deviation(
    closes: pd.Series,
    quote_volume: pd.Series,
    volume: pd.Series,
    horizon_bars: int = 12,
) -> pd.Series:
    """Price deviation from VWAP. (vwap_dev_1h)

    vwap_dev = (close - vwap) / vwap over rolling horizon.
    Positive = trading above VWAP (demand > supply).
    """
    cum_qvol = quote_volume.rolling(window=horizon_bars, min_periods=1).sum()
    cum_vol = volume.rolling(window=horizon_bars, min_periods=1).sum()
    vwap = cum_qvol / cum_vol.replace(0, np.nan)
    return (closes - vwap) / vwap.replace(0, np.nan)


def compute_spread_bps(
    best_bid: pd.Series,
    best_ask: pd.Series,
) -> pd.Series:
    """Current spread in basis points. (spread_bps)

    spread_bps = (ask - bid) / mid * 10000
    """
    mid = (best_bid + best_ask) / 2
    return (best_ask - best_bid) / mid.replace(0, np.nan) * 10000


def compute_spread_zscore(
    spread: pd.Series,
    horizon_bars: int = 288,
) -> pd.Series:
    """Spread z-score over horizon. (spread_z_1d)"""
    rolling_mean = spread.rolling(window=horizon_bars, min_periods=horizon_bars).mean()
    rolling_std = spread.rolling(window=horizon_bars, min_periods=horizon_bars).std(ddof=1)
    return (spread - rolling_mean) / rolling_std.replace(0, np.nan)


def compute_depth_top1(
    best_bid_size: pd.Series,
    best_ask_size: pd.Series,
    mid_price: pd.Series,
) -> pd.Series:
    """Top-of-book depth in USD. (depth_top1_usd)

    depth = (bid_size + ask_size) * mid_price / 2
    """
    return (best_bid_size + best_ask_size) * mid_price / 2


def compute_ob_imbalance(
    best_bid_size: pd.Series,
    best_ask_size: pd.Series,
) -> pd.Series:
    """Order book bid/ask imbalance. (ob_imbalance_1m)

    imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    Output in [-1, 1]. Positive = more bids.
    """
    total = best_bid_size + best_ask_size
    return (best_bid_size - best_ask_size) / total.replace(0, np.nan)
