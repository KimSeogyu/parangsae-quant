"""Block 4: Derivatives / Crowding features (8 features).

PRD Section 5:
  funding, funding_z_7d, funding_change_1d, oi_change_1h, oi_change_1d,
  oi_to_vol, basis, basis_z_7d

Purpose: Capture overheating, carry, and position crowding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_funding_rate(
    funding_rate: pd.Series,
) -> pd.Series:
    """Current funding rate pass-through. (funding)"""
    return funding_rate


def compute_funding_zscore(
    funding_rate: pd.Series,
    horizon_bars: int = 2016,
) -> pd.Series:
    """Funding rate z-score over horizon. (funding_z_7d)"""
    rolling_mean = funding_rate.rolling(window=horizon_bars, min_periods=horizon_bars).mean()
    rolling_std = funding_rate.rolling(window=horizon_bars, min_periods=horizon_bars).std(ddof=1)
    return (funding_rate - rolling_mean) / rolling_std.replace(0, np.nan)


def compute_funding_change(
    funding_rate: pd.Series,
    horizon_bars: int = 288,
) -> pd.Series:
    """Change in funding rate over horizon. (funding_change_1d)"""
    return funding_rate - funding_rate.shift(horizon_bars)


def compute_oi_change(
    open_interest: pd.Series,
    horizon_bars: int = 12,
) -> pd.Series:
    """Percentage change in open interest. (oi_change_1h/1d)

    oi_change = (oi_t - oi_{t-h}) / oi_{t-h}
    """
    return open_interest.pct_change(periods=horizon_bars)


def compute_oi_to_volume(
    open_interest: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """OI-to-volume ratio. (oi_to_vol)

    High ratio = large open positions relative to trading activity.
    """
    return open_interest / volume.replace(0, np.nan)


def compute_basis(
    futures_price: pd.Series,
    spot_price: pd.Series,
) -> pd.Series:
    """Futures-spot basis. (basis)

    basis = (futures - spot) / spot
    Positive = contango (futures premium).
    """
    return (futures_price - spot_price) / spot_price.replace(0, np.nan)


def compute_basis_zscore(
    basis: pd.Series,
    horizon_bars: int = 2016,
) -> pd.Series:
    """Basis z-score over horizon. (basis_z_7d)"""
    rolling_mean = basis.rolling(window=horizon_bars, min_periods=horizon_bars).mean()
    rolling_std = basis.rolling(window=horizon_bars, min_periods=horizon_bars).std(ddof=1)
    return (basis - rolling_mean) / rolling_std.replace(0, np.nan)
