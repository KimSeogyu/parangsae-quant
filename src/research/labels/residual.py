"""Residual return label computation.

PRD Section 4:
  r_i(t,h) = log(Mid_i(t+h) / Mid_i(t))
  y20_i,t = r_i(t,20m) - beta_i,t * r_BTC(t,20m)
  y60_i,t = r_i(t,60m) - beta_i,t * r_BTC(t,60m)
  score_i,t = 0.7 * ŷ20_i,t + 0.3 * ŷ60_i,t
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forward_log_return(
    mid_prices: pd.Series,
    horizon_bars: int,
) -> pd.Series:
    """Compute forward log return: log(mid(t+h) / mid(t)).

    Args:
        mid_prices: Mid price series.
        horizon_bars: Number of bars to look forward.

    Returns:
        Forward log return series. Last `horizon_bars` values are NaN.
    """
    future = mid_prices.shift(-horizon_bars)
    return np.log(future / mid_prices)


def compute_residual_return(
    forward_return_asset: pd.Series,
    forward_return_btc: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """Compute residual return: r_asset - beta * r_BTC.

    Args:
        forward_return_asset: Forward log return of the asset.
        forward_return_btc: Forward log return of BTC (same horizon).
        beta: Rolling beta estimate at each time step.

    Returns:
        Residual return series.
    """
    return forward_return_asset - beta * forward_return_btc


def build_labels(
    mid_prices: dict[str, pd.Series],
    mid_btc: pd.Series,
    betas: dict[str, pd.Series],
    primary_horizon_bars: int = 4,
    secondary_horizon_bars: int = 12,
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
) -> dict[str, pd.DataFrame]:
    """Build residual return labels for all symbols.

    Args:
        mid_prices: {symbol: mid_price_series} at 5-min frequency.
        mid_btc: BTC mid price series at 5-min frequency.
        betas: {symbol: beta_series} at 5-min frequency.
        primary_horizon_bars: 20min / 5min = 4 bars.
        secondary_horizon_bars: 60min / 5min = 12 bars.
        primary_weight: Weight for y20 in combined score.
        secondary_weight: Weight for y60 in combined score.

    Returns:
        {symbol: DataFrame with columns [y20, y60]} indexed by timestamp.
    """
    btc_fwd_20 = compute_forward_log_return(mid_btc, primary_horizon_bars)
    btc_fwd_60 = compute_forward_log_return(mid_btc, secondary_horizon_bars)

    labels = {}
    for sym, mid in mid_prices.items():
        asset_fwd_20 = compute_forward_log_return(mid, primary_horizon_bars)
        asset_fwd_60 = compute_forward_log_return(mid, secondary_horizon_bars)

        beta = betas.get(sym)
        if beta is None:
            continue

        y20 = compute_residual_return(asset_fwd_20, btc_fwd_20, beta)
        y60 = compute_residual_return(asset_fwd_60, btc_fwd_60, beta)

        labels[sym] = pd.DataFrame({"y20": y20, "y60": y60}, index=mid.index)

    return labels


def combine_predictions(
    pred_y20: pd.Series,
    pred_y60: pd.Series,
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
) -> pd.Series:
    """Combine y20 and y60 predictions into final score.

    score = primary_weight * ŷ20 + secondary_weight * ŷ60
    """
    return primary_weight * pred_y20 + secondary_weight * pred_y60
