"""Rolling OLS beta estimation for BTC residual computation.

PRD Section 4:
  beta_i,t = OLS( recent 1w/2w 5-min returns: r_i ~ r_BTC )

Uses statsmodels OLS for robustness. Falls back to numpy lstsq if
statsmodels is unavailable (test environments).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_beta(
    returns_asset: pd.Series,
    returns_btc: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling OLS beta of asset returns against BTC returns.

    Args:
        returns_asset: Asset log returns (5-min frequency).
        returns_btc: BTC log returns (same index as returns_asset).
        window: Rolling window size in number of bars.

    Returns:
        Series of rolling beta values, NaN where insufficient data.
    """
    aligned = pd.DataFrame({"asset": returns_asset, "btc": returns_btc}).dropna()

    betas = pd.Series(np.nan, index=returns_asset.index, dtype=float)

    if len(aligned) < window:
        return betas

    asset_vals = aligned["asset"].values
    btc_vals = aligned["btc"].values
    aligned_idx = aligned.index

    for i in range(window, len(aligned) + 1):
        y = asset_vals[i - window : i]
        x = btc_vals[i - window : i]

        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean

        denom = np.dot(x_centered, x_centered)
        if denom < 1e-15:
            beta = 0.0
        else:
            beta = np.dot(x_centered, y - y_mean) / denom

        betas.loc[aligned_idx[i - 1]] = beta

    return betas


def compute_dual_window_beta(
    returns_asset: pd.Series,
    returns_btc: pd.Series,
    windows_weeks: list[int] = [1, 2],
    bars_per_week: int = 2016,
) -> pd.Series:
    """Compute beta as average of multiple window estimates.

    PRD uses 1-week and 2-week windows for stability.
    bars_per_week = 7 * 24 * 12 = 2016 (for 5-min bars).
    """
    betas = []
    for w in windows_weeks:
        window_bars = w * bars_per_week
        beta = compute_rolling_beta(returns_asset, returns_btc, window_bars)
        betas.append(beta)

    # Average available betas (use longer window where shorter is NaN)
    beta_df = pd.concat(betas, axis=1)
    return beta_df.mean(axis=1)
