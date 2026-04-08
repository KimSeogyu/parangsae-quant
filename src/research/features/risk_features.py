"""Block 5: Risk / State features (8 features).

PRD Section 5:
  rvol_1h, rvol_1d, downside_vol_1d, corr_btc_1d,
  btc_trend_4h, alt_breadth_4h, median_funding_breadth, xs_corr_1d

Purpose: Gross exposure, veto signals, and regime diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_realized_vol(
    closes: pd.Series,
    horizon_bars: int = 12,
) -> pd.Series:
    """Realized volatility (std of log returns). (rvol_1h/1d)"""
    log_rets = np.log(closes / closes.shift(1))
    return log_rets.rolling(window=horizon_bars, min_periods=horizon_bars).std(ddof=1)


def compute_downside_vol(
    closes: pd.Series,
    horizon_bars: int = 288,
) -> pd.Series:
    """Downside volatility (std of negative log returns only). (downside_vol_1d)"""
    log_rets = np.log(closes / closes.shift(1))
    neg_rets = log_rets.where(log_rets < 0, 0.0)
    return neg_rets.rolling(window=horizon_bars, min_periods=horizon_bars).std(ddof=1)


def compute_corr_btc(
    returns_asset: pd.Series,
    returns_btc: pd.Series,
    horizon_bars: int = 288,
) -> pd.Series:
    """Rolling correlation with BTC. (corr_btc_1d)"""
    return returns_asset.rolling(window=horizon_bars, min_periods=horizon_bars).corr(returns_btc)


def compute_btc_trend(
    btc_closes: pd.Series,
    horizon_bars: int = 48,
) -> pd.Series:
    """BTC price trend (log momentum). (btc_trend_4h)

    Simple log return of BTC over the horizon. Positive = uptrend.
    """
    return np.log(btc_closes / btc_closes.shift(horizon_bars))


def compute_alt_breadth(
    returns_panel: pd.DataFrame,
    horizon_bars: int = 48,
) -> pd.Series:
    """Alt breadth: fraction of alts with positive return. (alt_breadth_4h)

    Computed from a panel of cumulative returns over the horizon.
    Output in [0, 1]. Low breadth during selloffs.
    """
    cum_rets = returns_panel.rolling(window=horizon_bars, min_periods=horizon_bars).sum()
    positive_frac = (cum_rets > 0).mean(axis=1)
    return positive_frac


def compute_median_funding_breadth(
    funding_panel: pd.DataFrame,
) -> pd.Series:
    """Median funding rate across universe. (median_funding_breadth)

    Cross-sectional median of current funding rates.
    """
    return funding_panel.median(axis=1)


def compute_cross_sectional_corr(
    returns_panel: pd.DataFrame,
    horizon_bars: int = 288,
) -> pd.Series:
    """Cross-sectional average pairwise correlation. (xs_corr_1d)

    Computes mean of upper triangle of rolling correlation matrix.
    High values indicate herding behavior.
    """
    result = pd.Series(np.nan, index=returns_panel.index, dtype=float)

    n_cols = returns_panel.shape[1]
    if n_cols < 2:
        return result

    for i in range(horizon_bars, len(returns_panel) + 1):
        window = returns_panel.iloc[i - horizon_bars : i]
        corr_matrix = window.corr()
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        upper = corr_matrix.values[mask]
        upper = upper[~np.isnan(upper)]
        if len(upper) > 0:
            result.iloc[i - 1] = upper.mean()

    return result
