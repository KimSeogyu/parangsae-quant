"""Block 1: Relative / Beta features (8 features).

PRD Section 5:
  beta_btc_1w, beta_btc_2w, resid_mom_1h, resid_mom_4h, resid_mom_1d,
  resid_z_1h, resid_z_4h, resid_rsi_14

Purpose: Measure relative strength after removing BTC common factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.labels.beta import compute_rolling_beta


def compute_beta_btc(
    returns_asset: pd.Series,
    returns_btc: pd.Series,
    window_weeks: int = 1,
    bars_per_week: int = 2016,
) -> pd.Series:
    """Rolling OLS beta of asset vs BTC. (beta_btc_1w, beta_btc_2w)"""
    return compute_rolling_beta(returns_asset, returns_btc, window_weeks * bars_per_week)


def compute_resid_momentum(
    returns_asset: pd.Series,
    returns_btc: pd.Series,
    beta: pd.Series,
    horizon_bars: int = 12,
) -> pd.Series:
    """Cumulative residual return over horizon. (resid_mom_1h/4h/1d)

    resid_mom = sum(r_asset - beta * r_btc) over trailing horizon_bars.
    """
    residuals = returns_asset - beta * returns_btc
    return residuals.rolling(window=horizon_bars, min_periods=horizon_bars).sum()


def compute_resid_zscore(
    returns_asset: pd.Series,
    returns_btc: pd.Series,
    beta: pd.Series,
    horizon_bars: int = 12,
) -> pd.Series:
    """Z-score of residual returns over horizon. (resid_z_1h/4h)

    z = mean(resid) / std(resid) over trailing horizon_bars.
    """
    residuals = returns_asset - beta * returns_btc
    rolling_mean = residuals.rolling(window=horizon_bars, min_periods=horizon_bars).mean()
    rolling_std = residuals.rolling(window=horizon_bars, min_periods=horizon_bars).std(ddof=1)
    return rolling_mean / rolling_std.replace(0, np.nan)


def compute_resid_rsi(
    returns_asset: pd.Series,
    returns_btc: pd.Series,
    beta: pd.Series,
    period: int = 14,
) -> pd.Series:
    """RSI of residual returns. (resid_rsi_14)

    Standard RSI formula applied to residual returns instead of prices.
    Output in [0, 100].
    """
    residuals = returns_asset - beta * returns_btc
    gains = residuals.clip(lower=0)
    losses = (-residuals).clip(lower=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
