"""Block 2: Trend / Path features (8 features).

PRD Section 5:
  mom_1h, mom_4h, mom_1d, mom_4h_voladj, mom_1d_voladj,
  fip_like_1d, trend_linearity_1d, wickiness_1d

Purpose: Separate good trends from jump-driven trends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum(
    closes: pd.Series,
    horizon_bars: int = 12,
) -> pd.Series:
    """Raw log momentum. (mom_1h/4h/1d)

    mom = log(close_t / close_{t-horizon})
    """
    return np.log(closes / closes.shift(horizon_bars))


def compute_momentum_voladj(
    closes: pd.Series,
    horizon_bars: int = 48,
    vol_window: int | None = None,
) -> pd.Series:
    """Volatility-adjusted momentum. (mom_4h_voladj/1d_voladj)

    voladj_mom = mom / realized_vol
    vol_window defaults to 2x horizon if not specified.
    """
    if vol_window is None:
        vol_window = horizon_bars * 2

    mom = compute_momentum(closes, horizon_bars)
    log_rets = np.log(closes / closes.shift(1))
    rvol = log_rets.rolling(window=vol_window, min_periods=vol_window).std(ddof=1)
    return mom / rvol.replace(0, np.nan)


def compute_fip_like(
    closes: pd.Series,
    horizon_bars: int = 288,
) -> pd.Series:
    """Fraction of Interval Positive (FIP-like). (fip_like_1d)

    Fraction of bars in the trailing horizon where return was positive.
    Output in [0, 1]. Higher = smoother uptrend.
    """
    log_rets = np.log(closes / closes.shift(1))
    positive = (log_rets > 0).astype(float)
    return positive.rolling(window=horizon_bars, min_periods=horizon_bars).mean()


def compute_trend_linearity(
    closes: pd.Series,
    horizon_bars: int = 288,
) -> pd.Series:
    """R-squared of log-price vs time. (trend_linearity_1d)

    High R² = linear trend. Low R² = choppy or mean-reverting.
    Output in [0, 1].
    """
    log_prices = np.log(closes)
    result = pd.Series(np.nan, index=closes.index, dtype=float)

    values = log_prices.values
    for i in range(horizon_bars, len(values) + 1):
        window = values[i - horizon_bars : i]
        if np.any(np.isnan(window)):
            continue
        x = np.arange(horizon_bars, dtype=float)
        x_mean = x.mean()
        y_mean = window.mean()
        ss_xy = np.dot(x - x_mean, window - y_mean)
        ss_xx = np.dot(x - x_mean, x - x_mean)
        ss_yy = np.dot(window - y_mean, window - y_mean)
        if ss_xx < 1e-15 or ss_yy < 1e-15:
            result.iloc[i - 1] = 0.0
        else:
            r = ss_xy / np.sqrt(ss_xx * ss_yy)
            result.iloc[i - 1] = r * r

    return result


def compute_wickiness(
    df: pd.DataFrame,
    horizon_bars: int = 288,
) -> pd.Series:
    """Average wick ratio over trailing horizon. (wickiness_1d)

    wick_ratio = (high - low - abs(close - open)) / (high - low)
    High wickiness = large shadows relative to body = rejection / indecision.
    Output in [0, 1].
    """
    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]
    wick_ratio = (total_range - body) / total_range.replace(0, np.nan)
    wick_ratio = wick_ratio.clip(0, 1)
    return wick_ratio.rolling(window=horizon_bars, min_periods=horizon_bars).mean()
