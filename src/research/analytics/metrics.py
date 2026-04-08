"""Performance metrics and analytics for the research pipeline.

PRD Section 10: Success Criteria
- Rank IC
- Net long-short Sharpe (cost-adjusted)
- BTC beta drift
- Turnover
- Maker fill ratio
- Regime-segmented performance
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sharpe(returns: pd.Series, annualization: float = np.sqrt(365 * 24 * 3)) -> float:
    """Annualized Sharpe ratio.

    Default annualization for 20-min bars: sqrt(365 * 24 * 3) = sqrt(26280).
    """
    if len(returns) < 2 or returns.std() < 1e-15:
        return 0.0
    return float(returns.mean() / returns.std() * annualization)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a positive fraction."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(-dd.min()) if len(dd) > 0 else 0.0


def compute_turnover(
    weights_history: list[dict[str, float]],
) -> pd.Series:
    """Compute per-period turnover from weight history.

    turnover_t = 0.5 * sum(|w_t - w_{t-1}|)
    """
    turnovers = []
    for i in range(1, len(weights_history)):
        all_syms = set(weights_history[i].keys()) | set(weights_history[i - 1].keys())
        t = 0.5 * sum(
            abs(weights_history[i].get(s, 0) - weights_history[i - 1].get(s, 0))
            for s in all_syms
        )
        turnovers.append(t)
    return pd.Series(turnovers)


def compute_beta_drift(
    portfolio_returns: pd.Series,
    btc_returns: pd.Series,
    window: int = 288,
) -> pd.Series:
    """Rolling beta of portfolio vs BTC returns."""
    aligned = pd.DataFrame({"port": portfolio_returns, "btc": btc_returns}).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    betas = []
    for i in range(window, len(aligned) + 1):
        y = aligned["port"].values[i - window : i]
        x = aligned["btc"].values[i - window : i]
        x_c = x - x.mean()
        denom = np.dot(x_c, x_c)
        if denom < 1e-15:
            betas.append(0.0)
        else:
            betas.append(np.dot(x_c, y - y.mean()) / denom)

    return pd.Series(betas, index=aligned.index[window - 1 :])


def classify_regime(
    btc_returns: pd.Series,
    window: int = 288 * 7,
) -> pd.Series:
    """Classify market regime into bull/bear/chop.

    Uses rolling 7-day BTC cumulative return and volatility.
    """
    cum_ret = btc_returns.rolling(window=window, min_periods=window).sum()
    vol = btc_returns.rolling(window=window, min_periods=window).std()

    regime = pd.Series("chop", index=btc_returns.index)
    regime[cum_ret > vol] = "bull"
    regime[cum_ret < -vol] = "bear"
    return regime


def compute_regime_performance(
    returns: pd.Series,
    regimes: pd.Series,
) -> dict[str, dict]:
    """Compute performance metrics per regime segment."""
    results = {}
    for regime in ["bull", "bear", "chop"]:
        mask = regimes == regime
        if mask.sum() < 10:
            results[regime] = {"count": int(mask.sum()), "sharpe": 0.0, "mean_return": 0.0}
            continue
        seg = returns[mask]
        results[regime] = {
            "count": int(mask.sum()),
            "sharpe": compute_sharpe(seg),
            "mean_return": float(seg.mean()),
            "cumulative_return": float((1 + seg).prod() - 1),
            "max_drawdown": compute_max_drawdown((1 + seg).cumprod()),
        }
    return results
