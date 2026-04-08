"""Research backtest report generation.

PRD Section 9 & 10: Report requirements
- Walk-forward results summary
- Fast vs Accurate mode comparison
- Regime-segmented performance
- Go/No-Go criteria evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.research.analytics.metrics import (
    classify_regime,
    compute_beta_drift,
    compute_max_drawdown,
    compute_regime_performance,
    compute_sharpe,
    compute_turnover,
)


@dataclass
class BacktestReport:
    """Complete backtest report."""

    # Overall metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_turnover: float = 0.0
    mean_beta: float = 0.0

    # Alpha metrics
    mean_rank_ic: float = 0.0
    pct_positive_ic: float = 0.0

    # Execution metrics
    maker_fill_ratio: float = 0.0
    mean_slippage_bps: float = 0.0

    # Cost metrics
    total_fees: float = 0.0
    net_sharpe_after_costs: float = 0.0

    # Regime performance
    regime_performance: dict = field(default_factory=dict)

    # Baseline comparison
    ridge_vs_zero_wins: float = 0.0
    ridge_vs_simple_wins: float = 0.0

    # Go/No-Go
    go_decisions: dict = field(default_factory=dict)


def generate_report(
    portfolio_returns: pd.Series,
    btc_returns: pd.Series,
    weights_history: list[dict[str, float]],
    walk_forward_summary: dict,
    maker_fill_ratio: float = 0.0,
    mean_slippage_bps: float = 0.0,
    total_fees: float = 0.0,
) -> BacktestReport:
    """Generate complete backtest report."""
    report = BacktestReport()

    # Overall performance
    equity = (1 + portfolio_returns).cumprod()
    report.total_return = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0.0
    n_years = len(portfolio_returns) / (365 * 24 * 3)  # 20-min bars
    if n_years > 0 and report.total_return > -1:
        report.annualized_return = (1 + report.total_return) ** (1 / n_years) - 1
    report.sharpe_ratio = compute_sharpe(portfolio_returns)
    report.max_drawdown = compute_max_drawdown(equity)

    # Turnover
    if weights_history:
        turnovers = compute_turnover(weights_history)
        report.avg_turnover = float(turnovers.mean()) if len(turnovers) > 0 else 0.0

    # Beta drift
    beta_drift = compute_beta_drift(portfolio_returns, btc_returns)
    report.mean_beta = float(beta_drift.mean()) if len(beta_drift) > 0 else 0.0

    # Walk-forward metrics
    report.mean_rank_ic = walk_forward_summary.get("mean_test_ic", 0.0)
    report.pct_positive_ic = walk_forward_summary.get("pct_positive_ic", 0.0)

    # Execution
    report.maker_fill_ratio = maker_fill_ratio
    report.mean_slippage_bps = mean_slippage_bps
    report.total_fees = total_fees

    # Cost-adjusted Sharpe
    if total_fees > 0 and len(portfolio_returns) > 0:
        cost_per_bar = total_fees / len(portfolio_returns)
        cost_adjusted = portfolio_returns - cost_per_bar
        report.net_sharpe_after_costs = compute_sharpe(cost_adjusted)
    else:
        report.net_sharpe_after_costs = report.sharpe_ratio

    # Regime analysis
    regimes = classify_regime(btc_returns)
    report.regime_performance = compute_regime_performance(portfolio_returns, regimes)

    # Baseline comparison
    report.ridge_vs_zero_wins = walk_forward_summary.get("ridge_vs_zero_wins", 0.0)
    report.ridge_vs_simple_wins = walk_forward_summary.get("ridge_vs_simple_rule_wins", 0.0)

    # Go/No-Go evaluation
    report.go_decisions = evaluate_go_no_go(report)

    return report


def evaluate_go_no_go(report: BacktestReport) -> dict[str, dict]:
    """Evaluate PRD Section 10 Go/No-Go criteria."""
    decisions = {}

    # Alpha: Ridge OOS rank IC > 0
    decisions["alpha_ic"] = {
        "criterion": "Ridge OOS rank IC > 0",
        "value": report.mean_rank_ic,
        "pass": report.mean_rank_ic > 0,
        "required": True,
    }

    # Performance: Sharpe >= 0.8
    decisions["sharpe"] = {
        "criterion": "Net long-short Sharpe >= 0.8",
        "value": report.net_sharpe_after_costs,
        "pass": report.net_sharpe_after_costs >= 0.8,
        "required": False,  # Recommended
    }

    # Risk: BTC beta cap
    decisions["beta_cap"] = {
        "criterion": "abs(BTC beta) <= 0.10",
        "value": abs(report.mean_beta),
        "pass": abs(report.mean_beta) <= 0.10,
        "required": True,
    }

    # Execution: maker fill ratio
    decisions["maker_fill"] = {
        "criterion": "Maker fill ratio >= 60%",
        "value": report.maker_fill_ratio,
        "pass": report.maker_fill_ratio >= 0.60,
        "required": False,  # Recommended
    }

    # Stability: positive PnL in 2/3 regime segments
    regime_positive = sum(
        1
        for r in report.regime_performance.values()
        if r.get("cumulative_return", 0) > 0
    )
    decisions["stability"] = {
        "criterion": "Net PnL positive in >= 2/3 regime segments",
        "value": regime_positive,
        "pass": regime_positive >= 2,
        "required": True,
    }

    return decisions
