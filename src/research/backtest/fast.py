"""Fast backtest mode utilities.

PRD Section 9: Fast mode
- Touch-based simplified maker fill
- Focus on: rank IC, spread, turnover, beta drift
"""

from __future__ import annotations




def simulate_fast_fills(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    prices: dict[str, float],
    fee_bps: float = 2.0,
) -> tuple[dict[str, float], float]:
    """Simulate instant fills at current prices (fast mode).

    Returns:
        (new_weights, fee_cost_fraction)
    """
    all_syms = set(target_weights) | set(current_weights)
    turnover = 0.5 * sum(
        abs(target_weights.get(s, 0) - current_weights.get(s, 0)) for s in all_syms
    )
    fee_cost = turnover * fee_bps / 10000
    return dict(target_weights), fee_cost


def compute_fast_portfolio_return(
    weights: dict[str, float],
    returns: dict[str, float],
) -> float:
    """Compute single-period portfolio return from weights and asset returns."""
    return sum(weights.get(s, 0) * returns.get(s, 0) for s in set(weights) | set(returns))
