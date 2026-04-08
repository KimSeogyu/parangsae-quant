"""Accurate backtest mode utilities.

PRD Section 9: Accurate mode
- 1-minute L1 snapshot-based passive fill proxy
- Tracks: maker fill ratio, realized slippage, post-fill failure rate
"""

from __future__ import annotations

import pandas as pd

from src.research.execution.passive import (
    OrderState,
    estimate_slippage_bps,
    simulate_passive_fill,
)


def simulate_accurate_period(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    bar_data: pd.DataFrame,
    tick_sizes: dict[str, float] | None = None,
    maker_fee_bps: float = 2.0,
    taker_fee_bps: float = 5.0,
) -> tuple[dict[str, float], list[OrderState], float]:
    """Simulate a rebalancing period with passive fill logic.

    Args:
        target_weights: Target portfolio weights.
        current_weights: Current portfolio weights.
        bar_data: DataFrame with columns for each symbol's OHLC data.
        tick_sizes: {symbol: tick_size}.
        maker_fee_bps: Maker fee in bps.
        taker_fee_bps: Taker fee in bps.

    Returns:
        (achieved_weights, orders, total_fee_cost)
    """
    if tick_sizes is None:
        tick_sizes = {}

    orders = []
    achieved = dict(current_weights)
    total_fee = 0.0

    for sym in set(target_weights) | set(current_weights):
        target = target_weights.get(sym, 0.0)
        current = current_weights.get(sym, 0.0)
        delta = target - current

        if abs(delta) < 1e-6:
            continue

        # Create passive limit order
        side = "buy" if delta > 0 else "sell"
        # Use first bar's bid/ask approximation
        if len(bar_data) > 0 and sym in bar_data.columns:
            mid = bar_data[sym].iloc[0] if sym in bar_data.columns else 0
        else:
            mid = 0

        if mid <= 0:
            continue

        tick = tick_sizes.get(sym, 0.01)
        limit_price = mid - tick if side == "buy" else mid + tick

        order = OrderState(sym, side, delta, limit_price, placed_at=0)

        # Simulate across bars
        for i in range(len(bar_data)):
            if f"{sym}_low" in bar_data.columns and f"{sym}_high" in bar_data.columns:
                low = bar_data[f"{sym}_low"].iloc[i]
                high = bar_data[f"{sym}_high"].iloc[i]
            else:
                low = mid * 0.999
                high = mid * 1.001

            order = simulate_passive_fill(
                order, bar_idx=i, bar_low=low, bar_high=high,
                tick_size=tick, requote_interval_bars=1, max_requotes=2,
                cancel_after_bars=36,
            )
            if order.filled or order.cancelled:
                break

        orders.append(order)

        if order.filled:
            achieved[sym] = target
            # Track slippage for reporting
            order.slippage_bps = estimate_slippage_bps(order.filled_price, mid, side)
            fee = maker_fee_bps / 10000 * abs(delta)
            total_fee += fee
        else:
            # Order cancelled, keep current weight
            fee = 0.0

    return achieved, orders, total_fee
