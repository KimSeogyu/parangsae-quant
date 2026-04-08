"""Passive execution simulation for backtesting.

PRD Section 8: Execution Specification
- New entries: passive limit at best bid (buy) / best ask (sell)
- Requote: every 60s, max 2 times, 1 tick step
- Unfilled after 3 min: cancel
- Emergency: taker allowed
"""

from __future__ import annotations

from dataclasses import dataclass



@dataclass
class OrderState:
    symbol: str
    side: str  # "buy" or "sell"
    target_weight: float
    limit_price: float
    placed_at: int  # bar index
    requotes: int = 0
    filled: bool = False
    filled_price: float = 0.0
    filled_at: int = 0
    cancelled: bool = False


def simulate_passive_fill(
    order: OrderState,
    bar_idx: int,
    bar_low: float,
    bar_high: float,
    tick_size: float = 0.01,
    requote_interval_bars: int = 12,
    max_requotes: int = 2,
    cancel_after_bars: int = 36,
) -> OrderState:
    """Simulate passive limit order fill on a 5-min bar.

    Fill condition (touch-based):
    - Buy: bar_low <= limit_price
    - Sell: bar_high >= limit_price

    Args:
        order: Current order state.
        bar_idx: Current bar index.
        bar_low: Low price of current bar.
        bar_high: High price of current bar.
        tick_size: Minimum price increment.
        requote_interval_bars: Bars between requotes (60s / 5min = 12 bars for 1m, or 1 for 5m).
        max_requotes: Maximum requote attempts.
        cancel_after_bars: Cancel if unfilled after this many bars.

    Returns:
        Updated order state.
    """
    if order.filled or order.cancelled:
        return order

    elapsed = bar_idx - order.placed_at

    # Check for fill
    if order.side == "buy" and bar_low <= order.limit_price:
        order.filled = True
        order.filled_price = order.limit_price
        order.filled_at = bar_idx
        return order

    if order.side == "sell" and bar_high >= order.limit_price:
        order.filled = True
        order.filled_price = order.limit_price
        order.filled_at = bar_idx
        return order

    # Requote logic
    if (
        elapsed > 0
        and elapsed % requote_interval_bars == 0
        and order.requotes < max_requotes
    ):
        if order.side == "buy":
            order.limit_price += tick_size
        else:
            order.limit_price -= tick_size
        order.requotes += 1

    # Cancel if too old
    if elapsed >= cancel_after_bars:
        order.cancelled = True

    return order


def compute_maker_fill_ratio(orders: list[OrderState]) -> float:
    """Fraction of orders filled as maker (at limit price)."""
    if not orders:
        return 0.0
    filled = [o for o in orders if o.filled]
    return len(filled) / len(orders) if orders else 0.0


def estimate_slippage_bps(
    filled_price: float,
    mid_price: float,
    side: str,
) -> float:
    """Estimate slippage in basis points.

    For buys: positive if filled above mid.
    For sells: positive if filled below mid.
    """
    if mid_price <= 0:
        return 0.0
    if side == "buy":
        return (filled_price - mid_price) / mid_price * 10000
    else:
        return (mid_price - filled_price) / mid_price * 10000
