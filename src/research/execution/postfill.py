"""Post-fill 5-minute validation rules.

PRD Section 8: Post-fill 5-min validation
- signed residual return <= 0 → reduce 50%
- signed residual return < -0.25 * residual_vol_1d → close 100%
- spread > 2x rolling median or OB imbalance spike → cancel new, reduce existing
- score sign flip → queue full close next cycle
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PostfillAction(Enum):
    HOLD = "hold"
    REDUCE_50 = "reduce_50"
    CLOSE_100 = "close_100"
    CANCEL_NEW = "cancel_new"
    QUEUE_CLOSE = "queue_close"


@dataclass
class PostfillResult:
    symbol: str
    action: PostfillAction
    reason: str
    residual_return: float = 0.0
    threshold: float = 0.0


def evaluate_postfill(
    symbol: str,
    side: str,
    signed_residual_return_5m: float,
    residual_vol_1d: float,
    current_spread_bps: float,
    rolling_median_spread_bps: float,
    current_score: float,
    original_score_sign: float,
    reduce_threshold: float = 0.0,
    close_vol_mult: float = -0.25,
    spread_multiple: float = 2.0,
) -> PostfillResult:
    """Evaluate post-fill validation rules.

    Args:
        symbol: Symbol identifier.
        side: "long" or "short".
        signed_residual_return_5m: Residual return in direction of position.
            For longs: positive if price went up.
            For shorts: positive if price went down.
        residual_vol_1d: 1-day residual return volatility.
        current_spread_bps: Current spread in bps.
        rolling_median_spread_bps: Rolling median spread in bps.
        current_score: Current model score for this symbol.
        original_score_sign: Sign of score when position was entered.
        reduce_threshold: Threshold for 50% reduction (default 0).
        close_vol_mult: Vol multiplier for 100% close (default -0.25).
        spread_multiple: Multiple of median spread that triggers cancel.

    Returns:
        PostfillResult with recommended action.
    """
    # Rule 1: Score sign flip → queue close
    if current_score * original_score_sign < 0:
        return PostfillResult(
            symbol=symbol,
            action=PostfillAction.QUEUE_CLOSE,
            reason="Score sign reversed",
        )

    # Rule 2: Spread anomaly → cancel/reduce
    if (
        rolling_median_spread_bps > 0
        and current_spread_bps > spread_multiple * rolling_median_spread_bps
    ):
        return PostfillResult(
            symbol=symbol,
            action=PostfillAction.CANCEL_NEW,
            reason=f"Spread {current_spread_bps:.1f} > {spread_multiple}x median "
                   f"{rolling_median_spread_bps:.1f}",
        )

    # Rule 3: Large adverse move → close 100%
    close_threshold = close_vol_mult * residual_vol_1d
    if signed_residual_return_5m < close_threshold:
        return PostfillResult(
            symbol=symbol,
            action=PostfillAction.CLOSE_100,
            reason=f"Adverse residual return {signed_residual_return_5m:.6f} "
                   f"< {close_threshold:.6f}",
            residual_return=signed_residual_return_5m,
            threshold=close_threshold,
        )

    # Rule 4: No follow-through → reduce 50%
    if signed_residual_return_5m <= reduce_threshold:
        return PostfillResult(
            symbol=symbol,
            action=PostfillAction.REDUCE_50,
            reason=f"No follow-through: residual return {signed_residual_return_5m:.6f} <= 0",
            residual_return=signed_residual_return_5m,
            threshold=reduce_threshold,
        )

    # All checks passed
    return PostfillResult(
        symbol=symbol,
        action=PostfillAction.HOLD,
        reason="Post-fill validation passed",
        residual_return=signed_residual_return_5m,
    )
