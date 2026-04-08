"""Tests for execution simulation and post-fill validation."""

from src.research.execution.passive import (
    OrderState,
    compute_maker_fill_ratio,
    estimate_slippage_bps,
    simulate_passive_fill,
)
from src.research.execution.postfill import PostfillAction, evaluate_postfill


class TestPassiveFill:
    def test_buy_fills_when_low_touches(self):
        order = OrderState("BTC", "buy", 0.1, 50000.0, placed_at=0)
        order = simulate_passive_fill(order, bar_idx=1, bar_low=49999.0, bar_high=50100.0)
        assert order.filled
        assert order.filled_price == 50000.0

    def test_buy_no_fill_when_low_above(self):
        order = OrderState("BTC", "buy", 0.1, 50000.0, placed_at=0)
        order = simulate_passive_fill(order, bar_idx=1, bar_low=50001.0, bar_high=50100.0)
        assert not order.filled

    def test_sell_fills_when_high_touches(self):
        order = OrderState("ETH", "sell", -0.1, 3000.0, placed_at=0)
        order = simulate_passive_fill(order, bar_idx=1, bar_low=2990.0, bar_high=3001.0)
        assert order.filled

    def test_requote_moves_price(self):
        order = OrderState("BTC", "buy", 0.1, 50000.0, placed_at=0)
        # Simulate for requote_interval_bars without fill
        for i in range(1, 13):
            order = simulate_passive_fill(
                order, bar_idx=i, bar_low=50100.0, bar_high=50200.0,
                requote_interval_bars=12,
            )
        assert order.requotes == 1
        assert order.limit_price == 50000.01  # Moved up 1 tick

    def test_cancel_after_timeout(self):
        order = OrderState("BTC", "buy", 0.1, 50000.0, placed_at=0)
        order = simulate_passive_fill(
            order, bar_idx=36, bar_low=50100.0, bar_high=50200.0,
            cancel_after_bars=36,
        )
        assert order.cancelled


class TestMakerFillRatio:
    def test_all_filled(self):
        orders = [
            OrderState("A", "buy", 0.1, 100.0, 0, filled=True),
            OrderState("B", "sell", -0.1, 200.0, 0, filled=True),
        ]
        assert compute_maker_fill_ratio(orders) == 1.0

    def test_half_filled(self):
        orders = [
            OrderState("A", "buy", 0.1, 100.0, 0, filled=True),
            OrderState("B", "sell", -0.1, 200.0, 0, cancelled=True),
        ]
        assert compute_maker_fill_ratio(orders) == 0.5

    def test_empty(self):
        assert compute_maker_fill_ratio([]) == 0.0


class TestSlippage:
    def test_buy_above_mid(self):
        slip = estimate_slippage_bps(50010.0, 50000.0, "buy")
        assert slip > 0  # Adverse

    def test_sell_below_mid(self):
        slip = estimate_slippage_bps(49990.0, 50000.0, "sell")
        assert slip > 0  # Adverse

    def test_buy_at_mid(self):
        slip = estimate_slippage_bps(50000.0, 50000.0, "buy")
        assert slip == 0.0


class TestPostfill:
    def test_hold_on_positive_follow_through(self):
        result = evaluate_postfill(
            "ETH", "long",
            signed_residual_return_5m=0.001,
            residual_vol_1d=0.02,
            current_spread_bps=5.0,
            rolling_median_spread_bps=4.0,
            current_score=1.0,
            original_score_sign=1.0,
        )
        assert result.action == PostfillAction.HOLD

    def test_reduce_on_no_follow_through(self):
        result = evaluate_postfill(
            "ETH", "long",
            signed_residual_return_5m=-0.0001,
            residual_vol_1d=0.02,
            current_spread_bps=5.0,
            rolling_median_spread_bps=4.0,
            current_score=1.0,
            original_score_sign=1.0,
        )
        assert result.action == PostfillAction.REDUCE_50

    def test_close_on_large_adverse(self):
        result = evaluate_postfill(
            "ETH", "long",
            signed_residual_return_5m=-0.01,
            residual_vol_1d=0.02,
            current_spread_bps=5.0,
            rolling_median_spread_bps=4.0,
            current_score=1.0,
            original_score_sign=1.0,
            close_vol_mult=-0.25,
        )
        # Threshold = -0.25 * 0.02 = -0.005, return -0.01 < -0.005
        assert result.action == PostfillAction.CLOSE_100

    def test_cancel_on_spread_anomaly(self):
        result = evaluate_postfill(
            "ETH", "long",
            signed_residual_return_5m=0.001,
            residual_vol_1d=0.02,
            current_spread_bps=20.0,
            rolling_median_spread_bps=5.0,
            current_score=1.0,
            original_score_sign=1.0,
            spread_multiple=2.0,
        )
        assert result.action == PostfillAction.CANCEL_NEW

    def test_queue_close_on_score_flip(self):
        result = evaluate_postfill(
            "ETH", "long",
            signed_residual_return_5m=0.001,
            residual_vol_1d=0.02,
            current_spread_bps=5.0,
            rolling_median_spread_bps=4.0,
            current_score=-0.5,
            original_score_sign=1.0,
        )
        assert result.action == PostfillAction.QUEUE_CLOSE
