import pytest
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from src.portfolio.construction import (
    apply_tiered_caps,
    apply_liquidity_caps,
    compute_target_deltas,
    compute_target_weights,
    apply_turnover_limit,
    filter_tradeable_deltas,
    quantize_order_quantity,
    select_holdings,
    should_rebalance,
    zscore_and_combine,
)


class TestZscoreCombine:
    def test_combines_two_alphas(self):
        alpha_scores = {
            "momentum": {"BTC": 0.5, "ETH": 0.3, "SOL": -0.1},
            "low_vol": {"BTC": -0.02, "ETH": -0.01, "SOL": -0.05},
        }
        weights = {"momentum": 0.75, "low_vol": 0.25}
        universe = {"BTC", "ETH", "SOL"}
        combined = zscore_and_combine(alpha_scores, weights, universe, clip=3.0)
        assert set(combined.keys()) == {"BTC", "ETH", "SOL"}
        assert combined["BTC"] > combined["ETH"]

    def test_clips_extreme_scores(self):
        alpha_scores = {"momentum": {"A": 100.0, "B": 0.0, "C": -100.0}}
        weights = {"momentum": 1.0}
        combined = zscore_and_combine(alpha_scores, weights, {"A", "B", "C"}, clip=3.0)
        assert all(-3.5 <= v <= 3.5 for v in combined.values())

    def test_single_symbol_returns_zero(self):
        alpha_scores = {"momentum": {"BTC": 0.5}}
        weights = {"momentum": 1.0}
        combined = zscore_and_combine(alpha_scores, weights, {"BTC"}, clip=3.0)
        assert combined["BTC"] == pytest.approx(0.0)

    def test_ignores_non_universe(self):
        alpha_scores = {"momentum": {"BTC": 0.5, "XRP": 0.8}}
        weights = {"momentum": 1.0}
        combined = zscore_and_combine(alpha_scores, weights, {"BTC"}, clip=3.0)
        assert "XRP" not in combined


class TestSelectHoldings:
    def test_selects_top_n(self):
        scores = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        current = set()
        selected = select_holdings(scores, current, num_holdings=3, entry_rank=2, exit_rank=4)
        assert "A" in selected
        assert "B" in selected

    def test_hysteresis_keeps_existing(self):
        scores = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        current = {"D"}  # D is rank 4, between entry_rank and exit_rank
        selected = select_holdings(scores, current, num_holdings=3, entry_rank=2, exit_rank=5)
        assert "D" in selected

    def test_hysteresis_removes_fallen(self):
        scores = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        current = {"E"}  # E is rank 5, beyond exit_rank
        selected = select_holdings(scores, current, num_holdings=3, entry_rank=2, exit_rank=4)
        assert "E" not in selected

    def test_negative_scores_excluded(self):
        scores = {"A": 0.9, "B": -0.5, "C": float("-inf")}
        selected = select_holdings(scores, set(), num_holdings=3, entry_rank=2, exit_rank=4)
        assert "B" not in selected
        assert "C" not in selected


class TestTieredCaps:
    def test_btc_cap_applied(self):
        weights = {"BTCUSDT.BINANCE": 0.30, "ETHUSDT.BINANCE": 0.10, "SOLUSDT.BINANCE": 0.05}
        capped = apply_tiered_caps(weights, max_btc=0.20, max_eth=0.15, max_other=0.07)
        assert capped["BTCUSDT.BINANCE"] == pytest.approx(0.20)
        assert capped["ETHUSDT.BINANCE"] == pytest.approx(0.10)
        assert capped["SOLUSDT.BINANCE"] == pytest.approx(0.05)

    def test_eth_cap_applied(self):
        weights = {"ETHUSDT.BINANCE": 0.25}
        capped = apply_tiered_caps(weights, max_btc=0.20, max_eth=0.15, max_other=0.07)
        assert capped["ETHUSDT.BINANCE"] == pytest.approx(0.15)

    def test_other_cap_applied(self):
        weights = {"SOLUSDT.BINANCE": 0.10}
        capped = apply_tiered_caps(weights, max_btc=0.20, max_eth=0.15, max_other=0.07)
        assert capped["SOLUSDT.BINANCE"] == pytest.approx(0.07)


class TestTargetWeights:
    def test_alpha_times_inv_vol(self):
        holdings = {"A", "B"}
        scores = {"A": 2.0, "B": 1.0}
        volatility = {"A": 0.02, "B": 0.04}
        weights = compute_target_weights(holdings, scores, volatility, risk_scale=1.0, min_position=0.01)
        assert weights["A"] > weights["B"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_risk_scale_reduces_weights(self):
        holdings = {"A"}
        scores = {"A": 1.0}
        volatility = {"A": 0.02}
        w_full = compute_target_weights(holdings, scores, volatility, risk_scale=1.0, min_position=0.01)
        w_half = compute_target_weights(holdings, scores, volatility, risk_scale=0.5, min_position=0.01)
        assert w_half["A"] == pytest.approx(w_full["A"] * 0.5, abs=0.01)

    def test_min_position_filter(self):
        holdings = {"A", "B"}
        scores = {"A": 1.0, "B": 0.001}
        volatility = {"A": 0.02, "B": 0.02}
        weights = compute_target_weights(holdings, scores, volatility, risk_scale=1.0, min_position=0.01)
        if "B" in weights:
            assert weights["B"] >= 0.01


class TestTargetDeltas:
    def test_basic_delta(self):
        deltas = compute_target_deltas({"A": 0.05, "B": 0.03}, {"A": 0.03}, threshold=0.005)
        assert deltas["A"] == pytest.approx(0.02)
        assert deltas["B"] == pytest.approx(0.03)

    def test_below_threshold_ignored(self):
        deltas = compute_target_deltas({"A": 0.050}, {"A": 0.049}, threshold=0.005)
        assert "A" not in deltas


class TestTurnoverLimit:
    def test_caps_total_turnover(self):
        deltas = {"A": 0.08, "B": 0.06, "C": -0.04}
        limited = apply_turnover_limit(deltas, max_turnover=0.10)
        total = sum(abs(v) for v in limited.values())
        assert total <= 0.10 + 1e-9

    def test_preserves_direction(self):
        deltas = {"A": 0.05, "B": -0.03}
        limited = apply_turnover_limit(deltas, max_turnover=1.0)
        assert limited["A"] > 0
        assert limited["B"] < 0

    def test_no_limit_when_below(self):
        deltas = {"A": 0.02, "B": 0.01}
        limited = apply_turnover_limit(deltas, max_turnover=0.10)
        assert limited == deltas

    def test_zero_budget_returns_empty_deltas(self):
        deltas = {"A": 0.02, "B": 0.01}
        limited = apply_turnover_limit(deltas, max_turnover=0.0)
        assert limited == {}


class TestTradeableDeltas:
    def test_filters_small_deltas_after_scaling(self):
        deltas = {"A": 0.004, "B": -0.006}
        filtered = filter_tradeable_deltas(deltas, threshold=0.005)
        assert filtered == {"B": -0.006}


class TestRebalanceCadence:
    def test_rebalances_on_first_bar(self):
        assert should_rebalance(current_hour=10, last_rebalance_hour=-1, interval_hours=3)

    def test_waits_for_interval(self):
        assert not should_rebalance(current_hour=11, last_rebalance_hour=10, interval_hours=3)
        assert should_rebalance(current_hour=13, last_rebalance_hour=10, interval_hours=3)


class TestOrderQuantization:
    def test_returns_none_when_quantity_rounds_to_zero(self):
        instrument = TestInstrumentProvider.adausdt_binance()
        assert quantize_order_quantity(instrument, 1e-11) is None

    def test_returns_quantity_when_above_minimum_increment(self):
        instrument = TestInstrumentProvider.adausdt_binance()
        rounded = quantize_order_quantity(instrument, 1.2)
        assert rounded is not None


class TestLiquidityCaps:
    def test_caps_weight_by_trailing_quote_volume(self):
        weights = {"ADAUSDT.BINANCE": 0.20}
        quote_volume_buffers = {"ADAUSDT.BINANCE": [1000.0] * 24}
        capped = apply_liquidity_caps(
            weights,
            quote_volume_buffers=quote_volume_buffers,
            total_equity=100_000.0,
            liquidity_cap_pct=0.01,
            min_position=0.001,
        )
        assert capped["ADAUSDT.BINANCE"] == pytest.approx(0.0024)

    def test_drops_positions_below_minimum_after_liquidity_cap(self):
        weights = {"ADAUSDT.BINANCE": 0.20}
        quote_volume_buffers = {"ADAUSDT.BINANCE": [100.0] * 24}
        capped = apply_liquidity_caps(
            weights,
            quote_volume_buffers=quote_volume_buffers,
            total_equity=100_000.0,
            liquidity_cap_pct=0.01,
            min_position=0.01,
        )
        assert capped == {}
