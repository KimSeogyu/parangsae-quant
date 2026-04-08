"""Tests for long/short portfolio construction."""

from src.research.portfolio.construction import (
    apply_beta_hedge,
    apply_net_cap,
    apply_position_caps,
    build_portfolio,
    compute_weights,
    select_long_short,
)


class TestSelectLongShort:
    def test_basic_selection(self):
        scores = {f"SYM{i}": float(i) for i in range(20)}  # SYM19 highest, SYM0 lowest
        longs, shorts = select_long_short(scores, set(), set(), long_count=4, short_count=4)
        assert len(longs) <= 4
        assert len(shorts) <= 4
        # All longs should have positive scores
        for s in longs:
            assert scores[s] > 0
        # All shorts should have negative or zero scores (but SYM0=0, no short since score=0)

    def test_hysteresis_keeps_existing(self):
        scores = {f"SYM{i}": float(20 - i) for i in range(20)}
        # SYM0=20 highest, SYM19=1 lowest
        current_longs = {"SYM5"}  # Still in top 35% exit zone
        longs, shorts = select_long_short(
            scores, current_longs, set(), long_count=4, short_count=4
        )
        assert "SYM5" in longs  # Should be kept due to hysteresis

    def test_empty_scores(self):
        longs, shorts = select_long_short({}, set(), set())
        assert longs == set()
        assert shorts == set()


class TestComputeWeights:
    def test_gross_normalization(self):
        longs = {"A", "B"}
        shorts = {"C", "D"}
        scores = {"A": 1.5, "B": 1.0, "C": -1.2, "D": -0.8}
        vols = {"A": 0.02, "B": 0.03, "C": 0.02, "D": 0.025}
        weights = compute_weights(longs, shorts, scores, vols, gross=1.0)
        total_abs = sum(abs(w) for w in weights.values())
        assert abs(total_abs - 1.0) < 1e-6

    def test_longs_positive_shorts_negative(self):
        longs = {"A"}
        shorts = {"B"}
        scores = {"A": 1.0, "B": -1.0}
        vols = {"A": 0.02, "B": 0.02}
        weights = compute_weights(longs, shorts, scores, vols)
        assert weights["A"] > 0
        assert weights["B"] < 0


class TestPositionCaps:
    def test_single_name_cap(self):
        weights = {"A": 0.20, "B": 0.15, "C": -0.10}
        capped = apply_position_caps(weights, single_name_cap=0.12)
        assert abs(capped["A"]) <= 0.12
        assert abs(capped["B"]) <= 0.12 + 1e-10

    def test_sector_cap(self):
        weights = {"A": 0.10, "B": 0.10, "C": 0.10, "D": 0.10}
        sector_map = {"A": "defi", "B": "defi", "C": "defi", "D": "l1"}
        capped = apply_position_caps(
            weights, single_name_cap=0.15, sector_map=sector_map, sector_soft_cap=0.20
        )
        defi_total = sum(abs(capped[s]) for s in ["A", "B", "C"])
        assert defi_total <= 0.20 + 1e-6


class TestBetaHedge:
    def test_no_hedge_within_cap(self):
        weights = {"A": 0.5, "B": -0.5}
        betas = {"A": 1.0, "B": 1.0}
        result = apply_beta_hedge(weights, betas, max_beta=0.10)
        # Net beta = 0.5*1 + (-0.5)*1 = 0.0, within cap
        assert "BTCUSDT" not in result or result.get("BTCUSDT", 0) == 0

    def test_hedge_when_exceeded(self):
        weights = {"A": 0.8, "B": -0.2}
        betas = {"A": 1.2, "B": 0.5, "BTCUSDT": 1.0}
        result = apply_beta_hedge(weights, betas, max_beta=0.10)
        # Original beta = 0.8*1.2 + (-0.2)*0.5 = 0.86, needs hedge
        assert "BTCUSDT" in result
        assert result["BTCUSDT"] < 0  # Short BTC to reduce beta


class TestNetCap:
    def test_within_cap(self):
        weights = {"A": 0.55, "B": -0.45}  # Net = 0.10
        result = apply_net_cap(weights, net_cap=0.15)
        assert abs(sum(result.values())) <= 0.15 + 1e-6

    def test_exceeds_cap_scales_down(self):
        weights = {"A": 0.7, "B": -0.1}  # Net = 0.6
        result = apply_net_cap(weights, net_cap=0.15)
        assert abs(sum(result.values())) <= 0.15 + 1e-6


class TestBuildPortfolio:
    def test_full_pipeline(self):
        scores = {f"SYM{i}": float(15 - i) for i in range(30)}
        vols = {f"SYM{i}": 0.02 for i in range(30)}
        betas = {f"SYM{i}": 1.0 for i in range(30)}
        weights, longs, shorts = build_portfolio(scores, vols, betas)
        assert len(longs) > 0
        assert len(shorts) > 0
        # All weights should respect single-name cap
        for w in weights.values():
            assert abs(w) <= 0.12 + 1e-6
