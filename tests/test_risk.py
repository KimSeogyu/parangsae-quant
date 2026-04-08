import pytest

from src.risk.model import (
    _compute_avg_pairwise_correlation,
    compute_btc_regime_scale,
    compute_combined_risk_scale,
    compute_drawdown,
    compute_drawdown_scale,
    compute_ema,
    compute_correlation_scale,
    compute_vol_target_scale,
)


class TestBtcRegime:
    def test_above_ema_full_exposure(self):
        scale = compute_btc_regime_scale(btc_price=100, btc_ema=100, lower=0.90, upper=1.00, min_exposure=0.3)
        assert scale == 1.0

    def test_at_lower_bound(self):
        scale = compute_btc_regime_scale(btc_price=90, btc_ema=100, lower=0.90, upper=1.00, min_exposure=0.3)
        assert scale == pytest.approx(0.3)

    def test_below_lower_bound(self):
        scale = compute_btc_regime_scale(btc_price=80, btc_ema=100, lower=0.90, upper=1.00, min_exposure=0.3)
        assert scale == 0.3

    def test_midpoint_interpolation(self):
        scale = compute_btc_regime_scale(btc_price=95, btc_ema=100, lower=0.90, upper=1.00, min_exposure=0.3)
        assert 0.3 < scale < 1.0
        assert scale == pytest.approx(0.65)


class TestVolTargeting:
    def test_at_target_no_scaling(self):
        scale = compute_vol_target_scale(portfolio_vol=0.30, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == pytest.approx(1.0)

    def test_high_vol_scales_down(self):
        scale = compute_vol_target_scale(portfolio_vol=0.60, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == pytest.approx(0.5)

    def test_low_vol_scales_up(self):
        scale = compute_vol_target_scale(portfolio_vol=0.15, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == pytest.approx(1.5)  # capped at max

    def test_near_zero_vol_capped(self):
        scale = compute_vol_target_scale(portfolio_vol=0.001, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == 1.5

    def test_min_floor(self):
        scale = compute_vol_target_scale(portfolio_vol=5.0, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == 0.1


class TestCorrelation:
    def test_below_threshold_no_reduction(self):
        scale = compute_correlation_scale(avg_corr=0.50, threshold_high=0.70, threshold_crisis=0.85, min_exposure=0.40)
        assert scale == 1.0

    def test_at_crisis_minimum(self):
        scale = compute_correlation_scale(avg_corr=0.85, threshold_high=0.70, threshold_crisis=0.85, min_exposure=0.40)
        assert scale == pytest.approx(0.40)

    def test_above_crisis_floored(self):
        scale = compute_correlation_scale(avg_corr=0.95, threshold_high=0.70, threshold_crisis=0.85, min_exposure=0.40)
        assert scale == 0.40

    def test_midpoint(self):
        scale = compute_correlation_scale(avg_corr=0.775, threshold_high=0.70, threshold_crisis=0.85, min_exposure=0.40)
        assert 0.40 < scale < 1.0


class TestDrawdownScaling:
    def test_no_drawdown_full_exposure(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.02, tiers=tiers)
        assert scale == 1.0

    def test_at_first_tier(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.05, tiers=tiers)
        assert scale == 1.0

    def test_between_tiers(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.075, tiers=tiers)
        assert scale == pytest.approx(0.875)

    def test_full_halt(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.30, tiers=tiers)
        assert scale == 0.0

    def test_at_10pct(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.10, tiers=tiers)
        assert scale == pytest.approx(0.75)


class TestCombinedRisk:
    def test_all_layers_full(self):
        scale = compute_combined_risk_scale(regime=1.0, vol_scale=1.0, corr_scale=1.0, dd_scale=1.0, min_total=0.05)
        assert scale == 1.0

    def test_floor_applied(self):
        scale = compute_combined_risk_scale(regime=0.3, vol_scale=0.1, corr_scale=0.4, dd_scale=0.0, min_total=0.05)
        assert scale == 0.05

    def test_multiplicative(self):
        scale = compute_combined_risk_scale(regime=0.5, vol_scale=0.8, corr_scale=1.0, dd_scale=0.75, min_total=0.05)
        assert scale == pytest.approx(0.3)


class TestEma:
    def test_ema_converges_to_constant(self):
        prices = [100.0] * 100
        ema = compute_ema(prices, period=20)
        assert ema == pytest.approx(100.0, abs=0.01)

    def test_ema_below_sma_in_downtrend(self):
        prices = list(range(200, 100, -1))  # 200 down to 101
        ema = compute_ema(prices, period=50)
        sma = sum(prices[-50:]) / 50
        assert ema < sma  # EMA weights recent lower prices more


class TestDrawdown:
    def test_no_loss(self):
        assert compute_drawdown([100, 110, 120]) == 0.0

    def test_returns_current_drawdown(self):
        dd = compute_drawdown([100, 110, 105, 95, 100])
        assert dd == pytest.approx((110 - 100) / 110, abs=0.001)

    def test_drawdown_recovers_to_zero_after_new_high(self):
        dd = compute_drawdown([100, 110, 95, 112])
        assert dd == 0.0


class TestCorrelationSampling:
    def test_prefers_high_liquidity_symbols(self):
        return_buffers = {
            "LOW": [0.05, -0.05, 0.05, -0.05],
            "HIGH_A": [0.01, 0.02, 0.03, 0.04],
            "HIGH_B": [0.01, 0.02, 0.03, 0.04],
        }
        liquidity_buffers = {
            "LOW": [10.0, 10.0, 10.0, 10.0],
            "HIGH_A": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
            "HIGH_B": [900.0, 900.0, 900.0, 900.0],
        }

        avg_corr = _compute_avg_pairwise_correlation(
            return_buffers,
            sample_coins=2,
            window=4,
            liquidity_buffers=liquidity_buffers,
        )

        assert avg_corr == pytest.approx(1.0)
