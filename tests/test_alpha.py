import math
import numpy as np
import pytest

from src.alpha.momentum import compute_qm_momentum


def _make_trending_up(n: int, base: float = 100.0, hourly_drift: float = 0.001) -> list[float]:
    """Generate monotonically trending-up hourly closes."""
    prices = [base]
    for i in range(1, n):
        prices.append(prices[-1] * (1 + hourly_drift))
    return prices


def _make_trending_down(n: int, base: float = 100.0, hourly_drift: float = 0.001) -> list[float]:
    """Generate monotonically trending-down hourly closes."""
    prices = [base]
    for i in range(1, n):
        prices.append(prices[-1] * (1 - hourly_drift))
    return prices


class TestQMMomentum:
    def test_positive_trending_produces_positive_alpha(self):
        closes = _make_trending_up(1100)
        alpha = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        assert alpha > 0

    def test_negative_trending_filtered_by_ts(self):
        closes = _make_trending_down(1100)
        alpha = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        assert alpha == float("-inf")

    def test_ts_filter_disabled_allows_negative(self):
        closes = _make_trending_down(1100)
        alpha = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=False,
        )
        assert alpha != float("-inf")
        assert alpha < 0

    def test_insufficient_data_returns_neg_inf(self):
        closes = [100.0] * 50
        alpha = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        assert alpha == float("-inf")

    def test_fip_quality_multiplier_bounds(self):
        """When daily FIP is positive (mixed/noisy uptrend has some negative days),
        quality is bounded from below by fip_floor.
        A higher fip_floor raises the minimum quality and thus increases alpha."""
        np.random.seed(7)
        # Build a noisy uptrend: net positive momentum but mixed daily returns
        n = 1100
        prices = [100.0]
        for i in range(1, n):
            # Drift up with moderate noise so some daily bars go negative
            prices.append(prices[-1] * (1 + 0.0005 + np.random.normal(0, 0.015)))
        # Ensure net positive for ts_filter (set endpoint above start)
        prices[-25] = prices[-361] * 1.05  # guarantee positive raw_mom at skip=24

        alpha_low_floor = compute_qm_momentum(
            prices, lookback=336, skip=24, vol_window=720, fip_floor=0.1, ts_filter=True,
        )
        alpha_high_floor = compute_qm_momentum(
            prices, lookback=336, skip=24, vol_window=720, fip_floor=0.5, ts_filter=True,
        )
        # Both should be valid (not -inf) given positive net momentum
        assert alpha_low_floor != float("-inf"), "Expected non-filtered alpha for low floor"
        assert alpha_high_floor != float("-inf"), "Expected non-filtered alpha for high floor"
        # Higher floor -> higher minimum quality -> higher (or equal) alpha
        assert alpha_high_floor >= alpha_low_floor

    def test_vol_adjustment_reduces_high_vol_signal(self):
        """Add noise to make vol higher; alpha should decrease vs smooth trend."""
        np.random.seed(42)
        smooth = _make_trending_up(1100, hourly_drift=0.002)
        noisy = [p * (1 + np.random.normal(0, 0.01)) for p in smooth]
        # Anchor skip boundary and start to ensure positive raw_mom
        noisy[-(24 + 1)] = smooth[-(24 + 1)]
        noisy[-(24 + 336 + 1)] = smooth[-(24 + 336 + 1)]

        alpha_smooth = compute_qm_momentum(
            smooth, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        alpha_noisy = compute_qm_momentum(
            noisy, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        assert alpha_noisy != float("-inf"), "Noisy series should have positive momentum"
        assert alpha_smooth > alpha_noisy
