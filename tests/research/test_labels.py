"""Tests for label pipeline (beta estimation and residual returns)."""

import numpy as np
import pandas as pd

from src.research.labels.beta import compute_dual_window_beta, compute_rolling_beta
from src.research.labels.residual import (
    build_labels,
    combine_predictions,
    compute_forward_log_return,
    compute_residual_return,
)


def _make_returns(n: int, beta: float = 1.0, noise: float = 0.01, seed: int = 42):
    """Generate synthetic asset returns = beta * btc + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    btc = pd.Series(rng.normal(0, 0.001, n), index=idx)
    asset = beta * btc + rng.normal(0, noise, n)
    return asset, btc


class TestRollingBeta:
    def test_known_beta(self):
        """With synthetic data where true beta=1.5, rolling beta should converge."""
        asset, btc = _make_returns(5000, beta=1.5, noise=0.0001)
        betas = compute_rolling_beta(asset, btc, window=500)
        # After warmup, beta should be close to 1.5
        valid = betas.dropna()
        assert len(valid) > 0
        assert abs(valid.iloc[-1] - 1.5) < 0.1

    def test_zero_btc_returns(self):
        """If BTC has no variance, beta should be 0."""
        idx = pd.date_range("2025-01-01", periods=100, freq="5min", tz="UTC")
        btc = pd.Series(0.0, index=idx)
        asset = pd.Series(np.random.default_rng(42).normal(0, 0.01, 100), index=idx)
        betas = compute_rolling_beta(asset, btc, window=50)
        valid = betas.dropna()
        assert all(valid == 0.0)

    def test_insufficient_data_returns_nan(self):
        asset, btc = _make_returns(50, beta=1.0)
        betas = compute_rolling_beta(asset, btc, window=100)
        assert betas.isna().all()


class TestDualWindowBeta:
    def test_averages_windows(self):
        asset, btc = _make_returns(10000, beta=2.0, noise=0.0001)
        # Use very small "weeks" for testing
        beta = compute_dual_window_beta(asset, btc, windows_weeks=[1, 2], bars_per_week=500)
        valid = beta.dropna()
        assert len(valid) > 0
        assert abs(valid.iloc[-1] - 2.0) < 0.2


class TestForwardLogReturn:
    def test_basic(self):
        prices = pd.Series([100.0, 110.0, 121.0, 100.0], dtype=float)
        fwd = compute_forward_log_return(prices, horizon_bars=1)
        assert abs(fwd.iloc[0] - np.log(1.1)) < 1e-10
        assert pd.isna(fwd.iloc[-1])

    def test_horizon_2(self):
        prices = pd.Series([100.0, 110.0, 121.0, 133.1], dtype=float)
        fwd = compute_forward_log_return(prices, horizon_bars=2)
        assert abs(fwd.iloc[0] - np.log(121.0 / 100.0)) < 1e-10
        assert pd.isna(fwd.iloc[-1])
        assert pd.isna(fwd.iloc[-2])


class TestResidualReturn:
    def test_zero_beta(self):
        """With beta=0, residual = raw return."""
        fwd_asset = pd.Series([0.01, 0.02, -0.01])
        fwd_btc = pd.Series([0.05, 0.05, 0.05])
        beta = pd.Series([0.0, 0.0, 0.0])
        resid = compute_residual_return(fwd_asset, fwd_btc, beta)
        assert abs(resid.iloc[0] - 0.01) < 1e-10

    def test_beta_one(self):
        """With beta=1, residual = asset - BTC."""
        fwd_asset = pd.Series([0.03, 0.02])
        fwd_btc = pd.Series([0.01, 0.01])
        beta = pd.Series([1.0, 1.0])
        resid = compute_residual_return(fwd_asset, fwd_btc, beta)
        assert abs(resid.iloc[0] - 0.02) < 1e-10


class TestBuildLabels:
    def test_multiple_symbols(self):
        idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
        rng = np.random.default_rng(42)
        mid_btc = pd.Series(50000 + np.cumsum(rng.normal(0, 10, 20)), index=idx)
        mid_eth = pd.Series(3000 + np.cumsum(rng.normal(0, 5, 20)), index=idx)
        betas = {"ETH": pd.Series(1.0, index=idx)}

        labels = build_labels(
            {"ETH": mid_eth}, mid_btc, betas,
            primary_horizon_bars=4, secondary_horizon_bars=12,
        )
        assert "ETH" in labels
        assert "y20" in labels["ETH"].columns
        assert "y60" in labels["ETH"].columns
        # Last 4 bars of y20 should be NaN (forward-looking)
        assert labels["ETH"]["y20"].iloc[-1:].isna().all()


class TestCombinePredictions:
    def test_weighted_combination(self):
        y20 = pd.Series([1.0, 2.0, 3.0])
        y60 = pd.Series([10.0, 20.0, 30.0])
        score = combine_predictions(y20, y60, 0.7, 0.3)
        expected = 0.7 * 1.0 + 0.3 * 10.0
        assert abs(score.iloc[0] - expected) < 1e-10
