"""Tests for all 5 feature blocks and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.research.features.relative import (
    compute_resid_momentum,
    compute_resid_rsi,
    compute_resid_zscore,
)
from src.research.features.trend import (
    compute_fip_like,
    compute_momentum,
    compute_momentum_voladj,
    compute_trend_linearity,
    compute_wickiness,
)
from src.research.features.liquidity import (
    compute_amihud,
    compute_ob_imbalance,
    compute_qvol_zscore,
    compute_spread_bps,
    compute_vwap_deviation,
)
from src.research.features.derivatives import (
    compute_basis,
    compute_funding_change,
    compute_funding_zscore,
    compute_oi_change,
    compute_oi_to_volume,
)
from src.research.features.risk_features import (
    compute_alt_breadth,
    compute_btc_trend,
    compute_corr_btc,
    compute_downside_vol,
    compute_realized_vol,
)
from src.research.features.preprocess import (
    check_coverage,
    clip_features,
    cross_sectional_zscore,
    fill_missing_cross_sectional_median,
    preprocess_features,
)
from src.research.features.registry import load_manifest, get_features_by_block


def _make_prices(n=500, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.Series(100 + np.cumsum(rng.normal(0, 0.1, n)), index=idx)


# ---------- Block 1: Relative ----------

class TestRelative:
    def test_resid_momentum_shape(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2025-01-01", periods=100, freq="5min", tz="UTC")
        asset = pd.Series(rng.normal(0, 0.01, 100), index=idx)
        btc = pd.Series(rng.normal(0, 0.01, 100), index=idx)
        beta = pd.Series(1.0, index=idx)
        result = compute_resid_momentum(asset, btc, beta, horizon_bars=12)
        assert len(result) == 100
        assert result.iloc[:11].isna().all()
        assert result.iloc[11:].notna().any()

    def test_resid_zscore(self):
        idx = pd.date_range("2025-01-01", periods=50, freq="5min", tz="UTC")
        asset = pd.Series(np.ones(50) * 0.01, index=idx)
        btc = pd.Series(np.zeros(50), index=idx)
        beta = pd.Series(0.0, index=idx)
        z = compute_resid_zscore(asset, btc, beta, horizon_bars=10)
        # Constant residuals → std=0 → NaN z-score
        valid = z.dropna()
        # All values are the same, so std=0, z=NaN
        assert len(valid) == 0 or valid.isna().all()

    def test_resid_rsi_bounds(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2025-01-01", periods=200, freq="5min", tz="UTC")
        asset = pd.Series(rng.normal(0, 0.01, 200), index=idx)
        btc = pd.Series(rng.normal(0, 0.01, 200), index=idx)
        beta = pd.Series(1.0, index=idx)
        rsi = compute_resid_rsi(asset, btc, beta, period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


# ---------- Block 2: Trend ----------

class TestTrend:
    def test_momentum_positive_trend(self):
        idx = pd.date_range("2025-01-01", periods=50, freq="5min", tz="UTC")
        prices = pd.Series(np.arange(100, 150, dtype=float), index=idx)
        mom = compute_momentum(prices, horizon_bars=10)
        assert mom.iloc[-1] > 0

    def test_voladj_momentum(self):
        prices = _make_prices(500)
        vadjm = compute_momentum_voladj(prices, horizon_bars=48)
        valid = vadjm.dropna()
        assert len(valid) > 0

    def test_fip_range(self):
        prices = _make_prices(500)
        fip = compute_fip_like(prices, horizon_bars=100)
        valid = fip.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_trend_linearity_perfect(self):
        idx = pd.date_range("2025-01-01", periods=50, freq="5min", tz="UTC")
        prices = pd.Series(np.exp(np.linspace(0, 1, 50)), index=idx)
        r2 = compute_trend_linearity(prices, horizon_bars=30)
        valid = r2.dropna()
        assert valid.iloc[-1] > 0.99  # Perfect linear log-price

    def test_wickiness_range(self):
        idx = pd.date_range("2025-01-01", periods=50, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {"open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0},
            index=idx,
        )
        wick = compute_wickiness(df, horizon_bars=10)
        valid = wick.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


# ---------- Block 3: Liquidity ----------

class TestLiquidity:
    def test_qvol_zscore(self):
        rng = np.random.default_rng(42)
        qvol = pd.Series(rng.uniform(1e6, 2e6, 500))
        z = compute_qvol_zscore(qvol, horizon_bars=100)
        valid = z.dropna()
        assert abs(valid.mean()) < 1.0  # Approximately zero-mean

    def test_amihud(self):
        idx = pd.date_range("2025-01-01", periods=50, freq="5min", tz="UTC")
        closes = pd.Series(100 + np.arange(50, dtype=float) * 0.1, index=idx)
        vol = pd.Series(1e6, index=idx)
        a = compute_amihud(closes, vol, horizon_bars=10)
        valid = a.dropna()
        assert (valid >= 0).all()

    def test_spread_bps(self):
        bid = pd.Series([100.0, 200.0])
        ask = pd.Series([100.05, 200.10])
        spread = compute_spread_bps(bid, ask)
        assert spread.iloc[0] == pytest.approx(5.0, rel=0.01)

    def test_vwap_deviation(self):
        closes = pd.Series([100.0, 102.0, 104.0])
        qvol = pd.Series([1000.0, 2040.0, 3120.0])
        vol = pd.Series([10.0, 20.0, 30.0])
        dev = compute_vwap_deviation(closes, qvol, vol, horizon_bars=3)
        assert dev.iloc[-1] is not None

    def test_ob_imbalance_range(self):
        bid_size = pd.Series([100.0, 50.0])
        ask_size = pd.Series([50.0, 100.0])
        imb = compute_ob_imbalance(bid_size, ask_size)
        assert imb.iloc[0] > 0  # More bids
        assert imb.iloc[1] < 0  # More asks
        assert (imb.abs() <= 1).all()


# ---------- Block 4: Derivatives ----------

class TestDerivatives:
    def test_funding_zscore(self):
        rng = np.random.default_rng(42)
        funding = pd.Series(rng.normal(0.0001, 0.00005, 3000))
        z = compute_funding_zscore(funding, horizon_bars=2016)
        valid = z.dropna()
        assert len(valid) > 0

    def test_funding_change(self):
        funding = pd.Series([0.0001, 0.0002, 0.0003, 0.0004])
        change = compute_funding_change(funding, horizon_bars=2)
        assert change.iloc[2] == pytest.approx(0.0002)

    def test_oi_change(self):
        oi = pd.Series([1e6, 1.1e6, 1.2e6, 1.0e6])
        change = compute_oi_change(oi, horizon_bars=1)
        assert change.iloc[1] == pytest.approx(0.1, rel=0.01)
        assert change.iloc[3] < 0

    def test_oi_to_volume(self):
        oi = pd.Series([1e6, 2e6])
        vol = pd.Series([1e5, 1e5])
        ratio = compute_oi_to_volume(oi, vol)
        assert ratio.iloc[0] == 10.0

    def test_basis(self):
        futures = pd.Series([50100.0, 50200.0])
        spot = pd.Series([50000.0, 50000.0])
        b = compute_basis(futures, spot)
        assert b.iloc[0] == pytest.approx(0.002, rel=0.01)


# ---------- Block 5: Risk ----------

class TestRisk:
    def test_realized_vol(self):
        prices = _make_prices(100)
        rvol = compute_realized_vol(prices, horizon_bars=12)
        valid = rvol.dropna()
        assert (valid >= 0).all()

    def test_downside_vol(self):
        prices = _make_prices(500)
        dvol = compute_downside_vol(prices, horizon_bars=100)
        rvol = compute_realized_vol(prices, horizon_bars=100)
        # Downside vol should be <= total vol (or close)
        both = pd.DataFrame({"dvol": dvol, "rvol": rvol}).dropna()
        assert (both["dvol"] <= both["rvol"] * 1.1).all()

    def test_corr_btc(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2025-01-01", periods=500, freq="5min", tz="UTC")
        btc = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        asset = btc * 1.5 + rng.normal(0, 0.001, 500)  # High correlation
        corr = compute_corr_btc(asset, btc, horizon_bars=100)
        valid = corr.dropna()
        assert valid.iloc[-1] > 0.8

    def test_btc_trend(self):
        idx = pd.date_range("2025-01-01", periods=100, freq="5min", tz="UTC")
        btc = pd.Series(np.linspace(50000, 51000, 100), index=idx)
        trend = compute_btc_trend(btc, horizon_bars=48)
        assert trend.iloc[-1] > 0

    def test_alt_breadth(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2025-01-01", periods=100, freq="5min", tz="UTC")
        panel = pd.DataFrame(rng.normal(0, 0.01, (100, 10)), index=idx)
        breadth = compute_alt_breadth(panel, horizon_bars=20)
        valid = breadth.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


# ---------- Preprocessing ----------

class TestPreprocessing:
    def test_zscore_zero_mean(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]})
        z = cross_sectional_zscore(df)
        # Each row should have mean ~0
        assert z.mean(axis=1).abs().max() < 1e-10

    def test_clip(self):
        df = pd.DataFrame({"A": [-10.0, 0.0, 10.0]})
        clipped = clip_features(df, 5.0)
        assert clipped["A"].max() == 5.0
        assert clipped["A"].min() == -5.0

    def test_fill_median(self):
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]})
        filled = fill_missing_cross_sectional_median(df)
        assert not filled.isna().any().any()
        # NaN at row 1, col A should be filled with median(NaN, 5, 8) = 6.5
        assert filled.loc[1, "A"] == 6.5

    def test_coverage_check(self):
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [1.0, np.nan, np.nan, np.nan, np.nan],
        })
        valid = check_coverage(df, min_coverage=0.80)
        assert "A" in valid
        assert "B" not in valid

    def test_full_pipeline(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(0, 1, (100, 10)))
        # Add some NaN
        df.iloc[0, 0] = np.nan
        result = preprocess_features(df, clip_val=5.0, min_coverage=0.5)
        assert not result.isna().any().any()
        assert result.max().max() <= 5.0
        assert result.min().min() >= -5.0


# ---------- Registry ----------

class TestRegistry:
    def test_load_manifest(self):
        specs = load_manifest()
        assert len(specs) == 40

    def test_blocks_coverage(self):
        specs = load_manifest()
        blocks = {s.block for s in specs}
        assert blocks == {"relative", "trend", "liquidity", "derivatives", "risk"}

    def test_each_block_has_8(self):
        specs = load_manifest()
        for block in ["relative", "trend", "liquidity", "derivatives", "risk"]:
            count = len(get_features_by_block(specs, block))
            assert count == 8, f"Block {block} has {count} features, expected 8"

    def test_params_parsed(self):
        specs = load_manifest()
        beta_1w = next(s for s in specs if s.name == "beta_btc_1w")
        assert beta_1w.params["window_weeks"] == "1"
