"""Tests for research data pipeline."""

import numpy as np
import pandas as pd

from src.research.data.aggregator import (
    aggregate_ohlcv,
    compute_log_returns,
    compute_mid_price,
    compute_vwap,
)
from src.research.data.alignment import (
    build_walk_forward_windows,
    embargo_split,
)


def _make_1m_bars(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.05, n),
            "high": close + rng.uniform(0, 0.1, n),
            "low": close - rng.uniform(0, 0.1, n),
            "close": close,
            "volume": rng.uniform(10, 100, n),
            "quote_volume": close * rng.uniform(10, 100, n),
        },
        index=idx,
    )


class TestAggregation:
    def test_aggregate_5m(self):
        df = _make_1m_bars(100)
        agg = aggregate_ohlcv(df, 5)
        assert len(agg) == 20
        # First 5m bar open should equal first 1m bar open
        assert agg.iloc[0]["open"] == df.iloc[0]["open"]
        # 5m high should be max of first 5 bars
        assert agg.iloc[0]["high"] == df.iloc[:5]["high"].max()
        # Volume should sum
        assert abs(agg.iloc[0]["volume"] - df.iloc[:5]["volume"].sum()) < 1e-6

    def test_aggregate_20m(self):
        df = _make_1m_bars(100)
        agg = aggregate_ohlcv(df, 20)
        assert len(agg) == 5

    def test_aggregate_preserves_close(self):
        df = _make_1m_bars(10)
        agg = aggregate_ohlcv(df, 5)
        # Close of 5m bar should equal close of 5th 1m bar
        assert agg.iloc[0]["close"] == df.iloc[4]["close"]


class TestMidPrice:
    def test_from_bid_ask(self):
        df = pd.DataFrame({"best_bid": [100.0, 200.0], "best_ask": [101.0, 201.0]})
        mid = compute_mid_price(df)
        assert mid.iloc[0] == 100.5

    def test_fallback_high_low(self):
        df = pd.DataFrame({"high": [105.0], "low": [95.0], "close": [100.0]})
        mid = compute_mid_price(df)
        assert mid.iloc[0] == 100.0


class TestLogReturns:
    def test_basic(self):
        prices = pd.Series([100.0, 110.0, 105.0])
        rets = compute_log_returns(prices)
        assert pd.isna(rets.iloc[0])
        assert abs(rets.iloc[1] - np.log(1.1)) < 1e-10


class TestVWAP:
    def test_basic(self):
        df = pd.DataFrame(
            {"close": [100.0, 200.0], "volume": [10.0, 20.0], "quote_volume": [1000.0, 4000.0]}
        )
        vwap = compute_vwap(df)
        # After bar 2: cumulative qvol = 5000, cumulative vol = 30
        assert abs(vwap.iloc[1] - 5000 / 30) < 1e-6


class TestEmbargoSplit:
    def test_embargo_gap(self):
        idx = pd.date_range("2025-01-01", periods=1000, freq="5min", tz="UTC")
        train_end = pd.Timestamp("2025-01-02", tz="UTC")
        val_start = pd.Timestamp("2025-01-02", tz="UTC")
        train_idx, val_idx = embargo_split(idx, train_end, val_start, embargo_minutes=60)
        # Validation should start 60 min after val_start
        assert val_idx.min() >= val_start + pd.Timedelta(minutes=60)
        # Train should end at or before train_end
        assert train_idx.max() <= train_end


class TestWalkForward:
    def test_windows_no_overlap(self):
        idx = pd.date_range("2025-01-01", periods=365 * 288, freq="5min", tz="UTC")
        windows = build_walk_forward_windows(idx, train_days=90, val_days=14, step_days=1)
        assert len(windows) > 0
        # Check first window
        w = windows[0]
        assert w["train_end"] < w["val_start"]  # embargo gap
        assert w["val_end"] <= w["test_start"]
        # Check no overlap between consecutive windows' test periods
        for i in range(len(windows) - 1):
            assert windows[i]["test_end"] <= windows[i + 1]["test_start"]
