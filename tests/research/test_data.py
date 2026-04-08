"""Tests for research data pipeline."""

import asyncio

import numpy as np
import pandas as pd
import pytest

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
from src.research.data.fetcher import (
    fetch_research_data,
    funding_to_dataframe,
    mark_price_to_dataframe,
    orderbook_to_dataframe,
    orderbook_to_record,
    resolve_orderbook_timestamp_ms,
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


class TestFetcherTransforms:
    def test_funding_dataframe_preserves_mark_price(self):
        raw = [
            {
                "timestamp": 1735689600000,
                "fundingRate": 0.0001,
                "markPrice": 100.5,
                "indexPrice": 100.0,
            }
        ]
        df = funding_to_dataframe(raw)
        assert list(df.columns) == ["funding_rate", "mark_price", "index_price"]
        assert df.iloc[0]["mark_price"] == 100.5

    def test_mark_price_dataframe_extracts_price_columns(self):
        raw = [
            {
                "timestamp": 1735689600000,
                "fundingRate": 0.0001,
                "markPrice": 100.5,
                "indexPrice": 100.0,
            },
            {
                "timestamp": 1735689900000,
                "fundingRate": 0.0002,
                "markPrice": None,
                "indexPrice": None,
            },
        ]
        df = mark_price_to_dataframe(raw)
        assert list(df.columns) == ["mark_price", "index_price"]
        assert len(df) == 1
        assert df.iloc[0]["index_price"] == 100.0

    def test_orderbook_dataframe_contains_top_of_book(self):
        ob = {
            "timestamp": 1735689600000,
            "bids": [[100.0, 3.0], [99.9, 2.0]],
            "asks": [[100.2, 4.0], [100.3, 1.0]],
        }
        ts_ms, ts_source = resolve_orderbook_timestamp_ms(ob, 1735689900000)
        record = orderbook_to_record(ob, ts_ms, timestamp_source=ts_source)
        df = orderbook_to_dataframe(ob, ts_ms, timestamp_source=ts_source)
        assert record["best_bid"] == 100.0
        assert record["best_ask_size"] == 4.0
        assert record["timestamp_source"] == "exchange_response"
        assert df.iloc[0]["spread_bps"] == pytest.approx(19.98, rel=0.01)

    def test_orderbook_timestamp_falls_back_to_exchange_clock(self):
        ob = {"bids": [[100.0, 1.0]], "asks": [[100.1, 1.5]]}
        ts_ms, ts_source = resolve_orderbook_timestamp_ms(ob, 1735689999000)
        assert ts_ms == 1735689999000
        assert ts_source == "exchange_clock"


class TestFetchResearchData:
    def test_persists_mark_price_and_orderbook_outputs(self, tmp_path, monkeypatch):
        from src.research.data import fetcher as fetcher_module

        class FakeExchange:
            def __init__(self, *_args, **_kwargs):
                self.markets = {}

            async def load_markets(self):
                return None

            async def close(self):
                return None

            async def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
                assert symbol == "BTC/USDT:USDT"
                assert timeframe == "1m"
                return [
                    [1735689600000, 100.0, 101.0, 99.5, 100.5, 10.0],
                    [1735689660000, 100.5, 101.5, 100.0, 101.0, 12.0],
                ]

            async def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
                assert symbol == "BTC/USDT:USDT"
                return [
                    {
                        "timestamp": 1735689600000,
                        "fundingRate": 0.0001,
                        "markPrice": 100.5,
                        "indexPrice": 100.0,
                    }
                ]

            async def fetch_open_interest_history(self, symbol, timeframe="5m", since=None, limit=500):
                assert symbol == "BTC/USDT:USDT"
                assert timeframe == "5m"
                return [
                    {
                        "timestamp": 1735689600000,
                        "openInterestAmount": 1000.0,
                        "openInterestValue": 100500.0,
                    }
                ]

            async def fetch_order_book(self, symbol, limit=5):
                assert symbol == "BTC/USDT:USDT"
                assert limit == 5
                return {
                    "timestamp": 1735689700000,
                    "bids": [[100.0, 3.0]],
                    "asks": [[100.2, 4.0]],
                }

            def milliseconds(self):
                return 1735689600000

        monkeypatch.setattr(fetcher_module.ccxt_async, "fake_exchange", FakeExchange, raising=False)

        counts = asyncio.run(
            fetch_research_data(
                exchange_id="fake_exchange",
                symbols=["BTC/USDT:USDT"],
                since_ms=1735689600000,
                storage_path=tmp_path,
            )
        )

        assert counts["ohlcv_1m"] == 2
        assert counts["funding"] == 1
        assert counts["oi"] == 1
        assert counts["mark_price"] == 1
        assert counts["orderbook_top1"] == 1

        safe = "BTCUSDT_USDT"
        mark_price_path = tmp_path / "mark_price" / f"{safe}.parquet"
        orderbook_path = tmp_path / "orderbook_top1" / f"{safe}.parquet"
        assert mark_price_path.exists()
        assert orderbook_path.exists()

        mark_price_df = pd.read_parquet(mark_price_path)
        orderbook_df = pd.read_parquet(orderbook_path)
        assert list(mark_price_df.columns) == ["mark_price", "index_price"]
        assert orderbook_df.iloc[0]["best_bid"] == 100.0
        assert orderbook_df.iloc[0]["timestamp_source"] == "exchange_response"
        assert orderbook_df.index[0] == pd.Timestamp(1735689700000, unit="ms", tz="UTC")
