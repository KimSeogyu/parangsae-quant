"""Tests for universe filter."""

from datetime import datetime

from src.research.config import UniverseSettings
from src.research.universe.filter import (
    build_universe,
    compute_reconstitution_dates,
    filter_by_data_coverage,
    filter_by_listing_age,
    filter_by_spread,
    get_symbol_sector,
    is_native_crypto,
    rank_by_liquidity,
)

import pandas as pd


class TestNativeCrypto:
    def test_btc_is_native(self):
        assert is_native_crypto("BTCUSDT-PERP", "BTC") is True

    def test_stablecoin_excluded(self):
        assert is_native_crypto("USDCUSDT-PERP", "USDC") is False

    def test_gold_token_excluded(self):
        assert is_native_crypto("PAXGUSDT-PERP", "PAXG") is False

    def test_leveraged_token_excluded(self):
        assert is_native_crypto("BTCUP", "BTCUP") is False
        assert is_native_crypto("ETHDOWN", "ETHDOWN") is False
        assert is_native_crypto("SOL3L", "SOL3L") is False

    def test_excluded_symbols_list(self):
        assert is_native_crypto("USDTUSDT", "USDT", excluded_symbols=["USDTUSDT"]) is False

    def test_normal_alt_passes(self):
        assert is_native_crypto("SOLUSDT-PERP", "SOL") is True


class TestListingAge:
    def test_180_day_filter(self):
        ref = datetime(2025, 7, 1)
        listing = {"BTC": datetime(2020, 1, 1), "NEW": datetime(2025, 6, 1)}
        result = filter_by_listing_age(["BTC", "NEW"], listing, ref, 180)
        assert result == ["BTC"]

    def test_missing_listing_date(self):
        ref = datetime(2025, 7, 1)
        result = filter_by_listing_age(["BTC"], {}, ref, 180)
        assert result == []


class TestDataCoverage:
    def test_coverage_filter(self):
        coverage = {"BTC": 0.99, "ETH": 0.90, "SOL": 0.96}
        result = filter_by_data_coverage(["BTC", "ETH", "SOL"], coverage, 0.95)
        assert "BTC" in result
        assert "SOL" in result
        assert "ETH" not in result


class TestLiquidityRank:
    def test_top_n(self):
        vols = {"BTC": 1e9, "ETH": 5e8, "SOL": 2e8, "DOGE": 1e8}
        result = rank_by_liquidity(list(vols.keys()), vols, top_n=2)
        assert result == ["BTC", "ETH"]


class TestSpreadFilter:
    def test_wide_spread_excluded(self):
        spreads = {"BTC": 1.0, "ETH": 2.0, "JUNK": 50.0}
        result = filter_by_spread(list(spreads.keys()), spreads, max_spread_bps=20.0)
        assert "JUNK" not in result
        assert "BTC" in result


class TestBuildUniverse:
    def test_full_pipeline(self):
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "USDCUSDT", "NEWUSDT", "JUNKUSDT"]
        bases = {
            "BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL",
            "USDCUSDT": "USDC", "NEWUSDT": "NEW", "JUNKUSDT": "JUNK",
        }
        listings = {
            "BTCUSDT": datetime(2020, 1, 1),
            "ETHUSDT": datetime(2020, 1, 1),
            "SOLUSDT": datetime(2020, 6, 1),
            "USDCUSDT": datetime(2020, 1, 1),
            "NEWUSDT": datetime(2025, 6, 1),
            "JUNKUSDT": datetime(2020, 1, 1),
        }
        volumes = {
            "BTCUSDT": 1e9, "ETHUSDT": 5e8, "SOLUSDT": 2e8,
            "USDCUSDT": 3e8, "NEWUSDT": 1e7, "JUNKUSDT": 5e7,
        }
        coverage = {
            "BTCUSDT": 0.99, "ETHUSDT": 0.98, "SOLUSDT": 0.96,
            "USDCUSDT": 0.99, "NEWUSDT": 0.50, "JUNKUSDT": 0.80,
        }
        spreads = {
            "BTCUSDT": 1.0, "ETHUSDT": 2.0, "SOLUSDT": 3.0,
            "USDCUSDT": 1.0, "NEWUSDT": 10.0, "JUNKUSDT": 50.0,
        }
        ref = datetime(2025, 7, 1)
        settings = UniverseSettings()

        result = build_universe(
            symbols, bases, listings, volumes, coverage, spreads, ref, settings
        )
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result
        assert "SOLUSDT" in result
        # USDC excluded (stablecoin)
        assert "USDCUSDT" not in result
        # NEW excluded (too new)
        assert "NEWUSDT" not in result
        # JUNK excluded (low coverage + wide spread)
        assert "JUNKUSDT" not in result


class TestReconstitution:
    def test_weekly_mondays(self):
        start = pd.Timestamp("2025-01-01", tz="UTC")
        end = pd.Timestamp("2025-02-01", tz="UTC")
        dates = compute_reconstitution_dates(start, end, "monday", 0)
        assert len(dates) >= 4
        for d in dates:
            assert d.day_name() == "Monday"


class TestSectorMap:
    def test_known_sector(self):
        sector_map = {"l1": ["BTC", "ETH", "SOL"], "defi": ["UNI", "AAVE"]}
        assert get_symbol_sector("BTCUSDT", "BTC", sector_map) == "l1"
        assert get_symbol_sector("UNIUSDT", "UNI", sector_map) == "defi"

    def test_unknown_sector(self):
        sector_map = {"l1": ["BTC"]}
        assert get_symbol_sector("RANDOMUSDT", "RANDOM", sector_map) == "other"
