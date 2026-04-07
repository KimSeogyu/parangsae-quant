from __future__ import annotations

import pandas as pd
import pytest

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from src.alpha.momentum import MomentumAlpha, MomentumAlphaConfig
from src.data.catalog import load_bars_from_parquet
from src.portfolio.construction import PortfolioConstruction, PortfolioConstructionConfig
from src.risk.model import RiskModel, RiskModelConfig
from src.universe.model import UniverseModel, UniverseModelConfig


def _make_ohlcv(
    hours: int,
    base_price: float,
    trend: float,
    volume: float,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a linear trend."""
    dates = pd.date_range("2025-01-01", periods=hours, freq="1h", tz="UTC")
    prices = [base_price + trend * i for i in range(hours)]
    return pd.DataFrame(
        {
            "open": [p - 10 for p in prices],
            "high": [p + 50 for p in prices],
            "low": [p - 50 for p in prices],
            "close": prices,
            "volume": [volume] * hours,
            "quote_volume": [p * volume for p in prices],
        },
        index=dates,
    )


@pytest.fixture
def synthetic_data(tmp_path):
    """Create synthetic parquet files for BTC (trending up) and ETH (flat)."""
    spot_dir = tmp_path / "spot"
    spot_dir.mkdir()

    btc = _make_ohlcv(hours=500, base_price=50000, trend=10, volume=100.0)
    btc.to_parquet(spot_dir / "BTCUSDT.parquet")

    eth = _make_ohlcv(hours=500, base_price=3000, trend=0, volume=50.0)
    eth.to_parquet(spot_dir / "ETHUSDT.parquet")

    return tmp_path


def test_backtest_runs_without_error(synthetic_data):
    """End-to-end integration test: full pipeline runs and processes bars."""
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="WARNING"),
        ),
    )

    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(100_000, USDT)],
    )

    btc = TestInstrumentProvider.btcusdt_binance()
    eth = TestInstrumentProvider.ethusdt_binance()
    engine.add_instrument(btc)
    engine.add_instrument(eth)

    for symbol, instrument in [("BTCUSDT", btc), ("ETHUSDT", eth)]:
        bars = load_bars_from_parquet(
            path=str(synthetic_data / "spot" / f"{symbol}.parquet"),
            instrument=instrument,
        )
        engine.add_data(bars, sort=False)
    engine.sort_data()

    engine.add_actor(
        UniverseModel(
            UniverseModelConfig(
                ranking_window=48,
                inclusion_rank=5,
                exclusion_rank=10,
            ),
        ),
    )

    engine.add_actor(
        MomentumAlpha(
            MomentumAlphaConfig(
                alpha_name="momentum",
                params={"lookback": 24},
            ),
        ),
    )

    engine.add_actor(
        RiskModel(
            RiskModelConfig(
                max_position_pct=0.3,
                max_drawdown=0.5,
                max_total_exposure=1.0,
                volatility_window=48,
            ),
        ),
    )

    engine.add_strategy(
        PortfolioConstruction(
            PortfolioConstructionConfig(
                rebalance_interval_hours=1,
                min_trade_threshold=0.001,
                max_position_pct=0.3,
                max_total_exposure=1.0,
                alpha_weights={"momentum": 1.0},
                order_id_tag="TEST001",
            ),
        ),
    )

    engine.run()
    assert engine.iteration > 0
    engine.dispose()
