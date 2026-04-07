from __future__ import annotations

import importlib
import sys
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money

from src.config.models import Settings
from src.data.catalog import load_bars_from_parquet
from src.data.instruments import load_instruments
from src.portfolio.construction import PortfolioConstruction, PortfolioConstructionConfig
from src.risk.model import RiskModel, RiskModelConfig
from src.universe.model import UniverseModel, UniverseModelConfig


_ALLOWED_MODULE_PREFIXES = ("src.alpha.",)


def _import_class(module_path: str):
    """Dynamically import a class from an allowed module path like 'src.alpha.momentum.MomentumAlpha'.

    Only modules under the allowed prefixes (src.alpha.*) can be loaded
    to prevent arbitrary code execution from untrusted config values.
    """
    module_name, _, class_name = module_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid module path: {module_path!r}")
    if not any(module_name.startswith(prefix) for prefix in _ALLOWED_MODULE_PREFIXES):
        raise ValueError(
            f"Module {module_name!r} is not in the allowed prefixes: {_ALLOWED_MODULE_PREFIXES}"
        )
    module = importlib.import_module(module_name)  # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
    return getattr(module, class_name)


def _create_instrument_fallback(symbol: str, market_type: str):
    """Fallback: create instrument from TestInstrumentProvider for known symbols."""
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    provider_map = {
        ("BTCUSDT", "spot"): TestInstrumentProvider.btcusdt_binance,
        ("ETHUSDT", "spot"): TestInstrumentProvider.ethusdt_binance,
        ("ADAUSDT", "spot"): TestInstrumentProvider.adausdt_binance,
        ("BTCUSDT-PERP", "futures"): TestInstrumentProvider.btcusdt_perp_binance,
        ("ETHUSDT-PERP", "futures"): TestInstrumentProvider.ethusdt_perp_binance,
    }
    factory = provider_map.get((symbol, market_type))
    if factory:
        return factory()
    return None


def build_backtest_engine(settings: Settings) -> BacktestEngine:
    """Assemble a fully configured BacktestEngine from a Settings object."""
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level=settings.system.log_level),
        ),
    )

    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(settings.portfolio.initial_capital, USDT)],
    )

    # Load instruments and bar data from parquet files
    for market_type in settings.data.market_types:
        data_dir = Path(settings.data.storage_path) / market_type
        if not data_dir.exists():
            continue

        # Try dynamic instruments from saved market info, fall back to test providers
        dynamic_instruments = load_instruments(settings.data.storage_path, market_type)

        for parquet_file in sorted(data_dir.glob("*.parquet")):
            if parquet_file.stem.startswith("_"):
                continue  # skip metadata files like _instruments.json
            symbol = parquet_file.stem
            instrument = dynamic_instruments.get(symbol) or _create_instrument_fallback(
                symbol, market_type
            )
            if instrument is None:
                continue
            engine.add_instrument(instrument)
            bars = load_bars_from_parquet(path=str(parquet_file), instrument=instrument)
            engine.add_data(bars, sort=False)

    engine.sort_data()

    # Universe Model
    engine.add_actor(
        UniverseModel(
            UniverseModelConfig(
                ranking_window=settings.universe.ranking_window,
                inclusion_rank=settings.universe.inclusion_rank,
                exclusion_rank=settings.universe.exclusion_rank,
            ),
        ),
    )

    # Alpha Models (dynamically loaded from config)
    alpha_weights: dict[str, float] = {}
    for alpha_name, alpha_cfg in settings.alphas.items():
        if not alpha_cfg.enabled:
            continue

        AlphaClass = _import_class(alpha_cfg.module)

        # Find the matching Config class in the same module (already imported by _import_class)
        config_module = sys.modules[AlphaClass.__module__]
        ConfigClass = None
        for attr_name in dir(config_module):
            attr = getattr(config_module, attr_name)
            if (
                isinstance(attr, type)
                and attr_name.endswith("Config")
                and attr_name != "BaseAlphaConfig"
            ):
                ConfigClass = attr
                break

        if ConfigClass is None:
            from src.alpha.base import BaseAlphaConfig

            ConfigClass = BaseAlphaConfig

        actor_config = ConfigClass(alpha_name=alpha_name, params=alpha_cfg.params)
        engine.add_actor(AlphaClass(actor_config))
        alpha_weights[alpha_name] = alpha_cfg.weight

    # Risk Model
    engine.add_actor(
        RiskModel(
            RiskModelConfig(
                max_position_pct=settings.risk.max_position_pct,
                max_drawdown=settings.risk.max_drawdown,
                max_total_exposure=settings.risk.max_total_exposure,
                volatility_window=settings.risk.volatility_window,
            ),
        ),
    )

    # Portfolio Construction Strategy
    engine.add_strategy(
        PortfolioConstruction(
            PortfolioConstructionConfig(
                rebalance_interval_hours=settings.portfolio.rebalance_interval_hours,
                min_trade_threshold=settings.portfolio.min_trade_threshold,
                max_position_pct=settings.risk.max_position_pct,
                max_total_exposure=settings.risk.max_total_exposure,
                alpha_weights=alpha_weights,
                order_id_tag="PC001",
            ),
        ),
    )

    return engine
