from __future__ import annotations

import logging

from nautilus_trader.backtest.engine import BacktestEngine

from src.config.loader import load_settings
from src.engine.factory import build_backtest_engine

logger = logging.getLogger(__name__)


def run_backtest(config_path: str = "config/settings.yaml") -> BacktestEngine:
    """Load settings, build the engine, run the backtest, and return the engine."""
    settings = load_settings(config_path)
    logger.info(
        "Running backtest: %s to %s",
        settings.backtest.start_date,
        settings.backtest.end_date,
    )
    engine = build_backtest_engine(settings)
    engine.run()
    logger.info("Backtest complete.")
    return engine
