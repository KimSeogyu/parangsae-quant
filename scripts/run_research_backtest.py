"""Entry point for the crypto-native residual ridge research backtest.

Usage:
    uv run python scripts/run_research_backtest.py [--mode fast|accurate]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.config import load_research_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run research backtest")
    parser.add_argument("--mode", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--data-path", default="data/research")
    args = parser.parse_args()

    settings = load_research_settings()
    logger.info(f"Running {args.mode} mode backtest")
    logger.info(f"Universe: top {settings.universe.liquidity.top_n} coins")
    logger.info(f"Model: Ridge with {settings.model.training.train_window_days}d train window")
    logger.info(f"Labels: {settings.model.labels.primary_weight}*y20 + "
                f"{settings.model.labels.secondary_weight}*y60")

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path {data_path} does not exist. Run data fetcher first.")
        logger.info("Usage: uv run python scripts/fetch_research_data.py")
        sys.exit(1)

    logger.info("Research backtest pipeline ready. Implement data loading to proceed.")
    logger.info("Pipeline: Data → Universe → Features → Labels → Ridge → Portfolio → Backtest")


if __name__ == "__main__":
    main()
