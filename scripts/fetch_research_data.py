"""Fetch extended research data (1m OHLCV, funding, OI) for the research pipeline.

Usage:
    uv run python scripts/fetch_research_data.py
"""

from __future__ import annotations

import asyncio
import logging

from src.research.config import load_research_settings
from src.research.data.fetcher import fetch_research_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main():
    settings = load_research_settings()
    logger.info("Fetching research data for crypto-native residual ridge pipeline")
    logger.info(f"Target: top {settings.universe.liquidity.top_n} coins")

    # Calculate since_ms from training window requirement
    import time
    days_needed = settings.model.training.train_window_days + 30  # Extra buffer
    since_ms = int((time.time() - days_needed * 86400) * 1000)

    counts = await fetch_research_data(
        exchange_id="binance",
        since_ms=since_ms,
    )

    logger.info(f"Fetched: {counts}")


if __name__ == "__main__":
    asyncio.run(main())
