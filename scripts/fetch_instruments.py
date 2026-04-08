"""Fetch instrument metadata from Binance via CCXT and save to data/{spot,futures}/_instruments.json.

This creates the metadata that factory.py needs to build Nautilus instruments
for all symbols in our Parquet data.
"""

from __future__ import annotations

import asyncio
import logging

import ccxt.async_support as ccxt_async

from src.data.instruments import save_market_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    exchange = ccxt_async.binance({"enableRateLimit": True})
    try:
        await exchange.load_markets()
        logger.info(f"Loaded {len(exchange.markets)} markets from Binance")

        save_market_info(exchange.markets, "spot", "data")
        save_market_info(exchange.markets, "futures", "data")
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
