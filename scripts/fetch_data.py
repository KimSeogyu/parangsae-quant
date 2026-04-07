from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.config.loader import load_settings
from src.data.fetcher import fetch_all_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    settings = load_settings()
    since = datetime.now(timezone.utc) - timedelta(days=settings.data.history_days)
    since_ms = int(since.timestamp() * 1000)

    for market_type in settings.data.market_types:
        logging.info(f"Fetching {market_type} data...")
        asyncio.run(
            fetch_all_symbols(
                exchange_id=settings.data.exchange,
                market_type=market_type,
                timeframe=settings.data.timeframe,
                since_ms=since_ms,
                storage_path=settings.data.storage_path,
            )
        )

    logging.info("Data fetch complete.")


if __name__ == "__main__":
    main()
