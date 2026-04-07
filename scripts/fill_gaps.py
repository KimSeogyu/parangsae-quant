"""Fill data gaps: supplement archive Parquet data with Binance API via CCXT.

Concurrent fetching with asyncio.Semaphore for rate-limited parallelism.
Multiple exchange instances to avoid single-connection bottleneck.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt.async_support as ccxt_async
import pandas as pd

from src.data.fetcher import ohlcv_to_dataframe, save_ohlcv
from src.data.instruments import ccxt_symbol_to_file_name, save_market_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_DAYS = 365
STORAGE_PATH = Path("data")
TIMEFRAME = "1h"
MAX_CONCURRENT = 3  # keep low to avoid Binance IP ban


async def fetch_range(
    exchange, ccxt_symbol: str, since_ms: int, until_ms: int, timeframe: str
) -> list[list]:
    """Fetch OHLCV between since_ms and until_ms with retry."""
    all_ohlcv = []
    current = since_ms
    while current < until_ms:
        for attempt in range(3):
            try:
                ohlcv = await exchange.fetch_ohlcv(
                    ccxt_symbol, timeframe, since=current, limit=1000
                )
                break
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"  Failed {ccxt_symbol} after 3 retries: {e}")
                    return all_ohlcv
                wait = 2 ** (attempt + 1)
                logger.debug(f"  Retry {ccxt_symbol} in {wait}s: {e}")
                await asyncio.sleep(wait)
        if not ohlcv:
            break
        ohlcv = [bar for bar in ohlcv if bar[0] < until_ms]
        all_ohlcv.extend(ohlcv)
        if len(ohlcv) < 1000:
            break
        current = ohlcv[-1][0] + 1
    return all_ohlcv


async def fill_one_symbol(
    sem: asyncio.Semaphore,
    exchange,
    ccxt_sym: str,
    file_name: str,
    data_dir: Path,
    target_start_ms: int,
    target_end_ms: int,
    progress: dict,
) -> bool:
    """Fill gaps for a single symbol. Returns True if data was fetched."""
    async with sem:
        parquet_path = data_dir / f"{file_name}.parquet"

        existing_start_ms = None
        existing_end_ms = None
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                if len(df) > 0:
                    existing_start_ms = int(df.index.min().timestamp() * 1000)
                    existing_end_ms = int(df.index.max().timestamp() * 1000)
            except Exception:
                pass

        fetch_ranges = []
        if existing_start_ms is None:
            fetch_ranges.append((target_start_ms, target_end_ms))
        else:
            if existing_start_ms > target_start_ms + 3_600_000:
                fetch_ranges.append((target_start_ms, existing_start_ms))
            if existing_end_ms < target_end_ms - 3_600_000:
                fetch_ranges.append((existing_end_ms + 1, target_end_ms))

        if not fetch_ranges:
            progress["skipped"] += 1
            progress["done"] += 1
            return False

        hours_missing = sum((e - s) / 3_600_000 for s, e in fetch_ranges)
        progress["done"] += 1
        logger.info(
            f"  [{progress['done']}/{progress['total']}] {file_name}: "
            f"{hours_missing:.0f}h gap(s)"
        )

        for since_ms, until_ms in fetch_ranges:
            ohlcv = await fetch_range(exchange, ccxt_sym, since_ms, until_ms, TIMEFRAME)
            if ohlcv:
                df = ohlcv_to_dataframe(ohlcv)
                save_ohlcv(df, str(parquet_path))

        progress["filled"] += 1
        return True


async def fill_market_gaps(
    markets: dict,
    market_type: str,
    target_start_ms: int,
    target_end_ms: int,
) -> None:
    """Fill gaps for all symbols in a market type using concurrent workers."""
    data_dir = STORAGE_PATH / market_type
    data_dir.mkdir(parents=True, exist_ok=True)

    if market_type == "spot":
        symbols = {
            s: m for s, m in markets.items()
            if m["active"] and m["quote"] == "USDT" and m["spot"]
        }
    else:
        symbols = {
            s: m for s, m in markets.items()
            if m["active"] and m["quote"] == "USDT" and m.get("swap") and m.get("linear")
        }

    to_process = [
        (ccxt_sym, ccxt_symbol_to_file_name(ccxt_sym, market_type))
        for ccxt_sym in symbols
    ]
    logger.info(f"[{market_type}] {len(to_process)} symbols to process")

    # Share a single exchange instance — asyncio is single-threaded so this is safe
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    exchange = ccxt_async.binance({"enableRateLimit": True})
    exchange.markets = markets  # reuse already-loaded markets, no extra API call

    progress = {"done": 0, "total": len(to_process), "filled": 0, "skipped": 0}

    tasks = []
    for ccxt_sym, file_name in to_process:
        tasks.append(
            fill_one_symbol(
                sem, exchange, ccxt_sym, file_name, data_dir,
                target_start_ms, target_end_ms, progress,
            )
        )

    await asyncio.gather(*tasks, return_exceptions=True)
    await exchange.close()

    logger.info(
        f"[{market_type}] Done: {progress['filled']} filled, "
        f"{progress['skipped']} already complete"
    )


async def main() -> None:
    target_end = datetime.now(timezone.utc)
    target_start = target_end - timedelta(days=TARGET_DAYS)
    target_start_ms = int(target_start.timestamp() * 1000)
    target_end_ms = int(target_end.timestamp() * 1000)

    logger.info(f"Target: {target_start.date()} to {target_end.date()} ({TARGET_DAYS}d)")
    logger.info(f"Concurrency: {MAX_CONCURRENT} parallel workers")

    # Load markets once
    exchange = ccxt_async.binance({"enableRateLimit": True})
    try:
        await exchange.load_markets()
        markets = exchange.markets
        logger.info(f"Loaded {len(markets)} markets")

        save_market_info(markets, "spot", str(STORAGE_PATH))
        save_market_info(markets, "futures", str(STORAGE_PATH))
    finally:
        await exchange.close()

    # Fill both markets
    for market_type in ["futures", "spot"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {market_type}")
        logger.info(f"{'='*60}")
        await fill_market_gaps(markets, market_type, target_start_ms, target_end_ms)

    # Summary
    for mt in ["spot", "futures"]:
        data_dir = STORAGE_PATH / mt
        files = [f for f in data_dir.glob("*.parquet") if not f.stem.startswith("_")]
        total_bars = 0
        for f in files:
            try:
                df = pd.read_parquet(f)
                total_bars += len(df)
            except Exception:
                pass
        logger.info(f"{mt}: {len(files)} symbols, {total_bars:,} total bars")


if __name__ == "__main__":
    asyncio.run(main())
