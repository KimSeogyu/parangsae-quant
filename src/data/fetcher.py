from __future__ import annotations

import logging
from pathlib import Path

import ccxt.async_support as ccxt_async
import pandas as pd

logger = logging.getLogger(__name__)


def ohlcv_to_dataframe(raw: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df["quote_volume"] = df["close"] * df["volume"]
    return df


def save_ohlcv(df: pd.DataFrame, path: str) -> None:
    path = Path(path)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


async def fetch_all_symbols(
    exchange_id: str,
    market_type: str,
    timeframe: str,
    since_ms: int,
    storage_path: str,
) -> None:
    exchange_class = getattr(ccxt_async, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    try:
        await exchange.load_markets()
        symbols = [
            s
            for s, m in exchange.markets.items()
            if m["active"]
            and m["quote"] == "USDT"
            and (
                (market_type == "spot" and m["spot"])
                or (market_type == "futures" and m["swap"] and m["linear"])
            )
        ]
        logger.info(f"Found {len(symbols)} {market_type} symbols")

        for symbol in symbols:
            safe_name = symbol.replace("/", "").replace(":", "-")
            out_path = Path(storage_path) / market_type / f"{safe_name}.parquet"

            symbol_since = since_ms
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                if len(existing) > 0:
                    last_ts = existing.index.max()
                    symbol_since = int(last_ts.timestamp() * 1000) + 1

            all_ohlcv = []
            current_since = symbol_since
            while True:
                try:
                    ohlcv = await exchange.fetch_ohlcv(
                        symbol, timeframe, since=current_since, limit=1000
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")
                    break

                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1

                if len(ohlcv) < 1000:
                    break

            if all_ohlcv:
                df = ohlcv_to_dataframe(all_ohlcv)
                save_ohlcv(df, str(out_path))
                logger.info(f"Saved {len(df)} bars for {symbol}")

    finally:
        await exchange.close()
