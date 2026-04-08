"""Extended data fetcher for the research pipeline.

Fetches 1m OHLCV, funding rates, open interest, mark-price history, and
top-of-book snapshots via CCXT. Stores each dataset as parquet under
data/research/.

PRD Section 3: Data Specification
"""

from __future__ import annotations

import logging
from pathlib import Path

import ccxt.async_support as ccxt_async
import pandas as pd

logger = logging.getLogger(__name__)

RESEARCH_DATA_ROOT = Path("data/research")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# OHLCV (1-minute bars)
# ---------------------------------------------------------------------------

def ohlcv_to_dataframe(raw: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df["quote_volume"] = df["close"] * df["volume"]
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
    _ensure_dir(path)
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# Aggregation: 1m -> 5m -> 20m
# ---------------------------------------------------------------------------

def aggregate_bars(df_1m: pd.DataFrame, freq_minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute OHLCV bars to a coarser frequency.

    Uses left-closed intervals aligned to midnight UTC.
    """
    rule = f"{freq_minutes}min"
    agg = df_1m.resample(rule, closed="left", label="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
        }
    )
    return agg.dropna(subset=["open"])


# ---------------------------------------------------------------------------
# Funding Rate
# ---------------------------------------------------------------------------

def funding_to_dataframe(raw: list[dict]) -> pd.DataFrame:
    """Convert CCXT funding rate history to DataFrame."""
    records = []
    for entry in raw:
        records.append(
            {
                "timestamp": pd.Timestamp(entry["timestamp"], unit="ms", tz="UTC"),
                "funding_rate": entry.get("fundingRate", 0.0),
                "mark_price": entry.get("markPrice"),
                "index_price": entry.get("indexPrice"),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.set_index("timestamp").sort_index()
    return df


def mark_price_to_dataframe(raw: list[dict]) -> pd.DataFrame:
    """Extract mark/index price history from funding history payloads."""
    df = funding_to_dataframe(raw)
    if df.empty:
        return df
    return df[["mark_price", "index_price"]].dropna(how="all")


# ---------------------------------------------------------------------------
# Open Interest
# ---------------------------------------------------------------------------

def oi_to_dataframe(raw: list[dict]) -> pd.DataFrame:
    """Convert CCXT open interest history to DataFrame."""
    records = []
    for entry in raw:
        ts = entry.get("timestamp")
        if ts is None:
            continue
        records.append(
            {
                "timestamp": pd.Timestamp(ts, unit="ms", tz="UTC"),
                "open_interest": entry.get("openInterestAmount", 0.0),
                "open_interest_value": entry.get("openInterestValue", 0.0),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# Orderbook Snapshot (top-of-book)
# ---------------------------------------------------------------------------

def orderbook_to_record(
    ob: dict,
    ts_ms: int,
    timestamp_source: str = "exchange_response",
) -> dict:
    """Extract top-of-book from CCXT orderbook response."""
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    return {
        "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
        "timestamp_source": timestamp_source,
        "best_bid": bids[0][0] if bids else None,
        "best_bid_size": bids[0][1] if bids else None,
        "best_ask": asks[0][0] if asks else None,
        "best_ask_size": asks[0][1] if asks else None,
        "spread_bps": (
            (asks[0][0] - bids[0][0]) / ((asks[0][0] + bids[0][0]) / 2) * 10000
            if bids and asks and bids[0][0] > 0
            else None
        ),
    }


def orderbook_to_dataframe(
    ob: dict,
    ts_ms: int,
    timestamp_source: str = "exchange_response",
) -> pd.DataFrame:
    """Convert a single orderbook snapshot to a timestamp-indexed DataFrame."""
    record = orderbook_to_record(ob, ts_ms, timestamp_source=timestamp_source)
    return pd.DataFrame([record]).set_index("timestamp").sort_index()


def resolve_orderbook_timestamp_ms(
    orderbook: dict,
    fallback_ms: int | None = None,
) -> tuple[int, str]:
    """Resolve the best timestamp for an orderbook snapshot.

    Preference order:
    1. Exchange-provided orderbook timestamp
    2. Exchange clock supplied by caller
    3. Current UTC clock
    """
    if orderbook.get("timestamp") is not None:
        return int(orderbook["timestamp"]), "exchange_response"
    if fallback_ms is not None:
        return int(fallback_ms), "exchange_clock"
    return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), "local_clock"


# ---------------------------------------------------------------------------
# Main fetch orchestrator
# ---------------------------------------------------------------------------

async def fetch_research_data(
    exchange_id: str = "binance",
    symbols: list[str] | None = None,
    since_ms: int | None = None,
    storage_path: Path | str = RESEARCH_DATA_ROOT,
) -> dict[str, int]:
    """Fetch all research data types for given symbols.

    Returns dict of {data_type: total_records_saved}.
    """
    storage_path = Path(storage_path)
    exchange_class = getattr(ccxt_async, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "options": {"defaultType": "swap"}})

    counts: dict[str, int] = {
        "ohlcv_1m": 0,
        "funding": 0,
        "oi": 0,
        "mark_price": 0,
        "orderbook_top1": 0,
    }

    try:
        await exchange.load_markets()

        if symbols is None:
            symbols = [
                s
                for s, m in exchange.markets.items()
                if m["active"] and m["swap"] and m["linear"] and m["quote"] == "USDT"
            ]

        logger.info(f"Fetching research data for {len(symbols)} symbols")

        for symbol in symbols:
            safe = symbol.replace("/", "").replace(":", "_")

            # --- 1m OHLCV ---
            ohlcv_path = storage_path / "ohlcv_1m" / f"{safe}.parquet"
            sym_since = since_ms
            if ohlcv_path.exists():
                existing = pd.read_parquet(ohlcv_path)
                if len(existing) > 0:
                    sym_since = int(existing.index.max().timestamp() * 1000) + 1

            all_ohlcv: list[list] = []
            current_since = sym_since
            while current_since is not None:
                try:
                    batch = await exchange.fetch_ohlcv(
                        symbol, "1m", since=current_since, limit=1000
                    )
                except Exception as e:
                    logger.warning(f"OHLCV fetch failed for {symbol}: {e}")
                    break
                if not batch:
                    break
                all_ohlcv.extend(batch)
                current_since = batch[-1][0] + 1
                if len(batch) < 1000:
                    break

            if all_ohlcv:
                df_1m = ohlcv_to_dataframe(all_ohlcv)
                save_parquet(df_1m, ohlcv_path)
                counts["ohlcv_1m"] += len(df_1m)

                # Pre-aggregate to 5m and 20m
                for freq in [5, 20]:
                    agg_path = storage_path / f"ohlcv_{freq}m" / f"{safe}.parquet"
                    agg_df = aggregate_bars(df_1m, freq)
                    if not agg_df.empty:
                        save_parquet(agg_df, agg_path)

            # --- Funding Rate ---
            funding_path = storage_path / "funding" / f"{safe}.parquet"
            mark_price_path = storage_path / "mark_price" / f"{safe}.parquet"
            try:
                funding_raw = await exchange.fetch_funding_rate_history(
                    symbol, since=since_ms, limit=1000
                )
                if funding_raw:
                    df_funding = funding_to_dataframe(funding_raw)
                    if not df_funding.empty:
                        save_parquet(df_funding, funding_path)
                        counts["funding"] += len(df_funding)

                    df_mark_price = mark_price_to_dataframe(funding_raw)
                    if not df_mark_price.empty:
                        save_parquet(df_mark_price, mark_price_path)
                        counts["mark_price"] += len(df_mark_price)
            except Exception as e:
                logger.warning(f"Funding fetch failed for {symbol}: {e}")

            # --- Open Interest ---
            oi_path = storage_path / "oi" / f"{safe}.parquet"
            try:
                if hasattr(exchange, "fetch_open_interest_history"):
                    oi_raw = await exchange.fetch_open_interest_history(
                        symbol, timeframe="5m", since=since_ms, limit=500
                    )
                    if oi_raw:
                        df_oi = oi_to_dataframe(oi_raw)
                        if not df_oi.empty:
                            save_parquet(df_oi, oi_path)
                            counts["oi"] += len(df_oi)
            except Exception as e:
                logger.warning(f"OI fetch failed for {symbol}: {e}")

            # --- Top-of-book orderbook snapshot ---
            orderbook_path = storage_path / "orderbook_top1" / f"{safe}.parquet"
            try:
                ob = await exchange.fetch_order_book(symbol, limit=5)
                fallback_ms = exchange.milliseconds() if hasattr(exchange, "milliseconds") else None
                ts_ms, timestamp_source = resolve_orderbook_timestamp_ms(ob, fallback_ms)
                df_orderbook = orderbook_to_dataframe(ob, ts_ms, timestamp_source=timestamp_source)
                if not df_orderbook.empty and df_orderbook[["best_bid", "best_ask"]].notna().any(axis=None):
                    save_parquet(df_orderbook, orderbook_path)
                    counts["orderbook_top1"] += len(df_orderbook)
            except Exception as e:
                logger.warning(f"Orderbook fetch failed for {symbol}: {e}")

    finally:
        await exchange.close()

    return counts
