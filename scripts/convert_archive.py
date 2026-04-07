"""Convert parangsae-archive raw 1m klines (ZIP/CSV) to 1h Parquet for backtesting.

Handles:
- spot_klines_1m: microsecond timestamps, no CSV header
- um_klines_1m: millisecond timestamps, has CSV header

Output: data/spot/*.parquet and data/futures/*.parquet in our standard schema.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_ROOT = Path("parangsae-archive/raw")
OUTPUT_ROOT = Path("data")

BINANCE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def read_zip_csv(zip_path: Path, has_header: bool, ts_unit: str) -> pd.DataFrame:
    """Read a single daily ZIP containing one CSV."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with zf.open(csv_names[0]) as f:
            if has_header:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip().str.lower()
                # Normalize column names
                col_map = {c: c for c in df.columns}
                if "open_time" in col_map:
                    pass  # already correct
                df = df.rename(columns=col_map)
            else:
                df = pd.read_csv(f, header=None, names=BINANCE_COLUMNS)

    if df.empty:
        return df

    # Convert timestamps
    if ts_unit == "us":
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="us", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    df = df.set_index("timestamp")

    # Keep only OHLCV + quote_volume
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["open", "high", "low", "close", "volume", "quote_volume"]]


def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m OHLCV to 1h."""
    resampled = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
    })
    return resampled.dropna(subset=["close"])


def convert_symbol(symbol_dir: Path, has_header: bool, ts_unit: str) -> pd.DataFrame:
    """Load all daily ZIPs for a symbol, concat, resample to 1h."""
    zip_files = sorted(symbol_dir.glob("*.zip"))
    if not zip_files:
        return pd.DataFrame()

    dfs = []
    for zf in zip_files:
        if zf.name.endswith(".CHECKSUM"):
            continue
        try:
            df = read_zip_csv(zf, has_header=has_header, ts_unit=ts_unit)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {zf}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return resample_to_1h(combined)


def convert_market(
    archive_dir: str,
    market_type: str,
    output_market_type: str,
    has_header: bool,
    ts_unit: str,
    suffix: str = "",
) -> None:
    """Convert all symbols in an archive market directory."""
    src_dir = ARCHIVE_ROOT / archive_dir
    if not src_dir.exists():
        logger.warning(f"Archive directory not found: {src_dir}")
        return

    out_dir = OUTPUT_ROOT / output_market_type
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    logger.info(f"Converting {len(symbol_dirs)} {output_market_type} symbols from {archive_dir}")

    for symbol_dir in symbol_dirs:
        symbol = symbol_dir.name + suffix  # e.g. "BTCUSDT" or "BTCUSDT-PERP"
        out_path = out_dir / f"{symbol}.parquet"

        if out_path.exists():
            logger.info(f"  {symbol}: already exists, skipping")
            continue

        logger.info(f"  {symbol}: converting...")
        df = convert_symbol(symbol_dir, has_header=has_header, ts_unit=ts_unit)
        if df.empty:
            logger.warning(f"  {symbol}: no data")
            continue

        df.to_parquet(out_path)
        logger.info(f"  {symbol}: {len(df)} bars saved ({df.index.min()} to {df.index.max()})")


def main() -> None:
    # Spot: microsecond timestamps, no header
    convert_market(
        archive_dir="spot_klines_1m",
        market_type="spot",
        output_market_type="spot",
        has_header=False,
        ts_unit="us",
    )

    # USDT-M Futures: millisecond timestamps, has header, add -PERP suffix
    convert_market(
        archive_dir="um_klines_1m",
        market_type="futures",
        output_market_type="futures",
        has_header=True,
        ts_unit="ms",
        suffix="-PERP",
    )

    # Summary
    for mt in ["spot", "futures"]:
        data_dir = OUTPUT_ROOT / mt
        if data_dir.exists():
            files = list(data_dir.glob("*.parquet"))
            logger.info(f"{mt}: {len(files)} symbols ready")


if __name__ == "__main__":
    main()
