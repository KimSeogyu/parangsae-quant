from __future__ import annotations

from pathlib import Path

import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.persistence.wranglers import BarDataWrangler


def load_bars_from_parquet(
    path: str | Path,
    instrument: Instrument,
    bar_step: int = 1,
    bar_aggregation: str = "HOUR",
) -> list[Bar]:
    df = pd.read_parquet(path)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    wrangler_df = df[["open", "high", "low", "close", "volume"]].copy()

    bar_type = BarType.from_str(
        f"{instrument.id}-{bar_step}-{bar_aggregation}-LAST-EXTERNAL"
    )
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)

    bars = wrangler.process(wrangler_df, ts_init_delta=0)
    return bars


def load_all_bars(
    storage_path: str,
    market_type: str,
    instruments: dict[str, Instrument],
    bar_step: int = 1,
    bar_aggregation: str = "HOUR",
) -> dict[str, list[Bar]]:
    result = {}
    base = Path(storage_path) / market_type
    if not base.exists():
        return result

    for parquet_file in sorted(base.glob("*.parquet")):
        symbol = parquet_file.stem
        if symbol not in instruments:
            continue
        bars = load_bars_from_parquet(
            path=parquet_file,
            instrument=instruments[symbol],
            bar_step=bar_step,
            bar_aggregation=bar_aggregation,
        )
        if bars:
            result[symbol] = bars
    return result
