import pandas as pd
from src.data.fetcher import ohlcv_to_dataframe, save_ohlcv


def test_ohlcv_to_dataframe():
    raw = [
        [1609459200000, 29000.0, 29500.0, 28800.0, 29300.0, 100.5],
        [1609462800000, 29300.0, 29700.0, 29100.0, 29600.0, 85.2],
    ]
    df = ohlcv_to_dataframe(raw)

    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "quote_volume"]
    assert df.index.name == "timestamp"
    assert df.index.tz is not None
    assert df.iloc[0]["close"] == 29300.0
    assert df.iloc[0]["quote_volume"] == 29300.0 * 100.5


def test_save_ohlcv_writes_parquet(tmp_path):
    raw = [
        [1609459200000, 29000.0, 29500.0, 28800.0, 29300.0, 100.5],
    ]
    df = ohlcv_to_dataframe(raw)
    out_path = tmp_path / "BTCUSDT.parquet"

    save_ohlcv(df, str(out_path))

    loaded = pd.read_parquet(out_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["close"] == 29300.0
