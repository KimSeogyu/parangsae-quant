from src.data.catalog import load_bars_from_parquet


def test_load_bars_from_parquet(tmp_path, btcusdt_instrument, sample_ohlcv_df):
    parquet_path = tmp_path / "BTCUSDT.parquet"
    sample_ohlcv_df.to_parquet(parquet_path)

    bars = load_bars_from_parquet(
        path=str(parquet_path),
        instrument=btcusdt_instrument,
        bar_step=1,
        bar_aggregation="HOUR",
    )

    assert len(bars) == 100
    assert float(bars[0].open) == 50000.0
    assert float(bars[0].close) == 50050.0
    assert bars[0].ts_init == bars[0].ts_event
