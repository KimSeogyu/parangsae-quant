import pandas as pd
import pytest
from nautilus_trader.test_kit.providers import TestInstrumentProvider


@pytest.fixture
def btcusdt_instrument():
    return TestInstrumentProvider.btcusdt_binance()


@pytest.fixture
def ethusdt_instrument():
    return TestInstrumentProvider.ethusdt_binance()


@pytest.fixture
def sample_ohlcv_df():
    dates = pd.date_range("2025-01-01", periods=100, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [50000.0 + i * 10 for i in range(100)],
            "high": [50100.0 + i * 10 for i in range(100)],
            "low": [49900.0 + i * 10 for i in range(100)],
            "close": [50050.0 + i * 10 for i in range(100)],
            "volume": [1.5] * 100,
            "quote_volume": [(50050.0 + i * 10) * 1.5 for i in range(100)],
        },
        index=dates,
    )
