import numpy as np
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
    """100 hourly bars — small fixture for unit tests."""
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


@pytest.fixture
def large_ohlcv_df():
    """1200 hourly bars — enough for QM momentum (336 lookback + 24 skip + 720 vol)."""
    np.random.seed(42)
    n = 1200
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    base_price = 50000.0
    prices = [base_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0.0002, 0.015)))
    prices = np.array(prices)
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.uniform(0.5, 3.0, n),
            "quote_volume": prices * np.random.uniform(0.5, 3.0, n),
        },
        index=dates,
    )
