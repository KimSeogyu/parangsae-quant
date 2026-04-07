import pytest
from src.alpha.momentum import compute_momentum


def test_compute_momentum_positive():
    closes = [100.0, 102.0, 105.0, 108.0, 110.0]
    result = compute_momentum(closes, lookback=4)
    assert result == pytest.approx(0.1)


def test_compute_momentum_negative():
    closes = [110.0, 108.0, 105.0, 102.0, 100.0]
    result = compute_momentum(closes, lookback=4)
    assert result == pytest.approx(-0.0909, abs=0.001)


def test_compute_momentum_insufficient_data():
    closes = [100.0, 102.0]
    result = compute_momentum(closes, lookback=4)
    assert result == 0.0
