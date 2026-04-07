import pytest
from src.risk.model import compute_rolling_volatility, compute_drawdown


def test_compute_rolling_volatility():
    returns = [0.0, 0.0, 0.0, 0.0]
    assert compute_rolling_volatility(returns, window=4) == 0.0


def test_compute_rolling_volatility_with_movement():
    returns = [0.01, -0.01, 0.02, -0.02, 0.01]
    vol = compute_rolling_volatility(returns, window=5)
    assert vol > 0
    assert vol < 1


def test_compute_drawdown():
    equity_curve = [100.0, 110.0, 105.0, 95.0, 100.0]
    dd = compute_drawdown(equity_curve)
    assert dd == pytest.approx(0.1364, abs=0.001)


def test_compute_drawdown_no_loss():
    equity_curve = [100.0, 110.0, 120.0]
    dd = compute_drawdown(equity_curve)
    assert dd == 0.0
