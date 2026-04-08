import numpy as np

from src.alpha.low_volatility import compute_low_volatility


def test_low_vol_returns_negative_volatility():
    np.random.seed(42)
    prices = [100.0]
    for _ in range(400):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    alpha = compute_low_volatility(prices, vol_window=336)
    assert alpha < 0


def test_low_vol_lower_for_higher_volatility():
    np.random.seed(42)
    calm = [100.0]
    for _ in range(400):
        calm.append(calm[-1] * (1 + np.random.normal(0, 0.005)))

    wild = [100.0]
    for _ in range(400):
        wild.append(wild[-1] * (1 + np.random.normal(0, 0.05)))

    alpha_calm = compute_low_volatility(calm, vol_window=336)
    alpha_wild = compute_low_volatility(wild, vol_window=336)
    assert alpha_calm > alpha_wild  # less negative = higher rank


def test_low_vol_insufficient_data():
    alpha = compute_low_volatility([100.0, 101.0], vol_window=336)
    assert alpha == float("-inf")


def test_low_vol_flat_prices():
    prices = [100.0] * 400
    alpha = compute_low_volatility(prices, vol_window=336)
    assert alpha == float("-inf")  # near-zero vol is filtered
