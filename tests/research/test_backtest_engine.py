"""Tests for the research backtest engine."""

import numpy as np
import pandas as pd

from src.research.backtest.engine import (
    BacktestConfig,
    run_accurate_backtest,
    run_fast_backtest,
)


def _make_engine_inputs():
    idx = pd.date_range("2025-01-01", periods=10 * 288, freq="5min", tz="UTC")
    alts = ["ALT1", "ALT2", "ALT3"]
    mi = pd.MultiIndex.from_product([idx, alts], names=["timestamp", "symbol"])

    base_signal = np.tile(np.array([1.0, 0.0, -1.0]), len(idx))
    time_wave = np.repeat(np.sin(np.linspace(0, 8, len(idx))), len(alts))
    feature_frame = {
        "f1": base_signal + 0.2 * time_wave,
        "f2": 0.5 * base_signal + 0.1 * np.cos(time_wave),
    }
    for i in range(3, 11):
        feature_frame[f"f{i}"] = 0.1 * np.sin(time_wave + i)
    features = pd.DataFrame(
        feature_frame,
        index=mi,
    )
    labels_y20 = pd.Series(0.01 * features["f1"], index=mi)
    labels_y60 = pd.Series(0.008 * features["f1"], index=mi)

    price_index = idx
    prices = pd.DataFrame(
        {
            "ALT1": 100.0 * np.exp(np.linspace(0.0, 0.12, len(idx))),
            "ALT2": 100.0 * np.exp(np.linspace(0.0, 0.01, len(idx))),
            "ALT3": 100.0 * np.exp(np.linspace(0.0, -0.10, len(idx))),
            "BTCUSDT": 20000.0 * np.exp(np.linspace(0.0, 0.04, len(idx))),
        },
        index=price_index,
    )
    btc_returns = prices["BTCUSDT"].pct_change().fillna(0.0)

    volatilities = pd.DataFrame(
        0.02,
        index=price_index,
        columns=prices.columns,
    )
    betas_panel = pd.DataFrame(
        {
            "ALT1": 1.10,
            "ALT2": 1.00,
            "ALT3": 0.90,
            "BTCUSDT": 1.00,
        },
        index=price_index,
    )

    orderbook_data = {}
    for sym in prices.columns:
        orderbook_data[sym] = prices[sym]
        orderbook_data[f"{sym}_low"] = prices[sym] * 0.999
        orderbook_data[f"{sym}_high"] = prices[sym] * 1.001
    orderbook_df = pd.DataFrame(orderbook_data, index=price_index)

    return (
        features,
        labels_y20,
        labels_y60,
        prices,
        btc_returns,
        volatilities,
        betas_panel,
        orderbook_df,
    )


class TestFastBacktest:
    def test_uses_prediction_panel_without_placeholder_randomness(self):
        inputs = _make_engine_inputs()
        report = run_fast_backtest(
            *inputs[:-1],
            config=BacktestConfig(rebalance_bars=12),
            train_days=4,
            val_days=2,
            alpha_grid=[0.1, 1.0],
        )
        assert report.mean_rank_ic > 0
        assert report.avg_turnover > 0
        assert np.isfinite(report.total_return)


class TestAccurateBacktest:
    def test_tracks_real_orders_and_fill_metrics(self):
        inputs = _make_engine_inputs()
        report = run_accurate_backtest(
            *inputs,
            config=BacktestConfig(mode="accurate", rebalance_bars=12),
            train_days=4,
            val_days=2,
            alpha_grid=[0.1, 1.0],
        )
        assert report.mean_rank_ic > 0
        assert 0.0 <= report.maker_fill_ratio <= 1.0
        assert report.maker_fill_ratio > 0.0
        assert np.isfinite(report.mean_slippage_bps)
