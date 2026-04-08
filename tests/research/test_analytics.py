"""Tests for analytics metrics and report generation."""

import numpy as np
import pandas as pd

from src.research.analytics.metrics import (
    classify_regime,
    compute_beta_drift,
    compute_max_drawdown,
    compute_regime_performance,
    compute_sharpe,
    compute_turnover,
)
from src.research.analytics.report import evaluate_go_no_go, BacktestReport


class TestSharpe:
    def test_positive_sharpe(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.01, 1000))
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_zero_vol(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        assert compute_sharpe(returns) == 0.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = pd.Series([1.0, 1.1, 1.2, 1.3])
        assert compute_max_drawdown(equity) == 0.0

    def test_known_drawdown(self):
        equity = pd.Series([1.0, 1.2, 0.9, 1.0])
        dd = compute_max_drawdown(equity)
        # From peak 1.2 to trough 0.9 = 25%
        assert abs(dd - 0.25) < 1e-6


class TestTurnover:
    def test_basic(self):
        history = [
            {"A": 0.5, "B": 0.5},
            {"A": 0.6, "B": 0.4},
            {"A": 0.3, "C": 0.7},
        ]
        to = compute_turnover(history)
        assert len(to) == 2
        assert to.iloc[0] > 0

    def test_no_change(self):
        history = [{"A": 0.5}, {"A": 0.5}]
        to = compute_turnover(history)
        assert to.iloc[0] == 0.0


class TestBetaDrift:
    def test_correlated_returns(self):
        rng = np.random.default_rng(42)
        n = 500
        btc = pd.Series(rng.normal(0, 0.01, n))
        port = 0.5 * btc + pd.Series(rng.normal(0, 0.005, n))
        drift = compute_beta_drift(port, btc, window=100)
        assert len(drift) > 0
        assert abs(drift.iloc[-1] - 0.5) < 0.3


class TestRegime:
    def test_classification(self):
        rng = np.random.default_rng(42)
        n = 5000
        returns = pd.Series(rng.normal(0, 0.01, n))
        regimes = classify_regime(returns, window=500)
        assert set(regimes.unique()).issubset({"bull", "bear", "chop"})

    def test_regime_performance(self):
        rng = np.random.default_rng(42)
        n = 5000
        returns = pd.Series(rng.normal(0.001, 0.01, n))
        regimes = pd.Series(["bull"] * 2000 + ["bear"] * 1500 + ["chop"] * 1500)
        perf = compute_regime_performance(returns, regimes)
        assert "bull" in perf
        assert "bear" in perf
        assert "chop" in perf
        assert perf["bull"]["count"] == 2000


class TestGoNoGo:
    def test_pass_criteria(self):
        report = BacktestReport(
            mean_rank_ic=0.05,
            net_sharpe_after_costs=1.0,
            mean_beta=0.05,
            maker_fill_ratio=0.70,
            regime_performance={
                "bull": {"cumulative_return": 0.10},
                "bear": {"cumulative_return": 0.02},
                "chop": {"cumulative_return": -0.01},
            },
        )
        decisions = evaluate_go_no_go(report)
        assert decisions["alpha_ic"]["pass"] is True
        assert decisions["sharpe"]["pass"] is True
        assert decisions["beta_cap"]["pass"] is True
        assert decisions["stability"]["pass"] is True

    def test_fail_criteria(self):
        report = BacktestReport(
            mean_rank_ic=-0.01,
            net_sharpe_after_costs=0.3,
            mean_beta=0.20,
            maker_fill_ratio=0.40,
            regime_performance={
                "bull": {"cumulative_return": 0.01},
                "bear": {"cumulative_return": -0.05},
                "chop": {"cumulative_return": -0.03},
            },
        )
        decisions = evaluate_go_no_go(report)
        assert decisions["alpha_ic"]["pass"] is False
        assert decisions["beta_cap"]["pass"] is False
        assert decisions["stability"]["pass"] is False
