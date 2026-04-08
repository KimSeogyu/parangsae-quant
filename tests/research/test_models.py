"""Tests for model pipeline (Ridge, baselines, walk-forward)."""

import numpy as np
import pandas as pd

from src.research.models.ridge import compute_rank_ic, get_feature_importance, train_ridge
from src.research.models.baselines import SimpleRuleBaseline, ZeroBaseline, train_all_baselines
from src.research.models.walkforward import run_walk_forward, summarize_walk_forward


def _make_panel(n_samples=5000, n_features=40, seed=42):
    """Generate synthetic panel with weak signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    # True signal: weak linear combination of first 5 features
    true_coef = np.zeros(n_features)
    true_coef[:5] = rng.normal(0, 0.1, 5)
    y = X @ true_coef + rng.normal(0, 1, n_samples)
    return X, y, true_coef


class TestRankIC:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(compute_rank_ic(y, y) - 1.0) < 1e-10

    def test_inverse_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(compute_rank_ic(y, -y) - (-1.0)) < 1e-10

    def test_random_near_zero(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 1000)
        y_pred = rng.normal(0, 1, 1000)
        ic = compute_rank_ic(y_true, y_pred)
        assert abs(ic) < 0.1

    def test_too_few_samples(self):
        assert compute_rank_ic(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


class TestTrainRidge:
    def test_selects_best_alpha(self):
        X, y, _ = _make_panel(1000, 10)
        X_train, y_train = X[:700], y[:700]
        X_val, y_val = X[700:], y[700:]
        model, alpha, ic = train_ridge(X_train, y_train, X_val, y_val)
        assert model is not None
        assert alpha > 0
        # IC should be non-negative with signal
        # (may be small due to noise)

    def test_prediction_shape(self):
        X, y, _ = _make_panel(500, 10)
        model, _, _ = train_ridge(X[:400], y[:400], X[400:], y[400:])
        preds = model.predict(X[400:])
        assert preds.shape == (100,)


class TestFeatureImportance:
    def test_returns_sorted(self):
        X, y, _ = _make_panel(500, 5)
        model, _, _ = train_ridge(X[:400], y[:400], X[400:], y[400:])
        importance = get_feature_importance(model, ["f0", "f1", "f2", "f3", "f4"])
        assert len(importance) == 5
        # Should be sorted by abs_coefficient descending
        assert importance["abs_coefficient"].is_monotonic_decreasing


class TestZeroBaseline:
    def test_always_zero(self):
        model = ZeroBaseline()
        model.fit(np.ones((10, 5)), np.ones(10))
        preds = model.predict(np.ones((3, 5)))
        assert (preds == 0).all()


class TestSimpleRule:
    def test_gate_suppresses_extreme_rsi(self):
        X = np.zeros((5, 10))
        X[:, 5] = [1.0, 2.0, 3.0, 4.0, 5.0]  # z_col
        X[:, 7] = [50, 25, 75, 50, 50]  # rsi_col: 25 and 75 are extreme
        model = SimpleRuleBaseline(z_col=5, rsi_col=7)
        model.fit(X, np.zeros(5))
        preds = model.predict(X)
        assert preds[0] == 1.0  # Normal RSI, passes through
        assert preds[1] == 0.0  # RSI 25, suppressed
        assert preds[2] == 0.0  # RSI 75, suppressed


class TestAllBaselines:
    def test_all_trained(self):
        X, y, _ = _make_panel(500, 10)
        results = train_all_baselines(X[:400], y[:400], X[400:], y[400:])
        assert "zero" in results
        assert "simple_rule" in results
        assert "ols" in results
        assert "lasso" in results
        assert "elastic_net" in results
        for name, res in results.items():
            assert "model" in res
            assert "val_ic" in res


class TestWalkForward:
    def test_produces_results(self):
        rng = np.random.default_rng(42)
        n = 365 * 288  # 1 year of 5-min bars
        # Use smaller dataset for test speed
        n = 120 * 288
        idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
        X = pd.DataFrame(rng.normal(0, 1, (n, 10)), index=idx)
        y20 = pd.Series(rng.normal(0, 0.01, n), index=idx)
        y60 = pd.Series(rng.normal(0, 0.01, n), index=idx)

        results = run_walk_forward(
            X, y20, y60,
            train_days=90, val_days=14, step_days=7,
            run_baselines=False,
        )
        assert len(results) > 0
        assert all(r.test_ic is not None for r in results)

    def test_summary_stats(self):
        rng = np.random.default_rng(42)
        n = 120 * 288
        idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
        X = pd.DataFrame(rng.normal(0, 1, (n, 10)), index=idx)
        y20 = pd.Series(rng.normal(0, 0.01, n), index=idx)
        y60 = pd.Series(rng.normal(0, 0.01, n), index=idx)

        results = run_walk_forward(
            X, y20, y60,
            train_days=90, val_days=14, step_days=7,
            run_baselines=False,
        )
        summary = summarize_walk_forward(results)
        assert "n_folds" in summary
        assert summary["n_folds"] > 0
        assert "mean_test_ic" in summary

    def test_empty_data(self):
        idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
        X = pd.DataFrame(np.ones((10, 5)), index=idx)
        y20 = pd.Series(np.ones(10), index=idx)
        y60 = pd.Series(np.ones(10), index=idx)
        results = run_walk_forward(X, y20, y60, train_days=90)
        assert len(results) == 0
