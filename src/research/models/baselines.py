"""Baseline models for comparison.

PRD Section 4: Benchmarks
- Zero benchmark: always predicts 0
- Simple rule baseline: beta-adjusted residual z-score + RSI/VWAP gate
- OLS: unregularized linear regression
- Lasso: L1 penalty (sparse selection comparison)
- Elastic Net: high-L2 setting (Ridge-adjacent comparison)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression

from src.research.models.ridge import compute_rank_ic


class ZeroBaseline:
    """Always predicts 0. Weak-signal environment must beat this."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> ZeroBaseline:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class SimpleRuleBaseline:
    """Simple rule: z-scored residual momentum + RSI gate.

    Predicts the residual z-score column (feature index 5 = resid_z_4h by default),
    gated by RSI (feature index 7 = resid_rsi_14). If RSI > 70 or < 30,
    signal is suppressed.
    """

    def __init__(self, z_col: int = 5, rsi_col: int = 7):
        self.z_col = z_col
        self.rsi_col = rsi_col

    def fit(self, X: np.ndarray, y: np.ndarray) -> SimpleRuleBaseline:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X[:, self.z_col].copy()
        rsi = X[:, self.rsi_col]
        # Gate: suppress when RSI extreme
        gate = (rsi > 30) & (rsi < 70)
        scores[~gate] = 0.0
        return scores


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, dict]:
    """Train all baseline models and return their validation rank IC.

    Returns:
        {model_name: {"model": model, "val_ic": float}}
    """
    baselines = {
        "zero": ZeroBaseline(),
        "simple_rule": SimpleRuleBaseline(),
        "ols": LinearRegression(),
        "lasso": Lasso(alpha=0.01, max_iter=5000),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=5000),
    }

    results = {}
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        ic = compute_rank_ic(y_val, y_pred)
        results[name] = {"model": model, "val_ic": ic}

    return results
