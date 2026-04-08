"""Ridge regression model for residual return prediction.

PRD Section 6: Model Specification
- Ridge/L2 with log-grid alpha search
- Panel cross-sectional regression
- Target metric: rank IC on validation set
- Two models: y20, y60, combined as 0.7*y20 + 0.3*y60
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def compute_rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank Information Coefficient.

    IC = correlation(rank(y_true), rank(y_pred))
    """
    from scipy.stats import spearmanr

    if len(y_true) < 3:
        return 0.0
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return 0.0
    corr, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(corr) if np.isfinite(corr) else 0.0


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha_grid: list[float] | None = None,
    fit_intercept: bool = True,
) -> tuple[Ridge, float, float]:
    """Train Ridge with alpha grid search, select by validation rank IC.

    Args:
        X_train: Training features (n_samples, n_features).
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        alpha_grid: List of alpha values to search.
        fit_intercept: Whether to fit intercept.

    Returns:
        (best_model, best_alpha, best_val_ic)
    """
    if alpha_grid is None:
        alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    best_model = None
    best_alpha = alpha_grid[0]
    best_ic = -np.inf

    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        ic = compute_rank_ic(y_val, y_pred_val)

        if ic > best_ic:
            best_ic = ic
            best_alpha = alpha
            best_model = model

    if best_model is None:
        best_model = Ridge(alpha=alpha_grid[0], fit_intercept=fit_intercept)
        best_model.fit(X_train, y_train)

    return best_model, best_alpha, best_ic


def predict_scores(
    model_y20: Ridge,
    model_y60: Ridge,
    X: np.ndarray,
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
) -> np.ndarray:
    """Generate combined prediction scores.

    score = primary_weight * y20_pred + secondary_weight * y60_pred
    """
    pred_y20 = model_y20.predict(X)
    pred_y60 = model_y60.predict(X)
    return primary_weight * pred_y20 + secondary_weight * pred_y60


def get_feature_importance(model: Ridge, feature_names: list[str]) -> pd.DataFrame:
    """Extract Ridge coefficient magnitudes as feature importance."""
    return pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
            "abs_coefficient": np.abs(model.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)
