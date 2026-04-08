"""Walk-forward training orchestration.

PRD Section 6 + Section 9:
- 90-day train window, 14-day purged validation, 60-min embargo
- Retrain cadence: 1 day
- Two models per window: y20, y60
- Score combination: 0.7 * y20 + 0.3 * y60
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.research.data.alignment import build_walk_forward_windows
from src.research.models.baselines import train_all_baselines
from src.research.models.ridge import compute_rank_ic, predict_scores, train_ridge


@dataclass
class WalkForwardResult:
    """Results from one walk-forward fold."""

    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_alpha_y20: float
    best_alpha_y60: float
    val_ic_y20: float
    val_ic_y60: float
    test_ic: float
    test_predictions: np.ndarray
    test_labels: np.ndarray
    baseline_ics: dict[str, float] = field(default_factory=dict)


def run_walk_forward(
    features: pd.DataFrame,
    labels_y20: pd.Series,
    labels_y60: pd.Series,
    train_days: int = 90,
    val_days: int = 14,
    embargo_minutes: int = 60,
    step_days: int = 1,
    alpha_grid: list[float] | None = None,
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
    run_baselines: bool = True,
) -> list[WalkForwardResult]:
    """Run walk-forward backtest with Ridge model.

    Args:
        features: DataFrame (timestamp index, feature columns).
        labels_y20: Primary (20-min) residual return labels.
        labels_y60: Secondary (60-min) residual return labels.
        train_days: Training window in days.
        val_days: Validation window in days.
        embargo_minutes: Gap between train and validation.
        step_days: Retrain cadence.
        alpha_grid: Ridge alpha search grid.
        primary_weight: Weight for y20 predictions.
        secondary_weight: Weight for y60 predictions.
        run_baselines: Whether to train baseline models for comparison.

    Returns:
        List of WalkForwardResult per fold.
    """
    # Align features and labels
    common_idx = features.index.intersection(labels_y20.index).intersection(labels_y60.index)
    features = features.loc[common_idx]
    labels_y20 = labels_y20.loc[common_idx]
    labels_y60 = labels_y60.loc[common_idx]

    # Drop rows with any NaN
    valid_mask = features.notna().all(axis=1) & labels_y20.notna() & labels_y60.notna()
    features = features[valid_mask]
    labels_y20 = labels_y20[valid_mask]
    labels_y60 = labels_y60[valid_mask]

    if len(features) == 0:
        return []

    windows = build_walk_forward_windows(
        features.index, train_days, val_days, embargo_minutes, step_days
    )

    results = []
    for w in windows:
        # Split data
        train_mask = (features.index >= w["train_start"]) & (features.index < w["train_end"])
        val_mask = (features.index >= w["val_start"]) & (features.index < w["val_end"])
        test_mask = (features.index >= w["test_start"]) & (features.index < w["test_end"])

        X_train = features[train_mask].values
        X_val = features[val_mask].values
        X_test = features[test_mask].values

        if len(X_train) < 10 or len(X_val) < 3 or len(X_test) == 0:
            continue

        # Train y20 model
        y20_train = labels_y20[train_mask].values
        y20_val = labels_y20[val_mask].values
        model_y20, alpha_y20, ic_y20 = train_ridge(
            X_train, y20_train, X_val, y20_val, alpha_grid
        )

        # Train y60 model
        y60_train = labels_y60[train_mask].values
        y60_val = labels_y60[val_mask].values
        model_y60, alpha_y60, ic_y60 = train_ridge(
            X_train, y60_train, X_val, y60_val, alpha_grid
        )

        # Predict on test set
        test_preds = predict_scores(model_y20, model_y60, X_test, primary_weight, secondary_weight)
        test_labels_combined = (
            primary_weight * labels_y20[test_mask].values
            + secondary_weight * labels_y60[test_mask].values
        )
        test_ic = compute_rank_ic(test_labels_combined, test_preds)

        # Baselines
        baseline_ics = {}
        if run_baselines:
            y_combined_train = primary_weight * y20_train + secondary_weight * y60_train
            y_combined_val = primary_weight * y20_val + secondary_weight * y60_val
            baseline_results = train_all_baselines(
                X_train, y_combined_train, X_val, y_combined_val
            )
            # Evaluate baselines on test set
            for name, res in baseline_results.items():
                bl_pred = res["model"].predict(X_test)
                baseline_ics[name] = compute_rank_ic(test_labels_combined, bl_pred)

        results.append(
            WalkForwardResult(
                test_start=w["test_start"],
                test_end=w["test_end"],
                best_alpha_y20=alpha_y20,
                best_alpha_y60=alpha_y60,
                val_ic_y20=ic_y20,
                val_ic_y60=ic_y60,
                test_ic=test_ic,
                test_predictions=test_preds,
                test_labels=test_labels_combined,
                baseline_ics=baseline_ics,
            )
        )

    return results


def summarize_walk_forward(results: list[WalkForwardResult]) -> dict:
    """Compute aggregate statistics across all walk-forward folds."""
    if not results:
        return {"n_folds": 0}

    test_ics = [r.test_ic for r in results]
    summary = {
        "n_folds": len(results),
        "mean_test_ic": float(np.mean(test_ics)),
        "median_test_ic": float(np.median(test_ics)),
        "std_test_ic": float(np.std(test_ics)),
        "pct_positive_ic": float(np.mean([ic > 0 for ic in test_ics])),
        "mean_val_ic_y20": float(np.mean([r.val_ic_y20 for r in results])),
        "mean_val_ic_y60": float(np.mean([r.val_ic_y60 for r in results])),
    }

    # Baseline comparison
    baseline_names = set()
    for r in results:
        baseline_names.update(r.baseline_ics.keys())
    for bl_name in baseline_names:
        ics = [r.baseline_ics.get(bl_name, 0.0) for r in results]
        summary[f"mean_test_ic_{bl_name}"] = float(np.mean(ics))
        summary[f"ridge_vs_{bl_name}_wins"] = float(
            np.mean([r.test_ic > r.baseline_ics.get(bl_name, 0.0) for r in results])
        )

    return summary
