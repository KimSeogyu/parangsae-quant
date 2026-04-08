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
    test_index: pd.Index = field(default_factory=lambda: pd.Index([]))
    baseline_ics: dict[str, float] = field(default_factory=dict)


def _to_timestamp_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        timestamps = index.get_level_values(0)
    else:
        timestamps = index
    return pd.DatetimeIndex(timestamps)


def _window_mask(index: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    timestamps = _to_timestamp_index(index)
    return np.asarray((timestamps >= start) & (timestamps < end))


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

    Supports both plain DatetimeIndex data and panel-style MultiIndex rows where
    level 0 is timestamp and level 1 is symbol.
    """
    common_idx = features.index.intersection(labels_y20.index).intersection(labels_y60.index)
    features = features.loc[common_idx]
    labels_y20 = labels_y20.loc[common_idx]
    labels_y60 = labels_y60.loc[common_idx]

    valid_mask = features.notna().all(axis=1) & labels_y20.notna() & labels_y60.notna()
    features = features[valid_mask]
    labels_y20 = labels_y20[valid_mask]
    labels_y60 = labels_y60[valid_mask]

    if len(features) == 0:
        return []

    full_timestamps = _to_timestamp_index(features.index)
    timestamp_grid = pd.DatetimeIndex(full_timestamps.unique()).sort_values()
    windows = build_walk_forward_windows(
        timestamp_grid, train_days, val_days, embargo_minutes, step_days
    )

    results: list[WalkForwardResult] = []
    for w in windows:
        train_mask = _window_mask(features.index, w["train_start"], w["train_end"])
        val_mask = _window_mask(features.index, w["val_start"], w["val_end"])
        test_mask = _window_mask(features.index, w["test_start"], w["test_end"])

        X_train = features.iloc[train_mask].values
        X_val = features.iloc[val_mask].values
        X_test = features.iloc[test_mask].values

        if len(X_train) < 10 or len(X_val) < 3 or len(X_test) == 0:
            continue

        y20_train = labels_y20.iloc[train_mask].values
        y20_val = labels_y20.iloc[val_mask].values
        model_y20, alpha_y20, ic_y20 = train_ridge(
            X_train, y20_train, X_val, y20_val, alpha_grid
        )

        y60_train = labels_y60.iloc[train_mask].values
        y60_val = labels_y60.iloc[val_mask].values
        model_y60, alpha_y60, ic_y60 = train_ridge(
            X_train, y60_train, X_val, y60_val, alpha_grid
        )

        test_preds = predict_scores(model_y20, model_y60, X_test, primary_weight, secondary_weight)
        test_labels_combined = (
            primary_weight * labels_y20.iloc[test_mask].values
            + secondary_weight * labels_y60.iloc[test_mask].values
        )
        test_ic = compute_rank_ic(test_labels_combined, test_preds)

        baseline_ics = {}
        if run_baselines:
            y_combined_train = primary_weight * y20_train + secondary_weight * y60_train
            y_combined_val = primary_weight * y20_val + secondary_weight * y60_val
            baseline_results = train_all_baselines(
                X_train, y_combined_train, X_val, y_combined_val
            )
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
                test_index=features.iloc[test_mask].index,
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
