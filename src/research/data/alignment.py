"""Point-in-time data alignment utilities.

PRD Section 3: "All features use information only from completed bars up to time t.
Labels are computed from intervals after t. No look-ahead."
"""

from __future__ import annotations

import pandas as pd


def align_point_in_time(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align features and labels ensuring no look-ahead bias.

    Features at time t use data up to and including t.
    Labels at time t use data strictly after t.

    Returns aligned (features, labels) with matching index.
    """
    common_idx = feature_df.index.intersection(label_df.index)
    return feature_df.loc[common_idx], label_df.loc[common_idx]


def embargo_split(
    index: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    embargo_minutes: int = 60,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split index into train and validation with embargo gap.

    Removes embargo_minutes worth of data between train and validation
    to prevent label leakage.
    """
    embargo_delta = pd.Timedelta(minutes=embargo_minutes)
    train_mask = index <= train_end
    val_mask = index >= (validation_start + embargo_delta)
    return index[train_mask], index[val_mask]


def build_walk_forward_windows(
    full_index: pd.DatetimeIndex,
    train_days: int = 90,
    val_days: int = 14,
    embargo_minutes: int = 60,
    step_days: int = 1,
) -> list[dict[str, pd.Timestamp]]:
    """Generate walk-forward train/validation/test windows.

    Each window:
      train: [start, start + train_days)
      embargo: [train_end, train_end + embargo)
      validation: [train_end + embargo, train_end + embargo + val_days)
      test: [val_end, val_end + step_days)  (the next retrain cadence)

    Returns list of dicts with keys: train_start, train_end, val_start, val_end,
    test_start, test_end.
    """
    min_ts = full_index.min()
    max_ts = full_index.max()

    train_delta = pd.Timedelta(days=train_days)
    val_delta = pd.Timedelta(days=val_days)
    embargo_delta = pd.Timedelta(minutes=embargo_minutes)
    step_delta = pd.Timedelta(days=step_days)

    windows = []
    cursor = min_ts

    while True:
        train_start = cursor
        train_end = train_start + train_delta
        val_start = train_end + embargo_delta
        val_end = val_start + val_delta
        test_start = val_end
        test_end = test_start + step_delta

        if test_end > max_ts:
            break

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor += step_delta

    return windows
