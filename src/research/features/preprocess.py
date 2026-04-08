"""Feature preprocessing pipeline.

PRD Section 6:
  - Cross-sectional z-score per timestamp
  - Clip at +/- 5
  - Missing values: cross-sectional median replacement
  - No feature selection (all 40 features pass through)
  - Coverage check: drop symbols below min_coverage threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Z-score each row (timestamp) across symbols.

    For each timestamp t, for each feature:
      z = (x - mean_xs) / std_xs

    Args:
        panel: DataFrame with (timestamp, symbol) multi-index or
               timestamp index and symbol columns.

    Returns:
        Z-scored DataFrame with same shape.
    """
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1, ddof=1)
    row_std = row_std.replace(0, np.nan)
    return panel.sub(row_mean, axis=0).div(row_std, axis=0)


def clip_features(
    panel: pd.DataFrame,
    clip_val: float = 5.0,
) -> pd.DataFrame:
    """Clip feature values to [-clip_val, +clip_val]."""
    return panel.clip(-clip_val, clip_val)


def fill_missing_cross_sectional_median(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Fill NaN with cross-sectional median (per row).

    For each timestamp, missing values are replaced with the median
    across all symbols at that timestamp.
    """
    row_median = panel.median(axis=1)
    return panel.T.fillna(row_median).T


def check_coverage(
    panel: pd.DataFrame,
    min_coverage: float = 0.80,
) -> list[str]:
    """Return list of columns (symbols) with coverage >= threshold.

    Coverage = fraction of non-NaN values across all timestamps.
    """
    coverage = panel.notna().mean(axis=0)
    return list(coverage[coverage >= min_coverage].index)


def preprocess_features(
    raw_features: pd.DataFrame,
    clip_val: float = 5.0,
    min_coverage: float = 0.80,
) -> pd.DataFrame:
    """Full preprocessing pipeline: coverage check -> fill -> z-score -> clip.

    Args:
        raw_features: DataFrame with timestamps as index, symbols as columns.
        clip_val: Clipping threshold (default ±5 per PRD).
        min_coverage: Minimum data coverage to keep a symbol.

    Returns:
        Preprocessed feature DataFrame.
    """
    # Step 1: Coverage filter
    valid_cols = check_coverage(raw_features, min_coverage)
    filtered = raw_features[valid_cols]

    # Step 2: Fill missing values
    filled = fill_missing_cross_sectional_median(filtered)

    # Step 3: Cross-sectional z-score
    zscored = cross_sectional_zscore(filled)

    # Step 4: Clip
    clipped = clip_features(zscored, clip_val)

    return clipped
