from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from src.alpha.base import BaseAlphaConfig, BaseAlphaModel


class QMMomentumAlphaConfig(BaseAlphaConfig):
    alpha_name: str = "crypto_qm_momentum"
    params: dict = Field(default_factory=lambda: {
        "lookback_hours": 336,
        "skip_hours": 24,
        "vol_window_hours": 720,
        "fip_enabled": True,
        "fip_floor": 0.3,
        "ts_filter": True,
    })


def compute_qm_momentum(
    closes: list[float],
    lookback: int = 336,
    skip: int = 24,
    vol_window: int = 720,
    fip_enabled: bool = True,
    fip_floor: float = 0.3,
    ts_filter: bool = True,
) -> float:
    """QM Momentum Alpha with FIP Quality Filter.

    Returns vol-adjusted momentum * quality multiplier, or -inf if filtered out.
    """
    required = skip + max(lookback, vol_window) + 2
    if len(closes) < required:
        return float("-inf")

    # Raw momentum (log return, skipping recent skip hours)
    p_end = closes[-(skip + 1)]
    p_start = closes[-(skip + lookback + 1)]
    if p_start <= 0 or p_end <= 0:
        return float("-inf")
    raw_mom = math.log(p_end / p_start)

    # Time-series filter: reject negative momentum
    if ts_filter and raw_mom <= 0.0:
        return float("-inf")

    # Realized volatility (vol_window ending at skip boundary)
    vol_slice = closes[-(skip + vol_window + 1) : -(skip)]
    arr = np.array(vol_slice, dtype=np.float64)
    log_rets = np.diff(np.log(arr))
    realized_vol = float(np.std(log_rets, ddof=1))
    # Apply minimum vol floor to avoid division by near-zero; use absolute floor rather than
    # filtering, so strongly trending assets get a capped (high) signal instead of -inf.
    vol_floor = 1e-5
    realized_vol = max(realized_vol, vol_floor)

    vol_adj_mom = raw_mom / realized_vol

    if not fip_enabled:
        return vol_adj_mom

    # FIP on daily bars (aggregate hourly to daily by sampling every 24th bar)
    lookback_slice = closes[-(skip + lookback + 1) : -(skip)]
    daily_closes = lookback_slice[::24]
    if len(daily_closes) < 2:
        return vol_adj_mom  # not enough daily data, skip FIP

    daily_arr = np.array(daily_closes, dtype=np.float64)
    daily_rets = np.diff(daily_arr) / daily_arr[:-1]

    pct_pos = float(np.sum(daily_rets > 0)) / len(daily_rets)
    pct_neg = float(np.sum(daily_rets < 0)) / len(daily_rets)
    sign = 1.0 if raw_mom > 0 else -1.0
    fip = sign * (pct_neg - pct_pos)  # range [-1, +1]

    # Transform FIP to quality multiplier [fip_floor, 1.0]
    quality = fip_floor + (1 - fip_floor) * (1 - fip) / 2

    return vol_adj_mom * quality


class QMMomentumAlpha(BaseAlphaModel):
    def __init__(self, config: QMMomentumAlphaConfig) -> None:
        super().__init__(config)
        p = config.params
        self._lookback = p.get("lookback_hours", 336)
        self._skip = p.get("skip_hours", 24)
        self._vol_window = p.get("vol_window_hours", 720)
        self._fip_enabled = p.get("fip_enabled", True)
        self._fip_floor = p.get("fip_floor", 0.3)
        self._ts_filter = p.get("ts_filter", True)
        self._max_buffer = self._skip + max(self._lookback, self._vol_window) + 10

    def compute_alpha(self, symbol: str, closes: list[float]) -> float:
        return compute_qm_momentum(
            closes,
            lookback=self._lookback,
            skip=self._skip,
            vol_window=self._vol_window,
            fip_enabled=self._fip_enabled,
            fip_floor=self._fip_floor,
            ts_filter=self._ts_filter,
        )
