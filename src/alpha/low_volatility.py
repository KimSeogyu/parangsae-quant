from __future__ import annotations

import numpy as np
from pydantic import Field

from src.alpha.base import BaseAlphaConfig, BaseAlphaModel


class LowVolatilityAlphaConfig(BaseAlphaConfig):
    alpha_name: str = "low_volatility"
    params: dict = Field(default_factory=lambda: {"vol_window_hours": 336})


def compute_low_volatility(closes: list[float], vol_window: int = 336) -> float:
    """Low Volatility Factor Alpha. Returns -realized_vol for cross-sectional ranking."""
    if len(closes) < vol_window + 1:
        return float("-inf")

    arr = np.array(closes[-(vol_window + 1) :], dtype=np.float64)
    log_rets = np.diff(np.log(arr))
    realized_vol = float(np.std(log_rets, ddof=1))

    if realized_vol < 1e-8:
        return float("-inf")

    return -realized_vol


class LowVolatilityAlpha(BaseAlphaModel):
    def __init__(self, config: LowVolatilityAlphaConfig) -> None:
        super().__init__(config)
        self._vol_window = config.params.get("vol_window_hours", 336)
        self._max_buffer = self._vol_window + 10

    def compute_alpha(self, symbol: str, closes: list[float]) -> float:
        return compute_low_volatility(closes, vol_window=self._vol_window)
