from __future__ import annotations

from pydantic import Field

from src.alpha.base import BaseAlphaConfig, BaseAlphaModel


class MomentumAlphaConfig(BaseAlphaConfig):
    alpha_name: str = "momentum"
    params: dict = Field(default_factory=lambda: {"lookback": 24})


def compute_momentum(closes: list[float], lookback: int) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    old = closes[-(lookback + 1)]
    current = closes[-1]
    if old == 0:
        return 0.0
    return (current - old) / old


class MomentumAlpha(BaseAlphaModel):
    def __init__(self, config: MomentumAlphaConfig) -> None:
        super().__init__(config)
        self._lookback = config.params.get("lookback", 24)
        self._max_buffer = self._lookback + 10

    def compute_alpha(self, symbol: str, closes: list[float]) -> float:
        return compute_momentum(closes, self._lookback)
