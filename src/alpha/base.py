from __future__ import annotations

import json
from collections import defaultdict

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType

from src.types import AlphaScore


class BaseAlphaConfig(ActorConfig):
    alpha_name: str = ""
    params: dict = {}


class BaseAlphaModel(Actor):
    def __init__(self, config: BaseAlphaConfig) -> None:
        super().__init__(config)
        self._close_buffers: dict[str, list[float]] = defaultdict(list)
        self._max_buffer: int = 500

    def on_start(self) -> None:
        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        buf = self._close_buffers[symbol]
        buf.append(float(bar.close))
        if len(buf) > self._max_buffer:
            buf.pop(0)

        alpha = self.compute_alpha(symbol, buf)
        self._publish_if_ready(symbol, alpha, bar.ts_event)

    def compute_alpha(self, symbol: str, closes: list[float]) -> float:
        raise NotImplementedError

    def _publish_if_ready(self, symbol: str, alpha: float, ts_event: int) -> None:
        score = AlphaScore(name=self.config.alpha_name, scores={symbol: alpha})
        self.publish_signal(
            name="ALPHA",
            value=json.dumps({"type": "AlphaScore", "name": score.name, "scores": score.scores}),
            ts_event=ts_event,
        )
