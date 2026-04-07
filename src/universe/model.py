from __future__ import annotations

import json
from collections import defaultdict

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType

from src.types import UniverseState


class UniverseModelConfig(ActorConfig):
    ranking_window: int = 336
    inclusion_rank: int = 100
    exclusion_rank: int = 150


def compute_rankings(volume_buffers: dict[str, list[float]]) -> dict[str, int]:
    averages = {}
    for symbol, volumes in volume_buffers.items():
        if volumes:
            averages[symbol] = sum(volumes) / len(volumes)
    sorted_symbols = sorted(averages, key=lambda s: averages[s], reverse=True)
    return {symbol: rank + 1 for rank, symbol in enumerate(sorted_symbols)}


def apply_hysteresis(
    current_universe: set[str],
    rankings: dict[str, int],
    inclusion_rank: int,
    exclusion_rank: int,
) -> tuple[set[str], tuple[str, ...], tuple[str, ...]]:
    new_universe = set()
    added = []
    removed = []

    for symbol, rank in rankings.items():
        if symbol in current_universe:
            if rank <= exclusion_rank:
                new_universe.add(symbol)
            else:
                removed.append(symbol)
        else:
            if rank <= inclusion_rank:
                new_universe.add(symbol)
                added.append(symbol)

    return new_universe, tuple(added), tuple(removed)


class UniverseModel(Actor):
    def __init__(self, config: UniverseModelConfig) -> None:
        super().__init__(config)
        self._volume_buffers: dict[str, list[float]] = defaultdict(list)
        self._universe: set[str] = set()
        self._last_ranking_hour: int = -1
        self._bars_seen: int = 0

    def on_start(self) -> None:
        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        notional_volume = float(bar.close) * float(bar.volume)

        buf = self._volume_buffers[symbol]
        buf.append(notional_volume)
        if len(buf) > self.config.ranking_window:
            buf.pop(0)

        self._bars_seen += 1
        current_hour = bar.ts_event // 3_600_000_000_000
        if current_hour > self._last_ranking_hour:
            self._last_ranking_hour = current_hour
            self._try_rebalance()

    def _try_rebalance(self) -> None:
        max_buf_len = max(
            (len(buf) for buf in self._volume_buffers.values()), default=0
        )
        if max_buf_len < self.config.ranking_window:
            return

        rankings = compute_rankings(self._volume_buffers)
        new_universe, added, removed = apply_hysteresis(
            current_universe=self._universe,
            rankings=rankings,
            inclusion_rank=self.config.inclusion_rank,
            exclusion_rank=self.config.exclusion_rank,
        )
        self._universe = new_universe

        state = UniverseState(
            current=frozenset(new_universe),
            added=added,
            removed=removed,
        )
        self.publish_signal(
            name="UNIVERSE",
            value=json.dumps({
                "type": "UniverseState",
                "current": list(state.current),
                "added": list(state.added),
                "removed": list(state.removed),
            }),
            ts_event=0,
        )
