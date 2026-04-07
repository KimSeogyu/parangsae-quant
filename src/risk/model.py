from __future__ import annotations

import json
import math
from collections import defaultdict

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType

from src.types import RiskState


class RiskModelConfig(ActorConfig):
    max_position_pct: float = 0.05
    max_drawdown: float = 0.15
    max_total_exposure: float = 1.0
    volatility_window: int = 168


def compute_rolling_volatility(returns: list[float], window: int) -> float:
    if len(returns) < 2:
        return 0.0
    tail = returns[-window:] if len(returns) >= window else returns
    n = len(tail)
    if n < 2:
        return 0.0
    mean = sum(tail) / n
    variance = sum((r - mean) ** 2 for r in tail) / (n - 1)
    return math.sqrt(variance)


def compute_drawdown(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
    return max_dd


class RiskModel(Actor):
    def __init__(self, config: RiskModelConfig) -> None:
        super().__init__(config)
        self._return_buffers: dict[str, list[float]] = defaultdict(list)
        self._prev_close: dict[str, float] = {}
        self._equity_curve: list[float] = []
        self._last_hour: int = -1

    def on_start(self) -> None:
        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        close = float(bar.close)

        if symbol in self._prev_close:
            prev = self._prev_close[symbol]
            if prev > 0:
                ret = (close - prev) / prev
                buf = self._return_buffers[symbol]
                buf.append(ret)
                if len(buf) > self.config.volatility_window:
                    buf.pop(0)
        self._prev_close[symbol] = close

        current_hour = bar.ts_event // 3_600_000_000_000
        if current_hour > self._last_hour:
            self._last_hour = current_hour
            self._publish_risk_state(bar.ts_event)

    def _publish_risk_state(self, ts_event: int) -> None:
        volatility = {}
        for symbol, returns in self._return_buffers.items():
            volatility[symbol] = compute_rolling_volatility(
                returns, self.config.volatility_window
            )

        account = None
        try:
            from nautilus_trader.model.currencies import USDT
            from nautilus_trader.model.identifiers import Venue
            account = self.portfolio.account(Venue("BINANCE"))
        except Exception:
            pass

        if account is not None:
            balance_money = account.balance_total(USDT)
            if balance_money is not None:
                self._equity_curve.append(float(balance_money.as_double()))
        drawdown = compute_drawdown(self._equity_curve) if self._equity_curve else 0.0

        risk_halt = drawdown > self.config.max_drawdown

        state = RiskState(
            volatility=volatility,
            drawdown=drawdown,
            total_exposure=0.0,
            risk_halt=risk_halt,
        )
        self.publish_signal(
            name="RISK",
            value=json.dumps({
                "type": "RiskState",
                "volatility": state.volatility,
                "drawdown": state.drawdown,
                "total_exposure": state.total_exposure,
                "risk_halt": state.risk_halt,
            }),
            ts_event=ts_event,
        )
