from __future__ import annotations

import json
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy

from src.types import AlphaScore, RiskState, UniverseState


class PortfolioConstructionConfig(StrategyConfig):
    rebalance_interval_hours: int = 1
    min_trade_threshold: float = 0.001
    max_position_pct: float = 0.05
    max_total_exposure: float = 1.0
    alpha_weights: dict[str, float] = {}
    order_id_tag: str = "PC"


def combine_alphas(
    alpha_scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    universe: set[str],
) -> dict[str, float]:
    combined: dict[str, float] = {}
    for alpha_name, scores in alpha_scores.items():
        w = weights.get(alpha_name, 0.0)
        for symbol, score in scores.items():
            if symbol in universe:
                combined[symbol] = combined.get(symbol, 0.0) + w * score
    return combined


def apply_constraints(
    raw_weights: dict[str, float],
    volatility: dict[str, float],
    max_position_pct: float,
    max_total_exposure: float,
    risk_halt: bool,
) -> dict[str, float]:
    if risk_halt:
        return {s: 0.0 for s in raw_weights}

    inv_vol_weights = {}
    for symbol, weight in raw_weights.items():
        vol = volatility.get(symbol, 1.0)
        inv_vol = 1.0 / vol if vol > 0 else 1.0
        inv_vol_weights[symbol] = weight * inv_vol

    total_abs = sum(abs(w) for w in inv_vol_weights.values())
    if total_abs > 0:
        scale = max_total_exposure / total_abs
        normalized = {s: w * scale for s, w in inv_vol_weights.items()}
    else:
        normalized = inv_vol_weights

    clipped = {}
    for symbol, weight in normalized.items():
        if weight > max_position_pct:
            clipped[symbol] = max_position_pct
        elif weight < -max_position_pct:
            clipped[symbol] = -max_position_pct
        else:
            clipped[symbol] = weight

    return clipped


def compute_target_deltas(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    threshold: float,
) -> dict[str, float]:
    all_symbols = set(target_weights) | set(current_weights)
    deltas = {}
    for symbol in all_symbols:
        target = target_weights.get(symbol, 0.0)
        current = current_weights.get(symbol, 0.0)
        delta = target - current
        if abs(delta) >= threshold:
            deltas[symbol] = delta
    return deltas


class PortfolioConstruction(Strategy):
    def __init__(self, config: PortfolioConstructionConfig) -> None:
        super().__init__(config)
        self._alpha_scores: dict[str, dict[str, float]] = {}
        self._risk_state: RiskState | None = None
        self._universe: frozenset[str] = frozenset()
        self._last_rebalance_hour: int = -1
        self._last_prices: dict[str, float] = {}

    def on_start(self) -> None:
        self.subscribe_signal("ALPHA")
        self.subscribe_signal("RISK")
        self.subscribe_signal("UNIVERSE")

        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_signal(self, signal) -> None:
        raw = signal.value
        if not isinstance(raw, str):
            return
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        signal_type = data.get("type")
        if signal_type == "AlphaScore":
            score = AlphaScore(name=data["name"], scores=data["scores"])
            self._alpha_scores.setdefault(score.name, {}).update(score.scores)
        elif signal_type == "RiskState":
            self._risk_state = RiskState(
                volatility=data["volatility"],
                drawdown=data["drawdown"],
                total_exposure=data["total_exposure"],
                risk_halt=data["risk_halt"],
            )
        elif signal_type == "UniverseState":
            state = UniverseState(
                current=frozenset(data["current"]),
                added=tuple(data["added"]),
                removed=tuple(data["removed"]),
            )
            self._universe = state.current
            for symbol_str in state.removed:
                try:
                    iid = InstrumentId.from_str(symbol_str)
                    self.close_all_positions(iid)
                except Exception:
                    pass

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        self._last_prices[symbol] = float(bar.close)

        current_hour = bar.ts_event // 3_600_000_000_000
        if current_hour <= self._last_rebalance_hour:
            return
        if not self._universe:
            return
        self._last_rebalance_hour = current_hour
        self._rebalance()

    def _rebalance(self) -> None:
        if not self._alpha_scores or self._risk_state is None:
            return

        combined = combine_alphas(
            self._alpha_scores,
            self.config.alpha_weights,
            set(self._universe),
        )

        targets = apply_constraints(
            raw_weights=combined,
            volatility=self._risk_state.volatility,
            max_position_pct=self.config.max_position_pct,
            max_total_exposure=self.config.max_total_exposure,
            risk_halt=self._risk_state.risk_halt,
        )

        current_weights = self._get_current_weights()

        deltas = compute_target_deltas(
            targets, current_weights, self.config.min_trade_threshold
        )

        for symbol_str, delta in deltas.items():
            self._execute_delta(symbol_str, delta)

    def _get_current_weights(self) -> dict[str, float]:
        from nautilus_trader.model.currencies import USDT

        venue = Venue("BINANCE")
        account = self.portfolio.account(venue)
        if account is None:
            return {}

        balance_money = account.balance_total(USDT)
        if balance_money is None:
            return {}
        total_equity = float(balance_money.as_double())
        if total_equity <= 0:
            return {}

        weights = {}
        for instrument in self.cache.instruments():
            symbol_str = str(instrument.id)
            net_pos = float(self.portfolio.net_position(instrument.id))
            if net_pos != 0.0:
                last_price = self._last_prices.get(symbol_str, 0.0)
                notional = abs(net_pos * last_price)
                sign = 1.0 if net_pos > 0 else -1.0
                weights[symbol_str] = sign * notional / total_equity
        return weights

    def _execute_delta(self, symbol_str: str, delta: float) -> None:
        try:
            instrument_id = InstrumentId.from_str(symbol_str)
        except Exception:
            return

        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            return

        from nautilus_trader.model.currencies import USDT

        venue = Venue("BINANCE")
        account = self.portfolio.account(venue)
        if account is None:
            return

        balance_money = account.balance_total(USDT)
        if balance_money is None:
            return
        total_equity = float(balance_money.as_double())
        notional = abs(delta) * total_equity

        last_price = self._last_prices.get(symbol_str, 0.0)
        if last_price == 0:
            return

        quantity = notional / last_price
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=instrument.make_qty(Decimal(str(quantity))),
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)

    def on_stop(self) -> None:
        for instrument in self.cache.instruments():
            self.cancel_all_orders(instrument.id)
            self.close_all_positions(instrument.id)
