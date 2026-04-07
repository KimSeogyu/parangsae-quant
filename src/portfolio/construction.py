from __future__ import annotations

import json
import math
from decimal import Decimal

import numpy as np
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy

from src.types import AlphaScore, RiskState, UniverseState


class PortfolioConstructionConfig(StrategyConfig):
    num_holdings: int = 15
    entry_rank: int = 12
    exit_rank: int = 18
    rebalance_interval_hours: int = 1
    min_weight_change: float = 0.005
    max_hourly_turnover: float = 0.10
    max_daily_turnover: float = 0.50
    max_position_btc: float = 0.20
    max_position_eth: float = 0.15
    max_position_other: float = 0.07
    min_position: float = 0.01
    zscore_clip: float = 3.0
    alpha_weights: dict[str, float] = {}
    order_id_tag: str = "PC"


def zscore_and_combine(
    alpha_scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    universe: set[str],
    clip: float = 3.0,
) -> dict[str, float]:
    """Cross-sectionally z-score each alpha, clip, then combine with weights.

    For each alpha, filter to universe symbols with finite values, compute
    z = (v - mean) / std (ddof=1). If < 2 values, skip (can't z-score).
    Returns dict of combined scores for all universe symbols.
    """
    combined: dict[str, float] = {sym: 0.0 for sym in universe}

    for alpha_name, scores in alpha_scores.items():
        w = weights.get(alpha_name, 0.0)
        if w == 0.0:
            continue

        # Filter to universe symbols with finite values
        eligible = {
            sym: v
            for sym, v in scores.items()
            if sym in universe and math.isfinite(v)
        }

        if len(eligible) < 2:
            # Can't z-score with fewer than 2 values; skip this alpha
            # But still initialize missing universe symbols to 0
            continue

        symbols = list(eligible.keys())
        vals = np.array([eligible[s] for s in symbols], dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1)

        if std == 0.0:
            zscores = {s: 0.0 for s in symbols}
        else:
            zscores = {
                s: float(np.clip((vals[i] - mean) / std, -clip, clip))
                for i, s in enumerate(symbols)
            }

        for sym, z in zscores.items():
            combined[sym] = combined.get(sym, 0.0) + w * z

    return combined


def select_holdings(
    scores: dict[str, float],
    current_holdings: set[str],
    num_holdings: int,
    entry_rank: int,
    exit_rank: int,
) -> set[str]:
    """Rank eligible symbols and apply hysteresis for holdings selection.

    Eligible = positive and finite score. Ranked descending.
    New entries must rank <= entry_rank (1-based).
    Existing holdings stay if rank <= exit_rank.
    Result capped at num_holdings total.
    """
    # Filter to positive, finite scores only
    eligible = {
        sym: v
        for sym, v in scores.items()
        if math.isfinite(v) and v > 0
    }

    # Rank descending (rank 1 = highest score)
    ranked = sorted(eligible.keys(), key=lambda s: eligible[s], reverse=True)
    rank_of = {sym: i + 1 for i, sym in enumerate(ranked)}  # 1-based

    selected: set[str] = set()

    # Add symbols that pass hysteresis criteria
    for sym, rank in rank_of.items():
        if sym in current_holdings:
            # Existing: keep if still within exit_rank
            if rank <= exit_rank:
                selected.add(sym)
        else:
            # New: must rank within entry_rank
            if rank <= entry_rank:
                selected.add(sym)

    # Enforce total cap: keep highest-ranked if over limit
    if len(selected) > num_holdings:
        selected = set(sorted(selected, key=lambda s: rank_of[s])[:num_holdings])

    return selected


def apply_tiered_caps(
    weights: dict[str, float],
    max_btc: float,
    max_eth: float,
    max_other: float,
) -> dict[str, float]:
    """Cap positions by tier: BTC symbols, ETH symbols, all others.

    Detection: case-insensitive substring match of 'BTC' or 'ETH' in symbol.
    """
    capped = {}
    for sym, w in weights.items():
        sym_upper = sym.upper()
        if "BTC" in sym_upper:
            cap = max_btc
        elif "ETH" in sym_upper:
            cap = max_eth
        else:
            cap = max_other
        capped[sym] = min(w, cap)
    return capped


def compute_target_weights(
    holdings: set[str],
    scores: dict[str, float],
    volatility: dict[str, float],
    risk_scale: float,
    min_position: float,
) -> dict[str, float]:
    """Compute target weights using alpha × (1/vol), normalized, risk-scaled.

    Steps:
    1. raw[sym] = score * (1 / vol)
    2. Normalize so sum(raw) = 1.0
    3. Multiply by risk_scale
    4. Drop positions below min_position
    """
    raws: dict[str, float] = {}
    for sym in holdings:
        alpha = scores.get(sym, 0.0)
        if not math.isfinite(alpha) or alpha <= 0:
            continue
        vol = volatility.get(sym, 1.0)
        inv_vol = 1.0 / vol if vol > 0 else 1.0
        raws[sym] = alpha * inv_vol

    total = sum(raws.values())
    if total <= 0:
        return {}

    weights = {sym: (v / total) * risk_scale for sym, v in raws.items()}

    # Drop below min_position
    weights = {sym: w for sym, w in weights.items() if w >= min_position}

    return weights


def compute_target_deltas(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    threshold: float,
) -> dict[str, float]:
    """Compute weight changes that exceed the threshold gate.

    Symbols in current but not in target are reduced to 0 (exit).
    Only changes >= threshold are included.
    """
    all_symbols = set(target_weights) | set(current_weights)
    deltas = {}
    for symbol in all_symbols:
        target = target_weights.get(symbol, 0.0)
        current = current_weights.get(symbol, 0.0)
        delta = target - current
        if abs(delta) >= threshold:
            deltas[symbol] = delta
    return deltas


def apply_turnover_limit(
    deltas: dict[str, float],
    max_turnover: float,
) -> dict[str, float]:
    """Scale all deltas proportionally if total abs turnover exceeds limit."""
    total = sum(abs(v) for v in deltas.values())
    if total <= max_turnover:
        return deltas
    scale = max_turnover / total
    return {sym: v * scale for sym, v in deltas.items()}


class PortfolioConstruction(Strategy):
    def __init__(self, config: PortfolioConstructionConfig) -> None:
        super().__init__(config)
        self._alpha_scores: dict[str, dict[str, float]] = {}
        self._risk_state: RiskState | None = None
        self._universe: frozenset[str] = frozenset()
        self._last_rebalance_hour: int = -1
        self._last_prices: dict[str, float] = {}
        self._current_holdings: set[str] = set()
        self._daily_turnover: float = 0.0
        self._last_day: int = -1

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
                risk_scale=data.get("risk_scale", 1.0),
                regime_scale=data.get("regime_scale", 1.0),
                vol_scale=data.get("vol_scale", 1.0),
                corr_scale=data.get("corr_scale", 1.0),
                dd_scale=data.get("dd_scale", 1.0),
                drawdown=data.get("drawdown", 0.0),
                portfolio_vol=data.get("portfolio_vol", 0.0),
                avg_correlation=data.get("avg_correlation", 0.0),
                volatility=data.get("volatility", {}),
            )
        elif signal_type == "UniverseState":
            state = UniverseState(
                current=frozenset(data["current"]),
                added=tuple(data["added"]),
                removed=tuple(data["removed"]),
            )
            self._universe = state.current
            for symbol_str in state.removed:
                self._current_holdings.discard(symbol_str)
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

        # Reset daily turnover counter on a new UTC day
        current_day = bar.ts_event // 86_400_000_000_000
        if current_day != self._last_day:
            self._daily_turnover = 0.0
            self._last_day = current_day

        self._rebalance()

    def _rebalance(self) -> None:
        if not self._alpha_scores or self._risk_state is None:
            return

        cfg = self.config
        universe = set(self._universe)

        # 1. Z-score and combine alphas
        combined = zscore_and_combine(
            self._alpha_scores,
            cfg.alpha_weights,
            universe,
            clip=cfg.zscore_clip,
        )

        # 2. Holdings selection with hysteresis
        new_holdings = select_holdings(
            combined,
            self._current_holdings,
            num_holdings=cfg.num_holdings,
            entry_rank=cfg.entry_rank,
            exit_rank=cfg.exit_rank,
        )
        self._current_holdings = new_holdings

        # 3. Compute target weights (alpha × inv-vol, risk-scaled)
        target_weights = compute_target_weights(
            new_holdings,
            combined,
            self._risk_state.volatility,
            risk_scale=self._risk_state.risk_scale,
            min_position=cfg.min_position,
        )

        # 4. Apply tiered position caps
        target_weights = apply_tiered_caps(
            target_weights,
            max_btc=cfg.max_position_btc,
            max_eth=cfg.max_position_eth,
            max_other=cfg.max_position_other,
        )

        # 5. Compute deltas with threshold gating
        current_weights = self._get_current_weights()
        deltas = compute_target_deltas(
            target_weights,
            current_weights,
            threshold=cfg.min_weight_change,
        )

        if not deltas:
            return

        # 6. Apply turnover limits (min of hourly cap and remaining daily budget)
        remaining_daily = max(0.0, cfg.max_daily_turnover - self._daily_turnover)
        effective_max = min(cfg.max_hourly_turnover, remaining_daily)
        deltas = apply_turnover_limit(deltas, max_turnover=effective_max)

        # 7. Execute deltas and track daily turnover
        executed_turnover = 0.0
        for symbol_str, delta in deltas.items():
            self._execute_delta(symbol_str, delta)
            executed_turnover += abs(delta)
        self._daily_turnover += executed_turnover

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

        rounded_qty = instrument.make_qty(Decimal(str(quantity)))
        if float(rounded_qty) <= 0.0:
            return

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=rounded_qty,
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)

    def on_stop(self) -> None:
        for instrument in self.cache.instruments():
            self.cancel_all_orders(instrument.id)
            self.close_all_positions(instrument.id)
