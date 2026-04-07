from __future__ import annotations

import json
import math
from collections import defaultdict
import numpy as np
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType
from pydantic import Field

from src.types import RiskState


# ---------------------------------------------------------------------------
# Part A: Pure functions (all exported, tested directly)
# ---------------------------------------------------------------------------


def compute_ema(prices: list[float], period: int) -> float:
    """EMA of the last `period` prices."""
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1)
    ema = float(prices[-period])
    for p in prices[-period + 1:]:
        ema = p * k + ema * (1 - k)
    return ema


def compute_btc_regime_scale(
    btc_price: float,
    btc_ema: float,
    lower: float = 0.90,
    upper: float = 1.00,
    min_exposure: float = 0.3,
) -> float:
    """Linear interpolation: BTC/EMA ratio -> exposure scale."""
    if btc_ema <= 0:
        return 1.0
    ratio = btc_price / btc_ema
    if ratio >= upper:
        return 1.0
    if ratio <= lower:
        return min_exposure
    t = (ratio - lower) / (upper - lower)
    return min_exposure + t * (1.0 - min_exposure)


def compute_vol_target_scale(
    portfolio_vol: float,
    target_vol: float = 0.30,
    min_scale: float = 0.1,
    max_scale: float = 1.5,
) -> float:
    """target_vol / realized_vol, clamped."""
    if portfolio_vol < 1e-8:
        return max_scale
    raw = target_vol / portfolio_vol
    return max(min_scale, min(raw, max_scale))


def compute_correlation_scale(
    avg_corr: float,
    threshold_high: float = 0.70,
    threshold_crisis: float = 0.85,
    min_exposure: float = 0.40,
) -> float:
    """Scale down when avg pairwise correlation exceeds threshold."""
    if avg_corr <= threshold_high:
        return 1.0
    if avg_corr >= threshold_crisis:
        return min_exposure
    t = (avg_corr - threshold_high) / (threshold_crisis - threshold_high)
    return 1.0 - t * (1.0 - min_exposure)


def compute_drawdown_scale(drawdown: float, tiers: list[list[float]]) -> float:
    """Five-tier linear drawdown scaling."""
    dd = abs(drawdown)
    if not tiers:
        return 1.0
    if dd <= tiers[0][0]:
        return tiers[0][1]
    for i in range(1, len(tiers)):
        dd_level, scale = tiers[i]
        prev_dd, prev_scale = tiers[i - 1]
        if dd <= dd_level:
            t = (dd - prev_dd) / (dd_level - prev_dd)
            return prev_scale + t * (scale - prev_scale)
    return tiers[-1][1]


def compute_combined_risk_scale(
    regime: float,
    vol_scale: float,
    corr_scale: float,
    dd_scale: float,
    min_total: float = 0.05,
) -> float:
    """Multiplicative combination with floor."""
    return max(regime * vol_scale * corr_scale * dd_scale, min_total)


def compute_drawdown(equity_curve: list[float]) -> float:
    """Current drawdown from peak."""
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


def compute_rolling_volatility(returns: list[float], window: int) -> float:
    """Rolling volatility from hourly returns."""
    if len(returns) < 2:
        return 0.0
    tail = returns[-window:] if len(returns) >= window else returns
    n = len(tail)
    if n < 2:
        return 0.0
    mean = sum(tail) / n
    variance = sum((r - mean) ** 2 for r in tail) / (n - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Part B: RiskModel Actor
# ---------------------------------------------------------------------------


def _compute_avg_pairwise_correlation(
    return_buffers: dict[str, list[float]],
    sample_coins: int,
    window: int,
) -> float:
    """Picks top N coins by buffer length, builds numpy correlation matrix,
    and averages upper triangle."""
    # Sort by buffer length descending and take top N
    sorted_symbols = sorted(
        return_buffers.keys(),
        key=lambda s: len(return_buffers[s]),
        reverse=True,
    )
    candidates = sorted_symbols[:sample_coins]

    # Need at least 2 coins and sufficient data
    valid = [s for s in candidates if len(return_buffers[s]) >= 2]
    if len(valid) < 2:
        return 0.0

    # Align all series to the same length (use the minimum window)
    min_len = min(min(len(return_buffers[s]), window) for s in valid)
    if min_len < 2:
        return 0.0

    matrix = np.array([return_buffers[s][-min_len:] for s in valid], dtype=float)
    # matrix shape: (n_coins, min_len)

    corr = np.corrcoef(matrix)  # (n_coins, n_coins)

    n = corr.shape[0]
    upper_triangle = []
    for i in range(n):
        for j in range(i + 1, n):
            val = corr[i, j]
            if np.isfinite(val):
                upper_triangle.append(val)

    if not upper_triangle:
        return 0.0
    return float(np.mean(upper_triangle))


class RiskModelConfig(ActorConfig):
    # BTC regime layer
    btc_regime_enabled: bool = True
    ema_period_hours: int = 4800
    regime_min_exposure: float = 0.3
    regime_transition_lower: float = 0.90
    regime_transition_upper: float = 1.00
    btc_instrument_id: str = "BTCUSDT.BINANCE"

    # Volatility targeting layer
    vol_targeting_enabled: bool = True
    target_vol_annual: float = 0.30
    vol_lookback_hours: int = 720
    vol_max_scale: float = 1.5
    vol_min_scale: float = 0.1

    # Correlation layer
    corr_enabled: bool = True
    corr_window_hours: int = 720
    corr_sample_coins: int = 20
    corr_update_interval_hours: int = 4
    corr_threshold_high: float = 0.70
    corr_threshold_crisis: float = 0.85
    corr_min_exposure: float = 0.40

    # Drawdown layer
    dd_enabled: bool = True
    dd_tiers: list = Field(
        default_factory=lambda: [
            [0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0],
        ]
    )
    min_total_scale: float = 0.05

    # Shared
    volatility_window: int = 168


class RiskModel(Actor):
    def __init__(self, config: RiskModelConfig) -> None:
        super().__init__(config)
        # BTC regime
        self._btc_closes: list[float] = []

        # Per-symbol return buffers (for correlation and per-symbol vol)
        self._return_buffers: dict[str, list[float]] = defaultdict(list)
        self._prev_close: dict[str, float] = {}

        # Portfolio-level tracking
        self._portfolio_returns: list[float] = []
        self._equity_curve: list[float] = []

        # Correlation cache
        self._cached_avg_corr: float = 0.0
        self._last_corr_hour: int = -1

        self._last_hour: int = -1

    def on_start(self) -> None:
        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        close = float(bar.close)

        # Track BTC closes for regime EMA
        if symbol == self.config.btc_instrument_id:
            self._btc_closes.append(close)
            max_btc_buf = self.config.ema_period_hours + 100
            if len(self._btc_closes) > max_btc_buf:
                self._btc_closes = self._btc_closes[-max_btc_buf:]

        # Track per-symbol returns
        if symbol in self._prev_close:
            prev = self._prev_close[symbol]
            if prev > 0:
                ret = (close - prev) / prev
                buf = self._return_buffers[symbol]
                buf.append(ret)
                max_buf = max(self.config.vol_lookback_hours, self.config.corr_window_hours) + 10
                if len(buf) > max_buf:
                    buf.pop(0)
        self._prev_close[symbol] = close

        current_hour = bar.ts_event // 3_600_000_000_000
        if current_hour > self._last_hour:
            self._last_hour = current_hour
            self._update_portfolio_equity(bar.ts_event)
            self._publish_risk_state(bar.ts_event)

    def _update_portfolio_equity(self, ts_event: int) -> None:
        account = None
        try:
            from nautilus_trader.model.currencies import USDT
            from nautilus_trader.model.identifiers import Venue
            account = self.portfolio.account(Venue("BINANCE"))
        except Exception:
            pass

        if account is not None:
            try:
                from nautilus_trader.model.currencies import USDT
                balance_money = account.balance_total(USDT)
                if balance_money is not None:
                    equity = float(balance_money.as_double())
                    if self._equity_curve:
                        prev_eq = self._equity_curve[-1]
                        if prev_eq > 0:
                            port_ret = (equity - prev_eq) / prev_eq
                            self._portfolio_returns.append(port_ret)
                            max_port_buf = self.config.vol_lookback_hours + 10
                            if len(self._portfolio_returns) > max_port_buf:
                                self._portfolio_returns.pop(0)
                    self._equity_curve.append(equity)
            except Exception:
                pass

    def _publish_risk_state(self, ts_event: int) -> None:
        # --- Layer 1: BTC regime ---
        if self.config.btc_regime_enabled and self._btc_closes:
            btc_ema = compute_ema(self._btc_closes, self.config.ema_period_hours)
            regime_scale = compute_btc_regime_scale(
                btc_price=self._btc_closes[-1],
                btc_ema=btc_ema,
                lower=self.config.regime_transition_lower,
                upper=self.config.regime_transition_upper,
                min_exposure=self.config.regime_min_exposure,
            )
        else:
            regime_scale = 1.0

        # --- Layer 2: Volatility targeting ---
        portfolio_vol = compute_rolling_volatility(
            self._portfolio_returns, self.config.vol_lookback_hours
        )
        # Annualize from hourly returns (sqrt(8760) hours/year)
        portfolio_vol_annual = portfolio_vol * math.sqrt(8760)

        if self.config.vol_targeting_enabled:
            vol_scale = compute_vol_target_scale(
                portfolio_vol=portfolio_vol_annual,
                target_vol=self.config.target_vol_annual,
                min_scale=self.config.vol_min_scale,
                max_scale=self.config.vol_max_scale,
            )
        else:
            vol_scale = 1.0

        # --- Layer 3: Correlation ---
        current_hour = ts_event // 3_600_000_000_000
        hours_since_update = current_hour - self._last_corr_hour
        if self.config.corr_enabled and hours_since_update >= self.config.corr_update_interval_hours:
            self._cached_avg_corr = _compute_avg_pairwise_correlation(
                self._return_buffers,
                self.config.corr_sample_coins,
                self.config.corr_window_hours,
            )
            self._last_corr_hour = current_hour

        if self.config.corr_enabled:
            corr_scale = compute_correlation_scale(
                avg_corr=self._cached_avg_corr,
                threshold_high=self.config.corr_threshold_high,
                threshold_crisis=self.config.corr_threshold_crisis,
                min_exposure=self.config.corr_min_exposure,
            )
        else:
            corr_scale = 1.0

        # --- Layer 4: Drawdown ---
        drawdown = compute_drawdown(self._equity_curve) if len(self._equity_curve) >= 2 else 0.0

        if self.config.dd_enabled:
            dd_scale = compute_drawdown_scale(
                drawdown=drawdown,
                tiers=self.config.dd_tiers,
            )
        else:
            dd_scale = 1.0

        # --- Combined ---
        risk_scale = compute_combined_risk_scale(
            regime=regime_scale,
            vol_scale=vol_scale,
            corr_scale=corr_scale,
            dd_scale=dd_scale,
            min_total=self.config.min_total_scale,
        )

        # Per-symbol volatility (hourly, not annualized — used by portfolio construction)
        volatility: dict[str, float] = {}
        for symbol, returns in self._return_buffers.items():
            volatility[symbol] = compute_rolling_volatility(returns, self.config.volatility_window)

        state = RiskState(
            risk_scale=risk_scale,
            regime_scale=regime_scale,
            vol_scale=vol_scale,
            corr_scale=corr_scale,
            dd_scale=dd_scale,
            drawdown=drawdown,
            portfolio_vol=portfolio_vol_annual,
            avg_correlation=self._cached_avg_corr,
            volatility=volatility,
        )

        self.publish_signal(
            name="RISK",
            value=json.dumps({
                "type": "RiskState",
                "risk_scale": state.risk_scale,
                "regime_scale": state.regime_scale,
                "vol_scale": state.vol_scale,
                "corr_scale": state.corr_scale,
                "dd_scale": state.dd_scale,
                "drawdown": state.drawdown,
                "portfolio_vol": state.portfolio_vol,
                "avg_correlation": state.avg_correlation,
                "volatility": state.volatility,
            }),
            ts_event=ts_event,
        )
