"""Walk-forward backtest engine orchestrating the full research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.research.analytics.report import BacktestReport, generate_report
from src.research.backtest.accurate import simulate_accurate_period
from src.research.backtest.fast import compute_fast_portfolio_return, simulate_fast_fills
from src.research.execution.passive import OrderState, compute_maker_fill_ratio
from src.research.models.walkforward import run_walk_forward, summarize_walk_forward
from src.research.portfolio.construction import build_portfolio


@dataclass
class BacktestConfig:
    mode: str = "fast"
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    initial_capital: float = 100000.0
    rebalance_bars: int = 4


@dataclass
class SimulationState:
    equity: float = 100000.0
    positions: dict[str, float] = field(default_factory=dict)
    longs: set[str] = field(default_factory=set)
    shorts: set[str] = field(default_factory=set)
    pending_orders: list[OrderState] = field(default_factory=list)
    returns_history: list[float] = field(default_factory=list)
    return_timestamps: list[pd.Timestamp] = field(default_factory=list)
    weights_history: list[dict[str, float]] = field(default_factory=list)
    total_fees: float = 0.0


def _to_score_frame(test_index: pd.Index, predictions: np.ndarray, symbols: list[str]) -> pd.DataFrame:
    if len(predictions) == 0:
        return pd.DataFrame(columns=["score"])

    score_frame = pd.DataFrame({"score": predictions}, index=test_index)
    if isinstance(test_index, pd.MultiIndex):
        score_frame.index = score_frame.index.set_names(["timestamp", "symbol"])
        return score_frame

    if not symbols:
        raise ValueError("At least one symbol is required for non-panel predictions.")

    score_frame["timestamp"] = pd.DatetimeIndex(test_index)
    score_frame["symbol"] = symbols[0]
    return score_frame.set_index(["timestamp", "symbol"])[["score"]]


def _score_map_by_timestamp(
    test_index: pd.Index,
    predictions: np.ndarray,
    tradable_symbols: list[str],
) -> dict[pd.Timestamp, dict[str, float]]:
    score_frame = _to_score_frame(test_index, predictions, tradable_symbols)
    if score_frame.empty:
        return {}

    result: dict[pd.Timestamp, dict[str, float]] = {}
    for timestamp, frame in score_frame.groupby(level=0):
        series = frame["score"]
        if isinstance(series.index, pd.MultiIndex):
            series.index = series.index.get_level_values(1)
        scores = {
            str(symbol): float(score)
            for symbol, score in series.items()
            if str(symbol) in tradable_symbols
        }
        if scores:
            result[pd.Timestamp(timestamp)] = scores
    return result


def _panel_row_as_dict(panel: pd.DataFrame, timestamp: pd.Timestamp, fallback: float) -> dict[str, float]:
    if timestamp not in panel.index:
        return {str(col): fallback for col in panel.columns}

    row = panel.loc[timestamp]
    if isinstance(row, pd.Series):
        return {
            str(symbol): float(value) if pd.notna(value) else fallback
            for symbol, value in row.items()
        }
    return {str(col): fallback for col in panel.columns}


def _price_return_map(
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    weights: dict[str, float],
) -> dict[str, float]:
    returns: dict[str, float] = {}
    if start == end or start not in prices.index or end not in prices.index:
        return returns

    for symbol in weights:
        if symbol not in prices.columns:
            continue
        p0 = prices.loc[start, symbol]
        p1 = prices.loc[end, symbol]
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            returns[symbol] = float(p1 / p0 - 1)
    return returns


def _is_btc_symbol(symbol: str) -> bool:
    return str(symbol).startswith("BTCUSDT")


def _build_report(
    state: SimulationState,
    btc_returns: pd.Series,
    wf_summary: dict,
    maker_fill_ratio: float = 0.0,
    mean_slippage_bps: float = 0.0,
) -> BacktestReport:
    if state.return_timestamps:
        portfolio_returns = pd.Series(
            state.returns_history,
            index=pd.DatetimeIndex(state.return_timestamps),
            dtype=float,
        )
    else:
        portfolio_returns = pd.Series(state.returns_history, dtype=float)
    btc_slice = (
        btc_returns.reindex(portfolio_returns.index).fillna(0.0)
        if isinstance(portfolio_returns.index, pd.DatetimeIndex)
        else btc_returns.iloc[: len(portfolio_returns)]
    )
    return generate_report(
        portfolio_returns=portfolio_returns,
        btc_returns=btc_slice if len(portfolio_returns) > 0 else pd.Series(dtype=float),
        weights_history=state.weights_history,
        walk_forward_summary=wf_summary,
        maker_fill_ratio=maker_fill_ratio,
        mean_slippage_bps=mean_slippage_bps,
        total_fees=state.total_fees,
    )


def run_fast_backtest(
    features: pd.DataFrame,
    labels_y20: pd.Series,
    labels_y60: pd.Series,
    prices: pd.DataFrame,
    btc_returns: pd.Series,
    volatilities: pd.DataFrame,
    betas_panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    train_days: int = 90,
    val_days: int = 14,
    alpha_grid: list[float] | None = None,
) -> BacktestReport:
    """Run fast-mode backtest with deterministic score-driven fills."""
    if config is None:
        config = BacktestConfig(mode="fast")

    wf_results = run_walk_forward(
        features,
        labels_y20,
        labels_y60,
        train_days=train_days,
        val_days=val_days,
        alpha_grid=alpha_grid,
        run_baselines=True,
    )
    wf_summary = summarize_walk_forward(wf_results)
    state = SimulationState(equity=config.initial_capital)

    tradable_symbols = [str(col) for col in prices.columns if not _is_btc_symbol(str(col))]

    for result in wf_results:
        score_map = _score_map_by_timestamp(result.test_index, result.test_predictions, tradable_symbols)
        timestamps = sorted(score_map)
        if len(timestamps) < 2:
            continue

        for idx in range(0, len(timestamps) - 1, config.rebalance_bars):
            timestamp = timestamps[idx]
            next_timestamp = timestamps[min(idx + config.rebalance_bars, len(timestamps) - 1)]
            scores = score_map.get(timestamp, {})
            if not scores:
                continue

            vols = _panel_row_as_dict(volatilities, timestamp, 0.02)
            betas = _panel_row_as_dict(betas_panel, timestamp, 1.0)
            target_weights, longs, shorts = build_portfolio(
                scores,
                vols,
                betas,
                state.longs,
                state.shorts,
            )

            achieved_weights, fee_cost = simulate_fast_fills(
                target_weights,
                state.positions,
                _panel_row_as_dict(prices, timestamp, 0.0),
                config.maker_fee_bps,
            )
            returns = _price_return_map(prices, timestamp, next_timestamp, achieved_weights)
            port_ret = compute_fast_portfolio_return(achieved_weights, returns)
            net_ret = port_ret - fee_cost

            state.longs = longs
            state.shorts = shorts
            if not state.weights_history:
                state.weights_history.append(dict(state.positions))
            state.positions = achieved_weights
            state.equity *= 1 + net_ret
            state.total_fees += fee_cost
            state.returns_history.append(net_ret)
            state.return_timestamps.append(next_timestamp)
            state.weights_history.append(dict(achieved_weights))

    return _build_report(state, btc_returns, wf_summary)


def run_accurate_backtest(
    features: pd.DataFrame,
    labels_y20: pd.Series,
    labels_y60: pd.Series,
    prices: pd.DataFrame,
    btc_returns: pd.Series,
    volatilities: pd.DataFrame,
    betas_panel: pd.DataFrame,
    orderbook_data: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    train_days: int = 90,
    val_days: int = 14,
    alpha_grid: list[float] | None = None,
) -> BacktestReport:
    """Run accurate-mode backtest with deterministic passive fill simulation."""
    if config is None:
        config = BacktestConfig(mode="accurate")

    wf_results = run_walk_forward(
        features,
        labels_y20,
        labels_y60,
        train_days=train_days,
        val_days=val_days,
        alpha_grid=alpha_grid,
        run_baselines=True,
    )
    wf_summary = summarize_walk_forward(wf_results)
    state = SimulationState(equity=config.initial_capital)

    tradable_symbols = [str(col) for col in prices.columns if not _is_btc_symbol(str(col))]
    all_orders: list[OrderState] = []
    all_slippages: list[float] = []

    for result in wf_results:
        score_map = _score_map_by_timestamp(result.test_index, result.test_predictions, tradable_symbols)
        timestamps = sorted(score_map)
        if len(timestamps) < 2:
            continue

        for idx in range(0, len(timestamps) - 1, config.rebalance_bars):
            timestamp = timestamps[idx]
            next_timestamp = timestamps[min(idx + config.rebalance_bars, len(timestamps) - 1)]
            scores = score_map.get(timestamp, {})
            if not scores:
                continue

            vols = _panel_row_as_dict(volatilities, timestamp, 0.02)
            betas = _panel_row_as_dict(betas_panel, timestamp, 1.0)
            target_weights, longs, shorts = build_portfolio(
                scores,
                vols,
                betas,
                state.longs,
                state.shorts,
            )

            bar_window = prices.loc[(prices.index >= timestamp) & (prices.index <= next_timestamp)]
            if orderbook_data is not None and not orderbook_data.empty:
                bar_window = orderbook_data.loc[
                    (orderbook_data.index >= timestamp) & (orderbook_data.index <= next_timestamp)
                ]

            achieved_weights, orders, fee_cost = simulate_accurate_period(
                target_weights,
                state.positions,
                bar_window,
                maker_fee_bps=config.maker_fee_bps,
                taker_fee_bps=config.taker_fee_bps,
            )
            returns = _price_return_map(prices, timestamp, next_timestamp, achieved_weights)
            port_ret = compute_fast_portfolio_return(achieved_weights, returns)
            net_ret = port_ret - fee_cost

            state.longs = longs
            state.shorts = shorts
            if not state.weights_history:
                state.weights_history.append(dict(state.positions))
            state.positions = achieved_weights
            state.equity *= 1 + net_ret
            state.total_fees += fee_cost
            state.returns_history.append(net_ret)
            state.return_timestamps.append(next_timestamp)
            state.weights_history.append(dict(achieved_weights))
            all_orders.extend(orders)
            all_slippages.extend(
                [float(order.slippage_bps) for order in orders if getattr(order, "slippage_bps", None) is not None]
            )

    maker_ratio = compute_maker_fill_ratio(all_orders)
    mean_slippage_bps = float(np.mean(all_slippages)) if all_slippages else 0.0
    return _build_report(state, btc_returns, wf_summary, maker_ratio, mean_slippage_bps)
