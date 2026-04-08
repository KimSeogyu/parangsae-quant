"""Manifest-driven end-to-end research backtest runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.research.backtest.engine import (
    BacktestConfig,
    run_accurate_backtest,
    run_fast_backtest,
)
from src.research.config import ResearchSettings
from src.research.data.aggregator import compute_log_returns, compute_mid_price
from src.research.features.derivatives import (
    compute_basis,
    compute_basis_zscore,
    compute_funding_change,
    compute_funding_rate,
    compute_funding_zscore,
    compute_oi_change,
    compute_oi_to_volume,
)
from src.research.features.liquidity import (
    compute_amihud,
    compute_depth_top1,
    compute_ob_imbalance,
    compute_qvol_zscore,
    compute_spread_bps,
    compute_spread_zscore,
    compute_vwap_deviation,
)
from src.research.features.preprocess import preprocess_features
from src.research.features.registry import FeatureSpec, load_manifest
from src.research.features.relative import (
    compute_beta_btc,
    compute_resid_momentum,
    compute_resid_rsi,
    compute_resid_zscore,
)
from src.research.features.risk_features import (
    compute_alt_breadth,
    compute_btc_trend,
    compute_corr_btc,
    compute_cross_sectional_corr,
    compute_downside_vol,
    compute_median_funding_breadth,
    compute_realized_vol,
)
from src.research.features.trend import (
    compute_fip_like,
    compute_momentum,
    compute_momentum_voladj,
    compute_trend_linearity,
    compute_wickiness,
)
from src.research.labels.beta import compute_dual_window_beta
from src.research.labels.residual import build_labels


@dataclass
class PreparedResearchInputs:
    features: pd.DataFrame
    labels_y20: pd.Series
    labels_y60: pd.Series
    prices: pd.DataFrame
    btc_returns: pd.Series
    volatilities: pd.DataFrame
    betas_panel: pd.DataFrame
    orderbook_data: pd.DataFrame
    btc_symbol: str
    tradable_symbols: list[str]


@dataclass
class SymbolContext:
    ohlcv: pd.DataFrame
    orderbook: pd.DataFrame
    close: pd.Series
    mid: pd.Series
    returns: pd.Series
    quote_volume: pd.Series
    volume: pd.Series
    funding_rate: pd.Series
    open_interest: pd.Series
    best_bid: pd.Series
    best_ask: pd.Series
    best_bid_size: pd.Series
    best_ask_size: pd.Series
    index_price: pd.Series
    beta: pd.Series | None = None


def _load_parquet_map(root: Path, subdir: str) -> dict[str, pd.DataFrame]:
    path = root / subdir
    if not path.exists():
        return {}

    frames: dict[str, pd.DataFrame] = {}
    for file in sorted(path.glob("*.parquet")):
        df = pd.read_parquet(file).sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        frames[file.stem] = df
    return frames


def _find_btc_symbol(symbols: list[str]) -> str:
    for symbol in symbols:
        if symbol.startswith("BTCUSDT"):
            return symbol
    raise ValueError("BTC research parquet not found. Expected a symbol starting with BTCUSDT.")


def _reindex_series(
    frame: pd.DataFrame | None,
    column: str,
    index: pd.DatetimeIndex,
    fill_value: float = 0.0,
) -> pd.Series:
    if frame is None or column not in frame.columns:
        return pd.Series(fill_value, index=index, dtype=float)
    series = frame[column].reindex(index).ffill()
    return series.fillna(fill_value).astype(float)


def _standardize_btc_column_name(frame: pd.DataFrame, btc_symbol: str) -> pd.DataFrame:
    if btc_symbol == "BTCUSDT":
        return frame
    return frame.rename(columns={btc_symbol: "BTCUSDT"})


def _int_param(spec: FeatureSpec, key: str, default: int) -> int:
    return int(spec.params.get(key, default))


def _build_symbol_contexts(
    common_index: pd.DatetimeIndex,
    symbols: list[str],
    ohlcv_frames: dict[str, pd.DataFrame],
    funding_frames: dict[str, pd.DataFrame],
    oi_frames: dict[str, pd.DataFrame],
    orderbook_frames: dict[str, pd.DataFrame],
) -> dict[str, SymbolContext]:
    contexts: dict[str, SymbolContext] = {}
    for symbol in symbols:
        ohlcv = ohlcv_frames[symbol].reindex(common_index).ffill()
        orderbook = orderbook_frames.get(symbol, pd.DataFrame(index=common_index)).reindex(common_index).ffill()
        close = ohlcv["close"].astype(float)
        mid = compute_mid_price(ohlcv).astype(float)
        contexts[symbol] = SymbolContext(
            ohlcv=ohlcv,
            orderbook=orderbook,
            close=close,
            mid=mid,
            returns=compute_log_returns(mid),
            quote_volume=ohlcv["quote_volume"].astype(float),
            volume=ohlcv["volume"].astype(float),
            funding_rate=_reindex_series(funding_frames.get(symbol), "funding_rate", common_index, 0.0),
            open_interest=_reindex_series(oi_frames.get(symbol), "open_interest", common_index, 0.0),
            best_bid=_reindex_series(orderbook, "best_bid", common_index, float("nan")),
            best_ask=_reindex_series(orderbook, "best_ask", common_index, float("nan")),
            best_bid_size=_reindex_series(orderbook, "best_bid_size", common_index, 0.0),
            best_ask_size=_reindex_series(orderbook, "best_ask_size", common_index, 0.0),
            index_price=_reindex_series(funding_frames.get(symbol), "index_price", common_index, float("nan")),
        )
    return contexts


def _compute_feature_panel(
    spec: FeatureSpec,
    tradable_symbols: list[str],
    contexts: dict[str, SymbolContext],
    btc_symbol: str,
    btc_returns: pd.Series,
    breadth: pd.Series,
    median_funding_breadth: pd.Series,
    xs_corr: pd.Series,
    common_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    btc_close = contexts[btc_symbol].close

    for symbol in tradable_symbols:
        ctx = contexts[symbol]
        if spec.function == "compute_beta_btc":
            data[symbol] = compute_beta_btc(
                ctx.returns,
                btc_returns,
                window_weeks=_int_param(spec, "window_weeks", 1),
            )
        elif spec.function == "compute_resid_momentum":
            data[symbol] = compute_resid_momentum(
                ctx.returns,
                btc_returns,
                ctx.beta,
                horizon_bars=_int_param(spec, "horizon_bars", 12),
            )
        elif spec.function == "compute_resid_zscore":
            data[symbol] = compute_resid_zscore(
                ctx.returns,
                btc_returns,
                ctx.beta,
                horizon_bars=_int_param(spec, "horizon_bars", 12),
            )
        elif spec.function == "compute_resid_rsi":
            data[symbol] = compute_resid_rsi(
                ctx.returns,
                btc_returns,
                ctx.beta,
                period=_int_param(spec, "period", 14),
            )
        elif spec.function == "compute_momentum":
            data[symbol] = compute_momentum(
                ctx.close,
                horizon_bars=_int_param(spec, "horizon_bars", 12),
            )
        elif spec.function == "compute_momentum_voladj":
            data[symbol] = compute_momentum_voladj(
                ctx.close,
                horizon_bars=_int_param(spec, "horizon_bars", 48),
            )
        elif spec.function == "compute_fip_like":
            data[symbol] = compute_fip_like(
                ctx.close,
                horizon_bars=_int_param(spec, "horizon_bars", 288),
            )
        elif spec.function == "compute_trend_linearity":
            data[symbol] = compute_trend_linearity(
                ctx.close,
                horizon_bars=_int_param(spec, "horizon_bars", 288),
            )
        elif spec.function == "compute_wickiness":
            data[symbol] = compute_wickiness(
                ctx.ohlcv,
                horizon_bars=_int_param(spec, "horizon_bars", 288),
            )
        elif spec.function == "compute_qvol_zscore":
            data[symbol] = compute_qvol_zscore(
                ctx.quote_volume,
                horizon_bars=_int_param(spec, "horizon_bars", 288),
            )
        elif spec.function == "compute_amihud":
            data[symbol] = compute_amihud(
                ctx.close,
                ctx.volume,
                horizon_bars=_int_param(spec, "horizon_bars", 288),
            )
        elif spec.function == "compute_vwap_deviation":
            data[symbol] = compute_vwap_deviation(
                ctx.close,
                ctx.quote_volume,
                ctx.volume,
                horizon_bars=_int_param(spec, "horizon_bars", 12),
            )
        elif spec.function == "compute_spread_bps":
            data[symbol] = compute_spread_bps(
                ctx.best_bid.fillna(ctx.mid * 0.999),
                ctx.best_ask.fillna(ctx.mid * 1.001),
            )
        elif spec.function == "compute_spread_zscore":
            spread = compute_spread_bps(
                ctx.best_bid.fillna(ctx.mid * 0.999),
                ctx.best_ask.fillna(ctx.mid * 1.001),
            )
            data[symbol] = compute_spread_zscore(
                spread,
                horizon_bars=_int_param(spec, "horizon_bars", 288),
            )
        elif spec.function == "compute_depth_top1":
            data[symbol] = compute_depth_top1(
                ctx.best_bid_size,
                ctx.best_ask_size,
                ctx.mid,
            )
        elif spec.function == "compute_ob_imbalance":
            data[symbol] = compute_ob_imbalance(
                ctx.best_bid_size,
                ctx.best_ask_size,
            )
        elif spec.function == "compute_funding_rate":
            data[symbol] = compute_funding_rate(ctx.funding_rate)
        elif spec.function == "compute_funding_zscore":
            data[symbol] = compute_funding_zscore(
                ctx.funding_rate,
                horizon_bars=_int_param(spec, "horizon_bars", 2016),
            )
        elif spec.function == "compute_funding_change":
            data[symbol] = compute_funding_change(
                ctx.funding_rate,
                horizon_bars=_int_param(spec, "horizon_bars", 288),
            )
        elif spec.function == "compute_oi_change":
            data[symbol] = compute_oi_change(
                ctx.open_interest,
                horizon_bars=_int_param(spec, "horizon_bars", 12),
            )
        elif spec.function == "compute_oi_to_volume":
            data[symbol] = compute_oi_to_volume(
                ctx.open_interest,
                ctx.volume,
            )
        elif spec.function == "compute_basis":
            spot = ctx.index_price.fillna(ctx.mid)
            data[symbol] = compute_basis(ctx.close, spot)
        elif spec.function == "compute_basis_zscore":
            spot = ctx.index_price.fillna(ctx.mid)
            basis = compute_basis(ctx.close, spot)
            data[symbol] = compute_basis_zscore(
                basis,
                horizon_bars=_int_param(spec, "horizon_bars", 2016),
            )
        elif spec.function == "compute_realized_vol":
            data[symbol] = compute_realized_vol(
                ctx.close,
                horizon_bars=_int_param(spec, "horizon_bars", 12),
            )
        elif spec.function == "compute_downside_vol":
            data[symbol] = compute_downside_vol(
                ctx.close,
                horizon_bars=_int_param(spec, "horizon_bars", 288),
            )
        elif spec.function == "compute_corr_btc":
            data[symbol] = compute_corr_btc(
                ctx.returns,
                btc_returns,
                horizon_bars=_int_param(spec, "horizon_bars", 288),
            )
        elif spec.function == "compute_btc_trend":
            data[symbol] = compute_btc_trend(
                btc_close,
                horizon_bars=_int_param(spec, "horizon_bars", 48),
            )
        elif spec.function == "compute_alt_breadth":
            data[symbol] = breadth
        elif spec.function == "compute_median_funding_breadth":
            data[symbol] = median_funding_breadth
        elif spec.function == "compute_cross_sectional_corr":
            data[symbol] = xs_corr
        else:
            raise ValueError(f"Unsupported manifest function: {spec.function}")

    return pd.DataFrame(data, index=common_index)


def prepare_research_inputs(
    data_path: str | Path,
    settings: ResearchSettings,
    max_symbols: int | None = None,
) -> PreparedResearchInputs:
    """Build research engine inputs from parquet data using the feature manifest."""
    data_path = Path(data_path)
    ohlcv_frames = _load_parquet_map(data_path, "ohlcv_5m")
    funding_frames = _load_parquet_map(data_path, "funding")
    oi_frames = _load_parquet_map(data_path, "oi")
    orderbook_frames = _load_parquet_map(data_path, "orderbook_top1")

    if len(ohlcv_frames) < 2:
        raise ValueError("Need at least BTC plus one alt in data/research/ohlcv_5m.")

    btc_symbol = _find_btc_symbol(list(ohlcv_frames))
    tradable_symbols = [symbol for symbol in sorted(ohlcv_frames) if symbol != btc_symbol]
    if max_symbols is not None:
        tradable_symbols = tradable_symbols[:max_symbols]
    if not tradable_symbols:
        raise ValueError("No tradable alt symbols found in research parquet data.")

    common_index = ohlcv_frames[btc_symbol].index
    for symbol in tradable_symbols:
        common_index = common_index.intersection(ohlcv_frames[symbol].index)
    common_index = pd.DatetimeIndex(common_index).sort_values()
    if len(common_index) == 0:
        raise ValueError("No common timestamps across BTC and alt parquet data.")

    contexts = _build_symbol_contexts(
        common_index,
        [btc_symbol, *tradable_symbols],
        ohlcv_frames,
        funding_frames,
        oi_frames,
        orderbook_frames,
    )
    btc_returns = contexts[btc_symbol].returns.fillna(0.0)
    beta_windows = settings.model.labels.beta_windows_weeks
    betas = {
        symbol: compute_dual_window_beta(
            contexts[symbol].returns,
            btc_returns,
            windows_weeks=beta_windows,
        )
        for symbol in tradable_symbols
    }
    for symbol in tradable_symbols:
        contexts[symbol].beta = betas[symbol]

    label_frames = build_labels(
        {symbol: contexts[symbol].mid for symbol in tradable_symbols},
        contexts[btc_symbol].mid,
        betas,
        primary_horizon_bars=settings.model.labels.primary_horizon_minutes // settings.model.data.research_bar_minutes,
        secondary_horizon_bars=settings.model.labels.secondary_horizon_minutes // settings.model.data.research_bar_minutes,
        primary_weight=settings.model.labels.primary_weight,
        secondary_weight=settings.model.labels.secondary_weight,
    )

    returns_panel = pd.DataFrame(
        {symbol: contexts[symbol].returns for symbol in tradable_symbols},
        index=common_index,
    )
    funding_panel = pd.DataFrame(
        {symbol: contexts[symbol].funding_rate for symbol in tradable_symbols},
        index=common_index,
    )
    breadth = compute_alt_breadth(returns_panel, horizon_bars=48)
    median_funding_breadth = compute_median_funding_breadth(funding_panel)
    xs_corr = compute_cross_sectional_corr(returns_panel, horizon_bars=288)

    processed_panels: dict[str, pd.DataFrame] = {}
    for spec in load_manifest():
        raw_panel = _compute_feature_panel(
            spec,
            tradable_symbols,
            contexts,
            btc_symbol,
            btc_returns,
            breadth,
            median_funding_breadth,
            xs_corr,
            common_index,
        )
        processed_panels[spec.name] = preprocess_features(
            raw_panel,
            clip_val=settings.model.preprocessing.clip,
            min_coverage=settings.model.preprocessing.min_coverage,
        ).fillna(0.0)

    tradable_symbols = [symbol for symbol in tradable_symbols if symbol in label_frames]
    if not tradable_symbols:
        raise ValueError("No tradable symbols remain after label construction.")

    feature_columns = []
    for name, panel in processed_panels.items():
        normalized = panel.reindex(columns=tradable_symbols, fill_value=0.0)
        stacked = normalized.stack(future_stack=True).rename(name)
        feature_columns.append(stacked)
    features = pd.concat(feature_columns, axis=1).sort_index()
    features.index = features.index.set_names(["timestamp", "symbol"])

    y20_panel = pd.DataFrame(
        {symbol: label_frames[symbol]["y20"] for symbol in tradable_symbols},
        index=common_index,
    )
    y60_panel = pd.DataFrame(
        {symbol: label_frames[symbol]["y60"] for symbol in tradable_symbols},
        index=common_index,
    )
    labels_y20 = y20_panel.stack(future_stack=True).rename("y20")
    labels_y60 = y60_panel.stack(future_stack=True).rename("y60")
    labels_y20.index = labels_y20.index.set_names(["timestamp", "symbol"])
    labels_y60.index = labels_y60.index.set_names(["timestamp", "symbol"])

    common_feature_index = features.index.intersection(labels_y20.index).intersection(labels_y60.index)
    features = features.loc[common_feature_index]
    labels_y20 = labels_y20.loc[common_feature_index]
    labels_y60 = labels_y60.loc[common_feature_index]

    prices = pd.DataFrame(
        {symbol: contexts[symbol].mid.reindex(common_index).ffill() for symbol in [btc_symbol, *tradable_symbols]},
        index=common_index,
    )
    prices = _standardize_btc_column_name(prices, btc_symbol)

    volatilities = pd.DataFrame(
        {
            symbol: compute_realized_vol(
                contexts[symbol].close,
                horizon_bars=288,
            ).reindex(common_index)
            for symbol in tradable_symbols
        },
        index=common_index,
    ).ffill().fillna(0.02)
    btc_vol = compute_realized_vol(
        contexts[btc_symbol].close,
        horizon_bars=288,
    ).reindex(common_index).ffill().fillna(0.02)
    volatilities["BTCUSDT"] = btc_vol

    betas_panel = pd.DataFrame(
        {symbol: betas[symbol].reindex(common_index) for symbol in tradable_symbols},
        index=common_index,
    ).ffill().fillna(1.0)
    betas_panel["BTCUSDT"] = 1.0

    orderbook_data = pd.DataFrame(index=common_index)
    for symbol in prices.columns:
        source_symbol = btc_symbol if symbol == "BTCUSDT" and btc_symbol != "BTCUSDT" else symbol
        ctx = contexts[source_symbol]
        orderbook_data[symbol] = prices[symbol]
        orderbook_data[f"{symbol}_low"] = ctx.best_bid.fillna(prices[symbol] * 0.999)
        orderbook_data[f"{symbol}_high"] = ctx.best_ask.fillna(prices[symbol] * 1.001)

    btc_returns = compute_log_returns(prices["BTCUSDT"]).fillna(0.0)

    return PreparedResearchInputs(
        features=features,
        labels_y20=labels_y20,
        labels_y60=labels_y60,
        prices=prices,
        btc_returns=btc_returns,
        volatilities=volatilities,
        betas_panel=betas_panel,
        orderbook_data=orderbook_data,
        btc_symbol="BTCUSDT",
        tradable_symbols=tradable_symbols,
    )


def run_research_backtest(
    data_path: str | Path,
    settings: ResearchSettings,
    mode: str = "fast",
    max_symbols: int | None = None,
) -> tuple[PreparedResearchInputs, object]:
    prepared = prepare_research_inputs(data_path, settings, max_symbols=max_symbols)
    config = BacktestConfig(
        mode=mode,
        maker_fee_bps=settings.model.backtest.maker_fee_bps,
        taker_fee_bps=settings.model.backtest.taker_fee_bps,
        rebalance_bars=max(
            1,
            settings.model.portfolio.rebalance_minutes // settings.model.data.research_bar_minutes,
        ),
    )
    common_kwargs = dict(
        features=prepared.features,
        labels_y20=prepared.labels_y20,
        labels_y60=prepared.labels_y60,
        prices=prepared.prices,
        btc_returns=prepared.btc_returns,
        volatilities=prepared.volatilities,
        betas_panel=prepared.betas_panel,
        config=config,
        train_days=settings.model.training.train_window_days,
        val_days=settings.model.training.validation_window_days,
        alpha_grid=settings.model.ridge.alpha_grid,
    )
    if mode == "accurate":
        report = run_accurate_backtest(
            **common_kwargs,
            orderbook_data=prepared.orderbook_data,
        )
    else:
        report = run_fast_backtest(**common_kwargs)
    return prepared, report


def summarize_report(report: object) -> str:
    return "\n".join(
        [
            f"total_return={getattr(report, 'total_return', 0.0):.6f}",
            f"sharpe_ratio={getattr(report, 'sharpe_ratio', 0.0):.6f}",
            f"mean_rank_ic={getattr(report, 'mean_rank_ic', 0.0):.6f}",
            f"maker_fill_ratio={getattr(report, 'maker_fill_ratio', 0.0):.6f}",
            f"mean_slippage_bps={getattr(report, 'mean_slippage_bps', 0.0):.6f}",
        ]
    )
