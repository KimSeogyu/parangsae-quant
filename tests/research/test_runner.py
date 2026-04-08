"""Tests for the research backtest runner."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.config import load_research_settings
from src.research.runner import prepare_research_inputs, run_research_backtest


def _write_parquet(root: Path, subdir: str, symbol: str, df: pd.DataFrame) -> None:
    path = root / subdir
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / f"{symbol}.parquet")


def _make_ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": 1000.0,
            "quote_volume": close * 1000.0,
        },
        index=close.index,
    )


def _seed_research_data(root: Path) -> None:
    idx = pd.date_range("2025-01-01", periods=16 * 288, freq="5min", tz="UTC")
    btc = pd.Series(20000 * np.exp(np.linspace(0.0, 0.08, len(idx))), index=idx)
    alt1 = pd.Series(100 * np.exp(np.linspace(0.0, 0.18, len(idx))), index=idx)
    alt2 = pd.Series(100 * np.exp(np.linspace(0.0, -0.10, len(idx))), index=idx)

    _write_parquet(root, "ohlcv_5m", "BTCUSDT", _make_ohlcv(btc))
    _write_parquet(root, "ohlcv_5m", "ALT1", _make_ohlcv(alt1))
    _write_parquet(root, "ohlcv_5m", "ALT2", _make_ohlcv(alt2))

    funding_index = idx
    for symbol, bias in [("ALT1", 0.0002), ("ALT2", -0.0001)]:
        funding = pd.DataFrame(
            {
                "funding_rate": bias,
                "mark_price": 100.0,
                "index_price": 99.9,
            },
            index=funding_index,
        )
        oi = pd.DataFrame(
            {
                "open_interest": np.linspace(1_000_000, 1_100_000, len(idx)),
                "open_interest_value": np.linspace(10_000_000, 11_000_000, len(idx)),
            },
            index=funding_index,
        )
        _write_parquet(root, "funding", symbol, funding)
        _write_parquet(root, "oi", symbol, oi)


class TestPrepareResearchInputs:
    def test_builds_panel_inputs_from_parquet(self, tmp_path):
        _seed_research_data(tmp_path)
        settings = load_research_settings()
        settings.model.training.train_window_days = 4
        settings.model.training.validation_window_days = 2
        settings.model.preprocessing.min_coverage = 0.05

        prepared = prepare_research_inputs(tmp_path, settings, max_symbols=2)

        assert not prepared.features.empty
        assert prepared.features.index.nlevels == 2
        assert set(prepared.tradable_symbols) == {"ALT1", "ALT2"}
        assert "BTCUSDT" in prepared.prices.columns
        assert set(prepared.labels_y20.index.get_level_values("symbol")) == {"ALT1", "ALT2"}


class TestRunResearchBacktest:
    def test_runs_fast_mode_end_to_end(self, tmp_path):
        _seed_research_data(tmp_path)
        settings = load_research_settings()
        settings.model.training.train_window_days = 4
        settings.model.training.validation_window_days = 2
        settings.model.preprocessing.min_coverage = 0.05
        settings.model.ridge.alpha_grid = [0.1, 1.0]

        prepared, report = run_research_backtest(tmp_path, settings, mode="fast", max_symbols=2)

        assert len(prepared.tradable_symbols) == 2
        assert np.isfinite(report.mean_rank_ic)
        assert np.isfinite(report.total_return)
