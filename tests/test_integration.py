import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def strategy_settings_yaml():
    return """\
system:
  mode: backtest
  log_level: WARNING

data:
  exchange: binance
  market_types:
    - spot
  timeframe: 1h
  history_days: 365
  storage_path: "{data_dir}"

universe:
  size: 100
  ranking_metric: avg_quote_volume_14d
  ranking_window: 50
  inclusion_rank: 5
  exclusion_rank: 8

alphas:
  crypto_qm_momentum:
    enabled: true
    module: src.alpha.momentum.QMMomentumAlpha
    weight: 0.75
    params:
      lookback_hours: 336
      skip_hours: 24
      vol_window_hours: 720
      fip_enabled: true
      fip_floor: 0.3
      ts_filter: true

  low_volatility:
    enabled: true
    module: src.alpha.low_volatility.LowVolatilityAlpha
    weight: 0.25
    params:
      vol_window_hours: 336

risk:
  btc_regime:
    enabled: false
  volatility_targeting:
    enabled: false
  correlation_monitor:
    enabled: false
  drawdown_scaling:
    enabled: true
    tiers:
      - [0.05, 1.0]
      - [0.10, 0.75]
      - [0.15, 0.50]
      - [0.20, 0.25]
      - [0.25, 0.0]
    min_total_scale: 0.05

portfolio:
  initial_capital: 100000
  num_holdings: 3
  entry_rank: 3
  exit_rank: 5
  rebalance_interval_hours: 1
  min_weight_change: 0.005
  max_hourly_turnover: 0.20
  max_daily_turnover: 0.80
  max_position_btc: 0.40
  max_position_eth: 0.30
  max_position_other: 0.20
  min_position: 0.01
  zscore_clip: 3.0

backtest:
  start_date: "2025-01-01"
  end_date: "2025-02-20"
  fee_rate: 0.001
  slippage_prob: 0.0

binance:
  api_key: ""
  api_secret: ""
"""


def _generate_parquet(data_dir: Path, symbol: str, n: int, base: float, drift: float):
    """Generate synthetic hourly parquet data for one symbol.

    The parquet is written with a UTC DatetimeIndex so that
    catalog.load_bars_from_parquet can read it without modification.
    """
    np.random.seed(hash(symbol) % 2**31)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(drift, 0.015)))
    prices = np.array(prices)

    df = pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.uniform(0.5, 3.0, n),
            "quote_volume": prices * np.random.uniform(0.5, 3.0, n),
        },
        index=dates,
    )
    spot_dir = data_dir / "spot"
    spot_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(spot_dir / f"{symbol}.parquet")


def test_full_strategy_backtest_runs(strategy_settings_yaml):
    """End-to-end: generate data, build engine, run backtest, verify no crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        n = 1200  # 50 days of hourly bars

        # Generate 3 synthetic coins with TestInstrumentProvider fallbacks
        _generate_parquet(data_dir, "BTCUSDT", n, 50000, 0.0003)
        _generate_parquet(data_dir, "ETHUSDT", n, 3000, 0.0002)
        _generate_parquet(data_dir, "ADAUSDT", n, 0.5, 0.0001)

        yaml_content = strategy_settings_yaml.format(data_dir=str(data_dir))
        config_path = Path(tmpdir) / "settings.yaml"
        config_path.write_text(yaml_content)

        from src.config.loader import load_settings
        from src.engine.factory import build_backtest_engine

        settings = load_settings(str(config_path))
        engine = build_backtest_engine(settings)
        engine.run()

        # Verify engine completed
        assert engine.iteration > 0
