# tests/test_config.py
from src.config.loader import load_settings
from src.config.models import Settings


def test_settings_parses_valid_config():
    raw = {
        "system": {"mode": "backtest", "log_level": "INFO"},
        "data": {
            "exchange": "binance",
            "market_types": ["spot", "futures"],
            "timeframe": "1h",
            "history_days": 365,
            "storage_path": "data/",
        },
        "universe": {
            "ranking_metric": "quote_volume",
            "ranking_window": 336,
            "inclusion_rank": 100,
            "exclusion_rank": 150,
        },
        "alphas": {
            "momentum": {
                "enabled": True,
                "module": "src.alpha.momentum.MomentumAlpha",
                "weight": 0.5,
                "params": {"lookback": 24},
            },
        },
        "risk": {
            "max_position_pct": 0.05,
            "max_drawdown": 0.15,
            "max_total_exposure": 1.0,
            "volatility_window": 168,
        },
        "portfolio": {
            "initial_capital": 100000,
            "rebalance_interval_hours": 1,
            "min_trade_threshold": 0.001,
        },
        "backtest": {
            "start_date": "2025-04-07",
            "end_date": "2026-04-07",
            "fee_rate": 0.001,
            "slippage_prob": 0.5,
        },
        "binance": {"api_key": "test", "api_secret": "test"},
    }
    settings = Settings(**raw)
    assert settings.system.mode == "backtest"
    assert settings.universe.inclusion_rank == 100
    assert settings.universe.exclusion_rank == 150
    assert settings.alphas["momentum"].weight == 0.5
    assert settings.portfolio.initial_capital == 100000


def test_settings_rejects_invalid_exclusion_rank():
    """exclusion_rank must be > inclusion_rank."""
    from pydantic import ValidationError
    import pytest

    raw = {
        "system": {"mode": "backtest", "log_level": "INFO"},
        "data": {
            "exchange": "binance",
            "market_types": ["spot"],
            "timeframe": "1h",
            "history_days": 365,
            "storage_path": "data/",
        },
        "universe": {
            "ranking_metric": "quote_volume",
            "ranking_window": 336,
            "inclusion_rank": 100,
            "exclusion_rank": 50,
        },
        "alphas": {},
        "risk": {
            "max_position_pct": 0.05,
            "max_drawdown": 0.15,
            "max_total_exposure": 1.0,
            "volatility_window": 168,
        },
        "portfolio": {
            "initial_capital": 100000,
            "rebalance_interval_hours": 1,
            "min_trade_threshold": 0.001,
        },
        "backtest": {
            "start_date": "2025-04-07",
            "end_date": "2026-04-07",
            "fee_rate": 0.001,
            "slippage_prob": 0.5,
        },
        "binance": {"api_key": "", "api_secret": ""},
    }
    with pytest.raises(ValidationError):
        Settings(**raw)


def test_load_settings_from_yaml(tmp_path):
    yaml_content = """\
system:
  mode: backtest
  log_level: DEBUG
data:
  exchange: binance
  market_types: [spot]
  timeframe: 1h
  history_days: 30
  storage_path: data/
universe:
  ranking_metric: quote_volume
  ranking_window: 336
  inclusion_rank: 100
  exclusion_rank: 150
alphas: {}
risk:
  max_position_pct: 0.05
  max_drawdown: 0.15
  max_total_exposure: 1.0
  volatility_window: 168
portfolio:
  initial_capital: 50000
  rebalance_interval_hours: 1
  min_trade_threshold: 0.001
backtest:
  start_date: "2025-01-01"
  end_date: "2025-12-31"
  fee_rate: 0.001
  slippage_prob: 0.5
binance:
  api_key: ${TEST_KEY}
  api_secret: ${TEST_SECRET}
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml_content)

    import os
    os.environ["TEST_KEY"] = "my_key"
    os.environ["TEST_SECRET"] = "my_secret"

    settings = load_settings(str(config_file))
    assert settings.system.log_level == "DEBUG"
    assert settings.binance.api_key == "my_key"
    assert settings.binance.api_secret == "my_secret"
    assert settings.portfolio.initial_capital == 50000

    del os.environ["TEST_KEY"]
    del os.environ["TEST_SECRET"]
