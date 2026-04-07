import pytest
from src.config.models import Settings


VALID_YAML = {
    "system": {"mode": "backtest", "log_level": "INFO"},
    "data": {
        "exchange": "binance",
        "market_types": ["spot", "futures"],
        "timeframe": "1h",
        "history_days": 365,
        "storage_path": "data/",
    },
    "universe": {
        "size": 100,
        "ranking_metric": "avg_quote_volume_14d",
        "ranking_window": 336,
        "inclusion_rank": 100,
        "exclusion_rank": 150,
        "exclude_stablecoins": True,
        "min_age_days": 30,
    },
    "alphas": {
        "crypto_qm_momentum": {
            "enabled": True,
            "module": "src.alpha.momentum.QMMomentumAlpha",
            "weight": 0.75,
            "params": {
                "lookback_hours": 336,
                "skip_hours": 24,
                "vol_window_hours": 720,
                "fip_enabled": True,
                "fip_floor": 0.3,
                "ts_filter": True,
            },
        },
        "low_volatility": {
            "enabled": True,
            "module": "src.alpha.low_volatility.LowVolatilityAlpha",
            "weight": 0.25,
            "params": {
                "vol_window_hours": 336,
            },
        },
    },
    "risk": {
        "btc_regime": {
            "enabled": True,
            "ema_period_hours": 4800,
            "min_exposure": 0.3,
            "transition_lower": 0.90,
            "transition_upper": 1.00,
        },
        "volatility_targeting": {
            "enabled": True,
            "target_vol_annual": 0.30,
            "lookback_hours": 720,
            "max_scale": 1.5,
            "min_scale": 0.1,
        },
        "correlation_monitor": {
            "enabled": True,
            "window_hours": 720,
            "sample_coins": 20,
            "update_interval_hours": 4,
            "threshold_high": 0.70,
            "threshold_crisis": 0.85,
            "min_exposure": 0.40,
        },
        "drawdown_scaling": {
            "enabled": True,
            "tiers": [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]],
            "min_total_scale": 0.05,
        },
    },
    "portfolio": {
        "initial_capital": 100000,
        "num_holdings": 15,
        "entry_rank": 12,
        "exit_rank": 18,
        "rebalance_interval_hours": 1,
        "min_weight_change": 0.005,
        "max_hourly_turnover": 0.10,
        "max_daily_turnover": 0.50,
        "max_position_btc": 0.20,
        "max_position_eth": 0.15,
        "max_position_other": 0.07,
        "min_position": 0.01,
        "zscore_clip": 3.0,
    },
    "backtest": {
        "start_date": "2025-04-07",
        "end_date": "2026-04-07",
        "fee_rate": 0.001,
        "slippage_prob": 0.5,
    },
    "binance": {"api_key": "", "api_secret": ""},
}


def test_settings_parses_valid_yaml():
    settings = Settings(**VALID_YAML)
    assert settings.alphas["crypto_qm_momentum"].weight == 0.75
    assert settings.risk.btc_regime.ema_period_hours == 4800
    assert settings.portfolio.num_holdings == 15
    assert settings.risk.drawdown_scaling.tiers[0] == [0.05, 1.0]


def test_settings_rejects_exclusion_le_inclusion():
    bad = {**VALID_YAML, "universe": {**VALID_YAML["universe"], "exclusion_rank": 50}}
    with pytest.raises(ValueError, match="exclusion_rank"):
        Settings(**bad)


def test_settings_rejects_negative_vol_target():
    bad_risk = {**VALID_YAML["risk"]}
    bad_risk["volatility_targeting"] = {**bad_risk["volatility_targeting"], "target_vol_annual": -0.1}
    with pytest.raises(ValueError):
        Settings(**{**VALID_YAML, "risk": bad_risk})
