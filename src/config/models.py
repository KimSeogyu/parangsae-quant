from __future__ import annotations

from pydantic import BaseModel, model_validator, field_validator


class SystemConfig(BaseModel):
    mode: str
    log_level: str = "INFO"


class DataConfig(BaseModel):
    exchange: str
    market_types: list[str]
    timeframe: str
    history_days: int
    storage_path: str


class UniverseConfig(BaseModel):
    size: int = 100
    ranking_metric: str = "avg_quote_volume_14d"
    ranking_window: int = 336
    inclusion_rank: int = 100
    exclusion_rank: int = 150
    exclude_stablecoins: bool = True
    min_age_days: int = 30

    @model_validator(mode="after")
    def check_rank_ordering(self) -> UniverseConfig:
        if self.exclusion_rank <= self.inclusion_rank:
            raise ValueError(
                f"exclusion_rank ({self.exclusion_rank}) must be > "
                f"inclusion_rank ({self.inclusion_rank})"
            )
        return self


class AlphaConfig(BaseModel):
    enabled: bool
    module: str
    weight: float
    params: dict = {}


class BtcRegimeConfig(BaseModel):
    enabled: bool = True
    ema_period_hours: int = 4800
    min_exposure: float = 0.3
    transition_lower: float = 0.90
    transition_upper: float = 1.00


class VolatilityTargetingConfig(BaseModel):
    enabled: bool = True
    target_vol_annual: float = 0.30
    lookback_hours: int = 720
    max_scale: float = 1.5
    min_scale: float = 0.1

    @field_validator("target_vol_annual")
    @classmethod
    def vol_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("target_vol_annual must be positive")
        return v


class CorrelationMonitorConfig(BaseModel):
    enabled: bool = True
    window_hours: int = 720
    sample_coins: int = 20
    update_interval_hours: int = 4
    threshold_high: float = 0.70
    threshold_crisis: float = 0.85
    min_exposure: float = 0.40


class DrawdownScalingConfig(BaseModel):
    enabled: bool = True
    tiers: list[list[float]] = [
        [0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0],
    ]
    min_total_scale: float = 0.05


class RiskConfig(BaseModel):
    btc_regime: BtcRegimeConfig = BtcRegimeConfig()
    volatility_targeting: VolatilityTargetingConfig = VolatilityTargetingConfig()
    correlation_monitor: CorrelationMonitorConfig = CorrelationMonitorConfig()
    drawdown_scaling: DrawdownScalingConfig = DrawdownScalingConfig()


class PortfolioConfig(BaseModel):
    initial_capital: float = 100000
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


class BacktestConfig(BaseModel):
    start_date: str
    end_date: str
    fee_rate: float
    slippage_prob: float


class BinanceConfig(BaseModel):
    api_key: str = ""
    api_secret: str = ""


class Settings(BaseModel):
    system: SystemConfig
    data: DataConfig
    universe: UniverseConfig
    alphas: dict[str, AlphaConfig]
    risk: RiskConfig
    portfolio: PortfolioConfig
    backtest: BacktestConfig
    binance: BinanceConfig
