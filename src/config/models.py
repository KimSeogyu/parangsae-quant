from __future__ import annotations

from pydantic import BaseModel, model_validator


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
    ranking_metric: str
    ranking_window: int
    inclusion_rank: int
    exclusion_rank: int

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


class RiskConfig(BaseModel):
    max_position_pct: float
    max_drawdown: float
    max_total_exposure: float
    volatility_window: int


class PortfolioConfig(BaseModel):
    initial_capital: float
    rebalance_interval_hours: int
    min_trade_threshold: float


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
