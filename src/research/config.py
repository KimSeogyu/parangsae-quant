"""Pydantic configuration models for the crypto-native residual ridge research pipeline.

Maps to config/universe.yaml + config/model.yaml + PRD Appendix A parameters.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Universe config (config/universe.yaml)
# ---------------------------------------------------------------------------

class InclusionConfig(BaseModel):
    asset_type: str = "native_crypto_perpetual"
    min_listing_days: int = 180
    min_funding_oi_coverage: float = 0.95
    max_spread_bps: float = 20.0


class ExclusionConfig(BaseModel):
    categories: list[str] = []
    symbols: list[str] = []


class LiquidityConfig(BaseModel):
    metric: str = "median_24h_quote_volume_30d"
    top_n: int = 30


class ReconstitutionConfig(BaseModel):
    frequency: str = "weekly"
    day_of_week: str = "monday"
    hour_utc: int = 0


class UniverseSettings(BaseModel):
    inclusion: InclusionConfig = InclusionConfig()
    exclusion: ExclusionConfig = ExclusionConfig()
    liquidity: LiquidityConfig = LiquidityConfig()
    reconstitution: ReconstitutionConfig = ReconstitutionConfig()
    sector_map: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Model config (config/model.yaml)
# ---------------------------------------------------------------------------

class DataBarConfig(BaseModel):
    research_bar_minutes: int = 5
    decision_bar_minutes: int = 20
    base_bar_minutes: int = 1


class LabelConfig(BaseModel):
    primary_horizon_minutes: int = 20
    secondary_horizon_minutes: int = 60
    beta_windows_weeks: list[int] = [1, 2]
    primary_weight: float = 0.7
    secondary_weight: float = 0.3

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> LabelConfig:
        total = self.primary_weight + self.secondary_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Label weights must sum to 1.0, got {total}")
        return self


class TrainingConfig(BaseModel):
    train_window_days: int = 90
    validation_window_days: int = 14
    embargo_minutes: int = 60
    retrain_cadence_days: int = 1


class RidgeConfig(BaseModel):
    alpha_grid: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    target_metric: str = "rank_ic"
    normalize: bool = True
    fit_intercept: bool = True


class PreprocessingConfig(BaseModel):
    zscore_method: str = "cross_sectional"
    clip: float = 5.0
    missing_fill: str = "cross_sectional_median"
    min_coverage: float = 0.80


class ResearchPortfolioConfig(BaseModel):
    direction: str = "long_short"
    gross: float = 1.0
    net_cap: float = 0.15
    entry_bucket_pct: float = 0.20
    exit_bucket_pct: float = 0.35
    long_count: int = 6
    short_count: int = 6
    rebalance_minutes: int = 20
    weight_formula: str = "clip(score_z, 2) / rvol_1d"
    single_name_cap: float = 0.12
    sector_soft_cap: float = 0.30
    btc_beta_cap: float = 0.10
    score_clip: float = 2.0

    @field_validator("direction")
    @classmethod
    def valid_direction(cls, v: str) -> str:
        if v not in ("long_short", "long_only"):
            raise ValueError(f"direction must be 'long_short' or 'long_only', got '{v}'")
        return v


class ExecutionConfig(BaseModel):
    mode: str = "passive_limit"
    requote_interval_seconds: int = 60
    max_requotes: int = 2
    requote_tick_step: int = 1
    unfilled_cancel_seconds: int = 180
    emergency_taker: bool = True


class PostfillConfig(BaseModel):
    check_horizon_minutes: int = 5
    reduce_threshold: float = 0.0
    reduce_pct: float = 0.50
    close_threshold_vol_mult: float = -0.25
    close_pct: float = 1.0
    spread_multiple: float = 2.0


class BacktestModeConfig(BaseModel):
    modes: list[str] = ["fast", "accurate"]
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    funding_realized: bool = True


class CostsConfig(BaseModel):
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0


class ModelSettings(BaseModel):
    data: DataBarConfig = DataBarConfig()
    labels: LabelConfig = LabelConfig()
    training: TrainingConfig = TrainingConfig()
    ridge: RidgeConfig = RidgeConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    portfolio: ResearchPortfolioConfig = ResearchPortfolioConfig()
    execution: ExecutionConfig = ExecutionConfig()
    postfill: PostfillConfig = PostfillConfig()
    backtest: BacktestModeConfig = BacktestModeConfig()
    costs: CostsConfig = CostsConfig()


# ---------------------------------------------------------------------------
# Combined settings
# ---------------------------------------------------------------------------

class ResearchSettings(BaseModel):
    universe: UniverseSettings
    model: ModelSettings


def load_research_settings(
    universe_path: str | Path = "config/universe.yaml",
    model_path: str | Path = "config/model.yaml",
) -> ResearchSettings:
    """Load research settings from YAML files."""
    with open(universe_path) as f:
        universe_raw = yaml.safe_load(f)
    with open(model_path) as f:
        model_raw = yaml.safe_load(f)
    return ResearchSettings(
        universe=UniverseSettings(**universe_raw),
        model=ModelSettings(**model_raw),
    )
