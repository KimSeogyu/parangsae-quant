# Quant Strategy Upgrade: Volatility-Adjusted QM Momentum with 4-Layer Risk

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing skeleton Grinold-Kahn system from a simple 24h momentum alpha with binary risk halt into a production-grade volatility-adjusted QM momentum strategy with FIP quality filtering, a low-volatility diversifier, 4-layer multiplicative risk scaling, tiered position caps, holdings hysteresis, and turnover-gated execution.

**Architecture:** The Nautilus Actor/Strategy signal bus pattern stays. Two Alpha Actors (QM Momentum, Low Volatility) publish `AlphaScore` signals. A redesigned Risk Actor publishes a `RiskState` containing a single multiplicative `risk_scale` float (0.05–1.5) from four layers. The Portfolio Construction Strategy z-scores and combines alphas, applies holdings hysteresis, tiered caps, risk scaling, and turnover limits before execution.

**Tech Stack:** Python 3.12+, nautilus_trader, numpy, pydantic v2, pyyaml

**Spec:** `quant_strategy.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `config/settings.yaml` | Full strategy parameter set |
| Modify | `src/config/models.py` | Pydantic models for new YAML structure |
| Modify | `src/types.py` | Extended RiskState with risk_scale, BTC regime, correlation |
| Modify | `src/alpha/base.py` | Increase default buffer to 1200 bars |
| Rewrite | `src/alpha/momentum.py` | QM Momentum: vol-adjusted + FIP quality + TS filter |
| Create | `src/alpha/low_volatility.py` | Low Volatility Alpha: rank-based inverse vol |
| Rewrite | `src/risk/model.py` | 4-layer multiplicative risk: regime + vol_target + corr + drawdown |
| Rewrite | `src/portfolio/construction.py` | Z-score combination, hysteresis, tiered caps, turnover limits |
| Modify | `src/engine/factory.py` | Wire new config → new components |
| Modify | `tests/test_alpha.py` | QM momentum + low-vol tests |
| Create | `tests/test_low_volatility.py` | Low volatility pure-function tests |
| Modify | `tests/test_risk.py` | 4-layer risk math tests |
| Modify | `tests/test_portfolio.py` | Z-score, hysteresis, turnover tests |
| Modify | `tests/test_config.py` | Validation tests for new config structure |
| Modify | `tests/test_integration.py` | E2E backtest with full strategy stack |
| Modify | `tests/conftest.py` | Larger synthetic data fixtures (1200+ bars) |

---

## Task 1: Config & Types Overhaul

**Files:**
- Modify: `config/settings.yaml`
- Modify: `src/config/models.py`
- Modify: `src/types.py`
- Modify: `tests/test_config.py`

This task rewrites configuration to match the full strategy parameter set from `quant_strategy.md` and extends signal types.

- [ ] **Step 1: Write failing test for new config structure**

```python
# tests/test_config.py — replace entire file
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_config.py -v`
Expected: FAIL — `Settings` model doesn't accept the new nested structure

- [ ] **Step 3: Rewrite src/config/models.py**

```python
# src/config/models.py — replace entire file
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
```

- [ ] **Step 4: Update src/types.py with extended RiskState**

```python
# src/types.py — replace entire file
"""Inter-component signal value types.

These frozen dataclasses define the contract between components.
Actors publish these as signal values; Strategy receives and pattern-matches them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlphaScore:
    """Published by Alpha Models. Consumed by Portfolio Construction."""

    name: str
    scores: dict[str, float]  # {instrument_id_str: alpha_value}


@dataclass(frozen=True)
class RiskState:
    """Published by Risk Model. Consumed by Portfolio Construction.

    risk_scale: multiplicative exposure scalar (0.05–1.5) from all four layers.
    """

    risk_scale: float  # combined multiplicative scale
    regime_scale: float  # BTC regime component
    vol_scale: float  # volatility targeting component
    corr_scale: float  # correlation monitor component
    dd_scale: float  # drawdown scaling component
    drawdown: float  # current drawdown fraction
    portfolio_vol: float  # realized portfolio vol (annualized)
    avg_correlation: float  # average pairwise correlation of top coins
    volatility: dict[str, float] = field(default_factory=dict)  # per-symbol vol


@dataclass(frozen=True)
class UniverseState:
    """Published by Universe Model. Consumed by Portfolio Construction."""

    current: frozenset[str]  # instrument_id strings in universe
    added: tuple[str, ...]
    removed: tuple[str, ...]
```

- [ ] **Step 5: Update config/settings.yaml**

```yaml
system:
  mode: backtest
  log_level: INFO

data:
  exchange: binance
  market_types:
    - spot
    - futures
  timeframe: 1h
  history_days: 365
  storage_path: data/

universe:
  size: 100
  ranking_metric: avg_quote_volume_14d
  ranking_window: 336
  inclusion_rank: 100
  exclusion_rank: 150
  exclude_stablecoins: true
  min_age_days: 30

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
    enabled: true
    ema_period_hours: 4800
    min_exposure: 0.3
    transition_lower: 0.90
    transition_upper: 1.00

  volatility_targeting:
    enabled: true
    target_vol_annual: 0.30
    lookback_hours: 720
    max_scale: 1.5
    min_scale: 0.1

  correlation_monitor:
    enabled: true
    window_hours: 720
    sample_coins: 20
    update_interval_hours: 4
    threshold_high: 0.70
    threshold_crisis: 0.85
    min_exposure: 0.40

  drawdown_scaling:
    enabled: true
    tiers:
      - [0.05, 1.00]
      - [0.10, 0.75]
      - [0.15, 0.50]
      - [0.20, 0.25]
      - [0.25, 0.00]
    min_total_scale: 0.05

portfolio:
  initial_capital: 100000
  num_holdings: 15
  entry_rank: 12
  exit_rank: 18
  rebalance_interval_hours: 1
  min_weight_change: 0.005
  max_hourly_turnover: 0.10
  max_daily_turnover: 0.50
  max_position_btc: 0.20
  max_position_eth: 0.15
  max_position_other: 0.07
  min_position: 0.01
  zscore_clip: 3.0

backtest:
  start_date: "2025-04-07"
  end_date: "2026-04-07"
  fee_rate: 0.001
  slippage_prob: 0.5

binance:
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}
```

- [ ] **Step 6: Run tests to verify config parses correctly**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_config.py -v`
Expected: All 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/config/models.py src/types.py config/settings.yaml tests/test_config.py
git commit -m "feat: config overhaul — nested risk/portfolio models for quant strategy"
```

---

## Task 2: QM Momentum Alpha with FIP Quality Filter

**Files:**
- Modify: `src/alpha/base.py` (buffer size increase)
- Rewrite: `src/alpha/momentum.py`
- Modify: `tests/test_alpha.py`

The QM momentum alpha computes: `α = [ln(P_skip / P_skip+lookback) / σ_realized] × quality(FIP)`. Coins with negative raw momentum are filtered out (TS filter). FIP quality transforms the fraction-of-positive-days metric into a [0.3, 1.0] multiplier that boosts "frog in the pan" continuous momentum.

- [ ] **Step 1: Write failing tests for QM momentum**

```python
# tests/test_alpha.py — replace entire file
import math
import numpy as np
import pytest

from src.alpha.momentum import compute_qm_momentum


def _make_trending_up(n: int, base: float = 100.0, hourly_drift: float = 0.001) -> list[float]:
    """Generate monotonically trending-up hourly closes."""
    prices = [base]
    for i in range(1, n):
        prices.append(prices[-1] * (1 + hourly_drift))
    return prices


def _make_trending_down(n: int, base: float = 100.0, hourly_drift: float = 0.001) -> list[float]:
    """Generate monotonically trending-down hourly closes."""
    prices = [base]
    for i in range(1, n):
        prices.append(prices[-1] * (1 - hourly_drift))
    return prices


class TestQMMomentum:
    def test_positive_trending_produces_positive_alpha(self):
        closes = _make_trending_up(1100)
        alpha = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        assert alpha > 0

    def test_negative_trending_filtered_by_ts(self):
        closes = _make_trending_down(1100)
        alpha = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        assert alpha == float("-inf")

    def test_ts_filter_disabled_allows_negative(self):
        closes = _make_trending_down(1100)
        alpha = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=False,
        )
        assert alpha != float("-inf")
        assert alpha < 0

    def test_insufficient_data_returns_neg_inf(self):
        closes = [100.0] * 50
        alpha = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        assert alpha == float("-inf")

    def test_fip_quality_multiplier_bounds(self):
        """Smooth uptrend (all positive days) should have quality near fip_floor,
        because FIP = sign(+) * (pct_neg - pct_pos) is very negative when all days are up."""
        closes = _make_trending_up(1100)
        alpha_low_floor = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.1, ts_filter=True,
        )
        alpha_high_floor = compute_qm_momentum(
            closes, lookback=336, skip=24, vol_window=720, fip_floor=0.5, ts_filter=True,
        )
        # Higher floor → higher minimum quality → higher alpha for smooth trends
        assert alpha_high_floor > alpha_low_floor

    def test_vol_adjustment_reduces_high_vol_signal(self):
        """Add noise to make vol higher; alpha should decrease vs smooth trend."""
        np.random.seed(42)
        smooth = _make_trending_up(1100, hourly_drift=0.002)
        noisy = [p * (1 + np.random.normal(0, 0.03)) for p in smooth]
        # Force same direction
        noisy[-25] = smooth[-25]  # anchor skip boundary

        alpha_smooth = compute_qm_momentum(
            smooth, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        alpha_noisy = compute_qm_momentum(
            noisy, lookback=336, skip=24, vol_window=720, fip_floor=0.3, ts_filter=True,
        )
        # Noisy version may be filtered or have lower alpha
        if alpha_noisy != float("-inf"):
            assert alpha_smooth > alpha_noisy
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_alpha.py -v`
Expected: FAIL — `compute_qm_momentum` does not exist

- [ ] **Step 3: Increase buffer in base.py**

In `src/alpha/base.py`, change line 22:

```python
# Change
    self._max_buffer: int = 500
# To
    self._max_buffer: int = 1200
```

- [ ] **Step 4: Implement QM momentum in src/alpha/momentum.py**

```python
# src/alpha/momentum.py — replace entire file
from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from src.alpha.base import BaseAlphaConfig, BaseAlphaModel


class QMMomentumAlphaConfig(BaseAlphaConfig):
    alpha_name: str = "crypto_qm_momentum"
    params: dict = Field(default_factory=lambda: {
        "lookback_hours": 336,
        "skip_hours": 24,
        "vol_window_hours": 720,
        "fip_enabled": True,
        "fip_floor": 0.3,
        "ts_filter": True,
    })


def compute_qm_momentum(
    closes: list[float],
    lookback: int = 336,
    skip: int = 24,
    vol_window: int = 720,
    fip_floor: float = 0.3,
    ts_filter: bool = True,
) -> float:
    """QM Momentum Alpha with FIP Quality Filter.

    Returns vol-adjusted momentum * quality multiplier, or -inf if filtered out.
    """
    required = skip + max(lookback, vol_window) + 2
    if len(closes) < required:
        return float("-inf")

    # Raw momentum (log return, skipping recent skip hours)
    p_end = closes[-(skip + 1)]
    p_start = closes[-(skip + lookback + 1)]
    if p_start <= 0 or p_end <= 0:
        return float("-inf")
    raw_mom = math.log(p_end / p_start)

    # Time-series filter: reject negative momentum
    if ts_filter and raw_mom <= 0.0:
        return float("-inf")

    # Realized volatility (vol_window ending at skip boundary)
    vol_slice = closes[-(skip + vol_window + 1) : -(skip)]
    arr = np.array(vol_slice, dtype=np.float64)
    log_rets = np.diff(np.log(arr))
    realized_vol = float(np.std(log_rets, ddof=1))
    if realized_vol < 1e-8:
        return float("-inf")

    vol_adj_mom = raw_mom / realized_vol

    # FIP on daily bars (aggregate hourly to daily by sampling every 24th bar)
    lookback_slice = closes[-(skip + lookback + 1) : -(skip)]
    daily_closes = lookback_slice[::24]
    if len(daily_closes) < 2:
        return vol_adj_mom  # not enough daily data, skip FIP

    daily_arr = np.array(daily_closes, dtype=np.float64)
    daily_rets = np.diff(daily_arr) / daily_arr[:-1]

    pct_pos = float(np.sum(daily_rets > 0)) / len(daily_rets)
    pct_neg = float(np.sum(daily_rets < 0)) / len(daily_rets)
    sign = 1.0 if raw_mom > 0 else -1.0
    fip = sign * (pct_neg - pct_pos)  # range [-1, +1]

    # Transform FIP to quality multiplier [fip_floor, 1.0]
    quality = fip_floor + (1 - fip_floor) * (1 - fip) / 2

    return vol_adj_mom * quality


class QMMomentumAlpha(BaseAlphaModel):
    def __init__(self, config: QMMomentumAlphaConfig) -> None:
        super().__init__(config)
        p = config.params
        self._lookback = p.get("lookback_hours", 336)
        self._skip = p.get("skip_hours", 24)
        self._vol_window = p.get("vol_window_hours", 720)
        self._fip_floor = p.get("fip_floor", 0.3)
        self._ts_filter = p.get("ts_filter", True)
        self._max_buffer = self._skip + max(self._lookback, self._vol_window) + 10

    def compute_alpha(self, symbol: str, closes: list[float]) -> float:
        return compute_qm_momentum(
            closes,
            lookback=self._lookback,
            skip=self._skip,
            vol_window=self._vol_window,
            fip_floor=self._fip_floor,
            ts_filter=self._ts_filter,
        )
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_alpha.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/alpha/base.py src/alpha/momentum.py tests/test_alpha.py
git commit -m "feat: QM momentum alpha with FIP quality filter and vol adjustment"
```

---

## Task 3: Low Volatility Alpha

**Files:**
- Create: `src/alpha/low_volatility.py`
- Create: `tests/test_low_volatility.py`

The low-vol alpha returns `-realized_vol` so that cross-sectional ranking favors low-volatility coins. This diversifies against momentum's tendency to favor high-vol assets.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_low_volatility.py — new file
import numpy as np
import pytest

from src.alpha.low_volatility import compute_low_volatility


def test_low_vol_returns_negative_volatility():
    np.random.seed(42)
    prices = [100.0]
    for _ in range(400):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    alpha = compute_low_volatility(prices, vol_window=336)
    assert alpha < 0


def test_low_vol_lower_for_higher_volatility():
    np.random.seed(42)
    calm = [100.0]
    for _ in range(400):
        calm.append(calm[-1] * (1 + np.random.normal(0, 0.005)))

    wild = [100.0]
    for _ in range(400):
        wild.append(wild[-1] * (1 + np.random.normal(0, 0.05)))

    alpha_calm = compute_low_volatility(calm, vol_window=336)
    alpha_wild = compute_low_volatility(wild, vol_window=336)
    assert alpha_calm > alpha_wild  # less negative = higher rank


def test_low_vol_insufficient_data():
    alpha = compute_low_volatility([100.0, 101.0], vol_window=336)
    assert alpha == float("-inf")


def test_low_vol_flat_prices():
    prices = [100.0] * 400
    alpha = compute_low_volatility(prices, vol_window=336)
    assert alpha == float("-inf")  # near-zero vol is filtered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_low_volatility.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement low volatility alpha**

```python
# src/alpha/low_volatility.py — new file
from __future__ import annotations

import numpy as np
from pydantic import Field

from src.alpha.base import BaseAlphaConfig, BaseAlphaModel


class LowVolatilityAlphaConfig(BaseAlphaConfig):
    alpha_name: str = "low_volatility"
    params: dict = Field(default_factory=lambda: {"vol_window_hours": 336})


def compute_low_volatility(closes: list[float], vol_window: int = 336) -> float:
    """Low Volatility Factor Alpha. Returns -realized_vol for cross-sectional ranking."""
    if len(closes) < vol_window + 1:
        return float("-inf")

    arr = np.array(closes[-(vol_window + 1) :], dtype=np.float64)
    log_rets = np.diff(np.log(arr))
    realized_vol = float(np.std(log_rets, ddof=1))

    if realized_vol < 1e-8:
        return float("-inf")

    return -realized_vol


class LowVolatilityAlpha(BaseAlphaModel):
    def __init__(self, config: LowVolatilityAlphaConfig) -> None:
        super().__init__(config)
        self._vol_window = config.params.get("vol_window_hours", 336)
        self._max_buffer = self._vol_window + 10

    def compute_alpha(self, symbol: str, closes: list[float]) -> float:
        return compute_low_volatility(closes, vol_window=self._vol_window)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_low_volatility.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha/low_volatility.py tests/test_low_volatility.py
git commit -m "feat: low volatility alpha — rank-based inverse vol diversifier"
```

---

## Task 4: Four-Layer Risk Model

**Files:**
- Rewrite: `src/risk/model.py`
- Modify: `tests/test_risk.py`

The risk model computes four independent scales (0–1 each) and multiplies them: `risk_scale = max(regime × vol_target × corr × drawdown, 0.05)`. Each layer has its own pure function.

- [ ] **Step 1: Write failing tests for all four risk layers**

```python
# tests/test_risk.py — replace entire file
import numpy as np
import pytest

from src.risk.model import (
    compute_btc_regime_scale,
    compute_vol_target_scale,
    compute_correlation_scale,
    compute_drawdown_scale,
    compute_combined_risk_scale,
    compute_drawdown,
    compute_ema,
)


class TestBtcRegime:
    def test_above_ema_full_exposure(self):
        scale = compute_btc_regime_scale(btc_price=100, btc_ema=100, lower=0.90, upper=1.00, min_exposure=0.3)
        assert scale == 1.0

    def test_at_lower_bound(self):
        scale = compute_btc_regime_scale(btc_price=90, btc_ema=100, lower=0.90, upper=1.00, min_exposure=0.3)
        assert scale == pytest.approx(0.3)

    def test_below_lower_bound(self):
        scale = compute_btc_regime_scale(btc_price=80, btc_ema=100, lower=0.90, upper=1.00, min_exposure=0.3)
        assert scale == 0.3

    def test_midpoint_interpolation(self):
        scale = compute_btc_regime_scale(btc_price=95, btc_ema=100, lower=0.90, upper=1.00, min_exposure=0.3)
        assert 0.3 < scale < 1.0
        assert scale == pytest.approx(0.65)


class TestVolTargeting:
    def test_at_target_no_scaling(self):
        scale = compute_vol_target_scale(portfolio_vol=0.30, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == pytest.approx(1.0)

    def test_high_vol_scales_down(self):
        scale = compute_vol_target_scale(portfolio_vol=0.60, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == pytest.approx(0.5)

    def test_low_vol_scales_up(self):
        scale = compute_vol_target_scale(portfolio_vol=0.15, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == pytest.approx(1.5)  # capped at max

    def test_near_zero_vol_capped(self):
        scale = compute_vol_target_scale(portfolio_vol=0.001, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == 1.5

    def test_min_floor(self):
        scale = compute_vol_target_scale(portfolio_vol=5.0, target_vol=0.30, min_scale=0.1, max_scale=1.5)
        assert scale == 0.1


class TestCorrelation:
    def test_below_threshold_no_reduction(self):
        scale = compute_correlation_scale(avg_corr=0.50, threshold_high=0.70, threshold_crisis=0.85, min_exposure=0.40)
        assert scale == 1.0

    def test_at_crisis_minimum(self):
        scale = compute_correlation_scale(avg_corr=0.85, threshold_high=0.70, threshold_crisis=0.85, min_exposure=0.40)
        assert scale == pytest.approx(0.40)

    def test_above_crisis_floored(self):
        scale = compute_correlation_scale(avg_corr=0.95, threshold_high=0.70, threshold_crisis=0.85, min_exposure=0.40)
        assert scale == 0.40

    def test_midpoint(self):
        scale = compute_correlation_scale(avg_corr=0.775, threshold_high=0.70, threshold_crisis=0.85, min_exposure=0.40)
        assert 0.40 < scale < 1.0


class TestDrawdownScaling:
    def test_no_drawdown_full_exposure(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.02, tiers=tiers)
        assert scale == 1.0

    def test_at_first_tier(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.05, tiers=tiers)
        assert scale == 1.0

    def test_between_tiers(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.075, tiers=tiers)
        assert scale == pytest.approx(0.875)

    def test_full_halt(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.30, tiers=tiers)
        assert scale == 0.0

    def test_at_10pct(self):
        tiers = [[0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0]]
        scale = compute_drawdown_scale(drawdown=0.10, tiers=tiers)
        assert scale == pytest.approx(0.75)


class TestCombinedRisk:
    def test_all_layers_full(self):
        scale = compute_combined_risk_scale(
            regime=1.0, vol_scale=1.0, corr_scale=1.0, dd_scale=1.0, min_total=0.05,
        )
        assert scale == 1.0

    def test_floor_applied(self):
        scale = compute_combined_risk_scale(
            regime=0.3, vol_scale=0.1, corr_scale=0.4, dd_scale=0.0, min_total=0.05,
        )
        assert scale == 0.05

    def test_multiplicative(self):
        scale = compute_combined_risk_scale(
            regime=0.5, vol_scale=0.8, corr_scale=1.0, dd_scale=0.75, min_total=0.05,
        )
        assert scale == pytest.approx(0.3)


class TestEma:
    def test_ema_converges_to_constant(self):
        prices = [100.0] * 100
        ema = compute_ema(prices, period=20)
        assert ema == pytest.approx(100.0, abs=0.01)

    def test_ema_below_sma_in_downtrend(self):
        prices = list(range(200, 100, -1))  # 200 down to 101
        ema = compute_ema(prices, period=50)
        sma = sum(prices[-50:]) / 50
        assert ema < sma  # EMA weights recent lower prices more


class TestDrawdown:
    def test_no_loss(self):
        assert compute_drawdown([100, 110, 120]) == 0.0

    def test_simple_drawdown(self):
        dd = compute_drawdown([100, 110, 105, 95, 100])
        assert dd == pytest.approx(0.1364, abs=0.001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_risk.py -v`
Expected: FAIL — new functions don't exist

- [ ] **Step 3: Implement 4-layer risk model**

```python
# src/risk/model.py — replace entire file
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType

from src.types import RiskState


class RiskModelConfig(ActorConfig):
    # BTC regime
    btc_regime_enabled: bool = True
    ema_period_hours: int = 4800
    regime_min_exposure: float = 0.3
    regime_transition_lower: float = 0.90
    regime_transition_upper: float = 1.00
    btc_instrument_id: str = "BTCUSDT.BINANCE"

    # Volatility targeting
    vol_targeting_enabled: bool = True
    target_vol_annual: float = 0.30
    vol_lookback_hours: int = 720
    vol_max_scale: float = 1.5
    vol_min_scale: float = 0.1

    # Correlation monitoring
    corr_enabled: bool = True
    corr_window_hours: int = 720
    corr_sample_coins: int = 20
    corr_update_interval_hours: int = 4
    corr_threshold_high: float = 0.70
    corr_threshold_crisis: float = 0.85
    corr_min_exposure: float = 0.40

    # Drawdown scaling
    dd_enabled: bool = True
    dd_tiers: list[list[float]] = [
        [0.05, 1.0], [0.10, 0.75], [0.15, 0.50], [0.20, 0.25], [0.25, 0.0],
    ]
    min_total_scale: float = 0.05

    # Legacy per-symbol volatility
    volatility_window: int = 168


def compute_ema(prices: list[float], period: int) -> float:
    """Exponential moving average of the last `period` prices."""
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1)
    ema = float(prices[-period])
    for p in prices[-period + 1 :]:
        ema = p * k + ema * (1 - k)
    return ema


def compute_btc_regime_scale(
    btc_price: float,
    btc_ema: float,
    lower: float = 0.90,
    upper: float = 1.00,
    min_exposure: float = 0.3,
) -> float:
    """Linear interpolation: BTC/EMA ratio → exposure scale."""
    if btc_ema <= 0:
        return 1.0
    ratio = btc_price / btc_ema
    if ratio >= upper:
        return 1.0
    if ratio <= lower:
        return min_exposure
    # Linear interpolation between [min_exposure, 1.0]
    t = (ratio - lower) / (upper - lower)
    return min_exposure + t * (1.0 - min_exposure)


def compute_vol_target_scale(
    portfolio_vol: float,
    target_vol: float = 0.30,
    min_scale: float = 0.1,
    max_scale: float = 1.5,
) -> float:
    """target_vol / realized_vol, clamped to [min_scale, max_scale]."""
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
    """Scale down when average pairwise correlation exceeds threshold."""
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

    # Below first tier → full exposure
    if dd <= tiers[0][0]:
        return tiers[0][1]

    # Interpolate between adjacent tiers
    for i in range(1, len(tiers)):
        dd_level, scale = tiers[i]
        prev_dd, prev_scale = tiers[i - 1]
        if dd <= dd_level:
            t = (dd - prev_dd) / (dd_level - prev_dd)
            return prev_scale + t * (scale - prev_scale)

    # Beyond last tier
    return tiers[-1][1]


def compute_combined_risk_scale(
    regime: float,
    vol_scale: float,
    corr_scale: float,
    dd_scale: float,
    min_total: float = 0.05,
) -> float:
    """Multiplicative combination of all risk layers with floor."""
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
    """Annualized rolling volatility from hourly returns."""
    if len(returns) < 2:
        return 0.0
    tail = returns[-window:] if len(returns) >= window else returns
    n = len(tail)
    if n < 2:
        return 0.0
    mean = sum(tail) / n
    variance = sum((r - mean) ** 2 for r in tail) / (n - 1)
    return math.sqrt(variance)


def _compute_avg_pairwise_correlation(
    return_buffers: dict[str, list[float]],
    sample_coins: int,
    window: int,
) -> float:
    """Average pairwise correlation of top N coins by buffer length."""
    # Pick coins with the most data
    eligible = sorted(return_buffers.keys(), key=lambda s: len(return_buffers[s]), reverse=True)
    coins = eligible[:sample_coins]
    if len(coins) < 2:
        return 0.0

    min_len = min(len(return_buffers[c]) for c in coins)
    usable = min(min_len, window)
    if usable < 10:
        return 0.0

    matrix = np.array([return_buffers[c][-usable:] for c in coins], dtype=np.float64)
    corr = np.corrcoef(matrix)
    n = len(coins)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(corr[i, j]):
                total += corr[i, j]
                count += 1
    return total / count if count > 0 else 0.0


class RiskModel(Actor):
    def __init__(self, config: RiskModelConfig) -> None:
        super().__init__(config)
        self._return_buffers: dict[str, list[float]] = defaultdict(list)
        self._prev_close: dict[str, float] = {}
        self._btc_closes: list[float] = []
        self._portfolio_returns: list[float] = []
        self._equity_curve: list[float] = []
        self._last_hour: int = -1
        self._last_corr_hour: int = -1
        self._cached_corr: float = 0.0

    def on_start(self) -> None:
        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        close = float(bar.close)

        # Track BTC closes for regime filter
        if symbol == self.config.btc_instrument_id:
            self._btc_closes.append(close)
            max_btc = self.config.ema_period_hours + 10
            if len(self._btc_closes) > max_btc:
                self._btc_closes.pop(0)

        # Track per-symbol returns
        if symbol in self._prev_close:
            prev = self._prev_close[symbol]
            if prev > 0:
                ret = (close - prev) / prev
                buf = self._return_buffers[symbol]
                buf.append(ret)
                max_buf = max(self.config.volatility_window, self.config.corr_window_hours) + 10
                if len(buf) > max_buf:
                    buf.pop(0)
        self._prev_close[symbol] = close

        current_hour = bar.ts_event // 3_600_000_000_000
        if current_hour > self._last_hour:
            self._last_hour = current_hour
            self._update_equity(bar.ts_event)
            self._publish_risk_state(bar.ts_event, current_hour)

    def _update_equity(self, ts_event: int) -> None:
        account = None
        try:
            from nautilus_trader.model.currencies import USDT
            from nautilus_trader.model.identifiers import Venue
            account = self.portfolio.account(Venue("BINANCE"))
        except Exception:
            pass

        if account is not None:
            from nautilus_trader.model.currencies import USDT
            balance_money = account.balance_total(USDT)
            if balance_money is not None:
                equity = float(balance_money.as_double())
                if self._equity_curve and self._equity_curve[-1] > 0:
                    ret = (equity - self._equity_curve[-1]) / self._equity_curve[-1]
                    self._portfolio_returns.append(ret)
                    max_pr = self.config.vol_lookback_hours + 10
                    if len(self._portfolio_returns) > max_pr:
                        self._portfolio_returns.pop(0)
                self._equity_curve.append(equity)

    def _publish_risk_state(self, ts_event: int, current_hour: int) -> None:
        # 1. BTC regime scale
        regime_scale = 1.0
        if self.config.btc_regime_enabled and len(self._btc_closes) >= 2:
            btc_price = self._btc_closes[-1]
            btc_ema = compute_ema(self._btc_closes, self.config.ema_period_hours)
            regime_scale = compute_btc_regime_scale(
                btc_price, btc_ema,
                lower=self.config.regime_transition_lower,
                upper=self.config.regime_transition_upper,
                min_exposure=self.config.regime_min_exposure,
            )

        # 2. Volatility targeting scale
        vol_scale = 1.0
        portfolio_vol = 0.0
        if self.config.vol_targeting_enabled and len(self._portfolio_returns) >= 10:
            hourly_vol = compute_rolling_volatility(
                self._portfolio_returns, self.config.vol_lookback_hours,
            )
            portfolio_vol = hourly_vol * math.sqrt(8760)  # annualize
            vol_scale = compute_vol_target_scale(
                portfolio_vol,
                target_vol=self.config.target_vol_annual,
                min_scale=self.config.vol_min_scale,
                max_scale=self.config.vol_max_scale,
            )

        # 3. Correlation monitoring (updated every N hours)
        corr_scale = 1.0
        if self.config.corr_enabled:
            hours_since = current_hour - self._last_corr_hour
            if hours_since >= self.config.corr_update_interval_hours or self._last_corr_hour < 0:
                self._cached_corr = _compute_avg_pairwise_correlation(
                    self._return_buffers,
                    self.config.corr_sample_coins,
                    self.config.corr_window_hours,
                )
                self._last_corr_hour = current_hour
            corr_scale = compute_correlation_scale(
                self._cached_corr,
                threshold_high=self.config.corr_threshold_high,
                threshold_crisis=self.config.corr_threshold_crisis,
                min_exposure=self.config.corr_min_exposure,
            )

        # 4. Drawdown scaling
        dd_scale = 1.0
        drawdown = compute_drawdown(self._equity_curve) if self._equity_curve else 0.0
        if self.config.dd_enabled:
            dd_scale = compute_drawdown_scale(drawdown, self.config.dd_tiers)

        # Combined
        risk_scale = compute_combined_risk_scale(
            regime_scale, vol_scale, corr_scale, dd_scale,
            min_total=self.config.min_total_scale,
        )

        # Per-symbol volatility (for portfolio sizing)
        volatility = {}
        for symbol, returns in self._return_buffers.items():
            volatility[symbol] = compute_rolling_volatility(
                returns, self.config.volatility_window,
            )

        state = RiskState(
            risk_scale=risk_scale,
            regime_scale=regime_scale,
            vol_scale=vol_scale,
            corr_scale=corr_scale,
            dd_scale=dd_scale,
            drawdown=drawdown,
            portfolio_vol=portfolio_vol,
            avg_correlation=self._cached_corr,
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
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_risk.py -v`
Expected: All 20 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/risk/model.py tests/test_risk.py
git commit -m "feat: 4-layer risk model — BTC regime, vol targeting, correlation, drawdown"
```

---

## Task 5: Portfolio Construction Upgrade

**Files:**
- Rewrite: `src/portfolio/construction.py`
- Modify: `tests/test_portfolio.py`

Major upgrade: z-score alpha combination, holdings hysteresis (entry rank 12, exit rank 18), tiered position caps (BTC 20%, ETH 15%, other 7%), risk-scale application, threshold-gated execution, and hourly/daily turnover caps.

- [ ] **Step 1: Write failing tests for new portfolio logic**

```python
# tests/test_portfolio.py — replace entire file
import pytest
from src.portfolio.construction import (
    zscore_and_combine,
    select_holdings,
    apply_tiered_caps,
    compute_target_weights,
    compute_target_deltas,
    apply_turnover_limit,
)


class TestZscoreCombine:
    def test_combines_two_alphas(self):
        alpha_scores = {
            "momentum": {"BTC": 0.5, "ETH": 0.3, "SOL": -0.1},
            "low_vol": {"BTC": -0.02, "ETH": -0.01, "SOL": -0.05},
        }
        weights = {"momentum": 0.75, "low_vol": 0.25}
        universe = {"BTC", "ETH", "SOL"}

        combined = zscore_and_combine(alpha_scores, weights, universe, clip=3.0)

        assert set(combined.keys()) == {"BTC", "ETH", "SOL"}
        # BTC should have highest combined score (strong momentum)
        assert combined["BTC"] > combined["ETH"]

    def test_clips_extreme_scores(self):
        alpha_scores = {
            "momentum": {"A": 100.0, "B": 0.0, "C": -100.0},
        }
        weights = {"momentum": 1.0}
        combined = zscore_and_combine(alpha_scores, weights, {"A", "B", "C"}, clip=3.0)
        # After z-scoring and clipping at ±3, range should be bounded
        assert all(-3.5 <= v <= 3.5 for v in combined.values())

    def test_single_symbol_returns_zero(self):
        alpha_scores = {"momentum": {"BTC": 0.5}}
        weights = {"momentum": 1.0}
        combined = zscore_and_combine(alpha_scores, weights, {"BTC"}, clip=3.0)
        assert combined["BTC"] == pytest.approx(0.0)

    def test_ignores_non_universe(self):
        alpha_scores = {"momentum": {"BTC": 0.5, "XRP": 0.8}}
        weights = {"momentum": 1.0}
        combined = zscore_and_combine(alpha_scores, weights, {"BTC"}, clip=3.0)
        assert "XRP" not in combined


class TestSelectHoldings:
    def test_selects_top_n(self):
        scores = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        current = set()
        selected = select_holdings(scores, current, num_holdings=3, entry_rank=2, exit_rank=4)
        assert "A" in selected
        assert "B" in selected

    def test_hysteresis_keeps_existing(self):
        scores = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        current = {"D"}  # D is rank 4, between entry_rank and exit_rank
        selected = select_holdings(scores, current, num_holdings=3, entry_rank=2, exit_rank=5)
        assert "D" in selected  # kept due to hysteresis

    def test_hysteresis_removes_fallen(self):
        scores = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        current = {"E"}  # E is rank 5, beyond exit_rank
        selected = select_holdings(scores, current, num_holdings=3, entry_rank=2, exit_rank=4)
        assert "E" not in selected

    def test_negative_scores_excluded(self):
        scores = {"A": 0.9, "B": -0.5, "C": float("-inf")}
        selected = select_holdings(scores, set(), num_holdings=3, entry_rank=2, exit_rank=4)
        assert "B" not in selected
        assert "C" not in selected


class TestTieredCaps:
    def test_btc_cap_applied(self):
        weights = {"BTCUSDT.BINANCE": 0.30, "ETHUSDT.BINANCE": 0.10, "SOLUSDT.BINANCE": 0.05}
        capped = apply_tiered_caps(weights, max_btc=0.20, max_eth=0.15, max_other=0.07)
        assert capped["BTCUSDT.BINANCE"] == pytest.approx(0.20)
        assert capped["ETHUSDT.BINANCE"] == pytest.approx(0.10)
        assert capped["SOLUSDT.BINANCE"] == pytest.approx(0.05)

    def test_eth_cap_applied(self):
        weights = {"ETHUSDT.BINANCE": 0.25}
        capped = apply_tiered_caps(weights, max_btc=0.20, max_eth=0.15, max_other=0.07)
        assert capped["ETHUSDT.BINANCE"] == pytest.approx(0.15)

    def test_other_cap_applied(self):
        weights = {"SOLUSDT.BINANCE": 0.10}
        capped = apply_tiered_caps(weights, max_btc=0.20, max_eth=0.15, max_other=0.07)
        assert capped["SOLUSDT.BINANCE"] == pytest.approx(0.07)


class TestTargetWeights:
    def test_alpha_times_inv_vol(self):
        holdings = {"A", "B"}
        scores = {"A": 2.0, "B": 1.0}
        volatility = {"A": 0.02, "B": 0.04}
        weights = compute_target_weights(holdings, scores, volatility, risk_scale=1.0, min_position=0.01)
        # A has higher alpha and lower vol → larger weight
        assert weights["A"] > weights["B"]
        # Weights should sum to ~1.0 (before caps)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_risk_scale_reduces_weights(self):
        holdings = {"A"}
        scores = {"A": 1.0}
        volatility = {"A": 0.02}
        w_full = compute_target_weights(holdings, scores, volatility, risk_scale=1.0, min_position=0.01)
        w_half = compute_target_weights(holdings, scores, volatility, risk_scale=0.5, min_position=0.01)
        assert w_half["A"] == pytest.approx(w_full["A"] * 0.5, abs=0.01)

    def test_min_position_filter(self):
        holdings = {"A", "B"}
        scores = {"A": 1.0, "B": 0.001}
        volatility = {"A": 0.02, "B": 0.02}
        weights = compute_target_weights(holdings, scores, volatility, risk_scale=1.0, min_position=0.01)
        # B should be dropped if its weight falls below min_position
        if "B" in weights:
            assert weights["B"] >= 0.01


class TestTargetDeltas:
    def test_basic_delta(self):
        deltas = compute_target_deltas({"A": 0.05, "B": 0.03}, {"A": 0.03}, threshold=0.005)
        assert deltas["A"] == pytest.approx(0.02)
        assert deltas["B"] == pytest.approx(0.03)

    def test_below_threshold_ignored(self):
        deltas = compute_target_deltas({"A": 0.050}, {"A": 0.049}, threshold=0.005)
        assert "A" not in deltas


class TestTurnoverLimit:
    def test_caps_total_turnover(self):
        deltas = {"A": 0.08, "B": 0.06, "C": -0.04}
        limited = apply_turnover_limit(deltas, max_turnover=0.10)
        total = sum(abs(v) for v in limited.values())
        assert total <= 0.10 + 1e-9

    def test_preserves_direction(self):
        deltas = {"A": 0.05, "B": -0.03}
        limited = apply_turnover_limit(deltas, max_turnover=1.0)
        assert limited["A"] > 0
        assert limited["B"] < 0

    def test_no_limit_when_below(self):
        deltas = {"A": 0.02, "B": 0.01}
        limited = apply_turnover_limit(deltas, max_turnover=0.10)
        assert limited == deltas
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_portfolio.py -v`
Expected: FAIL — new functions don't exist

- [ ] **Step 3: Implement upgraded portfolio construction**

```python
# src/portfolio/construction.py — replace entire file
from __future__ import annotations

import json
import math
from decimal import Decimal

import numpy as np
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy

from src.types import AlphaScore, RiskState, UniverseState


class PortfolioConstructionConfig(StrategyConfig):
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
    alpha_weights: dict[str, float] = {}
    order_id_tag: str = "PC"


def zscore_and_combine(
    alpha_scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    universe: set[str],
    clip: float = 3.0,
) -> dict[str, float]:
    """Cross-sectionally z-score each alpha, clip, and combine with weights."""
    combined: dict[str, float] = {s: 0.0 for s in universe}

    for alpha_name, scores in alpha_scores.items():
        w = weights.get(alpha_name, 0.0)
        if w == 0.0:
            continue

        # Filter to universe and finite values
        vals = {s: v for s, v in scores.items() if s in universe and math.isfinite(v)}
        if len(vals) < 2:
            # Can't z-score with < 2 values
            for s, v in vals.items():
                combined[s] += 0.0
            continue

        arr = np.array(list(vals.values()), dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std < 1e-10:
            continue

        for s, v in vals.items():
            z = (v - mean) / std
            z = max(-clip, min(z, clip))
            combined[s] += w * z

    return combined


def select_holdings(
    scores: dict[str, float],
    current_holdings: set[str],
    num_holdings: int = 15,
    entry_rank: int = 12,
    exit_rank: int = 18,
) -> set[str]:
    """Select holdings with entry/exit hysteresis. Only considers positive scores."""
    # Rank by score descending, exclude negative/inf
    eligible = {s: v for s, v in scores.items() if math.isfinite(v) and v > 0}
    ranked = sorted(eligible, key=lambda s: eligible[s], reverse=True)

    rank_map = {s: i + 1 for i, s in enumerate(ranked)}

    selected = set()
    for s in ranked:
        rank = rank_map[s]
        if s in current_holdings:
            # Existing holding: keep if within exit threshold
            if rank <= exit_rank:
                selected.add(s)
        else:
            # New entry: must be in top entry_rank
            if rank <= entry_rank and len(selected) < num_holdings:
                selected.add(s)

    # Re-add existing holdings that survived exit filter (up to num_holdings)
    for s in current_holdings:
        if s in rank_map and rank_map[s] <= exit_rank and len(selected) < num_holdings:
            selected.add(s)

    return selected


def apply_tiered_caps(
    weights: dict[str, float],
    max_btc: float = 0.20,
    max_eth: float = 0.15,
    max_other: float = 0.07,
) -> dict[str, float]:
    """Apply per-asset caps: BTC, ETH, and other coins."""
    capped = {}
    for symbol, w in weights.items():
        upper = symbol.upper()
        if "BTC" in upper and "ETH" not in upper:
            capped[symbol] = min(w, max_btc)
        elif "ETH" in upper:
            capped[symbol] = min(w, max_eth)
        else:
            capped[symbol] = min(w, max_other)
    return capped


def compute_target_weights(
    holdings: set[str],
    scores: dict[str, float],
    volatility: dict[str, float],
    risk_scale: float = 1.0,
    min_position: float = 0.01,
) -> dict[str, float]:
    """Alpha × inverse-vol weighting, normalized, then scaled by risk."""
    raw = {}
    for s in holdings:
        alpha = scores.get(s, 0.0)
        if alpha <= 0 or not math.isfinite(alpha):
            continue
        vol = volatility.get(s, 1.0)
        inv_vol = 1.0 / vol if vol > 1e-8 else 1.0
        raw[s] = alpha * inv_vol

    total = sum(raw.values())
    if total <= 0:
        return {}

    normalized = {s: v / total for s, v in raw.items()}

    # Apply risk scale
    scaled = {s: v * risk_scale for s, v in normalized.items()}

    # Remove positions below minimum
    filtered = {s: v for s, v in scaled.items() if v >= min_position}

    return filtered


def compute_target_deltas(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    threshold: float,
) -> dict[str, float]:
    """Compute deltas between target and current, filtering by threshold."""
    all_symbols = set(target_weights) | set(current_weights)
    deltas = {}
    for symbol in all_symbols:
        target = target_weights.get(symbol, 0.0)
        current = current_weights.get(symbol, 0.0)
        delta = target - current
        if abs(delta) >= threshold:
            deltas[symbol] = delta
    return deltas


def apply_turnover_limit(
    deltas: dict[str, float],
    max_turnover: float,
) -> dict[str, float]:
    """Proportionally scale deltas if total turnover exceeds limit."""
    total = sum(abs(d) for d in deltas.values())
    if total <= max_turnover or total <= 0:
        return dict(deltas)
    scale = max_turnover / total
    return {s: d * scale for s, d in deltas.items()}


class PortfolioConstruction(Strategy):
    def __init__(self, config: PortfolioConstructionConfig) -> None:
        super().__init__(config)
        self._alpha_scores: dict[str, dict[str, float]] = {}
        self._risk_state: RiskState | None = None
        self._universe: frozenset[str] = frozenset()
        self._current_holdings: set[str] = set()
        self._last_rebalance_hour: int = -1
        self._last_prices: dict[str, float] = {}
        self._daily_turnover: float = 0.0
        self._last_day: int = -1

    def on_start(self) -> None:
        self.subscribe_signal("ALPHA")
        self.subscribe_signal("RISK")
        self.subscribe_signal("UNIVERSE")

        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_signal(self, signal) -> None:
        raw = signal.value
        if not isinstance(raw, str):
            return
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        signal_type = data.get("type")
        if signal_type == "AlphaScore":
            score = AlphaScore(name=data["name"], scores=data["scores"])
            self._alpha_scores.setdefault(score.name, {}).update(score.scores)
        elif signal_type == "RiskState":
            self._risk_state = RiskState(
                risk_scale=data["risk_scale"],
                regime_scale=data["regime_scale"],
                vol_scale=data["vol_scale"],
                corr_scale=data["corr_scale"],
                dd_scale=data["dd_scale"],
                drawdown=data["drawdown"],
                portfolio_vol=data["portfolio_vol"],
                avg_correlation=data["avg_correlation"],
                volatility=data["volatility"],
            )
        elif signal_type == "UniverseState":
            state = UniverseState(
                current=frozenset(data["current"]),
                added=tuple(data["added"]),
                removed=tuple(data["removed"]),
            )
            self._universe = state.current
            for symbol_str in state.removed:
                try:
                    iid = InstrumentId.from_str(symbol_str)
                    self.close_all_positions(iid)
                except Exception:
                    pass

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        self._last_prices[symbol] = float(bar.close)

        current_hour = bar.ts_event // 3_600_000_000_000
        if current_hour <= self._last_rebalance_hour:
            return
        if not self._universe:
            return
        self._last_rebalance_hour = current_hour

        # Reset daily turnover counter at day boundary
        current_day = current_hour // 24
        if current_day > self._last_day:
            self._daily_turnover = 0.0
            self._last_day = current_day

        self._rebalance()

    def _rebalance(self) -> None:
        if not self._alpha_scores or self._risk_state is None:
            return

        # 1. Z-score and combine alphas
        combined = zscore_and_combine(
            self._alpha_scores,
            self.config.alpha_weights,
            set(self._universe),
            clip=self.config.zscore_clip,
        )

        # 2. Select holdings with hysteresis
        self._current_holdings = select_holdings(
            combined,
            self._current_holdings,
            num_holdings=self.config.num_holdings,
            entry_rank=self.config.entry_rank,
            exit_rank=self.config.exit_rank,
        )

        # 3. Compute target weights (alpha × inv-vol, risk-scaled)
        targets = compute_target_weights(
            self._current_holdings,
            combined,
            self._risk_state.volatility,
            risk_scale=self._risk_state.risk_scale,
            min_position=self.config.min_position,
        )

        # 4. Apply tiered position caps
        targets = apply_tiered_caps(
            targets,
            max_btc=self.config.max_position_btc,
            max_eth=self.config.max_position_eth,
            max_other=self.config.max_position_other,
        )

        # 5. Compute deltas with threshold gating
        current_weights = self._get_current_weights()
        deltas = compute_target_deltas(
            targets, current_weights, self.config.min_weight_change,
        )

        # 6. Apply turnover limits
        remaining_daily = max(0, self.config.max_daily_turnover - self._daily_turnover)
        hourly_limit = min(self.config.max_hourly_turnover, remaining_daily)
        deltas = apply_turnover_limit(deltas, hourly_limit)

        # 7. Execute
        turnover = sum(abs(d) for d in deltas.values())
        self._daily_turnover += turnover
        for symbol_str, delta in deltas.items():
            self._execute_delta(symbol_str, delta)

    def _get_current_weights(self) -> dict[str, float]:
        from nautilus_trader.model.currencies import USDT

        venue = Venue("BINANCE")
        account = self.portfolio.account(venue)
        if account is None:
            return {}

        balance_money = account.balance_total(USDT)
        if balance_money is None:
            return {}
        total_equity = float(balance_money.as_double())
        if total_equity <= 0:
            return {}

        weights = {}
        for instrument in self.cache.instruments():
            symbol_str = str(instrument.id)
            net_pos = float(self.portfolio.net_position(instrument.id))
            if net_pos != 0.0:
                last_price = self._last_prices.get(symbol_str, 0.0)
                notional = abs(net_pos * last_price)
                sign = 1.0 if net_pos > 0 else -1.0
                weights[symbol_str] = sign * notional / total_equity
        return weights

    def _execute_delta(self, symbol_str: str, delta: float) -> None:
        try:
            instrument_id = InstrumentId.from_str(symbol_str)
        except Exception:
            return

        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            return

        from nautilus_trader.model.currencies import USDT

        venue = Venue("BINANCE")
        account = self.portfolio.account(venue)
        if account is None:
            return

        balance_money = account.balance_total(USDT)
        if balance_money is None:
            return
        total_equity = float(balance_money.as_double())
        notional = abs(delta) * total_equity

        last_price = self._last_prices.get(symbol_str, 0.0)
        if last_price == 0:
            return

        quantity = notional / last_price
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=instrument.make_qty(Decimal(str(quantity))),
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)

    def on_stop(self) -> None:
        for instrument in self.cache.instruments():
            self.cancel_all_orders(instrument.id)
            self.close_all_positions(instrument.id)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_portfolio.py -v`
Expected: All 17 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/portfolio/construction.py tests/test_portfolio.py
git commit -m "feat: portfolio construction — z-score combination, hysteresis, tiered caps, turnover limits"
```

---

## Task 6: Engine Factory Wiring

**Files:**
- Modify: `src/engine/factory.py`

The factory must translate the new nested `RiskConfig` into the flat `RiskModelConfig` parameters and wire the new config fields to `PortfolioConstructionConfig`.

- [ ] **Step 1: Update factory.py to wire new config**

```python
# src/engine/factory.py — replace entire file
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money

from src.config.models import Settings
from src.data.catalog import load_bars_from_parquet
from src.data.instruments import load_instruments
from src.portfolio.construction import PortfolioConstruction, PortfolioConstructionConfig
from src.risk.model import RiskModel, RiskModelConfig
from src.universe.model import UniverseModel, UniverseModelConfig


_ALLOWED_MODULE_PREFIXES = ("src.alpha.",)


def _import_class(module_path: str):
    """Dynamically import a class from an allowed module path like 'src.alpha.momentum.QMMomentumAlpha'.

    Only modules under the allowed prefixes (src.alpha.*) can be loaded
    to prevent arbitrary code execution from untrusted config values.
    """
    module_name, _, class_name = module_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid module path: {module_path!r}")
    if not any(module_name.startswith(prefix) for prefix in _ALLOWED_MODULE_PREFIXES):
        raise ValueError(
            f"Module {module_name!r} is not in the allowed prefixes: {_ALLOWED_MODULE_PREFIXES}"
        )
    module = importlib.import_module(module_name)  # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
    return getattr(module, class_name)


def _create_instrument_fallback(symbol: str, market_type: str):
    """Fallback: create instrument from TestInstrumentProvider for known symbols."""
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    provider_map = {
        ("BTCUSDT", "spot"): TestInstrumentProvider.btcusdt_binance,
        ("ETHUSDT", "spot"): TestInstrumentProvider.ethusdt_binance,
        ("ADAUSDT", "spot"): TestInstrumentProvider.adausdt_binance,
        ("BTCUSDT-PERP", "futures"): TestInstrumentProvider.btcusdt_perp_binance,
        ("ETHUSDT-PERP", "futures"): TestInstrumentProvider.ethusdt_perp_binance,
    }
    factory = provider_map.get((symbol, market_type))
    if factory:
        return factory()
    return None


def build_backtest_engine(settings: Settings) -> BacktestEngine:
    """Assemble a fully configured BacktestEngine from a Settings object."""
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level=settings.system.log_level),
        ),
    )

    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(settings.portfolio.initial_capital, USDT)],
    )

    # Load instruments and bar data from parquet files
    for market_type in settings.data.market_types:
        data_dir = Path(settings.data.storage_path) / market_type
        if not data_dir.exists():
            continue

        dynamic_instruments = load_instruments(settings.data.storage_path, market_type)

        for parquet_file in sorted(data_dir.glob("*.parquet")):
            if parquet_file.stem.startswith("_"):
                continue
            symbol = parquet_file.stem
            instrument = dynamic_instruments.get(symbol) or _create_instrument_fallback(
                symbol, market_type
            )
            if instrument is None:
                continue
            engine.add_instrument(instrument)
            bars = load_bars_from_parquet(path=str(parquet_file), instrument=instrument)
            engine.add_data(bars, sort=False)

    engine.sort_data()

    # Universe Model
    engine.add_actor(
        UniverseModel(
            UniverseModelConfig(
                ranking_window=settings.universe.ranking_window,
                inclusion_rank=settings.universe.inclusion_rank,
                exclusion_rank=settings.universe.exclusion_rank,
            ),
        ),
    )

    # Alpha Models (dynamically loaded from config)
    alpha_weights: dict[str, float] = {}
    for alpha_name, alpha_cfg in settings.alphas.items():
        if not alpha_cfg.enabled:
            continue

        AlphaClass = _import_class(alpha_cfg.module)

        config_module = sys.modules[AlphaClass.__module__]
        ConfigClass = None
        for attr_name in dir(config_module):
            attr = getattr(config_module, attr_name)
            if (
                isinstance(attr, type)
                and attr_name.endswith("Config")
                and attr_name != "BaseAlphaConfig"
            ):
                ConfigClass = attr
                break

        if ConfigClass is None:
            from src.alpha.base import BaseAlphaConfig

            ConfigClass = BaseAlphaConfig

        actor_config = ConfigClass(alpha_name=alpha_name, params=alpha_cfg.params)
        engine.add_actor(AlphaClass(actor_config))
        alpha_weights[alpha_name] = alpha_cfg.weight

    # Risk Model — map nested config to flat RiskModelConfig
    rc = settings.risk
    engine.add_actor(
        RiskModel(
            RiskModelConfig(
                # BTC regime
                btc_regime_enabled=rc.btc_regime.enabled,
                ema_period_hours=rc.btc_regime.ema_period_hours,
                regime_min_exposure=rc.btc_regime.min_exposure,
                regime_transition_lower=rc.btc_regime.transition_lower,
                regime_transition_upper=rc.btc_regime.transition_upper,
                # Volatility targeting
                vol_targeting_enabled=rc.volatility_targeting.enabled,
                target_vol_annual=rc.volatility_targeting.target_vol_annual,
                vol_lookback_hours=rc.volatility_targeting.lookback_hours,
                vol_max_scale=rc.volatility_targeting.max_scale,
                vol_min_scale=rc.volatility_targeting.min_scale,
                # Correlation
                corr_enabled=rc.correlation_monitor.enabled,
                corr_window_hours=rc.correlation_monitor.window_hours,
                corr_sample_coins=rc.correlation_monitor.sample_coins,
                corr_update_interval_hours=rc.correlation_monitor.update_interval_hours,
                corr_threshold_high=rc.correlation_monitor.threshold_high,
                corr_threshold_crisis=rc.correlation_monitor.threshold_crisis,
                corr_min_exposure=rc.correlation_monitor.min_exposure,
                # Drawdown
                dd_enabled=rc.drawdown_scaling.enabled,
                dd_tiers=rc.drawdown_scaling.tiers,
                min_total_scale=rc.drawdown_scaling.min_total_scale,
            ),
        ),
    )

    # Portfolio Construction Strategy
    pc = settings.portfolio
    engine.add_strategy(
        PortfolioConstruction(
            PortfolioConstructionConfig(
                num_holdings=pc.num_holdings,
                entry_rank=pc.entry_rank,
                exit_rank=pc.exit_rank,
                rebalance_interval_hours=pc.rebalance_interval_hours,
                min_weight_change=pc.min_weight_change,
                max_hourly_turnover=pc.max_hourly_turnover,
                max_daily_turnover=pc.max_daily_turnover,
                max_position_btc=pc.max_position_btc,
                max_position_eth=pc.max_position_eth,
                max_position_other=pc.max_position_other,
                min_position=pc.min_position,
                zscore_clip=pc.zscore_clip,
                alpha_weights=alpha_weights,
                order_id_tag="PC001",
            ),
        ),
    )

    return engine
```

- [ ] **Step 2: Run all tests to verify nothing breaks**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All unit tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/engine/factory.py
git commit -m "feat: engine factory — wire nested risk config and new portfolio params"
```

---

## Task 7: Test Fixtures & Integration Test

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_integration.py`

Update fixtures to provide enough bars (1200+) for the QM momentum lookback+skip+vol windows. The integration test verifies the entire strategy stack runs end-to-end without errors.

- [ ] **Step 1: Update conftest.py with larger fixtures**

```python
# tests/conftest.py — replace entire file
import numpy as np
import pandas as pd
import pytest
from nautilus_trader.test_kit.providers import TestInstrumentProvider


@pytest.fixture
def btcusdt_instrument():
    return TestInstrumentProvider.btcusdt_binance()


@pytest.fixture
def ethusdt_instrument():
    return TestInstrumentProvider.ethusdt_binance()


@pytest.fixture
def sample_ohlcv_df():
    """100 hourly bars — small fixture for unit tests."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [50000.0 + i * 10 for i in range(100)],
            "high": [50100.0 + i * 10 for i in range(100)],
            "low": [49900.0 + i * 10 for i in range(100)],
            "close": [50050.0 + i * 10 for i in range(100)],
            "volume": [1.5] * 100,
            "quote_volume": [(50050.0 + i * 10) * 1.5 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def large_ohlcv_df():
    """1200 hourly bars — enough for QM momentum (336 lookback + 24 skip + 720 vol)."""
    np.random.seed(42)
    n = 1200
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    base_price = 50000.0
    prices = [base_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0.0002, 0.015)))
    prices = np.array(prices)
    return pd.DataFrame(
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
```

- [ ] **Step 2: Update integration test**

```python
# tests/test_integration.py — replace entire file
import os
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
    """Generate synthetic hourly parquet data for one symbol."""
    np.random.seed(hash(symbol) % 2**31)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(drift, 0.015)))
    prices = np.array(prices)

    df = pd.DataFrame(
        {
            "timestamp": [int(d.timestamp() * 1000) for d in dates],
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.uniform(0.5, 3.0, n),
        }
    )
    spot_dir = data_dir / "spot"
    spot_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(spot_dir / f"{symbol}.parquet", index=False)


def test_full_strategy_backtest_runs(strategy_settings_yaml):
    """End-to-end: generate data, build engine, run backtest, verify no crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        n = 1200  # 50 days of hourly bars

        # Generate 5 synthetic coins
        _generate_parquet(data_dir, "BTCUSDT", n, 50000, 0.0003)
        _generate_parquet(data_dir, "ETHUSDT", n, 3000, 0.0002)
        _generate_parquet(data_dir, "ADAUSDT", n, 0.5, 0.0001)
        _generate_parquet(data_dir, "SOLUSDT", n, 100, 0.0004)
        _generate_parquet(data_dir, "DOTUSDT", n, 7, -0.0001)

        yaml_content = strategy_settings_yaml.format(data_dir=str(data_dir))
        config_path = Path(tmpdir) / "settings.yaml"
        config_path.write_text(yaml_content)

        from src.config.loader import load_settings
        from src.engine.factory import build_backtest_engine

        settings = load_settings(str(config_path))
        engine = build_backtest_engine(settings)
        engine.run()

        # Verify engine completed
        assert engine.iteration_count > 0
```

- [ ] **Step 3: Run integration test**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/test_integration.py -v -s`
Expected: PASS — backtest runs end-to-end without crash

- [ ] **Step 4: Run all tests**

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_integration.py
git commit -m "feat: integration test — full strategy stack e2e with synthetic data"
```

---

## Dependency Check

Before starting, ensure `numpy` is installed:

Run: `cd /Users/seogyukim/Development/parangsae-quant && uv run python -c "import numpy; print(numpy.__version__)"`

If not installed: `uv add numpy`

---

## Summary of Changes by Component

| Component | Before | After |
|-----------|--------|-------|
| **Momentum Alpha** | 24h lookback, raw return | 336h lookback, 24h skip, vol-adjusted, FIP quality filter, TS filter |
| **Low-Vol Alpha** | — (not present) | 336h window, negative realized vol for cross-sectional ranking |
| **Alpha Combination** | Weighted sum of raw scores | Z-score normalization (clip ±3), 0.75/0.25 weighted combination |
| **Risk Model** | Per-symbol vol + binary halt | 4-layer multiplicative: BTC regime EMA, vol targeting, correlation, drawdown tiers |
| **Risk Signal** | `risk_halt: bool` | `risk_scale: float` (0.05–1.5) with per-layer breakdown |
| **Holdings Selection** | All universe coins weighted | Top 15 with entry/exit hysteresis (12/18) |
| **Position Sizing** | Flat cap (5%) | Tiered caps (BTC 20%, ETH 15%, other 7%), alpha × inv-vol |
| **Execution** | Any delta executes | Threshold-gated (0.5%), hourly turnover cap (10%), daily cap (50%) |
| **Config** | Flat YAML, simple Pydantic | Nested YAML with sub-models for each risk layer and portfolio rule |
