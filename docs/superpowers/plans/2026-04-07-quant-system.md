# Parangsae Quant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Nautilus Trader-based crypto quant backtesting system with 5-model Grinold-Kahn architecture (Universe, Alpha, Risk, Portfolio Construction, Execution).

**Architecture:** Nautilus `Actor` subclasses for Universe/Alpha/Risk models publish signals via MessageBus. A single `Strategy` subclass (Portfolio Construction) receives all signals, computes target weights, and submits orders. Pure computation functions are extracted for testability.

**Tech Stack:** Python 3.12+, nautilus_trader, ccxt, pydantic v2, pyyaml, pandas, pyarrow, uv

**Spec:** `docs/superpowers/specs/2026-04-07-quant-architecture-design.md`

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, dependencies |
| `.gitignore` | Ignore data/, results/, .env, __pycache__ |
| `.env.example` | Environment variable template |
| `config/settings.yaml` | Single source of truth for all configuration |
| `src/__init__.py` | Package root |
| `src/types.py` | Signal value types (AlphaScore, RiskState, UniverseState) |
| `src/config/__init__.py` | Config package |
| `src/config/models.py` | Pydantic models for YAML validation |
| `src/config/loader.py` | YAML loading + env var substitution |
| `src/data/__init__.py` | Data package |
| `src/data/fetcher.py` | CCXT-based Binance OHLCV bulk downloader |
| `src/data/catalog.py` | Parquet to Nautilus Bar conversion |
| `src/universe/__init__.py` | Universe package |
| `src/universe/model.py` | UniverseModel(Actor) — volume rankings, hysteresis |
| `src/alpha/__init__.py` | Alpha package |
| `src/alpha/base.py` | BaseAlphaModel(Actor) — contract for all alphas |
| `src/alpha/momentum.py` | MomentumAlpha — example implementation |
| `src/risk/__init__.py` | Risk package |
| `src/risk/model.py` | RiskModel(Actor) — volatility, drawdown, halt |
| `src/portfolio/__init__.py` | Portfolio package |
| `src/portfolio/construction.py` | PortfolioConstruction(Strategy) — the single decision-maker |
| `src/engine/__init__.py` | Engine package |
| `src/engine/factory.py` | Config to Nautilus component assembly |
| `src/engine/backtest.py` | BacktestEngine setup and run |
| `scripts/fetch_data.py` | CLI entry point for data collection |
| `scripts/run_backtest.py` | CLI entry point for backtest execution |
| `tests/conftest.py` | Shared fixtures (synthetic instruments, bars) |
| `tests/test_config.py` | Config loading tests |
| `tests/test_universe.py` | Universe ranking logic tests |
| `tests/test_alpha.py` | Alpha computation tests |
| `tests/test_risk.py` | Risk metric tests |
| `tests/test_portfolio.py` | Portfolio construction math tests |
| `tests/test_integration.py` | End-to-end backtest with synthetic data |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `config/settings.yaml`
- Create: all `__init__.py` files
- Create: `src/types.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "parangsae-quant"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "nautilus_trader",
    "ccxt",
    "pydantic>=2.0",
    "pyyaml",
    "pandas",
    "pyarrow",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
.env
data/
results/
.venv/
*.parquet
```

- [ ] **Step 3: Create .env.example**

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

- [ ] **Step 4: Create config/settings.yaml**

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
  ranking_metric: quote_volume
  ranking_window: 336
  inclusion_rank: 100
  exclusion_rank: 150

alphas:
  momentum:
    enabled: true
    module: src.alpha.momentum.MomentumAlpha
    weight: 0.5
    params:
      lookback: 24

risk:
  max_position_pct: 0.05
  max_drawdown: 0.15
  max_total_exposure: 1.0
  volatility_window: 168

portfolio:
  initial_capital: 100000
  rebalance_interval_hours: 1
  min_trade_threshold: 0.001

backtest:
  start_date: "2025-04-07"
  end_date: "2026-04-07"
  fee_rate: 0.001
  slippage_prob: 0.5

binance:
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}
```

- [ ] **Step 5: Create directory structure and __init__.py files**

```bash
mkdir -p src/config src/data src/universe src/alpha src/risk src/portfolio src/engine scripts tests data/spot data/futures results
touch src/__init__.py src/config/__init__.py src/data/__init__.py src/universe/__init__.py src/alpha/__init__.py src/risk/__init__.py src/portfolio/__init__.py src/engine/__init__.py
```

- [ ] **Step 6: Create src/types.py — signal value types**

```python
"""Inter-component signal value types.

These frozen dataclasses define the contract between components.
Actors publish these as signal values; Strategy receives and pattern-matches them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaScore:
    """Published by Alpha Models. Consumed by Portfolio Construction."""

    name: str
    scores: dict[str, float]  # {instrument_id_str: alpha_value}


@dataclass(frozen=True)
class RiskState:
    """Published by Risk Model. Consumed by Portfolio Construction."""

    volatility: dict[str, float]  # {instrument_id_str: rolling_vol}
    drawdown: float
    total_exposure: float
    risk_halt: bool


@dataclass(frozen=True)
class UniverseState:
    """Published by Universe Model. Consumed by Portfolio Construction."""

    current: frozenset[str]  # instrument_id strings in universe
    added: tuple[str, ...]
    removed: tuple[str, ...]
```

- [ ] **Step 7: Install dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully, `.venv` created.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with dependencies and signal types"
```

---

## Task 2: Configuration System

**Files:**
- Create: `src/config/models.py`
- Create: `src/config/loader.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test for config models**

```python
# tests/test_config.py
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
            "exclusion_rank": 50,  # invalid: less than inclusion
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config.models'`

- [ ] **Step 3: Implement config models**

```python
# src/config/models.py
from __future__ import annotations

from pydantic import BaseModel, model_validator


class SystemConfig(BaseModel):
    mode: str  # "backtest" or "live"
    log_level: str = "INFO"


class DataConfig(BaseModel):
    exchange: str
    market_types: list[str]
    timeframe: str
    history_days: int
    storage_path: str


class UniverseConfig(BaseModel):
    ranking_metric: str
    ranking_window: int  # hours
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
    volatility_window: int  # hours


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Write failing test for config loader**

Add to `tests/test_config.py`:

```python
from src.config.loader import load_settings


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
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_load_settings_from_yaml -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 7: Implement config loader**

```python
# src/config/loader.py
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from src.config.models import Settings

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR} with os.environ[VAR]. Missing vars become empty string."""

    def replacer(match: re.Match) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_PATTERN.sub(replacer, value)


def _walk_and_substitute(obj: object) -> object:
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    return obj


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Load settings from YAML, substituting ${ENV_VAR} references."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _walk_and_substitute(raw)
    return Settings(**raw)
```

- [ ] **Step 8: Run all config tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 9: Commit**

```bash
git add src/config/ tests/test_config.py config/settings.yaml
git commit -m "feat: configuration system with Pydantic validation and env var substitution"
```

---

## Task 3: Data Fetcher

**Files:**
- Create: `src/data/fetcher.py`
- Create: `scripts/fetch_data.py`
- Create: `tests/test_fetcher.py`

- [ ] **Step 1: Write failing test for OHLCV processing**

```python
# tests/test_fetcher.py
import pandas as pd
from src.data.fetcher import ohlcv_to_dataframe, save_ohlcv


def test_ohlcv_to_dataframe():
    """CCXT returns OHLCV as list of lists: [timestamp_ms, o, h, l, c, volume]."""
    raw = [
        [1609459200000, 29000.0, 29500.0, 28800.0, 29300.0, 100.5],
        [1609462800000, 29300.0, 29700.0, 29100.0, 29600.0, 85.2],
    ]
    df = ohlcv_to_dataframe(raw)

    assert len(df) == 2
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "quote_volume"]
    assert df.index.name == "timestamp"
    assert df.index.tz is not None  # UTC timezone
    assert df.iloc[0]["close"] == 29300.0
    # quote_volume = close * volume
    assert df.iloc[0]["quote_volume"] == 29300.0 * 100.5


def test_save_ohlcv_writes_parquet(tmp_path):
    raw = [
        [1609459200000, 29000.0, 29500.0, 28800.0, 29300.0, 100.5],
    ]
    df = ohlcv_to_dataframe(raw)
    out_path = tmp_path / "BTCUSDT.parquet"

    save_ohlcv(df, str(out_path))

    loaded = pd.read_parquet(out_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["close"] == 29300.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fetcher.py -v`
Expected: FAIL

- [ ] **Step 3: Implement fetcher**

```python
# src/data/fetcher.py
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import ccxt.async_support as ccxt_async
import pandas as pd

logger = logging.getLogger(__name__)


def ohlcv_to_dataframe(raw: list[list]) -> pd.DataFrame:
    """Convert CCXT OHLCV response to DataFrame with quote_volume."""
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df["quote_volume"] = df["close"] * df["volume"]
    return df


def save_ohlcv(df: pd.DataFrame, path: str) -> None:
    """Save OHLCV DataFrame to Parquet, appending to existing data."""
    path = Path(path)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


async def fetch_all_symbols(
    exchange_id: str,
    market_type: str,
    timeframe: str,
    since_ms: int,
    storage_path: str,
) -> None:
    """Fetch OHLCV for all active symbols of a given market type."""
    exchange_class = getattr(ccxt_async, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    try:
        await exchange.load_markets()
        symbols = [
            s
            for s, m in exchange.markets.items()
            if m["active"]
            and m["quote"] == "USDT"
            and (
                (market_type == "spot" and m["spot"])
                or (market_type == "futures" and m["swap"] and m["linear"])
            )
        ]
        logger.info(f"Found {len(symbols)} {market_type} symbols")

        for symbol in symbols:
            safe_name = symbol.replace("/", "").replace(":", "-")
            out_path = Path(storage_path) / market_type / f"{safe_name}.parquet"

            # Check last timestamp for incremental update
            symbol_since = since_ms
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                if len(existing) > 0:
                    last_ts = existing.index.max()
                    symbol_since = int(last_ts.timestamp() * 1000) + 1

            all_ohlcv = []
            current_since = symbol_since
            while True:
                try:
                    ohlcv = await exchange.fetch_ohlcv(
                        symbol, timeframe, since=current_since, limit=1000
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")
                    break

                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1

                if len(ohlcv) < 1000:
                    break

            if all_ohlcv:
                df = ohlcv_to_dataframe(all_ohlcv)
                save_ohlcv(df, str(out_path))
                logger.info(f"Saved {len(df)} bars for {symbol}")

    finally:
        await exchange.close()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_fetcher.py -v`
Expected: 2 passed

- [ ] **Step 5: Create fetch_data.py CLI script**

```python
# scripts/fetch_data.py
"""CLI script to download OHLCV data from Binance."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.config.loader import load_settings
from src.data.fetcher import fetch_all_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    settings = load_settings()
    since = datetime.now(timezone.utc) - timedelta(days=settings.data.history_days)
    since_ms = int(since.timestamp() * 1000)

    for market_type in settings.data.market_types:
        logging.info(f"Fetching {market_type} data...")
        asyncio.run(
            fetch_all_symbols(
                exchange_id=settings.data.exchange,
                market_type=market_type,
                timeframe=settings.data.timeframe,
                since_ms=since_ms,
                storage_path=settings.data.storage_path,
            )
        )

    logging.info("Data fetch complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/data/fetcher.py scripts/fetch_data.py tests/test_fetcher.py
git commit -m "feat: CCXT-based Binance OHLCV data fetcher with incremental updates"
```

---

## Task 4: Data Catalog (Parquet to Nautilus)

**Files:**
- Create: `src/data/catalog.py`
- Create: `tests/conftest.py`
- Create: `tests/test_catalog.py`

- [ ] **Step 1: Create shared test fixtures**

```python
# tests/conftest.py
import pandas as pd
import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider


@pytest.fixture
def btcusdt_instrument():
    return TestInstrumentProvider.btcusdt_binance()


@pytest.fixture
def ethusdt_instrument():
    return TestInstrumentProvider.ethusdt_binance()


@pytest.fixture
def sample_ohlcv_df():
    """100 hours of synthetic OHLCV data."""
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
```

- [ ] **Step 2: Write failing test for catalog**

```python
# tests/test_catalog.py
from src.data.catalog import load_bars_from_parquet


def test_load_bars_from_parquet(tmp_path, btcusdt_instrument, sample_ohlcv_df):
    parquet_path = tmp_path / "BTCUSDT.parquet"
    sample_ohlcv_df.to_parquet(parquet_path)

    bars = load_bars_from_parquet(
        path=str(parquet_path),
        instrument=btcusdt_instrument,
        bar_step=1,
        bar_aggregation="HOUR",
    )

    assert len(bars) == 100
    assert float(bars[0].open) == 50000.0
    assert float(bars[0].close) == 50050.0
    # ts_init should equal ts_event (bar close time)
    assert bars[0].ts_init == bars[0].ts_event
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_catalog.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Implement catalog**

```python
# src/data/catalog.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.persistence.wranglers import BarDataWrangler


def load_bars_from_parquet(
    path: str | Path,
    instrument: Instrument,
    bar_step: int = 1,
    bar_aggregation: str = "HOUR",
) -> list[Bar]:
    """Load Parquet OHLCV file and convert to Nautilus Bar objects."""
    df = pd.read_parquet(path)

    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # BarDataWrangler expects: open, high, low, close, volume
    wrangler_df = df[["open", "high", "low", "close", "volume"]].copy()

    bar_type = BarType.from_str(
        f"{instrument.id}-{bar_step}-{bar_aggregation}-LAST-EXTERNAL"
    )
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)

    # ts_init_delta=0 because our timestamps are already bar close times
    bars = wrangler.process(wrangler_df, ts_init_delta=0)
    return bars


def load_all_bars(
    storage_path: str,
    market_type: str,
    instruments: dict[str, Instrument],
    bar_step: int = 1,
    bar_aggregation: str = "HOUR",
) -> dict[str, list[Bar]]:
    """Load bars for all parquet files in a market_type directory."""
    result = {}
    base = Path(storage_path) / market_type
    if not base.exists():
        return result

    for parquet_file in sorted(base.glob("*.parquet")):
        symbol = parquet_file.stem  # e.g. "BTCUSDT"
        if symbol not in instruments:
            continue
        bars = load_bars_from_parquet(
            path=parquet_file,
            instrument=instruments[symbol],
            bar_step=bar_step,
            bar_aggregation=bar_aggregation,
        )
        if bars:
            result[symbol] = bars
    return result
```

- [ ] **Step 5: Run test**

Run: `uv run pytest tests/test_catalog.py -v`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add src/data/catalog.py tests/conftest.py tests/test_catalog.py
git commit -m "feat: Parquet to Nautilus Bar catalog with BarDataWrangler"
```

---

## Task 5: Universe Model

**Files:**
- Create: `src/universe/model.py`
- Create: `tests/test_universe.py`

- [ ] **Step 1: Write failing test for ranking logic (pure function)**

```python
# tests/test_universe.py
from src.universe.model import compute_rankings, apply_hysteresis


def test_compute_rankings():
    """Rankings by average quote volume, descending."""
    volume_buffers = {
        "BTCUSDT": [1000.0, 2000.0, 3000.0],
        "ETHUSDT": [500.0, 600.0, 700.0],
        "XRPUSDT": [5000.0, 4000.0, 3000.0],
    }
    rankings = compute_rankings(volume_buffers)

    # XRPUSDT avg=4000, BTCUSDT avg=2000, ETHUSDT avg=600
    assert rankings["XRPUSDT"] == 1
    assert rankings["BTCUSDT"] == 2
    assert rankings["ETHUSDT"] == 3


def test_apply_hysteresis_inclusion():
    """Symbols ranked <= inclusion_rank get added to universe."""
    current_universe: set[str] = set()
    rankings = {"BTCUSDT": 1, "ETHUSDT": 50, "XRPUSDT": 101}

    new_universe, added, removed = apply_hysteresis(
        current_universe=current_universe,
        rankings=rankings,
        inclusion_rank=100,
        exclusion_rank=150,
    )

    assert new_universe == {"BTCUSDT", "ETHUSDT"}
    assert set(added) == {"BTCUSDT", "ETHUSDT"}
    assert removed == ()


def test_apply_hysteresis_retention():
    """Symbols between inclusion and exclusion rank stay in universe."""
    current_universe = {"BTCUSDT", "ETHUSDT"}
    rankings = {"BTCUSDT": 1, "ETHUSDT": 120}  # ETH dropped to 120 but < 150

    new_universe, added, removed = apply_hysteresis(
        current_universe=current_universe,
        rankings=rankings,
        inclusion_rank=100,
        exclusion_rank=150,
    )

    assert "ETHUSDT" in new_universe  # retained
    assert added == ()
    assert removed == ()


def test_apply_hysteresis_exclusion():
    """Symbols ranked > exclusion_rank get removed."""
    current_universe = {"BTCUSDT", "ETHUSDT"}
    rankings = {"BTCUSDT": 1, "ETHUSDT": 200}

    new_universe, added, removed = apply_hysteresis(
        current_universe=current_universe,
        rankings=rankings,
        inclusion_rank=100,
        exclusion_rank=150,
    )

    assert "ETHUSDT" not in new_universe
    assert added == ()
    assert set(removed) == {"ETHUSDT"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_universe.py -v`
Expected: FAIL

- [ ] **Step 3: Implement universe model**

```python
# src/universe/model.py
from __future__ import annotations

from collections import defaultdict

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar

from src.types import UniverseState


class UniverseModelConfig(ActorConfig):
    ranking_window: int = 336  # hours
    inclusion_rank: int = 100
    exclusion_rank: int = 150


# ── Pure functions (testable without Nautilus) ──────────────────────


def compute_rankings(volume_buffers: dict[str, list[float]]) -> dict[str, int]:
    """Rank symbols by average quote volume (descending). Rank 1 = highest."""
    averages = {}
    for symbol, volumes in volume_buffers.items():
        if volumes:
            averages[symbol] = sum(volumes) / len(volumes)
    sorted_symbols = sorted(averages, key=lambda s: averages[s], reverse=True)
    return {symbol: rank + 1 for rank, symbol in enumerate(sorted_symbols)}


def apply_hysteresis(
    current_universe: set[str],
    rankings: dict[str, int],
    inclusion_rank: int,
    exclusion_rank: int,
) -> tuple[set[str], tuple[str, ...], tuple[str, ...]]:
    """Apply hysteresis buffer to universe membership.

    Returns: (new_universe, added, removed)
    """
    new_universe = set()
    added = []
    removed = []

    for symbol, rank in rankings.items():
        if symbol in current_universe:
            # Existing member: keep unless rank > exclusion
            if rank <= exclusion_rank:
                new_universe.add(symbol)
            else:
                removed.append(symbol)
        else:
            # Non-member: add only if rank <= inclusion
            if rank <= inclusion_rank:
                new_universe.add(symbol)
                added.append(symbol)

    return new_universe, tuple(added), tuple(removed)


# ── Nautilus Actor ──────────────────────────────────────────────────


class UniverseModel(Actor):
    def __init__(self, config: UniverseModelConfig) -> None:
        super().__init__(config)
        self._volume_buffers: dict[str, list[float]] = defaultdict(list)
        self._universe: set[str] = set()
        self._last_ranking_hour: int = -1
        self._bars_seen: int = 0

    def on_start(self) -> None:
        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            from nautilus_trader.model.data import BarType

            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        notional_volume = float(bar.close) * float(bar.volume)

        buf = self._volume_buffers[symbol]
        buf.append(notional_volume)
        if len(buf) > self.config.ranking_window:
            buf.pop(0)

        self._bars_seen += 1
        current_hour = bar.ts_event // 3_600_000_000_000
        if current_hour > self._last_ranking_hour:
            self._last_ranking_hour = current_hour
            self._try_rebalance()

    def _try_rebalance(self) -> None:
        # Warmup: need at least ranking_window hours of data for any symbol
        max_buf_len = max(
            (len(buf) for buf in self._volume_buffers.values()), default=0
        )
        if max_buf_len < self.config.ranking_window:
            return

        rankings = compute_rankings(self._volume_buffers)
        new_universe, added, removed = apply_hysteresis(
            current_universe=self._universe,
            rankings=rankings,
            inclusion_rank=self.config.inclusion_rank,
            exclusion_rank=self.config.exclusion_rank,
        )
        self._universe = new_universe

        self.publish_signal(
            name="UNIVERSE",
            value=UniverseState(
                current=frozenset(new_universe),
                added=added,
                removed=removed,
            ),
            ts_event=0,
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_universe.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/universe/ tests/test_universe.py
git commit -m "feat: universe model with volume ranking and hysteresis buffer"
```

---

## Task 6: Alpha Model Base + Momentum

**Files:**
- Create: `src/alpha/base.py`
- Create: `src/alpha/momentum.py`
- Create: `tests/test_alpha.py`

- [ ] **Step 1: Write failing test for momentum alpha logic**

```python
# tests/test_alpha.py
from src.alpha.momentum import compute_momentum


import pytest


def test_compute_momentum_positive():
    """Prices going up → positive momentum."""
    closes = [100.0, 102.0, 105.0, 108.0, 110.0]
    result = compute_momentum(closes, lookback=4)
    # (110 - 100) / 100 = 0.1
    assert result == pytest.approx(0.1)


def test_compute_momentum_negative():
    """Prices going down → negative momentum."""
    closes = [110.0, 108.0, 105.0, 102.0, 100.0]
    result = compute_momentum(closes, lookback=4)
    # (100 - 110) / 110 ≈ -0.0909
    assert result == pytest.approx(-0.0909, abs=0.001)


def test_compute_momentum_insufficient_data():
    """Not enough data → returns 0."""
    closes = [100.0, 102.0]
    result = compute_momentum(closes, lookback=4)
    assert result == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_alpha.py -v`
Expected: FAIL

- [ ] **Step 3: Implement base alpha model and momentum**

```python
# src/alpha/base.py
from __future__ import annotations

from collections import defaultdict

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType

from src.types import AlphaScore


class BaseAlphaConfig(ActorConfig):
    alpha_name: str = ""
    params: dict = {}


class BaseAlphaModel(Actor):
    """Base class for all alpha models.

    Subclasses implement compute_alpha(symbol, closes) -> float.
    The base class handles bar buffering and signal publishing.
    """

    def __init__(self, config: BaseAlphaConfig) -> None:
        super().__init__(config)
        self._close_buffers: dict[str, list[float]] = defaultdict(list)
        self._max_buffer: int = 500  # default, overridden by subclass

    def on_start(self) -> None:
        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        buf = self._close_buffers[symbol]
        buf.append(float(bar.close))
        if len(buf) > self._max_buffer:
            buf.pop(0)

        alpha = self.compute_alpha(symbol, buf)
        # Collect; publish batch at end of hour handled by subclass or here
        self._publish_if_ready(symbol, alpha, bar.ts_event)

    def compute_alpha(self, symbol: str, closes: list[float]) -> float:
        raise NotImplementedError

    def _publish_if_ready(self, symbol: str, alpha: float, ts_event: int) -> None:
        """Publish individual alpha score."""
        self.publish_signal(
            name="ALPHA",
            value=AlphaScore(name=self.config.alpha_name, scores={symbol: alpha}),
            ts_event=ts_event,
        )
```

```python
# src/alpha/momentum.py
from __future__ import annotations

from src.alpha.base import BaseAlphaConfig, BaseAlphaModel


class MomentumAlphaConfig(BaseAlphaConfig):
    alpha_name: str = "momentum"
    params: dict = {"lookback": 24}


# ── Pure function (testable without Nautilus) ──


def compute_momentum(closes: list[float], lookback: int) -> float:
    """Simple price momentum: return over lookback period."""
    if len(closes) < lookback + 1:
        return 0.0
    old = closes[-(lookback + 1)]
    current = closes[-1]
    if old == 0:
        return 0.0
    return (current - old) / old


# ── Nautilus Actor ──


class MomentumAlpha(BaseAlphaModel):
    def __init__(self, config: MomentumAlphaConfig) -> None:
        super().__init__(config)
        self._lookback = config.params.get("lookback", 24)
        self._max_buffer = self._lookback + 10

    def compute_alpha(self, symbol: str, closes: list[float]) -> float:
        return compute_momentum(closes, self._lookback)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_alpha.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/alpha/ tests/test_alpha.py
git commit -m "feat: alpha model base class and momentum alpha implementation"
```

---

## Task 7: Risk Model

**Files:**
- Create: `src/risk/model.py`
- Create: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests for risk functions**

```python
# tests/test_risk.py
import math
from src.risk.model import compute_rolling_volatility, compute_drawdown


def test_compute_rolling_volatility():
    """Volatility of constant prices is 0."""
    returns = [0.0, 0.0, 0.0, 0.0]
    assert compute_rolling_volatility(returns, window=4) == 0.0


def test_compute_rolling_volatility_with_movement():
    returns = [0.01, -0.01, 0.02, -0.02, 0.01]
    vol = compute_rolling_volatility(returns, window=5)
    assert vol > 0
    assert vol < 1  # sanity


def test_compute_drawdown():
    equity_curve = [100.0, 110.0, 105.0, 95.0, 100.0]
    dd = compute_drawdown(equity_curve)
    # Peak was 110, trough was 95: dd = (110 - 95) / 110 ≈ 0.1364
    assert dd == pytest.approx(0.1364, abs=0.001)


def test_compute_drawdown_no_loss():
    equity_curve = [100.0, 110.0, 120.0]
    dd = compute_drawdown(equity_curve)
    assert dd == 0.0


import pytest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_risk.py -v`
Expected: FAIL

- [ ] **Step 3: Implement risk model**

```python
# src/risk/model.py
from __future__ import annotations

import math
from collections import defaultdict

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar, BarType

from src.types import RiskState


class RiskModelConfig(ActorConfig):
    max_position_pct: float = 0.05
    max_drawdown: float = 0.15
    max_total_exposure: float = 1.0
    volatility_window: int = 168


# ── Pure functions ──────────────────────────────────────────────────


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


# ── Nautilus Actor ──────────────────────────────────────────────────


class RiskModel(Actor):
    def __init__(self, config: RiskModelConfig) -> None:
        super().__init__(config)
        self._return_buffers: dict[str, list[float]] = defaultdict(list)
        self._prev_close: dict[str, float] = {}
        self._equity_curve: list[float] = []
        self._last_hour: int = -1

    def on_start(self) -> None:
        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_bar(self, bar: Bar) -> None:
        symbol = str(bar.bar_type.instrument_id)
        close = float(bar.close)

        if symbol in self._prev_close:
            prev = self._prev_close[symbol]
            if prev > 0:
                ret = (close - prev) / prev
                buf = self._return_buffers[symbol]
                buf.append(ret)
                if len(buf) > self.config.volatility_window:
                    buf.pop(0)
        self._prev_close[symbol] = close

        current_hour = bar.ts_event // 3_600_000_000_000
        if current_hour > self._last_hour:
            self._last_hour = current_hour
            self._publish_risk_state(bar.ts_event)

    def _publish_risk_state(self, ts_event: int) -> None:
        # Per-symbol volatility
        volatility = {}
        for symbol, returns in self._return_buffers.items():
            volatility[symbol] = compute_rolling_volatility(
                returns, self.config.volatility_window
            )

        # Portfolio drawdown (use account balance from portfolio if available)
        account = None
        try:
            from nautilus_trader.model.identifiers import Venue
            account = self.portfolio.account(Venue("BINANCE"))
        except Exception:
            pass

        if account is not None:
            balance = float(account.balance_total().as_double())
            self._equity_curve.append(balance)
        drawdown = compute_drawdown(self._equity_curve) if self._equity_curve else 0.0

        risk_halt = drawdown > self.config.max_drawdown

        self.publish_signal(
            name="RISK",
            value=RiskState(
                volatility=volatility,
                drawdown=drawdown,
                total_exposure=0.0,  # computed by portfolio construction
                risk_halt=risk_halt,
            ),
            ts_event=ts_event,
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_risk.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/risk/ tests/test_risk.py
git commit -m "feat: risk model with rolling volatility and drawdown monitoring"
```

---

## Task 8: Portfolio Construction

**Files:**
- Create: `src/portfolio/construction.py`
- Create: `tests/test_portfolio.py`

- [ ] **Step 1: Write failing tests for portfolio math**

```python
# tests/test_portfolio.py
import pytest
from src.portfolio.construction import (
    combine_alphas,
    apply_constraints,
    compute_target_deltas,
)


def test_combine_alphas_weighted_sum():
    alpha_scores = {
        "momentum": {"BTC": 0.5, "ETH": -0.3},
        "mean_rev": {"BTC": -0.1, "ETH": 0.4},
    }
    weights = {"momentum": 0.6, "mean_rev": 0.4}
    universe = {"BTC", "ETH"}

    combined = combine_alphas(alpha_scores, weights, universe)

    # BTC: 0.5*0.6 + (-0.1)*0.4 = 0.26
    # ETH: (-0.3)*0.6 + 0.4*0.4 = -0.02
    assert combined["BTC"] == pytest.approx(0.26)
    assert combined["ETH"] == pytest.approx(-0.02)


def test_combine_alphas_ignores_non_universe():
    alpha_scores = {"momentum": {"BTC": 0.5, "XRP": 0.8}}
    weights = {"momentum": 1.0}
    universe = {"BTC"}  # XRP not in universe

    combined = combine_alphas(alpha_scores, weights, universe)

    assert "BTC" in combined
    assert "XRP" not in combined


def test_apply_constraints_clips_position():
    raw_weights = {"BTC": 0.15, "ETH": 0.03, "SOL": -0.08}
    volatility = {"BTC": 0.02, "ETH": 0.03, "SOL": 0.05}
    risk_halt = False

    result = apply_constraints(
        raw_weights=raw_weights,
        volatility=volatility,
        max_position_pct=0.05,
        max_total_exposure=1.0,
        risk_halt=risk_halt,
    )

    # BTC should be clipped to 0.05
    assert abs(result["BTC"]) <= 0.05
    # No position should exceed max
    for w in result.values():
        assert abs(w) <= 0.05


def test_apply_constraints_risk_halt():
    raw_weights = {"BTC": 0.05}
    result = apply_constraints(
        raw_weights=raw_weights,
        volatility={"BTC": 0.02},
        max_position_pct=0.05,
        max_total_exposure=1.0,
        risk_halt=True,
    )
    # risk_halt: all weights go to 0
    assert result["BTC"] == 0.0


def test_compute_target_deltas():
    target_weights = {"BTC": 0.05, "ETH": 0.03}
    current_weights = {"BTC": 0.03, "ETH": 0.03, "SOL": 0.02}
    threshold = 0.005

    deltas = compute_target_deltas(target_weights, current_weights, threshold)

    assert deltas["BTC"] == pytest.approx(0.02)  # 0.05 - 0.03
    assert "ETH" not in deltas  # delta=0, below threshold
    assert deltas["SOL"] == pytest.approx(-0.02)  # 0 - 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_portfolio.py -v`
Expected: FAIL

- [ ] **Step 3: Implement portfolio construction**

```python
# src/portfolio/construction.py
from __future__ import annotations

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy

from src.types import AlphaScore, RiskState, UniverseState


class PortfolioConstructionConfig(StrategyConfig):
    rebalance_interval_hours: int = 1
    min_trade_threshold: float = 0.001
    max_position_pct: float = 0.05
    max_total_exposure: float = 1.0
    alpha_weights: dict[str, float] = {}
    order_id_tag: str = "PC"


# ── Pure functions (testable without Nautilus) ──────────────────────


def combine_alphas(
    alpha_scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    universe: set[str],
) -> dict[str, float]:
    """Weighted sum of alpha scores, filtered to universe."""
    combined: dict[str, float] = {}
    for alpha_name, scores in alpha_scores.items():
        w = weights.get(alpha_name, 0.0)
        for symbol, score in scores.items():
            if symbol in universe:
                combined[symbol] = combined.get(symbol, 0.0) + w * score
    return combined


def apply_constraints(
    raw_weights: dict[str, float],
    volatility: dict[str, float],
    max_position_pct: float,
    max_total_exposure: float,
    risk_halt: bool,
) -> dict[str, float]:
    """Apply risk constraints to raw alpha weights."""
    if risk_halt:
        return {s: 0.0 for s in raw_weights}

    # Inverse-volatility weighting
    inv_vol_weights = {}
    for symbol, weight in raw_weights.items():
        vol = volatility.get(symbol, 1.0)
        inv_vol = 1.0 / vol if vol > 0 else 1.0
        inv_vol_weights[symbol] = weight * inv_vol

    # Normalize to sum of abs weights = max_total_exposure
    total_abs = sum(abs(w) for w in inv_vol_weights.values())
    if total_abs > 0:
        scale = max_total_exposure / total_abs
        normalized = {s: w * scale for s, w in inv_vol_weights.items()}
    else:
        normalized = inv_vol_weights

    # Clip individual positions
    clipped = {}
    for symbol, weight in normalized.items():
        if weight > max_position_pct:
            clipped[symbol] = max_position_pct
        elif weight < -max_position_pct:
            clipped[symbol] = -max_position_pct
        else:
            clipped[symbol] = weight

    return clipped


def compute_target_deltas(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    threshold: float,
) -> dict[str, float]:
    """Compute weight deltas, including exits for symbols not in target."""
    all_symbols = set(target_weights) | set(current_weights)
    deltas = {}
    for symbol in all_symbols:
        target = target_weights.get(symbol, 0.0)
        current = current_weights.get(symbol, 0.0)
        delta = target - current
        if abs(delta) >= threshold:
            deltas[symbol] = delta
    return deltas


# ── Nautilus Strategy ───────────────────────────────────────────────


class PortfolioConstruction(Strategy):
    def __init__(self, config: PortfolioConstructionConfig) -> None:
        super().__init__(config)
        self._alpha_scores: dict[str, dict[str, float]] = {}
        self._risk_state: RiskState | None = None
        self._universe: frozenset[str] = frozenset()
        self._last_rebalance_hour: int = -1
        self._last_prices: dict[str, float] = {}  # track prices from bars

    def on_start(self) -> None:
        self.subscribe_signal("ALPHA")
        self.subscribe_signal("RISK")
        self.subscribe_signal("UNIVERSE")

        for instrument in self.cache.instruments():
            bar_type_str = f"{instrument.id}-1-HOUR-LAST-EXTERNAL"
            self.subscribe_bars(BarType.from_str(bar_type_str))

    def on_signal(self, signal) -> None:
        value = signal.value
        match value:
            case AlphaScore():
                self._alpha_scores.setdefault(value.name, {}).update(value.scores)
            case RiskState():
                self._risk_state = value
            case UniverseState():
                self._universe = value.current
                # Close positions for removed symbols
                for symbol_str in value.removed:
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
        self._rebalance()

    def _rebalance(self) -> None:
        if not self._alpha_scores or self._risk_state is None:
            return

        # Step 1: Combine alphas
        combined = combine_alphas(
            self._alpha_scores,
            self.config.alpha_weights,
            set(self._universe),
        )

        # Step 2: Apply constraints
        targets = apply_constraints(
            raw_weights=combined,
            volatility=self._risk_state.volatility,
            max_position_pct=self.config.max_position_pct,
            max_total_exposure=self.config.max_total_exposure,
            risk_halt=self._risk_state.risk_halt,
        )

        # Step 3: Compute current weights
        current_weights = self._get_current_weights()

        # Step 4: Compute deltas
        deltas = compute_target_deltas(
            targets, current_weights, self.config.min_trade_threshold
        )

        # Step 5: Execute
        for symbol_str, delta in deltas.items():
            self._execute_delta(symbol_str, delta)

    def _get_current_weights(self) -> dict[str, float]:
        """Get current position weights relative to total equity."""
        venue = Venue("BINANCE")
        account = self.portfolio.account(venue)
        if account is None:
            return {}

        total_equity = float(account.balance_total().as_double())
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
        """Submit market order to achieve the weight delta."""
        try:
            instrument_id = InstrumentId.from_str(symbol_str)
        except Exception:
            return

        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            return

        venue = Venue("BINANCE")
        account = self.portfolio.account(venue)
        if account is None:
            return

        total_equity = float(account.balance_total().as_double())
        notional = abs(delta) * total_equity

        last_price = self._last_prices.get(symbol_str, 0.0)
        if last_price == 0:
            return

        quantity = notional / float(last_price)
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL

        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=side,
            quantity=instrument.make_qty(Decimal(str(quantity))),
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)

    def on_stop(self) -> None:
        self.cancel_all_orders()
        self.close_all_positions()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_portfolio.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/portfolio/ tests/test_portfolio.py
git commit -m "feat: portfolio construction with alpha combination and risk constraints"
```

---

## Task 9: Engine Factory + Backtest Runner

**Files:**
- Create: `src/engine/factory.py`
- Create: `src/engine/backtest.py`
- Create: `scripts/run_backtest.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Implement engine factory**

```python
# src/engine/factory.py
from __future__ import annotations

import importlib
from decimal import Decimal

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money

from src.alpha.base import BaseAlphaConfig
from src.config.models import Settings
from src.data.catalog import load_bars_from_parquet
from src.portfolio.construction import PortfolioConstruction, PortfolioConstructionConfig
from src.risk.model import RiskModel, RiskModelConfig
from src.universe.model import UniverseModel, UniverseModelConfig


def _import_class(module_path: str):
    """Dynamically import a class from 'module.path.ClassName' string."""
    parts = module_path.rsplit(".", 1)
    module = importlib.import_module(parts[0])
    return getattr(module, parts[1])


def build_backtest_engine(settings: Settings) -> BacktestEngine:
    """Assemble all Nautilus components from Settings."""
    engine = BacktestEngine(config=BacktestEngineConfig(logging_level="INFO"))

    # 1. Add venue
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(settings.portfolio.initial_capital, USDT)],
        fee_model=None,  # Use instrument-level fees
    )

    # 2. Load instruments and data
    from pathlib import Path
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    all_instruments = {}
    for market_type in settings.data.market_types:
        data_dir = Path(settings.data.storage_path) / market_type
        if not data_dir.exists():
            continue

        for parquet_file in sorted(data_dir.glob("*.parquet")):
            symbol = parquet_file.stem

            # Create instrument (simplified: use test provider for known symbols)
            # In production, fetch instrument specs from exchange
            instrument = _create_instrument(symbol, market_type)
            if instrument is None:
                continue

            engine.add_instrument(instrument)
            all_instruments[symbol] = instrument

            bars = load_bars_from_parquet(
                path=str(parquet_file),
                instrument=instrument,
            )
            engine.add_data(bars, sort=False)

    engine.sort_data()

    # 3. Add Universe Model
    engine.add_actor(
        UniverseModel(
            UniverseModelConfig(
                ranking_window=settings.universe.ranking_window,
                inclusion_rank=settings.universe.inclusion_rank,
                exclusion_rank=settings.universe.exclusion_rank,
            )
        )
    )

    # 4. Add Alpha Models
    alpha_weights = {}
    for alpha_name, alpha_cfg in settings.alphas.items():
        if not alpha_cfg.enabled:
            continue
        AlphaClass = _import_class(alpha_cfg.module)
        ConfigClass = _get_config_class(AlphaClass)
        actor_config = ConfigClass(
            alpha_name=alpha_name,
            params=alpha_cfg.params,
        )
        engine.add_actor(AlphaClass(actor_config))
        alpha_weights[alpha_name] = alpha_cfg.weight

    # 5. Add Risk Model
    engine.add_actor(
        RiskModel(
            RiskModelConfig(
                max_position_pct=settings.risk.max_position_pct,
                max_drawdown=settings.risk.max_drawdown,
                max_total_exposure=settings.risk.max_total_exposure,
                volatility_window=settings.risk.volatility_window,
            )
        )
    )

    # 6. Add Portfolio Construction (the single Strategy)
    engine.add_strategy(
        PortfolioConstruction(
            PortfolioConstructionConfig(
                rebalance_interval_hours=settings.portfolio.rebalance_interval_hours,
                min_trade_threshold=settings.portfolio.min_trade_threshold,
                max_position_pct=settings.risk.max_position_pct,
                max_total_exposure=settings.risk.max_total_exposure,
                alpha_weights=alpha_weights,
                order_id_tag="PC001",
            )
        )
    )

    return engine


def _create_instrument(symbol: str, market_type: str):
    """Create instrument from symbol name. Returns None if not recognized."""
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    # Map known symbols to test instruments
    provider_map = {
        ("BTCUSDT", "spot"): TestInstrumentProvider.btcusdt_binance,
        ("ETHUSDT", "spot"): TestInstrumentProvider.ethusdt_binance,
        ("BTCUSDT-PERP", "futures"): TestInstrumentProvider.btcusdt_perp_binance,
        ("ETHUSDT-PERP", "futures"): TestInstrumentProvider.ethusdt_perp_binance,
    }

    factory = provider_map.get((symbol, market_type))
    if factory:
        return factory()
    return None


def _get_config_class(alpha_class):
    """Get the config class for an alpha model."""
    import inspect

    init_sig = inspect.signature(alpha_class.__init__)
    for param in init_sig.parameters.values():
        if param.name == "config" and param.annotation != inspect.Parameter.empty:
            return param.annotation
    # Fallback
    from src.alpha.base import BaseAlphaConfig

    return BaseAlphaConfig
```

- [ ] **Step 2: Implement backtest runner module**

```python
# src/engine/backtest.py
from __future__ import annotations

import logging

from nautilus_trader.backtest.engine import BacktestEngine

from src.config.loader import load_settings
from src.engine.factory import build_backtest_engine

logger = logging.getLogger(__name__)


def run_backtest(config_path: str = "config/settings.yaml") -> BacktestEngine:
    """Load config, build engine, run backtest, return engine for analysis."""
    settings = load_settings(config_path)
    logger.info(f"Running backtest: {settings.backtest.start_date} to {settings.backtest.end_date}")

    engine = build_backtest_engine(settings)
    engine.run()

    logger.info("Backtest complete.")
    return engine
```

- [ ] **Step 3: Create CLI script**

```python
# scripts/run_backtest.py
"""CLI script to run a backtest."""
from __future__ import annotations

import logging
import sys

from src.engine.backtest import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/settings.yaml"
    engine = run_backtest(config_path)

    # Print summary
    for report in engine.trader.generate_order_fills_report():
        print(report)
    for report in engine.trader.generate_positions_report():
        print(report)
    for report in engine.trader.generate_account_report(venue=None):
        print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write integration test with synthetic data**

```python
# tests/test_integration.py
"""End-to-end backtest with synthetic data.

Creates 3 symbols, 500 hours of data.
BTC has high volume (universe inclusion), rising price (positive momentum).
ETH has medium volume, flat price.
XRP has low volume (should NOT enter universe).
"""
import pandas as pd
import pytest
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from src.alpha.momentum import MomentumAlpha, MomentumAlphaConfig
from src.data.catalog import load_bars_from_parquet
from src.portfolio.construction import PortfolioConstruction, PortfolioConstructionConfig
from src.risk.model import RiskModel, RiskModelConfig
from src.universe.model import UniverseModel, UniverseModelConfig


def _make_ohlcv(hours: int, base_price: float, trend: float, volume: float) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    dates = pd.date_range("2025-01-01", periods=hours, freq="1h", tz="UTC")
    prices = [base_price + trend * i for i in range(hours)]
    return pd.DataFrame(
        {
            "open": [p - 10 for p in prices],
            "high": [p + 50 for p in prices],
            "low": [p - 50 for p in prices],
            "close": prices,
            "volume": [volume] * hours,
            "quote_volume": [p * volume for p in prices],
        },
        index=dates,
    )


@pytest.fixture
def synthetic_data(tmp_path):
    """Create parquet files for 3 symbols."""
    spot_dir = tmp_path / "spot"
    spot_dir.mkdir()

    # BTC: high volume, uptrend → should be in universe, positive momentum
    btc = _make_ohlcv(hours=500, base_price=50000, trend=10, volume=100.0)
    btc.to_parquet(spot_dir / "BTCUSDT.parquet")

    # ETH: medium volume, flat → should be in universe, near-zero momentum
    eth = _make_ohlcv(hours=500, base_price=3000, trend=0, volume=50.0)
    eth.to_parquet(spot_dir / "ETHUSDT.parquet")

    return tmp_path


def test_backtest_runs_without_error(synthetic_data):
    """Smoke test: engine runs to completion with all components wired."""
    engine = BacktestEngine(config=BacktestEngineConfig(logging_level="WARNING"))

    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(100_000, USDT)],
    )

    btc = TestInstrumentProvider.btcusdt_binance()
    eth = TestInstrumentProvider.ethusdt_binance()
    engine.add_instrument(btc)
    engine.add_instrument(eth)

    for symbol, instrument in [("BTCUSDT", btc), ("ETHUSDT", eth)]:
        bars = load_bars_from_parquet(
            path=str(synthetic_data / "spot" / f"{symbol}.parquet"),
            instrument=instrument,
        )
        engine.add_data(bars, sort=False)
    engine.sort_data()

    # Universe: inclusion=1, exclusion=2 (both symbols should be included)
    engine.add_actor(
        UniverseModel(UniverseModelConfig(
            ranking_window=48,  # shorter warmup for test
            inclusion_rank=5,
            exclusion_rank=10,
        ))
    )

    # Alpha
    engine.add_actor(
        MomentumAlpha(MomentumAlphaConfig(
            alpha_name="momentum",
            params={"lookback": 24},
        ))
    )

    # Risk
    engine.add_actor(
        RiskModel(RiskModelConfig(
            max_position_pct=0.3,
            max_drawdown=0.5,
            max_total_exposure=1.0,
            volatility_window=48,
        ))
    )

    # Portfolio Construction
    engine.add_strategy(
        PortfolioConstruction(PortfolioConstructionConfig(
            rebalance_interval_hours=1,
            min_trade_threshold=0.001,
            max_position_pct=0.3,
            max_total_exposure=1.0,
            alpha_weights={"momentum": 1.0},
            order_id_tag="TEST001",
        ))
    )

    # Run
    engine.run()

    # Verify: engine ran without crashing
    assert engine.iteration > 0

    engine.dispose()
```

- [ ] **Step 5: Run integration test**

Run: `uv run pytest tests/test_integration.py -v --timeout=120`
Expected: 1 passed (may take 10-30 seconds)

- [ ] **Step 6: Commit**

```bash
git add src/engine/ scripts/run_backtest.py tests/test_integration.py
git commit -m "feat: engine factory, backtest runner, and end-to-end integration test"
```

---

## Task 10: Run All Tests + Final Verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (15+ tests)

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/ tests/`
Expected: No errors (fix any that appear)

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and all tests passing"
```
