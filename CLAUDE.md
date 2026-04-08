# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parangsae-quant is a crypto quantitative trading system built on **NautilusTrader**. It implements a Grinold-Kahn five-model architecture (Universe, Alpha, Risk, Portfolio, Execution) for backtesting and live trading on Binance. The system trades the top ~100 coins by volume using hourly bars.

## Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_alpha.py
uv run pytest tests/test_alpha.py::TestQMMomentum::test_positive_trending_produces_positive_alpha

# Lint
uv run ruff check src/ tests/

# Fetch market data (requires BINANCE_API_KEY/BINANCE_API_SECRET env vars)
uv run python scripts/fetch_data.py

# Run backtest
uv run python scripts/run_backtest.py
```

## Architecture

### Signal Flow

All components communicate through NautilusTrader's `publish_signal`/`subscribe_signal` mechanism using JSON-serialized messages. The flow is:

```
Bars → UniverseModel (Actor) → "UNIVERSE" signal
Bars → AlphaModels (Actors) → "ALPHA" signal
Bars → RiskModel (Actor)    → "RISK" signal
                                    ↓
                        PortfolioConstruction (Strategy)
                        subscribes to all three signals,
                        rebalances on each bar
```

### Key Design Patterns

- **Actors vs Strategy**: Universe, Alpha, and Risk models extend `nautilus_trader.common.actor.Actor`. Only PortfolioConstruction extends `Strategy` (which can submit orders). This is a NautilusTrader distinction — Actors observe, Strategies trade.

- **Pure functions + Actor wrappers**: Each model separates pure computation functions (tested directly) from the Actor/Strategy class that handles NautilusTrader lifecycle. For example, `compute_qm_momentum()` is a standalone function; `QMMomentumAlpha` is the Actor wrapper. Always test the pure functions directly.

- **Dynamic alpha loading**: Alpha models are loaded dynamically from `config/settings.yaml` via `src/engine/factory.py:_import_class()`. Only modules under `src.alpha.*` are allowed. Each alpha module must export a Config class (ending in `Config`) and a model class.

- **Signal serialization**: Signals are JSON strings with a `"type"` discriminator field (`"AlphaScore"`, `"RiskState"`, `"UniverseState"`). The frozen dataclasses in `src/types.py` define the contract but are reconstructed from JSON on the receiving end.

### Component Details

- **`src/universe/model.py`** — Ranks coins by trailing quote volume, applies hysteresis (inclusion_rank < exclusion_rank) to prevent churn.

- **`src/alpha/`** — Two alphas: QM Momentum (vol-adjusted momentum with FIP quality filter) and Low Volatility (negative realized vol). New alphas extend `BaseAlphaModel` and implement `compute_alpha()`.

- **`src/risk/model.py`** — Four-layer multiplicative risk model: BTC regime (EMA trend), volatility targeting, correlation monitor, drawdown scaling. Each layer produces a scale factor; they multiply together with a floor.

- **`src/portfolio/construction.py`** — Z-score combines alphas cross-sectionally, selects holdings with hysteresis, computes alpha×inv-vol weights, applies tiered position caps (BTC/ETH/other), liquidity caps, turnover limits (hourly + daily), then executes via market orders.

- **`src/engine/factory.py`** — `build_backtest_engine()` assembles the full NautilusTrader BacktestEngine from a `Settings` object. Loads instruments from `data/{market_type}/_instruments.json` with fallback to NautilusTrader test providers.

- **`src/data/instruments.py`** — Converts CCXT market metadata to NautilusTrader `CurrencyPair`/`CryptoPerpetual` instruments. Symbol convention: spot `BTCUSDT`, futures `BTCUSDT-PERP`.

### Config

All parameters live in `config/settings.yaml`, parsed via Pydantic models in `src/config/models.py`. Environment variables are substituted with `${VAR_NAME}` syntax (used for API keys).

### Bar Type Convention

All components use the string format `"{instrument_id}-1-HOUR-LAST-EXTERNAL"` when subscribing to bars via `BarType.from_str()`.

### Data Storage

Market data is stored as parquet files under `data/{spot,futures}/{SYMBOL}.parquet`. Instrument metadata is in `data/{spot,futures}/_instruments.json`.

## Research Pipeline (PRD v1: Crypto-Native Residual Ridge)

A separate ML research pipeline lives under `src/research/`, implementing the PRD (`parangsae_quant_research_prd_v1_crypto_native.md`). It does NOT use NautilusTrader — it's a standalone pandas/scikit-learn pipeline.

### Key Differences from Main System

| | Main (`src/`) | Research (`src/research/`) |
|---|---|---|
| Strategy | Rule-based momentum | Ridge regression on BTC residuals |
| Timeframe | 1H bars | 5m research / 20m decision |
| Direction | Long-only | Long/Short 6+6 |
| Universe | 100 coins | 30 coins, crypto-native only |
| Execution | Market orders | Passive limit + 5m post-fill |

### Research Config

- `config/universe.yaml` — Universe inclusion/exclusion rules, sector map
- `config/model.yaml` — All model parameters (PRD Appendix A)
- `features/manifest.csv` — 40 feature definitions

### Research Signal Flow

```
Data (1m OHLCV, funding, OI)
    ↓
Universe Filter (30 crypto-native coins, weekly)
    ↓
Feature Pipeline (40 features, 5 blocks)
    ↓ cross-sectional z-score, ±5 clip
Labels (rolling OLS beta → y20/y60 residual returns)
    ↓
Ridge Model (walk-forward: 90d train, 14d val, 60m embargo)
    ↓ score = 0.7 * ŷ20 + 0.3 * ŷ60
Portfolio (L/S 6+6, beta-neutral, 12% cap)
    ↓
Execution Sim (passive limit + 5m post-fill validation)
    ↓
Analytics (Sharpe, rank IC, regime breakdown, Go/No-Go)
```

### Running Research Tests

```bash
uv run pytest tests/research/ -v   # 126 tests
```

## Conventions

- Python 3.12+, line length 100 (ruff)
- The class is named `PortfolioConstruction` (typo is intentional in current codebase — do not rename without explicit request)
- NautilusTrader timestamps are nanoseconds (int). Hour extraction: `ts_event // 3_600_000_000_000`
- Alpha values of `-inf` mean "filtered out / insufficient data"
- All prices and returns use `float`, not `Decimal` (Decimal only for order quantities via `instrument.make_qty()`)
