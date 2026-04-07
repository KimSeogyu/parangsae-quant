# Parangsae Quant — Architecture Design Spec

## Overview

Binance Spot + Futures OHLCV(1h) 데이터를 활용한 퀀트 트레이딩 시스템. Nautilus Trader를 코어 엔진으로 사용하며, Grinold-Kahn 프레임워크 기반의 5-모델 아키텍처로 설계한다.

**초기 스코프**: 백테스트 전용. 아키텍처는 라이브 트레이딩까지 확장 가능하도록 설계.

## Decisions

| 항목 | 결정 | 근거 |
|---|---|---|
| 언어 | Python 3.12+ | 퀀트 생태계 표준 |
| 코어 엔진 | Nautilus Trader | 이벤트 기반 + Cython/Rust 성능 + Binance 내장 어댑터 |
| 데이터 수집 | CCXT | historical bulk download에 적합, 100+ 거래소 지원 |
| 데이터 저장 | Parquet (파일 기반) | 인프라 부담 없이 빠른 읽기, pandas/polars 친화 |
| 캔들 타임프레임 | 1h | 리밸런싱 주기와 동일 |
| 마켓 | Binance Spot + Futures | 사용자 요구사항 |
| 패키지 매니저 | uv | Nautilus 공식 권장, pip 대비 10-100x 빠름 |
| 설정 관리 | 단일 YAML + Pydantic 검증 | 파편화 방지, externally configurable |

## Architecture

### 5-Model Architecture (Grinold-Kahn Framework)

```
                    ┌─────────────────────────────┐
                    │       Data Pipeline          │
                    │   (DataEngine + Catalog)      │
                    └──────────┬──────────────────┘
                               │ bars
                    ┌──────────▼──────────────────┐
                    │      Universe Model          │
                    │         (Actor)               │
                    │                               │
                    │  2주 평균 거래대금 랭킹        │
                    │  편입: Top 100               │
                    │  추방: 150위 이하             │
                    └──────────┬──────────────────┘
                               │ universe events
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │ Alpha Model  │  │ Alpha Model  │  │ Alpha Model  │
   │   (Actor)    │  │   (Actor)    │  │   (Actor)    │
   │  momentum    │  │  mean-rev    │  │  extensible  │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │ alpha signals    │                  │
          └────────┬─────────┘──────────────────┘
                   ▼
        ┌─────────────────────┐
        │     Risk Model      │
        │      (Actor)        │
        │                     │
        │  volatility, dd,    │
        │  exposure, halt     │
        └─────────┬───────────┘
                  │ risk signals
                  ▼
     ┌────────────────────────────┐
     │  Portfolio Construction    │
     │       (Strategy)           │
     │                            │
     │  alpha 합산 → constraints  │
     │  → target weights → orders │
     └────────────┬───────────────┘
                  │ submit_order()
                  ▼
     ┌────────────────────────────┐
     │    Execution Model         │
     │  (Nautilus ExecutionEngine) │
     │                            │
     │  backtest: fill simulation │
     │  live: Binance adapter     │
     └────────────────────────────┘
```

### Nautilus Mapping

| 모델 | Nautilus 컴포넌트 | 역할 |
|---|---|---|
| Data Pipeline | DataEngine + ParquetDataCatalog | OHLCV 수집, 저장, 서빙 |
| Universe Model | `Actor` | 거래대금 기반 동적 유니버스 관리 |
| Alpha Model | `Actor` (N개) | 알파 시그널 계산 → `publish_signal()` |
| Risk Model | `Actor` | 포트폴리오 리스크 모니터링 → risk signal 발행 |
| Portfolio Construction | `Strategy` | 알파 + 리스크 종합 → target portfolio → 주문 생성 |
| Execution Model | Nautilus `ExecutionEngine` | 주문 라우팅, 체결, 슬리피지 |

### Key Principle: 단일 Strategy

`Portfolio Construction`만이 유일한 `Strategy`이다. 주문을 생성할 수 있는 컴포넌트는 이것 하나뿐. Alpha가 10개여도 주문을 만드는 곳은 단 하나이므로, 서로 다른 알파가 동시에 반대 방향 주문을 내는 사고가 구조적으로 불가능하다.

## Data Pipeline

### Data Fetcher (CCXT)

- Binance REST API를 통해 전 종목 1h OHLCV 1년치 수집
- 증분 업데이트: 마지막 timestamp 이후만 fetch
- CCXT의 rate limit 자동 관리

### Bootstrapping (초기 수집)

첫 실행 시에는 유니버스가 없으므로, Binance의 전체 활성 종목 목록을 먼저 가져온 뒤 **전 종목**의 1년치 데이터를 수집한다. 유니버스 필터링은 BacktestEngine 내에서 UniverseModel이 런타임에 수행한다. 이후 증분 업데이트 시에도 전 종목을 대상으로 하되, 마지막 timestamp 이후 데이터만 fetch한다.

### Parquet Schema

```
timestamp     (datetime64[ns, UTC])  — bar close time
open          (float64)
high          (float64)
low           (float64)
close         (float64)
volume        (float64)
quote_volume  (float64)              — 거래대금 (유니버스 랭킹 필수)
```

### Storage Layout

```
data/
├── spot/
│   ├── BTCUSDT.parquet
│   ├── ETHUSDT.parquet
│   └── ...
└── futures/
    ├── BTCUSDT-PERP.parquet
    └── ...
```

### Nautilus 연동

- `BarDataWrangler`로 Parquet → Nautilus `Bar` 객체 변환
- `ts_init` = bar close time (Nautilus 규약)
- `BacktestEngine.add_data(bars, sort=False)` 후 `sort_data()` 일괄 정렬

## Universe Model

### Hysteresis Buffer (100/150)

```
순위:  1 ──── 100 ──── 150 ──── 300+
       ├── 편입 존 ──┤
       ├────── 유지 존 ────────┤
                      ├─ 추방 ─┤
```

- **편입 조건**: 2주(336h) 평균 거래대금 순위 1~100위
- **추방 조건**: 150위 밖으로 하락
- **히스테리시스**: 편입/추방 임계값 갭으로 잦은 종목 교체 방지
- **워밍업**: 최초 336h(2주)는 랭킹 데이터가 충분하지 않으므로 유니버스를 빈 상태로 유지. 336h 이후 첫 랭킹 계산 시점부터 편입 시작

### Event 전파

Universe 변경은 MessageBus를 통해 모든 컴포넌트에 전파:
- Alpha Model → 새 종목 바 구독 시작
- Portfolio Construction → 추방 종목 포지션 청산

## Alpha Model

### Contract

```python
class BaseAlphaModel(Actor):
    def compute_alpha(self, bar: Bar) -> float:
        """리서처가 구현하는 유일한 메서드"""
        raise NotImplementedError
```

### Signal Convention

- 값 범위: 제한 없음 (z-score, raw return, 확률 등)
- 부호: 양수 = long, 음수 = short, 0 = 중립
- Portfolio Construction이 정규화 담당

### Extensibility

새 알파 추가 = 파일 하나 작성 + settings.yaml에 등록:

```yaml
alphas:
  my_new_alpha:
    enabled: true
    module: src.alpha.my_new_alpha.MyNewAlpha
    weight: 0.4
    params:
      lookback: 48
```

## Risk Model

### Published Constraints

| 제약 | 설명 | 용도 |
|---|---|---|
| `volatility[symbol]` | 종목별 롤링 변동성 | 역변동성 가중 |
| `max_position_pct` | 단일 종목 최대 비중 | 집중 리스크 방지 |
| `drawdown` | 현재 최대 낙폭 | 임계치 초과 시 진입 중단 |
| `total_exposure` | 총 노출도 | 레버리지 제한 |
| `risk_halt` | 긴급 정지 시그널 | 신규 진입 차단 (기존 포지션 유지, 강제 청산하지 않음) |

## Portfolio Construction

### Rebalancing Loop (매 1h)

1. **Alpha 합산**: 각 Alpha Model의 시그널을 config 가중치로 weighted sum
2. **제약 적용**: max_position_pct 클리핑, 역변동성 가중, 총 노출도 정규화, risk_halt 체크
3. **Delta 계산**: target_weights - current_weights
4. **주문 생성**: `min_trade_threshold` 초과 delta에 대해 Market Order 생성

### Alpha Combination

```
combined[symbol] = Σ (alpha_weight[i] × alpha_score[i][symbol])
```

Cross-sectional 정규화 후 제약 조건 적용.

## Configuration

### Single YAML (config/settings.yaml)

```yaml
system:
  mode: backtest
  log_level: INFO

data:
  exchange: binance
  market_types: [spot, futures]
  timeframe: 1h
  history_days: 365
  storage_path: data/

universe:
  ranking_metric: quote_volume
  ranking_window: 336
  inclusion_rank: 100
  exclusion_rank: 150
  rebalance_interval: 1h

alphas:
  momentum:
    enabled: true
    module: src.alpha.momentum.MomentumAlpha
    weight: 0.5
    params:
      lookback: 24
  mean_reversion:
    enabled: true
    module: src.alpha.mean_reversion.MeanReversionAlpha
    weight: 0.3
    params:
      lookback: 72
      z_threshold: 2.0

risk:
  max_position_pct: 0.05
  max_drawdown: 0.15
  max_total_exposure: 1.0
  volatility_window: 168

portfolio:
  initial_capital: 100000
  rebalance_interval: 1h
  min_trade_threshold: 0.001

backtest:
  start_date: "2025-04-07"
  end_date: "2026-04-07"
  fee_rate: 0.001
  slippage_model: default

binance:
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}
```

### Design Principles

- **SSOT**: 이 파일 하나만 수정하면 시스템 전체 행동 변경
- **Alpha plug-in**: alphas 섹션에 항목 추가 = 새 전략 활성화
- **Secret 분리**: 환경변수 참조 (`${ENV_VAR}`)
- **Pydantic 검증**: YAML → Pydantic model로 파싱, 스키마 위반 시 즉시 에러

## Project Structure

```
parangsae-quant/
├── config/
│   └── settings.yaml
├── src/
│   ├── config/
│   │   ├── loader.py           # YAML → dict, 환경변수 치환
│   │   └── models.py           # Pydantic models
│   ├── data/
│   │   ├── fetcher.py          # CCXT Binance OHLCV 수집
│   │   └── catalog.py          # Parquet ↔ Nautilus Bar 변환
│   ├── universe/
│   │   └── model.py            # UniverseModel(Actor)
│   ├── alpha/
│   │   ├── base.py             # BaseAlphaModel(Actor)
│   │   └── momentum.py         # 예시 알파
│   ├── risk/
│   │   └── model.py            # RiskModel(Actor)
│   ├── portfolio/
│   │   └── construction.py     # PortfolioConstruction(Strategy)
│   └── engine/
│       ├── backtest.py         # BacktestEngine 셋업 & 실행
│       └── factory.py          # Config → Nautilus 컴포넌트 조립
├── scripts/
│   ├── fetch_data.py           # CLI: 데이터 수집
│   └── run_backtest.py         # CLI: 백테스트 실행
├── tests/
│   ├── test_universe.py
│   ├── test_alpha.py
│   ├── test_risk.py
│   ├── test_portfolio.py
│   └── test_backtest_integration.py
├── data/                       # (gitignore)
├── results/                    # (gitignore)
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Dependencies

```toml
[project]
name = "parangsae-quant"
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
dev = ["pytest", "ruff"]
```

## CLI Workflow

```bash
# 1. 데이터 수집
uv run python scripts/fetch_data.py

# 2. 백테스트 실행
uv run python scripts/run_backtest.py

# 3. 설정 변경 후 재실행
#    config/settings.yaml 수정 → 다시 2번
```

## Testing Strategy

- **Unit**: 각 모델(Universe, Alpha, Risk, Portfolio) 독립 테스트
- **Integration**: 전체 파이프라인 end-to-end 백테스트
- **Alpha 검증**: 알파 시그널의 방향성/범위 sanity check
