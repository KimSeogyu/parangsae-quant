# Crypto-Native Residual Ridge Implementation Plan

**PRD**: `parangsae_quant_research_prd_v1_crypto_native.md`
**Date**: 2026-04-08
**Status**: Planning

## Gap Analysis

### Current System vs PRD Requirements

| 영역 | 현재 | PRD | 변경 규모 |
|---|---|---|---|
| **전략 패러다임** | Rule-based momentum+lowvol | ML (Ridge) residual prediction | **완전 재설계** |
| **타임프레임** | 1H bars | 5m research / 20m decision | **신규** |
| **알파 생성** | 2개 팩터 (모멘텀, 저변동성) | 40개 feature → Ridge regression | **신규** |
| **예측 대상** | Raw return 기반 ranking | BTC beta-adjusted residual return | **신규** |
| **포트폴리오** | Long-only 15종목 | Long/Short 6+6, beta-neutral | **대폭 수정** |
| **유니버스** | 100종목, hourly ranking | 30종목, weekly, 180d+, funding/OI 커버리지 | **대폭 수정** |
| **실행** | Market order (IOC) | Passive limit + 5m post-fill 검증 | **신규** |
| **백테스트** | 단일 모드 | Fast + Accurate 2단계 | **신규** |
| **데이터** | OHLCV only | OHLCV + Funding + OI + Orderbook + Mark Price | **대폭 확장** |
| **학습** | 없음 (rule-based) | Walk-forward Ridge, 90d train, 14d validation | **신규** |

### Architecture Decision

PRD는 현재의 NautilusTrader event-driven 아키텍처와 근본적으로 다른 **research/ML pipeline**이다.

**결정**: 기존 `src/` 모듈은 유지하고, PRD 연구 파이프라인을 `src/research/` 하위에 별도 구축한다. 추후 Ridge 모델의 예측 결과를 NautilusTrader Actor로 감싸서 실행 엔진과 통합할 수 있다.

---

## Module Structure

```
src/research/                          # NEW - PRD research pipeline
├── __init__.py
├── config.py                          # PRD-specific Pydantic config models
├── universe/
│   ├── __init__.py
│   └── filter.py                      # Crypto-native universe filter
├── data/
│   ├── __init__.py
│   ├── fetcher.py                     # Extended: funding, OI, orderbook, mark price
│   ├── aggregator.py                  # 1m → 5m → 20m bar aggregation
│   └── alignment.py                   # Point-in-time alignment enforcement
├── labels/
│   ├── __init__.py
│   ├── beta.py                        # Rolling OLS beta estimation
│   └── residual.py                    # y20, y60 residual return computation
├── features/
│   ├── __init__.py
│   ├── registry.py                    # Feature manifest & registry
│   ├── relative.py                    # Block 1: beta, residual momentum (8)
│   ├── trend.py                       # Block 2: momentum, path quality (8)
│   ├── liquidity.py                   # Block 3: volume, spread, depth (8)
│   ├── derivatives.py                 # Block 4: funding, OI, basis (8)
│   ├── risk_features.py               # Block 5: vol, correlation, regime (8)
│   └── preprocess.py                  # Cross-sectional z-score, clip, imputation
├── models/
│   ├── __init__.py
│   ├── ridge.py                       # Ridge training & prediction pipeline
│   ├── baselines.py                   # Zero, simple rule, OLS, Lasso, ElasticNet
│   └── walkforward.py                 # Walk-forward train/validate/test orchestration
├── portfolio/
│   ├── __init__.py
│   └── construction.py                # Long/short, beta-neutral, position sizing
├── execution/
│   ├── __init__.py
│   ├── passive.py                     # Passive limit order fill simulation
│   └── postfill.py                    # 5-min post-fill validation
├── backtest/
│   ├── __init__.py
│   ├── engine.py                      # Walk-forward backtest orchestration
│   ├── fast.py                        # Fast mode: touch-based fill
│   └── accurate.py                    # Accurate mode: L1 snapshot fill
└── analytics/
    ├── __init__.py
    ├── metrics.py                     # Rank IC, Sharpe, turnover, beta drift
    └── report.py                      # Performance report by regime segment

config/
├── universe.yaml                      # NEW - Universe inclusion/exclusion rules
└── model.yaml                         # NEW - Model hyperparams, feature manifest

features/
└── manifest.csv                       # NEW - 40 feature definitions

scripts/
├── fetch_research_data.py             # NEW - Extended data fetcher script
└── run_research_backtest.py           # NEW - Research backtest entry point

tests/research/                        # NEW - All research pipeline tests
├── __init__.py
├── test_universe_filter.py
├── test_beta.py
├── test_residual_labels.py
├── test_features_relative.py
├── test_features_trend.py
├── test_features_liquidity.py
├── test_features_derivatives.py
├── test_features_risk.py
├── test_preprocess.py
├── test_ridge.py
├── test_walkforward.py
├── test_portfolio_ls.py
├── test_execution.py
└── test_backtest_engine.py
```

---

## Implementation Tasks (10 tasks)

### Task 1: Config & Project Setup
**Files**: `src/research/__init__.py`, `src/research/config.py`, `config/universe.yaml`, `config/model.yaml`, `features/manifest.csv`, `pyproject.toml`

- Pydantic config models for PRD parameters (Appendix A 전체)
- `config/universe.yaml`: inclusion/exclusion rules, liquidity thresholds, sector map
- `config/model.yaml`: train window(90d), validation window(14d), embargo(60m), Ridge alpha grid, weight caps
- `features/manifest.csv`: 40개 feature 정의 (name, block, function, params)
- `pyproject.toml`에 scikit-learn, statsmodels 의존성 추가
- Tests: config parsing, validation

### Task 2: Extended Data Fetcher
**Files**: `src/research/data/fetcher.py`, `src/research/data/aggregator.py`, `src/research/data/alignment.py`

- CCXT를 통한 1분 OHLCV, funding rate, open interest 수집
- Mark/Index price, top-of-book bid/ask snapshot 수집
- 1m → 5m research bar, 5m → 20m decision bar 집계
- Point-in-time alignment 강제 (t 시점 feature는 t까지의 데이터만 사용)
- Contract metadata (tick size, lot size, fee schema) 수집
- Parquet 저장: `data/research/{ohlcv_1m, funding, oi, orderbook, mark_price}/`
- Tests: aggregation correctness, alignment enforcement

### Task 3: Universe Filter
**Files**: `src/research/universe/filter.py`

- Native crypto risk asset만 포함 (stablecoin, tokenized gold/bond/stock, leveraged token 제외)
- 180일 이상 상장 이력 필터
- 30일 median 24h quote volume 기준 상위 30개
- Funding/OI coverage ≥ 95% 요구
- 과도한 spread 종목 제외
- 주 1회 재구성 (output: `universe_history.parquet`)
- Tests: 각 필터 조건별 unit test

### Task 4: Label Pipeline (Beta & Residual)
**Files**: `src/research/labels/beta.py`, `src/research/labels/residual.py`

- Rolling OLS beta estimation: `r_i ~ r_BTC` (1주, 2주 window, 5분 수익률)
- Mid price 기반 log return: `r_i(t,h) = log(Mid_i(t+h) / Mid_i(t))`
- Primary label (y20): `r_i(t,20m) - beta_i,t * r_BTC(t,20m)`
- Secondary label (y60): `r_i(t,60m) - beta_i,t * r_BTC(t,60m)`
- Label은 t 이후 구간에서만 계산 (look-ahead bias 방지)
- Output: `labels_residual.parquet`
- Tests: beta estimation accuracy, residual return calculation, no future data leakage

### Task 5: Feature Pipeline (40 features, 5 blocks)
**Files**: `src/research/features/relative.py`, `trend.py`, `liquidity.py`, `derivatives.py`, `risk_features.py`, `registry.py`

**Block 1 - Relative/Beta (8)**:
`beta_btc_1w`, `beta_btc_2w`, `resid_mom_1h`, `resid_mom_4h`, `resid_mom_1d`, `resid_z_1h`, `resid_z_4h`, `resid_rsi_14`

**Block 2 - Trend/Path (8)**:
`mom_1h`, `mom_4h`, `mom_1d`, `mom_4h_voladj`, `mom_1d_voladj`, `fip_like_1d`, `trend_linearity_1d`, `wickiness_1d`

**Block 3 - Liquidity/Execution (8)**:
`qvol_z_1d`, `qvol_z_7d`, `amihud_1d`, `vwap_dev_1h`, `spread_bps`, `spread_z_1d`, `depth_top1_usd`, `ob_imbalance_1m`

**Block 4 - Derivatives/Crowding (8)**:
`funding`, `funding_z_7d`, `funding_change_1d`, `oi_change_1h`, `oi_change_1d`, `oi_to_vol`, `basis`, `basis_z_7d`

**Block 5 - Risk/State (8)**:
`rvol_1h`, `rvol_1d`, `downside_vol_1d`, `corr_btc_1d`, `btc_trend_4h`, `alt_breadth_4h`, `median_funding_breadth`, `xs_corr_1d`

**Feature Registry**: manifest.csv와 연결된 함수 매핑
- Tests: 각 block별 feature 계산 correctness

### Task 6: Feature Preprocessing
**Files**: `src/research/features/preprocess.py`

- 시점별 cross-sectional z-score
- ±5 clip
- 결측값: 시점별 단면 중앙값 대체
- Coverage 미달 종목은 유니버스 제외 판정
- Feature selection 없음 (전체 40개 투입)
- Tests: z-score, clipping, imputation, coverage check

### Task 7: Model Pipeline (Ridge + Baselines)
**Files**: `src/research/models/ridge.py`, `baselines.py`, `walkforward.py`

- **Ridge**: scikit-learn Ridge, alpha log-grid 탐색
- **Walk-forward**: 90일 train → 14일 purged validation (60분 embargo) → test
- y20, y60 각각 별도 모델 학습
- Score 결합: `score = 0.7 * ŷ20 + 0.3 * ŷ60`
- **Baselines**: Zero (항상 0), Simple rule (z-score+RSI+VWAP gate), OLS, Lasso, ElasticNet
- 목표 지표: validation rank IC, cost-aware spread
- 재학습 주기: 1일
- 주 1회 coefficient stability / feature block contribution 리포트
- Tests: walk-forward correctness, no data leakage, Ridge vs baselines

### Task 8: Portfolio Construction (Long/Short)
**Files**: `src/research/portfolio/construction.py`

- Cross-sectional long/short: 상위 20% long, 하위 20% short + score 부호 일치
- 종목 수: long 6 / short 6 (유니버스 30개의 상하 20%)
- 청산: 상위/하위 35% 밖 이탈 또는 veto (히스테리시스)
- 가중치: `clip(score_z, ±2) / rvol_1d`, gross 1.0 기준 정규화
- Single-name cap 12%, sector soft cap 30%
- BTC beta cap: `abs(portfolio beta) ≤ 0.10` (BTC perp overlay hedge)
- Net exposure cap: `abs(net) ≤ 0.15`
- 리밸런싱: 20분 주기
- Tests: weight normalization, cap enforcement, beta neutrality

### Task 9: Execution & Post-fill Validation
**Files**: `src/research/execution/passive.py`, `postfill.py`

**Passive Execution**:
- 신규 진입: best bid/ask에 passive limit
- 재호가: 60초마다, 최대 2회, 1 tick 이동
- 미체결: 3분 경과 시 취소
- 긴급 청산만 taker 허용

**Post-fill 5분 검증**:
- signed residual return ≤ 0 → 50% 축소
- signed residual return < -0.25 × residual_vol_1d → 100% 청산
- spread > 2 × rolling median 또는 OB imbalance 급변 → 신규 취소 + 기존 축소
- score 부호 반전 → 다음 사이클 전량 청산 검토
- Tests: fill simulation, post-fill trigger conditions

### Task 10: Backtest Engine & Analytics
**Files**: `src/research/backtest/engine.py`, `fast.py`, `accurate.py`, `src/research/analytics/metrics.py`, `report.py`

**Fast Mode**: touch-based simplified maker fill → rank IC, spread, turnover, beta drift
**Accurate Mode**: 1분 L1 snapshot 기반 passive fill proxy → maker fill ratio, realized slippage, post-fill failure rate

**공통 규칙**:
- Walk-forward: train → validate → test 순서 엄격 유지
- 비용: maker_fee_bps, taker_fee_bps, funding_realized config 분리
- 유니버스 재구성 전후 turnover 별도 보고
- 누수 방지: t 시점 feature는 t 이후 데이터 절대 사용 금지

**Analytics**:
- Rank IC (information coefficient)
- Net long-short Sharpe (비용 포함)
- BTC beta drift
- Turnover, maker fill ratio, realized slippage
- 리포트 분할: 전체 / bull-trend / bear-stress / chop 구간별
- Go/No-Go 기준 평가 (PRD Section 10)

**Entry point**: `scripts/run_research_backtest.py`
- Tests: end-to-end fast mode, accurate mode correctness

---

## Dependencies to Add

```toml
# pyproject.toml additions
dependencies = [
    # ... existing ...
    "scikit-learn>=1.5",     # Ridge, Lasso, ElasticNet
    "statsmodels>=0.14",     # OLS for beta estimation
    "scipy>=1.14",           # Statistical functions
]
```

---

## Execution Order

```
Task 1 (Config)
    ↓
Task 2 (Data) ──→ Task 3 (Universe)
    ↓                    ↓
Task 4 (Labels) ←────────┘
    ↓
Task 5 (Features) → Task 6 (Preprocessing)
    ↓                       ↓
Task 7 (Model) ←────────────┘
    ↓
Task 8 (Portfolio) → Task 9 (Execution)
    ↓                       ↓
Task 10 (Backtest & Analytics) ←──┘
```

Tasks 1-3 can partially overlap. Tasks 5-6 can run in parallel with Task 4. Tasks 8-9 can proceed together.

---

## Go/No-Go Criteria (from PRD Section 10)

| 영역 | 기준 | 판정 |
|---|---|---|
| 데이터 | BTC/ETH coverage 99%+, 알트 97%+, 누수 0건 | 필수 |
| 알파 | Ridge OOS rank IC > 0, simple rule/zero 대비 우위 | 필수 |
| 성과 | net long-short Sharpe ≥ 0.8 (비용 포함) | 권고 |
| 순수성 | broad universe 대비 beta drift/turnover 안정성 개선 | 필수 |
| 리스크 | abs(BTC beta) ≤ 0.10, single-name/sector cap 준수 | 필수 |
| 실행 | maker fill ratio ≥ 60% | 권고 |
| 5분 규칙 | MDD/realized slippage 개선 | 권고 |
| 안정성 | 3개 하위 구간 중 최소 2개에서 net PnL 양수 | 필수 |
