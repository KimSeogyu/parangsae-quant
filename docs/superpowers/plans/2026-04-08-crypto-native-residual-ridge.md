# Crypto-Native Residual Ridge Implementation Plan

**PRD**: `parangsae_quant_research_prd_v1_crypto_native.md`
**Date**: 2026-04-08
**Status**: MVP executable — manifest-driven runner wired, production-grade expansion pending

---

## Implementation Progress

### Overall Status

| Phase | Status | Tests | Notes |
|---|---|---|---|
| **Task 1**: Config & Project Setup | ✅ Complete | 8/8 | Pydantic models, YAML configs, manifest.csv |
| **Task 2**: Extended Data Fetcher | ✅ Complete | 13/13 | 1m OHLCV + funding + OI + mark_price + orderbook_top1 |
| **Task 3**: Universe Filter | ✅ Complete | 11/11 | 30 coins, crypto-native only, weekly reconstitution |
| **Task 4**: Label Pipeline | ✅ Complete | 10/10 | Rolling OLS beta, y20/y60 residual returns |
| **Task 5**: Feature Pipeline | ✅ Complete | 32/32 | 40 features × 5 blocks, registry |
| **Task 6**: Feature Preprocessing | ✅ Complete | (included above) | z-score, ±5 clip, median imputation |
| **Task 7**: Model Pipeline | ✅ Complete | 13/13 | Ridge + walk-forward + 5 baselines |
| **Task 8**: Portfolio Construction | ✅ Complete | 12/12 | L/S 6+6, beta-neutral, caps |
| **Task 9**: Execution & Post-fill | ✅ Complete | 17/17 | Passive limit, 5m validation, deterministic execution path |
| **Task 10**: Backtest & Analytics | ✅ Complete | 19/19 | Fast/accurate modes, Go/No-Go, real CLI entrypoint |
| **Total** | **10/10 tasks** | **135/135** | **lint clean** |

### What's Done (Code)

모든 핵심 모듈의 순수 함수(pure function) 구현과 단위 테스트가 완료됨.
실행 가능한 CLI/backtest runner가 추가됐고, manifest 기반의 broader feature wiring이 연결된 상태다.
새 코드와 테스트가 `src/research/` 및 `tests/research/` 하위에 추가됐다.
`tests/research`는 현재 135개 테스트를 통과한다.

### What's NOT Done Yet (Requires Local Data)

다음 작업들은 실제 Binance 데이터와 API 키가 필요하여 로컬에서 수행해야 함:

1. **데이터 수집**: `scripts/fetch_research_data.py` 실행
   - 1m OHLCV, funding rate, open interest, mark price, top-of-book 수집
   - 예상 저장 경로: `data/research/{ohlcv_1m, ohlcv_5m, ohlcv_20m, funding, oi, mark_price, orderbook_top1}/`

2. **End-to-end 파이프라인 확장**: `run_research_backtest.py`는 실제 CLI로 동작하고 manifest-driven feature assembly를 사용한다
   - 데이터 로딩 → manifest 기반 feature 계산 → label 생성 → walk-forward Ridge → portfolio sim → report
   - 실데이터 기준 커버리지/품질 튜닝과 PRD 수준 fidelity 검증은 다음 단계

3. **Backtest engine 확장**: `engine.py`는 더 이상 스캐폴딩이 아니며 deterministic execution path를 사용한다
   - 현재는 runner가 제공하는 parquet-derived 입력 기준으로 동작
   - PRD 전체 시뮬레이션 fidelity는 추가 확장 필요

4. **Jupyter 리포트 노트북**: PRD에서 권장하는 `analytics/report.ipynb` (성과 시각화, feature block contribution, coefficient stability)

5. **universe_history.parquet 생성**: 주간 재구성 이력 저장

---

## Gap Analysis

### Current System vs PRD Requirements

| 영역 | 기존 시스템 (`src/`) | PRD 연구 파이프라인 (`src/research/`) | 변경 규모 |
|---|---|---|---|
| **전략 패러다임** | Rule-based momentum+lowvol | ML (Ridge) residual prediction | 완전 재설계 |
| **타임프레임** | 1H bars | 5m research / 20m decision | 신규 |
| **알파 생성** | 2개 팩터 (모멘텀, 저변동성) | 40개 feature → Ridge regression | 신규 |
| **예측 대상** | Raw return 기반 ranking | BTC beta-adjusted residual return | 신규 |
| **포트폴리오** | Long-only 15종목 | Long/Short 6+6, beta-neutral | 대폭 수정 |
| **유니버스** | 100종목, hourly ranking | 30종목, weekly, 180d+, funding/OI 커버리지 | 대폭 수정 |
| **실행** | Market order (IOC) | Passive limit + 5m post-fill 검증 | 신규 |
| **백테스트** | 단일 모드 | Fast + Accurate 2단계 | 신규 |
| **데이터** | OHLCV only | OHLCV + Funding + OI + Orderbook + Mark Price | 대폭 확장 |
| **학습** | 없음 (rule-based) | Walk-forward Ridge, 90d train, 14d validation | 신규 |

### Architecture Decision

기존 `src/` NautilusTrader 모듈은 유지하고, PRD 연구 파이프라인을 `src/research/` 하위에 별도 구축.
추후 Ridge 모델의 예측 결과를 NautilusTrader Actor로 감싸서 실행 엔진과 통합 가능.

---

## Module Structure

```
src/research/
├── __init__.py
├── config.py                          # Pydantic config (universe + model settings)
├── universe/
│   └── filter.py                      # Crypto-native universe filter (5 stages)
├── data/
│   ├── fetcher.py                     # CCXT: 1m OHLCV, funding, OI
│   ├── aggregator.py                  # 1m → 5m → 20m aggregation, mid/VWAP/log returns
│   └── alignment.py                   # Point-in-time, embargo split, walk-forward windows
├── labels/
│   ├── beta.py                        # Rolling OLS beta (1w/2w dual-window)
│   └── residual.py                    # y20/y60 residual return, score combination
├── features/
│   ├── registry.py                    # manifest.csv → FeatureSpec lookup
│   ├── relative.py                    # Block 1 (8): beta, resid_mom, resid_z, resid_rsi
│   ├── trend.py                       # Block 2 (8): mom, voladj, fip, linearity, wickiness
│   ├── liquidity.py                   # Block 3 (8): qvol_z, amihud, vwap_dev, spread, depth, ob_imbalance
│   ├── derivatives.py                 # Block 4 (8): funding, oi_change, oi_to_vol, basis
│   ├── risk_features.py               # Block 5 (8): rvol, downside_vol, corr_btc, btc_trend, breadth, xs_corr
│   └── preprocess.py                  # Cross-sectional z-score → clip ±5 → median fill
├── models/
│   ├── ridge.py                       # Ridge train (alpha grid → rank IC), feature importance
│   ├── baselines.py                   # Zero, SimpleRule, OLS, Lasso, ElasticNet
│   └── walkforward.py                 # Walk-forward orchestration + summary stats
├── portfolio/
│   └── construction.py                # L/S 6+6, hysteresis, weight, caps, beta hedge, net cap
├── execution/
│   ├── passive.py                     # Passive limit fill sim (requote, cancel)
│   └── postfill.py                    # 5m validation (reduce/close/cancel/queue_close)
├── backtest/
│   ├── engine.py                      # run_fast_backtest / run_accurate_backtest
│   ├── fast.py                        # Touch-based instant fill utilities
│   └── accurate.py                    # L1 snapshot passive fill utilities
└── analytics/
    ├── metrics.py                     # Sharpe, MDD, turnover, beta drift, regime classification
    └── report.py                      # BacktestReport generation + Go/No-Go evaluation

config/
├── universe.yaml                      # Inclusion/exclusion rules, sector map
└── model.yaml                         # All PRD Appendix A parameters

features/
└── manifest.csv                       # 40 feature definitions (name, block, function, params)

scripts/
├── fetch_research_data.py             # Data collection entry point
└── run_research_backtest.py           # Backtest entry point

tests/research/                        # 135 tests across 11 test files
├── test_config.py                     # Config parsing, validation (8 tests)
├── test_data.py                       # Aggregation, alignment, walk-forward (9 tests)
├── test_universe_filter.py            # All filter stages + reconstitution (11 tests)
├── test_labels.py                     # Beta estimation, residual returns (10 tests)
├── test_features.py                   # All 5 blocks + preprocessing + registry (32 tests)
├── test_models.py                     # Ridge, baselines, walk-forward (13 tests)
├── test_portfolio_ls.py               # L/S selection, weights, caps, hedge (12 tests)
├── test_execution.py                  # Passive fill, slippage, post-fill (17 tests)
├── test_analytics.py                  # Metrics, regime, Go/No-Go (11 tests)
├── test_backtest_engine.py            # Deterministic fast/accurate engine coverage
└── test_runner.py                     # Manifest-driven runner / CLI input prep
```

---

## PRD Coverage Matrix

PRD 각 섹션이 코드의 어디에 매핑되는지:

| PRD Section | 코드 위치 | 구현 상태 |
|---|---|---|
| §1. 제품 정의 | — | N/A (문서) |
| §2. 유니버스 규칙 | `universe/filter.py`, `config/universe.yaml` | ✅ 전체 구현 |
| §3. 데이터 사양 | `data/fetcher.py`, `data/aggregator.py`, `data/alignment.py` | ✅ 구조 구현 + mark_price/orderbook_top1 저장 |
| §4. 라벨과 비교 기준 | `labels/beta.py`, `labels/residual.py`, `models/baselines.py` | ✅ 전체 구현 |
| §5. 피처 사양 (40개) | `features/{relative,trend,liquidity,derivatives,risk_features}.py` | ✅ 40/40 구현 |
| §6. 모델 사양 | `models/ridge.py`, `models/walkforward.py`, `features/preprocess.py` | ✅ 전체 구현 |
| §7. 포트폴리오 구성 | `portfolio/construction.py` | ✅ 전체 구현 |
| §8. 실행 사양 | `execution/passive.py`, `execution/postfill.py` | ✅ 전체 구현 |
| §9. 백테스트 엔진 규칙 | `backtest/engine.py`, `backtest/fast.py`, `backtest/accurate.py` | ✅ deterministic fast/accurate modes |
| §10. 성공 기준 | `analytics/report.py` (Go/No-Go evaluation) | ✅ 전체 구현 |
| §11. 빌드 로드맵 | 이 문서 | ✅ Task 1-10 완료, CLI runner 연결 |
| §12. Phase 2 확장 | — | ❌ MVP 이후 |
| Appendix A. 파라미터 | `config/model.yaml`, `src/research/config.py` | ✅ 전체 반영 |

---

## Local 실행 가이드

```bash
# 1. 브랜치 가져오기
git fetch origin
git checkout claude/apply-crypto-quant-framework-kBRtM

# 2. 의존성 설치
uv sync

# 3. 테스트 확인
uv run pytest tests/research/ -v

# 4. 데이터 수집 (API 키 필요)
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
uv run python scripts/fetch_research_data.py

# 5. 백테스트 실행
uv run python scripts/run_research_backtest.py --mode fast
```

---

## Next Steps (로컬에서 수행)

1. **데이터 수집 + 검증**: fetch → coverage 확인 → gap fill
2. **End-to-end 파이프라인 glue code**: `run_research_backtest.py`에서 전체 흐름 연결
3. **Fast backtest 실행**: rank IC, Sharpe, turnover 확인
4. **Accurate backtest 실행**: maker fill ratio, slippage 확인
5. **Go/No-Go 판정**: PRD §10 기준에 따라 Ridge vs zero/simple rule 비교
6. **Parameter freeze v1**: 결과에 따라 `config/model.yaml` 확정
7. **Phase 2 결정**: shallow NN, sector residual, COIN-M 트랙 등

---

## Go/No-Go Criteria (PRD §10)

| 영역 | 기준 | 판정 | 코드 위치 |
|---|---|---|---|
| 데이터 | BTC/ETH coverage 99%+, 알트 97%+, 누수 0건 | 필수 | (데이터 수집 후 확인) |
| 알파 | Ridge OOS rank IC > 0, zero/simple rule 대비 우위 | 필수 | `analytics/report.py` → `go_decisions["alpha_ic"]` |
| 성과 | net long-short Sharpe ≥ 0.8 (비용 포함) | 권고 | `go_decisions["sharpe"]` |
| 순수성 | broad universe 대비 beta drift/turnover 안정성 개선 | 필수 | `metrics.py` → `compute_beta_drift()` |
| 리스크 | abs(BTC beta) ≤ 0.10, single-name/sector cap 준수 | 필수 | `go_decisions["beta_cap"]` |
| 실행 | maker fill ratio ≥ 60% | 권고 | `go_decisions["maker_fill"]` |
| 5분 규칙 | MDD/realized slippage 개선 | 권고 | `postfill.py` → `PostfillAction` |
| 안정성 | 3개 하위 구간 중 최소 2개에서 net PnL 양수 | 필수 | `go_decisions["stability"]` |
