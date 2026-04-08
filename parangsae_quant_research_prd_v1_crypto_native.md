# Parangsae Quant Research PRD v1

**Crypto-Native Residual Ridge · Backtest-Ready Spec**  
**Research PRD / Backtest Specification**  
**v1 · 2026-04-08**

## 한 페이지 요약

- 코어 유니버스는 native crypto risk asset만 사용한다. 달러·금·채권·주식 등 외부 기준자산에 anchor된 토큰과 피처는 제외한다.
- 예측 대상은 raw return이 아니라 BTC beta-adjusted residual return이다. 즉 BTC 공통 움직임을 제거한 뒤 알트의 상대 강약을 맞힌다.
- MVP 모델은 Ridge/L2다. feature selection은 하지 않고, crypto-native 약한 신호를 다수 결합한다.
- 실행은 1호가 passive quote를 기본으로 하고, filled 후 5분 안에 follow-through가 없으면 축소 또는 청산한다.
- 문서 목적은 전략 설명이 아니라 즉시 백테스트 가능한 구현 사양을 고정하는 것이다.

## 1. 제품 정의

이 문서는 v4 전략 문서를 실제 연구·백테스트·리포팅 파이프라인으로 옮기기 위한 PRD다. 핵심 질문은 하나다.

BTC 공통 요인을 제거하고도, crypto-native 피처만으로 알트의 단기 residual return을 예측할 수 있는가?

MVP는 해석 가능하고 운영 가능한 구조를 우선한다. 따라서 외부 거시 변수, RWA 토큰, 복잡한 딥러닝, 과도한 파라미터 최적화, discretionary override는 범위 밖으로 둔다.

| 항목 | 정의 |
|---|---|
| 제품 목표 | crypto-only 상대가치 long/short 전략의 백테스트 가능 MVP 구축 |
| 핵심 가설 | 약한 신호 환경에서는 BTC residual label + Ridge/L2 + 단순 execution rule 조합이 sparse selection보다 낫다 |
| 주요 시장 | Binance perpetual 선물 (초기 production은 USDT-M, 엄격 순수성은 COIN-M 연구 트랙으로 분리) |
| 주요 의사결정 주기 | 5분 연구 바, 20분 리밸런싱, 5분 post-fill 검증 |
| 비범위 | gold / dollar / bond / stock-linked token, 외부 macro feature, deep model 우선주의, 온체인 heavy feature |

## 2. 유니버스 규칙

유니버스는 '크립토 내부 수급과 내러티브가 가격결정의 1차 기준점인 자산'으로 제한한다. 실행 편의 때문에 USDT-M을 사용하더라도, USDT는 알파 입력이 아니라 결제 단위로만 본다.

| 구분 | 포함 / 제외 규칙 | MVP 기본값 |
|---|---|---|
| 포함 | native crypto risk asset perpetual | BTC, ETH, 주요 L1/L2, DeFi, 인프라, 밈 중 유동성 충족 종목 |
| 제외 | 외부 anchor 자산 | stablecoin, tokenized gold, tokenized bond/T-bill, tokenized stock/ETF, FX proxy, leveraged/inverse token |
| 상장 이력 | 충분한 beta 추정 길이 필요 | 180일 이상 |
| 유동성 | 최근 30일 median 24h quote volume 기준 상위 종목 유지 | 상위 30개 내외 |
| 시장 마이크로구조 | 스프레드·깊이·OI·funding 데이터 필요 | funding/OI coverage ≥ 95%, 과도한 스프레드 종목 제외 |
| 재구성 주기 | 유니버스 고정과 교체의 균형 | 주 1회 |

## 3. 데이터 사양

모든 피처는 t 시점에 완료된 마지막 5분 바까지의 정보만 사용한다. label은 t 이후 구간에서 계산한다. 데이터는 point-in-time alignment를 강제하며, 거래 가능 가격과 학습 label을 분리해 관리한다.

| 데이터셋 | 해상도 | 용도 | 비고 |
|---|---|---|---|
| OHLCV | 1분 | 기본 가격·거래량·변동성·VWAP | 5분 연구 바와 20분 의사결정 바로 집계 |
| Mark / Index Price | 1분 | basis·label·리스크 진단 | last trade 노이즈 분리 목적 |
| Funding Rate | 거래소 제공 주기 | crowding / carry | 최근 값 보간 후 bar에 정렬 |
| Open Interest | 1분 또는 5분 | 포지션 유입·청산 | 결측 구간은 종목 제외 판정에 반영 |
| Top-of-Book Bid/Ask | 1분 snapshot | spread / depth / passive fill | 초기 정확 backtest용 핵심 |
| Contract Metadata | 이벤트성 | tick size / lot size / fee schema | 실행기와 비용 모듈에서 사용 |
| Funding Realization | 실현 값 | PnL 반영 | 예측용 피처와 분리 저장 |

## 4. 라벨과 비교 기준

MVP의 중심 라벨은 20분 residual return이다. 보조 라벨은 60분 residual return이다. 5분 horizon은 alpha label이 아니라 execution validation용으로만 사용한다.

### 수식 및 점수 정의

```text
r_i(t,h) = log(Mid_i(t+h) / Mid_i(t))
beta_i,t = OLS( 최근 1주·2주 5분 수익률에서 r_i ~ r_BTC )
y20_i,t = r_i(t,20m) - beta_i,t * r_BTC(t,20m)
y60_i,t = r_i(t,60m) - beta_i,t * r_BTC(t,60m)
score_i,t = 0.7 * ŷ20_i,t + 0.3 * ŷ60_i,t
```

| 구분 | 정의 | 역할 |
|---|---|---|
| Zero benchmark | 항상 0 예측 | weak-signal 환경에서 반드시 이겨야 하는 기준선 |
| Simple rule baseline | beta-adjusted residual z-score + RSI/VWAP gate | 사용자 힌트 기반의 단순 기술지표 비교군 |
| OLS | 무정규화 선형회귀 | 과적합 비교군 |
| Lasso | L1 penalty | sparse selection 비교군 |
| Elastic Net | high-L2 세팅 | Ridge 인접 비교군 |
| Ridge | MVP 기본 학습기 | production 기본값 |

## 5. 피처 사양 (MVP 40개)

MVP는 40개 crypto-native feature로 시작한다. 숫자를 억지로 90개 이상으로 늘리지 않는다. 다만 전부 크립토 내부 변수로만 구성하고, feature selection은 하지 않는다.

단순 기술지표는 독립 알파가 아니라 residual 중심 점수의 modifier 또는 gate로만 사용한다. 파라미터는 14, 20, 24, 48 같은 표준값만 허용한다.

| 블록 | 개수 | 피처 | 역할 |
|---|---|---|---|
| Relative / Beta | 8 | beta_btc_1w, beta_btc_2w, resid_mom_1h, resid_mom_4h, resid_mom_1d, resid_z_1h, resid_z_4h, resid_rsi_14 | BTC 공통요인 제거 이후 상대 강도 측정 |
| Trend / Path | 8 | mom_1h, mom_4h, mom_1d, mom_4h_voladj, mom_1d_voladj, fip_like_1d, trend_linearity_1d, wickiness_1d | 좋은 추세와 점프성 추세 분리 |
| Liquidity / Execution | 8 | qvol_z_1d, qvol_z_7d, amihud_1d, vwap_dev_1h, spread_bps, spread_z_1d, depth_top1_usd, ob_imbalance_1m | 실행 가능성과 정보성 동시 반영 |
| Derivatives / Crowding | 8 | funding, funding_z_7d, funding_change_1d, oi_change_1h, oi_change_1d, oi_to_vol, basis, basis_z_7d | 과열, carry, 포지션 편중 포착 |
| Risk / State | 8 | rvol_1h, rvol_1d, downside_vol_1d, corr_btc_1d, btc_trend_4h, alt_breadth_4h, median_funding_breadth, xs_corr_1d | gross·veto·regime 진단 |

## 6. 모델 사양

모델은 panel 형태의 시계열-단면 회귀다. 각 의사결정 시점에서 직전 90일 데이터를 사용해 Ridge를 학습하고, 14일 purged validation으로 alpha를 선택한다. 재학습 주기는 1일이다.

입력 전처리는 단순해야 한다. 시점별 cross-sectional z-score, ±5 clip, 결측값의 단면 중앙값 대체만 허용한다. feature selection, 복잡한 scaling stack, deep interaction learner는 MVP 밖이다.

| 항목 | MVP 기본값 |
|---|---|
| 학습 샘플 | 최근 90일 panel observations |
| 검증 샘플 | 최근 14일 purged validation, 60분 embargo |
| 재학습 주기 | 1일 |
| 정규화 | 시점별 cross-sectional z-score + clip ±5 |
| 결측 처리 | 시점별 단면 중앙값 대체, coverage 미달 종목은 유니버스 제외 |
| 하이퍼파라미터 | Ridge alpha: log-grid 탐색, 목표지표는 validation rank IC와 cost-aware spread |
| 모델 수 | y20, y60용 2개 모델 학습 후 가중 결합 |
| 해석 리포트 | 주 1회 coefficient stability / feature block contribution 리포트 생성 |

## 7. 포트폴리오 구성

포트폴리오는 cross-sectional long/short 구조다. 목표는 '큰 베팅'이 아니라 residual relative value를 작고 반복 가능하게 회수하는 것이다. beta-neutral이 1차, dollar-neutral은 2차다.

| 항목 | MVP 기본값 | 비고 |
|---|---|---|
| 리밸런싱 | 20분 | 5분 bar를 4개 묶은 의사결정 주기 |
| 종목 수 | long 6 / short 6 시작 | 유니버스 크기에 따라 상하 20% 사용 |
| 진입 | 상위 20% / 하위 20% + score 부호 일치 | 단순 rank + sign gate |
| 청산 | 상위/하위 35% 밖 이탈 또는 veto 발생 | 히스테리시스 적용 |
| 가중치 | clip(score_z, ±2) / rvol_1d | gross 1.0 기준 정규화 |
| single-name cap | 12% | 유동성 낮은 종목은 더 낮춤 |
| sector soft cap | 30% | crypto sector는 라벨 기반 soft cap만 사용 |
| BTC beta cap | abs(portfolio beta) ≤ 0.10 | 필요 시 BTC perp overlay hedge |
| net exposure cap | abs(net) ≤ 0.15 | 상대가치 성격 유지 |

## 8. 실행 사양

실행 규칙은 사용자 힌트를 그대로 MVP에 반영한다. 즉 1호가 단위 maker-first, filled 후 5분 검증, 과도한 quote chasing 금지다.

새 진입은 spread를 crossing하지 않는다. 리스크 청산이나 kill-switch 상황만 예외로 둔다.

| 단계 | 규칙 |
|---|---|
| 신규 진입 | 매수는 best bid, 매도는 best ask에 passive limit 주문 |
| 재호가 | 60초마다 한 번, 최대 2회, 매번 1 tick만 이동 |
| 미체결 | 3분 경과 시 취소하고 다음 사이클에서 재판단 |
| 부분체결 | 잔여 수량은 동일 규칙 유지, all-in/out 금지 |
| 긴급 청산 | DD breach, mark/index 괴리 급등, 거래소 이상 시 taker 허용 |

### Post-fill 5분 검증 규칙

| 조건 | 조치 |
|---|---|
| filled 후 5분 signed residual return ≤ 0 | 포지션 50% 축소 |
| filled 후 5분 signed residual return < -0.25 × residual_vol_1d | 포지션 100% 청산 |
| spread > 2 × rolling median 또는 order book imbalance 급변 | 신규 진입 취소, 기존 포지션 축소 |
| score 부호 반전 | 다음 사이클에서 전량 청산 우선 검토 |

## 9. 백테스트 엔진 규칙

백테스트는 두 단계로 나눈다. Fast mode는 연구 반복 속도를 우선하고, Accurate mode는 execution 가정을 검증한다. 결과 보고서는 두 모드를 항상 나란히 제시한다.

| 모드 | 목적 | fill 가정 | 필수 출력 |
|---|---|---|---|
| Fast | feature / model iteration | touch-based simplified maker fill | rank IC, spread, turnover, beta drift |
| Accurate | 실행 가정 검증 | 1분 L1 snapshot 기반 passive fill proxy | maker fill ratio, realized slippage, post-fill failure rate |

### 공통 엔진 규칙

| 항목 | 규칙 |
|---|---|
| walk-forward | train → validate → test 순서를 엄격히 유지 |
| 비용 | maker_fee_bps, taker_fee_bps, funding_realized를 config로 분리 |
| 유니버스 재구성 | 주 1회, 재구성 전후 turnover 별도 보고 |
| 누수 방지 | t 시점 feature는 t 이후 체결·funding·OI 변화를 절대 사용 금지 |
| 리포트 분할 | 전체, bull/trend, bear/stress, chop 구간별 성과 별도 산출 |

## 10. 성공 기준 / Go-No-Go

아래 기준은 첫 번째 production gate다. 수치는 초기값이므로 1차 분포를 확인한 뒤 조정할 수 있다. 중요한 것은 'Ridge가 zero와 simple rule을 넘어서는가'와 'pure universe가 더 깨끗한가'다.

| 영역 | 기준 | 판정 |
|---|---|---|
| 데이터 | BTC/ETH coverage 99%+, 알트 97%+, 누수 0건 | 필수 |
| 알파 | Ridge OOS rank IC > 0 and simple rule / zero 대비 우위 | 필수 |
| 성과 | net long-short Sharpe ≥ 0.8 (비용 포함) | 권고 |
| 순수성 | broad universe 대비 beta drift 또는 turnover 안정성 개선 | 필수 |
| 리스크 | abs(BTC beta) ≤ 0.10, single-name/sector cap 준수 | 필수 |
| 실행 | maker fill ratio ≥ 60% 또는 동일 Sharpe 대비 비용 절감 입증 | 권고 |
| 5분 규칙 | MDD 또는 realized slippage 개선 | 권고 |
| 안정성 | 주요 3개 하위 구간 중 최소 2개 구간에서 net PnL 양수 | 필수 |

## 11. 빌드 로드맵과 산출물

문서의 목적은 백테스트 시작점을 모호하지 않게 만드는 것이다. 따라서 구현 산출물도 명시한다.

| 주차 | 작업 | 산출물 |
|---|---|---|
| 1주차 | data ingestion + universe filter + contract metadata 정리 | universe_history.parquet, contract_specs.yaml |
| 2주차 | label / feature pipeline 구축 | feature_manifest.csv, labels_residual.parquet |
| 3주차 | ridge training + walk-forward backtest | model_config.yaml, backtest_fast_report |
| 4주차 | accurate execution mode + 5분 validation rule | execution_report, cost_decomposition |
| 5주차 | benchmark comparison + go/no-go review | research memo, parameter freeze v1 |

### 권장 리포지토리 구조

| 경로 | 내용 |
|---|---|
| config/universe.yaml | 포함/제외 규칙, 유동성 threshold, sector map |
| config/model.yaml | train window, validation window, alpha grid, weight caps |
| features/manifest.csv | 40개 MVP feature 정의와 계산 함수 매핑 |
| labels/residual.py | beta 추정과 y20 / y60 생성 |
| models/ridge.py | 학습·검증·예측 파이프라인 |
| backtest/executor.py | fast / accurate fill engine |
| analytics/report.ipynb | 성과·비용·안정성 리포트 |

## 12. Phase 2 이후에만 허용할 확장

다음 항목은 MVP가 통과된 뒤에만 연구한다. 처음부터 넣지 않는다.

| 확장 항목 | 조건 |
|---|---|
| shallow NN + L2 | Ridge 대비 추가 이득이 비용 이후에도 확인될 때 |
| Random Forest residual learner | interaction benefit 가설을 별도 검증할 때 |
| sector residual label | 기본 BTC residual이 안정화된 뒤 |
| COIN-M collateral 트랙 | 실행 커버리지와 담보 리스크를 분리 평가할 때 |
| 온체인 / liquidation data | 기본 microstructure feature가 안정화된 뒤 |
| state-based gross allocator | static gross book이 재현 가능하게 나온 뒤 |

## Appendix A. 초기 고정 파라미터

| 항목 | 기본값 |
|---|---|
| Research bar | 5분 |
| Decision bar | 20분 |
| Primary label | 20분 BTC residual return |
| Secondary label | 60분 BTC residual return |
| Beta window | 1주, 2주 |
| Train window | 90일 |
| Validation window | 14일 |
| Embargo | 60분 |
| Retrain cadence | 1일 |
| Universe size | 유동성 상위 30개 내외 |
| Entry / Exit bucket | 20% / 35% |
| Gross | 1.0x 시작 |
| Net cap | ±0.15 |
| Single-name cap | 12% |
| Sector soft cap | 30% |

## Appendix B. 참고 문헌

- Wesley R. Gray, Jack R. Vogel, Quantitative Momentum
- Bill Jiang, Investment Strategies
- Shihao Gu, Bryan Kelly, Dacheng Xiu, Empirical Asset Pricing via Machine Learning
- Zhouyu Shen, Dacheng Xiu, Can Machines Learn Weak Signals?
- Ridge Ensemble Strategy 초안 (사용자 제공)