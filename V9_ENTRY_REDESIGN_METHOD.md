# V9 KOSPI 진입 구조 재설계

이 실험은 V6 장기 백테스트에서 확인된 약점을 진입부부터 다시 검증합니다. V6를 삭제하거나 실전 스캐너를 바꾸지 않습니다. `v6_baseline`을 그대로 재현한 뒤 새 조건을 단계별로 비교합니다.

## 바꾸는 이유

1996~2024년 V6는 1,129건에서 거래당 평균 순수익 -0.50%, KOSPI 대비 -0.45%p였습니다. RS만 쓴 경우보다 눌림 setup은 손실 폭을 줄였지만, 3일 고가 돌파는 추가 기여가 작았습니다. 진입 후 5~10일 성과가 가장 약했고 다음 시가 +3% 초과 갭 사례도 부진했습니다. 따라서 exit이나 손절보다 entry timing, RS 정의, 추세 품질을 먼저 다시 봅니다.

## 고정 기준선

V6는 아래를 그대로 유지합니다.

- KOSPI, ADV20 20억원 이상, 당시 원주가 1,000원 이상
- 종가 > SMA120, SMA20 > SMA60 > SMA120
- SMA20 및 SMA60 상승
- 종가가 SMA20 대비 -3%~+4%
- 60일 고점 대비 -16%~-2%
- RSI 40~72
- 하락일 평균 거래량이 상승일 평균 거래량의 1.1배 이하
- RS20 > 0, RS20 percentile >= 50%, RS60 >= 60%, RS120 >= 70%
- 주가/KOSPI RS ratio 5일 기울기 > 0
- 종가가 직전 3일 고가 돌파, 종가 > SMA5, SMA5 상승
- 신호 다음 시가 진입, 20번째 관측일 종가 청산, 왕복 비용 0.4%, 종목별 20세션 cooldown

`v6_baseline`은 2025년 이전 완료 거래 1,129건을 재현해야 합니다. 재현 실패 시 V9 결과를 채택하지 않습니다.

## 새 진입 후보

트리거는 V6 setup을 유지한 채 별도로 비교합니다.

- `diag_setup_no_trigger`: setup만으로 진입 시점 자체를 진단
- `diag_prev_high`: 전일 고가 돌파
- `diag_ma5_reclaim`: 전일 종가가 SMA5 이하였고 당일 SMA5 재돌파, SMA5 상승
- `diag_ma20_reclaim`: 전일 종가가 SMA20 이하였고 당일 SMA20 재돌파, SMA20 상승

V9의 주 후보는 `ma5_reclaim`입니다. V6의 3일 고가 돌파보다 빠른 재상승 확인을 의도합니다.

## RS 재설계

V6의 RS20 percentile 하한을 새 모델에서는 제거합니다. 장기 리더십은 유지합니다.

- RS60 percentile >= 60%
- RS120 percentile >= 70%
- RS20 절대 초과수익 > 0
- 현재 5일 RS ratio 기울기 > 0
- 현재 5일 RS ratio 기울기가 직전 5일 기울기보다 개선

즉 장기적으로 강하지만 단기 RS가 다시 가속하는 종목을 찾습니다. `diag_longrs_ma5`와 `diag_reaccel_ma5`를 따로 두어 RS20 percentile 제거 효과와 재가속 효과를 분리합니다.

## V9 core

`v9_core`는 아래를 모두 요구합니다.

- 기존 V6 setup
- 새 RS reacceleration
- SMA5 reclaim
- SMA120의 20거래일 기울기 > 0
- KOSPI bear regime이 아님
- 다음 시가가 신호 종가 대비 +3%를 초과하면 주문 취소

+3% 갭 제한은 신호 생성 규칙이 아니라 다음 시가에서 관찰하는 주문 실행 규칙입니다. 취소된 주문은 cooldown을 발생시키지 않습니다.

## 품질 오버레이

`v9_core_dryup`은 눌림 전 5일 평균 거래량이 직전 20일 평균의 85% 이하이고 하락일 거래량도 상승일 거래량의 90% 이하인 경우만 허용합니다.

`v9_core_contraction`은 신호 전 Bollinger width가 자체 120일 하위 30%이거나 5/20/60일선 spread가 자체 120일 하위 40%인 경우만 허용합니다.

`v9_full_quality`은 dry-up 또는 contraction 중 하나 이상을 요구합니다. 둘 다 강제로 요구해 표본을 과도하게 줄이지 않습니다.

## 평가

1996~2024 전체 결과 외에 아래 창을 따로 냅니다.

- development: 1996~2012
- validation: 2013~2019
- recent pre-2025: 2020~2024
- 2025~2026: 이미 여러 번 본 재사용 평가 구간

이 세 구간은 이제 모두 관측된 과거이므로 진정한 out-of-sample이라고 부르지 않습니다.

`v9_core` 자동 채택 조건은 다음과 같습니다.

- development 평균 초과수익 > 0
- validation 평균 순수익 및 초과수익 > 0, 완료 거래 50건 이상
- 2020~2024 평균 순수익 및 초과수익 > 0, 완료 거래 40건 이상

하나라도 실패하면 V6를 연구 기준선으로 유지합니다.

## 출력

- `v9_all_trades.csv.gz`
- `v9_signal_funnel.csv`
- `v9_pre2025_comparison.csv`
- `v9_periods.csv`
- `v9_research_windows.csv`
- `v9_yearly.csv`
- `v9_adoption_gate.csv`
- `v9_portfolio_pre2025.csv`
- V6, V9 core, V9 full-quality의 일별 equity curve
- `manifest.json`

실전 스캐너와 `main.py/scanner.py`는 이 연구에서 변경하지 않습니다.
