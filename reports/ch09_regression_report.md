# Chapter 09 회귀 분석 보고서

예측 대상은 주문 한 건의 주문금액 합계(order_total)이다.
예측 시점은 주문 메타데이터와 고객 정보만 알고 주문상세는 모르는 시점으로 가정했다.

## 데이터 분할
| split   |   rows |   ratio_pct | start_date   | end_date   |
|:--------|-------:|------------:|:-------------|:-----------|
| train   |    242 |        80.7 | 2025-07-09   | 2026-04-08 |
| test    |     58 |        19.3 | 2026-04-10   | 2026-07-08 |

## 훈련 기간 TimeSeriesSplit 결과
| model             |   n_splits |   cv_MAE_mean |   cv_MAE_std |   cv_R2_mean |
|:------------------|-----------:|--------------:|-------------:|-------------:|
| Baseline Mean     |          5 |       95350.5 |      14097.4 |        -0.05 |
| Linear Regression |          5 |      115128   |      12535.5 |        -0.57 |
| Random Forest     |          5 |       99490.6 |      11229.8 |        -0.22 |

비베이스라인 중 cv_MAE_mean이 가장 낮은 **Random Forest** 을 Final Test 전에 고정했다.

## Final Test (Baseline vs 고정 모델)
| model         | selection_role       |   train_MAE |   test_MAE |   test_RMSE |   test_R2 |   MAE_improvement_vs_baseline_pct |
|:--------------|:---------------------|------------:|-----------:|------------:|----------:|----------------------------------:|
| Baseline Mean | baseline             |     95324.1 |     127791 |      156604 |     -0    |                              0    |
| Random Forest | selected_by_train_cv |     74649.5 |     121477 |      160267 |     -0.05 |                              4.94 |

## 해석
- 선택 모델이 Baseline보다 MAE가 4.9% 낮았다.
- R²가 음수다. 테스트 목표값의 평균으로 예측하는 것보다 못했다는 뜻이다. 숨기지 않고 기록한다.
- train MAE와 test MAE 차이: 46,827
- Baseline test MAE: 127,791
- 테스트 결과를 보고 모델을 다시 고르지 않았다.
- 개선이 없다면 운영하지 않는 결정도 올바른 결과라고 책에 나와 있다.

## 자동 Validation
| check                                     |   value | status   | passed   | detail                              |
|:------------------------------------------|--------:|:---------|:---------|:------------------------------------|
| forbidden_feature_overlap                 |       0 | PASS     | True     | []                                  |
| strict_train_before_test                  |    True | PASS     | True     | 2026-04-08 < 2026-04-10             |
| no_shared_calendar_dates                  |       0 | PASS     | True     | 겹치는 날짜 수                      |
| selected_model_exists_in_train_cv         |    True | PASS     | True     | Random Forest                       |
| selected_model_is_not_baseline            |    True | PASS     | True     | Random Forest                       |
| final_test_contains_baseline              |    True | PASS     | True     |                                     |
| final_test_contains_frozen_selected_model |    True | PASS     | True     | Random Forest                       |
| final_test_only_two_models                |       2 | PASS     | True     | ['Baseline Mean', 'Random Forest']  |
| test_rows_for_r2                          |      58 | PASS     | True     | R² 계산에는 test 행이 2개 이상 필요 |
