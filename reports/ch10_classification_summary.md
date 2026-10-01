# Chapter 10 분류 분석 요약 보고서

## 1. 분석 목적과 타깃
완료 주문과 취소 주문만 사용해 주문 취소 여부를 예측합니다.

- completed = 0
- cancelled = 1
- refunded 및 기타 상태 = 모델링 범위 제외

## 2. 모델링 데이터
- 행 수: 203
- 예측 대상: is_cancelled

## 3. 타깃 클래스 분포
```text
 is_cancelled class_label  count  ratio
            0   completed    149  0.734
            1   cancelled     54  0.266
```

## 4. Feature Audit
```text
           column  selected            role                 reason
              age      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
       item_count      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
   total_quantity      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
     order_amount      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
      order_month      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
  order_dayofweek      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
days_since_signup      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
           gender      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
             city      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
   payment_method      True allowed_feature 교육용 예측 시점에 사용 가능하다고 가정
    cancel_reason     False       forbidden       취소 이후 생성되는 사후 정보
     cancelled_at     False       forbidden       취소 이후 생성되는 사후 정보
      customer_id     False       forbidden                 고객 식별자
     is_cancelled     False       forbidden               예측 대상 자체
         order_id     False       forbidden                 주문 식별자
     order_status     False       forbidden   예측 결과와 직접 연결되는 주문 상태
       product_id     False       forbidden            주문 상세 식별 정보
```

## 5. 병합 검증
```text
                         merge  before_rows  after_rows  row_count_preserved  unmatched_count status
order_id → order_item_features          203         203                 True                0   PASS
       customer_id → customers          203         203                 True                0   PASS
```

## 6. Train / Validation / Test
```text
     split  is_cancelled class_label  count  ratio
     train             0   completed     89 0.7355
     train             1   cancelled     32 0.2645
validation             0   completed     30 0.7317
validation             1   cancelled     11 0.2683
      test             0   completed     30 0.7317
      test             1   cancelled     11 0.2683
```

## 7. Validation 모델 비교
```text
              model evaluation_split  accuracy  precision   recall       f1
Logistic Regression       validation  0.634146   0.400000 0.727273 0.516129
      Random Forest       validation  0.658537   0.285714 0.181818 0.222222
Dummy Most Frequent       validation  0.731707   0.000000 0.000000 0.000000
```

선택 모델: **Logistic Regression**

## 8. Validation Threshold 비교
```text
 threshold  accuracy  precision   recall       f1
      0.20  0.365854   0.297297 1.000000 0.458333
      0.25  0.463415   0.322581 0.909091 0.476190
      0.30  0.463415   0.322581 0.909091 0.476190
      0.35  0.512195   0.333333 0.818182 0.473684
      0.40  0.560976   0.360000 0.818182 0.500000
      0.45  0.658537   0.428571 0.818182 0.562500
      0.50  0.634146   0.400000 0.727273 0.516129
      0.55  0.682927   0.444444 0.727273 0.551724
      0.60  0.780488   0.583333 0.636364 0.608696
      0.65  0.756098   0.555556 0.454545 0.500000
      0.70  0.804878   0.800000 0.363636 0.500000
      0.75  0.804878   0.800000 0.363636 0.500000
      0.80  0.756098   0.666667 0.181818 0.285714
```

선택 threshold: **0.60**

## 9. Final Test
```text
evaluation_split  threshold   selection_status  accuracy  precision   recall       f1
            test        0.6 frozen_before_test  0.682927        0.4 0.363636 0.380952
```

## 10. Confusion Matrix
```text
                  pred_completed  pred_cancelled
actual_completed              24               6
actual_cancelled               7               4
```

## 11. 자동 Validation Evidence
```text
                               check  value status
  target_exactly_completed_cancelled [0, 1]   PASS
           forbidden_feature_overlap      0   PASS
              merge_rows_and_matches   True   PASS
        all_splits_have_both_classes   True   PASS
        model_selected_on_validation   True   PASS
    threshold_selected_on_validation   True   PASS
public_prediction_identifier_columns   none   PASS
```

## 12. 사람 검토 체크리스트
```text
                                        check_item status
           completed와 cancelled만 사용해 이진 타깃을 만들었는가?      □
                 refunded 등 다른 상태를 0 클래스에 섞지 않았는가?      □
                 예측 시점과 feature 가용성 가정을 설명할 수 있는가?      □
        line_total = quantity × unit_price를 검증했는가?      □
                 병합에 validate를 사용하고 미매칭 0건을 확인했는가?      □
 order_status, target, ID, 사후 정보를 feature에서 제외했는가?      □
                   train, validation, test를 분리했는가?      □
                모델과 threshold는 validation에서 선택했는가?      □
            모델과 threshold를 고정한 뒤 test를 한 번만 사용했는가?      □
                               Dummy 기준 모델과 비교했는가?      □
accuracy 외 precision, recall, F1과 FP/FN을 함께 확인했는가?      □
      공개 prediction에서 내부 source index와 식별자를 제거했는가?      □
                      random split의 교육용 한계를 기록했는가?      □
                         모델 결과를 취소 원인으로 단정하지 않았는가?      □
```

## 13. 해석 시 주의사항
- 모델과 threshold는 Validation에서 선택했습니다.
- Final Test는 선택이 끝난 뒤 마지막 평가에만 사용했습니다.
- accuracy뿐 아니라 precision, recall, F1과 FP/FN을 함께 봅니다.
- random split은 교육용 설계이며 실제 운영 전에는 out-of-time 평가가 필요합니다.
- 모델이 학습한 예측 패턴을 취소의 원인으로 단정하지 않습니다.
- 공개 prediction에는 내부 source index나 원본 식별자를 포함하지 않습니다.
