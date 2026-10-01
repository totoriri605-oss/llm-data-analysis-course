# llm-data-analysis-course

AI Data Analysis 책 따라하면서 공부하는 저장소.

## Chapter 09. 회귀 분석으로 숫자 예측하기
- 노트북: `notebooks/ch09/ch09_regression.ipynb`
- 함수 모음: `src/regression.py`
- 실행 순서
  1. `python scripts/preprocess_data.py`
  2. `python scripts/run_regression_analysis.py`
- 결과: `reports/` (csv, 보고서 md, 그래프), 스크린샷: `docs/ch09/`

주의: `data/raw/customers.csv` 가 주문 파일이랑 내용이 같아서, 고객 정보와 주문상세는
`scripts/preprocess_data.py` 에서 seed 고정으로 가상 데이터를 만들어 썼음.
주문번호가 들어간 `*_internal.csv` 는 .gitignore 로 올리지 않음.

## 제출용
- Chapter 09 답안 양식 노트북: `chapter09/chapter09.ipynb` (이미지: `chapter09/images/`)
- Chapter 10 답안 양식 노트북: `chapter10/chapter10.ipynb` (이미지: `chapter10/images/`)
  - 강의 공개 저장소의 `src/classification.py` 등을 복사해 사용 (데이터는 `data/ch10/`)
  - 공식 raw 에 가입일보다 주문일이 앞선 주문이 있어서 `scripts/prepare_ch10_data.py` 에 제외 단계를 추가함 (자세한 내용은 노트북 맨 위)
