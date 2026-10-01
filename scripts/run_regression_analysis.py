"""
Chapter 09 회귀 분석 전체 실행
실행: python scripts/run_regression_analysis.py
(먼저 python scripts/preprocess_data.py 를 해둬야 함)
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.regression import run_regression_analysis  # noqa: E402

result = run_regression_analysis(
    processed_dir=project_root / "data" / "processed",
    report_dir=project_root / "reports",
    test_size=0.2,
    random_state=42,
)

print("분할 요약")
print(result["split_summary"].to_string(index=False))
print()
print("훈련 CV 결과")
print(result["cv_summary"].round(2).to_string(index=False))
print()
print("선택 모델:", result["selected_model_name"])
print()
print("Final Test")
print(result["model_comparison"].round(2).to_string(index=False))
print()
print("Validation 통과:", bool(result["validation"]["passed"].all()))
