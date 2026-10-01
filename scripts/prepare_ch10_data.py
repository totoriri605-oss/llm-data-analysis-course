"""Prepare validated Chapter10 classification input from common project raw data.

Run from the project root:

    python scripts/prepare_ch10_data.py

Input:
    data/raw/customers.csv
    data/raw/products.csv
    data/raw/orders.csv
    data/raw/order_items.csv

Output:
    data/processed/customers_clean.csv
    data/processed/products_clean.csv
    data/processed/orders_clean.csv
    data/processed/order_items_clean.csv

Chapter05의 전용 오류 탐지 데이터(`practice/chapter05/data/raw`)는 사용하지 않습니다.
Chapter10은 Chapter08·09와 같은 공통 프로젝트 데이터 흐름을 이어받습니다.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_sales_data  # noqa: E402
from src.preprocessing import (  # noqa: E402
    preprocess_sales_data,
    save_processed_data,
    validate_relationships,
)


RAW_DIR = PROJECT_ROOT / "data" / "ch10" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "ch10" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"


def prepare_ch10_data(
    raw_dir: str | Path = RAW_DIR,
    processed_dir: str | Path = PROCESSED_DIR,
) -> dict[str, object]:
    """Preprocess common raw data and fail on core relationship errors."""
    raw_data = load_sales_data(raw_dir)
    processed_data = preprocess_sales_data(raw_data)
    relationship_checks = validate_relationships(processed_data)

    if (
        not relationship_checks.empty
        and not relationship_checks["invalid_count"].eq(0).all()
    ):
        failed = relationship_checks.loc[
            relationship_checks["invalid_count"].ne(0)
        ]
        raise ValueError(
            "Chapter10 입력 관계 검증에 실패했습니다:\n"
            + failed.to_string(index=False)
        )

    # [내가 추가한 단계] 가입일(signup_date)보다 주문일이 앞선 주문은 논리적으로 맞지 않는 데이터다.
    # src/classification.py 는 이런 주문이 있으면 에러를 내는데(공식 raw 로 돌려보니 실제로 에러),
    # 몰래 고치거나 날짜를 바꾸지 않고 "제외 + 목록 저장" 으로 처리한다.
    processed_data, excluded = exclude_orders_before_signup(processed_data)
    if not excluded.empty:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        excluded.to_csv(
            REPORT_DIR / "ch10_excluded_pre_signup_orders_internal.csv",
            index=False,
            encoding="utf-8-sig",
        )

    saved_paths = save_processed_data(processed_data, processed_dir)
    return {
        "raw_data": raw_data,
        "processed_data": processed_data,
        "relationship_checks": relationship_checks,
        "excluded_orders": excluded,
        "saved_paths": saved_paths,
    }


def exclude_orders_before_signup(processed_data):
    """order_date < signup_date 인 주문(과 그 주문상세)을 제외하고, 제외 목록을 돌려준다."""
    import pandas as pd

    customers = processed_data["customers"]
    orders = processed_data["orders"]
    order_items = processed_data["order_items"]

    check = orders[["order_id", "customer_id", "order_date", "order_status"]].merge(
        customers[["customer_id", "signup_date"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    days = (pd.to_datetime(check["order_date"]) - pd.to_datetime(check["signup_date"])).dt.days
    bad_ids = set(check.loc[days.lt(0), "order_id"])
    excluded = check[check["order_id"].isin(bad_ids)].assign(days_before_signup=-days[days.lt(0)])

    result = dict(processed_data)
    result["orders"] = orders[~orders["order_id"].isin(bad_ids)].copy()
    result["order_items"] = order_items[~order_items["order_id"].isin(bad_ids)].copy()
    return result, excluded


def main() -> None:
    result = prepare_ch10_data()

    print("Chapter10 모델링 입력 준비 완료")
    print(f"입력 Raw 경로: {RAW_DIR}")
    print("\n[관계 검증]")
    print(result["relationship_checks"].to_string(index=False))
    print("\n[저장된 processed 파일]")
    for path in result["saved_paths"]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
