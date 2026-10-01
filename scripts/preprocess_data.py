"""
전처리 스크립트 (Chapter 05 대신 일단 이렇게 만들었음)

data/raw/order.csv 는 진짜로 있는데,
customers.csv 는 열어보니 order.csv 랑 똑같은 내용이고
주문상세(order_items) 파일은 아예 없었다.
그래서 고객 정보랑 주문상세는 seed 고정해서 가상으로 만들어 쓴다.
(책에서도 가상 데이터라고 했으니까 일단 이렇게 진행)

실행: python scripts/preprocess_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# 1. 주문 (진짜 데이터)
orders = pd.read_csv(raw_dir / "order.csv")
orders["order_date"] = pd.to_datetime(orders["order_date"])
orders = orders.drop_duplicates("order_id").sort_values(["order_date", "order_id"])

# 2. 고객 (가상) - 주문에 나오는 customer_id 만 만든다
customer_ids = np.sort(orders["customer_id"].unique())
customers = pd.DataFrame({
    "customer_id": customer_ids,
    "gender": rng.choice(["F", "M"], size=len(customer_ids)),
    "age": rng.integers(19, 66, size=len(customer_ids)),
    "city": rng.choice(["서울", "부산", "대구", "인천", "광주"], size=len(customer_ids)),
})

# 3. 주문상세 (가상) - 주문 하나에 상품 1~4개
unit_prices = [3000, 8000, 12000, 25000, 49000, 89000]
rows = []
for order_id in orders["order_id"]:
    for _ in range(rng.integers(1, 5)):
        rows.append({
            "order_id": order_id,
            "product_id": int(rng.integers(1, 7)),
            "quantity": int(rng.integers(1, 4)),
            "unit_price": int(rng.choice(unit_prices)),
        })
order_items = pd.DataFrame(rows)
order_items["line_total"] = order_items["quantity"] * order_items["unit_price"]

customers.to_csv(processed_dir / "customers_clean.csv", index=False, encoding="utf-8-sig")
orders.to_csv(processed_dir / "orders_clean.csv", index=False, encoding="utf-8-sig")
order_items.to_csv(processed_dir / "order_items_clean.csv", index=False, encoding="utf-8-sig")

print("customers:", customers.shape)
print("orders:", orders.shape)
print("order_items:", order_items.shape)
