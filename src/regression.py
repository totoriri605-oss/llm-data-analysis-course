"""
Chapter 09 회귀 분석 함수 모음

예측 대상: 주문 한 건의 주문금액 합계(order_total)
예측 시점: 주문 날짜/결제수단/고객 정보만 알고, 주문상세(수량, 단가)는 모르는 시점
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 png 로 저장만 할 거라서
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "order_total"

NUMERIC_FEATURES = ["order_month", "order_dayofweek", "age"]
CATEGORICAL_FEATURES = ["payment_method", "gender", "city"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# 입력에 넣으면 안 되는 컬럼들 (목표값, 목표 계산 재료, 사후정보, 식별자)
FORBIDDEN_FEATURES = {
    "order_total",
    "line_total",
    "quantity",
    "unit_price",
    "item_count",
    "total_quantity",
    "avg_unit_price",
    "order_status",
    "order_id",
    "customer_id",
    "product_id",
}

FORBIDDEN_REASONS = {
    "order_total": "목표값 자체",
    "line_total": "목표값 계산 재료",
    "quantity": "목표값 계산 재료",
    "unit_price": "목표값 계산 재료",
    "item_count": "주문상세 집계 (목표값과 연결)",
    "total_quantity": "주문상세 집계 (목표값과 연결)",
    "avg_unit_price": "주문상세 집계 (목표값과 연결)",
    "order_status": "주문 처리 후에 정해지는 사후정보 가능성",
    "order_id": "식별자",
    "customer_id": "식별자",
    "product_id": "식별자",
}

ALLOWED_REASONS = {
    "order_month": "주문일에서 만든 값, 예측 시점에 알 수 있음",
    "order_dayofweek": "주문일에서 만든 값, 예측 시점에 알 수 있음",
    "age": "고객 비식별 특성",
    "payment_method": "주문 메타데이터",
    "gender": "고객 비식별 특성",
    "city": "고객 비식별 특성",
}

BASELINE_NAME = "Baseline Mean"


# ---------------------------------------------------------------
# 데이터 불러오기 / 목표값 만들기
# ---------------------------------------------------------------
def load_regression_source_data(processed_dir):
    processed_dir = Path(processed_dir)
    names = {
        "customers": "customers_clean.csv",
        "orders": "orders_clean.csv",
        "order_items": "order_items_clean.csv",
    }
    missing = [f for f in names.values() if not (processed_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            "전처리 파일이 없습니다: "
            + ", ".join(missing)
            + "\n먼저 python scripts/preprocess_data.py를 실행하세요."
        )
    return {key: pd.read_csv(processed_dir / f) for key, f in names.items()}


def build_order_totals(order_items):
    """주문상세 -> 주문별 order_total"""
    items = order_items.copy()
    if items["order_id"].isna().any():
        raise ValueError("order_items에 order_id 결측이 있습니다.")

    items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce")
    items["unit_price"] = pd.to_numeric(items["unit_price"], errors="coerce")
    if items[["quantity", "unit_price"]].isna().any().any():
        raise ValueError("quantity/unit_price에 결측이거나 숫자가 아닌 값이 있습니다.")

    expected = items["quantity"] * items["unit_price"]
    if "line_total" in items.columns:
        diff = (items["line_total"] - expected).abs()
        if diff.gt(1e-6).any():
            raise ValueError(
                f"line_total이 quantity × unit_price와 다른 행이 {int(diff.gt(1e-6).sum())}개 있습니다."
            )
    items["line_total"] = expected

    totals = items.groupby("order_id", as_index=False)["line_total"].sum()
    totals = totals.rename(columns={"line_total": TARGET_COLUMN})
    if totals[TARGET_COLUMN].le(0).any():
        raise ValueError("order_total이 0 이하인 주문이 있습니다.")
    return totals


def build_regression_dataset(customers, orders, order_items):
    """주문 한 건이 한 행인 모델링 데이터"""
    if not customers["customer_id"].is_unique:
        raise ValueError("customers의 customer_id가 중복입니다.")
    if not orders["order_id"].is_unique:
        raise ValueError("orders의 order_id가 중복입니다.")

    totals = build_order_totals(order_items)

    # 주문 : 목표값 = 1:1
    data = orders.merge(totals, on="order_id", how="outer", validate="one_to_one", indicator=True)
    if not data["_merge"].eq("both").all():
        raise ValueError("주문과 목표값이 1:1로 맞지 않습니다. (주문상세 없는 주문 또는 반대)")
    data = data.drop(columns="_merge")

    # 주문 : 고객 = 다:1
    n_before = len(data)
    data = data.merge(
        customers[["customer_id", "gender", "age", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    if len(data) != n_before:
        raise ValueError("고객과 합치는 중에 행 수가 달라졌습니다.")

    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    must_have = ["order_id", "customer_id", "order_date", TARGET_COLUMN]
    if data[must_have].isna().any().any():
        raise ValueError("필수 컬럼(order_id, customer_id, order_date, order_total)에 결측이 있습니다.")

    data["order_month"] = data["order_date"].dt.month
    data["order_dayofweek"] = data["order_date"].dt.dayofweek
    data = data.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    return data


# ---------------------------------------------------------------
# 입력 변수 검사
# ---------------------------------------------------------------
def validate_feature_columns(feature_columns):
    leaked = set(feature_columns) & FORBIDDEN_FEATURES
    if leaked:
        raise ValueError("누수 위험 입력 변수가 있습니다: " + str(sorted(leaked)))


def build_feature_audit():
    rows = []
    for col in FEATURE_COLUMNS:
        rows.append({"column": col, "selected": True, "role": "allowed_feature", "reason": ALLOWED_REASONS[col]})
    for col in sorted(FORBIDDEN_FEATURES):
        rows.append({"column": col, "selected": False, "role": "forbidden", "reason": FORBIDDEN_REASONS[col]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 시간 순서 분할 (같은 날짜는 한쪽에만)
# ---------------------------------------------------------------
def split_model_data_by_time(model_data, test_size=0.2):
    data = model_data.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    days = data["order_date"].dt.normalize()
    unique_days = np.sort(days.unique())
    if len(unique_days) < 2:
        raise ValueError("시간 순서 분할에는 최소 2개의 서로 다른 주문일이 필요합니다.")

    rows_per_day = days.value_counts().sort_index()
    cum_rows = rows_per_day.cumsum().to_numpy()
    target_train_rows = (1 - test_size) * len(data)

    # 목표 행 수를 처음으로 넘는 날짜까지 train (단, test 날짜는 최소 하루 남김)
    cut = int(np.searchsorted(cum_rows, target_train_rows))
    cut = min(cut, len(unique_days) - 2)
    last_train_day = rows_per_day.index[cut]

    train = data[days <= last_train_day].copy()
    test = data[days > last_train_day].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_split_summary(train_data, test_data):
    total = len(train_data) + len(test_data)
    rows = []
    for name, df in [("train", train_data), ("test", test_data)]:
        rows.append({
            "split": name,
            "rows": len(df),
            "ratio_pct": round(len(df) / total * 100, 1),
            "start_date": df["order_date"].min().date(),
            "end_date": df["order_date"].max().date(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 전처리 / 모델 Pipeline
# ---------------------------------------------------------------
def make_one_hot_encoder():
    # sklearn 버전마다 인자 이름이 달라서 둘 다 시도
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def make_regression_models(random_state=42):
    # 모델마다 preprocessor를 따로 만들어서 같은 객체를 공유하지 않게 함
    return {
        BASELINE_NAME: Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Linear Regression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }


# ---------------------------------------------------------------
# 훈련 기간 TimeSeriesSplit 후보 비교 / 모델 선택
# ---------------------------------------------------------------
def cross_validate_regression_models(models, X_train, y_train, max_splits=5):
    n = len(X_train)
    if n < 6:
        raise ValueError("시간 순서 교차검증에는 최소 6개의 훈련 행이 필요합니다.")

    # validation fold 가 최소 2행은 되도록 split 수를 줄인다 (R² 계산 때문에)
    n_splits = max_splits
    while n_splits > 2 and n // (n_splits + 1) < 2:
        n_splits -= 1
    cv = TimeSeriesSplit(n_splits=n_splits)

    rows = []
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1,
        )
        mae = -scores["test_mae"]
        rows.append({
            "model": name,
            "n_splits": n_splits,
            "cv_MAE_mean": mae.mean(),
            "cv_MAE_std": mae.std(),
            "cv_R2_mean": scores["test_r2"].mean(),
        })
    return pd.DataFrame(rows)


def select_diagnostic_model(cv_summary):
    """Baseline 은 빼고 cv_MAE_mean 이 가장 낮은 모델 이름"""
    candidates = cv_summary[cv_summary["model"] != BASELINE_NAME]
    best = candidates.sort_values(["cv_MAE_mean", "model"]).iloc[0]
    return best["model"]


# ---------------------------------------------------------------
# Final Test (Baseline + 고정한 모델만)
# ---------------------------------------------------------------
def train_and_evaluate_models(models, selected_model_name, X_train, X_test, y_train, y_test):
    targets = [BASELINE_NAME, selected_model_name]
    rows = []
    predictions = {}
    for name in targets:
        pipeline = models[name]
        pipeline.fit(X_train, y_train)
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        predictions[name] = test_pred
        rows.append({
            "model": name,
            "selection_role": "baseline" if name == BASELINE_NAME else "selected_by_train_cv",
            "train_MAE": mean_absolute_error(y_train, train_pred),
            "test_MAE": mean_absolute_error(y_test, test_pred),
            "test_RMSE": root_mean_squared_error(y_test, test_pred),
            "test_R2": r2_score(y_test, test_pred),
        })
    result = pd.DataFrame(rows)

    base_mae = result.loc[result["model"] == BASELINE_NAME, "test_MAE"].iloc[0]
    result["MAE_improvement_vs_baseline_pct"] = (base_mae - result["test_MAE"]) / base_mae * 100
    return result, predictions


def build_regression_validation(train_data, test_data, cv_summary, selected_model_name, model_comparison):
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    overlap = set(FEATURE_COLUMNS) & FORBIDDEN_FEATURES
    add("forbidden feature overlap = 0", len(overlap) == 0, str(sorted(overlap)))

    train_max = train_data["order_date"].max().normalize()
    test_min = test_data["order_date"].min().normalize()
    add("train max date < test min date", train_max < test_min, f"{train_max.date()} < {test_min.date()}")

    days_overlap = set(train_data["order_date"].dt.normalize()) & set(test_data["order_date"].dt.normalize())
    add("같은 날짜가 train/test에 없음", len(days_overlap) == 0, f"겹치는 날짜 {len(days_overlap)}개")

    add("선택 모델이 CV 결과에 있음", selected_model_name in set(cv_summary["model"]), selected_model_name)
    add("선택 모델이 Baseline이 아님", selected_model_name != BASELINE_NAME, selected_model_name)

    final_models = set(model_comparison["model"])
    add("Final Test에 Baseline 있음", BASELINE_NAME in final_models, "")
    add("Final Test에 고정 모델 있음", selected_model_name in final_models, "")
    add("Final Test는 두 모델만 비교", final_models == {BASELINE_NAME, selected_model_name}, str(sorted(final_models)))
    add("R² 계산 가능한 테스트 크기", len(test_data) >= 2, f"test rows = {len(test_data)}")
    return pd.DataFrame(checks)


# ---------------------------------------------------------------
# 예측 결과 / 그래프
# ---------------------------------------------------------------
def create_prediction_result(test_data, y_test, y_pred, model_name):
    result = pd.DataFrame({
        "order_id": test_data["order_id"].to_numpy(),
        "order_date": test_data["order_date"].dt.date.to_numpy(),
        "actual_order_total": np.asarray(y_test, dtype=float),
        "predicted_order_total": np.asarray(y_pred, dtype=float),
    })
    result["residual"] = result["actual_order_total"] - result["predicted_order_total"]
    result["abs_error"] = result["residual"].abs()
    result["model"] = model_name
    return result.sort_values("abs_error", ascending=False).reset_index(drop=True)


def public_prediction_result(prediction_result):
    # 공개용에서는 order_id 빼기
    return prediction_result.drop(columns=["order_id"], errors="ignore").copy()


def configure_korean_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def create_diagnostic_figures(prediction_result, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    korean = configure_korean_font()

    def t(ko, en):
        return ko if korean else en

    actual = prediction_result["actual_order_total"]
    pred = prediction_result["predicted_order_total"]

    # 실제값 vs 예측값
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, alpha=0.6)
    low = min(actual.min(), pred.min())
    high = max(actual.max(), pred.max())
    ax.plot([low, high], [low, high], color="red", linestyle="--", label=t("완벽한 예측선", "perfect prediction"))
    ax.set_title(t("실제값 대 예측값", "Actual vs Predicted"))
    ax.set_xlabel(t("실제 주문금액", "Actual order_total"))
    ax.set_ylabel(t("예측 주문금액", "Predicted order_total"))
    ax.legend()
    fig.tight_layout()
    scatter_path = figure_dir / "ch09_actual_vs_predicted.png"
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)

    # 잔차 히스토그램
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(prediction_result["residual"], bins=20, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(t("잔차 히스토그램", "Residual histogram"))
    ax.set_xlabel(t("잔차 (실제값 - 예측값)", "Residual (actual - predicted)"))
    ax.set_ylabel(t("주문 수", "Count"))
    fig.tight_layout()
    hist_path = figure_dir / "ch09_residual_histogram.png"
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)

    return {"actual_vs_predicted": scatter_path, "residual_histogram": hist_path}


# ---------------------------------------------------------------
# 체크리스트 / 저장 / 전체 실행
# ---------------------------------------------------------------
def build_leakage_checklist():
    items = [
        ("예측 시점", "입력 변수가 예측 시점에 실제로 사용 가능한가?"),
        ("목표 계산 재료", "quantity, unit_price, line_total 등이 입력에 없는가?"),
        ("사후정보", "order_status 같은 처리 후 정보가 입력에 없는가?"),
        ("식별자", "order_id, customer_id, product_id가 입력에 없는가?"),
        ("Pipeline 전처리", "전처리가 Pipeline 안에서 train fold에만 fit되는가?"),
        ("날짜 분리", "같은 날짜가 train/test에 나뉘지 않았는가?"),
        ("train CV 선택", "모델 선택이 훈련 기간 TimeSeriesSplit에서 끝났는가?"),
        ("test 최종 평가", "test에서는 Baseline과 고정 모델만 비교했는가?"),
        ("Dummy 비교", "Dummy 베이스라인과 비교했는가?"),
        ("지표 해석", "MAE, RMSE, R²를 함께 해석했는가?"),
        ("낮은 성능 기록", "낮은 성능이나 음수 R²를 숨기지 않았는가?"),
        ("내부/공개 분리", "공개 결과에서 order_id를 제거했는가?"),
    ]
    return pd.DataFrame(items, columns=["item", "question"]).assign(checked=True)


def _write_report(path, split_summary, cv_summary, selected_model_name, model_comparison, validation):
    base = model_comparison[model_comparison["model"] == BASELINE_NAME].iloc[0]
    sel = model_comparison[model_comparison["model"] == selected_model_name].iloc[0]
    improve = sel["MAE_improvement_vs_baseline_pct"]

    if improve > 0:
        verdict = f"선택 모델이 Baseline보다 MAE가 {improve:.1f}% 낮았다."
    else:
        verdict = (
            f"선택 모델이 Baseline보다 MAE가 {abs(improve):.1f}% 높았다 (개선 없음). "
            "현재 입력 변수만으로는 주문금액을 잘 설명하지 못하는 것 같다."
        )
    if sel["test_R2"] < 0:
        r2_note = "R²가 음수다. 테스트 목표값의 평균으로 예측하는 것보다 못했다는 뜻이다. 숨기지 않고 기록한다."
    else:
        r2_note = "R²가 0 이상이다."

    gap = sel["test_MAE"] - sel["train_MAE"]
    lines = [
        "# Chapter 09 회귀 분석 보고서",
        "",
        "예측 대상은 주문 한 건의 주문금액 합계(order_total)이다.",
        "예측 시점은 주문 메타데이터와 고객 정보만 알고 주문상세는 모르는 시점으로 가정했다.",
        "",
        "## 데이터 분할",
        split_summary.to_markdown(index=False),
        "",
        "## 훈련 기간 TimeSeriesSplit 결과",
        cv_summary.round(2).to_markdown(index=False),
        "",
        f"비베이스라인 중 cv_MAE_mean이 가장 낮은 **{selected_model_name}** 을 Final Test 전에 고정했다.",
        "",
        "## Final Test (Baseline vs 고정 모델)",
        model_comparison.round(2).to_markdown(index=False),
        "",
        "## 해석",
        f"- {verdict}",
        f"- {r2_note}",
        f"- train MAE와 test MAE 차이: {gap:,.0f}",
        f"- Baseline test MAE: {base['test_MAE']:,.0f}",
        "- 테스트 결과를 보고 모델을 다시 고르지 않았다.",
        "- 개선이 없다면 운영하지 않는 결정도 올바른 결과라고 책에 나와 있다.",
        "",
        "## 자동 Validation",
        validation.to_markdown(index=False),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_regression_outputs(
    model_data,
    train_data,
    test_data,
    split_summary,
    feature_audit,
    cv_summary,
    selected_model_name,
    model_comparison,
    prediction_result,
    validation,
    checklist,
    report_dir,
):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "split_summary": report_dir / "ch09_regression_split_summary.csv",
        "feature_audit": report_dir / "ch09_regression_feature_audit.csv",
        "cv_summary": report_dir / "ch09_regression_cv_summary.csv",
        "model_comparison": report_dir / "ch09_regression_model_comparison.csv",
        "validation": report_dir / "ch09_regression_validation.csv",
        "checklist": report_dir / "ch09_regression_checklist.csv",
        "model_data_internal": report_dir / "ch09_regression_model_data_internal.csv",
        "predictions_internal": report_dir / "ch09_regression_predictions_internal.csv",
        "report": report_dir / "ch09_regression_report.md",
    }
    tables = {
        "split_summary": split_summary,
        "feature_audit": feature_audit,
        "cv_summary": cv_summary,
        "model_comparison": model_comparison,
        "validation": validation,
        "checklist": checklist,
        "model_data_internal": model_data,
        "predictions_internal": prediction_result,
    }
    for key, df in tables.items():
        df.to_csv(paths[key], index=False, encoding="utf-8-sig")

    _write_report(paths["report"], split_summary, cv_summary, selected_model_name, model_comparison, validation)
    return paths


def run_regression_analysis(processed_dir, report_dir, test_size=0.2, random_state=42):
    report_dir = Path(report_dir)
    figure_dir = report_dir / "figures"

    datasets = load_regression_source_data(processed_dir)
    model_data = build_regression_dataset(datasets["customers"], datasets["orders"], datasets["order_items"])

    validate_feature_columns(FEATURE_COLUMNS)
    feature_audit = build_feature_audit()

    train_data, test_data = split_model_data_by_time(model_data, test_size=test_size)
    split_summary = build_split_summary(train_data, test_data)

    X_train, y_train = train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN]
    X_test, y_test = test_data[FEATURE_COLUMNS], test_data[TARGET_COLUMN]

    models = make_regression_models(random_state=random_state)
    cv_summary = cross_validate_regression_models(models, X_train, y_train, max_splits=5)
    selected_model_name = select_diagnostic_model(cv_summary)

    model_comparison, predictions = train_and_evaluate_models(
        models, selected_model_name, X_train, X_test, y_train, y_test
    )
    validation = build_regression_validation(
        train_data, test_data, cv_summary, selected_model_name, model_comparison
    )
    if not validation["passed"].all():
        raise RuntimeError("자동 Validation 실패:\n" + validation.to_string())

    prediction_result = create_prediction_result(
        test_data, y_test, predictions[selected_model_name], selected_model_name
    )
    figure_paths = create_diagnostic_figures(prediction_result, figure_dir)
    checklist = build_leakage_checklist()

    output_paths = save_regression_outputs(
        model_data=model_data,
        train_data=train_data,
        test_data=test_data,
        split_summary=split_summary,
        feature_audit=feature_audit,
        cv_summary=cv_summary,
        selected_model_name=selected_model_name,
        model_comparison=model_comparison,
        prediction_result=prediction_result,
        validation=validation,
        checklist=checklist,
        report_dir=report_dir,
    )
    output_paths.update(figure_paths)

    return {
        "split_summary": split_summary,
        "feature_audit": feature_audit,
        "cv_summary": cv_summary,
        "selected_model_name": selected_model_name,
        "model_comparison": model_comparison,
        "validation": validation,
        "prediction_result": prediction_result,
        "output_paths": output_paths,
    }
