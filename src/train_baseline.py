import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample"
)

PROJECT_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\malware_detection"
)

MODEL_DIR = PROJECT_DIR / "models"

FEATURE_NAMES = [
    "has_certificate",
    "has_pe_warning",
    "entropy",
    "import_api_count",
]


def extract_features(record):
    # 인증서 존재 여부
    authenticode = record.get("authenticode") or {}
    cert_count = authenticode.get("num_certs", 0) or 0
    has_certificate = 1 if cert_count > 0 else 0

    # PE Warning 존재 여부
    warnings = record.get("pefilewarnings") or []
    has_pe_warning = 1 if warnings else 0

    # 전체 Entropy
    general = record.get("general") or {}
    entropy = general.get("entropy")

    if entropy is None:
        entropy = 0.0

    # Import API 개수
    imports = record.get("imports") or {}

    import_api_count = 0

    for api_list in imports.values():
        if isinstance(api_list, list):
            import_api_count += len(api_list)

    return [
        has_certificate,
        has_pe_warning,
        entropy,
        import_api_count,
    ]


def load_week(file_path):
    X = []
    y = []

    json_errors = 0

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            label = record.get("label")

            if label not in (0, 1):
                continue

            X.append(extract_features(record))
            y.append(label)

    print(
        f"[LOAD] {file_path.name} | "
        f"{len(y):,}건 | JSON 오류={json_errors:,}"
    )

    return X, y


def main():
    files = sorted(DATA_DIR.glob("*_Win32_train.jsonl"))

    if len(files) < 5:
        print("[ERROR] 최소 5주 데이터가 필요합니다.")
        print(f"현재 파일 수: {len(files)}")
        return

    # -------------------------------------------------
    # 시간 순서 유지
    #
    # 앞 4주 = 학습
    # 마지막 1주 = 검증
    #
    # 랜덤 분할하지 않음
    # -------------------------------------------------

    train_files = files[:4]
    validation_file = files[4]

    print("=" * 70)
    print("EMBER2024 Win32 Baseline")
    print("=" * 70)

    print("\n[Train]")
    for file_path in train_files:
        print(file_path.name)

    print("\n[Validation]")
    print(validation_file.name)

    X_train = []
    y_train = []

    print("\n[데이터 로딩]")

    for file_path in train_files:
        X_week, y_week = load_week(file_path)

        X_train.extend(X_week)
        y_train.extend(y_week)

    X_val, y_val = load_week(validation_file)

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int8)

    X_val = np.asarray(X_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.int8)

    print()
    print(f"Train      : {len(y_train):,}")
    print(f"Validation : {len(y_val):,}")

    # -------------------------------------------------
    # Scaling
    # -------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # -------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------

    print("\n[MODEL] Logistic Regression 학습 시작")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    print("[MODEL] 학습 완료")

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    probabilities = model.predict_proba(X_val_scaled)[:, 1]

    threshold = 0.5

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print()
    print("=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print()
    print("[Confusion Matrix]")
    print(f"TN 정상→정상 : {tn:,}")
    print(f"FP 정상→악성 : {fp:,}")
    print(f"FN 악성→정상 : {fn:,}")
    print(f"TP 악성→악성 : {tp:,}")

    print()
    print(f"FPR : {fpr:.4%}")
    print(f"FNR : {fnr:.4%}")

    # -------------------------------------------------
    # Feature coefficients
    # -------------------------------------------------

    print()
    print("[Feature Coefficients]")

    for name, coefficient in zip(
        FEATURE_NAMES,
        model.coef_[0],
    ):
        print(
            f"{name:<20} : "
            f"{coefficient:+.4f}"
        )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "baseline_logistic.joblib"
    scaler_path = MODEL_DIR / "baseline_scaler.joblib"

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    print()
    print(f"[SAVE] {model_path}")
    print(f"[SAVE] {scaler_path}")

    print()
    print("[DONE] Baseline 완료")


if __name__ == "__main__":
    main()