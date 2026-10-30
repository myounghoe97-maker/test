import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample"
)

CORE_DIM = 16
HISTOGRAM_DIM = 256
BYTEENTROPY_DIM = 256
TOTAL_DIM = CORE_DIM + HISTOGRAM_DIM + BYTEENTROPY_DIM


def number(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    return default


def extract_core_features(record):
    authenticode = record.get("authenticode") or {}
    warnings = record.get("pefilewarnings") or []
    general = record.get("general") or {}
    imports = record.get("imports") or {}
    strings = record.get("strings") or {}

    cert_count = number(authenticode.get("num_certs"))

    import_api_count = 0
    for api_list in imports.values():
        if isinstance(api_list, list):
            import_api_count += len(api_list)

    v0 = [
        1.0 if cert_count > 0 else 0.0,
        1.0 if warnings else 0.0,
        number(general.get("entropy")),
        float(import_api_count),
    ]

    section = record.get("section")
    section_missing = 1.0 if not section else 0.0
    section = section or {}

    sections = section.get("sections") or []

    entropies = []
    for item in sections:
        value = item.get("entropy")
        if isinstance(value, (int, float)):
            entropies.append(float(value))

    if entropies:
        mean_entropy = sum(entropies) / len(entropies)
        max_entropy = max(entropies)
    else:
        mean_entropy = 0.0
        max_entropy = 0.0

    overlay = section.get("overlay") or {}

    section_overlay = [
        section_missing,
        float(len(sections)),
        mean_entropy,
        max_entropy,
        number(overlay.get("size")),
        number(overlay.get("size_ratio")),
        number(overlay.get("entropy")),
    ]

    general_strings = [
        number(general.get("size")),
        number(strings.get("numstrings")),
        number(strings.get("avlength")),
        number(strings.get("entropy")),
        number(strings.get("printables")),
    ]

    return np.asarray(
        v0 + section_overlay + general_strings,
        dtype=np.float32,
    )


def normalize_256_feature(record, key):
    values = record.get(key)

    if not isinstance(values, list) or len(values) != 256:
        return np.zeros(256, dtype=np.float32), 1

    values = np.asarray(values, dtype=np.float32)

    total = float(values.sum())
    if total <= 0.0:
        return np.zeros(256, dtype=np.float32), 1

    return values / total, 0


def count_valid_records(files):
    count = 0

    for file_path in files:
        print(f"[COUNT] {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record.get("label") in (0, 1):
                    count += 1

    return count


def build_dataset(files):
    row_count = count_valid_records(files)

    X = np.empty(
        (row_count, TOTAL_DIM),
        dtype=np.float32,
    )
    y = np.empty(
        row_count,
        dtype=np.int8,
    )

    row_index = 0
    histogram_missing = 0
    byteentropy_missing = 0

    for file_path in files:
        print(f"[LOAD] {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                label = record.get("label")
                if label not in (0, 1):
                    continue

                core = extract_core_features(record)
                histogram, hist_missing = normalize_256_feature(
                    record,
                    "histogram",
                )
                byteentropy, be_missing = normalize_256_feature(
                    record,
                    "byteentropy",
                )

                X[row_index, :CORE_DIM] = core
                X[
                    row_index,
                    CORE_DIM:CORE_DIM + HISTOGRAM_DIM
                ] = histogram
                X[
                    row_index,
                    CORE_DIM + HISTOGRAM_DIM:
                ] = byteentropy

                histogram_missing += hist_missing
                byteentropy_missing += be_missing

                y[row_index] = label
                row_index += 1

    return (
        X,
        y,
        histogram_missing,
        byteentropy_missing,
    )


def calculate_metrics(name, y_true, probabilities, train_seconds, predict_seconds):
    predictions = (
        probabilities >= 0.5
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    pr_precision, pr_recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    return {
        "name": name,
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": auc(
            pr_recall,
            pr_precision,
        ),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "fp": int(fp),
        "fn": int(fn),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
    }


def evaluate_logistic(X_train, y_train, X_val, y_val):
    print()
    print("=" * 84)
    print("Logistic Regression")
    print("=" * 84)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    start = time.perf_counter()
    model.fit(
        X_train_scaled,
        y_train,
    )
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    probabilities = model.predict_proba(
        X_val_scaled
    )[:, 1]
    predict_seconds = time.perf_counter() - start

    return calculate_metrics(
        "Logistic Regression",
        y_val,
        probabilities,
        train_seconds,
        predict_seconds,
    )


def evaluate_random_forest(X_train, y_train, X_val, y_val):
    print()
    print("=" * 84)
    print("Random Forest")
    print("=" * 84)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
    )
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    probabilities = model.predict_proba(
        X_val
    )[:, 1]
    predict_seconds = time.perf_counter() - start

    return calculate_metrics(
        "Random Forest",
        y_val,
        probabilities,
        train_seconds,
        predict_seconds,
    )


def print_result(result):
    print(
        f"ROC-AUC={result['roc_auc']:.4f} | "
        f"PR-AUC={result['pr_auc']:.4f} | "
        f"Recall={result['recall']:.4f} | "
        f"FPR={result['fpr']:.4f} | "
        f"FNR={result['fnr']:.4f} | "
        f"FP={result['fp']:,} | "
        f"FN={result['fn']:,} | "
        f"Train={result['train_seconds']:.1f}s | "
        f"Predict={result['predict_seconds']:.1f}s"
    )


def print_summary(results):
    print()
    print("=" * 136)
    print("MODEL COMPARISON SUMMARY - 528 FEATURES")
    print("=" * 136)

    print(
        f"{'Model':<24}"
        f"{'ROC-AUC':>11}"
        f"{'PR-AUC':>11}"
        f"{'Recall':>10}"
        f"{'FPR':>10}"
        f"{'FNR':>10}"
        f"{'FP':>9}"
        f"{'FN':>9}"
        f"{'Train(s)':>12}"
        f"{'Predict(s)':>12}"
    )

    print("-" * 136)

    for result in results:
        print(
            f"{result['name']:<24}"
            f"{result['roc_auc']:>11.4f}"
            f"{result['pr_auc']:>11.4f}"
            f"{result['recall']:>10.4f}"
            f"{result['fpr']:>10.4f}"
            f"{result['fnr']:>10.4f}"
            f"{result['fp']:>9,}"
            f"{result['fn']:>9,}"
            f"{result['train_seconds']:>12.1f}"
            f"{result['predict_seconds']:>12.1f}"
        )

    print("=" * 136)


def main():
    files = sorted(
        DATA_DIR.glob("*_Win32_train.jsonl")
    )

    if len(files) < 5:
        print("[ERROR] train_sample 폴더에 최소 5개의 Win32 train JSONL 파일이 필요합니다.")
        return

    train_files = files[:4]
    validation_files = [files[4]]

    print("[1/2] Train dataset build")

    (
        X_train,
        y_train,
        train_hist_missing,
        train_be_missing,
    ) = build_dataset(train_files)

    print(f"Train rows      : {len(y_train):,}")
    print(f"Feature count   : {X_train.shape[-1]}")
    print(f"Histogram missing/invalid : {train_hist_missing:,}")
    print(f"ByteEntropy missing/invalid: {train_be_missing:,}")

    print()
    print("[2/2] Validation dataset build")

    (
        X_val,
        y_val,
        val_hist_missing,
        val_be_missing,
    ) = build_dataset(validation_files)

    print(f"Validation rows : {len(y_val):,}")
    print(f"Histogram missing/invalid : {val_hist_missing:,}")
    print(f"ByteEntropy missing/invalid: {val_be_missing:,}")

    results = []

    logistic_result = evaluate_logistic(
        X_train,
        y_train,
        X_val,
        y_val,
    )
    print_result(logistic_result)
    results.append(logistic_result)

    random_forest_result = evaluate_random_forest(
        X_train,
        y_train,
        X_val,
        y_val,
    )
    print_result(random_forest_result)
    results.append(random_forest_result)

    print_summary(results)


if __name__ == "__main__":
    main()
