import json
from pathlib import Path

import numpy as np
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


def number(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    return default


def extract_core_features(record):
    """
    V1-Core 후보:
    V0(4) + SectionOverlay(7) + GeneralStrings(5) = 16 features
    """
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


def extract_histogram(record):
    """
    EMBER2024 ByteHistogram:
    raw 256-bin byte counts -> normalized by total count.
    """
    histogram = record.get("histogram")

    if not isinstance(histogram, list) or len(histogram) != HISTOGRAM_DIM:
        return np.zeros(HISTOGRAM_DIM, dtype=np.float32), 1

    values = np.asarray(histogram, dtype=np.float32)

    total = float(values.sum())
    if total <= 0.0:
        return np.zeros(HISTOGRAM_DIM, dtype=np.float32), 1

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


def build_dataset(files, include_histogram):
    feature_dim = CORE_DIM + (HISTOGRAM_DIM if include_histogram else 0)

    row_count = count_valid_records(files)

    X = np.empty(
        (row_count, feature_dim),
        dtype=np.float32,
    )
    y = np.empty(
        row_count,
        dtype=np.int8,
    )

    row_index = 0
    histogram_missing = 0

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

                if include_histogram:
                    histogram, missing = extract_histogram(record)

                    X[row_index, :CORE_DIM] = core
                    X[row_index, CORE_DIM:] = histogram

                    histogram_missing += missing
                else:
                    X[row_index] = core

                y[row_index] = label
                row_index += 1

    return X, y, histogram_missing


def evaluate(
    name,
    train_files,
    validation_files,
    include_histogram,
):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    X_train, y_train, train_hist_missing = build_dataset(
        train_files,
        include_histogram,
    )

    X_val, y_val, val_hist_missing = build_dataset(
        validation_files,
        include_histogram,
    )

    print(f"Train rows      : {len(y_train):,}")
    print(f"Validation rows : {len(y_val):,}")
    print(f"Feature count   : {X_train.shape[1]}")

    if include_histogram:
        print(f"Train histogram missing/invalid : {train_hist_missing:,}")
        print(f"Val histogram missing/invalid   : {val_hist_missing:,}")

    scaler = StandardScaler(copy=False)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

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

    pr_precision, pr_recall, _ = precision_recall_curve(
        y_val,
        probabilities,
    )

    result = {
        "name": name,
        "features": X_train.shape[1],
        "accuracy": accuracy_score(
            y_val,
            predictions,
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(
            y_val,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_val,
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
    }

    print(
        f"ROC-AUC={result['roc_auc']:.4f} | "
        f"PR-AUC={result['pr_auc']:.4f} | "
        f"Recall={result['recall']:.4f} | "
        f"FPR={result['fpr']:.4f} | "
        f"FNR={result['fnr']:.4f} | "
        f"FP={result['fp']:,} | "
        f"FN={result['fn']:,}"
    )

    return result


def print_summary(results):
    print()
    print("=" * 116)
    print("HISTOGRAM INCREMENTAL VALUE SUMMARY")
    print("=" * 116)

    print(
        f"{'Model':<34}"
        f"{'Feat':>6}"
        f"{'ROC-AUC':>11}"
        f"{'PR-AUC':>11}"
        f"{'Recall':>10}"
        f"{'FPR':>10}"
        f"{'FNR':>10}"
        f"{'FP':>9}"
        f"{'FN':>9}"
    )
    print("-" * 116)

    for result in results:
        print(
            f"{result['name']:<34}"
            f"{result['features']:>6}"
            f"{result['roc_auc']:>11.4f}"
            f"{result['pr_auc']:>11.4f}"
            f"{result['recall']:>10.4f}"
            f"{result['fpr']:>10.4f}"
            f"{result['fnr']:>10.4f}"
            f"{result['fp']:>9,}"
            f"{result['fn']:>9,}"
        )

    print("=" * 116)


def main():
    files = sorted(
        DATA_DIR.glob("*_Win32_train.jsonl")
    )

    if len(files) < 5:
        print("[ERROR] train_sample 폴더에 최소 5개의 Win32 train JSONL 파일이 필요합니다.")
        return

    train_files = files[:4]
    validation_files = [files[4]]

    print("Train files:")
    for file_path in train_files:
        print(f"  - {file_path.name}")

    print("Validation file:")
    print(f"  - {validation_files[0].name}")

    results = []

    results.append(
        evaluate(
            "V1-Core",
            train_files,
            validation_files,
            include_histogram=False,
        )
    )

    results.append(
        evaluate(
            "V1-Core + Histogram",
            train_files,
            validation_files,
            include_histogram=True,
        )
    )

    print_summary(results)


if __name__ == "__main__":
    main()
