import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


DATA_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample"
)

CORE_DIM = 16
HISTOGRAM_DIM = 256
BYTEENTROPY_DIM = 256
TOTAL_DIM = CORE_DIM + HISTOGRAM_DIM + BYTEENTROPY_DIM

LOW_THRESHOLDS = [0.10, 0.20, 0.30, 0.40]
HIGH_THRESHOLDS = [0.60, 0.70, 0.80, 0.90]


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

    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    max_entropy = max(entropies) if entropies else 0.0

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
        return np.zeros(256, dtype=np.float32)

    values = np.asarray(values, dtype=np.float32)

    total = float(values.sum())
    if total <= 0.0:
        return np.zeros(256, dtype=np.float32)

    return values / total


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

    X = np.empty((row_count, TOTAL_DIM), dtype=np.float32)
    y = np.empty(row_count, dtype=np.int8)

    row_index = 0

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
                histogram = normalize_256_feature(record, "histogram")
                byteentropy = normalize_256_feature(record, "byteentropy")

                X[row_index, :CORE_DIM] = core
                X[
                    row_index,
                    CORE_DIM:CORE_DIM + HISTOGRAM_DIM
                ] = histogram
                X[
                    row_index,
                    CORE_DIM + HISTOGRAM_DIM:
                ] = byteentropy

                y[row_index] = label
                row_index += 1

    return X, y


def print_baseline(y_val, scores):
    predictions = (scores >= 0.5).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    print()
    print("=" * 90)
    print("RANDOM FOREST BASELINE @ 0.50")
    print("=" * 90)
    print(f"ROC-AUC : {roc_auc_score(y_val, scores):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_val, scores):.4f}")
    print(f"FP      : {fp:,}")
    print(f"FN      : {fn:,}")
    print(f"FPR     : {fp / (fp + tn):.4f}")
    print(f"FNR     : {fn / (fn + tp):.4f}")


def evaluate_uncertain_zone(y_true, scores, low_threshold, high_threshold):
    low_mask = scores < low_threshold
    high_mask = scores >= high_threshold
    uncertain_mask = ~(low_mask | high_mask)

    total = len(y_true)
    benign_total = int(np.sum(y_true == 0))
    malicious_total = int(np.sum(y_true == 1))

    low_count = int(np.sum(low_mask))
    high_count = int(np.sum(high_mask))
    uncertain_count = int(np.sum(uncertain_mask))

    low_malicious = int(np.sum((y_true == 1) & low_mask))
    high_benign = int(np.sum((y_true == 0) & high_mask))

    auto_count = low_count + high_count
    auto_correct = int(
        np.sum((y_true == 0) & low_mask)
        + np.sum((y_true == 1) & high_mask)
    )

    return {
        "low": low_threshold,
        "high": high_threshold,
        "auto_rate": auto_count / total,
        "sandbox_rate": uncertain_count / total,
        "low_malicious": low_malicious,
        "low_malicious_rate": low_malicious / malicious_total,
        "high_benign": high_benign,
        "high_benign_rate": high_benign / benign_total,
        "auto_accuracy": auto_correct / auto_count if auto_count else 0.0,
    }


def print_zone_summary(results):
    print()
    print("=" * 132)
    print("UNCERTAIN ZONE CANDIDATE SUMMARY")
    print("=" * 132)

    print(
        f"{'Low':>6}"
        f"{'High':>7}"
        f"{'Auto%':>10}"
        f"{'Sandbox%':>11}"
        f"{'Low→Mal':>11}"
        f"{'MalLeak%':>11}"
        f"{'High→Ben':>11}"
        f"{'BenLeak%':>11}"
        f"{'AutoAcc%':>11}"
    )

    print("-" * 132)

    for result in results:
        print(
            f"{result['low']:>6.2f}"
            f"{result['high']:>7.2f}"
            f"{result['auto_rate'] * 100:>9.2f}%"
            f"{result['sandbox_rate'] * 100:>10.2f}%"
            f"{result['low_malicious']:>11,}"
            f"{result['low_malicious_rate'] * 100:>10.2f}%"
            f"{result['high_benign']:>11,}"
            f"{result['high_benign_rate'] * 100:>10.2f}%"
            f"{result['auto_accuracy'] * 100:>10.2f}%"
        )

    print("=" * 132)


def main():
    files = sorted(DATA_DIR.glob("*_Win32_train.jsonl"))

    if len(files) < 5:
        print("[ERROR] train_sample 폴더에 최소 5개의 Win32 train JSONL 파일이 필요합니다.")
        return

    train_files = files[:4]
    validation_files = [files[4]]

    print("[1/2] Train dataset build")
    X_train, y_train = build_dataset(train_files)

    print(f"Train rows    : {len(y_train):,}")
    print(f"Feature count : {X_train.shape[-1]}")

    print()
    print("[2/2] Validation dataset build")
    X_val, y_val = build_dataset(validation_files)

    print(f"Validation rows: {len(y_val):,}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    print()
    print("[MODEL] Random Forest training")
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_val)[:, 1]

    print_baseline(y_val, scores)

    results = []

    for low_threshold in LOW_THRESHOLDS:
        for high_threshold in HIGH_THRESHOLDS:
            if low_threshold >= high_threshold:
                continue

            results.append(
                evaluate_uncertain_zone(
                    y_val,
                    scores,
                    low_threshold,
                    high_threshold,
                )
            )

    print_zone_summary(results)


if __name__ == "__main__":
    main()
