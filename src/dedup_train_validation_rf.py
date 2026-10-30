import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


DATA_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample"
)

LOW_THRESHOLDS = [0.01, 0.03, 0.05, 0.08, 0.10]
HIGH_THRESHOLDS = [0.75, 0.80, 0.85, 0.90]


def number(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    return default


def get_sha256(record):
    for key in ("sha256", "sha_256", "hash"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def extract_core(record):
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

    section = record.get("section")
    section_missing = 1.0 if not section else 0.0
    section = section or {}

    sections = section.get("sections") or []
    entropies = [
        float(item["entropy"])
        for item in sections
        if isinstance(item.get("entropy"), (int, float))
    ]

    mean_section_entropy = (
        sum(entropies) / len(entropies)
        if entropies
        else 0.0
    )

    max_section_entropy = max(entropies) if entropies else 0.0

    overlay = section.get("overlay") or {}

    return np.asarray(
        [
            1.0 if cert_count > 0 else 0.0,
            1.0 if warnings else 0.0,
            number(general.get("entropy")),
            float(import_api_count),
            section_missing,
            float(len(sections)),
            mean_section_entropy,
            max_section_entropy,
            number(overlay.get("size")),
            number(overlay.get("size_ratio")),
            number(overlay.get("entropy")),
            number(general.get("size")),
            number(strings.get("numstrings")),
            number(strings.get("avlength")),
            number(strings.get("entropy")),
            number(strings.get("printables")),
        ],
        dtype=np.float32,
    )


def normalize_256(record, key):
    values = record.get(key)

    if not isinstance(values, list) or len(values) != 256:
        return np.zeros(256, dtype=np.float32)

    arr = np.asarray(values, dtype=np.float32)
    total = float(arr.sum())

    if total <= 0.0:
        return np.zeros(256, dtype=np.float32)

    return arr / total


def build_vector(record):
    return np.concatenate(
        [
            extract_core(record),
            normalize_256(record, "histogram"),
            normalize_256(record, "byteentropy"),
        ]
    ).astype(np.float32)


def load_dedup(files, name):
    X = []
    y = []
    seen_sha = set()

    total_valid = 0
    missing_sha = 0
    duplicate_rows = 0

    for file_path in files:
        print(f"[{name}] {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                label = record.get("label")
                if label not in (0, 1):
                    continue

                total_valid += 1

                sha = get_sha256(record)

                if not sha:
                    missing_sha += 1
                    continue

                if sha in seen_sha:
                    duplicate_rows += 1
                    continue

                seen_sha.add(sha)

                X.append(build_vector(record))
                y.append(label)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int8),
        {
            "total_valid": total_valid,
            "unique_sha": len(seen_sha),
            "duplicate_rows": duplicate_rows,
            "missing_sha": missing_sha,
        },
    )


def print_dataset_stats(name, stats):
    print()
    print(f"{name} DATASET")
    print("-" * 70)
    print(f"Valid rows           : {stats['total_valid']:,}")
    print(f"Unique SHA           : {stats['unique_sha']:,}")
    print(f"Duplicate rows removed: {stats['duplicate_rows']:,}")
    print(f"Missing SHA          : {stats['missing_sha']:,}")


def main():
    files = sorted(DATA_DIR.glob("*_Win32_train.jsonl"))

    if len(files) < 5:
        print("[ERROR] train_sample 폴더에 최소 5개의 JSONL이 필요합니다.")
        return

    train_files = files[:4]
    validation_files = [files[4]]

    print("[1/4] Load + dedup TRAIN")
    X_train, y_train, train_stats = load_dedup(
        train_files,
        "TRAIN",
    )

    print("[2/4] Load + dedup VALIDATION")
    X_val, y_val, val_stats = load_dedup(
        validation_files,
        "VAL",
    )

    print_dataset_stats("TRAIN", train_stats)
    print_dataset_stats("VALIDATION", val_stats)

    print()
    print("[3/4] Train Random Forest")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    scores = model.predict_proba(X_val)[:, 1]
    preds = (scores >= 0.5).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        preds,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print()
    print("=" * 100)
    print("DEDUP TRAIN + DEDUP VALIDATION - RF 528")
    print("=" * 100)
    print(f"Accuracy   : {accuracy_score(y_val, preds):.4f}")
    print(f"Precision  : {precision_score(y_val, preds):.4f}")
    print(f"Recall     : {recall_score(y_val, preds):.4f}")
    print(f"F1         : {f1_score(y_val, preds):.4f}")
    print(f"ROC-AUC    : {roc_auc_score(y_val, scores):.4f}")
    print(f"PR-AUC     : {average_precision_score(y_val, scores):.4f}")
    print(f"FPR        : {fpr:.4f}")
    print(f"FNR        : {fnr:.4f}")
    print(f"TN         : {tn:,}")
    print(f"FP         : {fp:,}")
    print(f"FN         : {fn:,}")
    print(f"TP         : {tp:,}")

    print()
    print("[4/4] Uncertain zone")

    print()
    print("=" * 136)
    print("DEDUP TRAIN + DEDUP VALIDATION - UNCERTAIN ZONE")
    print("=" * 136)
    print(
        f"{'Low':>6}"
        f"{'High':>7}"
        f"{'Auto%':>10}"
        f"{'Sandbox%':>11}"
        f"{'Low→Mal':>11}"
        f"{'MalLeak%':>11}"
        f"{'High→Ben':>11}"
        f"{'BenLeak%':>11}"
        f"{'AutoAcc%':>10}"
    )
    print("-" * 136)

    total = len(y_val)
    total_mal = int((y_val == 1).sum())
    total_ben = int((y_val == 0).sum())

    for low in LOW_THRESHOLDS:
        for high in HIGH_THRESHOLDS:
            low_mask = scores < low
            high_mask = scores >= high
            auto_mask = low_mask | high_mask
            sandbox_mask = ~auto_mask

            low_mal = int(
                np.sum(low_mask & (y_val == 1))
            )

            high_ben = int(
                np.sum(high_mask & (y_val == 0))
            )

            auto_count = int(auto_mask.sum())
            sandbox_count = int(sandbox_mask.sum())

            auto_correct = int(
                np.sum(
                    (low_mask & (y_val == 0))
                    | (high_mask & (y_val == 1))
                )
            )

            auto_pct = 100.0 * auto_count / total
            sandbox_pct = 100.0 * sandbox_count / total

            mal_leak = (
                100.0 * low_mal / total_mal
                if total_mal
                else 0.0
            )

            ben_leak = (
                100.0 * high_ben / total_ben
                if total_ben
                else 0.0
            )

            auto_acc = (
                100.0 * auto_correct / auto_count
                if auto_count
                else 0.0
            )

            print(
                f"{low:>6.2f}"
                f"{high:>7.2f}"
                f"{auto_pct:>9.2f}%"
                f"{sandbox_pct:>10.2f}%"
                f"{low_mal:>11}"
                f"{mal_leak:>10.2f}%"
                f"{high_ben:>11}"
                f"{ben_leak:>10.2f}%"
                f"{auto_acc:>9.2f}%"
            )


if __name__ == "__main__":
    main()
