import json
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)
from sklearn.preprocessing import StandardScaler

from feature_extractor import (
    V0_FEATURE_NAMES,
    V1_FEATURE_NAMES,
    extract_v0,
    extract_v1,
)


DATA_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample"
)


def load_data(files, extractor):
    X = []
    y = []

    for file_path in files:
        print(f"[LOAD] {file_path.name}")

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                label = record.get("label")

                if label not in (0, 1):
                    continue

                X.append(
                    extractor(record)
                )

                y.append(label)

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int8),
    )


def evaluate(
    name,
    train_files,
    validation_file,
    extractor,
    feature_names,
):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    X_train, y_train = load_data(
        train_files,
        extractor,
    )

    X_val, y_val = load_data(
        [validation_file],
        extractor,
    )

    print(
        f"Feature 수 : {len(feature_names)}"
    )

    print(
        f"Train      : {len(y_train):,}"
    )

    print(
        f"Validation : {len(y_val):,}"
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

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

    pr_precision, pr_recall, _ = (
        precision_recall_curve(
            y_val,
            probabilities,
        )
    )

    pr_auc = auc(
        pr_recall,
        pr_precision,
    )

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    result = {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    print()
    print("[RESULT]")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print()
    print(f"TN : {tn:,}")
    print(f"FP : {fp:,}")
    print(f"FN : {fn:,}")
    print(f"TP : {tp:,}")

    print()

    print(f"FPR : {fpr:.4%}")
    print(f"FNR : {fnr:.4%}")

    return result


def print_comparison(v0, v1):
    print()
    print("=" * 70)
    print("V0 vs V1")
    print("=" * 70)

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("FPR", "fpr"),
        ("FNR", "fnr"),
    ]

    for display_name, key in metrics:
        old = v0[key]
        new = v1[key]

        difference = new - old

        print(
            f"{display_name:<10} "
            f"V0={old:.4f} | "
            f"V1={new:.4f} | "
            f"차이={difference:+.4f}"
        )

    print()
    print(
        f"FN 변화: "
        f"{v0['fn']:,} → {v1['fn']:,} "
        f"({v1['fn'] - v0['fn']:+,})"
    )

    print(
        f"FP 변화: "
        f"{v0['fp']:,} → {v1['fp']:,} "
        f"({v1['fp'] - v0['fp']:+,})"
    )


def main():
    files = sorted(
        DATA_DIR.glob(
            "*_Win32_train.jsonl"
        )
    )

    if len(files) < 5:
        print(
            "[ERROR] 5주 데이터가 필요합니다."
        )
        return

    train_files = files[:4]
    validation_file = files[4]

    print("[TRAIN]")
    for file_path in train_files:
        print(file_path.name)

    print()
    print("[VALIDATION]")
    print(validation_file.name)

    v0_result = evaluate(
        "V0 - 4 Features",
        train_files,
        validation_file,
        extract_v0,
        V0_FEATURE_NAMES,
    )

    v1_result = evaluate(
        "V1 - Expanded Features",
        train_files,
        validation_file,
        extract_v1,
        V1_FEATURE_NAMES,
    )

    print_comparison(
        v0_result,
        v1_result,
    )


if __name__ == "__main__":
    main()