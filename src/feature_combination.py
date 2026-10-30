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


# ============================================================
# Feature 추출
# ============================================================

def number(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    return default


def extract_feature_groups(record):
    authenticode = record.get("authenticode") or {}
    warnings = record.get("pefilewarnings") or []
    general = record.get("general") or {}
    imports = record.get("imports") or {}

    cert_count = number(authenticode.get("num_certs"))

    import_api_count = 0
    for api_list in imports.values():
        if isinstance(api_list, list):
            import_api_count += len(api_list)

    groups = {}

    # V0
    groups["V0"] = [
        1.0 if cert_count > 0 else 0.0,
        1.0 if warnings else 0.0,
        number(general.get("entropy")),
        float(import_api_count),
    ]

    # General + Strings
    strings = record.get("strings") or {}

    groups["GENERAL_STRINGS"] = [
        number(general.get("size")),
        number(strings.get("numstrings")),
        number(strings.get("avlength")),
        number(strings.get("entropy")),
        number(strings.get("printables")),
    ]

    # PE Header
    header = record.get("header")
    header_missing = 1.0 if not header else 0.0
    header = header or {}

    coff = header.get("coff") or {}
    optional = header.get("optional") or {}

    groups["HEADER"] = [
        header_missing,
        number(coff.get("number_of_sections")),
        number(optional.get("sizeof_code")),
        number(optional.get("sizeof_image")),
        number(optional.get("sizeof_headers")),
        number(optional.get("address_of_entrypoint")),
    ]

    # Section + Overlay
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

    groups["SECTION_OVERLAY"] = [
        section_missing,
        float(len(sections)),
        mean_entropy,
        max_entropy,
        number(overlay.get("size")),
        number(overlay.get("size_ratio")),
        number(overlay.get("entropy")),
    ]

    # Import 구조
    groups["IMPORT"] = [
        float(len(imports)),
    ]

    # Authenticode 세부
    groups["AUTHENTICODE"] = [
        cert_count,
        number(authenticode.get("self_signed")),
        number(authenticode.get("parse_error")),
    ]

    # Warning 세부
    groups["WARNING"] = [
        float(len(warnings)),
    ]

    return groups


# ============================================================
# 데이터 로드
# ============================================================

def load_records(files):
    records = []

    for file_path in files:
        print(f"[LOAD] {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record.get("label") not in (0, 1):
                    continue

                records.append(record)

    return records


def build_dataset(records, selected_groups):
    X = []
    y = []

    for record in records:
        groups = extract_feature_groups(record)

        features = []
        for group_name in selected_groups:
            features.extend(groups[group_name])

        X.append(features)
        y.append(record["label"])

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int8),
    )


# ============================================================
# 모델 평가
# ============================================================

def evaluate(
    name,
    selected_groups,
    train_records,
    validation_records,
):
    print()
    print("=" * 78)
    print(name)
    print("Groups:", " + ".join(selected_groups))
    print("=" * 78)

    X_train, y_train = build_dataset(
        train_records,
        selected_groups,
    )

    X_val, y_val = build_dataset(
        validation_records,
        selected_groups,
    )

    print(f"Feature count: {X_train.shape[-1]}")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    probabilities = model.predict_proba(
        X_val_scaled
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
        "features": X_train.shape[-1],
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
        "fp": fp,
        "fn": fn,
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


# ============================================================
# 결과 요약
# ============================================================

def print_summary(results):
    print()
    print("=" * 118)
    print("SECTIONOVERLAY-BASED FEATURE COMBINATION SUMMARY")
    print("=" * 118)

    header = (
        f"{'Model':<40}"
        f"{'Feat':>6}"
        f"{'ROC-AUC':>11}"
        f"{'PR-AUC':>11}"
        f"{'Recall':>10}"
        f"{'FPR':>10}"
        f"{'FNR':>10}"
        f"{'FP':>9}"
        f"{'FN':>9}"
    )

    print(header)
    print("-" * 118)

    for result in results:
        print(
            f"{result['name']:<40}"
            f"{result['features']:>6}"
            f"{result['roc_auc']:>11.4f}"
            f"{result['pr_auc']:>11.4f}"
            f"{result['recall']:>10.4f}"
            f"{result['fpr']:>10.4f}"
            f"{result['fnr']:>10.4f}"
            f"{result['fp']:>9,}"
            f"{result['fn']:>9,}"
        )

    print("=" * 118)


def main():
    files = sorted(
        DATA_DIR.glob("*_Win32_train.jsonl")
    )

    if len(files) < 5:
        print("[ERROR] train_sample 폴더에 최소 5개의 Win32 train JSONL 파일이 필요합니다.")
        return

    train_files = files[:4]
    validation_files = [files[4]]

    print("[1/2] Train 데이터 로딩")

    train_records = load_records(
        train_files
    )

    print(
        f"Train: {len(train_records):,}"
    )

    print("\n[2/2] Validation 데이터 로딩")

    validation_records = load_records(
        validation_files
    )

    print(
        f"Validation: {len(validation_records):,}"
    )

    experiments = [
        (
            "V0",
            ["V0"],
        ),
        (
            "V0 + SectionOverlay",
            ["V0", "SECTION_OVERLAY"],
        ),
        (
            "V0 + SectionOverlay + GeneralStrings",
            ["V0", "SECTION_OVERLAY", "GENERAL_STRINGS"],
        ),
        (
            "V0 + SectionOverlay + Header",
            ["V0", "SECTION_OVERLAY", "HEADER"],
        ),
        (
            "V0 + SectionOverlay + Import",
            ["V0", "SECTION_OVERLAY", "IMPORT"],
        ),
        (
            "V0 + SectionOverlay + Authenticode",
            ["V0", "SECTION_OVERLAY", "AUTHENTICODE"],
        ),
        (
            "V0 + SectionOverlay + Warning",
            ["V0", "SECTION_OVERLAY", "WARNING"],
        ),
        (
            "V1 ALL",
            [
                "V0",
                "GENERAL_STRINGS",
                "HEADER",
                "SECTION_OVERLAY",
                "IMPORT",
                "AUTHENTICODE",
                "WARNING",
            ],
        ),
    ]

    results = []

    for name, groups in experiments:
        result = evaluate(
            name,
            groups,
            train_records,
            validation_records,
        )

        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
