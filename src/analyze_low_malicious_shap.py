import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


DATA_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample"
)

REPORT_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\malware_detection\reports"
)

LOW_THRESHOLD = 0.03

CORE_NAMES = [
    "core_has_certificate",
    "core_has_pe_warning",
    "core_general_entropy",
    "core_import_api_count",
    "core_section_missing",
    "core_section_count",
    "core_mean_section_entropy",
    "core_max_section_entropy",
    "core_overlay_size",
    "core_overlay_size_ratio",
    "core_overlay_entropy",
    "core_file_size",
    "core_numstrings",
    "core_string_avlength",
    "core_string_entropy",
    "core_printables",
]

HIST_NAMES = [f"hist_byte_{i:03d}" for i in range(256)]
BYTE_NAMES = [f"byteentropy_bin_{i:03d}" for i in range(256)]
FEATURE_NAMES = CORE_NAMES + HIST_NAMES + BYTE_NAMES

CORE_DIM = len(CORE_NAMES)
HIST_DIM = len(HIST_NAMES)


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


def load_train(files):
    X, y = [], []

    for file_path in files:
        print(f"[TRAIN LOAD] {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                label = record.get("label")
                if label not in (0, 1):
                    continue

                X.append(build_vector(record))
                y.append(label)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)


def load_validation_dedup(file_path):
    X, y, metadata = [], [], []
    seen_sha = set()

    print(f"[VAL LOAD] {file_path.name}")

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            label = record.get("label")
            if label not in (0, 1):
                continue

            sha = get_sha256(record)
            if not sha or sha in seen_sha:
                continue

            seen_sha.add(sha)

            X.append(build_vector(record))
            y.append(label)
            metadata.append(
                {
                    "sha256": sha,
                    "line_number": line_number,
                }
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int8),
        metadata,
    )


def group_name(feature_index):
    if feature_index < CORE_DIM:
        return "CORE"
    if feature_index < CORE_DIM + HIST_DIM:
        return "HIST"
    return "BYTE"


def normalize_shap_output(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) < 2:
            raise RuntimeError("Class 1 SHAP output not found.")
        return np.asarray(shap_values[1])

    values = np.asarray(shap_values)

    if values.ndim == 3 and values.shape[-1] >= 2:
        return values[:, :, 1]

    if values.ndim == 2:
        return values

    raise RuntimeError(
        f"Unsupported SHAP output shape: {values.shape}"
    )


def main():
    try:
        import shap
    except ImportError:
        print("[ERROR] shap 패키지가 설치되어 있지 않습니다.")
        print(
            r"설치 명령: .\.venv\Scripts\python.exe -m pip install shap"
        )
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*_Win32_train.jsonl"))
    if len(files) < 5:
        print("[ERROR] train_sample 폴더에 최소 5개의 JSONL이 필요합니다.")
        return

    train_files = files[:4]
    validation_file = files[4]

    print("[1/4] Dataset build")
    X_train, y_train = load_train(train_files)
    X_val, y_val, metadata = load_validation_dedup(validation_file)

    print(f"Train rows        : {len(y_train):,}")
    print(f"Validation unique : {len(y_val):,}")
    print(f"Feature count     : {X_train.shape[-1]}")

    print()
    print("[2/4] Random Forest training")

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

    target_indices = np.where(
        (y_val == 1) & (scores < LOW_THRESHOLD)
    )[0]

    print(f"Unique low malicious: {len(target_indices):,}")

    if len(target_indices) == 0:
        print("[INFO] 분석 대상 없음")
        return

    X_target = X_val[target_indices]

    print()
    print("[3/4] SHAP calculation")

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(
        X_target,
        check_additivity=False,
    )
    class1_shap = normalize_shap_output(raw_shap_values)

    if class1_shap.shape != X_target.shape:
        raise RuntimeError(
            f"SHAP shape mismatch: expected {X_target.shape}, "
            f"got {class1_shap.shape}"
        )

    print()
    print("[4/4] Per-sample explanation")

    detail_rows = []
    group_rows = []

    print()
    print("=" * 160)
    print("SHAP - UNIQUE LOW MALICIOUS TOP FEATURES")
    print("=" * 160)

    for local_idx, validation_idx in enumerate(target_indices):
        shap_row = class1_shap[local_idx]
        feature_row = X_target[local_idx]

        top_indices = np.argsort(
            np.abs(shap_row)
        )[::-1][:15]

        print()
        print(
            f"idx={validation_idx} | "
            f"score={scores[validation_idx]:.6f} | "
            f"sha256={metadata[validation_idx]['sha256']}"
        )
        print(
            f"{'Rank':>4} "
            f"{'Group':<6} "
            f"{'Feature':<30} "
            f"{'Value':>12} "
            f"{'SHAP':>12} "
            f"{'Direction':<10}"
        )
        print("-" * 90)

        for rank, feature_index in enumerate(top_indices, start=1):
            shap_value = float(shap_row[feature_index])
            direction = "MALICIOUS" if shap_value > 0 else "BENIGN"
            group = group_name(int(feature_index))
            feature_name = FEATURE_NAMES[feature_index]
            feature_value = float(feature_row[feature_index])

            print(
                f"{rank:>4} "
                f"{group:<6} "
                f"{feature_name:<30} "
                f"{feature_value:>12.6f} "
                f"{shap_value:>12.6f} "
                f"{direction:<10}"
            )

            detail_rows.append(
                {
                    "validation_index": int(validation_idx),
                    "sha256": metadata[validation_idx]["sha256"],
                    "score": float(scores[validation_idx]),
                    "rank": rank,
                    "group": group,
                    "feature": feature_name,
                    "feature_value": feature_value,
                    "shap_value": shap_value,
                    "direction": direction,
                }
            )

        core_sum = float(shap_row[:CORE_DIM].sum())
        hist_sum = float(
            shap_row[
                CORE_DIM:
                CORE_DIM + HIST_DIM
            ].sum()
        )
        byte_sum = float(
            shap_row[
                CORE_DIM + HIST_DIM:
            ].sum()
        )

        group_rows.append(
            {
                "validation_index": int(validation_idx),
                "sha256": metadata[validation_idx]["sha256"],
                "score": float(scores[validation_idx]),
                "core_shap_sum": core_sum,
                "hist_shap_sum": hist_sum,
                "byte_shap_sum": byte_sum,
                "total_shap_sum": core_sum + hist_sum + byte_sum,
            }
        )

    print()
    print("=" * 160)
    print("SHAP GROUP CONTRIBUTION SUMMARY")
    print("=" * 160)

    print(
        f"{'idx':>6}"
        f"{'score':>10}"
        f"{'CORE':>12}"
        f"{'HIST':>12}"
        f"{'BYTE':>12}"
        f"{'TOTAL':>12}"
    )
    print("-" * 80)

    for row in group_rows:
        print(
            f"{row['validation_index']:>6}"
            f"{row['score']:>10.4f}"
            f"{row['core_shap_sum']:>12.5f}"
            f"{row['hist_shap_sum']:>12.5f}"
            f"{row['byte_shap_sum']:>12.5f}"
            f"{row['total_shap_sum']:>12.5f}"
        )

    detail_path = REPORT_DIR / "low_malicious_shap_top_features.csv"
    with detail_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(detail_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    group_path = REPORT_DIR / "low_malicious_shap_group_summary.csv"
    with group_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(group_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(group_rows)

    print()
    print("Saved:")
    print(detail_path)
    print(group_path)


if __name__ == "__main__":
    main()
