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

CORE_DIM = 16
HISTOGRAM_DIM = 256
BYTEENTROPY_DIM = 256

CORE_NAMES = [
    "has_certificate",
    "has_pe_warning",
    "general_entropy",
    "import_api_count",
    "section_missing",
    "section_count",
    "mean_section_entropy",
    "max_section_entropy",
    "overlay_size",
    "overlay_size_ratio",
    "overlay_entropy",
    "file_size",
    "numstrings",
    "string_avlength",
    "string_entropy",
    "printables",
]


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

    entropies = []
    for item in sections:
        value = item.get("entropy")
        if isinstance(value, (int, float)):
            entropies.append(float(value))

    mean_section_entropy = (
        sum(entropies) / len(entropies)
        if entropies
        else 0.0
    )

    max_section_entropy = (
        max(entropies)
        if entropies
        else 0.0
    )

    overlay = section.get("overlay") or {}

    values = [
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
    ]

    return np.asarray(values, dtype=np.float32)


def normalize_256(record, key):
    values = record.get(key)

    if not isinstance(values, list) or len(values) != 256:
        return np.zeros(256, dtype=np.float32)

    arr = np.asarray(values, dtype=np.float32)

    total = float(arr.sum())
    if total <= 0.0:
        return np.zeros(256, dtype=np.float32)

    return arr / total


def build_feature_vector(record):
    core = extract_core(record)
    histogram = normalize_256(record, "histogram")
    byteentropy = normalize_256(record, "byteentropy")

    return np.concatenate(
        [core, histogram, byteentropy]
    ).astype(np.float32)


def load_records(file_path, dedup=False):
    records = []
    seen_sha = set()

    print(f"[LOAD] {file_path.name}")

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

            if dedup:
                if not sha:
                    continue
                if sha in seen_sha:
                    continue
                seen_sha.add(sha)

            records.append(
                {
                    "line_number": line_number,
                    "sha256": sha,
                    "label": label,
                    "record": record,
                }
            )

    return records


def z_distance(value, mean, std):
    if std <= 1e-12:
        return 0.0
    return abs((value - mean) / std)


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        DATA_DIR.glob("*_Win32_train.jsonl")
    )

    if len(files) < 5:
        print("[ERROR] train_sample 폴더에 최소 5개의 JSONL이 필요합니다.")
        return

    train_files = files[:4]
    validation_file = files[4]

    print("[1/4] Train load")

    train_X = []
    train_y = []

    for file_path in train_files:
        records = load_records(file_path, dedup=False)

        for item in records:
            train_X.append(
                build_feature_vector(item["record"])
            )
            train_y.append(item["label"])

    X_train = np.asarray(train_X, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.int8)

    print(f"Train rows: {len(y_train):,}")

    print()
    print("[2/4] Validation dedup load")

    val_records = load_records(
        validation_file,
        dedup=True,
    )

    X_val = np.asarray(
        [
            build_feature_vector(item["record"])
            for item in val_records
        ],
        dtype=np.float32,
    )

    y_val = np.asarray(
        [item["label"] for item in val_records],
        dtype=np.int8,
    )

    print(f"Validation unique rows: {len(y_val):,}")

    print()
    print("[3/4] Random Forest")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    scores = model.predict_proba(
        X_val
    )[:, 1]

    low_mal_idx = np.where(
        (y_val == 1)
        & (scores < LOW_THRESHOLD)
    )[0]

    low_ben_idx = np.where(
        (y_val == 0)
        & (scores < LOW_THRESHOLD)
    )[0]

    all_mal_idx = np.where(
        y_val == 1
    )[0]

    print(f"Unique low malicious: {len(low_mal_idx):,}")
    print(f"Low benign         : {len(low_ben_idx):,}")

    if len(low_mal_idx) == 0:
        print("[INFO] low malicious case 없음")
        return

    print()
    print("[4/4] Group distance analysis")

    core_val = X_val[:, :CORE_DIM]
    hist_val = X_val[
        :,
        CORE_DIM:CORE_DIM + HISTOGRAM_DIM
    ]
    byte_val = X_val[
        :,
        CORE_DIM + HISTOGRAM_DIM:
    ]

    low_ben_core_mean = core_val[low_ben_idx].mean(axis=0)
    low_ben_core_std = core_val[low_ben_idx].std(axis=0)

    all_mal_core_mean = core_val[all_mal_idx].mean(axis=0)
    all_mal_core_std = core_val[all_mal_idx].std(axis=0)

    low_ben_hist_mean = hist_val[low_ben_idx].mean(axis=0)
    all_mal_hist_mean = hist_val[all_mal_idx].mean(axis=0)

    low_ben_byte_mean = byte_val[low_ben_idx].mean(axis=0)
    all_mal_byte_mean = byte_val[all_mal_idx].mean(axis=0)

    output_rows = []

    print()
    print("=" * 128)
    print("UNIQUE LOW MALICIOUS - DISTANCE ANALYSIS")
    print("=" * 128)

    for idx in low_mal_idx:
        item = val_records[idx]

        core = core_val[idx]
        hist = hist_val[idx]
        byte = byte_val[idx]

        core_dist_to_benign = float(
            np.linalg.norm(
                (core - low_ben_core_mean)
                / np.where(
                    low_ben_core_std > 1e-12,
                    low_ben_core_std,
                    1.0,
                )
            )
        )

        core_dist_to_malicious = float(
            np.linalg.norm(
                (core - all_mal_core_mean)
                / np.where(
                    all_mal_core_std > 1e-12,
                    all_mal_core_std,
                    1.0,
                )
            )
        )

        hist_dist_to_benign = float(
            np.linalg.norm(
                hist - low_ben_hist_mean
            )
        )

        hist_dist_to_malicious = float(
            np.linalg.norm(
                hist - all_mal_hist_mean
            )
        )

        byte_dist_to_benign = float(
            np.linalg.norm(
                byte - low_ben_byte_mean
            )
        )

        byte_dist_to_malicious = float(
            np.linalg.norm(
                byte - all_mal_byte_mean
            )
        )

        core_closer = (
            "BENIGN"
            if core_dist_to_benign < core_dist_to_malicious
            else "MALICIOUS"
        )

        hist_closer = (
            "BENIGN"
            if hist_dist_to_benign < hist_dist_to_malicious
            else "MALICIOUS"
        )

        byte_closer = (
            "BENIGN"
            if byte_dist_to_benign < byte_dist_to_malicious
            else "MALICIOUS"
        )

        output_rows.append({
            "validation_index": int(idx),
            "sha256": item["sha256"],
            "score": float(scores[idx]),
            "core_closer": core_closer,
            "core_dist_benign": core_dist_to_benign,
            "core_dist_malicious": core_dist_to_malicious,
            "hist_closer": hist_closer,
            "hist_dist_benign": hist_dist_to_benign,
            "hist_dist_malicious": hist_dist_to_malicious,
            "byte_closer": byte_closer,
            "byte_dist_benign": byte_dist_to_benign,
            "byte_dist_malicious": byte_dist_to_malicious,
        })

        print(
            f"idx={idx:>6} | "
            f"score={scores[idx]:.6f} | "
            f"CORE={core_closer:<9} | "
            f"HIST={hist_closer:<9} | "
            f"BYTE={byte_closer:<9}"
        )

    print()
    print("=" * 128)
    print("CORE FEATURE DIRECTION")
    print("=" * 128)

    for feature_index, feature_name in enumerate(CORE_NAMES):
        low_mal_mean = float(
            core_val[
                low_mal_idx,
                feature_index
            ].mean()
        )

        benign_mean = float(
            low_ben_core_mean[
                feature_index
            ]
        )

        malicious_mean = float(
            all_mal_core_mean[
                feature_index
            ]
        )

        d_b = abs(
            low_mal_mean - benign_mean
        )

        d_m = abs(
            low_mal_mean - malicious_mean
        )

        closer = (
            "BENIGN"
            if d_b < d_m
            else "MALICIOUS"
        )

        print(
            f"{feature_name:<24} "
            f"lowMal={low_mal_mean:>10.4f} | "
            f"lowBen={benign_mean:>10.4f} | "
            f"allMal={malicious_mean:>10.4f} | "
            f"closer={closer}"
        )

    output_path = (
        REPORT_DIR
        / "low_malicious_003_distance_analysis.csv"
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print("Saved:")
    print(output_path)


if __name__ == "__main__":
    main()
