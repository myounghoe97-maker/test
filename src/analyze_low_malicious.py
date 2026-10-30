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
TOTAL_DIM = CORE_DIM + HISTOGRAM_DIM + BYTEENTROPY_DIM


def number(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    return default


def get_sha256(record):
    for key in ("sha256", "sha_256", "hash"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def extract_named_core_features(record):
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
    max_section_entropy = max(entropies) if entropies else 0.0

    overlay = section.get("overlay") or {}

    named = {
        "has_certificate": 1.0 if cert_count > 0 else 0.0,
        "has_pe_warning": 1.0 if warnings else 0.0,
        "general_entropy": number(general.get("entropy")),
        "import_api_count": float(import_api_count),
        "section_missing": section_missing,
        "section_count": float(len(sections)),
        "mean_section_entropy": mean_section_entropy,
        "max_section_entropy": max_section_entropy,
        "overlay_size": number(overlay.get("size")),
        "overlay_size_ratio": number(overlay.get("size_ratio")),
        "overlay_entropy": number(overlay.get("entropy")),
        "file_size": number(general.get("size")),
        "numstrings": number(strings.get("numstrings")),
        "string_avlength": number(strings.get("avlength")),
        "string_entropy": number(strings.get("entropy")),
        "printables": number(strings.get("printables")),
    }

    vector = np.asarray(
        list(named.values()),
        dtype=np.float32,
    )

    return vector, named


def normalize_256_feature(record, key):
    values = record.get(key)

    if not isinstance(values, list) or len(values) != 256:
        return np.zeros(256, dtype=np.float32), 1

    values = np.asarray(values, dtype=np.float32)

    total = float(values.sum())
    if total <= 0.0:
        return np.zeros(256, dtype=np.float32), 1

    return values / total, 0


def iter_valid_records(files):
    for file_path in files:
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

                yield file_path.name, line_number, record


def build_dataset(files, keep_metadata=False):
    rows = []
    labels = []
    metadata = []

    for file_name, line_number, record in iter_valid_records(files):
        core, named = extract_named_core_features(record)
        histogram, hist_missing = normalize_256_feature(
            record,
            "histogram",
        )
        byteentropy, be_missing = normalize_256_feature(
            record,
            "byteentropy",
        )

        feature_vector = np.concatenate(
            [core, histogram, byteentropy]
        ).astype(np.float32)

        rows.append(feature_vector)
        labels.append(record["label"])

        if keep_metadata:
            metadata.append({
                "file_name": file_name,
                "line_number": line_number,
                "sha256": get_sha256(record),
                "histogram_missing": hist_missing,
                "byteentropy_missing": be_missing,
                **named,
            })

    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        metadata,
    )


def safe_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if len(arr) else 0.0


def summarize_group(name, indices, metadata, scores):
    if len(indices) == 0:
        return None

    fields = [
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
        "histogram_missing",
        "byteentropy_missing",
    ]

    summary = {
        "group": name,
        "count": len(indices),
        "score_mean": safe_mean(scores[indices]),
        "score_min": float(np.min(scores[indices])),
        "score_max": float(np.max(scores[indices])),
    }

    for field in fields:
        summary[field] = safe_mean(
            [metadata[i][field] for i in indices]
        )

    return summary


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
    validation_files = [files[4]]

    print("[1/3] Train dataset")
    X_train, y_train, _ = build_dataset(
        train_files,
        keep_metadata=False,
    )

    print(f"Train rows      : {len(y_train):,}")
    print(f"Feature count   : {X_train.shape[-1]}")

    print()
    print("[2/3] Validation dataset")
    X_val, y_val, metadata = build_dataset(
        validation_files,
        keep_metadata=True,
    )

    print(f"Validation rows : {len(y_val):,}")

    print()
    print("[3/3] Random Forest")

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

    low_malicious_indices = np.where(
        (y_val == 1)
        & (scores < LOW_THRESHOLD)
    )[0]

    low_benign_indices = np.where(
        (y_val == 0)
        & (scores < LOW_THRESHOLD)
    )[0]

    all_malicious_indices = np.where(
        y_val == 1
    )[0]

    print()
    print("=" * 80)
    print("LOW-RISK MALICIOUS ANALYSIS")
    print("=" * 80)
    print(
        f"Threshold              : < {LOW_THRESHOLD:.2f}"
    )
    print(
        f"Low malicious count    : {len(low_malicious_indices):,}"
    )
    print(
        f"Low benign count       : {len(low_benign_indices):,}"
    )

    detail_path = (
        REPORT_DIR
        / "low_malicious_003_details.csv"
    )

    detail_fields = [
        "validation_index",
        "score",
        "file_name",
        "line_number",
        "sha256",
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
        "histogram_missing",
        "byteentropy_missing",
    ]

    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=detail_fields,
        )
        writer.writeheader()

        for index in low_malicious_indices:
            row = {
                "validation_index": int(index),
                "score": float(scores[index]),
                **metadata[index],
            }
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in detail_fields
                }
            )

    summaries = []

    for group_name, indices in [
        (
            "LOW_MALICIOUS",
            low_malicious_indices,
        ),
        (
            "LOW_BENIGN",
            low_benign_indices,
        ),
        (
            "ALL_MALICIOUS",
            all_malicious_indices,
        ),
    ]:
        summary = summarize_group(
            group_name,
            indices,
            metadata,
            scores,
        )

        if summary is not None:
            summaries.append(summary)

    summary_path = (
        REPORT_DIR
        / "low_malicious_003_group_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    print()
    print("LOW MALICIOUS CASES")
    print("-" * 80)

    for index in low_malicious_indices:
        item = metadata[index]

        print(
            f"idx={index:>6} | "
            f"score={scores[index]:.6f} | "
            f"cert={item['has_certificate']:.0f} | "
            f"warning={item['has_pe_warning']:.0f} | "
            f"entropy={item['general_entropy']:.3f} | "
            f"imports={item['import_api_count']:.0f} | "
            f"sections={item['section_count']:.0f} | "
            f"section_ent={item['mean_section_entropy']:.3f} | "
            f"overlay_ratio={item['overlay_size_ratio']:.4f}"
        )

    print()
    print("GROUP SUMMARY")
    print("-" * 80)

    for summary in summaries:
        print(
            f"{summary['group']:<15} "
            f"count={summary['count']:>6} | "
            f"score_mean={summary['score_mean']:.4f} | "
            f"cert={summary['has_certificate']:.3f} | "
            f"warning={summary['has_pe_warning']:.3f} | "
            f"entropy={summary['general_entropy']:.3f} | "
            f"imports={summary['import_api_count']:.1f} | "
            f"sections={summary['section_count']:.2f} | "
            f"section_ent={summary['mean_section_entropy']:.3f} | "
            f"overlay_ratio={summary['overlay_size_ratio']:.4f}"
        )

    print()
    print("Saved:")
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
