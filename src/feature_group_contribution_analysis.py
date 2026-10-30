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
HIST_DIM = 256
BYTE_DIM = 256

GROUPS = {
    "ALL_528": ("CORE", "HIST", "BYTE"),
    "NO_CORE": ("HIST", "BYTE"),
    "NO_HIST": ("CORE", "BYTE"),
    "NO_BYTE": ("CORE", "HIST"),
    "CORE_ONLY": ("CORE",),
    "HIST_ONLY": ("HIST",),
    "BYTE_ONLY": ("BYTE",),
}


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


def extract_groups(record):
    return {
        "CORE": extract_core(record),
        "HIST": normalize_256(record, "histogram"),
        "BYTE": normalize_256(record, "byteentropy"),
    }


def make_vector(group_data, selected_groups):
    return np.concatenate(
        [group_data[name] for name in selected_groups]
    ).astype(np.float32)


def load_train_records(files):
    records = []

    for file_path in files:
        print(f"[TRAIN LOAD] {file_path.name}")

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


def load_validation_dedup(file_path):
    records = []
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
            if not sha:
                continue

            if sha in seen_sha:
                continue

            seen_sha.add(sha)

            records.append({
                "record": record,
                "sha256": sha,
                "line_number": line_number,
                "label": label,
            })

    return records


def build_xy(records, selected_groups, validation=False):
    X = []
    y = []

    for item in records:
        record = item["record"] if validation else item
        group_data = extract_groups(record)

        X.append(
            make_vector(
                group_data,
                selected_groups,
            )
        )

        y.append(
            item["label"] if validation else record["label"]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int8),
    )


def train_and_score(
    train_records,
    val_records,
    selected_groups,
):
    X_train, y_train = build_xy(
        train_records,
        selected_groups,
        validation=False,
    )

    X_val, y_val = build_xy(
        val_records,
        selected_groups,
        validation=True,
    )

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

    return y_val, scores


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

    print("[1/3] Load datasets")

    train_records = load_train_records(
        train_files
    )

    val_records = load_validation_dedup(
        validation_file
    )

    print(f"Train rows          : {len(train_records):,}")
    print(f"Validation unique   : {len(val_records):,}")

    print()
    print("[2/3] Baseline ALL_528")

    y_val, baseline_scores = train_and_score(
        train_records,
        val_records,
        GROUPS["ALL_528"],
    )

    target_indices = np.where(
        (y_val == 1)
        & (baseline_scores < LOW_THRESHOLD)
    )[0]

    print(f"Unique low malicious: {len(target_indices):,}")

    if len(target_indices) == 0:
        print("[INFO] 분석 대상 없음")
        return

    results = {}

    results["ALL_528"] = baseline_scores

    print()
    print("[3/3] Group ablation scoring")

    for name, selected_groups in GROUPS.items():
        if name == "ALL_528":
            continue

        print(
            f"[MODEL] {name} -> "
            + "+".join(selected_groups)
        )

        _, scores = train_and_score(
            train_records,
            val_records,
            selected_groups,
        )

        results[name] = scores

    print()
    print("=" * 150)
    print("FEATURE GROUP CONTRIBUTION - UNIQUE LOW MALICIOUS")
    print("=" * 150)

    header = (
        f"{'idx':>6}"
        f"{'ALL':>10}"
        f"{'NO_CORE':>10}"
        f"{'NO_HIST':>10}"
        f"{'NO_BYTE':>10}"
        f"{'CORE':>10}"
        f"{'HIST':>10}"
        f"{'BYTE':>10}"
    )

    print(header)
    print("-" * 150)

    output_rows = []

    for idx in target_indices:
        row = {
            "validation_index": int(idx),
            "sha256": val_records[idx]["sha256"],
            "all_528": float(results["ALL_528"][idx]),
            "no_core": float(results["NO_CORE"][idx]),
            "no_hist": float(results["NO_HIST"][idx]),
            "no_byte": float(results["NO_BYTE"][idx]),
            "core_only": float(results["CORE_ONLY"][idx]),
            "hist_only": float(results["HIST_ONLY"][idx]),
            "byte_only": float(results["BYTE_ONLY"][idx]),
        }

        output_rows.append(row)

        print(
            f"{idx:>6}"
            f"{row['all_528']:>10.4f}"
            f"{row['no_core']:>10.4f}"
            f"{row['no_hist']:>10.4f}"
            f"{row['no_byte']:>10.4f}"
            f"{row['core_only']:>10.4f}"
            f"{row['hist_only']:>10.4f}"
            f"{row['byte_only']:>10.4f}"
        )

    print()
    print("=" * 150)
    print("AVERAGE SCORE - 9 UNIQUE LOW MALICIOUS")
    print("=" * 150)

    for key in [
        "ALL_528",
        "NO_CORE",
        "NO_HIST",
        "NO_BYTE",
        "CORE_ONLY",
        "HIST_ONLY",
        "BYTE_ONLY",
    ]:
        mean_score = float(
            np.mean(
                results[key][target_indices]
            )
        )

        print(
            f"{key:<12} : {mean_score:.4f}"
        )

    print()
    print("=" * 150)
    print("AVERAGE SCORE DELTA VS ALL_528")
    print("=" * 150)

    baseline_mean = float(
        np.mean(
            results["ALL_528"][target_indices]
        )
    )

    for key in [
        "NO_CORE",
        "NO_HIST",
        "NO_BYTE",
    ]:
        mean_score = float(
            np.mean(
                results[key][target_indices]
            )
        )

        delta = mean_score - baseline_mean

        print(
            f"{key:<12} : "
            f"{mean_score:.4f} "
            f"(delta {delta:+.4f})"
        )

    output_path = (
        REPORT_DIR
        / "low_malicious_group_contribution.csv"
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
