import csv
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample"
)

REPORT_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\malware_detection\reports"
)

LOW_MALICIOUS_DETAILS = REPORT_DIR / "low_malicious_003_details.csv"

SHA_KEYS = ("sha256", "sha_256", "hash")


def get_sha256(record):
    for key in SHA_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def iter_valid_records(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("label") not in (0, 1):
                continue

            yield line_number, record


def load_sha_index(files):
    """
    Returns:
        sha_to_rows:
            sha -> list of {file_name, line_number, label}
        missing_sha_rows:
            rows without usable SHA-256
    """
    sha_to_rows = {}
    missing_sha_rows = []

    for file_path in files:
        print(f"[SCAN] {file_path.name}")

        for line_number, record in iter_valid_records(file_path):
            sha = get_sha256(record)

            row = {
                "file_name": file_path.name,
                "line_number": line_number,
                "label": int(record["label"]),
            }

            if not sha:
                missing_sha_rows.append(row)
                continue

            sha_to_rows.setdefault(sha, []).append(row)

    return sha_to_rows, missing_sha_rows


def load_low_malicious_details():
    if not LOW_MALICIOUS_DETAILS.exists():
        raise FileNotFoundError(
            f"Missing file: {LOW_MALICIOUS_DETAILS}"
        )

    rows = []

    with LOW_MALICIOUS_DETAILS.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            sha = (row.get("sha256") or "").strip().lower()

            rows.append({
                "validation_index": row.get("validation_index", ""),
                "score": row.get("score", ""),
                "file_name": row.get("file_name", ""),
                "line_number": row.get("line_number", ""),
                "sha256": sha,
            })

    return rows


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

    print("=" * 88)
    print("1. LOW MALICIOUS 18 CASES - SHA-256 UNIQUENESS")
    print("=" * 88)

    low_rows = load_low_malicious_details()

    low_shas = [
        row["sha256"]
        for row in low_rows
        if row["sha256"]
    ]

    sha_counts = Counter(low_shas)

    duplicate_low_shas = {
        sha: count
        for sha, count in sha_counts.items()
        if count > 1
    }

    print(f"Low malicious rows        : {len(low_rows):,}")
    print(f"Rows with SHA-256         : {len(low_shas):,}")
    print(f"Rows without SHA-256      : {len(low_rows) - len(low_shas):,}")
    print(f"Unique SHA-256 count      : {len(sha_counts):,}")
    print(f"Duplicated SHA-256 groups : {len(duplicate_low_shas):,}")

    if duplicate_low_shas:
        print()
        print("[LOW MALICIOUS DUPLICATED SHA-256]")
        for sha, count in sorted(
            duplicate_low_shas.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"{sha} | count={count}")

    print()
    print("=" * 88)
    print("2. TRAIN / VALIDATION SHA-256 OVERLAP")
    print("=" * 88)

    train_index, train_missing = load_sha_index(
        train_files
    )

    val_index, val_missing = load_sha_index(
        [validation_file]
    )

    train_shas = set(train_index)
    val_shas = set(val_index)

    overlap = train_shas & val_shas

    print()
    print(f"Train unique SHA-256      : {len(train_shas):,}")
    print(f"Validation unique SHA-256 : {len(val_shas):,}")
    print(f"Train missing SHA rows    : {len(train_missing):,}")
    print(f"Validation missing SHA    : {len(val_missing):,}")
    print(f"Train/Validation overlap  : {len(overlap):,}")

    overlap_label_conflicts = 0

    overlap_rows = []

    for sha in sorted(overlap):
        train_labels = sorted(
            set(row["label"] for row in train_index[sha])
        )

        val_labels = sorted(
            set(row["label"] for row in val_index[sha])
        )

        conflict = train_labels != val_labels

        if conflict:
            overlap_label_conflicts += 1

        overlap_rows.append({
            "sha256": sha,
            "train_count": len(train_index[sha]),
            "validation_count": len(val_index[sha]),
            "train_labels": ",".join(map(str, train_labels)),
            "validation_labels": ",".join(map(str, val_labels)),
            "label_conflict": int(conflict),
        })

    print(f"Overlap label conflicts   : {overlap_label_conflicts:,}")

    overlap_path = (
        REPORT_DIR
        / "train_validation_sha256_overlap.csv"
    )

    with overlap_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "sha256",
            "train_count",
            "validation_count",
            "train_labels",
            "validation_labels",
            "label_conflict",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(overlap_rows)

    print()
    print("=" * 88)
    print("3. LOW MALICIOUS SHA-256 VS TRAIN")
    print("=" * 88)

    low_in_train = []

    for row in low_rows:
        sha = row["sha256"]

        if sha and sha in train_index:
            train_labels = sorted(
                set(item["label"] for item in train_index[sha])
            )

            low_in_train.append({
                **row,
                "train_count": len(train_index[sha]),
                "train_labels": ",".join(
                    map(str, train_labels)
                ),
            })

    print(
        f"Low malicious rows whose SHA exists in Train: "
        f"{len(low_in_train):,}"
    )

    low_train_path = (
        REPORT_DIR
        / "low_malicious_003_train_overlap.csv"
    )

    with low_train_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "validation_index",
            "score",
            "file_name",
            "line_number",
            "sha256",
            "train_count",
            "train_labels",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(low_in_train)

    low_unique_path = (
        REPORT_DIR
        / "low_malicious_003_sha256_summary.csv"
    )

    with low_unique_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "sha256",
            "count_in_low_malicious",
        ])

        for sha, count in sorted(
            sha_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            writer.writerow([
                sha,
                count,
            ])

    print()
    print("Saved:")
    print(low_unique_path)
    print(overlap_path)
    print(low_train_path)


if __name__ == "__main__":
    main()
