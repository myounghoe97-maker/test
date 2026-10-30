import json
from collections import Counter
from pathlib import Path


DATA_FILE = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋"
    r"\train_sample"
    r"\2023-09-24_2023-09-30_Win32_train.jsonl"
)


def analyze_jsonl(file_path: Path) -> None:
    total = 0
    labels = Counter()
    file_types = Counter()

    missing = Counter()
    json_errors = 0

    print(f"[INFO] 분석 시작: {file_path.name}")

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            total += 1

            # -------------------------
            # 기본 정보
            # -------------------------
            label = record.get("label")
            file_type = record.get("file_type")

            labels[label] += 1
            file_types[file_type] += 1

            # -------------------------
            # 주요 Feature 존재 여부
            # -------------------------
            if label is None:
                missing["label"] += 1

            if not record.get("general"):
                missing["general"] += 1

            if not record.get("header"):
                missing["header"] += 1

            if not record.get("section"):
                missing["section"] += 1

            if record.get("imports") is None:
                missing["imports"] += 1

            if not record.get("strings"):
                missing["strings"] += 1

            if record.get("authenticode") is None:
                missing["authenticode"] += 1

            if record.get("histogram") is None:
                missing["histogram"] += 1

            if record.get("byteentropy") is None:
                missing["byteentropy"] += 1

            # 진행 상황 표시
            if total % 5000 == 0:
                print(f"[INFO] {total:,}건 처리 완료")

    # -------------------------
    # 결과 출력
    # -------------------------

    benign = labels.get(0, 0)
    malicious = labels.get(1, 0)
    other = total - benign - malicious

    print()
    print("=" * 60)
    print("EMBER2024 Win32 Train - 첫 주 데이터 검사")
    print("=" * 60)

    print(f"전체 데이터        : {total:,}")
    print(f"JSON 오류          : {json_errors:,}")

    print()
    print("[Label]")
    print(f"정상 (label=0)     : {benign:,}")
    print(f"악성 (label=1)     : {malicious:,}")
    print(f"기타/결측          : {other:,}")

    if total > 0:
        print(f"정상 비율          : {benign / total * 100:.2f}%")
        print(f"악성 비율          : {malicious / total * 100:.2f}%")

    print()
    print("[File Type]")

    for file_type, count in file_types.items():
        print(f"{file_type}: {count:,}")

    print()
    print("[주요 데이터 결측]")

    feature_names = [
        "label",
        "general",
        "header",
        "section",
        "imports",
        "strings",
        "authenticode",
        "histogram",
        "byteentropy",
    ]

    for feature in feature_names:
        count = missing[feature]

        if total > 0:
            ratio = count / total * 100
        else:
            ratio = 0

        print(f"{feature:<15}: {count:>8,} ({ratio:.2f}%)")

    print()
    print("=" * 60)
    print("[DONE] 분석 완료")
    print("=" * 60)


def main() -> None:

    if not DATA_FILE.exists():
        print("[ERROR] 데이터 파일을 찾을 수 없습니다.")
        print(DATA_FILE)
        return

    analyze_jsonl(DATA_FILE)


if __name__ == "__main__":
    main()