import json
from pathlib import Path


DATA_DIR = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample"
)


def new_stats():
    return {
        0: {
            "total": 0,
            "signed": 0,
            "warning": 0,
            "entropy_sum": 0.0,
            "entropy_count": 0,
            "api_sum": 0,
        },
        1: {
            "total": 0,
            "signed": 0,
            "warning": 0,
            "entropy_sum": 0.0,
            "entropy_count": 0,
            "api_sum": 0,
        },
    }


def analyze_file(file_path):
    stats = new_stats()
    json_errors = 0

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            label = record.get("label")

            if label not in (0, 1):
                continue

            s = stats[label]
            s["total"] += 1

            # 인증서
            authenticode = record.get("authenticode") or {}

            if (authenticode.get("num_certs", 0) or 0) > 0:
                s["signed"] += 1

            # PE Warning
            warnings = record.get("pefilewarnings") or []

            if warnings:
                s["warning"] += 1

            # 전체 Entropy
            general = record.get("general") or {}
            entropy = general.get("entropy")

            if entropy is not None:
                s["entropy_sum"] += entropy
                s["entropy_count"] += 1

            # Import API 개수
            imports = record.get("imports") or {}

            api_count = 0

            for api_list in imports.values():
                if isinstance(api_list, list):
                    api_count += len(api_list)

            s["api_sum"] += api_count

    return stats, json_errors


def percent(value, total):
    if total == 0:
        return 0.0

    return value / total * 100


def average(value, count):
    if count == 0:
        return 0.0

    return value / count


def main():
    files = sorted(DATA_DIR.glob("*_Win32_train.jsonl"))

    if not files:
        print("[ERROR] Train JSONL 파일이 없습니다.")
        return

    print("=" * 90)
    print("EMBER2024 Win32 - 주차별 Feature 안정성 검사")
    print("=" * 90)

    print(f"분석 파일 수: {len(files)}")

    for index, file_path in enumerate(files, start=1):

        print()
        print(f"[{index}주차] {file_path.name}")

        stats, json_errors = analyze_file(file_path)

        for label, name in [(0, "정상"), (1, "악성")]:
            s = stats[label]

            signed_ratio = percent(
                s["signed"],
                s["total"],
            )

            warning_ratio = percent(
                s["warning"],
                s["total"],
            )

            entropy_avg = average(
                s["entropy_sum"],
                s["entropy_count"],
            )

            api_avg = average(
                s["api_sum"],
                s["total"],
            )

            print(
                f"{name} | "
                f"N={s['total']:,} | "
                f"인증서={signed_ratio:6.2f}% | "
                f"Warning={warning_ratio:6.2f}% | "
                f"Entropy={entropy_avg:.3f} | "
                f"API평균={api_avg:.2f}"
            )

        print(f"JSON 오류: {json_errors:,}")

    print()
    print("=" * 90)
    print("[DONE] 주차별 안정성 검사 완료")
    print("=" * 90)


if __name__ == "__main__":
    main()
    