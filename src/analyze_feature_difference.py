import json
import statistics
from collections import Counter
from pathlib import Path


DATA_FILE = Path(
    r"C:\Users\user\Desktop\최종프로젝트\데이터셋"
    r"\train_sample"
    r"\2023-09-24_2023-09-30_Win32_train.jsonl"
)


def safe_median(values):
    if not values:
        return None
    return statistics.median(values)


def safe_mean(values):
    if not values:
        return None
    return statistics.fmean(values)


def format_number(value):
    if value is None:
        return "N/A"
    return f"{value:,.2f}"


def analyze():
    # 정상=0, 악성=1
    data = {
        0: {
            "size": [],
            "entropy": [],
            "numstrings": [],
            "string_entropy": [],
            "section_count": [],
            "import_dll_count": [],
            "import_api_count": [],
            "overlay_ratio": [],
            "cert_count": [],
            "warning_count": [],
        },
        1: {
            "size": [],
            "entropy": [],
            "numstrings": [],
            "string_entropy": [],
            "section_count": [],
            "import_dll_count": [],
            "import_api_count": [],
            "overlay_ratio": [],
            "cert_count": [],
            "warning_count": [],
        },
    }

    total = Counter()

    missing_header = Counter()
    missing_section = Counter()

    signed = Counter()
    has_warning = Counter()

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            label = record.get("label")

            if label not in (0, 1):
                continue

            total[label] += 1

            # -------------------------
            # General
            # -------------------------
            general = record.get("general") or {}

            if general.get("size") is not None:
                data[label]["size"].append(general["size"])

            if general.get("entropy") is not None:
                data[label]["entropy"].append(general["entropy"])

            # -------------------------
            # Strings
            # -------------------------
            strings = record.get("strings") or {}

            if strings.get("numstrings") is not None:
                data[label]["numstrings"].append(strings["numstrings"])

            if strings.get("entropy") is not None:
                data[label]["string_entropy"].append(strings["entropy"])

            # -------------------------
            # Header
            # -------------------------
            header = record.get("header")

            if not header:
                missing_header[label] += 1

            # -------------------------
            # Sections / Overlay
            # -------------------------
            section = record.get("section")

            if not section:
                missing_section[label] += 1

            else:
                sections = section.get("sections") or []
                data[label]["section_count"].append(len(sections))

                overlay = section.get("overlay") or {}

                if overlay.get("size_ratio") is not None:
                    data[label]["overlay_ratio"].append(
                        overlay["size_ratio"]
                    )

            # -------------------------
            # Imports
            # -------------------------
            imports = record.get("imports") or {}

            dll_count = len(imports)

            api_count = 0

            for api_list in imports.values():
                if isinstance(api_list, list):
                    api_count += len(api_list)

            data[label]["import_dll_count"].append(dll_count)
            data[label]["import_api_count"].append(api_count)

            # -------------------------
            # Authenticode
            # -------------------------
            authenticode = record.get("authenticode") or {}

            cert_count = authenticode.get("num_certs", 0) or 0

            data[label]["cert_count"].append(cert_count)

            if cert_count > 0:
                signed[label] += 1

            # -------------------------
            # PE warnings
            # -------------------------
            warnings = record.get("pefilewarnings") or []

            warning_count = len(warnings)

            data[label]["warning_count"].append(warning_count)

            if warning_count > 0:
                has_warning[label] += 1

            if sum(total.values()) % 5000 == 0:
                print(
                    f"[INFO] {sum(total.values()):,}건 처리 완료"
                )

    print()
    print("=" * 72)
    print("EMBER2024 Win32 - 정상 vs 악성 Feature 비교")
    print("=" * 72)

    print("\n[1] Header / Section 결측 96건 분석")

    for label, name in [(0, "정상"), (1, "악성")]:
        print(
            f"{name:<4} | "
            f"Header 결측: {missing_header[label]:,} | "
            f"Section 결측: {missing_section[label]:,}"
        )

    print("\n[2] 주요 수치형 Feature")
    print("-" * 72)

    feature_names = {
        "size": "파일 크기",
        "entropy": "전체 Entropy",
        "numstrings": "문자열 개수",
        "string_entropy": "문자열 Entropy",
        "section_count": "Section 개수",
        "import_dll_count": "Import DLL 개수",
        "import_api_count": "Import API 개수",
        "overlay_ratio": "Overlay 비율",
        "cert_count": "인증서 개수",
        "warning_count": "PE Warning 개수",
    }

    for key, display_name in feature_names.items():
        benign_values = data[0][key]
        malware_values = data[1][key]

        print(f"\n[{display_name}]")

        print(
            "정상 | "
            f"평균: {format_number(safe_mean(benign_values))} | "
            f"중앙값: {format_number(safe_median(benign_values))}"
        )

        print(
            "악성 | "
            f"평균: {format_number(safe_mean(malware_values))} | "
            f"중앙값: {format_number(safe_median(malware_values))}"
        )

    print("\n[3] 디지털 인증서 존재 비율")
    print("-" * 72)

    for label, name in [(0, "정상"), (1, "악성")]:
        ratio = (
            signed[label] / total[label] * 100
            if total[label]
            else 0
        )

        print(
            f"{name}: "
            f"{signed[label]:,}/{total[label]:,} "
            f"({ratio:.2f}%)"
        )

    print("\n[4] PE Warning 존재 비율")
    print("-" * 72)

    for label, name in [(0, "정상"), (1, "악성")]:
        ratio = (
            has_warning[label] / total[label] * 100
            if total[label]
            else 0
        )

        print(
            f"{name}: "
            f"{has_warning[label]:,}/{total[label]:,} "
            f"({ratio:.2f}%)"
        )

    print()
    print("=" * 72)
    print("[DONE] Feature 비교 완료")
    print("=" * 72)


def main():
    if not DATA_FILE.exists():
        print("[ERROR] 데이터 파일을 찾을 수 없습니다.")
        print(DATA_FILE)
        return

    analyze()


if __name__ == "__main__":
    main()