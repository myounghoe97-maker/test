from typing import Any


V0_FEATURE_NAMES = [
    "has_certificate",
    "has_pe_warning",
    "entropy",
    "import_api_count",
]


V1_FEATURE_NAMES = V0_FEATURE_NAMES + [
    # General
    "file_size",

    # Strings
    "num_strings",
    "avg_string_length",
    "string_entropy",
    "num_printables",

    # PE Header
    "header_missing",
    "num_sections_header",
    "size_of_code",
    "size_of_image",
    "size_of_headers",
    "entrypoint",

    # Section / Overlay
    "section_missing",
    "section_count",
    "mean_section_entropy",
    "max_section_entropy",
    "overlay_size",
    "overlay_ratio",
    "overlay_entropy",

    # Imports
    "import_dll_count",

    # Authenticode
    "certificate_count",
    "self_signed",
    "certificate_parse_error",

    # PE warnings
    "warning_count",
]


def number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    return default


def extract_v0(record: dict) -> list[float]:
    authenticode = record.get("authenticode") or {}
    cert_count = number(authenticode.get("num_certs"))

    warnings = record.get("pefilewarnings") or []

    general = record.get("general") or {}
    entropy = number(general.get("entropy"))

    imports = record.get("imports") or {}

    import_api_count = 0

    for api_list in imports.values():
        if isinstance(api_list, list):
            import_api_count += len(api_list)

    return [
        1.0 if cert_count > 0 else 0.0,
        1.0 if warnings else 0.0,
        entropy,
        float(import_api_count),
    ]


def extract_v1(record: dict) -> list[float]:
    # -----------------------------
    # V0
    # -----------------------------
    features = extract_v0(record)

    # -----------------------------
    # General
    # -----------------------------
    general = record.get("general") or {}

    file_size = number(general.get("size"))

    # -----------------------------
    # Strings
    # -----------------------------
    strings = record.get("strings") or {}

    num_strings = number(strings.get("numstrings"))
    avg_string_length = number(strings.get("avlength"))
    string_entropy = number(strings.get("entropy"))
    num_printables = number(strings.get("printables"))

    # -----------------------------
    # Header
    # -----------------------------
    header = record.get("header")
    header_missing = 1.0 if not header else 0.0

    header = header or {}

    coff = header.get("coff") or {}
    optional = header.get("optional") or {}

    num_sections_header = number(
        coff.get("number_of_sections")
    )

    size_of_code = number(
        optional.get("sizeof_code")
    )

    size_of_image = number(
        optional.get("sizeof_image")
    )

    size_of_headers = number(
        optional.get("sizeof_headers")
    )

    entrypoint = number(
        optional.get("address_of_entrypoint")
    )

    # -----------------------------
    # Sections
    # -----------------------------
    section = record.get("section")
    section_missing = 1.0 if not section else 0.0

    section = section or {}

    sections = section.get("sections") or []

    section_count = float(len(sections))

    section_entropies = []

    for item in sections:
        entropy_value = item.get("entropy")

        if isinstance(entropy_value, (int, float)):
            section_entropies.append(
                float(entropy_value)
            )

    if section_entropies:
        mean_section_entropy = (
            sum(section_entropies)
            / len(section_entropies)
        )

        max_section_entropy = max(
            section_entropies
        )
    else:
        mean_section_entropy = 0.0
        max_section_entropy = 0.0

    overlay = section.get("overlay") or {}

    overlay_size = number(
        overlay.get("size")
    )

    overlay_ratio = number(
        overlay.get("size_ratio")
    )

    overlay_entropy = number(
        overlay.get("entropy")
    )

    # -----------------------------
    # Imports
    # -----------------------------
    imports = record.get("imports") or {}

    import_dll_count = float(len(imports))

    # -----------------------------
    # Authenticode
    # -----------------------------
    authenticode = record.get("authenticode") or {}

    certificate_count = number(
        authenticode.get("num_certs")
    )

    self_signed = number(
        authenticode.get("self_signed")
    )

    certificate_parse_error = number(
        authenticode.get("parse_error")
    )

    # -----------------------------
    # PE warnings
    # -----------------------------
    warnings = record.get("pefilewarnings") or []

    warning_count = float(len(warnings))

    # -----------------------------
    # 최종 Feature
    # -----------------------------
    features.extend(
        [
            file_size,

            num_strings,
            avg_string_length,
            string_entropy,
            num_printables,

            header_missing,
            num_sections_header,
            size_of_code,
            size_of_image,
            size_of_headers,
            entrypoint,

            section_missing,
            section_count,
            mean_section_entropy,
            max_section_entropy,
            overlay_size,
            overlay_ratio,
            overlay_entropy,

            import_dll_count,

            certificate_count,
            self_signed,
            certificate_parse_error,

            warning_count,
        ]
    )

    return features