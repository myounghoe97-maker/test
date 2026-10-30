from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


DEFAULT_TIMEOUT_SECONDS = 30


class Verdict(str, Enum):
    SAFE_CANDIDATE = "SAFE_CANDIDATE"
    MALICIOUS_EVIDENCE = "MALICIOUS_EVIDENCE"
    SANDBOX_REQUIRED = "SANDBOX_REQUIRED"
    ERROR = "ERROR"


class ToolStatus(str, Enum):
    DETECTED = "DETECTED"
    CLEAN = "CLEAN"
    UNAVAILABLE = "UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"


@dataclass
class ToolResult:
    status: ToolStatus
    detail: str = ""


@dataclass
class SecondaryStaticResult:
    file_path: str
    sha256: str
    benign_hash_match: bool
    malicious_hash_match: bool
    clamav: ToolResult
    yara: ToolResult
    static_risk_score: int
    static_risk_reasons: list[str]
    verdict: Verdict
    reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_hash_set(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()

    hashes: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip().lower()

            if not value or value.startswith("#"):
                continue

            if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
                hashes.add(value)

    return hashes


def run_clamav(
    file_path: Path,
    clamav_command: str = "clamscan",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ToolResult:
    try:
        proc = subprocess.run(
            [
                clamav_command,
                "--no-summary",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return ToolResult(
            status=ToolStatus.UNAVAILABLE,
            detail="ClamAV command not found",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            status=ToolStatus.TIMEOUT,
            detail=f"ClamAV timeout after {timeout_seconds}s",
        )
    except OSError as exc:
        return ToolResult(
            status=ToolStatus.ERROR,
            detail=f"ClamAV OS error: {exc}",
        )

    output = "\n".join(
        x.strip()
        for x in (proc.stdout, proc.stderr)
        if x and x.strip()
    )

    # clamscan:
    # 0 = no virus found
    # 1 = virus found
    # >1 = error
    if proc.returncode == 1:
        return ToolResult(
            status=ToolStatus.DETECTED,
            detail=output,
        )

    if proc.returncode == 0:
        return ToolResult(
            status=ToolStatus.CLEAN,
            detail=output,
        )

    return ToolResult(
        status=ToolStatus.ERROR,
        detail=output or f"ClamAV exit code {proc.returncode}",
    )


def run_yara(
    file_path: Path,
    yara_rules: Path | None,
    yara_command: str = "yara",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ToolResult:
    if yara_rules is None or not yara_rules.exists():
        return ToolResult(
            status=ToolStatus.UNAVAILABLE,
            detail="YARA rule file not configured",
        )

    try:
        proc = subprocess.run(
            [
                yara_command,
                str(yara_rules),
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return ToolResult(
            status=ToolStatus.UNAVAILABLE,
            detail="YARA command not found",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            status=ToolStatus.TIMEOUT,
            detail=f"YARA timeout after {timeout_seconds}s",
        )
    except OSError as exc:
        return ToolResult(
            status=ToolStatus.ERROR,
            detail=f"YARA OS error: {exc}",
        )

    output = "\n".join(
        x.strip()
        for x in (proc.stdout, proc.stderr)
        if x and x.strip()
    )

    # yara CLI:
    # 0 = scan completed, matching rules may be printed
    # non-zero = command/rule error
    if proc.returncode != 0:
        return ToolResult(
            status=ToolStatus.ERROR,
            detail=output or f"YARA exit code {proc.returncode}",
        )

    if proc.stdout.strip():
        return ToolResult(
            status=ToolStatus.DETECTED,
            detail=proc.stdout.strip(),
        )

    return ToolResult(
        status=ToolStatus.CLEAN,
        detail="No YARA rule matched",
    )


def static_risk_from_record(record: dict | None) -> tuple[int, list[str]]:
    """
    기존 EMBER JSONL 형태의 정적 특징이 제공된 경우에만 보조 위험도를 계산한다.

    중요:
    - 이 점수 단독으로 SAFE 판정을 하지 않는다.
    - 이 점수 단독으로 MALICIOUS 확정을 하지 않는다.
    - Sandbox 우선순위/설명용 보조 근거다.
    """
    if not record:
        return 0, []

    score = 0
    reasons: list[str] = []

    auth = record.get("authenticode") or {}
    warnings = record.get("pefilewarnings") or []
    general = record.get("general") or {}
    section = record.get("section") or {}

    sections = section.get("sections") or []
    overlay = section.get("overlay") or {}

    num_certs = auth.get("num_certs")
    if isinstance(num_certs, (int, float)) and num_certs <= 0:
        score += 1
        reasons.append("certificate_missing")

    if warnings:
        score += 2
        reasons.append("pe_warning_present")

    entropy = general.get("entropy")
    if isinstance(entropy, (int, float)) and entropy >= 7.2:
        score += 2
        reasons.append("high_file_entropy")

    section_entropies = [
        float(s["entropy"])
        for s in sections
        if isinstance(s, dict) and isinstance(s.get("entropy"), (int, float))
    ]

    if section_entropies and max(section_entropies) >= 7.5:
        score += 2
        reasons.append("high_section_entropy")

    overlay_ratio = overlay.get("size_ratio")
    if isinstance(overlay_ratio, (int, float)) and overlay_ratio >= 0.30:
        score += 1
        reasons.append("large_overlay_ratio")

    return score, reasons


def decide_verdict(
    benign_hash_match: bool,
    malicious_hash_match: bool,
    clamav: ToolResult,
    yara: ToolResult,
) -> tuple[Verdict, str]:
    if malicious_hash_match:
        return (
            Verdict.MALICIOUS_EVIDENCE,
            "Known malicious SHA256 matched",
        )

    if clamav.status == ToolStatus.DETECTED:
        return (
            Verdict.MALICIOUS_EVIDENCE,
            "ClamAV detected malware",
        )

    if yara.status == ToolStatus.DETECTED:
        return (
            Verdict.MALICIOUS_EVIDENCE,
            "YARA rule matched",
        )

    if benign_hash_match:
        return (
            Verdict.SAFE_CANDIDATE,
            "Trusted benign SHA256 matched",
        )

    # 핵심 안전 규칙:
    # ClamAV/YARA의 미탐지는 정상 증거로 사용하지 않는다.
    return (
        Verdict.SANDBOX_REQUIRED,
        "No strong static evidence; dynamic analysis required",
    )


def analyze_uncertain_file(
    file_path: Path,
    benign_hashes: Iterable[str] = (),
    malicious_hashes: Iterable[str] = (),
    yara_rules: Path | None = None,
    extracted_record: dict | None = None,
    clamav_command: str = "clamscan",
    yara_command: str = "yara",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> SecondaryStaticResult:
    if not file_path.exists() or not file_path.is_file():
        return SecondaryStaticResult(
            file_path=str(file_path),
            sha256="",
            benign_hash_match=False,
            malicious_hash_match=False,
            clamav=ToolResult(ToolStatus.NOT_RUN, "File not found"),
            yara=ToolResult(ToolStatus.NOT_RUN, "File not found"),
            static_risk_score=0,
            static_risk_reasons=[],
            verdict=Verdict.ERROR,
            reason="Input file does not exist",
        )

    try:
        file_sha256 = sha256_file(file_path)
    except OSError as exc:
        return SecondaryStaticResult(
            file_path=str(file_path),
            sha256="",
            benign_hash_match=False,
            malicious_hash_match=False,
            clamav=ToolResult(ToolStatus.NOT_RUN, "Hash failed"),
            yara=ToolResult(ToolStatus.NOT_RUN, "Hash failed"),
            static_risk_score=0,
            static_risk_reasons=[],
            verdict=Verdict.ERROR,
            reason=f"SHA256 calculation failed: {exc}",
        )

    benign_set = {x.lower() for x in benign_hashes}
    malicious_set = {x.lower() for x in malicious_hashes}

    benign_hash_match = file_sha256 in benign_set
    malicious_hash_match = file_sha256 in malicious_set

    # 악성 hash가 정확히 일치하면 외부 스캐너를 돌릴 필요가 없다.
    if malicious_hash_match:
        clamav_result = ToolResult(
            ToolStatus.NOT_RUN,
            "Skipped: known malicious SHA256",
        )
        yara_result = ToolResult(
            ToolStatus.NOT_RUN,
            "Skipped: known malicious SHA256",
        )
    else:
        clamav_result = run_clamav(
            file_path=file_path,
            clamav_command=clamav_command,
            timeout_seconds=timeout_seconds,
        )

        yara_result = run_yara(
            file_path=file_path,
            yara_rules=yara_rules,
            yara_command=yara_command,
            timeout_seconds=timeout_seconds,
        )

    risk_score, risk_reasons = static_risk_from_record(extracted_record)

    verdict, reason = decide_verdict(
        benign_hash_match=benign_hash_match,
        malicious_hash_match=malicious_hash_match,
        clamav=clamav_result,
        yara=yara_result,
    )

    return SecondaryStaticResult(
        file_path=str(file_path),
        sha256=file_sha256,
        benign_hash_match=benign_hash_match,
        malicious_hash_match=malicious_hash_match,
        clamav=clamav_result,
        yara=yara_result,
        static_risk_score=risk_score,
        static_risk_reasons=risk_reasons,
        verdict=verdict,
        reason=reason,
    )


def load_json_record(path: Path | None) -> dict | None:
    if path is None:
        return None

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("record JSON must be an object")

    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Second-stage static filter for LightGBM uncertain files"
    )

    parser.add_argument(
        "file",
        type=Path,
        help="PE file path to inspect",
    )
    parser.add_argument(
        "--benign-hashes",
        type=Path,
        default=None,
        help="Trusted benign SHA256 text file",
    )
    parser.add_argument(
        "--malicious-hashes",
        type=Path,
        default=None,
        help="Known malicious SHA256 text file",
    )
    parser.add_argument(
        "--yara-rules",
        type=Path,
        default=None,
        help="YARA rule file",
    )
    parser.add_argument(
        "--record-json",
        type=Path,
        default=None,
        help="Optional extracted static-feature JSON",
    )
    parser.add_argument(
        "--clamav-command",
        default="clamscan",
    )
    parser.add_argument(
        "--yara-command",
        default="yara",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )

    args = parser.parse_args()

    benign_hashes = load_hash_set(args.benign_hashes)
    malicious_hashes = load_hash_set(args.malicious_hashes)
    record = load_json_record(args.record_json)

    result = analyze_uncertain_file(
        file_path=args.file,
        benign_hashes=benign_hashes,
        malicious_hashes=malicious_hashes,
        yara_rules=args.yara_rules,
        extracted_record=record,
        clamav_command=args.clamav_command,
        yara_command=args.yara_command,
        timeout_seconds=args.timeout,
    )

    print(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.value if isinstance(x, Enum) else str(x),
        )
    )


if __name__ == "__main__":
    main()
