from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


EICAR_TEXT = (
    "X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
    "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
)

YARA_SAMPLE_TEXT = "THIS_IS_A_SAFE_YARA_CONNECTION_TEST"
YARA_RULE_TEXT = '''
rule safe_connection_test
{
    strings:
        $marker = "THIS_IS_A_SAFE_YARA_CONNECTION_TEST"

    condition:
        $marker
}
'''.strip()


def resolve_command(explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)

        found = shutil.which(explicit)
        if found:
            return found

        return None

    for name in candidates:
        found = shutil.which(name)
        if found:
            return found

    return None


def run_command(args: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return -2, "TIMEOUT"
    except OSError as exc:
        return -3, f"OS_ERROR: {exc}"

    output = "\n".join(
        x.strip()
        for x in (proc.stdout, proc.stderr)
        if x and x.strip()
    )
    return proc.returncode, output


def test_clamav(command: str | None) -> bool:
    print("=" * 80)
    print("CLAMAV CONNECTION TEST")
    print("=" * 80)

    if not command:
        print("RESULT: FAIL")
        print("REASON: clamscan command not found")
        return False

    print(f"Command: {command}")

    rc, out = run_command([command, "--version"])
    print(f"Version exit code: {rc}")
    print(out or "(no output)")

    if rc != 0:
        print("RESULT: FAIL")
        print("REASON: clamscan --version failed")
        return False

    try:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "eicar_test.txt"

            try:
                sample.write_text(
                    EICAR_TEXT,
                    encoding="ascii",
                )
            except OSError as exc:
                print("RESULT: BLOCKED/INCONCLUSIVE")
                print(
                    "REASON: EICAR test file could not be created. "
                    f"Another antivirus may have blocked it: {exc}"
                )
                return False

            rc, out = run_command(
                [
                    command,
                    "--no-summary",
                    str(sample),
                ]
            )

            print(f"Scan exit code: {rc}")
            print(out or "(no output)")

            if rc == 1:
                print("RESULT: PASS")
                print("ClamAV command works and detected the EICAR test signature.")
                return True

            if not sample.exists():
                print("RESULT: BLOCKED/INCONCLUSIVE")
                print(
                    "REASON: test file disappeared during scan. "
                    "Another antivirus may have quarantined it."
                )
                return False

            print("RESULT: FAIL")
            print("REASON: ClamAV ran but did not detect the EICAR test signature.")
            return False

    except OSError as exc:
        print("RESULT: FAIL")
        print(f"REASON: temporary file error: {exc}")
        return False


def test_yara(command: str | None) -> bool:
    print()
    print("=" * 80)
    print("YARA CONNECTION TEST")
    print("=" * 80)

    if not command:
        print("RESULT: FAIL")
        print("REASON: yara command not found")
        return False

    print(f"Command: {command}")

    rc, out = run_command([command, "--version"])
    print(f"Version exit code: {rc}")
    print(out or "(no output)")

    if rc != 0:
        print("RESULT: FAIL")
        print("REASON: yara --version failed")
        return False

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            sample = tmp_dir / "safe_yara_sample.txt"
            rule = tmp_dir / "safe_connection_test.yar"

            sample.write_text(
                YARA_SAMPLE_TEXT,
                encoding="utf-8",
            )
            rule.write_text(
                YARA_RULE_TEXT,
                encoding="utf-8",
            )

            rc, out = run_command(
                [
                    command,
                    str(rule),
                    str(sample),
                ]
            )

            print(f"Scan exit code: {rc}")
            print(out or "(no output)")

            if rc == 0 and "safe_connection_test" in out:
                print("RESULT: PASS")
                print("YARA command works and matched the safe test rule.")
                return True

            print("RESULT: FAIL")
            print("REASON: YARA ran but the safe test rule did not match.")
            return False

    except OSError as exc:
        print("RESULT: FAIL")
        print(f"REASON: temporary file error: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safe ClamAV/YARA connection test"
    )
    parser.add_argument(
        "--clamscan",
        default=None,
        help="Path or command name for clamscan.exe",
    )
    parser.add_argument(
        "--yara",
        default=None,
        help="Path or command name for yara.exe",
    )
    args = parser.parse_args()

    clamav_command = resolve_command(
        args.clamscan,
        ["clamscan.exe", "clamscan"],
    )
    yara_command = resolve_command(
        args.yara,
        ["yara64.exe", "yara.exe", "yara"],
    )

    clamav_ok = test_clamav(clamav_command)
    yara_ok = test_yara(yara_command)

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"ClamAV: {'PASS' if clamav_ok else 'FAIL/INCONCLUSIVE'}")
    print(f"YARA  : {'PASS' if yara_ok else 'FAIL/INCONCLUSIVE'}")

    if clamav_ok and yara_ok:
        print("OVERALL: PASS")
    else:
        print("OVERALL: NOT READY")


if __name__ == "__main__":
    main()
