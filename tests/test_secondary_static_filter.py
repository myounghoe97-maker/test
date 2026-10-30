
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from secondary_static_filter import (
    ToolResult,
    ToolStatus,
    Verdict,
    decide_verdict,
    sha256_file,
)


class SecondaryStaticFilterTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"abc")

            expected = hashlib.sha256(b"abc").hexdigest()

            self.assertEqual(
                sha256_file(path),
                expected,
            )

    def test_known_malicious_hash_has_priority(self):
        verdict, _ = decide_verdict(
            benign_hash_match=True,
            malicious_hash_match=True,
            clamav=ToolResult(ToolStatus.CLEAN),
            yara=ToolResult(ToolStatus.CLEAN),
        )

        self.assertEqual(
            verdict,
            Verdict.MALICIOUS_EVIDENCE,
        )

    def test_clean_scanners_do_not_mean_safe(self):
        verdict, _ = decide_verdict(
            benign_hash_match=False,
            malicious_hash_match=False,
            clamav=ToolResult(ToolStatus.CLEAN),
            yara=ToolResult(ToolStatus.CLEAN),
        )

        self.assertEqual(
            verdict,
            Verdict.SANDBOX_REQUIRED,
        )

    def test_scanner_failure_does_not_mean_safe(self):
        verdict, _ = decide_verdict(
            benign_hash_match=False,
            malicious_hash_match=False,
            clamav=ToolResult(ToolStatus.ERROR),
            yara=ToolResult(ToolStatus.TIMEOUT),
        )

        self.assertEqual(
            verdict,
            Verdict.SANDBOX_REQUIRED,
        )

    def test_trusted_benign_hash_can_be_safe_candidate(self):
        verdict, _ = decide_verdict(
            benign_hash_match=True,
            malicious_hash_match=False,
            clamav=ToolResult(ToolStatus.CLEAN),
            yara=ToolResult(ToolStatus.CLEAN),
        )

        self.assertEqual(
            verdict,
            Verdict.SAFE_CANDIDATE,
        )


if __name__ == "__main__":
    unittest.main()
