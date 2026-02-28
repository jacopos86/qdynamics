#!/usr/bin/env python3
"""Dry-run tests for drive flag passthrough in compare_hardcoded_vs_qiskit_pipeline.py.

Strategy
--------
All tests exercise ``_build_drive_args`` and ``parse_args`` directly (no
subprocess launched, no filesystem touched).  This makes the suite fast,
hermetic, and independent of Qiskit/matplotlib availability.

Test classes
------------
TestBuildDriveArgsDisabled
    Verify that _build_drive_args returns [] when --enable-drive is absent.
    This is the backward-compatibility guarantee.

TestBuildDriveArgsEnabled
    Verify every expected flag appears in the token list when --enable-drive
    is set, with the correct values forwarded.

TestBuildDriveArgsCustomWeights
    Verify --drive-custom-s is forwarded only when provided, and that the
    --drive-include-identity flag is a bare flag (not --drive-include-identity=True).

TestParseDriveDefaults
    Verify that parse_args() assigns the correct defaults for all drive flags
    so that the forward is stable even when a user omits them.

TestCommandStructure
    End-to-end: simulate the exact hc_cmd / qk_cmd construction logic and
    verify (a) drive tokens appear at the end after --skip-qpe, (b) no drive
    tokens appear when drive is disabled, (c) token list is valid shlex.
"""

from __future__ import annotations

import shlex
import sys
import types
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = REPO_ROOT / "Fermi-Hamil-JW-VQE-TROTTER-PIPELINE"
for p in (REPO_ROOT, PIPELINE_ROOT):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

# Import only the pure functions — no side-effect module-level code is
# triggered by the import because compare_hardcoded_vs_qiskit_pipeline.py
# guards its main() behind ``if __name__ == "__main__":``
import pipelines.compare_hardcoded_vs_qiskit_pipeline as cmp

_build_drive_args = cmp._build_drive_args
_DRIVE_FLAG_DEFAULTS = cmp._DRIVE_FLAG_DEFAULTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> types.SimpleNamespace:
    """Return a SimpleNamespace populated with drive-flag defaults, then overrides."""
    merged = dict(_DRIVE_FLAG_DEFAULTS)
    merged.update(overrides)
    return types.SimpleNamespace(**merged)


# ---------------------------------------------------------------------------

class TestBuildDriveArgsDisabled(unittest.TestCase):
    """_build_drive_args returns [] when drive is disabled — backward compat."""

    def test_returns_empty_list_by_default(self) -> None:
        args = _make_args()  # enable_drive=False
        result = _build_drive_args(args)
        self.assertEqual(result, [])

    def test_returns_empty_list_when_explicitly_false(self) -> None:
        args = _make_args(enable_drive=False)
        result = _build_drive_args(args)
        self.assertEqual(result, [])

    def test_result_is_a_list(self) -> None:
        args = _make_args()
        result = _build_drive_args(args)
        self.assertIsInstance(result, list)


class TestBuildDriveArgsEnabled(unittest.TestCase):
    """Every numeric/string drive flag is forwarded correctly."""

    def setUp(self) -> None:
        self.args = _make_args(
            enable_drive=True,
            drive_A=0.42,
            drive_omega=3.14,
            drive_tbar=7.0,
            drive_phi=0.5,
            drive_pattern="staggered",
            drive_custom_s=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
        )
        self.tokens = _build_drive_args(self.args)

    def test_enable_drive_flag_present(self) -> None:
        self.assertIn("--enable-drive", self.tokens)

    def test_drive_A_forwarded(self) -> None:
        idx = self.tokens.index("--drive-A")
        self.assertAlmostEqual(float(self.tokens[idx + 1]), 0.42)

    def test_drive_omega_forwarded(self) -> None:
        idx = self.tokens.index("--drive-omega")
        self.assertAlmostEqual(float(self.tokens[idx + 1]), 3.14)

    def test_drive_tbar_forwarded(self) -> None:
        idx = self.tokens.index("--drive-tbar")
        self.assertAlmostEqual(float(self.tokens[idx + 1]), 7.0)

    def test_drive_phi_forwarded(self) -> None:
        idx = self.tokens.index("--drive-phi")
        self.assertAlmostEqual(float(self.tokens[idx + 1]), 0.5)

    def test_drive_pattern_forwarded(self) -> None:
        idx = self.tokens.index("--drive-pattern")
        self.assertEqual(self.tokens[idx + 1], "staggered")

    def test_drive_time_sampling_forwarded(self) -> None:
        idx = self.tokens.index("--drive-time-sampling")
        self.assertEqual(self.tokens[idx + 1], "midpoint")

    def test_drive_t0_forwarded(self) -> None:
        idx = self.tokens.index("--drive-t0")
        self.assertAlmostEqual(float(self.tokens[idx + 1]), 0.0)

    def test_no_custom_s_when_none(self) -> None:
        self.assertNotIn("--drive-custom-s", self.tokens)

    def test_no_include_identity_flag_when_false(self) -> None:
        self.assertNotIn("--drive-include-identity", self.tokens)

    def test_token_count_is_reasonable(self) -> None:
        # 1 bare flag + 8 pairs = 17 tokens (no custom-s, no identity)
        # 8 pairs: A, omega, tbar, phi, pattern, time-sampling, t0, exact-steps-multiplier
        self.assertEqual(len(self.tokens), 17)

    def test_tokens_are_all_strings(self) -> None:
        for tok in self.tokens:
            self.assertIsInstance(tok, str, f"Non-string token: {tok!r}")

    def test_tokens_are_shlex_round_trippable(self) -> None:
        joined = " ".join(shlex.quote(t) for t in self.tokens)
        roundtripped = shlex.split(joined)
        self.assertEqual(roundtripped, self.tokens)


class TestBuildDriveArgsCustomWeights(unittest.TestCase):
    """Custom weights and identity flag are conditional."""

    def test_custom_s_forwarded_when_provided(self) -> None:
        args = _make_args(
            enable_drive=True,
            drive_custom_s="[1.0,-0.5,0.3]",
            drive_pattern="custom",
        )
        tokens = _build_drive_args(args)
        self.assertIn("--drive-custom-s", tokens)
        idx = tokens.index("--drive-custom-s")
        self.assertEqual(tokens[idx + 1], "[1.0,-0.5,0.3]")

    def test_custom_s_absent_when_none(self) -> None:
        args = _make_args(enable_drive=True, drive_custom_s=None)
        tokens = _build_drive_args(args)
        self.assertNotIn("--drive-custom-s", tokens)

    def test_include_identity_bare_flag_when_true(self) -> None:
        args = _make_args(enable_drive=True, drive_include_identity=True)
        tokens = _build_drive_args(args)
        self.assertIn("--drive-include-identity", tokens)
        # Must be a bare flag — not followed by a boolean string.
        idx = tokens.index("--drive-include-identity")
        # Either it's the last token or the next token starts with "--"
        if idx + 1 < len(tokens):
            self.assertTrue(
                tokens[idx + 1].startswith("--"),
                f"Expected next token to be another flag, got {tokens[idx+1]!r}",
            )

    def test_include_identity_absent_when_false(self) -> None:
        args = _make_args(enable_drive=True, drive_include_identity=False)
        tokens = _build_drive_args(args)
        self.assertNotIn("--drive-include-identity", tokens)

    def test_all_patterns_forwarded_correctly(self) -> None:
        for pattern in ("staggered", "dimer_bias", "custom"):
            with self.subTest(pattern=pattern):
                args = _make_args(enable_drive=True, drive_pattern=pattern)
                tokens = _build_drive_args(args)
                idx = tokens.index("--drive-pattern")
                self.assertEqual(tokens[idx + 1], pattern)

    def test_all_time_sampling_values_forwarded(self) -> None:
        for sampling in ("midpoint", "left", "right"):
            with self.subTest(sampling=sampling):
                args = _make_args(enable_drive=True, drive_time_sampling=sampling)
                tokens = _build_drive_args(args)
                idx = tokens.index("--drive-time-sampling")
                self.assertEqual(tokens[idx + 1], sampling)


class TestParseDriveDefaults(unittest.TestCase):
    """parse_args() assigns correct defaults for all drive flags."""

    @classmethod
    def setUpClass(cls) -> None:
        # parse_args() reads sys.argv; inject a minimal valid argv.
        cls._orig_argv = sys.argv[:]
        sys.argv = ["compare_hardcoded_vs_qiskit_pipeline.py"]

    @classmethod
    def tearDownClass(cls) -> None:
        sys.argv = cls._orig_argv

    def setUp(self) -> None:
        self.args = cmp.parse_args()

    def test_enable_drive_default_false(self) -> None:
        self.assertFalse(self.args.enable_drive)

    def test_drive_A_default(self) -> None:
        self.assertAlmostEqual(self.args.drive_A, 1.0)

    def test_drive_omega_default(self) -> None:
        self.assertAlmostEqual(self.args.drive_omega, 1.0)

    def test_drive_tbar_default(self) -> None:
        self.assertAlmostEqual(self.args.drive_tbar, 5.0)

    def test_drive_phi_default(self) -> None:
        self.assertAlmostEqual(self.args.drive_phi, 0.0)

    def test_drive_pattern_default(self) -> None:
        self.assertEqual(self.args.drive_pattern, "staggered")

    def test_drive_custom_s_default_none(self) -> None:
        self.assertIsNone(self.args.drive_custom_s)

    def test_drive_include_identity_default_false(self) -> None:
        self.assertFalse(self.args.drive_include_identity)

    def test_drive_time_sampling_default(self) -> None:
        self.assertEqual(self.args.drive_time_sampling, "midpoint")

    def test_drive_t0_default(self) -> None:
        self.assertAlmostEqual(self.args.drive_t0, 0.0)

    def test_fidelity_subspace_energy_tol_default(self) -> None:
        self.assertAlmostEqual(float(self.args.fidelity_subspace_energy_tol), 1e-8)

    def test_report_verbose_default_false(self) -> None:
        self.assertFalse(self.args.report_verbose)

    def test_safe_test_near_threshold_factor_default(self) -> None:
        self.assertAlmostEqual(float(self.args.safe_test_near_threshold_factor), 100.0)

    def test_defaults_match_DRIVE_FLAG_DEFAULTS_dict(self) -> None:
        """All defaults in parse_args must match _DRIVE_FLAG_DEFAULTS sentinel."""
        for key, expected in _DRIVE_FLAG_DEFAULTS.items():
            got = getattr(self.args, key)
            if isinstance(expected, float):
                self.assertAlmostEqual(
                    float(got), float(expected),
                    msg=f"Mismatch for {key}: got {got!r}, expected {expected!r}",
                )
            else:
                self.assertEqual(
                    got, expected,
                    msg=f"Mismatch for {key}: got {got!r}, expected {expected!r}",
                )


class TestCommandStructure(unittest.TestCase):
    """Simulate the exact hc_cmd/qk_cmd construction and verify token order."""

    def _build_hc_cmd(self, args: types.SimpleNamespace) -> list[str]:
        """Mirror the hc_cmd construction from main(), without subprocess."""
        cmd = [
            sys.executable,
            "pipelines/hardcoded_hubbard_pipeline.py",
            "--L", "2",
            "--t", "1.0",
            "--u", "4.0",
            "--dv", "0.0",
            "--boundary", "periodic",
            "--ordering", "blocked",
            "--t-final", "10.0",
            "--num-times", "51",
            "--suzuki-order", "2",
            "--trotter-steps", "16",
            "--term-order", "sorted",
            "--vqe-reps", "1",
            "--vqe-restarts", "1",
            "--vqe-seed", "7",
            "--vqe-maxiter", "40",
            "--qpe-eval-qubits", "4",
            "--qpe-shots", "64",
            "--qpe-seed", "11",
            "--initial-state-source", "exact",
            "--output-json", "/tmp/hc.json",
            "--output-pdf", "/tmp/hc.pdf",
            "--skip-pdf",
        ]
        if getattr(args, "skip_qpe", False):
            cmd.append("--skip-qpe")
        cmd.extend(_build_drive_args(args))
        return cmd

    def test_no_drive_tokens_when_disabled(self) -> None:
        args = _make_args(enable_drive=False)
        cmd = self._build_hc_cmd(args)
        for tok in cmd:
            self.assertFalse(
                tok.startswith("--drive"),
                f"Unexpected drive token {tok!r} when drive is disabled",
            )
        self.assertNotIn("--enable-drive", cmd)

    def test_drive_tokens_appended_at_end_when_enabled(self) -> None:
        args = _make_args(enable_drive=True, drive_A=0.9, drive_omega=2.5, drive_tbar=4.0)
        cmd = self._build_hc_cmd(args)
        # --enable-drive must be present.
        self.assertIn("--enable-drive", cmd)
        # The first drive token must come after --skip-pdf.
        skip_pdf_idx = cmd.index("--skip-pdf")
        enable_idx = cmd.index("--enable-drive")
        self.assertGreater(enable_idx, skip_pdf_idx,
                           "--enable-drive should come after --skip-pdf")

    def test_skip_qpe_before_drive_tokens(self) -> None:
        args = _make_args(enable_drive=True, skip_qpe=True)
        cmd = self._build_hc_cmd(args)
        self.assertIn("--skip-qpe", cmd)
        self.assertIn("--enable-drive", cmd)
        skip_qpe_idx = cmd.index("--skip-qpe")
        enable_idx = cmd.index("--enable-drive")
        self.assertLess(skip_qpe_idx, enable_idx,
                        "--skip-qpe must come before --enable-drive")

    def test_drive_A_value_in_cmd(self) -> None:
        args = _make_args(enable_drive=True, drive_A=0.77)
        cmd = self._build_hc_cmd(args)
        idx = cmd.index("--drive-A")
        self.assertAlmostEqual(float(cmd[idx + 1]), 0.77)

    def test_cmd_is_shlex_round_trippable(self) -> None:
        args = _make_args(
            enable_drive=True,
            drive_A=0.5,
            drive_custom_s="[1.0, -1.0]",
        )
        cmd = self._build_hc_cmd(args)
        joined = " ".join(shlex.quote(t) for t in cmd)
        roundtripped = shlex.split(joined)
        self.assertEqual(roundtripped, cmd)

    def test_drive_flags_absent_in_no_drive_shlex(self) -> None:
        args = _make_args(enable_drive=False)
        cmd = self._build_hc_cmd(args)
        joined = " ".join(shlex.quote(t) for t in cmd)
        self.assertNotIn("--drive", joined)
        self.assertNotIn("--enable-drive", joined)

    def test_qk_cmd_receives_same_drive_tokens(self) -> None:
        """Both hc_cmd and qk_cmd should receive identical drive token lists."""
        args = _make_args(enable_drive=True, drive_A=0.3, drive_tbar=3.0, drive_phi=0.1)
        drive_tokens = _build_drive_args(args)

        # Build both commands and extract their drive suffixes.
        hc = self._build_hc_cmd(args)
        # qk_cmd is identical in structure; reuse same helper with different script.
        qk = [t.replace("hardcoded_hubbard_pipeline", "qiskit_hubbard_baseline_pipeline")
              for t in hc]

        # Both contain identical drive tokens.
        self.assertEqual(drive_tokens, hc[-len(drive_tokens):])
        self.assertEqual(drive_tokens, qk[-len(drive_tokens):])


if __name__ == "__main__":
    unittest.main()
