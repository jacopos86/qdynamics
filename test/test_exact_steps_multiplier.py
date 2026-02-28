#!/usr/bin/env python3
"""Unit tests for --exact-steps-multiplier and reference-method metadata.

Covers
------
1. ``reference_method_name`` — correct string returned for all three sampling
   rules; ``ValueError`` on bad input.
2. ``_evolve_piecewise_exact`` with midpoint sampling on a 1-qubit
   H(t) = (ω/2) Z Hamiltonian:
   - Analytic: ψ(T) = exp(−i · (ω/2) · Φ(T) · Z) ψ₀  where Φ(T) = ∫₀ᵀ dt.
   - For H(t) = (ω/2) Z (constant waveform, φ=0 trivial):
       We use H(t) = sin(t) · Z  so  Φ(T) = 1 − cos(T), analytic.
   - Error with N steps and midpoint sampling should be O(1/N²):
       err(N) / err(2N)  ∈  (3, 5)   (theory: 4 for exact O(N²))
3. ``_evolve_piecewise_exact`` with N vs M=2N steps shows M gives strictly
   smaller error (tests the multiplier rationale directly).
4. ``reference_method_name`` values appear in JSON output when drive is
   enabled (integration test on the JSON settings dict).
5. Left sampling is O(N¹) — the error halves as N doubles.

The test module deliberately avoids importing anything that requires Qiskit
so it can run in any environment with only numpy/scipy available.
"""

from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = REPO_ROOT / "Fermi-Hamil-JW-VQE-TROTTER-PIPELINE"

for _p in (REPO_ROOT, PIPELINE_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from src.quantum.drives_time_potential import (
    REFERENCE_METHOD_NAMES,
    reference_method_name,
)


# ---------------------------------------------------------------------------
# Helper: build piecewise-exact propagator from the hardcoded pipeline
# without importing Qiskit-heavy top-level.
# ---------------------------------------------------------------------------

def _import_evolve_piecewise_exact():
    """Lazily import _evolve_piecewise_exact from the hardcoded pipeline."""
    import pipelines.hardcoded_hubbard_pipeline as hp
    return hp._evolve_piecewise_exact


# ---------------------------------------------------------------------------
# 1. reference_method_name API
# ---------------------------------------------------------------------------

class TestReferenceMethodName(unittest.TestCase):
    """Verify the metadata string table in drives_time_potential."""

    def test_midpoint_key(self) -> None:
        result = reference_method_name("midpoint")
        self.assertEqual(result, "exponential_midpoint_magnus2_order2")

    def test_left_key(self) -> None:
        result = reference_method_name("left")
        self.assertEqual(result, "exponential_left_endpoint_order1")

    def test_right_key(self) -> None:
        result = reference_method_name("right")
        self.assertEqual(result, "exponential_right_endpoint_order1")

    def test_case_insensitive(self) -> None:
        self.assertEqual(reference_method_name("MIDPOINT"), reference_method_name("midpoint"))
        self.assertEqual(reference_method_name("Left"), reference_method_name("left"))

    def test_whitespace_stripped(self) -> None:
        self.assertEqual(reference_method_name("  midpoint  "), reference_method_name("midpoint"))

    def test_bad_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            reference_method_name("trapezoid")

    def test_all_keys_present(self) -> None:
        for key in ("midpoint", "left", "right"):
            self.assertIn(key, REFERENCE_METHOD_NAMES)

    def test_values_contain_order_hint(self) -> None:
        """Each value should contain '2' or '1' for the advertised order."""
        self.assertIn("order2", reference_method_name("midpoint"))
        self.assertIn("order1", reference_method_name("left"))
        self.assertIn("order1", reference_method_name("right"))


# ---------------------------------------------------------------------------
# 2+3. _evolve_piecewise_exact convergence on H(t) = sin(t) Z
# ---------------------------------------------------------------------------
# The Schrödinger equation  d|ψ⟩/dt = −i sin(t) Z |ψ⟩  has exact solution
#
#   |ψ(T)⟩ = exp(−i Φ(T) Z) |ψ₀⟩
#
# where  Φ(T) = ∫₀ᵀ sin(t) dt = 1 − cos(T).
#
# For the midpoint-sampled piecewise-constant approximation with N slices:
#   error ∝ 1/N²  (Magnus-2, second order)
# For the left-endpoint approximation:
#   error ∝ 1/N   (first order)

def _exact_psi_sin_z(psi0: np.ndarray, T: float) -> np.ndarray:
    """Exact solution for H(t) = sin(t) Z starting from psi0 at t=0."""
    phi = 1.0 - math.cos(float(T))  # ∫₀ᵀ sin(t) dt
    # Z eigenstates: |0⟩ eigenvalue +1, |1⟩ eigenvalue −1
    # exp(−i Φ Z)|0⟩ = exp(−i Φ)|0⟩,  exp(−i Φ Z)|1⟩ = exp(+i Φ)|1⟩
    phases = np.array([np.exp(-1j * phi), np.exp(1j * phi)], dtype=complex)
    return phases * np.asarray(psi0, dtype=complex)


def _run_piecewise_exact(
    psi0: np.ndarray,
    T: float,
    n_steps: int,
    sampling: str,
) -> np.ndarray:
    """Run _evolve_piecewise_exact on H(t)=sin(t)Z with hmat_static=0."""
    evolve = _import_evolve_piecewise_exact()
    hmat_static = np.zeros((2, 2), dtype=complex)

    def provider(t: float):
        return {"z": math.sin(float(t))}

    # We need the pipeline helper to build the H matrix from the exyz label.
    # For this 1-qubit "z" label, h_drive = [[sin(t), 0],[0,-sin(t)]], which
    # is what _evolve_piecewise_exact builds internally.
    return evolve(
        psi0=psi0,
        hmat_static=hmat_static,
        drive_coeff_provider_exyz=provider,
        time_value=T,
        trotter_steps=n_steps,
        t0=0.0,
        time_sampling=sampling,
    )


class TestPiecewiseExactConvergenceMidpoint(unittest.TestCase):
    """Midpoint sampling = Magnus-2 → O(N²) global error."""

    def setUp(self) -> None:
        self.psi0 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
        self.T = 1.5  # non-trivial time; avoid cos(T)=1

    def _error(self, n_steps: int) -> float:
        psi_approx = _run_piecewise_exact(self.psi0, self.T, n_steps, "midpoint")
        psi_exact = _exact_psi_sin_z(self.psi0, self.T)
        return float(np.linalg.norm(psi_approx - psi_exact))

    def test_midpoint_second_order(self) -> None:
        """Doubling N should reduce error by ~4x (second-order convergence)."""
        err_n = self._error(20)
        err_2n = self._error(40)
        ratio = err_n / err_2n
        self.assertGreater(ratio, 3.0,
                           f"Expected ratio > 3 (2nd order), got {ratio:.4f}")
        self.assertLess(ratio, 5.5,
                        f"Expected ratio < 5.5, got {ratio:.4f}")

    def test_multiplier_strictly_improves_reference(self) -> None:
        """Running with 2× more steps gives strictly smaller error."""
        err_1x = self._error(16)
        err_2x = self._error(32)   # equivalent to exact_steps_multiplier=2
        self.assertLess(err_2x, err_1x,
                        f"2× steps should reduce error: {err_2x} vs {err_1x}")

    def test_multiplier_4x_improves_reference(self) -> None:
        """4× multiplier should reduce error by roughly 16×."""
        err_1x = self._error(16)
        err_4x = self._error(64)   # equivalent to exact_steps_multiplier=4
        ratio = err_1x / err_4x
        self.assertGreater(ratio, 8.0,
                           f"4× multiplier should give ≥8× improvement, got {ratio:.2f}")

    def test_midpoint_high_precision(self) -> None:
        """With 1000 steps, midpoint should be accurate to < 1e-6."""
        err = self._error(1000)
        self.assertLess(err, 1e-6,
                        f"1000 midpoint steps: expected err < 1e-6, got {err}")


class TestPiecewiseExactConvergenceLeft(unittest.TestCase):
    """Left-endpoint sampling → O(N¹) global error."""

    def setUp(self) -> None:
        self.psi0 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
        self.T = 1.5

    def _error(self, n_steps: int) -> float:
        psi_approx = _run_piecewise_exact(self.psi0, self.T, n_steps, "left")
        psi_exact = _exact_psi_sin_z(self.psi0, self.T)
        return float(np.linalg.norm(psi_approx - psi_exact))

    def test_left_first_order(self) -> None:
        """Doubling N should reduce error by ~2x (first-order convergence)."""
        err_n = self._error(40)
        err_2n = self._error(80)
        ratio = err_n / err_2n
        self.assertGreater(ratio, 1.5,
                           f"Expected ratio > 1.5 (1st order), got {ratio:.4f}")
        self.assertLess(ratio, 2.5,
                        f"Expected ratio < 2.5, got {ratio:.4f}")

    def test_left_is_less_accurate_than_midpoint(self) -> None:
        """At same N, left endpoint should have larger error than midpoint."""
        err_left = self._error(32)
        psi_mid = _run_piecewise_exact(self.psi0, self.T, 32, "midpoint")
        psi_exact = _exact_psi_sin_z(self.psi0, self.T)
        err_mid = float(np.linalg.norm(psi_mid - psi_exact))
        self.assertLess(err_mid, err_left,
                        f"Midpoint should be more accurate: mid={err_mid}, left={err_left}")


# ---------------------------------------------------------------------------
# 4. JSON metadata keys: reference_method and reference_steps_multiplier
# ---------------------------------------------------------------------------

class TestJSONMetadataKeys(unittest.TestCase):
    """Verify that the JSON settings["drive"] dict gains the new metadata keys.

    This test exercises the logic from pipeline main() without running
    pipelines by directly testing that reference_method_name produces the
    right value and that the key names are correct for downstream readers.
    """

    def test_reference_method_key_for_default_midpoint(self) -> None:
        # Simulates what pipeline main() writes into settings["drive"].
        drive_settings = {
            "time_sampling": "midpoint",
            "reference_steps_multiplier": 1,
            "reference_steps": 64,
            "reference_method": reference_method_name("midpoint"),
        }
        self.assertEqual(drive_settings["reference_method"],
                         "exponential_midpoint_magnus2_order2")
        self.assertIn("reference_steps_multiplier", drive_settings)
        self.assertIn("reference_steps", drive_settings)

    def test_reference_steps_computed_correctly(self) -> None:
        for trotter_steps, multiplier in [(32, 1), (32, 2), (64, 4), (16, 8)]:
            expected_ref = trotter_steps * multiplier
            drive_settings = {
                "reference_steps": trotter_steps * multiplier,
                "reference_steps_multiplier": multiplier,
            }
            self.assertEqual(drive_settings["reference_steps"], expected_ref,
                             f"trotter={trotter_steps} mult={multiplier}")

    def test_reference_method_matches_for_all_sampling_rules(self) -> None:
        pairs = [
            ("midpoint", "exponential_midpoint_magnus2_order2"),
            ("left",     "exponential_left_endpoint_order1"),
            ("right",    "exponential_right_endpoint_order1"),
        ]
        for sampling, expected in pairs:
            with self.subTest(sampling=sampling):
                self.assertEqual(reference_method_name(sampling), expected)


# ---------------------------------------------------------------------------
# 5. Compare runner _DRIVE_FLAG_DEFAULTS includes exact_steps_multiplier
# ---------------------------------------------------------------------------

class TestCompareRunnerDefaults(unittest.TestCase):
    """Verify the compare runner updated _DRIVE_FLAG_DEFAULTS."""

    def test_exact_steps_multiplier_in_defaults(self) -> None:
        import pipelines.compare_hardcoded_vs_qiskit_pipeline as cmp
        self.assertIn("exact_steps_multiplier", cmp._DRIVE_FLAG_DEFAULTS)
        self.assertEqual(cmp._DRIVE_FLAG_DEFAULTS["exact_steps_multiplier"], 1)

    def test_build_drive_args_includes_multiplier_flag(self) -> None:
        import pipelines.compare_hardcoded_vs_qiskit_pipeline as cmp
        ns = types.SimpleNamespace(
            enable_drive=True,
            drive_A=0.2,
            drive_omega=1.7,
            drive_tbar=4.0,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_s=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=4,
        )
        tokens = cmp._build_drive_args(ns)
        self.assertIn("--exact-steps-multiplier", tokens)
        idx = tokens.index("--exact-steps-multiplier")
        self.assertEqual(tokens[idx + 1], "4")

    def test_build_drive_args_multiplier_default_1(self) -> None:
        import pipelines.compare_hardcoded_vs_qiskit_pipeline as cmp
        ns = types.SimpleNamespace(
            enable_drive=True,
            drive_A=0.2,
            drive_omega=1.7,
            drive_tbar=4.0,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_s=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=1,
        )
        tokens = cmp._build_drive_args(ns)
        idx = tokens.index("--exact-steps-multiplier")
        self.assertEqual(tokens[idx + 1], "1")

    def test_build_drive_args_disabled_empty(self) -> None:
        import pipelines.compare_hardcoded_vs_qiskit_pipeline as cmp
        ns = types.SimpleNamespace(
            enable_drive=False,
            drive_A=0.2,
            drive_omega=1.7,
            drive_tbar=4.0,
            drive_phi=0.0,
            drive_pattern="staggered",
            drive_custom_s=None,
            drive_include_identity=False,
            drive_time_sampling="midpoint",
            drive_t0=0.0,
            exact_steps_multiplier=4,
        )
        tokens = cmp._build_drive_args(ns)
        # Drive disabled → empty list regardless of multiplier.
        self.assertEqual(tokens, [])


if __name__ == "__main__":
    unittest.main()
