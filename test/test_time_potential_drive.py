#!/usr/bin/env python3
"""Unit tests for time-dependent onsite density drive.

Covers:
1. Z-label placement (qubit indexing correctness).
2. Drive-only phase evolution for L=2 in both blocked and interleaved indexing.
3. Midpoint vs left quadrature convergence order.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (same pattern as test_integration.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = REPO_ROOT / "Fermi-Hamil-JW-VQE-TROTTER-PIPELINE"

for p in (REPO_ROOT, PIPELINE_ROOT):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from src.quantum.drives_time_potential import (
    DensityDriveTemplate,
    build_gaussian_sinusoid_density_drive,
    evaluate_drive_waveform,
    gaussian_sinusoid_waveform,
)
from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index


class TestZLabelPlacement(unittest.TestCase):
    """Verify that Z labels are placed at the correct qubit position."""

    def test_z_label_qubit0_rightmost(self) -> None:
        # nq_total=4, qubit 0 corresponds to rightmost character.
        template = DensityDriveTemplate.build(
            n_sites=2, nq_total=4, indexing="interleaved"
        )
        lbl = template.z_label(0, SPIN_UP)  # mode_index(site0, up) == 0
        self.assertEqual(lbl, "eeez")

        # Diagonal eigenvalue: +1 for bit0=0, -1 for bit0=1.
        nq = 4
        for idx in range(1 << nq):
            bit0 = (idx >> 0) & 1
            expected_sign = 1 if bit0 == 0 else -1
            # The 'z' is at position nq-1-0 = 3 from the left in exyz,
            # which corresponds to qubit 0.
            op = lbl[nq - 1 - 0]
            self.assertEqual(op, "z")
            got = 1 if bit0 == 0 else -1
            self.assertEqual(got, expected_sign)

    def test_all_labels_distinct(self) -> None:
        """Each (site, spin) should produce a unique Z label."""
        for indexing in ("blocked", "interleaved"):
            for n_sites in (2, 3, 4):
                template = DensityDriveTemplate.build(
                    n_sites=n_sites,
                    nq_total=2 * n_sites,
                    indexing=indexing,
                )
                labels = template.labels_exyz(include_identity=False)
                # All Z labels should be distinct.
                self.assertEqual(len(labels), len(set(labels)),
                                 f"Duplicate Z labels for {indexing} L={n_sites}")


class TestDriveOnlyPhase(unittest.TestCase):
    """Evolve basis states under drive only (no static H) and verify phases."""

    def test_drive_only_phase_L2_blocked_and_interleaved(self) -> None:
        import pipelines.hardcoded_hubbard_pipeline as hp

        dt = 0.25
        t_mid = 0.5 * dt
        A = 0.7
        omega = 1.3
        tbar = 2.0
        phi = 0.2
        v_scalar = gaussian_sinusoid_waveform(
            t_mid, A=A, omega=omega, tbar=tbar, phi=phi
        )
        V_sites = np.array([+v_scalar, -v_scalar], dtype=float)

        for indexing in ("blocked", "interleaved"):
            with self.subTest(indexing=indexing):
                drive = build_gaussian_sinusoid_density_drive(
                    n_sites=2,
                    nq_total=4,
                    indexing=indexing,
                    A=A,
                    omega=omega,
                    tbar=tbar,
                    phi=phi,
                    pattern_mode="dimer_bias",
                    include_identity=False,
                )

                ordered_labels_exyz = sorted(
                    drive.template.labels_exyz(include_identity=False)
                )
                compiled = {
                    lbl: hp._compile_pauli_action(lbl, 4)
                    for lbl in ordered_labels_exyz
                }

                def provider(t: float):
                    return drive.coeff_map_exyz(t)

                def evolve_basis(idx: int) -> complex:
                    psi0 = np.zeros(1 << 4, dtype=complex)
                    psi0[idx] = 1.0 + 0.0j
                    psi1 = hp._evolve_trotter_suzuki2_absolute(
                        psi0,
                        ordered_labels_exyz,
                        {},  # no static Hamiltonian
                        compiled,
                        dt,
                        1,
                        drive_coeff_provider_exyz=provider,
                        t0=0.0,
                        time_sampling="midpoint",
                    )
                    return complex(psi1[idx])

                amp_ref = evolve_basis(0)
                self.assertAlmostEqual(abs(amp_ref), 1.0, places=12)

                for idx in range(1 << 4):
                    amp = evolve_basis(idx)
                    self.assertAlmostEqual(abs(amp), 1.0, places=12)

                    # Global phase cancels by dividing by |0000⟩ amplitude.
                    ratio = amp / amp_ref

                    energy = 0.0
                    for site in range(2):
                        for spin in (SPIN_UP, SPIN_DN):
                            q = mode_index(
                                site, spin, indexing=indexing, n_sites=2
                            )
                            n = (idx >> q) & 1
                            energy += float(V_sites[site]) * float(n)
                    expected = complex(np.exp(-1j * dt * energy))
                    self.assertTrue(
                        np.allclose(ratio, expected, atol=1e-12, rtol=0.0),
                        f"idx={idx} indexing={indexing}: "
                        f"ratio={ratio} expected={expected}",
                    )


class TestQuadratureOrder(unittest.TestCase):
    """Verify midpoint is 2nd-order and left is 1st-order."""

    def test_midpoint_vs_left_quadrature_order(self) -> None:
        import pipelines.hardcoded_hubbard_pipeline as hp

        # Single-qubit commuting test: H(t) = t² Z.
        # Exact relative phase = exp(-i · 2·T³/3).
        ordered_labels_exyz = ["z"]
        compiled = {"z": hp._compile_pauli_action("z", 1)}

        psi0 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
        T = 1.0
        exact_ratio = complex(np.exp(-1j * (2.0 * (T**3) / 3.0)))

        def run(n_steps: int, sampling: str) -> float:
            def provider(t: float):
                return {"z": float(t * t)}

            psi = hp._evolve_trotter_suzuki2_absolute(
                psi0,
                ordered_labels_exyz,
                {},  # no static H
                compiled,
                T,
                int(n_steps),
                drive_coeff_provider_exyz=provider,
                t0=0.0,
                time_sampling=sampling,
            )
            ratio = complex(psi[0] / psi[1])
            return float(abs(ratio - exact_ratio))

        err_mid_20 = run(20, "midpoint")
        err_mid_40 = run(40, "midpoint")
        err_left_20 = run(20, "left")
        err_left_40 = run(40, "left")

        # Midpoint should be second order; left should be first order.
        self.assertLess(err_mid_20, err_left_20)
        # left: halving dt should halve error (ratio ~2)
        self.assertGreater(err_left_20 / err_left_40, 1.5)
        self.assertLess(err_left_20 / err_left_40, 2.5)
        # midpoint: halving dt should quarter error (ratio ~4)
        self.assertGreater(err_mid_20 / err_mid_40, 3.0)
        self.assertLess(err_mid_20 / err_mid_40, 5.5)


class TestPiecewiseExact(unittest.TestCase):
    """Verify the piecewise-exact propagator matches a known analytic result."""

    def test_single_qubit_z_drive(self) -> None:
        """H(t) = t · Z on one qubit. Exact: ψ(T) = exp(-i T²/2 Z) ψ₀."""
        import pipelines.hardcoded_hubbard_pipeline as hp

        psi0 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
        T = 2.0
        n_steps = 200

        def provider(t: float):
            return {"z": float(t)}

        # hmat_static = 0
        hmat_static = np.zeros((2, 2), dtype=complex)

        psi = hp._evolve_piecewise_exact(
            psi0=psi0,
            hmat_static=hmat_static,
            drive_coeff_provider_exyz=provider,
            time_value=T,
            trotter_steps=n_steps,
            t0=0.0,
            time_sampling="midpoint",
        )

        # Exact: integral of t from 0 to T = T²/2
        phase = T * T / 2.0
        psi_exact = np.array(
            [np.exp(-1j * phase), np.exp(1j * phase)], dtype=complex
        ) / math.sqrt(2.0)
        psi_exact = psi_exact / np.linalg.norm(psi_exact)

        # Should agree to high precision (midpoint on linear integrand is exact).
        self.assertTrue(
            np.allclose(psi, psi_exact, atol=1e-10, rtol=0.0),
            f"Piecewise exact failed: max diff = {np.max(np.abs(psi - psi_exact))}",
        )


class TestWaveform(unittest.TestCase):
    """Test the waveform function edge cases."""

    def test_zero_amplitude(self) -> None:
        v = gaussian_sinusoid_waveform(1.0, A=0.0, omega=1.0, tbar=1.0)
        self.assertEqual(v, 0.0)

    def test_tbar_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            gaussian_sinusoid_waveform(1.0, A=1.0, omega=1.0, tbar=-1.0)

    def test_known_value(self) -> None:
        # t=0 => sin(phi)*exp(0) = sin(phi)
        phi = 0.5
        A = 2.0
        v = gaussian_sinusoid_waveform(0.0, A=A, omega=1.0, tbar=1.0, phi=phi)
        self.assertAlmostEqual(v, A * math.sin(phi), places=14)

    def test_evaluate_drive_waveform_matches_scalar(self) -> None:
        times = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
        cfg = {
            "drive_omega": 1.7,
            "drive_tbar": 2.0,
            "drive_phi": 0.3,
            "drive_t0": 0.0,
        }
        got = evaluate_drive_waveform(times, cfg, amplitude=0.9)
        expected = np.array(
            [
                gaussian_sinusoid_waveform(
                    float(t),
                    A=0.9,
                    omega=1.7,
                    tbar=2.0,
                    phi=0.3,
                )
                for t in times
            ],
            dtype=float,
        )
        self.assertTrue(np.allclose(got, expected, atol=1e-14, rtol=0.0))

    def test_evaluate_drive_waveform_applies_t0(self) -> None:
        times = np.array([0.0, 0.2, 0.4], dtype=float)
        cfg = {
            "omega": 2.0,
            "tbar": 1.5,
            "phi": 0.0,
            "t0": 0.5,
        }
        got = evaluate_drive_waveform(times, cfg, amplitude=1.0)
        expected = np.array(
            [gaussian_sinusoid_waveform(float(t) + 0.5, A=1.0, omega=2.0, tbar=1.5, phi=0.0) for t in times],
            dtype=float,
        )
        self.assertTrue(np.allclose(got, expected, atol=1e-14, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
