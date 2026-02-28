#!/usr/bin/env python3
"""Unit tests for projector-based trajectory subspace fidelity helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = REPO_ROOT / "Fermi-Hamil-JW-VQE-TROTTER-PIPELINE"

for p in (REPO_ROOT, PIPELINE_ROOT):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

import pipelines.hardcoded_hubbard_pipeline as hp


class TestProjectorFidelityMath(unittest.TestCase):
    def test_projector_fidelity_is_bounded(self) -> None:
        rng = np.random.default_rng(123)
        raw_basis = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
        basis = hp._orthonormalize_basis_columns(raw_basis)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        psi = hp._normalize_state(psi)
        got = hp._projector_fidelity_from_basis(basis, psi)
        self.assertGreaterEqual(got, 0.0)
        self.assertLessEqual(got, 1.0)

    def test_dimension_one_matches_scalar_overlap(self) -> None:
        rng = np.random.default_rng(456)
        vec = hp._normalize_state(rng.normal(size=16) + 1j * rng.normal(size=16))
        psi = hp._normalize_state(rng.normal(size=16) + 1j * rng.normal(size=16))
        got = hp._projector_fidelity_from_basis(vec.reshape(-1, 1), psi)
        expected = float(abs(np.vdot(vec, psi)) ** 2)
        self.assertAlmostEqual(got, expected, places=12)

    def test_two_state_degenerate_subspace_detects_membership(self) -> None:
        e1 = np.zeros(4, dtype=complex)
        e2 = np.zeros(4, dtype=complex)
        e1[0] = 1.0
        e2[1] = 1.0
        basis = np.column_stack([e1, e2])
        psi = (e1 + e2) / np.sqrt(2.0)

        subspace_fid = hp._projector_fidelity_from_basis(basis, psi)
        single_vec_fid = float(abs(np.vdot(e1, psi)) ** 2)

        self.assertAlmostEqual(subspace_fid, 1.0, places=12)
        self.assertLess(single_vec_fid, 0.75)


class TestGroundManifoldTolerance(unittest.TestCase):
    def test_energy_tolerance_includes_near_degenerate_state(self) -> None:
        num_sites = 2
        num_particles = hp._half_filled_particles(num_sites)  # (1,1)
        sector = hp._sector_basis_indices(num_sites, num_particles, ordering="blocked")
        dim = 1 << (2 * num_sites)
        diag = np.full(dim, 7.0, dtype=float)
        diag[int(sector[0])] = 0.0
        diag[int(sector[1])] = 5e-9
        hmat = np.diag(diag).astype(complex)

        e0, basis = hp._ground_manifold_basis_sector_filtered(
            hmat=hmat,
            num_sites=num_sites,
            num_particles=num_particles,
            ordering="blocked",
            energy_tol=1e-8,
        )

        self.assertAlmostEqual(e0, 0.0, places=14)
        self.assertEqual(basis.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
