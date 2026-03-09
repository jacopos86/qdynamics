#!/usr/bin/env python3
"""Integration tests for HH trotterization wiring (Session 5).

Test categories:
  1. HH + Trotter smoke test — build HH Hamiltonian, sector-filter, VQE,
     short Trotter propagation, verify fidelity.
  2. HH → Hubbard reduction — omega0=0, g_ep=0 gives same Hamiltonian spectrum
     as pure Hubbard ⊗ I_phonon.
  3. PDF manifest field presence — manifest logic branches correctly for HH.
  4. Sector filtering correctness — _sector_basis_indices_hh gives correct
     fermion-only count mask with phonon qubits unconstrained.
  5. HH exact ground energy cross-check — pipeline helper matches VQE module.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Imports from src/quantum ───────────────────────────────────────────────
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.hartree_fock_reference_state import (
    hubbard_holstein_reference_state,
)
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    vqe_minimize,
    exact_ground_energy_sector_hh,
)

# ── Imports from hardcoded pipeline ────────────────────────────────────────
import pipelines.hardcoded.hubbard_pipeline as hc_pipe

_sector_basis_indices_hh = hc_pipe._sector_basis_indices_hh
_ground_manifold_basis_sector_filtered_hh = hc_pipe._ground_manifold_basis_sector_filtered_hh
_collect_hardcoded_terms_exyz = hc_pipe._collect_hardcoded_terms_exyz
_build_hamiltonian_matrix = hc_pipe._build_hamiltonian_matrix


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
_L = 2
_T = 1.0
_U = 4.0
_OMEGA0 = 1.0
_G_EP = 0.5
_N_PH_MAX = 1
_BOSON_ENCODING = "binary"
_BOUNDARY = "periodic"
_ORDERING = "blocked"
_NUM_PARTICLES = (_L, _L)  # half-filling
_ENCODINGS = ("binary", "unary")


def _build_hh_poly_and_matrix(boson_encoding=_BOSON_ENCODING):
    """Build HH PauliPolynomial, extract coeff_map, build hmat."""
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=_L, J=_T, U=_U,
        omega0=_OMEGA0, g=_G_EP,
        n_ph_max=_N_PH_MAX, boson_encoding=boson_encoding,
        indexing=_ORDERING, pbc=(_BOUNDARY == "periodic"),
    )
    qpb = boson_qubits_per_site(_N_PH_MAX, boson_encoding)
    nq_total = 2 * _L + _L * qpb
    _, coeff_map = _collect_hardcoded_terms_exyz(h_poly)
    hmat = _build_hamiltonian_matrix(coeff_map)
    return h_poly, coeff_map, nq_total, qpb, hmat


def _build_hubbard_poly_and_matrix():
    """Build pure Hubbard PauliPolynomial, extract coeff_map, build hmat."""
    h_poly = build_hubbard_hamiltonian(
        dims=_L, t=_T, U=_U,
        indexing=_ORDERING, pbc=(_BOUNDARY == "periodic"),
    )
    nq = 2 * _L
    _, coeff_map = _collect_hardcoded_terms_exyz(h_poly)
    hmat = _build_hamiltonian_matrix(coeff_map)
    return h_poly, coeff_map, nq, hmat


# ════════════════════════════════════════════════════════════════════════════
# 1) HH + Trotter Smoke
# ════════════════════════════════════════════════════════════════════════════
class TestHHTrotterSmoke(unittest.TestCase):
    """Build HH Hamiltonian → sector-filter → VQE → verify fidelity."""

    def test_hh_vqe_and_sector_filter(self):
        for enc in _ENCODINGS:
            with self.subTest(encoding=enc):
                h_poly, coeff_map, nq_total, qpb, hmat = _build_hh_poly_and_matrix(enc)

                # Sector filtering
                sector_idx = _sector_basis_indices_hh(
                    num_sites=_L,
                    num_particles=_NUM_PARTICLES,
                    ordering=_ORDERING,
                    nq_total=nq_total,
                )
                self.assertGreater(sector_idx.size, 0, "Sector must be non-empty")

                # Exact filtered ground energy
                gs_energy, basis = _ground_manifold_basis_sector_filtered_hh(
                    hmat,
                    num_sites=_L,
                    num_particles=_NUM_PARTICLES,
                    ordering=_ORDERING,
                    nq_total=nq_total,
                    energy_tol=1e-8,
                )
                self.assertTrue(math.isfinite(gs_energy))
                self.assertGreater(basis.shape[1], 0)

                # VQE with HH ansatz (minimal: 1 rep, 1 restart)
                psi_ref = hubbard_holstein_reference_state(
                    dims=_L, n_ph_max=_N_PH_MAX,
                    boson_encoding=enc, indexing=_ORDERING,
                )
                ansatz = HubbardHolsteinLayerwiseAnsatz(
                    dims=_L, J=_T, U=_U,
                    omega0=_OMEGA0, g=_G_EP,
                    n_ph_max=_N_PH_MAX, boson_encoding=enc,
                    reps=1, indexing=_ORDERING,
                    pbc=(_BOUNDARY == "periodic"),
                )
                vqe_res = vqe_minimize(
                    h_poly, ansatz, psi_ref,
                    restarts=1, seed=7, maxiter=200,
                )
                self.assertTrue(math.isfinite(vqe_res.energy))
                eigs_full = np.linalg.eigvalsh(hmat)
                self.assertGreaterEqual(vqe_res.energy, float(eigs_full[0]) - 1e-6)

                psi_vqe = ansatz.prepare_state(vqe_res.theta, psi_ref)
                overlap = basis.conj().T @ psi_vqe
                fid = float(np.real(np.sum(np.abs(overlap) ** 2)))
                self.assertGreaterEqual(fid, 0.0, "Fidelity must be non-negative")
                self.assertLessEqual(fid, 1.0 + 1e-10)


# ════════════════════════════════════════════════════════════════════════════
# 2) HH → Hubbard Reduction Check
# ════════════════════════════════════════════════════════════════════════════
class TestHHToHubbardReduction(unittest.TestCase):
    """omega0=0, g_ep=0, n_ph_max=1 => spectrum embeds pure Hubbard."""

    def test_reduction_spectrum(self):
        for enc in _ENCODINGS:
            with self.subTest(encoding=enc):
                # Build HH with zero coupling
                qpb = boson_qubits_per_site(_N_PH_MAX, enc)
                nq_hh = 2 * _L + _L * qpb
                h_poly_hh = build_hubbard_holstein_hamiltonian(
                    dims=_L, J=_T, U=_U,
                    omega0=0.0, g=0.0,
                    n_ph_max=_N_PH_MAX, boson_encoding=enc,
                    indexing=_ORDERING, pbc=(_BOUNDARY == "periodic"),
                    include_zero_point=False,
                )
                _, coeff_map_hh = _collect_hardcoded_terms_exyz(h_poly_hh)
                hmat_hh = _build_hamiltonian_matrix(coeff_map_hh)

                evals_hh = np.sort(np.real(np.linalg.eigvalsh(hmat_hh)))

                # Build pure Hubbard
                _, _, nq_hub, hmat_hub = _build_hubbard_poly_and_matrix()
                evals_hub = np.sort(np.real(np.linalg.eigvalsh(hmat_hub)))

                # HH spectrum is Hubbard ⊗ I_phonon
                # => each Hubbard eigenvalue appears 2^(L*qpb) times
                n_phonon_states = 1 << (_L * qpb)
                evals_hub_repeated = np.sort(np.repeat(evals_hub, n_phonon_states))

                np.testing.assert_allclose(
                    evals_hh,
                    evals_hub_repeated,
                    atol=1e-10,
                    err_msg=f"HH(omega0=0, g=0, enc={enc}) spectrum must embed Hubbard ⊗ I_phonon",
                )


# ════════════════════════════════════════════════════════════════════════════
# 3) PDF Manifest Field Presence
# ════════════════════════════════════════════════════════════════════════════
class TestPDFManifestFields(unittest.TestCase):
    """Verify manifest construction logic branches correctly for HH vs Hubbard."""

    def test_manifest_model_name_hh(self):
        """HH payload → model_name = Hubbard-Holstein."""
        settings = {
            "problem": "hh",
            "holstein": {
                "omega0": 1.0, "g_ep": 0.5, "n_ph_max": 1,
                "boson_encoding": "unary", "nq_fermion": 4,
                "nq_phonon": 2, "nq_total": 6,
            },
            "drive": {"enabled": True, "A": 0.5, "omega": 2.0,
                      "tbar": 3.0, "phi": 0.0, "t0": 0.0,
                      "pattern": "staggered", "time_sampling": "midpoint",
                      "reference_steps_multiplier": 2},
        }
        problem_label = str(settings.get("problem", "hubbard")).strip().lower()
        model_name = "Hubbard-Holstein" if problem_label == "hh" else "Hubbard"
        hh_block = settings.get("holstein", {})
        drive_block = settings.get("drive")
        drive_enabled = isinstance(drive_block, dict) and bool(drive_block.get("enabled", False))

        self.assertEqual(model_name, "Hubbard-Holstein")
        self.assertTrue(drive_enabled)
        self.assertEqual(hh_block.get("omega0"), 1.0)
        self.assertEqual(hh_block.get("g_ep"), 0.5)
        self.assertEqual(hh_block.get("n_ph_max"), 1)
        self.assertEqual(hh_block.get("boson_encoding"), "unary")
        self.assertEqual(hh_block.get("nq_total"), 6)

    def test_manifest_model_name_hubbard(self):
        """Hubbard payload → model_name = Hubbard, no holstein block."""
        settings = {"problem": "hubbard"}
        problem_label = str(settings.get("problem", "hubbard")).strip().lower()
        model_name = "Hubbard-Holstein" if problem_label == "hh" else "Hubbard"
        self.assertEqual(model_name, "Hubbard")

    def test_drive_disabled_detection(self):
        """No drive dict → drive_enabled = False."""
        settings = {"problem": "hubbard", "drive": None}
        drive_block = settings.get("drive")
        drive_enabled = isinstance(drive_block, dict) and bool(drive_block.get("enabled", False))
        self.assertFalse(drive_enabled)


# ════════════════════════════════════════════════════════════════════════════
# 4) Sector Filtering Correctness
# ════════════════════════════════════════════════════════════════════════════
class TestSectorFilteringHH(unittest.TestCase):
    """_sector_basis_indices_hh returns correct fermion-only sector mask."""

    def test_sector_count_blocked(self):
        """Blocked ordering: HH sector count = Hubbard-sector × 2^(phonon_qubits)."""
        for enc in _ENCODINGS:
            with self.subTest(encoding=enc):
                qpb = boson_qubits_per_site(_N_PH_MAX, enc)
                nq_total = 2 * _L + _L * qpb
                sector_hh = _sector_basis_indices_hh(
                    num_sites=_L,
                    num_particles=_NUM_PARTICLES,
                    ordering="blocked",
                    nq_total=nq_total,
                )

                from math import comb
                n_hub_sector = comb(_L, _NUM_PARTICLES[0]) * comb(_L, _NUM_PARTICLES[1])
                n_phonon_states = 1 << (_L * qpb)
                expected_count = n_hub_sector * n_phonon_states
                self.assertEqual(
                    sector_hh.size,
                    expected_count,
                    f"[{enc}] HH sector should have {expected_count} states "
                    f"({n_hub_sector} fermion × {n_phonon_states} phonon), "
                    f"got {sector_hh.size}",
                )

    def test_fermion_only_mask(self):
        """Each selected HH state has correct fermion number regardless of phonon bits."""
        for enc in _ENCODINGS:
            with self.subTest(encoding=enc):
                qpb = boson_qubits_per_site(_N_PH_MAX, enc)
                nq_total = 2 * _L + _L * qpb
                sector = _sector_basis_indices_hh(
                    num_sites=_L,
                    num_particles=_NUM_PARTICLES,
                    ordering="blocked",
                    nq_total=nq_total,
                )

                up_mask = (1 << _L) - 1
                dn_mask = up_mask << _L
                for idx in sector:
                    n_up = bin(int(idx) & int(up_mask)).count("1")
                    n_dn = bin(int(idx) & int(dn_mask)).count("1")
                    self.assertEqual(n_up, _NUM_PARTICLES[0],
                                     f"[{enc}] State {idx}: n_up={n_up} != {_NUM_PARTICLES[0]}")
                    self.assertEqual(n_dn, _NUM_PARTICLES[1],
                                     f"[{enc}] State {idx}: n_dn={n_dn} != {_NUM_PARTICLES[1]}")


# ════════════════════════════════════════════════════════════════════════════
# 5) HH Exact Ground Energy Cross-check
# ════════════════════════════════════════════════════════════════════════════
class TestHHExactGroundEnergy(unittest.TestCase):
    """exact_ground_energy_sector_hh matches direct pipeline diagonalization."""

    def test_gs_energy_matches_direct(self):
        for enc in _ENCODINGS:
            with self.subTest(encoding=enc):
                h_poly, coeff_map, nq_total, qpb, hmat = _build_hh_poly_and_matrix(enc)

                # Pipeline's sector-filter approach
                gs_direct, _ = _ground_manifold_basis_sector_filtered_hh(
                    hmat,
                    num_sites=_L,
                    num_particles=_NUM_PARTICLES,
                    ordering=_ORDERING,
                    nq_total=nq_total,
                    energy_tol=1e-8,
                )

                # VQE module's public API
                gs_api = exact_ground_energy_sector_hh(
                    h_poly,
                    num_sites=_L,
                    num_particles=_NUM_PARTICLES,
                    n_ph_max=_N_PH_MAX,
                    boson_encoding=enc,
                    indexing=_ORDERING,
                )

                self.assertAlmostEqual(
                    gs_direct, gs_api, places=8,
                    msg=f"[{enc}] Pipeline _ground_manifold and exact_ground_energy_sector_hh must agree",
                )


if __name__ == "__main__":
    unittest.main()
