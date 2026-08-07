"""
Independent ED ↔ Pauli-polynomial Hamiltonian cross-check.

For several parameter sets and both boson encodings (binary, unary), this test:
  1. Builds the HH Hamiltonian via the *independent* ED module (occupation-number basis,
     no Pauli algebra).
  2. Builds the HH Hamiltonian via the *Pauli-polynomial* path (PauliPolynomial →
     matrix → sector projection).
  3. Verifies that the sector eigenvalues agree to machine precision.

This is the strongest correctness test: if it passes, the entire Pauli-to-matrix
pipeline for Hubbard-Holstein is correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT,):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

# ── Pauli-polynomial path ─────────────────────────────────────────────────
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

# ── Independent ED path ───────────────────────────────────────────────────
from src.quantum.ed_hubbard_holstein import (
    build_hh_sector_basis,
    build_hh_sector_hamiltonian_ed,
    hermiticity_residual,
    matrix_to_dense,
)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _sector_eigenvalues_pauli(
    *,
    dims,
    J, U, omega0, g, n_ph_max,
    boson_encoding, indexing, pbc,
    num_particles,
    include_zero_point=True,
    delta_v=None,
) -> np.ndarray:
    """Build HH via Pauli polynomial, project to sector, return sorted eigenvalues."""
    # delta_v in the ED module means: H_drive = sum_i,sigma delta_v_i * n_i,sigma
    # In the Pauli builder: pass as v_t (static values), v0=None  →  delta = v_t - 0
    # build_hubbard_potential applies: (-v_for_existing)*n where v_for_existing = -delta
    # Net: +delta_v * n  — matches.
    v_t_arg = None
    if delta_v is not None:
        v_t_arg = delta_v  # pass dict/list directly as static v_t
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=dims, J=J, U=U, omega0=omega0, g=g,
        n_ph_max=n_ph_max, boson_encoding=boson_encoding,
        indexing=indexing, pbc=pbc,
        include_zero_point=include_zero_point,
        v_t=v_t_arg, v0=None,
    )
    # Full Hilbert-space matrix
    hmat_full = hamiltonian_matrix(h_poly)

    # Build ED basis to get the sector indices
    basis = build_hh_sector_basis(
        dims=dims, n_ph_max=n_ph_max,
        num_particles=num_particles,
        indexing=indexing, boson_encoding=boson_encoding,
    )
    idx = np.array(basis.basis_indices, dtype=int)

    # Project: H_sector = P^T H P  where P selects sector rows
    hmat_sector = hmat_full[np.ix_(idx, idx)]
    evals = np.sort(np.real(np.linalg.eigvalsh(hmat_sector)))
    return evals


def _sector_eigenvalues_ed(
    *,
    dims,
    J, U, omega0, g, n_ph_max,
    boson_encoding, indexing, pbc,
    num_particles,
    include_zero_point=True,
    delta_v=None,
) -> np.ndarray:
    """Build HH via independent ED, return sorted eigenvalues."""
    h_mat = build_hh_sector_hamiltonian_ed(
        dims=dims, J=J, U=U, omega0=omega0, g=g,
        n_ph_max=n_ph_max, num_particles=num_particles,
        indexing=indexing, boson_encoding=boson_encoding,
        pbc=pbc, delta_v=delta_v,
        include_zero_point=include_zero_point,
        sparse=False,
    )
    h_dense = matrix_to_dense(h_mat)
    evals = np.sort(np.real(np.linalg.eigvalsh(h_dense)))
    return evals


# ════════════════════════════════════════════════════════════════════════════
# Parameter sets
# ════════════════════════════════════════════════════════════════════════════

_ENCODINGS = ("binary", "unary")

# Each tuple: (label, dims, J, U, omega0, g, n_ph_max, pbc, num_particles, include_zp, delta_v)
_PARAM_SETS = [
    # Basic half-filling with moderate coupling
    ("L2_half_fill_g0.5", 2, 1.0, 4.0, 1.0, 0.5, 1, True, (1, 1), True, None),
    # Zero coupling  → pure Hubbard ⊗ I_phonon
    ("L2_zero_coupling", 2, 1.0, 4.0, 0.0, 0.0, 1, True, (1, 1), False, None),
    # Strong coupling
    ("L2_strong_g", 2, 1.0, 2.0, 2.0, 1.5, 1, True, (1, 1), True, None),
    # Open boundary
    ("L2_obc", 2, 1.0, 4.0, 1.0, 0.5, 1, False, (1, 1), True, None),
    # Asymmetric filling (1 up, 0 down)
    ("L2_1up_0dn", 2, 1.0, 4.0, 1.0, 0.5, 1, True, (1, 0), True, None),
    # n_ph_max=2 (larger phonon space)
    ("L2_nph2", 2, 1.0, 4.0, 1.0, 0.5, 2, True, (1, 1), True, None),
    # Site-dependent potential
    ("L2_delta_v", 2, 1.0, 4.0, 1.0, 0.5, 1, True, (1, 1), True, {0: 0.3, 1: -0.3}),
]


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

class TestEDvsPauliSpectrumMatch:
    """Sector eigenvalues from ED must match Pauli-polynomial projection."""

    @pytest.mark.parametrize("enc", _ENCODINGS)
    @pytest.mark.parametrize(
        "label, dims, J, U, omega0, g, n_ph_max, pbc, num_particles, include_zp, delta_v",
        _PARAM_SETS,
        ids=[p[0] for p in _PARAM_SETS],
    )
    def test_spectrum_match(
        self, enc, label, dims, J, U, omega0, g, n_ph_max, pbc,
        num_particles, include_zp, delta_v,
    ):
        evals_pauli = _sector_eigenvalues_pauli(
            dims=dims, J=J, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding=enc,
            indexing="blocked", pbc=pbc,
            num_particles=num_particles,
            include_zero_point=include_zp, delta_v=delta_v,
        )
        evals_ed = _sector_eigenvalues_ed(
            dims=dims, J=J, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding=enc,
            indexing="blocked", pbc=pbc,
            num_particles=num_particles,
            include_zero_point=include_zp, delta_v=delta_v,
        )
        assert evals_pauli.shape == evals_ed.shape, (
            f"[{enc}/{label}] dimension mismatch: "
            f"Pauli={evals_pauli.shape}, ED={evals_ed.shape}"
        )
        np.testing.assert_allclose(
            evals_pauli, evals_ed, atol=1e-10,
            err_msg=f"[{enc}/{label}] eigenvalue mismatch",
        )


class TestEDHermiticity:
    """ED Hamiltonian must be Hermitian."""

    @pytest.mark.parametrize("enc", _ENCODINGS)
    @pytest.mark.parametrize(
        "label, dims, J, U, omega0, g, n_ph_max, pbc, num_particles, include_zp, delta_v",
        _PARAM_SETS,
        ids=[p[0] for p in _PARAM_SETS],
    )
    def test_hermiticity(
        self, enc, label, dims, J, U, omega0, g, n_ph_max, pbc,
        num_particles, include_zp, delta_v,
    ):
        h_mat = build_hh_sector_hamiltonian_ed(
            dims=dims, J=J, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, num_particles=num_particles,
            indexing="blocked", boson_encoding=enc,
            pbc=pbc, delta_v=delta_v,
            include_zero_point=include_zp,
            sparse=False,
        )
        h_dense = matrix_to_dense(h_mat)
        resid = hermiticity_residual(h_dense)
        assert resid < 1e-13, (
            f"[{enc}/{label}] Hermiticity residual {resid} exceeds tolerance"
        )


class TestEDBasisDimension:
    """ED sector basis dimension must match Pauli projection dimension."""

    @pytest.mark.parametrize("enc", _ENCODINGS)
    def test_basis_dim_matches_sector_projection(self, enc):
        """For a fixed parameter set, verify dim(ED basis) == dim(Pauli sector)."""
        dims = 2
        n_ph_max = 1
        num_particles = (1, 1)
        basis = build_hh_sector_basis(
            dims=dims, n_ph_max=n_ph_max,
            num_particles=num_particles,
            indexing="blocked", boson_encoding=enc,
        )
        # Physical dimension: C(L, n_up) * C(L, n_dn) * (n_ph_max+1)^L
        from math import comb
        n_sites = 2
        expected = comb(n_sites, num_particles[0]) * comb(n_sites, num_particles[1]) * (n_ph_max + 1) ** n_sites
        assert basis.dimension == expected, (
            f"[{enc}] ED basis dim={basis.dimension}, expected={expected}"
        )


class TestEDGroundStateEnergy:
    """ED ground-state energy matches VQE module's exact_ground_energy_sector_hh."""

    @pytest.mark.parametrize("enc", _ENCODINGS)
    def test_gs_energy_matches_vqe_module(self, enc):
        from src.quantum.vqe_latex_python_pairs import exact_ground_energy_sector_hh

        dims = 2
        J, U, omega0, g = 1.0, 4.0, 1.0, 0.5
        n_ph_max = 1
        num_particles = (1, 1)

        # ED path
        evals_ed = _sector_eigenvalues_ed(
            dims=dims, J=J, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding=enc,
            indexing="blocked", pbc=True,
            num_particles=num_particles,
            include_zero_point=True,
        )
        e_gs_ed = float(evals_ed[0])

        # VQE module path
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=dims, J=J, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding=enc,
            indexing="blocked", pbc=True,
        )
        e_gs_vqe = exact_ground_energy_sector_hh(
            h_poly, num_sites=dims, num_particles=num_particles,
            n_ph_max=n_ph_max, boson_encoding=enc,
            indexing="blocked",
        )

        assert abs(e_gs_ed - e_gs_vqe) < 1e-10, (
            f"[{enc}] ED gs={e_gs_ed}, VQE module gs={e_gs_vqe}"
        )


class TestL3PaperISectorReference:
    """L=3 Paper-I weak-Holstein sector reference must remain fixed."""

    def test_l3_strong_weak_binary_open_matches_sector_gold(self):
        dims = 3
        J, U, omega0, g = 1.0, 8.0, 1.0, 0.3535533905932738
        n_ph_max = 1
        num_particles = (2, 1)
        kwargs = dict(
            dims=dims,
            J=J,
            U=U,
            omega0=omega0,
            g=g,
            n_ph_max=n_ph_max,
            boson_encoding="binary",
            indexing="blocked",
            pbc=False,
            num_particles=num_particles,
            include_zero_point=True,
        )

        evals_ed = _sector_eigenvalues_ed(**kwargs)
        evals_pauli = _sector_eigenvalues_pauli(**kwargs)
        h_sparse = build_hh_sector_hamiltonian_ed(
            dims=dims,
            J=J,
            U=U,
            omega0=omega0,
            g=g,
            n_ph_max=n_ph_max,
            num_particles=num_particles,
            indexing="blocked",
            boson_encoding="binary",
            pbc=False,
            include_zero_point=True,
            sparse=True,
        )
        evals_sparse = np.sort(np.real(np.linalg.eigvalsh(matrix_to_dense(h_sparse))))

        np.testing.assert_allclose(evals_ed, evals_pauli, atol=1e-10)
        np.testing.assert_allclose(evals_ed, evals_sparse, atol=1e-10)
        assert abs(float(evals_ed[0]) - 0.79018469638518) < 1e-10

        no_zp = _sector_eigenvalues_ed(
            dims=dims,
            J=J,
            U=U,
            omega0=omega0,
            g=g,
            n_ph_max=n_ph_max,
            boson_encoding="binary",
            indexing="blocked",
            pbc=False,
            num_particles=num_particles,
            include_zero_point=False,
        )
        assert abs(float(evals_ed[0] - no_zp[0]) - 1.5) < 1e-10


class TestEncodingConsistency:
    """Binary and unary encodings must give the same physical eigenvalues."""

    @pytest.mark.parametrize(
        "label, dims, J, U, omega0, g, n_ph_max, pbc, num_particles, include_zp, delta_v",
        _PARAM_SETS,
        ids=[p[0] for p in _PARAM_SETS],
    )
    def test_binary_unary_same_spectrum(
        self, label, dims, J, U, omega0, g, n_ph_max, pbc,
        num_particles, include_zp, delta_v,
    ):
        """Both encodings must give identical sector eigenvalues."""
        evals_bin = _sector_eigenvalues_ed(
            dims=dims, J=J, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding="binary",
            indexing="blocked", pbc=pbc,
            num_particles=num_particles,
            include_zero_point=include_zp, delta_v=delta_v,
        )
        evals_uni = _sector_eigenvalues_ed(
            dims=dims, J=J, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding="unary",
            indexing="blocked", pbc=pbc,
            num_particles=num_particles,
            include_zero_point=include_zp, delta_v=delta_v,
        )
        assert evals_bin.shape == evals_uni.shape, (
            f"[{label}] dim mismatch binary={evals_bin.shape} vs unary={evals_uni.shape}"
        )
        np.testing.assert_allclose(
            evals_bin, evals_uni, atol=1e-10,
            err_msg=f"[{label}] binary vs unary eigenvalue mismatch",
        )


class TestRandomizedCrosscheck:
    """Randomized parameter sweep for extra confidence."""

    @pytest.mark.parametrize("enc", _ENCODINGS)
    @pytest.mark.parametrize("seed", [42, 137, 271])
    def test_random_params(self, enc, seed):
        rng = np.random.default_rng(seed)
        J = float(rng.uniform(0.5, 2.0))
        U = float(rng.uniform(0.0, 6.0))
        omega0 = float(rng.uniform(0.0, 3.0))
        g = float(rng.uniform(0.0, 2.0))
        n_ph_max = int(rng.integers(1, 3))
        pbc = bool(rng.choice([True, False]))
        # L=2, half-filling
        dims = 2
        num_particles = (1, 1)

        evals_pauli = _sector_eigenvalues_pauli(
            dims=dims, J=J, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding=enc,
            indexing="blocked", pbc=pbc,
            num_particles=num_particles,
        )
        evals_ed = _sector_eigenvalues_ed(
            dims=dims, J=J, U=U, omega0=omega0, g=g,
            n_ph_max=n_ph_max, boson_encoding=enc,
            indexing="blocked", pbc=pbc,
            num_particles=num_particles,
        )
        np.testing.assert_allclose(
            evals_pauli, evals_ed, atol=1e-10,
            err_msg=f"[{enc}/seed={seed}] J={J:.3f} U={U:.3f} ω₀={omega0:.3f} g={g:.3f}",
        )
