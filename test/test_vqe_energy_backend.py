from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on path (same pattern as other integration tests).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hartree_fock_reference_state import (
    hartree_fock_statevector,
    hubbard_holstein_reference_state,
)
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinLayerwiseAnsatz,
    HubbardLayerwiseAnsatz,
    vqe_minimize,
)


def test_vqe_energy_backend_one_apply_matches_legacy_hubbard():
    H = build_hubbard_hamiltonian(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.1,
        indexing="blocked",
        pbc=True,
    )
    ansatz = HubbardLayerwiseAnsatz(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.1,
        reps=1,
        indexing="blocked",
        pbc=True,
    )
    psi_ref = hartree_fock_statevector(n_sites=2, num_particles=(1, 1), indexing="blocked")

    legacy = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=1,
        seed=123,
        maxiter=120,
        energy_backend="legacy",
    )
    fast = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=1,
        seed=123,
        maxiter=120,
        energy_backend="one_apply_compiled",
    )

    assert np.isfinite(legacy.energy)
    assert np.isfinite(fast.energy)
    assert abs(fast.energy - legacy.energy) < 1e-9


def test_vqe_energy_backend_one_apply_matches_legacy_hh():
    H = build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
        pbc=True,
    )
    ansatz = HubbardHolsteinLayerwiseAnsatz(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        indexing="blocked",
        pbc=True,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=2,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
    )

    legacy = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=1,
        seed=321,
        maxiter=180,
        energy_backend="legacy",
    )
    fast = vqe_minimize(
        H,
        ansatz,
        psi_ref,
        restarts=1,
        seed=321,
        maxiter=180,
        energy_backend="one_apply_compiled",
    )

    assert np.isfinite(legacy.energy)
    assert np.isfinite(fast.energy)
    assert abs(fast.energy - legacy.energy) < 1e-9
