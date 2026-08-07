from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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
from src.quantum import vqe_latex_python_pairs as vqe_module
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


class _IdentityOneParameterAnsatz:
    num_parameters = 1

    def prepare_state(self, theta, psi_ref):  # noqa: ANN001, ANN201
        np.asarray(theta, dtype=float).reshape(1)
        return np.asarray(psi_ref, dtype=complex)


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


def test_vqe_minimize_objective_value_transform_default_unchanged_and_seen_by_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vqe_module, "_try_import_scipy_minimize", lambda: None)
    hamiltonian = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    ansatz = _IdentityOneParameterAnsatz()
    psi_ref = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

    default = vqe_module.vqe_minimize(
        hamiltonian,
        ansatz,
        psi_ref,
        restarts=1,
        seed=1,
        initial_point=np.array([0.0]),
        use_initial_point_first_restart=True,
        maxiter=0,
    )

    seen: list[dict[str, object]] = []

    def _transform(event: dict[str, object]) -> float:
        seen.append(event)
        assert event["surface"] == "vqe_objective"
        assert event["energy_ideal"] == pytest.approx(1.0)
        return -2.0

    transformed = vqe_module.vqe_minimize(
        hamiltonian,
        ansatz,
        psi_ref,
        restarts=1,
        seed=1,
        initial_point=np.array([0.0]),
        use_initial_point_first_restart=True,
        maxiter=0,
        objective_value_transform=_transform,
    )

    assert default.energy == pytest.approx(1.0)
    assert transformed.energy == pytest.approx(-2.0)
    assert seen
    assert np.asarray(seen[0]["theta"], dtype=float).shape == (1,)
