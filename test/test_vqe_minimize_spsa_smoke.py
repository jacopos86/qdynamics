from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.vqe_latex_python_pairs import (
    apply_pauli_rotation,
    basis_state,
    expval_pauli_polynomial,
    vqe_minimize,
)


class _RxAnsatz:
    num_parameters = 1

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        angle = float(np.asarray(theta, dtype=float)[0])
        return apply_pauli_rotation(np.asarray(psi_ref, dtype=complex), "x", angle)


def test_vqe_minimize_supports_spsa_method_smoke() -> None:
    hamiltonian = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    ansatz = _RxAnsatz()
    psi_ref = basis_state(1, "0")

    # Mirror vqe_minimize restart initialization for reproducible baseline.
    rng = np.random.default_rng(0)
    theta0 = 0.3 * rng.normal(size=1)
    initial_energy = expval_pauli_polynomial(ansatz.prepare_state(theta0, psi_ref), hamiltonian)

    result = vqe_minimize(
        hamiltonian,
        ansatz,
        psi_ref,
        method="SPSA",
        maxiter=200,
        restarts=1,
        seed=0,
        spsa_a=0.2,
        spsa_c=0.1,
    )

    assert result.success is True
    assert np.isfinite(result.energy)
    assert result.energy < initial_energy
