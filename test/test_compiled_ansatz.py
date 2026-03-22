from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on path (same pattern as other integration tests).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HubbardHolsteinTermwiseAnsatz,
    HubbardTermwiseAnsatz,
    PauliPolynomial,
    PauliTerm,
    apply_exp_pauli_polynomial_termwise,
)


def _random_state(nq: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    psi = rng.normal(size=1 << int(nq)) + 1j * rng.normal(size=1 << int(nq))
    psi = np.asarray(psi, dtype=complex)
    return psi / np.linalg.norm(psi)


def test_compiled_ansatz_parity_hubbard_termwise():
    ansatz = HubbardTermwiseAnsatz(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.2,
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
        include_potential_terms=True,
    )
    rng = np.random.default_rng(111)
    theta = rng.normal(scale=0.3, size=ansatz.num_parameters)
    psi_ref = _random_state(ansatz.nq, seed=112)

    psi_slow = ansatz.prepare_state(
        theta,
        psi_ref,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        ansatz.base_terms,
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
    )
    psi_fast = executor.prepare_state(theta, psi_ref)

    assert np.linalg.norm(psi_slow - psi_fast) < 1e-10


def test_compiled_ansatz_parity_hh_termwise():
    ansatz = HubbardHolsteinTermwiseAnsatz(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        v=[0.1, -0.1],
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
        include_zero_point=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    assert int(ansatz.nq) <= 10

    rng = np.random.default_rng(211)
    theta = rng.normal(scale=0.2, size=ansatz.num_parameters)
    psi_ref = _random_state(ansatz.nq, seed=212)

    psi_slow = ansatz.prepare_state(
        theta,
        psi_ref,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        ansatz.base_terms,
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
    )
    psi_fast = executor.prepare_state(theta, psi_ref)

    assert np.linalg.norm(psi_slow - psi_fast) < 1e-10


def test_compiled_ansatz_parity_per_pauli_term_mode():
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xx", pc=1.0),
            PauliTerm(2, ps="zz", pc=0.5),
        ],
    )
    term = AnsatzTerm(label="multi", polynomial=poly)
    theta_runtime = np.array([0.2, -0.15], dtype=float)
    psi_ref = _random_state(2, seed=313)

    psi_slow = apply_exp_pauli_polynomial_termwise(
        psi_ref,
        poly,
        theta_runtime,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        [term],
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="per_pauli_term",
    )
    psi_fast = executor.prepare_state(theta_runtime, psi_ref)

    assert np.linalg.norm(psi_slow - psi_fast) < 1e-10
