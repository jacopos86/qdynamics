from __future__ import annotations

import numpy as np
import pytest

from pipelines.qse_spectra.adaptive_qse_benchmark import (
    STOP_RESIDUAL_CONVERGED,
    guarded_davidson_correction,
    orthogonalize_adaptive_direction,
    run_adaptive_qse_benchmark,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def test_random_hermitian_converges_to_exact_lowest_roots() -> None:
    rng = np.random.default_rng(20260826)
    raw = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    hamiltonian = 0.5 * (raw + raw.conj().T)
    prepared = rng.normal(size=16) + 1j * rng.normal(size=16)

    result = run_adaptive_qse_benchmark(
        hamiltonian,
        prepared,
        target_roots=3,
        eps_residual=1.0e-11,
        max_dimension=16,
        seed_elements=(),
    )

    exact = np.linalg.eigvalsh(hamiltonian)[:3]
    assert result["stop_reason"] == STOP_RESIDUAL_CONVERGED
    assert result["root_energies"] == pytest.approx(exact, abs=1.0e-10)
    assert result["max_root_residual"] <= 1.0e-11
    assert result["iterations"][-1]["max_root_residual"] <= 1.0e-11


def test_preconditioner_guard_is_finite_at_zero_denominator() -> None:
    diagonal = np.asarray([0.0, 1.0, 2.0])
    residual = np.asarray([1.0 + 2.0j, -3.0, 4.0])

    correction = guarded_davidson_correction(
        diagonal,
        1.0,
        residual,
        denominator_floor=1.0e-8,
    )

    assert np.all(np.isfinite(correction.real))
    assert np.all(np.isfinite(correction.imag))
    assert correction[1] == pytest.approx(-3.0e8 + 0.0j)


def test_direction_already_in_retained_span_is_rejected() -> None:
    frame = np.eye(4, dtype=complex)[:, :2]
    candidate = 2.0 * frame[:, 0] - 0.5j * frame[:, 1]

    admitted, novelty = orthogonalize_adaptive_direction(
        candidate,
        frame,
        independence_floor=1.0e-12,
    )

    assert admitted is None
    assert novelty < 1.0e-12


def test_seed_directions_use_q0_projection_and_direction_cost() -> None:
    hamiltonian = np.diag([0.0, 1.0, 2.0, 3.0])
    prepared = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex)
    parallel_plus_new = np.asarray([3.0, 4.0, 0.0, 0.0], dtype=complex)

    result = run_adaptive_qse_benchmark(
        hamiltonian,
        prepared,
        target_roots=2,
        eps_residual=1.0e-12,
        max_dimension=2,
        seed_elements=(parallel_plus_new,),
        direction_resources={"n2q": 5.0, "d2q": 7.0, "dc": 11.0},
    )

    seed = result["seed_policy"]
    assert seed["declared_seed_set_size"] == 1
    assert seed["admitted_seed_direction_count"] == 1
    assert seed["seed_rows"][0]["q0_projected_norm"] == pytest.approx(4.0)
    assert result["resources"] == {"n2q": 5.0, "d2q": 7.0, "dc": 11.0}


def test_pauli_polynomial_uses_compiled_action_and_exact_diagonal() -> None:
    hamiltonian = PauliPolynomial("JW")
    for label, coefficient in (("ze", 0.7), ("ez", -0.2), ("xx", 0.4)):
        hamiltonian.add_term(PauliTerm(2, ps=label, pc=coefficient))
    prepared = np.asarray([1.0, 1.0j, -0.5, 0.25j], dtype=complex)

    result = run_adaptive_qse_benchmark(
        hamiltonian,
        prepared,
        target_roots=2,
        eps_residual=1.0e-12,
        max_dimension=4,
    )

    dense = np.asarray(
        [
            [0.5, 0.0, 0.0, 0.4],
            [0.0, 0.9, 0.4, 0.0],
            [0.0, 0.4, -0.9, 0.0],
            [0.4, 0.0, 0.0, -0.5],
        ],
        dtype=complex,
    )
    assert result["stop_reason"] == STOP_RESIDUAL_CONVERGED
    assert result["root_energies"] == pytest.approx(np.linalg.eigvalsh(dense)[:2], abs=1.0e-11)
