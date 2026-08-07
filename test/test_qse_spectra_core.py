from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    QSEMatrices,
    QSEPruningConfig,
    build_qse_matrices,
    computational_basis_state,
    compute_qse_spectra,
    pauli_string_basis_element,
    pauli_string_observable,
    polynomial_basis_element,
    solve_qse_generalized_eigenproblem,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


def _poly(nq: int, terms: list[tuple[str, complex]]) -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for label, coeff in terms:
        out.add_term(PauliTerm(int(nq), ps=str(label), pc=complex(coeff)))
    return out


def _exact_eigs(poly: PauliPolynomial) -> np.ndarray:
    dense = hamiltonian_matrix(poly, tol=1e-15)
    return np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))


def test_identity_only_basis_returns_prepared_state_energy() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    psi = computational_basis_state(1, "0")
    result = compute_qse_spectra(
        hamiltonian,
        psi,
        [pauli_string_basis_element("e", nq=1)],
    )

    assert result.retained_rank == 1
    assert result.eigenvalues.tolist() == pytest.approx([1.0], abs=1e-12)
    assert result.matrices.reference_energy == pytest.approx(1.0, abs=1e-12)
    assert result.generalized_residual_norms[0] < 1e-12


def test_one_qubit_full_qse_basis_matches_dense_exact_spectrum() -> None:
    hamiltonian = _poly(1, [("z", 1.0), ("x", 0.3)])
    psi = computational_basis_state(1, "0")
    basis = [
        pauli_string_basis_element("e", nq=1),
        pauli_string_basis_element("x", nq=1),
    ]

    result = compute_qse_spectra(hamiltonian, psi, basis)

    assert result.retained_rank == 2
    assert result.discarded_rank == 0
    assert np.allclose(result.eigenvalues, _exact_eigs(hamiltonian), atol=1e-12)
    s_metric = result.eigenvectors_basis.conj().T @ result.matrices.overlap @ result.eigenvectors_basis
    assert np.allclose(s_metric, np.eye(2), atol=1e-12)


def test_two_qubit_pauli_basis_spans_full_hilbert_space() -> None:
    hamiltonian = _poly(
        2,
        [
            ("ze", 0.7),
            ("ez", -0.2),
            ("xx", 0.4),
            ("xe", 0.1),
        ],
    )
    psi = computational_basis_state(2, "00")
    basis = [pauli_string_basis_element(label, nq=2) for label in ["ee", "ex", "xe", "xx"]]

    result = compute_qse_spectra(hamiltonian, psi, basis)

    assert result.retained_rank == 4
    assert np.allclose(result.eigenvalues, _exact_eigs(hamiltonian), atol=1e-12)
    assert max(result.generalized_residual_norms) < 1e-12


def test_duplicate_basis_labels_are_pruned_by_overlap_rank() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    psi = computational_basis_state(1, "0")
    basis = [
        pauli_string_basis_element("e", nq=1, name="e0"),
        pauli_string_basis_element("e", nq=1, name="e_duplicate"),
        pauli_string_basis_element("x", nq=1, name="x"),
    ]

    result = compute_qse_spectra(hamiltonian, psi, basis)

    assert result.retained_rank == 2
    assert result.discarded_rank == 1
    assert np.allclose(result.eigenvalues, [-1.0, 1.0], atol=1e-12)
    assert result.overlap_condition_estimate == pytest.approx(2.0, abs=1e-12)


def test_polynomial_basis_element_is_supported() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    psi = computational_basis_state(1, "0")
    flip_poly = _poly(1, [("x", 1.0)])
    basis = [
        pauli_string_basis_element("e", nq=1),
        polynomial_basis_element(flip_poly, name="x_poly"),
    ]

    result = compute_qse_spectra(hamiltonian, psi, basis)

    assert result.retained_rank == 2
    assert np.allclose(result.eigenvalues, [-1.0, 1.0], atol=1e-12)


def test_q0_projection_removes_reference_parallel_records_but_keeps_flip() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    psi = computational_basis_state(1, "0")
    basis = [
        pauli_string_basis_element("I", nq=1, name="identity"),
        pauli_string_basis_element("Z", nq=1, name="z_parallel"),
        pauli_string_basis_element("X", nq=1, name="x_flip"),
    ]
    policy = QSEBasisVectorPolicy(reference_projection="q0", basis_vector_normalization="raw_projected")

    matrices = build_qse_matrices(hamiltonian, psi, basis, basis_vector_policy=policy)

    assert matrices.basis_action_norms == pytest.approx((1.0, 1.0, 1.0), abs=1e-12)
    assert matrices.basis_projected_norms == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)
    assert matrices.basis_matrix_vector_norms == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)
    assert [row.projected_out_by_q0 for row in matrices.basis_vector_diagnostics] == [True, True, False]
    assert [row.zero_vector for row in matrices.basis_vector_diagnostics] == [True, True, False]
    assert matrices.basis_vector_diagnostics[0].reference_overlap_after_projection_abs == pytest.approx(0.0, abs=1e-12)
    assert matrices.basis_vector_diagnostics[2].reference_overlap_before_projection_abs == pytest.approx(0.0, abs=1e-12)

    result = compute_qse_spectra(hamiltonian, psi, basis, basis_vector_policy=policy)
    assert result.retained_rank == 1
    assert result.discarded_rank == 2
    assert result.eigenvalues.tolist() == pytest.approx([-1.0], abs=1e-12)


def test_raw_projected_mode_preserves_scaled_vector_norm_diagnostics() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    psi = computational_basis_state(1, "0")
    scaled_flip = _poly(1, [("x", 2.0)])
    basis = [polynomial_basis_element(scaled_flip, name="two_x")]

    legacy = build_qse_matrices(hamiltonian, psi, basis)
    raw = build_qse_matrices(
        hamiltonian,
        psi,
        basis,
        basis_vector_policy=QSEBasisVectorPolicy(basis_vector_normalization="raw_projected"),
    )

    assert legacy.basis_action_norms == pytest.approx((2.0,), abs=1e-12)
    assert legacy.basis_projected_norms == pytest.approx((2.0,), abs=1e-12)
    assert legacy.basis_matrix_vector_norms == pytest.approx((1.0,), abs=1e-12)
    assert legacy.overlap[0, 0] == pytest.approx(1.0 + 0.0j, abs=1e-12)
    assert legacy.basis_vector_diagnostics[0].normalized_for_matrices is True

    assert raw.basis_action_norms == pytest.approx((2.0,), abs=1e-12)
    assert raw.basis_projected_norms == pytest.approx((2.0,), abs=1e-12)
    assert raw.basis_matrix_vector_norms == pytest.approx((2.0,), abs=1e-12)
    assert raw.overlap[0, 0] == pytest.approx(4.0 + 0.0j, abs=1e-12)
    assert raw.basis_vector_diagnostics[0].normalized_for_matrices is False


def test_transition_observable_one_qubit_q0_analytic_strength() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    psi = computational_basis_state(1, "0")
    policy = QSEBasisVectorPolicy(reference_projection="q0", basis_vector_normalization="raw_projected")

    result = compute_qse_spectra(
        hamiltonian,
        psi,
        [pauli_string_basis_element("X", nq=1, name="x_excitation")],
        basis_vector_policy=policy,
        transition_observables=[pauli_string_observable("X", nq=1, name="dipole")],
    )

    assert result.eigenvalues.tolist() == pytest.approx([-1.0], abs=1e-12)
    assert len(result.transition_observables) == 1
    transition = result.transition_observables[0]
    assert transition.observable.name == "dipole"
    assert transition.observable_matrix[0, 0] == pytest.approx(0.0 + 0.0j, abs=1e-12)
    assert transition.transition_vector[0] == pytest.approx(1.0 + 0.0j, abs=1e-12)
    assert transition.transition_amplitudes[0] == pytest.approx(1.0 + 0.0j, abs=1e-12)
    assert transition.transition_strengths[0] == pytest.approx(1.0, abs=1e-12)


def test_scaled_polynomial_basis_is_not_pruned_by_absolute_overlap_cutoff() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    psi = computational_basis_state(1, "0")
    tiny_flip = _poly(1, [("x", 1.0e-9)])
    basis = [
        pauli_string_basis_element("e", nq=1),
        polynomial_basis_element(tiny_flip, name="tiny_x"),
    ]

    result = compute_qse_spectra(hamiltonian, psi, basis)

    assert result.matrices.basis_vector_norms[1] == pytest.approx(1.0e-9, rel=1e-12)
    assert result.retained_rank == 2
    assert np.allclose(result.eigenvalues, [-1.0, 1.0], atol=1e-12)


def test_zero_polynomial_basis_element_is_pruned_not_fatal() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    psi = computational_basis_state(1, "0")
    zero_like = _poly(1, [("x", 1.0e-20)])
    basis = [
        pauli_string_basis_element("e", nq=1),
        polynomial_basis_element(zero_like, name="zero_like"),
    ]

    result = compute_qse_spectra(hamiltonian, psi, basis)

    assert result.matrices.basis_vector_norms[1] == pytest.approx(0.0, abs=1e-30)
    assert result.retained_rank == 1
    assert result.discarded_rank == 1
    assert result.eigenvalues.tolist() == pytest.approx([1.0], abs=1e-12)


def test_public_solver_rejects_nonhermitian_matrices() -> None:
    matrices = QSEMatrices(
        nq=1,
        hilbert_dim=2,
        basis_elements=(pauli_string_basis_element("e", nq=1),),
        reference_energy=0.0,
        reference_energy_imag_abs=0.0,
        basis_vector_norms=(1.0,),
        overlap=np.asarray([[1.0 + 0.0j]], dtype=complex),
        hamiltonian=np.asarray([[1.0 + 0.0j, 1.0 + 0.0j]], dtype=complex),
        overlap_hermitian_residual_max_abs_raw=0.0,
        hamiltonian_hermitian_residual_max_abs_raw=1.0,
        hamiltonian_coeff_imag_max_abs=0.0,
    )

    with pytest.raises(ValueError, match="shapes must match"):
        solve_qse_generalized_eigenproblem(matrices)

    matrices = QSEMatrices(
        nq=1,
        hilbert_dim=2,
        basis_elements=(pauli_string_basis_element("e", nq=1), pauli_string_basis_element("x", nq=1)),
        reference_energy=0.0,
        reference_energy_imag_abs=0.0,
        basis_vector_norms=(1.0, 1.0),
        overlap=np.eye(2, dtype=complex),
        hamiltonian=np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
        overlap_hermitian_residual_max_abs_raw=0.0,
        hamiltonian_hermitian_residual_max_abs_raw=1.0,
        hamiltonian_coeff_imag_max_abs=0.0,
    )

    with pytest.raises(ValueError, match="non-Hermitian"):
        solve_qse_generalized_eigenproblem(matrices)


def test_nonreal_hamiltonian_coefficient_is_rejected() -> None:
    hamiltonian = _poly(1, [("x", 1.0j)])
    psi = computational_basis_state(1, "0")

    with pytest.raises(ValueError, match="imaginary part"):
        compute_qse_spectra(
            hamiltonian,
            psi,
            [pauli_string_basis_element("e", nq=1)],
        )


def test_invalid_inputs_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="At least one overlap cutoff"):
        QSEPruningConfig(overlap_relative_cutoff=0.0, overlap_absolute_cutoff=0.0)

    with pytest.raises(ValueError, match="power of two"):
        compute_qse_spectra(
            _poly(1, [("z", 1.0)]),
            np.ones(3, dtype=complex),
            [pauli_string_basis_element("e", nq=1)],
        )

    with pytest.raises(ValueError, match="length"):
        pauli_string_basis_element("ee", nq=1)
