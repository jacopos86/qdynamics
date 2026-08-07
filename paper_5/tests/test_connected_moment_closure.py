from __future__ import annotations

import numpy as np
import pytest

from paper5.stability import (
    DimerParameters,
    conditional_k_closed_scalar_rhs,
    conditional_k_matrix_dimer_rhs,
    conditional_k_pauli_repaired_closed_scalar_rhs,
    conditional_pauli_regression_mixed_moment,
    conditional_pauli_regression_velocity_correction,
)
from paper5.stability.matrix_reference import (
    MatrixDimerState,
    closed_scalar_rhs,
    matrix_derivative_to_closed_scalar,
    matrix_state_to_closed_scalar_coordinates,
    pauli_repaired_closed_scalar_rhs,
    electron_phonon_moment_matrix,
)
from paper5.stability.connected_moment_closure_analysis import (
    run_analysis as run_closure_gate_analysis,
)
from paper5.stability.connected_moment_propagation_analysis import (
    run_analysis as run_propagation_analysis,
)


def _physical_state(*, correlation_scale: float = 1.0) -> MatrixDimerState:
    correlation_0 = correlation_scale * np.array(
        [
            [0.0, 0.01 + 0.02j],
            [-0.015 + 0.005j, 0.0],
        ],
        dtype=complex,
    )
    return MatrixDimerState(
        electron_density=np.array(
            [
                [0.62, 0.12 + 0.03j],
                [0.12 - 0.03j, 0.38],
            ],
            dtype=complex,
        ),
        coherent_phonon=np.zeros(2, dtype=complex),
        phonon_density=0.3 * np.eye(2, dtype=complex),
        anomalous_phonon_density=np.zeros((2, 2), dtype=complex),
        electron_phonon_correlation=np.stack(
            [correlation_0, -correlation_0]
        ),
    )


def test_regression_solves_state_weighted_normal_equations() -> None:
    state = _physical_state()
    result = conditional_pauli_regression_mixed_moment(state)
    joint = electron_phonon_moment_matrix(state)
    electronic_gram = joint[4:, 4:]
    cross_gram = joint[:4, 4:]

    assert result.electronic_support_rank == 3
    assert result.electronic_gram_minimum_eigenvalue > -1e-14
    assert result.maximum_normal_equation_relative_residual < 2e-14
    for direction in range(4):
        np.testing.assert_allclose(
            electronic_gram
            @ result.phonon_to_pauli_coefficients[direction],
            np.conjugate(cross_gram[direction]),
            atol=2e-14,
            rtol=2e-14,
        )


def test_regressed_mixed_moment_preserves_commutator_trace_structure() -> None:
    result = conditional_pauli_regression_mixed_moment(_physical_state())
    mixed = result.mixed_moment

    assert mixed.shape == (2, 2, 2, 2)
    assert np.all(np.isfinite(mixed))
    np.testing.assert_allclose(
        np.diagonal(mixed, axis1=-2, axis2=-1),
        0.0,
        atol=2e-16,
    )
    np.testing.assert_allclose(
        np.trace(mixed, axis1=-2, axis2=-1),
        0.0,
        atol=2e-16,
    )
    assert np.linalg.norm(mixed) > 1e-8


def test_regressed_mixed_moment_vanishes_without_retained_cross_correlation() -> None:
    result = conditional_pauli_regression_mixed_moment(
        _physical_state(correlation_scale=0.0)
    )

    np.testing.assert_array_equal(result.mixed_moment, 0.0)
    np.testing.assert_array_equal(result.phonon_to_pauli_coefficients, 0.0)


def test_velocity_correction_vanishes_in_decoupled_control() -> None:
    correction = conditional_pauli_regression_velocity_correction(
        _physical_state(),
        DimerParameters(lambda_ep=0.0, gamma=0.5),
    )

    np.testing.assert_array_equal(correction, 0.0)


def test_conditional_k_rhs_changes_only_correlation_velocity() -> None:
    state = _physical_state()
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    coordinates = matrix_state_to_closed_scalar_coordinates(state)
    raw = closed_scalar_rhs(0.37, coordinates, parameters)
    conditional = conditional_k_closed_scalar_rhs(
        0.37,
        coordinates,
        parameters,
    )

    np.testing.assert_allclose(conditional[:17], raw[:17], atol=2e-14)
    assert np.linalg.norm(conditional[17:] - raw[17:]) > 1e-8


def test_conditional_k_scalar_and_matrix_rhs_agree() -> None:
    state = _physical_state()
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    coordinates = matrix_state_to_closed_scalar_coordinates(state)

    scalar = conditional_k_closed_scalar_rhs(0.37, coordinates, parameters)
    matrix = matrix_derivative_to_closed_scalar(
        conditional_k_matrix_dimer_rhs(0.37, state, parameters)
    )

    np.testing.assert_allclose(scalar, matrix, atol=2e-14, rtol=2e-14)


def test_combined_rhs_adds_pauli_repair_to_conditional_k_rhs() -> None:
    state = _physical_state()
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    coordinates = matrix_state_to_closed_scalar_coordinates(state)
    raw = closed_scalar_rhs(0.37, coordinates, parameters)
    pauli = pauli_repaired_closed_scalar_rhs(0.37, coordinates, parameters)
    conditional = conditional_k_closed_scalar_rhs(
        0.37,
        coordinates,
        parameters,
    )
    combined = conditional_k_pauli_repaired_closed_scalar_rhs(
        0.37,
        coordinates,
        parameters,
    )

    np.testing.assert_allclose(
        combined,
        conditional + pauli - raw,
        atol=2e-14,
        rtol=2e-14,
    )


def test_analysis_driver_emits_complete_offline_gate(tmp_path) -> None:
    run_directory = tmp_path / "conditional_k_gate"

    summary = run_closure_gate_analysis(
        run_directory,
        parameters=DimerParameters(lambda_ep=0.5, gamma=0.5),
        final_time=0.02,
        sample_step=0.02,
        phonon_cutoffs=(1, 2),
        decision_cutoff=2,
        maximum_step=0.02,
    )

    assert set(summary["cutoff_metrics"]) == {"1", "2"}
    assert isinstance(summary["gate"]["short_propagation_authorized"], bool)
    assert (run_directory / "conditional_k_gate.npz").is_file()
    assert (run_directory / "conditional_k_gate.png").is_file()
    assert (run_directory / "summary.json").is_file()
    assert (run_directory / "runtime_manifest.json").is_file()


def test_analysis_driver_requires_two_cutoffs(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least two distinct"):
        run_closure_gate_analysis(
            tmp_path / "conditional_k_gate",
            parameters=DimerParameters(lambda_ep=0.5, gamma=0.5),
            phonon_cutoffs=(2,),
            decision_cutoff=2,
        )


def test_short_propagation_driver_emits_complete_comparison(tmp_path) -> None:
    run_directory = tmp_path / "conditional_k_propagation"

    summary = run_propagation_analysis(
        run_directory,
        parameters=DimerParameters(lambda_ep=0.5, gamma=0.5),
        final_time=0.02,
        time_step=0.01,
        phonon_cutoff=1,
        exact_maximum_step=0.01,
    )

    assert set(summary["lanes"]) == {
        "controller",
        "pauli_controller",
        "conditional_k_controller",
        "conditional_k_pauli_controller",
    }
    assert isinstance(summary["gate"]["step_refinement_authorized"], bool)
    assert (run_directory / "trajectories.npz").is_file()
    assert (run_directory / "propagation_gate.png").is_file()
    assert (run_directory / "summary.json").is_file()
    assert (run_directory / "runtime_manifest.json").is_file()
