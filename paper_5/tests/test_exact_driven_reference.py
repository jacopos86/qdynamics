from __future__ import annotations

import json

import numpy as np

from paper5.stability import (
    CLOSED_PROTOCOLS,
    ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE,
    DimerParameters,
    closed_scalar_rhs,
    compare_exact_and_archive_closure,
    compare_exact_and_closed_protocols,
    correlation_source_by_phonon_mode,
    electron_phonon_moment_matrix,
    exact_holstein_correlation_closure_trajectory,
    exact_holstein_driven_trajectory,
    exact_holstein_fourth_cumulant_trajectory,
    exact_holstein_third_cumulant_trajectory,
    fourth_cumulant_rhs,
    FOURTH_CUMULANT_MOMENT_KEYS,
    third_cumulant_rhs,
    THIRD_CUMULANT_MOMENT_KEYS,
)
from paper5.stability.exact_compare import run_diagnostic
from paper5.stability.matrix_reference import (
    _correlation_homogeneous_rhs,
    matrix_dimer_rhs,
    matrix_state_to_closed_scalar_coordinates,
    same_spin_pauli_velocity_correction,
)


def _correlation_coordinates(correlation: np.ndarray) -> np.ndarray:
    shared_trace = 0.5 * (
        np.trace(correlation[0]) + np.trace(correlation[1])
    )
    values = [shared_trace.real, shared_trace.imag]
    for q in range(2):
        diagonal_difference = correlation[q, 0, 0] - correlation[q, 1, 1]
        values.extend(
            [
                diagonal_difference.real,
                diagonal_difference.imag,
                correlation[q, 0, 1].real,
                correlation[q, 0, 1].imag,
                correlation[q, 1, 0].real,
                correlation[q, 1, 0].imag,
            ]
        )
    return np.asarray(values, dtype=float)


def test_exact_driven_reference_preserves_decoupled_phonon_vacuum() -> None:
    parameters = DimerParameters(
        lambda_ep=0.0,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    sample_times = np.linspace(0.0, 2.0, 9)

    trajectory = exact_holstein_driven_trajectory(
        parameters,
        sample_times=sample_times,
        phonon_cutoff=3,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        maximum_step=0.02,
    )

    np.testing.assert_allclose(trajectory.times, sample_times)
    assert trajectory.success
    assert np.max(np.abs(trajectory.state_norms - 1.0)) < 2e-10

    for state in trajectory.matrix_states:
        np.testing.assert_allclose(
            np.trace(state.electron_density),
            1.0,
            atol=2e-10,
        )
        np.testing.assert_allclose(state.coherent_phonon, 0.0, atol=2e-10)
        np.testing.assert_allclose(state.phonon_density, 0.0, atol=2e-10)
        np.testing.assert_allclose(
            state.anomalous_phonon_density,
            0.0,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            state.electron_phonon_correlation,
            0.0,
            atol=2e-10,
        )


def test_archive_closure_matches_exact_decoupled_driven_dimer() -> None:
    parameters = DimerParameters(
        lambda_ep=0.0,
        gamma=0.5,
        drive_amplitude=1.0,
    )

    comparison = compare_exact_and_archive_closure(
        parameters,
        sample_times=np.linspace(0.0, 2.0, 41),
        phonon_cutoff=3,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        maximum_step=0.02,
    )

    assert comparison.success
    assert comparison.exact_coordinates.shape == (41, 31)
    assert comparison.archive_coordinates.shape == (41, 31)
    assert np.max(np.abs(comparison.coordinate_errors)) < 3e-9


def test_all_closed_protocols_match_exact_decoupled_stationary_dimer() -> None:
    parameters = DimerParameters(
        lambda_ep=0.0,
        gamma=0.5,
        drive_amplitude=0.0,
    )

    comparisons = compare_exact_and_closed_protocols(
        parameters,
        sample_times=np.linspace(0.0, 0.2, 5),
        phonon_cutoff=2,
        protocols=CLOSED_PROTOCOLS,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        maximum_step=0.05,
    )

    assert tuple(comparisons) == CLOSED_PROTOCOLS
    exact_trajectory = comparisons["archive"].exact_trajectory
    for protocol, comparison in comparisons.items():
        assert comparison.protocol == protocol
        assert comparison.success
        assert comparison.exact_trajectory is exact_trajectory
        assert comparison.closed_coordinates.shape == (5, 31)
        assert np.max(np.abs(comparison.coordinate_errors)) < 3e-9


def test_diagnostic_writes_retrievable_cutoff_artifacts(tmp_path) -> None:
    plan = {
        "schema_version": 1,
        "run_id": "test_exact_closure_diagnostic",
        "classification": "diagnostic",
        "scientific_question": "test",
        "evidence_status": "exploratory_local_not_promoted",
        "execution_authorized": False,
        "parameters": {
            "hopping": 1.0,
            "gamma": 0.5,
            "lambda_ep": 0.0,
            "drive_amplitude": 0.0,
            "pulse_width": 1.0,
        },
        "cutoff_execution_order": [2],
        "protocols": ["archive"],
        "correction": {
            "activation_margin": 1e-5,
            "target_flux": 0.0,
            "barrier_rate": 5.0,
            "energy_neutral": True,
            "require_convergence": True,
        },
        "integration": {
            "initial_time": 0.0,
            "final_time": 0.1,
            "sample_step": 0.05,
            "maximum_step": 0.05,
            "relative_tolerance": 1e-10,
            "absolute_tolerance": 1e-12,
            "eigensolver_tolerance": 1e-12,
        },
        "source_hashes": {},
    }
    authorization = {
        "run_id": plan["run_id"],
        "authorized": True,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "authorization.json").write_text(
        json.dumps(authorization),
        encoding="utf-8",
    )

    summary = run_diagnostic(tmp_path)

    assert summary["status"] == "complete"
    assert summary["cutoffs"]["2"]["exact"][
        "hilbert_space_dimension"
    ] == 36
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "trajectories_cutoff_2.npz").is_file()
    manifest = json.loads(
        (tmp_path / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert "trajectories_cutoff_2.npz" in manifest["artifact_hashes"]


def test_strong_coupling_closure_defect_starts_in_correlation_rhs() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    undriven = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=0.0,
    )
    target_time = 0.5
    trajectory = exact_holstein_driven_trajectory(
        parameters,
        sample_times=np.array([0.0, target_time]),
        phonon_cutoff=16,
        relative_tolerance=1e-11,
        absolute_tolerance=1e-13,
        maximum_step=0.01,
    )
    assert min(
        float(np.linalg.eigvalsh(electron_phonon_moment_matrix(state))[0])
        for state in trajectory.matrix_states
    ) > 0.0
    coordinates = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in trajectory.matrix_states
        ]
    )
    exact_derivative = matrix_state_to_closed_scalar_coordinates(
        trajectory.matrix_derivatives[1]
    )
    initial_residual = closed_scalar_rhs(
        0.0,
        coordinates[0],
        undriven,
    )
    closure_derivative = (
        closed_scalar_rhs(target_time, coordinates[1], parameters)
        - initial_residual
    )

    derivative_error = closure_derivative - exact_derivative
    noncorrelation_error = np.linalg.norm(derivative_error[:17])
    correlation_error = np.linalg.norm(derivative_error[17:])

    assert noncorrelation_error < 1e-5
    assert correlation_error > 4e-2


def test_correlation_source_mode_partition_reconstructs_eq14d() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    trajectory = exact_holstein_driven_trajectory(
        parameters,
        sample_times=np.array([0.0, 0.1]),
        phonon_cutoff=6,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        maximum_step=0.02,
    )
    state = trajectory.matrix_states[-1]
    full_derivative = matrix_dimer_rhs(0.1, state, parameters)
    transport = _correlation_homogeneous_rhs(
        0.1,
        state,
        state.electron_phonon_correlation,
        parameters,
    )
    source_by_mode = correlation_source_by_phonon_mode(state, parameters)

    assert source_by_mode.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(
        np.sum(source_by_mode, axis=1),
        full_derivative.electron_phonon_correlation - transport,
        atol=2e-14,
    )


def test_exact_missing_moments_collapse_residual_subtracted_c_defect() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    audit = exact_holstein_correlation_closure_trajectory(
        parameters,
        sample_times=np.array([0.0, 0.5]),
        phonon_cutoff=16,
        relative_tolerance=1e-11,
        absolute_tolerance=1e-13,
        maximum_step=0.01,
    )

    for index, state in enumerate(audit.exact_trajectory.matrix_states):
        np.testing.assert_allclose(
            same_spin_pauli_velocity_correction(state, parameters),
            audit.same_spin_pauli_velocity_corrections[index],
            atol=2e-14,
            rtol=2e-14,
        )

    assert audit.exact_mixed_moment.shape == (2, 2, 2, 2, 2)
    assert audit.opposite_spin_covariance.shape == (2, 2, 2, 2)
    exact_derivatives = np.asarray(
        [
            derivative.electron_phonon_correlation
            for derivative in audit.exact_trajectory.matrix_derivatives
        ]
    )
    archive_errors = audit.archive_correlation_derivatives - exact_derivatives
    corrected_errors = (
        archive_errors
        + audit.mixed_moment_velocity_corrections
        + audit.same_spin_pauli_velocity_corrections
        + audit.opposite_spin_velocity_corrections
    )

    old_defect = _correlation_coordinates(
        archive_errors[1] - archive_errors[0]
    )
    corrected_defect = _correlation_coordinates(
        corrected_errors[1] - corrected_errors[0]
    )
    mixed_change = _correlation_coordinates(
        audit.mixed_moment_velocity_corrections[1]
        - audit.mixed_moment_velocity_corrections[0]
    )
    opposite_spin_change = _correlation_coordinates(
        audit.opposite_spin_velocity_corrections[1]
        - audit.opposite_spin_velocity_corrections[0]
    )
    pauli_change = _correlation_coordinates(
        audit.same_spin_pauli_velocity_corrections[1]
        - audit.same_spin_pauli_velocity_corrections[0]
    )

    assert 0.0484 < np.linalg.norm(old_defect) < 0.0487
    assert 0.0480 < np.linalg.norm(mixed_change) < 0.0483
    assert 0.0077 < np.linalg.norm(pauli_change) < 0.0080
    assert 0.0056 < np.linalg.norm(opposite_spin_change) < 0.0059
    assert np.linalg.norm(corrected_defect) < 2e-5
    assert np.linalg.norm(corrected_defect) < 5e-4 * np.linalg.norm(old_defect)

    reconstructed = (
        audit.archive_correlation_derivatives
        + audit.mixed_moment_velocity_corrections
        + audit.same_spin_pauli_velocity_corrections
        + audit.opposite_spin_velocity_corrections
        + audit.cutoff_velocity_remainders
    )
    np.testing.assert_allclose(reconstructed, exact_derivatives, atol=2e-13)

    exact_diagonal = np.diagonal(
        audit.exact_mixed_moment,
        axis1=-2,
        axis2=-1,
    )
    factorized_diagonal = np.diagonal(
        audit.factorized_mixed_moment,
        axis1=-2,
        axis2=-1,
    )
    np.testing.assert_allclose(exact_diagonal, 0.0, atol=2e-14)
    np.testing.assert_allclose(factorized_diagonal, 0.0, atol=2e-14)
    np.testing.assert_allclose(
        audit.opposite_spin_covariance,
        audit.opposite_spin_covariance.conjugate().swapaxes(-1, -2),
        atol=2e-14,
    )
    np.testing.assert_allclose(
        audit.opposite_spin_covariance[:, 0]
        + audit.opposite_spin_covariance[:, 1],
        0.0,
        atol=2e-14,
    )


def test_exact_adapter_separates_algebra_from_terminal_cumulant_error() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    trajectory = exact_holstein_third_cumulant_trajectory(
        parameters,
        sample_times=np.array([0.0, 0.5]),
        phonon_cutoff=20,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        maximum_step=0.01,
    )

    assert trajectory.coordinates.shape == (2, 47)
    assert trajectory.coordinate_derivatives.shape == (2, 47)
    assert np.max(np.abs(trajectory.state_norms - 1.0)) < 2e-10

    approximate = third_cumulant_rhs(
        0.5,
        trajectory.coordinates[1],
        parameters,
    )
    defect = approximate - trajectory.coordinate_derivatives[1]
    degree_one = np.array(
        [
            index + 2
            for index, key in enumerate(THIRD_CUMULANT_MOMENT_KEYS)
            if key.degree == 1
        ]
    )
    degree_two = np.array(
        [
            index + 2
            for index, key in enumerate(THIRD_CUMULANT_MOMENT_KEYS)
            if key.degree == 2
        ]
    )
    degree_three = np.array(
        [
            index + 2
            for index, key in enumerate(THIRD_CUMULANT_MOMENT_KEYS)
            if key.degree == 3
        ]
    )

    # Degrees one and two need no closure and agree down to the finite-cutoff
    # commutator floor.  Degree three is the first block that invokes the
    # zero-connected-fourth-cumulant approximation.
    assert np.max(np.abs(defect[:2])) < 2e-6
    assert np.max(np.abs(defect[degree_one])) < 2e-8
    assert np.max(np.abs(defect[degree_two])) < 2e-5
    assert np.linalg.norm(defect[degree_three]) > 0.2


def test_third_cumulant_closure_is_exact_in_decoupled_control() -> None:
    parameters = DimerParameters(
        lambda_ep=0.0,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    trajectory = exact_holstein_third_cumulant_trajectory(
        parameters,
        sample_times=np.linspace(0.0, 2.0, 9),
        phonon_cutoff=3,
        relative_tolerance=1e-11,
        absolute_tolerance=1e-13,
        maximum_step=0.01,
    )
    approximate_derivatives = np.asarray(
        [
            third_cumulant_rhs(float(time), coordinates, parameters)
            for time, coordinates in zip(
                trajectory.times,
                trajectory.coordinates,
                strict=True,
            )
        ]
    )
    np.testing.assert_allclose(
        approximate_derivatives,
        trajectory.coordinate_derivatives,
        atol=8e-13,
    )


def test_fourth_hierarchy_repairs_degree_three_and_exposes_terminal_error() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    trajectory = exact_holstein_fourth_cumulant_trajectory(
        parameters,
        sample_times=np.array([0.0, 0.5]),
        phonon_cutoff=20,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        maximum_step=0.01,
    )
    approximate = fourth_cumulant_rhs(
        0.5,
        trajectory.coordinates[1],
        parameters,
    )
    defect = approximate - trajectory.coordinate_derivatives[1]
    indices = {
        degree: np.asarray(
            [
                index + 2
                for index, key in enumerate(FOURTH_CUMULANT_MOMENT_KEYS)
                if key.degree == degree
            ]
        )
        for degree in (1, 2, 3, 4)
    }

    assert trajectory.coordinates.shape == (2, 82)
    assert trajectory.maximum_degree == 4
    assert np.max(np.abs(defect[indices[1]])) < 2e-8
    assert np.max(np.abs(defect[indices[2]])) < 2e-5
    assert np.max(np.abs(defect[indices[3]])) < 3e-6
    assert np.linalg.norm(defect[indices[4]]) > 3.0

    adapted = fourth_cumulant_rhs(
        0.5,
        trajectory.coordinates[1],
        parameters,
        closure=ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE,
    )
    adapted_defect = adapted - trajectory.coordinate_derivatives[1]
    assert np.linalg.norm(adapted_defect[indices[4]]) < 2.0
    assert np.linalg.norm(adapted_defect[indices[4]]) > 1.9


def test_fourth_hierarchy_is_exact_in_decoupled_control() -> None:
    parameters = DimerParameters(
        lambda_ep=0.0,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    trajectory = exact_holstein_fourth_cumulant_trajectory(
        parameters,
        sample_times=np.linspace(0.0, 2.0, 9),
        phonon_cutoff=3,
        relative_tolerance=1e-11,
        absolute_tolerance=1e-13,
        maximum_step=0.01,
    )
    approximate_derivatives = np.asarray(
        [
            fourth_cumulant_rhs(float(time), coordinates, parameters)
            for time, coordinates in zip(
                trajectory.times,
                trajectory.coordinates,
                strict=True,
            )
        ]
    )
    np.testing.assert_allclose(
        approximate_derivatives,
        trajectory.coordinate_derivatives,
        atol=1e-12,
    )
    adapted_derivatives = np.asarray(
        [
            fourth_cumulant_rhs(
                float(time),
                coordinates,
                parameters,
                closure=ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE,
            )
            for time, coordinates in zip(
                trajectory.times,
                trajectory.coordinates,
                strict=True,
            )
        ]
    )
    np.testing.assert_allclose(
        adapted_derivatives,
        trajectory.coordinate_derivatives,
        atol=1e-12,
    )
