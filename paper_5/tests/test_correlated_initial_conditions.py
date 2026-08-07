from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp

from paper5.stability import (
    DimerParameters,
    closed_boson_moment_eigenvalues,
    closed_electron_eigenvalues,
    closed_phonon_eigenvalues,
    closed_residual_subtracted_rhs,
    closed_scalar_rhs,
    electron_density_eigenvalues,
    exact_ground_closed_scalar_coordinates,
    exact_ground_extended_scalar_coordinates,
    exact_ground_scalar_coordinates,
    extended_residual_subtracted_rhs,
    fan_migdal_with_anomalous_rhs,
    hartree_fock_closed_scalar_coordinates,
    hartree_fock_zero_correlation_state,
    phonon_density_eigenvalues,
    relative_boson_uncertainty_margin,
    residual_subtracted_rhs,
    source_connected_stationary_state,
)
from paper5.stability.exact_reference import exact_holstein_ground_state
from paper5.stability.matrix_reference import matrix_total_energy
from paper5.stability.matrix_reference import (
    boson_boundary_flux_decomposition,
    closed_eq14d_history_flux_decomposition,
    closed_scalar_to_matrix_state,
)


def test_exact_contractions_reproduce_exact_ground_energy() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=0.0,
    )
    exact = exact_holstein_ground_state(parameters, phonon_cutoff=12)

    assert abs(
        matrix_total_energy(exact.matrix_state, parameters) - exact.energy
    ) < 2e-13
    np.testing.assert_allclose(
        np.trace(exact.matrix_state.electron_density),
        1.0,
        atol=2e-13,
    )


def test_source_connected_scalar_root_is_stationary_and_basic_psd() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    stationary = source_connected_stationary_state(
        parameters,
        phonon_cutoff=12,
    )

    assert stationary.residual_norm < 1e-11
    assert stationary.electron_eigenvalues[0] > 0.0
    assert stationary.phonon_eigenvalues[0] > 0.0
    assert stationary.energy > stationary.exact_seed_energy


def test_residual_subtraction_keeps_strong_case_bounded_and_psd_to_t140() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = exact_ground_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    rhs = residual_subtracted_rhs(parameters, initial_state)
    sample_times = np.linspace(0.0, 140.0, 1401)
    solution = solve_ivp(
        rhs,
        (0.0, 140.0),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.1,
    )

    assert solution.success
    assert float(np.max(np.abs(solution.y))) < 3.0
    electron_minimum = min(
        electron_density_eigenvalues(solution.y[:, index])[0]
        for index in range(solution.y.shape[1])
    )
    phonon_minimum = min(
        phonon_density_eigenvalues(solution.y[:, index])[0]
        for index in range(solution.y.shape[1])
    )
    assert electron_minimum > 0.0
    assert phonon_minimum > 0.0


def test_eq14c_extension_delays_amplitude_failure_but_loses_physicality() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = np.concatenate(
        [hartree_fock_zero_correlation_state(), np.zeros(2)]
    )
    sample_times = np.linspace(0.0, 140.0, 1401)
    solution = solve_ivp(
        lambda time, state: fan_migdal_with_anomalous_rhs(
            time,
            state,
            parameters,
        ),
        (0.0, 140.0),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.1,
    )

    assert solution.success
    assert float(np.max(np.abs(solution.y))) < 3.0
    assert min(
        relative_boson_uncertainty_margin(solution.y[:, index])
        for index in range(solution.y.shape[1])
    ) < -1.0
    assert min(
        electron_density_eigenvalues(solution.y[:, index])[0]
        for index in range(solution.y.shape[1])
    ) < 0.0


def test_extended_eq112_is_bounded_but_not_boson_representable() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = exact_ground_extended_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    rhs = extended_residual_subtracted_rhs(parameters, initial_state)
    sample_times = np.linspace(0.0, 140.0, 1401)
    solution = solve_ivp(
        rhs,
        (0.0, 140.0),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.1,
    )

    assert solution.success
    assert float(np.max(np.abs(solution.y))) < 4.0
    assert min(
        electron_density_eigenvalues(solution.y[:, index])[0]
        for index in range(solution.y.shape[1])
    ) > 0.0
    assert min(
        phonon_density_eigenvalues(solution.y[:, index])[0]
        for index in range(solution.y.shape[1])
    ) > 0.0
    assert min(
        relative_boson_uncertainty_margin(solution.y[:, index])
        for index in range(solution.y.shape[1])
    ) < -1.0


def test_closed_hf_protocol_reproduces_full_matrix_failure_chain() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = hartree_fock_closed_scalar_coordinates(parameters)

    def threshold_event(_time: float, state: np.ndarray) -> float:
        return 1e4 - float(np.max(np.abs(state)))

    threshold_event.terminal = True
    threshold_event.direction = -1
    solution = solve_ivp(
        lambda time, state: closed_scalar_rhs(time, state, parameters),
        (0.0, 60.0),
        initial_state,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
        max_step=0.1,
        events=threshold_event,
        dense_output=True,
    )

    assert solution.success
    assert len(solution.t_events[0]) == 1
    assert 54.4 < solution.t_events[0][0] < 54.6
    assert solution.sol is not None
    state_at_two = solution.sol(2.0)
    assert closed_boson_moment_eigenvalues(state_at_two)[0] < 0.0


def test_closed_eq112_is_bounded_but_loses_boson_physicality() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    rhs = closed_residual_subtracted_rhs(parameters, initial_state)
    sample_times = np.linspace(0.0, 20.0, 401)
    solution = solve_ivp(
        rhs,
        (0.0, 20.0),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.1,
    )

    assert solution.success
    assert float(np.max(np.abs(solution.y))) < 3.0
    assert min(
        closed_electron_eigenvalues(solution.y[:, index])[0]
        for index in range(solution.y.shape[1])
    ) > 0.0
    assert min(
        closed_phonon_eigenvalues(solution.y[:, index])[0]
        for index in range(solution.y.shape[1])
    ) < 0.0
    assert min(
        closed_boson_moment_eigenvalues(solution.y[:, index])[0]
        for index in range(solution.y.shape[1])
    ) < 0.0


def test_eq112_boundary_flux_decomposition_reconstructs_outward_crossing() -> None:
    """Eq. (14b)--(14d) terms must explain the first bosonic PSD loss."""

    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )
    rhs = closed_residual_subtracted_rhs(parameters, initial_state)

    def boundary_event(_time: float, state: np.ndarray) -> float:
        return float(closed_boson_moment_eigenvalues(state)[0])

    boundary_event.terminal = True
    boundary_event.direction = -1
    solution = solve_ivp(
        rhs,
        (0.0, 2.0),
        initial_state,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.02,
        events=boundary_event,
    )

    assert solution.success
    assert len(solution.t_events[0]) == 1
    crossing_time = float(solution.t_events[0][0])
    crossing_state = np.asarray(solution.y_events[0][0], dtype=float)
    undriven = replace(parameters, drive_amplitude=0.0)
    initial_residual = closed_scalar_rhs(0.0, initial_state, undriven)
    decomposition = boson_boundary_flux_decomposition(
        crossing_time,
        closed_scalar_to_matrix_state(crossing_state),
        parameters,
        residual_subtraction=initial_residual,
    )

    assert 1.48 < crossing_time < 1.50
    assert abs(decomposition["minimum_eigenvalue"]) < 2e-10
    assert decomposition["total_flux"] < -1e-3
    assert decomposition["eq14b_correlation_source_flux"] < -3e-3
    assert 0.0 < decomposition["eq14c_correlation_source_flux"] < 3e-6
    assert (
        0.0
        < decomposition["eq112_residual_subtraction_flux"]
        < 2e-7
    )
    assert abs(decomposition["eq14d_direct_flux"]) < 1e-15
    assert decomposition["reconstruction_error"] < 2e-13
    assert decomposition["finite_difference_error"] < 1e-8


def test_eq14d_history_reconstructs_boundary_correlation_and_eq14b_flux() -> None:
    """Term-resolved Eq. (14d) histories must reconstruct the bad flux."""

    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )
    undriven = replace(parameters, drive_amplitude=0.0)
    initial_residual = closed_scalar_rhs(0.0, initial_state, undriven)
    decomposition = closed_eq14d_history_flux_decomposition(
        parameters,
        initial_state,
        residual_subtraction=initial_residual,
        maximum_time=2.0,
    )

    assert 1.48 < decomposition["crossing_time"] < 1.50
    assert decomposition["correlation_reconstruction_error"] < 2e-9
    assert decomposition["eq14b_flux_reconstruction_error"] < 2e-10
    assert decomposition["eq14c_flux_reconstruction_error"] < 2e-10
    assert decomposition["dominant_outward_history"] in (
        decomposition["eq14b_flux_by_history"]
    )
    assert all(
        np.isfinite(value)
        for value in decomposition["eq14b_flux_by_history"].values()
    )
    eq14b_histories = decomposition["eq14b_flux_by_history"]
    assert (
        decomposition["dominant_outward_history"]
        == "eq112_correlation_subtraction"
    )
    assert eq14b_histories["eq112_correlation_subtraction"] < -0.19
    assert eq14b_histories["eq14d_bare_pauli_source"] > 0.19
    assert abs(
        eq14b_histories["eq14d_anomalous_first_source"]
        + eq14b_histories["eq14d_anomalous_second_source"]
    ) < 1e-8
    assert abs(
        eq14b_histories["eq14d_normal_particle_source"]
        + eq14b_histories["eq14d_normal_hole_source"]
    ) < 2e-8

    without_correlation_subtraction = initial_residual.copy()
    without_correlation_subtraction[17:] = 0.0
    ablated = closed_eq14d_history_flux_decomposition(
        parameters,
        initial_state,
        residual_subtraction=without_correlation_subtraction,
        maximum_time=4.0,
    )
    assert 3.67 < ablated["crossing_time"] < 3.70
