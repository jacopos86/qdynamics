from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from paper5.stability import (
    DimerParameters,
    closed_boson_moment_eigenvalues,
    closed_cone_projected_rhs,
    closed_electron_phonon_cone_projected_rhs,
    closed_full_state_joint_cone_projected_rhs,
    closed_scalar_rhs,
    closed_residual_subtracted_rhs,
    closed_joint_cone_projected_rhs,
    closed_electron_eigenvalues,
    closed_state_lifted_frobenius_metric,
    closed_state_lifted_frobenius_norm,
    closed_state_correction_energy_gradient,
    exact_ground_closed_scalar_coordinates,
    electron_phonon_moment_matrix,
    joint_correction_energy_gradient,
    structured_electron_velocity_lift,
    structured_electron_phonon_barrier_correction,
    structured_electron_phonon_moment_velocity_lift,
    structured_full_state_joint_barrier_correction,
    structured_boson_barrier_correction,
    structured_boson_cone_correction,
    structured_boson_velocity_lift,
    structured_closed_state_velocity_lift,
    structured_joint_barrier_correction,
)
from paper5.stability.matrix_reference import (
    closed_scalar_to_matrix_state,
    matrix_total_energy,
)
from paper5.stability.cone_correction import _project_origin_onto_halfspaces


def test_halfspace_projection_returns_minimum_norm_intersection() -> None:
    point, converged = _project_origin_onto_halfspaces(
        [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        ],
        [1.0, 1.0, 3.0],
        tolerance=1e-12,
        maximum_cycles=10_000,
    )

    assert converged
    np.testing.assert_allclose(point, [1.5, 1.5], atol=1e-10)


def test_structured_velocity_lift_preserves_boson_block_structure() -> None:
    coordinates = np.arange(1.0, 11.0)
    lifted = structured_boson_velocity_lift(coordinates)

    np.testing.assert_allclose(lifted, lifted.conjugate().T)
    np.testing.assert_allclose(lifted[:2, :2], lifted[2:, 2:].T)
    np.testing.assert_allclose(lifted[:2, 2:], lifted[2:, :2].conjugate())
    np.testing.assert_allclose(lifted[2:, :2], lifted[2:, :2].T)


def test_electron_velocity_lift_is_hermitian_and_traceless() -> None:
    lifted = structured_electron_velocity_lift(
        np.array([0.7, -0.2, 0.3])
    )

    np.testing.assert_allclose(lifted, lifted.conjugate().T)
    np.testing.assert_allclose(np.trace(lifted), 0.0, atol=1e-15)


def test_lifted_frobenius_metric_matches_explicit_matrix_blocks() -> None:
    direction = np.random.default_rng(161803).normal(size=31)
    lifted = structured_closed_state_velocity_lift(direction)
    explicit_squared = sum(
        float(np.vdot(value, value).real)
        for value in (
            lifted.electron_density,
            lifted.coherent_phonon,
            lifted.phonon_density,
            lifted.anomalous_phonon_density,
            lifted.electron_phonon_correlation,
        )
    )
    metric = closed_state_lifted_frobenius_metric()

    np.testing.assert_allclose(metric, metric.T, atol=1e-15)
    assert np.linalg.eigvalsh(metric)[0] > 0.0
    np.testing.assert_allclose(
        direction @ metric @ direction,
        explicit_squared,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        closed_state_lifted_frobenius_norm(direction) ** 2,
        explicit_squared,
        atol=1e-13,
    )


def test_joint_energy_gradient_matches_total_energy_directional_derivative() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    direction = np.array(
        [
            0.2,
            -0.3,
            0.4,
            0.5,
            -0.1,
            0.7,
            -0.2,
            0.6,
            -0.8,
            0.9,
            -0.4,
            0.3,
            -0.5,
        ]
    )
    closed_direction = np.zeros_like(state)
    closed_direction[:3] = direction[:3]
    closed_direction[7:17] = direction[3:]
    step = 1e-7
    plus = matrix_total_energy(
        closed_scalar_to_matrix_state(state + step * closed_direction),
        parameters,
    )
    minus = matrix_total_energy(
        closed_scalar_to_matrix_state(state - step * closed_direction),
        parameters,
    )
    finite_difference = (plus - minus) / (2.0 * step)
    analytic = joint_correction_energy_gradient(state, parameters) @ direction

    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-8, atol=1e-8)


def test_full_state_energy_gradient_matches_directional_derivative() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    direction = np.random.default_rng(314159).normal(size=state.size)
    step = 1e-7
    plus = matrix_total_energy(
        closed_scalar_to_matrix_state(state + step * direction),
        parameters,
    )
    minus = matrix_total_energy(
        closed_scalar_to_matrix_state(state - step * direction),
        parameters,
    )
    finite_difference = (plus - minus) / (2.0 * step)
    analytic = (
        closed_state_correction_energy_gradient(state, parameters)
        @ direction
    )

    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-8, atol=1e-8)


def test_joint_moment_velocity_lift_matches_directional_derivative() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    direction = np.random.default_rng(271828).normal(size=state.size)
    step = 1e-7
    plus = electron_phonon_moment_matrix(
        closed_scalar_to_matrix_state(state + step * direction)
    )
    minus = electron_phonon_moment_matrix(
        closed_scalar_to_matrix_state(state - step * direction)
    )
    finite_difference = (plus - minus) / (2.0 * step)
    analytic = structured_electron_phonon_moment_velocity_lift(
        state,
        direction,
    )

    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-8, atol=1e-8)


def test_structured_cone_correction_is_minimum_norm_for_active_mode() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    derivative = np.zeros_like(state)
    derivative[7] = -1.0
    derivative[8] = -0.5
    derivative[11] = 0.75
    derivative[16] = -0.25

    result = structured_boson_cone_correction(
        state,
        derivative,
        activation_margin=1.0,
        target_flux=0.2,
    )

    assert result.active
    assert result.raw_flux < result.target_flux
    np.testing.assert_allclose(result.corrected_flux, result.target_flux)
    np.testing.assert_allclose(
        result.response_vector @ result.correction_coordinates,
        result.target_flux - result.raw_flux,
    )

    response = result.response_vector
    correction = result.correction_coordinates
    rng = np.random.default_rng(271828)
    for _ in range(20):
        trial_offset = rng.normal(size=response.size)
        trial_offset -= response * (
            response @ trial_offset / (response @ response)
        )
        feasible_trial = correction + trial_offset
        assert np.linalg.norm(feasible_trial) >= np.linalg.norm(correction)


def test_cone_projected_rhs_changes_only_normal_and_anomalous_velocities() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    base_rhs = closed_residual_subtracted_rhs(parameters, initial_state)
    corrected_rhs = closed_cone_projected_rhs(
        parameters,
        initial_state,
        activation_margin=1.0,
        target_flux=1.0,
    )
    base = base_rhs(0.3, initial_state)
    corrected = corrected_rhs(0.3, initial_state)

    np.testing.assert_allclose(corrected[:7], base[:7])
    np.testing.assert_allclose(corrected[17:], base[17:])
    assert np.linalg.norm(corrected[7:17] - base[7:17]) > 0.0


def test_barrier_correction_uses_continuous_flux_floor() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    derivative = np.zeros_like(state)
    derivative[7] = -10.0
    result = structured_boson_cone_correction(
        state,
        derivative,
        activation_margin=1e-5,
        target_flux=0.0,
        barrier_rate=5.0,
    )

    expected_floor = 5.0 * (
        1e-5 - result.minimum_eigenvalue
    )
    assert result.active
    np.testing.assert_allclose(result.target_flux, expected_floor)
    np.testing.assert_allclose(result.corrected_flux, expected_floor)


def test_full_matrix_barrier_is_psd_for_competing_modes() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    derivative = np.zeros_like(state)
    derivative[7:17] = np.array(
        [-1.0, -0.8, 0.4, -0.3, 0.7, -0.2, -0.6, 0.5, 0.3, -0.4]
    )
    result = structured_boson_barrier_correction(
        state,
        derivative,
        activation_margin=1e-5,
        target_flux=0.0,
        barrier_rate=5.0,
    )

    assert result.raw_barrier_minimum_eigenvalue < 0.0
    assert result.corrected_barrier_minimum_eigenvalue > -2e-11
    assert result.constraint_count >= 1
    assert result.correction_norm > 0.0


def test_energy_neutral_barrier_has_zero_normal_trace() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    derivative = np.zeros_like(state)
    derivative[7:17] = np.array(
        [-1.0, 0.4, 0.2, -0.1, 0.7, -0.2, -0.6, 0.5, 0.3, -0.4]
    )
    result = structured_boson_barrier_correction(
        state,
        derivative,
        activation_margin=1e-5,
        target_flux=0.0,
        barrier_rate=5.0,
        energy_neutral=True,
    )

    np.testing.assert_allclose(
        result.correction_coordinates[0]
        + result.correction_coordinates[1],
        0.0,
        atol=1e-12,
    )
    assert result.corrected_barrier_minimum_eigenvalue > -2e-11


def test_joint_barrier_enforces_both_electron_bounds_and_boson_cone() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    derivative = np.zeros_like(state)
    derivative[:3] = np.array([-3.0, 4.0, -2.0])
    derivative[7:17] = np.array(
        [-1.0, -0.8, 0.4, -0.3, 0.7, -0.2, -0.6, 0.5, 0.3, -0.4]
    )

    result = structured_joint_barrier_correction(
        state,
        derivative,
        parameters,
        activation_margin=1e-5,
        target_flux=0.0,
        barrier_rate=5.0,
        energy_neutral=False,
    )

    assert result.raw_electron_lower_barrier_minimum_eigenvalue < 0.0
    assert result.raw_electron_upper_barrier_minimum_eigenvalue < 0.0
    assert result.raw_boson_barrier_minimum_eigenvalue < 0.0
    assert result.corrected_electron_lower_barrier_minimum_eigenvalue > -2e-10
    assert result.corrected_electron_upper_barrier_minimum_eigenvalue > -2e-10
    assert result.corrected_boson_barrier_minimum_eigenvalue > -2e-10
    assert result.constraint_count >= 3
    assert result.correction_norm > 0.0
    np.testing.assert_allclose(
        np.trace(structured_electron_velocity_lift(result.correction_coordinates[:3])),
        0.0,
        atol=1e-14,
    )


def test_joint_energy_neutral_barrier_has_zero_total_energy_flux() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    derivative = np.zeros_like(state)
    derivative[:3] = 0.1 * np.array([-3.0, 4.0, -2.0])
    derivative[7:17] = 0.1 * np.array(
        [-1.0, -0.8, 0.4, -0.3, 0.7, -0.2, -0.6, 0.5, 0.3, -0.4]
    )

    result = structured_joint_barrier_correction(
        state,
        derivative,
        parameters,
        activation_margin=1e-5,
        target_flux=0.0,
        barrier_rate=5.0,
        energy_neutral=True,
    )

    assert result.converged
    assert result.corrected_electron_lower_barrier_minimum_eigenvalue > -2e-10
    assert result.corrected_electron_upper_barrier_minimum_eigenvalue > -2e-10
    assert result.corrected_boson_barrier_minimum_eigenvalue > -2e-10
    np.testing.assert_allclose(result.correction_energy_flux, 0.0, atol=2e-11)
    np.testing.assert_allclose(
        joint_correction_energy_gradient(state, parameters)
        @ result.correction_coordinates,
        0.0,
        atol=2e-11,
    )


def test_joint_barrier_resolves_many_mode_boundary_state() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = np.array(
        [
            0.7662599441693244,
            0.3185411422473864,
            -0.01115079189201803,
            -2.1897012390333686,
            -0.02713696803672531,
            -0.2598300166699305,
            0.0268320966681872,
            0.05157980514704288,
            0.05387980537542401,
            0.02531057209827398,
            0.00570431464057819,
            -0.12319079966143419,
            -0.07357498667227906,
            -0.1442595496867259,
            -0.0811783137101723,
            -0.12226194551989604,
            0.0533402299477054,
            0.03189753541252562,
            -0.04477272034802068,
            -0.01545782611023151,
            0.12373661961246038,
            0.02264060891823499,
            0.07741487256292515,
            0.01349794769920067,
            -0.10126786256467246,
            0.00532681760879519,
            -0.04326461452515489,
            0.04372270503908616,
            0.00452225409655276,
            -0.08020506745800526,
            -0.0733309237050589,
        ]
    )
    derivative = closed_scalar_rhs(28.545, state, parameters)

    result = structured_joint_barrier_correction(
        state,
        derivative,
        parameters,
        activation_margin=1e-5,
        target_flux=0.0,
        barrier_rate=5.0,
        energy_neutral=True,
    )

    assert result.converged
    assert result.constraint_count > 96
    assert result.corrected_electron_lower_barrier_minimum_eigenvalue > -1e-11
    assert result.corrected_electron_upper_barrier_minimum_eigenvalue > -1e-11
    assert result.corrected_boson_barrier_minimum_eigenvalue > -1e-11

    direct = structured_joint_barrier_correction(
        state,
        derivative,
        parameters,
        activation_margin=1e-5,
        target_flux=0.0,
        barrier_rate=5.0,
        energy_neutral=True,
        solver="direct_eigenvalue",
    )
    assert direct.converged
    assert direct.constraint_count == 8
    np.testing.assert_allclose(
        direct.correction_norm,
        result.correction_norm,
        rtol=1e-8,
        atol=1e-10,
    )


def test_full_state_controls_restore_energy_neutral_barrier_feasibility() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = np.array(
        [
            0.7658644803220191,
            0.3190934652437296,
            -0.0099051815846953,
            -2.1900372789975457,
            -0.02680042823650692,
            -0.2594977841122353,
            0.02649610053452796,
            0.04571073581886077,
            0.04782851580479525,
            0.02247876557189533,
            0.00502103096238836,
            -0.11192811847520523,
            -0.06636865373365847,
            -0.13170540471051598,
            -0.07320404792141004,
            -0.11020798994684143,
            0.04809745998758826,
            0.03133442826929894,
            -0.04518961555825599,
            -0.00507170953082324,
            0.12001561984005256,
            0.02197824334396107,
            0.07803017665079814,
            0.00324762241900395,
            -0.09764653111955272,
            0.00868015659297977,
            -0.04647030099004339,
            0.04227952874720384,
            0.00702428613852605,
            -0.0840637843259177,
            -0.07172678995556854,
        ]
    )
    derivative = closed_scalar_rhs(28.57, state, parameters)
    restricted = structured_joint_barrier_correction(
        state,
        derivative,
        parameters,
        activation_margin=1e-5,
        barrier_rate=5.0,
        energy_neutral=True,
        cone_tolerance=1e-9,
        solver="direct_eigenvalue",
    )
    full = structured_full_state_joint_barrier_correction(
        state,
        derivative,
        parameters,
        activation_margin=1e-5,
        barrier_rate=5.0,
        energy_neutral=True,
        cone_tolerance=1e-9,
    )

    assert not restricted.converged
    assert full.converged
    assert full.corrected_electron_lower_barrier_minimum_eigenvalue > -1e-9
    assert full.corrected_electron_upper_barrier_minimum_eigenvalue > -1e-9
    assert full.corrected_boson_barrier_minimum_eigenvalue > -1e-9
    assert np.linalg.norm(full.correction_coordinates[17:]) > 0.0
    np.testing.assert_allclose(
        closed_state_correction_energy_gradient(state, parameters)
        @ full.correction_coordinates,
        0.0,
        atol=2e-11,
    )


def test_electron_phonon_barrier_controls_c_and_preserves_its_trace() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )
    derivative = closed_scalar_rhs(0.0, state, parameters)

    cutting_plane = structured_electron_phonon_barrier_correction(
        state,
        derivative,
        parameters,
        activation_margin=1e-5,
        energy_neutral=True,
        preserve_correlation_trace=True,
    )
    direct = structured_electron_phonon_barrier_correction(
        state,
        derivative,
        parameters,
        activation_margin=1e-5,
        energy_neutral=True,
        preserve_correlation_trace=True,
        solver="direct_eigenvalue",
    )

    assert cutting_plane.converged
    assert direct.converged
    assert cutting_plane.raw_joint_barrier_minimum_eigenvalue < -2e-3
    assert cutting_plane.corrected_joint_barrier_minimum_eigenvalue > -1e-9
    assert abs(cutting_plane.corrected_correlation_trace_velocity) < 1e-12
    assert np.linalg.norm(cutting_plane.correction_coordinates[17:]) > 5e-2
    np.testing.assert_allclose(
        cutting_plane.correction_energy_flux,
        0.0,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        cutting_plane.correction_norm,
        direct.correction_norm,
        rtol=2e-7,
        atol=1e-9,
    )


def test_frobenius_metric_selects_the_frobenius_minimum_correction() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )
    derivative = closed_scalar_rhs(0.0, state, parameters)
    common = {
        "activation_margin": 1e-5,
        "energy_neutral": True,
        "preserve_correlation_trace": True,
    }
    default = structured_electron_phonon_barrier_correction(
        state,
        derivative,
        parameters,
        **common,
    )
    euclidean = structured_electron_phonon_barrier_correction(
        state,
        derivative,
        parameters,
        correction_metric="euclidean",
        **common,
    )
    frobenius = structured_electron_phonon_barrier_correction(
        state,
        derivative,
        parameters,
        correction_metric="frobenius",
        **common,
    )
    frobenius_direct = structured_electron_phonon_barrier_correction(
        state,
        derivative,
        parameters,
        correction_metric="frobenius",
        solver="direct_eigenvalue",
        **common,
    )

    assert euclidean.converged
    assert frobenius.converged
    assert frobenius_direct.converged
    np.testing.assert_allclose(
        default.correction_coordinates,
        euclidean.correction_coordinates,
        atol=1e-13,
    )
    assert euclidean.correction_norm <= frobenius.correction_norm + 1e-10
    assert (
        frobenius.lifted_frobenius_norm
        <= euclidean.lifted_frobenius_norm + 1e-10
    )
    assert np.linalg.norm(
        euclidean.correction_coordinates - frobenius.correction_coordinates
    ) > 1e-4
    np.testing.assert_allclose(
        frobenius.correction_coordinates,
        frobenius_direct.correction_coordinates,
        rtol=2e-6,
        atol=2e-8,
    )


def test_joint_moment_barrier_prevents_the_first_raw_cone_exit() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )

    raw_solution = solve_ivp(
        lambda time, state: closed_scalar_rhs(time, state, parameters),
        (0.0, 0.2),
        initial,
        method="DOP853",
        t_eval=np.linspace(0.0, 0.2, 41),
        rtol=1e-10,
        atol=1e-12,
        max_step=0.01,
    )
    assert raw_solution.success
    raw_joint_minimum = min(
        float(
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    closed_scalar_to_matrix_state(raw_solution.y[:, index])
                )
            )[0]
        )
        for index in range(raw_solution.y.shape[1])
    )
    assert raw_joint_minimum < -1e-3

    corrected_rhs = closed_electron_phonon_cone_projected_rhs(
        parameters,
        initial,
        activation_margin=1e-5,
        barrier_rate=5.0,
        energy_neutral=True,
        preserve_correlation_trace=True,
        cone_tolerance=1e-8,
    )
    state = initial.copy()
    time_value = 0.0
    time_step = 0.01
    corrected_joint_minimum = float("inf")
    for _ in range(20):
        k1 = corrected_rhs(time_value, state)
        k2 = corrected_rhs(
            time_value + 0.5 * time_step,
            state + 0.5 * time_step * k1,
        )
        k3 = corrected_rhs(
            time_value + 0.5 * time_step,
            state + 0.5 * time_step * k2,
        )
        k4 = corrected_rhs(
            time_value + time_step,
            state + time_step * k3,
        )
        state = state + (time_step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        time_value += time_step
        corrected_joint_minimum = min(
            corrected_joint_minimum,
            float(
                np.linalg.eigvalsh(
                    electron_phonon_moment_matrix(
                        closed_scalar_to_matrix_state(state)
                    )
                )[0]
            ),
        )

    assert corrected_joint_minimum > 1e-5
    np.testing.assert_allclose(state[17:19], 0.0, atol=2e-12)


def test_joint_barrier_is_noop_when_all_raw_barriers_are_psd() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    result = structured_joint_barrier_correction(
        state,
        np.zeros_like(state),
        parameters,
        activation_margin=0.0,
        target_flux=0.0,
        barrier_rate=5.0,
        energy_neutral=True,
    )

    assert result.converged
    assert result.constraint_count == 0
    np.testing.assert_allclose(result.correction_coordinates, 0.0)


def test_joint_projected_rhs_changes_only_electron_and_boson_moments() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=12,
    )
    base_rhs = closed_residual_subtracted_rhs(parameters, initial_state)
    corrected_rhs = closed_joint_cone_projected_rhs(
        parameters,
        initial_state,
        activation_margin=1e-5,
        target_flux=0.4,
        barrier_rate=5.0,
        energy_neutral=True,
    )
    base = base_rhs(0.3, initial_state)
    corrected = corrected_rhs(0.3, initial_state)

    np.testing.assert_allclose(corrected[3:7], base[3:7])
    np.testing.assert_allclose(corrected[17:], base[17:])
    assert np.linalg.norm(corrected[:3] - base[:3]) > 0.0
    assert np.linalg.norm(corrected[7:17] - base[7:17]) > 0.0


def test_cone_projected_strong_trajectory_stays_inside_boson_cone() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )
    rhs = closed_cone_projected_rhs(
        parameters,
        initial_state,
        activation_margin=1e-5,
        target_flux=0.0,
    )
    sample_times = np.linspace(0.0, 4.0, 401)
    solution = solve_ivp(
        rhs,
        (0.0, 4.0),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.02,
    )

    assert solution.success
    minimum_moment_eigenvalue = min(
        closed_boson_moment_eigenvalues(solution.y[:, index])[0]
        for index in range(solution.y.shape[1])
    )
    assert minimum_moment_eigenvalue > -1e-8
    assert float(np.max(np.abs(solution.y))) < 5.0
