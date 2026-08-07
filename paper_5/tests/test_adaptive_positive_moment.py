from __future__ import annotations

import numpy as np

from paper5.stability.adaptive_positive_moment import (
    ARCHIVE_RELATIVE_MOMENT_KEYS,
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    HIDDEN_RELATIVE_MOMENT_KEYS,
    INITIAL_PROMOTION_CANDIDATE_KEYS,
    connected_k_from_relative_moments,
    matrix_derivative_to_raw_moment_velocity,
    matrix_state_to_raw_moment_coordinates,
    opposite_spin_covariance_from_relative_moments,
    raw_moment_coordinates_to_matrix_state,
    raw_moment_schur_complement,
    relative_moments_from_matrix_state,
)
from paper5.stability.exact_reference import (
    exact_holstein_correlation_closure_trajectory,
    exact_holstein_moment_hierarchy_trajectory,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import (
    MatrixDimerState,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    matrix_derivative_to_closed_scalar,
    matrix_dimer_rhs,
    matrix_state_to_closed_scalar_coordinates,
)
from paper5.stability.moment_hierarchy import THIRD_ORDER_HIERARCHY


def _parameters() -> DimerParameters:
    return DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )


def test_raw_29_coordinate_chart_round_trips_exact_contractions() -> None:
    exact = exact_holstein_correlation_closure_trajectory(
        _parameters(),
        sample_times=np.asarray([0.0, 0.2]),
        phonon_cutoff=8,
        maximum_step=0.02,
    ).exact_trajectory
    for state in exact.matrix_states:
        coordinates = matrix_state_to_raw_moment_coordinates(state)
        reconstructed = raw_moment_coordinates_to_matrix_state(coordinates)
        for field in (
            "electron_density",
            "coherent_phonon",
            "phonon_density",
            "anomalous_phonon_density",
            "electron_phonon_correlation",
        ):
            np.testing.assert_allclose(
                getattr(reconstructed, field),
                getattr(state, field),
                atol=3e-10,
                rtol=3e-10,
            )


def test_uncentered_affine_matrix_has_joint_gram_schur_complement() -> None:
    exact = exact_holstein_correlation_closure_trajectory(
        _parameters(),
        sample_times=np.asarray([0.0, 0.01]),
        phonon_cutoff=8,
        maximum_step=0.01,
    ).exact_trajectory
    state = exact.matrix_states[0]
    coordinates = matrix_state_to_raw_moment_coordinates(state)
    np.testing.assert_allclose(
        raw_moment_schur_complement(coordinates),
        electron_phonon_moment_matrix(state),
        atol=4e-11,
        rtol=4e-11,
    )


def test_raw_moment_velocity_is_the_chart_directional_derivative() -> None:
    parameters = _parameters()
    exact = exact_holstein_correlation_closure_trajectory(
        parameters,
        sample_times=np.asarray([0.0, 0.2]),
        phonon_cutoff=8,
        maximum_step=0.02,
    ).exact_trajectory
    state = exact.matrix_states[-1]
    derivative = matrix_dimer_rhs(0.2, state, parameters)
    correlation_velocity = derivative.electron_phonon_correlation.copy()
    for mode in range(2):
        correlation_velocity[mode] -= (
            0.5
            * np.trace(correlation_velocity[mode])
            * np.eye(2, dtype=complex)
        )
    derivative = MatrixDimerState(
        electron_density=derivative.electron_density,
        coherent_phonon=derivative.coherent_phonon,
        phonon_density=derivative.phonon_density,
        anomalous_phonon_density=derivative.anomalous_phonon_density,
        electron_phonon_correlation=correlation_velocity,
    )
    velocity = matrix_derivative_to_raw_moment_velocity(state, derivative)
    closed_state = matrix_state_to_closed_scalar_coordinates(state)
    closed_velocity = matrix_derivative_to_closed_scalar(derivative)
    step = 1e-7
    plus = matrix_state_to_raw_moment_coordinates(
        closed_scalar_to_matrix_state(closed_state + step * closed_velocity)
    )
    minus = matrix_state_to_raw_moment_coordinates(
        closed_scalar_to_matrix_state(closed_state - step * closed_velocity)
    )
    np.testing.assert_allclose(
        velocity,
        (plus - minus) / (2.0 * step),
        atol=2e-8,
        rtol=2e-8,
    )


def test_relative_moment_decoder_matches_exact_k_and_d_audit() -> None:
    parameters = _parameters()
    times = np.asarray([0.0, 0.2])
    audit = exact_holstein_correlation_closure_trajectory(
        parameters,
        sample_times=times,
        phonon_cutoff=16,
        maximum_step=0.02,
    )
    hierarchy = exact_holstein_moment_hierarchy_trajectory(
        parameters,
        hierarchy=THIRD_ORDER_HIERARCHY,
        sample_times=times,
        phonon_cutoff=16,
        maximum_step=0.02,
    )
    for index, coordinates in enumerate(hierarchy.coordinates):
        _, moments = THIRD_ORDER_HIERARCHY.unpack(coordinates)
        np.testing.assert_allclose(
            connected_k_from_relative_moments(moments),
            audit.exact_mixed_moment[index]
            - audit.factorized_mixed_moment[index],
            # The relative-mode decoder uses canonical CCR.  The local-mode
            # cutoff-16 scorer differs by its top-projector commutator.
            atol=5e-5,
            rtol=5e-5,
        )
        np.testing.assert_allclose(
            opposite_spin_covariance_from_relative_moments(moments),
            audit.opposite_spin_covariance[index],
            atol=2e-10,
            rtol=2e-10,
        )


def test_hidden_relative_dictionary_is_six_pair_plus_25_third_moments() -> None:
    pair = [key for key in HIDDEN_RELATIVE_MOMENT_KEYS if key.degree == 2]
    third = [key for key in HIDDEN_RELATIVE_MOMENT_KEYS if key.degree == 3]
    assert len(pair) == 6
    assert len(third) == 25


def test_adaptive_dictionary_starts_from_tu_and_leaves_16_candidates() -> None:
    assert len(ARCHIVE_RELATIVE_MOMENT_KEYS) == 14
    assert len(ENTRANCE_RELATIVE_MOMENT_KEYS) == 15
    assert len(INITIAL_PROMOTION_CANDIDATE_KEYS) == 16
    assert set(ENTRANCE_RELATIVE_MOMENT_KEYS).isdisjoint(
        INITIAL_PROMOTION_CANDIDATE_KEYS
    )
    assert set(ENTRANCE_RELATIVE_MOMENT_KEYS).union(
        INITIAL_PROMOTION_CANDIDATE_KEYS
    ) == set(HIDDEN_RELATIVE_MOMENT_KEYS)


def test_archive_tuple_and_hidden_coordinates_reconstruct_hierarchy_state() -> None:
    parameters = _parameters()
    times = np.asarray([0.0, 0.2])
    audit = exact_holstein_correlation_closure_trajectory(
        parameters,
        sample_times=times,
        phonon_cutoff=16,
        maximum_step=0.02,
    )
    hierarchy = exact_holstein_moment_hierarchy_trajectory(
        parameters,
        hierarchy=THIRD_ORDER_HIERARCHY,
        sample_times=times,
        phonon_cutoff=16,
        maximum_step=0.02,
    )
    for index, coordinates in enumerate(hierarchy.coordinates):
        center, exact_moments = THIRD_ORDER_HIERARCHY.unpack(coordinates)
        hidden = {
            key: exact_moments[key] for key in HIDDEN_RELATIVE_MOMENT_KEYS
        }
        reconstructed_center, reconstructed = relative_moments_from_matrix_state(
            audit.exact_trajectory.matrix_states[index],
            hidden,
        )
        assert abs(reconstructed_center - center) < 2e-10
        maximum_error = max(
            abs(reconstructed[key] - exact_moments[key])
            for key in THIRD_ORDER_HIERARCHY.moment_keys
        )
        assert maximum_error < 4e-5
