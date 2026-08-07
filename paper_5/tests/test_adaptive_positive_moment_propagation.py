from __future__ import annotations

import numpy as np

from paper5.stability.adaptive_positive_moment_propagation import (
    APCMSettings,
    ArchiveBackedAPCM,
    initialize_apcm_state,
    integrate_apcm_ssprk3,
    pack_apcm_state,
    unpack_apcm_state,
)
from paper5.stability.adaptive_positive_moment import (
    matrix_derivative_to_raw_moment_velocity,
    raw_moment_coordinates_to_matrix_state,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import (
    MatrixDimerState,
    matrix_dimer_rhs,
)


def _parameters() -> DimerParameters:
    return DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )


def test_apcm_state_pack_round_trip() -> None:
    state = initialize_apcm_state(_parameters(), phonon_cutoff=8)
    raw, hidden = unpack_apcm_state(state)
    np.testing.assert_allclose(pack_apcm_state(raw, hidden), state)


def test_apcm_without_enrichment_or_controller_retains_archive_rhs() -> None:
    parameters = _parameters()
    state = initialize_apcm_state(parameters, phonon_cutoff=16)
    model = ArchiveBackedAPCM(
        parameters,
        settings=APCMSettings(
            include_k=False,
            include_pauli=False,
            include_opposite_spin=False,
            apply_physicality_controller=False,
        ),
    )
    evaluation = model.evaluate(0.0, state)
    raw, _ = unpack_apcm_state(state)
    matrix_state = raw_moment_coordinates_to_matrix_state(raw)
    matrix_velocity = matrix_dimer_rhs(0.0, matrix_state, parameters)
    correlation_velocity = matrix_velocity.electron_phonon_correlation.copy()
    for mode in range(2):
        correlation_velocity[mode] -= (
            0.5
            * np.trace(correlation_velocity[mode])
            * np.eye(2, dtype=complex)
        )
    matrix_velocity = MatrixDimerState(
        electron_density=matrix_velocity.electron_density,
        coherent_phonon=matrix_velocity.coherent_phonon,
        phonon_density=matrix_velocity.phonon_density,
        anomalous_phonon_density=matrix_velocity.anomalous_phonon_density,
        electron_phonon_correlation=correlation_velocity,
    )
    np.testing.assert_allclose(
        evaluation.derivative[: raw.size],
        matrix_derivative_to_raw_moment_velocity(
            matrix_state,
            matrix_velocity,
        ),
        atol=3e-13,
        rtol=3e-13,
    )


def test_enriched_apcm_rhs_is_finite_and_completion_is_positive() -> None:
    parameters = _parameters()
    state = initialize_apcm_state(parameters, phonon_cutoff=16)
    evaluation = ArchiveBackedAPCM(parameters).evaluate(0.0, state)
    assert np.all(np.isfinite(evaluation.derivative))
    assert evaluation.completion.success
    assert evaluation.completion.minimum_moment_matrix_eigenvalue > -1e-6
    assert evaluation.controller is not None
    assert evaluation.controller.converged
    assert np.linalg.norm(evaluation.kpd_correction) > 1e-4


def test_zero_cumulant_ablation_skips_extended_cone_retraction() -> None:
    parameters = _parameters()
    state = initialize_apcm_state(parameters, phonon_cutoff=8)
    model = ArchiveBackedAPCM(
        parameters,
        settings=APCMSettings(
            terminal_completion="zero_cumulant_prior",
        ),
    )
    trajectory = integrate_apcm_ssprk3(
        model,
        state,
        final_time=0.0025,
        time_step=0.0025,
    )
    assert trajectory.success
    assert trajectory.hidden_retraction_norms[-1] == 0.0
    assert np.isfinite(trajectory.completion_minimum_eigenvalues[-1])


def test_ssprk3_continuation_uses_absolute_initial_time() -> None:
    parameters = _parameters()
    state = initialize_apcm_state(parameters, phonon_cutoff=8)
    model = ArchiveBackedAPCM(
        parameters,
        settings=APCMSettings(terminal_completion="zero_cumulant_prior"),
    )
    trajectory = integrate_apcm_ssprk3(
        model,
        state,
        initial_time=1.0,
        final_time=1.0025,
        time_step=0.0025,
    )
    np.testing.assert_allclose(trajectory.times, [1.0, 1.0025])
