from __future__ import annotations

import numpy as np

from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_derivative,
    _contract_matrix_state,
    _ground_state,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import (
    matrix_state_to_closed_scalar_coordinates,
)
from pipelines.open_dynamics.analyze_trajectory_closure_identifiability import (
    _correlation_coordinates,
    _exact_c_derivative_batch,
    _source_subspace_scan,
)


def test_batched_exact_c_derivative_matches_established_contraction() -> None:
    parameters = DimerParameters(
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    drive = GaussianSineDrive(
        amplitude=1.0,
        pulse_width=1.0,
        delays=(0.0, 8.0),
    )
    model = _build_exact_dimer_model(parameters, phonon_cutoff=2)
    _, state = _ground_state(model, eigensolver_tolerance=1e-12)
    time = 0.37
    hamiltonian = (
        model.static_hamiltonian
        + drive.difference(time) * model.drive_operator
    )
    state_derivative = -1j * (hamiltonian @ state)
    matrix_state = _contract_matrix_state(model, state)
    matrix_derivative = _contract_matrix_derivative(
        model,
        state,
        state_derivative,
        matrix_state,
    )
    closed = matrix_state_to_closed_scalar_coordinates(matrix_state)

    actual = _exact_c_derivative_batch(
        model,
        state[None, :],
        np.array([time]),
        closed[None, :],
        drive,
    )[0]
    expected = _correlation_coordinates(
        matrix_derivative.electron_phonon_correlation
    )

    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_source_subspace_scan_recovers_shared_low_rank_source() -> None:
    rng = np.random.default_rng(260804)
    basis, _ = np.linalg.qr(rng.normal(size=(14, 2)))
    times = np.linspace(0.0, 4.0, 41)
    target = np.empty((3, times.size, 14), dtype=float)
    for member in range(3):
        coefficients = np.column_stack(
            (
                np.sin(times) + 0.01 * member,
                np.cos(2.0 * times) - 0.02 * member,
            )
        )
        target[member] = coefficients @ basis.T

    summary, _, _, _ = _source_subspace_scan(target, np.ones(31))

    assert summary["rank_scan"][0][
        "all_member_normalized_reconstruction_rms"
    ] > 0.1
    assert summary["rank_scan"][1][
        "all_member_normalized_reconstruction_rms"
    ] < 1e-12
