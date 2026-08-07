from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.sparse import eye, kron

from paper5.stability import DimerParameters
from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
    _ground_state,
)
from paper5.stability.krylov_memory_closure import (
    RAW_MOMENT_NAMES,
    _apply_liouvillian,
    _block_gram,
    _hs_inner,
    _operator_block_norm,
    _project_block_out,
    build_krylov_closure_construction,
    build_raw_moment_basis,
    centered_jacobian_from_orthonormal,
    closed_coordinates_to_orthonormal,
    closed_coordinates_to_raw_moments,
    orthonormal_to_closed_coordinates,
    raw_moments_to_closed_coordinates,
    raw_to_closed_jacobian,
)
from paper5.stability.krylov_memory_analysis import teacher_forced_krylov_gate
from paper5.stability.matrix_reference import (
    matrix_state_to_closed_scalar_coordinates,
)


def _random_raw_moments(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(scale=0.2, size=len(RAW_MOMENT_NAMES))
    raw[7:9] += 0.8
    return raw


def test_raw_and_centered_physical_charts_round_trip() -> None:
    rng = np.random.default_rng(2026080301)
    for _ in range(100):
        raw = _random_raw_moments(rng)
        closed = raw_moments_to_closed_coordinates(raw)

        np.testing.assert_allclose(
            closed_coordinates_to_raw_moments(closed),
            raw,
            atol=3e-16,
            rtol=3e-16,
        )
        assert closed[17] == 0.0
        assert closed[18] == 0.0


def test_raw_to_centered_analytic_jacobian_matches_finite_differences() -> None:
    rng = np.random.default_rng(2026080302)
    raw = _random_raw_moments(rng)
    analytic = raw_to_closed_jacobian(raw)
    step = 1e-6
    finite_difference = np.empty_like(analytic)
    for column in range(raw.size):
        offset = np.zeros_like(raw)
        offset[column] = step
        finite_difference[:, column] = (
            raw_moments_to_closed_coordinates(raw + offset)
            - raw_moments_to_closed_coordinates(raw - offset)
        ) / (2.0 * step)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        atol=2e-10,
        rtol=2e-10,
    )


def test_hilbert_schmidt_raw_basis_is_orthonormal_and_has_expected_identity() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    basis = build_raw_moment_basis(parameters, phonon_cutoff=2)
    overlap = _block_gram(
        basis.orthonormal_observables,
        basis.orthonormal_observables,
        dimension=basis.hilbert_dimension,
    )

    np.testing.assert_allclose(overlap, np.eye(29), atol=2e-13, rtol=2e-13)
    expected_identity = np.zeros(29)
    expected_identity[7:9] = 1.0
    np.testing.assert_allclose(
        basis.identity_expectations,
        expected_identity,
        atol=2e-15,
        rtol=2e-15,
    )


def test_exact_ground_state_contractions_match_raw_and_centered_charts() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    cutoff = 2
    model = _build_exact_dimer_model(parameters, phonon_cutoff=cutoff)
    _, state = _ground_state(model, eigensolver_tolerance=1e-12)
    basis = build_raw_moment_basis(parameters, phonon_cutoff=cutoff)

    orthonormal = basis.contract_state(state)
    reconstructed = orthonormal_to_closed_coordinates(basis, orthonormal)
    contracted = matrix_state_to_closed_scalar_coordinates(
        _contract_matrix_state(model, state)
    )

    np.testing.assert_allclose(reconstructed, contracted, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        closed_coordinates_to_orthonormal(basis, contracted),
        orthonormal,
        atol=2e-12,
        rtol=2e-12,
    )


def test_orthonormal_centering_jacobian_matches_direct_reconstruction() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    basis = build_raw_moment_basis(parameters, phonon_cutoff=2)
    rng = np.random.default_rng(2026080303)
    coordinates = rng.normal(scale=0.2, size=29)
    analytic = centered_jacobian_from_orthonormal(basis, coordinates)
    step = 1e-6
    finite_difference = np.empty_like(analytic)
    for column in range(coordinates.size):
        offset = np.zeros_like(coordinates)
        offset[column] = step
        finite_difference[:, column] = (
            orthonormal_to_closed_coordinates(basis, coordinates + offset)
            - orthonormal_to_closed_coordinates(basis, coordinates - offset)
        ) / (2.0 * step)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        atol=2e-10,
        rtol=2e-10,
    )


def test_gate_a_measures_full_finite_cutoff_force_and_drive_tangency() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    construction = build_krylov_closure_construction(
        parameters,
        phonon_cutoff=2,
        shell_count=5,
    )

    # The exact cutoff construction must report, rather than force, the rank.
    assert construction.force_rank == 19
    assert construction.shell_dimensions == (19, 19, 19, 19, 19)
    dimension = construction.hilbert_dimension
    drive_action = _apply_liouvillian(
        construction.drive_hamiltonian,
        construction.raw_basis.orthonormal_observables,
    )
    assert (
        _operator_block_norm(construction.drive_force, dimension=dimension)
        / _operator_block_norm(drive_action, dimension=dimension)
        < 1e-12
    )
    assert max(construction.retained_symmetric_leakage.values()) < 1e-12


def test_gate_a_augmented_velocity_matches_exact_retained_velocity() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    cutoff = 2
    construction = build_krylov_closure_construction(
        parameters,
        phonon_cutoff=cutoff,
        shell_count=3,
    )
    coefficients = construction.coefficients(3)
    model = _build_exact_dimer_model(parameters, phonon_cutoff=cutoff)

    rng = np.random.default_rng(2026080304)
    vector = rng.normal(size=construction.hilbert_dimension) + 1j * rng.normal(
        size=construction.hilbert_dimension
    )
    oscillator_dimension = cutoff + 1
    electron_swap = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )
    swap = kron(
        electron_swap,
        eye(oscillator_dimension**2, format="csc", dtype=complex),
        format="csc",
    )
    vector = vector + swap @ vector
    vector /= np.linalg.norm(vector)

    drive_value = 0.37
    hamiltonian = (
        model.static_hamiltonian + drive_value * model.drive_operator
    ).tocsc()
    state_velocity = -1j * (hamiltonian @ vector)
    exact_velocity = np.asarray(
        [
            (
                np.vdot(state_velocity, operator @ vector)
                + np.vdot(vector, operator @ state_velocity)
            ).real
            for operator in construction.raw_basis.orthonormal_observables
        ]
    )
    retained = construction.raw_basis.contract_state(vector)
    auxiliary = coefficients.contract_auxiliary_state(vector)
    modeled_velocity, _ = coefficients.orthonormal_velocity(
        retained,
        auxiliary,
        drive_value=drive_value,
    )

    np.testing.assert_allclose(
        modeled_velocity,
        exact_velocity,
        atol=8e-13,
        rtol=8e-13,
    )


def test_gate_a_energy_direction_is_invariant_under_augmented_generator() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    construction = build_krylov_closure_construction(
        parameters,
        phonon_cutoff=2,
        shell_count=3,
    )
    coefficients = construction.coefficients(3)
    dimension = construction.hilbert_dimension
    identity = eye(dimension, format="csc", dtype=complex)
    basis = (
        *construction.raw_basis.orthonormal_observables,
        *coefficients.auxiliary_observables,
    )

    for drive_value in (0.0, 0.7):
        hamiltonian = (
            construction.static_hamiltonian
            + drive_value * construction.drive_hamiltonian
        ).tocsc()
        identity_part = _hs_inner(
            identity,
            hamiltonian,
            dimension=dimension,
        ).real
        centered_hamiltonian = (hamiltonian - identity_part * identity).tocsc()
        energy_coefficients = np.asarray(
            [
                _hs_inner(operator, centered_hamiltonian, dimension=dimension).real
                for operator in basis
            ]
        )
        span_residual = _project_block_out(
            (centered_hamiltonian,),
            basis,
            dimension=dimension,
            passes=2,
        )[0]
        generator = np.block(
            [
                [
                    coefficients.retained_static
                    + drive_value * coefficients.retained_drive,
                    -coefficients.retained_to_auxiliary.T,
                ],
                [
                    coefficients.retained_to_auxiliary,
                    coefficients.auxiliary_static
                    + drive_value * coefficients.auxiliary_drive,
                ],
            ]
        )

        assert sqrt(
            max(
                0.0,
                _hs_inner(
                    span_residual,
                    span_residual,
                    dimension=dimension,
                ).real,
            )
        ) < 1e-12
        assert np.linalg.norm(generator @ energy_coefficients) < 1e-12


def test_gate_a_decoupled_force_deflates_to_zero_rank() -> None:
    construction = build_krylov_closure_construction(
        DimerParameters(lambda_ep=0.0, gamma=0.5),
        phonon_cutoff=2,
        shell_count=2,
    )

    assert construction.force_rank == 0
    assert construction.shell_dimensions == ()
    assert (
        _operator_block_norm(
            construction.static_force,
            dimension=construction.hilbert_dimension,
        )
        < 1e-12
    )


def test_teacher_forced_gate_integrates_auxiliary_state_without_exact_feedback() -> None:
    result = teacher_forced_krylov_gate(
        DimerParameters(lambda_ep=1.5, gamma=0.5),
        phonon_cutoff=2,
        final_time=0.02,
        sample_step=0.01,
        orders=(2, 3),
        maximum_step=0.01,
    )

    assert result.times.shape == (3,)
    assert result.exact_closed_coordinates.shape == (3, 31)
    assert result.exact_closed_derivatives.shape == (3, 31)
    for order, order_result in result.orders.items():
        assert order_result.auxiliary_coordinates.shape == (
            3,
            sum(result.construction.shell_dimensions[:order]),
        )
        np.testing.assert_allclose(
            order_result.modeled_derivatives[0],
            result.exact_closed_derivatives[0],
            atol=1e-12,
            rtol=1e-12,
        )
        assert np.all(np.isfinite(order_result.total_residual_norms))
