from __future__ import annotations

from dataclasses import fields

import numpy as np

from paper5.stability import (
    CLOSED_SCALAR_STATE_NAMES,
    EXTENDED_FAN_MIGDAL_STATE_NAMES,
    FAN_MIGDAL_STATE_NAMES,
    DimerParameters,
    closed_scalar_rhs,
    fan_migdal_rhs,
    fan_migdal_with_anomalous_rhs,
    hartree_fock_zero_correlation_state,
)
from paper5.stability.matrix_reference import (
    MatrixDimerState,
    closed_scalar_embedding_normal_residual,
    closed_scalar_to_matrix_state,
    discover_invariant_closure,
    extended_scalar_embedding_normal_residual,
    extended_scalar_to_matrix_state,
    matrix_derivative_to_closed_scalar,
    matrix_derivative_to_scalar,
    matrix_derivative_to_extended_scalar,
    matrix_dimer_rhs,
    matrix_total_energy,
    matrix_state_to_closed_scalar_coordinates,
    pauli_repaired_closed_scalar_rhs,
    pauli_repaired_matrix_dimer_rhs,
    pack_matrix_state,
    same_spin_density_covariance,
    same_spin_pauli_velocity_correction,
    scalar_embedding_normal_residual,
    scalar_to_matrix_state,
    unpack_matrix_state,
)


def test_matrix_state_real_vector_round_trip() -> None:
    rng = np.random.default_rng(1401)
    scalar_state = rng.normal(size=len(FAN_MIGDAL_STATE_NAMES))
    state = scalar_to_matrix_state(scalar_state)

    round_trip = unpack_matrix_state(pack_matrix_state(state))

    for field in fields(MatrixDimerState):
        np.testing.assert_array_equal(
            getattr(round_trip, field.name),
            getattr(state, field.name),
        )


def test_scalar_rhs_matches_projected_matrix_equations() -> None:
    """Lock the component mapping independently of hand-simplified formulas."""

    rng = np.random.default_rng(260622233)
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )

    for _ in range(100):
        scalar_state = rng.normal(
            scale=0.15,
            size=len(FAN_MIGDAL_STATE_NAMES),
        )
        bloch = rng.normal(size=3)
        bloch *= rng.uniform(0.0, 0.9) / np.linalg.norm(bloch)
        scalar_state[0] = bloch[2]
        scalar_state[1] = 0.5 * bloch[0]
        scalar_state[2] = 0.5 * bloch[1]
        scalar_state[5] = rng.uniform(0.05, 0.5)
        scalar_state[6] = rng.uniform(
            -scalar_state[5],
            scalar_state[5],
        )
        time = float(rng.uniform(0.0, 3.0))

        matrix_derivative = matrix_dimer_rhs(
            time,
            scalar_to_matrix_state(scalar_state),
            parameters,
        )
        projected_derivative = matrix_derivative_to_scalar(matrix_derivative)

        np.testing.assert_allclose(
            fan_migdal_rhs(time, scalar_state, parameters),
            projected_derivative,
            atol=2e-15,
            rtol=2e-15,
        )


def test_omitted_anomalous_phonon_field_is_generated_after_initial_time() -> None:
    """The thirteen-scalar slice is not closed under matrix Eq. (14c)."""

    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial_state = hartree_fock_zero_correlation_state()
    initial_derivative = fan_migdal_rhs(0.0, initial_state, parameters)

    initial_normal = scalar_embedding_normal_residual(
        matrix_dimer_rhs(
            0.0,
            scalar_to_matrix_state(initial_state),
            parameters,
        )
    )
    assert initial_normal["anomalous_phonon_rhs_norm"] == 0.0

    first_order_state = initial_state + 1e-3 * initial_derivative
    first_order_normal = scalar_embedding_normal_residual(
        matrix_dimer_rhs(
            1e-3,
            scalar_to_matrix_state(first_order_state),
            parameters,
        )
    )
    assert first_order_normal["anomalous_phonon_rhs_norm"] > 1e-4


def test_two_anomalous_coordinates_capture_14c_but_not_full_closure() -> None:
    """Lock the Eq. (14c) projection and expose the remaining normal sector."""

    rng = np.random.default_rng(140315)
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )

    maximum_preexisting_normal_residual = 0.0
    for _ in range(100):
        extended_state = rng.normal(
            scale=0.15,
            size=len(EXTENDED_FAN_MIGDAL_STATE_NAMES),
        )
        bloch = rng.normal(size=3)
        bloch *= rng.uniform(0.0, 0.9) / np.linalg.norm(bloch)
        extended_state[0] = bloch[2]
        extended_state[1] = 0.5 * bloch[0]
        extended_state[2] = 0.5 * bloch[1]
        extended_state[5] = rng.uniform(0.05, 0.5)
        extended_state[6] = rng.uniform(
            -extended_state[5],
            extended_state[5],
        )
        time = float(rng.uniform(0.0, 3.0))

        matrix_derivative = matrix_dimer_rhs(
            time,
            extended_scalar_to_matrix_state(extended_state),
            parameters,
        )
        projected_derivative = matrix_derivative_to_extended_scalar(
            matrix_derivative
        )

        np.testing.assert_allclose(
            fan_migdal_with_anomalous_rhs(
                time,
                extended_state,
                parameters,
            ),
            projected_derivative,
            atol=2e-15,
            rtol=2e-15,
        )
        normal_residual = extended_scalar_embedding_normal_residual(
            matrix_derivative
        )
        for name, value in normal_residual.items():
            if name.startswith("anomalous_"):
                assert value < 2e-15
            else:
                maximum_preexisting_normal_residual = max(
                    maximum_preexisting_normal_residual,
                    value,
                )

    # Eq. (14c) is now tangent, but the original scalar projection also
    # discarded correlation trace/mode-sum components.  Two extra coordinates
    # therefore repair Eq. (14c), not the complete matrix closure.
    assert maximum_preexisting_normal_residual > 1e-3


def test_enlarged_scalar_system_is_matrix_exact_and_tangent() -> None:
    """The next scalar model must close every generated matrix direction."""

    rng = np.random.default_rng(14031599)
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    for _ in range(100):
        state = rng.normal(
            scale=0.1,
            size=len(CLOSED_SCALAR_STATE_NAMES),
        )
        time = float(rng.uniform(0.0, 3.0))
        matrix_derivative = matrix_dimer_rhs(
            time,
            closed_scalar_to_matrix_state(state),
            parameters,
        )
        np.testing.assert_allclose(
            closed_scalar_rhs(time, state, parameters),
            matrix_derivative_to_closed_scalar(matrix_derivative),
            atol=3e-14,
            rtol=3e-14,
        )
        assert max(
            closed_scalar_embedding_normal_residual(
                matrix_derivative
            ).values(),
            default=0.0,
        ) < 3e-14


def test_same_spin_pauli_repair_is_autonomous_and_changes_only_c_velocity() -> None:
    rng = np.random.default_rng(314159)
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    state = rng.normal(
        scale=0.1,
        size=len(CLOSED_SCALAR_STATE_NAMES),
    )
    time = 0.37
    matrix_state = closed_scalar_to_matrix_state(state)
    raw = closed_scalar_rhs(time, state, parameters)
    repaired = pauli_repaired_closed_scalar_rhs(time, state, parameters)
    matrix_repaired = pauli_repaired_matrix_dimer_rhs(
        time,
        matrix_state,
        parameters,
    )

    np.testing.assert_allclose(repaired[:17], raw[:17], atol=2e-15, rtol=2e-15)
    assert np.linalg.norm(repaired[17:] - raw[17:]) > 1e-6
    np.testing.assert_allclose(
        repaired,
        matrix_derivative_to_closed_scalar(matrix_repaired),
        atol=3e-14,
        rtol=3e-14,
    )
    np.testing.assert_allclose(
        repaired[17:] - raw[17:],
        matrix_derivative_to_closed_scalar(
            MatrixDimerState(
                electron_density=np.zeros((2, 2), dtype=complex),
                coherent_phonon=np.zeros(2, dtype=complex),
                phonon_density=np.zeros((2, 2), dtype=complex),
                anomalous_phonon_density=np.zeros((2, 2), dtype=complex),
                electron_phonon_correlation=(
                    same_spin_pauli_velocity_correction(
                        matrix_state,
                        parameters,
                    )
                ),
            )
        )[17:],
        atol=3e-14,
        rtol=3e-14,
    )


def test_same_spin_density_covariance_obeys_fixed_sector_formula() -> None:
    rho = np.array(
        [[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]],
        dtype=complex,
    )
    covariance = same_spin_density_covariance(rho)

    for q in range(2):
        for i in range(2):
            for j in range(2):
                expected = (
                    (rho[q, j] if i == q else 0.0)
                    - rho[i, j] * rho[q, q]
                )
                assert covariance[q, i, j] == expected
    np.testing.assert_allclose(
        np.trace(covariance, axis1=1, axis2=2),
        0.0,
        atol=2e-16,
    )


def test_closed_scalar_matrix_round_trip() -> None:
    rng = np.random.default_rng(3100)
    for _ in range(100):
        state = rng.normal(
            scale=0.2,
            size=len(CLOSED_SCALAR_STATE_NAMES),
        )
        np.testing.assert_allclose(
            matrix_state_to_closed_scalar_coordinates(
                closed_scalar_to_matrix_state(state)
            ),
            state,
            atol=2e-16,
            rtol=2e-16,
        )


def test_invariant_closure_discovery_reaches_31_dimensions() -> None:
    result = discover_invariant_closure(
        DimerParameters(
            lambda_ep=1.5,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        samples_per_iteration=100,
        validation_samples=100,
    )

    assert result["ambient_real_dimension"] == 44
    assert result["initial_dimension"] == 15
    assert result["closure_dimension"] == len(CLOSED_SCALAR_STATE_NAMES) == 31
    assert result["maximum_validation_residual"] < 3e-14


def test_matrix_equations_conserve_energy_without_drive_instantaneously() -> None:
    rng = np.random.default_rng(22)
    parameters = DimerParameters(
        lambda_ep=0.5,
        gamma=0.5,
        drive_amplitude=0.0,
    )
    epsilon = 1e-7

    for _ in range(10):
        scalar_state = rng.normal(
            scale=0.1,
            size=len(FAN_MIGDAL_STATE_NAMES),
        )
        scalar_state[0] = rng.uniform(-0.4, 0.4)
        scalar_state[1] = rng.uniform(-0.3, 0.3)
        state = scalar_to_matrix_state(scalar_state)
        derivative = matrix_dimer_rhs(0.0, state, parameters)

        plus = MatrixDimerState(
            *(
                getattr(state, field.name)
                + epsilon * getattr(derivative, field.name)
                for field in fields(MatrixDimerState)
            )
        )
        minus = MatrixDimerState(
            *(
                getattr(state, field.name)
                - epsilon * getattr(derivative, field.name)
                for field in fields(MatrixDimerState)
            )
        )
        directional_derivative = (
            matrix_total_energy(plus, parameters)
            - matrix_total_energy(minus, parameters)
        ) / (2.0 * epsilon)

        assert abs(directional_derivative) < 2e-8
