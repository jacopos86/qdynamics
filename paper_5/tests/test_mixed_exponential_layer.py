from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from paper5.stability.archive_gram_tangent_pilot import (
    packet_archive_mixed_frames,
)
from paper5.stability.mixed_exponential_layer import (
    mixed_exponential_layer_state,
    mixed_exponential_origin_tangent,
    mixed_layer_centers,
    pack_mixed_coordinates,
    retract_mixed_exponential_layer,
    unpack_mixed_coordinates,
)
from paper5.stability.multi_coherent import (
    multi_coherent_state,
    pack_multi_coherent_parameters,
)


def _symmetric_packet_parameters() -> np.ndarray:
    coefficients = np.asarray(
        [[0.55], [0.32], [0.32], [0.45]],
        dtype=complex,
    )
    displacements = np.asarray(
        [[0.10 + 0.05j], [0.18 - 0.07j], [0.18 - 0.07j], [-0.08j]],
        dtype=complex,
    )
    return pack_multi_coherent_parameters(coefficients, displacements)


def _phase_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    value = candidate / np.linalg.norm(candidate)
    overlap = np.vdot(reference, value)
    return value * np.exp(-1j * np.angle(overlap))


def test_mixed_coordinate_packing_round_trip() -> None:
    values = np.asarray(
        [
            [0.1 + 0.2j, -0.3j, 0.4 - 0.1j],
            [-0.2 + 0.05j, 0.7j, -0.11 - 0.09j],
        ]
    )
    np.testing.assert_allclose(
        unpack_mixed_coordinates(pack_mixed_coordinates(values)),
        values,
    )


def test_chart_origin_matches_packet_and_low_boundary_truncated_frame() -> None:
    parameters = _symmetric_packet_parameters()
    relative_dimension = 11
    state, tangent = mixed_exponential_origin_tangent(
        parameters,
        relative_dimension=relative_dimension,
    )
    pilot = packet_archive_mixed_frames(
        parameters,
        relative_dimension=relative_dimension,
    )

    np.testing.assert_allclose(state, pilot.relative_state, atol=2e-14)
    np.testing.assert_allclose(
        tangent,
        pilot.relative_mixed_tangent[: 4 * relative_dimension],
        atol=5e-12,
    )


def test_analytic_packet_union_has_the_declared_origin_derivatives() -> None:
    parameters = _symmetric_packet_parameters()
    relative_dimension = 15
    centers = mixed_layer_centers(
        parameters,
        relative_dimension=relative_dimension,
    )
    state, tangent = mixed_exponential_origin_tangent(
        parameters,
        relative_dimension=relative_dimension,
        centers=centers,
    )
    step = 2e-6
    for column in range(12):
        offset = np.zeros(12)
        offset[column] = step
        plus = mixed_exponential_layer_state(
            parameters,
            offset,
            relative_dimension=relative_dimension,
            centers=centers,
        ).state
        minus = mixed_exponential_layer_state(
            parameters,
            -offset,
            relative_dimension=relative_dimension,
            centers=centers,
        ).state
        derivative = (plus - minus) / (2.0 * step)
        derivative -= state * np.vdot(state, derivative)
        np.testing.assert_allclose(derivative, tangent[:, column], atol=3e-9)


def test_origin_tangent_matches_analytic_layer_at_cutoff_boundary() -> None:
    coefficients = np.asarray(
        [[0.55], [0.32], [0.32], [0.45]],
        dtype=complex,
    )
    displacements = np.asarray(
        [[5.0], [-5.0], [-5.0], [5.0]],
        dtype=complex,
    )
    parameters = pack_multi_coherent_parameters(coefficients, displacements)
    relative_dimension = 33
    centers = mixed_layer_centers(
        parameters,
        relative_dimension=relative_dimension,
    )
    state, tangent = mixed_exponential_origin_tangent(
        parameters,
        relative_dimension=relative_dimension,
        centers=centers,
    )
    step = 2e-6
    for column in range(12):
        offset = np.zeros(12)
        offset[column] = step
        plus = mixed_exponential_layer_state(
            parameters,
            offset,
            relative_dimension=relative_dimension,
            centers=centers,
        ).state
        minus = mixed_exponential_layer_state(
            parameters,
            -offset,
            relative_dimension=relative_dimension,
            centers=centers,
        ).state
        derivative = (plus - minus) / (2.0 * step)
        derivative -= state * np.vdot(state, derivative)
        np.testing.assert_allclose(derivative, tangent[:, column], atol=3e-8)


def test_analytic_action_agrees_with_low_boundary_dense_exponential() -> None:
    parameters = _symmetric_packet_parameters()
    relative_dimension = 19
    centers = mixed_layer_centers(
        parameters,
        relative_dimension=relative_dimension,
    )
    values = np.asarray(
        [
            [0.012 - 0.004j, -0.006 + 0.003j, 0.009 + 0.002j],
            [-0.005 + 0.004j, 0.007 - 0.002j, -0.003 + 0.005j],
        ]
    )
    coordinates = pack_mixed_coordinates(values)
    analytic = mixed_exponential_layer_state(
        parameters,
        coordinates,
        relative_dimension=relative_dimension,
        centers=centers,
    ).state
    origin = mixed_exponential_layer_state(
        parameters,
        np.zeros(12),
        relative_dimension=relative_dimension,
        centers=centers,
    ).state

    annihilation = np.zeros((relative_dimension, relative_dimension), complex)
    for occupation in range(1, relative_dimension):
        annihilation[occupation - 1, occupation] = np.sqrt(occupation)
    creation = annihilation.conjugate().T
    oscillator_identity = np.eye(relative_dimension)
    electronic_identity = np.eye(4)
    pauli = (
        np.array([[0.0, 1.0], [1.0, 0.0]], complex),
        np.array([[0.0, -1j], [1j, 0.0]], complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], complex),
    )
    dense = origin.copy()
    for index in range(3):
        electronic = np.kron(pauli[index], np.eye(2))
        electronic -= centers.pauli_means[index] * electronic_identity
        boson_minus = (
            annihilation
            - centers.relative_amplitude * oscillator_identity
        )
        boson_plus = (
            creation
            - centers.relative_amplitude.conjugate() * oscillator_identity
        )
        generator = (
            values[0, index] * np.kron(electronic, boson_minus)
            + values[1, index] * np.kron(electronic, boson_plus)
        )
        dense = expm(generator) @ dense
    dense = _phase_align(origin, dense)
    np.testing.assert_allclose(analytic, dense, atol=2e-12)


def test_mixed_layer_retraction_recovers_packet_union_without_capacity_cap() -> None:
    parameters = _symmetric_packet_parameters()
    relative_dimension = 15
    centers = mixed_layer_centers(
        parameters,
        relative_dimension=relative_dimension,
    )
    coordinates = pack_mixed_coordinates(
        np.asarray(
            [
                [0.004 - 0.002j, -0.003j, 0.002 + 0.001j],
                [-0.003 + 0.001j, 0.002 - 0.001j, 0.001j],
            ]
        )
    )
    target = mixed_exponential_layer_state(
        parameters,
        coordinates,
        relative_dimension=relative_dimension,
        centers=centers,
    ).state
    retraction = retract_mixed_exponential_layer(
        parameters,
        coordinates,
        relative_dimension=relative_dimension,
        centers=centers,
        relative_tolerance=1e-10,
    )
    reconstructed = multi_coherent_state(
        retraction.parameters,
        relative_dimension=relative_dimension,
    )

    assert retraction.packet_count <= relative_dimension
    assert retraction.state_error < 2e-10
    assert retraction.fidelity > 1.0 - 1e-12
    np.testing.assert_allclose(reconstructed, target, atol=2e-10)
