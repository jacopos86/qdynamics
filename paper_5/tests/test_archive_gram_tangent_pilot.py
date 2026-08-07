from __future__ import annotations

import numpy as np

from paper5.stability.archive_gram_tangent_pilot import (
    archive_mixed_tangent_pilot_point,
    full_state_matrix_derivative,
    packet_archive_mixed_frames,
)
from paper5.stability.hubbard_dimer import (
    DimerParameters,
    GaussianSineDrive,
)
from paper5.stability.matrix_reference import (
    matrix_derivative_to_closed_scalar,
)
from paper5.stability.moment_hierarchy import moment_hierarchy
from paper5.stability.multi_coherent import (
    pack_multi_coherent_parameters,
    relative_state_moment_coordinates,
    relative_state_moment_derivative,
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


def test_archive_operator_gram_matches_contracted_moment_gram() -> None:
    frames = packet_archive_mixed_frames(
        _symmetric_packet_parameters(),
        relative_dimension=9,
    )

    assert frames.spin_swap_fidelity > 1.0 - 1e-14
    assert frames.symmetry_projection_fidelity > 1.0 - 1e-14
    assert frames.archive_gram_max_error < 1e-12
    np.testing.assert_allclose(
        frames.full_state.conj() @ frames.packet_tangent,
        0.0,
        atol=1e-13,
    )


def test_full_tangent_contraction_matches_relative_hierarchy_map() -> None:
    rng = np.random.default_rng(260804)
    relative_dimension = 7
    state = rng.normal(size=4 * relative_dimension) + 1j * rng.normal(
        size=4 * relative_dimension
    )
    tangent = rng.normal(size=state.size) + 1j * rng.normal(size=state.size)
    state.reshape(4, relative_dimension)[:, -1] = 0.0
    tangent.reshape(4, relative_dimension)[:, -1] = 0.0
    swap = np.kron(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        np.eye(relative_dimension),
    )
    state = 0.5 * (state + swap @ state)
    state /= np.linalg.norm(state)
    tangent = 0.5 * (tangent + swap @ tangent)
    tangent -= state * np.vdot(state, tangent)

    full_state = np.concatenate((state, np.zeros_like(state)))
    full_tangent = np.concatenate((tangent, np.zeros_like(tangent)))
    contracted = matrix_derivative_to_closed_scalar(
        full_state_matrix_derivative(
            full_state,
            full_tangent,
            relative_dimension=relative_dimension,
        )
    )

    hierarchy = moment_hierarchy(4)
    coordinates = relative_state_moment_coordinates(
        state,
        hierarchy,
        center_amplitude=0.0j,
    )
    coordinate_derivative = relative_state_moment_derivative(
        state,
        tangent,
        hierarchy,
        center_derivative=0.0j,
    )
    expected = matrix_derivative_to_closed_scalar(
        hierarchy.matrix_derivative(coordinates, coordinate_derivative)
    )
    np.testing.assert_allclose(contracted, expected, atol=2e-13)


def test_nested_tangent_augmentations_cannot_increase_residual() -> None:
    parameters = DimerParameters(lambda_ep=1.5, drive_amplitude=1.0)
    result = archive_mixed_tangent_pilot_point(
        _symmetric_packet_parameters(),
        time=1.25,
        parameters=parameters,
        drive_protocol=GaussianSineDrive.from_parameters(parameters),
        relative_dimension=9,
        coordinate_scales=np.ones(31),
    )
    index = {name: offset for offset, name in enumerate(result.space_names)}

    assert (
        result.hilbert_relative_residual[index["archive_mixed"]]
        <= result.hilbert_relative_residual[index["archive"]] + 1e-12
    )
    assert (
        abs(
            result.hilbert_relative_residual[
                index["archive_relative_mixed"]
            ]
            - result.hilbert_relative_residual[index["archive_mixed"]]
        )
        < 1e-12
    )
    assert (
        result.hilbert_relative_residual[index["packet_archive"]]
        <= result.hilbert_relative_residual[index["packet_geometric"]]
        + 1e-12
    )
    assert (
        abs(
            result.hilbert_relative_residual[
                index["packet_relative_mixed"]
            ]
            - result.hilbert_relative_residual[index["packet_mixed"]]
        )
        < 1e-12
    )
    assert (
        result.hilbert_relative_residual[index["packet_mixed"]]
        <= result.hilbert_relative_residual[index["packet_archive"]]
        + 1e-12
    )
    assert np.all(result.mixed_candidate_residual_reduction >= -1e-14)
    assert np.all(result.mixed_candidate_residual_reduction <= 1.0 + 1e-12)
