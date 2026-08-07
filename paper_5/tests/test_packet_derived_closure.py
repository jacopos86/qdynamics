from __future__ import annotations

import numpy as np

from paper5.stability import DimerParameters, GaussianSineDrive
from paper5.stability.moment_hierarchy import moment_hierarchy
from paper5.stability.multi_coherent import (
    multi_coherent_state,
    pack_multi_coherent_parameters,
    relative_state_closed_coordinates,
)
from paper5.stability.packet_derived_closure import (
    normalized_scaled_source_error,
    packet_closed_velocity_pair,
    reconstruct_frozen_source_subspace,
    scaled_source_fluctuation_rms,
)


def _one_packet_parameters() -> np.ndarray:
    coefficients = np.array(
        [[0.52], [0.31], [0.27], [0.48]],
        dtype=complex,
    )
    displacements = np.array(
        [[-0.4 + 0.1j], [0.2 - 0.1j], [-0.1 + 0.3j], [0.5 - 0.2j]],
        dtype=complex,
    )
    return pack_multi_coherent_parameters(coefficients, displacements)


def test_packet_closed_velocity_matches_state_direction_finite_difference() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    drive = GaussianSineDrive.from_parameters(parameters)
    hierarchy = moment_hierarchy(4)
    packed = _one_packet_parameters()
    result = packet_closed_velocity_pair(
        packed,
        time=0.37,
        parameters=parameters,
        drive_protocol=drive,
        relative_dimension=7,
        hierarchy=hierarchy,
        center_amplitude=-0.6 + 0.0j,
    )
    np.testing.assert_allclose(
        result.closed_coordinates,
        relative_state_closed_coordinates(
            multi_coherent_state(packed, relative_dimension=7),
            hierarchy,
            center_amplitude=-0.6 + 0.0j,
        ),
        atol=1e-14,
        rtol=0.0,
    )
    assert result.hierarchy_coordinate_max_error == 0.0
    state = multi_coherent_state(packed, relative_dimension=7)
    normalized_state = state / np.linalg.norm(state)
    # Recover the projected state tangent indirectly from the parameter result
    # by re-evaluating the public helper through the packet velocity routine's
    # exact finite difference target.  The Schrodinger tangent is sufficient to
    # verify the moment-chain contraction independently of parameter packing.
    from paper5.stability.multi_coherent import (
        project_schrodinger_velocity,
        relative_holstein_hamiltonian,
    )

    projection = project_schrodinger_velocity(
        packed,
        relative_holstein_hamiltonian(
            0.37,
            parameters,
            relative_dimension=7,
            drive_protocol=drive,
        ),
        relative_dimension=7,
        regularization="tikhonov",
        relative_damping=3e-4,
        relative_singular_value_cutoff=1e-2,
    )
    step = 1e-6
    plus = relative_state_closed_coordinates(
        normalized_state + step * projection.target_velocity,
        hierarchy,
        center_amplitude=-0.6 + 0.0j,
    )
    minus = relative_state_closed_coordinates(
        normalized_state - step * projection.target_velocity,
        hierarchy,
        center_amplitude=-0.6 + 0.0j,
    )
    np.testing.assert_allclose(
        (plus - minus) / (2.0 * step),
        result.schrodinger_closed_velocity,
        atol=2e-8,
        rtol=2e-8,
    )
    projected_plus = relative_state_closed_coordinates(
        multi_coherent_state(
            packed + step * projection.parameter_velocity,
            relative_dimension=7,
        ),
        hierarchy,
        center_amplitude=-0.6 + 0.0j,
    )
    projected_minus = relative_state_closed_coordinates(
        multi_coherent_state(
            packed - step * projection.parameter_velocity,
            relative_dimension=7,
        ),
        hierarchy,
        center_amplitude=-0.6 + 0.0j,
    )
    np.testing.assert_allclose(
        (projected_plus - projected_minus) / (2.0 * step),
        result.projected_closed_velocity,
        atol=2e-8,
        rtol=2e-8,
    )


def test_frozen_source_projection_and_normalized_error() -> None:
    source = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 5.0]],
            [[0.0, 1.0, 2.0], [3.0, 5.0, 8.0]],
        ]
    )
    scales = np.array([1.0, 2.0, 4.0])
    center = np.array([1.0, 1.0, 1.0])
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coefficients, reconstructed = reconstruct_frozen_source_subspace(
        source,
        scales,
        center,
        basis,
    )
    assert coefficients.shape == (2, 2, 2)
    np.testing.assert_allclose(reconstructed[..., :2], source[..., :2])
    np.testing.assert_allclose(reconstructed[..., 2], 1.0)
    assert scaled_source_fluctuation_rms(source, scales) > 0.0
    assert normalized_scaled_source_error(source, source, scales) == 0.0
