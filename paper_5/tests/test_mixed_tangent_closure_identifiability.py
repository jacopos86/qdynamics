from __future__ import annotations

import numpy as np

from paper5.stability.hubbard_dimer import (
    DimerParameters,
    GaussianSineDrive,
)
from paper5.stability.mixed_tangent_closure_identifiability import (
    mixed_tangent_closure_point,
)
from paper5.stability.multi_coherent import pack_multi_coherent_parameters


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


def test_mixed_complement_exactly_factors_same_state_c_velocity() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    point = mixed_tangent_closure_point(
        _symmetric_packet_parameters(),
        time=1.25,
        parameters=parameters,
        drive_protocol=GaussianSineDrive.from_parameters(parameters),
        relative_dimension=9,
    )

    assert point.mixed_coefficients.shape == (12,)
    assert point.mixed_response.shape == (14, 12)
    assert point.mixed_complement_rank == 12
    np.testing.assert_allclose(
        point.archive_frame_source
        + point.mixed_response @ point.mixed_coefficients,
        point.target_source,
        atol=2e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(point.unresolved_source, 0.0, atol=2e-13)
    np.testing.assert_allclose(
        point.enriched_correlation_velocity,
        point.target_correlation_velocity,
        atol=2e-13,
        rtol=0.0,
    )
