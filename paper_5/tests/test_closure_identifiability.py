from __future__ import annotations

import numpy as np
import pytest

from paper5.stability import (
    DimerParameters,
    closed_scalar_rhs,
    closure_identifiability_witness,
)


@pytest.mark.parametrize("phonon_cutoff", (2, 3))
def test_physical_states_with_one_retained_state_have_distinct_c_rates(
    phonon_cutoff: int,
) -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    witness = closure_identifiability_witness(
        parameters,
        phonon_cutoff=phonon_cutoff,
        reference_identity_weight=0.1,
    )

    for density in (witness.density_plus, witness.density_minus):
        np.testing.assert_allclose(density, density.conjugate().T, atol=1e-14)
        assert abs(np.trace(density) - 1.0) < 1e-13
        assert np.min(np.linalg.eigvalsh(density)) > 0.0

    assert witness.minimum_density_eigenvalue > 0.0
    assert witness.minimum_joint_gram_eigenvalue > 0.0
    assert witness.spin_swap_residual < 1e-12
    assert witness.maximum_constraint_overlap < 1e-12
    assert witness.maximum_coordinate_difference < 1e-12
    assert witness.lower_derivative_difference_norm < 1e-12
    assert witness.correlation_derivative_difference_norm > 1e-2
    assert witness.maximum_correlation_derivative_difference > 1e-2
    assert witness.target_relative_residual > 0.5

    np.testing.assert_allclose(
        closed_scalar_rhs(
            witness.time,
            witness.coordinates_plus,
            parameters,
        ),
        closed_scalar_rhs(
            witness.time,
            witness.coordinates_minus,
            parameters,
        ),
        atol=1e-12,
        rtol=0.0,
    )


def test_identifiability_witness_rejects_invalid_interior_weight() -> None:
    parameters = DimerParameters()
    with pytest.raises(ValueError, match="reference_identity_weight"):
        closure_identifiability_witness(
            parameters,
            phonon_cutoff=1,
            reference_identity_weight=0.0,
        )
