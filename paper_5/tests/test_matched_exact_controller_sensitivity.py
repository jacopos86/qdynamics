from __future__ import annotations

import numpy as np

from paper5.stability import DimerParameters
from paper5.stability.exact_reference_sensitivity import (
    exact_reference_sensitivity,
)
from paper5.stability.matched_exact_controller_sensitivity import (
    matched_controller_sensitivity,
)


def test_matched_controller_uses_exact_induced_pair_without_rescaling() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    exact = exact_reference_sensitivity(
        parameters,
        sample_times=np.array([0.0, 0.02]),
        phonon_cutoff=2,
        perturbation_amplitudes=(1e-5,),
        direction_names=("drive",),
        maximum_step=0.01,
    )
    initial_base = exact.base_coordinates[0]
    initial_shadow = exact.shadow_coordinates[0, 0, 0]
    initial_distance = exact.coordinate_frobenius_distances[0, 0, 0]

    result = matched_controller_sensitivity(
        parameters,
        initial_base,
        initial_shadow[None, :],
        labels=("drive",),
        final_time=0.02,
        time_step=0.01,
        sample_step=0.01,
    )

    assert result.sampled_states.shape == (3, 2, 31)
    assert abs(result.step_frobenius_distances[0, 0] - initial_distance) < 1e-15
    assert np.all(result.converged)
    assert np.min(result.sampled_margins) > -1e-8
    assert np.max(result.sampled_trace_residuals) < 2e-12
    np.testing.assert_allclose(
        result.total_growth_contributions,
        result.raw_growth_contributions + result.correction_growth_contributions,
    )
