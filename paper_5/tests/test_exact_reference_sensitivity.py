from __future__ import annotations

import numpy as np

from paper5.stability import DimerParameters
from paper5.stability.exact_reference_sensitivity import (
    exact_reference_sensitivity,
)


def test_exact_sensitivity_preserves_fidelity_and_returns_physical_moments() -> None:
    result = exact_reference_sensitivity(
        DimerParameters(
            lambda_ep=1.5,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        sample_times=np.linspace(0.0, 0.2, 5),
        phonon_cutoff=2,
        perturbation_amplitudes=(1e-4,),
        direction_names=("drive",),
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        maximum_step=0.02,
    )

    assert result.success
    assert result.base_coordinates.shape == (5, 31)
    assert result.shadow_coordinates.shape == (1, 1, 5, 31)
    assert np.max(np.abs(result.basis_norms - 1.0)) < 2e-10
    fidelity = result.state_fidelities[0, 0]
    assert np.max(np.abs(fidelity - fidelity[0])) < 2e-12
    assert np.min(result.coordinate_frobenius_distances) > 0.0
    assert np.min(result.base_margins) > -2e-12
    assert np.min(result.shadow_margins) > -2e-12
    assert np.max(result.base_trace_residuals) < 2e-12
    assert np.max(result.shadow_trace_residuals) < 2e-12


def test_contracted_distance_is_linear_in_small_wavefunction_amplitude() -> None:
    result = exact_reference_sensitivity(
        DimerParameters(
            lambda_ep=1.5,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        sample_times=np.array([0.0, 0.05]),
        phonon_cutoff=2,
        perturbation_amplitudes=(5e-5, 1e-4),
        direction_names=("relative_phonon_position",),
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        maximum_step=0.02,
    )

    distance_ratio = (
        result.coordinate_frobenius_distances[0, 1]
        / result.coordinate_frobenius_distances[0, 0]
    )
    np.testing.assert_allclose(distance_ratio, 2.0, rtol=3e-4, atol=0.0)
