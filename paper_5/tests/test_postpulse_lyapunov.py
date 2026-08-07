from __future__ import annotations

import numpy as np

from paper5.stability import (
    DimerParameters,
    closed_state_correction_energy_gradient,
    closed_state_lifted_frobenius_norm,
    exact_ground_closed_scalar_coordinates,
)
from paper5.stability.postpulse_lyapunov import (
    postpulse_lyapunov_estimate,
    tangent_projected_direction,
)


def test_tangent_projection_enforces_equalities_and_normalization() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=0.0,
    )
    base = exact_ground_closed_scalar_coordinates(parameters, phonon_cutoff=2)
    direction = np.random.default_rng(7).normal(size=31)
    projected = tangent_projected_direction(direction, base, parameters)

    assert abs(closed_state_correction_energy_gradient(base, parameters) @ projected) < 2e-11
    assert abs(projected[17]) < 2e-11
    assert abs(projected[18]) < 2e-11
    assert abs(closed_state_lifted_frobenius_norm(projected) - 1.0) < 2e-14


def test_short_postpulse_estimate_stays_physical_and_records_directions() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=0.0,
    )
    base = exact_ground_closed_scalar_coordinates(parameters, phonon_cutoff=2)
    directions = np.random.default_rng(11).normal(size=(2, 31))
    estimate = postpulse_lyapunov_estimate(
        parameters,
        base,
        directions,
        labels=("a", "b"),
        initial_time=4.0,
        final_time=4.04,
        time_step=0.01,
        renormalization_interval=0.02,
        perturbation_size=1e-5,
    )

    assert estimate.local_exponents.shape == (2, 2)
    assert estimate.cumulative_exponents.shape == (2, 2)
    assert np.all(np.isfinite(estimate.local_exponents))
    assert np.all(estimate.projection_retention > 0.0)
    assert np.all(estimate.projection_retention <= 1.0 + 1e-12)
    assert np.min(estimate.base_margins) > -1e-8
    assert np.min(estimate.shadow_margins) > -1e-8
    assert np.max(estimate.base_trace_residuals) < 2e-12
    assert np.max(estimate.shadow_trace_residuals) < 2e-12
