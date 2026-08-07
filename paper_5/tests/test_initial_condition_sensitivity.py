from __future__ import annotations

import numpy as np

from paper5.stability import (
    DimerParameters,
    closed_state_correction_energy_gradient,
    closed_state_lifted_frobenius_norm,
    exact_ground_closed_scalar_coordinates,
)
from paper5.stability.initial_condition_sensitivity import (
    benettin_nearby_trajectory,
    constrained_initial_direction,
    physicality_diagnostics,
)


def test_benettin_recovers_known_linear_growth_rate() -> None:
    growth_rate = 0.3

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        derivative = -0.2 * state
        derivative[0] = growth_rate * state[0]
        return derivative

    initial = np.ones(31)
    direction = np.zeros(31)
    direction[0] = 1.0

    estimate = benettin_nearby_trajectory(
        rhs,
        initial,
        direction,
        final_time=2.0,
        time_step=0.01,
        renormalization_interval=0.1,
        perturbation_size=1e-6,
        selected_metric="euclidean",
        record_physicality=False,
    )

    assert abs(estimate.cumulative_euclidean_exponents[-1] - growth_rate) < 2e-8


def test_dimer_direction_is_tangent_normalized_and_admissible() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=4,
    )
    direction = constrained_initial_direction(
        initial,
        parameters,
        seed=20260803,
        metric="frobenius",
    )

    energy_gradient = closed_state_correction_energy_gradient(
        initial,
        parameters,
    )
    assert abs(float(energy_gradient @ direction)) < 2e-12
    assert abs(direction[17]) < 2e-12
    assert abs(direction[18]) < 2e-12
    assert abs(closed_state_lifted_frobenius_norm(direction) - 1.0) < 2e-14

    margins, trace_residual = physicality_diagnostics(initial + 1e-6 * direction)
    assert np.min(margins) > 0.0
    assert trace_residual < 2e-13


def test_short_dimer_growth_records_both_norms_and_margins() -> None:
    parameters = DimerParameters(
        lambda_ep=1.5,
        gamma=0.5,
        drive_amplitude=1.0,
    )
    initial = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=4,
    )
    direction = constrained_initial_direction(
        initial,
        parameters,
        seed=7,
    )
    estimate = benettin_nearby_trajectory(
        lambda time, state: np.zeros_like(state),
        initial,
        direction,
        final_time=0.02,
        time_step=0.01,
        renormalization_interval=0.01,
        perturbation_size=1e-6,
        selected_metric="frobenius",
    )

    np.testing.assert_allclose(estimate.local_euclidean_exponents, 0.0, atol=1e-9)
    np.testing.assert_allclose(estimate.local_frobenius_exponents, 0.0, atol=1e-9)
    assert estimate.base_margins.shape == (3, 4)
    assert estimate.shadow_margins.shape == (3, 4)
    assert np.min(estimate.base_margins) > 0.0
