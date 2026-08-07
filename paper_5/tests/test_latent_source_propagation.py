from __future__ import annotations

import numpy as np

from paper5.stability.latent_source_closure import (
    LatentSourceBasis,
    StableSecondOrderLatentSourceEvolutionModel,
)
from paper5.stability.latent_source_propagation import (
    integrate_latent_augmented_rk4,
    latent_augmented_velocity,
)


def _basis() -> LatentSourceBasis:
    basis = np.zeros((2, 14), dtype=float)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    return LatentSourceBasis(
        center=np.linspace(0.1, 1.4, 14),
        basis=basis,
        coordinate_scales=np.linspace(1.0, 2.3, 14),
        singular_values=np.ones(2),
    )


def _model() -> StableSecondOrderLatentSourceEvolutionModel:
    return StableSecondOrderLatentSourceEvolutionModel(
        acceleration_intercept=np.array([0.7, -0.2]),
        state_coefficients=np.zeros((2, 31)),
        source_coefficients=np.zeros((2, 2)),
        rate_coefficients=np.zeros((2, 2)),
        drive_coefficients=np.zeros(2),
        coordinate_scales=np.ones(31),
        ridge_penalty=1.0,
        stability_margin=0.01,
        stability_shift=0.1,
        maximum_real_part_before_shift=0.09,
    )


def test_latent_source_is_added_only_to_c_velocity_before_controller() -> None:
    moment_state = np.linspace(-0.3, 0.3, 31)
    latent_source = np.array([2.0, -3.0])
    latent_rate = np.array([0.4, -0.5])
    augmented = np.concatenate((moment_state, latent_source, latent_rate))
    base_velocity = np.linspace(1.0, 4.0, 31)
    captured: dict[str, np.ndarray] = {}

    def moment_rhs(time: float, state: np.ndarray) -> np.ndarray:
        assert time == 0.25
        np.testing.assert_allclose(state, moment_state)
        return base_velocity.copy()

    def correction(
        time: float,
        state: np.ndarray,
        proposed_velocity: np.ndarray,
    ) -> np.ndarray:
        assert time == 0.25
        np.testing.assert_allclose(state, moment_state)
        captured["proposed"] = proposed_velocity.copy()
        result = np.zeros(31)
        result[0] = 0.125
        return result

    velocity = latent_augmented_velocity(
        0.25,
        augmented,
        moment_rhs=moment_rhs,
        drive_difference=lambda _time: 0.0,
        basis=_basis(),
        model=_model(),
        moment_correction=correction,
    )

    expected_source = _basis().center.copy()
    expected_source[0] += 2.0 * _basis().coordinate_scales[0]
    expected_source[1] -= 3.0 * _basis().coordinate_scales[1]
    expected_proposed = base_velocity.copy()
    expected_proposed[17:31] += expected_source
    np.testing.assert_allclose(captured["proposed"], expected_proposed)
    np.testing.assert_allclose(velocity[:31], expected_proposed + np.eye(31)[0] * 0.125)
    np.testing.assert_allclose(velocity[31:33], latent_rate)
    np.testing.assert_allclose(velocity[33:35], np.array([0.7, -0.2]))


def test_rk4_recomputes_velocity_at_all_four_stages() -> None:
    evaluations: list[tuple[float, np.ndarray]] = []

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        evaluations.append((time, state.copy()))
        return state

    result = integrate_latent_augmented_rk4(
        rhs,
        np.array([1.0]),
        final_time=0.1,
        time_step=0.1,
        sample_step=0.1,
    )

    assert result.rhs_evaluations == 4
    np.testing.assert_allclose(
        [evaluation[0] for evaluation in evaluations],
        [0.0, 0.05, 0.05, 0.1],
    )
    np.testing.assert_allclose(evaluations[1][1], [1.05])
    np.testing.assert_allclose(evaluations[2][1], [1.0525])
    np.testing.assert_allclose(evaluations[3][1], [1.10525])
    np.testing.assert_allclose(result.times, [0.0, 0.1])
    np.testing.assert_allclose(result.states[:, 0], [1.0, 1.1051708333333332])
