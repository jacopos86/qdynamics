"""Autonomous propagation seam for the reduced latent ``C`` source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .latent_source_closure import (
    LatentSourceBasis,
    StableSecondOrderLatentSourceEvolutionModel,
    predict_stable_second_order_latent_source_evolution,
    reconstruct_missing_source,
)


Vector = np.ndarray
Rhs = Callable[[float, Vector], Vector]
MomentRhs = Callable[[float, Vector], Vector]
DriveDifference = Callable[[float], float]
MomentCorrection = Callable[[float, Vector, Vector], Vector]


@dataclass(frozen=True)
class LatentAugmentedTrajectory:
    """Sampled states from fixed-step RK4 propagation."""

    times: np.ndarray
    states: np.ndarray
    rhs_evaluations: int


def latent_augmented_velocity(
    time: float,
    state: Vector,
    *,
    moment_rhs: MomentRhs,
    drive_difference: DriveDifference,
    basis: LatentSourceBasis,
    model: StableSecondOrderLatentSourceEvolutionModel,
    moment_correction: MomentCorrection | None = None,
) -> Vector:
    """Evaluate the autonomous 31-moment plus latent-source velocity."""

    values = np.asarray(state, dtype=float)
    rank = basis.rank
    expected_shape = (31 + 2 * rank,)
    if values.shape != expected_shape:
        raise ValueError(
            f"expected augmented state shape {expected_shape}, got {values.shape}"
        )
    if model.source_coefficients.shape != (rank, rank):
        raise ValueError("latent basis and evolution model ranks do not match")

    moments = values[:31]
    latent_source = values[31 : 31 + rank]
    latent_rate = values[31 + rank :]
    proposed_moment_velocity = np.asarray(
        moment_rhs(float(time), moments),
        dtype=float,
    ).copy()
    if proposed_moment_velocity.shape != (31,):
        raise ValueError("moment_rhs must return shape (31,)")
    proposed_moment_velocity[17:31] += reconstruct_missing_source(
        latent_source,
        basis,
    )

    if moment_correction is None:
        correction = np.zeros(31, dtype=float)
    else:
        correction = np.asarray(
            moment_correction(
                float(time),
                moments,
                proposed_moment_velocity,
            ),
            dtype=float,
        )
        if correction.shape != (31,):
            raise ValueError("moment_correction must return shape (31,)")

    latent_velocity = predict_stable_second_order_latent_source_evolution(
        model,
        moments[None, :],
        latent_source[None, :],
        latent_rate[None, :],
        np.asarray([drive_difference(float(time))], dtype=float),
    )[0]
    result = np.concatenate(
        (proposed_moment_velocity + correction, latent_velocity)
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("latent augmented velocity is non-finite")
    return result


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    ratio = numerator / denominator
    rounded = int(round(ratio))
    if rounded < 1 or not np.isclose(ratio, rounded, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be a positive integer multiple")
    return rounded


def integrate_latent_augmented_rk4(
    rhs: Rhs,
    initial_state: Vector,
    *,
    final_time: float,
    time_step: float,
    sample_step: float,
) -> LatentAugmentedTrajectory:
    """Propagate an autonomous augmented state and sample a fixed time grid."""

    if final_time <= 0.0 or time_step <= 0.0 or sample_step <= 0.0:
        raise ValueError("final_time, time_step, and sample_step must be positive")
    total_steps = _integer_ratio(final_time, time_step, "final_time")
    sample_stride = _integer_ratio(sample_step, time_step, "sample_step")
    if total_steps % sample_stride != 0:
        raise ValueError("final_time must be an integer multiple of sample_step")

    state = np.asarray(initial_state, dtype=float).copy()
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must be a finite vector")
    times = [0.0]
    states = [state.copy()]
    rhs_evaluations = 0
    half_step = 0.5 * time_step
    for step_index in range(total_steps):
        time = step_index * time_step
        k1 = np.asarray(rhs(time, state), dtype=float)
        k2 = np.asarray(rhs(time + half_step, state + half_step * k1), dtype=float)
        k3 = np.asarray(rhs(time + half_step, state + half_step * k2), dtype=float)
        k4 = np.asarray(rhs(time + time_step, state + time_step * k3), dtype=float)
        rhs_evaluations += 4
        for stage in (k1, k2, k3, k4):
            if stage.shape != state.shape:
                raise ValueError("rhs must preserve the state shape")
        state = state + (time_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(state)):
            raise RuntimeError(f"non-finite state reached at t={time + time_step}")
        if (step_index + 1) % sample_stride == 0:
            times.append((step_index + 1) * time_step)
            states.append(state.copy())

    return LatentAugmentedTrajectory(
        times=np.asarray(times, dtype=float),
        states=np.asarray(states, dtype=float),
        rhs_evaluations=rhs_evaluations,
    )
