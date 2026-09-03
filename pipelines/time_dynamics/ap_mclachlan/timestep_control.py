"""Shared component-bounded Euler timestep control for variational dynamics.

The control component may be the raw parameter velocity or a local coordinate
derived from the already measured McLachlan metric.  In every case the state
update remains ``theta <- theta + dt * theta_dot``; the component rate only
chooses ``dt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.inverse import SupportedInverse


COMPONENT_CONTROLLED_EULER_V1 = "component_controlled_euler_v1"
RETAINED_METRIC_EIGENMODE_STEPPING_V1 = "retained_metric_eigenmode_controlled_euler_v1"


@dataclass(frozen=True)
class ControlledVelocity:
    """One physical parameter velocity and its scalar timestep-control rate."""

    theta_dot: np.ndarray
    control_rate: float
    retained_metric_rank: int | None = None
    raw_metric_threshold: float | None = None


@dataclass(frozen=True)
class ComponentControlledSteppingResult:
    """Outcome of one reporting interval under component-bounded Euler steps."""

    theta_next: np.ndarray
    substep_count: int
    min_dt: float
    max_dt: float
    max_theta_dot: float
    max_control_rate: float
    step_max: float
    policy: str = COMPONENT_CONTROLLED_EULER_V1
    min_retained_metric_rank: int | None = None
    max_retained_metric_rank: int | None = None
    min_raw_metric_threshold: float | None = None
    max_raw_metric_threshold: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "policy": str(self.policy),
            "substep_count": int(self.substep_count),
            "rhs_evaluation_count": int(self.substep_count),
            "min_dt": float(self.min_dt),
            "max_dt": float(self.max_dt),
            "max_theta_dot": float(self.max_theta_dot),
            "max_control_rate": float(self.max_control_rate),
            "step_max": float(self.step_max),
            "min_retained_metric_rank": self.min_retained_metric_rank,
            "max_retained_metric_rank": self.max_retained_metric_rank,
            "min_raw_metric_threshold": self.min_raw_metric_threshold,
            "max_raw_metric_threshold": self.max_raw_metric_threshold,
        }


def retained_metric_eigenmode_velocity(
    *,
    theta_dot: np.ndarray,
    supported_inverse: SupportedInverse,
    epsilon: float,
) -> ControlledVelocity:
    r"""Return the local metric-whitened retained-eigenmode control rate.

    For a raw Gram eigensystem ``G = V diag(lambda) V^T``, the local whitened
    velocity is ``dot y = sqrt(lambda) V^T dot theta``.  The supported inverse
    diagonalizes ``G + ridge * I``; isotropic ridge leaves ``V`` unchanged, so
    the physical eigenvalues are recovered by subtracting the ridge.  Modes
    must be retained by the declared solve and positive above the raw-metric
    cutoff.  The infinity norm of ``dot y`` controls the step.

    This is sign-invariant but, like every eigenmode infinity norm, depends on
    the chosen basis inside an exactly degenerate eigenspace.
    """

    if not np.isfinite(float(epsilon)) or float(epsilon) < 0.0:
        raise ValueError("epsilon must be finite and non-negative.")
    velocity = np.asarray(theta_dot, dtype=float).reshape(-1)
    eigenvalues = np.asarray(supported_inverse.eigenvalues, dtype=float).reshape(-1)
    retained = np.asarray(supported_inverse.retained, dtype=bool).reshape(-1)
    vectors = np.asarray(supported_inverse.eigenvectors, dtype=float)
    n = int(velocity.size)
    if eigenvalues.shape != (n,) or retained.shape != (n,) or vectors.shape != (n, n):
        raise ValueError(
            "supported inverse eigensystem must match theta_dot; "
            f"got theta_dot {(n,)}, eigenvalues {eigenvalues.shape}, "
            f"retained {retained.shape}, eigenvectors {vectors.shape}."
        )
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(vectors)):
        raise ValueError("theta_dot and supported inverse eigenvectors must be finite.")
    raw_eigenvalues = eigenvalues - float(supported_inverse.ridge_lambda)
    if not np.all(np.isfinite(raw_eigenvalues)):
        raise ValueError("raw metric eigenvalues must be finite.")
    raw_scale = float(np.max(np.abs(raw_eigenvalues))) if raw_eigenvalues.size else 0.0
    threshold = max(float(epsilon), float(supported_inverse.pinv_rcond) * raw_scale)
    positive_retained = retained & (raw_eigenvalues > threshold)
    rank = int(np.count_nonzero(positive_retained))
    if rank == 0:
        control_rate = 0.0
    else:
        coefficients = vectors[:, positive_retained].T @ velocity
        whitened_velocity = np.sqrt(raw_eigenvalues[positive_retained]) * coefficients
        control_rate = float(np.max(np.abs(whitened_velocity)))
    if not np.isfinite(control_rate):
        raise ValueError("retained metric eigenmode velocity must be finite.")
    return ControlledVelocity(
        theta_dot=velocity,
        control_rate=control_rate,
        retained_metric_rank=rank,
        raw_metric_threshold=float(threshold),
    )


def advance_interval_component_controlled(
    *,
    theta: np.ndarray,
    time: float,
    dt_interval: float,
    controlled_rhs: Callable[[np.ndarray, float], ControlledVelocity],
    step_max: float,
    policy: str = COMPONENT_CONTROLLED_EULER_V1,
    max_substeps: int = 100000,
    runaway_label: str = "component-controlled stepping",
) -> ComponentControlledSteppingResult:
    """Advance one interval with ``dt * control_rate <= step_max``."""

    if not np.isfinite(float(step_max)) or float(step_max) <= 0.0:
        raise ValueError("step_max must be finite and positive.")
    if not np.isfinite(float(dt_interval)) or float(dt_interval) < 0.0:
        raise ValueError("dt_interval must be finite and non-negative.")
    if int(max_substeps) < 1:
        raise ValueError("max_substeps must be positive.")

    theta_current = np.asarray(theta, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta_current)):
        raise ValueError("theta must contain only finite values.")
    t_current = float(time)
    t_end = float(time) + float(dt_interval)
    if not np.isfinite(t_current) or not np.isfinite(t_end):
        raise ValueError("time and interval endpoint must be finite.")
    substeps = 0
    min_dt = float("inf")
    max_dt = 0.0
    max_theta_dot_seen = 0.0
    max_control_rate_seen = 0.0
    metric_ranks: list[int] = []
    raw_thresholds: list[float] = []

    while t_current < t_end - 1.0e-15:
        if substeps >= int(max_substeps):
            raise RuntimeError(
                f"{runaway_label} exceeded {max_substeps} substeps over one "
                f"reporting interval (step_max={step_max:g}); refusing to "
                "report a partially advanced interval."
            )
        controlled = controlled_rhs(theta_current, t_current)
        theta_dot = np.asarray(controlled.theta_dot, dtype=float).reshape(-1)
        if theta_dot.shape != theta_current.shape:
            raise ValueError(
                "controlled RHS theta_dot must match theta; "
                f"got {theta_dot.shape} and {theta_current.shape}."
            )
        if not np.all(np.isfinite(theta_dot)):
            raise ValueError("controlled RHS theta_dot must be finite.")
        rate = float(controlled.control_rate)
        if not np.isfinite(rate) or rate < 0.0:
            raise ValueError("controlled RHS rate must be finite and non-negative.")
        theta_peak = float(np.max(np.abs(theta_dot))) if theta_dot.size else 0.0
        max_theta_dot_seen = max(max_theta_dot_seen, theta_peak)
        max_control_rate_seen = max(max_control_rate_seen, rate)
        if controlled.retained_metric_rank is not None:
            metric_ranks.append(int(controlled.retained_metric_rank))
        if controlled.raw_metric_threshold is not None:
            raw_thresholds.append(float(controlled.raw_metric_threshold))

        remaining = t_end - t_current
        dt_allowed = (
            remaining if rate <= 0.0 else min(remaining, float(step_max) / rate)
        )
        if not np.isfinite(dt_allowed) or dt_allowed <= 0.0:
            raise RuntimeError(
                "component-controlled stepping produced an invalid step."
            )
        theta_current = theta_current + dt_allowed * theta_dot
        t_current += dt_allowed
        substeps += 1
        min_dt = min(min_dt, dt_allowed)
        max_dt = max(max_dt, dt_allowed)

    return ComponentControlledSteppingResult(
        theta_next=theta_current,
        substep_count=int(substeps),
        min_dt=0.0 if not np.isfinite(min_dt) else float(min_dt),
        max_dt=float(max_dt),
        max_theta_dot=float(max_theta_dot_seen),
        max_control_rate=float(max_control_rate_seen),
        step_max=float(step_max),
        policy=str(policy),
        min_retained_metric_rank=min(metric_ranks) if metric_ranks else None,
        max_retained_metric_rank=max(metric_ranks) if metric_ranks else None,
        min_raw_metric_threshold=min(raw_thresholds) if raw_thresholds else None,
        max_raw_metric_threshold=max(raw_thresholds) if raw_thresholds else None,
    )


__all__ = [
    "COMPONENT_CONTROLLED_EULER_V1",
    "RETAINED_METRIC_EIGENMODE_STEPPING_V1",
    "ComponentControlledSteppingResult",
    "ControlledVelocity",
    "advance_interval_component_controlled",
    "retained_metric_eigenmode_velocity",
]
