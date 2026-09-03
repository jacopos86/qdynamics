"""AVQDS parameter-controlled adaptive stepping (Yao et al., PRX Quantum 2, 030307).

AVQDS does not integrate on a fixed step.  The published method adjusts the
step at every step so that no variational parameter moves more than a fixed
budget::

    the time step size dt is dynamically adjusted at each time step such that
    max(|d theta_mu|) <= d theta_max

and the paper calls this out as its stabilization mechanism -- "an effective
way to stabilize AVQDS simulations by fixing a maximal step size d theta_max"
-- with ``d theta_max = 5e-3`` in the reported comparisons.

This matters for a comparison against this route.  Running AVQDS at a fixed
reporting step is not a faithful reproduction; measured on the driven
Hubbard--Holstein seed at a fixed dt of 0.04, it produced mean energy errors
between 3.5e-1 and 2.6e0, two to four orders of magnitude worse than any other
policy -- an artifact of the imposed step, not a property of the method.

It also corrects the relationship between the two methods' stabilization
layers: AVQDS bounds *parameter* motion per step, this route bounds *state*
motion per step and subdivides.  These are variants of one idea, not an
addition on one side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.timestep_control import (
    ControlledVelocity,
    advance_interval_component_controlled,
)

AVQDS_STEPPING_V1 = "avqds_delta_theta_controlled_euler_v1"

# The value used in the source's reported comparisons.
DEFAULT_DELTA_THETA_MAX = 5.0e-3


@dataclass(frozen=True)
class AVQDSSteppingResult:
    """Outcome of advancing one reporting interval under the AVQDS rule."""

    theta_next: np.ndarray
    substep_count: int
    min_dt: float
    max_dt: float
    max_theta_dot: float
    delta_theta_max: float
    policy: str = AVQDS_STEPPING_V1

    def to_json_dict(self) -> dict[str, object]:
        return {
            "policy": str(self.policy),
            "substep_count": int(self.substep_count),
            "min_dt": float(self.min_dt),
            "max_dt": float(self.max_dt),
            "max_theta_dot": float(self.max_theta_dot),
            "delta_theta_max": float(self.delta_theta_max),
        }


def advance_interval_delta_theta_controlled(
    *,
    theta: np.ndarray,
    time: float,
    dt_interval: float,
    theta_dot_rhs: Callable[[np.ndarray, float], np.ndarray],
    delta_theta_max: float = DEFAULT_DELTA_THETA_MAX,
    max_substeps: int = 100000,
) -> AVQDSSteppingResult:
    """Advance one reporting interval with Euler steps bounded by ``delta_theta_max``.

    Each substep takes ``dt = min(remaining, delta_theta_max / max|theta_dot|)``,
    which is the published control law.  Euler is the published integrator: the
    source notes that Runge--Kutta would scale more favourably and states "we
    adopt the Euler method".

    ``max_substeps`` is a runaway guard, not part of the rule; exhausting it
    raises rather than silently returning a partially advanced interval, since
    a short interval reported as a full one is a wrong trajectory with no
    outward sign.
    """

    if not np.isfinite(float(delta_theta_max)) or float(delta_theta_max) <= 0.0:
        raise ValueError("delta_theta_max must be finite and positive.")

    def _controlled_rhs(theta_value: np.ndarray, t_value: float) -> ControlledVelocity:
        theta_dot = np.asarray(theta_dot_rhs(theta_value, t_value), dtype=float).reshape(-1)
        peak = float(np.max(np.abs(theta_dot))) if theta_dot.size else 0.0
        return ControlledVelocity(theta_dot=theta_dot, control_rate=peak)

    stepped = advance_interval_component_controlled(
        theta=theta,
        time=time,
        dt_interval=dt_interval,
        controlled_rhs=_controlled_rhs,
        step_max=float(delta_theta_max),
        policy=AVQDS_STEPPING_V1,
        max_substeps=max_substeps,
        runaway_label="AVQDS stepping",
    )
    return AVQDSSteppingResult(
        theta_next=np.asarray(stepped.theta_next, dtype=float).reshape(-1),
        substep_count=int(stepped.substep_count),
        min_dt=float(stepped.min_dt),
        max_dt=float(stepped.max_dt),
        max_theta_dot=float(stepped.max_theta_dot),
        delta_theta_max=float(delta_theta_max),
    )


__all__ = [
    "AVQDS_STEPPING_V1",
    "DEFAULT_DELTA_THETA_MAX",
    "AVQDSSteppingResult",
    "advance_interval_delta_theta_controlled",
]
