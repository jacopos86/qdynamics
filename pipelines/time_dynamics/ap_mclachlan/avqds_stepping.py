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

    if float(delta_theta_max) <= 0.0:
        raise ValueError("delta_theta_max must be positive.")
    if float(dt_interval) < 0.0:
        raise ValueError("dt_interval must be non-negative.")

    theta_current = np.asarray(theta, dtype=float).reshape(-1)
    t_current = float(time)
    t_end = float(time) + float(dt_interval)
    substeps = 0
    min_dt = float("inf")
    max_dt = 0.0
    max_theta_dot_seen = 0.0

    while t_current < t_end - 1.0e-15:
        if substeps >= int(max_substeps):
            raise RuntimeError(
                f"AVQDS stepping exceeded {max_substeps} substeps over one "
                f"reporting interval (delta_theta_max={delta_theta_max:g}); "
                "refusing to report a partially advanced interval."
            )
        theta_dot = np.asarray(
            theta_dot_rhs(theta_current, t_current), dtype=float
        ).reshape(-1)
        peak = float(np.max(np.abs(theta_dot))) if theta_dot.size else 0.0
        max_theta_dot_seen = max(max_theta_dot_seen, peak)
        remaining = t_end - t_current
        # The control law: no parameter may move more than delta_theta_max.
        dt_allowed = (
            remaining if peak <= 0.0
            else min(remaining, float(delta_theta_max) / peak)
        )
        if not np.isfinite(dt_allowed) or dt_allowed <= 0.0:
            dt_allowed = remaining
        theta_current = theta_current + dt_allowed * theta_dot
        t_current += dt_allowed
        substeps += 1
        min_dt = min(min_dt, dt_allowed)
        max_dt = max(max_dt, dt_allowed)

    return AVQDSSteppingResult(
        theta_next=theta_current,
        substep_count=int(substeps),
        min_dt=0.0 if not np.isfinite(min_dt) else float(min_dt),
        max_dt=float(max_dt),
        max_theta_dot=float(max_theta_dot_seen),
        delta_theta_max=float(delta_theta_max),
    )


__all__ = [
    "AVQDS_STEPPING_V1",
    "DEFAULT_DELTA_THETA_MAX",
    "AVQDSSteppingResult",
    "advance_interval_delta_theta_controlled",
]
