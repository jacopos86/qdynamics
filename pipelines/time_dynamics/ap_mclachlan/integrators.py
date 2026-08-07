"""ODE integration methods for AP-McLachlan parameter propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


INTEGRATOR_EULER = "euler"
INTEGRATOR_RK4 = "rk4"
SUPPORTED_INTEGRATORS: tuple[str, ...] = (INTEGRATOR_EULER, INTEGRATOR_RK4)

RhsFunction = Callable[[np.ndarray, float], np.ndarray | Sequence[float]]


@dataclass(frozen=True)
class IntegrationStep:
    """One parameter-integration step for ``d theta / dt = rhs(theta,t)``."""

    theta_next: np.ndarray
    theta_dot: np.ndarray
    method: str
    t: float
    dt: float
    rhs_evaluation_count: int
    local_subdivision_applied: bool = False
    local_subdivision_depth: int = 0
    local_substep_count: int = 1
    local_subdivision_reason: str | None = None
    repair_summary: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "integrator_method": str(self.method),
            "t": float(self.t),
            "dt": float(self.dt),
            "rhs_evaluation_count": int(self.rhs_evaluation_count),
            "theta_dot": [float(x) for x in self.theta_dot.tolist()],
            "theta_next": [float(x) for x in self.theta_next.tolist()],
            "local_subdivision_applied": bool(self.local_subdivision_applied),
            "local_subdivision_depth": int(self.local_subdivision_depth),
            "local_substep_count": int(self.local_substep_count),
            "local_subdivision_reason": (
                None if self.local_subdivision_reason is None else str(self.local_subdivision_reason)
            ),
            "repair_summary": _json_safe(dict(self.repair_summary or {})),
        }


def _theta(value: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _rhs(rhs: RhsFunction, theta: np.ndarray, t: float) -> np.ndarray:
    value = _theta(rhs(np.asarray(theta, dtype=float), float(t)), name="rhs(theta,t)")
    if value.shape != theta.shape:
        raise ValueError(f"rhs shape {value.shape} does not match theta shape {theta.shape}.")
    return value


def integrate_theta_step(
    *,
    theta: np.ndarray | Sequence[float],
    t: float,
    dt: float,
    rhs: RhsFunction,
    method: str = INTEGRATOR_EULER,
) -> IntegrationStep:
    """Advance one ODE step with an explicit integration method."""

    theta0 = _theta(theta, name="theta")
    h = float(dt)
    if not np.isfinite(h):
        raise ValueError("dt must be finite.")
    method_id = str(method).lower()
    if method_id == INTEGRATOR_EULER:
        k1 = _rhs(rhs, theta0, float(t))
        return IntegrationStep(
            theta_next=np.asarray(theta0 + h * k1, dtype=float),
            theta_dot=k1,
            method=INTEGRATOR_EULER,
            t=float(t),
            dt=h,
            rhs_evaluation_count=1,
        )
    if method_id == INTEGRATOR_RK4:
        k1 = _rhs(rhs, theta0, float(t))
        k2 = _rhs(rhs, theta0 + 0.5 * h * k1, float(t) + 0.5 * h)
        k3 = _rhs(rhs, theta0 + 0.5 * h * k2, float(t) + 0.5 * h)
        k4 = _rhs(rhs, theta0 + h * k3, float(t) + h)
        theta_dot = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return IntegrationStep(
            theta_next=np.asarray(theta0 + h * theta_dot, dtype=float),
            theta_dot=np.asarray(theta_dot, dtype=float),
            method=INTEGRATOR_RK4,
            t=float(t),
            dt=h,
            rhs_evaluation_count=4,
        )
    raise ValueError(f"Unsupported integrator method {method!r}; expected one of {SUPPORTED_INTEGRATORS}.")


def integration_step_with_metadata(
    step: IntegrationStep,
    *,
    local_subdivision_applied: bool,
    local_subdivision_depth: int,
    local_substep_count: int,
    local_subdivision_reason: str | None,
    repair_summary: Mapping[str, Any] | None = None,
) -> IntegrationStep:
    """Return ``step`` with subdivision metadata attached."""

    return IntegrationStep(
        theta_next=np.asarray(step.theta_next, dtype=float).reshape(-1),
        theta_dot=np.asarray(step.theta_dot, dtype=float).reshape(-1),
        method=str(step.method),
        t=float(step.t),
        dt=float(step.dt),
        rhs_evaluation_count=int(step.rhs_evaluation_count),
        local_subdivision_applied=bool(local_subdivision_applied),
        local_subdivision_depth=int(local_subdivision_depth),
        local_substep_count=int(local_substep_count),
        local_subdivision_reason=local_subdivision_reason,
        repair_summary=dict(repair_summary or {}),
    )


def aggregate_integration_substeps(
    *,
    theta_start: np.ndarray | Sequence[float],
    substeps: Sequence[IntegrationStep],
    t: float,
    dt: float,
    method: str,
    depth: int,
    reason: str,
    repair_summary: Mapping[str, Any] | None = None,
) -> IntegrationStep:
    """Collapse a sequence of local substeps into one interval record."""

    steps = tuple(substeps)
    if not steps:
        raise ValueError("substeps must contain at least one IntegrationStep.")
    theta0 = _theta(theta_start, name="theta_start")
    theta_next = np.asarray(steps[-1].theta_next, dtype=float).reshape(-1)
    h = float(dt)
    theta_dot = np.zeros_like(theta_next, dtype=float) if h == 0.0 else (theta_next - theta0) / h
    return IntegrationStep(
        theta_next=theta_next,
        theta_dot=np.asarray(theta_dot, dtype=float).reshape(-1),
        method=str(method).lower(),
        t=float(t),
        dt=h,
        rhs_evaluation_count=int(sum(int(step.rhs_evaluation_count) for step in steps)),
        local_subdivision_applied=True,
        local_subdivision_depth=int(depth),
        local_substep_count=int(len(steps)),
        local_subdivision_reason=str(reason),
        repair_summary=dict(repair_summary or {}),
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


__all__ = [
    "INTEGRATOR_EULER",
    "INTEGRATOR_RK4",
    "SUPPORTED_INTEGRATORS",
    "IntegrationStep",
    "RhsFunction",
    "aggregate_integration_substeps",
    "integrate_theta_step",
    "integration_step_with_metadata",
]
