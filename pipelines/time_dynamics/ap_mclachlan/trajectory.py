"""Fixed-support AP-McLachlan trajectory propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.fixed_step import (
    FixedMcLachlanStep,
    solve_fixed_mclachlan_step,
)
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    GeometryEvaluation,
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.integrators import (
    INTEGRATOR_EULER,
    IntegrationStep,
    integrate_theta_step,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.state import APMcLachlanState


FIXED_TRAJECTORY_SCHEMA_V1 = "ap_mclachlan_fixed_trajectory_v1"


@dataclass(frozen=True)
class FixedTrajectoryPoint:
    """One recorded fixed-support AP-McLachlan time point."""

    index: int
    time: float
    theta_runtime: np.ndarray
    energy_expectation: float
    geometry: GeometryEvaluation
    fixed_step: FixedMcLachlanStep
    integration_to_next: IntegrationStep | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "time": float(self.time),
            "theta_runtime": [float(x) for x in np.asarray(self.theta_runtime, dtype=float).reshape(-1)],
            "energy_expectation": float(self.energy_expectation),
            "fixed_step": self.fixed_step.to_json_dict(),
            "integration_to_next": (
                None
                if self.integration_to_next is None
                else self.integration_to_next.to_json_dict()
            ),
        }


@dataclass(frozen=True)
class FixedMclachlanTrajectory:
    """Fixed-support AP-McLachlan trajectory output."""

    points: tuple[FixedTrajectoryPoint, ...]
    integrator_method: str
    inverse_policy: McLachlanInversePolicy
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def final_theta_runtime(self) -> np.ndarray:
        if not self.points:
            return np.zeros(0, dtype=float)
        return np.asarray(self.points[-1].theta_runtime, dtype=float).reshape(-1)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": FIXED_TRAJECTORY_SCHEMA_V1,
            "integrator_method": str(self.integrator_method),
            "inverse_policy_id": str(self.inverse_policy.policy_id),
            "pinv_rcond": float(self.inverse_policy.pinv_rcond),
            "ridge_lambda": float(self.inverse_policy.ridge_lambda),
            "point_count": int(len(self.points)),
            "points": [point.to_json_dict() for point in self.points],
            "metadata": _json_safe(dict(self.metadata or {})),
        }


def run_fixed_mclachlan_trajectory(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    times: Sequence[float],
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
    integrator_method: str = INTEGRATOR_EULER,
    metadata: Mapping[str, Any] | None = None,
) -> FixedMclachlanTrajectory:
    """Propagate a fixed AP-McLachlan support over a time grid."""

    time_grid = _time_grid(times)
    theta_current = np.asarray(state.theta_runtime, dtype=float).reshape(-1)
    points: list[FixedTrajectoryPoint] = []

    def theta_dot_rhs(theta_value: np.ndarray, time_value: float) -> np.ndarray:
        evaluation = evaluate_mclachlan_geometry(
            state=state,
            hamiltonian=hamiltonian,
            theta_runtime=np.asarray(theta_value, dtype=float).reshape(-1),
            time=float(time_value),
        )
        step = solve_fixed_mclachlan_step(
            evaluation.geometry,
            inverse_policy=inverse_policy,
        )
        return np.asarray(step.theta_dot, dtype=float).reshape(-1)

    for index, time_value in enumerate(time_grid):
        evaluation = evaluate_mclachlan_geometry(
            state=state,
            hamiltonian=hamiltonian,
            theta_runtime=theta_current,
            time=float(time_value),
        )
        fixed_step = solve_fixed_mclachlan_step(
            evaluation.geometry,
            inverse_policy=inverse_policy,
        )
        integration: IntegrationStep | None = None
        if index + 1 < len(time_grid):
            dt = float(time_grid[index + 1] - time_value)
            integration = integrate_theta_step(
                theta=theta_current,
                t=float(time_value),
                dt=dt,
                rhs=theta_dot_rhs,
                method=str(integrator_method),
            )
            theta_next = np.asarray(integration.theta_next, dtype=float).reshape(-1)
        else:
            theta_next = theta_current
        points.append(
            FixedTrajectoryPoint(
                index=int(index),
                time=float(time_value),
                theta_runtime=np.asarray(theta_current, dtype=float).reshape(-1),
                energy_expectation=float(evaluation.energy_expectation),
                geometry=evaluation,
                fixed_step=fixed_step,
                integration_to_next=integration,
            )
        )
        theta_current = np.asarray(theta_next, dtype=float).reshape(-1)

    return FixedMclachlanTrajectory(
        points=tuple(points),
        integrator_method=str(integrator_method).lower(),
        inverse_policy=inverse_policy,
        metadata={
            "trajectory_kind": "fixed_support",
            "uses_reference_for_decision": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "uses_statevector_as_ideal_observable_estimator": True,
            **dict(metadata or {}),
        },
    )


def _time_grid(times: Sequence[float]) -> np.ndarray:
    grid = np.asarray(times, dtype=float).reshape(-1)
    if int(grid.size) == 0:
        raise ValueError("times must contain at least one time point.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("times must contain only finite values.")
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("times must be monotonically nondecreasing.")
    return grid


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
    "FIXED_TRAJECTORY_SCHEMA_V1",
    "FixedMclachlanTrajectory",
    "FixedTrajectoryPoint",
    "run_fixed_mclachlan_trajectory",
]
