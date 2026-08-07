"""Array-level McLachlan geometry payloads for AP-McLachlan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    as_float_matrix,
    as_float_vector,
    solve_theta_dot,
)


GEOMETRY_SCHEMA_V1 = "ap_mclachlan_geometry_v1"


@dataclass(frozen=True)
class McLachlanGeometry:
    """Validated ``K``, ``f``, and ``V=||b||^2`` for one time point."""

    K: np.ndarray | Sequence[Sequence[float]]
    f: np.ndarray | Sequence[float]
    norm_b_sq: float
    support_indices: tuple[int, ...] | None = None
    support_labels: tuple[str, ...] = ()
    time: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        K = as_float_matrix(self.K, name="K")
        f = as_float_vector(self.f, name="f")
        if K.shape != (int(f.size), int(f.size)):
            raise ValueError(f"K must have shape ({f.size}, {f.size}); got {K.shape}.")
        norm = float(self.norm_b_sq)
        if not np.isfinite(norm) or norm < 0.0:
            raise ValueError("norm_b_sq must be finite and non-negative.")
        indices = (
            tuple(range(int(f.size)))
            if self.support_indices is None
            else tuple(int(i) for i in self.support_indices)
        )
        if len(indices) != int(f.size):
            raise ValueError("support_indices must match len(f).")
        if any(i < 0 for i in indices):
            raise ValueError("support_indices must be non-negative.")
        labels = tuple(str(label) for label in self.support_labels)
        if labels and len(labels) != int(f.size):
            raise ValueError("support_labels must be empty or match len(f).")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "norm_b_sq", norm)
        object.__setattr__(self, "support_indices", indices)
        object.__setattr__(self, "support_labels", labels)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.f, dtype=float).reshape(-1).size)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": GEOMETRY_SCHEMA_V1,
            "time": None if self.time is None else float(self.time),
            "dimension": int(self.dimension),
            "support_indices": [int(i) for i in self.support_indices or ()],
            "support_labels": [str(label) for label in self.support_labels],
            "norm_b_sq": float(self.norm_b_sq),
            "metadata": _json_safe(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class McLachlanGeometryDiagnostics:
    """Residual and conditioning diagnostics for one geometry object."""

    gamma: float
    residual_sq: float
    residual_ratio: float
    rank: int
    condition_number: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "gamma": float(self.gamma),
            "residual_sq": float(self.residual_sq),
            "residual_ratio": float(self.residual_ratio),
            "rank": int(self.rank),
            "condition_number": (
                None if self.condition_number is None else float(self.condition_number)
            ),
        }


@dataclass(frozen=True)
class StateSpaceSolveMetrics:
    """State-space diagnostics for one McLachlan velocity on a fixed tangent plane."""

    theta_dot_l2: float
    projected_velocity_sq: float
    projected_velocity_l2: float
    realized_residual_sq: float
    rho_real: float
    best_case_residual_sq: float
    rho_expr: float
    rho_num: float
    denominator: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "theta_dot_l2": float(self.theta_dot_l2),
            "projected_velocity_sq": float(self.projected_velocity_sq),
            "projected_velocity_l2": float(self.projected_velocity_l2),
            "realized_residual_sq": float(self.realized_residual_sq),
            "rho_real": float(self.rho_real),
            "best_case_residual_sq": float(self.best_case_residual_sq),
            "rho_expr": float(self.rho_expr),
            "rho_num": float(self.rho_num),
            "denominator": float(self.denominator),
        }


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


def residual_denominator(norm_b_sq: float, epsilon: float) -> float:
    denom = float(norm_b_sq) + max(0.0, float(epsilon))
    if not np.isfinite(denom) or denom <= 0.0:
        return max(float(epsilon), 1.0e-14)
    return float(denom)


def state_space_solve_metrics(
    geometry: McLachlanGeometry,
    theta_dot: np.ndarray | Sequence[float],
    *,
    best_case_residual_sq: float,
    epsilon: float,
) -> StateSpaceSolveMetrics:
    """Compute realized McLachlan residuals from ``K``, ``f``, and ``||b||^2``.

    The large tangent matrix is not needed for same-plane diagnostics because
    ``||T v||^2 = v^T K v`` and
    ``||T v - b||^2 = ||b||^2 - 2 f^T v + v^T K v``.
    """

    theta = np.asarray(theta_dot, dtype=float).reshape(-1)
    if int(theta.size) != int(geometry.dimension):
        raise ValueError(
            "theta_dot dimension must match McLachlan geometry dimension: "
            f"got {theta.size}, expected {geometry.dimension}."
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta_dot must contain only finite values.")
    K = np.asarray(geometry.K, dtype=float)
    f = np.asarray(geometry.f, dtype=float).reshape(-1)
    projected_sq_raw = float(theta @ K @ theta) if theta.size else 0.0
    projected_sq = float(max(0.0, projected_sq_raw))
    realized_raw = float(geometry.norm_b_sq) - 2.0 * float(f @ theta) + projected_sq_raw
    realized_sq = float(_clip_nonnegative(realized_raw))
    best_sq = float(_clip_interval(best_case_residual_sq, 0.0, float(geometry.norm_b_sq)))
    denom = residual_denominator(float(geometry.norm_b_sq), float(epsilon))
    rho_real = float(_clip_interval(realized_sq / denom, 0.0, 1.0))
    rho_expr = float(_clip_interval(best_sq / denom, 0.0, 1.0))
    rho_num = float(max(0.0, realized_sq - best_sq) / denom)
    return StateSpaceSolveMetrics(
        theta_dot_l2=float(np.linalg.norm(theta)),
        projected_velocity_sq=projected_sq,
        projected_velocity_l2=float(np.sqrt(projected_sq)),
        realized_residual_sq=realized_sq,
        rho_real=rho_real,
        best_case_residual_sq=best_sq,
        rho_expr=rho_expr,
        rho_num=rho_num,
        denominator=float(denom),
    )


def state_space_kink_eta(
    geometry: McLachlanGeometry,
    theta_dot_plus: np.ndarray | Sequence[float],
    theta_dot_minus: np.ndarray | Sequence[float],
    *,
    epsilon: float,
) -> float:
    """Return the state-space velocity jump between two same-plane solves."""

    plus = np.asarray(theta_dot_plus, dtype=float).reshape(-1)
    minus = np.asarray(theta_dot_minus, dtype=float).reshape(-1)
    if plus.shape != minus.shape:
        raise ValueError(f"theta_dot shapes must match; got {plus.shape} and {minus.shape}.")
    if int(plus.size) != int(geometry.dimension):
        raise ValueError(
            "theta_dot dimension must match McLachlan geometry dimension: "
            f"got {plus.size}, expected {geometry.dimension}."
        )
    if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
        raise ValueError("theta_dot values must be finite.")
    delta = np.asarray(plus - minus, dtype=float).reshape(-1)
    jump_sq = float(delta @ np.asarray(geometry.K, dtype=float) @ delta) if delta.size else 0.0
    denom = residual_denominator(float(geometry.norm_b_sq), float(epsilon))
    return float(max(0.0, jump_sq) / denom)


def _clip_nonnegative(value: float) -> float:
    if not np.isfinite(float(value)):
        raise ValueError("state-space metric produced a non-finite value.")
    return float(max(0.0, float(value)))


def _clip_interval(value: float, lower: float, upper: float) -> float:
    if not np.isfinite(float(value)):
        raise ValueError("state-space metric produced a non-finite value.")
    lo = float(lower)
    hi = float(upper)
    if hi < lo:
        hi = lo
    return float(min(max(float(value), lo), hi))


def geometry_diagnostics(
    geometry: McLachlanGeometry,
    *,
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
) -> McLachlanGeometryDiagnostics:
    solve = solve_theta_dot(geometry.K, geometry.f, policy=inverse_policy)
    residual_sq = float(max(0.0, float(geometry.norm_b_sq) - float(solve.gamma)))
    residual_ratio = float(
        residual_sq / residual_denominator(geometry.norm_b_sq, inverse_policy.epsilon)
    )
    return McLachlanGeometryDiagnostics(
        gamma=float(solve.gamma),
        residual_sq=residual_sq,
        residual_ratio=residual_ratio,
        rank=int(solve.inverse.rank),
        condition_number=solve.inverse.condition_number,
    )


__all__ = [
    "GEOMETRY_SCHEMA_V1",
    "McLachlanGeometry",
    "McLachlanGeometryDiagnostics",
    "StateSpaceSolveMetrics",
    "geometry_diagnostics",
    "residual_denominator",
    "state_space_kink_eta",
    "state_space_solve_metrics",
]
