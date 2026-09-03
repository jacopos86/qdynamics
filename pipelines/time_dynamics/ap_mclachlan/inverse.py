"""Supported inverse convention for AP-McLachlan geometry.

This module owns the shared ridge, retained-eigenspace, and solve convention
used by fixed McLachlan propagation and support-patch scoring.  It is purely
array-level and does not know about Hamiltonians, ansatz objects, exact
references, candidate generation, or controller commits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


INVERSE_POLICY_SUPPORTED_EIGH_V1 = "supported_eigh_ridge_v1"
INVERSE_POLICY_NUMPY_PINV_RCOND_V1 = INVERSE_POLICY_SUPPORTED_EIGH_V1
DEFAULT_MCLACHLAN_RIDGE_LAMBDA = 1.0e-7
DEFAULT_MCLACHLAN_SOLVE_DAMPING = 0.0


@dataclass(frozen=True)
class McLachlanInversePolicy:
    """Ridge and retained-eigenspace convention for McLachlan solves."""

    pinv_rcond: float = 1.0e-10
    ridge_lambda: float = DEFAULT_MCLACHLAN_RIDGE_LAMBDA
    solve_damping: float = DEFAULT_MCLACHLAN_SOLVE_DAMPING
    epsilon: float = 1.0e-14
    policy_id: str = INVERSE_POLICY_SUPPORTED_EIGH_V1

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.pinv_rcond)) or float(self.pinv_rcond) < 0.0:
            raise ValueError("pinv_rcond must be finite and non-negative.")
        if not np.isfinite(float(self.ridge_lambda)) or float(self.ridge_lambda) < 0.0:
            raise ValueError("ridge_lambda must be finite and non-negative.")
        if not np.isfinite(float(self.solve_damping)) or float(self.solve_damping) < 0.0:
            raise ValueError("solve_damping must be finite and non-negative.")
        if not np.isfinite(float(self.epsilon)) or float(self.epsilon) < 0.0:
            raise ValueError("epsilon must be finite and non-negative.")


@dataclass(frozen=True)
class SupportedInverse:
    """Telemetry for a supported symmetric inverse."""

    inverse: np.ndarray
    eigenvalues: np.ndarray
    retained: np.ndarray
    rank: int
    condition_number: float | None
    policy_id: str
    pinv_rcond: float
    ridge_lambda: float
    solve_damping: float
    max_abs_eigenvalue: float | None = None
    retained_threshold: float = 0.0
    retained_support_empty: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "inverse_policy_id": str(self.policy_id),
            "inverse_pinv_rcond": float(self.pinv_rcond),
            "inverse_ridge_lambda": float(self.ridge_lambda),
            "inverse_solve_damping": float(self.solve_damping),
            "inverse_rank_retained": int(self.rank),
            "inverse_condition_number": (
                None if self.condition_number is None else float(self.condition_number)
            ),
            "inverse_max_abs_eigenvalue": (
                None if self.max_abs_eigenvalue is None else float(self.max_abs_eigenvalue)
            ),
            "inverse_retained_threshold": float(self.retained_threshold),
            "inverse_retained_support_empty": bool(self.retained_support_empty),
            "inverse_eigenvalues": [float(x) for x in self.eigenvalues.tolist()],
            "inverse_retained_mask": [bool(x) for x in self.retained.tolist()],
        }


@dataclass(frozen=True)
class McLachlanSolve:
    """Result of one supported McLachlan solve under the shared policy.

    ``gamma = f^T theta_dot`` is the historical objective value.  It equals the
    captured drift only for the exact unridged, untruncated, undamped solve;
    under the actual policy it overstates what the realized velocity achieves.

    ``captured_drift`` is the realized reduction of the McLachlan miss by the
    velocity this solve actually returns:

        Q = ||b||^2 - ||T theta_dot - b||^2
          = 2 f^T theta_dot - theta_dot^T K theta_dot,

    evaluated against the *raw* (unridged) ``K``.  The identity holds exactly
    under retained-mode truncation, ridge, and solve damping because it uses
    the realized ``theta_dot`` directly.  Support-patch gains and losses must
    use this quantity; ``gamma`` remains for propagation telemetry and legacy
    comparison.
    """

    theta_dot: np.ndarray
    gamma: float
    inverse: SupportedInverse
    captured_drift: float

    def to_json_dict(self) -> dict[str, Any]:
        payload = self.inverse.to_json_dict()
        payload.update(
            {
                "theta_dot": [float(x) for x in self.theta_dot.tolist()],
                "gamma": float(self.gamma),
                "captured_drift": float(self.captured_drift),
            }
        )
        return payload


def as_float_matrix(value: Any, *, name: str = "matrix") -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix; got shape {arr.shape}.")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(arr, dtype=float)


def as_float_vector(value: Any, *, name: str = "f_vec") -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(arr, dtype=float)


def symmetrize_matrix(matrix: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    mat = as_float_matrix(matrix, name="matrix")
    return 0.5 * (mat + mat.T)


def apply_ridge(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    ridge_lambda: float,
) -> np.ndarray:
    mat = symmetrize_matrix(matrix)
    ridge = float(ridge_lambda)
    if ridge == 0.0 or mat.size == 0:
        return mat
    return mat + ridge * np.eye(int(mat.shape[0]), dtype=float)


def supported_inverse(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    policy: McLachlanInversePolicy = McLachlanInversePolicy(),
) -> SupportedInverse:
    """Return the supported inverse under the shared AP-McLachlan convention."""

    mat = apply_ridge(matrix, ridge_lambda=float(policy.ridge_lambda))
    n = int(mat.shape[0])
    if n == 0:
        return SupportedInverse(
            inverse=np.zeros((0, 0), dtype=float),
            eigenvalues=np.zeros(0, dtype=float),
            retained=np.zeros(0, dtype=bool),
            rank=0,
            condition_number=None,
            policy_id=str(policy.policy_id),
            pinv_rcond=float(policy.pinv_rcond),
            ridge_lambda=float(policy.ridge_lambda),
            solve_damping=float(policy.solve_damping),
            max_abs_eigenvalue=None,
            retained_threshold=0.0,
            retained_support_empty=True,
        )

    try:
        eigenvalues, vectors = np.linalg.eigh(mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError("McLachlan matrix eigendecomposition failed.") from exc

    abs_eigs = np.abs(eigenvalues)
    max_abs = float(np.max(abs_eigs)) if abs_eigs.size else 0.0
    threshold = float(policy.pinv_rcond) * max_abs
    retained = abs_eigs > threshold
    inv_eigs = np.zeros_like(eigenvalues, dtype=float)
    damping = float(policy.solve_damping)
    if damping == 0.0:
        inv_eigs[retained] = 1.0 / eigenvalues[retained]
    else:
        signs = np.sign(eigenvalues[retained])
        inv_eigs[retained] = signs / (abs_eigs[retained] + damping)
    inv = (vectors * inv_eigs) @ vectors.T

    retained_abs = abs_eigs[retained]
    condition = None
    if retained_abs.size:
        condition = float(np.max(retained_abs) / max(np.min(retained_abs), float(policy.epsilon)))

    return SupportedInverse(
        inverse=np.asarray(inv, dtype=float),
        eigenvalues=np.asarray(eigenvalues, dtype=float),
        retained=np.asarray(retained, dtype=bool),
        rank=int(np.count_nonzero(retained)),
        condition_number=condition,
        policy_id=str(policy.policy_id),
        pinv_rcond=float(policy.pinv_rcond),
        ridge_lambda=float(policy.ridge_lambda),
        solve_damping=float(policy.solve_damping),
        max_abs_eigenvalue=float(max_abs),
        retained_threshold=float(threshold),
        retained_support_empty=bool(np.count_nonzero(retained) == 0),
    )


def solve_theta_dot(
    matrix: np.ndarray | Sequence[Sequence[float]],
    f_vec: np.ndarray | Sequence[float],
    *,
    policy: McLachlanInversePolicy = McLachlanInversePolicy(),
) -> McLachlanSolve:
    mat = as_float_matrix(matrix, name="matrix")
    force = as_float_vector(f_vec, name="f_vec")
    if mat.shape != (int(force.size), int(force.size)):
        raise ValueError(
            "matrix must be square with dimension len(f_vec); "
            f"got matrix {mat.shape}, f_vec {force.shape}."
        )
    inverse = supported_inverse(mat, policy=policy)
    theta_dot = np.asarray(inverse.inverse @ force, dtype=float).reshape(-1)
    gamma = float(force @ theta_dot)
    # Realized captured drift against the raw (unridged) K; see McLachlanSolve.
    captured_drift = float(2.0 * gamma - float(theta_dot @ mat @ theta_dot))
    if (
        not np.all(np.isfinite(theta_dot))
        or not np.isfinite(gamma)
        or not np.isfinite(captured_drift)
    ):
        raise ValueError("McLachlan solve produced non-finite values.")
    return McLachlanSolve(
        theta_dot=theta_dot,
        gamma=gamma,
        inverse=inverse,
        captured_drift=captured_drift,
    )


def gamma_for_support(
    *,
    matrix: np.ndarray | Sequence[Sequence[float]],
    f_vec: np.ndarray | Sequence[float],
    indices: Sequence[int],
    policy: McLachlanInversePolicy = McLachlanInversePolicy(),
) -> McLachlanSolve:
    """Return ``Gamma(S)=f_S^T K_SS^+ f_S`` on a runtime support."""

    force = as_float_vector(f_vec, name="f_vec")
    mat_full = as_float_matrix(matrix, name="matrix")
    if mat_full.shape != (int(force.size), int(force.size)):
        raise ValueError(
            "matrix must be square with dimension len(f_vec); "
            f"got matrix {mat_full.shape}, f_vec {force.shape}."
        )
    idx = tuple(int(i) for i in indices if 0 <= int(i) < int(force.size))
    if not idx:
        return McLachlanSolve(
            theta_dot=np.zeros(0, dtype=float),
            gamma=0.0,
            inverse=supported_inverse(np.zeros((0, 0), dtype=float), policy=policy),
            captured_drift=0.0,
        )
    mat = np.asarray(mat_full[np.ix_(idx, idx)], dtype=float)
    force_slice = np.asarray(force[list(idx)], dtype=float).reshape(-1)
    return solve_theta_dot(mat, force_slice, policy=policy)


__all__ = [
    "DEFAULT_MCLACHLAN_RIDGE_LAMBDA",
    "DEFAULT_MCLACHLAN_SOLVE_DAMPING",
    "INVERSE_POLICY_NUMPY_PINV_RCOND_V1",
    "INVERSE_POLICY_SUPPORTED_EIGH_V1",
    "McLachlanInversePolicy",
    "McLachlanSolve",
    "SupportedInverse",
    "apply_ridge",
    "as_float_matrix",
    "as_float_vector",
    "gamma_for_support",
    "solve_theta_dot",
    "supported_inverse",
    "symmetrize_matrix",
]
