"""Pure support-patch scoring for AP-McLachlan.

This module is the array-level implementation target for append-prune
McLachlan support edits. It consumes already-estimated geometry/force arrays and
returns JSON-safe telemetry. It does not know about controller state, exact
references, candidate generation, commits, Optuna, or reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.inverse import (
    INVERSE_POLICY_NUMPY_PINV_RCOND_V1,
    McLachlanInversePolicy,
    apply_ridge as _shared_apply_ridge,
    gamma_for_support as _shared_gamma_for_support,
    supported_inverse as _shared_supported_inverse,
)

SUPPORT_PATCH_SCORE_V1 = "support_patch_score_v1"
SCHUR_NOVELTY_V1 = "checkpoint_schur_novelty_v1"
PRUNE_CONDITIONING_DIAGNOSTICS_V1 = "prune_conditioning_diagnostics_v1"

PATCH_NO_EDIT = "no_edit"
PATCH_APPEND = "append"
# Backward-compatibility token for old trajectory manifests. New pure-growth
# support patches must emit PATCH_APPEND.
PATCH_INSERT = "insert"
PATCH_DELETE = "delete"
PATCH_EXCHANGE = "exchange"

DENOM_NORM_B_PLUS_EPS_V1 = "norm_b_sq_plus_epsilon_v1"
SUPPORT_BEFORE_AFTER_V1 = "before_after_runtime_support_v1"
DAMPED_DELTA_K_V1 = "damped_delta_k_v1"


@dataclass(frozen=True)
class SupportPatch:
    """Uniform patch shape ``(B-, R+)`` for AP-McLachlan support edits."""

    removed_runtime_indices: tuple[int, ...] = ()
    inserted_count: int = 0
    inserted_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        removed = tuple(int(i) for i in self.removed_runtime_indices)
        inserted = int(self.inserted_count)
        labels = tuple(str(label) for label in self.inserted_labels)
        if inserted < 0:
            raise ValueError("inserted_count must be non-negative.")
        if labels and len(labels) != inserted:
            raise ValueError("inserted_labels must be empty or match inserted_count.")
        object.__setattr__(self, "removed_runtime_indices", removed)
        object.__setattr__(self, "inserted_count", inserted)
        object.__setattr__(self, "inserted_labels", labels)

    @property
    def kind(self) -> str:
        return _patch_kind(
            removed_runtime_indices=self.removed_runtime_indices,
            inserted_count=self.inserted_count,
        )


@dataclass(frozen=True)
class SupportPatchGeometry:
    """Before-support geometry plus optional appended block geometry."""

    K_before: np.ndarray | Sequence[Sequence[float]]
    f_before: np.ndarray | Sequence[float]
    norm_b_sq: float
    K_insert_cross: np.ndarray | Sequence[Sequence[float]] | None = None
    K_insert_insert: np.ndarray | Sequence[Sequence[float]] | None = None
    f_insert: np.ndarray | Sequence[float] | None = None


@dataclass(frozen=True)
class SupportPatchBeforeCache:
    """Checkpoint-local before-support linear algebra cache.

    The before support is shared by every append/prune/exchange candidate at one
    time point.  This cache stores only quantities that are invariant across
    those candidates.
    """

    runtime_parameter_count: int
    K_shape: tuple[int, int]
    inverse_policy_signature: tuple[str, float, float, float, float]
    before_indices: tuple[int, ...]
    before_solve: Any | None
    before_solve_error: str | None
    rank_before: int | None
    before_inverse: Any | None
    before_inverse_error: str | None


@dataclass(frozen=True)
class SupportPatchAfterCache:
    """Candidate-local after-support solve reused by score diagnostics."""

    before_runtime_parameter_count: int
    removed_runtime_indices: tuple[int, ...]
    inserted_count: int
    after_indices_before_part: tuple[int, ...]
    K_shape: tuple[int, int]
    inverse_policy_signature: tuple[str, float, float, float, float]
    after_solve: Any | None
    after_solve_error: str | None
    rank_after: int | None


@dataclass(frozen=True)
class AugmentedSolveConfirmation:
    """Pre-commit solve check on the appended/exchanged support."""

    confirmed: bool
    reason: str
    support_size: int
    theta_dot_l2: float | None
    gamma: float | None
    residual_sq: float | None
    residual_ratio: float | None
    rank_retained: int | None
    condition_number: float | None
    pinv_policy_id: str
    pinv_rcond: float
    ridge_lambda: float
    solve_damping: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "confirmed": bool(self.confirmed),
            "reason": str(self.reason),
            "support_size": int(self.support_size),
            "theta_dot_l2": _finite_or_none(self.theta_dot_l2),
            "gamma": _finite_or_none(self.gamma),
            "residual_sq": _finite_or_none(self.residual_sq),
            "residual_ratio": _finite_or_none(self.residual_ratio),
            "rank_retained": (
                None if self.rank_retained is None else int(self.rank_retained)
            ),
            "condition_number": _finite_or_none(self.condition_number),
            "pinv_policy_id": str(self.pinv_policy_id),
            "pinv_rcond": float(self.pinv_rcond),
            "ridge_lambda": float(self.ridge_lambda),
            "solve_damping": float(self.solve_damping),
        }


@dataclass(frozen=True)
class SupportPatchScore:
    """JSON-safe AP-McLachlan support-patch score payload."""

    patch_kind: str
    before_indices: tuple[int, ...]
    after_indices_before_part: tuple[int, ...]
    removed_runtime_indices: tuple[int, ...]
    inserted_count: int
    inserted_labels: tuple[str, ...]
    before_gain: float | None
    after_gain: float | None
    signed_delta_gain: float | None
    normalized_score: float | None
    insertion_gain: float | None
    deletion_loss: float | None
    denominator: float
    denominator_kind: str
    support_kind: str
    score_kind: str
    pinv_policy_id: str
    pinv_rcond: float
    ridge_lambda: float
    solve_damping: float
    rank_before: int | None
    rank_after: int | None
    cost_terms: dict[str, Any] = field(default_factory=dict)
    cost_weight: float = 0.0
    rank_score: float | None = None
    schur_novelty: "SchurNovelty | None" = None
    augmented_solve_confirmation: AugmentedSolveConfirmation | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "support_patch_score_kind": self.score_kind,
            "support_patch_kind": self.patch_kind,
            "support_patch_before_indices": [int(i) for i in self.before_indices],
            "support_patch_after_indices_before_part": [
                int(i) for i in self.after_indices_before_part
            ],
            "support_patch_removed_runtime_indices": [
                int(i) for i in self.removed_runtime_indices
            ],
            "support_patch_removed_count": int(len(self.removed_runtime_indices)),
            "support_patch_deleted_count": int(len(self.removed_runtime_indices)),
            "support_patch_appended_count": int(self.inserted_count),
            "support_patch_appended_labels": [str(v) for v in self.inserted_labels],
            "support_patch_inserted_count": int(self.inserted_count),
            "support_patch_inserted_labels": [str(v) for v in self.inserted_labels],
            "support_patch_before_gain": _finite_or_none(self.before_gain),
            "support_patch_after_gain": _finite_or_none(self.after_gain),
            "support_patch_signed_delta_gain": _finite_or_none(self.signed_delta_gain),
            "support_patch_normalized_score": _finite_or_none(self.normalized_score),
            "support_patch_append_gain": _finite_or_none(self.insertion_gain),
            "support_patch_insertion_gain": _finite_or_none(self.insertion_gain),
            "support_patch_deletion_loss": _finite_or_none(self.deletion_loss),
            "support_patch_denominator": float(self.denominator),
            "support_patch_denominator_kind": str(self.denominator_kind),
            "support_patch_support_kind": str(self.support_kind),
            "support_patch_pinv_policy_id": str(self.pinv_policy_id),
            "support_patch_pinv_rcond": float(self.pinv_rcond),
            "support_patch_ridge_lambda": float(self.ridge_lambda),
            "support_patch_solve_damping": float(self.solve_damping),
            "support_patch_rank_before": self.rank_before,
            "support_patch_rank_after": self.rank_after,
            "support_patch_cost_terms": _json_safe(self.cost_terms),
            "support_patch_cost_weight": float(self.cost_weight),
            "support_patch_rank_score": _finite_or_none(self.rank_score),
            "support_patch_schur_novelty": (
                None
                if self.schur_novelty is None
                else self.schur_novelty.to_json_dict()
            ),
            "support_patch_augmented_solve_confirmation": (
                None
                if self.augmented_solve_confirmation is None
                else self.augmented_solve_confirmation.to_json_dict()
            ),
        }


@dataclass(frozen=True)
class SchurNovelty:
    """Checkpoint-local Schur novelty of candidate directions against support."""

    matrix: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    candidate_count: int
    available_count: int
    retained_threshold: float
    min_eigenvalue: float | None
    max_eigenvalue: float | None
    min_retained_eigenvalue: float | None
    max_retained_eigenvalue: float | None
    condition_number: float | None
    full_rank: bool
    psd_within_tolerance: bool
    negative_tolerance: float
    policy_id: str
    pinv_rcond: float
    ridge_lambda: float
    solve_damping: float
    novelty_kind: str = SCHUR_NOVELTY_V1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schur_novelty_kind": str(self.novelty_kind),
            "candidate_count": int(self.candidate_count),
            "available_count": int(self.available_count),
            "rank": int(self.rank),
            "full_rank": bool(self.full_rank),
            "psd_within_tolerance": bool(self.psd_within_tolerance),
            "condition_number": _finite_or_none(self.condition_number),
            "min_eigenvalue": _finite_or_none(self.min_eigenvalue),
            "max_eigenvalue": _finite_or_none(self.max_eigenvalue),
            "min_retained_eigenvalue": _finite_or_none(self.min_retained_eigenvalue),
            "max_retained_eigenvalue": _finite_or_none(self.max_retained_eigenvalue),
            "retained_threshold": float(self.retained_threshold),
            "negative_tolerance": float(self.negative_tolerance),
            "pinv_policy_id": str(self.policy_id),
            "pinv_rcond": float(self.pinv_rcond),
            "ridge_lambda": float(self.ridge_lambda),
            "solve_damping": float(self.solve_damping),
        }


@dataclass(frozen=True)
class PruneConditioningDiagnostics:
    """Whole-batch conditioning diagnostics for prune nomination pressure."""

    available: bool
    reason: str
    removed_runtime_indices: tuple[int, ...]
    retained_runtime_indices: tuple[int, ...]
    condition_number_before: float | None
    condition_number_after: float | None
    log_condition_before: float
    log_condition_after: float
    d_kappa_rel: float
    d_kappa_dam: float
    d_schur: float
    schur_rank: int | None
    schur_candidate_count: int
    schur_condition_number: float | None
    schur_rank_deficit_fraction: float
    diagnostics_kind: str = PRUNE_CONDITIONING_DIAGNOSTICS_V1

    @property
    def conditioning_toxicity(self) -> float:
        return float(max(0.0, self.d_kappa_rel) + max(0.0, self.d_schur))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "prune_conditioning_diagnostics_kind": str(self.diagnostics_kind),
            "available": bool(self.available),
            "reason": str(self.reason),
            "removed_runtime_indices": [int(i) for i in self.removed_runtime_indices],
            "retained_runtime_indices": [int(i) for i in self.retained_runtime_indices],
            "condition_number_before": _finite_or_none(self.condition_number_before),
            "condition_number_after": _finite_or_none(self.condition_number_after),
            "log_condition_before": float(self.log_condition_before),
            "log_condition_after": float(self.log_condition_after),
            "d_kappa_rel": float(self.d_kappa_rel),
            "d_kappa_dam": float(self.d_kappa_dam),
            "d_schur": float(self.d_schur),
            "d_conditioning_toxicity": float(self.conditioning_toxicity),
            "schur_rank": None if self.schur_rank is None else int(self.schur_rank),
            "schur_candidate_count": int(self.schur_candidate_count),
            "schur_condition_number": _finite_or_none(self.schur_condition_number),
            "schur_rank_deficit_fraction": float(self.schur_rank_deficit_fraction),
        }


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


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


def _float_matrix(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _float_vector(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validate_before_geometry(
    geometry: SupportPatchGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    K_before = _float_matrix(geometry.K_before, name="K_before")
    f_before = _float_vector(geometry.f_before, name="f_before")
    n = int(f_before.size)
    if K_before.shape != (n, n):
        raise ValueError(
            f"K_before must have shape ({n}, {n}) for f_before; got {K_before.shape}."
        )
    return K_before, f_before


def _valid_removed_indices(values: Sequence[int], *, size: int) -> tuple[int, ...]:
    valid = sorted({int(i) for i in values if 0 <= int(i) < int(size)})
    return tuple(valid)


def _patch_kind(*, removed_runtime_indices: Sequence[int], inserted_count: int) -> str:
    has_removed = bool(tuple(removed_runtime_indices))
    has_inserted = int(inserted_count) > 0
    if not has_removed and not has_inserted:
        return PATCH_NO_EDIT
    if not has_removed and has_inserted:
        return PATCH_APPEND
    if has_removed and not has_inserted:
        return PATCH_DELETE
    return PATCH_EXCHANGE


def _denominator(norm_b_sq: float, epsilon: float) -> float:
    eps = max(0.0, float(epsilon))
    denom = float(norm_b_sq) + eps
    if not np.isfinite(denom) or denom <= 0.0:
        denom = eps if eps > 0.0 else 1.0e-14
    return float(denom)


def _matrix_rank(matrix: np.ndarray) -> int | None:
    if matrix.size <= 0:
        return 0
    try:
        return int(np.linalg.matrix_rank(matrix))
    except np.linalg.LinAlgError:
        return None


def _matrix_rank_from_symmetric_solve(
    solve: Any | None,
    *,
    fallback_matrix: np.ndarray,
) -> int | None:
    """Reuse solve eigenvalues for NumPy's default symmetric matrix rank."""

    if solve is None:
        return _matrix_rank(fallback_matrix)
    eigenvalues = np.asarray(solve.inverse.eigenvalues, dtype=float).reshape(-1)
    if eigenvalues.size == 0:
        return 0
    singular_values = np.abs(eigenvalues)
    tolerance = (
        float(np.max(singular_values))
        * int(eigenvalues.size)
        * float(np.finfo(eigenvalues.dtype).eps)
    )
    return int(np.count_nonzero(singular_values > tolerance))


def _apply_ridge(matrix: np.ndarray, ridge_lambda: float) -> np.ndarray:
    return _shared_apply_ridge(matrix, ridge_lambda=float(ridge_lambda))


def _inverse_policy_signature(
    policy: McLachlanInversePolicy,
) -> tuple[str, float, float, float, float]:
    return (
        str(policy.policy_id),
        float(policy.pinv_rcond),
        float(policy.ridge_lambda),
        float(policy.solve_damping),
        float(policy.epsilon),
    )


def _validate_before_cache(
    cache: SupportPatchBeforeCache | None,
    *,
    K_before: np.ndarray,
    f_before: np.ndarray,
    inverse_policy: McLachlanInversePolicy,
) -> SupportPatchBeforeCache | None:
    if cache is None:
        return None
    expected_shape = (int(f_before.size), int(f_before.size))
    if tuple(cache.K_shape) != expected_shape:
        raise ValueError(
            "support-patch before cache shape does not match current geometry: "
            f"got {cache.K_shape}, expected {expected_shape}."
        )
    if int(cache.runtime_parameter_count) != int(f_before.size):
        raise ValueError(
            "support-patch before cache parameter count does not match current "
            f"force size: got {cache.runtime_parameter_count}, expected "
            f"{int(f_before.size)}."
        )
    if tuple(cache.inverse_policy_signature) != _inverse_policy_signature(
        inverse_policy
    ):
        raise ValueError(
            "support-patch before cache inverse policy does not match current "
            "inverse policy."
        )
    return cache


def _validate_after_cache(
    cache: SupportPatchAfterCache | None,
    *,
    K_before: np.ndarray,
    patch: SupportPatch,
    K_after: np.ndarray,
    keep: tuple[int, ...],
    inverse_policy: McLachlanInversePolicy,
) -> SupportPatchAfterCache | None:
    if cache is None:
        return None
    removed = _valid_removed_indices(
        patch.removed_runtime_indices,
        size=int(K_before.shape[0]),
    )
    if int(cache.before_runtime_parameter_count) != int(K_before.shape[0]):
        raise ValueError("support-patch after cache before-support size mismatch.")
    if tuple(cache.removed_runtime_indices) != tuple(removed):
        raise ValueError("support-patch after cache removed indices mismatch.")
    if int(cache.inserted_count) != int(patch.inserted_count):
        raise ValueError("support-patch after cache inserted count mismatch.")
    if tuple(cache.after_indices_before_part) != tuple(keep):
        raise ValueError("support-patch after cache retained indices mismatch.")
    if tuple(cache.K_shape) != tuple(int(value) for value in K_after.shape):
        raise ValueError("support-patch after cache matrix shape mismatch.")
    if tuple(cache.inverse_policy_signature) != _inverse_policy_signature(
        inverse_policy
    ):
        raise ValueError("support-patch after cache inverse policy mismatch.")
    return cache


def _safe_log_condition_number(value: Any) -> float:
    finite = _finite_or_none(value)
    if finite is None or finite <= 1.0:
        return 0.0
    capped = min(float(finite), 1.0e300)
    return float(math.log(capped))


def _schur_degeneracy_pressure(schur: SchurNovelty) -> tuple[float, float]:
    count = int(schur.candidate_count)
    if count <= 0:
        return 0.0, 0.0
    rank = max(0, min(int(schur.rank), count))
    rank_deficit_fraction = float((count - rank) / count)
    log_condition = _safe_log_condition_number(schur.condition_number)
    return float(rank_deficit_fraction + log_condition), rank_deficit_fraction


def _cost_penalty(cost_terms: Mapping[str, Any] | None) -> float:
    if not cost_terms:
        return 0.0
    value = _finite_or_none(cost_terms.get("cost_penalty"))
    return 0.0 if value is None else float(value)


def gain_for_support(
    *,
    matrix: np.ndarray | Sequence[Sequence[float]],
    f_vec: np.ndarray | Sequence[float],
    indices: Sequence[int],
    pinv_rcond: float | None = None,
    inverse_policy: McLachlanInversePolicy | None = None,
) -> float | None:
    """Return ``f_A.T @ matrix_A^+ @ f_A`` for a runtime support."""

    policy = inverse_policy or McLachlanInversePolicy(
        pinv_rcond=1.0e-10 if pinv_rcond is None else float(pinv_rcond)
    )
    try:
        solve = _shared_gamma_for_support(
            matrix=matrix,
            f_vec=f_vec,
            indices=indices,
            policy=policy,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    value = float(solve.gamma)
    if not np.isfinite(value):
        return None
    return float(value)


def _solve_for_support(
    *,
    matrix: np.ndarray | Sequence[Sequence[float]],
    f_vec: np.ndarray | Sequence[float],
    indices: Sequence[int],
    inverse_policy: McLachlanInversePolicy,
) -> tuple[Any | None, str | None]:
    try:
        solve = _shared_gamma_for_support(
            matrix=matrix,
            f_vec=f_vec,
            indices=indices,
            policy=inverse_policy,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, str(exc)
    if not np.all(np.isfinite(solve.theta_dot)) or not np.isfinite(float(solve.gamma)):
        return None, "augmented_solve_nonfinite"
    return solve, None


def build_support_patch_before_cache(
    *,
    geometry: SupportPatchGeometry,
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
) -> SupportPatchBeforeCache:
    """Build reusable before-support solve/rank/inverse data for one checkpoint."""

    K_before, f_before = _validate_before_geometry(geometry)
    n = int(f_before.size)
    before_indices = tuple(range(n))
    before_solve, before_solve_error = _solve_for_support(
        matrix=K_before,
        f_vec=f_before,
        indices=before_indices,
        inverse_policy=inverse_policy,
    )
    rank_before = _matrix_rank_from_symmetric_solve(
        before_solve,
        fallback_matrix=_apply_ridge(K_before, inverse_policy.ridge_lambda),
    )
    before_inverse = None
    before_inverse_error = None
    if before_solve is not None:
        before_inverse = before_solve.inverse
    else:
        try:
            before_inverse = _shared_supported_inverse(K_before, policy=inverse_policy)
        except (ValueError, np.linalg.LinAlgError) as exc:
            before_inverse_error = str(exc)
    return SupportPatchBeforeCache(
        runtime_parameter_count=n,
        K_shape=tuple(int(v) for v in K_before.shape),
        inverse_policy_signature=_inverse_policy_signature(inverse_policy),
        before_indices=before_indices,
        before_solve=before_solve,
        before_solve_error=before_solve_error,
        rank_before=rank_before,
        before_inverse=before_inverse,
        before_inverse_error=before_inverse_error,
    )


def build_support_patch_after_cache(
    *,
    geometry: SupportPatchGeometry,
    patch: SupportPatch,
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
) -> SupportPatchAfterCache:
    """Build one candidate's reusable after-support solve data."""

    K_before, _f_before = _validate_before_geometry(geometry)
    K_after, f_after, keep = build_after_geometry(geometry=geometry, patch=patch)
    after_indices = tuple(range(int(f_after.size)))
    after_solve, after_solve_error = _solve_for_support(
        matrix=K_after,
        f_vec=f_after,
        indices=after_indices,
        inverse_policy=inverse_policy,
    )
    rank_after = _matrix_rank_from_symmetric_solve(
        after_solve,
        fallback_matrix=_apply_ridge(K_after, inverse_policy.ridge_lambda),
    )
    return SupportPatchAfterCache(
        before_runtime_parameter_count=int(K_before.shape[0]),
        removed_runtime_indices=_valid_removed_indices(
            patch.removed_runtime_indices,
            size=int(K_before.shape[0]),
        ),
        inserted_count=int(patch.inserted_count),
        after_indices_before_part=tuple(keep),
        K_shape=tuple(int(value) for value in K_after.shape),
        inverse_policy_signature=_inverse_policy_signature(inverse_policy),
        after_solve=after_solve,
        after_solve_error=after_solve_error,
        rank_after=rank_after,
    )


def _residual_sq_from_gamma(
    *,
    norm_b_sq: float,
    gamma: float,
) -> float:
    total = max(0.0, float(norm_b_sq))
    value = float(total - float(gamma))
    if not np.isfinite(value):
        return float("nan")
    return float(min(max(value, 0.0), total))


def _augmented_solve_confirmation(
    *,
    solve: Any | None,
    solve_error: str | None,
    support_size: int,
    norm_b_sq: float,
    denominator: float,
    inverse_policy: McLachlanInversePolicy,
) -> AugmentedSolveConfirmation:
    if solve is None:
        return AugmentedSolveConfirmation(
            confirmed=False,
            reason=solve_error or "augmented_solve_failed",
            support_size=int(support_size),
            theta_dot_l2=None,
            gamma=None,
            residual_sq=None,
            residual_ratio=None,
            rank_retained=None,
            condition_number=None,
            pinv_policy_id=str(inverse_policy.policy_id),
            pinv_rcond=float(inverse_policy.pinv_rcond),
            ridge_lambda=float(inverse_policy.ridge_lambda),
            solve_damping=float(inverse_policy.solve_damping),
        )

    theta_dot = np.asarray(solve.theta_dot, dtype=float).reshape(-1)
    gamma = float(solve.gamma)
    residual_sq = _residual_sq_from_gamma(norm_b_sq=float(norm_b_sq), gamma=gamma)
    residual_ratio = float(residual_sq / float(denominator))
    values_finite = bool(
        np.all(np.isfinite(theta_dot))
        and np.isfinite(gamma)
        and np.isfinite(residual_sq)
        and np.isfinite(residual_ratio)
    )
    rank = int(solve.inverse.rank)
    return AugmentedSolveConfirmation(
        confirmed=values_finite,
        reason="confirmed" if values_finite else "augmented_solve_nonfinite",
        support_size=int(support_size),
        theta_dot_l2=float(np.linalg.norm(theta_dot)) if values_finite else None,
        gamma=gamma if np.isfinite(gamma) else None,
        residual_sq=residual_sq if np.isfinite(residual_sq) else None,
        residual_ratio=residual_ratio if np.isfinite(residual_ratio) else None,
        rank_retained=rank,
        condition_number=solve.inverse.condition_number,
        pinv_policy_id=str(solve.inverse.policy_id),
        pinv_rcond=float(solve.inverse.pinv_rcond),
        ridge_lambda=float(solve.inverse.ridge_lambda),
        solve_damping=float(solve.inverse.solve_damping),
    )


def build_after_geometry(
    *,
    geometry: SupportPatchGeometry,
    patch: SupportPatch,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Build after-patch geometry over retained before directions plus inserts."""

    K_before, f_before = _validate_before_geometry(geometry)
    n = int(f_before.size)
    removed = set(_valid_removed_indices(patch.removed_runtime_indices, size=n))
    keep = tuple(i for i in range(n) if i not in removed)
    m = int(patch.inserted_count)

    if m == 0:
        K_after = np.asarray(K_before[np.ix_(keep, keep)], dtype=float)
        f_after = np.asarray(f_before[list(keep)], dtype=float)
        return K_after, f_after, keep

    if (
        geometry.K_insert_cross is None
        or geometry.K_insert_insert is None
        or geometry.f_insert is None
    ):
        raise ValueError("Append/exchange patches require appended block geometry.")

    K_cross = _float_matrix(geometry.K_insert_cross, name="K_insert_cross")
    K_insert = _float_matrix(geometry.K_insert_insert, name="K_insert_insert")
    f_insert = _float_vector(geometry.f_insert, name="f_insert")
    if K_cross.shape != (n, m):
        raise ValueError(
            f"K_insert_cross must have shape ({n}, {m}); got {K_cross.shape}."
        )
    if K_insert.shape != (m, m):
        raise ValueError(
            f"K_insert_insert must have shape ({m}, {m}); got {K_insert.shape}."
        )
    if f_insert.shape != (m,):
        raise ValueError(f"f_insert must have shape ({m},); got {f_insert.shape}.")

    keep_len = int(len(keep))
    after_size = keep_len + m
    K_after = np.zeros((after_size, after_size), dtype=float)
    if keep_len:
        K_after[:keep_len, :keep_len] = K_before[np.ix_(keep, keep)]
        K_after[:keep_len, keep_len:] = K_cross[list(keep), :]
        K_after[keep_len:, :keep_len] = K_cross[list(keep), :].T
    K_after[keep_len:, keep_len:] = K_insert
    f_after = np.concatenate((np.asarray(f_before[list(keep)], dtype=float), f_insert))
    return K_after, f_after, keep


def prune_conditioning_diagnostics(
    *,
    geometry: SupportPatchGeometry,
    patch: SupportPatch,
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
    schur_inverse_policy: McLachlanInversePolicy | None = None,
    before_cache: SupportPatchBeforeCache | None = None,
    after_cache: SupportPatchAfterCache | None = None,
) -> PruneConditioningDiagnostics:
    """Return whole-batch conditioning pressure diagnostics for a delete patch.

    These diagnostics are for prune nomination only.  They do not define whether
    a deletion is physically safe; that authority remains the grouped deletion
    loss and downstream refit/shadow checks.
    """

    K_before, f_before = _validate_before_geometry(geometry)
    cache = _validate_before_cache(
        before_cache,
        K_before=K_before,
        f_before=f_before,
        inverse_policy=inverse_policy,
    )
    n = int(K_before.shape[0])
    removed = _valid_removed_indices(patch.removed_runtime_indices, size=n)
    retained = tuple(i for i in range(n) if i not in set(removed))
    K_after = np.asarray(K_before[np.ix_(retained, retained)], dtype=float)
    validated_after_cache = _validate_after_cache(
        after_cache,
        K_before=K_before,
        patch=patch,
        K_after=K_after,
        keep=retained,
        inverse_policy=inverse_policy,
    )
    if not removed:
        return PruneConditioningDiagnostics(
            available=False,
            reason="no_removed_runtime_indices",
            removed_runtime_indices=removed,
            retained_runtime_indices=retained,
            condition_number_before=None,
            condition_number_after=None,
            log_condition_before=0.0,
            log_condition_after=0.0,
            d_kappa_rel=0.0,
            d_kappa_dam=0.0,
            d_schur=0.0,
            schur_rank=None,
            schur_candidate_count=0,
            schur_condition_number=None,
            schur_rank_deficit_fraction=0.0,
        )

    try:
        if cache is None:
            before_inverse = _shared_supported_inverse(
                K_before,
                policy=inverse_policy,
            )
        elif cache.before_inverse is None:
            raise ValueError(
                cache.before_inverse_error or "cached before inverse unavailable"
            )
        else:
            before_inverse = cache.before_inverse
        if validated_after_cache is None:
            after_inverse = _shared_supported_inverse(K_after, policy=inverse_policy)
        elif validated_after_cache.after_solve is None:
            raise ValueError(
                validated_after_cache.after_solve_error
                or "cached after inverse unavailable"
            )
        else:
            after_inverse = validated_after_cache.after_solve.inverse
        schur_policy = schur_inverse_policy or inverse_policy
        schur_available_inverse = None
        if _inverse_policy_signature(schur_policy) == _inverse_policy_signature(
            inverse_policy
        ):
            schur_available_inverse = after_inverse.inverse
        schur = schur_novelty(
            K_available=np.asarray(K_before[np.ix_(retained, retained)], dtype=float),
            K_candidate=np.asarray(K_before[np.ix_(removed, removed)], dtype=float),
            K_available_candidate=np.asarray(
                K_before[np.ix_(retained, removed)], dtype=float
            ),
            inverse_policy=schur_policy,
            candidate_ridge_lambda=0.0,
            available_inverse=schur_available_inverse,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return PruneConditioningDiagnostics(
            available=False,
            reason=f"conditioning_diagnostics_failed:{exc}",
            removed_runtime_indices=removed,
            retained_runtime_indices=retained,
            condition_number_before=None,
            condition_number_after=None,
            log_condition_before=0.0,
            log_condition_after=0.0,
            d_kappa_rel=0.0,
            d_kappa_dam=0.0,
            d_schur=0.0,
            schur_rank=None,
            schur_candidate_count=len(removed),
            schur_condition_number=None,
            schur_rank_deficit_fraction=0.0,
        )

    log_before = _safe_log_condition_number(before_inverse.condition_number)
    log_after = _safe_log_condition_number(after_inverse.condition_number)
    d_schur, rank_deficit_fraction = _schur_degeneracy_pressure(schur)
    return PruneConditioningDiagnostics(
        available=True,
        reason="available",
        removed_runtime_indices=removed,
        retained_runtime_indices=retained,
        condition_number_before=before_inverse.condition_number,
        condition_number_after=after_inverse.condition_number,
        log_condition_before=log_before,
        log_condition_after=log_after,
        d_kappa_rel=float(max(0.0, log_before - log_after)),
        d_kappa_dam=float(max(0.0, log_after - log_before)),
        d_schur=float(d_schur),
        schur_rank=int(schur.rank),
        schur_candidate_count=int(schur.candidate_count),
        schur_condition_number=schur.condition_number,
        schur_rank_deficit_fraction=float(rank_deficit_fraction),
    )


def schur_novelty(
    *,
    K_available: np.ndarray | Sequence[Sequence[float]],
    K_candidate: np.ndarray | Sequence[Sequence[float]],
    K_available_candidate: np.ndarray | Sequence[Sequence[float]],
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
    candidate_ridge_lambda: float | None = None,
    available_inverse: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> SchurNovelty:
    """Return checkpoint-local Schur novelty of candidate tangent directions.

    ``K_available`` is the current retained support Gram block, ``K_candidate``
    is the candidate block Gram matrix, and ``K_available_candidate`` is the
    current-support by candidate cross block.
    """

    K_w = _float_matrix(K_available, name="K_available")
    K_a = _float_matrix(K_candidate, name="K_candidate")
    K_wa = _float_matrix(K_available_candidate, name="K_available_candidate")
    if K_w.shape[0] != K_w.shape[1]:
        raise ValueError(f"K_available must be square; got {K_w.shape}.")
    if K_a.shape[0] != K_a.shape[1]:
        raise ValueError(f"K_candidate must be square; got {K_a.shape}.")
    m = int(K_a.shape[0])
    n = int(K_w.shape[0])
    if K_wa.shape != (n, m):
        raise ValueError(
            f"K_available_candidate must have shape ({n}, {m}); got {K_wa.shape}."
        )

    ridge_a = (
        float(inverse_policy.ridge_lambda)
        if candidate_ridge_lambda is None
        else float(candidate_ridge_lambda)
    )
    if not np.isfinite(ridge_a) or ridge_a < 0.0:
        raise ValueError("candidate_ridge_lambda must be finite and non-negative.")

    if n == 0:
        projected = np.zeros((m, m), dtype=float)
    else:
        if available_inverse is None:
            inverse_matrix = _shared_supported_inverse(
                K_w,
                policy=inverse_policy,
            ).inverse
        else:
            inverse_matrix = _float_matrix(
                available_inverse,
                name="available_inverse",
            )
            if inverse_matrix.shape != (n, n):
                raise ValueError(
                    "available_inverse shape does not match K_available: "
                    f"got {inverse_matrix.shape}, expected ({n}, {n})."
                )
        projected = np.asarray(K_wa.T @ inverse_matrix @ K_wa, dtype=float)
    candidate = _apply_ridge(K_a, ridge_a)
    matrix = np.asarray(candidate - projected, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)

    if m == 0:
        eigenvalues = np.zeros(0, dtype=float)
        return SchurNovelty(
            matrix=matrix,
            eigenvalues=eigenvalues,
            rank=0,
            candidate_count=0,
            available_count=n,
            retained_threshold=0.0,
            min_eigenvalue=None,
            max_eigenvalue=None,
            min_retained_eigenvalue=None,
            max_retained_eigenvalue=None,
            condition_number=None,
            full_rank=True,
            psd_within_tolerance=True,
            negative_tolerance=float(inverse_policy.epsilon),
            policy_id=str(inverse_policy.policy_id),
            pinv_rcond=float(inverse_policy.pinv_rcond),
            ridge_lambda=float(inverse_policy.ridge_lambda),
            solve_damping=float(inverse_policy.solve_damping),
        )

    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Schur novelty eigendecomposition failed.") from exc
    eigenvalues = np.asarray(eigenvalues, dtype=float).reshape(-1)
    max_abs = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    retained_threshold = float(inverse_policy.pinv_rcond) * max_abs
    negative_tolerance = max(float(inverse_policy.epsilon), retained_threshold)
    retained = eigenvalues > retained_threshold
    retained_values = eigenvalues[retained]
    rank = int(np.count_nonzero(retained))
    min_retained = (
        None if retained_values.size == 0 else float(np.min(retained_values))
    )
    max_retained = (
        None if retained_values.size == 0 else float(np.max(retained_values))
    )
    condition = None
    if min_retained is not None and max_retained is not None:
        condition = float(max_retained / max(min_retained, float(inverse_policy.epsilon)))

    return SchurNovelty(
        matrix=matrix,
        eigenvalues=eigenvalues,
        rank=rank,
        candidate_count=m,
        available_count=n,
        retained_threshold=retained_threshold,
        min_eigenvalue=float(np.min(eigenvalues)),
        max_eigenvalue=float(np.max(eigenvalues)),
        min_retained_eigenvalue=min_retained,
        max_retained_eigenvalue=max_retained,
        condition_number=condition,
        full_rank=bool(rank == m),
        psd_within_tolerance=bool(float(np.min(eigenvalues)) >= -negative_tolerance),
        negative_tolerance=negative_tolerance,
        policy_id=str(inverse_policy.policy_id),
        pinv_rcond=float(inverse_policy.pinv_rcond),
        ridge_lambda=float(inverse_policy.ridge_lambda),
        solve_damping=float(inverse_policy.solve_damping),
    )


def score_support_patch(
    *,
    geometry: SupportPatchGeometry,
    patch: SupportPatch,
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
    schur_inverse_policy: McLachlanInversePolicy | None = None,
    schur_candidate_ridge_lambda: float | None = None,
    cost_terms: Mapping[str, Any] | None = None,
    cost_weight: float = 0.0,
    before_cache: SupportPatchBeforeCache | None = None,
    after_cache: SupportPatchAfterCache | None = None,
    schur_available_inverse: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> SupportPatchScore:
    """Score an AP-McLachlan support patch from already-estimated arrays."""

    K_before_raw, f_before = _validate_before_geometry(geometry)
    cache = _validate_before_cache(
        before_cache,
        K_before=K_before_raw,
        f_before=f_before,
        inverse_policy=inverse_policy,
    )
    n = int(f_before.size)
    removed = _valid_removed_indices(patch.removed_runtime_indices, size=n)
    before_indices = tuple(range(n))
    denom = _denominator(geometry.norm_b_sq, inverse_policy.epsilon)

    K_after_raw, f_after, keep = build_after_geometry(geometry=geometry, patch=patch)
    validated_after_cache = _validate_after_cache(
        after_cache,
        K_before=K_before_raw,
        patch=patch,
        K_after=K_after_raw,
        keep=keep,
        inverse_policy=inverse_policy,
    )

    schur_score = None
    if int(patch.inserted_count) > 0:
        if geometry.K_insert_cross is None or geometry.K_insert_insert is None:
            raise ValueError("Append/exchange patches require appended block geometry.")
        K_cross = _float_matrix(geometry.K_insert_cross, name="K_insert_cross")
        K_insert = _float_matrix(geometry.K_insert_insert, name="K_insert_insert")
        m = int(patch.inserted_count)
        if K_cross.shape != (n, m):
            raise ValueError(
                "K_insert_cross shape does not match appended count: "
                f"got {K_cross.shape}, expected ({n}, {m})."
            )
        if K_insert.shape != (m, m):
            raise ValueError(
                "K_insert_insert shape does not match appended count: "
                f"got {K_insert.shape}, expected ({m}, {m})."
            )
        schur_score = schur_novelty(
            K_available=np.asarray(K_before_raw[np.ix_(keep, keep)], dtype=float),
            K_candidate=K_insert,
            K_available_candidate=np.asarray(K_cross[list(keep), :], dtype=float),
            inverse_policy=schur_inverse_policy or inverse_policy,
            candidate_ridge_lambda=schur_candidate_ridge_lambda,
            available_inverse=schur_available_inverse,
        )

    if cache is None:
        before_solve, _before_solve_error = _solve_for_support(
            matrix=K_before_raw,
            f_vec=f_before,
            indices=before_indices,
            inverse_policy=inverse_policy,
        )
        rank_before = _matrix_rank_from_symmetric_solve(
            before_solve,
            fallback_matrix=_apply_ridge(
                K_before_raw,
                inverse_policy.ridge_lambda,
            ),
        )
    else:
        before_solve = cache.before_solve
        _before_solve_error = cache.before_solve_error
        rank_before = cache.rank_before
    after_indices = tuple(range(int(f_after.size)))
    if validated_after_cache is None:
        after_solve, after_solve_error = _solve_for_support(
            matrix=K_after_raw,
            f_vec=f_after,
            indices=after_indices,
            inverse_policy=inverse_policy,
        )
        rank_after = _matrix_rank_from_symmetric_solve(
            after_solve,
            fallback_matrix=_apply_ridge(
                K_after_raw,
                inverse_policy.ridge_lambda,
            ),
        )
    else:
        after_solve = validated_after_cache.after_solve
        after_solve_error = validated_after_cache.after_solve_error
        rank_after = validated_after_cache.rank_after
    before_gain = None if before_solve is None else float(before_solve.gamma)
    after_gain = None if after_solve is None else float(after_solve.gamma)
    augmented_confirmation = None
    if int(patch.inserted_count) > 0:
        augmented_confirmation = _augmented_solve_confirmation(
            solve=after_solve,
            solve_error=after_solve_error,
            support_size=int(f_after.size),
            norm_b_sq=float(geometry.norm_b_sq),
            denominator=denom,
            inverse_policy=inverse_policy,
        )

    signed_delta = normalized = insertion_gain = deletion_loss = rank_score = None
    if before_gain is not None and after_gain is not None:
        signed_delta = float(after_gain) - float(before_gain)
        normalized = float(signed_delta / denom)
        insertion_gain = float(max(0.0, signed_delta) / denom)
        deletion_loss = float(max(0.0, -signed_delta) / denom)
        penalty = _cost_penalty(cost_terms)
        cw = float(cost_weight)
        if not np.isfinite(cw):
            raise ValueError("cost_weight must be finite.")
        rank_score = float(normalized - cw * penalty)

    return SupportPatchScore(
        patch_kind=_patch_kind(
            removed_runtime_indices=removed,
            inserted_count=patch.inserted_count,
        ),
        before_indices=before_indices,
        after_indices_before_part=keep,
        removed_runtime_indices=removed,
        inserted_count=int(patch.inserted_count),
        inserted_labels=patch.inserted_labels,
        before_gain=before_gain,
        after_gain=after_gain,
        signed_delta_gain=signed_delta,
        normalized_score=normalized,
        insertion_gain=insertion_gain,
        deletion_loss=deletion_loss,
        denominator=denom,
        denominator_kind=DENOM_NORM_B_PLUS_EPS_V1,
        support_kind=SUPPORT_BEFORE_AFTER_V1,
        score_kind=SUPPORT_PATCH_SCORE_V1,
        pinv_policy_id=str(inverse_policy.policy_id),
        pinv_rcond=float(inverse_policy.pinv_rcond),
        ridge_lambda=float(inverse_policy.ridge_lambda),
        solve_damping=float(inverse_policy.solve_damping),
        rank_before=rank_before,
        rank_after=rank_after,
        cost_terms=dict(cost_terms or {}),
        cost_weight=float(cost_weight),
        rank_score=rank_score,
        schur_novelty=schur_score,
        augmented_solve_confirmation=augmented_confirmation,
    )


def score_patch_payload(
    *,
    geometry: SupportPatchGeometry,
    patch: SupportPatch,
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
    schur_inverse_policy: McLachlanInversePolicy | None = None,
    schur_candidate_ridge_lambda: float | None = None,
    cost_terms: Mapping[str, Any] | None = None,
    cost_weight: float = 0.0,
) -> dict[str, Any]:
    """Return ``score_support_patch(...).to_json_dict()``."""

    return score_support_patch(
        geometry=geometry,
        patch=patch,
        inverse_policy=inverse_policy,
        schur_inverse_policy=schur_inverse_policy,
        schur_candidate_ridge_lambda=schur_candidate_ridge_lambda,
        cost_terms=cost_terms,
        cost_weight=cost_weight,
    ).to_json_dict()


def legacy_prune_payload_from_score(score: SupportPatchScore) -> dict[str, Any]:
    """Return delete-only prune-loss compatibility fields for migration tests."""

    if score.patch_kind != PATCH_DELETE or score.inserted_count != 0:
        raise ValueError("legacy prune payload requires a delete-only support patch.")
    return {
        "prune_loss_selected": _finite_or_none(score.deletion_loss),
        "prune_loss_selected_kind": DAMPED_DELTA_K_V1,
        "prune_loss_delta_k_damped": _finite_or_none(score.deletion_loss),
        "prune_loss_denominator": float(score.denominator),
        "prune_loss_denominator_kind": str(score.denominator_kind),
        "prune_loss_removed_runtime_indices": [
            int(i) for i in score.removed_runtime_indices
        ],
        "prune_loss_support_size": int(len(score.after_indices_before_part)),
        "prune_loss_pinv_policy_id": str(score.pinv_policy_id),
        "prune_loss_pinv_rcond": float(score.pinv_rcond),
        "prune_loss_regularization_lambda": float(score.ridge_lambda),
        "prune_loss_solve_damping": float(score.solve_damping),
    }


__all__ = [
    "AugmentedSolveConfirmation",
    "DAMPED_DELTA_K_V1",
    "DENOM_NORM_B_PLUS_EPS_V1",
    "INVERSE_POLICY_NUMPY_PINV_RCOND_V1",
    "McLachlanInversePolicy",
    "PATCH_APPEND",
    "PATCH_DELETE",
    "PATCH_EXCHANGE",
    "PATCH_INSERT",
    "PATCH_NO_EDIT",
    "PRUNE_CONDITIONING_DIAGNOSTICS_V1",
    "PruneConditioningDiagnostics",
    "SCHUR_NOVELTY_V1",
    "SchurNovelty",
    "SUPPORT_BEFORE_AFTER_V1",
    "SUPPORT_PATCH_SCORE_V1",
    "SupportPatch",
    "SupportPatchAfterCache",
    "SupportPatchBeforeCache",
    "SupportPatchGeometry",
    "SupportPatchScore",
    "build_support_patch_after_cache",
    "build_support_patch_before_cache",
    "build_after_geometry",
    "gain_for_support",
    "legacy_prune_payload_from_score",
    "prune_conditioning_diagnostics",
    "score_patch_payload",
    "score_support_patch",
    "schur_novelty",
]
