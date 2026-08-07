"""Pure prune-loss telemetry for HH realtime AP McLachlan pruning.

This module is intentionally narrow: it computes local McLachlan prune-loss
quantities from already-built geometry arrays. It does not know about controller
state, exact references, projection, persistence, cooldown, or branch commits.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


THEOREM_DELTA_G_V1 = "theorem_delta_g_v1"
DAMPED_DELTA_K_V1 = "damped_delta_k_v1"
LEGACY_PROXY_V1 = "legacy_proxy_v1"
COMPAT_SCHUR_NORMALIZED_V1 = "compat_schur_normalized_v1"
DENOM_NORM_B_PLUS_EPS_V1 = "norm_b_sq_plus_epsilon_v1"
DENOM_MAX_NORM_B_EPS_COMPAT_V1 = "max_norm_b_sq_epsilon_compat_v1"
SUPPORT_FULL_MINUS_REMOVED_V1 = "full_minus_removed_v1"
MATRIX_GRAM_G = "gram_g"
MATRIX_DAMPED_K = "damped_k"
MATRIX_LEGACY_PROXY = "legacy_proxy"
MATRIX_COMPAT_SCHUR_K = "compat_schur_k"
REG_SOURCE_BASELINE_K = "baseline_k"
REG_SOURCE_CONSTRUCTED_FROM_G_LAMBDA = "constructed_from_g_lambda"
REG_SOURCE_NONE = "none"
PINV_POLICY_NUMPY_PINV_RCOND_V1 = "numpy_pinv_rcond_v1"


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _int_tuple(values: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _matrix_rank(matrix: np.ndarray) -> int | None:
    if matrix.size <= 0:
        return 0
    try:
        return int(np.linalg.matrix_rank(matrix))
    except np.linalg.LinAlgError:
        return None


def compute_gain_for_support(
    *,
    matrix: np.ndarray,
    f_vec: np.ndarray,
    indices: Sequence[int],
    pinv_rcond: float,
) -> float | None:
    """Return ``f_A.T @ matrix_A^+ @ f_A`` for a runtime support.

    Empty support has zero gain. Non-finite or invalid solves return ``None`` so
    the caller can distinguish missing telemetry from a real zero.
    """

    idx = [int(i) for i in indices if 0 <= int(i) < int(f_vec.size)]
    if not idx:
        return 0.0
    mat = np.asarray(matrix[np.ix_(idx, idx)], dtype=float)
    force = np.asarray(f_vec[idx], dtype=float).reshape(-1)
    if mat.size <= 0 or force.size <= 0:
        return 0.0
    try:
        pinv = np.linalg.pinv(mat, rcond=float(pinv_rcond))
        value = float(force @ (pinv @ force))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _delta_from_gains(
    *,
    full_gain: float | None,
    reduced_gain: float | None,
    denominator: float,
) -> tuple[float | None, float | None, bool]:
    if full_gain is None or reduced_gain is None or not np.isfinite(float(denominator)):
        return None, None, False
    signed = float(full_gain) - float(reduced_gain)
    clipped = float(max(0.0, signed) / float(denominator))
    return float(clipped), float(signed / float(denominator)), bool(signed < 0.0)


def _support_payload(
    *,
    full_indices: tuple[int, ...],
    removed_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    full_set = set(int(i) for i in full_indices)
    removed = tuple(int(i) for i in removed_indices if int(i) in full_set)
    removed_set = set(removed)
    support = tuple(int(i) for i in full_indices if int(i) not in removed_set)
    return removed, support


def compute_prune_loss_payload(
    *,
    G: np.ndarray | Sequence[Sequence[float]] | None,
    K: np.ndarray | Sequence[Sequence[float]] | None,
    f_vec: np.ndarray | Sequence[float],
    norm_b_sq: float,
    removed_runtime_indices: Sequence[int],
    full_runtime_indices: Sequence[int] | None = None,
    pinv_rcond: float = 1.0e-10,
    regularization_lambda: float = 0.0,
    epsilon: float = 1.0e-14,
    selected_loss: float | None = None,
    selected_loss_kind: str | None = None,
    selected_denominator: float | None = None,
    selected_denominator_kind: str | None = None,
    selected_matrix_for_selection: str | None = None,
    legacy_proxy_loss: float | None = None,
    monotonicity_status: str | None = None,
    include_support_indices: bool = True,
) -> dict[str, Any]:
    """Build JSON-safe prune-loss telemetry without changing controller state."""

    force = np.asarray(f_vec, dtype=float).reshape(-1)
    full_indices = (
        tuple(range(int(force.size)))
        if full_runtime_indices is None
        else _int_tuple(full_runtime_indices)
    )
    removed, support = _support_payload(
        full_indices=full_indices,
        removed_indices=_int_tuple(removed_runtime_indices),
    )
    eps = max(0.0, float(epsilon))
    theorem_denominator = float(norm_b_sq) + eps
    if not np.isfinite(theorem_denominator) or theorem_denominator <= 0.0:
        theorem_denominator = eps if eps > 0.0 else 1.0e-14

    g_matrix = None if G is None else np.asarray(G, dtype=float)
    if g_matrix is not None and g_matrix.shape != (int(force.size), int(force.size)):
        g_matrix = None

    k_matrix = None if K is None else np.asarray(K, dtype=float)
    regularization_source = REG_SOURCE_BASELINE_K
    if k_matrix is not None and k_matrix.shape != (int(force.size), int(force.size)):
        k_matrix = None
    if k_matrix is None and g_matrix is not None:
        k_matrix = np.asarray(
            g_matrix + float(regularization_lambda) * np.eye(int(force.size)),
            dtype=float,
        )
        regularization_source = REG_SOURCE_CONSTRUCTED_FROM_G_LAMBDA
    elif k_matrix is None:
        regularization_source = REG_SOURCE_NONE

    full_gain_g = reduced_gain_g = None
    delta_g = delta_g_signed = None
    delta_g_clip = False
    rank_full_g = rank_reduced_g = None
    if g_matrix is not None:
        full_gain_g = compute_gain_for_support(
            matrix=g_matrix,
            f_vec=force,
            indices=full_indices,
            pinv_rcond=float(pinv_rcond),
        )
        reduced_gain_g = compute_gain_for_support(
            matrix=g_matrix,
            f_vec=force,
            indices=support,
            pinv_rcond=float(pinv_rcond),
        )
        delta_g, delta_g_signed, delta_g_clip = _delta_from_gains(
            full_gain=full_gain_g,
            reduced_gain=reduced_gain_g,
            denominator=theorem_denominator,
        )
        rank_full_g = _matrix_rank(np.asarray(g_matrix[np.ix_(full_indices, full_indices)], dtype=float))
        rank_reduced_g = _matrix_rank(np.asarray(g_matrix[np.ix_(support, support)], dtype=float))

    full_gain_k = reduced_gain_k = None
    delta_k = delta_k_signed = None
    delta_k_clip = False
    rank_full_k = rank_reduced_k = None
    if k_matrix is not None:
        full_gain_k = compute_gain_for_support(
            matrix=k_matrix,
            f_vec=force,
            indices=full_indices,
            pinv_rcond=float(pinv_rcond),
        )
        reduced_gain_k = compute_gain_for_support(
            matrix=k_matrix,
            f_vec=force,
            indices=support,
            pinv_rcond=float(pinv_rcond),
        )
        delta_k, delta_k_signed, delta_k_clip = _delta_from_gains(
            full_gain=full_gain_k,
            reduced_gain=reduced_gain_k,
            denominator=theorem_denominator,
        )
        rank_full_k = _matrix_rank(np.asarray(k_matrix[np.ix_(full_indices, full_indices)], dtype=float))
        rank_reduced_k = _matrix_rank(np.asarray(k_matrix[np.ix_(support, support)], dtype=float))

    selected_kind = str(selected_loss_kind or "")
    selected_value = _finite_or_none(selected_loss)
    if selected_value is None:
        if selected_kind == THEOREM_DELTA_G_V1:
            selected_value = delta_g
        elif selected_kind == DAMPED_DELTA_K_V1:
            selected_value = delta_k
        elif selected_kind == LEGACY_PROXY_V1:
            selected_value = _finite_or_none(legacy_proxy_loss)
    if not selected_kind:
        if selected_value is not None and legacy_proxy_loss is not None:
            selected_kind = LEGACY_PROXY_V1
        elif selected_value is not None:
            selected_kind = COMPAT_SCHUR_NORMALIZED_V1
        else:
            selected_kind = "unselected"

    selected_matrix = str(selected_matrix_for_selection or "")
    if not selected_matrix:
        if selected_kind == THEOREM_DELTA_G_V1:
            selected_matrix = MATRIX_GRAM_G
        elif selected_kind == DAMPED_DELTA_K_V1:
            selected_matrix = MATRIX_DAMPED_K
        elif selected_kind == LEGACY_PROXY_V1:
            selected_matrix = MATRIX_LEGACY_PROXY
        elif selected_kind == COMPAT_SCHUR_NORMALIZED_V1:
            selected_matrix = MATRIX_COMPAT_SCHUR_K
        else:
            selected_matrix = "unknown"

    payload: dict[str, Any] = {
        "prune_loss_delta_g_theorem": delta_g,
        "prune_loss_delta_g_theorem_signed": delta_g_signed,
        "prune_loss_delta_k_damped": delta_k,
        "prune_loss_delta_k_damped_signed": delta_k_signed,
        "prune_loss_legacy_proxy": _finite_or_none(legacy_proxy_loss),
        "prune_loss_selected": selected_value,
        "prune_loss_selected_kind": str(selected_kind),
        "prune_loss_denominator": float(
            selected_denominator
            if selected_denominator is not None and np.isfinite(float(selected_denominator))
            else theorem_denominator
        ),
        "prune_loss_denominator_kind": str(
            selected_denominator_kind or DENOM_NORM_B_PLUS_EPS_V1
        ),
        "prune_loss_theorem_denominator": float(theorem_denominator),
        "prune_loss_theorem_denominator_kind": DENOM_NORM_B_PLUS_EPS_V1,
        "prune_loss_support_kind": SUPPORT_FULL_MINUS_REMOVED_V1,
        "prune_loss_removed_runtime_indices": [int(i) for i in removed],
        "prune_loss_support_size": int(len(support)),
        "prune_loss_matrix_for_selection": str(selected_matrix),
        "prune_loss_pinv_policy_id": PINV_POLICY_NUMPY_PINV_RCOND_V1,
        "prune_loss_pinv_rcond": float(pinv_rcond),
        "prune_loss_regularization_lambda": (
            None if k_matrix is None else float(regularization_lambda)
        ),
        "prune_loss_regularization_source": str(regularization_source),
        "prune_loss_rank_full_g": rank_full_g,
        "prune_loss_rank_reduced_g": rank_reduced_g,
        "prune_loss_rank_full_k": rank_full_k,
        "prune_loss_rank_reduced_k": rank_reduced_k,
        "prune_loss_negative_clip_applied": bool(delta_g_clip or delta_k_clip),
        "prune_loss_monotonicity_status": monotonicity_status,
        "prune_loss_full_gain_g": _finite_or_none(full_gain_g),
        "prune_loss_reduced_gain_g": _finite_or_none(reduced_gain_g),
        "prune_loss_full_gain_k": _finite_or_none(full_gain_k),
        "prune_loss_reduced_gain_k": _finite_or_none(reduced_gain_k),
    }
    if include_support_indices:
        payload["prune_loss_support_runtime_indices"] = [int(i) for i in support]
    else:
        payload["prune_loss_support_runtime_indices_omitted"] = True
    return payload


def selected_prune_loss_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return selected-row mirrors for canonical prune-loss candidate fields."""

    mapping = {
        "prune_loss_selected": "selected_prune_loss",
        "prune_loss_selected_kind": "selected_prune_loss_kind",
        "prune_loss_delta_g_theorem": "selected_prune_loss_delta_g_theorem",
        "prune_loss_delta_k_damped": "selected_prune_loss_delta_k_damped",
        "prune_loss_legacy_proxy": "selected_prune_loss_legacy_proxy",
        "prune_loss_denominator": "selected_prune_loss_denominator",
        "prune_loss_denominator_kind": "selected_prune_loss_denominator_kind",
        "prune_loss_support_kind": "selected_prune_loss_support_kind",
        "prune_loss_removed_runtime_indices": "selected_prune_loss_removed_runtime_indices",
        "prune_loss_support_runtime_indices": "selected_prune_loss_support_runtime_indices",
        "prune_loss_support_size": "selected_prune_loss_support_size",
        "prune_loss_matrix_for_selection": "selected_prune_loss_matrix_for_selection",
        "prune_loss_pinv_policy_id": "selected_prune_loss_pinv_policy_id",
        "prune_loss_pinv_rcond": "selected_prune_loss_pinv_rcond",
        "prune_loss_regularization_lambda": "selected_prune_loss_regularization_lambda",
        "prune_loss_regularization_source": "selected_prune_loss_regularization_source",
        "prune_rank_score": "selected_prune_rank_score",
        "prune_rank_score_kind": "selected_prune_rank_score_kind",
        "prune_rank_score_terms": "selected_prune_rank_score_terms",
    }
    return {
        selected_key: candidate[source_key]
        for source_key, selected_key in mapping.items()
        if source_key in candidate
    }


__all__ = [
    "COMPAT_SCHUR_NORMALIZED_V1",
    "DAMPED_DELTA_K_V1",
    "DENOM_MAX_NORM_B_EPS_COMPAT_V1",
    "DENOM_NORM_B_PLUS_EPS_V1",
    "LEGACY_PROXY_V1",
    "MATRIX_COMPAT_SCHUR_K",
    "MATRIX_LEGACY_PROXY",
    "THEOREM_DELTA_G_V1",
    "compute_gain_for_support",
    "compute_prune_loss_payload",
    "selected_prune_loss_payload",
]
