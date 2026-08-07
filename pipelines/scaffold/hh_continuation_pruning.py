#!/usr/bin/env python3
"""Mature-scaffold pruning helpers for HH continuation."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_types import PruneDecision
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
    JointLinearSolveConfig,
    factor_supported_metric,
    solve_joint_linear_model,
)

PRUNE_POLICY_RECOVERABILITY_LADDER_V1 = "recoverability_ladder_v1"
PRUNE_TOLERANCE_AUTO = "auto"
PRUNE_TOLERANCE_FIXED = "fixed"
PRUNE_TOLERANCE_ADAPTIVE_V1 = "adaptive_v1"
PRUNE_CURVATURE_GUARD_OFF = "off"
PRUNE_CURVATURE_GUARD_CONSERVATIVE_V1 = "conservative_v1"
PRUNE_SCHUR_ROUTE_HESSIAN_COUPLING_V1 = "hessian_coupling_v1"
PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1 = "metric_regularized_v1"
PRUNE_SCHUR_ROUTE_FULL_LOGICAL_FS_TRUST_DELETE_REFIT_V1 = (
    "full_logical_fs_trust_delete_refit_v1"
)
PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1 = "stationary_gw_zero_v1"
PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1 = "gradient_corrected_v1"
PRUNE_METRIC_SCHUR_SOLVE_AFFINE_DELETION_GLOBAL_TRUST_V1 = (
    "affine_deletion_global_trust_v1"
)
PRUNE_METRIC_COST_WEIGHT_ANSATZ_ENTRY_DENOMINATOR_V1 = "ansatz_entry_denominator_v1"
PRUNE_METRIC_COST_WEIGHT_OFF = "off"
AFFINE_DELETION_FS_TRUST_SOLVER_V1 = "full_logical_affine_deletion_fs_trust_v1"
AFFINE_DELETION_FS_TRUST_TRIAL_RECEIPT_V1 = (
    "affine_deletion_fs_trust_same_trial_receipt_v1"
)


@dataclass(frozen=True)
class AffineDeletionFSTrustConfig:
    """Numerical policy for the full-logical affine deletion trust solve.

    Every logical coordinate enters ``G`` before the shared supported-metric
    factorization is applied.  The factorization and trust solve are classical;
    this configuration never requests estimator data.
    """

    rank_relative_tolerance: float = 1.0e-8
    energy_resolution: float = 1.0e-12
    affine_tolerance: float = 1.0e-10
    feasibility_tolerance: float = 1.0e-10
    kkt_residual_accuracy: float = 1.0e-8
    metric_distortion_budget: float = 5.0e-2

    def __post_init__(self) -> None:
        for name in (
            "rank_relative_tolerance",
            "energy_resolution",
            "affine_tolerance",
            "feasibility_tolerance",
            "kkt_residual_accuracy",
            "metric_distortion_budget",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if float(self.kkt_residual_accuracy) <= 0.0:
            raise ValueError("kkt_residual_accuracy must be positive.")
        if float(self.kkt_residual_accuracy) > 1.0:
            raise ValueError("kkt_residual_accuracy must not exceed one.")
        if float(self.metric_distortion_budget) >= 1.0:
            raise ValueError("metric_distortion_budget must be less than one.")


@dataclass(frozen=True)
class AffineDeletionFSTrustResult:
    """Certified classical solution of one affine deletion response model."""

    feasible: bool
    reason: str
    deletion_index: int
    joint_step: np.ndarray
    predicted_energy_change: float
    predicted_reduction: float
    fubini_study_displacement_sq: float
    trust_lambda: float
    metric_damping: float
    telemetry: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": AFFINE_DELETION_FS_TRUST_SOLVER_V1,
            "feasible": bool(self.feasible),
            "reason": str(self.reason),
            "deletion_index": int(self.deletion_index),
            "joint_step": [float(value) for value in self.joint_step.tolist()],
            "predicted_energy_change": float(self.predicted_energy_change),
            "predicted_reduction": float(self.predicted_reduction),
            "fubini_study_displacement_sq": float(
                self.fubini_study_displacement_sq
            ),
            "trust_lambda": float(self.trust_lambda),
            "metric_damping": float(self.metric_damping),
            **dict(self.telemetry),
        }


@dataclass(frozen=True)
class AffineDeletionFSTrustState:
    """Conservative branch-local prune radius and metric damping state."""

    radius: float
    metric_damping: float = 0.0
    update_count: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.radius)) or float(self.radius) <= 0.0:
            raise ValueError("radius must be finite and positive.")
        if (
            not math.isfinite(float(self.metric_damping))
            or float(self.metric_damping) < 0.0
        ):
            raise ValueError("metric_damping must be finite and nonnegative.")
        if int(self.update_count) < 0:
            raise ValueError("update_count must be nonnegative.")


@dataclass(frozen=True)
class AffineDeletionFSTrustUpdateConfig:
    """Contraction-only radius and one-way damping update policy."""

    radius_contraction_factor: float = 0.5
    radius_floor: float = 1.0e-8
    damping_initial_increment: float = 1.0e-6
    damping_growth_factor: float = 2.0
    damping_maximum: float = 1.0e6

    def __post_init__(self) -> None:
        contraction = float(self.radius_contraction_factor)
        if not math.isfinite(contraction) or not 0.0 < contraction <= 1.0:
            raise ValueError(
                "radius_contraction_factor must be finite and in (0, 1]."
            )
        for name in (
            "radius_floor",
            "damping_initial_increment",
            "damping_maximum",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        growth = float(self.damping_growth_factor)
        if not math.isfinite(growth) or growth <= 1.0:
            raise ValueError("damping_growth_factor must be finite and exceed one.")
        if float(self.damping_initial_increment) > float(self.damping_maximum):
            raise ValueError(
                "damping_initial_increment must not exceed damping_maximum."
            )


def _empty_affine_deletion_fs_trust_result(
    *,
    reason: str,
    dimension: int,
    deletion_index: int,
    metric_damping: float,
    telemetry: Mapping[str, Any],
) -> AffineDeletionFSTrustResult:
    return AffineDeletionFSTrustResult(
        feasible=False,
        reason=str(reason),
        deletion_index=int(deletion_index),
        joint_step=np.zeros(int(max(0, dimension)), dtype=float),
        predicted_energy_change=0.0,
        predicted_reduction=0.0,
        fubini_study_displacement_sq=0.0,
        trust_lambda=0.0,
        metric_damping=float(metric_damping),
        telemetry={
            "solver_policy": AFFINE_DELETION_FS_TRUST_SOLVER_V1,
            "classical_quantum_query_charge": 0,
            **dict(telemetry),
        },
    )


def _affine_null_basis(vector: np.ndarray) -> np.ndarray:
    """Return a deterministic orthonormal basis for ``vector.T @ x == 0``."""

    row = np.asarray(vector, dtype=float).reshape(1, -1)
    dimension = int(row.shape[1])
    if dimension == 0:
        return np.zeros((0, 0), dtype=float)
    _u, _singular, vh = np.linalg.svd(row, full_matrices=True)
    return np.asarray(vh[1:, :].T, dtype=float).reshape(dimension, dimension - 1)


def solve_full_logical_affine_deletion_fs_trust(
    *,
    theta: Sequence[float] | np.ndarray,
    gradient: Sequence[float] | np.ndarray,
    hessian: Sequence[Sequence[float]] | np.ndarray,
    metric: Sequence[Sequence[float]] | np.ndarray,
    deletion_index: int,
    trust_radius: float,
    metric_damping: float = 0.0,
    config: AffineDeletionFSTrustConfig | None = None,
) -> AffineDeletionFSTrustResult:
    """Minimize a full-logical affine deletion model in an FS trust ball.

    The solved model is

    ``g.T @ d + 0.5 * d.T @ (H + mu * G) @ d``

    subject to ``d[j] = -theta[j]`` and ``d.T @ G @ d <= rho**2``.
    All logical coordinates enter the raw matrices before genuine Gram-null
    directions are removed by the shared supported-metric factorization.
    """

    resolved = config if config is not None else AffineDeletionFSTrustConfig()
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    gradient_vec = np.asarray(gradient, dtype=float).reshape(-1)
    H_raw = np.asarray(hessian, dtype=float)
    G_raw = np.asarray(metric, dtype=float)
    dimension = int(theta_vec.size)
    index = int(deletion_index)
    radius = float(trust_radius)
    mu = float(metric_damping)
    base_telemetry: dict[str, Any] = {
        "solver_policy": AFFINE_DELETION_FS_TRUST_SOLVER_V1,
        "pre_support_coordinate_count": int(dimension),
        "all_logical_coordinates_entered_before_support_reduction": True,
        "supported_rank_projection_after_full_coordinate_entry": True,
        "deletion_index": int(index),
        "trust_radius": float(radius),
        "metric_damping": float(mu),
        "classical_quantum_query_charge": 0,
    }
    if dimension <= 0:
        raise ValueError("theta must contain at least one logical coordinate.")
    if gradient_vec.size != dimension:
        raise ValueError("gradient length must match theta length.")
    if H_raw.shape != (dimension, dimension):
        raise ValueError("hessian shape must match theta length.")
    if G_raw.shape != (dimension, dimension):
        raise ValueError("metric shape must match theta length.")
    if index < 0 or index >= dimension:
        raise ValueError("deletion_index is out of range.")
    if not (
        np.all(np.isfinite(theta_vec))
        and np.all(np.isfinite(gradient_vec))
        and np.all(np.isfinite(H_raw))
        and np.all(np.isfinite(G_raw))
    ):
        raise ValueError("affine deletion model contains nonfinite values.")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("trust_radius must be finite and nonnegative.")
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError("metric_damping must be finite and nonnegative.")

    H = 0.5 * (H_raw + H_raw.T)
    G = 0.5 * (G_raw + G_raw.T)
    effective_hessian = np.asarray(H + mu * G, dtype=float)
    factor = factor_supported_metric(
        G,
        rank_relative_tolerance=float(resolved.rank_relative_tolerance),
        metric_regularization=0.0,
    )
    base_telemetry.update(
        {
            "metric_support_status": (
                "resolved" if factor.feasible else "unresolved"
            ),
            "metric_support_reason": str(factor.reason),
            "metric_supported_rank": int(factor.rank),
            "metric_retained_mask": [
                bool(value) for value in factor.retained_mask.tolist()
            ],
            "raw_metric_eigenvalues": [
                float(value) for value in factor.raw_eigenvalues.tolist()
            ],
            "supported_metric_provenance_id": str(factor.provenance_id),
        }
    )
    if not factor.feasible:
        return _empty_affine_deletion_fs_trust_result(
            reason=f"metric_support_unresolved::{factor.reason}",
            dimension=dimension,
            deletion_index=index,
            metric_damping=mu,
            telemetry=base_telemetry,
        )

    support_basis = np.asarray(factor.raw_orthonormalizer, dtype=float)
    supported_rank = int(support_basis.shape[1])
    gradient_supported = np.asarray(support_basis.T @ gradient_vec, dtype=float)
    hessian_supported = np.asarray(
        support_basis.T @ effective_hessian @ support_basis,
        dtype=float,
    )
    hessian_supported = 0.5 * (
        hessian_supported + hessian_supported.T
    )
    affine_row = np.asarray(support_basis[index, :], dtype=float).reshape(-1)
    affine_target = float(-theta_vec[index])
    affine_row_norm_sq = float(affine_row @ affine_row)
    affine_scale = float(max(1.0, abs(affine_target), np.linalg.norm(affine_row)))
    affine_zero_tolerance = float(resolved.affine_tolerance * affine_scale)

    if affine_row_norm_sq <= float(affine_zero_tolerance**2):
        if abs(affine_target) > affine_zero_tolerance:
            return _empty_affine_deletion_fs_trust_result(
                reason="deletion_coordinate_not_in_supported_metric_range",
                dimension=dimension,
                deletion_index=index,
                metric_damping=mu,
                telemetry={
                    **base_telemetry,
                    "affine_target": float(affine_target),
                    "affine_row_norm_sq": float(affine_row_norm_sq),
                    "minimum_affine_fubini_study_displacement_sq": None,
                },
            )
        affine_origin = np.zeros(supported_rank, dtype=float)
        affine_null_basis = np.eye(supported_rank, dtype=float)
    else:
        affine_origin = np.asarray(
            affine_target * affine_row / affine_row_norm_sq,
            dtype=float,
        )
        affine_null_basis = _affine_null_basis(affine_row)

    minimum_displacement_sq = float(affine_origin @ affine_origin)
    radius_sq = float(radius * radius)
    feasibility_scale = float(max(1.0, radius_sq, minimum_displacement_sq))
    feasibility_tolerance = float(
        resolved.feasibility_tolerance * feasibility_scale
    )
    base_telemetry.update(
        {
            "affine_target": float(affine_target),
            "affine_row_norm_sq": float(affine_row_norm_sq),
            "minimum_affine_fubini_study_displacement_sq": float(
                minimum_displacement_sq
            ),
            "affine_free_supported_dimension": int(
                affine_null_basis.shape[1]
            ),
        }
    )
    if minimum_displacement_sq > radius_sq + feasibility_tolerance:
        return _empty_affine_deletion_fs_trust_result(
            reason="affine_deletion_outside_trust_radius",
            dimension=dimension,
            deletion_index=index,
            metric_damping=mu,
            telemetry=base_telemetry,
        )

    residual_radius_sq = float(max(0.0, radius_sq - minimum_displacement_sq))
    residual_radius = float(math.sqrt(residual_radius_sq))
    free_dimension = int(affine_null_basis.shape[1])
    reduced_gradient = np.asarray(
        affine_null_basis.T
        @ (gradient_supported + hessian_supported @ affine_origin),
        dtype=float,
    ).reshape(-1)
    reduced_hessian = np.asarray(
        affine_null_basis.T @ hessian_supported @ affine_null_basis,
        dtype=float,
    )
    reduced_hessian = 0.5 * (reduced_hessian + reduced_hessian.T)
    inner_payload: dict[str, Any]
    trust_lambda = 0.0
    if free_dimension == 0 or residual_radius <= math.sqrt(
        np.finfo(float).eps
    ) * max(1.0, radius):
        free_step = np.zeros(free_dimension, dtype=float)
        inner_payload = {
            "feasible": True,
            "reason": "affine_constraint_exhausts_free_trust_radius",
            "joint_step": [float(value) for value in free_step.tolist()],
            "classical_quantum_query_charge": 0,
        }
    else:
        inner_result = solve_joint_linear_model(
            gram=np.eye(free_dimension, dtype=float),
            hessian=np.asarray(reduced_hessian, dtype=float),
            gradient=np.asarray(-reduced_gradient, dtype=float),
            active_coordinate_count=int(free_dimension),
            config=JointLinearSolveConfig(
                policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
                rank_relative_tolerance=float(
                    resolved.rank_relative_tolerance
                ),
                metric_regularization=0.0,
                energy_regularization=float(resolved.energy_resolution),
                max_fubini_study_step=float(residual_radius),
                global_trust_kkt_residual_accuracy=float(
                    resolved.kkt_residual_accuracy
                ),
                global_trust_metric_distortion_budget=float(
                    resolved.metric_distortion_budget
                ),
            ),
        )
        inner_payload = inner_result.as_dict()
        if not inner_result.feasible:
            return _empty_affine_deletion_fs_trust_result(
                reason=f"reduced_trust_solve_unresolved::{inner_result.reason}",
                dimension=dimension,
                deletion_index=index,
                metric_damping=mu,
                telemetry={
                    **base_telemetry,
                    "residual_trust_radius": float(residual_radius),
                    "reduced_trust_solve": inner_payload,
                },
            )
        free_step = np.asarray(inner_result.joint_step, dtype=float).reshape(-1)
        trust_lambda = float(inner_result.trust_lambda)

    supported_step = np.asarray(
        affine_origin + affine_null_basis @ free_step,
        dtype=float,
    )
    joint_step = np.asarray(support_basis @ supported_step, dtype=float).reshape(-1)
    predicted_energy_change = float(
        gradient_vec @ joint_step
        + 0.5 * joint_step @ effective_hessian @ joint_step
    )
    displacement_sq = float(max(0.0, joint_step @ G @ joint_step))
    affine_residual = float(joint_step[index] - affine_target)
    supported_stationarity = np.asarray(
        support_basis.T
        @ (
            gradient_vec
            + effective_hessian @ joint_step
            + trust_lambda * (G @ joint_step)
        ),
        dtype=float,
    )
    tangent_stationarity = np.asarray(
        affine_null_basis.T @ supported_stationarity,
        dtype=float,
    )
    tangent_stationarity_norm = float(np.linalg.norm(tangent_stationarity))
    certified = bool(
        abs(affine_residual) <= affine_zero_tolerance + feasibility_tolerance
        and displacement_sq <= radius_sq + feasibility_tolerance
        and np.all(np.isfinite(joint_step))
        and math.isfinite(predicted_energy_change)
    )
    telemetry = {
        **base_telemetry,
        "residual_trust_radius": float(residual_radius),
        "affine_constraint_residual": float(affine_residual),
        "affine_constraint_tolerance": float(
            affine_zero_tolerance + feasibility_tolerance
        ),
        "fubini_study_displacement_sq": float(displacement_sq),
        "trust_constraint_slack_sq": float(radius_sq - displacement_sq),
        "trust_radius_binding": bool(
            abs(displacement_sq - radius_sq) <= feasibility_tolerance
        ),
        "reduced_tangent_stationarity_norm": float(
            tangent_stationarity_norm
        ),
        "reduced_trust_solve": inner_payload,
        "affine_deletion_certificate_status": (
            "resolved" if certified else "unresolved"
        ),
        "classical_quantum_query_charge": 0,
    }
    if not certified:
        return _empty_affine_deletion_fs_trust_result(
            reason="affine_deletion_certificate_failed",
            dimension=dimension,
            deletion_index=index,
            metric_damping=mu,
            telemetry=telemetry,
        )
    return AffineDeletionFSTrustResult(
        feasible=True,
        reason="full_logical_affine_deletion_fs_trust_solve",
        deletion_index=index,
        joint_step=np.asarray(joint_step, dtype=float),
        predicted_energy_change=float(predicted_energy_change),
        predicted_reduction=float(-predicted_energy_change),
        fubini_study_displacement_sq=float(displacement_sq),
        trust_lambda=float(trust_lambda),
        metric_damping=float(mu),
        telemetry=telemetry,
    )


def initialize_affine_deletion_fs_trust_state(
    *, radius: float
) -> AffineDeletionFSTrustState:
    """Create a new state with the required undamped ``mu=0`` start."""

    return AffineDeletionFSTrustState(
        radius=float(radius),
        metric_damping=0.0,
        update_count=0,
    )


def _same_trial_underprediction_receipt(
    receipt: Mapping[str, Any] | None,
) -> tuple[str, bool, float | None]:
    if receipt is None:
        return "missing_receipt", False, None
    payload = dict(receipt)
    if str(payload.get("schema", "")) != AFFINE_DELETION_FS_TRUST_TRIAL_RECEIPT_V1:
        return "schema_mismatch", False, None
    if not (
        bool(payload.get("prediction_complete", False))
        and bool(payload.get("realization_complete", False))
        and bool(payload.get("energy_receipt_complete", False))
    ):
        return "incomplete_receipt", False, None
    trial_id = str(payload.get("trial_id", "")).strip()
    prediction_trial_id = str(payload.get("prediction_trial_id", "")).strip()
    realization_trial_id = str(payload.get("realization_trial_id", "")).strip()
    if (
        not trial_id
        or prediction_trial_id != trial_id
        or realization_trial_id != trial_id
    ):
        return "trial_identity_mismatch", False, None
    try:
        predicted = float(payload.get("predicted_energy_change"))
        realized = float(payload.get("realized_energy_change"))
        comparison_width = float(payload.get("energy_comparison_width", 0.0))
    except (TypeError, ValueError):
        return "nonfinite_energy_comparison", False, None
    if (
        not math.isfinite(predicted)
        or not math.isfinite(realized)
        or not math.isfinite(comparison_width)
        or comparison_width < 0.0
    ):
        return "nonfinite_energy_comparison", False, None
    underprediction_gap = float(realized - predicted)
    return (
        "complete_same_trial_receipt",
        bool(underprediction_gap > comparison_width),
        float(underprediction_gap),
    )


def update_affine_deletion_fs_trust_state(
    state: AffineDeletionFSTrustState,
    *,
    contract_radius: bool,
    trial_receipt: Mapping[str, Any] | None = None,
    config: AffineDeletionFSTrustUpdateConfig | None = None,
) -> tuple[AffineDeletionFSTrustState, dict[str, Any]]:
    """Apply a contraction-only radius and fail-closed damping update.

    ``mu`` can rise only when one complete receipt links the prediction and
    realized exact energy to the same trial and resolves material
    underprediction.  Missing, mismatched, or incomplete receipts hold ``mu``.
    """

    resolved = (
        config if config is not None else AffineDeletionFSTrustUpdateConfig()
    )
    radius_before = float(state.radius)
    damping_before = float(state.metric_damping)
    if bool(contract_radius):
        contracted = float(
            max(
                float(resolved.radius_floor),
                radius_before * float(resolved.radius_contraction_factor),
            )
        )
        radius_after = float(min(radius_before, contracted))
        radius_reason = (
            "contracted" if radius_after < radius_before else "held_at_floor"
        )
    else:
        radius_after = float(radius_before)
        radius_reason = "hold_no_contraction_requested"

    receipt_status, underpredicted, underprediction_gap = (
        _same_trial_underprediction_receipt(trial_receipt)
    )
    damping_after = float(damping_before)
    if bool(underpredicted) and damping_before < float(resolved.damping_maximum):
        proposed = (
            float(resolved.damping_initial_increment)
            if damping_before == 0.0
            else damping_before * float(resolved.damping_growth_factor)
        )
        damping_after = float(
            min(float(resolved.damping_maximum), max(damping_before, proposed))
        )
        damping_reason = (
            "complete_same_trial_underprediction_increase"
            if damping_after > damping_before
            else "held_at_maximum"
        )
    elif bool(underpredicted):
        damping_reason = "held_at_maximum"
    else:
        damping_reason = f"hold::{receipt_status}"

    next_state = AffineDeletionFSTrustState(
        radius=float(radius_after),
        metric_damping=float(damping_after),
        update_count=int(state.update_count) + 1,
    )
    telemetry = {
        "schema": "affine_deletion_fs_trust_state_update_v1",
        "radius_policy": "contraction_only_v1",
        "radius_before": float(radius_before),
        "radius_after": float(radius_after),
        "radius_action": str(radius_reason),
        "radius_never_increased": bool(radius_after <= radius_before),
        "metric_damping_before": float(damping_before),
        "metric_damping_after": float(damping_after),
        "metric_damping_action": str(damping_reason),
        "metric_damping_never_decreased": bool(damping_after >= damping_before),
        "damping_receipt_status": str(receipt_status),
        "complete_same_trial_underprediction": bool(underpredicted),
        "underprediction_gap": underprediction_gap,
        "update_count_before": int(state.update_count),
        "update_count_after": int(next_state.update_count),
        "classical_quantum_query_charge": 0,
    }
    return next_state, telemetry


@dataclass(frozen=True)
class PruneConfig:
    policy: str = PRUNE_POLICY_RECOVERABILITY_LADDER_V1
    max_candidates: int = 6
    min_candidates: int = 2
    fraction_candidates: float = 0.25
    max_regression: float = 1e-8
    retained_gain_ratio: float = 0.5
    protect_steps: int = 2
    cooldown_steps: int = 2
    local_window_size: int = 4
    old_fraction: float = 0.25
    surrogate_enabled: bool = False
    surrogate_ridge: float = 1e-6
    surrogate_psd_floor: float = 1e-12
    surrogate_curvature_eta: float = 1e-8
    surrogate_monotonicity_tol: float = 1e-10
    surrogate_nomination_gate_enabled: bool = False
    surrogate_nomination_gate_factor: float = 1.0
    surrogate_exact_trial_cap: int = 1
    surrogate_recovery_trust_radius: float = 0.0
    schur_nomination_route: str = PRUNE_SCHUR_ROUTE_HESSIAN_COUPLING_V1
    metric_schur_mu: float = 1e-6
    metric_schur_solve_mode: str = PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1
    metric_schur_cost_weighting: str = PRUNE_METRIC_COST_WEIGHT_ANSATZ_ENTRY_DENOMINATOR_V1
    tolerance_mode_requested: str = PRUNE_TOLERANCE_AUTO
    tolerance_mode: str = PRUNE_TOLERANCE_FIXED
    tolerance_shot_coeff: float = 0.0
    tolerance_screen_coeff: float = 0.01
    tolerance_chem: float = 0.0
    tolerance_rel_coeff: float = 0.0
    tolerance_target_energy: float | None = None


@dataclass(frozen=True)
class StaticPruneCurvatureCache:
    """Damped quasi-Newton curvature cache for static ADAPT prune ranking.

    This cache is intentionally non-authoritative. Its Schur scores may rank
    candidates or suggest compensation windows, but a deletion must still pass
    the measured remove-refit energy check.
    """

    labels: tuple[str, ...]
    hessian: np.ndarray
    last_theta: np.ndarray | None = None
    last_gradient: np.ndarray | None = None
    ridge: float = 1e-6
    psd_floor: float = 1e-12
    update_count: int = 0
    skipped_update_count: int = 0
    health: str = "initialized"
    last_skip_reason: str | None = None
    surrogate_authority: str = "rank_window_diag_only"


def _normalized_prune_policy(policy: str | None) -> str:
    raw = str(policy or PRUNE_POLICY_RECOVERABILITY_LADDER_V1).strip().lower()
    aliases = {
        "recoverability": PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        "recoverability_v1": PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        "ladder": PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
    }
    raw = aliases.get(raw, raw)
    if raw != PRUNE_POLICY_RECOVERABILITY_LADDER_V1:
        raise ValueError(f"Unsupported prune policy: {policy}")
    return str(raw)


def resolve_prune_tolerance_mode(*, mode: str | None, prune_policy: str | None) -> str:
    """Resolve public prune tolerance mode to an effective implementation mode.

    ``auto`` resolves to the adaptive recoverability-ladder tolerance.
    """

    raw = str(mode or PRUNE_TOLERANCE_AUTO).strip().lower()
    aliases = {
        "legacy": PRUNE_TOLERANCE_FIXED,
        "none": PRUNE_TOLERANCE_FIXED,
        "off": PRUNE_TOLERANCE_FIXED,
        "adaptive": PRUNE_TOLERANCE_ADAPTIVE_V1,
        "scale_aware": PRUNE_TOLERANCE_ADAPTIVE_V1,
        "scale-aware": PRUNE_TOLERANCE_ADAPTIVE_V1,
    }
    raw = aliases.get(raw, raw)
    if raw == PRUNE_TOLERANCE_AUTO:
        _normalized_prune_policy(prune_policy)
        return PRUNE_TOLERANCE_ADAPTIVE_V1
    if raw not in {PRUNE_TOLERANCE_FIXED, PRUNE_TOLERANCE_ADAPTIVE_V1}:
        raise ValueError(f"Unsupported prune tolerance mode: {mode}")
    return str(raw)


def _finite_nonnegative_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0.0:
        return None
    return float(out)


def compute_prune_regression_tolerance(
    *,
    delta_num: float,
    mode: str,
    sigma_e: float | None = None,
    delta_scr: float | None = None,
    delta_chem: float | None = None,
    current_energy: float | None = None,
    target_energy: float | None = None,
    c_shot: float = 0.0,
    c_scr: float = 0.0,
    c_rel: float = 0.0,
) -> dict[str, Any]:
    """Compute the scale-aware prune regression tolerance.

    This helper only produces the scalar gate and audit telemetry.  The
    authoritative delete/no-delete decision remains the remove-refit energy
    safety check in :func:`recoverability_prune_ladder`.
    """

    mode_eff = str(mode or PRUNE_TOLERANCE_FIXED).strip().lower()
    if mode_eff not in {PRUNE_TOLERANCE_FIXED, PRUNE_TOLERANCE_ADAPTIVE_V1}:
        raise ValueError(f"Unsupported effective prune tolerance mode: {mode}")
    delta_num_f = _finite_nonnegative_or_none(delta_num)
    delta_num_f = 0.0 if delta_num_f is None else float(delta_num_f)
    components: dict[str, dict[str, Any]] = {
        "delta_num": {
            "value": float(delta_num_f),
            "source": "phase1_prune_max_regression",
        }
    }
    dropped: dict[str, str] = {}

    def _add_scaled_component(
        name: str,
        *,
        raw_value: float | None,
        coeff: float,
    ) -> None:
        coeff_f = _finite_nonnegative_or_none(coeff)
        raw_f = _finite_nonnegative_or_none(raw_value)
        if coeff_f is None or coeff_f <= 0.0:
            dropped[name] = "coefficient_zero_or_unavailable"
            return
        if raw_f is None or raw_f <= 0.0:
            dropped[name] = "raw_value_unavailable"
            return
        components[name] = {
            "value": float(coeff_f * raw_f),
            "raw": float(raw_f),
            "coeff": float(coeff_f),
        }

    if mode_eff == PRUNE_TOLERANCE_ADAPTIVE_V1:
        _add_scaled_component("shot", raw_value=sigma_e, coeff=c_shot)
        _add_scaled_component("screen", raw_value=delta_scr, coeff=c_scr)
        chem_f = _finite_nonnegative_or_none(delta_chem)
        if chem_f is not None and chem_f > 0.0:
            components["chem"] = {
                "value": float(chem_f),
                "source": "explicit_phase1_prune_tolerance_chem",
            }
        else:
            dropped["chem"] = "explicit_value_zero_or_unavailable"
        coeff_rel_f = _finite_nonnegative_or_none(c_rel)
        if coeff_rel_f is None or coeff_rel_f <= 0.0:
            dropped["relative_target"] = "coefficient_zero_or_unavailable"
        elif current_energy is None or target_energy is None:
            dropped["relative_target"] = "target_energy_unavailable"
        else:
            try:
                rel_raw = abs(float(current_energy) - float(target_energy))
            except (TypeError, ValueError):
                rel_raw = math.nan
            rel_raw_f = _finite_nonnegative_or_none(rel_raw)
            if rel_raw_f is None or rel_raw_f <= 0.0:
                dropped["relative_target"] = "raw_value_unavailable"
            else:
                components["relative_target"] = {
                    "value": float(coeff_rel_f * rel_raw_f),
                    "raw": float(rel_raw_f),
                    "coeff": float(coeff_rel_f),
                }
    else:
        dropped.update(
            {
                "shot": "fixed_mode",
                "screen": "fixed_mode",
                "chem": "fixed_mode",
                "relative_target": "fixed_mode",
            }
        )

    used_name, used_payload = max(
        components.items(),
        key=lambda item: float(item[1].get("value", 0.0)),
    )
    return {
        "schema": "prune_regression_tolerance_v1",
        "mode": str(mode_eff),
        "effective_tolerance": float(used_payload.get("value", 0.0)),
        "delta_num": float(delta_num_f),
        "components": components,
        "dropped_components": dropped,
        "used_component": str(used_name),
    }


def evaluate_prune_permission(
    *,
    policy: str | None,
    mode_enabled: bool,
    has_min_scaffold: bool,
    mature_open: bool,
    stable_refit: bool = True,
    accepted_admission: bool = False,
    plateau: bool = False,
    checkpoint_due: bool = False,
    terminal: bool = False,
    snr_low_enough: bool = False,
    snr_adm: float | None = None,
) -> dict[str, Any]:
    """Evaluate Prune-0 permission for static ADAPT pruning.

    For the recoverability-ladder policy, low SNR is only a pressure signal; it
    does not open the deletion cadence.
    """

    policy_eff = _normalized_prune_policy(policy)
    base_gates = {
        "mode_enabled": bool(mode_enabled),
        "has_min_scaffold": bool(has_min_scaffold),
        "stable_refit": bool(stable_refit),
        "mature_open": bool(mature_open),
    }
    base_ok = bool(all(base_gates.values()))
    permission_triggers = {
        "accepted_admission": bool(accepted_admission),
        "plateau": bool(plateau),
        "checkpoint_due": bool(checkpoint_due),
        "terminal": bool(terminal),
    }
    pressure_signals = {
        "snr_low_enough": bool(snr_low_enough),
        "snr_adm": None if snr_adm is None else float(snr_adm),
    }

    reason = "open"
    permission_open = False
    if not base_ok:
        for key, value in base_gates.items():
            if not bool(value):
                reason = str(key)
                break
    else:
        for key in ("accepted_admission", "plateau", "checkpoint_due", "terminal"):
            if bool(permission_triggers[key]):
                permission_open = True
                reason = str(key)
                break
        if not permission_open:
            reason = "awaiting_recoverability_cadence"

    return {
        "permission_schema": "static_prune_permission_v1",
        "permission_policy": str(policy_eff),
        "permission_open": bool(permission_open),
        "permission_reason": str(reason),
        "permission_base_gates": dict(base_gates),
        "permission_triggers": dict(permission_triggers),
        "pressure_signals": dict(pressure_signals),
    }


def _normalized_curvature_guard_mode(mode: str | None) -> str:
    raw = str(mode or PRUNE_CURVATURE_GUARD_OFF).strip().lower()
    aliases = {
        "none": PRUNE_CURVATURE_GUARD_OFF,
        "disabled": PRUNE_CURVATURE_GUARD_OFF,
        "conservative": PRUNE_CURVATURE_GUARD_CONSERVATIVE_V1,
        "conservative-v1": PRUNE_CURVATURE_GUARD_CONSERVATIVE_V1,
    }
    raw = aliases.get(raw, raw)
    if raw not in {PRUNE_CURVATURE_GUARD_OFF, PRUNE_CURVATURE_GUARD_CONSERVATIVE_V1}:
        raise ValueError(f"Unsupported prune curvature guard mode: {mode}")
    return str(raw)


def evaluate_recoverability_curvature_guard(
    *,
    rung_index: int,
    rung_kind: str,
    confidence_upper_regression: float,
    regression_threshold: float,
    mode: str | None = PRUNE_CURVATURE_GUARD_OFF,
    context: Mapping[str, Any] | None = None,
    retained_gain: float | None = None,
    admitted_gain: float | None = None,
    retained_gain_ratio: float = 0.0,
) -> dict[str, Any]:
    """Guard high/curvature-compensated prune rungs.

    The guard is intentionally inactive by default so existing callers preserve
    behavior.  In conservative mode it only applies to noncommuting/terminal
    recoverability rungs, and uses the confidence upper regression.
    """

    mode_eff = _normalized_curvature_guard_mode(mode)
    ctx = dict(context or {})
    rung_kind_eff = str(rung_kind or "")
    rung_kind_key = rung_kind_eff.strip().lower()
    curvature_compensated = bool(
        int(rung_index) >= 3
        or rung_kind_key in {"comm_corr_nc_refit", "terminal_refit"}
        or "nc" in rung_kind_key
    )
    terminal_rung_s4 = bool(int(rung_index) >= 4 or rung_kind_key == "terminal_refit")
    if mode_eff == PRUNE_CURVATURE_GUARD_OFF or not curvature_compensated:
        return {
            "curvature_guard_mode": str(mode_eff),
            "curvature_guard_active": False,
            "curvature_guard_ok": True,
            "curvature_guard_reason": "guard_off" if mode_eff == PRUNE_CURVATURE_GUARD_OFF else "flat_or_commuting_rung",
            "curvature_compensated_rung": bool(curvature_compensated),
            "strict_regression_ok": True,
            "near_terminal_breadth": False,
            "compression_mode": bool(ctx.get("compression_mode", False)),
            "terminal_full": bool(ctx.get("terminal_full", False)),
            "terminal_rung_s4": bool(terminal_rung_s4),
            "strong_retained_gain_ok": False,
            "active_window_fraction": float(ctx.get("active_window_fraction", 0.0) or 0.0),
        }

    upper = float(confidence_upper_regression)
    threshold = float(max(0.0, regression_threshold))
    gamma_curv = float(max(0.0, ctx.get("gamma_curv", ctx.get("strict_regression_factor", 0.25))))
    strict_regression_ok = bool(math.isfinite(upper) and upper <= float(gamma_curv * threshold))
    active_window_fraction = float(max(0.0, min(1.0, ctx.get("active_window_fraction", 0.0) or 0.0)))
    near_terminal_threshold = float(max(0.0, min(1.0, ctx.get("near_terminal_window_fraction", 0.75))))
    near_terminal_breadth = bool(active_window_fraction >= near_terminal_threshold)
    compression_mode = bool(ctx.get("compression_mode", False))
    terminal_full = bool(ctx.get("terminal_full", False))

    admitted = None if admitted_gain is None else float(admitted_gain)
    retained = None if retained_gain is None else float(retained_gain)
    activation = float(max(0.0, ctx.get("retained_gain_activation", 0.0) or 0.0))
    strong_ratio = float(max(1.0, ctx.get("strong_retained_gain_ratio", max(1.0, retained_gain_ratio))))
    strong_retained_gain_ok = bool(
        admitted is not None
        and retained is not None
        and math.isfinite(admitted)
        and math.isfinite(retained)
        and admitted > max(activation, 0.0)
        and retained >= float(strong_ratio * admitted)
    )
    reasons = []
    if strict_regression_ok:
        reasons.append("strict_regression")
    if terminal_rung_s4:
        reasons.append("terminal_rung_s4")
    if compression_mode:
        reasons.append("compression_mode")
    if near_terminal_breadth:
        reasons.append("compression_broad_window" if compression_mode else "near_terminal_breadth")
    if terminal_full:
        reasons.append("terminal_full")
    if strong_retained_gain_ok:
        reasons.append("strong_retained_gain")
    ok = bool(reasons)
    return {
        "curvature_guard_mode": str(mode_eff),
        "curvature_guard_active": True,
        "curvature_guard_ok": bool(ok),
        "curvature_guard_reason": str(reasons[0] if reasons else "curvature_compensated_guard_failed"),
        "curvature_compensated_rung": True,
        "strict_regression_ok": bool(strict_regression_ok),
        "near_terminal_breadth": bool(near_terminal_breadth),
        "compression_mode": bool(compression_mode),
        "terminal_full": bool(terminal_full),
        "terminal_rung_s4": bool(terminal_rung_s4),
        "strong_retained_gain_ok": bool(strong_retained_gain_ok),
        "active_window_fraction": float(active_window_fraction),
        "gamma_curv": float(gamma_curv),
        "strong_retained_gain_ratio": float(strong_ratio),
    }


def cheap_prune_score(*, frozen_regression: float, selector_burden: float) -> float:
    return float(max(0.0, float(frozen_regression)) / (1.0 + max(0.0, float(selector_burden))))


def _sym_psd_project(matrix: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    if arr.size == 0:
        return np.zeros_like(arr, dtype=float)
    sym = 0.5 * (arr + arr.T)
    try:
        vals, vecs = np.linalg.eigh(sym)
    except np.linalg.LinAlgError:
        diag = np.diag(np.maximum(np.diag(sym), float(floor)))
        return np.asarray(diag, dtype=float)
    vals = np.maximum(np.asarray(vals, dtype=float), float(floor))
    return np.asarray((vecs * vals) @ vecs.T, dtype=float)


def initialize_static_prune_curvature_cache(
    *,
    labels: Sequence[str],
    ridge: float = 1e-6,
    psd_floor: float = 1e-12,
    hessian: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> StaticPruneCurvatureCache:
    """Create a PSD/ridged static-prune curvature cache.

    The default is a diagonal ridge cache.  Supplying a Hessian seeds ranking
    evidence only; it does not alter prune acceptance semantics.
    """

    labels_tuple = tuple(str(x) for x in labels)
    n = int(len(labels_tuple))
    ridge_f = float(max(0.0, ridge))
    floor_f = float(max(0.0, psd_floor))
    if hessian is None:
        H = np.eye(n, dtype=float) * max(ridge_f, floor_f)
    else:
        H_in = np.asarray(hessian, dtype=float)
        if H_in.shape != (n, n):
            raise ValueError(f"hessian shape {H_in.shape} does not match {n} labels")
        H = _sym_psd_project(H_in, floor=floor_f) + ridge_f * np.eye(n, dtype=float)
    return StaticPruneCurvatureCache(
        labels=labels_tuple,
        hessian=np.asarray(H, dtype=float),
        ridge=float(ridge_f),
        psd_floor=float(floor_f),
    )


def update_static_prune_curvature_cache(
    cache: StaticPruneCurvatureCache | None,
    *,
    labels: Sequence[str],
    theta: Sequence[float] | np.ndarray,
    gradient: Sequence[float] | np.ndarray,
    ridge: float = 1e-6,
    psd_floor: float = 1e-12,
    curvature_eta: float = 1e-8,
    epsilon: float = 1e-12,
) -> tuple[StaticPruneCurvatureCache, dict[str, Any]]:
    """Apply a damped BFGS update to the static prune curvature cache.

    The update is conservative: shape/label mismatches reset the cache; bad
    secant pairs are damped toward ``H s``; the result is symmetrized and
    projected PSD.  The cache remains ranking/window telemetry only.
    """

    labels_tuple = tuple(str(x) for x in labels)
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    grad_vec = np.asarray(gradient, dtype=float).reshape(-1)
    if theta_vec.size != len(labels_tuple) or grad_vec.size != len(labels_tuple):
        raise ValueError("theta and gradient must match labels")
    if cache is None or tuple(cache.labels) != labels_tuple or cache.hessian.shape != (len(labels_tuple), len(labels_tuple)):
        new_cache = initialize_static_prune_curvature_cache(
            labels=labels_tuple,
            ridge=float(ridge),
            psd_floor=float(psd_floor),
        )
        return (
            StaticPruneCurvatureCache(
                labels=new_cache.labels,
                hessian=new_cache.hessian,
                last_theta=np.asarray(theta_vec, dtype=float),
                last_gradient=np.asarray(grad_vec, dtype=float),
                ridge=float(new_cache.ridge),
                psd_floor=float(new_cache.psd_floor),
                health="reset_scaffold",
                last_skip_reason="scaffold_or_shape_mismatch",
            ),
            {
                "updated": False,
                "reason": "scaffold_or_shape_mismatch",
                "surrogate_authority": "rank_window_diag_only",
            },
        )
    if cache.last_theta is None or cache.last_gradient is None:
        return (
            StaticPruneCurvatureCache(
                labels=labels_tuple,
                hessian=np.asarray(cache.hessian, dtype=float),
                last_theta=np.asarray(theta_vec, dtype=float),
                last_gradient=np.asarray(grad_vec, dtype=float),
                ridge=float(cache.ridge),
                psd_floor=float(cache.psd_floor),
                update_count=int(cache.update_count),
                skipped_update_count=int(cache.skipped_update_count),
                health="seeded",
                surrogate_authority="rank_window_diag_only",
            ),
            {
                "updated": False,
                "reason": "seeded_initial_pair",
                "surrogate_authority": "rank_window_diag_only",
            },
        )

    H = np.asarray(cache.hessian, dtype=float)
    s = np.asarray(theta_vec - np.asarray(cache.last_theta, dtype=float), dtype=float).reshape(-1)
    y_raw = np.asarray(grad_vec - np.asarray(cache.last_gradient, dtype=float), dtype=float).reshape(-1)
    s_norm = float(np.linalg.norm(s))
    y_norm = float(np.linalg.norm(y_raw))
    eta = float(max(0.0, curvature_eta))
    eps = float(max(1e-30, epsilon))
    if s_norm <= eps or y_norm <= eps:
        skipped = int(cache.skipped_update_count + 1)
        return (
            StaticPruneCurvatureCache(
                labels=labels_tuple,
                hessian=H.copy(),
                last_theta=np.asarray(theta_vec, dtype=float),
                last_gradient=np.asarray(grad_vec, dtype=float),
                ridge=float(cache.ridge),
                psd_floor=float(cache.psd_floor),
                update_count=int(cache.update_count),
                skipped_update_count=skipped,
                health="skipped_secant",
                last_skip_reason="zero_secant_norm",
            ),
            {
                "updated": False,
                "reason": "zero_secant_norm",
                "s_norm": float(s_norm),
                "y_norm": float(y_norm),
                "surrogate_authority": "rank_window_diag_only",
            },
        )

    y = y_raw.copy()
    sty_raw = float(np.dot(s, y_raw))
    curvature_floor = float(eta * s_norm * y_norm)
    damped = False
    if sty_raw < curvature_floor:
        Hs = np.asarray(H @ s, dtype=float)
        sHs = float(np.dot(s, Hs))
        # Powell-style damping toward Hs.  This preserves a positive curvature
        # pair without allowing noisy gradients to dominate the cache.
        target = float(max(curvature_floor, eps))
        denom = float(max(sHs - sty_raw, eps))
        beta = float(max(0.0, min(1.0, (sHs - target) / denom)))
        y = beta * y_raw + (1.0 - beta) * Hs
        damped = True
    sty = float(np.dot(s, y))
    if sty <= eps:
        skipped = int(cache.skipped_update_count + 1)
        return (
            StaticPruneCurvatureCache(
                labels=labels_tuple,
                hessian=H.copy(),
                last_theta=np.asarray(theta_vec, dtype=float),
                last_gradient=np.asarray(grad_vec, dtype=float),
                ridge=float(cache.ridge),
                psd_floor=float(cache.psd_floor),
                update_count=int(cache.update_count),
                skipped_update_count=skipped,
                health="skipped_secant",
                last_skip_reason="nonpositive_secant_after_damping",
            ),
            {
                "updated": False,
                "reason": "nonpositive_secant_after_damping",
                "sTy_raw": float(sty_raw),
                "sTy": float(sty),
                "surrogate_authority": "rank_window_diag_only",
            },
        )
    Hs = np.asarray(H @ s, dtype=float)
    sHs = float(np.dot(s, Hs))
    H_new = H.copy()
    if sHs > eps:
        H_new = H_new - np.outer(Hs, Hs) / float(sHs)
    H_new = H_new + np.outer(y, y) / float(sty)
    ridge_f = float(max(0.0, ridge if ridge is not None else cache.ridge))
    floor_f = float(max(0.0, psd_floor if psd_floor is not None else cache.psd_floor))
    H_new = _sym_psd_project(H_new, floor=floor_f) + ridge_f * np.eye(len(labels_tuple), dtype=float)
    updated_cache = StaticPruneCurvatureCache(
        labels=labels_tuple,
        hessian=np.asarray(H_new, dtype=float),
        last_theta=np.asarray(theta_vec, dtype=float),
        last_gradient=np.asarray(grad_vec, dtype=float),
        ridge=float(ridge_f),
        psd_floor=float(floor_f),
        update_count=int(cache.update_count + 1),
        skipped_update_count=int(cache.skipped_update_count),
        health="healthy",
        last_skip_reason=None,
        surrogate_authority="rank_window_diag_only",
    )
    return (
        updated_cache,
        {
            "updated": True,
            "damped": bool(damped),
            "s_norm": float(s_norm),
            "y_norm": float(y_norm),
            "sTy_raw": float(sty_raw),
            "sTy": float(sty),
            "sHs": float(sHs),
            "surrogate_authority": "rank_window_diag_only",
        },
    )


def static_prune_schur_surrogate_ladder(
    *,
    theta: Sequence[float] | np.ndarray,
    hessian: Sequence[Sequence[float]] | np.ndarray,
    block_indices: Sequence[int],
    windows: Sequence[Sequence[int]],
    ridge: float = 1e-6,
    monotonicity_tol: float = 1e-10,
    recovery_trust_radius: float = 0.0,
    gradient: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute non-authoritative Schur prune-loss ladder rows.

    The returned values are ranking/windowing evidence only.  They must not be
    used as deletion certificates.
    """

    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    H = _sym_psd_project(np.asarray(hessian, dtype=float), floor=0.0)
    n = int(theta_vec.size)
    if H.shape != (n, n):
        raise ValueError("hessian shape must match theta")
    gradient_vec: np.ndarray | None = None
    if gradient is not None:
        gradient_vec = np.asarray(gradient, dtype=float).reshape(-1)
        if int(gradient_vec.size) != n:
            raise ValueError("gradient shape must match theta")
        if not np.all(np.isfinite(gradient_vec)):
            raise ValueError("gradient entries must be finite")
    block = [int(i) for i in block_indices if 0 <= int(i) < n]
    if not block:
        return {
            "rows": [],
            "monotone": True,
            "health": "empty_block",
            "surrogate_authority": "rank_window_diag_only",
            "used_for_acceptance": False,
        }
    B = np.asarray(block, dtype=int)
    theta_B = theta_vec[B]
    H_BB = H[np.ix_(B, B)]
    values: list[float] = []
    bounded_values: list[float] = []
    rows: list[dict[str, Any]] = []
    prev_value: float | None = None
    monotone = True
    trust_radius = float(max(0.0, recovery_trust_radius))
    bounded_recovery_active = bool(math.isfinite(trust_radius) and trust_radius > 0.0)

    def _bounded_compensation(
        H_WW: np.ndarray,
        b_vec: np.ndarray,
        *,
        radius: float,
    ) -> tuple[np.ndarray, bool]:
        if b_vec.size == 0:
            return np.zeros(0, dtype=float), False
        try:
            unconstrained = -np.linalg.pinv(H_WW) @ b_vec
        except np.linalg.LinAlgError:
            unconstrained = -np.linalg.pinv(H_WW + float(max(0.0, ridge)) * np.eye(int(H_WW.shape[0]))) @ b_vec
        unconstrained = np.asarray(unconstrained, dtype=float).reshape(-1)
        norm_unconstrained = float(np.linalg.norm(unconstrained))
        if (not bounded_recovery_active) or norm_unconstrained <= float(radius) + 1e-15:
            return unconstrained, False
        if float(radius) <= 0.0:
            return np.zeros_like(unconstrained), True
        eye = np.eye(int(H_WW.shape[0]), dtype=float)

        def _solve(lam: float) -> np.ndarray:
            return -np.linalg.pinv(H_WW + float(lam) * eye) @ b_vec

        lo = 0.0
        hi = 1.0
        for _ in range(80):
            trial = np.asarray(_solve(hi), dtype=float).reshape(-1)
            if float(np.linalg.norm(trial)) <= float(radius):
                break
            hi *= 2.0
        bounded = unconstrained
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            trial = np.asarray(_solve(mid), dtype=float).reshape(-1)
            if float(np.linalg.norm(trial)) > float(radius):
                lo = mid
            else:
                hi = mid
                bounded = trial
        return np.asarray(bounded, dtype=float).reshape(-1), True

    for rung_idx, window_raw in enumerate(windows):
        W_list = [
            int(i)
            for i in window_raw
            if 0 <= int(i) < n and int(i) not in set(block)
        ]
        W = np.asarray(W_list, dtype=int)
        frozen_value = float(0.5 * theta_B.T @ H_BB @ theta_B)
        frozen_value = float(max(0.0, frozen_value))
        compensation_norm = 0.0
        bounded_value = frozen_value
        bounded_active_for_row = False
        bounded_clipped = False
        compensation = np.zeros(int(W.size), dtype=float)
        g_survivor_row: np.ndarray | None = None
        H_WW_raw_row = np.zeros((int(W.size), int(W.size)), dtype=float)
        H_WJ_row = np.zeros((int(W.size), int(B.size)), dtype=float)
        if W.size == 0:
            reduced = H_BB
        else:
            H_BW = H[np.ix_(B, W)]
            H_WW_raw_row = H[np.ix_(W, W)]
            H_WJ_row = H[np.ix_(W, B)]
            H_WW = H_WW_raw_row + float(max(0.0, ridge)) * np.eye(int(W.size), dtype=float)
            b_vec = np.asarray(H_WJ_row @ theta_B, dtype=float).reshape(-1)
            if gradient_vec is not None:
                g_survivor_row = np.asarray(gradient_vec[W], dtype=float).reshape(-1)
            compensation, bounded_clipped = _bounded_compensation(
                H_WW,
                b_vec,
                radius=float(trust_radius),
            )
            compensation_norm = float(np.linalg.norm(compensation))
            bounded_value = float(
                frozen_value
                + float(b_vec.T @ compensation)
                + 0.5 * float(compensation.T @ H_WW @ compensation)
            )
            bounded_value = float(max(0.0, bounded_value))
            bounded_active_for_row = bool(bounded_recovery_active)
            reduced = H_BB - H_BW @ np.linalg.pinv(H_WW) @ H_BW.T
        value = float(0.5 * theta_B.T @ reduced @ theta_B)
        value = float(max(0.0, value))
        if not bounded_recovery_active:
            bounded_value = float(value)
        elif W.size == 0:
            bounded_value = float(value)
        if prev_value is not None and value > prev_value + float(max(0.0, monotonicity_tol)):
            monotone = False
        prev_value = float(value)
        values.append(float(value))
        bounded_values.append(float(bounded_value))
        rows.append(
            {
                "rung_index": int(rung_idx),
                "window_indices": [int(x) for x in W_list],
                "schur_value": float(value),
                "frozen_value": float(frozen_value),
                "bounded_value": float(bounded_value),
                "bounded_recovery_active": bool(bounded_active_for_row),
                "bounded_recovery_clipped": bool(bounded_clipped),
                "recovery_trust_radius": float(trust_radius),
                "compensation_norm": float(compensation_norm),
                "theta_removed": [float(x) for x in np.asarray(theta_B, dtype=float).reshape(-1).tolist()],
                "survivor_window_indices": [int(x) for x in W_list],
                "g_survivor": (
                    None
                    if g_survivor_row is None
                    else [float(x) for x in np.asarray(g_survivor_row, dtype=float).reshape(-1).tolist()]
                ),
                "H_survivor_survivor": [
                    [float(x) for x in row]
                    for row in np.asarray(H_WW_raw_row, dtype=float).reshape(int(W.size), int(W.size)).tolist()
                ],
                "H_survivor_removed": [
                    [float(x) for x in row]
                    for row in np.asarray(H_WJ_row, dtype=float).reshape(int(W.size), int(B.size)).tolist()
                ],
                "ridge_used": float(max(0.0, ridge)),
                "compensation_solve": [
                    float(x) for x in np.asarray(compensation, dtype=float).reshape(-1).tolist()
                ],
                "warm_start_compensation_solve": [
                    float(x) for x in np.asarray(-compensation, dtype=float).reshape(-1).tolist()
                ],
                "compensation_semantics": (
                    "coupling_only_trust_limited"
                    if bool(bounded_active_for_row)
                    else "coupling_only_unbounded"
                ),
                "surrogate_authority": "rank_window_diag_only",
                "used_for_acceptance": False,
            }
        )
    return {
        "rows": rows,
        "values": [float(x) for x in values],
        "bounded_values": [float(x) for x in bounded_values],
        "bounded_recovery_active": bool(bounded_recovery_active),
        "recovery_trust_radius": float(trust_radius),
        "monotone": bool(monotone),
        "health": "ok" if bool(monotone) else "nonmonotone",
        "surrogate_authority": "rank_window_diag_only",
        "used_for_acceptance": False,
    }


def _normalize_metric_schur_solve_mode(mode: str | None) -> str:
    raw = str(mode or PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1).strip().lower()
    aliases = {
        "stationary": PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
        "stationary_v1": PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
        "gw_zero": PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
        "g_w_zero": PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
        "gradient": PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
        "gradient_corrected": PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
        "grad_corrected": PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
    }
    raw = aliases.get(raw, raw)
    if raw not in {
        PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
        PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
    }:
        raise ValueError(f"Unsupported metric Schur solve mode: {mode}")
    return str(raw)


def _sanitize_entry_cost_denominators(
    values: Sequence[float] | np.ndarray | None,
    *,
    n: int,
) -> list[float]:
    if values is None:
        return [1.0 for _ in range(int(n))]
    out: list[float] = []
    for raw in list(values)[: int(n)]:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = 1.0
        if not math.isfinite(val) or val <= 0.0:
            val = 1.0
        out.append(float(max(1.0, val)))
    while len(out) < int(n):
        out.append(1.0)
    return out


def metric_regularized_prune_schur_surrogate_ladder(
    *,
    theta: Sequence[float] | np.ndarray,
    hessian: Sequence[Sequence[float]] | np.ndarray,
    metric: Sequence[Sequence[float]] | np.ndarray,
    block_indices: Sequence[int],
    windows: Sequence[Sequence[int]],
    ridge: float = 1e-6,
    metric_mu: float = 1e-6,
    solve_mode: str = PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
    monotonicity_tol: float = 1e-10,
    recovery_trust_radius: float = 0.0,
    gradient: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute metric-regularized Schur prune-loss ladder rows.

    This is still a nomination model.  It ranks or filters delete trials, while
    measured remove/refit energy safety remains the deletion authority.
    """

    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    n = int(theta_vec.size)
    H = _sym_psd_project(np.asarray(hessian, dtype=float), floor=0.0)
    G = _sym_psd_project(np.asarray(metric, dtype=float), floor=0.0)
    if H.shape != (n, n):
        raise ValueError("hessian shape must match theta")
    if G.shape != (n, n):
        raise ValueError("metric shape must match theta")
    mu = float(max(0.0, float(metric_mu)))
    solve_mode_key = _normalize_metric_schur_solve_mode(solve_mode)
    gradient_vec: np.ndarray | None = None
    if gradient is not None:
        gradient_vec = np.asarray(gradient, dtype=float).reshape(-1)
        if int(gradient_vec.size) != n:
            raise ValueError("gradient shape must match theta")
        if not np.all(np.isfinite(gradient_vec)):
            raise ValueError("gradient entries must be finite")
    block = [int(i) for i in block_indices if 0 <= int(i) < n]
    if not block:
        return {
            "rows": [],
            "values": [],
            "bounded_values": [],
            "monotone": True,
            "health": "empty_block",
            "schur_model": PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
            "surrogate_authority": "metric_regularized_rank_window_diag_only",
            "used_for_acceptance": False,
        }
    B = np.asarray(block, dtype=int)
    theta_B = theta_vec[B]
    A = H + float(mu) * G
    A_BB = A[np.ix_(B, B)]
    G_BB = G[np.ix_(B, B)]
    values: list[float] = []
    bounded_values: list[float] = []
    rows: list[dict[str, Any]] = []
    prev_value: float | None = None
    monotone = True
    trust_radius = float(max(0.0, recovery_trust_radius))
    bounded_recovery_active = bool(math.isfinite(trust_radius) and trust_radius > 0.0)
    block_set = {int(x) for x in block}

    def _solve_response(
        R: np.ndarray,
        rhs: np.ndarray,
        *,
        radius: float,
    ) -> tuple[np.ndarray, bool]:
        rhs_vec = np.asarray(rhs, dtype=float).reshape(-1)
        if rhs_vec.size == 0:
            return np.zeros(0, dtype=float), False
        R_use = 0.5 * (np.asarray(R, dtype=float) + np.asarray(R, dtype=float).T)
        try:
            unconstrained = np.linalg.pinv(R_use) @ rhs_vec
        except np.linalg.LinAlgError:
            unconstrained = np.linalg.pinv(
                R_use + float(max(0.0, ridge)) * np.eye(int(R_use.shape[0]))
            ) @ rhs_vec
        unconstrained = np.asarray(unconstrained, dtype=float).reshape(-1)
        norm_unconstrained = float(np.linalg.norm(unconstrained))
        if (not bounded_recovery_active) or norm_unconstrained <= float(radius) + 1e-15:
            return unconstrained, False
        if float(radius) <= 0.0:
            return np.zeros_like(unconstrained), True
        eye = np.eye(int(R_use.shape[0]), dtype=float)

        def _solve(lam: float) -> np.ndarray:
            return np.linalg.pinv(R_use + float(lam) * eye) @ rhs_vec

        lo = 0.0
        hi = 1.0
        bounded = unconstrained
        for _ in range(80):
            trial = np.asarray(_solve(hi), dtype=float).reshape(-1)
            if float(np.linalg.norm(trial)) <= float(radius):
                break
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            trial = np.asarray(_solve(mid), dtype=float).reshape(-1)
            if float(np.linalg.norm(trial)) > float(radius):
                lo = mid
            else:
                hi = mid
                bounded = trial
        return np.asarray(bounded, dtype=float).reshape(-1), True

    def _loss_terms(
        *,
        W: np.ndarray,
        R: np.ndarray,
        rhs: np.ndarray,
        delta: np.ndarray,
    ) -> tuple[float, float, float, float]:
        frozen = float(0.5 * theta_B.T @ A_BB @ theta_B)
        frozen = float(max(0.0, frozen))
        delta_vec = np.asarray(delta, dtype=float).reshape(-1)
        rhs_vec = np.asarray(rhs, dtype=float).reshape(-1)
        R_use = 0.5 * (np.asarray(R, dtype=float) + np.asarray(R, dtype=float).T)
        credit = float(rhs_vec.T @ delta_vec - 0.5 * delta_vec.T @ R_use @ delta_vec)
        loss = float(max(0.0, frozen - credit))
        if W.size == 0:
            metric_residual = float(theta_B.T @ G_BB @ theta_B)
        else:
            G_BW = G[np.ix_(B, W)]
            G_WW = G[np.ix_(W, W)]
            metric_residual = float(
                theta_B.T @ G_BB @ theta_B
                - 2.0 * theta_B.T @ G_BW @ delta_vec
                + delta_vec.T @ G_WW @ delta_vec
            )
        metric_residual = float(max(0.0, metric_residual))
        return float(loss), float(frozen), float(credit), float(metric_residual)

    for rung_idx, window_raw in enumerate(windows):
        W_list = [
            int(i)
            for i in window_raw
            if 0 <= int(i) < n and int(i) not in block_set
        ]
        W = np.asarray(W_list, dtype=int)
        if W.size == 0:
            R_raw = np.zeros((0, 0), dtype=float)
            R = np.zeros((0, 0), dtype=float)
            rhs = np.zeros(0, dtype=float)
            g_survivor_row = None
            delta_unbounded = np.zeros(0, dtype=float)
            delta_bounded = np.zeros(0, dtype=float)
            bounded_clipped = False
        else:
            R_raw = A[np.ix_(W, W)]
            R = R_raw + float(max(0.0, ridge)) * np.eye(int(W.size), dtype=float)
            rhs = np.asarray(A[np.ix_(W, B)] @ theta_B, dtype=float).reshape(-1)
            g_survivor_row = (
                None
                if gradient_vec is None
                else np.asarray(gradient_vec[W], dtype=float).reshape(-1)
            )
            if (
                solve_mode_key == PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1
                and g_survivor_row is not None
            ):
                rhs = np.asarray(rhs - g_survivor_row, dtype=float).reshape(-1)
            try:
                delta_unbounded = np.asarray(np.linalg.pinv(R) @ rhs, dtype=float).reshape(-1)
            except np.linalg.LinAlgError:
                delta_unbounded = np.asarray(
                    np.linalg.pinv(R + float(max(0.0, ridge)) * np.eye(int(W.size))) @ rhs,
                    dtype=float,
                ).reshape(-1)
            delta_bounded, bounded_clipped = _solve_response(
                R,
                rhs,
                radius=float(trust_radius),
            )
        value, frozen_value, credit_value, metric_residual = _loss_terms(
            W=W,
            R=R,
            rhs=rhs,
            delta=delta_unbounded,
        )
        bounded_value, _frozen_b, bounded_credit, bounded_metric_residual = _loss_terms(
            W=W,
            R=R,
            rhs=rhs,
            delta=delta_bounded,
        )
        if not bounded_recovery_active or W.size == 0:
            bounded_value = float(value)
            bounded_credit = float(credit_value)
            bounded_metric_residual = float(metric_residual)
            delta_bounded = np.asarray(delta_unbounded, dtype=float).reshape(-1)
            bounded_clipped = False
        if prev_value is not None and value > prev_value + float(max(0.0, monotonicity_tol)):
            monotone = False
        prev_value = float(value)
        values.append(float(value))
        bounded_values.append(float(bounded_value))
        rows.append(
            {
                "rung_index": int(rung_idx),
                "window_indices": [int(x) for x in W_list],
                "schur_value": float(value),
                "metric_regularized_loss": float(value),
                "frozen_value": float(frozen_value),
                "credit_value": float(credit_value),
                "bounded_value": float(bounded_value),
                "bounded_credit_value": float(bounded_credit),
                "metric_residual": float(metric_residual),
                "bounded_metric_residual": float(bounded_metric_residual),
                "bounded_recovery_active": bool(bounded_recovery_active),
                "bounded_recovery_clipped": bool(bounded_clipped),
                "recovery_trust_radius": float(trust_radius),
                "compensation_norm": float(np.linalg.norm(delta_bounded)),
                "theta_removed": [
                    float(x) for x in np.asarray(theta_B, dtype=float).reshape(-1).tolist()
                ],
                "survivor_window_indices": [int(x) for x in W_list],
                "g_survivor": (
                    None
                    if g_survivor_row is None
                    else [float(x) for x in np.asarray(g_survivor_row, dtype=float).reshape(-1).tolist()]
                ),
                "H_survivor_survivor": [
                    [float(x) for x in row]
                    for row in np.asarray(H[np.ix_(W, W)], dtype=float).reshape(int(W.size), int(W.size)).tolist()
                ],
                "H_survivor_removed": [
                    [float(x) for x in row]
                    for row in np.asarray(H[np.ix_(W, B)], dtype=float).reshape(int(W.size), int(B.size)).tolist()
                ],
                "G_survivor_survivor": [
                    [float(x) for x in row]
                    for row in np.asarray(G[np.ix_(W, W)], dtype=float).reshape(int(W.size), int(W.size)).tolist()
                ],
                "G_survivor_removed": [
                    [float(x) for x in row]
                    for row in np.asarray(G[np.ix_(W, B)], dtype=float).reshape(int(W.size), int(B.size)).tolist()
                ],
                "A_survivor_survivor": [
                    [float(x) for x in row]
                    for row in np.asarray(R_raw, dtype=float).reshape(int(W.size), int(W.size)).tolist()
                ],
                "rhs": [float(x) for x in np.asarray(rhs, dtype=float).reshape(-1).tolist()],
                "ridge_used": float(max(0.0, ridge)),
                "metric_mu": float(mu),
                "metric_schur_solve_mode": str(solve_mode_key),
                "compensation_solve": [
                    float(x) for x in np.asarray(delta_bounded, dtype=float).reshape(-1).tolist()
                ],
                "warm_start_compensation_solve": [
                    float(x) for x in np.asarray(delta_bounded, dtype=float).reshape(-1).tolist()
                ],
                "compensation_semantics": "metric_regularized_added_survivor_delta",
                "schur_model": PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
                "surrogate_authority": "metric_regularized_rank_window_diag_only",
                "used_for_acceptance": False,
            }
        )
    return {
        "rows": rows,
        "values": [float(x) for x in values],
        "bounded_values": [float(x) for x in bounded_values],
        "bounded_recovery_active": bool(bounded_recovery_active),
        "recovery_trust_radius": float(trust_radius),
        "metric_mu": float(mu),
        "metric_schur_solve_mode": str(solve_mode_key),
        "monotone": bool(monotone),
        "health": "ok" if bool(monotone) else "nonmonotone",
        "schur_model": PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
        "surrogate_authority": "metric_regularized_rank_window_diag_only",
        "used_for_acceptance": False,
    }


def build_metric_regularized_prune_surrogate_scores(
    *,
    theta: Sequence[float] | np.ndarray,
    labels: Sequence[str],
    hessian: Sequence[Sequence[float]] | np.ndarray,
    metric: Sequence[Sequence[float]] | np.ndarray,
    gradient: Sequence[float] | np.ndarray | None = None,
    local_window_size: int = 4,
    ridge: float = 1e-6,
    metric_mu: float = 1e-6,
    solve_mode: str = PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
    monotonicity_tol: float = 1e-10,
    recovery_trust_radius: float = 0.0,
    entry_cost_denominators: Sequence[float] | np.ndarray | None = None,
) -> dict[int, dict[str, Any]]:
    """Build scalar cost-weighted metric-Schur prune nomination rows."""

    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    labels_list = [str(x) for x in labels]
    n = int(theta_vec.size)
    if len(labels_list) != n:
        raise ValueError("labels must match theta")
    H = np.asarray(hessian, dtype=float)
    G = np.asarray(metric, dtype=float)
    if H.shape != (n, n):
        raise ValueError("hessian shape must match theta")
    if G.shape != (n, n):
        raise ValueError("metric shape must match theta")
    gradient_vec: np.ndarray | None = None
    if gradient is not None:
        gradient_vec = np.asarray(gradient, dtype=float).reshape(-1)
        if int(gradient_vec.size) != n:
            raise ValueError("gradient must match theta")
    denominators = _sanitize_entry_cost_denominators(entry_cost_denominators, n=n)
    out: dict[int, dict[str, Any]] = {}
    for idx in range(n):
        survivor = [j for j in range(n) if int(j) != int(idx)]
        local_window_raw = int(local_window_size)
        if local_window_raw <= 0:
            local = list(survivor)
        else:
            omega = int(max(0, min(local_window_raw, len(survivor))))
            if omega <= 0:
                local = []
            else:
                start = max(0, min(int(idx) - (omega - 1) // 2, len(survivor) - omega))
                local = survivor[start : start + omega]
        windows = [[], list(local), survivor]
        ladder = metric_regularized_prune_schur_surrogate_ladder(
            theta=theta_vec,
            hessian=H,
            metric=G,
            block_indices=[int(idx)],
            windows=windows,
            ridge=float(ridge),
            metric_mu=float(metric_mu),
            solve_mode=str(solve_mode),
            monotonicity_tol=float(monotonicity_tol),
            recovery_trust_radius=float(recovery_trust_radius),
            gradient=gradient_vec,
        )
        values = [float(x) for x in ladder.get("values", [])]
        bounded_values = [float(x) for x in ladder.get("bounded_values", [])]
        bounded_active = bool(ladder.get("bounded_recovery_active", False))
        schur_score = float(min(values)) if values else float("inf")
        bounded_score = float(min(bounded_values)) if bounded_values else float(schur_score)
        unweighted_score = float(bounded_score if bounded_active else schur_score)
        denominator = float(denominators[int(idx)])
        score = float(unweighted_score / max(1.0, denominator))
        out[int(idx)] = {
            "index": int(idx),
            "label": str(labels_list[int(idx)]),
            "score": float(score),
            "unweighted_score": float(unweighted_score),
            "schur_min": float(schur_score),
            "bounded_score": float(bounded_score),
            "entry_cost_denominator": float(denominator),
            "cost_weighting": PRUNE_METRIC_COST_WEIGHT_ANSATZ_ENTRY_DENOMINATOR_V1,
            "bounded_recovery_active": bool(bounded_active),
            "recovery_trust_radius": float(ladder.get("recovery_trust_radius", 0.0)),
            "metric_mu": float(ladder.get("metric_mu", metric_mu)),
            "metric_schur_solve_mode": str(ladder.get("metric_schur_solve_mode", solve_mode)),
            "schur_rows": list(ladder.get("rows", [])),
            "schur_health": str(ladder.get("health", "unavailable")),
            "schur_monotone": bool(ladder.get("monotone", False)),
            "schur_model": PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
            "surrogate_authority": "metric_regularized_rank_window_diag_only",
            "used_for_acceptance": False,
        }
    return out


def build_static_prune_surrogate_scores(
    *,
    theta: Sequence[float] | np.ndarray,
    labels: Sequence[str],
    hessian: Sequence[Sequence[float]] | np.ndarray,
    gradient: Sequence[float] | np.ndarray | None = None,
    local_window_size: int = 4,
    ridge: float = 1e-6,
    monotonicity_tol: float = 1e-10,
    recovery_trust_radius: float = 0.0,
) -> dict[int, dict[str, Any]]:
    """Build per-coordinate Schur surrogate ranking rows.

    This helper deliberately returns dictionaries rather than mutating prune
    decisions.  Callers may pass the rows to ``rank_prune_candidates`` as
    ``surrogate_scores``; the measured recoverability ladder remains the
    commit boundary.
    """

    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    labels_list = [str(x) for x in labels]
    n = int(theta_vec.size)
    if len(labels_list) != n:
        raise ValueError("labels must match theta")
    H = np.asarray(hessian, dtype=float)
    if H.shape != (n, n):
        raise ValueError("hessian shape must match theta")
    gradient_vec: np.ndarray | None = None
    if gradient is not None:
        gradient_vec = np.asarray(gradient, dtype=float).reshape(-1)
        if int(gradient_vec.size) != n:
            raise ValueError("gradient must match theta")
    out: dict[int, dict[str, Any]] = {}
    for idx in range(n):
        survivor = [j for j in range(n) if int(j) != int(idx)]
        local_window_raw = int(local_window_size)
        if local_window_raw <= 0:
            local = list(survivor)
        else:
            omega = int(max(0, min(local_window_raw, len(survivor))))
            if omega <= 0:
                local = []
            else:
                start = max(0, min(int(idx) - (omega - 1) // 2, len(survivor) - omega))
                local = survivor[start : start + omega]
        if not local:
            local = []
        windows = [[], list(local), survivor]
        ladder = static_prune_schur_surrogate_ladder(
            theta=theta_vec,
            hessian=H,
            block_indices=[int(idx)],
            windows=windows,
            ridge=float(ridge),
            monotonicity_tol=float(monotonicity_tol),
            recovery_trust_radius=float(recovery_trust_radius),
            gradient=gradient_vec,
        )
        values = [float(x) for x in ladder.get("values", [])]
        bounded_values = [float(x) for x in ladder.get("bounded_values", [])]
        bounded_active = bool(ladder.get("bounded_recovery_active", False))
        schur_score = float(min(values)) if values else float("inf")
        bounded_score = float(min(bounded_values)) if bounded_values else float(schur_score)
        score = float(bounded_score if bounded_active else schur_score)
        out[int(idx)] = {
            "index": int(idx),
            "label": str(labels_list[int(idx)]),
            "score": float(score),
            "schur_min": float(schur_score),
            "bounded_score": float(bounded_score),
            "bounded_recovery_active": bool(bounded_active),
            "recovery_trust_radius": float(ladder.get("recovery_trust_radius", 0.0)),
            "schur_rows": list(ladder.get("rows", [])),
            "schur_health": str(ladder.get("health", "unavailable")),
            "schur_monotone": bool(ladder.get("monotone", False)),
            "surrogate_authority": "rank_window_diag_only",
            "used_for_acceptance": False,
        }
    return out


def rank_prune_candidates(
    *,
    theta: np.ndarray,
    labels: list[str],
    marginal_proxy_benefit: list[float] | None,
    max_candidates: int,
    min_candidates: int,
    fraction_candidates: float,
    selector_burden: Sequence[float] | None = None,
    admission_steps: Sequence[int] | None = None,
    cooldown_remaining: Sequence[int] | None = None,
    current_step: int | None = None,
    protect_steps: int = 0,
    policy: str = PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
    surrogate_scores: Mapping[int, Mapping[str, Any]] | None = None,
    surrogate_score_threshold: float | None = None,
    surrogate_candidate_cap: int | None = None,
    surrogate_score_primary_only: bool = False,
) -> list[int]:
    n = int(theta.size)
    if n <= 0:
        return []
    _normalized_prune_policy(policy)
    benefits = list(marginal_proxy_benefit) if marginal_proxy_benefit is not None else []

    def _benefit_key(i: int) -> float:
        if i >= len(benefits):
            return float("inf")
        val = float(benefits[i])
        if not np.isfinite(val):
            return float("inf")
        return float(val)

    surrogate_rows: Mapping[int, Mapping[str, Any]] = surrogate_scores or {}

    def _surrogate_key(i: int) -> float:
        row = surrogate_rows.get(int(i))
        if not isinstance(row, Mapping):
            return float("inf")
        for key in ("score", "schur_min", "schur_score", "surrogate_score"):
            if key in row:
                try:
                    val = float(row[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(val):
                    return float(val)
        return float("inf")

    def _surrogate_gated(indices: Sequence[int]) -> list[int]:
        if (
            not surrogate_rows
            or surrogate_score_threshold is None
        ):
            out = [int(i) for i in indices]
        else:
            try:
                threshold = float(surrogate_score_threshold)
            except (TypeError, ValueError):
                threshold = float("nan")
            if not np.isfinite(threshold) or threshold < 0.0:
                return []
            out = [
                int(i)
                for i in indices
                if np.isfinite(_surrogate_key(int(i)))
                and float(_surrogate_key(int(i))) <= float(threshold) + 1e-15
            ]
        if (
            surrogate_candidate_cap is not None
        ):
            try:
                cap = int(surrogate_candidate_cap)
            except (TypeError, ValueError):
                cap = 0
            if cap <= 0:
                return []
            out = [int(i) for i in out[:cap]]
        return [int(i) for i in out]

    target = int(np.ceil(float(fraction_candidates) * float(n)))
    target = max(int(min_candidates), target)
    target = min(int(max_candidates), target, n)

    metadata_ready = all(
        seq is not None
        for seq in (
            admission_steps,
            cooldown_remaining,
        )
    ) and current_step is not None
    if not metadata_ready:
        order = sorted(
            range(n),
            key=lambda i: (
                _surrogate_key(int(i)),
                _benefit_key(int(i)),
                str(labels[i]),
            ),
        )
        return _surrogate_gated([int(i) for i in order[:target]])

    admission_steps_list = [int(x) for x in list(admission_steps or [])[:n]]
    cooldown_list = [int(x) for x in list(cooldown_remaining or [])[:n]]
    while len(admission_steps_list) < n:
        admission_steps_list.append(0)
    while len(cooldown_list) < n:
        cooldown_list.append(0)

    eligible = [
        int(i)
        for i in range(n)
        if int(current_step) - int(admission_steps_list[i]) >= int(protect_steps)
        and int(cooldown_list[i]) <= 0
    ]
    if not eligible:
        return []

    burden_list = [float(x) for x in list(selector_burden or [])[:n]]
    while len(burden_list) < n:
        burden_list.append(0.0)
    target = min(int(max_candidates), len(eligible))
    order = sorted(
        eligible,
        key=lambda i: (
            _surrogate_key(int(i)),
            _benefit_key(int(i)),
            float(max(0.0, burden_list[i])),
            str(labels[i]),
        ),
    )
    return _surrogate_gated([int(i) for i in order[:target]])


def _prune_reason(
    *,
    safe_regression_ok: bool,
    retained_ok: bool,
    curvature_guard_ok: bool = True,
    curvature_guard_reason: str | None = None,
) -> str:
    failures: list[str] = []
    if not bool(safe_regression_ok):
        failures.append("safe_regression_exceeded")
    if not bool(retained_ok):
        failures.append("retained_gain_below_ratio")
    if not bool(curvature_guard_ok):
        failures.append(str(curvature_guard_reason or "curvature_guard_failed"))
    return "accepted" if not failures else "_and_".join(failures)


def recoverability_prune_ladder(
    *,
    theta: np.ndarray,
    labels: list[str],
    candidate_indices: list[int],
    rung_windows_by_index: Mapping[int, Sequence[tuple[str, Sequence[int]]]],
    eval_with_removal_window: Callable[..., tuple[float, np.ndarray]],
    energy_before: float,
    max_regression: float,
    retained_reference_energy: float | None = None,
    admitted_gain: float | None = None,
    retained_gain_ratio: float = 0.0,
    retained_gain_activation: float | None = None,
    curvature_guard_mode: str = PRUNE_CURVATURE_GUARD_OFF,
    curvature_guard_context: Mapping[str, Any] | None = None,
    max_trial_evaluations: int | None = None,
) -> tuple[np.ndarray, list[str], list[PruneDecision], float, list[dict[str, Any]]]:
    """Run the static ADAPT recoverability remove-refit ladder.

    Each rung is a real remove-refit trial.  Surrogate values may decide which
    candidates/windows deserve a trial, but they are never deletion
    certificates; acceptance is only from refit-energy safety plus optional
    retained-gain/curvature guards.
    """

    cur_theta = np.asarray(theta, dtype=float).copy()
    cur_labels = list(labels)
    decisions: list[PruneDecision] = []
    ladder_rows: list[dict[str, Any]] = []
    energy0 = float(energy_before)
    max_reg = float(max(0.0, max_regression))
    admitted = None if admitted_gain is None else float(max(0.0, admitted_gain))
    retained_activation = (
        None
        if retained_gain_activation is None
        else float(max(0.0, retained_gain_activation))
    )
    retained_guard_active = bool(
        admitted is not None
        and retained_reference_energy is not None
        and float(retained_gain_ratio) > 0.0
        and (retained_activation is None or float(admitted) >= float(retained_activation))
    )
    trial_budget = None
    if max_trial_evaluations is not None:
        try:
            trial_budget = int(max_trial_evaluations)
        except (TypeError, ValueError):
            trial_budget = 0
        trial_budget = int(max(0, trial_budget))
    trial_count = 0

    # Resolve the legacy callback arity before any scientific objective can be
    # evaluated.  Catching ``TypeError`` around the callback invocation itself
    # is unsafe: an internal error raised after a quantum/objective evaluation
    # would otherwise retry the same deletion and charge it twice.
    callback_accepts_rung_kind = True
    try:
        callback_signature = inspect.signature(eval_with_removal_window)
    except (TypeError, ValueError):
        callback_signature = None
    if callback_signature is not None:
        five_arg_probe = (0, np.zeros(0, dtype=float), [], [], "probe")
        four_arg_probe = five_arg_probe[:-1]
        try:
            callback_signature.bind(*five_arg_probe)
        except TypeError as five_arg_error:
            try:
                callback_signature.bind(*four_arg_probe)
            except TypeError:
                raise TypeError(
                    "eval_with_removal_window must accept either four legacy "
                    "arguments or five arguments including rung_kind."
                ) from five_arg_error
            callback_accepts_rung_kind = False

    for original_idx in [int(x) for x in candidate_indices]:
        if original_idx < 0 or original_idx >= len(cur_labels):
            continue
        label = str(cur_labels[original_idx])
        rung_specs = list(rung_windows_by_index.get(int(original_idx), ()))
        if not rung_specs:
            survivors = [int(i) for i in range(len(cur_labels) - 1)]
            rung_specs = [("local_refit", survivors)]
        for rung_idx, (rung_kind_raw, active_indices_raw) in enumerate(rung_specs):
            if trial_budget is not None and int(trial_count) >= int(trial_budget):
                return cur_theta, cur_labels, decisions, float(energy0), ladder_rows
            rung_kind = str(rung_kind_raw or f"rung_{int(rung_idx)}")
            active_indices = [int(x) for x in active_indices_raw]
            if callback_accepts_rung_kind:
                trial_energy, trial_theta = eval_with_removal_window(
                    int(original_idx),
                    np.asarray(cur_theta, dtype=float),
                    list(cur_labels),
                    list(active_indices),
                    str(rung_kind),
                )
            else:
                trial_energy, trial_theta = eval_with_removal_window(
                    int(original_idx),
                    np.asarray(cur_theta, dtype=float),
                    list(cur_labels),
                    list(active_indices),
                )
            trial_count += 1
            trial_energy_f = float(trial_energy)
            regression = float(trial_energy_f - energy0)
            safe_regression_ok = bool(regression <= max_reg)
            confidence_sigma = 0.0
            confidence_upper_regression = float(regression)
            retained_gain = (
                None
                if retained_reference_energy is None
                else float(retained_reference_energy) - float(trial_energy_f)
            )
            retained_gain_threshold = (
                None
                if not retained_guard_active or admitted is None
                else float(retained_gain_ratio) * float(admitted)
            )
            retained_ok = True
            if retained_guard_active and retained_gain is not None and retained_gain_threshold is not None:
                retained_ok = bool(float(retained_gain) >= float(retained_gain_threshold))
            guard_context = dict(curvature_guard_context or {})
            if "active_window_fraction" not in guard_context:
                survivor_count = max(1, len(cur_labels) - 1)
                guard_context["active_window_fraction"] = float(
                    min(1.0, max(0.0, len(active_indices) / float(survivor_count)))
                )
            if retained_activation is not None and "retained_gain_activation" not in guard_context:
                guard_context["retained_gain_activation"] = float(retained_activation)
            curvature_guard = evaluate_recoverability_curvature_guard(
                rung_index=int(rung_idx),
                rung_kind=str(rung_kind),
                confidence_upper_regression=float(confidence_upper_regression),
                regression_threshold=float(max_reg),
                mode=str(curvature_guard_mode),
                context=guard_context,
                retained_gain=retained_gain,
                admitted_gain=admitted,
                retained_gain_ratio=float(retained_gain_ratio),
            )
            curvature_guard_ok = bool(curvature_guard.get("curvature_guard_ok", True))
            accepted = (
                bool(safe_regression_ok)
                and bool(retained_ok)
                and bool(curvature_guard_ok)
            )
            reason = _prune_reason(
                safe_regression_ok=bool(safe_regression_ok),
                retained_ok=bool(retained_ok),
                curvature_guard_ok=bool(curvature_guard_ok),
                curvature_guard_reason=str(curvature_guard.get("curvature_guard_reason", "")),
            )
            decision = PruneDecision(
                index=int(original_idx),
                label=str(label),
                accepted=bool(accepted),
                energy_before=float(energy0),
                energy_after=float(trial_energy_f),
                regression=float(regression),
                reason=str(reason),
                safe_regression_ok=bool(safe_regression_ok),
                retained_gain_ok=bool(retained_ok),
                regression_threshold=float(max_reg),
                retained_gain=(None if retained_gain is None else float(retained_gain)),
                retained_gain_threshold=(
                    None if retained_gain_threshold is None else float(retained_gain_threshold)
                ),
                rung_index=int(rung_idx),
                rung_kind=str(rung_kind),
                confidence_model="deterministic_sigma0",
                confidence_sigma=float(confidence_sigma),
                confidence_upper_regression=float(confidence_upper_regression),
                confidence_guard_ok=bool(safe_regression_ok),
                curvature_guard_mode=str(curvature_guard.get("curvature_guard_mode", curvature_guard_mode)),
                curvature_guard_active=bool(curvature_guard.get("curvature_guard_active", False)),
                curvature_guard_ok=bool(curvature_guard_ok),
                curvature_guard_reason=str(curvature_guard.get("curvature_guard_reason", "")),
                acceptance_source="remove_refit_energy_safety",
                surrogate_used_for_acceptance=False,
            )
            decisions.append(decision)
            ladder_rows.append(
                {
                    "candidate_index": int(original_idx),
                    "candidate_label": str(label),
                    "rung_index": int(rung_idx),
                    "rung_kind": str(rung_kind),
                    "active_logical_indices": [int(x) for x in active_indices],
                    "energy_before": float(energy0),
                    "energy_after": float(trial_energy_f),
                    "regression": float(regression),
                    "paired_energy_regression": float(regression),
                    "confidence_model": "deterministic_sigma0",
                    "confidence_sigma": float(confidence_sigma),
                    "confidence_upper_regression": float(confidence_upper_regression),
                    "confidence_guard_ok": bool(safe_regression_ok),
                    "regression_threshold": float(max_reg),
                    "safe_regression_ok": bool(safe_regression_ok),
                    "retained_gain": (None if retained_gain is None else float(retained_gain)),
                    "retained_gain_threshold": (
                        None if retained_gain_threshold is None else float(retained_gain_threshold)
                    ),
                    "retained_guard_active": bool(retained_guard_active),
                    "retained_gain_ok": bool(retained_ok),
                    "curvature_guard_mode": str(curvature_guard.get("curvature_guard_mode", curvature_guard_mode)),
                    "curvature_guard_active": bool(curvature_guard.get("curvature_guard_active", False)),
                    "curvature_guard_ok": bool(curvature_guard_ok),
                    "curvature_guard_reason": str(curvature_guard.get("curvature_guard_reason", "")),
                    "curvature_compensated_rung": bool(curvature_guard.get("curvature_compensated_rung", False)),
                    "active_window_fraction": float(curvature_guard.get("active_window_fraction", guard_context.get("active_window_fraction", 0.0))),
                    "compression_mode": bool(curvature_guard.get("compression_mode", guard_context.get("compression_mode", False))),
                    "terminal_full": bool(curvature_guard.get("terminal_full", guard_context.get("terminal_full", False))),
                    "terminal_rung_s4": bool(curvature_guard.get("terminal_rung_s4", False)),
                    "accepted": bool(accepted),
                    "reason": str(reason),
                    "acceptance_source": "remove_refit_energy_safety",
                    "surrogate_authority": "rank_window_diag_only",
                    "surrogate_used_for_acceptance": False,
                }
            )
            if accepted:
                labels_out = list(cur_labels)
                del labels_out[int(original_idx)]
                return (
                    np.asarray(trial_theta, dtype=float).copy(),
                    labels_out,
                    decisions,
                    float(trial_energy_f),
                    ladder_rows,
                )

    return cur_theta, cur_labels, decisions, float(energy0), ladder_rows
