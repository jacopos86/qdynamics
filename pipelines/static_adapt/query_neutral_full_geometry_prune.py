"""Query-neutral full-geometry pruning for the Paper-I SR-SNAKE route.

This module is deliberately classical.  It accepts the already measured
Phase-III full active-plus-singleton coordinate model, solves at most one
affine deletion trust problem, and returns a warm start for the route's one
ordinary accepted-state refit.  It must never call an estimator, state
preparation routine, or objective function.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_pruning import (
    AffineDeletionFSTrustState,
    solve_full_logical_affine_deletion_fs_trust,
)
from pipelines.scaffold.hh_continuation_scoring import (
    BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
    HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
)


QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_MODEL_V1 = (
    "query_neutral_full_geometry_prune_model_v1"
)
QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_PROPOSAL_V1 = (
    "query_neutral_full_geometry_prune_proposal_v1"
)
QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_HOLD_V1 = (
    "query_neutral_full_geometry_prune_hold_v1"
)
QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_TRANSACTION_V1 = (
    "query_neutral_full_geometry_prune_transaction_v1"
)
PAPER_I_QUERY_NEUTRAL_PRUNE_TARGET_ABS_DELTA_E = 2.0e-4
PAPER_I_QUERY_NEUTRAL_PRUNE_MODELED_ENERGY_CHANGE_MAX = -2.0e-6
PAPER_I_QUERY_NEUTRAL_PRUNE_ENERGY_GUARD_ABS_TOL = 1.0e-12


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_vector(raw: Any, *, size: int, field: str) -> np.ndarray:
    value = np.asarray(raw, dtype=float).reshape(-1)
    if value.shape != (int(size),) or not np.all(np.isfinite(value)):
        raise RuntimeError(
            f"Query-neutral prune source {field} has the wrong shape or "
            "contains nonfinite values."
        )
    return value


def _finite_matrix(
    raw: Any,
    *,
    shape: tuple[int, int],
    field: str,
) -> np.ndarray:
    value = np.asarray(raw, dtype=float)
    normalized_shape = tuple(int(x) for x in shape)
    if value.size == 0 and int(np.prod(normalized_shape, dtype=int)) == 0:
        # JSON cannot preserve the distinction among [], a 0x0 matrix, and a
        # 0x1 block.  At the zero-active initialization the authoritative
        # ansatz/candidate dimensions restore that shape without measurement.
        value = value.reshape(normalized_shape)
    if value.shape != normalized_shape or not np.all(
        np.isfinite(value)
    ):
        raise RuntimeError(
            f"Query-neutral prune source {field} has the wrong shape or "
            "contains nonfinite values."
        )
    return value


def normalize_full_geometry_prune_source(
    *,
    selector_summary: Mapping[str, Any],
    pre_admission_labels: Sequence[str],
    pre_admission_theta: Sequence[float] | np.ndarray,
    candidate_label: str,
    candidate_pool_index: int,
    candidate_position: int,
) -> dict[str, Any]:
    """Normalize the locked full active-plus-singleton Phase-III model.

    The selector records descent gradients ``g_A/g_B``.  The affine-deletion
    solver consumes energy gradients, so the returned gradient is their
    negative.  All numeric arrays are copied; no estimator work is performed.
    """

    summary = dict(selector_summary)
    if str(summary.get("schema", "")) != (
        HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA
    ):
        raise RuntimeError(
            "Query-neutral pruning requires the historical singleton "
            "coordinate-model schema."
        )
    if str(summary.get("joint_batch_context_mode", "")) != (
        BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1
    ):
        raise RuntimeError(
            "Query-neutral pruning requires the full-ansatz Phase-III "
            "coordinate model; a material or rolling window is forbidden."
        )
    if str(summary.get("joint_linear_solve_policy_effective", "")) != (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
    ):
        raise RuntimeError(
            "Query-neutral pruning requires the projected generalized "
            "Phase-III solve."
        )
    if not bool(summary.get("feasible", False)):
        raise RuntimeError(
            "Query-neutral pruning cannot nominate from an infeasible "
            "Phase-III coordinate model."
        )

    labels = tuple(str(value) for value in pre_admission_labels)
    theta_old = _finite_vector(
        pre_admission_theta,
        size=len(labels),
        field="pre_admission_theta",
    )
    n_old = int(len(labels))
    position = int(candidate_position)
    pool_index = int(candidate_pool_index)
    label = str(candidate_label)
    if not label:
        raise RuntimeError("Query-neutral pruning requires a candidate label.")
    if position < 0 or position > n_old:
        raise RuntimeError(
            "Query-neutral prune candidate insertion position is out of range."
        )
    if int(summary.get("position_id", -1)) != position:
        raise RuntimeError(
            "Query-neutral prune source candidate position drifted from the "
            "authoritative admission."
        )
    if int(summary.get("candidate_pool_index", -1)) != pool_index:
        raise RuntimeError(
            "Query-neutral prune source candidate pool index drifted from the "
            "authoritative admission."
        )
    summary_label = str(summary.get("candidate_label", ""))
    if summary_label and summary_label != label:
        raise RuntimeError(
            "Query-neutral prune source candidate label drifted from the "
            "authoritative admission."
        )

    active_identities = tuple(
        str(value)
        for value in summary.get("active_coordinate_identities", ())
    )
    if active_identities != labels:
        raise RuntimeError(
            "Query-neutral prune source active-coordinate identities do not "
            "match the pre-admission ansatz."
        )
    candidate_identities = summary.get("batch_coordinate_identities")
    if not isinstance(candidate_identities, Sequence) or len(
        candidate_identities
    ) != 1:
        raise RuntimeError(
            "Query-neutral pruning requires exactly one candidate identity."
        )
    identity = candidate_identities[0]
    if not isinstance(identity, Mapping):
        raise RuntimeError(
            "Query-neutral prune candidate identity is not a mapping."
        )
    if int(identity.get("candidate_pool_index", -1)) != pool_index or int(
        identity.get("position_id", -1)
    ) != position:
        raise RuntimeError(
            "Query-neutral prune candidate identity disagrees with the "
            "authoritative admission."
        )
    identity_label = str(identity.get("candidate_label", ""))
    if identity_label and identity_label != label:
        raise RuntimeError(
            "Query-neutral prune candidate identity has a conflicting label."
        )

    G_AA = _finite_matrix(
        summary.get("G_AA_raw"),
        shape=(n_old, n_old),
        field="G_AA_raw",
    )
    G_AB = _finite_matrix(
        summary.get("G_AB_raw"),
        shape=(n_old, 1),
        field="G_AB_raw",
    )
    G_BB = _finite_matrix(
        summary.get("G_BB_raw"),
        shape=(1, 1),
        field="G_BB_raw",
    )
    H_AA = _finite_matrix(
        summary.get("H_AA_raw"),
        shape=(n_old, n_old),
        field="H_AA_raw",
    )
    H_AB = _finite_matrix(
        summary.get("H_AB_raw"),
        shape=(n_old, 1),
        field="H_AB_raw",
    )
    H_BB = _finite_matrix(
        summary.get("H_BB_raw"),
        shape=(1, 1),
        field="H_BB_raw",
    )
    descent_old = _finite_vector(
        summary.get("g_A"),
        size=n_old,
        field="g_A",
    )
    descent_candidate = _finite_vector(
        summary.get("g_B"),
        size=1,
        field="g_B",
    )
    metric = np.block([[G_AA, G_AB], [G_AB.T, G_BB]])
    hessian = np.block([[H_AA, H_AB], [H_AB.T, H_BB]])
    metric = 0.5 * (metric + metric.T)
    hessian = 0.5 * (hessian + hessian.T)
    gradient = -np.concatenate((descent_old, descent_candidate))
    theta_model = np.concatenate((theta_old, np.zeros(1, dtype=float)))
    old_post_indices = tuple(
        int(index if index < position else index + 1)
        for index in range(n_old)
    )
    model_post_indices = (*old_post_indices, int(position))

    identity_receipt = {
        "schema": QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_MODEL_V1,
        "phase3_coordinate_scope": "full_active_plus_singleton_v1",
        "pre_admission_labels": list(labels),
        "candidate_label": label,
        "candidate_pool_index": pool_index,
        "candidate_position": position,
        "model_post_indices": [int(value) for value in model_post_indices],
        "active_coordinate_count": n_old,
        "model_coordinate_count": int(n_old + 1),
        "candidate_label_placeholder_filled": bool(not identity_label),
        "source_gradient_convention": "descent_gradient_g",
        "prune_gradient_convention": "energy_gradient_minus_g",
        "source_geometry_reused": True,
        "duplicate_measurement_performed": False,
        "incremental_quantum_query_charge": 0,
    }
    identity_receipt["source_identity_sha256"] = _json_digest(
        identity_receipt
    )
    return {
        "theta": theta_model,
        "gradient": gradient,
        "hessian": hessian,
        "metric": metric,
        "model_post_indices": tuple(int(value) for value in model_post_indices),
        "old_post_indices": old_post_indices,
        "candidate_post_index": int(position),
        "receipt": identity_receipt,
    }


def build_query_neutral_prune_source_unavailable_hold(
    *,
    selector_summary: Mapping[str, Any],
    trust_state: AffineDeletionFSTrustState,
) -> dict[str, Any]:
    """Hold pruning when Phase III produced no feasible coordinate model.

    The parent no-overlap route can legitimately admit through its
    geometry-expansion fallback without a coordinate prediction.  That round
    has no complete old-plus-singleton model from which pruning may be
    certified, so pruning is ineligible and performs no replacement
    measurement.
    """

    summary = dict(selector_summary)
    if bool(summary.get("feasible", False)):
        raise RuntimeError(
            "A feasible Phase-III coordinate model must use the ordinary "
            "query-neutral prune normalizer."
        )
    return {
        "schema": QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_HOLD_V1,
        "nominated": False,
        "reason": "phase3_coordinate_model_infeasible_hold",
        "source_reason": str(
            summary.get(
                "reason",
                summary.get(
                    "joint_linear_solve_reason",
                    "no_coordinate_prediction",
                ),
            )
        ),
        "source_geometry_reused": False,
        "source_geometry_available": False,
        "duplicate_measurement_performed": False,
        "incremental_quantum_query_charge": 0,
        "trust_radius": float(trust_state.radius),
        "metric_damping": float(trust_state.metric_damping),
    }


def build_query_neutral_prune_proposal(
    *,
    model: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    trust_state: AffineDeletionFSTrustState,
    selector_step: int,
    protect_steps: int,
    modeled_energy_change_max: float = 0.0,
) -> dict[str, Any]:
    """Return at most one conservative, measurement-free deletion proposal."""

    theta = np.asarray(model.get("theta"), dtype=float).reshape(-1)
    gradient = np.asarray(model.get("gradient"), dtype=float).reshape(-1)
    hessian = np.asarray(model.get("hessian"), dtype=float)
    metric = np.asarray(model.get("metric"), dtype=float)
    model_post_indices = tuple(
        int(value) for value in model.get("model_post_indices", ())
    )
    n_model = int(theta.size)
    if (
        gradient.shape != (n_model,)
        or hessian.shape != (n_model, n_model)
        or metric.shape != (n_model, n_model)
        or len(model_post_indices) != n_model
    ):
        raise RuntimeError(
            "Query-neutral prune model is internally inconsistent."
        )
    metadata = [dict(row) for row in metadata_rows]
    n_old = int(n_model - 1)
    if len(metadata) != n_old:
        raise RuntimeError(
            "Query-neutral prune metadata count must match the inherited "
            "active coordinates."
        )
    max_loss = float(modeled_energy_change_max)
    if not math.isfinite(max_loss):
        raise RuntimeError(
            "Query-neutral prune modeled-energy budget must be finite."
        )

    candidates: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for model_index, row in enumerate(metadata):
        first_seen = int(row.get("first_seen_step", selector_step))
        age = int(max(0, int(selector_step) - first_seen))
        cooldown = int(max(0, int(row.get("cooldown_remaining", 0) or 0)))
        protected = bool(age < int(max(0, protect_steps)))
        if protected or cooldown > 0:
            ineligible.append(
                {
                    "model_index": int(model_index),
                    "post_index": int(model_post_indices[model_index]),
                    "reason": "protected" if protected else "cooldown",
                    "age": age,
                    "cooldown_remaining": cooldown,
                }
            )
            continue
        solve = solve_full_logical_affine_deletion_fs_trust(
            theta=theta,
            gradient=gradient,
            hessian=hessian,
            metric=metric,
            deletion_index=int(model_index),
            trust_radius=float(trust_state.radius),
            metric_damping=float(trust_state.metric_damping),
        )
        solve_payload = solve.as_dict()
        if not solve.feasible:
            ineligible.append(
                {
                    "model_index": int(model_index),
                    "post_index": int(model_post_indices[model_index]),
                    "reason": str(solve.reason),
                    "solve": solve_payload,
                }
            )
            continue
        if float(solve.predicted_energy_change) > max_loss:
            ineligible.append(
                {
                    "model_index": int(model_index),
                    "post_index": int(model_post_indices[model_index]),
                    "reason": "modeled_energy_change_exceeds_budget",
                    "solve": solve_payload,
                }
            )
            continue
        candidates.append(
            {
                "model_index": int(model_index),
                "post_index": int(model_post_indices[model_index]),
                "predicted_energy_change": float(
                    solve.predicted_energy_change
                ),
                "predicted_reduction": float(solve.predicted_reduction),
                "fubini_study_displacement_sq": float(
                    solve.fubini_study_displacement_sq
                ),
                "joint_step": [
                    float(value) for value in solve.joint_step.tolist()
                ],
                "solve": solve_payload,
            }
        )

    if not candidates:
        return {
            "schema": QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_HOLD_V1,
            "nominated": False,
            "reason": "no_conservative_eligible_deletion",
            "eligible_solve_count": 0,
            "ineligible": ineligible,
            "trust_radius": float(trust_state.radius),
            "metric_damping": float(trust_state.metric_damping),
            "source_geometry_reused": True,
            "duplicate_measurement_performed": False,
            "incremental_quantum_query_charge": 0,
        }

    chosen = min(
        candidates,
        key=lambda row: (
            float(row["predicted_energy_change"]),
            float(row["fubini_study_displacement_sq"]),
            int(row["model_index"]),
        ),
    )
    joint_step = np.asarray(chosen["joint_step"], dtype=float)
    modeled_theta = theta + joint_step
    deletion_model_index = int(chosen["model_index"])
    deletion_post_index = int(chosen["post_index"])
    full_post_theta = np.zeros(n_model, dtype=float)
    for model_index, post_index in enumerate(model_post_indices):
        full_post_theta[int(post_index)] = float(modeled_theta[model_index])
    warm_start_post_delete = np.delete(full_post_theta, deletion_post_index)
    receipt = {
        "schema": QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_PROPOSAL_V1,
        "nominated": True,
        "reason": "conservative_full_geometry_affine_deletion",
        "deletion_model_index": deletion_model_index,
        "deletion_post_index": deletion_post_index,
        "predicted_energy_change": float(
            chosen["predicted_energy_change"]
        ),
        "predicted_reduction": float(chosen["predicted_reduction"]),
        "fubini_study_displacement_sq": float(
            chosen["fubini_study_displacement_sq"]
        ),
        "joint_step_source_model_order": [
            float(value) for value in joint_step.tolist()
        ],
        "warm_start_post_delete_logical_theta": [
            float(value) for value in warm_start_post_delete.tolist()
        ],
        "model_post_indices": [int(value) for value in model_post_indices],
        "eligible_solve_count": int(len(candidates)),
        "ineligible": ineligible,
        "chosen_solve": dict(chosen["solve"]),
        "trust_radius": float(trust_state.radius),
        "metric_damping": float(trust_state.metric_damping),
        "modeled_energy_change_max": max_loss,
        "source_geometry_reused": True,
        "duplicate_measurement_performed": False,
        "incremental_quantum_query_charge": 0,
    }
    receipt["proposal_sha256"] = _json_digest(receipt)
    return receipt


def combined_transition_energy_guard(
    *,
    energy_before: float,
    energy_after: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    """Classify the one-refit combined admission/deletion transaction."""

    before = float(energy_before)
    after = float(energy_after)
    tolerance = float(max(0.0, absolute_tolerance))
    if not (math.isfinite(before) and math.isfinite(after)):
        raise RuntimeError(
            "Query-neutral prune energy guard received a nonfinite energy."
        )
    accepted = bool(after <= before + tolerance)
    return {
        "schema": QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_TRANSACTION_V1,
        "accepted": accepted,
        "action": (
            "commit_combined_admission_deletion"
            if accepted
            else "restore_pre_round_accepted_state"
        ),
        "energy_before": before,
        "energy_after_refit": after,
        "realized_energy_change": float(after - before),
        "absolute_tolerance": tolerance,
        "second_refit_performed": False,
        "rollback_classical": bool(not accepted),
        "rollback_quantum_query_charge": 0,
        "incremental_prune_quantum_query_charge": 0,
    }


def realized_source_model_step_after_deletion(
    *,
    model: Mapping[str, Any],
    proposal: Mapping[str, Any],
    final_post_delete_logical_theta: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Embed the accepted reduced endpoint back into old-plus-candidate order."""

    theta_source = np.asarray(model.get("theta"), dtype=float).reshape(-1)
    model_post_indices = tuple(
        int(value) for value in model.get("model_post_indices", ())
    )
    if len(model_post_indices) != int(theta_source.size):
        raise RuntimeError(
            "Query-neutral prune source mapping has the wrong dimension."
        )
    deletion_post_index = int(proposal.get("deletion_post_index", -1))
    if deletion_post_index < 0 or deletion_post_index >= int(
        theta_source.size
    ):
        raise RuntimeError(
            "Query-neutral prune deletion post index is out of range."
        )
    final_reduced = _finite_vector(
        final_post_delete_logical_theta,
        size=int(theta_source.size - 1),
        field="final_post_delete_logical_theta",
    )
    full_post = np.insert(
        final_reduced,
        deletion_post_index,
        0.0,
    )
    final_model = np.asarray(
        [full_post[post_index] for post_index in model_post_indices],
        dtype=float,
    )
    return np.asarray(final_model - theta_source, dtype=float)


__all__ = [
    "PAPER_I_QUERY_NEUTRAL_PRUNE_ENERGY_GUARD_ABS_TOL",
    "PAPER_I_QUERY_NEUTRAL_PRUNE_MODELED_ENERGY_CHANGE_MAX",
    "PAPER_I_QUERY_NEUTRAL_PRUNE_TARGET_ABS_DELTA_E",
    "QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_HOLD_V1",
    "QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_MODEL_V1",
    "QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_PROPOSAL_V1",
    "QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_TRANSACTION_V1",
    "build_query_neutral_prune_source_unavailable_hold",
    "build_query_neutral_prune_proposal",
    "combined_transition_energy_guard",
    "normalize_full_geometry_prune_source",
    "realized_source_model_step_after_deletion",
]
