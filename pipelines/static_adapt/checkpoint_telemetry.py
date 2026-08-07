"""Current-checkpoint telemetry compaction helpers for static ADAPT."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.prune_risk_dataset import PRUNE_PREFILTER_OFF
from pipelines.static_adapt.route_c_plateau import route_c_plateau_compact_event_payload

__all__ = [
    "_compact_prune_audit",
    "_current_float",
    "_current_int",
    "_current_jsonable",
    "_current_str_or_none",
    "_compact_history_row_for_checkpoint",
    "_compact_exact_outer_control",
    "_compact_schur_warm_start",
    "_history_tail_for_checkpoint",
    "_selected_records_from_history_row",
    "_surface_rows_summary",
    "_phase3_surface_audit_payload",
    "_active_hh_pool_summary_payload",
    "_scaffold_fingerprint_payload",
    "_optimizer_memory_contract_summary_payload",
    "_controller_runtime_boundary_summary_payload",
]


def _optional_runtime_mapping(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    """Serialize an optional runtime receipt without fabricating a default."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping when present.")
    return _current_jsonable(value)


def _compact_prune_audit(summary_raw: Mapping[str, Any] | None) -> dict[str, Any]:
    summary = dict(summary_raw) if isinstance(summary_raw, Mapping) else {}
    prune_tolerance = dict(summary.get("prune_tolerance", {}) or {})
    reanchor_raw = summary.get("formal_manifold_post_prune_reanchor")
    reanchor = dict(reanchor_raw) if isinstance(reanchor_raw, Mapping) else {}
    query_receipt_raw = reanchor.get("query_receipt")
    query_receipt = (
        dict(query_receipt_raw)
        if isinstance(query_receipt_raw, Mapping)
        else {}
    )
    return {
        "enabled": bool(summary.get("enabled", False)),
        "prune_mode": str(summary.get("prune_mode", "live")),
        "permission_open": bool(summary.get("permission_open", False)),
        "permission_reason": str(summary.get("permission_reason", "unknown")),
        "executed": bool(summary.get("executed", False)),
        "accepted_count": int(summary.get("accepted_count", 0) or 0),
        "candidate_count": int(summary.get("candidate_count", 0) or 0),
        "prune_prefilter_policy": str(summary.get("prune_prefilter_policy", PRUNE_PREFILTER_OFF)),
        "prune_prefilter_active": bool(summary.get("prune_prefilter_active", False)),
        "prune_prefilter_input_count": int(summary.get("prune_prefilter_input_count", 0) or 0),
        "prune_prefilter_allowed_count": int(summary.get("prune_prefilter_allowed_count", 0) or 0),
        "prune_prefilter_blocked_count": int(summary.get("prune_prefilter_blocked_count", 0) or 0),
        "prune_prefilter_blocked_indices": [
            int(x) for x in summary.get("prune_prefilter_blocked_indices", [])
        ],
        "prune_prefilter_blocked_labels": [
            str(x) for x in summary.get("prune_prefilter_blocked_labels", [])
        ],
        "selected_index": summary.get("selected_index"),
        "selected_label": summary.get("selected_label"),
        "snr_adm": float(summary.get("snr_adm", 0.0) or 0.0),
        "u_sat": float(summary.get("u_sat", 0.0) or 0.0),
        "probe_indices": [int(x) for x in summary.get("probe_indices", [])],
        "max_regression": float(summary.get("max_regression", 0.0) or 0.0),
        "max_regression_effective": float(
            summary.get("max_regression_effective", summary.get("max_regression", 0.0)) or 0.0
        ),
        "phase1_prune_tolerance_mode": str(summary.get("phase1_prune_tolerance_mode", "fixed")),
        "prune_tolerance_effective": float(
            prune_tolerance.get(
                "effective_tolerance",
                summary.get("max_regression_effective", summary.get("max_regression", 0.0)),
            )
            or 0.0
        ),
        "prune_tolerance_used_component": str(
            prune_tolerance.get("used_component", "delta_num")
        ),
        "nfev_formal_manifold_prune_reanchor": int(
            summary.get("nfev_formal_manifold_prune_reanchor", 0) or 0
        ),
        "formal_manifold_post_prune_reanchor": (
            None
            if not reanchor
            else {
                "schema": str(reanchor.get("schema", "")),
                "state_action": str(reanchor.get("state_action", "")),
                "curvature_action": str(reanchor.get("curvature_action", "")),
                "qbroyd_action": str(reanchor.get("qbroyd_action", "")),
                "nfev": int(reanchor.get("nfev", 0) or 0),
                "whitening_id": str(reanchor.get("whitening_id", "")),
                "frame_id": str(reanchor.get("frame_id", "")),
                "logical_range_id": str(
                    reanchor.get("logical_range_id", "")
                ),
                "query_primitive_ids_requested": [
                    str(value)
                    for value in query_receipt.get(
                        "primitive_ids_requested", []
                    )
                ],
                "query_primitive_ids_reused": [
                    str(value)
                    for value in query_receipt.get(
                        "primitive_ids_reused", []
                    )
                ],
            }
        ),
        "phase1_prune_trust_state_before": _optional_runtime_mapping(
            summary.get("phase1_prune_trust_state_before"),
            field_name="phase1_prune_trust_state_before",
        ),
        "phase1_prune_trust_state_after": _optional_runtime_mapping(
            summary.get("phase1_prune_trust_state_after"),
            field_name="phase1_prune_trust_state_after",
        ),
        "phase1_prune_trust_update": _optional_runtime_mapping(
            summary.get("phase1_prune_trust_update"),
            field_name="phase1_prune_trust_update",
        ),
        "phase1_prune_trial_receipt": _optional_runtime_mapping(
            summary.get("phase1_prune_trial_receipt"),
            field_name="phase1_prune_trial_receipt",
        ),
        "phase1_prune_affine_deletion_model": _optional_runtime_mapping(
            summary.get("phase1_prune_affine_deletion_model"),
            field_name="phase1_prune_affine_deletion_model",
        ),
        "phase1_prune_exact_refit_work_accounting": _optional_runtime_mapping(
            summary.get("phase1_prune_exact_refit_work_accounting"),
            field_name="phase1_prune_exact_refit_work_accounting",
        ),
        "phase1_prune_no_feasible_model": _optional_runtime_mapping(
            summary.get("phase1_prune_no_feasible_model"),
            field_name="phase1_prune_no_feasible_model",
        ),
        "repeat_label_guard": _optional_runtime_mapping(
            summary.get("repeat_label_guard"),
            field_name="repeat_label_guard",
        ),
    }


def _current_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _current_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _current_jsonable(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_current_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _current_float(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return float(value_f)


def _current_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _current_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return str(value)


def _compact_schur_warm_start(
    payload_raw: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(payload_raw, Mapping):
        return None
    payload = dict(payload_raw)
    compact_keys = (
        "schema",
        "enabled",
        "mode",
        "attempted",
        "status",
        "reason",
        "chosen_source",
        "guard_objective_evals",
        "fallback_to_incumbent",
        "step_source",
        "prediction_authority",
        "objective_guard_required",
        "legacy_diagonal_geometry_used",
        "selected_labels",
        "batch_window",
        "active_pre_logical_indices",
        "active_post_logical_indices",
        "candidate_post_logical_indices",
        "active_reduced_position_groups",
        "candidate_reduced_position_groups",
        "active_parameter_relaxation",
        "batch_coordinate_step",
        "runtime_delta_l2",
        "selector_applied_predicted_reduction",
        "mapped_seed_incumbent_energy",
        "mapped_seed_proposal_energy",
        "mapped_seed_exact_gain",
        "prediction_to_exact_seed_ratio",
    )
    compact = {
        str(key): _current_jsonable(payload.get(key))
        for key in compact_keys
        if key in payload
    }
    guard_raw = payload.get("guard")
    if isinstance(guard_raw, Mapping):
        guard = dict(guard_raw)
        evaluations_raw = guard.get("evaluations")
        evaluations = (
            [
                {
                    str(key): _current_jsonable(dict(row).get(key))
                    for key in ("name", "status", "energy", "exception")
                    if key in dict(row)
                }
                for row in evaluations_raw[:8]
                if isinstance(row, Mapping)
            ]
            if isinstance(evaluations_raw, Sequence)
            and not isinstance(evaluations_raw, (str, bytes, bytearray))
            else []
        )
        compact["guard"] = {
            str(key): _current_jsonable(guard.get(key))
            for key in (
                "proposal_count",
                "incumbent_energy",
                "chosen_energy",
                "guard_tolerance",
                "guard_objective_evals",
            )
            if key in guard
        }
        compact["guard"]["evaluations"] = evaluations
    return compact


def _current_jsonable_without_event_rows(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _current_jsonable_without_event_rows(item)
            for key, item in value.items()
            if str(key) != "events"
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_current_jsonable_without_event_rows(item) for item in value]
    return _current_jsonable(value)


def _primitive_id_inventory_summary(values: Any) -> dict[str, Any]:
    ids = (
        sorted({str(value) for value in values})
        if isinstance(values, Sequence)
        and not isinstance(values, (str, bytes, bytearray))
        else []
    )
    payload = json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return {
        "count": int(len(ids)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _compact_exact_outer_control(
    payload_raw: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Retain the exact-vs-predicted transport certificate in current.json."""

    if not isinstance(payload_raw, Mapping):
        return None
    payload = dict(payload_raw)
    diagnostic = dict(payload.get("diagnostic", {}) or {})
    orientation = dict(diagnostic.get("orientation", {}) or {})
    gradient = dict(diagnostic.get("gradient", {}) or {})
    curvature = dict(diagnostic.get("curvature", {}) or {})
    stationarity = dict(diagnostic.get("stationarity", {}) or {})
    registration = dict(
        diagnostic.get("endpoint_frame_registration", {}) or {}
    )
    exact_registration = dict(
        orientation.get("exact_registration_telemetry", {}) or {}
    )
    exact_source_spectrum = dict(
        diagnostic.get("exact_source_spectrum", {}) or {}
    )
    exact_endpoint_spectrum = dict(
        diagnostic.get("exact_endpoint_spectrum", {}) or {}
    )
    query = dict(payload.get("query_accounting", {}) or {})
    authority = payload.get("authority")
    return {
        "schema": str(payload.get("schema", "")),
        "mode": str(payload.get("mode", "")),
        "available": bool(payload.get("available", False)),
        "prediction_available": bool(
            payload.get("prediction_available", False)
        ),
        "reason": _current_str_or_none(payload.get("reason")),
        "controller_state_mutated": bool(
            payload.get("controller_state_mutated", False)
        ),
        "exact_supported_stationarity_passed": bool(
            payload.get("exact_supported_stationarity_passed", False)
        ),
        "shadow_does_not_affect_controller": bool(
            payload.get("shadow_does_not_affect_controller", False)
        ),
        "diagnostic": {
            "schema": str(diagnostic.get("schema", "")),
            "predicted_frame_id": _current_str_or_none(
                diagnostic.get("predicted_frame_id")
            ),
            "exact_frame_id": _current_str_or_none(
                diagnostic.get("exact_frame_id")
            ),
            "exact_source_frame_id": _current_str_or_none(
                diagnostic.get("exact_source_frame_id")
            ),
            "predicted_support_rank": _current_int(
                diagnostic.get("predicted_support_rank")
            ),
            "exact_support_rank": _current_int(
                diagnostic.get("exact_support_rank")
            ),
            "exact_source_support_rank": _current_int(
                diagnostic.get("exact_source_support_rank")
            ),
            "exact_source_spectrum": {
                str(key): _current_jsonable(exact_source_spectrum.get(key))
                for key in (
                    "raw_metric_sha256",
                    "raw_eigenvalues_sha256",
                    "retained_eigenvalues_sha256",
                    "raw_eigenvalue_min",
                    "raw_eigenvalue_max",
                    "retained_eigenvalue_min",
                    "retained_eigenvalue_max",
                    "retained_condition_number",
                    "support_threshold",
                    "metric_ridge",
                    "rank_relative_tolerance",
                )
                if key in exact_source_spectrum
            },
            "exact_endpoint_spectrum": {
                str(key): _current_jsonable(
                    exact_endpoint_spectrum.get(key)
                )
                for key in (
                    "raw_metric_sha256",
                    "raw_eigenvalues_sha256",
                    "retained_eigenvalues_sha256",
                    "raw_eigenvalue_min",
                    "raw_eigenvalue_max",
                    "retained_eigenvalue_min",
                    "retained_eigenvalue_max",
                    "retained_condition_number",
                    "support_threshold",
                    "metric_ridge",
                    "rank_relative_tolerance",
                )
                if key in exact_endpoint_spectrum
            },
            "support_rank_match": bool(
                diagnostic.get("support_rank_match", False)
            ),
            "support_projector_defect": _current_float(
                diagnostic.get("support_projector_defect")
            ),
            "raw_metric_error_fro": _current_float(
                diagnostic.get("raw_metric_error_fro")
            ),
            "raw_metric_error_spectral": _current_float(
                diagnostic.get("raw_metric_error_spectral")
            ),
            "raw_metric_relative_error_fro": _current_float(
                diagnostic.get("raw_metric_relative_error_fro")
            ),
            "predicted_whitener_exact_metric_identity_residual": (
                _current_float(
                    diagnostic.get(
                        "predicted_whitener_exact_metric_identity_residual"
                    )
                )
            ),
            "endpoint_frame_registration": {
                "available": bool(registration.get("available", True)),
                "reason": _current_str_or_none(registration.get("reason")),
                "sigma_min": _current_float(registration.get("sigma_min")),
                "singular_values": [
                    float(value)
                    for value in registration.get("singular_values", [])
                ],
                "orthogonality_residual": _current_float(
                    registration.get("orthogonality_residual")
                ),
            },
            "orientation": {
                "available": bool(orientation.get("available", False)),
                "reason": _current_str_or_none(orientation.get("reason")),
                "transport_fully_compared": bool(
                    orientation.get("transport_fully_compared", False)
                ),
                "transport_error_fro": _current_float(
                    orientation.get("transport_error_fro")
                ),
                "transport_error_spectral": _current_float(
                    orientation.get("transport_error_spectral")
                ),
                "exact_registration_sigma_min": _current_float(
                    exact_registration.get("sigma_min")
                ),
                "exact_registration_singular_values": [
                    float(value)
                    for value in exact_registration.get(
                        "singular_values", []
                    )
                ],
            },
            "gradient": {
                str(key): _current_jsonable(gradient.get(key))
                for key in (
                    "available",
                    "exact_available",
                    "comparison_available",
                    "reason",
                    "absolute_error",
                    "relative_error",
                    "cosine",
                    "exact_frame_gradient_norm",
                    "exact_raw_null_compatibility",
                )
                if key in gradient
            },
            "curvature": {
                str(key): _current_jsonable(curvature.get(key))
                for key in (
                    "available",
                    "reason",
                    "absolute_error_fro",
                    "relative_error_fro",
                    "operator_error",
                )
                if key in curvature
            },
            "stationarity": {
                str(key): _current_jsonable(stationarity.get(key))
                for key in (
                    "schema",
                    "policy",
                    "supported_gradient",
                    "supported_gradient_norm",
                    "tolerance",
                    "passed",
                    "powell_success_is_not_stationarity_certificate",
                    "prediction_reuse_eligible",
                    "next_outer_exact_finalist_authority_required",
                    "structural_rollback_performed",
                )
                if key in stationarity
            },
        },
        "authority": (
            _current_jsonable(authority)
            if isinstance(authority, Mapping)
            else None
        ),
        "query_accounting": {
            "source_metric_elements_charged": _current_int(
                query.get("source_metric_elements_charged")
            ),
            "endpoint_metric_elements_charged": _current_int(
                query.get("endpoint_metric_elements_charged")
            ),
            "endpoint_gradient_elements_charged": _current_int(
                query.get("endpoint_gradient_elements_charged")
            ),
            "cross_state_tangent_elements_charged": _current_int(
                query.get("cross_state_tangent_elements_charged")
            ),
            "endpoint_primitive_id_inventory": (
                _primitive_id_inventory_summary(
                    query.get("endpoint_primitive_ids", [])
                )
            ),
            "source_primitive_id_inventory": (
                _primitive_id_inventory_summary(
                    query.get("source_primitive_ids", [])
                )
            ),
            "cross_frame_primitive_id_inventory": (
                _primitive_id_inventory_summary(
                    query.get("cross_frame_primitive_ids", [])
                )
            ),
        },
    }


def _selected_records_from_history_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels_raw = row.get("selected_ops")
    if isinstance(labels_raw, Sequence) and not isinstance(labels_raw, (str, bytes, bytearray)):
        labels_step = [str(x) for x in labels_raw]
    elif row.get("selected_op") not in {None, ""}:
        labels_step = [str(row.get("selected_op"))]
    else:
        labels_step = []
    positions_raw = row.get("selected_positions")
    if isinstance(positions_raw, Sequence) and not isinstance(positions_raw, (str, bytes, bytearray)):
        positions_step = [_current_int(x) for x in positions_raw]
    else:
        positions_step = [_current_int(row.get("selected_position", 0))]
    pool_indices_raw = row.get("selected_pool_indices")
    if isinstance(pool_indices_raw, Sequence) and not isinstance(pool_indices_raw, (str, bytes, bytearray)):
        pool_indices_step = [_current_int(x) for x in pool_indices_raw]
    else:
        pool_indices_step = [_current_int(row.get("pool_index"))]
    feature_rows_raw = row.get("selected_feature_rows")
    feature_rows = (
        list(feature_rows_raw)
        if isinstance(feature_rows_raw, Sequence) and not isinstance(feature_rows_raw, (str, bytes, bytearray))
        else []
    )
    records: list[dict[str, Any]] = []
    for item_idx, label_step in enumerate(labels_step):
        feature_mapping = (
            feature_rows[item_idx]
            if item_idx < len(feature_rows) and isinstance(feature_rows[item_idx], Mapping)
            else row
        )
        position_id = (
            positions_step[item_idx]
            if item_idx < len(positions_step) and positions_step[item_idx] is not None
            else _current_int(row.get("selected_position", 0))
        )
        candidate_pool_index = (
            pool_indices_step[item_idx]
            if item_idx < len(pool_indices_step) and pool_indices_step[item_idx] is not None
            else _current_int(feature_mapping.get("candidate_pool_index", row.get("pool_index")))
        )
        records.append(
            {
                "operator_label": str(label_step),
                "generator_label": str(feature_mapping.get("candidate_label", label_step)),
                "generator_id": _current_str_or_none(feature_mapping.get("generator_id")),
                "parent_generator_id": _current_str_or_none(feature_mapping.get("parent_generator_id")),
                "template_id": _current_str_or_none(feature_mapping.get("template_id")),
                "position_id": position_id,
                "candidate_pool_index": candidate_pool_index,
                "selection_mode": str(feature_mapping.get("selection_mode", row.get("selection_mode", ""))),
                "runtime_split_mode": str(
                    feature_mapping.get("runtime_split_mode", row.get("runtime_split_mode", "off"))
                ),
                "runtime_split_chosen_representation": _current_str_or_none(
                    feature_mapping.get(
                        "runtime_split_chosen_representation",
                        row.get("runtime_split_chosen_representation"),
                    )
                ),
                "runtime_split_child_generator_ids": [
                    str(x)
                    for x in (feature_mapping.get("runtime_split_child_generator_ids", []) or [])
                ],
                "route_a_child_identity": _current_str_or_none(
                    feature_mapping.get("route_a_child_identity")
                ),
                "route_a_global_pauli_identity": _current_str_or_none(
                    feature_mapping.get("route_a_global_pauli_identity")
                ),
                "route_a_child_parent_labels": [
                    str(value)
                    for value in feature_mapping.get(
                        "route_a_child_parent_labels", []
                    )
                ],
                "route_a_child_parent_count": _current_int(
                    feature_mapping.get("route_a_child_parent_count")
                ),
                "route_a_child_direction_normalization": (
                    _current_jsonable(
                        feature_mapping.get(
                            "route_a_child_direction_normalization"
                        )
                    )
                    if isinstance(
                        feature_mapping.get(
                            "route_a_child_direction_normalization"
                        ),
                        Mapping,
                    )
                    else None
                ),
            }
        )
    return records


def _compact_history_row_for_checkpoint(
    row: Mapping[str, Any],
    *,
    fallback_depth: int,
) -> dict[str, Any]:
    branch_id = _current_int(row.get("branch_id", row.get("beam_branch_id")))
    parent_branch_id = _current_int(
        row.get("parent_branch_id", row.get("beam_parent_branch_id"))
    )
    return {
        "depth": _current_int(row.get("depth", fallback_depth)),
        "branch_id": branch_id,
        "parent_branch_id": parent_branch_id,
        "selected_op": _current_str_or_none(row.get("selected_op")),
        "selected_position": _current_int(row.get("selected_position")),
        "selection_mode": _current_str_or_none(row.get("selection_mode")),
        "beam_structural_mode": _current_str_or_none(row.get("beam_structural_mode")),
        "selected_records": _selected_records_from_history_row(row),
        "batch_size": _current_int(row.get("batch_size")),
        "energy_before_opt": _current_float(row.get("energy_before_opt")),
        "energy_after_opt": _current_float(row.get("energy_after_opt")),
        "delta_energy": _current_float(row.get("delta_energy")),
        "delta_abs_current": _current_float(row.get("delta_abs_current")),
        "benchmark_target_abs_delta_current": _current_float(
            row.get("benchmark_target_abs_delta_current")
        ),
        "benchmark_target_abs_delta_e": _current_float(row.get("benchmark_target_abs_delta_e")),
        "benchmark_target_reference_energy": _current_float(
            row.get("benchmark_target_reference_energy")
        ),
        "selector_score": _current_float(row.get("selector_score")),
        "selector_burden": _current_float(row.get("selector_burden")),
        "simple_score": _current_float(row.get("simple_score")),
        "phase2_raw_score": _current_float(row.get("phase2_raw_score")),
        "full_v2_score": _current_float(row.get("full_v2_score")),
        "nfev_opt": _current_int(row.get("nfev_opt")),
        "nfev_seed_probe": _current_int(row.get("nfev_seed_probe")),
        "initial_energy_nfev": _current_int(row.get("initial_energy_nfev")),
        "nfev_schur_warm_start_guard": _current_int(
            row.get("nfev_schur_warm_start_guard")
        ),
        "schur_warm_start": _compact_schur_warm_start(
            row.get("schur_warm_start")
        ),
        "outer_nfev": _current_int(row.get("outer_nfev")),
        "nfev_total_before_step": _current_int(
            row.get("nfev_total_before_step")
        ),
        "nfev_total_after_step": _current_int(
            row.get("nfev_total_after_step")
        ),
        "nfev_step_total_delta": _current_int(
            row.get("nfev_step_total_delta")
        ),
        "nit_opt": _current_int(row.get("nit_opt")),
        "opt_success": (
            bool(row.get("opt_success"))
            if row.get("opt_success") is not None
            else None
        ),
        "opt_message": _current_str_or_none(row.get("opt_message")),
        "controller_measurement_work_proxy": (
            _current_jsonable_without_event_rows(
                row.get("controller_measurement_work_proxy")
            )
            if isinstance(row.get("controller_measurement_work_proxy"), Mapping)
            else None
        ),
        "stop_reason": _current_str_or_none(row.get("stop_reason")),
        "route_c_plateau_acquisition": (
            _current_jsonable(
                route_c_plateau_compact_event_payload(
                    row.get("route_c_plateau_acquisition")
                )
            )
            if isinstance(row.get("route_c_plateau_acquisition"), Mapping)
            else None
        ),
        "route_a_trust_region_update": (
            _current_jsonable(row.get("route_a_trust_region_update"))
            if isinstance(row.get("route_a_trust_region_update"), Mapping)
            else None
        ),
        "phase3_shadow_damping_receipt": _optional_runtime_mapping(
            row.get("phase3_shadow_damping_receipt"),
            field_name="phase3_shadow_damping_receipt",
        ),
        "accepted_refit": (
            _current_jsonable(row.get("accepted_refit"))
            if isinstance(row.get("accepted_refit"), Mapping)
            else None
        ),
        "post_admission_prune": _compact_prune_audit(row.get("post_admission_prune")),
        "active_prefix_checkpoint": _optional_runtime_mapping(
            row.get("active_prefix_checkpoint"),
            field_name="active_prefix_checkpoint",
        ),
        "formal_outer_exact_control": _compact_exact_outer_control(
            (
                row.get("formal_manifold_warm_start", {}) or {}
            ).get("exact_outer_control")
            if isinstance(row.get("formal_manifold_warm_start"), Mapping)
            else None
        ),
    }


def _history_tail_for_checkpoint(
    history_rows: Sequence[Mapping[str, Any]] | None,
    *,
    keep_history_tail: int,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in history_rows or [] if isinstance(row, Mapping)]
    if int(keep_history_tail) <= 0:
        return []
    first_depth = max(1, len(rows) - int(keep_history_tail) + 1)
    return [
        _compact_history_row_for_checkpoint(row, fallback_depth=int(first_depth + offset))
        for offset, row in enumerate(rows[-int(keep_history_tail):])
    ]


def _surface_rows_summary(rows_raw: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    rows = [dict(row) for row in rows_raw if isinstance(row, Mapping)] if isinstance(rows_raw, Sequence) else []
    return {
        "count": int(len(rows)),
        "operator_labels": list(
            dict.fromkeys(
                str(row.get("candidate_label", ""))
                for row in rows
                if str(row.get("candidate_label", "")) != ""
            )
        ),
        "generator_ids": list(
            dict.fromkeys(
                str(row.get("generator_id", ""))
                for row in rows
                if str(row.get("generator_id", "")) != ""
            )
        ),
        "position_ids": list(
            dict.fromkeys(
                int(row.get("position_id"))
                for row in rows
                if row.get("position_id") is not None
            )
        ),
        "runtime_split_modes": list(
            dict.fromkeys(
                str(row.get("runtime_split_mode", "off"))
                for row in rows
            )
        ),
    }


def _phase3_surface_audit_payload(
    *,
    scored_rows: Sequence[Mapping[str, Any]] | None,
    retained_rows: Sequence[Mapping[str, Any]] | None,
    admitted_rows: Sequence[Mapping[str, Any]] | None,
    beam_enabled: bool,
) -> dict[str, Any]:
    return {
        "scored_surface_notation": ("R_3(b)" if beam_enabled else "R_3(t)"),
        "scored_surface_key": "phase2_scored_rows",
        "scored_surface_semantics": "last_scored_candidate_surface",
        "retained_shortlist_notation": ("S_3(b)" if beam_enabled else "S_3(t)"),
        "retained_shortlist_key": "phase2_retained_shortlist_rows",
        "retained_shortlist_semantics": "controller_retained_shortlist",
        "admitted_set_notation": ("A_b" if beam_enabled else "B_t^*"),
        "admitted_set_key": "phase2_admitted_rows",
        "admitted_set_semantics": (
            "branch_local_retained_admission_set"
            if beam_enabled
            else "reduced_plane_admitted_set"
        ),
        "scored_surface": _surface_rows_summary(scored_rows),
        "retained_shortlist": _surface_rows_summary(retained_rows),
        "admitted_set": _surface_rows_summary(admitted_rows),
    }


def _active_hh_pool_summary_payload(
    *,
    phase1_rows: Sequence[Mapping[str, Any]] | None,
    phase2_rows: Sequence[Mapping[str, Any]] | None,
    phase3_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    phase1_rows_list = (
        [dict(row) for row in phase1_rows if isinstance(row, Mapping)]
        if isinstance(phase1_rows, Sequence)
        else []
    )
    phase2_rows_list = (
        [dict(row) for row in phase2_rows if isinstance(row, Mapping)]
        if isinstance(phase2_rows, Sequence)
        else []
    )
    phase3_rows_list = (
        [dict(row) for row in phase3_rows if isinstance(row, Mapping)]
        if isinstance(phase3_rows, Sequence)
        else []
    )
    phase1_summary = _surface_rows_summary(phase1_rows_list)
    phase2_summary = _surface_rows_summary(phase2_rows_list)
    phase3_summary = _surface_rows_summary(phase3_rows_list)

    def _split_closed_labels(
        seed_labels: set[str],
        rows_extra: Sequence[Mapping[str, Any]],
    ) -> set[str]:
        closed = set(str(x) for x in seed_labels)
        changed = True
        while changed:
            changed = False
            for row in rows_extra:
                label = str(row.get("candidate_label", ""))
                parent_label = str(row.get("runtime_split_parent_label", ""))
                if not label:
                    continue
                if parent_label and parent_label in closed and label not in closed:
                    closed.add(label)
                    changed = True
        return closed

    phase1_labels_raw = set(str(x) for x in phase1_summary.get("operator_labels", []))
    phase2_labels_raw = set(str(x) for x in phase2_summary.get("operator_labels", []))
    phase3_labels_raw = set(str(x) for x in phase3_summary.get("operator_labels", []))
    phase1_labels_effective = _split_closed_labels(
        phase1_labels_raw,
        [*phase2_rows_list, *phase3_rows_list],
    )
    phase2_labels_effective = _split_closed_labels(phase2_labels_raw, phase3_rows_list)
    phase3_labels_effective = set(phase3_labels_raw)
    phase1_summary["generator_image_labels_effective"] = sorted(phase1_labels_effective)
    phase1_summary["generator_image_count_effective"] = int(len(phase1_labels_effective))
    phase2_summary["generator_image_labels_effective"] = sorted(phase2_labels_effective)
    phase2_summary["generator_image_count_effective"] = int(len(phase2_labels_effective))
    phase3_summary["generator_image_labels_effective"] = sorted(phase3_labels_effective)
    phase3_summary["generator_image_count_effective"] = int(len(phase3_labels_effective))
    return {
        "summary_label": "Omega_HH_active",
        "omega_chain": ["Omega_HH^(1)", "Omega_HH^(2)", "Omega_HH^(3)"],
        "nested_generator_image_inclusion": {
            "phase2_in_phase1": bool(phase2_labels_effective.issubset(phase1_labels_effective)),
            "phase3_in_phase2": bool(phase3_labels_effective.issubset(phase2_labels_effective)),
            "phase3_in_phase1": bool(phase3_labels_effective.issubset(phase1_labels_effective)),
        },
        "phases": {
            "phase1": {
                "omega_label": "Omega_HH^(1)",
                "generator_image_notation": "G^(1)",
                "rows_key": "phase1_retained_rows",
                "rows_semantics": "phase1_retained_record_shortlist",
                "generator_family_notation": "G_adapt^(1)",
                **dict(phase1_summary),
            },
            "phase2": {
                "omega_label": "Omega_HH^(2)",
                "generator_image_notation": "G^(2)",
                "rows_key": "phase2_geometric_shortlist_rows",
                "rows_semantics": "phase2_retained_geometric_shortlist",
                "generator_family_notation": "G_adapt^(2)",
                **dict(phase2_summary),
            },
            "phase3": {
                "omega_label": "Omega_HH^(3)",
                "generator_image_notation": "G^(3)",
                "rows_key": "phase2_retained_shortlist_rows",
                "rows_semantics": "phase3_retained_shortlist_generator_image",
                "generator_family_notation": "G_adapt^(3)",
                **dict(phase3_summary),
            },
        },
    }


def _scaffold_fingerprint_payload(
    *,
    operator_labels: Sequence[str],
    generator_ids: Sequence[str],
    num_parameters: int,
) -> dict[str, Any]:
    payload = {
        "selected_operator_labels": [str(x) for x in operator_labels],
        "selected_generator_ids": [str(x) for x in generator_ids if str(x) != ""],
        "num_parameters": int(num_parameters),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {
        "fingerprint_notation": "fp(O_*)",
        "fingerprint_version": "scaffold_labels_generator_ids_params_v1",
        "fingerprint_sha256": str(digest),
        **payload,
    }


def _optimizer_memory_contract_summary_payload(
    *,
    beam_enabled: bool,
    branch_id: int | None,
    memory_state: Mapping[str, Any] | None,
    operator_labels: Sequence[str],
    generator_ids: Sequence[str],
    num_parameters: int,
    last_active_subset_source: str | None,
    last_active_subset_reused: bool,
) -> dict[str, Any]:
    state = dict(memory_state) if isinstance(memory_state, Mapping) else {}
    remap_events = [
        dict(row)
        for row in state.get("remap_events", [])
        if isinstance(row, Mapping)
    ]
    structural_transport = any(
        str(row.get("op", "")) in {"insert", "remove"} for row in remap_events
    )
    memory_source_value = (
        str(state.get("source"))
        if state.get("source") not in {None, ""}
        else (
            str(last_active_subset_source)
            if last_active_subset_source not in {None, ""}
            else "unavailable"
        )
    )
    if not bool(state.get("available", False)):
        observed_transport_mode = "unavailable"
    elif structural_transport:
        observed_transport_mode = "canonical_embedding_or_index_remap"
    else:
        observed_transport_mode = "same_scaffold_active_subset"
    return {
        "contract_label": "phase2_optimizer_memory_contract",
        "beam_enabled": bool(beam_enabled),
        "branch_id": (None if branch_id is None else int(branch_id)),
        "scaffold_fingerprint": _scaffold_fingerprint_payload(
            operator_labels=operator_labels,
            generator_ids=generator_ids,
            num_parameters=int(num_parameters),
        ),
        "exact_reuse_rule": "requires_matching_scaffold_fingerprint",
        "fingerprint_match_required": True,
        "canonical_embedding_notation": "theta -> theta⊕_p 0",
        "refit_window_notation": "W(r;t)",
        "memory_available": bool(state.get("available", False)),
        "memory_optimizer": str(state.get("optimizer", "unknown")),
        "memory_parameter_count": int(state.get("parameter_count", max(0, int(num_parameters)))),
        "memory_source": str(memory_source_value),
        "last_active_subset_source": (
            None
            if last_active_subset_source in {None, ""}
            else str(last_active_subset_source)
        ),
        "last_active_subset_reused": bool(last_active_subset_reused),
        "structural_transport_detected": bool(structural_transport),
        "observed_transport_mode": str(observed_transport_mode),
        "remap_event_count": int(len(remap_events)),
        "remap_event_tail": [dict(row) for row in remap_events[-8:]],
    }


def _controller_runtime_boundary_summary_payload(
    *,
    phase_enabled: bool,
    cfg: Any,
    stage_controller_payload: Mapping[str, Any] | None,
    current_snapshot_payload: Mapping[str, Any] | None,
    beam_enabled: bool,
    branch_id: int | None,
) -> dict[str, Any]:
    return {
        "summary_label": "appendix_a_runtime_boundary",
        "beam_enabled": bool(beam_enabled),
        "branch_id": (None if branch_id is None else int(branch_id)),
        "phase_enabled": bool(phase_enabled),
        "symbolic_result_keys": [
            "selected_scaffold_summary",
            "selected_scaffold_final_choice",
            "selected_scaffold_branch_state",
            "selected_state_summary",
            "selected_scaffold_history",
            "selected_scaffold_record_chain",
            "active_hh_pool_summary",
            "active_phase3_surface_summary",
        ],
        "runtime_controller_keys": [
            "stage_controller",
            "selected_scaffold_branch_state.controller_telemetry",
            "selected_scaffold_optimizer_memory_contract",
        ],
        "runtime_law_notation": {
            "thresholds": "tau_k(t)",
            "caps": "N_k(t)",
            "shots_phase1": "N_shot,1(t)",
            "shots_phasek": "N_shot,k(t)",
        },
        "runtime_dependencies": [
            "available_depth",
            "wall_clock",
            "sampling_budget",
            "device_noise",
        ],
        "calibration_status": "runtime_calibrated_not_symbolic",
        "configured_bounds": {
            "tau_phase1_min": float(cfg.tau_phase1_min),
            "tau_phase1_max": float(cfg.tau_phase1_max),
            "tau_phase2_min": float(cfg.tau_phase2_min),
            "tau_phase2_max": float(cfg.tau_phase2_max),
            "tau_phase3_min": float(cfg.tau_phase3_min),
            "tau_phase3_max": float(cfg.tau_phase3_max),
            "cap_phase1_min": int(cfg.cap_phase1_min),
            "cap_phase1_max": int(cfg.cap_phase1_max),
            "cap_phase2_min": int(cfg.cap_phase2_min),
            "cap_phase2_max": int(cfg.cap_phase2_max),
            "cap_phase3_min": int(cfg.cap_phase3_min),
            "cap_phase3_max": int(cfg.cap_phase3_max),
            "shot_min": int(cfg.shot_min),
            "shot_max": int(cfg.shot_max),
            "shot_cap_phase1": int(cfg.shot_cap_phase1),
            "shot_cap_phase2": int(cfg.shot_cap_phase2),
            "shot_cap_phase3": int(cfg.shot_cap_phase3),
        },
        "stage_controller_payload": (
            dict(stage_controller_payload)
            if isinstance(stage_controller_payload, Mapping)
            else None
        ),
        "current_controller_snapshot": (
            dict(current_snapshot_payload)
            if isinstance(current_snapshot_payload, Mapping)
            else None
        ),
    }
