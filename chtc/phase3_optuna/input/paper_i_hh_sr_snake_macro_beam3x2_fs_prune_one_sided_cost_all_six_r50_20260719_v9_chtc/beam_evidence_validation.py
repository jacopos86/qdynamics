#!/usr/bin/env python3
"""Fail-closed evidence validation for macro/beam/prune cost-arm runs.

The historical beam archive may select a shallow terminal winner while its
recoverable controller frontier reaches the requested horizon.  This module
validates those two paths separately, validates the complete materialized-beam
estimator receipt graph, and never changes branch selection or archive policy.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from evidence_validation import (
    FULL_RESPONSE_SCOPE,
    PHASE1_ENERGY_MODEL,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY,
    PHASE2_CURVATURE_POLICY,
    _mapping,
    _sequence,
    validate_checkpoint,
    validate_ledger,
    validate_live_prune_round,
    validate_phase2_curvature_receipt,
    validate_post_prune_depth,
    validate_supported_rank_or_exact_fallback,
)


COMPONENTS = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
LANE_FIELDS = (
    "static_lane_route",
    "physical_operator_lane",
    "physical_operator_quality",
    "physical_operator_hh_full_meta_class",
    "physical_operator_lane_source",
    "physical_operator_lane_health",
    "physical_operator_lane_relative_health",
    "physical_operator_lane_live",
)
STATE_IDENTITY_FIELDS = (
    "active_ansatz_depth",
    "ordered_active_operator_labels",
    "ordered_active_operators",
    "signed_unwrapped_logical_parameters",
    "signed_unwrapped_runtime_parameters",
    "parameterization",
    "projective_state_fingerprint",
)
COMMON_RUNTIME_SETTINGS: Mapping[str, Any] = {
    "adapt_finite_angle_fallback": False,
    "phase3_enable_rescue": False,
    "adapt_final_full_refit": "false",
    "adapt_full_refit_every": 0,
    "phase2_enable_batching": False,
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "live",
    "phase1_prune_max_candidates": 1,
    "phase1_prune_local_window_size": 0,
    "phase1_prune_recovery_trust_radius": 0.125,
    "phase1_prune_schur_nomination_route": (
        "full_logical_fs_trust_delete_refit_v1"
    ),
    "phase1_prune_metric_schur_mu": 0.0,
    "phase1_prune_metric_schur_solve_mode": (
        "affine_deletion_global_trust_v1"
    ),
    "phase1_prune_metric_schur_cost_weighting": "off",
    "phase1_prune_trust_update_policy": "modeled_local_fs_conservative_v1",
    "phase1_prune_metric_mu_update_policy": "off",
    "phase1_prune_endpoint_overlap_policy": "off",
    "phase2_gram_novelty_policy": "fallback_only_v1",
    "phase3_gram_novelty_policy": "fallback_only_v1",
    "phase3_shadow_damping_policy": "off",
    "phase3_response_coordinate_scope": FULL_RESPONSE_SCOPE,
    "historical_singleton_coordinate_solve_scope": "phase3_only_v1",
    "historical_singleton_coordinate_solve_policy": (
        "supported_metric_whitened_eigh_v1"
    ),
    "historical_singleton_trust_region_update_policy": (
        "displacement_calibrated_unbounded_v2"
    ),
    "sr_powell_coordinate_chart_policy": (
        "expanded_runtime_projected_logical_v1"
    ),
}
EXPECTED_BEAM_RUNTIME: Mapping[str, Any] = {
    "beam_enabled": True,
    "live_branches": 3,
    "children_per_parent": 2,
    "terminated_keep": 3,
    "lambda_beam": 0.005,
}


def _runtime_value(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if key == "adapt_final_full_refit" and value is False:
        return "false"
    return value


def _validate_runtime_settings(
    settings: Mapping[str, Any], *, expected_cost_mode: str
) -> None:
    if not expected_cost_mode:
        raise ValueError("expected cost mode is absent")
    required = dict(COMMON_RUNTIME_SETTINGS)
    required["phase3_hardware_cost_normalization_mode"] = expected_cost_mode
    for key, expected in required.items():
        if _runtime_value(settings, key) != expected:
            raise ValueError(f"normalized candidate setting drift: {key}")



SOURCE_ONLY_RUNTIME_SETTINGS: Mapping[str, Any] = {
    "phase_live_hysteresis_enabled": False,
    "phase0_pilot_enabled": False,
    "phase3_enable_batching": False,
    "adapt_beam_live_branches": 3,
    "adapt_beam_children_per_parent": 2,
    "adapt_beam_terminated_keep": 3,
    "adapt_beam_terminal_archive_mode": "legacy",
    "adapt_beam_lambda": 0.005,
    "adapt_beam_parent_workers": 1,
    "adapt_accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
}


def validate_source_only_runtime_settings_receipt(
    normalized_manifest: Mapping[str, Any], *, digest: str
) -> dict[str, Any]:
    route = _mapping(
        normalized_manifest.get("route_identity"),
        field="normalized route identity",
    )
    contract = _mapping(route.get("profile_contract"), field="route contract")
    execution = _mapping(
        contract.get("execution_settings"), field="route execution settings"
    )
    semantic = _mapping(
        contract.get("semantic_invariants"), field="route semantic invariants"
    )
    argv = [
        str(value)
        for value in _sequence(
            normalized_manifest.get("command_argv"),
            field="normalized command argv",
        )
    ]
    if len(digest) != 64 or route.get("profile_contract_sha256") != digest:
        raise ValueError("source-only route digest drift")
    for key, expected in SOURCE_ONLY_RUNTIME_SETTINGS.items():
        if execution.get(key) != expected:
            raise ValueError(f"source-only route setting drift: {key}")
    if int(semantic.get("beam_expanded_child_cap_per_round", -1)) != 6:
        raise ValueError("source-only beam expanded-child cap drift")
    if (
        argv.count("--phase-live-hysteresis-disabled") != 1
        or "--phase-live-hysteresis-enabled" in argv
    ):
        raise ValueError("hysteresis-disabled command receipt drift")
    return {
        "schema": "paper_i_sr_source_only_runtime_settings_receipt_v1",
        "status": "pass",
        "phase_live_hysteresis_disabled": True,
        "command_flag": "--phase-live-hysteresis-disabled",
        "profile_contract_sha256": digest,
        "source_only_runtime_settings": dict(SOURCE_ONLY_RUNTIME_SETTINGS),
        "beam_expanded_child_cap_per_round": 6,
        "result_settings_fields_required": False,
        "behavioral_closure": "full_response_validated_each_controller_round_v1",
    }

def _validate_phase12(payload_raw: Any, *, label: str) -> int:
    payload = _mapping(payload_raw, field=f"{label} Phase-I/II telemetry")
    full = int(payload.get("phase2_full_candidate_occurrences", -1))
    if (
        payload.get("schema") != "sr_snake_phase12_energy_model_runtime_v1"
        or payload.get("phase1_energy_model") != PHASE1_ENERGY_MODEL
        or payload.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY
        or payload.get("phase2_cheap_curvature_proxy_policy")
        != PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        or full <= 0
        or int(payload.get("validated_phase2_curvature_receipt_occurrences", -2))
        != full
        or any(
            int(payload.get(field, -1)) != 0
            for field in (
                "phase1_lambda_f_proxy_occurrences",
                "phase2_lambda_f_proxy_occurrences",
                "phase2_missing_curvature_fallback_occurrences",
            )
        )
    ):
        raise ValueError(f"{label} Phase-I/II closure drift")
    return full


def _validate_fallback(
    payload_raw: Any,
    *,
    label: str,
    expected_policy: str,
    observed_rounds: Sequence[int],
) -> None:
    payload = _mapping(payload_raw, field=f"{label} fallback telemetry")
    rounds = [
        int(value)
        for value in _sequence(
            payload.get("controller_rounds"),
            field=f"{label} fallback controller rounds",
        )
    ]
    if (
        not expected_policy
        or payload.get("schema")
        != "sr_all_energy_models_infeasible_novelty_fallback_telemetry_v1"
        or payload.get("enabled") is not True
        or payload.get("policy") != expected_policy
        or payload.get("ordinary_phase2_multiplier_active") is not False
        or payload.get("ordinary_phase3_multiplier_active") is not False
        or payload.get("phase2_curvature_failure_can_trigger") is not False
        or payload.get("paper_reporting_scope") != "telemetry_gate_only_v1"
        or int(payload.get("query_charge_total", -1)) != 0
        or rounds != list(observed_rounds)
        or rounds != sorted(set(rounds))
        or int(payload.get("activation_count", -1)) != len(rounds)
        or (payload.get("fired") is True) != bool(rounds)
    ):
        raise ValueError(f"{label} fallback closure drift")


def _validate_trust_state(
    payload_raw: Any, *, label: str, expected_rounds: int, expected_radius: float
) -> None:
    payload = _mapping(payload_raw, field=f"{label} trust state")
    radius = float(payload.get("radius", float("nan")))
    if (
        payload.get("schema") != "route_a_trust_region_state_v1"
        or int(payload.get("update_count", -1)) != expected_rounds
        or not math.isfinite(radius)
        or radius != expected_radius
    ):
        raise ValueError(f"{label} trust-state drift")


def validate_controller_winner_relationship(
    *,
    beam: Mapping[str, Any],
    current_adapt: Mapping[str, Any],
    current_checkpoint: Mapping[str, Any],
    segment: Mapping[str, Any],
    digest: str,
    target_round: int,
    target_new_admissions: int,
) -> dict[str, Any]:
    rounds = list(_sequence(beam.get("rounds"), field="beam controller rounds"))
    if len(rounds) != target_round:
        raise ValueError("beam controller did not complete exact target rounds")
    for key, expected in EXPECTED_BEAM_RUNTIME.items():
        if beam.get(key) != expected:
            raise ValueError(f"historical beam runtime telemetry drift: {key}")

    selected_round = int(segment.get("final_controller_round", -1))
    selected_admissions = int(segment.get("new_admission_records", -1))
    if (
        int(segment.get("source_controller_round", -1)) != 0
        or int(segment.get("target_controller_round", -1)) != target_round
        or int(segment.get("max_new_admissions", -1)) != target_new_admissions
        or selected_round < 1
        or selected_round > target_round
        or selected_admissions != selected_round
    ):
        raise ValueError("selected-winner segment receipt drift")

    relationship = _mapping(
        beam.get("final_checkpoint_relationship"),
        field="beam final checkpoint relationship",
    )
    diagnostic = _mapping(
        relationship.get("diagnostic_terminal_branch"),
        field="diagnostic terminal branch",
    )
    selected_branch_id = int(
        relationship.get("diagnostic_terminal_branch_id", -1)
    )
    if (
        relationship.get("schema_version")
        != "static_adapt_beam_final_checkpoint_relationship_v1"
        or selected_branch_id < 0
        or int(diagnostic.get("branch_id", -2)) != selected_branch_id
        or int(diagnostic.get("depth_local", -1)) != selected_round
        or int(diagnostic.get("history_count", -1)) != selected_round
        or bool(diagnostic.get("terminated"))
        != (diagnostic.get("status") == "terminal")
    ):
        raise ValueError("diagnostic winner relationship drift")

    recoverable = _mapping(
        relationship.get("recoverable_frontier_branch"),
        field="recoverable frontier branch",
    )
    recoverable_id = int(relationship.get("recoverable_frontier_branch_id", -1))
    if (
        recoverable_id < 0
        or int(recoverable.get("branch_id", -2)) != recoverable_id
        or int(recoverable.get("depth_local", -1)) != target_round
        or int(recoverable.get("history_count", -1)) != target_round
        or recoverable.get("status") != "frontier"
        or recoverable.get("terminated") is not False
        or relationship.get("checkpoint_branch_policy") != "best_frontier_branch"
    ):
        raise ValueError("recoverable-frontier relationship drift")

    relationship_present = relationship.get("relationship_present") is True
    if selected_round < target_round:
        if (
            not relationship_present
            or relationship.get("reason")
            != "non_target_terminal_selected_with_recoverable_frontier"
            or relationship.get("recoverable_frontier_deeper_than_terminal")
            is not True
            or diagnostic.get("status") != "terminal"
        ):
            raise ValueError("shallow terminal/recoverable-frontier drift")
    elif (
        relationship_present
        or relationship.get("reason") != "not_applicable"
        or relationship.get("recoverable_frontier_deeper_than_terminal") is not False
    ):
        raise ValueError("round-target winner relationship drift")

    if (
        current_adapt.get("partial_checkpoint") is not True
        or current_adapt.get("success") is not False
        or int(current_adapt.get("history_count", -1)) != target_round
        or int(current_adapt.get("history_tail_count", -1)) != target_round
        or current_adapt.get("history_checkpoint_complete") is not True
        or int(current_adapt.get("branch_id", -1)) != recoverable_id
        or int(current_checkpoint.get("depth", -1)) != target_round
        or int(current_checkpoint.get("branch_id", -1)) != recoverable_id
        or current_checkpoint.get("checkpoint_branch_policy")
        != "best_frontier_branch"
        or current_checkpoint.get("sr_route_profile_contract_sha256") != digest
    ):
        raise ValueError("controller-frontier current-pointer drift")
    return {
        "selected_round": selected_round,
        "selected_branch_id": selected_branch_id,
        "recoverable_frontier_branch_id": recoverable_id,
        "relationship": relationship,
    }


def _validate_lane_receipt(row: Mapping[str, Any], *, outer_iteration: int) -> None:
    selected = list(
        _sequence(
            row.get("selected_feature_rows"),
            field=f"round {outer_iteration} selected feature rows",
        )
    )
    if len(selected) != 1:
        raise ValueError(f"round {outer_iteration}: non-singleton beam admission")
    feature = _mapping(
        selected[0], field=f"round {outer_iteration} selected feature row"
    )
    for field in LANE_FIELDS:
        if field not in row or field not in feature or row.get(field) != feature.get(field):
            raise ValueError(
                f"round {outer_iteration}: beam lane field {field} is absent or changed"
            )
    if (
        row.get("static_lane_route") != "physical_operator_type"
        or not str(row.get("physical_operator_lane") or "")
    ):
        raise ValueError(f"round {outer_iteration}: physical lane receipt drift")


def _validate_path(
    history_raw: Any,
    *,
    expected_rounds: int,
    require_supported_rank: bool,
) -> dict[str, Any]:
    history = list(_sequence(history_raw, field="beam branch history"))
    if len(history) != expected_rounds:
        raise ValueError("beam branch history length drift")
    checkpoints: list[Mapping[str, Any]] = []
    previous_depth = 0
    previous_radius: float | None = None
    fallback_rounds: list[int] = []
    max_sector = 0.0
    max_padding = 0.0
    prune_executed = 0
    prune_accepted = 0
    for outer_iteration, raw in enumerate(history, start=1):
        row = _mapping(raw, field=f"beam history round {outer_iteration}")
        if int(row.get("depth", -1)) != outer_iteration:
            raise ValueError(f"round {outer_iteration}: branch depth drift")
        _validate_lane_receipt(row, outer_iteration=outer_iteration)
        if str(row.get("selected_op", "")).startswith(
            ("guarded_singleton::", "projected_singleton::")
        ):
            raise ValueError("macro-only route admitted a generated child")
        if (
            row.get("phase1_energy_model") != PHASE1_ENERGY_MODEL
            or row.get("phase1_lambda_f_proxy_applied") is not False
            or row.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY
            or row.get("phase2_cheap_curvature_proxy_policy")
            != PHASE2_CHEAP_CURVATURE_PROXY_POLICY
            or row.get("phase2_lambda_f_proxy_applied") is not False
            or row.get("phase2_missing_curvature_fallback_used") is not False
        ):
            raise ValueError(f"round {outer_iteration}: Phase-I/II policy drift")
        validate_phase2_curvature_receipt(
            row.get("phase2_curvature_receipt"),
            outer_iteration=outer_iteration,
        )

        active_count = int(row.get("phase3_active_logical_coordinate_count", -1))
        pre_support = int(row.get("phase3_response_pre_support_count", -1))
        indices = [
            int(value)
            for value in _sequence(
                row.get("phase3_response_coordinate_indices"),
                field=f"round {outer_iteration} response indices",
            )
        ]
        if (
            row.get("phase3_response_coordinate_scope") != FULL_RESPONSE_SCOPE
            or active_count != previous_depth
            or pre_support != active_count + 1
            or indices != list(range(pre_support))
        ):
            raise ValueError(f"round {outer_iteration}: full-response invariant failed")
        rank = validate_supported_rank_or_exact_fallback(
            row,
            pre_support=pre_support,
            outer_iteration=outer_iteration,
            require_supported_rank=require_supported_rank,
        )
        if rank["fallback_fired"]:
            fallback_rounds.append(outer_iteration)
        if int(row.get("phase3_accepted_refit_coordinate_count", -1)) != pre_support:
            raise ValueError(f"round {outer_iteration}: accepted-refit scope drift")
        accepted = _mapping(
            row.get("accepted_refit"),
            field=f"round {outer_iteration} accepted refit",
        )
        invocation = _mapping(
            accepted.get("accepted_refit_invocation"),
            field=f"round {outer_iteration} accepted-refit invocation",
        )
        config = _mapping(
            invocation.get("config"),
            field=f"round {outer_iteration} refit config",
        )
        if (
            config.get("scope") != "full_ansatz_v1"
            or config.get("full_ansatz") is not True
            or config.get("coordinate_chart") != "supported_fs_whitened_fixed_v1"
            or config.get("supported_fs_whitened") is not True
            or config.get("base_chart_policy")
            != "expanded_runtime_projected_logical_v1"
        ):
            raise ValueError(f"round {outer_iteration}: accepted-refit chart drift")
        trust = _mapping(
            row.get("route_a_trust_region_update"),
            field=f"round {outer_iteration} trust update",
        )
        before = float(trust.get("radius_before", float("nan")))
        after = float(trust.get("radius_after", float("nan")))
        if (
            trust.get("schema") != "route_a_trust_region_update_v1"
            or trust.get("policy") != "displacement_calibrated_unbounded_v2"
            or trust.get("full_coordinate_refit") is not True
            or not math.isfinite(before)
            or not math.isfinite(after)
            or before < 0.0
            or after < 0.0
            or (previous_radius is not None and before != previous_radius)
        ):
            raise ValueError(f"round {outer_iteration}: trust-chain drift")
        previous_radius = after

        prune = validate_live_prune_round(
            row.get("post_admission_prune"), outer_iteration=outer_iteration
        )
        prune_executed += int(prune["executed"] is True)
        prune_accepted += int(
            prune["executed"] is True and prune.get("accepted") is True
        )
        checkpoint = _mapping(
            row.get("active_prefix_checkpoint"),
            field=f"round {outer_iteration} active-prefix checkpoint",
        )
        summary = validate_checkpoint(
            checkpoint,
            outer_iteration=outer_iteration,
            checkpoint_kind="post_admission_prune",
        )
        previous_depth = validate_post_prune_depth(
            previous_active_depth=previous_depth,
            pre_support=pre_support,
            checkpoint_depth=int(summary["active_ansatz_depth"]),
            prune_summary=prune,
            outer_iteration=outer_iteration,
        )
        max_sector = max(max_sector, float(summary["fixed_sector_leakage"]))
        max_padding = max(max_padding, float(summary["binary_padding_leakage"]))
        checkpoints.append(checkpoint)
    return {
        "history": history,
        "checkpoints": checkpoints,
        "final_active_depth": previous_depth,
        "final_radius": previous_radius,
        "fallback_rounds": fallback_rounds,
        "max_fixed_sector_leakage": max_sector,
        "max_binary_padding_leakage": max_padding,
        "prune_rounds_executed": prune_executed,
        "prune_rounds_accepted": prune_accepted,
    }



def _validate_compact_current_history(
    history_raw: Any,
    *,
    selected_path: Mapping[str, Any],
    expected_rounds: int,
) -> dict[str, Any]:
    compact = list(_sequence(history_raw, field="compact current history"))
    selected = list(
        _sequence(selected_path.get("history"), field="selected full history")
    )
    if len(compact) != expected_rounds or len(selected) != expected_rounds:
        raise ValueError("compact/full current history length drift")
    scalar_fields = (
        "depth",
        "branch_id",
        "parent_branch_id",
        "selected_op",
        "selected_position",
        "batch_size",
        "energy_before_opt",
        "energy_after_opt",
        "delta_energy",
        "nfev_total_before_step",
        "nfev_total_after_step",
        "nfev_step_total_delta",
    )
    selected_checkpoints = list(
        _sequence(selected_path.get("checkpoints"), field="selected checkpoints")
    )
    for outer_iteration, (compact_raw, selected_raw, checkpoint_raw) in enumerate(
        zip(compact, selected, selected_checkpoints), start=1
    ):
        compact_row = _mapping(
            compact_raw, field=f"compact current round {outer_iteration}"
        )
        selected_row = _mapping(
            selected_raw, field=f"selected full round {outer_iteration}"
        )
        checkpoint = _mapping(
            compact_row.get("active_prefix_checkpoint"),
            field=f"compact checkpoint round {outer_iteration}",
        )
        selected_checkpoint = _mapping(
            checkpoint_raw, field=f"selected checkpoint round {outer_iteration}"
        )
        for field in scalar_fields:
            if compact_row.get(field) != selected_row.get(field):
                raise ValueError(
                    f"compact/full current history drift at round {outer_iteration}: {field}"
                )
        if checkpoint.get("checkpoint_sha256") != selected_checkpoint.get(
            "checkpoint_sha256"
        ):
            raise ValueError(
                f"compact/full checkpoint drift at round {outer_iteration}"
            )
    return {
        "schema": "paper_i_sr_compact_current_full_winner_crosscheck_v1",
        "status": "pass",
        "rounds": expected_rounds,
    }

def _validate_selected_checkpoint_list(
    adapt: Mapping[str, Any], *, path: Mapping[str, Any]
) -> None:
    explicit = list(
        _sequence(
            adapt.get("active_prefix_checkpoints"),
            field="selected active-prefix checkpoints",
        )
    )
    embedded = list(path["checkpoints"])
    if len(explicit) != len(embedded):
        raise ValueError("selected checkpoint-list length drift")
    for index, (left, right) in enumerate(zip(explicit, embedded), start=1):
        if _mapping(left, field=f"selected checkpoint {index}") != right:
            raise ValueError(f"selected checkpoint {index} differs from history")


def _validate_current_pointer(
    *,
    current: Mapping[str, Any],
    current_adapt: Mapping[str, Any],
    current_checkpoint: Mapping[str, Any],
    controller_path: Mapping[str, Any],
    target_round: int,
) -> None:
    if (
        current.get("schema_version") != "static_adapt_current_checkpoint_v1"
        or current.get("no_credentials_serialized") is not True
        or current_checkpoint.get("complete") is not False
        or not str(current_checkpoint.get("reason") or "")
        or int(current_checkpoint.get("ansatz_depth", -1))
        != int(controller_path["final_active_depth"])
        or int(current_adapt.get("ansatz_depth", -1))
        != int(controller_path["final_active_depth"])
    ):
        raise ValueError("current pointer metadata drift")
    last = _mapping(
        controller_path["checkpoints"][-1], field="controller terminal checkpoint"
    )
    receipt = _mapping(
        last.get("estimator_ledger_receipt"),
        field="controller terminal checkpoint receipt",
    )
    if (
        int(current_checkpoint.get("depth", -1)) != target_round
        or int(last.get("outer_iteration", -1))
        != int(current_checkpoint.get("depth", -2))
        or last.get("checkpoint_kind") != "post_admission_prune"
        or int(last.get("active_ansatz_depth", -1))
        != int(current_checkpoint.get("ansatz_depth", -2))
        or int(receipt.get("outer_iteration", -1)) != target_round
        or str(receipt.get("branch_id")) != str(current_checkpoint.get("branch_id"))
        or str(receipt.get("parent_branch_id"))
        != str(current_checkpoint.get("parent_branch_id"))
    ):
        raise ValueError("current pointer/checkpoint receipt mismatch")


def _validate_path_closure(
    *,
    adapt: Mapping[str, Any],
    phase12_raw: Any,
    path: Mapping[str, Any],
    label: str,
    expected_rounds: int,
    expected_fallback_policy: str,
) -> int:
    _validate_trust_state(
        adapt.get("route_a_trust_region_state"),
        label=label,
        expected_rounds=expected_rounds,
        expected_radius=float(path["final_radius"]),
    )
    _validate_fallback(
        adapt.get("all_energy_models_infeasible_novelty_fallback_telemetry"),
        label=label,
        expected_policy=expected_fallback_policy,
        observed_rounds=path["fallback_rounds"],
    )
    return _validate_phase12(phase12_raw, label=label)


def _validate_beam_receipts(
    *,
    adapt: Mapping[str, Any],
    controller_checkpoints: Sequence[Mapping[str, Any]],
    selected_checkpoints: Sequence[Mapping[str, Any]],
    selected_terminal: Mapping[str, Any],
    selected_round: int,
    selected_branch_id: int,
    target_round: int,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    continuation = _mapping(adapt.get("continuation"), field="result continuation")
    receipts = list(
        _sequence(
            continuation.get("all_active_prefix_estimator_ledger_receipts"),
            field="all beam estimator receipts",
        )
    )
    if len(receipts) < target_round + 1:
        raise ValueError("beam receipt stream is shorter than controller horizon")

    prior_raw = 0
    prior_unique = 0
    summed_raw = {key: 0 for key in COMPONENTS}
    summed_unique = {key: 0 for key in COMPONENTS}
    ordinary_rounds: list[int] = []
    branch_round: dict[str, int] = {}
    root_parent_ids: set[str] = set()
    for sequence, raw in enumerate(receipts, start=1):
        receipt = _mapping(raw, field=f"beam estimator receipt {sequence}")
        if (
            receipt.get("schema")
            != "paper_i_active_prefix_estimator_ledger_receipt_v1"
            or receipt.get("enabled") is not True
            or receipt.get("status") != "complete"
            or int(receipt.get("checkpoint_sequence", -1)) != sequence
            or receipt.get("canonical_same_state_deduplication_active") is not True
            or receipt.get("raw_occurrences_preserved") is not True
        ):
            raise ValueError(f"beam estimator receipt {sequence} identity drift")
        outer = int(receipt.get("outer_iteration", -1))
        kind = str(receipt.get("checkpoint_kind") or "")
        branch_id = str(receipt.get("branch_id") or "")
        parent_branch_id = str(receipt.get("parent_branch_id") or "")
        if kind == "post_admission_prune":
            if (
                outer < 1
                or outer > target_round
                or not branch_id
                or not parent_branch_id
                or branch_id in branch_round
            ):
                raise ValueError(f"beam estimator receipt {sequence} branch drift")
            if outer == 1:
                root_parent_ids.add(parent_branch_id)
            elif (
                parent_branch_id not in branch_round
                or branch_round[parent_branch_id] != outer - 1
            ):
                raise ValueError(f"beam estimator receipt {sequence} parent-link drift")
            ordinary_rounds.append(outer)
            branch_round[branch_id] = outer
        elif kind == "terminal_post_final_refit_and_prune":
            if (
                sequence != len(receipts)
                or outer != selected_round
                or branch_id != str(selected_branch_id)
                or branch_id not in branch_round
                or branch_round[branch_id] != selected_round
            ):
                raise ValueError("beam terminal estimator receipt drift")
        else:
            raise ValueError(f"beam estimator receipt {sequence} kind drift")

        raw_delta = _mapping(receipt.get("raw_occurrence_delta"), field="raw delta")
        unique_delta = _mapping(
            receipt.get("unique_primitive_delta"), field="unique delta"
        )
        cumulative_raw = _mapping(
            receipt.get("cumulative_raw_occurrences"), field="cumulative raw"
        )
        cumulative_unique = _mapping(
            receipt.get("cumulative_unique_primitives"), field="cumulative unique"
        )
        raw_components = _mapping(raw_delta.get("components"), field="raw components")
        unique_components = _mapping(
            unique_delta.get("components"), field="unique components"
        )
        cumulative_raw_components = _mapping(
            cumulative_raw.get("components"), field="cumulative raw components"
        )
        cumulative_unique_components = _mapping(
            cumulative_unique.get("components"),
            field="cumulative unique components",
        )
        raw_values = {key: int(raw_components.get(key, -1)) for key in COMPONENTS}
        unique_values = {
            key: int(unique_components.get(key, -1)) for key in COMPONENTS
        }
        raw_total = int(raw_delta.get("total", -1))
        unique_total = int(unique_delta.get("S_alg", -1))
        current_raw = int(cumulative_raw.get("total", -1))
        current_unique = int(cumulative_unique.get("S_alg", -1))
        if min(raw_values.values()) < 0 or min(unique_values.values()) < 0:
            raise ValueError(f"beam estimator receipt {sequence} negative delta")
        for key in COMPONENTS:
            summed_raw[key] += raw_values[key]
            summed_unique[key] += unique_values[key]
        if (
            raw_total != sum(raw_values.values())
            or unique_total != sum(unique_values.values())
            or int(receipt.get("occurrence_sequence_start_exclusive", -1))
            != prior_raw
            or current_raw != prior_raw + raw_total
            or current_unique != prior_unique + unique_total
            or any(
                int(cumulative_raw_components.get(key, -1)) != summed_raw[key]
                or int(cumulative_unique_components.get(key, -1))
                != summed_unique[key]
                for key in COMPONENTS
            )
            or current_raw != sum(summed_raw.values())
            or current_unique != sum(summed_unique.values())
        ):
            raise ValueError(f"beam estimator receipt {sequence} arithmetic drift")
        prior_raw = current_raw
        prior_unique = current_unique

    if (
        len(root_parent_ids) != 1
        or ordinary_rounds != sorted(ordinary_rounds)
        or set(ordinary_rounds) != set(range(1, target_round + 1))
        or max(ordinary_rounds, default=-1) != target_round
        or len(branch_round) != len(receipts) - 1
    ):
        raise ValueError("beam receipt graph coverage drift")

    def _crosscheck_checkpoint(checkpoint: Mapping[str, Any], index: int) -> None:
        receipt = _mapping(
            checkpoint.get("estimator_ledger_receipt"),
            field=f"checkpoint {index} estimator receipt",
        )
        if (
            receipt not in receipts[:-1]
            or int(receipt.get("outer_iteration", -1)) != index
            or branch_round.get(str(receipt.get("branch_id") or "")) != index
        ):
            raise ValueError(f"checkpoint {index} estimator receipt mismatch")

    for index, checkpoint in enumerate(controller_checkpoints, start=1):
        _crosscheck_checkpoint(checkpoint, index)
    for index, checkpoint in enumerate(selected_checkpoints, start=1):
        _crosscheck_checkpoint(checkpoint, index)
    if selected_terminal.get("estimator_ledger_receipt") != receipts[-1]:
        raise ValueError("selected terminal receipt mismatch")
    if (
        str(
            _mapping(
                selected_checkpoints[-1].get("estimator_ledger_receipt"),
                field="selected final ordinary receipt",
            ).get("branch_id")
        )
        != str(selected_branch_id)
    ):
        raise ValueError("selected path does not end on selected branch")

    closure = _mapping(
        continuation.get("active_prefix_estimator_ledger_closure"),
        field="beam estimator closure",
    )
    if (
        closure.get("schema")
        != "paper_i_active_prefix_estimator_ledger_closure_v1"
        or closure.get("enabled") is not True
        or closure.get("status") != "complete"
        or closure.get("passed") is not True
        or closure.get("includes_discarded_branch_checkpoints") is not True
        or int(closure.get("receipt_count", -1)) != len(receipts)
        or closure.get("summed_raw_occurrences")
        != closure.get("terminal_raw_occurrences")
        or closure.get("summed_unique_primitives")
        != closure.get("terminal_unique_primitives")
    ):
        raise ValueError("beam estimator closure drift")
    terminal_raw = _mapping(
        closure.get("terminal_raw_occurrences"), field="terminal raw closure"
    )
    terminal_unique = _mapping(
        closure.get("terminal_unique_primitives"), field="terminal unique closure"
    )
    if (
        terminal_raw.get("components") != summed_raw
        or int(terminal_raw.get("total", -1)) != prior_raw
        or terminal_unique.get("components") != summed_unique
        or int(terminal_unique.get("S_alg", -1)) != prior_unique
        or prior_raw != int(ledger["raw_occurrence_count"])
        or prior_unique != int(ledger["all_branch_s_alg"])
    ):
        raise ValueError("beam estimator receipts do not close to ledger")
    return {
        "receipt_count": len(receipts),
        "controller_horizon_rounds": target_round,
        "terminal_receipt_count": 1,
        "materialized_branch_count": len(branch_round),
        "all_branch_S_alg": prior_unique,
        "closure_passed": True,
    }


def validate_beam_parent_evidence(
    *,
    result: Mapping[str, Any],
    current: Mapping[str, Any],
    ledger_sidecar: Mapping[str, Any],
    profile: str,
    digest: str,
    expected_cost_mode: str,
    expected_fallback_policy: str,
    target_round: int,
    target_new_admissions: int,
    require_supported_rank: bool = True,
) -> dict[str, Any]:
    adapt = _mapping(result.get("adapt_vqe"), field="result adapt_vqe")
    settings = _mapping(result.get("settings"), field="result settings")
    current_settings = _mapping(current.get("settings"), field="current settings")
    current_adapt = _mapping(current.get("adapt_vqe"), field="current adapt_vqe")
    current_checkpoint = _mapping(current.get("checkpoint"), field="current checkpoint")
    for name, payload in (
        ("settings", settings),
        ("result", adapt),
        ("current settings", current_settings),
    ):
        if payload.get("sr_route_profile_resolved") != profile:
            raise ValueError(f"{name} profile drift")
        if payload.get("sr_route_profile_contract_sha256") != digest:
            raise ValueError(f"{name} route digest drift")
        if payload.get("phase1_energy_model") != PHASE1_ENERGY_MODEL:
            raise ValueError(f"{name} Phase-I policy drift")
        if payload.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY:
            raise ValueError(f"{name} Phase-II policy drift")
        if payload.get("phase2_cheap_curvature_proxy_policy") != (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ):
            raise ValueError(f"{name} Phase-II proxy policy drift")
    _validate_runtime_settings(settings, expected_cost_mode=expected_cost_mode)

    continuation = _mapping(adapt.get("continuation"), field="result continuation")
    beam = _mapping(continuation.get("beam_search"), field="beam search telemetry")
    if adapt.get("adapt_beam_enabled") is not True or adapt.get("success") is not True:
        raise ValueError("beam ADAPT execution status drift")
    if adapt.get("finite_angle_fallback") is not False:
        raise ValueError("finite-angle fallback was enabled at runtime")
    if (
        str(continuation.get("oracle_gradient_scope")) != "off"
        or continuation.get("oracle_gradient_config") is not None
        or int(continuation.get("oracle_gradient_calls_total", -1)) != 0
        or int(continuation.get("oracle_gradient_raw_records_total", -1)) != 0
    ):
        raise ValueError("Phase-III oracle-gradient route was active")

    segment = _mapping(result.get("adapt_segment"), field="result segment")
    relationship_receipt = validate_controller_winner_relationship(
        beam=beam,
        current_adapt=current_adapt,
        current_checkpoint=current_checkpoint,
        segment=segment,
        digest=digest,
        target_round=target_round,
        target_new_admissions=target_new_admissions,
    )
    selected_round = int(relationship_receipt["selected_round"])
    selected_branch_id = int(relationship_receipt["selected_branch_id"])
    relationship = _mapping(
        relationship_receipt["relationship"], field="validated relationship"
    )

    selected_path = _validate_path(
        adapt.get("history"),
        expected_rounds=selected_round,
        require_supported_rank=require_supported_rank,
    )
    if (
        selected_round != target_round
        or relationship.get("relationship_present") is True
        or int(current_adapt.get("branch_id", -1)) != selected_branch_id
    ):
        raise ValueError(
            "frozen validator repair requires the round-target selected winner "
            "to equal the checkpoint frontier"
        )
    compact_current_receipt = _validate_compact_current_history(
        current_adapt.get("history"),
        selected_path=selected_path,
        expected_rounds=target_round,
    )
    controller_path = selected_path
    _validate_current_pointer(
        current=current,
        current_adapt=current_adapt,
        current_checkpoint=current_checkpoint,
        controller_path=controller_path,
        target_round=target_round,
    )
    _validate_selected_checkpoint_list(adapt, path=selected_path)
    if (
        int(adapt.get("ansatz_depth", -1))
        != int(selected_path["final_active_depth"])
        or int(segment.get("final_depth", -1))
        != int(selected_path["final_active_depth"])
    ):
        raise ValueError("selected-winner active depth drift")

    controller_full = _validate_path_closure(
        adapt=current_adapt,
        phase12_raw=current_checkpoint.get("phase12_energy_model_telemetry"),
        path=controller_path,
        label="controller frontier",
        expected_rounds=target_round,
        expected_fallback_policy=expected_fallback_policy,
    )
    selected_full = _validate_path_closure(
        adapt=adapt,
        phase12_raw=adapt.get("phase12_energy_model_telemetry"),
        path=selected_path,
        label="selected winner",
        expected_rounds=selected_round,
        expected_fallback_policy=expected_fallback_policy,
    )

    final_refit = _mapping(adapt.get("final_full_refit"), field="final full refit")
    terminal_prune = _mapping(
        adapt.get("post_prune_refit"), field="terminal post-prune refit"
    )
    if (
        final_refit.get("requested") is not False
        or final_refit.get("executed") is not False
        or terminal_prune.get("executed") is not False
    ):
        raise ValueError("terminal-only refit/prune alteration executed")

    selected_terminal = _mapping(
        adapt.get("terminal_active_prefix_checkpoint"),
        field="selected terminal checkpoint",
    )
    validate_checkpoint(
        selected_terminal,
        outer_iteration=selected_round,
        checkpoint_kind="terminal_post_final_refit_and_prune",
    )
    selected_last = selected_path["checkpoints"][-1]
    for field in STATE_IDENTITY_FIELDS:
        if selected_terminal.get(field) != selected_last.get(field):
            raise ValueError(f"terminal-only selected-state alteration: {field}")

    ledger = validate_ledger(
        ledger_sidecar,
        _mapping(adapt.get("estimator_call_accounting"), field="result accounting"),
    )
    receipts = _validate_beam_receipts(
        adapt=adapt,
        controller_checkpoints=controller_path["checkpoints"],
        selected_checkpoints=selected_path["checkpoints"],
        selected_terminal=selected_terminal,
        selected_round=selected_round,
        selected_branch_id=selected_branch_id,
        target_round=target_round,
        ledger=ledger,
    )
    return {
        "controller_rounds": target_round,
        "controller_frontier_active_depth": controller_path["final_active_depth"],
        "selected_final_controller_round": selected_round,
        "selected_final_active_depth": selected_path["final_active_depth"],
        "selected_terminal_checkpoint_sha256": str(
            selected_last["checkpoint_sha256"]
        ),
        "relationship_present": bool(
            relationship.get("relationship_present") is True
        ),
        "checkpoint_branch_policy": str(
            relationship.get("checkpoint_branch_policy") or ""
        ),
        "expected_cost_mode": expected_cost_mode,
        "expected_fallback_policy": expected_fallback_policy,
        "max_fixed_sector_leakage": max(
            controller_path["max_fixed_sector_leakage"],
            selected_path["max_fixed_sector_leakage"],
        ),
        "max_binary_padding_leakage": max(
            controller_path["max_binary_padding_leakage"],
            selected_path["max_binary_padding_leakage"],
        ),
        "controller_phase2_full_candidate_occurrences": controller_full,
        "selected_phase2_full_candidate_occurrences": selected_full,
        "controller_frontier_prune_rounds_executed": controller_path[
            "prune_rounds_executed"
        ],
        "controller_frontier_prune_rounds_accepted": controller_path[
            "prune_rounds_accepted"
        ],
        "selected_prune_rounds_executed": selected_path["prune_rounds_executed"],
        "selected_prune_rounds_accepted": selected_path["prune_rounds_accepted"],
        "ledger": ledger,
        "active_prefix_estimator_ledger_receipts": receipts,
        "compact_current_history_receipt": compact_current_receipt,
    }
