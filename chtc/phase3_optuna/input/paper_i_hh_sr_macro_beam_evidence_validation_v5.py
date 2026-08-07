#!/usr/bin/env python3
"""Fail-closed A/B validation for historical beam terminal-archive runs.

A is the controller/frontier receipt: all requested controller rounds and the
best recoverable frontier checkpoint.  B is the selected result branch, which
may be a shallower terminal retained by the configured legacy archive.  The
two are validated separately; this module never changes branch selection.
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
    if selected_round < target_round:
        terminal_branch = _mapping(
            relationship.get("diagnostic_terminal_branch"),
            field="diagnostic terminal branch",
        )
        frontier_branch = _mapping(
            relationship.get("recoverable_frontier_branch"),
            field="recoverable frontier branch",
        )
        if (
            relationship.get("schema_version")
            != "static_adapt_beam_final_checkpoint_relationship_v1"
            or relationship.get("relationship_present") is not True
            or relationship.get("reason")
            != "non_target_terminal_selected_with_recoverable_frontier"
            or relationship.get("checkpoint_branch_policy")
            != "best_frontier_branch"
            or relationship.get("recoverable_frontier_deeper_than_terminal")
            is not True
            or int(terminal_branch.get("depth_local", -1)) != selected_round
            or int(terminal_branch.get("history_count", -1)) != selected_round
            or terminal_branch.get("status") != "terminal"
            or int(frontier_branch.get("depth_local", -1)) != target_round
            or int(frontier_branch.get("history_count", -1)) != target_round
            or frontier_branch.get("status") != "frontier"
            or frontier_branch.get("terminated") is not False
        ):
            raise ValueError("shallow terminal/recoverable-frontier relationship drift")
    elif relationship.get("relationship_present") is True:
        raise ValueError("ordinary round-target winner has contradictory relationship")
    if (
        int(current_checkpoint.get("depth", -1)) != target_round
        or current_checkpoint.get("checkpoint_branch_policy")
        != "best_frontier_branch"
        or int(current_adapt.get("history_count", -1)) != target_round
        or current_adapt.get("history_checkpoint_complete") is not True
        or current_checkpoint.get("sr_route_profile_contract_sha256") != digest
    ):
        raise ValueError("controller-frontier current checkpoint drift")
    if selected_round < target_round and (
        int(current_checkpoint.get("branch_id", -1))
        != int(relationship.get("recoverable_frontier_branch_id", -2))
    ):
        raise ValueError("current frontier/relationship branch mismatch")
    return {
        "selected_round": selected_round,
        "relationship": relationship,
    }


def _validate_path(
    history_raw: Any,
    *,
    expected_rounds: int,
    require_supported_rank: bool,
    require_lane_receipt: bool,
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
        if require_lane_receipt:
            for field in (
                "static_lane_route",
                "physical_operator_lane",
                "physical_operator_quality",
                "physical_operator_hh_full_meta_class",
                "physical_operator_lane_source",
                "physical_operator_lane_health",
                "physical_operator_lane_relative_health",
                "physical_operator_lane_live",
            ):
                if field not in row:
                    raise ValueError(
                        f"round {outer_iteration}: missing beam lane field {field}"
                    )
            if not str(row.get("static_lane_route") or ""):
                raise ValueError(f"round {outer_iteration}: lane route absent")
            if not str(row.get("physical_operator_lane") or ""):
                raise ValueError(f"round {outer_iteration}: physical lane absent")
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
            row.get("accepted_refit"), field=f"round {outer_iteration} accepted refit"
        )
        invocation = _mapping(
            accepted.get("accepted_refit_invocation"),
            field=f"round {outer_iteration} accepted-refit invocation",
        )
        config = _mapping(
            invocation.get("config"), field=f"round {outer_iteration} refit config"
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


def _validate_beam_receipts(
    *,
    adapt: Mapping[str, Any],
    controller_checkpoints: Sequence[Mapping[str, Any]],
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
    ordinary_branch_ids: set[str] = set()
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
        parent_branch_id = receipt.get("parent_branch_id")
        if kind == "post_admission_prune":
            if outer < 1 or outer > target_round:
                raise ValueError(f"beam estimator receipt {sequence} round drift")
            if not branch_id or parent_branch_id is None or branch_id in ordinary_branch_ids:
                raise ValueError(f"beam estimator receipt {sequence} branch-link drift")
            ordinary_rounds.append(outer)
            ordinary_branch_ids.add(branch_id)
        elif kind == "terminal_post_final_refit_and_prune":
            if (
                sequence != len(receipts)
                or outer != selected_round
                or branch_id != str(selected_branch_id)
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
        cumulative_raw_components = _mapping(
            cumulative_raw.get("components"), field="cumulative raw components"
        )
        cumulative_unique_components = _mapping(
            cumulative_unique.get("components"),
            field="cumulative unique components",
        )
        raw_components = _mapping(raw_delta.get("components"), field="raw components")
        unique_components = _mapping(
            unique_delta.get("components"), field="unique components"
        )
        raw_values = {key: int(raw_components.get(key, -1)) for key in COMPONENTS}
        unique_values = {
            key: int(unique_components.get(key, -1)) for key in COMPONENTS
        }
        raw_total = int(raw_delta.get("total", -1))
        unique_total = int(unique_delta.get("S_alg", -1))
        current_raw = int(cumulative_raw.get("total", -1))
        current_unique = int(cumulative_unique.get("S_alg", -1))
        if (
            min(raw_values.values()) < 0
            or min(unique_values.values()) < 0
            or raw_total != sum(raw_values.values())
            or unique_total != sum(unique_values.values())
            or int(receipt.get("occurrence_sequence_start_exclusive", -1))
            != prior_raw
            or current_raw != prior_raw + raw_total
            or current_unique != prior_unique + unique_total
            or current_raw
            != sum(int(cumulative_raw_components.get(key, -1)) for key in COMPONENTS)
            or current_unique
            != sum(
                int(cumulative_unique_components.get(key, -1))
                for key in COMPONENTS
            )
        ):
            raise ValueError(f"beam estimator receipt {sequence} arithmetic drift")
        for key in COMPONENTS:
            summed_raw[key] += raw_values[key]
            summed_unique[key] += unique_values[key]
        prior_raw = current_raw
        prior_unique = current_unique
    if (
        ordinary_rounds != sorted(ordinary_rounds)
        or set(ordinary_rounds) != set(range(1, target_round + 1))
        or max(ordinary_rounds, default=-1) != target_round
        or len(ordinary_branch_ids) != len(receipts) - 1
    ):
        raise ValueError("beam ordinary receipt round/branch coverage drift")
    for index, checkpoint in enumerate(controller_checkpoints, start=1):
        if checkpoint.get("estimator_ledger_receipt") not in receipts:
            raise ValueError(f"controller checkpoint {index} receipt missing")
    if selected_terminal.get("estimator_ledger_receipt") != receipts[-1]:
        raise ValueError("selected terminal receipt mismatch")
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
    target_round: int,
    target_new_admissions: int,
    require_supported_rank: bool = True,
) -> dict[str, Any]:
    adapt = _mapping(result.get("adapt_vqe"), field="result adapt_vqe")
    settings = _mapping(result.get("settings"), field="result settings")
    current_adapt = _mapping(current.get("adapt_vqe"), field="current adapt_vqe")
    current_checkpoint = _mapping(current.get("checkpoint"), field="current checkpoint")
    for name, payload in (("settings", settings), ("result", adapt)):
        if payload.get("sr_route_profile_resolved") != profile:
            raise ValueError(f"{name} profile drift")
        if payload.get("sr_route_profile_contract_sha256") != digest:
            raise ValueError(f"{name} route digest drift")
    continuation = _mapping(adapt.get("continuation"), field="result continuation")
    beam = _mapping(continuation.get("beam_search"), field="beam search telemetry")
    if adapt.get("adapt_beam_enabled") is not True or adapt.get("success") is not True:
        raise ValueError("beam ADAPT execution status drift")
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
    relationship = _mapping(
        relationship_receipt["relationship"], field="validated relationship"
    )

    controller_path = _validate_path(
        current_adapt.get("history"),
        expected_rounds=target_round,
        require_supported_rank=require_supported_rank,
        require_lane_receipt=True,
    )
    selected_path = _validate_path(
        adapt.get("history"),
        expected_rounds=selected_round,
        require_supported_rank=require_supported_rank,
        require_lane_receipt=True,
    )
    if int(current_adapt.get("ansatz_depth", -1)) != int(
        controller_path["final_active_depth"]
    ):
        raise ValueError("controller-frontier active depth drift")
    if (
        int(adapt.get("ansatz_depth", -1)) != int(selected_path["final_active_depth"])
        or int(segment.get("final_depth", -1))
        != int(selected_path["final_active_depth"])
    ):
        raise ValueError("selected-winner active depth drift")
    route_state = _mapping(
        current_adapt.get("route_a_trust_region_state"),
        field="controller-frontier trust state",
    )
    if (
        route_state.get("schema") != "route_a_trust_region_state_v1"
        or int(route_state.get("update_count", -1)) != target_round
        or float(route_state.get("radius", float("nan")))
        != controller_path["final_radius"]
    ):
        raise ValueError("controller-frontier trust state drift")
    fallback = _mapping(
        current_adapt.get("all_energy_models_infeasible_novelty_fallback_telemetry"),
        field="controller-frontier fallback telemetry",
    )
    fallback_rounds = [
        int(value)
        for value in _sequence(
            fallback.get("controller_rounds"), field="fallback controller rounds"
        )
    ]
    if (
        fallback.get("enabled") is not True
        or int(fallback.get("query_charge_total", -1)) != 0
        or fallback_rounds != controller_path["fallback_rounds"]
        or int(fallback.get("activation_count", -1)) != len(fallback_rounds)
        or (fallback.get("fired") is True) != bool(fallback_rounds)
    ):
        raise ValueError("controller-frontier fallback closure drift")
    phase12 = _mapping(
        current_checkpoint.get("phase12_energy_model_telemetry"),
        field="controller-frontier Phase-I/II telemetry",
    )
    full = int(phase12.get("phase2_full_candidate_occurrences", -1))
    if (
        phase12.get("phase1_energy_model") != PHASE1_ENERGY_MODEL
        or phase12.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY
        or phase12.get("phase2_cheap_curvature_proxy_policy")
        != PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        or full <= 0
        or int(phase12.get("validated_phase2_curvature_receipt_occurrences", -2))
        != full
        or any(
            int(phase12.get(field, -1)) != 0
            for field in (
                "phase1_lambda_f_proxy_occurrences",
                "phase2_lambda_f_proxy_occurrences",
                "phase2_missing_curvature_fallback_occurrences",
            )
        )
    ):
        raise ValueError("controller-frontier Phase-I/II closure drift")
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
    for field in (
        "active_ansatz_depth",
        "ordered_active_operator_labels",
        "ordered_active_operators",
        "signed_unwrapped_logical_parameters",
        "signed_unwrapped_runtime_parameters",
        "parameterization",
        "projective_state_fingerprint",
    ):
        if selected_terminal.get(field) != selected_last.get(field):
            raise ValueError(f"terminal-only selected-state alteration: {field}")
    ledger = validate_ledger(
        ledger_sidecar,
        _mapping(adapt.get("estimator_call_accounting"), field="result accounting"),
    )
    receipts = _validate_beam_receipts(
        adapt=adapt,
        controller_checkpoints=controller_path["checkpoints"],
        selected_terminal=selected_terminal,
        selected_round=selected_round,
        selected_branch_id=int(relationship.get("diagnostic_terminal_branch_id", -1)),
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
        "max_fixed_sector_leakage": max(
            controller_path["max_fixed_sector_leakage"],
            selected_path["max_fixed_sector_leakage"],
        ),
        "max_binary_padding_leakage": max(
            controller_path["max_binary_padding_leakage"],
            selected_path["max_binary_padding_leakage"],
        ),
        "phase2_full_candidate_occurrences": full,
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
    }
