#!/usr/bin/env python3
"""Beam-search policy helpers for static ADAPT.

This module intentionally keeps behavior-level beam execution in
``adapt_pipeline.py``.  The helpers here are pure policy adapters around the
existing branch ranking and pruning primitives.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.checkpoint_telemetry import (
    _compact_prune_audit,
    _current_str_or_none,
    _history_tail_for_checkpoint,
    _optimizer_memory_contract_summary_payload,
    _phase3_surface_audit_payload,
)
from pipelines.static_adapt.controller_telemetry import (
    _branch_state_summary_payload,
)
from pipelines.static_adapt.engine_support import (
    _BeamBranchState,
    _BranchStepScratch,
    _beam_dedup,
    _beam_energy_cost_prune_key_payload,
    _beam_energy_cost_sort_key,
    _beam_gain_per_added_cost_prune_key_payload,
    _beam_gain_per_added_cost_sort_key,
    _beam_prune,
    _beam_prune_energy_cost_pareto_with_audit,
    _beam_prune_gain_per_added_cost_pareto_with_audit,
    _beam_prune_key,
    _beam_prune_key_payload,
)
from pipelines.static_adapt.run_control import (
    _benchmark_target_error_from_energy,
    _benchmark_target_hit_classification_payload,
)


@dataclass(frozen=True)
class _BeamParentRoundPolicy:
    parent_workers_requested: int
    parent_workers_effective: int
    parent_parallel_enabled: bool
    parent_parallel_disabled_reason: str | None
    branch_worker_budget: int


@dataclass(frozen=True)
class _BeamRoundPruneAuditSummary:
    audits: list[dict[str, Any]]
    permission_reason_counts: dict[str, int]
    child_count: int
    permission_open_count: int
    executed_count: int
    accepted_count: int


def _beam_prune_key_payload_for_policy(
    branch: _BeamBranchState,
    *,
    ordered_batch_beam_mode: bool,
    lambda_beam: float,
    canonical_beam_survival: bool = False,
    energy_root: float | None = None,
) -> dict[str, Any]:
    if bool(canonical_beam_survival):
        if energy_root is None:
            raise ValueError("Canonical beam survival requires energy_root.")
        return dict(
            _beam_gain_per_added_cost_prune_key_payload(
                branch,
                energy_root=float(energy_root),
                legacy_lambda_beam=float(lambda_beam),
            )
        )
    if bool(ordered_batch_beam_mode):
        return dict(
            _beam_energy_cost_prune_key_payload(
                branch,
                lambda_beam=float(lambda_beam),
            )
        )
    return dict(_beam_prune_key_payload(branch))


def _beam_sort_key_for_policy(
    branch: _BeamBranchState,
    *,
    ordered_batch_beam_mode: bool,
    lambda_beam: float,
    canonical_beam_survival: bool = False,
    energy_root: float | None = None,
) -> tuple[Any, ...]:
    if bool(canonical_beam_survival):
        if energy_root is None:
            raise ValueError("Canonical beam survival requires energy_root.")
        return _beam_gain_per_added_cost_sort_key(
            branch,
            energy_root=float(energy_root),
            legacy_lambda_beam=float(lambda_beam),
        )
    if bool(ordered_batch_beam_mode):
        return _beam_energy_cost_sort_key(
            branch,
            lambda_beam=float(lambda_beam),
        )
    return _beam_prune_key(branch)


def _beam_prune_for_policy(
    branches: Sequence[_BeamBranchState],
    *,
    cap: int,
    ordered_batch_beam_mode: bool,
    lambda_beam: float,
    source: str | None = None,
    canonical_beam_survival: bool = False,
    energy_root: float | None = None,
    cost_contract: Mapping[str, Any] | None = None,
) -> tuple[list[_BeamBranchState], dict[str, Any] | None]:
    if bool(canonical_beam_survival):
        if energy_root is None:
            raise ValueError("Canonical beam survival requires energy_root.")
        kept, audit = _beam_prune_gain_per_added_cost_pareto_with_audit(
            branches,
            cap=int(cap),
            energy_root=float(energy_root),
            legacy_lambda_beam=float(lambda_beam),
            cost_contract=cost_contract,
        )
        if source is not None:
            audit["source"] = str(source)
        return kept, dict(audit)
    if bool(ordered_batch_beam_mode):
        kept, audit = _beam_prune_energy_cost_pareto_with_audit(
            branches,
            cap=int(cap),
            lambda_beam=float(lambda_beam),
        )
        if source is not None:
            audit["source"] = str(source)
        return kept, dict(audit)
    return _beam_prune(branches, cap=int(cap)), None


def _beam_dedup_for_policy(
    branches: Sequence[_BeamBranchState],
    *,
    ordered_batch_beam_mode: bool,
    lambda_beam: float,
    source: str | None = None,
    canonical_beam_survival: bool = False,
    energy_root: float | None = None,
    cost_contract: Mapping[str, Any] | None = None,
) -> tuple[list[_BeamBranchState], dict[str, Any] | None]:
    if bool(canonical_beam_survival):
        return _beam_prune_for_policy(
            branches,
            cap=int(len(branches)),
            ordered_batch_beam_mode=bool(ordered_batch_beam_mode),
            lambda_beam=float(lambda_beam),
            source=source,
            canonical_beam_survival=True,
            energy_root=energy_root,
            cost_contract=cost_contract,
        )
    if bool(ordered_batch_beam_mode):
        return _beam_prune_for_policy(
            branches,
            cap=int(len(branches)),
            ordered_batch_beam_mode=True,
            lambda_beam=float(lambda_beam),
            source=source,
        )
    return _beam_dedup(branches), None


def _resolve_beam_parent_round_policy(
    *,
    frontier_input_count: int,
    requested_parent_workers: int,
    adapt_parallel_gradient_workers: int,
    finite_angle_fallback: bool,
    cap_worker_limit_for_items: Callable[[int, int], int],
) -> _BeamParentRoundPolicy:
    parent_workers_requested = int(max(1, int(requested_parent_workers)))
    parent_workers_effective = int(
        cap_worker_limit_for_items(
            int(parent_workers_requested),
            int(frontier_input_count),
        )
    )
    parent_parallel_enabled = bool(
        int(parent_workers_effective) > 1
        and int(frontier_input_count) > 1
        and not bool(finite_angle_fallback)
    )
    parent_parallel_disabled_reason = None
    if int(parent_workers_requested) <= 1:
        parent_parallel_disabled_reason = "workers_leq_one"
    elif int(frontier_input_count) <= 1:
        parent_parallel_disabled_reason = "single_parent_frontier"
    elif bool(finite_angle_fallback):
        parent_parallel_disabled_reason = "finite_angle_fallback_mutates_nfev"
    branch_worker_budget = int(
        max(
            1,
            int(adapt_parallel_gradient_workers)
            // max(1, int(parent_workers_effective))
            if bool(parent_parallel_enabled)
            else int(adapt_parallel_gradient_workers),
        )
    )
    return _BeamParentRoundPolicy(
        parent_workers_requested=int(parent_workers_requested),
        parent_workers_effective=int(parent_workers_effective),
        parent_parallel_enabled=bool(parent_parallel_enabled),
        parent_parallel_disabled_reason=parent_parallel_disabled_reason,
        branch_worker_budget=int(branch_worker_budget),
    )


def _beam_round_best_record_value(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    best_value: float | None = None
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        try:
            numeric_value = float(rec.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_value):
            continue
        if best_value is None or numeric_value > best_value:
            best_value = float(numeric_value)
    return best_value


def _beam_round_update_best(
    diagnostic: dict[str, Any],
    field: str,
    value: Any,
) -> None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(numeric_value):
        return
    current_value = diagnostic.get(field)
    if current_value is None or numeric_value > float(current_value):
        diagnostic[field] = float(numeric_value)


def _accumulate_beam_round_frontier_diagnostic(
    diagnostic: dict[str, Any],
    scratch: _BranchStepScratch,
) -> None:
    def _scratch_count(field: str, fallback: int) -> int:
        value = getattr(scratch, field, None)
        return int(fallback if value is None else value)

    diagnostic["raw_candidate_record_count"] = int(
        diagnostic["raw_candidate_record_count"]
    ) + _scratch_count("phase1_raw_record_count", 0)
    diagnostic["phase2_raw_candidate_record_count"] = int(
        diagnostic["phase2_raw_candidate_record_count"]
    ) + _scratch_count("phase2_raw_record_count", len(scratch.phase2_last_shortlist_eval_records))
    diagnostic["phase1_shortlist_size"] = int(
        diagnostic["phase1_shortlist_size"]
    ) + _scratch_count("phase1_shortlist_size", len(scratch.phase1_last_retained_records))
    diagnostic["phase2_shortlist_size"] = int(
        diagnostic["phase2_shortlist_size"]
    ) + _scratch_count("phase2_shortlist_size", len(scratch.phase2_last_geometric_shortlist_records))
    diagnostic["phase3_shortlist_size"] = int(
        diagnostic["phase3_shortlist_size"]
    ) + _scratch_count("phase3_shortlist_size", len(scratch.phase2_last_retained_shortlist_records))
    _beam_round_update_best(diagnostic, "best_available_gradient", scratch.max_grad)
    records_for_best: list[Mapping[str, Any]] = [
        *[dict(row) for row in scratch.phase1_last_retained_records],
        *[dict(row) for row in scratch.phase2_last_shortlist_eval_records],
        *[dict(row) for row in scratch.phase2_last_geometric_shortlist_records],
        *[dict(row) for row in scratch.phase2_last_retained_shortlist_records],
        *[dict(row) for row in scratch.phase2_last_admitted_records],
    ]
    for key, field in (
        ("simple_score", "best_available_simple_score"),
        ("phase2_raw_score", "best_available_phase2_raw_score"),
        ("full_v2_score", "best_available_full_v2_score"),
        ("phase2_raw_trust_gain", "best_available_gain"),
    ):
        _beam_round_update_best(diagnostic, field, _beam_round_best_record_value(records_for_best, key))
    parent_reason = (
        str(scratch.stop_reason)
        if scratch.stop_reason is not None
        else ("expanded" if scratch.proposals else "empty")
    )
    reason_counts = dict(diagnostic.get("parent_stop_reason_counts", {}))
    reason_counts[parent_reason] = int(reason_counts.get(parent_reason, 0)) + 1
    diagnostic["parent_stop_reason_counts"] = reason_counts


def _beam_base_branch_from_parent_scratch(
    parent: _BeamBranchState,
    scratch: _BranchStepScratch,
    *,
    branch_id: int,
    parent_branch_id: int | None,
) -> _BeamBranchState:
    base_branch = parent.clone_for_child(branch_id=int(branch_id))
    base_branch.parent_branch_id = (
        None if parent_branch_id is None else int(parent_branch_id)
    )
    base_branch.energy_current = float(scratch.energy_current)
    base_branch.available_indices = set(
        int(x) for x in scratch.available_indices_after_transition
    )
    base_branch.phase1_stage = scratch.phase1_stage_after_transition.clone()
    base_branch.phase1_residual_opened = bool(scratch.phase1_residual_opened)
    base_branch.phase1_stage_events = [
        dict(x) for x in scratch.phase1_stage_events_after_transition
    ]
    base_branch.phase1_last_probe_reason = str(scratch.phase1_last_probe_reason)
    base_branch.phase1_last_positions_considered = [
        int(x) for x in scratch.phase1_last_positions_considered
    ]
    base_branch.phase1_last_trough_detected = bool(
        scratch.phase1_last_trough_detected
    )
    base_branch.phase1_last_trough_probe_triggered = bool(
        scratch.phase1_last_trough_probe_triggered
    )
    base_branch.phase1_last_selected_score = scratch.phase1_last_selected_score
    base_branch.phase1_last_retained_records = [
        dict(x) for x in scratch.phase1_last_retained_records
    ]
    base_branch.phase2_last_shortlist_records = [
        dict(x) for x in scratch.phase2_last_shortlist_records
    ]
    base_branch.phase2_last_geometric_shortlist_records = [
        dict(x) for x in scratch.phase2_last_geometric_shortlist_records
    ]
    base_branch.phase2_last_retained_shortlist_records = [
        dict(x) for x in scratch.phase2_last_retained_shortlist_records
    ]
    base_branch.phase2_last_admitted_records = [
        dict(x) for x in scratch.phase2_last_admitted_records
    ]
    base_branch.phase2_last_batch_selected = bool(
        scratch.phase2_last_batch_selected
    )
    base_branch.phase2_last_batch_penalty_total = float(
        scratch.phase2_last_batch_penalty_total
    )
    base_branch.phase2_last_batch_schur_context = copy.deepcopy(
        scratch.phase2_last_batch_schur_context
    )
    base_branch.phase2_last_optimizer_memory_reused = bool(
        scratch.phase2_last_optimizer_memory_reused
    )
    base_branch.phase2_last_optimizer_memory_source = str(
        scratch.phase2_last_optimizer_memory_source
    )
    base_branch.phase2_last_shortlist_eval_records = [
        dict(x) for x in scratch.phase2_last_shortlist_eval_records
    ]
    base_branch.phase3_runtime_split_summary = copy.deepcopy(
        scratch.phase3_runtime_split_summary_after_eval
    )
    base_branch.controller_measurement_work = scratch.controller_measurement_work_after_eval.clone()
    return base_branch


def _beam_terminal_child_from_scratch(
    base_branch: _BeamBranchState,
    scratch: _BranchStepScratch,
) -> _BeamBranchState:
    terminal_branch = base_branch.clone_for_child(branch_id=int(base_branch.branch_id))
    terminal_branch.parent_branch_id = (
        None if base_branch.parent_branch_id is None else int(base_branch.parent_branch_id)
    )
    terminal_branch.last_transition_kind = "stop_child"
    terminal_branch.last_admission_record_count = 0
    terminal_branch.terminated = True
    terminal_branch.stop_reason = (
        str(scratch.stop_reason)
        if scratch.stop_reason is not None
        else ("stop" if scratch.proposals else "empty")
    )
    return terminal_branch


def _beam_round_prune_audit_summary(
    round_admission_children: Sequence[_BeamBranchState],
    *,
    compact_prune_audit: Callable[[Mapping[str, Any] | None], dict[str, Any]],
) -> _BeamRoundPruneAuditSummary:
    audits = [
        dict(compact_prune_audit(branch.phase1_last_prune_summary))
        for branch in round_admission_children
    ]
    reason_counts: dict[str, int] = {}
    for audit in audits:
        reason_key = str(audit.get("permission_reason", "unknown"))
        reason_counts[reason_key] = int(reason_counts.get(reason_key, 0)) + 1
    return _BeamRoundPruneAuditSummary(
        audits=[dict(audit) for audit in audits],
        permission_reason_counts=dict(reason_counts),
        child_count=int(len(round_admission_children)),
        permission_open_count=int(
            sum(1 for audit in audits if bool(audit.get("permission_open", False)))
        ),
        executed_count=int(
            sum(1 for audit in audits if bool(audit.get("executed", False)))
        ),
        accepted_count=int(
            sum(int(audit.get("accepted_count", 0) or 0) for audit in audits)
        ),
    )


def _beam_round_stop_reason(
    *,
    frontier_input_count: int,
    frontier_kept_count: int,
    proposal_family_count: int,
    parent_stop_reason_counts: Mapping[str, Any],
) -> str | None:
    if not (
        int(frontier_input_count) > 0
        and int(frontier_kept_count) == 0
        and int(proposal_family_count) == 0
    ):
        return None
    reason_counts = dict(parent_stop_reason_counts)
    if len(reason_counts) == 1:
        return str(next(iter(reason_counts)))
    if reason_counts:
        return "mixed"
    return "empty"


def _beam_round_diagnostics_payload(
    *,
    depth: int,
    frontier_input_count: int,
    parents_expanded_count: int,
    proposals_selected_count: int,
    proposal_family_count: int,
    stop_children_count: int,
    child_frontier_count: int,
    round_terminal_count: int,
    active_children_unique_count: int,
    frontier_kept_count: int,
    round_live_cap: int,
    terminal_pool_candidate_count: int,
    terminal_pool_unique_count: int,
    terminal_kept_count: int,
    round_stop_reason: str | None,
    beam_parent_workers_requested: int,
    beam_parent_workers_effective: int,
    beam_parent_parallel_enabled: bool,
    beam_parent_parallel_disabled_reason: str | None,
    beam_parent_result_elapsed_s: Sequence[float],
    round_frontier_diagnostic: Mapping[str, Any],
    round_prune_audit_summary: _BeamRoundPruneAuditSummary,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "depth": int(depth + 1),
        "frontier_input_count": int(frontier_input_count),
        "parents_expanded_count": int(parents_expanded_count),
        "proposals_selected_count": int(proposals_selected_count),
        "proposal_family_count": int(proposal_family_count),
        "stop_children_count": int(stop_children_count),
        "children_materialized_count": int(
            child_frontier_count + round_terminal_count
        ),
        "active_children_raw_count": int(child_frontier_count),
        "active_children_unique_count": int(active_children_unique_count),
        "frontier_kept_count": int(frontier_kept_count),
        "frontier_cap_effective": int(max(1, int(round_live_cap))),
        "round_terminals_raw_count": int(round_terminal_count),
        "terminal_pool_candidate_count": int(terminal_pool_candidate_count),
        "terminal_pool_unique_count": int(terminal_pool_unique_count),
        "terminal_kept_count": int(terminal_kept_count),
        "stop_reason": round_stop_reason,
        "beam_parent_workers_requested": int(beam_parent_workers_requested),
        "beam_parent_workers_effective": int(beam_parent_workers_effective),
        "beam_parent_parallel_enabled": bool(beam_parent_parallel_enabled),
        "beam_parent_parallel_disabled_reason": beam_parent_parallel_disabled_reason,
        "beam_parent_parallel_merge_order": "frontier_order",
        "beam_parent_result_elapsed_s": [
            float(elapsed_s) for elapsed_s in beam_parent_result_elapsed_s
        ],
    }
    payload.update(dict(round_frontier_diagnostic))
    payload.update(
        {
            "prune_child_count": int(round_prune_audit_summary.child_count),
            "prune_permission_open_count": int(
                round_prune_audit_summary.permission_open_count
            ),
            "prune_executed_count": int(round_prune_audit_summary.executed_count),
            "prune_accepted_count": int(round_prune_audit_summary.accepted_count),
            "prune_permission_reason_counts": dict(
                round_prune_audit_summary.permission_reason_counts
            ),
            "prune_audits": [
                dict(audit) for audit in round_prune_audit_summary.audits
            ],
        }
    )
    return payload


def _beam_round_done_log_payload(
    *,
    depth: int,
    frontier_input_count: int,
    parents_expanded_count: int,
    proposals_selected_count: int,
    proposal_family_count: int,
    stop_children_count: int,
    frontier_kept_count: int,
    round_live_cap: int,
    terminal_kept_count: int,
    round_stop_reason: str | None,
    beam_parent_workers_requested: int,
    beam_parent_workers_effective: int,
    beam_parent_parallel_enabled: bool,
    beam_parent_parallel_disabled_reason: str | None,
    round_frontier_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "depth": int(depth + 1),
        "frontier_input_count": int(frontier_input_count),
        "parents_expanded_count": int(parents_expanded_count),
        "proposals_selected_count": int(proposals_selected_count),
        "proposal_family_count": int(proposal_family_count),
        "stop_children_count": int(stop_children_count),
        "frontier_kept_count": int(frontier_kept_count),
        "frontier_cap_effective": int(max(1, int(round_live_cap))),
        "terminal_kept_count": int(terminal_kept_count),
        "stop_reason": round_stop_reason,
        "beam_parent_workers_requested": int(beam_parent_workers_requested),
        "beam_parent_workers_effective": int(beam_parent_workers_effective),
        "beam_parent_parallel_enabled": bool(beam_parent_parallel_enabled),
        "beam_parent_parallel_disabled_reason": beam_parent_parallel_disabled_reason,
        "beam_parent_parallel_merge_order": "frontier_order",
    }
    payload.update(dict(round_frontier_diagnostic))
    return payload


def _formal_manifold_branch_payloads(
    branch: _BeamBranchState,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    runtime = getattr(branch, "formal_manifold_runtime", None)
    if runtime is None:
        return None, None
    checkpoint_builder = getattr(runtime, "checkpoint_payload", None)
    behavior_builder = getattr(runtime, "behavioral_fingerprint_payload", None)
    if not callable(checkpoint_builder) or not callable(behavior_builder):
        raise TypeError(
            "formal_manifold_runtime must expose checkpoint_payload() and "
            "behavioral_fingerprint_payload()."
        )
    checkpoint = checkpoint_builder()
    behavior = behavior_builder()
    if not isinstance(checkpoint, Mapping) or not isinstance(behavior, Mapping):
        raise TypeError("formal-manifold runtime payloads must be mappings.")
    return copy.deepcopy(dict(checkpoint)), copy.deepcopy(dict(behavior))


def _beam_branch_replay_summary_payload(
    branch: _BeamBranchState,
    *,
    keep_history_tail: int,
    benchmark_stop_reference_energy: float,
    benchmark_target_abs_delta_e: float | None,
    beam_prune_key_payload: Callable[[_BeamBranchState], Mapping[str, Any]],
) -> dict[str, Any]:
    formal_checkpoint, formal_behavior = _formal_manifold_branch_payloads(branch)
    history_tail = _history_tail_for_checkpoint(
        branch.history,
        keep_history_tail=int(keep_history_tail),
    )
    target_error = _benchmark_target_error_from_energy(
        energy_value=branch.energy_current,
        reference_energy=float(benchmark_stop_reference_energy),
    )
    target_classification = _benchmark_target_hit_classification_payload(
        stop_reason_snapshot=(
            None if branch.stop_reason is None else str(branch.stop_reason)
        ),
        target_error=target_error,
        target_threshold=benchmark_target_abs_delta_e,
        source="beam_branch_replay_summary",
    )
    return {
        "branch_id": int(branch.branch_id),
        "parent_branch_id": (
            None if branch.parent_branch_id is None else int(branch.parent_branch_id)
        ),
        "status": "terminal" if bool(branch.terminated) else "frontier",
        "terminated": bool(branch.terminated),
        "stop_reason": _current_str_or_none(branch.stop_reason),
        "depth_local": int(branch.depth_local),
        "ansatz_depth": int(len(branch.selected_ops)),
        "energy": float(branch.energy_current),
        "benchmark_target_abs_delta_current": target_error,
        "benchmark_target_abs_delta_e": benchmark_target_abs_delta_e,
        "benchmark_target_error_within_threshold": bool(
            target_classification.get("target_error_within_threshold", False)
        ),
        "benchmark_target_hit": bool(
            target_classification.get("target_hit_success", False)
        ),
        "benchmark_target_classification": dict(target_classification),
        "operator_labels": [str(op.label) for op in branch.selected_ops],
        "last_selected_records": (
            list(history_tail[-1].get("selected_records", [])) if history_tail else []
        ),
        "history_count": int(len(branch.history)),
        "history_tail_count": int(len(history_tail)),
        "history_tail": history_tail,
        "route_a_trust_region_state": (
            None
            if branch.route_a_trust_region_state is None
            else branch.route_a_trust_region_state.as_dict()
        ),
        "formal_manifold_runtime_checkpoint": formal_checkpoint,
        "formal_manifold_behavioral_fingerprint": formal_behavior,
        "frontier_prune_key": dict(beam_prune_key_payload(branch)),
    }


def _beam_branch_summary_payload(
    branch: _BeamBranchState,
    *,
    benchmark_stop_reference_energy: float,
    benchmark_target_abs_delta_e: float | None,
    generator_ids: Sequence[str],
    beam_prune_key_payload: Callable[[_BeamBranchState], Mapping[str, Any]],
) -> dict[str, Any]:
    formal_checkpoint, formal_behavior = _formal_manifold_branch_payloads(branch)
    prune_history = [
        _compact_prune_audit(row.get("post_admission_prune"))
        for row in branch.history
        if isinstance(row, Mapping)
        and isinstance(row.get("post_admission_prune"), Mapping)
    ]
    branch_controller_snapshot = branch.phase1_stage.snapshot().get("last_snapshot")
    target_error = _benchmark_target_error_from_energy(
        energy_value=branch.energy_current,
        reference_energy=float(benchmark_stop_reference_energy),
    )
    target_classification = _benchmark_target_hit_classification_payload(
        stop_reason_snapshot=(
            None if branch.stop_reason is None else str(branch.stop_reason)
        ),
        target_error=target_error,
        target_threshold=benchmark_target_abs_delta_e,
        source="beam_branch_summary",
    )
    return {
        "branch_id": int(branch.branch_id),
        "parent_branch_id": (
            None if branch.parent_branch_id is None else int(branch.parent_branch_id)
        ),
        "depth_local": int(branch.depth_local),
        "stop_reason": (
            None if branch.stop_reason is None else str(branch.stop_reason)
        ),
        "terminated": bool(branch.terminated),
        "status": ("terminal" if bool(branch.terminated) else "frontier"),
        "termination_label": (
            str(branch.stop_reason)
            if bool(branch.terminated) and branch.stop_reason is not None
            else None
        ),
        "last_transition_kind": str(branch.last_transition_kind),
        "last_admission_record_count": int(branch.last_admission_record_count),
        "energy": float(branch.energy_current),
        "benchmark_target_abs_delta_current": target_error,
        "benchmark_target_abs_delta_e": benchmark_target_abs_delta_e,
        "benchmark_target_error_within_threshold": bool(
            target_classification.get("target_error_within_threshold", False)
        ),
        "benchmark_target_hit": bool(
            target_classification.get("target_hit_success", False)
        ),
        "benchmark_target_classification": dict(target_classification),
        "cumulative_selector_score": float(branch.cumulative_selector_score),
        "cumulative_selector_burden": float(branch.cumulative_selector_burden),
        "scored_surface_count": int(len(branch.phase2_last_shortlist_records)),
        "retained_shortlist_count": int(
            len(branch.phase2_last_retained_shortlist_records)
        ),
        "admitted_count": int(len(branch.phase2_last_admitted_records)),
        "phase3_surface_summary": _phase3_surface_audit_payload(
            scored_rows=branch.phase2_last_shortlist_records,
            retained_rows=branch.phase2_last_retained_shortlist_records,
            admitted_rows=branch.phase2_last_admitted_records,
            beam_enabled=True,
        ),
        "prune_key": dict(beam_prune_key_payload(branch)),
        "last_prune": _compact_prune_audit(branch.phase1_last_prune_summary),
        "prune_history": [dict(x) for x in prune_history],
        "route_a_trust_region_state": (
            None
            if branch.route_a_trust_region_state is None
            else branch.route_a_trust_region_state.as_dict()
        ),
        "formal_manifold_runtime_checkpoint": formal_checkpoint,
        "formal_manifold_behavioral_fingerprint": formal_behavior,
        "branch_state_summary": _branch_state_summary_payload(
            beam_enabled=True,
            branch_id=int(branch.branch_id),
            parent_branch_id=(
                None if branch.parent_branch_id is None else int(branch.parent_branch_id)
            ),
            history_rows=branch.history,
            depth_local=int(branch.depth_local),
            ansatz_depth=int(len(branch.selected_ops)),
            terminated=bool(branch.terminated),
            termination_label=(
                None if branch.stop_reason is None else str(branch.stop_reason)
            ),
            cumulative_selector_score=float(branch.cumulative_selector_score),
            cumulative_selector_burden=float(branch.cumulative_selector_burden),
            stage_name=str(branch.phase1_stage.stage_name),
            residual_opened=bool(branch.phase1_residual_opened),
            last_probe_reason=str(branch.phase1_last_probe_reason),
            stage_events=branch.phase1_stage_events,
            last_snapshot=branch_controller_snapshot,
        ),
        "optimizer_memory_contract_summary": _optimizer_memory_contract_summary_payload(
            beam_enabled=True,
            branch_id=int(branch.branch_id),
            memory_state=branch.phase2_optimizer_memory,
            operator_labels=[str(op.label) for op in branch.selected_ops],
            generator_ids=[str(x) for x in generator_ids],
            num_parameters=int(np.asarray(branch.theta, dtype=float).size),
            last_active_subset_source=str(branch.phase2_last_optimizer_memory_source),
            last_active_subset_reused=bool(branch.phase2_last_optimizer_memory_reused),
        ),
    }


def _beam_final_diagnostics_payload(
    *,
    frontier: Sequence[_BeamBranchState],
    terminals: Sequence[_BeamBranchState],
    finalists: Sequence[_BeamBranchState],
    winner_branch: _BeamBranchState,
    winner_target_classification: Mapping[str, Any],
    beam_sort_key: Callable[[_BeamBranchState], Any],
    branch_state_fingerprint: Callable[[_BeamBranchState], Any],
    beam_prune_key_payload: Callable[[_BeamBranchState], Mapping[str, Any]],
    beam_branch_replay_summary: Callable[[_BeamBranchState], Mapping[str, Any]],
    beam_branch_summary: Callable[[_BeamBranchState], Mapping[str, Any]],
    beam_survival_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final_frontier_sorted = sorted(list(frontier), key=beam_sort_key)
    recoverable_frontier_branch = (
        final_frontier_sorted[0] if final_frontier_sorted else None
    )
    relationship_present = bool(
        recoverable_frontier_branch is not None
        and bool(winner_branch.terminated)
        and not bool(winner_target_classification.get("target_hit_success", False))
    )
    final_checkpoint_relationship = {
        "schema_version": "static_adapt_beam_final_checkpoint_relationship_v1",
        "relationship_present": relationship_present,
        "reason": (
            "non_target_terminal_selected_with_recoverable_frontier"
            if relationship_present
            else "not_applicable"
        ),
        "diagnostic_terminal_branch_id": int(winner_branch.branch_id),
        "diagnostic_terminal_stop_reason": str(
            winner_branch.stop_reason or "max_depth"
        ),
        "diagnostic_terminal_target_hit_classification": dict(
            winner_target_classification
        ),
        "recoverable_frontier_branch_id": (
            None
            if recoverable_frontier_branch is None
            else int(recoverable_frontier_branch.branch_id)
        ),
        "recoverable_frontier_parent_branch_id": (
            None
            if recoverable_frontier_branch is None
            or recoverable_frontier_branch.parent_branch_id is None
            else int(recoverable_frontier_branch.parent_branch_id)
        ),
        "recoverable_frontier_deeper_than_terminal": bool(
            recoverable_frontier_branch is not None
            and int(recoverable_frontier_branch.depth_local)
            > int(winner_branch.depth_local)
        ),
        "checkpoint_branch_policy": (
            "best_frontier_branch"
            if recoverable_frontier_branch is not None
            else "best_terminal_or_root_branch"
        ),
        "diagnostic_terminal_branch": dict(
            beam_branch_replay_summary(winner_branch)
        ),
        "recoverable_frontier_branch": (
            None
            if recoverable_frontier_branch is None
            else dict(beam_branch_replay_summary(recoverable_frontier_branch))
        ),
    }
    winner_prune_key = dict(beam_prune_key_payload(winner_branch))
    winner_branch_summary = dict(beam_branch_summary(winner_branch))
    return {
        "frontier_final_count": int(len(frontier)),
        "terminal_final_count": int(len(terminals)),
        "finalist_count": int(len(finalists)),
        "winner_branch_id": int(winner_branch.branch_id),
        "winner_parent_branch_id": (
            None
            if winner_branch.parent_branch_id is None
            else int(winner_branch.parent_branch_id)
        ),
        "winner_stop_reason": str(winner_branch.stop_reason or "max_depth"),
        "winner_target_hit_success": bool(
            winner_target_classification.get("target_hit_success", False)
        ),
        "winner_target_non_hit_reason": winner_target_classification.get(
            "non_hit_reason"
        ),
        "winner_target_hit_classification": dict(winner_target_classification),
        "final_checkpoint_relationship": copy.deepcopy(
            final_checkpoint_relationship
        ),
        "winner_fingerprint": str(branch_state_fingerprint(winner_branch)),
        "winner_prune_key": winner_prune_key,
        "winner_survival_key": dict(winner_prune_key),
        "winner_prune_summary": _compact_prune_audit(
            winner_branch.phase1_last_prune_summary
        ),
        "winner_branch_summary": winner_branch_summary,
        "winner_branch_state_summary": dict(
            winner_branch_summary.get("branch_state_summary", {})
        ),
        "survival_audits": [dict(audit) for audit in beam_survival_audits],
        "winner_optimizer_memory_contract": dict(
            winner_branch_summary.get("optimizer_memory_contract_summary", {})
        ),
        "finalist_summaries": [
            beam_branch_summary(branch)
            for branch in sorted(finalists, key=beam_sort_key)
        ],
    }


def _beam_replay_round_payload(
    *,
    depth: int,
    frontier_input_count: int,
    parents_expanded_count: int,
    proposals_selected_count: int,
    proposal_family_count: int,
    stop_children_count: int,
    round_live_cap: int,
    round_stop_reason: str | None,
    round_frontier_diagnostic: Mapping[str, Any],
    frontier_branches: Sequence[_BeamBranchState],
    terminal_branches: Sequence[_BeamBranchState],
    round_terminal_branches: Sequence[_BeamBranchState],
    beam_sort_key: Callable[[_BeamBranchState], Any],
    beam_branch_replay_summary: Callable[[_BeamBranchState], dict[str, Any]],
    current_str_or_none: Callable[[Any], str | None],
    current_float: Callable[[Any], float | None],
) -> dict[str, Any]:
    frontier_sorted = sorted(list(frontier_branches), key=beam_sort_key)
    terminal_sorted = sorted(list(terminal_branches), key=beam_sort_key)
    round_terminal_sorted = sorted(
        list(round_terminal_branches),
        key=beam_sort_key,
    )
    return {
        "schema_version": "static_adapt_beam_replay_round_v1",
        "depth": int(depth),
        "stop_reason": current_str_or_none(round_stop_reason),
        "frontier_input_count": int(frontier_input_count),
        "parents_expanded_count": int(parents_expanded_count),
        "proposals_selected_count": int(proposals_selected_count),
        "proposal_family_count": int(proposal_family_count),
        "stop_children_count": int(stop_children_count),
        "parent_stop_reason_counts": dict(
            round_frontier_diagnostic.get("parent_stop_reason_counts", {})
        ),
        "best_available": {
            "gradient": current_float(
                round_frontier_diagnostic.get("best_available_gradient")
            ),
            "simple_score": current_float(
                round_frontier_diagnostic.get("best_available_simple_score")
            ),
            "phase2_raw_score": current_float(
                round_frontier_diagnostic.get("best_available_phase2_raw_score")
            ),
            "full_v2_score": current_float(
                round_frontier_diagnostic.get("best_available_full_v2_score")
            ),
            "gain": current_float(
                round_frontier_diagnostic.get("best_available_gain")
            ),
        },
        "frontier_summary": {
            "kept_count": int(len(frontier_sorted)),
            "cap_effective": int(max(1, int(round_live_cap))),
            "branch_ids": [int(branch.branch_id) for branch in frontier_sorted],
        },
        "terminal_summary": {
            "kept_count": int(len(terminal_sorted)),
            "round_terminal_count": int(len(round_terminal_sorted)),
            "branch_ids": [int(branch.branch_id) for branch in terminal_sorted],
            "round_branch_ids": [
                int(branch.branch_id) for branch in round_terminal_sorted
            ],
            "stop_reason_counts": {
                reason: sum(
                    1
                    for branch in terminal_sorted
                    if str(branch.stop_reason) == reason
                )
                for reason in sorted(
                    {str(branch.stop_reason) for branch in terminal_sorted}
                )
            },
        },
        "frontier": {
            "branches": [
                beam_branch_replay_summary(branch) for branch in frontier_sorted
            ],
        },
        "terminal": {
            "branches": [
                beam_branch_replay_summary(branch) for branch in terminal_sorted
            ],
            "round_branches": [
                beam_branch_replay_summary(branch)
                for branch in round_terminal_sorted
            ],
        },
    }


def _beam_replay_telemetry_payload(
    *,
    depth: int,
    round_replay_payload: Mapping[str, Any],
    beam_replay_rounds: Sequence[Mapping[str, Any]],
    replay_tail_count: int,
    leading_branch: _BeamBranchState,
    checkpoint_branch: _BeamBranchState,
    has_checkpoint_frontier_candidates: bool,
    beam_branch_replay_summary: Callable[[_BeamBranchState], dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "static_adapt_beam_replay_telemetry_v1",
        "depth": int(depth + 1),
        "current_round": copy.deepcopy(dict(round_replay_payload)),
        "rounds": copy.deepcopy(
            list(beam_replay_rounds)[-int(replay_tail_count):]
        ),
        "leading_branch": beam_branch_replay_summary(leading_branch),
        "checkpoint_branch": beam_branch_replay_summary(checkpoint_branch),
        "checkpoint_branch_policy": (
            "best_frontier_branch"
            if bool(has_checkpoint_frontier_candidates)
            else "best_terminal_or_root_branch"
        ),
    }
