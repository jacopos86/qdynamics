"""Fail-closed evidence checks for the Paper-I query-neutral prune route.

This module is copied into an immutable CHTC bundle beside a frozen copy of
the validated no-overlap parent validator, imported as
``evidence_validation_parent``.  It intentionally separates:

* direct prune measurement overhead, which must be exactly zero; and
* trajectory-induced work through first target hit, which remains part of the
  ordinary strict estimator ledger.
"""

from __future__ import annotations

import collections
import copy
import math
from typing import Any, Mapping, Sequence

from evidence_validation_parent import (
    validate_active_prefix_estimator_receipts,
    validate_checkpoint,
    validate_ledger,
    validate_no_overlap_trust_evidence,
    validate_parent_evidence,
    validate_projected_generalized_phase3_evidence,
)


TARGET_ABS_DELTA_E = 2.0e-4
MODELED_ENERGY_CHANGE_MAX = -2.0e-6
ENERGY_GUARD_ABS_TOL = 1.0e-12
INITIAL_PRUNE_FS_RADIUS = 0.00390625
MIN_PRUNE_FS_RADIUS = 1.0e-8
QUERY_NEUTRAL_SCHEMA = "query_neutral_full_geometry_prune_round_v1"
PROPOSAL_SCHEMA = "query_neutral_full_geometry_prune_proposal_v1"
TRANSACTION_SCHEMA = "query_neutral_full_geometry_prune_transaction_v1"
FULL_RESPONSE_SCOPE = "full_active_plus_singleton_v1"
ROUTE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_query_neutral_fs_prune_v1"
)
DIGEST = "326ae05091b24fcb580d33f86f25add4c1252bcdd64316b82ae14c14c6bb3372"
PARENT_ROUTE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_v1"
)
PARENT_DIGEST = "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
QUERY_DELTA_FIELDS = (
    "prune_source_geometry_query_delta",
    "prune_energy_query_delta",
    "prune_refit_query_delta",
    "prune_gradient_query_delta",
    "prune_metric_query_delta",
    "prune_hessian_query_delta",
    "prune_endpoint_overlap_query_delta",
    "prune_explicit_query_delta",
)
S_ALG_COMPONENTS = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")


def _expected_query_neutral_optimizer_nfev(
    *,
    refit_occurrence_total: int,
    outer_occurrence_total: int,
    occurrence_scopes: Mapping[str, int],
) -> int:
    """Reconcile optimizer nfev without charging reporting-only energies."""

    initial_energy_count = int(occurrence_scopes.get("energy:initial_state", 0))
    final_verification_count = int(
        occurrence_scopes.get("final_state_verification", 0)
    )
    outer_refresh_count = int(occurrence_scopes.get("outer_state_refresh", 0))
    if (
        initial_energy_count != 1
        or final_verification_count != 1
        or (
            int(outer_occurrence_total)
            != initial_energy_count
            + outer_refresh_count
            + final_verification_count
        )
    ):
        raise ValueError("query-neutral outer-energy occurrence taxonomy drift")
    return int(refit_occurrence_total) + initial_energy_count


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} is not a mapping")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} is not a sequence")
    return value


def _first_hit_contract(
    result: Mapping[str, Any],
    *,
    safety_cap: int,
) -> dict[str, Any]:
    adapt = _mapping(result.get("adapt_vqe"), field="adapt_vqe")
    segment = _mapping(result.get("adapt_segment"), field="adapt_segment")
    history = list(_sequence(adapt.get("history"), field="ADAPT history"))
    final_round = int(segment.get("final_controller_round", -1))
    if (
        adapt.get("success") is not True
        or segment.get("stop_reason") != "benchmark_abs_delta_e_target"
        or final_round <= 0
        or final_round > int(safety_cap)
        or len(history) != final_round
    ):
        raise ValueError("run did not stop successfully at the first target hit")
    reference_raw = adapt.get("benchmark_target_reference_energy")
    if reference_raw is None:
        reference_raw = adapt.get("benchmark_stop_reference_energy")
    reference = float(reference_raw)
    target = float(adapt.get("adapt_benchmark_target_abs_delta_e"))
    if not math.isclose(target, TARGET_ABS_DELTA_E, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Paper-I target threshold drift")
    energies = [float(row["energy_after_opt"]) for row in history]
    errors = [abs(value - reference) for value in energies]
    if errors[-1] > target:
        raise ValueError("terminal state did not reach the target")
    if any(value <= target for value in errors[:-1]):
        raise ValueError("run continued after an earlier target hit")
    return {
        "target_abs_delta_e": target,
        "reference_energy": reference,
        "first_hit_round": final_round,
        "first_hit_error": errors[-1],
        "previous_round_error": errors[-2] if len(errors) > 1 else None,
    }


def _ledger_closure(
    *,
    result: Mapping[str, Any],
    ledger_sidecar: Mapping[str, Any],
    target_round: int,
) -> dict[str, Any]:
    adapt = _mapping(result.get("adapt_vqe"), field="adapt_vqe")
    result_accounting = _mapping(
        adapt.get("estimator_call_accounting"),
        field="result estimator accounting",
    )
    ledger = validate_ledger(ledger_sidecar, result_accounting)
    prefix = validate_active_prefix_estimator_receipts(
        adapt=adapt,
        ledger_summary=ledger,
        target_round=int(target_round),
    )
    all_work = _mapping(
        result_accounting.get("all_branch_search_work"),
        field="all-work accounting",
    )
    components = {key: int(all_work.get(key, -1)) for key in S_ALG_COMPONENTS}
    s_alg = int(all_work.get("S_alg", -1))
    if min(components.values()) < 0 or sum(components.values()) != s_alg:
        raise ValueError("strict S_alg component closure failed")
    return {
        "ledger": ledger,
        "active_prefix": prefix,
        "components": components,
        "S_alg": s_alg,
    }


def _raw_ledger_prune_accounting_audit(
    *,
    result: Mapping[str, Any],
    ledger_sidecar: Mapping[str, Any],
    target_round: int,
) -> dict[str, Any]:
    """Reconstruct S_alg and prove that pruning opened no query stream."""

    adapt = _mapping(result.get("adapt_vqe"), field="adapt_vqe")
    history = list(_sequence(adapt.get("history"), field="ADAPT history"))
    if len(history) != int(target_round):
        raise ValueError("raw-ledger audit history length drift")
    raw_ledger = _mapping(
        ledger_sidecar.get("ledger"),
        field="raw estimator ledger",
    )
    entries = list(
        _sequence(raw_ledger.get("entries"), field="raw estimator entries")
    )
    occurrences = list(
        _sequence(
            raw_ledger.get("occurrences"),
            field="raw estimator occurrences",
        )
    )

    entry_by_id: dict[str, Mapping[str, Any]] = {}
    unique_components = collections.Counter()
    for raw_entry in entries:
        entry = _mapping(raw_entry, field="raw estimator entry")
        primitive_id = str(entry.get("primitive_id", ""))
        component = str(entry.get("charged_component", ""))
        if (
            not primitive_id
            or primitive_id in entry_by_id
            or component not in S_ALG_COMPONENTS
        ):
            raise ValueError("raw estimator entry identity/component drift")
        entry_by_id[primitive_id] = entry
        unique_components[component] += 1

    accounting = _mapping(
        ledger_sidecar.get("accounting"),
        field="ledger accounting",
    )
    all_work = _mapping(
        accounting.get("all_branch_search_work"),
        field="all-work accounting",
    )
    expected_unique_components = {
        key: int(all_work.get(key, -1)) for key in S_ALG_COMPONENTS
    }
    if (
        len(entry_by_id) != int(all_work.get("S_alg", -1))
        or {
            key: int(unique_components.get(key, 0))
            for key in S_ALG_COMPONENTS
        }
        != expected_unique_components
    ):
        raise ValueError("raw unique entries do not reconstruct strict S_alg")

    occurrence_components = collections.Counter()
    occurrence_scopes = collections.Counter()
    occurrence_branches = collections.Counter()
    charged_by_id = collections.Counter()
    occurrence_ids: set[str] = set()
    normalized_occurrences: list[Mapping[str, Any]] = []
    for expected_sequence, raw_occurrence in enumerate(occurrences, start=1):
        occurrence = _mapping(
            raw_occurrence,
            field=f"raw estimator occurrence {expected_sequence}",
        )
        if int(occurrence.get("sequence", -1)) != expected_sequence:
            raise ValueError("raw estimator occurrence sequence is not contiguous")
        primitive_id = str(occurrence.get("primitive_id", ""))
        component = str(occurrence.get("component", ""))
        scope = str(occurrence.get("consumer_scope", ""))
        branch_id = occurrence.get("branch_id")
        if (
            primitive_id not in entry_by_id
            or component not in S_ALG_COMPONENTS
            or not scope
        ):
            raise ValueError("raw estimator occurrence identity drift")
        if branch_id is not None:
            raise ValueError("query-neutral route recorded branch estimator work")
        lowered_scope = scope.lower()
        if any(
            token in lowered_scope
            for token in ("prune", "rollback", "finite_angle", "energy_guard")
        ):
            raise ValueError(
                f"direct prune/rollback/guard estimator scope recorded: {scope}"
            )
        if component == "N_H_refit" and scope != "energy:depth_opt":
            raise ValueError(f"non-ordinary refit energy scope recorded: {scope}")
        if bool(occurrence.get("charged", False)):
            charged_by_id[primitive_id] += 1
            if str(entry_by_id[primitive_id].get("charged_component")) != component:
                raise ValueError("charged occurrence component drift")
        occurrence_components[component] += 1
        occurrence_scopes[scope] += 1
        occurrence_branches[
            "__unbranched__" if branch_id is None else str(branch_id)
        ] += 1
        occurrence_ids.add(primitive_id)
        normalized_occurrences.append(occurrence)
    if (
        occurrence_ids != set(entry_by_id)
        or set(charged_by_id) != set(entry_by_id)
        or any(count != 1 for count in charged_by_id.values())
    ):
        raise ValueError("raw occurrence charge/unique-entry bijection failed")

    executed = _mapping(
        accounting.get("executed_occurrence_accounting"),
        field="executed occurrence accounting",
    )
    all_execution = _mapping(
        executed.get("all_execution"),
        field="all execution occurrence summary",
    )
    if (
        int(all_execution.get("total_call_occurrences", -1))
        != len(normalized_occurrences)
        or {
            key: int(occurrence_components.get(key, 0))
            for key in S_ALG_COMPONENTS
        }
        != {key: int(all_execution.get(key, -1)) for key in S_ALG_COMPONENTS}
        or dict(sorted(occurrence_scopes.items()))
        != dict(all_execution.get("occurrence_count_by_consumer_scope", {}))
        or dict(occurrence_branches)
        != dict(all_execution.get("occurrence_count_by_consumer_branch", {}))
        or int(all_execution.get("unique_primitive_count", -1))
        != len(entry_by_id)
        or int(all_execution.get("same_identity_reuse_occurrence_count", -1))
        != len(normalized_occurrences) - len(entry_by_id)
    ):
        raise ValueError("raw occurrences do not reconstruct execution accounting")

    winning = _mapping(
        accounting.get("winning_lineage"),
        field="winning-lineage accounting",
    )
    discarded = _mapping(
        accounting.get("discarded_branch_only_by_unique_set_difference"),
        field="discarded-branch accounting",
    )
    if any(
        int(winning.get(key, -1)) != int(all_work.get(key, -2))
        for key in (*S_ALG_COMPONENTS, "S_alg")
    ):
        raise ValueError("query-neutral all-work/winning-lineage mismatch")
    if (
        int(discarded.get("S_alg", -1)) != 0
        or int(discarded.get("unique_primitive_count", -1)) != 0
        or list(discarded.get("primitive_ids", ()))
    ):
        raise ValueError("query-neutral route recorded discarded branch work")

    refit_occurrence_total = 0
    refit_metric_occurrence_total = 0
    for outer_iteration, row_raw in enumerate(history, start=1):
        row = _mapping(row_raw, field=f"history round {outer_iteration}")
        checkpoint = _mapping(
            row.get("active_prefix_checkpoint"),
            field=f"round {outer_iteration} active-prefix checkpoint",
        )
        receipt = _mapping(
            checkpoint.get("estimator_ledger_receipt"),
            field=f"round {outer_iteration} estimator-ledger receipt",
        )
        start = int(receipt.get("occurrence_sequence_start_exclusive", -1))
        end = int(receipt.get("occurrence_sequence_end_inclusive", -1))
        if start < 0 or end < start:
            raise ValueError("round estimator occurrence boundary drift")
        round_occurrences = normalized_occurrences[start:end]
        if len(round_occurrences) != end - start:
            raise ValueError("round estimator occurrence slice drift")
        refit_occurrences = [
            occurrence
            for occurrence in round_occurrences
            if occurrence.get("component") == "N_H_refit"
        ]
        expected_nfev = int(row.get("nfev_step_total_delta", -1))
        if (
            expected_nfev < 0
            or len(refit_occurrences) != expected_nfev
            or any(
                occurrence.get("consumer_scope") != "energy:depth_opt"
                for occurrence in refit_occurrences
            )
        ):
            raise ValueError(
                f"round {outer_iteration} ordinary refit/nfev reconciliation failed"
            )
        metric_occurrences = [
            occurrence
            for occurrence in round_occurrences
            if (
                occurrence.get("component") == "N_metric"
                and occurrence.get("consumer_scope")
                == "accepted_refit:native_depth_reopt"
            )
        ]
        refit_coordinate_count = int(
            row.get("phase3_accepted_refit_coordinate_count", -1)
        )
        expected_metric_occurrences = (
            refit_coordinate_count * (refit_coordinate_count + 1) // 2
        )
        if (
            refit_coordinate_count < 0
            or len(metric_occurrences) != expected_metric_occurrences
        ):
            raise ValueError(
                f"round {outer_iteration} accepted-refit metric charge drift"
            )
        refit_occurrence_total += len(refit_occurrences)
        refit_metric_occurrence_total += len(metric_occurrences)

    if refit_occurrence_total != int(
        occurrence_components.get("N_H_refit", 0)
    ):
        raise ValueError("roundwise refit occurrences do not close globally")
    expected_optimizer_nfev = _expected_query_neutral_optimizer_nfev(
        refit_occurrence_total=refit_occurrence_total,
        outer_occurrence_total=int(occurrence_components.get("N_H_outer", 0)),
        occurrence_scopes=occurrence_scopes,
    )
    if (
        int(executed.get("optimizer_and_guard_nfev_reported", -1))
        != expected_optimizer_nfev
    ):
        raise ValueError("optimizer/refit nfev total does not close")
    return {
        "schema": "paper_i_query_neutral_prune_raw_ledger_audit_v1",
        "status": "pass",
        "strict_unique_entry_count": len(entry_by_id),
        "strict_S_alg_components": expected_unique_components,
        "raw_occurrence_count": len(normalized_occurrences),
        "raw_occurrence_components": {
            key: int(occurrence_components.get(key, 0))
            for key in S_ALG_COMPONENTS
        },
        "roundwise_ordinary_refit_occurrences": refit_occurrence_total,
        "optimizer_and_initial_energy_nfev": expected_optimizer_nfev,
        "reporting_final_energy_excluded_from_optimizer_nfev": True,
        "roundwise_accepted_refit_metric_occurrences": (
            refit_metric_occurrence_total
        ),
        "prune_specific_estimator_scope_count": 0,
        "branch_estimator_occurrence_count": 0,
        "discarded_branch_S_alg": 0,
        "all_work_equals_winning_lineage": True,
    }


def validate_parent_first_hit_evidence(
    *,
    result: Mapping[str, Any],
    current: Mapping[str, Any],
    ledger_sidecar: Mapping[str, Any],
    safety_cap: int = 50,
) -> dict[str, Any]:
    """Validate an early-stopped parent anchor without changing its science."""

    hit = _first_hit_contract(result, safety_cap=safety_cap)
    target_round = int(hit["first_hit_round"])
    validation_view = copy.deepcopy(dict(result))
    validation_view["adapt_segment"]["max_new_admissions"] = target_round
    parent = validate_parent_evidence(
        result=validation_view,
        current=current,
        ledger_sidecar=ledger_sidecar,
        profile=PARENT_ROUTE,
        digest=PARENT_DIGEST,
        target_round=target_round,
        target_new_admissions=target_round,
        require_supported_rank=True,
    )
    projected = validate_projected_generalized_phase3_evidence(
        result=dict(result),
        target_round=target_round,
    )
    no_overlap = validate_no_overlap_trust_evidence(
        result=dict(result),
        target_round=target_round,
    )
    return {
        "schema": "paper_i_query_neutral_prune_parent_first_hit_evidence_v1",
        "status": "pass",
        "first_hit": hit,
        "parent": parent,
        "projected_phase3": projected,
        "no_overlap_trust": no_overlap,
    }


def validate_query_neutral_prune_evidence(
    *,
    result: Mapping[str, Any],
    current: Mapping[str, Any],
    ledger_sidecar: Mapping[str, Any],
    safety_cap: int = 50,
) -> dict[str, Any]:
    """Validate the zero-direct-query pruning route through first target hit."""

    del current  # The immutable result and sidecar carry the complete evidence.
    hit = _first_hit_contract(result, safety_cap=safety_cap)
    target_round = int(hit["first_hit_round"])
    adapt = _mapping(result.get("adapt_vqe"), field="adapt_vqe")
    settings = _mapping(result.get("settings"), field="settings")
    if (
        settings.get("sr_route_profile_resolved") != ROUTE
        or settings.get("sr_route_profile_contract_sha256") != DIGEST
        or adapt.get("sr_route_profile_resolved") != ROUTE
        or adapt.get("sr_route_profile_contract_sha256") != DIGEST
    ):
        raise ValueError("query-neutral prune route identity drift")
    expected_settings = {
        "phase1_prune_enabled": True,
        "phase1_prune_mode": "live",
        "phase1_prune_max_candidates": 1,
        "phase1_prune_local_window_size": 0,
        "phase1_prune_recovery_trust_radius": INITIAL_PRUNE_FS_RADIUS,
        "phase1_prune_schur_nomination_route": (
            "full_logical_fs_trust_delete_refit_v1"
        ),
        "phase1_prune_metric_schur_mu": 0.0,
        "phase1_prune_metric_schur_cost_weighting": "off",
        "phase1_prune_endpoint_overlap_policy": "off",
        "phase3_response_coordinate_scope": FULL_RESPONSE_SCOPE,
    }
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            raise ValueError(f"normalized query-neutral setting drift: {key}")
    invariants = _mapping(
        _mapping(
            adapt.get("sr_route_profile_contract"),
            field="route contract",
        ).get("semantic_invariants"),
        field="route semantic invariants",
    )
    expected_invariants = {
        "phase3_support_projection_active": True,
        "phase3_supported_metric_inverse_sqrt_active": False,
        "phase3_supported_whitening_active": False,
        "accepted_refit_scope": "full_ansatz_v1",
        "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
        "prune_source_geometry_query_delta": 0,
        "prune_derivative_query_delta": 0,
        "prune_metric_query_delta": 0,
        "prune_hessian_query_delta": 0,
        "prune_energy_query_delta": 0,
        "prune_endpoint_overlap_query_delta": 0,
        "prune_explicit_query_delta": 0,
    }
    for key, expected in expected_invariants.items():
        if invariants.get(key) != expected:
            raise ValueError(f"query-neutral semantic invariant drift: {key}")
    if adapt.get("adapt_beam_enabled") is not False:
        raise ValueError("historical admission beam became active")

    history = list(_sequence(adapt.get("history"), field="ADAPT history"))
    checkpoints = list(
        _sequence(
            adapt.get("active_prefix_checkpoints"),
            field="active-prefix checkpoints",
        )
    )
    if len(checkpoints) != target_round:
        raise ValueError("checkpoint count does not match first-hit round")
    prior_depth = 0
    prior_radius = INITIAL_PRUNE_FS_RADIUS
    nominated = accepted = rolled_back = 0
    query_delta_totals = {key: 0 for key in QUERY_DELTA_FIELDS}
    validated_checkpoints: list[dict[str, Any]] = []
    for outer_iteration, (row_raw, checkpoint_raw) in enumerate(
        zip(history, checkpoints),
        start=1,
    ):
        row = _mapping(row_raw, field=f"history round {outer_iteration}")
        checkpoint = _mapping(
            checkpoint_raw,
            field=f"checkpoint round {outer_iteration}",
        )
        if row.get("active_prefix_checkpoint") != checkpoint:
            raise ValueError(f"round {outer_iteration} checkpoint round-trip drift")
        validated = validate_checkpoint(
            checkpoint,
            outer_iteration=outer_iteration,
            checkpoint_kind="post_admission_prune",
        )
        validated_checkpoints.append(validated)
        prune = _mapping(
            row.get("post_admission_prune"),
            field=f"round {outer_iteration} prune receipt",
        )
        if (
            prune.get("schema") != QUERY_NEUTRAL_SCHEMA
            or prune.get("enabled") is not True
            or int(prune.get("ordinary_combined_refit_count", -1)) != 1
            or prune.get("second_refit_performed") is not False
            or prune.get("terminal_prune_active") is not False
            or prune.get("minimal_keep_prune_verification_beam") is not None
        ):
            raise ValueError(f"round {outer_iteration} prune execution drift")
        for key in QUERY_DELTA_FIELDS:
            value = int(prune.get(key, -1))
            if value != 0:
                raise ValueError(
                    f"round {outer_iteration} direct prune query charge: {key}"
                )
            query_delta_totals[key] += value
        proposal = _mapping(
            prune.get("query_neutral_proposal"),
            field=f"round {outer_iteration} proposal",
        )
        transaction = _mapping(
            prune.get("query_neutral_transaction"),
            field=f"round {outer_iteration} transaction",
        )
        is_nominated = proposal.get("nominated") is True
        is_accepted = transaction.get("accepted") is True
        is_rollback = transaction.get("rollback_classical") is True
        if transaction.get("schema") != TRANSACTION_SCHEMA:
            raise ValueError(f"round {outer_iteration} transaction schema drift")
        if (
            int(transaction.get("incremental_prune_quantum_query_charge", -1))
            != 0
            or int(transaction.get("rollback_quantum_query_charge", -1)) != 0
            or transaction.get("second_refit_performed") is not False
        ):
            raise ValueError(f"round {outer_iteration} transaction added work")
        state_before = _mapping(
            prune.get("phase1_prune_trust_state_before"),
            field=f"round {outer_iteration} prune trust before",
        )
        state_after = _mapping(
            prune.get("phase1_prune_trust_state_after"),
            field=f"round {outer_iteration} prune trust after",
        )
        radius_before = float(state_before.get("radius"))
        radius_after = float(state_after.get("radius"))
        if not math.isclose(
            radius_before,
            prior_radius,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"round {outer_iteration} prune-radius chain broke")
        if is_nominated:
            nominated += 1
            if (
                proposal.get("schema") != PROPOSAL_SCHEMA
                or proposal.get("source_geometry_reused") is not True
                or float(proposal.get("predicted_energy_change"))
                > MODELED_ENERGY_CHANGE_MAX
            ):
                raise ValueError(f"round {outer_iteration} nomination drift")
            receipt = _mapping(
                proposal.get("source_geometry_receipt"),
                field=f"round {outer_iteration} source geometry",
            )
            if (
                receipt.get("phase3_coordinate_scope") != FULL_RESPONSE_SCOPE
                or receipt.get("source_geometry_reused") is not True
                or receipt.get("duplicate_measurement_performed") is not False
                or int(receipt.get("incremental_quantum_query_charge", -1)) != 0
            ):
                raise ValueError(
                    f"round {outer_iteration} source geometry was remeasured"
                )
            if is_accepted:
                accepted += 1
                if is_rollback:
                    raise ValueError(
                        f"round {outer_iteration} accepted transaction rolled back"
                    )
                realized = float(transaction.get("realized_energy_change"))
                if realized > ENERGY_GUARD_ABS_TOL:
                    raise ValueError(
                        f"round {outer_iteration} accepted energy regression"
                    )
                expected_depth = prior_depth
                expected_radius = radius_before
            else:
                rolled_back += 1
                if not is_rollback:
                    raise ValueError(
                        f"round {outer_iteration} rejected prune did not roll back"
                    )
                expected_depth = prior_depth
                expected_radius = max(
                    MIN_PRUNE_FS_RADIUS,
                    0.5 * radius_before,
                )
        else:
            if is_accepted or is_rollback:
                raise ValueError(
                    f"round {outer_iteration} non-nomination transaction drift"
                )
            expected_depth = prior_depth + 1
            expected_radius = radius_before
        if int(validated["active_ansatz_depth"]) != expected_depth:
            raise ValueError(f"round {outer_iteration} active-depth transition drift")
        if not math.isclose(
            radius_after,
            expected_radius,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"round {outer_iteration} prune-radius update drift")
        prior_depth = expected_depth
        prior_radius = radius_after

    projected = validate_projected_generalized_phase3_evidence(
        result=dict(result),
        target_round=target_round,
    )
    no_overlap = validate_no_overlap_trust_evidence(
        result=dict(result),
        target_round=target_round,
    )
    ledger = _ledger_closure(
        result=result,
        ledger_sidecar=ledger_sidecar,
        target_round=target_round,
    )
    raw_ledger_audit = _raw_ledger_prune_accounting_audit(
        result=result,
        ledger_sidecar=ledger_sidecar,
        target_round=target_round,
    )
    return {
        "schema": "paper_i_query_neutral_full_geometry_prune_evidence_v1",
        "status": "pass",
        "first_hit": hit,
        "controller_rounds": target_round,
        "terminal_active_depth": prior_depth,
        "prune_nomination_count": nominated,
        "prune_accept_count": accepted,
        "prune_rollback_count": rolled_back,
        "direct_prune_query_delta_totals": query_delta_totals,
        "direct_prune_query_overhead": 0,
        "trajectory_total_S_alg": int(ledger["S_alg"]),
        "trajectory_S_alg_components": dict(ledger["components"]),
        "raw_ledger_prune_accounting_audit": raw_ledger_audit,
        "projected_phase3": projected,
        "no_overlap_trust": no_overlap,
        "checkpoint_count": len(validated_checkpoints),
    }


__all__ = [
    "DIGEST",
    "PARENT_DIGEST",
    "PARENT_ROUTE",
    "ROUTE",
    "validate_parent_first_hit_evidence",
    "validate_query_neutral_prune_evidence",
]
