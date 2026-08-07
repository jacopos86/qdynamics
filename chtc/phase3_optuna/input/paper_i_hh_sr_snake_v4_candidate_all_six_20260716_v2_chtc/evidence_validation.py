#!/usr/bin/env python3
"""Shared fail-closed evidence checks for SR-SNAKE-v4 parent jobs.

This module is transferred beside ``run_job.py`` and is also used after fetch.
It validates scientific telemetry only; it does not reconstruct or alter a run.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


CHECKPOINT_SCHEMA = "paper_i_signed_active_prefix_checkpoint_v1"
LEDGER_SCHEMA = "paper_i_estimator_call_ledger_sidecar_v1"
ACCOUNTING_SCHEMA = "paper_i_current_s_alg_accounting_v1"
FULL_RESPONSE_SCOPE = "full_active_plus_singleton_v1"
DEFAULT_LEAKAGE_TOLERANCE = 1.0e-10


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an array")
    return value


def checkpoint_sha256(checkpoint: Mapping[str, Any]) -> str:
    payload = dict(checkpoint)
    payload.pop("checkpoint_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    outer_iteration: int,
    checkpoint_kind: str,
    leakage_tolerance: float = DEFAULT_LEAKAGE_TOLERANCE,
) -> dict[str, Any]:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"round {outer_iteration}: unexpected checkpoint schema")
    if int(checkpoint.get("outer_iteration", -1)) != outer_iteration:
        raise ValueError(f"round {outer_iteration}: checkpoint iteration mismatch")
    if str(checkpoint.get("checkpoint_kind")) != checkpoint_kind:
        raise ValueError(f"round {outer_iteration}: checkpoint kind mismatch")
    recorded_sha = str(checkpoint.get("checkpoint_sha256") or "")
    if len(recorded_sha) != 64 or recorded_sha != checkpoint_sha256(checkpoint):
        raise ValueError(f"round {outer_iteration}: checkpoint SHA-256 mismatch")

    depth = int(checkpoint.get("active_ansatz_depth", -1))
    labels = list(_sequence(
        checkpoint.get("ordered_active_operator_labels"),
        field=f"round {outer_iteration} ordered labels",
    ))
    operators = list(_sequence(
        checkpoint.get("ordered_active_operators"),
        field=f"round {outer_iteration} ordered operators",
    ))
    logical = list(_sequence(
        checkpoint.get("signed_unwrapped_logical_parameters"),
        field=f"round {outer_iteration} signed logical parameters",
    ))
    runtime = list(_sequence(
        checkpoint.get("signed_unwrapped_runtime_parameters"),
        field=f"round {outer_iteration} signed runtime parameters",
    ))
    parameterization = _mapping(
        checkpoint.get("parameterization"),
        field=f"round {outer_iteration} parameterization",
    )
    if depth < 0 or len(labels) != depth or len(operators) != depth:
        raise ValueError(f"round {outer_iteration}: ordered prefix/depth mismatch")
    if len(logical) != int(parameterization.get("logical_operator_count", -1)):
        raise ValueError(f"round {outer_iteration}: logical parameter count mismatch")
    if len(runtime) != int(parameterization.get("runtime_parameter_count", -1)):
        raise ValueError(f"round {outer_iteration}: runtime parameter count mismatch")
    if len(logical) != depth:
        raise ValueError(f"round {outer_iteration}: one-logical-coordinate-per-prefix mismatch")
    for position, operator in enumerate(operators):
        record = _mapping(operator, field=f"round {outer_iteration} operator {position}")
        if int(record.get("active_position", -1)) != position:
            raise ValueError(f"round {outer_iteration}: noncanonical operator position")
        if str(record.get("label")) != str(labels[position]):
            raise ValueError(f"round {outer_iteration}: label/operator ordering mismatch")
        terms = _sequence(
            record.get("serialized_terms_exyz_in_execution_order"),
            field=f"round {outer_iteration} operator {position} execution terms",
        )
        if not terms:
            raise ValueError(f"round {outer_iteration}: operator has no serialized terms")

    for contract_name in (
        "generator_sector_contract",
        "generator_pool_sector_contract",
        "active_generator_sector_contract",
        "state_sector_contract",
        "strict_replay",
    ):
        contract = _mapping(
            checkpoint.get(contract_name),
            field=f"round {outer_iteration} {contract_name}",
        )
        if contract.get("passed") is not True:
            raise ValueError(f"round {outer_iteration}: {contract_name} did not pass")
    sector_leakage = float(checkpoint.get("fixed_spin_sector_illegal_probability", float("inf")))
    padding_leakage = float(checkpoint.get("boson_illegal_codeword_probability", float("inf")))
    if sector_leakage > leakage_tolerance:
        raise ValueError(f"round {outer_iteration}: fixed-sector leakage exceeds tolerance")
    if padding_leakage > leakage_tolerance:
        raise ValueError(f"round {outer_iteration}: binary-padding leakage exceeds tolerance")
    return {
        "outer_iteration": outer_iteration,
        "checkpoint_sha256": recorded_sha,
        "active_ansatz_depth": depth,
        "logical_parameter_count": len(logical),
        "runtime_parameter_count": len(runtime),
        "fixed_sector_leakage": sector_leakage,
        "binary_padding_leakage": padding_leakage,
    }


def validate_ledger(
    ledger_sidecar: Mapping[str, Any],
    result_accounting: Mapping[str, Any],
) -> dict[str, Any]:
    if ledger_sidecar.get("schema") != LEDGER_SCHEMA:
        raise ValueError("unexpected estimator-ledger sidecar schema")
    if ledger_sidecar.get("adapt_success") is not True or ledger_sidecar.get("adapt_error") is not None:
        raise ValueError("estimator ledger records an unsuccessful ADAPT execution")
    accounting = _mapping(ledger_sidecar.get("accounting"), field="ledger accounting")
    for name, payload in (("ledger", accounting), ("result", result_accounting)):
        if payload.get("schema") != ACCOUNTING_SCHEMA:
            raise ValueError(f"{name} accounting schema mismatch")
        if payload.get("enabled") is not True or payload.get("complete") is not True:
            raise ValueError(f"{name} estimator accounting is incomplete")
        blockers = list(_sequence(payload.get("exact_blockers"), field=f"{name} exact blockers"))
        if blockers:
            raise ValueError(f"{name} estimator accounting has blockers: {blockers}")
        if not str(payload.get("status", "")).startswith("resolved_"):
            raise ValueError(f"{name} estimator accounting status is unresolved")
    occurrence = _mapping(
        accounting.get("executed_occurrence_accounting"),
        field="executed occurrence accounting",
    )
    if occurrence.get("allocation_is_disjoint_by_consumer") is not True:
        raise ValueError("executed occurrence allocation is not disjoint by consumer")
    winning = _mapping(accounting.get("winning_lineage"), field="winning-lineage accounting")
    all_work = _mapping(accounting.get("all_branch_search_work"), field="all-work accounting")
    for label, payload in (("winning", winning), ("all", all_work)):
        s_alg = int(payload.get("S_alg", -1))
        components = sum(int(payload.get(key, -1)) for key in (
            "N_H_outer", "N_H_refit", "N_grad", "N_metric"
        ))
        if s_alg < 0 or components != s_alg:
            raise ValueError(f"{label} S_alg component closure failed")
    for key in ("S_alg", "N_H_outer", "N_H_refit", "N_grad", "N_metric"):
        if int(winning.get(key, -1)) != int(result_accounting.get("winning_lineage", {}).get(key, -2)):
            raise ValueError(f"result/ledger winning-lineage mismatch: {key}")
        if int(all_work.get(key, -1)) != int(result_accounting.get("all_branch_search_work", {}).get(key, -2)):
            raise ValueError(f"result/ledger all-work mismatch: {key}")
    ledger = _mapping(ledger_sidecar.get("ledger"), field="raw estimator ledger")
    if not str(ledger.get("ledger_fingerprint", "")):
        raise ValueError("raw estimator ledger fingerprint missing")
    _sequence(ledger.get("entries"), field="raw estimator ledger entries")
    occurrences = list(_sequence(
        ledger.get("occurrences"), field="raw estimator ledger occurrences"
    ))
    finite_angle_occurrences = [
        row for row in occurrences
        if isinstance(row, Mapping)
        and str(row.get("consumer_scope", "")) == "finite_angle_objective_guard"
    ]
    if finite_angle_occurrences:
        raise ValueError("finite-angle objective guard recorded estimator work")
    return {
        "ledger_fingerprint": str(ledger["ledger_fingerprint"]),
        "winning_lineage_s_alg": int(winning["S_alg"]),
        "all_branch_s_alg": int(all_work["S_alg"]),
        "raw_entry_count": len(ledger["entries"]),
        "raw_occurrence_count": len(ledger["occurrences"]),
        "finite_angle_guard_occurrence_count": 0,
    }


def validate_live_prune_round(
    prune_raw: Any,
    *,
    outer_iteration: int,
) -> dict[str, Any]:
    """Validate the v4 live-only affine-deletion prune receipt.

    A round is allowed to leave pruning closed or to find no feasible affine
    deletion.  If it executes a delete-and-refit trial, however, every model,
    query-accounting, exact-energy-authority, and monotone trust-state receipt
    must be present and mutually consistent.
    """

    prune = _mapping(prune_raw, field=f"round {outer_iteration} live prune")
    if prune.get("enabled") is not True:
        raise ValueError(f"round {outer_iteration}: v4 live pruning is disabled")
    if str(prune.get("prune_mode")) != "live":
        raise ValueError(f"round {outer_iteration}: prune mode is not live")
    if prune.get("affine_deletion_fs_trust_route_active") is not True:
        raise ValueError(
            f"round {outer_iteration}: affine-deletion FS trust route is inactive"
        )

    nomination_raw = prune.get("schur_surrogate_nomination")
    model_count: int | None = None
    feasible_count: int | None = None
    if isinstance(nomination_raw, Mapping):
        nomination = _mapping(
            nomination_raw,
            field=f"round {outer_iteration} prune nomination",
        )
        count_raw = nomination.get("affine_deletion_model_count")
        feasible_raw = nomination.get("affine_deletion_feasible_count")
        rows_raw = nomination.get("affine_deletion_models")
        if count_raw is not None or feasible_raw is not None or rows_raw is not None:
            model_count = int(count_raw)
            feasible_count = int(feasible_raw)
            rows = list(_sequence(
                rows_raw,
                field=f"round {outer_iteration} affine-deletion models",
            ))
            if model_count < 0 or feasible_count < 0 or len(rows) != model_count:
                raise ValueError(
                    f"round {outer_iteration}: affine-deletion model counts do not close"
                )
            observed_feasible = 0
            for index, row_raw in enumerate(rows):
                row = _mapping(
                    row_raw,
                    field=f"round {outer_iteration} affine model {index}",
                )
                pre_support = int(row.get("pre_support_coordinate_count", -1))
                supported_rank = int(row.get("metric_supported_rank", -1))
                if pre_support != model_count:
                    raise ValueError(
                        f"round {outer_iteration}: affine model omitted a logical coordinate"
                    )
                if supported_rank < 1 or supported_rank > pre_support:
                    raise ValueError(
                        f"round {outer_iteration}: affine-model supported rank is invalid"
                    )
                if row.get(
                    "all_logical_coordinates_entered_before_support_reduction"
                ) is not True:
                    raise ValueError(
                        f"round {outer_iteration}: affine model was reduced before full entry"
                    )
                if row.get(
                    "supported_rank_projection_after_full_coordinate_entry"
                ) is not True:
                    raise ValueError(
                        f"round {outer_iteration}: affine support projection ordering drift"
                    )
                if int(row.get("classical_quantum_query_charge", -1)) != 0:
                    raise ValueError(
                        f"round {outer_iteration}: affine nomination charged a query"
                    )
                observed_feasible += int(bool(row.get("feasible", False)))
            if observed_feasible != feasible_count:
                raise ValueError(
                    f"round {outer_iteration}: feasible affine-model count mismatch"
                )

    candidate_count = int(prune.get("candidate_count", -1))
    probe_indices = list(_sequence(
        prune.get("probe_indices"),
        field=f"round {outer_iteration} prune probe indices",
    ))
    if candidate_count < 0 or candidate_count > 1 or len(probe_indices) > 1:
        raise ValueError(f"round {outer_iteration}: prune exact-trial cap drift")
    if len(probe_indices) != candidate_count:
        raise ValueError(f"round {outer_iteration}: prune candidate/probe count mismatch")

    executed = prune.get("executed") is True
    if not executed:
        if model_count and feasible_count == 0:
            if str(prune.get("permission_reason")) != "all_affine_deletion_models_infeasible":
                raise ValueError(
                    f"round {outer_iteration}: all-infeasible prune did not fail closed"
                )
            if candidate_count != 0 or probe_indices:
                raise ValueError(
                    f"round {outer_iteration}: all-infeasible prune attempted a trial"
                )
            if str(prune.get("frozen_probe_policy")) != (
                "disabled_no_feasible_affine_deletion_model_v1"
            ):
                raise ValueError(
                    f"round {outer_iteration}: all-infeasible fallback policy drift"
                )
            if int(prune.get("nfev_formal_frozen_prune_energy_probes", -1)) != 0:
                raise ValueError(
                    f"round {outer_iteration}: all-infeasible prune charged a probe"
                )
            no_feasible = _mapping(
                prune.get("phase1_prune_no_feasible_model"),
                field=f"round {outer_iteration} all-infeasible receipt",
            )
            if no_feasible.get("legacy_nomination_fallback_used") is not False:
                raise ValueError(
                    f"round {outer_iteration}: legacy prune fallback was used"
                )
            if int(no_feasible.get("exact_delete_refit_trial_count", -1)) != 0:
                raise ValueError(
                    f"round {outer_iteration}: all-infeasible receipt records a trial"
                )
            if no_feasible.get("trust_state_before") != no_feasible.get(
                "trust_state_after"
            ):
                raise ValueError(
                    f"round {outer_iteration}: all-infeasible trust state changed"
                )
            update = _mapping(
                no_feasible.get("trust_update"),
                field=f"round {outer_iteration} all-infeasible trust update",
            )
            if (
                int(update.get("classical_quantum_query_charge", -1)) != 0
                or int(update.get("update_count_after", -1))
                != int(update.get("update_count_before", -2))
            ):
                raise ValueError(
                    f"round {outer_iteration}: all-infeasible trust hold drift"
                )
        return {
            "executed": False,
            "model_count": model_count,
            "feasible_count": feasible_count,
        }

    if model_count is None or feasible_count is None or feasible_count < 1:
        raise ValueError(
            f"round {outer_iteration}: executed prune lacks a feasible model"
        )
    if candidate_count != 1 or len(probe_indices) != 1:
        raise ValueError(
            f"round {outer_iteration}: executed prune was not one nominee/one trial"
        )
    if str(prune.get("frozen_probe_policy")) != (
        "disabled_model_nomination_single_exact_trial_v1"
    ):
        raise ValueError(f"round {outer_iteration}: frozen prune probe policy drift")
    if int(prune.get("nfev_formal_frozen_prune_energy_probes", -1)) != 0:
        raise ValueError(f"round {outer_iteration}: frozen prune probe charged work")

    decisions = list(_sequence(
        prune.get("decisions"),
        field=f"round {outer_iteration} prune decisions",
    ))
    ladder_rows = list(_sequence(
        prune.get("recoverability_ladder_rows"),
        field=f"round {outer_iteration} prune ladder rows",
    ))
    if len(decisions) != 1 or len(ladder_rows) != 1:
        raise ValueError(
            f"round {outer_iteration}: measured prune decision count drift"
        )
    decision = _mapping(decisions[0], field=f"round {outer_iteration} prune decision")
    accepted = bool(decision.get("accepted", False))

    work = _mapping(
        prune.get("phase1_prune_exact_refit_work_accounting"),
        field=f"round {outer_iteration} prune work accounting",
    )
    nfev = int(work.get("nfev", -1))
    committed = int(work.get("committed_prune_nfev", -1))
    discarded = int(work.get("discarded_prune_nfev", -1))
    if nfev < 0 or committed < 0 or discarded < 0 or committed + discarded != nfev:
        raise ValueError(f"round {outer_iteration}: prune nfev buckets do not close")
    expected_class = "committed_prune" if accepted else "discarded_prune"
    if (
        str(work.get("classification")) != expected_class
        or work.get("included_in_total_nfev") is not True
        or work.get("included_in_total_estimator_ledger") is not True
    ):
        raise ValueError(f"round {outer_iteration}: prune work classification drift")
    reconciliation = _mapping(
        work.get("hamiltonian_occurrence_reconciliation"),
        field=f"round {outer_iteration} prune occurrence reconciliation",
    )
    if reconciliation.get("closed") is not True:
        raise ValueError(f"round {outer_iteration}: prune occurrence accounting is open")
    if int(reconciliation.get("recorded_N_H_refit_occurrences", -1)) != int(
        reconciliation.get("expected_N_H_refit_occurrences", -2)
    ):
        raise ValueError(f"round {outer_iteration}: prune occurrence count mismatch")

    receipt = _mapping(
        prune.get("phase1_prune_trial_receipt"),
        field=f"round {outer_iteration} prune trial receipt",
    )
    if (
        receipt.get("prediction_complete") is not True
        or receipt.get("realization_complete") is not True
        or receipt.get("energy_receipt_complete") is not True
        or receipt.get("measured_delete_refit_is_acceptance_authority") is not True
        or receipt.get("endpoint_overlap_measured") is not False
        or int(receipt.get("added_endpoint_overlap_query_count", -1)) != 0
    ):
        raise ValueError(f"round {outer_iteration}: prune trial authority/query drift")
    trial_id = str(receipt.get("trial_id") or "")
    if (
        not trial_id
        or receipt.get("prediction_trial_id") != trial_id
        or receipt.get("realization_trial_id") != trial_id
    ):
        raise ValueError(f"round {outer_iteration}: prune trial identity mismatch")

    before = _mapping(
        prune.get("phase1_prune_trust_state_before"),
        field=f"round {outer_iteration} prune trust state before",
    )
    after = _mapping(
        prune.get("phase1_prune_trust_state_after"),
        field=f"round {outer_iteration} prune trust state after",
    )
    update = _mapping(
        prune.get("phase1_prune_trust_update"),
        field=f"round {outer_iteration} prune trust update",
    )
    radius_before = float(before.get("radius", float("nan")))
    radius_after = float(after.get("radius", float("nan")))
    mu_before = float(before.get("metric_damping", float("nan")))
    mu_after = float(after.get("metric_damping", float("nan")))
    count_before = int(before.get("update_count", -1))
    count_after = int(after.get("update_count", -1))
    if not (radius_after <= radius_before and mu_after >= mu_before):
        raise ValueError(f"round {outer_iteration}: prune rho/mu monotonicity failed")
    if count_after != count_before + 1:
        raise ValueError(f"round {outer_iteration}: prune trust update count drift")
    if (
        float(update.get("radius_before", float("nan"))) != radius_before
        or float(update.get("radius_after", float("nan"))) != radius_after
        or float(update.get("metric_damping_before", float("nan"))) != mu_before
        or float(update.get("metric_damping_after", float("nan"))) != mu_after
        or int(update.get("update_count_before", -1)) != count_before
        or int(update.get("update_count_after", -1)) != count_after
        or update.get("radius_never_increased") is not True
        or update.get("metric_damping_never_decreased") is not True
        or int(update.get("classical_quantum_query_charge", -1)) != 0
    ):
        raise ValueError(f"round {outer_iteration}: prune trust update receipt drift")
    return {
        "executed": True,
        "accepted": accepted,
        "model_count": model_count,
        "feasible_count": feasible_count,
        "nfev": nfev,
        "radius_before": radius_before,
        "radius_after": radius_after,
        "metric_damping_before": mu_before,
        "metric_damping_after": mu_after,
    }


def validate_parent_evidence(
    *,
    result: Mapping[str, Any],
    current: Mapping[str, Any],
    ledger_sidecar: Mapping[str, Any],
    profile: str,
    digest: str,
    target_round: int = 30,
    target_new_admissions: int = 30,
    require_supported_rank: bool = True,
    leakage_tolerance: float = DEFAULT_LEAKAGE_TOLERANCE,
) -> dict[str, Any]:
    adapt = _mapping(result.get("adapt_vqe"), field="result adapt_vqe")
    settings = _mapping(result.get("settings"), field="result settings")
    for name, payload in (("settings", settings), ("adapt_vqe", adapt)):
        if payload.get("sr_route_profile_resolved") != profile:
            raise ValueError(f"{name} route profile drift")
        if payload.get("sr_route_profile_contract_sha256") != digest:
            raise ValueError(f"{name} route digest drift")
    if settings.get("adapt_finite_angle_fallback") is not False:
        raise ValueError("v4 finite-angle fallback setting drift")
    if adapt.get("finite_angle_fallback") is not False:
        raise ValueError("v4 finite-angle fallback runtime drift")
    if settings.get("phase3_enable_rescue") is not False:
        raise ValueError("v4 Phase-III rescue setting drift")
    continuation = _mapping(
        adapt.get("continuation"), field="result continuation telemetry"
    )
    if str(continuation.get("oracle_gradient_scope")) != "off":
        raise ValueError("v4 Phase-III oracle-gradient scope is not off")
    if continuation.get("oracle_gradient_config") is not None:
        raise ValueError("v4 Phase-III oracle-gradient config is present")
    if int(continuation.get("oracle_gradient_calls_total", -1)) != 0:
        raise ValueError("v4 Phase-III oracle-gradient calls were recorded")
    if int(continuation.get("oracle_gradient_raw_records_total", -1)) != 0:
        raise ValueError("v4 Phase-III oracle-gradient raw records were recorded")
    if adapt.get("success") is not True:
        raise ValueError("result does not record successful ADAPT execution")
    if settings.get("adapt_final_full_refit") not in {False, "false"}:
        raise ValueError("v4 terminal full-refit setting drift")
    if settings.get("phase1_prune_enabled") is not True:
        raise ValueError("v4 live pruning is disabled in normalized settings")
    if str(settings.get("phase1_prune_mode")) != "live":
        raise ValueError("v4 prune mode is not live")
    if int(settings.get("phase1_prune_max_candidates", -1)) != 1:
        raise ValueError("v4 prune nominee cap drift")
    if str(settings.get("phase1_prune_endpoint_overlap_policy")) != "off":
        raise ValueError("v4 prune added an endpoint-overlap query")
    final_refit = _mapping(adapt.get("final_full_refit"), field="final full-refit telemetry")
    if final_refit.get("requested") is not False or final_refit.get("executed") is not False:
        raise ValueError("unapproved terminal full refit executed")
    terminal_post_prune_refit = _mapping(
        adapt.get("post_prune_refit"), field="terminal post-prune refit telemetry"
    )
    if terminal_post_prune_refit.get("executed") is not False:
        raise ValueError("unapproved terminal prune/refit executed")

    segment = _mapping(result.get("adapt_segment"), field="result adapt_segment")
    if int(segment.get("source_controller_round", -1)) != 0:
        raise ValueError("parent source controller round is not zero")
    if int(segment.get("final_controller_round", -1)) != target_round:
        raise ValueError("parent did not finish the target controller round")
    if int(segment.get("new_admission_records", -1)) != target_new_admissions:
        raise ValueError("parent did not commit the requested number of admissions")
    if int(segment.get("max_new_admissions", -1)) != target_new_admissions:
        raise ValueError("parent max-new-admissions drift")

    history = list(_sequence(adapt.get("history"), field="ADAPT history"))
    checkpoints = list(_sequence(
        adapt.get("active_prefix_checkpoints"), field="active-prefix checkpoints"
    ))
    if len(history) != target_new_admissions or len(checkpoints) != target_new_admissions:
        raise ValueError("history/checkpoint count does not equal the admission horizon")
    validated_checkpoints: list[dict[str, Any]] = []
    previous_active_depth = 0
    previous_route_radius: float | None = None
    supported_rank_recorded_each_round = True
    prune_round_summaries: list[dict[str, Any]] = []
    for expected_round, (history_raw, checkpoint_raw) in enumerate(
        zip(history, checkpoints), start=1
    ):
        history_row = _mapping(history_raw, field=f"history round {expected_round}")
        checkpoint = _mapping(checkpoint_raw, field=f"checkpoint round {expected_round}")
        if int(history_row.get("depth", -1)) != expected_round:
            raise ValueError(f"history round {expected_round}: nonconsecutive controller depth")
        embedded = _mapping(
            history_row.get("active_prefix_checkpoint"),
            field=f"history round {expected_round} embedded checkpoint",
        )
        if dict(embedded) != dict(checkpoint):
            raise ValueError(f"history round {expected_round}: checkpoint serialization mismatch")
        response_scope = str(history_row.get("phase3_response_coordinate_scope"))
        if response_scope != FULL_RESPONSE_SCOPE:
            raise ValueError(f"history round {expected_round}: Phase-III scope drift")
        active_count = int(history_row.get("phase3_active_logical_coordinate_count", -1))
        pre_support = int(history_row.get("phase3_response_pre_support_count", -1))
        response_indices = [int(value) for value in _sequence(
            history_row.get("phase3_response_coordinate_indices"),
            field=f"history round {expected_round} Phase-III indices",
        )]
        if active_count != previous_active_depth or pre_support != active_count + 1:
            raise ValueError(f"history round {expected_round}: full-response count invariant failed")
        if response_indices != list(range(pre_support)):
            raise ValueError(f"history round {expected_round}: full-response index ordering failed")
        supported_rank_raw = history_row.get("phase3_response_supported_rank")
        if require_supported_rank and supported_rank_raw is None:
            raise ValueError(f"history round {expected_round}: supported rank missing")
        if supported_rank_raw is None:
            supported_rank_recorded_each_round = False
        if supported_rank_raw is not None:
            supported_rank = int(supported_rank_raw)
            if supported_rank < 1 or supported_rank > pre_support:
                raise ValueError(f"history round {expected_round}: supported rank out of bounds")
        accepted_refit_count = int(
            history_row.get("phase3_accepted_refit_coordinate_count", -1)
        )
        if accepted_refit_count != pre_support:
            raise ValueError(f"history round {expected_round}: full accepted-refit count mismatch")

        trust_update = _mapping(
            history_row.get("route_a_trust_region_update"),
            field=f"history round {expected_round} adaptive-trust update",
        )
        if (
            trust_update.get("schema") != "route_a_trust_region_update_v1"
            or trust_update.get("policy") != "displacement_calibrated_unbounded_v2"
            or trust_update.get("full_coordinate_refit") is not True
        ):
            raise ValueError(
                f"history round {expected_round}: adaptive-trust update policy drift"
            )
        radius_before = float(trust_update.get("radius_before", float("nan")))
        radius_after = float(trust_update.get("radius_after", float("nan")))
        if (
            not math.isfinite(radius_before)
            or not math.isfinite(radius_after)
            or radius_before < 0.0
            or radius_after < 0.0
        ):
            raise ValueError(
                f"history round {expected_round}: adaptive-trust radius is invalid"
            )
        if previous_route_radius is not None and radius_before != previous_route_radius:
            raise ValueError(
                f"history round {expected_round}: adaptive-trust radius chain broke"
            )
        previous_route_radius = radius_after

        shadow = _mapping(
            history_row.get("phase3_shadow_damping_receipt"),
            field=f"history round {expected_round} shadow damping receipt",
        )
        if (
            shadow.get("schema") != "route_a_phase3_shadow_damping_receipt_v1"
            or shadow.get("policy") != "zero_query_mapped_seed_shadow_v1"
            or shadow.get("requested_policy") != "mapped_seed_zero_query_v1"
            or shadow.get("uses_existing_mapped_seed_evidence_only") is not True
            or shadow.get("applied_to_phase3_model") is not False
            or shadow.get("damping_applied") is not False
            or float(shadow.get("applied_mu", float("nan"))) != 0.0
            or shadow.get("trust_radius_mutated") is not False
            or shadow.get("hamiltonian_probe_performed") is not False
            or shadow.get("objective_evaluation_performed") is not False
            or int(shadow.get("added_query_count", -1)) != 0
        ):
            raise ValueError(
                f"history round {expected_round}: shadow damping was not diagnostic-only"
            )
        if str(shadow.get("status")) == "unresolved":
            if any(
                shadow.get(key) is not None
                for key in (
                    "mapped_seed_predicted_gain",
                    "mapped_seed_exact_gain",
                    "modeled_displacement_squared",
                    "positive_gain_overprediction",
                )
            ):
                raise ValueError(
                    f"history round {expected_round}: unresolved shadow receipt invented evidence"
                )
        prune_round_summaries.append(validate_live_prune_round(
            history_row.get("post_admission_prune"),
            outer_iteration=expected_round,
        ))
        checkpoint_summary = validate_checkpoint(
            checkpoint,
            outer_iteration=expected_round,
            checkpoint_kind="post_admission_prune",
            leakage_tolerance=leakage_tolerance,
        )
        if checkpoint_summary["active_ansatz_depth"] > pre_support:
            raise ValueError(f"history round {expected_round}: prune increased active depth")
        previous_active_depth = checkpoint_summary["active_ansatz_depth"]
        validated_checkpoints.append(checkpoint_summary)

    final_depth = int(segment.get("final_depth", -1))
    if final_depth != previous_active_depth or int(adapt.get("ansatz_depth", -1)) != final_depth:
        raise ValueError("segment/result/checkpoint final active-depth mismatch")
    route_state = _mapping(
        adapt.get("route_a_trust_region_state"), field="final adaptive-trust state"
    )
    if (
        route_state.get("schema") != "route_a_trust_region_state_v1"
        or int(route_state.get("update_count", -1)) != target_new_admissions
        or float(route_state.get("radius", float("nan"))) != previous_route_radius
    ):
        raise ValueError("adaptive trust did not update exactly once per admission")
    terminal = _mapping(
        adapt.get("terminal_active_prefix_checkpoint"),
        field="terminal active-prefix checkpoint",
    )
    validate_checkpoint(
        terminal,
        outer_iteration=target_round,
        checkpoint_kind="terminal_post_final_refit_and_prune",
        leakage_tolerance=leakage_tolerance,
    )
    last = _mapping(checkpoints[-1], field="last ordinary checkpoint")
    for field in (
        "active_ansatz_depth",
        "ordered_active_operator_labels",
        "ordered_active_operators",
        "signed_unwrapped_logical_parameters",
        "signed_unwrapped_runtime_parameters",
        "parameterization",
        "projective_state_fingerprint",
    ):
        if terminal.get(field) != last.get(field):
            raise ValueError(f"terminal-only state alteration detected: {field}")

    current_adapt = _mapping(current.get("adapt_vqe"), field="current adapt_vqe")
    current_checkpoint = _mapping(current.get("checkpoint"), field="current checkpoint")
    if int(current_checkpoint.get("depth", -1)) != target_round:
        raise ValueError("current JSON is not the target-round checkpoint")
    if int(current_adapt.get("history_count", -1)) != target_new_admissions:
        raise ValueError("current JSON history count mismatch")
    if current_adapt.get("history_checkpoint_complete") is not True:
        raise ValueError("current JSON history checkpoint is incomplete")
    current_history = list(_sequence(
        current_adapt.get("history"), field="current ADAPT history"
    ))
    if len(current_history) != len(history):
        raise ValueError("current/result history length mismatch")
    for expected_round, (current_raw, result_raw) in enumerate(
        zip(current_history, history), start=1
    ):
        current_row = _mapping(
            current_raw, field=f"current history round {expected_round}"
        )
        result_row = _mapping(
            result_raw, field=f"result history round {expected_round}"
        )
        if int(current_row.get("depth", -1)) != expected_round:
            raise ValueError(
                f"current history round {expected_round}: nonconsecutive depth"
            )
        if current_row.get("active_prefix_checkpoint") != result_row.get(
            "active_prefix_checkpoint"
        ):
            raise ValueError(
                f"current/result round {expected_round}: ordered checkpoint round-trip mismatch"
            )
    if int(current_adapt.get("ansatz_depth", -1)) != final_depth:
        raise ValueError("current/result active-depth mismatch")
    if current_checkpoint.get("sr_route_profile_contract_sha256") != digest:
        raise ValueError("current checkpoint route digest drift")

    ledger_summary = validate_ledger(
        ledger_sidecar,
        _mapping(adapt.get("estimator_call_accounting"), field="result accounting"),
    )
    return {
        "controller_rounds": target_round,
        "new_admissions": target_new_admissions,
        "final_active_depth": final_depth,
        "terminal_checkpoint_sha256": str(last["checkpoint_sha256"]),
        "max_fixed_sector_leakage": max(
            row["fixed_sector_leakage"] for row in validated_checkpoints
        ),
        "max_binary_padding_leakage": max(
            row["binary_padding_leakage"] for row in validated_checkpoints
        ),
        "phase3_response_scope": FULL_RESPONSE_SCOPE,
        "supported_rank_recorded_each_round": supported_rank_recorded_each_round,
        "adaptive_trust_updates": target_new_admissions,
        "shadow_damping_diagnostic_only_each_round": True,
        "terminal_state_unchanged_from_last_ordinary_round": True,
        "live_prune_rounds_executed": sum(
            1 for row in prune_round_summaries if row["executed"]
        ),
        "live_prune_rounds_accepted": sum(
            1 for row in prune_round_summaries
            if row["executed"] and row.get("accepted") is True
        ),
        "ledger": ledger_summary,
    }
