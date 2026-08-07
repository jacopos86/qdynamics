#!/usr/bin/env python3
"""Fail-closed evidence checks for the SR-SNAKE symmetric-cost candidate.

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
PHASE1_ENERGY_MODEL = "first_order_fs_trust_v1"
PHASE2_CURVATURE_POLICY = "measured_required_fail_closed_v1"
PHASE2_CHEAP_CURVATURE_PROXY_POLICY = "off"
PHASE2_CURVATURE_RECEIPT_SCHEMA = (
    "sr_snake_phase2_directional_curvature_receipt_v1"
)
PHASE2_CURVATURE_PROVENANCE_SCHEMA = (
    "sr_snake_phase2_directional_curvature_provenance_v1"
)


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


def validate_phase2_curvature_receipt(
    receipt_raw: Any,
    *,
    outer_iteration: int,
) -> dict[str, Any]:
    receipt = _mapping(
        receipt_raw,
        field=f"history round {outer_iteration} Phase-II curvature receipt",
    )
    if receipt.get("schema") != PHASE2_CURVATURE_RECEIPT_SCHEMA:
        raise ValueError(
            f"history round {outer_iteration}: Phase-II curvature receipt schema drift"
        )
    if receipt.get("status") != "computed_finite":
        raise ValueError(
            f"history round {outer_iteration}: Phase-II curvature is unresolved"
        )
    h_raw = receipt.get("h_raw")
    if isinstance(h_raw, bool) or not isinstance(h_raw, (int, float)):
        raise ValueError(
            f"history round {outer_iteration}: Phase-II curvature is malformed"
        )
    curvature = float(h_raw)
    if not math.isfinite(curvature):
        raise ValueError(
            f"history round {outer_iteration}: Phase-II curvature is nonfinite"
        )
    expected_negative = curvature < 0.0
    if receipt.get("negative_curvature") is not expected_negative:
        raise ValueError(
            f"history round {outer_iteration}: negative-curvature receipt drift"
        )
    for field in (
        "state_fingerprint",
        "ordered_scaffold_fingerprint",
        "theta_fingerprint",
        "hamiltonian_fingerprint",
        "candidate_coordinate_fingerprint",
    ):
        if len(str(receipt.get(field) or "")) != 64:
            raise ValueError(
                f"history round {outer_iteration}: missing receipt identity {field}"
            )
    if int(receipt.get("candidate_position_id", -1)) < 0:
        raise ValueError(
            f"history round {outer_iteration}: invalid receipt candidate position"
        )
    if receipt.get("derivative_convention") != (
        "compiled_ansatz_exact_parameter_derivatives_v1"
    ):
        raise ValueError(
            f"history round {outer_iteration}: curvature derivative convention drift"
        )
    provenance = _mapping(
        receipt.get("measurement_provenance"),
        field=f"history round {outer_iteration} Phase-II curvature provenance",
    )
    if (
        provenance.get("schema") != PHASE2_CURVATURE_PROVENANCE_SCHEMA
        or provenance.get("required_primitives_resolved") is not True
        or provenance.get("source")
        != "compiled_directional_energy_hessian_v1"
        or provenance.get("candidate_derivative_source")
        != "compiled_ansatz_exact_parameter_derivatives_v1"
        or provenance.get("hamiltonian_action_source")
        != "existing_compiled_hamiltonian_actions_v1"
        or int(provenance.get("added_query_count", -1)) != 0
    ):
        raise ValueError(
            f"history round {outer_iteration}: Phase-II curvature provenance drift"
        )
    return {
        "h_raw": curvature,
        "negative_curvature": expected_negative,
        "candidate_coordinate_fingerprint": str(
            receipt["candidate_coordinate_fingerprint"]
        ),
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
        if payload.get("phase1_energy_model") != PHASE1_ENERGY_MODEL:
            raise ValueError(f"{name} Phase-I energy model drift")
        if payload.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY:
            raise ValueError(f"{name} Phase-II curvature policy drift")
        if payload.get("phase2_cheap_curvature_proxy_policy") != (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ):
            raise ValueError(f"{name} Phase-II cheap-proxy policy drift")
    phase12_telemetry = _mapping(
        adapt.get("phase12_energy_model_telemetry"),
        field="Phase-I/II runtime energy-model telemetry",
    )
    if (
        phase12_telemetry.get("schema")
        != "sr_snake_phase12_energy_model_runtime_v1"
        or phase12_telemetry.get("phase1_energy_model")
        != PHASE1_ENERGY_MODEL
        or phase12_telemetry.get("phase2_curvature_policy")
        != PHASE2_CURVATURE_POLICY
        or phase12_telemetry.get("phase2_cheap_curvature_proxy_policy")
        != PHASE2_CHEAP_CURVATURE_PROXY_POLICY
    ):
        raise ValueError("Phase-I/II runtime policy telemetry drift")
    phase2_full_candidate_occurrences = int(
        phase12_telemetry.get("phase2_full_candidate_occurrences", -1)
    )
    validated_phase2_receipt_occurrences = int(
        phase12_telemetry.get(
            "validated_phase2_curvature_receipt_occurrences", -2
        )
    )
    if (
        phase2_full_candidate_occurrences <= 0
        or validated_phase2_receipt_occurrences
        != phase2_full_candidate_occurrences
    ):
        raise ValueError("Phase-II curvature-receipt occurrence accounting is open")
    for field in (
        "phase1_lambda_f_proxy_occurrences",
        "phase2_lambda_f_proxy_occurrences",
        "phase2_missing_curvature_fallback_occurrences",
    ):
        if int(phase12_telemetry.get(field, -1)) != 0:
            raise ValueError(f"forbidden v4 Phase-I/II occurrence recorded: {field}")
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
        if history_row.get("phase1_energy_model") != PHASE1_ENERGY_MODEL:
            raise ValueError(
                f"history round {expected_round}: Phase-I energy model drift"
            )
        if history_row.get("phase1_lambda_f_proxy_applied") is not False:
            raise ValueError(
                f"history round {expected_round}: Phase-I lambda-F proxy applied"
            )
        if history_row.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY:
            raise ValueError(
                f"history round {expected_round}: Phase-II curvature policy drift"
            )
        if history_row.get("phase2_cheap_curvature_proxy_policy") != (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ):
            raise ValueError(
                f"history round {expected_round}: Phase-II cheap-proxy policy drift"
            )
        if history_row.get("phase2_lambda_f_proxy_applied") is not False:
            raise ValueError(
                f"history round {expected_round}: Phase-II lambda-F proxy applied"
            )
        if history_row.get("phase2_missing_curvature_fallback_used") is not False:
            raise ValueError(
                f"history round {expected_round}: missing-curvature fallback used"
            )
        validate_phase2_curvature_receipt(
            history_row.get("phase2_curvature_receipt"),
            outer_iteration=expected_round,
        )
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
        "phase1_energy_model": PHASE1_ENERGY_MODEL,
        "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ),
        "phase2_full_candidate_occurrences": phase2_full_candidate_occurrences,
        "validated_phase2_curvature_receipt_occurrences": (
            validated_phase2_receipt_occurrences
        ),
        "phase1_lambda_f_proxy_occurrences": 0,
        "phase2_lambda_f_proxy_occurrences": 0,
        "phase2_missing_curvature_fallback_occurrences": 0,
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


# This final definition intentionally supersedes the retained v4-prune helper
# above.  Keeping the helper available makes fetched historical v4 evidence
# inspectable, while this bundle's public validator enforces the no-prune route.
def validate_parent_evidence(
    *,
    result: Mapping[str, Any],
    current: Mapping[str, Any],
    ledger_sidecar: Mapping[str, Any],
    profile: str,
    digest: str,
    target_round: int,
    target_new_admissions: int,
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
        if payload.get("phase1_energy_model") != PHASE1_ENERGY_MODEL:
            raise ValueError(f"{name} Phase-I energy-model drift")
        if payload.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY:
            raise ValueError(f"{name} Phase-II curvature-policy drift")
        if payload.get("phase2_cheap_curvature_proxy_policy") != (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY
        ):
            raise ValueError(f"{name} Phase-II cheap-proxy drift")

    required_settings = {
        "adapt_finite_angle_fallback": False,
        "phase3_enable_rescue": False,
        "adapt_final_full_refit": "false",
        "adapt_full_refit_every": 0,
        "phase1_prune_enabled": False,
        "phase2_gram_novelty_policy": "fallback_only_v1",
        "phase3_gram_novelty_policy": "fallback_only_v1",
        "phase3_shadow_damping_policy": "off",
        "phase3_hardware_cost_normalization_mode": (
            "family_robust_symmetric_arctan_v1"
        ),
        "historical_singleton_coordinate_solve_scope": "phase3_only_v1",
        "historical_singleton_coordinate_solve_policy": (
            "supported_metric_whitened_eigh_v1"
        ),
        "historical_singleton_trust_region_update_policy": (
            "displacement_calibrated_unbounded_v2"
        ),
    }
    for key, expected in required_settings.items():
        actual = settings.get(key)
        if key == "adapt_final_full_refit" and actual is False:
            actual = "false"
        if actual != expected:
            raise ValueError(f"normalized candidate setting drift: {key}")
    if adapt.get("adapt_beam_enabled") is not False:
        raise ValueError("beam execution was not disabled")
    if adapt.get("finite_angle_fallback") is not False:
        raise ValueError("finite-angle fallback was enabled at runtime")
    if adapt.get("success") is not True:
        raise ValueError("result does not record successful ADAPT execution")

    continuation = _mapping(
        adapt.get("continuation"), field="result continuation telemetry"
    )
    if (
        str(continuation.get("oracle_gradient_scope")) != "off"
        or continuation.get("oracle_gradient_config") is not None
        or int(continuation.get("oracle_gradient_calls_total", -1)) != 0
        or int(continuation.get("oracle_gradient_raw_records_total", -1)) != 0
    ):
        raise ValueError("Phase-III oracle-gradient route was active")

    phase12 = _mapping(
        adapt.get("phase12_energy_model_telemetry"),
        field="Phase-I/II runtime energy-model telemetry",
    )
    if (
        phase12.get("schema") != "sr_snake_phase12_energy_model_runtime_v1"
        or phase12.get("phase1_energy_model") != PHASE1_ENERGY_MODEL
        or phase12.get("phase2_curvature_policy") != PHASE2_CURVATURE_POLICY
        or phase12.get("phase2_cheap_curvature_proxy_policy")
        != PHASE2_CHEAP_CURVATURE_PROXY_POLICY
    ):
        raise ValueError("Phase-I/II runtime policy telemetry drift")
    full_candidate_occurrences = int(
        phase12.get("phase2_full_candidate_occurrences", -1)
    )
    validated_receipts = int(
        phase12.get("validated_phase2_curvature_receipt_occurrences", -2)
    )
    if full_candidate_occurrences <= 0 or validated_receipts != full_candidate_occurrences:
        raise ValueError("Phase-II curvature-receipt occurrence accounting is open")
    for field in (
        "phase1_lambda_f_proxy_occurrences",
        "phase2_lambda_f_proxy_occurrences",
        "phase2_missing_curvature_fallback_occurrences",
    ):
        if int(phase12.get(field, -1)) != 0:
            raise ValueError(f"forbidden Phase-I/II occurrence recorded: {field}")

    fallback = _mapping(
        adapt.get("all_energy_models_infeasible_novelty_fallback_telemetry"),
        field="all-energy-models-infeasible fallback telemetry",
    )
    if (
        fallback.get("schema")
        != "sr_all_energy_models_infeasible_novelty_fallback_telemetry_v1"
        or fallback.get("enabled") is not True
        or fallback.get("policy")
        != "collective_span_novelty_over_symmetric_cost_v1"
        or fallback.get("ordinary_phase2_multiplier_active") is not False
        or fallback.get("ordinary_phase3_multiplier_active") is not False
        or fallback.get("phase2_curvature_failure_can_trigger") is not False
        or fallback.get("paper_reporting_scope") != "telemetry_gate_only_v1"
    ):
        raise ValueError("infeasible-model fallback policy/telemetry drift")
    fallback_rounds = [int(value) for value in _sequence(
        fallback.get("controller_rounds"), field="fallback controller rounds"
    )]
    if fallback_rounds != sorted(set(fallback_rounds)):
        raise ValueError("fallback controller rounds are unordered or duplicated")
    fallback_count = int(fallback.get("activation_count", -1))
    fallback_fired = fallback.get("fired") is True
    if fallback_count != len(fallback_rounds) or fallback_fired != (fallback_count > 0):
        raise ValueError("fallback activation count/fired flag does not close")
    if int(fallback.get("query_charge_total", -1)) < 0:
        raise ValueError("fallback query charge is invalid")

    final_refit = _mapping(
        adapt.get("final_full_refit"), field="final full-refit telemetry"
    )
    terminal_prune_refit = _mapping(
        adapt.get("post_prune_refit"), field="terminal post-prune refit"
    )
    if (
        final_refit.get("requested") is not False
        or final_refit.get("executed") is not False
        or terminal_prune_refit.get("executed") is not False
    ):
        raise ValueError("terminal-only refit/prune alteration executed")

    segment = _mapping(result.get("adapt_segment"), field="result adapt_segment")
    if int(segment.get("source_controller_round", -1)) != 0:
        raise ValueError("source controller round is not zero")
    if int(segment.get("final_controller_round", -1)) != target_round:
        raise ValueError("run did not finish the exact target controller round")
    if int(segment.get("new_admission_records", -1)) != target_new_admissions:
        raise ValueError("run did not commit the exact requested admissions")
    if int(segment.get("max_new_admissions", -1)) != target_new_admissions:
        raise ValueError("max-new-admissions drift")

    history = list(_sequence(adapt.get("history"), field="ADAPT history"))
    checkpoints = list(_sequence(
        adapt.get("active_prefix_checkpoints"), field="active-prefix checkpoints"
    ))
    if len(history) != target_new_admissions or len(checkpoints) != target_new_admissions:
        raise ValueError("history/checkpoint count does not equal the horizon")

    validated_checkpoints: list[dict[str, Any]] = []
    previous_active_depth = 0
    previous_radius: float | None = None
    supported_rank_each_round = True
    observed_fallback_rounds: list[int] = []
    for outer_iteration, (history_raw, checkpoint_raw) in enumerate(
        zip(history, checkpoints), start=1
    ):
        row = _mapping(history_raw, field=f"history round {outer_iteration}")
        checkpoint = _mapping(
            checkpoint_raw, field=f"checkpoint round {outer_iteration}"
        )
        if int(row.get("depth", -1)) != outer_iteration:
            raise ValueError(f"round {outer_iteration}: nonconsecutive depth")
        embedded = _mapping(
            row.get("active_prefix_checkpoint"),
            field=f"round {outer_iteration} embedded checkpoint",
        )
        if dict(embedded) != dict(checkpoint):
            raise ValueError(f"round {outer_iteration}: checkpoint round-trip drift")
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
            row.get("phase2_curvature_receipt"), outer_iteration=outer_iteration
        )

        active_count = int(row.get("phase3_active_logical_coordinate_count", -1))
        pre_support = int(row.get("phase3_response_pre_support_count", -1))
        indices = [int(value) for value in _sequence(
            row.get("phase3_response_coordinate_indices"),
            field=f"round {outer_iteration} Phase-III response indices",
        )]
        if (
            row.get("phase3_response_coordinate_scope") != FULL_RESPONSE_SCOPE
            or active_count != previous_active_depth
            or pre_support != active_count + 1
            or indices != list(range(pre_support))
        ):
            raise ValueError(f"round {outer_iteration}: full-response invariant failed")
        supported_raw = row.get("phase3_response_supported_rank")
        if require_supported_rank and supported_raw is None:
            raise ValueError(f"round {outer_iteration}: supported rank missing")
        if supported_raw is None:
            supported_rank_each_round = False
        else:
            supported_rank = int(supported_raw)
            if supported_rank < 1 or supported_rank > pre_support:
                raise ValueError(f"round {outer_iteration}: supported rank out of bounds")
        if int(row.get("phase3_accepted_refit_coordinate_count", -1)) != pre_support:
            raise ValueError(f"round {outer_iteration}: accepted-refit scope drift")
        accepted_refit = _mapping(
            row.get("accepted_refit"), field=f"round {outer_iteration} accepted refit"
        )
        invocation = _mapping(
            accepted_refit.get("accepted_refit_invocation"),
            field=f"round {outer_iteration} accepted-refit invocation",
        )
        config = _mapping(
            invocation.get("config"),
            field=f"round {outer_iteration} accepted-refit config",
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
            field=f"round {outer_iteration} adaptive-trust update",
        )
        if (
            trust.get("schema") != "route_a_trust_region_update_v1"
            or trust.get("policy") != "displacement_calibrated_unbounded_v2"
            or trust.get("full_coordinate_refit") is not True
        ):
            raise ValueError(f"round {outer_iteration}: adaptive-trust policy drift")
        radius_before = float(trust.get("radius_before", float("nan")))
        radius_after = float(trust.get("radius_after", float("nan")))
        if (
            not math.isfinite(radius_before)
            or not math.isfinite(radius_after)
            or radius_before < 0.0
            or radius_after < 0.0
        ):
            raise ValueError(f"round {outer_iteration}: invalid trust radius")
        if previous_radius is not None and radius_before != previous_radius:
            raise ValueError(f"round {outer_iteration}: trust-radius chain broke")
        previous_radius = radius_after
        if row.get("phase3_shadow_damping_receipt") is not None:
            raise ValueError(f"round {outer_iteration}: shadow damping receipt exists")

        prune = _mapping(
            row.get("post_admission_prune"),
            field=f"round {outer_iteration} inactive prune telemetry",
        )
        if (
            prune.get("enabled") is not False
            or prune.get("executed") is not False
            or int(prune.get("candidate_count", -1)) != 0
            or int(prune.get("accepted_count", -1)) != 0
            or list(_sequence(prune.get("decisions"), field="inactive prune decisions"))
        ):
            raise ValueError(f"round {outer_iteration}: prune route became active")

        row_fallback = row.get("all_energy_models_infeasible_novelty_fallback_fired") is True
        if row.get("all_energy_models_infeasible_novelty_fallback_enabled") is not True:
            raise ValueError(f"round {outer_iteration}: fallback safety disabled")
        if row_fallback:
            observed_fallback_rounds.append(outer_iteration)
        summary = validate_checkpoint(
            checkpoint,
            outer_iteration=outer_iteration,
            checkpoint_kind="post_admission_prune",
            leakage_tolerance=leakage_tolerance,
        )
        if summary["active_ansatz_depth"] != outer_iteration:
            raise ValueError(f"round {outer_iteration}: no-prune depth invariant failed")
        previous_active_depth = summary["active_ansatz_depth"]
        validated_checkpoints.append(summary)

    if observed_fallback_rounds != fallback_rounds:
        raise ValueError("per-round/cumulative fallback telemetry mismatch")
    final_depth = int(segment.get("final_depth", -1))
    if final_depth != target_round or int(adapt.get("ansatz_depth", -1)) != target_round:
        raise ValueError("final no-prune ansatz depth does not equal exact horizon")
    route_state = _mapping(
        adapt.get("route_a_trust_region_state"), field="final trust-region state"
    )
    if (
        route_state.get("schema") != "route_a_trust_region_state_v1"
        or int(route_state.get("update_count", -1)) != target_new_admissions
        or float(route_state.get("radius", float("nan"))) != previous_radius
    ):
        raise ValueError("adaptive trust did not update once per admission")

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
    if (
        int(current_checkpoint.get("depth", -1)) != target_round
        or int(current_adapt.get("history_count", -1)) != target_new_admissions
        or current_adapt.get("history_checkpoint_complete") is not True
        or int(current_adapt.get("ansatz_depth", -1)) != final_depth
        or current_checkpoint.get("sr_route_profile_contract_sha256") != digest
    ):
        raise ValueError("current checkpoint/result closure failed")
    current_history = list(_sequence(
        current_adapt.get("history"), field="current ADAPT history"
    ))
    if len(current_history) != len(history):
        raise ValueError("current/result history length mismatch")
    for outer_iteration, (current_row, result_row) in enumerate(
        zip(current_history, history), start=1
    ):
        if _mapping(current_row, field="current history row").get(
            "active_prefix_checkpoint"
        ) != _mapping(result_row, field="result history row").get(
            "active_prefix_checkpoint"
        ):
            raise ValueError(
                f"round {outer_iteration}: current/result checkpoint mismatch"
            )

    ledger = validate_ledger(
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
        "supported_rank_recorded_each_round": supported_rank_each_round,
        "adaptive_trust_updates": target_new_admissions,
        "phase2_full_candidate_occurrences": full_candidate_occurrences,
        "validated_phase2_curvature_receipt_occurrences": validated_receipts,
        "phase1_lambda_f_proxy_occurrences": 0,
        "phase2_lambda_f_proxy_occurrences": 0,
        "phase2_missing_curvature_fallback_occurrences": 0,
        "ordinary_phase2_novelty_multiplier_active": False,
        "ordinary_phase3_novelty_multiplier_active": False,
        "infeasible_model_fallback_enabled": True,
        "infeasible_model_fallback_fired": fallback_fired,
        "infeasible_model_fallback_activation_count": fallback_count,
        "infeasible_model_fallback_controller_rounds": fallback_rounds,
        "prune_rounds_executed": 0,
        "terminal_state_unchanged_from_last_ordinary_round": True,
        "ledger": ledger,
    }
