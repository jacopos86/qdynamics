#!/usr/bin/env python3
"""Execute and validate the source-locked Append-S weak--strong continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BUNDLE_ID = (
    "paper_i_hh_append_projected_singleton_weak_strong_warm_continuation_"
    "k50_to_firsthit_or_k80_20260721_v1_chtc"
)
STOP_POLICY = "append_warm_start_first_hit_or_max_iterations_v1"
CAP_POLICY = "accept_finite_nonincreasing_v1"
SOURCE_ITERATIONS = 50
SOURCE_DEPTH = 50
SOURCE_S_ALG = 1_276_060.0
TARGET_ERROR = 2.0e-4
MAX_TOTAL_ITERATIONS = 80


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: {actual!r}!={expected!r}")


def _require_close(actual: Any, expected: Any, label: str, tol: float = 1.0e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=float(tol)):
        raise RuntimeError(f"{label} mismatch: {actual!r}!={expected!r}")


def _validate_input(job: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    _require_equal(job.get("bundle_id"), BUNDLE_ID, "bundle id")
    _require_equal(job.get("algorithm_id"), "static_full_meta_append_adapt_vqe", "algorithm")
    _require_equal(job.get("seed"), 7, "seed")
    _require_equal(job["regime"]["slug"], "weak_strong", "regime")
    _require_equal(job["physics"]["n_ph_work"], 7, "working cutoff")
    _require_equal(job["physics"]["n_ph_ref"], 7, "reference cutoff")
    _require_equal(job["physics"]["same_cutoff_reference"], True, "same-cutoff reference")
    _require_equal(job["candidate_pool"]["shared_pauli_pool_mode"], "projected_singleton_children_only_v1", "pool mode")
    _require_equal(job["candidate_pool"]["shared_pauli_pool_symmetry_policy"], "hard_guard", "pool guard")
    _require_equal(job["candidate_pool"]["shared_pauli_pool_max_subset_size"], 1, "pool subset size")
    _require_equal(job["optimizer"]["kind"], "powell", "optimizer")
    _require_equal(job["optimizer"]["maxiter"], 200, "optimizer maxiter")
    _require_equal(job["optimizer"]["powell_maxiter_cap_policy"], CAP_POLICY, "Powell cap policy")
    controller = job["controller"]
    _require_equal(controller["max_adapt_iterations"], MAX_TOTAL_ITERATIONS, "total horizon")
    _require_close(controller["energy_stop_target"], TARGET_ERROR, "target error", 0.0)
    _require_equal(controller["first_hit_thresholds"], [TARGET_ERROR], "first-hit thresholds")
    _require_equal(controller["gradient_threshold"], 0.0, "gradient threshold")
    _require_equal(controller["allow_repeats"], True, "repeat policy")
    _require_equal(controller["stop_policy"], STOP_POLICY, "stop policy")
    _require_equal(controller["initial_selected_operator_count"], SOURCE_DEPTH, "source depth")
    _require_equal(controller["initial_history_count"], SOURCE_ITERATIONS, "source history")
    _require_equal(job["continuation"]["source_S_alg"], SOURCE_S_ALG, "source S_alg")
    _require_equal(state.get("schema"), "paper_i_append_projected_singleton_warm_start_state_v1", "state schema")
    labels = state.get("selected_operator_labels")
    batches = state.get("selected_operator_batches")
    theta = state.get("theta")
    history = state.get("adapt_history")
    if not all(isinstance(value, list) for value in (labels, batches, theta, history)):
        raise RuntimeError("warm state arrays are malformed")
    if not (len(labels) == len(batches) == len(theta) == len(history) == SOURCE_DEPTH):
        raise RuntimeError("warm-state labels/batches/theta/history do not align at 50")
    if [batch[0] for batch in batches if isinstance(batch, list) and len(batch) == 1] != labels:
        raise RuntimeError("warm-state singleton batches do not flatten to labels")
    for index, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"warm history item {index} is malformed")
        _require_equal(row.get("iteration"), index, "warm history iteration")
        _require_equal(row.get("depth_after"), index + 1, "warm history depth")
        _require_equal(row.get("appended_operator_labels"), [labels[index]], "warm history operator")
        _require_equal(row.get("optimizer_success"), True, "warm prefix optimizer")
    state_core = {
        "selected_operator_labels": labels,
        "selected_operator_batches": batches,
        "theta": theta,
        "adapt_history": history,
    }
    _require_equal(state.get("state_core_sha256"), _canonical_sha256(state_core), "state core hash")
    prefix = state.get("prefix_identity")
    if not isinstance(prefix, Mapping):
        raise RuntimeError("prefix identity is absent")
    prefix_without_self = dict(prefix)
    expected_prefix_sha = prefix_without_self.pop("prefix_identity_sha256", None)
    _require_equal(expected_prefix_sha, _canonical_sha256(prefix_without_self), "prefix identity hash")
    _require_equal(prefix.get("source_iterations"), SOURCE_ITERATIONS, "prefix iterations")
    _require_equal(prefix.get("source_depth"), SOURCE_DEPTH, "prefix depth")
    _require_equal(prefix.get("source_S_alg"), SOURCE_S_ALG, "prefix S_alg")
    _require_equal(prefix.get("ordered_operator_labels_sha256"), _canonical_sha256(labels), "label hash")
    _require_equal(prefix.get("selected_operator_batches_sha256"), _canonical_sha256(batches), "batch hash")
    _require_equal(prefix.get("theta_sha256"), _canonical_sha256(theta), "theta hash")
    _require_equal(prefix.get("compact_history_sha256"), _canonical_sha256(history), "history hash")
    _require_equal(job["continuation"]["prefix_identity_sha256"], expected_prefix_sha, "job prefix hash")
    return {
        "schema": "paper_i_append_warm_continuation_input_preflight_v1",
        "validated_utc": _utc_now(),
        "status": "pass",
        "bundle_id": BUNDLE_ID,
        "job_id": job["job_id"],
        "source_iterations": SOURCE_ITERATIONS,
        "source_depth": SOURCE_DEPTH,
        "source_S_alg": SOURCE_S_ALG,
        "source_prefix_identity_sha256": expected_prefix_sha,
        "warm_start_state_core_sha256": state["state_core_sha256"],
        "target_error": TARGET_ERROR,
        "max_total_iterations": MAX_TOTAL_ITERATIONS,
    }


def _terminal_receipt_delta(row: Mapping[str, Any]) -> float:
    receipt = row.get("adapt_terminal_estimator_call_receipt")
    if not isinstance(receipt, Mapping):
        return 0.0
    return float(receipt.get("S_alg_delta", 0.0))


def _validate_result(
    job: Mapping[str, Any], state: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_equal(payload.get("status"), "completed", "top-level status")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise RuntimeError("generic result must contain exactly one row")
    row = dict(rows[0])
    _require_equal(row.get("status"), "ok", "row status")
    _require_equal(row.get("algorithm_id"), job["algorithm_id"], "algorithm id")
    _require_equal(row.get("case_id"), job["regime"]["case_id"], "case id")
    _require_equal(row.get("seed"), job["seed"], "seed")
    _require_equal(row.get("adapt_max_iterations"), MAX_TOTAL_ITERATIONS, "ADAPT horizon")
    _require_equal(row.get("generic_adapt_stop_policy"), STOP_POLICY, "stop policy")
    _require_equal(row.get("adapt_continuation_mode"), "warm_start_selected_theta_v1", "continuation mode")
    _require_equal(row.get("adapt_warm_start_source_depth"), SOURCE_DEPTH, "warm source depth")
    _require_equal(row.get("adapt_warm_start_source_iterations"), SOURCE_ITERATIONS, "warm source iterations")
    _require_equal(row.get("adapt_selection_with_replacement"), True, "replacement policy")
    _require_equal(row.get("adapt_append_only"), True, "append-only policy")
    _require_equal(row.get("optimizer_kind"), "powell", "optimizer")
    _require_equal(row.get("optimizer_maxiter"), 200, "optimizer maxiter")
    _require_equal(row.get("powell_maxiter_cap_policy"), CAP_POLICY, "Powell cap policy")
    _require_equal(row.get("optimizer_success_all"), True, "continuation optimizer success")
    total_iterations = int(row.get("adapt_num_iterations"))
    depth = int(row.get("adapt_depth_reached"))
    if not SOURCE_ITERATIONS < total_iterations <= MAX_TOTAL_ITERATIONS:
        raise RuntimeError(f"continuation total iteration count is invalid: {total_iterations}")
    _require_equal(depth, total_iterations, "Append depth/iteration identity")
    stop_reason = str(row.get("adapt_stop_reason"))
    if stop_reason not in {"benchmark_abs_delta_e_target", "max_adapt_iterations"}:
        raise RuntimeError(f"unexpected continuation stop reason: {stop_reason}")
    history = row.get("adapt_history")
    if not isinstance(history, list) or len(history) != total_iterations:
        raise RuntimeError("continuation history length mismatch")
    prefix_history = history[:SOURCE_ITERATIONS]
    _require_equal(
        _canonical_sha256(prefix_history),
        _canonical_sha256(state["adapt_history"]),
        "preserved 50-round history prefix",
    )
    labels = row.get("selected_operators")
    batches = row.get("selected_operator_batches")
    if not isinstance(labels, list) or labels[:SOURCE_DEPTH] != state["selected_operator_labels"]:
        raise RuntimeError("selected operator prefix drifted")
    if not isinstance(batches, list) or batches[:SOURCE_DEPTH] != state["selected_operator_batches"]:
        raise RuntimeError("selected batch prefix drifted")
    suffix = history[SOURCE_ITERATIONS:]
    for offset, item in enumerate(suffix, start=SOURCE_ITERATIONS):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"continuation history item {offset} is malformed")
        _require_equal(item.get("iteration"), offset, "continuation iteration")
        _require_equal(item.get("depth_after"), offset + 1, "continuation depth")
        _require_equal(item.get("optimizer_success"), True, "continuation optimizer success")
        _require_equal(item.get("optimizer_cap_policy"), CAP_POLICY, "continuation cap policy")
        if item.get("optimizer_capped") is True:
            _require_equal(item.get("optimizer_capped_accepted"), True, "capped acceptance")
            _require_equal(
                item.get("optimizer_cap_acceptance_reason"),
                "finite_nonincreasing_powell_maxiter_accepted",
                "capped acceptance reason",
            )
    errors = [float(item["abs_delta_e_same_cutoff_after"]) for item in suffix]
    crossing_offset = next((index for index, value in enumerate(errors) if value <= TARGET_ERROR), None)
    crossing_iteration_zero_based = (
        None if crossing_offset is None else SOURCE_ITERATIONS + crossing_offset
    )
    crossing_k_one_based = (
        None if crossing_iteration_zero_based is None else crossing_iteration_zero_based + 1
    )
    final_error = float(row.get("abs_delta_e_same_cutoff"))
    if stop_reason == "benchmark_abs_delta_e_target":
        if crossing_k_one_based != total_iterations or final_error > TARGET_ERROR:
            raise RuntimeError("target-stop result is not the first continuation crossing")
        _require_equal(row.get("adapt_target_stop_policy"), "warm_start_first_hit_or_max_iterations", "target-stop label")
        _require_equal(
            row.get("adapt_terminal_diagnostic_queries_in_S_alg"),
            False,
            "reporting-only terminal diagnostic accounting",
        )
    else:
        if total_iterations != MAX_TOTAL_ITERATIONS or crossing_k_one_based is not None:
            raise RuntimeError("k80 stop conflicts with observed crossing history")
        if final_error <= TARGET_ERROR:
            raise RuntimeError("k80 result should have stopped at its target crossing")
    _require_equal(row.get("hh_adaptive_pool_profile"), "full_meta_unfiltered", "HH pool profile")
    pool = job["candidate_pool"]
    for field in (
        "shared_pauli_pool_mode",
        "shared_pauli_pool_symmetry_policy",
        "shared_pauli_pool_max_subset_size",
    ):
        _require_equal(row.get(field), pool[field], field)
    _require_equal(row.get("generic_adapt_runtime_split_mode"), "off", "runtime split")
    _require_equal(payload.get("n_ph_work"), 7, "working cutoff")
    _require_equal(payload.get("n_ph_reference"), 7, "reference cutoff")
    _require_equal(payload.get("same_cutoff_reference"), True, "same-cutoff reference")
    _require_close(row.get("same_cutoff_exact_gs_energy"), job["exact_reference"]["energy"], "exact energy", 1.0e-12)
    _require_equal(row.get("primary_energy_metric"), "same_cutoff_abs_delta_e", "primary metric")
    _require_equal(row.get("same_cutoff_error_role"), "primary", "metric role")
    _require_equal(row.get("sector_leak_flag"), False, "fixed-sector leakage")
    _require_equal(row.get("boson_truncation_leak_flag"), False, "padding leakage")
    fidelity = float(row.get("exact_state_fidelity"))
    if not math.isfinite(fidelity) or not 0.0 <= fidelity <= 1.0:
        raise RuntimeError(f"invalid exact-state fidelity: {fidelity!r}")
    _require_equal(row.get("exact_state_fidelity_s_alg_charged"), False, "fidelity accounting")
    _require_equal(row.get("compiled_circuit_stats_status"), "ok", "Qiskit compile status")
    _require_equal(row.get("compiled_resource_qiskit_validated"), True, "Qiskit validation")
    _require_equal(row.get("qiskit_transpile_seed"), 7, "Qiskit seed")
    ledger = row.get("table_i_measurement_event_ledger")
    if not isinstance(ledger, Mapping):
        raise RuntimeError("measurement ledger is absent")
    _require_equal(ledger.get("status"), "ok", "measurement ledger status")
    _require_equal(ledger.get("source_kind"), "state_keyed_estimator_call_ledger_v1", "ledger source")
    receipts = row.get("estimator_call_round_receipts")
    if not isinstance(receipts, list) or len(receipts) != total_iterations - SOURCE_ITERATIONS:
        raise RuntimeError("continuation estimator round receipt count mismatch")
    _require_equal(
        [int(receipt["iteration"]) for receipt in receipts],
        list(range(SOURCE_ITERATIONS, total_iterations)),
        "continuation receipt iteration identity",
    )
    incremental_s_alg = float(row.get("S_alg"))
    component_sum = sum(
        float(row.get(field, 0.0))
        for field in (
            "S_alg_N_H_outer_eval",
            "S_alg_N_H_refit_eval",
            "S_alg_N_grad_probe",
            "S_alg_N_metric_probe",
            "S_alg_N_other_quantum",
        )
    )
    _require_close(component_sum, incremental_s_alg, "incremental S_alg components")
    round_sum = sum(float(receipt.get("S_alg_delta", 0.0)) for receipt in receipts)
    terminal_delta = _terminal_receipt_delta(row)
    _require_close(round_sum + terminal_delta, incremental_s_alg, "incremental receipt graph")
    if stop_reason == "benchmark_abs_delta_e_target":
        _require_close(terminal_delta, 0.0, "target terminal delta")
    cumulative_s_alg = SOURCE_S_ALG + incremental_s_alg
    accounting = {
        "schema": "paper_i_append_warm_continuation_accounting_v1",
        "created_utc": _utc_now(),
        "status": "pass",
        "source_S_alg": SOURCE_S_ALG,
        "source_S_alg_scope": "validated exact 50-round prefix",
        "continuation_incremental_S_alg": incremental_s_alg,
        "continuation_round_receipt_S_alg": round_sum,
        "continuation_terminal_receipt_S_alg": terminal_delta,
        "continuation_cumulative_S_alg": cumulative_s_alg,
        "cumulative_formula": "source_S_alg + continuation_incremental_S_alg",
        "core_row_S_alg_semantics": "incremental_continuation_only",
        "reporting_S_alg_field": "continuation_cumulative_S_alg",
        "terminal_diagnostic_semantics": "computed_for_reporting_but_uncharged_in_frozen_parent_source",
        "source_iterations": SOURCE_ITERATIONS,
        "new_iterations": total_iterations - SOURCE_ITERATIONS,
        "total_iterations": total_iterations,
        "stop_reason": stop_reason,
        "target_error": TARGET_ERROR,
        "crossing_iteration_zero_based": crossing_iteration_zero_based,
        "crossing_k_one_based": crossing_k_one_based,
        "terminal_error": final_error,
    }
    receipt = {
        "schema": "paper_i_hh_append_warm_continuation_validation_receipt_v1",
        "validated_utc": _utc_now(),
        "status": "pass",
        "job_id": job["job_id"],
        "regime": "weak-strong",
        "source_prefix_identity_sha256": state["prefix_identity"]["prefix_identity_sha256"],
        "source_iterations": SOURCE_ITERATIONS,
        "source_depth": SOURCE_DEPTH,
        "total_iterations": total_iterations,
        "active_depth": depth,
        "new_iterations": total_iterations - SOURCE_ITERATIONS,
        "stop_reason": stop_reason,
        "target_error": TARGET_ERROR,
        "crossing_iteration_zero_based": crossing_iteration_zero_based,
        "crossing_k_one_based": crossing_k_one_based,
        "energy": float(row.get("energy")),
        "same_cutoff_abs_error": final_error,
        "fidelity": fidelity,
        "continuation_incremental_S_alg": incremental_s_alg,
        "continuation_cumulative_S_alg": cumulative_s_alg,
        "compiled_count_2q_total": int(row["compiled_count_2q_total"]),
        "compiled_depth_2q_total": int(row["compiled_depth_2q_total"]),
        "compiled_depth_total": int(row["compiled_depth_total"]),
        "sector_leak_flag": False,
        "boson_truncation_leak_flag": False,
        "ledger_closure": "pass",
        "prefix_identity": "pass",
    }
    return receipt, accounting


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_manifest", type=Path)
    parser.add_argument("warm_start_state", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--validate-input-only", action="store_true")
    args = parser.parse_args()
    job = _load(args.job_manifest)
    state = _load(args.warm_start_state)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    input_receipt = _validate_input(job, state)
    _write(output / "input_preflight_receipt.json", input_receipt)
    if args.validate_input_only:
        print(json.dumps(input_receipt, indent=2, sort_keys=True))
        return 0
    execution: dict[str, Any] = {
        "schema": "paper_i_hh_append_warm_continuation_execution_v1",
        "job_id": job.get("job_id"),
        "started_utc": _utc_now(),
        "status": "running",
        "input_preflight": input_receipt,
    }
    _write(output / "execution.json", execution)
    os.environ["TABLE_I_STATIC_SUITE_PROFILE"] = str(job["physics"]["suite_profile"])
    os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "OFF"
    os.environ["STATIC_ADAPT_HH_POOL_CACHE_SCOPE"] = "paper-i-holstein-sector"
    os.environ["GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP"] = str(
        int(job["candidate_pool"]["expanded_pool_term_cap"])
    )
    from pipelines.exact_bench.generic_static_adapt_variants import (  # noqa: PLC0415
        run_generic_static_adapt_variant_single,
    )

    controller = job["controller"]
    optimizer = job["optimizer"]
    pool = job["candidate_pool"]
    result = run_generic_static_adapt_variant_single(
        family=job["family"],
        case_id=job["regime"]["case_id"],
        algorithm_id=job["algorithm_id"],
        output_dir=output,
        max_adapt_iterations=controller["max_adapt_iterations"],
        optimizer_maxiter=optimizer["maxiter"],
        gradient_threshold=controller["gradient_threshold"],
        seed=job["seed"],
        energy_stop_target=controller["energy_stop_target"],
        first_hit_thresholds=tuple(controller["first_hit_thresholds"]),
        same_cutoff_exact_gs_energy=job["exact_reference"]["energy"],
        exact_reference_energy=job["exact_reference"]["energy"],
        exact_reference_n_ph_max=job["exact_reference"]["n_ph_max"],
        primary_energy_metric="same_cutoff_abs_delta_e",
        same_cutoff_error_role="primary",
        selected_logical_route="standard",
        selected_logical_source_json=None,
        allow_repeats=True,
        progress_jsonl_path=output / "adapt_iteration_progress.jsonl",
        generic_adapt_stop_policy=STOP_POLICY,
        powell_maxiter_cap_policy=optimizer["powell_maxiter_cap_policy"],
        generic_adapt_runtime_split_mode="off",
        generic_adapt_runtime_split_symmetry_policy="off",
        generic_adapt_runtime_split_max_subset_size=3,
        shared_pauli_pool_mode=pool["shared_pauli_pool_mode"],
        shared_pauli_pool_symmetry_policy=pool["shared_pauli_pool_symmetry_policy"],
        shared_pauli_pool_max_subset_size=pool["shared_pauli_pool_max_subset_size"],
        initial_selected_operator_labels=state["selected_operator_labels"],
        initial_selected_operator_batches=state["selected_operator_batches"],
        initial_theta=state["theta"],
        initial_adapt_history=state["adapt_history"],
        adapt_optimizer_kind="powell",
        optimizer_overlay_source=optimizer["overlay_source"],
        hh_adaptive_pool_profile="full_meta_unfiltered",
    )
    if result.get("status") != "completed":
        raise RuntimeError(f"continuation failed: {result.get('reason', result.get('status'))}")
    receipt, accounting = _validate_result(job, state, result)
    _write(output / "validation_receipt.json", receipt)
    _write(output / "continuation_accounting.json", accounting)
    _write(output / "normalized_job_manifest.json", job)
    _write(output / "warm_start_state.json", state)
    _write(output / "source_prefix_identity.json", state["prefix_identity"])
    execution.update(
        {
            "status": "completed",
            "finished_utc": _utc_now(),
            "validation": receipt,
            "continuation_accounting": accounting,
        }
    )
    _write(output / "execution.json", execution)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
