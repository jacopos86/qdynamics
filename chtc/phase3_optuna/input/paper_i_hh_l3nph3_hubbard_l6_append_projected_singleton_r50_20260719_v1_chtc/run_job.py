#!/usr/bin/env python3
"""Execute and validate one source-locked higher-L Append child-pool job."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


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


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: {actual!r}!={expected!r}")


def _validate_result(job: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    horizon = int(job["controller"]["max_adapt_iterations"])
    payload = _load(output_dir / "generic_static_single.json")
    _require_equal(payload.get("status"), "completed", "top-level status")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("generic result must contain exactly one row")
    row = rows[0]
    _require_equal(row.get("status"), "ok", "row status")
    _require_equal(payload.get("family"), job["family"], "family")
    _require_equal(
        row.get("num_qubits"),
        int(job["runtime_limits"]["resource_qubit_cap"]),
        "declared qubit count",
    )
    _require_equal(row.get("algorithm_id"), job["algorithm_id"], "algorithm id")
    _require_equal(row.get("case_id"), job["regime"]["case_id"], "case id")
    _require_equal(row.get("seed"), job["seed"], "seed")
    _require_equal(row.get("adapt_num_iterations"), horizon, "completed ADAPT iterations")
    _require_equal(row.get("adapt_max_iterations"), horizon, "ADAPT horizon")
    _require_equal(row.get("adapt_stop_reason"), "max_adapt_iterations", "stop reason")
    _require_equal(row.get("generic_adapt_stop_policy"), "fixed_horizon_no_target_v1", "stop policy")
    _require_equal(row.get("adapt_continuation_mode"), "fresh", "continuation mode")
    _require_equal(row.get("adapt_warm_start_source_depth"), 0, "warm-start depth")
    _require_equal(row.get("adapt_warm_start_source_iterations"), 0, "warm-start iterations")
    _require_equal(row.get("adapt_selection_with_replacement"), True, "replacement policy")
    _require_equal(row.get("adapt_append_only"), True, "append-only policy")
    _require_equal(row.get("optimizer_kind"), "powell", "optimizer")
    _require_equal(row.get("optimizer_maxiter"), 200, "optimizer maxiter")
    cap_policy = str(job["optimizer"]["powell_maxiter_cap_policy"])
    _require_equal(row.get("powell_maxiter_cap_policy"), cap_policy, "Powell cap policy")
    _require_equal(row.get("optimizer_success_all"), True, "optimizer success")
    history = row.get("adapt_history")
    if not isinstance(history, list) or len(history) != horizon:
        raise RuntimeError(
            f"expected {horizon} completed Append rounds, got {len(history or [])}"
        )
    capped_iterations: list[int] = []
    for expected_iteration, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Append history item {expected_iteration} is malformed")
        _require_equal(item.get("iteration"), expected_iteration, "Append history iteration")
        _require_equal(item.get("optimizer_success"), True, "Append optimizer success")
        _require_equal(item.get("optimizer_cap_policy"), cap_policy, "Append cap policy receipt")
        if item.get("optimizer_capped") is True:
            capped_iterations.append(expected_iteration)
            _require_equal(item.get("optimizer_capped_accepted"), True, "Append capped acceptance")
            _require_equal(item.get("optimizer_cap_status"), 2, "Append capped status")
            _require_equal(item.get("optimizer_cap_nit"), job["optimizer"]["maxiter"], "Append capped nit")
            _require_equal(item.get("optimizer_cap_maxiter"), job["optimizer"]["maxiter"], "Append cap maxiter")
            _require_equal(item.get("optimizer_cap_message_match"), True, "Append capped message")
            _require_equal(item.get("optimizer_cap_parameters_finite"), True, "Append capped parameters")
            _require_equal(item.get("optimizer_cap_objective_finite"), True, "Append capped objective")
            _require_equal(item.get("optimizer_cap_energy_nonincreasing"), True, "Append capped energy")
            _require_equal(
                item.get("optimizer_cap_acceptance_reason"),
                "finite_nonincreasing_powell_maxiter_accepted",
                "Append capped acceptance reason",
            )
        else:
            _require_equal(item.get("optimizer_raw_success"), True, "Append raw optimizer success")
            _require_equal(item.get("optimizer_capped_accepted"), False, "Append uncapped acceptance")
            _require_equal(
                item.get("optimizer_cap_acceptance_reason"),
                "optimizer_success",
                "Append optimizer acceptance reason",
            )
    _require_equal(row.get("optimizer_capped_count"), len(capped_iterations), "Append capped count")
    _require_equal(row.get("optimizer_capped_iterations"), capped_iterations, "Append capped iterations")
    _require_equal(
        row.get("optimizer_capped_accepted_count"),
        len(capped_iterations),
        "Append capped accepted count",
    )
    _require_equal(
        row.get("optimizer_capped_accepted_iterations"),
        capped_iterations,
        "Append capped accepted iterations",
    )
    expected_hh_profile = "full_meta_unfiltered" if job["family"] == "hh" else None
    _require_equal(
        row.get("hh_adaptive_pool_profile"),
        expected_hh_profile,
        "family-aware HH pool profile",
    )
    pool = job["candidate_pool"]
    for field in (
        "shared_pauli_pool_mode",
        "shared_pauli_pool_symmetry_policy",
        "shared_pauli_pool_max_subset_size",
    ):
        _require_equal(row.get(field), pool[field], field)
    expanded_pool_count = row.get("shared_pauli_pool_expanded_pool_term_count")
    _require_equal(
        expanded_pool_count,
        int(pool["expected_projected_singleton_child_count"]),
        "projected-singleton child count",
    )
    if expanded_pool_count > int(pool["expanded_pool_term_cap"]):
        raise RuntimeError(
            f"projected-singleton pool exceeds recorded execution cap: {expanded_pool_count!r}"
        )
    _require_equal(
        row.get("shared_pauli_pool_base_pool_term_count"),
        int(pool["expected_parent_pool_count"]),
        "source parent pool count",
    )
    contract = row.get("shared_pauli_pool_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("shared Pauli-child contract receipt is absent")
    _require_equal(
        contract.get("projected_singleton_source_term_count"),
        int(pool["expected_raw_pauli_term_count"]),
        "raw Pauli term count",
    )
    _require_equal(
        contract.get("projected_singleton_null_count"),
        int(pool["expected_null_child_count"]),
        "null child count",
    )
    _require_equal(
        row.get("shared_pauli_pool_ordered_label_hash"),
        pool["expected_ordered_label_hash"],
        "projected-child ordered label hash",
    )
    _require_equal(
        row.get("shared_pauli_pool_ordered_pool_hash"),
        pool["expected_ordered_pool_hash"],
        "projected-child ordered pool hash",
    )
    _require_equal(row.get("generic_adapt_runtime_split_mode"), "off", "runtime split")
    _require_equal(payload.get("n_ph_work"), job["physics"]["n_ph_work"], "working cutoff")
    _require_equal(payload.get("n_ph_reference"), job["physics"]["n_ph_ref"], "reference cutoff")
    _require_equal(
        payload.get("same_cutoff_reference"),
        job["physics"]["same_cutoff_reference"],
        "same-cutoff reference",
    )
    exact = float(job["exact_reference"]["energy"])
    observed_exact = float(row.get("same_cutoff_exact_gs_energy"))
    if not math.isclose(observed_exact, exact, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"same-cutoff exact energy mismatch: {observed_exact}!={exact}")
    _require_equal(row.get("primary_energy_metric"), "same_cutoff_abs_delta_e", "primary metric")
    _require_equal(row.get("same_cutoff_error_role"), "primary", "same-cutoff metric role")
    _require_equal(row.get("sector_leak_flag"), False, "fixed-sector leakage")
    _require_equal(row.get("boson_truncation_leak_flag"), False, "binary-padding leakage")
    sector_diagnostics = row.get("sector_diagnostics")
    if not isinstance(sector_diagnostics, Mapping):
        raise RuntimeError("sector diagnostics are absent")
    fixed_counts = sector_diagnostics.get("fixed_count_constraints_evaluated")
    if not isinstance(fixed_counts, list):
        raise RuntimeError("fixed-count sector receipts are absent")
    observed_particles = [int(item["value"]) for item in fixed_counts]
    _require_equal(
        observed_particles,
        list(job["sector_lock"]["num_particles"]),
        "fixed-count particle sector",
    )
    fidelity = float(row.get("exact_state_fidelity"))
    if not math.isfinite(fidelity) or not 0.0 <= fidelity <= 1.0:
        raise RuntimeError(f"invalid exact-state fidelity: {fidelity!r}")
    _require_equal(row.get("exact_state_fidelity_s_alg_charged"), False, "fidelity accounting")
    if not row.get("exact_state_fidelity_reference_convention"):
        raise RuntimeError("ground-space fidelity convention is absent")
    _require_equal(row.get("compiled_circuit_stats_status"), "ok", "Qiskit compile status")
    _require_equal(row.get("compiled_resource_qiskit_validated"), True, "Qiskit validation")
    _require_equal(row.get("qiskit_transpile_seed"), 7, "Qiskit seed")
    for field in ("compiled_count_2q_total", "compiled_depth_2q_total", "compiled_depth_total"):
        value = row.get(field)
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid compiled cost {field}: {value!r}")
    ledger = row.get("table_i_measurement_event_ledger")
    if not isinstance(ledger, dict):
        raise RuntimeError("measurement ledger is absent")
    _require_equal(ledger.get("status"), "ok", "measurement ledger status")
    _require_equal(ledger.get("source_kind"), "state_keyed_estimator_call_ledger_v1", "ledger source")
    receipts = row.get("estimator_call_round_receipts")
    if not isinstance(receipts, list) or len(receipts) != horizon:
        raise RuntimeError(
            f"expected {horizon} estimator round receipts, got {len(receipts or [])}"
        )
    receipt_sum = sum(float(receipt["S_alg_delta"]) for receipt in receipts)
    s_alg = float(row.get("S_alg"))
    if not math.isclose(receipt_sum, s_alg, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"round-receipt ledger does not close: {receipt_sum}!={s_alg}")
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
    if not math.isclose(component_sum, s_alg, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"terminal S_alg components do not close: {component_sum}!={s_alg}")
    return {
        "schema": "paper_i_higher_l_append_validation_receipt_v1",
        "validated_utc": _utc_now(),
        "status": "pass",
        "job_id": job["job_id"],
        "variant": job["variant"]["slug"],
        "family": job["family"],
        "case_id": job["regime"]["case_id"],
        "regime": job["regime"]["label"],
        "adapt_iterations": horizon,
        "active_depth": int(row.get("adapt_depth_reached")),
        "energy": float(row.get("energy")),
        "same_cutoff_abs_error": float(row.get("abs_delta_e_same_cutoff")),
        "fidelity": fidelity,
        "S_alg": s_alg,
        "compiled_count_2q_total": row["compiled_count_2q_total"],
        "compiled_depth_2q_total": row["compiled_depth_2q_total"],
        "compiled_depth_total": row["compiled_depth_total"],
        "sector_leak_flag": False,
        "boson_truncation_leak_flag": False,
        "sector_label": job["sector_lock"]["label"],
        "sector_num_particles": job["sector_lock"]["num_particles"],
        "ledger_closure": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    job = _load(args.job_manifest)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    execution: dict[str, Any] = {
        "schema": "paper_i_higher_l_append_execution_v1",
        "job_id": job.get("job_id"),
        "started_utc": started,
        "status": "running",
    }
    _write(output / "execution.json", execution)
    os.environ["TABLE_I_STATIC_SUITE_PROFILE"] = str(job["physics"]["suite_profile"])
    os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "OFF"
    os.environ["STATIC_ADAPT_HH_POOL_CACHE_SCOPE"] = "paper-i-holstein-sector"
    pool = job["candidate_pool"]
    os.environ["GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP"] = str(
        int(pool["expanded_pool_term_cap"])
    )
    runtime_limits = job["runtime_limits"]
    os.environ["GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP"] = str(
        int(runtime_limits["resource_qubit_cap"])
    )
    os.environ["GENERIC_STATIC_TABLE_EXACT_FIDELITY_MAX_QUBITS"] = str(
        int(runtime_limits["exact_fidelity_max_qubits"])
    )
    from pipelines.exact_bench.generic_static_adapt_variants import (  # noqa: PLC0415
        run_generic_static_adapt_variant_single,
    )

    controller = job["controller"]
    optimizer = job["optimizer"]
    result = run_generic_static_adapt_variant_single(
        family=job["family"],
        case_id=job["regime"]["case_id"],
        algorithm_id=job["algorithm_id"],
        output_dir=output,
        max_adapt_iterations=controller["max_adapt_iterations"],
        optimizer_maxiter=optimizer["maxiter"],
        gradient_threshold=controller["gradient_threshold"],
        seed=job["seed"],
        energy_stop_target=None,
        same_cutoff_exact_gs_energy=job["exact_reference"]["energy"],
        exact_reference_energy=job["exact_reference"]["energy"],
        exact_reference_n_ph_max=job["exact_reference"]["n_ph_max"],
        primary_energy_metric="same_cutoff_abs_delta_e",
        same_cutoff_error_role="primary",
        selected_logical_route="standard",
        selected_logical_source_json=None,
        allow_repeats=True,
        progress_jsonl_path=output / "adapt_iteration_progress.jsonl",
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        powell_maxiter_cap_policy=optimizer["powell_maxiter_cap_policy"],
        generic_adapt_runtime_split_mode="off",
        generic_adapt_runtime_split_symmetry_policy="off",
        generic_adapt_runtime_split_max_subset_size=3,
        shared_pauli_pool_mode=pool["shared_pauli_pool_mode"],
        shared_pauli_pool_symmetry_policy=pool["shared_pauli_pool_symmetry_policy"],
        shared_pauli_pool_max_subset_size=pool["shared_pauli_pool_max_subset_size"],
        adapt_optimizer_kind="powell",
        optimizer_overlay_source=optimizer["overlay_source"],
        hh_adaptive_pool_profile=(
            "full_meta_unfiltered" if job["family"] == "hh" else None
        ),
    )
    if result.get("status") != "completed":
        raise RuntimeError(f"comparator run failed: {result.get('reason', result.get('status'))}")
    receipt = _validate_result(job, output)
    _write(output / "validation_receipt.json", receipt)
    execution.update({"status": "completed", "finished_utc": _utc_now(), "validation": receipt})
    _write(output / "execution.json", execution)
    _write(output / "normalized_job_manifest.json", job)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
