#!/usr/bin/env python3
"""Execute and validate one source-locked Geo-ADAPT completion job."""

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


def _validate_geo_history(row: Mapping[str, Any]) -> dict[str, Any]:
    history = row.get("adapt_history")
    if not isinstance(history, list) or len(history) != 50:
        raise RuntimeError(f"expected 50 completed Geo rounds, got {len(history or [])}")
    immediate_repeat_skips = 0
    for expected_iteration, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Geo history item {expected_iteration} is malformed")
        _require_equal(item.get("iteration"), expected_iteration, "Geo history iteration")
        _require_equal(item.get("geo_stop_rule"), "fubini_study_natural_gradient_norm", "Geo stop rule")
        before = item.get("geo_metric_candidate_count_before_screen")
        after = item.get("geo_metric_candidate_count_after_screen")
        if not isinstance(before, int) or before <= 0:
            raise RuntimeError(f"Geo round {expected_iteration} has invalid candidate count: {before!r}")
        _require_equal(after, before, "Geo full-pool metric prescreen count")
        top = item.get("top_candidates")
        if not isinstance(top, list) or not top or not isinstance(top[0], Mapping):
            raise RuntimeError(f"Geo round {expected_iteration} has no top-candidate receipt")
        receipt = top[0]
        _require_equal(
            receipt.get("geo_metric_factorization"),
            "exact_real_tangent_span_svd_v1",
            "Geo metric factorization",
        )
        _require_equal(receipt.get("geo_metric_solve_kind"), "moore_penrose_pseudoinverse", "Geo solve kind")
        _require_equal(
            receipt.get("geo_metric_regularization_used_in_selector_solve"),
            False,
            "Geo selector regularization",
        )
        _require_equal(receipt.get("geo_metric_prescreen_mode"), "full_candidate_set", "Geo prescreen")
        _require_equal(
            receipt.get("geo_selector_numerical_tie_policy"),
            "machine_precision_collinear_alias_then_label_v1",
            "Geo numerical tie policy",
        )
        supported_rank = receipt.get("geo_metric_pinv_supported_rank")
        if not isinstance(supported_rank, int) or supported_rank <= 0 or supported_rank > before:
            raise RuntimeError(f"Geo round {expected_iteration} has invalid supported rank: {supported_rank!r}")
        if item.get("geo_immediate_repeat_skipped") is True:
            immediate_repeat_skips += 1
    return {"immediate_repeat_skip_count": immediate_repeat_skips}


def _validate_result(job: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    payload = _load(output_dir / "generic_static_single.json")
    _require_equal(payload.get("status"), "completed", "top-level status")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("generic result must contain exactly one row")
    row = rows[0]
    _require_equal(row.get("status"), "ok", "row status")
    _require_equal(row.get("algorithm_id"), job["algorithm_id"], "algorithm id")
    _require_equal(row.get("case_id"), job["regime"]["case_id"], "case id")
    _require_equal(row.get("seed"), job["seed"], "seed")
    _require_equal(row.get("adapt_num_iterations"), 50, "completed ADAPT iterations")
    _require_equal(row.get("adapt_max_iterations"), 50, "ADAPT horizon")
    _require_equal(row.get("adapt_stop_reason"), "max_adapt_iterations", "stop reason")
    _require_equal(row.get("generic_adapt_stop_policy"), "fixed_horizon_no_target_v1", "stop policy")
    _require_equal(row.get("adapt_continuation_mode"), "fresh", "continuation mode")
    _require_equal(row.get("adapt_warm_start_source_depth"), 0, "warm-start depth")
    _require_equal(row.get("adapt_warm_start_source_iterations"), 0, "warm-start iterations")
    _require_equal(row.get("adapt_selection_with_replacement"), True, "replacement policy")
    _require_equal(row.get("adapt_repeat_policy"), "with_replacement_except_immediate_repeat", "repeat policy")
    _require_equal(row.get("adapt_append_only"), True, "append-only policy")
    _require_equal(row.get("geo_immediate_repeat_blocked"), True, "immediate-repeat policy")
    _require_equal(
        row.get("geo_immediate_repeat_policy_stage"),
        "post_full_pool_selection_skip_append",
        "immediate-repeat policy stage",
    )
    _require_equal(
        row.get("geo_replacement_policy"),
        "score_full_pool_with_replacement; skip_append_after_immediate_repeat_wins",
        "Geo replacement policy",
    )
    _require_equal(row.get("geo_selector_mode"), "full_pool_projected_natural_gradient", "Geo selector mode")
    _require_equal(row.get("geo_inner_optimizer"), "powell", "Geo inner optimizer")
    _require_equal(row.get("position_policy"), "append", "Geo insertion policy")
    _require_equal(row.get("position_optimized_geo_adapt"), False, "position optimization")
    _require_equal(row.get("optimizer_kind"), "powell", "optimizer")
    _require_equal(row.get("optimizer_maxiter"), 200, "optimizer maxiter")
    _require_equal(row.get("powell_maxiter_cap_policy"), "strict_failure_v1", "Powell cap policy")
    _require_equal(row.get("optimizer_success_all"), True, "optimizer success")
    _require_equal(row.get("hh_adaptive_pool_profile"), "full_meta_unfiltered", "HH pool profile")
    pool = job["candidate_pool"]
    for field in (
        "shared_pauli_pool_mode",
        "shared_pauli_pool_symmetry_policy",
        "shared_pauli_pool_max_subset_size",
    ):
        _require_equal(row.get(field), pool[field], field)
    expanded_pool_count = row.get("shared_pauli_pool_expanded_pool_term_count")
    if not isinstance(expanded_pool_count, int) or expanded_pool_count <= 256:
        raise RuntimeError(f"projected-singleton pool was not fully expanded: {expanded_pool_count!r}")
    if expanded_pool_count > int(pool["expanded_pool_term_cap"]):
        raise RuntimeError(
            f"projected-singleton pool exceeds recorded execution cap: {expanded_pool_count!r}"
        )
    _require_equal(row.get("generic_adapt_runtime_split_mode"), "off", "runtime split")
    _require_equal(payload.get("n_ph_work"), job["physics"]["n_ph_work"], "working cutoff")
    _require_equal(payload.get("n_ph_reference"), job["physics"]["n_ph_ref"], "reference cutoff")
    _require_equal(payload.get("same_cutoff_reference"), True, "same-cutoff reference")
    exact = float(job["exact_reference"]["energy"])
    observed_exact = float(row.get("same_cutoff_exact_gs_energy"))
    if not math.isclose(observed_exact, exact, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"same-cutoff exact energy mismatch: {observed_exact}!={exact}")
    _require_equal(row.get("primary_energy_metric"), "same_cutoff_abs_delta_e", "primary metric")
    _require_equal(row.get("same_cutoff_error_role"), "primary", "same-cutoff metric role")
    _require_equal(row.get("sector_leak_flag"), False, "fixed-sector leakage")
    _require_equal(row.get("boson_truncation_leak_flag"), False, "binary-padding leakage")
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
    history_validation = _validate_geo_history(row)
    ledger = row.get("table_i_measurement_event_ledger")
    if not isinstance(ledger, dict):
        raise RuntimeError("measurement ledger is absent")
    _require_equal(ledger.get("status"), "ok", "measurement ledger status")
    _require_equal(ledger.get("source_kind"), "state_keyed_estimator_call_ledger_v1", "ledger source")
    receipts = row.get("estimator_call_round_receipts")
    if not isinstance(receipts, list) or len(receipts) != 50:
        raise RuntimeError(f"expected 50 estimator round receipts, got {len(receipts or [])}")
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
        "schema": "paper_i_hh_geo_completion_validation_receipt_v1",
        "validated_utc": _utc_now(),
        "status": "pass",
        "job_id": job["job_id"],
        "variant": job["variant"]["slug"],
        "regime": job["regime"]["label"],
        "adapt_iterations": 50,
        "active_depth": int(row.get("adapt_depth_reached")),
        "immediate_repeat_skip_count": history_validation["immediate_repeat_skip_count"],
        "energy": float(row.get("energy")),
        "same_cutoff_abs_error": float(row.get("abs_delta_e_same_cutoff")),
        "fidelity": fidelity,
        "S_alg": s_alg,
        "compiled_count_2q_total": row["compiled_count_2q_total"],
        "compiled_depth_2q_total": row["compiled_depth_2q_total"],
        "compiled_depth_total": row["compiled_depth_total"],
        "geo_metric_factorization": "exact_real_tangent_span_svd_v1",
        "sector_leak_flag": False,
        "boson_truncation_leak_flag": False,
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
    execution: dict[str, Any] = {
        "schema": "paper_i_hh_geo_completion_execution_v1",
        "job_id": job.get("job_id"),
        "source_lock": job.get("source_lock"),
        "started_utc": _utc_now(),
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
        metric_floor=job["geo_selector"]["metric_pinv_rcond"],
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
        powell_maxiter_cap_policy="strict_failure_v1",
        generic_adapt_runtime_split_mode="off",
        generic_adapt_runtime_split_symmetry_policy="off",
        generic_adapt_runtime_split_max_subset_size=3,
        shared_pauli_pool_mode=pool["shared_pauli_pool_mode"],
        shared_pauli_pool_symmetry_policy=pool["shared_pauli_pool_symmetry_policy"],
        shared_pauli_pool_max_subset_size=pool["shared_pauli_pool_max_subset_size"],
        adapt_optimizer_kind="powell",
        optimizer_overlay_source=optimizer["overlay_source"],
        hh_adaptive_pool_profile="full_meta_unfiltered",
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
