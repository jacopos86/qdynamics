#!/usr/bin/env python3
"""Focused tests for the submission-enabled Append-S continuation bundle."""

from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_module("append_warm_builder", BUNDLE / "build_bundle.py")
worker = _load_module("append_warm_worker", BUNDLE / "run_job.py")


def _load(path: Path):  # noqa: ANN202
    return json.loads(path.read_text(encoding="utf-8"))


def test_bundle_is_one_row_and_submission_enabled() -> None:
    queue_rows = (BUNDLE / "queue.tsv").read_text(encoding="utf-8").splitlines()
    assert len(queue_rows) == 1
    assert queue_rows[0].startswith(builder.JOB_ID + "\t")
    submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
    assert "requirements = True" in submit
    assert "Submission enabled" in submit
    manifest = _load(BUNDLE / "bundle_manifest.json")
    assert manifest["row_count"] == 1
    assert manifest["submission_enabled"] is True
    assert manifest["remote_image_preflight"] == "exact_uploaded_v2_required_before_condor_submit"


def test_source_archive_is_exact_parent_plus_surgical_patch() -> None:
    source_manifest = _load(BUNDLE / "source_archive_manifest.json")
    assert source_manifest["parent_archive_sha256"] == builder.BASE_SOURCE_SHA256
    assert [row["path"] for row in source_manifest["changed_members"]] == [
        builder.GENERIC_REL.as_posix(),
        builder.TEST_REL.as_posix(),
    ]
    with tarfile.open(BUNDLE / "source_locked.tar.gz", "r:gz") as tar:
        generic = tar.extractfile(builder.GENERIC_REL.as_posix()).read().decode("utf-8")
        tests = tar.extractfile(builder.TEST_REL.as_posix()).read().decode("utf-8")
    assert builder.STOP_POLICY in generic
    assert "warm_start_first_hit_or_max_iterations" in generic
    assert "test_append_warm_first_hit_policy_rejects_incomplete_contract" in tests


def test_manifest_changes_are_authorized_and_baseline_science_is_identical() -> None:
    audit = _load(BUNDLE / "source_locked_sensitivity_audit.json")
    assert audit["status"] == "pass"
    assert audit["unauthorized_manifest_diff"] == []
    assert all(audit["scientific_settings_preserved"].values())
    assert audit["submission_authorized"] is True
    job = _load(BUNDLE / "jobs" / f"{builder.JOB_ID}.json")
    assert job["controller"]["max_adapt_iterations"] == 80
    assert job["controller"]["energy_stop_target"] == 2.0e-4
    assert job["controller"]["stop_policy"] == builder.STOP_POLICY
    assert job["continuation"]["source_S_alg"] == 1_276_060.0


def test_warm_state_and_input_preflight_close() -> None:
    state = _load(BUNDLE / "warm_start_state.json")
    job = _load(BUNDLE / "jobs" / f"{builder.JOB_ID}.json")
    receipt = worker._validate_input(job, state)
    assert receipt["status"] == "pass"
    assert receipt["source_iterations"] == 50
    assert len(state["selected_operator_labels"]) == 50
    assert len(state["selected_operator_batches"]) == 50
    assert len(state["theta"]) == 50
    assert len(state["adapt_history"]) == 50
    assert [batch[0] for batch in state["selected_operator_batches"]] == state["selected_operator_labels"]


def test_target_crossing_reports_incremental_and_cumulative_s_alg() -> None:
    state = _load(BUNDLE / "warm_start_state.json")
    job = _load(BUNDLE / "jobs" / f"{builder.JOB_ID}.json")
    suffix = {
        "iteration": 50,
        "depth_after": 51,
        "optimizer_success": True,
        "optimizer_cap_policy": builder.CAP_POLICY,
        "optimizer_capped": False,
        "optimizer_capped_accepted": False,
        "optimizer_cap_acceptance_reason": "optimizer_success",
        "abs_delta_e_same_cutoff_after": 1.5e-4,
    }
    row = {
        "status": "ok",
        "algorithm_id": job["algorithm_id"],
        "case_id": job["regime"]["case_id"],
        "seed": 7,
        "adapt_max_iterations": 80,
        "generic_adapt_stop_policy": builder.STOP_POLICY,
        "adapt_continuation_mode": "warm_start_selected_theta_v1",
        "adapt_warm_start_source_depth": 50,
        "adapt_warm_start_source_iterations": 50,
        "adapt_selection_with_replacement": True,
        "adapt_append_only": True,
        "optimizer_kind": "powell",
        "optimizer_maxiter": 200,
        "powell_maxiter_cap_policy": builder.CAP_POLICY,
        "optimizer_success_all": True,
        "adapt_num_iterations": 51,
        "adapt_depth_reached": 51,
        "adapt_stop_reason": "benchmark_abs_delta_e_target",
        "adapt_target_stop_policy": "warm_start_first_hit_or_max_iterations",
        "adapt_terminal_diagnostic_queries_in_S_alg": False,
        "adapt_history": state["adapt_history"] + [suffix],
        "selected_operators": state["selected_operator_labels"] + ["continuation_operator"],
        "selected_operator_batches": state["selected_operator_batches"] + [["continuation_operator"]],
        "abs_delta_e_same_cutoff": 1.5e-4,
        "energy": float(job["exact_reference"]["energy"]) + 1.5e-4,
        "hh_adaptive_pool_profile": "full_meta_unfiltered",
        "shared_pauli_pool_mode": job["candidate_pool"]["shared_pauli_pool_mode"],
        "shared_pauli_pool_symmetry_policy": job["candidate_pool"]["shared_pauli_pool_symmetry_policy"],
        "shared_pauli_pool_max_subset_size": 1,
        "generic_adapt_runtime_split_mode": "off",
        "same_cutoff_exact_gs_energy": job["exact_reference"]["energy"],
        "primary_energy_metric": "same_cutoff_abs_delta_e",
        "same_cutoff_error_role": "primary",
        "sector_leak_flag": False,
        "boson_truncation_leak_flag": False,
        "exact_state_fidelity": 0.999,
        "exact_state_fidelity_s_alg_charged": False,
        "compiled_circuit_stats_status": "ok",
        "compiled_resource_qiskit_validated": True,
        "qiskit_transpile_seed": 7,
        "compiled_count_2q_total": 300,
        "compiled_depth_2q_total": 240,
        "compiled_depth_total": 940,
        "table_i_measurement_event_ledger": {
            "status": "ok",
            "source_kind": "state_keyed_estimator_call_ledger_v1",
        },
        "estimator_call_round_receipts": [{"iteration": 50, "S_alg_delta": 123.0}],
        "S_alg": 123.0,
        "S_alg_N_H_outer_eval": 1.0,
        "S_alg_N_H_refit_eval": 20.0,
        "S_alg_N_grad_probe": 102.0,
        "S_alg_N_metric_probe": 0.0,
        "S_alg_N_other_quantum": 0.0,
        "adapt_terminal_estimator_call_receipt": None,
    }
    payload = {
        "status": "completed",
        "rows": [row],
        "n_ph_work": 7,
        "n_ph_reference": 7,
        "same_cutoff_reference": True,
    }
    receipt, accounting = worker._validate_result(job, state, payload)
    assert receipt["crossing_k_one_based"] == 51
    assert receipt["continuation_incremental_S_alg"] == 123.0
    assert receipt["continuation_cumulative_S_alg"] == 1_276_183.0
    assert accounting["reporting_S_alg_field"] == "continuation_cumulative_S_alg"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
