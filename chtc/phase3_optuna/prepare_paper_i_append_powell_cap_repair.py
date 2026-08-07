#!/usr/bin/env python3
"""Prepare the single Append-ADAPT Powell-cap repair row; never submit it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna import generate_paper_i_scaling_matrix_records as scaling  # noqa: E402
from chtc.phase3_optuna import run_paper_i_scaling_matrix_cell as cell_runner  # noqa: E402


SOURCE_BATCH_ID = "paper_i_scaling_matrix_parent_powell200_20260710_v1"
SOURCE_RECORD_ID = (
    "paper_i_scaling_matrix_parent_powell200_20260710_v1__hubbard__"
    "hubbard_L4_scaling_strong__append__parent_powell200__iter30"
)
SOURCE_CLUSTER = 8772847
SOURCE_PROCESS = 53
SOURCE_RESULT_SHA256 = "00dca6c25128958ee7e7b5a9c85714098aaddb7b80dbd9fc011d78a4b6babdce"
SOURCE_CELL_MANIFEST_SHA256 = "340d419416ea173a868136d7bf1fff85ccdaa3d4cfc3dc29dc7d4ffa2f4d2297"
REPAIR_BATCH_ID = "paper_i_scaling_matrix_append_powell_cap_repair_20260711_v1"
REPAIR_RECORD_ID = (
    "paper_i_scaling_matrix_append_powell_cap_repair_20260711_v1__hubbard__"
    "hubbard_L4_scaling_strong__append__parent_powell200__"
    "powellcap_finite_nonincreasing_v1__iter30"
)
CAP_POLICY = "accept_finite_nonincreasing_v1"
LOCAL_SOURCE_EVIDENCE_DIR = (
    ROOT / "output" / "chtc_retrievals" / "paper_i_append_powell_cap_repair_source_20260711"
)
LOCAL_SOURCE_RESULT = LOCAL_SOURCE_EVIDENCE_DIR / "result" / "generic_static_single.json"
LOCAL_SOURCE_CELL_MANIFEST = LOCAL_SOURCE_EVIDENCE_DIR / "cell_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _load_source_row(records_path: Path) -> dict[str, str]:
    with records_path.open(newline="", encoding="utf-8") as handle:
        matches = [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
            if str(row.get("record_id") or "") == SOURCE_RECORD_ID
        ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one proc-53 source row, found {len(matches)}")
    row = matches[0]
    expected = {
        "family": "hubbard",
        "case_id": "hubbard_L4_scaling_strong",
        "method_key": "append",
        "algorithm_id": "static_full_meta_append_adapt_vqe",
        "adapt_optimizer_kind": "powell",
        "optimizer": "POWELL",
        "budget": "200",
        "phase3_adapt_maxiter": "200",
        "phase3_refit_maxiter": "200",
        "phase3_final_maxiter": "200",
        "expected_horizon": "30",
        "generic_adapt_stop_policy": "fixed_horizon_no_target_v1",
        "child_policy": "macro_only",
        "pool_contract": "full_meta_unfiltered",
        "shared_pauli_pool_mode": "off",
        "generic_adapt_runtime_split_mode": "off",
        "phase2_batching": "off",
        "phase3_batching": "off",
    }
    problems = [
        f"{field}={row.get(field)!r}, expected {value!r}"
        for field, value in expected.items()
        if str(row.get(field) or "") != value
    ]
    if problems:
        raise ValueError("proc-53 source contract drifted: " + "; ".join(problems))
    return row


def _science_settings(row: Mapping[str, str]) -> dict[str, str]:
    fields = (
        "family",
        "case_id",
        "algorithm_id",
        "method_key",
        "L",
        "n_ph_work",
        "n_ph_ref",
        "exact_reference_n_ph_max",
        "same_cutoff_exact_gs_energy",
        "exact_reference_energy",
        "primary_energy_metric",
        "same_cutoff_error_role",
        "optimizer",
        "adapt_optimizer_kind",
        "budget",
        "phase3_adapt_maxiter",
        "phase3_refit_maxiter",
        "phase3_final_maxiter",
        "max_depth",
        "phase3_adapt_max_depth",
        "expected_horizon",
        "generic_adapt_stop_policy",
        "pool_contract",
        "child_policy",
        "parent_generator_policy",
        "generic_adapt_runtime_split_mode",
        "generic_adapt_runtime_split_symmetry_policy",
        "generic_adapt_runtime_split_max_subset_size",
        "shared_pauli_pool_mode",
        "shared_pauli_pool_symmetry_policy",
        "shared_pauli_pool_max_subset_size",
        "phase2_batching",
        "phase3_batching",
        "one_accepted_parent_per_outer_iteration",
        "adapt_allow_repeats",
        "geo_immediate_repeat_policy",
        "append_selection_policy",
        "exact_fidelity_max_qubits",
        "resource_qubit_cap",
        "resource_pool_term_cap",
    )
    out = {field: str(row.get(field) or "") for field in fields}
    out["powell_maxiter_cap_policy"] = str(
        row.get("powell_maxiter_cap_policy") or "strict_failure_v1"
    )
    return out


def prepare(
    *,
    output_dir: Path,
    submit_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    submit_path = Path(submit_path).expanduser().resolve()
    scaling._ensure_fresh_targets(output_dir, submit_path, force=bool(force))

    source_dir = ROOT / "chtc" / "phase3_optuna" / "input" / SOURCE_BATCH_ID
    source_records = source_dir / "paper_i_scaling_matrix_records.tsv"
    source_exact_manifest = source_dir / "exact_energy_manifest.json"
    source_row = _load_source_row(source_records)
    source_science = _science_settings(source_row)

    source_evidence_dir = output_dir / "source_evidence"
    source_evidence_dir.mkdir(parents=True, exist_ok=True)
    packaged_source_result = source_evidence_dir / "proc53_generic_static_single.json"
    packaged_source_cell_manifest = source_evidence_dir / "proc53_cell_manifest.json"
    shutil.copy2(LOCAL_SOURCE_RESULT, packaged_source_result)
    shutil.copy2(LOCAL_SOURCE_CELL_MANIFEST, packaged_source_cell_manifest)
    if _sha256(packaged_source_result) != SOURCE_RESULT_SHA256:
        raise ValueError("fetched proc-53 result JSON hash mismatch")
    if _sha256(packaged_source_cell_manifest) != SOURCE_CELL_MANIFEST_SHA256:
        raise ValueError("fetched proc-53 cell manifest hash mismatch")

    exact_manifest = output_dir / "exact_energy_manifest.json"
    shutil.copy2(source_exact_manifest, exact_manifest)
    exact_manifest_sha256 = _sha256(exact_manifest)
    if exact_manifest_sha256 != str(source_row["exact_energy_manifest_sha256"]):
        raise ValueError("copied exact-energy manifest hash does not match proc-53")

    code_bundle = scaling._write_code_bundle(output_dir)
    implementation_lock, implementation_lock_path = scaling._write_implementation_lock(
        output_dir,
        code_bundle,
    )
    implementation_lock_sha256 = _sha256(implementation_lock_path)

    row = dict(source_row)
    record_output_dir = f"raw_outputs/{REPAIR_BATCH_ID}/{REPAIR_RECORD_ID}"
    row.update(
        {
            "record_id": REPAIR_RECORD_ID,
            "batch_id": REPAIR_BATCH_ID,
            "powell_maxiter_cap_policy": CAP_POLICY,
            "implementation_contract_id": "append_powell_maxiter_cap_finite_nonincreasing_20260711_v1",
            "repair_scope": (
                "single_append_hubbard_L4_strong_powell_cap_finite_nonincreasing_v1"
            ),
            "repair_source_batch_id": SOURCE_BATCH_ID,
            "repair_source_record_id": SOURCE_RECORD_ID,
            "source_record_id": SOURCE_RECORD_ID,
            "source_cluster": str(SOURCE_CLUSTER),
            "source_process": str(SOURCE_PROCESS),
            "source_result_json_sha256": SOURCE_RESULT_SHA256,
            "source_cell_manifest_sha256": SOURCE_CELL_MANIFEST_SHA256,
            "source_result_json_local": _repo_path(packaged_source_result),
            "source_cell_manifest_local": _repo_path(packaged_source_cell_manifest),
            "source_code_bundle_sha256": str(source_row["code_bundle_sha256"]),
            "source_implementation_lock_sha256": str(source_row["implementation_lock_sha256"]),
            "source_settings_changed": str(source_row.get("settings_changed") or ""),
            "implementation_repair_audit": _repo_path(
                output_dir / "implementation_repair_audit.json"
            ),
            "settings_diff_json": _repo_path(output_dir / "settings_diff.json"),
            "settings_changed": "powell_maxiter_cap_policy;new_output_identity",
            "settings_reused": (
                "all_proc53_declared_physics_pool_optimizer_budget_horizon_and_accounting_fields"
            ),
            "exact_energy_manifest": _repo_path(exact_manifest),
            "exact_energy_manifest_sha256": exact_manifest_sha256,
            "code_bundle": str(code_bundle["path"]),
            "code_bundle_sha256": str(code_bundle["sha256"]),
            "implementation_lock": _repo_path(implementation_lock_path),
            "implementation_lock_sha256": implementation_lock_sha256,
            "record_output_dir": record_output_dir,
            "result_json_rel": f"{record_output_dir}/result/generic_static_single.json",
            "current_json_rel": f"{record_output_dir}/adapt_iteration_progress.jsonl",
            "stdout_rel": f"{record_output_dir}/stdout.log",
            "stderr_rel": f"{record_output_dir}/stderr.log",
            "cell_manifest_rel": f"{record_output_dir}/cell_manifest.json",
        }
    )
    cell_runner.validate_record(row)

    repair_science = _science_settings(row)
    science_diff = {
        key: {"source": source_science.get(key), "repair": repair_science.get(key)}
        for key in sorted(set(source_science) | set(repair_science))
        if source_science.get(key) != repair_science.get(key)
    }
    if science_diff != {
        "powell_maxiter_cap_policy": {
            "source": "strict_failure_v1",
            "repair": CAP_POLICY,
        }
    }:
        raise ValueError(f"unexpected science-settings drift: {science_diff}")

    records_path = output_dir / "paper_i_scaling_matrix_records.tsv"
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted(row),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)
    ids_path = output_dir / "paper_i_scaling_matrix_record_ids.txt"
    ids_path.write_text(REPAIR_RECORD_ID + "\n", encoding="utf-8")
    queue_path = output_dir / "paper_i_scaling_matrix_record_queue.tsv"
    queue_path.write_text(
        f"{REPAIR_RECORD_ID}\t{row['request_cpus']}\t{row['request_memory_mb']}\t{row['request_disk_mb']}\n",
        encoding="utf-8",
    )
    scaling._write_submit(
        submit_path=submit_path,
        batch_id=REPAIR_BATCH_ID,
        records_path=records_path,
        queue_path=queue_path,
        output_dir=output_dir,
        job_batch_name="paper-i-append-powell-cap-repair",
    )

    generated_utc = datetime.now(timezone.utc).isoformat()
    source_result_path = str(source_row["result_json_rel"])
    source_cell_manifest_path = str(source_row["cell_manifest_rel"])
    settings_diff_path = output_dir / "settings_diff.json"
    settings_diff = {
        "schema": "paper_i_scaling_matrix_single_row_settings_diff_v1",
        "source_record_id": SOURCE_RECORD_ID,
        "repair_record_id": REPAIR_RECORD_ID,
        "declared_record_science_settings_diff": science_diff,
        "approved_science_behavior_change": ["powell_maxiter_cap_policy"],
        "output_and_provenance_changes": [
            "batch_id",
            "record_id",
            "record_output_dir",
            "result/current/stdout/stderr/cell_manifest paths",
            "code_bundle path and sha256",
            "implementation_lock path and sha256",
            "exact_energy_manifest path only; content sha256 unchanged",
            "implementation_contract_id",
        ],
        "unchanged_declared_science_settings_count": sum(
            1 for key in source_science if key not in science_diff
        ),
        "source_lock_status": "not_claimed_not_evaluated",
        "status": "pass_declared_record_diff_not_source_lock",
    }
    _write_json(settings_diff_path, settings_diff)

    repair_audit_path = output_dir / "implementation_repair_audit.json"
    repair_audit = {
        "schema": "paper_i_user_approved_implementation_repair_audit_v1",
        "classification": "implementation_repair_not_sensitivity_sweep",
        "source_locked_sensitivity_claim": False,
        "source": {
            "table_label": "paper_i_scaling_matrix_appendix_scaling_row",
            "method": "Append-ADAPT",
            "regime_or_case": "hubbard_L4_scaling_strong",
            "cluster": SOURCE_CLUSTER,
            "process": SOURCE_PROCESS,
            "source_json": source_result_path,
            "source_sha256": SOURCE_RESULT_SHA256,
            "source_json_local": _repo_path(packaged_source_result),
            "source_cell_manifest": source_cell_manifest_path,
            "source_cell_manifest_sha256": SOURCE_CELL_MANIFEST_SHA256,
            "source_cell_manifest_local": _repo_path(packaged_source_cell_manifest),
            "source_command_or_manifest": _repo_path(source_records),
            "source_command_or_manifest_sha256": _sha256(source_records),
            "runner_mode": "paper_i_scaling_matrix_single_cell",
            "route_or_profile_id": str(source_row["suite_profile"]),
            "settings_hash": _canonical_sha256(source_science),
            "original_behavior": "strict_failure_v1",
            "observed_terminal_status": "completed_quality_nonpassing",
            "observed_terminal_reason": "powell_optimizer_failed",
            "observed_outer_iterations": 13,
            "observed_powell_nit": 200,
            "observed_powell_nfev": 31012,
        },
        "repair": {
            "run_class": "candidate_single_row_repair",
            "approved_behavior_field": "powell_maxiter_cap_policy",
            "approved_behavior_value": CAP_POLICY,
            "runner_mode": "paper_i_scaling_matrix_single_cell",
            "wrapper_used": False,
            "wrapper_kind": None,
            "baseline_materialization_status": (
                "proc53 TSV record plus hash-verified local result/cell bytes materialized; "
                "no current-code strict-policy anchor claimed"
            ),
            "unresolved_source_fields": [],
            "settings_changed": ["powell_maxiter_cap_policy"],
        },
        "prepared_rows": [
            {
                "powell_maxiter_cap_policy": CAP_POLICY,
                "record_id": REPAIR_RECORD_ID,
                "settings_hash": _canonical_sha256(repair_science),
                "changed_fields_vs_source": ["powell_maxiter_cap_policy"],
                "declared_record_non_changed_fields_diff": [],
                "output_and_provenance_diff_json": _repo_path(settings_diff_path),
            }
        ],
        "source_value_anchor": {
            "required_for_this_artifact": False,
            "status": "not_claimed",
            "reason": (
                "User approved a narrow implementation failure-policy repair, not a "
                "scientific sensitivity sweep. The original proc-53 result is retained as "
                "failure provenance and is not represented as a new-executable anchor."
            ),
        },
        "regression_evidence": {
            "strict_default_unchanged": {
                "test": (
                    "test/test_generic_static_adapt_variants.py::"
                    "test_scipy_optimizer_failure_stops_and_marks_quality_nonpassing[powell]"
                ),
                "expected": "strict_failure_v1 remains the default and stops on success=false",
            },
            "opt_in_cap_continues_fixed_horizon": {
                "test": (
                    "test/test_generic_static_adapt_variants.py::"
                    "test_fixed_horizon_append_accepts_only_finite_nonincreasing_powell_maxiter_caps"
                ),
                "expected": "only the opt-in capped result continues the outer horizon",
            },
            "unsafe_or_nonmaxiter_failures_rejected": {
                "test": (
                    "test/test_generic_static_adapt_variants.py::"
                    "test_powell_cap_rejects_nonfinite_increasing_or_nonmaxiter_failures"
                ),
                "expected": "nonfinite, energy-increasing, wrong-status, and wrong-message failures remain failures",
            },
            "dispatch_and_scope": {
                "test": "test/test_generic_static_powell_cap_dispatch.py",
                "expected": "the policy is explicit and generic-comparator-only",
            },
        },
        "source_evidence_fetch_status": "local_bytes_verified_and_packaged_for_transfer",
        "status": "approved_implementation_repair_prepared",
    }
    _write_json(repair_audit_path, repair_audit)

    manifest_path = output_dir / "paper_i_scaling_matrix_manifest.json"
    manifest = {
        "schema": "paper_i_scaling_matrix_single_row_repair_manifest_v1",
        "generated_utc": generated_utc,
        "batch_id": REPAIR_BATCH_ID,
        "record_count": 1,
        "record_ids": [REPAIR_RECORD_ID],
        "source_cluster": SOURCE_CLUSTER,
        "source_process": SOURCE_PROCESS,
        "source_record_id": SOURCE_RECORD_ID,
        "source_evidence": {
            "result_json": _repo_path(packaged_source_result),
            "result_json_sha256": SOURCE_RESULT_SHA256,
            "cell_manifest": _repo_path(packaged_source_cell_manifest),
            "cell_manifest_sha256": SOURCE_CELL_MANIFEST_SHA256,
            "status": "local_bytes_verified_and_packaged_for_transfer",
        },
        "records_path": _repo_path(records_path),
        "record_ids_path": _repo_path(ids_path),
        "record_queue_path": _repo_path(queue_path),
        "submit_path": _repo_path(submit_path),
        "settings_diff": _repo_path(settings_diff_path),
        "implementation_repair_audit": _repo_path(repair_audit_path),
        "exact_energy_manifest": _repo_path(exact_manifest),
        "exact_energy_manifest_sha256": exact_manifest_sha256,
        "code_bundle": code_bundle,
        "implementation_lock": _repo_path(implementation_lock_path),
        "implementation_lock_sha256": implementation_lock_sha256,
        "approved_behavior_change": {
            "powell_maxiter_cap_policy": CAP_POLICY,
            "scope": "Append-ADAPT fixed-horizon row only",
            "acceptance": (
                "SciPy Powell status=2 maxiter exhaustion only; finite objective and parameters; "
                "finite non-increasing exact refit energy within rel_tol=abs_tol=1e-10"
            ),
        },
        "submission_authority": "prepared_only_not_submitted",
        "status": "prepared_not_submitted",
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "chtc" / "phase3_optuna" / "input" / REPAIR_BATCH_ID,
    )
    parser.add_argument(
        "--submit-path",
        type=Path,
        default=ROOT / "chtc" / "phase3_optuna" / f"submit_{REPAIR_BATCH_ID}.sub",
    )
    parser.add_argument("--force", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare(
        output_dir=Path(args.output_dir),
        submit_path=Path(args.submit_path),
        force=bool(args.force),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
