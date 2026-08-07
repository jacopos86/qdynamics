#!/usr/bin/env python3
"""Write the disabled, archive-free main SR-SNAKE pre-freeze manifests.

This helper never snapshots source, creates an archive, executes a scientific
job, or submits to CHTC.  It exists so the six generated job/normalized
manifests remain mechanically synchronized while final source hashes are still
unknown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_bundle as spec


BUNDLE = Path(__file__).resolve().parent
TODO_SHA = "TODO_FINAL_SOURCE_ARCHIVE_SHA256"


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def environment(paths: dict[str, str]) -> dict[str, str]:
    root = paths["output_root"]
    return {
        "PYTHONPATH": "/work",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": f"{root}/cache/matplotlib",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": (
            f"{root}/cache/candidate_records"
        ),
        "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": (
            f"{root}/cache/hh_generator_registry"
        ),
        "STATIC_ADAPT_HH_POOL_CACHE": "disk",
        "STATIC_ADAPT_HH_POOL_CACHE_DIR": f"{root}/cache/hh_pool",
        "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "exact",
    }


def main() -> int:
    if spec.SOURCE_FREEZE_COMPLETE or spec.SUBMISSION_ENABLED:
        raise RuntimeError("prefreeze helper requires both freeze and submission disabled")
    template = json.loads((BUNDLE / "jobs/strong_weak_u8.json").read_text())
    contract = template["route_identity"]["profile_contract"]
    queue: list[str] = []
    jobs: list[str] = []
    normalized: list[str] = []
    physics_rows: list[dict[str, Any]] = []
    base = BUNDLE.relative_to(spec.REPO).as_posix()
    for row in spec.REGIMES:
        slug = str(row["slug"])
        argv, paths = spec.job_command(row)
        env = environment(paths)
        source_lock = {
            "prefreeze_placeholder": True,
            "git_commit": spec.EXPECTED_HEAD,
            "git_tree": spec.EXPECTED_TREE,
            "source_archive": f"{base}/source_locked.tar.gz",
            "source_archive_sha256": TODO_SHA,
            "source_archive_manifest": f"{base}/source_archive_manifest.json",
            "source_archive_manifest_sha256": "TODO_FINAL_SOURCE_ARCHIVE_MANIFEST_SHA256",
            "source_revision_manifest": f"{base}/source_revision_manifest.json",
            "source_revision_manifest_sha256": "TODO_FINAL_SOURCE_REVISION_MANIFEST_SHA256",
            "physics_reference_lock": f"{base}/physics_and_exact_reference_lock.json",
            "physics_reference_lock_sha256": "TODO_FINAL_PHYSICS_LOCK_SHA256",
            "worker_source_mode": "pending_hash_locked_source_freeze_v1",
            "source_inventory": "TODO_FINAL_COMPLETE_PER_FILE_SHA256_INVENTORY",
            "historical_manifest": (
                spec.HISTORICAL_MANIFEST_ROOT / f"{slug}.json"
            ).as_posix(),
            "historical_manifest_sha256": str(row["manifest_sha256"]),
            "historical_result": spec.source_result_path(slug).as_posix(),
            "historical_result_sha256": str(row["result_sha256"]),
        }
        physics = {
            "problem": "hh",
            "L": 2,
            "ordering": "blocked",
            "boundary": "open",
            "t": 1.0,
            "dv": 0.0,
            "omega0": 1.0,
            "u_over_t": float(row["u"]),
            "lambda": float(row["lambda"]),
            "g_ep": float(row["g_ep"]),
            "g_ep_decimal_12": str(row["g_ep"]),
            "n_ph_work": int(row["n_ph"]),
            "n_ph_reference": int(row["n_ph"]),
            "same_cutoff_reference": True,
            "expected_exact_energy": float(row["exact_energy"]),
            "expected_exact_energy_decimal": str(row["exact_energy_decimal"]),
            "exact_energy_tolerance": 1.0e-12,
        }
        target = int(row["target_round"])
        segment = {
            "source_controller_round": 0,
            "source_depth": 0,
            "target_controller_round": target,
            "target_depth": target,
            "max_new_admissions": target,
            "future_continuation_required_after_validation": False,
            "future_continuation_target": None,
            "terminal_qiskit_sidecar_outer_iteration": target,
            "terminal_qiskit_sidecar_required": True,
            "terminal_qiskit_checkpoint_order_policy": (
                "repair_permutation_only_execution_order_fail_closed_v1"
            ),
            "post_run_projector_fidelity_required": True,
            "post_run_projector_fidelity_policy": spec.FIDELITY_POLICY,
        }
        route_identity = {
            "family": "singleton_response_snake",
            "profile_request": spec.PROFILE_REQUEST,
            "profile_resolved": spec.PROFILE_RESOLVED,
            "profile_contract_sha256": spec.PROFILE_CONTRACT_SHA256,
            "profile_contract": contract,
            "phase12_energy_model_contract": {
                "phase1_energy_model": spec.PHASE1_ENERGY_MODEL,
                "phase2_curvature_policy": spec.PHASE2_CURVATURE_POLICY,
                "phase2_cheap_curvature_proxy_policy": (
                    spec.PHASE2_CHEAP_CURVATURE_PROXY_POLICY
                ),
                "lambda_f_proxy_flags_forbidden": True,
                "missing_curvature_failure_policy": "abort_run_v1",
            },
        }
        evidence = {
            "exact_s_alg_ledger_closure_required": True,
            "active_prefix_estimator_receipt_each_round_required": True,
            "terminal_estimator_closure_receipt_required": True,
            "fallback_telemetry_required": True,
            "full_active_plus_singleton_response_each_round_required": True,
            "full_accepted_refit_each_round_required": True,
            "symmetry_and_padding_leakage_gate_required": True,
            "exact_round_50_horizon_required": True,
            "post_run_projector_fidelity": spec.FIDELITY_POLICY,
        }
        job = {
            "schema": "paper_i_hh_sr_main_r50_candidate_job_v1",
            "prefreeze_status": "blocked_pending_final_source_hashes",
            "bundle_id": spec.BUNDLE_ID,
            "batch_name": spec.BATCH_NAME,
            "run_class": "main_route_matrix",
            "regime_slug": slug,
            "route_identity": route_identity,
            "physics": physics,
            "segment": segment,
            "command": {
                "argv": argv,
                "method_configuration_surface": (
                    "sr_route_profile_plus_exact_round50_horizon"
                ),
                "explicit_method_overrides": ["adapt_max_depth"],
                "manual_exact_reference_override": False,
            },
            "environment": env,
            "cache_policy": spec.RUN_CACHE_POLICY,
            "evidence_requirements": evidence,
            "paths": paths,
            "source_lock": source_lock,
            "resource_request": {
                "cpus": 4,
                "memory_mb": int(row["memory_mb"]),
                "disk_mb": int(row["disk_mb"]),
                "max_runtime_s": 259200,
            },
        }
        job_path = BUNDLE / "jobs" / f"{slug}.json"
        normalized_path = BUNDLE / "normalized_manifests" / f"{slug}.json"
        dump(job_path, job)
        dump(normalized_path, {
            "schema": "paper_i_hh_sr_main_r50_normalized_prefreeze_manifest_v1",
            "prefreeze_status": "blocked_pending_final_source_hashes",
            "bundle_id": spec.BUNDLE_ID,
            "regime_slug": slug,
            "route_identity": route_identity,
            "physics": physics,
            "segment": segment,
            "command_argv": argv,
            "environment": env,
            "cache_policy": spec.RUN_CACHE_POLICY,
            "evidence_requirements": evidence,
            "source_lock": source_lock,
            "resource_request": job["resource_request"],
        })
        jobs.append(job_path.relative_to(spec.REPO).as_posix())
        normalized.append(normalized_path.relative_to(spec.REPO).as_posix())
        queue.append("\t".join((
            slug,
            job_path.relative_to(spec.REPO).as_posix(),
            normalized_path.relative_to(spec.REPO).as_posix(),
            str(row["memory_mb"]),
            str(row["disk_mb"]),
        )))
        physics_rows.append({
            "regime_slug": slug,
            "u_over_t_decimal": str(row["u"]),
            "lambda_decimal": str(row["lambda"]),
            "g_ep_decimal_12": str(row["g_ep"]),
            "n_ph_work": int(row["n_ph"]),
            "n_ph_reference": int(row["n_ph"]),
            "same_cutoff_reference": True,
            "expected_exact_energy_decimal": str(row["exact_energy_decimal"]),
            "target_controller_round": target,
            "target_depth": target,
        })
    (BUNDLE / "queue.tsv").write_text("\n".join(queue) + "\n", encoding="utf-8")
    (BUNDLE / "submit.sub").write_text(spec.submit_text(TODO_SHA), encoding="utf-8")
    dump(BUNDLE / "physics_and_exact_reference_lock.json", {
        "schema": "paper_i_hh_main_sr_round50_physics_reference_lock_v1",
        "status": "locked_pending_source_freeze",
        "cutoff_policy": "same_working_and_reference_cutoff_v1",
        "g_ep_precision": "12_digits_after_decimal",
        "rows": physics_rows,
    })
    dump(BUNDLE / "bundle_manifest.json", {
        "schema": "paper_i_hh_sr_main_r50_prefreeze_bundle_manifest_v1",
        "status": "blocked_pending_final_source_hashes",
        "submission_enabled": False,
        "source_freeze_complete": False,
        "bundle_id": spec.BUNDLE_ID,
        "batch_name": spec.BATCH_NAME,
        "profile_request": spec.PROFILE_REQUEST,
        "profile_resolved": spec.PROFILE_RESOLVED,
        "profile_contract_sha256": spec.PROFILE_CONTRACT_SHA256,
        "all_six_fresh_round_zero_to_50": True,
        "cache_policy": spec.RUN_CACHE_POLICY,
        "post_run_projector_fidelity_policy": spec.FIDELITY_POLICY,
        "jobs": jobs,
        "normalized_manifests": normalized,
        "required_freeze_placeholders": [
            spec.EXPECTED_HEAD,
            spec.EXPECTED_TREE,
            TODO_SHA,
            "TODO_FINAL_SOURCE_ARCHIVE_MANIFEST_SHA256",
            "TODO_FINAL_SOURCE_REVISION_MANIFEST_SHA256",
            "TODO_FINAL_PHYSICS_LOCK_SHA256",
            "TODO_FINAL_COMPLETE_PER_FILE_SHA256_INVENTORY",
            "CRITICAL_SOURCE_SHA256",
            "NONSCIENTIFIC_ARCHIVE_OVERLAYS",
            "REQUIRED_UNTRACKED_SOURCE_MODULES",
        ],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
