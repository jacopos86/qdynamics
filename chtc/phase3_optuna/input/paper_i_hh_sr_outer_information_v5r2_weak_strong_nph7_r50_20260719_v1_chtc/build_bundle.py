#!/usr/bin/env python3
"""Build the immutable weak-strong NPH7 control/reuse CHTC pair."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pair_contract import (
    AUTHORITATIVE_JOB_REL,
    AUTHORITATIVE_JOB_SHA256,
    BUNDLE_ID,
    CONTROL_MODE,
    EXPECTED_TARGET_ROUND,
    HISTORICAL_REPAIRED_FAILURE,
    IMAGE_REL,
    IMAGE_SHA256,
    MODES,
    OUTER_PROFILE,
    PACKAGED_SOURCE_ARCHIVE,
    PACKAGED_SOURCE_AUDIT,
    PAIR_ID,
    REUSE_MODE,
    RUNTIME_ROOT,
    SOURCE_ARCHIVE_REL,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_AUDIT_REL,
    SOURCE_AUDIT_SHA256,
    SOURCE_RUNTIME_REVISION,
    bundle_dir,
    copy_exact,
    digest_jsonable,
    dump_json,
    load_json,
    option_map,
    repo_root,
    sha256,
    scientific_command_view,
    validate_job,
    validate_pair_diff,
    validate_source_lock,
)


def replace_option(argv: list[str], option: str, value: str) -> None:
    index = argv.index(option)
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ValueError(f"option has no value: {option}")
    argv[index + 1] = value


def build_job(authoritative: dict[str, Any], mode: str) -> dict[str, Any]:
    job = copy.deepcopy(authoritative)
    output = f"raw_outputs/{BUNDLE_ID}/{mode}"
    paths = {
        "output_root": output,
        "current_json": f"{output}/json/current.json",
        "ledger_json": f"{output}/json/estimator_call_ledger.json",
        "result_json": f"{output}/json/result.json",
        "execution_json": f"{output}/execution.json",
        "normalized_runtime_manifest_json": f"{output}/normalized_run_manifest.json",
        "validation_json": f"{output}/validation.json",
        "qiskit_cost_sidecar_json": f"{output}/qiskit_cost_sidecar.json",
        "repaired_terminal_checkpoint_json": (
            f"{output}/terminal_checkpoint.execution_order_repaired.json"
        ),
        "anchor_gate_json": f"{output}/anchor_gate.json",
        "wrapper_exit_receipt_json": f"{output}/wrapper_exit_receipt.json",
    }
    job["schema"] = "paper_i_sr_outer_information_matched_pair_job_v1"
    job["bundle_id"] = BUNDLE_ID
    job["batch_name"] = f"paper-i-sr-outer-v5r2-ws-nph7-r50-{mode}-20260719-v1"
    job["run_class"] = "matched_control" if mode == CONTROL_MODE else "matched_candidate"
    job["paths"] = paths
    environment = copy.deepcopy(job["environment"])
    environment.update(
        {
            "PYTHONPATH": f"/work/runtime_source/{RUNTIME_ROOT}",
            "MPLCONFIGDIR": f"{output}/cache/matplotlib",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": (
                f"{output}/cache/candidate_records"
            ),
            "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": (
                f"{output}/cache/hh_generator_registry"
            ),
            "STATIC_ADAPT_HH_POOL_CACHE_DIR": f"{output}/cache/hh_pool",
        }
    )
    job["environment"] = environment
    argv = [str(token) for token in job["command"]["argv"]]
    replace_option(
        argv,
        "--adapt-segment-id",
        f"weak_strong-sr-outer-v5r2-{mode}-r0-r50-20260719-v1",
    )
    replace_option(argv, "--adapt-current-json", paths["current_json"])
    replace_option(argv, "--adapt-estimator-call-ledger-json", paths["ledger_json"])
    replace_option(argv, "--output-json", paths["result_json"])
    if "--adapt-formal-manifold-route-profile" in argv:
        raise ValueError("authoritative control unexpectedly contains an outer profile")
    if mode == REUSE_MODE:
        insert_at = argv.index("--sr-route-profile") + 2
        argv[insert_at:insert_at] = [
            "--adapt-formal-manifold-route-profile",
            OUTER_PROFILE,
        ]
    job["command"]["argv"] = argv
    job["command"]["explicit_method_overrides"] = (
        ["adapt_max_depth"]
        if mode == CONTROL_MODE
        else ["adapt_max_depth", "adapt_formal_manifold_route_profile"]
    )
    job["command"]["method_configuration_surface"] = (
        "source_locked_sr_control_v1"
        if mode == CONTROL_MODE
        else "source_locked_sr_plus_outer_information_profile_v1"
    )
    authoritative_source_lock = copy.deepcopy(job["source_lock"])
    job["source_lock"] = {
        "worker_source_mode": "frozen_archive_nested_root_v1",
        "archive": (
            f"chtc/phase3_optuna/input/{BUNDLE_ID}/{PACKAGED_SOURCE_ARCHIVE}"
        ),
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "audit": f"chtc/phase3_optuna/input/{BUNDLE_ID}/{PACKAGED_SOURCE_AUDIT}",
        "audit_sha256": SOURCE_AUDIT_SHA256,
        "runtime_root": RUNTIME_ROOT,
        "source_runtime_revision": SOURCE_RUNTIME_REVISION,
        "image": IMAGE_REL,
        "image_sha256": IMAGE_SHA256,
        "authoritative_job_source_lock": authoritative_source_lock,
    }
    job["source_job_lock"] = {
        "path": AUTHORITATIVE_JOB_REL,
        "sha256": AUTHORITATIVE_JOB_SHA256,
        "copied_path": (
            f"chtc/phase3_optuna/input/{BUNDLE_ID}/"
            "authoritative_weak_strong_job_lock.json"
        ),
    }
    job["route_identity"]["outer_information_overlay"] = {
        "mode": mode,
        "profile": "off" if mode == CONTROL_MODE else OUTER_PROFILE,
        "selector_owner": "source_locked_sr",
        "accepted_reoptimizer_owner": "source_locked_sr_supported_fs_powell_v1",
        "structural_rollback": False,
    }
    job["pair_contract"] = {
        "pair_id": PAIR_ID,
        "mode": mode,
        "control_precedes_reuse": True,
        "sole_scientific_difference": "adapt_formal_manifold_route_profile",
        "control_gate_path": f"chtc/phase3_optuna/input/{BUNDLE_ID}/anchor_gate.control.json",
        "expected_control_job_manifest_sha256": None,
        "historical_anchor_reproduction_status": "not_claimed",
    }
    return job


def write_submit_support() -> None:
    """Ensure manually authored executable helpers are executable."""

    for name in (
        "build_bundle.py",
        "execute_source_locked_job.sh",
        "post_control_gate.py",
        "run_job.py",
        "validate_fetched.py",
    ):
        path = bundle_dir() / name
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o111)


def main() -> int:
    root = repo_root()
    bundle = bundle_dir()
    if (bundle / "anchor_gate.control.json").exists():
        raise RuntimeError(
            "refusing to rebuild over a dynamic control gate; archive or remove it first"
        )
    authoritative_path = root / AUTHORITATIVE_JOB_REL
    if sha256(authoritative_path) != AUTHORITATIVE_JOB_SHA256:
        raise ValueError("authoritative weak-strong job lock drifted")
    authoritative = load_json(authoritative_path)
    copy_exact(
        root / SOURCE_ARCHIVE_REL,
        bundle / PACKAGED_SOURCE_ARCHIVE,
        SOURCE_ARCHIVE_SHA256,
    )
    copy_exact(
        root / SOURCE_AUDIT_REL,
        bundle / PACKAGED_SOURCE_AUDIT,
        SOURCE_AUDIT_SHA256,
    )
    copy_exact(
        authoritative_path,
        bundle / "authoritative_weak_strong_job_lock.json",
        AUTHORITATIVE_JOB_SHA256,
    )
    if not (bundle / "evidence_validation.py").is_file():
        raise ValueError("bundle-local resume-aware evidence validator is missing")
    inventory = validate_source_lock(bundle)
    dump_json(bundle / "source_archive_inventory.json", inventory)

    jobs = {mode: build_job(authoritative, mode) for mode in MODES}
    control_path = bundle / "jobs/control.json"
    dump_json(control_path, jobs[CONTROL_MODE])
    control_hash = sha256(control_path)
    for mode in MODES:
        jobs[mode]["pair_contract"][
            "expected_control_job_manifest_sha256"
        ] = control_hash
    # Updating the control manifest with its own hash would be recursive.  The
    # control stores a null expectation; only the reuse manifest consumes it.
    jobs[CONTROL_MODE]["pair_contract"][
        "expected_control_job_manifest_sha256"
    ] = None
    dump_json(control_path, jobs[CONTROL_MODE])
    control_hash = sha256(control_path)
    jobs[REUSE_MODE]["pair_contract"][
        "expected_control_job_manifest_sha256"
    ] = control_hash
    reuse_path = bundle / "jobs/reuse.json"
    dump_json(reuse_path, jobs[REUSE_MODE])
    for mode, path in ((CONTROL_MODE, control_path), (REUSE_MODE, reuse_path)):
        validate_job(jobs[mode], expected_mode=mode)
        dump_json(
            bundle / f"normalized_manifests/{mode}.json",
            {
                "schema": "paper_i_sr_outer_information_pair_prelaunch_manifest_v1",
                "status": "pass",
                "pair_id": PAIR_ID,
                "mode": mode,
                "job_manifest": f"jobs/{mode}.json",
                "job_manifest_sha256": sha256(path),
                "scientific_command_view": scientific_command_view(jobs[mode]),
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "source_audit_sha256": SOURCE_AUDIT_SHA256,
                "source_runtime_revision": SOURCE_RUNTIME_REVISION,
            },
        )
    changed = validate_pair_diff(jobs[CONTROL_MODE], jobs[REUSE_MODE])
    pair_audit = {
        "schema": "paper_i_sr_outer_information_source_locked_sensitivity_audit_v1",
        "status": "pass",
        "pair_id": PAIR_ID,
        "regime": "weak_strong",
        "physics": jobs[CONTROL_MODE]["physics"],
        "target_controller_round": EXPECTED_TARGET_ROUND,
        "authoritative_job_lock": AUTHORITATIVE_JOB_REL,
        "authoritative_job_lock_sha256": AUTHORITATIVE_JOB_SHA256,
        "control_job_manifest_sha256": sha256(control_path),
        "reuse_job_manifest_sha256": sha256(reuse_path),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "changed_fields_control_to_reuse": changed,
        "control_first": True,
        "cold_disjoint_caches": True,
        "historical_anchor_reproduction_status": "not_claimed",
        "current_runtime_control_validation_status": "pending",
        "source_runtime_revision": SOURCE_RUNTIME_REVISION,
        "historical_repaired_failure": HISTORICAL_REPAIRED_FAILURE,
        "required_suppressed_eps_grad_fallback_support": True,
        "strong_weak_live_boundary_validation": {
            "status": "passed_through_round_33_nonterminal",
            "claim_scope": "repair-boundary validation only; not terminal evidence",
        },
    }
    dump_json(bundle / "source_locked_sensitivity_audit.json", pair_audit)
    dump_json(
        bundle / "weak_strong_nph7_source_lock_audit.json",
        {
            "schema": "paper_i_weak_strong_nph7_source_lock_audit_v1",
            "status": "pass",
            "regime": "weak_strong",
            "n_ph_work": 7,
            "n_ph_reference": 7,
            "same_cutoff_exact_energy": jobs[CONTROL_MODE]["physics"][
                "expected_exact_energy"
            ],
            "exact_energy_tolerance": jobs[CONTROL_MODE]["physics"][
                "exact_energy_tolerance"
            ],
            "target_controller_round": 50,
            "source_job_lock_sha256": AUTHORITATIVE_JOB_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_runtime_revision": SOURCE_RUNTIME_REVISION,
            "sr_contract_sha256": jobs[CONTROL_MODE]["route_identity"][
                "profile_contract_sha256"
            ],
            "historical_repaired_failure": HISTORICAL_REPAIRED_FAILURE,
            "suppressed_eps_grad_fallback_support": "required_and_verified_in_v5r2",
        },
    )
    dump_json(
        bundle / "submission_gate.json",
        {
            "schema": "paper_i_chtc_submission_gate_v1",
            "status": "pass",
            "submission_enabled": True,
            "source_runtime_revision": SOURCE_RUNTIME_REVISION,
            "historical_repaired_failure": HISTORICAL_REPAIRED_FAILURE,
            "local_source_archive_gate": "pass",
            "remote_image_and_quota_preflight": "required_at_later_submission",
            "control_before_reuse_gate": "enforced_by_pair_dag_post_script",
        },
    )
    generated = [
        "authoritative_weak_strong_job_lock.json",
        "source_locked_v5r2.tar.gz",
        "source_lock_audit_v5r2.json",
        "source_archive_inventory.json",
        "jobs/control.json",
        "jobs/reuse.json",
        "normalized_manifests/control.json",
        "normalized_manifests/reuse.json",
        "source_locked_sensitivity_audit.json",
        "weak_strong_nph7_source_lock_audit.json",
        "submission_gate.json",
    ]
    authored = [
        "README.md",
        "build_bundle.py",
        "execute_source_locked_job.sh",
        "evidence_validation.py",
        "pair_contract.py",
        "pair.dag",
        "post_control_gate.py",
        "run_job.py",
        "submit_control.sub",
        "submit_reuse.sub",
        "test_bundle.py",
        "validate_fetched.py",
    ]
    artifact_hashes = {
        relative: {
            "sha256": sha256(bundle / relative),
            "size_bytes": (bundle / relative).stat().st_size,
        }
        for relative in sorted(generated + authored)
        if (bundle / relative).is_file()
    }
    dump_json(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_outer_information_submission_artifact_hashes_v1",
            "status": "pass",
            "bundle_id": BUNDLE_ID,
            "artifacts": artifact_hashes,
        },
    )
    upload = sorted(
        authored
        + generated
        + ["submission_artifact_hashes.json", "bundle_manifest.json", "upload_artifact_list.txt"]
    )
    (bundle / "upload_artifact_list.txt").write_text(
        "\n".join(
            f"chtc/phase3_optuna/input/{BUNDLE_ID}/{relative}"
            for relative in upload
        )
        + "\n",
        encoding="utf-8",
    )
    dump_json(
        bundle / "bundle_manifest.json",
        {
            "schema": "paper_i_sr_outer_information_chtc_bundle_manifest_v1",
            "status": "built_local_gates_passed_remote_preflight_pending",
            "bundle_id": BUNDLE_ID,
            "pair_id": PAIR_ID,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_digest": digest_jsonable(artifact_hashes),
            "artifact_hashes": artifact_hashes,
            "source_archive_inventory_sha256": sha256(
                bundle / "source_archive_inventory.json"
            ),
            "pair_audit_sha256": sha256(
                bundle / "source_locked_sensitivity_audit.json"
            ),
            "submission_gate_sha256": sha256(bundle / "submission_gate.json"),
        },
    )
    write_submit_support()
    print(
        json.dumps(
            {
                "status": "built_local_gates_passed_remote_preflight_pending",
                "bundle_id": BUNDLE_ID,
                "changed_fields": changed,
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "source_runtime_revision": SOURCE_RUNTIME_REVISION,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
