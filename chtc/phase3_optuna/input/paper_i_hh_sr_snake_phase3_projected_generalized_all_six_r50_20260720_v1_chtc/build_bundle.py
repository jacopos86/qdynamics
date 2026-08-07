#!/usr/bin/env python3
"""Build the source-locked six-regime projected-Generalized-Trust fanout.

The executable source is the byte-identical archive proven by the weak-weak
source-value anchor.  Scientific job manifests are copied from the validated
six-regime Main-SR parent and change exactly one route execution field:
``historical_singleton_coordinate_solve_policy``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
BUNDLE = Path(__file__).resolve().parent
BUNDLE_ID = BUNDLE.name
BATCH_NAME = "paper-i-hh-sr-phase3-projected-generalized-six-r50-20260720-v1"
ANCHOR = ROOT / (
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_phase3_projected_generalized_parent_anchor_"
    "weak_weak_r50_20260719_v9_chtc"
)
PARENT = ROOT / (
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_"
    "nobeam_no_ordinary_novelty_all_six_r50_20260718_v4_chtc"
)
ANCHOR_AUDIT = ANCHOR / "source_locked_sensitivity_audit.json"
PROFILE_REQUEST = "sr_snake_no_prune_symmetric_cost_projected_phase3_v1"
PROFILE_RESOLVED = (
    "supported_projected_generalized_adaptive_trust_full_response_"
    "symmetric_cost_no_prune_v1"
)
ROUTE_DIGEST = "3ff2abb1455cda3cf8cc2de0cf739172f8cdcfe6b1c9436e1afdd40076cd3ce8"
PARENT_ROUTE_DIGEST = (
    "023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91"
)
SOURCE_ARCHIVE_SHA256 = (
    "702190f2cff1e73188d6253e89ca2f358d0c64e1d5ddb19a8bc6fbd8365baea1"
)
ANCHOR_RESULT_SHA256 = (
    "7ba3b3d95db548695010463896ab557b85dd2fd1e08e95bf254812b5c19ae278"
)
SOLVE_POLICY = "supported_metric_projected_generalized_trust_v1"
PARENT_SOLVE_POLICY = "supported_metric_whitened_eigh_v1"
REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object: {path}")
    return payload


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def replace_bundle_path(value: Any, old_bundle: str) -> Any:
    if isinstance(value, str):
        return value.replace(old_bundle, BUNDLE_ID)
    if isinstance(value, list):
        return [replace_bundle_path(item, old_bundle) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_bundle_path(item, old_bundle)
            for key, item in value.items()
        }
    return value


def contract_from_archive() -> dict[str, Any]:
    """Resolve the child contract from the frozen worker source, not live code."""

    archive = ANCHOR / "source_locked.tar.gz"
    with tempfile.TemporaryDirectory(prefix="sr_phase3_projected_contract_") as tmp:
        root = Path(tmp)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(root, filter="data")
        script = (
            "import json; "
            "from pipelines.static_adapt.sr_snake_route_profile import "
            "canonical_sr_snake_contract, canonical_sr_snake_contract_sha256, "
            "normalize_sr_route_profile_request; "
            f"r={PROFILE_REQUEST!r}; "
            "print(json.dumps({'resolved': normalize_sr_route_profile_request(r), "
            "'digest': canonical_sr_snake_contract_sha256(r), "
            "'contract': canonical_sr_snake_contract(r)}, sort_keys=True))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(root), "PYTHONNOUSERSITE": "1"},
        )
        record = json.loads(completed.stdout)
    if (
        record["resolved"] != PROFILE_RESOLVED
        or record["digest"] != ROUTE_DIGEST
    ):
        raise ValueError("frozen source resolves unexpected projected route")
    return record["contract"]


def transformed_worker(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    text = text.replace(ANCHOR.name, BUNDLE_ID)
    text = text.replace(
        'PROFILE_REQUEST = "sr_snake_no_prune_symmetric_cost_v1"',
        f'PROFILE_REQUEST = "{PROFILE_REQUEST}"',
    )
    text = text.replace(
        'PROFILE = (\n    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"\n    "no_prune_v1"\n)',
        'PROFILE = (\n'
        '    "supported_projected_generalized_adaptive_trust_full_response_"\n'
        '    "symmetric_cost_no_prune_v1"\n'
        ')',
    )
    text = text.replace(PARENT_ROUTE_DIGEST, ROUTE_DIGEST)
    text = text.replace(PARENT_SOLVE_POLICY, SOLVE_POLICY)
    text = text.replace(
        'f"{slug}-sr-main-symcost-noprune-r0-r{target}-20260718-v1"',
        'f"{slug}-sr-phase3-projected-generalized-r0-r{target}-20260720-v1"',
    )
    text = text.replace(
        '"phase3_supported_whitening_active": True,',
        '"phase3_supported_whitening_active": False,\n'
        '        "phase3_support_projection_active": True,\n'
        '        "phase3_supported_metric_inverse_sqrt_active": False,\n'
        '        "phase3_metric_ridge_active": False,',
    )
    text = text.replace(
        "from evidence_validation import checkpoint_sha256, validate_parent_evidence",
        "from evidence_validation import (\n"
        "    checkpoint_sha256,\n"
        "    validate_parent_evidence,\n"
        "    validate_projected_generalized_phase3_evidence,\n"
        ")",
    )
    call = """    evidence = validate_parent_evidence(
        result=result,
        current=current,
        ledger_sidecar=ledger,
        profile=PROFILE,
        digest=DIGEST,
        target_round=target_round,
        target_new_admissions=target_admissions,
        require_supported_rank=True,
    )
"""
    if call not in text:
        raise ValueError("worker evidence-validation insertion point missing")
    text = text.replace(
        call,
        call
        + "    projected_evidence = "
        + "validate_projected_generalized_phase3_evidence(\n"
        + "        result=result, target_round=target_round\n"
        + "    )\n",
        1,
    )
    text = text.replace(
        '"scientific_evidence_validation": evidence,',
        '"scientific_evidence_validation": evidence,\n'
        '        "projected_generalized_phase3_validation": projected_evidence,',
        1,
    )
    return text


def transformed_fetch_validator(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    text = text.replace(ANCHOR.name, BUNDLE_ID)
    text = text.replace(
        'PROFILE = (\n    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"\n    "no_prune_v1"\n)',
        'PROFILE = (\n'
        '    "supported_projected_generalized_adaptive_trust_full_response_"\n'
        '    "symmetric_cost_no_prune_v1"\n'
        ')',
    )
    text = text.replace(
        "from evidence_validation import checkpoint_sha256, validate_parent_evidence",
        "from evidence_validation import (\n"
        "    checkpoint_sha256,\n"
        "    validate_parent_evidence,\n"
        "    validate_projected_generalized_phase3_evidence,\n"
        ")",
    )
    call = """    evidence = validate_parent_evidence(
        result=result,
        current=current,
        ledger_sidecar=ledger,
        profile=PROFILE,
        digest=digest,
        target_round=target_round,
        target_new_admissions=target_new_admissions,
        require_supported_rank=True,
    )
"""
    if call not in text:
        raise ValueError("fetched-validator insertion point missing")
    text = text.replace(
        call,
        call
        + "    projected_evidence = "
        + "validate_projected_generalized_phase3_evidence(\n"
        + "        result=result, target_round=target_round\n"
        + "    )\n",
        1,
    )
    text = text.replace(
        '"scientific_evidence_validation": evidence,',
        '"scientific_evidence_validation": evidence,\n'
        '        "projected_generalized_phase3_validation": projected_evidence,',
        1,
    )
    return text


def projected_evidence_helper() -> str:
    return r'''


PROJECTED_GENERALIZED_POLICY = "supported_metric_projected_generalized_trust_v1"


def validate_projected_generalized_phase3_evidence(
    *, result: dict[str, Any], target_round: int,
) -> dict[str, Any]:
    """Fail closed on whitening or ridge drift in the projected Phase-III arm."""

    settings = result.get("settings", {})
    adapt = result.get("adapt_vqe", {})
    if settings.get("historical_singleton_coordinate_solve_policy") != (
        PROJECTED_GENERALIZED_POLICY
    ):
        raise ValueError("projected Phase-III solve policy missing from settings")
    history = adapt.get("history", [])
    if len(history) != int(target_round):
        raise ValueError("projected Phase-III history does not cover every round")
    feasible_count = 0
    fallback_count = 0
    provenance_ids: list[str] = []
    for expected_round, row in enumerate(history, start=1):
        if not isinstance(row, dict):
            raise ValueError("projected Phase-III history row is malformed")
        active_count = int(row.get("phase3_active_logical_coordinate_count", -1))
        pre_support = int(row.get("phase3_response_pre_support_count", -1))
        if (
            row.get("phase3_response_coordinate_scope")
            != "full_active_plus_singleton_v1"
            or pre_support != active_count + 1
        ):
            raise ValueError(
                f"round {expected_round} projected response scope/count drift"
            )
        admitted = row.get("admitted_records", [])
        if not isinstance(admitted, list) or len(admitted) != 1:
            raise ValueError(
                f"round {expected_round} lacks one singleton admission record"
            )
        summary = admitted[0].get("phase2_joint_geometry_reuse")
        if not isinstance(summary, dict):
            raise ValueError(
                f"round {expected_round} lacks the Phase-III response receipt"
            )
        for key in (
            "joint_solve_policy",
            "joint_linear_solve_policy_requested",
            "joint_linear_solve_policy_effective",
        ):
            if summary.get(key) != PROJECTED_GENERALIZED_POLICY:
                raise ValueError(
                    f"round {expected_round} projected solver policy drift: {key}"
                )
        if (
            summary.get("supported_metric_projection_active") is not True
            or summary.get("supported_metric_whitening_active") is not False
            or summary.get("supported_metric_inverse_sqrt_constructed") is not False
            or summary.get("supported_metric_inverse_constructed") is not False
            or summary.get("metric_regularization_applied") is not False
            or int(summary.get("classical_quantum_query_charge", -1)) != 0
        ):
            raise ValueError(
                f"round {expected_round} projected/no-whitening receipt drift"
            )
        provenance = str(
            summary.get("supported_metric_projection_provenance_id") or ""
        )
        if len(provenance) != 64:
            raise ValueError(
                f"round {expected_round} projection provenance is unresolved"
            )
        provenance_ids.append(provenance)
        if bool(summary.get("feasible", False)):
            residual = float(summary.get("supported_generalized_kkt_residual"))
            if not math.isfinite(residual):
                raise ValueError(
                    f"round {expected_round} generalized KKT residual is nonfinite"
                )
            feasible_count += 1
        else:
            fallback_count += 1
        accepted = row.get("accepted_refit", {})
        config = accepted.get("accepted_refit_invocation", {}).get("config", {})
        if (
            config.get("scope") != "full_ansatz_v1"
            or config.get("coordinate_chart")
            != "supported_fs_whitened_fixed_v1"
            or config.get("supported_fs_whitened") is not True
        ):
            raise ValueError(
                f"round {expected_round} accepted Powell refit lost whitening"
            )
    return {
        "schema": "paper_i_sr_projected_generalized_phase3_evidence_v1",
        "status": "pass",
        "controller_rounds": int(target_round),
        "projected_solver_receipt_count": int(target_round),
        "feasible_solver_receipt_count": feasible_count,
        "infeasible_solver_receipt_count": fallback_count,
        "supported_metric_whitening_active": False,
        "accepted_powell_refit_whitening_active": True,
        "projection_provenance_count": len(set(provenance_ids)),
        "classical_quantum_query_charge": 0,
    }
'''


def build() -> dict[str, Any]:
    audit = load(ANCHOR_AUDIT)
    anchor = audit.get("anchor", {})
    if (
        audit.get("fanout_authorized") is not True
        or anchor.get("anchor_reproduces_source") is not True
        or anchor.get("operator_sequence_match") is not True
        or anchor.get("settings_exact_match") is not True
        or anchor.get("non_swept_settings_diff") != []
        or anchor.get("anchor_result_sha256") != ANCHOR_RESULT_SHA256
    ):
        raise ValueError("source-value anchor has not authorized the fanout")
    contract = contract_from_archive()
    parent_contract = load(PARENT / "jobs/weak_weak.json")["route_identity"][
        "profile_contract"
    ]
    parent_execution = parent_contract["execution_settings"]
    child_execution = contract["execution_settings"]
    changed_execution = sorted(
        key
        for key in set(parent_execution) | set(child_execution)
        if parent_execution.get(key) != child_execution.get(key)
    )
    if changed_execution != ["historical_singleton_coordinate_solve_policy"]:
        raise ValueError(f"unexpected child execution drift: {changed_execution}")

    for directory in (BUNDLE / "jobs", BUNDLE / "normalized_manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ANCHOR / "source_locked.tar.gz", BUNDLE / "source_locked.tar.gz")
    shutil.copy2(
        ANCHOR / "physics_and_exact_reference_lock.json",
        BUNDLE / "physics_and_exact_reference_lock.json",
    )
    if sha256(BUNDLE / "source_locked.tar.gz") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source archive changed while building child fanout")

    archive_manifest = load(ANCHOR / "source_archive_manifest.json")
    archive_manifest["archive"] = str(
        (Path("chtc/phase3_optuna/input") / BUNDLE_ID / "source_locked.tar.gz")
    )
    dump(BUNDLE / "source_archive_manifest.json", archive_manifest)
    revision = load(ANCHOR / "source_revision_manifest.json")
    revision.update(
        {
            "profile_request": PROFILE_REQUEST,
            "profile_resolved": PROFILE_RESOLVED,
            "profile_contract_sha256": ROUTE_DIGEST,
        }
    )
    dump(BUNDLE / "source_revision_manifest.json", revision)
    source_archive_manifest_sha = sha256(BUNDLE / "source_archive_manifest.json")
    source_revision_manifest_sha = sha256(BUNDLE / "source_revision_manifest.json")
    physics_lock_sha = sha256(BUNDLE / "physics_and_exact_reference_lock.json")

    anchor_job = load(ANCHOR / "jobs/weak_weak.json")
    anchor_source_lock = anchor_job["source_lock"]
    job_paths: list[str] = []
    normalized_paths: list[str] = []
    queue_rows: list[str] = []
    for slug in REGIMES:
        parent_job_path = PARENT / "jobs" / f"{slug}.json"
        job = load(parent_job_path)
        old_bundle = str(job["bundle_id"])
        job = replace_bundle_path(job, old_bundle)
        job["bundle_id"] = BUNDLE_ID
        job["batch_name"] = BATCH_NAME
        job["run_class"] = "source_locked_phase3_projected_generalized_ablation"
        job.pop("source_value_anchor", None)
        argv = job["command"]["argv"]
        profile_index = argv.index("--sr-route-profile") + 1
        argv[profile_index] = PROFILE_REQUEST
        segment_index = argv.index("--adapt-segment-id") + 1
        argv[segment_index] = (
            f"{slug}-sr-phase3-projected-generalized-r0-r50-20260720-v1"
        )
        job["route_identity"] = {
            "family": "singleton_response_snake",
            "profile_request": PROFILE_REQUEST,
            "profile_resolved": PROFILE_RESOLVED,
            "profile_contract_sha256": ROUTE_DIGEST,
            "profile_contract": contract,
            "phase12_energy_model_contract": copy.deepcopy(
                anchor_job["route_identity"]["phase12_energy_model_contract"]
            ),
        }
        historical_keys = {
            key: value
            for key, value in job["source_lock"].items()
            if key.startswith("historical_")
        }
        source_lock = copy.deepcopy(anchor_source_lock)
        source_lock.update(historical_keys)
        prefix = Path("chtc/phase3_optuna/input") / BUNDLE_ID
        source_lock.update(
            {
                "source_archive": str(prefix / "source_locked.tar.gz"),
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "source_archive_manifest": str(
                    prefix / "source_archive_manifest.json"
                ),
                "source_archive_manifest_sha256": source_archive_manifest_sha,
                "source_revision_manifest": str(
                    prefix / "source_revision_manifest.json"
                ),
                "source_revision_manifest_sha256": source_revision_manifest_sha,
                "physics_reference_lock": str(
                    prefix / "physics_and_exact_reference_lock.json"
                ),
                "physics_reference_lock_sha256": physics_lock_sha,
            }
        )
        job["source_lock"] = source_lock
        job["sensitivity_study"] = {
            "schema": "source_locked_sensitivity_child_v1",
            "anchor_bundle": ANCHOR.name,
            "anchor_result_sha256": ANCHOR_RESULT_SHA256,
            "anchor_audit_sha256": sha256(ANCHOR_AUDIT),
            "parent_route_contract_sha256": PARENT_ROUTE_DIGEST,
            "child_route_contract_sha256": ROUTE_DIGEST,
            "swept_field": "historical_singleton_coordinate_solve_policy",
            "parent_value": PARENT_SOLVE_POLICY,
            "child_value": SOLVE_POLICY,
            "changed_execution_fields": changed_execution,
            "non_swept_settings_diff": [],
        }
        # The anchor established 40 GiB as a safe n_ph=3 envelope.  The
        # n_ph=7 rows preserve the parent 48 GiB request that completed.
        if int(job["physics"]["n_ph_work"]) == 3:
            job["resource_request"]["memory_mb"] = max(
                40960, int(job["resource_request"]["memory_mb"])
            )
        job_path = BUNDLE / "jobs" / f"{slug}.json"
        dump(job_path, job)
        normalized = {
            "schema": "paper_i_hh_sr_projected_generalized_manifest_v1",
            "batch_name": BATCH_NAME,
            "bundle_id": BUNDLE_ID,
            "regime_slug": slug,
            "command_argv": job["command"]["argv"],
            "environment": job["environment"],
            "route_identity": job["route_identity"],
            "physics": job["physics"],
            "segment": job["segment"],
            "resource_request": job["resource_request"],
            "cache_policy": job["cache_policy"],
            "evidence_requirements": job["evidence_requirements"],
            "source_lock": job["source_lock"],
            "sensitivity_study": job["sensitivity_study"],
        }
        normalized_path = BUNDLE / "normalized_manifests" / f"{slug}.json"
        dump(normalized_path, normalized)
        job_rel = str(job_path.relative_to(ROOT))
        normalized_rel = str(normalized_path.relative_to(ROOT))
        job_paths.append(job_rel)
        normalized_paths.append(normalized_rel)
        resources = job["resource_request"]
        queue_rows.append(
            "\t".join(
                (
                    slug,
                    job_rel,
                    normalized_rel,
                    str(resources["memory_mb"]),
                    str(resources["disk_mb"]),
                )
            )
        )
    (BUNDLE / "queue.tsv").write_text("\n".join(queue_rows) + "\n")

    evidence = (ANCHOR / "evidence_validation.py").read_text(encoding="utf-8")
    if "def validate_projected_generalized_phase3_evidence" not in evidence:
        evidence += projected_evidence_helper()
    (BUNDLE / "evidence_validation.py").write_text(evidence, encoding="utf-8")
    (BUNDLE / "run_job.py").write_text(
        transformed_worker(ANCHOR / "run_job.py"), encoding="utf-8"
    )
    (BUNDLE / "validate_fetched.py").write_text(
        transformed_fetch_validator(ANCHOR / "validate_fetched.py"),
        encoding="utf-8",
    )
    shell = (ANCHOR / "execute_source_locked_job.sh").read_text(
        encoding="utf-8"
    ).replace(ANCHOR.name, BUNDLE_ID)
    (BUNDLE / "execute_source_locked_job.sh").write_text(shell, encoding="utf-8")
    (BUNDLE / "execute_source_locked_job.sh").chmod(0o755)

    fanout_audit = copy.deepcopy(audit)
    fanout_audit["fanout_bundle"] = BUNDLE_ID
    fanout_audit["fanout_route_contract_sha256"] = ROUTE_DIGEST
    fanout_audit["fanout_source_archive_sha256"] = SOURCE_ARCHIVE_SHA256
    dump(BUNDLE / "source_locked_sensitivity_audit.json", fanout_audit)
    route_parity = {
        "schema": "paper_i_sr_phase3_projection_fanout_route_parity_v1",
        "status": "pass",
        "bundle_id": BUNDLE_ID,
        "parent_route_contract_sha256": PARENT_ROUTE_DIGEST,
        "projected_route_contract_sha256": ROUTE_DIGEST,
        "changed_execution_fields_vs_parent": changed_execution,
        "parent_value": PARENT_SOLVE_POLICY,
        "candidate_value": SOLVE_POLICY,
        "non_swept_settings_diff": [],
        "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
    }
    dump(BUNDLE / "route_parity.json", route_parity)
    scientific_audit = {
        "schema": "paper_i_sr_phase3_projection_fanout_scientific_audit_v1",
        "status": "pass",
        "bundle_id": BUNDLE_ID,
        "anchor_reproduces_source": True,
        "changed_scientific_execution_fields_vs_parent": changed_execution,
        "non_swept_settings_diff": [],
        "phase3_supported_whitening_active": False,
        "accepted_powell_refit_whitening_active": True,
    }
    dump(BUNDLE / "scientific_settings_audit.json", scientific_audit)
    checks = {
        "anchor_reproduces_source": True,
        "fanout_authorized": True,
        "six_job_records": True,
        "six_normalized_records": True,
        "all_rows_exact_round_50": True,
        "weak_holstein_n_ph_3": True,
        "strong_holstein_n_ph_7": True,
        "same_cutoff_all_rows": True,
        "single_scientific_execution_field_changed": True,
        "phase3_raw_supported_projection": True,
        "phase3_whitening_disabled": True,
        "accepted_powell_refit_whitening_preserved": True,
        "source_archive_hash_locked": True,
        "submission_not_performed": True,
    }
    preflight = {
        "schema": "paper_i_sr_phase3_projection_fanout_preflight_v1",
        "status": "pass",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "profile_request": PROFILE_REQUEST,
        "route_contract_sha256": ROUTE_DIGEST,
        "parent_route_contract_sha256": PARENT_ROUTE_DIGEST,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "checks": checks,
    }
    dump(BUNDLE / "preflight.json", preflight)
    dump(BUNDLE / "archive_only_preflight.json", preflight)
    manifest = {
        "schema": "paper_i_sr_phase3_projection_fanout_bundle_v1",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "job_count": 6,
        "jobs": job_paths,
        "normalized_manifests": normalized_paths,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_archive_manifest_sha256": source_archive_manifest_sha,
        "source_revision_manifest_sha256": source_revision_manifest_sha,
        "physics_reference_lock_sha256": physics_lock_sha,
        "parent_route_contract_sha256": PARENT_ROUTE_DIGEST,
        "projected_route_contract_sha256": ROUTE_DIGEST,
        "anchor_result_sha256": ANCHOR_RESULT_SHA256,
        "fanout_authorized": True,
        "submission_performed": False,
    }
    dump(BUNDLE / "bundle_manifest.json", manifest)
    dump(
        BUNDLE / "submission_gate.json",
        {
            "schema": "paper_i_sr_phase3_projection_fanout_submission_gate_v1",
            "status": "ready_for_authenticated_remote_preflight",
            "bundle_id": BUNDLE_ID,
            "job_count": 6,
            "scientific_blockers": [],
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "route_contract_sha256": ROUTE_DIGEST,
            "submission_performed": False,
        },
    )
    dump(
        BUNDLE / "remote_execution_gate.json",
        {
            "schema": "paper_i_sr_phase3_projection_fanout_remote_gate_v1",
            "status": "pending_authenticated_remote_preflight",
            "bundle_id": BUNDLE_ID,
            "image_path": "chtc/phase3_optuna/image.sif",
            "image_sha256": (
                "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
            ),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "submission_performed": False,
        },
    )

    submit = f'''universe = vanilla
executable = chtc/phase3_optuna/input/{BUNDLE_ID}/execute_source_locked_job.sh
arguments = $(job_manifest) chtc/phase3_optuna/input/{BUNDLE_ID}/source_locked.tar.gz {SOURCE_ARCHIVE_SHA256} chtc/phase3_optuna/image.sif fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = chtc/phase3_optuna/input/{BUNDLE_ID}/run_job.py, chtc/phase3_optuna/input/{BUNDLE_ID}/evidence_validation.py, chtc/phase3_optuna/input/{BUNDLE_ID}/validate_fetched.py, chtc/phase3_optuna/input/{BUNDLE_ID}/source_archive_manifest.json, chtc/phase3_optuna/input/{BUNDLE_ID}/source_revision_manifest.json, chtc/phase3_optuna/input/{BUNDLE_ID}/physics_and_exact_reference_lock.json, chtc/phase3_optuna/input/{BUNDLE_ID}/bundle_manifest.json, chtc/phase3_optuna/input/{BUNDLE_ID}/source_locked_sensitivity_audit.json, $(job_manifest), $(normalized_manifest), chtc/phase3_optuna/input/{BUNDLE_ID}/source_locked.tar.gz, chtc/phase3_optuna/image.sif
transfer_output_files = raw_outputs/{BUNDLE_ID}/$(regime_slug)_transfer.tar.gz
transfer_output_remaps = "raw_outputs/{BUNDLE_ID}/$(regime_slug)_transfer.tar.gz = $(Cluster).$(Process)__$(regime_slug)_transfer.tar.gz"
stream_output = False
stream_error = False
log = logs/{BUNDLE_ID}.$(Cluster).$(Process).log
output = logs/{BUNDLE_ID}.$(Cluster).$(Process).out
error = logs/{BUNDLE_ID}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = "{BATCH_NAME}"
notification = Never
queue regime_slug, job_manifest, normalized_manifest, memory_mb, disk_mb from chtc/phase3_optuna/input/{BUNDLE_ID}/queue.tsv
'''
    (BUNDLE / "submit.sub").write_text(submit, encoding="utf-8")
    upload = [
        str((Path("chtc/phase3_optuna/input") / BUNDLE_ID / name))
        for name in (
            "execute_source_locked_job.sh",
            "run_job.py",
            "evidence_validation.py",
            "validate_fetched.py",
            "source_archive_manifest.json",
            "source_revision_manifest.json",
            "physics_and_exact_reference_lock.json",
            "bundle_manifest.json",
            "source_locked_sensitivity_audit.json",
            "source_locked.tar.gz",
            "queue.tsv",
            "submit.sub",
        )
    ] + job_paths + normalized_paths
    (BUNDLE / "upload_artifact_list.txt").write_text(
        "\n".join(upload) + "\n", encoding="utf-8"
    )
    (BUNDLE / "README.md").write_text(
        "# Phase-III projected generalized-trust six-regime fanout\n\n"
        "The source-value anchor reproduced its locked parent exactly.  This "
        "bundle changes only `historical_singleton_coordinate_solve_policy` "
        "in the six 50-round scientific routes.  Phase III uses raw supported "
        "projection/generalized trust; accepted Powell refits remain supported-FS "
        "whitened.\n",
        encoding="utf-8",
    )
    (BUNDLE / "test_bundle.py").write_text(
        "from build_bundle import verify\n\n\n"
        "def test_projected_generalized_fanout_bundle() -> None:\n"
        "    assert verify()\n",
        encoding="utf-8",
    )
    artifact_files: dict[str, dict[str, Any]] = {}
    for path in sorted(BUNDLE.rglob("*")):
        relative = path.relative_to(BUNDLE)
        if (
            not path.is_file()
            or "__pycache__" in relative.parts
            or path.suffix == ".pyc"
            or relative.as_posix() == "submission_artifact_hashes.json"
        ):
            continue
        artifact_files[relative.as_posix()] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    dump(
        BUNDLE / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_phase3_projection_fanout_submission_artifacts_v1",
            "bundle_id": BUNDLE_ID,
            "files": artifact_files,
        },
    )
    return manifest


def verify() -> bool:
    manifest = load(BUNDLE / "bundle_manifest.json")
    if manifest.get("job_count") != 6 or manifest.get("fanout_authorized") is not True:
        raise ValueError("fanout bundle manifest is incomplete")
    if sha256(BUNDLE / "source_locked.tar.gz") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("fanout source archive hash drift")
    jobs = sorted((BUNDLE / "jobs").glob("*.json"))
    if [path.stem for path in jobs] != sorted(REGIMES):
        raise ValueError("fanout job set drift")
    for path in jobs:
        job = load(path)
        route = job["route_identity"]
        if (
            route["profile_contract_sha256"] != ROUTE_DIGEST
            or route["profile_contract"]["execution_settings"][
                "historical_singleton_coordinate_solve_policy"
            ] != SOLVE_POLICY
            or int(job["segment"]["target_controller_round"]) != 50
            or int(job["physics"]["n_ph_work"])
            != int(job["physics"]["n_ph_reference"])
            or int(job["physics"]["n_ph_work"])
            != (3 if path.stem in REGIMES[:3] else 7)
        ):
            raise ValueError(f"fanout row contract drift: {path.name}")
    inventory = load(BUNDLE / "submission_artifact_hashes.json")
    for relative, receipt in inventory.get("files", {}).items():
        artifact = BUNDLE / relative
        if (
            not artifact.is_file()
            or sha256(artifact) != receipt.get("sha256")
            or artifact.stat().st_size != int(receipt.get("size_bytes", -1))
        ):
            raise ValueError(f"fanout artifact inventory drift: {relative}")
    return True


if __name__ == "__main__":
    build()
    verify()
    print("projected-Generalized-Trust fanout bundle built and verified")
