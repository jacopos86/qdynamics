#!/usr/bin/env python3
"""Validate the explicit retention-v2 operational overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from build_operational_overlay_v2 import (  # noqa: E402
    _assert_one_member_delta,
    _derive_all_protocols,
    _draft_jobs,
)
from operational_overlay_v2_contract import (  # noqa: E402
    BUILD_RECEIPT_V2_NAME,
    BUILD_RECEIPT_V2_SCHEMA,
    CHECKPOINT_MEMBER,
    COLLISION_EVIDENCE_NAME,
    COLLISION_EVIDENCE_SCHEMA,
    EFFECTIVE_ALWAYS_FAMILY,
    EFFECTIVE_CORE_FAMILY,
    EFFECTIVE_EXECUTION_CONTRACT_SCHEMA,
    EFFECTIVE_FAMILY_BY_BASE_FAMILY,
    EFFECTIVE_SOURCES_NAME,
    EFFECTIVE_SOURCES_SCHEMA,
    EXECUTION_PLAN_V2_NAME,
    EXECUTION_PLAN_V2_SCHEMA,
    JOBS_V2_DIR,
    JOB_V2_SCHEMA,
    OVERLAY_CONTROL_FILES,
    OVERLAY_ID,
    OVERLAY_MANIFEST_NAME,
    OVERLAY_MANIFEST_SCHEMA,
    OVERLAY_PACKAGE_ID,
    PARENT_CHECKPOINT_SHA256,
    QUEUE_V2_NAME,
    REPAIRED_CHECKPOINT_SHA256,
    SOURCE_LOCK_AUDIT_V2_NAME,
    SOURCE_LOCK_AUDIT_V2_SCHEMA,
    build_effective_execution_contract,
    effective_contract_sha256,
    normalized_protocol_without_horizon,
)
from package_contract import (  # noqa: E402
    CELL_COUNT,
    COLLISION_QUEUE_RELATIVE,
    COLLISION_SUBMISSION_RECEIPT_RELATIVE,
    FRESH_COUNT,
    PACKAGE_ID,
    PACKAGE_MANIFEST_NAME,
    RESUME_COUNT,
    RESUME_INPUTS_NAME,
    SOURCE_ARCHIVES_NAME,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    PackageContractError,
    canonical_sha256,
    load_json,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)
from validate_package import validate_package  # noqa: E402


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageContractError(f"{label} must be a list.")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageContractError(message)


def _package_path(value: Any, *, label: str) -> Path:
    relative = safe_relative_path(value, label=label)
    path = PACKAGE_DIR / relative
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(
            f"{label} escaped the package."
        ) from exc
    return path


def _verify_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    path = _package_path(binding.get("path"), label=f"{label} path")
    _require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size
        == int(binding.get("size_bytes", -1))
        and sha256_file(path) == binding.get("sha256"),
        f"{label} file binding drifted.",
    )
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    observed = verify_self_digest(payload, label=label)
    _require(
        observed == binding.get("canonical_sha256"),
        f"{label} canonical binding drifted.",
    )
    return path, payload


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(
            _contains_key(item, key) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _validate_effective_sources(
    *,
    payload: Mapping[str, Any],
    source_archives: Mapping[str, Any],
) -> None:
    verify_self_digest(payload, label="effective sources")
    families = _mapping(
        payload.get("families"), label="effective source families"
    )
    _require(
        payload.get("schema") == EFFECTIVE_SOURCES_SCHEMA
        and payload.get("overlay_id") == OVERLAY_ID
        and payload.get("package_id") == OVERLAY_PACKAGE_ID
        and payload.get("base_package_id") == PACKAGE_ID
        and payload.get("status") == "passed"
        and int(payload.get("family_count", -1)) == 2
        and set(families)
        == {EFFECTIVE_CORE_FAMILY, EFFECTIVE_ALWAYS_FAMILY}
        and payload.get("base_to_effective_family")
        == EFFECTIVE_FAMILY_BY_BASE_FAMILY
        and payload.get("changed_source_members")
        == [CHECKPOINT_MEMBER]
        and payload.get("protocol_members_byte_identical") is True
        and payload.get("scientific_settings_changed") == []
        and payload.get("controller_semantics_changed") is False
        and payload.get("observation_retention_only") is True,
        "Effective-source index identity drifted.",
    )
    base_families = _mapping(
        source_archives.get("families"),
        label="base source families",
    )
    pairs = (
        (
            "core",
            base_families["stationary_core_v11"],
            families[EFFECTIVE_CORE_FAMILY],
        ),
        (
            "always",
            base_families["always_factorial_v1"],
            families[EFFECTIVE_ALWAYS_FAMILY],
        ),
    )
    for label, base, effective in pairs:
        parent_archive, _ = _verify_binding(
            _mapping(
                base.get("packaged_archive"),
                label=f"{label} base archive",
            ),
            label=f"{label} base archive",
        )
        parent_manifest, _ = _verify_binding(
            _mapping(
                base.get("packaged_manifest"),
                label=f"{label} base manifest",
            ),
            label=f"{label} base manifest",
        )
        effective_archive, _ = _verify_binding(
            _mapping(
                effective.get("effective_archive"),
                label=f"{label} effective archive",
            ),
            label=f"{label} effective archive",
        )
        effective_manifest, _ = _verify_binding(
            _mapping(
                effective.get("effective_manifest"),
                label=f"{label} effective manifest",
            ),
            label=f"{label} effective manifest",
            canonical=True,
        )
        _delta_path, delta = _verify_binding(
            _mapping(
                effective.get("delta_receipt"),
                label=f"{label} source delta",
            ),
            label=f"{label} source delta",
            canonical=True,
        )
        proof = _assert_one_member_delta(
            parent_archive=parent_archive,
            parent_manifest=parent_manifest,
            effective_archive=effective_archive,
            effective_manifest=effective_manifest,
            label=f"{label} retention-v2",
        )
        changed = _sequence(
            delta.get("changed_members"),
            label=f"{label} changed members",
        )
        _require(
            delta.get("schema")
            == "paper_i_checkpoint_sidecar_retention_source_delta_v2"
            and delta.get("parent_source_archive_sha256")
            == sha256_file(parent_archive)
            and delta.get("repaired_source_archive_sha256")
            == sha256_file(effective_archive)
            and int(delta.get("changed_member_count", -1)) == 1
            and len(changed) == 1
            and changed[0].get("path") == CHECKPOINT_MEMBER
            and changed[0].get("parent_sha256")
            == PARENT_CHECKPOINT_SHA256
            and changed[0].get("repaired_sha256")
            == REPAIRED_CHECKPOINT_SHA256
            and changed[0].get("scientific_protocol_change")
            is False
            and changed[0].get("controller_semantics_change")
            is False
            and delta.get("protocol_members_byte_identical")
            is True
            and delta.get("scientific_settings_changed") == []
            and effective.get("delta_proof") == proof
            and effective.get("parent_checkpoint_sha256")
            == PARENT_CHECKPOINT_SHA256
            and effective.get("effective_checkpoint_sha256")
            == REPAIRED_CHECKPOINT_SHA256,
            f"{label} effective-source delta drifted.",
        )


def _validate_collision_evidence(
    *, payload: Mapping[str, Any], repo_root: Path
) -> None:
    verify_self_digest(payload, label="collision evidence v2")
    receipt = _mapping(
        payload.get("submission_receipt"),
        label="collision submission receipt binding",
    )
    queue = _mapping(
        payload.get("bound_queue"),
        label="collision queue binding",
    )
    receipt_path = (
        repo_root / COLLISION_SUBMISSION_RECEIPT_RELATIVE
    )
    queue_path = repo_root / COLLISION_QUEUE_RELATIVE
    receipt_payload = load_json(
        receipt_path, label="collision submission receipt"
    )
    verify_self_digest(
        receipt_payload, label="collision submission receipt"
    )
    receipt_queue = _mapping(
        receipt_payload.get("bindings", {}).get("queue_manifest"),
        label="receipt queue binding",
    )
    _require(
        payload.get("schema") == COLLISION_EVIDENCE_SCHEMA
        and payload.get("overlay_id") == OVERLAY_ID
        and payload.get("package_id") == OVERLAY_PACKAGE_ID
        and payload.get("status") == "blocked_stale_local_evidence"
        and payload.get("blocking") is True
        and payload.get("cluster_id") == 9397758
        and payload.get("proc_ids") == list(range(9))
        and payload.get("external_state_revalidation_required")
        is True
        and payload.get("fresh_execution_requires_sealed_clearance")
        is True
        and payload.get("submission_ready") is False
        and payload.get("may_submit") is False
        and receipt.get("path")
        == COLLISION_SUBMISSION_RECEIPT_RELATIVE
        and receipt.get("sha256") == sha256_file(receipt_path)
        and receipt.get("canonical_sha256")
        == receipt_payload["sha256"]
        and queue.get("path") == COLLISION_QUEUE_RELATIVE
        and queue.get("sha256") == sha256_file(queue_path)
        and queue.get("sha256")
        == receipt_queue.get("sha256"),
        "Collision receipt/queue evidence drifted.",
    )


def validate_overlay(
    *,
    full_archive_scan: bool = True,
    rederive_protocols: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    baseline_result = validate_package(
        full_archive_scan=full_archive_scan
    )
    _require(
        baseline_result.get("status")
        == "passed_inert_collision_blocked"
        and baseline_result.get("authenticated_resume_count")
        == RESUME_COUNT,
        "Baseline package validation failed.",
    )
    _require(
        not (PACKAGE_DIR / "submit.sub").exists()
        and not (PACKAGE_DIR / "authority").exists()
        and not (PACKAGE_DIR / "collision_clearance").exists(),
        "The inert overlay gained submission/authority state.",
    )
    manifest = load_json(
        PACKAGE_DIR / OVERLAY_MANIFEST_NAME,
        label="overlay manifest",
    )
    verify_self_digest(manifest, label="overlay manifest")
    _require(
        manifest.get("schema") == OVERLAY_MANIFEST_SCHEMA
        and manifest.get("overlay_id") == OVERLAY_ID
        and manifest.get("package_id") == OVERLAY_PACKAGE_ID
        and manifest.get("base_package_id") == PACKAGE_ID
        and manifest.get("status")
        == "passed_inert_collision_blocked"
        and int(manifest.get("cell_count", -1)) == CELL_COUNT
        and int(manifest.get("authenticated_resume_count", -1))
        == RESUME_COUNT
        and int(manifest.get("fresh_count", -1)) == FRESH_COUNT
        and int(manifest.get("source_horizon", -1))
        == SOURCE_HORIZON
        and int(manifest.get("target_horizon", -1))
        == TARGET_HORIZON
        and manifest.get("submit_descriptor_present") is False
        and manifest.get("authority_overlay_present") is False
        and manifest.get("collision_clearance_overlay_present")
        is False
        and manifest.get("execution_authorized") is False
        and manifest.get("submission_authorized") is False
        and manifest.get("submission_ready") is False
        and manifest.get("submitted") is False
        and manifest.get("remote_stage") is False
        and manifest.get("condor_submit") is False,
        "Overlay manifest inert identity drifted.",
    )

    base_manifest_binding = _mapping(
        manifest.get("base_package_manifest"),
        label="base package binding",
    )
    _base_path, base_manifest = _verify_binding(
        base_manifest_binding,
        label="base package manifest",
        canonical=True,
    )
    resume_binding = _mapping(
        manifest.get("immutable_resume_inputs"),
        label="immutable resume-input binding",
    )
    _resume_path, resume_inputs = _verify_binding(
        resume_binding,
        label="immutable resume inputs",
        canonical=True,
    )
    _require(
        base_manifest.get("package_id") == PACKAGE_ID
        and resume_binding.get("full_hash_and_member_scan_passed")
        is True
        and resume_binding.get("reused_without_copy_or_mutation")
        is True
        and resume_inputs.get("resume_cell_count") == RESUME_COUNT
        and resume_binding.get("canonical_sha256")
        == base_manifest["resume_inputs"]["sha256"],
        "Immutable compact-input reuse binding drifted.",
    )

    document_bindings = {
        "effective_sources": (
            EFFECTIVE_SOURCES_NAME,
            "effective sources",
        ),
        "collision_evidence": (
            COLLISION_EVIDENCE_NAME,
            "collision evidence",
        ),
        "source_lock_audit": (
            SOURCE_LOCK_AUDIT_V2_NAME,
            "source-lock audit v2",
        ),
        "execution_plan": (
            EXECUTION_PLAN_V2_NAME,
            "execution plan v2",
        ),
        "build_receipt": (
            BUILD_RECEIPT_V2_NAME,
            "overlay build receipt",
        ),
    }
    documents: dict[str, dict[str, Any]] = {}
    for key, (name, label) in document_bindings.items():
        binding = _mapping(
            manifest.get(key), label=f"{label} binding"
        )
        _require(
            binding.get("path") == name,
            f"{label} path drifted.",
        )
        _path, payload = _verify_binding(
            binding, label=label, canonical=True
        )
        assert payload is not None
        documents[key] = payload

    source_archives = load_json(
        PACKAGE_DIR / SOURCE_ARCHIVES_NAME,
        label="base source archives",
    )
    verify_self_digest(source_archives, label="base source archives")
    _validate_effective_sources(
        payload=documents["effective_sources"],
        source_archives=source_archives,
    )
    _validate_collision_evidence(
        payload=documents["collision_evidence"],
        repo_root=repo_root,
    )

    audit = documents["source_lock_audit"]
    verify_self_digest(audit, label="source-lock audit v2")
    audit_rows = _sequence(
        audit.get("planned_rows"), label="audit rows"
    )
    _require(
        audit.get("schema") == SOURCE_LOCK_AUDIT_V2_SCHEMA
        and audit.get("overlay_id") == OVERLAY_ID
        and audit.get("package_id") == OVERLAY_PACKAGE_ID
        and len(audit_rows) == CELL_COUNT
        and audit.get("sweep", {}).get("settings_changed")
        == ["maximum_controller_rounds"]
        and audit.get("sweep", {}).get(
            "scientific_settings_changed_by_operational_overlay"
        )
        == []
        and audit.get("anchor", {}).get(
            "authenticated_checkpoint_metadata_count"
        )
        == RESUME_COUNT
        and audit.get("anchor", {}).get(
            "blocked_fresh_anchor_count"
        )
        == FRESH_COUNT
        and audit.get("anchor", {}).get(
            "operator_sequence_match_claim_count"
        )
        == 0
        and all(
            row.get("anchor", {}).get(
                "operator_sequence_match_claimed"
            )
            is False
            and row.get("anchor", {}).get(
                "operator_sequence_digest_available"
            )
            is False
            and "operator_sequence_match"
            not in row.get("anchor", {})
            for row in audit_rows
            if isinstance(row, Mapping)
        ),
        "Source-lock audit v2 drifted.",
    )

    plan = documents["execution_plan"]
    verify_self_digest(plan, label="execution plan v2")
    contracts_index = _mapping(
        plan.get("effective_execution_contracts"),
        label="effective contract index",
    )
    _require(
        plan.get("schema") == EXECUTION_PLAN_V2_SCHEMA
        and plan.get("overlay_id") == OVERLAY_ID
        and plan.get("package_id") == OVERLAY_PACKAGE_ID
        and plan.get("only_scientific_change")
        == "maximum_controller_rounds_50_to_70"
        and plan.get("operational_overlay", {}).get(
            "changed_source_members"
        )
        == [CHECKPOINT_MEMBER]
        and plan.get("operational_overlay", {}).get(
            "scientific_settings_changed"
        )
        == []
        and plan.get("submission_ready") is False
        and plan.get("submitted") is False
        and len(plan.get("execution_ids", [])) == CELL_COUNT
        and len(contracts_index) == CELL_COUNT,
        "Execution plan v2 drifted.",
    )

    build_receipt = documents["build_receipt"]
    verify_self_digest(build_receipt, label="overlay build receipt")
    _require(
        build_receipt.get("schema") == BUILD_RECEIPT_V2_SCHEMA
        and build_receipt.get("overlay_id") == OVERLAY_ID
        and build_receipt.get("status")
        == "passed_inert_collision_blocked"
        and build_receipt.get(
            "immutable_compact_resume_inputs", {}
        ).get("full_hash_and_member_scan_passed")
        is True
        and build_receipt.get(
            "immutable_compact_resume_inputs", {}
        ).get("reused_without_copy_or_mutation")
        is True
        and build_receipt.get("changed_source_members")
        == [CHECKPOINT_MEMBER]
        and build_receipt.get("scientific_settings_changed") == []
        and build_receipt.get("derived_protocol_count")
        == CELL_COUNT
        and build_receipt.get(
            "effective_execution_contract_count"
        )
        == CELL_COUNT
        and build_receipt.get("settings_hash_removed") is True
        and build_receipt.get(
            "separate_scientific_and_operational_hashes"
        )
        is True
        and build_receipt.get(
            "operator_sequence_match_claim_count"
        )
        == 0,
        "Overlay build receipt drifted.",
    )

    job_bindings = _sequence(
        manifest.get("jobs"), label="v2 job bindings"
    )
    jobs_by_id = {
        str(row["execution_id"]): row
        for row in job_bindings
        if isinstance(row, Mapping)
    }
    _require(
        len(job_bindings) == CELL_COUNT
        and set(jobs_by_id) == set(plan["execution_ids"])
        == set(contracts_index),
        "V2 job matrix drifted.",
    )
    jobs: dict[str, dict[str, Any]] = {}
    resume_count = 0
    fresh_count = 0
    for execution_id in plan["execution_ids"]:
        binding = _mapping(
            jobs_by_id[execution_id],
            label=f"{execution_id} v2 job binding",
        )
        path, _ = _verify_binding(
            binding, label=f"{execution_id} v2 job"
        )
        _require(
            path
            == PACKAGE_DIR
            / JOBS_V2_DIR
            / f"{execution_id}.json",
            f"{execution_id} v2 job path drifted.",
        )
        job = load_json(path, label=f"{execution_id} v2 job")
        job_digest = verify_self_digest(
            job, label=f"{execution_id} v2 job"
        )
        contract = _mapping(
            job.get("effective_execution_contract"),
            label=f"{execution_id} effective contract",
        )
        contract_digest = effective_contract_sha256(contract)
        derived = _mapping(
            contract.get("scientific_settings", {}).get(
                "derived_protocol_payload"
            ),
            label=f"{execution_id} derived protocol",
        )
        source_protocol_path = (
            repo_root
            / safe_relative_path(
                job["source_protocol"]["path"],
                label=f"{execution_id} source protocol path",
            )
        )
        source_protocol = load_json(
            source_protocol_path,
            label=f"{execution_id} source protocol",
        )
        rebuilt = build_effective_execution_contract(
            job=job, derived_protocol_payload=derived
        )
        mode = job.get("execution_mode")
        if mode == "authenticated_resume_50_to_70":
            resume_count += 1
        elif mode == "fresh_0_to_70":
            fresh_count += 1
        else:
            raise PackageContractError(
                f"{execution_id} execution mode drifted."
            )
        _require(
            job.get("schema") == JOB_V2_SCHEMA
            and job.get("package_id") == OVERLAY_PACKAGE_ID
            and job.get("overlay_id") == OVERLAY_ID
            and binding.get("canonical_sha256") == job_digest
            and job.get("base_package_manifest_sha256")
            == base_manifest["sha256"]
            and job.get("effective_sources_sha256")
            == documents["effective_sources"]["sha256"]
            and job.get("execution_plan_sha256") == plan["sha256"]
            and job.get("source_lock_audit_sha256")
            == audit["sha256"]
            and job.get("collision_evidence_sha256")
            == documents["collision_evidence"]["sha256"]
            and job.get("effective_execution_contract_sha256")
            == contract_digest
            == contracts_index[execution_id]["sha256"]
            and job.get("scientific_settings_sha256")
            == contract["scientific_settings_sha256"]
            == contracts_index[execution_id][
                "scientific_settings_sha256"
            ]
            and job.get("operational_settings_sha256")
            == contract["operational_settings_sha256"]
            == contracts_index[execution_id][
                "operational_settings_sha256"
            ]
            and rebuilt == contract
            and contract.get("schema")
            == EFFECTIVE_EXECUTION_CONTRACT_SCHEMA
            and normalized_protocol_without_horizon(
                source_protocol
            )
            == normalized_protocol_without_horizon(derived)
            and not _contains_key(job, "settings_hash")
            and job.get("execution_authorized") is False
            and job.get("submission_authorized") is False
            and job.get("submission_ready") is False
            and job.get("submitted") is False
            and job.get("collision_clearance_required")
            is (mode == "fresh_0_to_70"),
            f"{execution_id} truthful runtime contract drifted.",
        )
        jobs[execution_id] = job
    _require(
        resume_count == RESUME_COUNT
        and fresh_count == FRESH_COUNT,
        "V2 resume/fresh counts drifted.",
    )

    queue_binding = _mapping(
        manifest.get("queue"), label="v2 queue binding"
    )
    queue_path, _ = _verify_binding(
        queue_binding, label="v2 planning queue"
    )
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    _require(
        queue_binding.get("path") == QUEUE_V2_NAME
        and queue_binding.get("kind")
        == "inert_planning_queue_not_condor_queue"
        and queue_binding.get("row_count") == CELL_COUNT
        and queue_binding.get("column_count") == 8
        and len(lines) == CELL_COUNT,
        "V2 planning queue binding drifted.",
    )
    for line, execution_id in zip(lines, plan["execution_ids"]):
        fields = line.split("\t")
        job = jobs[execution_id]
        expected = [
            execution_id,
            str(job["execution_mode"]),
            str(job["collision_status"]),
            str(job["effective_source_family"]),
            str(job["resources"]["request_cpus"]),
            str(job["resources"]["request_memory_mb"]),
            str(job["resources"]["request_disk_mb"]),
            str(job["resources"]["max_runtime_seconds"]),
        ]
        _require(
            fields == expected,
            f"{execution_id} queue resources/runtime drifted.",
        )

    control_rows = {
        str(row["path"]): row
        for row in _sequence(
            manifest.get("control_files"),
            label="overlay control bindings",
        )
        if isinstance(row, Mapping)
    }
    _require(
        set(control_rows) == set(OVERLAY_CONTROL_FILES),
        "Overlay control-file closure drifted.",
    )
    for name in OVERLAY_CONTROL_FILES:
        _verify_binding(
            _mapping(
                control_rows[name], label=f"{name} binding"
            ),
            label=name,
        )

    if rederive_protocols:
        drafts = _draft_jobs(
            baseline_manifest=base_manifest,
            effective_sources=documents["effective_sources"],
        )
        observed = _derive_all_protocols(drafts)
        for execution_id, row in observed.items():
            expected = jobs[execution_id][
                "effective_execution_contract"
            ]["scientific_settings"]["derived_protocol_payload"]
            _require(
                row["protocol"] == expected,
                f"{execution_id} runtime rederivation drifted.",
            )

    return {
        "status": "passed_inert_collision_blocked",
        "overlay_id": OVERLAY_ID,
        "package_id": OVERLAY_PACKAGE_ID,
        "cell_count": CELL_COUNT,
        "authenticated_resume_count": RESUME_COUNT,
        "fresh_count": FRESH_COUNT,
        "effective_source_family_count": 2,
        "changed_source_members": [CHECKPOINT_MEMBER],
        "immutable_compact_resume_inputs": True,
        "full_archive_scan": full_archive_scan,
        "all_protocols_rederived": rederive_protocols,
        "settings_hash_removed": True,
        "operator_sequence_match_claim_count": 0,
        "execution_authorized": False,
        "submission_authorized": False,
        "submission_ready": False,
        "submitted": False,
        "overlay_manifest_sha256": manifest["sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip the second full compact-archive payload scan.",
    )
    parser.add_argument(
        "--rederive-protocols",
        action="store_true",
        help="Rebuild all 36 exact-source r70 protocols before passing.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = validate_overlay(
            full_archive_scan=not args.metadata_only,
            rederive_protocols=args.rederive_protocols,
        )
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
