#!/usr/bin/env python3
"""Build the inert, source-locked six-row fresh round-100 CHTC package."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from derived_protocol import (  # noqa: E402
    activate_source_root,
    build_derived_protocol,
)
from package_contract import (  # noqa: E402
    ANCHOR_EVIDENCE_NAME,
    CAMPAIGN_ID,
    CANDIDATE_REPRESENTATIONS,
    DELTA_AUDIT_SCHEMA,
    DIRECT_EXECUTION_COUNT,
    EXECUTION_PLAN_SCHEMA,
    EXPECTED_EXECUTION_IDS,
    EXPECTED_RESOURCES_BY_NPH,
    EXPECTED_SOURCE_EXECUTION_IDS,
    GENERATED_CONTROL_FILES,
    GENERATED_DIRECTORIES,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REPO_ROOT,
    RUN_CLASS,
    SOURCE_ARCHIVE_MANIFEST_FILE_SHA256,
    SOURCE_ARCHIVE_MANIFEST_NAME,
    SOURCE_ARCHIVE_MANIFEST_SHA256,
    SOURCE_ARCHIVE_NAME,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_ARCHIVE_SIZE_BYTES,
    SOURCE_AUTHORITY_SCHEMA,
    SOURCE_AUTHORIZATION_FILE_SHA256,
    SOURCE_AUTHORIZATION_SHA256,
    SOURCE_EXECUTION_PLAN_FILE_SHA256,
    SOURCE_EXECUTION_PLAN_SHA256,
    SOURCE_FINAL_RECEIPT_FILE_SHA256,
    SOURCE_FINAL_RECEIPT_SHA256,
    SOURCE_HORIZON,
    SOURCE_PACKAGE_DIR,
    SOURCE_PACKAGE_ID,
    SOURCE_PACKAGE_RELATIVE_ROOT,
    SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
    SOURCE_PACKAGE_MANIFEST_SHA256,
    STATIC_CONTROL_FILES,
    TARGET_HORIZON,
    PackageContractError,
    _expected_artifact_paths,
    _manifest_file_binding,
    audit_anchor_payload,
    atomic_write_json,
    canonical_json_bytes,
    digested,
    load_json_object,
    sha256_file,
    validate_package,
    verify_self_digest,
)


VISIBLE_R70_ANCHOR_PATH = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_append_adapt_stationary_core12_r70_fresh_20260731_v1_chtc/"
    "anchor_evidence.json"
)
VISIBLE_R70_ANCHOR_FILE_SHA256 = (
    "6ed9e0a7f2c6df75854a5fcd584a576c1fb2ef356458029813b0f3de637e91e1"
)
VISIBLE_R70_ANCHOR_CANONICAL_SHA256 = (
    "5241f3ad71799f9ca139a0921ed4ae626c6ae396f77625626acf1f0ec98a9cef"
)


def _load_visible_r70_anchor() -> dict[str, Any]:
    if (
        not VISIBLE_R70_ANCHOR_PATH.is_file()
        or VISIBLE_R70_ANCHOR_PATH.is_symlink()
        or sha256_file(VISIBLE_R70_ANCHOR_PATH)
        != VISIBLE_R70_ANCHOR_FILE_SHA256
    ):
        raise PackageContractError("Visible round-70 Append anchor drifted.")
    payload = load_json_object(
        VISIBLE_R70_ANCHOR_PATH,
        label="visible round-70 Append anchor",
    )
    verify_self_digest(payload, label="visible round-70 Append anchor")
    if payload.get("sha256") != VISIBLE_R70_ANCHOR_CANONICAL_SHA256:
        raise PackageContractError(
            "Visible round-70 Append anchor digest drifted."
        )
    return payload


def _assert_file_anchor(
    path: Path,
    *,
    file_sha256: str,
    canonical_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        raise PackageContractError(f"Source authority file is missing: {path}")
    if sha256_file(path) != file_sha256:
        raise PackageContractError(f"Source authority bytes drifted: {path}")
    if canonical_sha256 is None:
        return None
    payload = load_json_object(path, label=f"source {path.name}")
    verify_self_digest(payload, label=f"source {path.name}")
    if payload["sha256"] != canonical_sha256:
        raise PackageContractError(
            f"Source authority digest drifted: {path}"
        )
    return payload


def _validate_source_package() -> dict[str, Any]:
    manifest = _assert_file_anchor(
        SOURCE_PACKAGE_DIR / "package_manifest.json",
        file_sha256=SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
        canonical_sha256=SOURCE_PACKAGE_MANIFEST_SHA256,
    )
    plan = _assert_file_anchor(
        SOURCE_PACKAGE_DIR / "execution_plan.json",
        file_sha256=SOURCE_EXECUTION_PLAN_FILE_SHA256,
        canonical_sha256=SOURCE_EXECUTION_PLAN_SHA256,
    )
    archive_manifest = _assert_file_anchor(
        SOURCE_PACKAGE_DIR / "source_archive_manifest.json",
        file_sha256=SOURCE_ARCHIVE_MANIFEST_FILE_SHA256,
        canonical_sha256=SOURCE_ARCHIVE_MANIFEST_SHA256,
    )
    _assert_file_anchor(
        SOURCE_PACKAGE_DIR
        / "authority"
        / "core_final_publication_receipt.json",
        file_sha256=SOURCE_FINAL_RECEIPT_FILE_SHA256,
        canonical_sha256=SOURCE_FINAL_RECEIPT_SHA256,
    )
    _assert_file_anchor(
        SOURCE_PACKAGE_DIR
        / "authority"
        / "submission_authorization_receipt.json",
        file_sha256=SOURCE_AUTHORIZATION_FILE_SHA256,
        canonical_sha256=SOURCE_AUTHORIZATION_SHA256,
    )
    archive = SOURCE_PACKAGE_DIR / "source_locked.tar.gz"
    if (
        sha256_file(archive) != SOURCE_ARCHIVE_SHA256
        or archive.stat().st_size != SOURCE_ARCHIVE_SIZE_BYTES
    ):
        raise PackageContractError("Source package archive drifted.")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    # Validate an exact temporary mirror so unrelated local bytecode cache
    # contamination cannot weaken or block the sealed source-package check.
    # Every authority byte used below is still anchored to the visible v6
    # package before this mirror is made.
    with tempfile.TemporaryDirectory(
        prefix="paper_i_append_r100_source_package_validation_"
    ) as raw:
        validation_package_dir = (
            Path(raw) / SOURCE_PACKAGE_DIR.name
        )
        shutil.copytree(
            SOURCE_PACKAGE_DIR,
            validation_package_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        process = subprocess.run(
            [
                sys.executable,
                str(validation_package_dir / "validate_package.py"),
                "--require-authorization",
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    if process.returncode != 0:
        raise PackageContractError(
            "The complete round-50 source package no longer validates: "
            + process.stderr.strip()
        )
    try:
        validation = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise PackageContractError(
            "Source package validation did not emit JSON."
        ) from exc
    if (
        not isinstance(validation, dict)
        or validation.get("status") != "passed"
        or validation.get("package_manifest_sha256")
        != SOURCE_PACKAGE_MANIFEST_SHA256
        or validation.get("execution_plan_sha256")
        != SOURCE_EXECUTION_PLAN_SHA256
        or validation.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
    ):
        raise PackageContractError(
            "Source package validation authority drifted."
        )
    assert manifest is not None
    assert plan is not None
    assert archive_manifest is not None
    return {
        "manifest": manifest,
        "plan": plan,
        "archive_manifest": archive_manifest,
        "validation": validation,
    }


def _source_manifest_files(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise PackageContractError("Source package manifest has no files.")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PackageContractError("Malformed source package file row.")
        relative = str(row.get("path", ""))
        if relative in result:
            raise PackageContractError(
                f"Duplicate source package file binding: {relative}"
            )
        result[relative] = row
    return result


def _source_archive_members(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("members")
    if not isinstance(rows, list):
        raise PackageContractError("Source archive manifest has no members.")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PackageContractError("Malformed source archive member.")
        relative = str(row.get("path", ""))
        if relative in result:
            raise PackageContractError(
                f"Duplicate source archive member: {relative}"
            )
        result[relative] = row
    return result


def _base_job_binding(
    source_execution_id: str,
    *,
    manifest_files: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = f"jobs/{source_execution_id}.json"
    path = SOURCE_PACKAGE_DIR / relative
    payload = load_json_object(path, label=f"source job {source_execution_id}")
    verify_self_digest(payload, label=f"source job {source_execution_id}")
    declared = manifest_files.get(relative)
    observed = {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "executable": bool(path.stat().st_mode & 0o111),
    }
    if declared != observed:
        raise PackageContractError(
            f"Source job file binding drifted: {source_execution_id}"
        )
    if (
        payload.get("execution_id") != source_execution_id
        or payload.get("execution_entrypoint") != "run_append_adapt"
        or payload.get("run_class") != RUN_CLASS
        or payload.get("execution_authorized") is not False
        or payload.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
    ):
        raise PackageContractError(
            f"Source job semantics drifted: {source_execution_id}"
        )
    return payload, {
        "path": relative,
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
        "canonical_sha256": payload["sha256"],
    }


def _preliminary_job(
    *,
    execution_id: str,
    source_execution_id: str,
    source_job: Mapping[str, Any],
    source_job_binding: Mapping[str, Any],
    archive_members: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    route_id = execution_id.rsplit("__", 1)[-1]
    protocol = source_job.get("protocol")
    if not isinstance(protocol, Mapping):
        raise PackageContractError(
            f"Source job lacks a protocol: {source_execution_id}"
        )
    protocol_path = str(protocol.get("path", ""))
    archive_binding = archive_members.get(protocol_path)
    if (
        archive_binding is None
        or protocol.get("sha256") != archive_binding.get("sha256")
        or int(protocol.get("size_bytes", -1))
        != int(archive_binding.get("size_bytes", -2))
    ):
        raise PackageContractError(
            f"Source protocol/archive binding drifted: {source_execution_id}"
        )
    regime_id = str(source_job["regime_id"])
    nph = int(source_job["nph"])
    if (
        execution_id
        != f"r100_fresh__{regime_id}__nph{nph}__{route_id}"
        or source_job.get("route_id") != route_id
        or source_job.get("candidate_representation")
        != CANDIDATE_REPRESENTATIONS[route_id]
    ):
        raise PackageContractError(
            f"Source/new job identity drifted: {execution_id}"
        )
    return {
        "schema": JOB_SCHEMA,
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "run_class": RUN_CLASS,
        "execution_target": "chtc",
        "execution_id": execution_id,
        "cell_id": execution_id,
        "source_execution_id": source_execution_id,
        "source_job": dict(source_job_binding),
        "regime_id": regime_id,
        "nph": nph,
        "route_id": route_id,
        "candidate_representation": str(
            source_job["candidate_representation"]
        ),
        "execution_entrypoint": "run_append_adapt",
        "source_lock_id": str(source_job["source_lock_id"]),
        "source_protocol": {
            "path": protocol_path,
            "sha256": str(protocol["sha256"]),
            "size_bytes": int(protocol["size_bytes"]),
            "canonical_sha256": str(protocol["canonical_sha256"]),
        },
        "source_archive": {
            "path": SOURCE_ARCHIVE_NAME,
            "sha256": SOURCE_ARCHIVE_SHA256,
            "size_bytes": SOURCE_ARCHIVE_SIZE_BYTES,
        },
        "horizon": {
            "source": SOURCE_HORIZON,
            "target": TARGET_HORIZON,
        },
        "fresh_start_contract": {
            "kind": "fresh_start",
            "source_checkpoint_consumed": False,
            "source_result_consumed": False,
            "resume_claimed": False,
            "controller_round_origin": 0,
        },
        "artifact_paths": _expected_artifact_paths(execution_id),
        # Scheduler sizing is operational metadata. Use measured round-70
        # singleton peaks with headroom; all scientific settings remain bound
        # to the source job and derived protocol.
        "resources": dict(EXPECTED_RESOURCES_BY_NPH[nph]),
        "execution_authorized": False,
        "submission_authorized": False,
        "submission_state": "not_submitted",
    }


def _copy_source_authority_files() -> None:
    source_archive = SOURCE_PACKAGE_DIR / "source_locked.tar.gz"
    destination_archive = PACKAGE_DIR / SOURCE_ARCHIVE_NAME
    destination_manifest = PACKAGE_DIR / SOURCE_ARCHIVE_MANIFEST_NAME
    if destination_archive.exists() or destination_manifest.exists():
        raise PackageContractError(
            "Refusing to overwrite copied source authority."
        )
    shutil.copyfile(source_archive, destination_archive)
    shutil.copyfile(
        SOURCE_PACKAGE_DIR / "source_archive_manifest.json",
        destination_manifest,
    )


def _extract_source(destination: Path) -> None:
    with tarfile.open(PACKAGE_DIR / SOURCE_ARCHIVE_NAME, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")


def _control_plane_receipt() -> dict[str, Any]:
    return digested(
        {
            "schema": (
                "paper_i_append_adapt_stationary_core_r100_"
                "control_plane_v1"
            ),
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "status": "passed",
            "files": [
                _manifest_file_binding(PACKAGE_DIR, relative)
                for relative in sorted(STATIC_CONTROL_FILES)
            ],
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "remote_stage": False,
            "condor_submit": False,
        }
    )


def build_package() -> dict[str, Any]:
    if PACKAGE_DIR.name != PACKAGE_ID or REPO_ROOT != Path.cwd().resolve():
        raise PackageContractError(
            "Build from the active repository root and fixed package path."
        )
    for relative in STATIC_CONTROL_FILES:
        path = PACKAGE_DIR / relative
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Static package file is unavailable: {relative}"
            )
    generated = [
        *(PACKAGE_DIR / relative for relative in GENERATED_CONTROL_FILES),
        PACKAGE_DIR / "package_manifest.json",
        *(PACKAGE_DIR / relative for relative in GENERATED_DIRECTORIES),
    ]
    collisions = [str(path) for path in generated if path.exists()]
    if collisions:
        raise PackageContractError(
            "Refusing an in-place package rebuild: " + ", ".join(collisions)
        )

    source = _validate_source_package()
    manifest_files = _source_manifest_files(source["manifest"])
    archive_members = _source_archive_members(source["archive_manifest"])
    source_rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for execution_id, source_execution_id in zip(
        EXPECTED_EXECUTION_IDS,
        EXPECTED_SOURCE_EXECUTION_IDS,
        strict=True,
    ):
        source_job, source_job_binding = _base_job_binding(
            source_execution_id, manifest_files=manifest_files
        )
        source_rows.append(
            {
                "execution_id": execution_id,
                "source_execution_id": source_execution_id,
                "source_job": source_job_binding,
                "source_protocol": dict(source_job["protocol"]),
                "source_lock_id": source_job["source_lock_id"],
            }
        )
        jobs.append(
            _preliminary_job(
                execution_id=execution_id,
                source_execution_id=source_execution_id,
                source_job=source_job,
                source_job_binding=source_job_binding,
                archive_members=archive_members,
            )
        )
    if len(jobs) != DIRECT_EXECUTION_COUNT:
        raise PackageContractError("Append source row count drifted.")

    _copy_source_authority_files()
    control = _control_plane_receipt()
    atomic_write_json(PACKAGE_DIR / "control_plane_receipt.json", control)
    source_authority = digested(
        {
            "schema": SOURCE_AUTHORITY_SCHEMA,
            "status": "passed",
            "source_package_id": SOURCE_PACKAGE_ID,
            "source_package_manifest_sha256": (
                SOURCE_PACKAGE_MANIFEST_SHA256
            ),
            "source_package_manifest_file_sha256": (
                SOURCE_PACKAGE_MANIFEST_FILE_SHA256
            ),
            "source_execution_plan_sha256": SOURCE_EXECUTION_PLAN_SHA256,
            "source_execution_plan_file_sha256": (
                SOURCE_EXECUTION_PLAN_FILE_SHA256
            ),
            "source_archive_manifest_sha256": (
                SOURCE_ARCHIVE_MANIFEST_SHA256
            ),
            "source_archive_manifest_file_sha256": (
                SOURCE_ARCHIVE_MANIFEST_FILE_SHA256
            ),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_archive_size_bytes": SOURCE_ARCHIVE_SIZE_BYTES,
            "source_final_receipt_sha256": SOURCE_FINAL_RECEIPT_SHA256,
            "source_final_receipt_file_sha256": (
                SOURCE_FINAL_RECEIPT_FILE_SHA256
            ),
            "source_authorization_sha256": SOURCE_AUTHORIZATION_SHA256,
            "source_authorization_file_sha256": (
                SOURCE_AUTHORIZATION_FILE_SHA256
            ),
            "source_package_validation_status": "passed",
            "source_append_row_count": len(source_rows),
            "source_rows": source_rows,
            "source_role": (
                "completed_visible_round50_settings_authority_only_v1"
            ),
            "source_checkpoint_consumed": False,
            "source_result_consumed": False,
        }
    )
    atomic_write_json(PACKAGE_DIR / "source_authority.json", source_authority)
    # The original independent round-50 anchor archives were intentionally
    # cleaned after the sealed round-70 package was validated. Reuse that
    # package's exact, self-digested anchor receipt rather than reconstructing
    # or weakening the source-value proof.
    anchor_evidence = _load_visible_r70_anchor()
    atomic_write_json(
        PACKAGE_DIR / ANCHOR_EVIDENCE_NAME,
        anchor_evidence,
    )

    audit_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="paper_i_append_r100_source_"
    ) as raw:
        source_root = Path(raw)
        _extract_source(source_root)
        activate_source_root(source_root)
        for index, job in enumerate(jobs):
            protocol, _problem, row = build_derived_protocol(
                job=job,
                source_root=source_root,
                validate_entire_bundle=(index == 0),
            )
            job["derived_protocol_sha256"] = protocol.sha256
            audit_rows.append(row)
    jobs = [digested(job) for job in jobs]

    audit = digested(
        {
            "schema": DELTA_AUDIT_SCHEMA,
            "status": "pass",
            "source": {
                "table_label": (
                    "paper_i_stationary_core_conventional_append_rows"
                ),
                "method": "Append-ADAPT",
                "regime_or_case": "six_regime_singleton_matrix",
                "source_package_id": SOURCE_PACKAGE_ID,
                "source_package_manifest_sha256": (
                    SOURCE_PACKAGE_MANIFEST_SHA256
                ),
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "source_horizon": SOURCE_HORIZON,
                "source_variable_value": SOURCE_HORIZON,
                "runner_mode": "run_append_adapt",
                "route_or_profile_id": (
                    "conventional_append_single_pauli_word_v1"
                ),
                "source_row_count": DIRECT_EXECUTION_COUNT,
                "source_json": (
                    f"{anchor_evidence['source_execution']['archive']['path']}"
                    "#"
                    f"{anchor_evidence['source_execution']['result']['path']}"
                ),
                "source_sha256": anchor_evidence[
                    "source_execution"
                ]["result"]["sha256"],
                "source_command_or_manifest": (
                    f"{SOURCE_PACKAGE_RELATIVE_ROOT}/jobs/"
                    f"{anchor_evidence['execution_id']}.json"
                ),
                "source_command_or_manifest_sha256": (
                    anchor_evidence["source_execution"]["job"]["sha256"]
                ),
                "settings_hash": anchor_evidence["protocol_sha256"],
            },
            "sweep": {
                "run_class": RUN_CLASS,
                "variable": "maximum_controller_rounds",
                "grid": [TARGET_HORIZON],
                "runner_mode": "fresh_full_replay",
                "wrapper_used": False,
                "wrapper_kind": None,
                "baseline_materialization_status": "complete",
                "unresolved_source_fields": [],
                "fields_added_by_current_defaults": [],
                "settings_changed": [
                    "maximum_controller_rounds",
                    "output_identity",
                ],
            },
            "planned_rows": audit_rows,
            "anchor": audit_anchor_payload(anchor_evidence),
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
        }
    )
    atomic_write_json(PACKAGE_DIR / "horizon_delta_audit.json", audit)

    jobs_dir = PACKAGE_DIR / "jobs"
    jobs_dir.mkdir(parents=False, exist_ok=False)
    for job in jobs:
        atomic_write_json(
            jobs_dir / f"{job['execution_id']}.json", job
        )
    queue_lines = []
    for job in jobs:
        resources = job["resources"]
        relative = f"jobs/{job['execution_id']}.json"
        queue_lines.append(
            "\t".join(
                (
                    str(job["execution_id"]),
                    relative,
                    sha256_file(PACKAGE_DIR / relative),
                    SOURCE_ARCHIVE_SHA256,
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                )
            )
        )
    queue_path = PACKAGE_DIR / "queue.tsv"
    queue_path.write_bytes(("\n".join(queue_lines) + "\n").encode("utf-8"))

    plan = digested(
        {
            "schema": EXECUTION_PLAN_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "execution_target": "chtc",
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "fresh_start": True,
            "resume_claimed": False,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_authority_sha256": source_authority["sha256"],
            "anchor_evidence_sha256": anchor_evidence["sha256"],
            "horizon_delta_audit_sha256": audit["sha256"],
            "control_plane_receipt_sha256": control["sha256"],
            "direct_execution_count": DIRECT_EXECUTION_COUNT,
            "execution_ids": list(EXPECTED_EXECUTION_IDS),
            "direct_executions": [
                {
                    "execution_id": job["execution_id"],
                    "source_execution_id": job["source_execution_id"],
                    "regime_id": job["regime_id"],
                    "nph": job["nph"],
                    "route_id": job["route_id"],
                    "job_spec_path": f"jobs/{job['execution_id']}.json",
                    "job_spec_sha256": job["sha256"],
                    "derived_protocol_sha256": (
                        job["derived_protocol_sha256"]
                    ),
                }
                for job in jobs
            ],
            "activation_contract": {
                "required_before_submission": True,
                "execution_authorization_schema": (
                    "paper_i_append_adapt_stationary_core_r100_"
                    "execution_authorization_v1"
                ),
                "submit_surface_present": False,
            },
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    atomic_write_json(PACKAGE_DIR / "execution_plan.json", plan)

    package_files = {
        *STATIC_CONTROL_FILES,
        *GENERATED_CONTROL_FILES,
        *(f"jobs/{execution_id}.json" for execution_id in EXPECTED_EXECUTION_IDS),
    }
    manifest = digested(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "execution_target": "chtc",
            "execution_plan_sha256": plan["sha256"],
            "source_authority_sha256": source_authority["sha256"],
            "anchor_evidence_sha256": anchor_evidence["sha256"],
            "horizon_delta_audit_sha256": audit["sha256"],
            "control_plane_receipt_sha256": control["sha256"],
            "source_archive": {
                "path": SOURCE_ARCHIVE_NAME,
                "sha256": SOURCE_ARCHIVE_SHA256,
                "size_bytes": SOURCE_ARCHIVE_SIZE_BYTES,
            },
            "source_archive_manifest": {
                "path": SOURCE_ARCHIVE_MANIFEST_NAME,
                "canonical_sha256": SOURCE_ARCHIVE_MANIFEST_SHA256,
                "file_sha256": SOURCE_ARCHIVE_MANIFEST_FILE_SHA256,
            },
            "direct_execution_count": DIRECT_EXECUTION_COUNT,
            "files": [
                _manifest_file_binding(PACKAGE_DIR, relative)
                for relative in sorted(package_files)
            ],
            "mutable_runtime_directories": [
                "worker_outputs",
                "worker_receipts",
            ],
            "activation_required_before_submission": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    atomic_write_json(PACKAGE_DIR / "package_manifest.json", manifest)
    return validate_package(
        full_archive_scan=True,
        # The sealed round-70 package preserves the exact anchor receipt, but
        # its original multi-hundred-MB round-50 attempt archives were cleaned
        # after validation. The receipt bytes and canonical digest are checked
        # above; do not require the deliberately removed external archives.
        full_anchor_scan=False,
    )


def main() -> int:
    try:
        result = build_package()
    except (
        OSError,
        PackageContractError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
