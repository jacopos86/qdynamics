#!/usr/bin/env python3
"""Materialize exact authority for the sealed 12-cell package."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ACTIVATION_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(ACTIVATION_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVATION_DIR))

from activation_contract import (  # noqa: E402
    ACTIVATION_ID,
    ACTIVATION_RELATIVE,
    ACTIVATION_SCHEMA,
    AUTHORIZATION_SCHEMA,
    BATCH_NAME,
    CALIBRATION_CANONICAL_SHA256,
    CALIBRATION_FILE_SHA256,
    CONTROL_FILES,
    DIRECT_EXECUTION_COUNT,
    EXECUTION_PLAN_CANONICAL_SHA256,
    EXECUTION_PLAN_FILE_SHA256,
    JOB_RESOURCE_KEYS,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_CANONICAL_SHA256,
    PACKAGE_MANIFEST_FILE_SHA256,
    PACKAGE_RELATIVE,
    QUEUE_VARIABLES,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    RESOURCE_STATUS,
    SOURCE_ARCHIVE_SHA256,
    ActivationContractError,
    canonical_json_bytes,
    digested,
    file_binding,
    json_binding,
    load_json,
    remote_dry_run_contract,
    repo_root_from_script,
    validate_sealed_package,
    verify_self_digest,
)


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ActivationContractError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _exclusive_write(path, canonical_json_bytes(payload) + b"\n")


def _job_resources(job: Mapping[str, Any]) -> dict[str, Any]:
    raw = job.get("resources")
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {*JOB_RESOURCE_KEYS, "status"}
        or raw.get("status")
        != "provisional_not_demonstrated_by_bounded_calibration"
    ):
        raise ActivationContractError("Job resource contract drifted.")
    resources: dict[str, Any] = {}
    for name in JOB_RESOURCE_KEYS:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ActivationContractError(
                f"Job resource {name} must be a positive integer."
            )
        resources[name] = int(value)
    resources["status"] = raw["status"]
    return resources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-utc", required=True)
    parser.add_argument("--remote-image-verified-utc", required=True)
    args = parser.parse_args()

    repo_root = repo_root_from_script(__file__)
    if ACTIVATION_DIR != repo_root / ACTIVATION_RELATIVE:
        raise ActivationContractError(
            "Activation directory identity drifted."
        )
    for name in (
        "activation_manifest.json",
        "queue.tsv",
        "authorizations",
    ):
        path = ACTIVATION_DIR / name
        if path.exists() or path.is_symlink():
            raise ActivationContractError(f"Refusing to overwrite: {path}")

    package = validate_sealed_package(repo_root)
    package_dir = package["package_dir"]
    package_manifest = package["manifest"]
    plan = package["plan"]
    calibration = package["calibration"]
    if (
        calibration.get("sha256") != CALIBRATION_CANONICAL_SHA256
        or calibration.get("package_resource_status")
        != RESOURCE_STATUS
        or calibration.get("package_resources_demonstrated") is not False
    ):
        raise ActivationContractError(
            "Provisional calibration gate did not close."
        )

    controls = [
        file_binding(ACTIVATION_DIR / name, relative_to=ACTIVATION_DIR)
        for name in CONTROL_FILES
    ]
    control_digest = hashlib.sha256(
        canonical_json_bytes(controls)
    ).hexdigest()
    jobs = package_manifest.get("jobs")
    plan_ids = plan.get("execution_ids")
    if (
        not isinstance(jobs, list)
        or len(jobs) != DIRECT_EXECUTION_COUNT
        or not isinstance(plan_ids, list)
        or len(plan_ids) != DIRECT_EXECUTION_COUNT
    ):
        raise ActivationContractError(
            "Sealed package execution count drifted."
        )

    authorizations_dir = ACTIVATION_DIR / "authorizations"
    authorizations_dir.mkdir()
    executions: list[dict[str, Any]] = []
    queue_lines: list[str] = []
    for queue_index, package_job in enumerate(jobs):
        if not isinstance(package_job, Mapping):
            raise ActivationContractError(
                "Sealed package job binding is malformed."
            )
        job_relative = package_job.get("path")
        if not isinstance(job_relative, str):
            raise ActivationContractError(
                "Sealed package job path is malformed."
            )
        job_path = package_dir / job_relative
        job = load_json(job_path, label=f"job {queue_index}")
        verify_self_digest(job, label=f"job {queue_index}")
        execution_id = str(job.get("execution_id"))
        if (
            job.get("schema") != JOB_SCHEMA
            or job.get("package_id") != PACKAGE_ID
            or execution_id != str(plan_ids[queue_index])
            or package_job.get("canonical_sha256") != job.get("sha256")
        ):
            raise ActivationContractError(
                "Sealed package job identity drifted."
            )
        job_binding = json_binding(job_path, relative_to=repo_root)
        if (
            package_job.get("sha256") != job_binding["sha256"]
            or int(package_job.get("size_bytes", -1))
            != job_binding["size_bytes"]
        ):
            raise ActivationContractError(
                "Sealed package job file binding drifted."
            )
        resources = _job_resources(job)
        authorization = digested(
            {
                "schema": AUTHORIZATION_SCHEMA,
                "authorization_id": (
                    f"{ACTIVATION_ID}__{execution_id}"
                ),
                "authorized_utc": args.authorized_utc,
                "package_id": PACKAGE_ID,
                "activation_id": ACTIVATION_ID,
                "batch_name": BATCH_NAME,
                "execution_id": execution_id,
                "job_sha256": job["sha256"],
                "job_file_sha256": job_binding["sha256"],
                "package_manifest_sha256": (
                    PACKAGE_MANIFEST_CANONICAL_SHA256
                ),
                "execution_plan_sha256": (
                    EXECUTION_PLAN_CANONICAL_SHA256
                ),
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "open_plateau_calibration_sha256": (
                    CALIBRATION_CANONICAL_SHA256
                ),
                "resource_status": RESOURCE_STATUS,
                "activation_control_plane_sha256": control_digest,
                "remote_image_path": REMOTE_IMAGE_PATH,
                "remote_image_sha256": REMOTE_IMAGE_SHA256,
                "remote_image_byte_verification_passed": True,
                "remote_image_verified_utc": (
                    args.remote_image_verified_utc
                ),
                "execution_authorized": True,
                "submission_authorized": True,
                "submission_state": "authorized_not_submitted",
                "remote_stage": False,
                "condor_submit": False,
                "submitted": False,
            }
        )
        authorization_path = authorizations_dir / f"{execution_id}.json"
        _write_json(authorization_path, authorization)
        authorization_binding = json_binding(
            authorization_path,
            relative_to=ACTIVATION_DIR,
        )
        executions.append(
            {
                "queue_index": queue_index,
                "execution_id": execution_id,
                "job": job_binding,
                "authorization": authorization_binding,
                "resources": resources,
            }
        )
        queue_lines.append(
            "\t".join(
                (
                    execution_id,
                    job_path.relative_to(repo_root).as_posix(),
                    str(job_binding["sha256"]),
                    authorization_path.relative_to(repo_root).as_posix(),
                    str(authorization_binding["sha256"]),
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                    str(resources["max_runtime_seconds"]),
                )
            )
            + "\n"
        )

    queue_path = ACTIVATION_DIR / "queue.tsv"
    _exclusive_write(queue_path, "".join(queue_lines).encode("utf-8"))
    manifest = digested(
        {
            "schema": ACTIVATION_SCHEMA,
            "activation_id": ACTIVATION_ID,
            "package_id": PACKAGE_ID,
            "batch_name": BATCH_NAME,
            "campaign_id": package_manifest["campaign_id"],
            "run_class": package_manifest["run_class"],
            "execution_target": "chtc",
            "authorized_utc": args.authorized_utc,
            "direct_execution_count": len(executions),
            "resource_status": RESOURCE_STATUS,
            "sealed_package": {
                "path": PACKAGE_RELATIVE.as_posix(),
                "manifest_canonical_sha256": (
                    PACKAGE_MANIFEST_CANONICAL_SHA256
                ),
                "manifest_file_sha256": (
                    PACKAGE_MANIFEST_FILE_SHA256
                ),
                "execution_plan_canonical_sha256": (
                    EXECUTION_PLAN_CANONICAL_SHA256
                ),
                "execution_plan_file_sha256": (
                    EXECUTION_PLAN_FILE_SHA256
                ),
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "open_plateau_calibration_canonical_sha256": (
                    CALIBRATION_CANONICAL_SHA256
                ),
                "open_plateau_calibration_file_sha256": (
                    CALIBRATION_FILE_SHA256
                ),
                "resource_status": RESOURCE_STATUS,
            },
            "remote_image": {
                "path": REMOTE_IMAGE_PATH,
                "sha256": REMOTE_IMAGE_SHA256,
                "byte_verification_passed": True,
                "verified_utc": args.remote_image_verified_utc,
            },
            "control_plane": controls,
            "activation_control_plane_sha256": control_digest,
            "executions": executions,
            "queue": file_binding(
                queue_path,
                relative_to=ACTIVATION_DIR,
            ),
            "queue_variables": list(QUEUE_VARIABLES),
            "remote_dry_run_validation_contract": (
                remote_dry_run_contract()
            ),
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_state": "authorized_not_submitted",
            "remote_stage": False,
            "condor_submit": False,
            "submitted": False,
            "paper_evidence_adopted": False,
        }
    )
    _write_json(ACTIVATION_DIR / "activation_manifest.json", manifest)
    print(
        canonical_json_bytes(
            {
                "status": "passed",
                "activation_id": ACTIVATION_ID,
                "activation_manifest_sha256": manifest["sha256"],
                "direct_execution_count": len(executions),
                "resource_status": RESOURCE_STATUS,
                "submission_state": "authorized_not_submitted",
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
