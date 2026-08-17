#!/usr/bin/env python3
"""Materialize the authorized ordinary overlay after remote image proof."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Mapping


sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from activation_contract import (
    ACTIVATION_ID,
    ACTIVATION_SCHEMA,
    AUTHORIZATION_SCHEMA,
    BATCH_NAME,
    CONTROL_FILES,
    DIRECT_EXECUTION_COUNT,
    EXECUTION_PLAN_CANONICAL_SHA256,
    EXECUTION_PLAN_FILE_SHA256,
    GENERATED_FILES,
    HORIZON_AUDIT_CANONICAL_SHA256,
    IMAGE_PATH,
    IMAGE_SHA256,
    PACKAGE_ID,
    PACKAGE_MANIFEST_CANONICAL_SHA256,
    PACKAGE_MANIFEST_FILE_SHA256,
    QUEUE_VARIABLES,
    SOURCE_ARCHIVE_SHA256,
    ActivationContractError,
    activation_relative,
    canonical_json_bytes,
    digested,
    file_binding,
    json_binding,
    load_json,
    package_relative,
    quota_release_contract,
    repo_root_from_script,
    sha256_file,
    validate_sealed_package,
    verify_self_digest,
)


ACTIVATION_DIR = Path(__file__).resolve().parent


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-utc", required=True)
    parser.add_argument("--remote-image-verified-utc", required=True)
    parser.add_argument("--remote-image-path", default=IMAGE_PATH)
    parser.add_argument("--remote-image-sha256", required=True)
    args = parser.parse_args()
    if (
        args.remote_image_path != IMAGE_PATH
        or args.remote_image_sha256 != IMAGE_SHA256
    ):
        raise ActivationContractError("Remote image identity is not locked.")

    repo_root = repo_root_from_script(__file__)
    if ACTIVATION_DIR != repo_root / activation_relative():
        raise ActivationContractError("Activation directory identity drifted.")
    for name in GENERATED_FILES:
        path = ACTIVATION_DIR / name
        if path.exists() or path.is_symlink():
            raise ActivationContractError(f"Refusing to overwrite: {path}")

    package = validate_sealed_package(repo_root)
    package_dir = repo_root / package_relative()
    controls = [
        file_binding(ACTIVATION_DIR / name, relative_to=ACTIVATION_DIR)
        for name in CONTROL_FILES
    ]
    control_sha = hashlib.sha256(canonical_json_bytes(controls)).hexdigest()
    execution_ids = package["plan"].get("execution_ids")
    planned_rows = package["plan"].get("direct_executions")
    if (
        not isinstance(execution_ids, list)
        or len(execution_ids) != DIRECT_EXECUTION_COUNT
        or not isinstance(planned_rows, list)
        or len(planned_rows) != DIRECT_EXECUTION_COUNT
    ):
        raise ActivationContractError("Execution-plan closure drifted.")

    authorization = digested(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "passed",
            "authorization_id": f"{ACTIVATION_ID}__all6",
            "authorized_utc": args.authorized_utc,
            "activation_id": ACTIVATION_ID,
            "package_id": PACKAGE_ID,
            "campaign_id": package["manifest"]["campaign_id"],
            "package_manifest_sha256": (
                PACKAGE_MANIFEST_CANONICAL_SHA256
            ),
            "execution_plan_sha256": EXECUTION_PLAN_CANONICAL_SHA256,
            "horizon_delta_audit_sha256": (
                HORIZON_AUDIT_CANONICAL_SHA256
            ),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "activation_control_plane_sha256": control_sha,
            "remote_image_path": IMAGE_PATH,
            "remote_image_sha256": IMAGE_SHA256,
            "remote_image_byte_verification_passed": True,
            "remote_image_verified_utc": args.remote_image_verified_utc,
            "authorized_execution_ids": execution_ids,
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_state": "authorized_not_submitted",
            "remote_stage": False,
            "condor_submit": False,
            "submitted": False,
        }
    )
    authorization_path = ACTIVATION_DIR / "execution_authorization.json"
    _write_json(authorization_path, authorization)
    authorization_file_sha = sha256_file(authorization_path)

    queue_lines: list[str] = []
    execution_bindings: list[dict[str, Any]] = []
    for index, planned in enumerate(planned_rows):
        if not isinstance(planned, Mapping):
            raise ActivationContractError("Planned row is malformed.")
        execution_id = str(planned.get("execution_id"))
        if execution_id != execution_ids[index]:
            raise ActivationContractError("Execution ordering drifted.")
        job_path = package_dir / str(planned.get("job_spec_path"))
        job = load_json(job_path, label=f"job {index}")
        verify_self_digest(job, label=f"job {index}")
        if (
            job.get("execution_id") != execution_id
            or job.get("package_id") != PACKAGE_ID
            or job.get("run_class") != "paper_facing"
            or job.get("execution_authorized") is not False
            or job.get("submission_authorized") is not False
        ):
            raise ActivationContractError("Sealed job identity drifted.")
        resources = job.get("resources")
        if not isinstance(resources, Mapping):
            raise ActivationContractError("Job resources are absent.")
        resource_values = [
            resources.get("request_cpus"),
            resources.get("request_memory_mb"),
            resources.get("request_disk_mb"),
            resources.get("max_runtime_seconds"),
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in resource_values
        ):
            raise ActivationContractError("Job resource value drifted.")
        job_file_sha = sha256_file(job_path)
        queue_lines.append(
            "\t".join(
                (
                    execution_id,
                    job_path.relative_to(repo_root).as_posix(),
                    job_file_sha,
                    authorization_path.relative_to(repo_root).as_posix(),
                    authorization_file_sha,
                    *(str(value) for value in resource_values),
                )
            )
            + "\n"
        )
        execution_bindings.append(
            {
                "queue_index": index,
                "execution_id": execution_id,
                "job": json_binding(job_path, relative_to=repo_root),
                "resources": {
                    "request_cpus": resource_values[0],
                    "request_memory_mb": resource_values[1],
                    "request_disk_mb": resource_values[2],
                    "max_runtime_seconds": resource_values[3],
                },
            }
        )

    queue_path = ACTIVATION_DIR / "queue.tsv"
    _exclusive_write(queue_path, "".join(queue_lines).encode("utf-8"))
    manifest = digested(
        {
            "schema": ACTIVATION_SCHEMA,
            "activation_id": ACTIVATION_ID,
            "package_id": PACKAGE_ID,
            "campaign_id": package["manifest"]["campaign_id"],
            "batch_name": BATCH_NAME,
            "run_class": "paper_facing",
            "execution_target": "chtc",
            "authorized_utc": args.authorized_utc,
            "direct_execution_count": DIRECT_EXECUTION_COUNT,
            "sealed_package": {
                "path": package_relative().as_posix(),
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
                "horizon_delta_audit_sha256": (
                    HORIZON_AUDIT_CANONICAL_SHA256
                ),
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            },
            "remote_image": {
                "path": IMAGE_PATH,
                "sha256": IMAGE_SHA256,
                "byte_verification_passed": True,
                "verified_utc": args.remote_image_verified_utc,
            },
            "control_plane": controls,
            "activation_control_plane_sha256": control_sha,
            "execution_authorization": json_binding(
                authorization_path, relative_to=ACTIVATION_DIR
            ),
            "executions": execution_bindings,
            "queue": file_binding(queue_path, relative_to=ACTIVATION_DIR),
            "queue_variables": list(QUEUE_VARIABLES),
            "operational_mode": "ordinary_unheld_v1",
            "quota_release_contract": quota_release_contract(),
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
                "direct_execution_count": DIRECT_EXECUTION_COUNT,
                "submission_state": "authorized_not_submitted",
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
