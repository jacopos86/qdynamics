#!/usr/bin/env python3
"""Materialize the authorized weak-only staging activation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ACTIVATION_DIR = Path(__file__).resolve().parent
if str(ACTIVATION_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVATION_DIR))
sys.dont_write_bytecode = True

from activation_contract import (  # noqa: E402
    ACTIVATION_ID,
    ACTIVATION_RELATIVE,
    ACTIVATION_SCHEMA,
    BATCH_NAME,
    CAMPAIGN_ID,
    CONTROL_FILES,
    GENERATED_PATHS,
    IMAGE_PATH,
    IMAGE_SHA256,
    LOADER_PACKAGE_RELATIVE,
    OUTPUT_DESTINATION,
    OUTPUT_URI_TEMPLATE,
    PACKAGE_ID,
    PACKAGE_RELATIVE,
    QUEUE_VARIABLES,
    RUNTIME_RELATIVE,
    SOURCE_PACKAGE_RELATIVE,
    STAGING_ROOT,
    WEAK_ARCHIVE_BASENAME,
    WEAK_ARCHIVE_SHA256,
    WEAK_ARCHIVE_SIZE_BYTES,
    WEAK_EXECUTION_ID,
    WEAK_INPUT_URI,
    ActivationContractError,
    authorization_payload,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    file_binding,
    json_binding,
    load_json,
    render_submit,
    repo_root_from_script,
    sha256_file,
    validate_package,
    verify_self_digest,
    weak_job,
)


def _exclusive_write(path: Path, data: bytes, *, created: list[Path]) -> None:
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
        created.append(path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(
    path: Path, payload: Mapping[str, Any], *, created: list[Path]
) -> None:
    _exclusive_write(
        path, canonical_json_bytes(payload) + b"\n", created=created
    )


def materialize(*, authorized_utc: str) -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    if ACTIVATION_DIR != repo_root / ACTIVATION_RELATIVE:
        raise ActivationContractError("Activation directory identity drifted.")
    if not authorized_utc.strip():
        raise ActivationContractError("Authorization timestamp is required.")
    for name in GENERATED_PATHS:
        path = ACTIVATION_DIR / name
        if path.exists() or path.is_symlink():
            raise ActivationContractError(f"Refusing to overwrite: {path}")
    created: list[Path] = []
    try:
        package_validation = validate_package(repo_root)
        package_manifest_path = repo_root / PACKAGE_RELATIVE / "package_manifest.json"
        package_manifest = load_json(
            package_manifest_path, label="package manifest"
        )
        verify_self_digest(package_manifest, label="package manifest")
        job_path, job = weak_job(repo_root)
        controls = [
            file_binding(ACTIVATION_DIR / name, relative_to=ACTIVATION_DIR)
            for name in CONTROL_FILES
        ]
        control_sha = canonical_sha256({"controls": controls})
        authorization = digested(
            authorization_payload(
                job=job,
                package_manifest=package_manifest,
                control_sha256=control_sha,
                authorized_utc=authorized_utc,
            )
        )
        authorization_path = (
            ACTIVATION_DIR / "authorizations" / f"{WEAK_EXECUTION_ID}.json"
        )
        _write_json(authorization_path, authorization, created=created)
        authorization_binding = json_binding(
            authorization_path, relative_to=ACTIVATION_DIR
        )
        resources = job["resources"]
        queue_row = [
            WEAK_EXECUTION_ID,
            job_path.relative_to(repo_root).as_posix(),
            sha256_file(job_path),
            authorization_path.relative_to(repo_root).as_posix(),
            authorization_binding["sha256"],
            WEAK_INPUT_URI,
            WEAK_ARCHIVE_BASENAME,
            WEAK_ARCHIVE_SHA256,
            str(resources["request_cpus"]),
            str(resources["request_memory_mb"]),
            str(resources["request_disk_mb"]),
            str(resources["max_runtime_seconds"]),
        ]
        if len(queue_row) != len(QUEUE_VARIABLES):
            raise ActivationContractError("Queue field cardinality drifted.")
        queue_path = ACTIVATION_DIR / "queue.tsv"
        _exclusive_write(
            queue_path,
            ("\t".join(queue_row) + "\n").encode("utf-8"),
            created=created,
        )
        submit_path = ACTIVATION_DIR / "submit.sub"
        _exclusive_write(
            submit_path, render_submit().encode("utf-8"), created=created
        )
        manifest = digested(
            {
                "schema": ACTIVATION_SCHEMA,
                "status": "passed_authorized_pending_staging",
                "activation_id": ACTIVATION_ID,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "batch_name": BATCH_NAME,
                "run_class": "diagnostic",
                "execution_target": "chtc",
                "authorized_utc": authorized_utc,
                "direct_execution_count": 1,
                "execution_id": WEAK_EXECUTION_ID,
                "regime_id": "weak_strong",
                "resume_controller_round": 49,
                "target_horizon": 70,
                "sealed_package": {
                    "path": PACKAGE_RELATIVE.as_posix(),
                    "manifest": json_binding(
                        package_manifest_path, relative_to=repo_root
                    ),
                    "validator_status": package_validation["status"],
                    "source_preflight_count": package_validation[
                        "source_preflight_count"
                    ],
                },
                "source_package_path": SOURCE_PACKAGE_RELATIVE.as_posix(),
                "loader_package_path": LOADER_PACKAGE_RELATIVE.as_posix(),
                "job": json_binding(job_path, relative_to=repo_root),
                "authorization": {
                    "path": authorization_path.relative_to(repo_root).as_posix(),
                    **{
                        key: authorization_binding[key]
                        for key in (
                            "sha256",
                            "canonical_sha256",
                            "size_bytes",
                        )
                    },
                },
                "resources": resources,
                "remote_image": {
                    "path": IMAGE_PATH,
                    "sha256": IMAGE_SHA256,
                    "byte_verification_required_before_submit": True,
                    "byte_verification_passed": False,
                },
                "staging": {
                    "path": STAGING_ROOT,
                    "expected_quota_gb": 100,
                    "expected_item_limit": 1000,
                    "provisioned": False,
                    "input_uri": WEAK_INPUT_URI,
                    "input_basename": WEAK_ARCHIVE_BASENAME,
                    "input_sha256": WEAK_ARCHIVE_SHA256,
                    "input_size_bytes": WEAK_ARCHIVE_SIZE_BYTES,
                    "input_uploaded": False,
                    "output_uri_template": OUTPUT_URI_TEMPLATE,
                    "output_destination": OUTPUT_DESTINATION,
                    "output_exists": False,
                    "transfer_host": "transfer.chtc.wisc.edu",
                },
                "control_plane": controls,
                "activation_control_plane_sha256": control_sha,
                "queue": file_binding(queue_path, relative_to=ACTIVATION_DIR),
                "submit_descriptor": file_binding(
                    submit_path, relative_to=ACTIVATION_DIR
                ),
                "queue_variables": list(QUEUE_VARIABLES),
                "runtime_root": RUNTIME_RELATIVE.as_posix(),
                "ordinary_cluster": True,
                "bounded_factory": False,
                "intentional_hold": False,
                "source_held_jobs_preserved": True,
                "source_held_job_removal_authorized": False,
                "only_scientific_change": (
                    "maximum_controller_rounds_50_to_70"
                ),
                "non_swept_settings_diff": [],
                "pre_submit_requirements": [
                    "staging_quota_100gb_1000_items_present",
                    "staging_input_uploaded_via_transfer_server",
                    "staging_input_exact_size_and_sha256_verified",
                    "staging_output_path_absent",
                    "remote_image_exact_size_and_sha256_verified",
                    "exact_batch_and_execution_id_collision_check",
                    "condor_submit_dry_run_exact_descriptor",
                ],
                "execution_authorized": True,
                "submission_authorized": True,
                "submission_ready": False,
                "submission_state": "authorized_pending_staging",
                "remote_stage": False,
                "condor_submit": False,
                "submitted": False,
                "paper_evidence_adopted": False,
            }
        )
        _write_json(
            ACTIVATION_DIR / "activation_manifest.json",
            manifest,
            created=created,
        )
        return {
            "status": manifest["status"],
            "activation_id": ACTIVATION_ID,
            "activation_manifest_sha256": manifest["sha256"],
            "direct_execution_count": 1,
            "execution_id": WEAK_EXECUTION_ID,
            "submission_state": manifest["submission_state"],
        }
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        directory = ACTIVATION_DIR / "authorizations"
        if directory.is_dir() and not directory.is_symlink():
            try:
                directory.rmdir()
            except OSError:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-utc", required=True)
    args = parser.parse_args()
    print(
        canonical_json_bytes(
            materialize(authorized_utc=args.authorized_utc)
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
