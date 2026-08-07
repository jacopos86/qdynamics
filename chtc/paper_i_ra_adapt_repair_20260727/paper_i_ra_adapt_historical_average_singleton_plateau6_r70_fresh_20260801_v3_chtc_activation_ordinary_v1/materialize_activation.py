#!/usr/bin/env python3
"""Materialize the authorized six-row held activation after package sealing."""

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
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from activation_contract import (  # noqa: E402
    ACTIVATION_ID,
    ACTIVATION_SCHEMA,
    BATCH_NAME,
    CONTROL_FILES,
    GENERATED_PATHS,
    IMAGE_PATH,
    IMAGE_SHA256,
    PACKAGE_ID,
    QUEUE_VARIABLES,
    ActivationContractError,
    activation_relative,
    authorization_payload,
    authorization_relative,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    file_binding,
    json_binding,
    package_relative,
    render_submit,
    repo_root_from_script,
    sha256_file,
    validate_activation,
    validate_sealed_package,
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
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    created: list[Path],
) -> None:
    _exclusive_write(
        path,
        canonical_json_bytes(payload) + b"\n",
        created=created,
    )


def _sealed_package_binding(
    package: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    return {
        "path": package_relative().as_posix(),
        "manifest": json_binding(package["manifest_path"], relative_to=repo_root),
        "execution_plan": json_binding(package["plan_path"], relative_to=repo_root),
        "source_lock_audit": json_binding(package["audit_path"], relative_to=repo_root),
        "source_archive": file_binding(package["source_archive"], relative_to=repo_root),
        "validator_status": package["validator_receipt"]["status"],
    }


def _materialize_once(
    *, authorized_utc: str, created: list[Path]
) -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    if ACTIVATION_DIR != repo_root / activation_relative():
        raise ActivationContractError("Activation directory identity drifted.")
    for name in GENERATED_PATHS:
        path = ACTIVATION_DIR / name
        if path.exists() or path.is_symlink():
            raise ActivationContractError(f"Refusing to overwrite: {path}")
    if not authorized_utc.strip():
        raise ActivationContractError("Authorization timestamp is required.")

    package = validate_sealed_package(repo_root)
    controls = [
        file_binding(ACTIVATION_DIR / name, relative_to=ACTIVATION_DIR)
        for name in CONTROL_FILES
    ]
    control_sha = canonical_sha256(controls)
    authorization_bindings: list[dict[str, Any]] = []
    authorization_paths: dict[str, Path] = {}
    for row in package["jobs"]:
        execution_id = row["execution_id"]
        authorization = digested(
            authorization_payload(
                package=package,
                job_row=row,
                control_sha256=control_sha,
                authorized_utc=authorized_utc,
            )
        )
        authorization_path = ACTIVATION_DIR.joinpath(
            *authorization_relative(execution_id).parts
        )
        _write_json(authorization_path, authorization, created=created)
        authorization_paths[execution_id] = authorization_path
        authorization_bindings.append(
            {
                "execution_id": execution_id,
                **json_binding(authorization_path, relative_to=ACTIVATION_DIR),
            }
        )

    queue_lines: list[str] = []
    execution_bindings: list[dict[str, Any]] = []
    for index, row in enumerate(package["jobs"]):
        resources = row["resources"]
        job_path = row["path"]
        authorization_path = authorization_paths[row["execution_id"]]
        authorization_file_sha = sha256_file(authorization_path)
        queue_lines.append(
            "\t".join(
                (
                    row["execution_id"],
                    job_path.relative_to(repo_root).as_posix(),
                    sha256_file(job_path),
                    authorization_path.relative_to(repo_root).as_posix(),
                    authorization_file_sha,
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                    str(resources["max_runtime_seconds"]),
                )
            )
            + "\n"
        )
        execution_bindings.append(
            {
                "queue_index": index,
                "execution_id": row["execution_id"],
                "job": json_binding(job_path, relative_to=repo_root),
                "resources": {
                    key: resources[key]
                    for key in (
                        "request_cpus",
                        "request_memory_mb",
                        "request_disk_mb",
                        "max_runtime_seconds",
                    )
                },
            }
        )
    queue_path = ACTIVATION_DIR / "queue.tsv"
    _exclusive_write(
        queue_path,
        "".join(queue_lines).encode("utf-8"),
        created=created,
    )

    submit_path = ACTIVATION_DIR / "submit.sub"
    submit_text = render_submit(
        source_archive_sha256=package["source_archive_binding"]["sha256"]
    )
    _exclusive_write(
        submit_path,
        submit_text.encode("utf-8"),
        created=created,
    )
    manifest = digested(
        {
            "schema": ACTIVATION_SCHEMA,
            "activation_id": ACTIVATION_ID,
            "package_id": PACKAGE_ID,
            "campaign_id": package["manifest"]["campaign_id"],
            "batch_name": BATCH_NAME,
            "run_class": "candidate",
            "execution_target": "chtc",
            "authorized_utc": authorized_utc,
            "direct_execution_count": 6,
            "sealed_package": _sealed_package_binding(package, repo_root),
            "remote_image": {
                "path": IMAGE_PATH,
                "sha256": IMAGE_SHA256,
                "byte_verification_required_before_submit": True,
                "byte_verification_passed": False,
            },
            "control_plane": controls,
            "activation_control_plane_sha256": control_sha,
            "execution_authorizations": authorization_bindings,
            "executions": execution_bindings,
            "queue": file_binding(queue_path, relative_to=ACTIVATION_DIR),
            "submit_descriptor": file_binding(
                submit_path, relative_to=ACTIVATION_DIR
            ),
            "queue_variables": list(QUEUE_VARIABLES),
            "operational_mode": "ordinary_unheld_v1",
            "pre_submit_requirements": [
                "verify_remote_image_exact_path_size_and_sha256",
                "validate_exact_submit_descriptor_lifecycle",
                "condor_submit_dry_run_exact_descriptor",
                "check_exact_batch_and_execution_id_collisions",
                "check_home_quota_available",
            ],
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_state": "authorized_pending_remote_preflight",
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
    validated = validate_activation(repo_root)
    if validated.get("activation_manifest_sha256") != manifest["sha256"]:
        raise ActivationContractError("Post-materialization receipt drifted.")
    return {
        "status": validated["status"],
        "activation_id": validated["activation_id"],
        "activation_manifest_sha256": validated[
            "activation_manifest_sha256"
        ],
        "direct_execution_count": validated["direct_execution_count"],
        "submission_state": validated["submission_state"],
    }


def materialize(*, authorized_utc: str) -> dict[str, Any]:
    """Materialize atomically enough to leave controls-only on any failure."""
    created: list[Path] = []
    try:
        return _materialize_once(
            authorized_utc=authorized_utc,
            created=created,
        )
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        authorization_dir = ACTIVATION_DIR / "authorizations"
        if authorization_dir.is_dir() and not authorization_dir.is_symlink():
            try:
                authorization_dir.rmdir()
            except OSError:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-utc", required=True)
    args = parser.parse_args()
    print(canonical_json_bytes(materialize(authorized_utc=args.authorized_utc)).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


