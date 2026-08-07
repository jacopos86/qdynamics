#!/usr/bin/env python3
"""Materialize the authorized ordinary three-row resume activation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
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
    IMPLEMENTATION_REPAIR,
    PACKAGE_ID,
    PACKAGE_RELATIVE,
    QUEUE_VARIABLES,
    RUNTIME_RELATIVE,
    SOURCE_PACKAGE_RELATIVE,
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
    safe_relative_path,
    sha256_file,
    verify_self_digest,
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


def _validated_package(repo_root: Path) -> dict[str, Any]:
    package_root = repo_root / PACKAGE_RELATIVE
    validator = package_root / "validate_package.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(validator),
            "--metadata-only",
            "--source-preflight",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        raise ActivationContractError(
            f"Sealed package validation failed: {completed.stderr}"
        )
    receipt = load_json(
        package_root / "package_manifest.json", label="package manifest"
    )
    verify_self_digest(receipt, label="package manifest")
    validation = __import__("json").loads(completed.stdout)
    if (
        validation.get("status")
        != "passed_inert_three_authenticated_resumes"
        or validation.get("source_preflight_count") != 3
        or validation.get("scientific_protocol_changed") is not False
        or validation.get("scientific_settings_changed") != []
        or validation.get("implementation_repair") != IMPLEMENTATION_REPAIR
        or validation.get("source_held_jobs_preserved") is not True
    ):
        raise ActivationContractError("Sealed package validation drifted.")
    jobs = []
    for binding in receipt["jobs"]:
        path = package_root / safe_relative_path(
            binding["path"], label="job path"
        )
        job = load_json(path, label="resume job")
        if (
            verify_self_digest(job, label="resume job")
            != binding["canonical_sha256"]
            or sha256_file(path) != binding["sha256"]
        ):
            raise ActivationContractError("Resume job binding drifted.")
        jobs.append((path, job))
    return {
        "root": package_root,
        "manifest": receipt,
        "jobs": jobs,
        "validator": validation,
    }


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
        package = _validated_package(repo_root)
        controls = [
            file_binding(ACTIVATION_DIR / name, relative_to=ACTIVATION_DIR)
            for name in CONTROL_FILES
        ]
        control_sha = canonical_sha256({"controls": controls})
        authorization_bindings: list[dict[str, Any]] = []
        execution_bindings: list[dict[str, Any]] = []
        queue_lines: list[str] = []
        for index, (job_path, job) in enumerate(package["jobs"]):
            identifier = str(job["execution_id"])
            authorization = digested(
                authorization_payload(
                    job=job,
                    package_manifest=package["manifest"],
                    control_sha256=control_sha,
                    authorized_utc=authorized_utc,
                )
            )
            authorization_path = (
                ACTIVATION_DIR / "authorizations" / f"{identifier}.json"
            )
            _write_json(
                authorization_path, authorization, created=created
            )
            authorization_binding = json_binding(
                authorization_path, relative_to=ACTIVATION_DIR
            )
            authorization_bindings.append(
                {"execution_id": identifier, **authorization_binding}
            )
            resources = job["resources"]
            resume = job["resume_input"]["archive"]
            queue_lines.append(
                "\t".join(
                    (
                        identifier,
                        job_path.relative_to(repo_root).as_posix(),
                        sha256_file(job_path),
                        authorization_path.relative_to(repo_root).as_posix(),
                        authorization_binding["sha256"],
                        str(resume["path"]),
                        str(resume["sha256"]),
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
                    "execution_id": identifier,
                    "source_cluster_id": job["source_cluster_id"],
                    "source_proc_id": job["source_proc_id"],
                    "source_held_job_preserved": True,
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
                    "resume_archive": resume,
                    "resources": resources,
                    "implementation_repair": IMPLEMENTATION_REPAIR,
                }
            )
        queue_path = ACTIVATION_DIR / "queue.tsv"
        _exclusive_write(
            queue_path, "".join(queue_lines).encode("utf-8"), created=created
        )
        submit_path = ACTIVATION_DIR / "submit.sub"
        _exclusive_write(
            submit_path, render_submit().encode("utf-8"), created=created
        )
        manifest = digested(
            {
                "schema": ACTIVATION_SCHEMA,
                "status": "passed_authorized_not_submitted",
                "activation_id": ACTIVATION_ID,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "batch_name": BATCH_NAME,
                "run_class": "diagnostic",
                "execution_target": "chtc",
                "authorized_utc": authorized_utc,
                "direct_execution_count": 3,
                "sealed_package": {
                    "path": PACKAGE_RELATIVE.as_posix(),
                    "manifest": json_binding(
                        package["root"] / "package_manifest.json",
                        relative_to=repo_root,
                    ),
                    "validator_status": package["validator"]["status"],
                    "source_preflight_count": package["validator"][
                        "source_preflight_count"
                    ],
                },
                "source_package_path": SOURCE_PACKAGE_RELATIVE.as_posix(),
                "implementation_repair": IMPLEMENTATION_REPAIR,
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
                "queue": file_binding(
                    queue_path, relative_to=ACTIVATION_DIR
                ),
                "submit_descriptor": file_binding(
                    submit_path, relative_to=ACTIVATION_DIR
                ),
                "queue_variables": list(QUEUE_VARIABLES),
                "runtime_root": RUNTIME_RELATIVE.as_posix(),
                "ordinary_cluster": True,
                "bounded_factory": False,
                "source_held_jobs_preserved": True,
                "source_held_job_removal_authorized": False,
                "scientific_protocol_changed": False,
                "scientific_settings_changed": [],
                "pre_submit_requirements": [
                    "verify_remote_image_exact_path_size_and_sha256",
                    "verify_three_remote_resume_archives_exact_size_and_sha256",
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
        return {
            "status": manifest["status"],
            "activation_id": ACTIVATION_ID,
            "activation_manifest_sha256": manifest["sha256"],
            "direct_execution_count": 3,
            "implementation_repair": IMPLEMENTATION_REPAIR,
            "submission_state": manifest["submission_state"],
        }
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
    print(
        canonical_json_bytes(
            materialize(authorized_utc=args.authorized_utc)
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
