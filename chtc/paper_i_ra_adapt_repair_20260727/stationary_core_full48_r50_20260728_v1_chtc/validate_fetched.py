#!/usr/bin/env python3
"""Validate fetched attempt archives without selecting or adopting evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    EXPECTED_ARTIFACT_ROLES,
    FETCH_VALIDATION_SCHEMA,
    JOB_SPEC_SCHEMA,
    PACKAGE_ID,
    PACKAGE_RELATIVE_ROOT,
    RUNTIME_RELATIVE_ROOT,
    SUBMISSION_AUTHORIZATION_RELATIVE,
    WORKER_RECEIPT_SCHEMA,
    PackageContractError,
    atomic_write_json,
    canonical_json_bytes,
    direct_execution_ids,
    digested,
    load_json_object,
    repo_root_from_script,
    sha256_file,
    verify_self_digest,
)


ATTEMPT_RE = re.compile(
    r"^(?P<execution>core__[A-Za-z0-9_.-]+)__cluster_"
    r"(?P<cluster>[0-9]+)__proc_(?P<proc>[0-9]+)\.tar\.gz$"
)


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise PackageContractError(f"Unsafe attempt member: {name}")
    return path


def _json_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    label: str,
) -> dict[str, Any]:
    source = archive.extractfile(member)
    if source is None:
        raise PackageContractError(f"{label} has no bytes.")
    try:
        payload = json.loads(source.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"{label} is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return payload


def _stream_member_sha256_and_size(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    label: str,
) -> tuple[str, int]:
    source = archive.extractfile(member)
    if source is None:
        raise PackageContractError(f"{label} has no bytes.")
    digest = hashlib.sha256()
    size_bytes = 0
    while True:
        block = source.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
        size_bytes += len(block)
    return digest.hexdigest(), size_bytes


def validate_attempt(path: Path, *, relative_path: str) -> dict[str, Any]:
    match = ATTEMPT_RE.fullmatch(relative_path)
    if match is None:
        raise PackageContractError(
            f"Unknown attempt path: {relative_path}"
        )
    execution_id = match.group("execution")
    if execution_id not in direct_execution_ids():
        raise PackageContractError(
            f"Attempt is outside the 48-cell matrix: {execution_id}"
        )
    members: dict[str, tarfile.TarInfo] = {}
    directories: set[str] = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            relative = _safe_member(member.name).as_posix()
            if member.isdir():
                if relative in directories:
                    raise PackageContractError(
                        f"Duplicate attempt directory: {relative}"
                    )
                directories.add(relative)
                continue
            if relative in members or not member.isfile():
                raise PackageContractError(
                    f"Unsafe/duplicate attempt member: {relative}"
                )
            members[relative] = member
        if directories != {"worker_outputs"}:
            raise PackageContractError(
                "Attempt directory allowlist must be exactly worker_outputs/."
            )
        status_name = "worker_outputs/worker_exit_status.txt"
        scheduler_attempt_name = (
            "worker_outputs/scheduler_attempt_ordinal.txt"
        )
        status_member = members.get(status_name)
        if status_member is None:
            raise PackageContractError("Attempt has no worker exit status.")
        status_stream = archive.extractfile(status_member)
        if status_stream is None:
            raise PackageContractError("Cannot read worker exit status.")
        try:
            exit_status = int(status_stream.read().decode("ascii").strip())
        except ValueError as exc:
            raise PackageContractError("Worker exit status is invalid.") from exc
        scheduler_member = members.get(scheduler_attempt_name)
        if scheduler_member is None:
            raise PackageContractError(
                "Attempt has no scheduler attempt ordinal."
            )
        scheduler_stream = archive.extractfile(scheduler_member)
        if scheduler_stream is None:
            raise PackageContractError(
                "Cannot read scheduler attempt ordinal."
            )
        try:
            scheduler_attempt_ordinal = int(
                scheduler_stream.read().decode("ascii").strip()
            )
        except ValueError as exc:
            raise PackageContractError(
                "Scheduler attempt ordinal is invalid."
            ) from exc
        if scheduler_attempt_ordinal < 1:
            raise PackageContractError(
                "Scheduler attempt ordinal must be positive."
            )
        receipt_name = "worker_outputs/worker_receipt.json"
        receipt = None
        artifact_checks: list[dict[str, Any]] = []
        if exit_status == 0:
            receipt_member = members.get(receipt_name)
            if receipt_member is None:
                raise PackageContractError(
                    "Successful attempt has no worker receipt."
                )
            receipt = _json_member(
                archive, receipt_member, label="worker receipt"
            )
            verify_self_digest(receipt, label="worker receipt")
            job_name = (
                f"{PACKAGE_RELATIVE_ROOT}/jobs/{execution_id}.json"
            )
            job_member = members.get(job_name)
            if job_member is None:
                raise PackageContractError(
                    "Attempt does not retain its exact job spec."
                )
            job = _json_member(archive, job_member, label="attempt job spec")
            verify_self_digest(job, label="attempt job spec")
            job_stream = archive.extractfile(job_member)
            if job_stream is None:
                raise PackageContractError("Cannot reread attempt job spec.")
            job_bytes = job_stream.read()
            if (
                job.get("schema") != JOB_SPEC_SCHEMA
                or job.get("execution_id") != execution_id
                or receipt.get("job_spec_path")
                != f"jobs/{execution_id}.json"
                or receipt.get("job_spec_sha256") != job["sha256"]
                or receipt.get("job_spec_file_sha256")
                != hashlib.sha256(job_bytes).hexdigest()
            ):
                raise PackageContractError("Attempt job binding drifted.")
            raw_artifacts = receipt.get("artifact_bindings")
            artifact_roles = [
                str(row.get("role", ""))
                for row in raw_artifacts
                if isinstance(row, Mapping)
            ] if isinstance(raw_artifacts, list) else []
            if (
                receipt.get("schema") != WORKER_RECEIPT_SCHEMA
                or receipt.get("package_id") != PACKAGE_ID
                or receipt.get("execution_id") != execution_id
                or receipt.get("status") != "passed"
                or int(receipt.get("scheduler_attempt_ordinal", -1))
                != scheduler_attempt_ordinal
                or not isinstance(raw_artifacts, list)
                or artifact_roles != list(EXPECTED_ARTIFACT_ROLES)
            ):
                raise PackageContractError("Worker receipt closure drifted.")
            local_paths = {
                "execution_manifest": "execution_manifest.json",
                "checkpoint": "checkpoint.json",
                "estimator_ledger": "estimator_ledger.json",
                "result": "result.json",
                "summary": "summary.json",
            }
            for raw in raw_artifacts:
                if not isinstance(raw, Mapping):
                    raise PackageContractError("Invalid artifact binding.")
                if raw.get("path") != local_paths.get(str(raw.get("role"))):
                    raise PackageContractError(
                        "Worker artifact local path drifted."
                    )
                member_name = f"worker_outputs/{raw['path']}"
                member = members.get(member_name)
                if member is None:
                    raise PackageContractError(
                        f"Missing worker artifact: {member_name}"
                    )
                actual_sha256, actual_size_bytes = (
                    _stream_member_sha256_and_size(
                        archive,
                        member,
                        label=f"worker artifact {member_name}",
                    )
                )
                if (
                    actual_sha256 != raw.get("sha256")
                    or actual_size_bytes
                    != int(raw.get("size_bytes", -1))
                ):
                    raise PackageContractError(
                        f"Worker artifact binding drifted: {member_name}"
                    )
                artifact_checks.append(dict(raw))
            expected_members = {
                status_name,
                scheduler_attempt_name,
                receipt_name,
                job_name,
                *(
                    f"worker_outputs/{raw['path']}"
                    for raw in raw_artifacts
                ),
            }
            if set(members) != expected_members:
                raise PackageContractError(
                    "Attempt recursive file allowlist drifted."
                )
            package_manifest = load_json_object(
                PACKAGE_DIR / "package_manifest.json",
                label="current package manifest",
            )
            execution_plan = load_json_object(
                PACKAGE_DIR / "execution_plan.json",
                label="current execution plan",
            )
            authorization_path = (
                PACKAGE_DIR / SUBMISSION_AUTHORIZATION_RELATIVE
            )
            authorization = load_json_object(
                authorization_path,
                label="current submission authorization",
            )
            if (
                receipt.get("package_manifest_sha256")
                != package_manifest["sha256"]
                or receipt.get("package_manifest_file_sha256")
                != sha256_file(PACKAGE_DIR / "package_manifest.json")
                or receipt.get("execution_plan_sha256")
                != execution_plan["sha256"]
                or receipt.get("execution_plan_file_sha256")
                != sha256_file(PACKAGE_DIR / "execution_plan.json")
                or receipt.get("submission_authorization_sha256")
                != authorization["sha256"]
                or receipt.get("submission_authorization_file_sha256")
                != sha256_file(authorization_path)
                or receipt.get("source_archive_sha256")
                != execution_plan["source_archive"]["sha256"]
            ):
                raise PackageContractError(
                    "Attempt package/plan/authorization binding drifted."
                )
            for raw in raw_artifacts:
                role = str(raw["role"])
                if (
                    raw.get("declared_canonical_path")
                    != job["artifact_paths"][role]
                    or raw.get("mapping_kind")
                    != "worker_archive_copy_of_declared_output_v1"
                ):
                    raise PackageContractError(
                        f"Attempt destination mapping drifted: {role}"
                    )
        else:
            job_name = (
                f"{PACKAGE_RELATIVE_ROOT}/jobs/{execution_id}.json"
            )
            allowed_partial = {
                status_name,
                scheduler_attempt_name,
                job_name,
                "worker_outputs/worker_receipt.json",
                "worker_outputs/execution_manifest.json",
                "worker_outputs/checkpoint.json",
                "worker_outputs/estimator_ledger.json",
                "worker_outputs/result.json",
                "worker_outputs/summary.json",
            }
            if (
                not {
                    status_name,
                    scheduler_attempt_name,
                    job_name,
                }.issubset(members)
                or not set(members).issubset(allowed_partial)
            ):
                raise PackageContractError(
                    "Failed attempt contains undeclared partial members."
                )
    return {
        "execution_id": execution_id,
        "cluster_id": int(match.group("cluster")),
        "proc_id": int(match.group("proc")),
        "attempt_ordinal": scheduler_attempt_ordinal,
        "path": relative_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "worker_exit_status": exit_status,
        "worker_receipt_sha256": (
            None if receipt is None else receipt["sha256"]
        ),
        "artifact_bindings": artifact_checks,
        "status": "passed" if exit_status == 0 else "failed_attempt_retained",
    }


def validate_fetched(*, fetched_dir: Path, output: Path) -> dict[str, Any]:
    if not fetched_dir.is_dir() or fetched_dir.is_symlink():
        raise PackageContractError("Fetched directory is unavailable/unsafe.")
    entries = sorted(fetched_dir.rglob("*"))
    unknown = [
        path.relative_to(fetched_dir).as_posix()
        for path in entries
        if path.is_symlink()
        or (
            path.is_file()
            and ATTEMPT_RE.fullmatch(
                path.relative_to(fetched_dir).as_posix()
            )
            is None
        )
        or (not path.is_file() and not path.is_dir())
    ]
    if unknown:
        raise PackageContractError(
            f"Unsafe or unknown fetched entries: {unknown}"
        )
    attempt_files = [path for path in entries if path.is_file()]
    attempts = [
        validate_attempt(
            path,
            relative_path=path.relative_to(fetched_dir).as_posix(),
        )
        for path in attempt_files
    ]
    observed_directories = {
        path.relative_to(fetched_dir).as_posix()
        for path in entries
        if path.is_dir()
    }
    if observed_directories:
        raise PackageContractError(
            "Fetched runtime root must contain only terminal attempt files."
        )
    receipt = digested(
        {
            "schema": FETCH_VALIDATION_SCHEMA,
            "package_id": PACKAGE_ID,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "execution_ids_with_passed_attempts": sorted(
                {
                    row["execution_id"]
                    for row in attempts
                    if row["status"] == "passed"
                }
            ),
            "automatic_attempt_selection_performed": False,
            "paper_evidence_adopted": False,
            "status": "validated_no_selection",
        }
    )
    atomic_write_json(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fetched-dir",
        type=Path,
        default=(
            repo_root_from_script(__file__)
            / RUNTIME_RELATIVE_ROOT
            / "fetched"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = validate_fetched(
            fetched_dir=args.fetched_dir.resolve(),
            output=args.output.resolve(),
        )
        print(canonical_json_bytes(receipt).decode("utf-8"))
        return 0
    except (OSError, PackageContractError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
