#!/usr/bin/env python3
"""Validate the inert three-row accepted-state resume package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True

from resume_contract import (  # noqa: E402
    CAMPAIGN_ID,
    CONTROL_FILES,
    IMPLEMENTATION_REPAIR_ID,
    PACKAGE_ID,
    PACKAGE_RELATIVE,
    PACKAGE_SCHEMA,
    RESUME_LOADER_AFTER_SHA256,
    RESUME_LOADER_BEFORE_SHA256,
    RESUME_LOADER_PATCH_PATH,
    ResumeContractError,
    canonical_json_bytes,
    expected_jobs,
    file_binding,
    load_json,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResumeContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResumeContractError(f"{label} must be a list.")
    return value


def _verify_binding(
    root: Path, raw: Any, *, label: str, canonical: bool = False
) -> tuple[Path, dict[str, Any] | None]:
    binding = _mapping(raw, label=f"{label} binding")
    path = root / safe_relative_path(binding.get("path"), label=f"{label} path")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ResumeContractError(f"{label} escaped package root.") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise ResumeContractError(f"{label} exact bytes drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != binding.get(
        "canonical_sha256"
    ):
        raise ResumeContractError(f"{label} canonical binding drifted.")
    return path, payload


def _scan_resume_archive(
    repo_root: Path, resume: Mapping[str, Any]
) -> None:
    archive_binding = _mapping(resume.get("archive"), label="resume archive")
    archive_path = repo_root / safe_relative_path(
        archive_binding.get("path"), label="resume archive path"
    )
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or archive_path.stat().st_size
        != int(archive_binding.get("size_bytes", -1))
        or sha256_file(archive_path) != archive_binding.get("sha256")
    ):
        raise ResumeContractError("Resume archive exact bytes drifted.")
    rows = _sequence(resume.get("members"), label="resume members")
    expected = {
        str(row["path"]): row for row in rows if isinstance(row, Mapping)
    }
    if len(expected) != len(rows) or len(expected) != 3:
        raise ResumeContractError("Resume archive member index drifted.")
    observed: set[str] = set()
    import hashlib

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            row = expected.get(member.name)
            if (
                row is None
                or member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row.get("size_bytes", -1))
            ):
                raise ResumeContractError(
                    f"Unsafe resume archive member: {member.name}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise ResumeContractError(
                    f"Unreadable resume archive member: {member.name}"
                )
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            if size != member.size or digest.hexdigest() != row.get("sha256"):
                raise ResumeContractError(
                    f"Resume archive member digest drifted: {member.name}"
                )
            observed.add(member.name)
    if observed != set(expected):
        raise ResumeContractError("Resume archive closure is incomplete.")


def _source_preflight(repo_root: Path, job_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "run_resume_cell.py"),
            "--job",
            str(job_path),
            "--preflight",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if completed.returncode != 0:
        raise ResumeContractError(
            f"Source preflight failed for {job_path.name}: {completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "passed"
        or payload.get("scientific_protocol_changed") is not False
        or payload.get("scientific_settings_changed") != []
        or payload.get("implementation_repair_id")
        != IMPLEMENTATION_REPAIR_ID
        or payload.get("resume_loader_sha256")
        != RESUME_LOADER_AFTER_SHA256
        or payload.get("source_held_job_preserved") is not True
    ):
        raise ResumeContractError(
            f"Source preflight closure drifted for {job_path.name}."
        )
    return payload


def validate_package(
    *, full_archive_scan: bool, source_preflight: bool
) -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    if PACKAGE_DIR != repo_root / PACKAGE_RELATIVE:
        raise ResumeContractError("Package directory identity drifted.")
    if (PACKAGE_DIR / "authorizations").exists() or (
        PACKAGE_DIR / "submit.sub"
    ).exists():
        raise ResumeContractError("Inert package contains activation state.")
    manifest = load_json(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    verify_self_digest(manifest, label="package manifest")
    expected_repair = {
        "repair_id": IMPLEMENTATION_REPAIR_ID,
        "path": RESUME_LOADER_PATCH_PATH,
        "before_sha256": RESUME_LOADER_BEFORE_SHA256,
        "after_sha256": RESUME_LOADER_AFTER_SHA256,
        "scientific_protocol_changed": False,
        "scientific_settings_changed": [],
    }
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status")
        != "passed_inert_three_authenticated_resumes"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("row_count") != 3
        or manifest.get("scientific_protocol_changed") is not False
        or manifest.get("scientific_settings_changed") != []
        or manifest.get("implementation_repair") != expected_repair
        or manifest.get("source_held_jobs_preserved") is not True
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submitted") is not False
    ):
        raise ResumeContractError("Package manifest identity drifted.")
    expected_controls = [
        file_binding(PACKAGE_DIR / name, relative_to=PACKAGE_DIR)
        for name in CONTROL_FILES
    ]
    if manifest.get("control_files") != expected_controls:
        raise ResumeContractError("Package control-plane bytes drifted.")
    expected = expected_jobs(repo_root, hash_archives=full_archive_scan)
    expected_ids = [str(job["execution_id"]) for job in expected]
    if manifest.get("execution_ids") != expected_ids:
        raise ResumeContractError("Package execution order drifted.")
    bindings = _sequence(manifest.get("jobs"), label="job bindings")
    if len(bindings) != len(expected):
        raise ResumeContractError("Package job cardinality drifted.")
    materialized: list[tuple[Path, dict[str, Any]]] = []
    for binding, expected_job in zip(bindings, expected, strict=True):
        path, job = _verify_binding(
            PACKAGE_DIR, binding, label="resume job", canonical=True
        )
        assert job is not None
        if (
            binding.get("execution_id") != expected_job["execution_id"]
            or job != expected_job
            or path.name != f"{expected_job['execution_id']}.json"
        ):
            raise ResumeContractError("Materialized resume job drifted.")
        materialized.append((path, job))
        if full_archive_scan:
            _scan_resume_archive(repo_root, job["resume_input"])

    resume_path, resume_manifest = _verify_binding(
        PACKAGE_DIR,
        manifest.get("resume_inputs_manifest"),
        label="resume inputs manifest",
        canonical=True,
    )
    plan_path, plan = _verify_binding(
        PACKAGE_DIR,
        manifest.get("execution_plan"),
        label="execution plan",
        canonical=True,
    )
    assert resume_manifest is not None and plan is not None
    if (
        resume_manifest.get("status") != "passed"
        or resume_manifest.get("cell_count") != 3
        or set(resume_manifest.get("cells", {})) != set(expected_ids)
        or resume_manifest.get("archive_bytes_duplicated_locally") is not False
        or plan.get("status") != "passed_inert"
        or plan.get("execution_ids") != expected_ids
        or plan.get("request_memory_mb") != 131_072
        or plan.get("request_disk_mb") != 81_920
        or plan.get("scientific_protocol_changed") is not False
        or plan.get("scientific_settings_changed") != []
        or plan.get("implementation_repair") != expected_repair
        or plan.get("source_held_jobs_preserved") is not True
    ):
        raise ResumeContractError("Plan/resume-input closure drifted.")
    queue_path, _ = _verify_binding(
        PACKAGE_DIR, manifest.get("queue"), label="queue"
    )
    expected_queue = "".join(
        "\t".join(
            (
                str(job["execution_id"]),
                path.relative_to(repo_root).as_posix(),
                sha256_file(path),
                str(job["resume_input"]["archive"]["path"]),
                str(job["resume_input"]["archive"]["sha256"]),
                str(job["resources"]["request_cpus"]),
                str(job["resources"]["request_memory_mb"]),
                str(job["resources"]["request_disk_mb"]),
                str(job["resources"]["max_runtime_seconds"]),
            )
        )
        + "\n"
        for path, job in materialized
    )
    if queue_path.read_text(encoding="utf-8") != expected_queue:
        raise ResumeContractError("Queue rows drifted from resume jobs.")
    preflights = (
        [_source_preflight(repo_root, path) for path, _job in materialized]
        if source_preflight
        else []
    )
    return {
        "status": "passed_inert_three_authenticated_resumes",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "row_count": 3,
        "execution_ids": expected_ids,
        "full_archive_scan_count": 3 if full_archive_scan else 0,
        "source_preflight_count": len(preflights),
        "scientific_protocol_changed": False,
        "scientific_settings_changed": [],
        "implementation_repair": expected_repair,
        "source_held_jobs_preserved": True,
        "request_memory_mb": 131_072,
        "request_disk_mb": 81_920,
        "resume_inputs_manifest": resume_path.relative_to(repo_root).as_posix(),
        "execution_plan": plan_path.relative_to(repo_root).as_posix(),
        "submission_ready": False,
        "submitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--full-archive-scan", action="store_true")
    parser.add_argument("--source-preflight", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_package(
            full_archive_scan=args.full_archive_scan,
            source_preflight=args.source_preflight,
        )
    except (OSError, ResumeContractError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
