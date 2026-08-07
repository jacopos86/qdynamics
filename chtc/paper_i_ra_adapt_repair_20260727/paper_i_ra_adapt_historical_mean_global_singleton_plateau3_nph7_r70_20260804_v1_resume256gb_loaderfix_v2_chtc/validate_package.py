#!/usr/bin/env python3
"""Validate the inert exact-prefix r70 continuation package."""

from __future__ import annotations

import argparse
import hashlib
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

from package_contract import (  # noqa: E402
    CAMPAIGN_ID,
    CONTROL_FILES,
    HORIZON_CHANGED_PATHS,
    PACKAGE_ID,
    PACKAGE_RELATIVE,
    PACKAGE_SCHEMA,
    RESOURCE_ENVELOPE,
    SCIENTIFIC_SETTINGS_CHANGED,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    expected_materialization,
    file_binding,
    implementation_repair,
    load_json,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageContractError(f"{label} must be a list.")
    return value


def _verify_binding(
    root: Path, raw: Any, *, label: str, canonical: bool = False
) -> tuple[Path, dict[str, Any] | None]:
    binding = _mapping(raw, label=f"{label} binding")
    path = root / safe_relative_path(binding.get("path"), label=f"{label} path")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped package root.") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError(f"{label} bytes drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != binding.get(
        "canonical_sha256"
    ):
        raise PackageContractError(f"{label} digest drifted.")
    return path, payload


def _scan_archive(repo_root: Path, resume: Mapping[str, Any]) -> None:
    binding = _mapping(resume.get("local_archive"), label="archive")
    path = repo_root / safe_relative_path(
        binding.get("path"), label="archive path"
    )
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError("Resume archive bytes drifted.")
    rows = _sequence(resume.get("members"), label="archive members")
    expected = {
        str(row["path"]): row for row in rows if isinstance(row, Mapping)
    }
    if len(expected) != 3 or len(expected) != len(rows):
        raise PackageContractError("Archive member index drifted.")
    observed: set[str] = set()
    with tarfile.open(path, mode="r|gz") as archive:
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
                raise PackageContractError(
                    f"Unsafe archive member: {member.name}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"Unreadable archive member: {member.name}"
                )
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            if size != member.size or digest.hexdigest() != row.get("sha256"):
                raise PackageContractError(
                    f"Archive member digest drifted: {member.name}"
                )
            observed.add(member.name)
    if observed != set(expected):
        raise PackageContractError("Archive member closure is incomplete.")


def _preflight(repo_root: Path, job_path: Path) -> dict[str, Any]:
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
        timeout=900,
    )
    if completed.returncode != 0:
        raise PackageContractError(
            f"Source preflight failed for {job_path.name}: {completed.stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PackageContractError("Preflight returned malformed JSON.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "passed"
        or payload.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or payload.get("changed_protocol_paths")
        != list(HORIZON_CHANGED_PATHS)
        or payload.get("non_swept_settings_diff") != []
        or payload.get("target_horizon") != TARGET_HORIZON
        or payload.get("request_memory_mb")
        != RESOURCE_ENVELOPE["request_memory_mb"]
        or payload.get("request_disk_mb")
        != RESOURCE_ENVELOPE["request_disk_mb"]
        or payload.get("source_held_job_preserved") is not True
    ):
        raise PackageContractError("Source preflight closure drifted.")
    return payload


def validate(
    *, full_archive_scan: bool, source_preflight: bool
) -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    if PACKAGE_DIR != repo_root / PACKAGE_RELATIVE:
        raise PackageContractError("Package directory identity drifted.")
    if (PACKAGE_DIR / "authorizations").exists() or (
        PACKAGE_DIR / "submit.sub"
    ).exists():
        raise PackageContractError("Inert package contains activation state.")
    manifest = load_json(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status")
        != "passed_inert_three_authenticated_r70_resumes"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("row_count") != 3
        or manifest.get("source_horizon") != SOURCE_HORIZON
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or manifest.get("changed_protocol_paths")
        != list(HORIZON_CHANGED_PATHS)
        or manifest.get("non_swept_settings_diff") != []
        or manifest.get("implementation_repair") != implementation_repair()
        or manifest.get("source_held_jobs_preserved") is not True
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Package manifest identity drifted.")
    expected_controls = [
        file_binding(PACKAGE_DIR / name, relative_to=PACKAGE_DIR)
        for name in CONTROL_FILES
    ]
    if manifest.get("control_files") != expected_controls:
        raise PackageContractError("Package control-plane bytes drifted.")
    expected = expected_materialization(
        repo_root, hash_archives=full_archive_scan
    )
    expected_ids = [str(row["execution_id"]) for row in expected]
    if manifest.get("execution_ids") != expected_ids:
        raise PackageContractError("Execution order drifted.")
    bindings = _sequence(manifest.get("jobs"), label="job bindings")
    if len(bindings) != len(expected):
        raise PackageContractError("Job cardinality drifted.")
    jobs: list[tuple[Path, dict[str, Any]]] = []
    for raw, expected_row in zip(bindings, expected, strict=True):
        path, job = _verify_binding(
            PACKAGE_DIR, raw, label="job", canonical=True
        )
        assert job is not None
        derived_path, derived = _verify_binding(
            repo_root,
            job.get("derived_protocol"),
            label="derived protocol",
            canonical=True,
        )
        assert derived is not None
        resume = _mapping(job.get("resume_input"), label="resume input")
        if (
            raw.get("execution_id") != expected_row["execution_id"]
            or job.get("execution_id") != expected_row["execution_id"]
            or path.name != f"{expected_row['execution_id']}.json"
            or derived != expected_row["derived_protocol"]
            or job.get("derived_protocol_sha256") != derived.get("sha256")
            or derived_path.name != f"{expected_row['execution_id']}.json"
            or job.get("resume_input") != expected_row["resume_input"]
            or job.get("resources") != RESOURCE_ENVELOPE
            or job.get("scientific_protocol_changed") is not True
            or job.get("scientific_settings_changed")
            != list(SCIENTIFIC_SETTINGS_CHANGED)
            or job.get("only_scientific_change")
            != "maximum_controller_rounds_50_to_70"
            or job.get("non_swept_settings_diff") != []
        ):
            raise PackageContractError("Materialized job drifted.")
        if full_archive_scan:
            _scan_archive(repo_root, resume)
        jobs.append((path, job))
    _resume_path, resume_manifest = _verify_binding(
        PACKAGE_DIR,
        manifest.get("resume_inputs_manifest"),
        label="resume inputs manifest",
        canonical=True,
    )
    _audit_path, audit = _verify_binding(
        PACKAGE_DIR,
        manifest.get("source_lock_audit"),
        label="source lock audit",
        canonical=True,
    )
    _plan_path, plan = _verify_binding(
        PACKAGE_DIR,
        manifest.get("execution_plan"),
        label="execution plan",
        canonical=True,
    )
    queue_path, _ = _verify_binding(
        PACKAGE_DIR, manifest.get("queue"), label="queue"
    )
    assert resume_manifest is not None and audit is not None and plan is not None
    if (
        resume_manifest.get("status") != "passed"
        or resume_manifest.get("cell_count") != 3
        or resume_manifest.get("resume_controller_rounds") != [49, 45, 31]
        or resume_manifest.get("archive_bytes_duplicated_locally") is not False
        or audit.get("status") != "passed"
        or audit.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or audit.get("changed_protocol_paths")
        != list(HORIZON_CHANGED_PATHS)
        or audit.get("non_swept_settings_diff") != []
        or audit.get("paper_evidence_adopted") is not False
        or plan.get("status") != "passed_inert"
        or plan.get("activation_cardinality") != 1
        or plan.get("resources") != RESOURCE_ENVELOPE
        or plan.get("submission_ready") is not False
    ):
        raise PackageContractError("Package audit/plan closure drifted.")
    expected_queue = "".join(
        "\t".join(
            (
                str(job["execution_id"]),
                path.relative_to(repo_root).as_posix(),
                sha256_file(path),
                str(job["resume_input"]["local_archive"]["path"]),
                str(job["resume_input"]["local_archive"]["sha256"]),
                str(RESOURCE_ENVELOPE["request_cpus"]),
                str(RESOURCE_ENVELOPE["request_memory_mb"]),
                str(RESOURCE_ENVELOPE["request_disk_mb"]),
                str(RESOURCE_ENVELOPE["max_runtime_seconds"]),
            )
        )
        + "\n"
        for path, job in jobs
    )
    if queue_path.read_text(encoding="utf-8") != expected_queue:
        raise PackageContractError("Queue rows drifted.")
    preflights = (
        [_preflight(repo_root, path) for path, _job in jobs]
        if source_preflight
        else []
    )
    return {
        "status": "passed_inert_three_authenticated_r70_resumes",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "row_count": 3,
        "execution_ids": expected_ids,
        "resume_controller_rounds": [49, 45, 31],
        "full_archive_scan_count": 3 if full_archive_scan else 0,
        "source_preflight_count": len(preflights),
        "only_scientific_change": "maximum_controller_rounds_50_to_70",
        "changed_protocol_paths": list(HORIZON_CHANGED_PATHS),
        "non_swept_settings_diff": [],
        "resources": dict(RESOURCE_ENVELOPE),
        "source_held_jobs_preserved": True,
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
        result = validate(
            full_archive_scan=args.full_archive_scan,
            source_preflight=args.source_preflight,
        )
    except (OSError, ValueError, json.JSONDecodeError, PackageContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
