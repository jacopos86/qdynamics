#!/usr/bin/env python3
"""Validate the repaired three-cell nph7 global-singleton package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ALLOWED_SOURCE_TO_TARGET_DIFF_PATHS,
    CONTROL_FILES,
    EXECUTION_MODE,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REGIME_ROWS,
    RUNTIME_SOURCE_ARCHIVE_FILE_SHA256,
    SOURCE_IMPLEMENTATION_INVENTORY_SHA256,
    SOURCE_PATCH_BINDINGS,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    expected_source_cell_ids,
    load_json,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)
from run_cell import _load_closed_job  # noqa: E402


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageContractError(f"{label} must be a list.")
    return value


def _bound_file(raw: Any, *, label: str, canonical: bool = False) -> tuple[Path, Any]:
    row = _mapping(raw, label=f"{label} binding")
    relative = safe_relative_path(row.get("path"), label=f"{label} path")
    path = PACKAGE_DIR / relative
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped package.") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("size_bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != row.get("canonical_sha256"):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _subprocess_worker(flag: str, job_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "run_cell.py"),
            flag,
            "--job",
            str(job_path),
        ],
        cwd=PACKAGE_DIR,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "off",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PackageContractError(
            f"Worker {flag} failed for {job_path.name}: {completed.stderr.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines or any(not line.startswith("AI_LOG ") for line in lines[:-1]):
        raise PackageContractError("Worker emitted unauthenticated stdout noise.")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise PackageContractError(
            "Worker final stdout line is not a JSON receipt."
        ) from exc
    if not isinstance(payload, dict) or not str(payload.get("status", "")).startswith("passed"):
        raise PackageContractError(f"Worker {flag} did not pass.")
    verify_self_digest(payload, label=f"worker {flag} receipt")
    return payload


def validate_package(
    *, deep: bool = False, smoke_one_round: bool = False
) -> dict[str, Any]:
    forbidden = [
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in PACKAGE_DIR.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    ]
    if forbidden:
        raise PackageContractError(f"Unbound bytecode present: {forbidden}")
    if (PACKAGE_DIR / "authorizations").exists():
        raise PackageContractError("Inert package must not contain authorizations.")

    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    expected_ids = list(expected_execution_ids())
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_three_rows"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("row_count") != 3
        or manifest.get("execution_ids") != expected_ids
        or manifest.get("source_cell_ids") != list(expected_source_cell_ids())
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submit_template_present") is not True
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Package manifest semantic closure drifted.")

    controls = _sequence(manifest.get("control_files"), label="control bindings")
    if [row.get("path") for row in controls if isinstance(row, Mapping)] != list(CONTROL_FILES):
        raise PackageContractError("Control-file order or closure drifted.")
    for row in controls:
        control_path, _payload = _bound_file(
            row, label=f"control {row.get('path')}"
        )
        if row.get("path") == "execute_authorized_job.sh" and not os.access(
            control_path, os.X_OK
        ):
            raise PackageContractError("Worker wrapper is not executable.")

    canonical_names = (
        "protocol_bundle_manifest",
        "source_locks_snapshot",
        "source_archive_manifest",
        "source_lock_audit",
        "execution_plan",
    )
    documents: dict[str, Mapping[str, Any]] = {}
    for name in canonical_names:
        _path, payload = _bound_file(manifest.get(name), label=name, canonical=True)
        assert isinstance(payload, Mapping)
        documents[name] = payload
    archive_path, _ = _bound_file(manifest.get("source_archive"), label="source archive")
    queue_path, _ = _bound_file(manifest.get("queue"), label="queue")

    plan = documents["execution_plan"]
    audit = documents["source_lock_audit"]
    bundle = documents["protocol_bundle_manifest"]
    locks = documents["source_locks_snapshot"]
    source_manifest = documents["source_archive_manifest"]
    if (
        plan.get("execution_ids") != expected_ids
        or plan.get("row_count") != 3
        or plan.get("execution_mode") != EXECUTION_MODE
        or plan.get("ordinary_cluster") is not True
        or plan.get("bounded_factory") is not False
        or plan.get("success_rows_leave_queue") is not False
        or plan.get("per_job_checkpoint") is not True
        or plan.get("per_job_estimator_ledger") is not True
        or plan.get("execution_authorized") is not False
        or plan.get("submission_ready") is not False
        or audit.get("status") != "passed"
        or audit.get("cell_count") != 3
        or bundle.get("cells") is None
        or len(bundle["cells"]) != 3
        or bundle.get("execution_authorized") is not False
        or locks.get("implementation_sources", {}).get("sha256")
        != plan.get("implementation_source_inventory_sha256", locks.get("implementation_sources", {}).get("sha256"))
    ):
        raise PackageContractError("Plan/audit/protocol-bundle closure drifted.")

    allowed_paths = {tuple(path) for path in ALLOWED_SOURCE_TO_TARGET_DIFF_PATHS}
    audit_rows = _sequence(audit.get("rows"), label="audit rows")
    if [row.get("execution_id") for row in audit_rows if isinstance(row, Mapping)] != expected_ids:
        raise PackageContractError("Source-lock audit row order drifted.")
    for row in audit_rows:
        if (
            row.get("status") != "passed"
            or {
                tuple(item["path"])
                for item in row.get("source_to_target_differences", [])
            }
            != allowed_paths
        ):
            raise PackageContractError("Source-to-target delta audit drifted.")

    member_rows = _sequence(source_manifest.get("members"), label="source members")
    declared = {
        safe_relative_path(row.get("path"), label="source member").as_posix(): row
        for row in member_rows
        if isinstance(row, Mapping)
    }
    expected_globals = sorted(
        str(row["path"])
        for row in _mapping(locks.get("global_sources"), label="global sources").values()
    )
    observed: set[str] = set()
    if (
        source_manifest.get("status") != "passed"
        or source_manifest.get("archive") != manifest.get("source_archive")
        or source_manifest.get("member_count") != len(member_rows)
        or len(declared) != len(member_rows)
        or source_manifest.get("global_source_paths") != expected_globals
        or source_manifest.get("runtime_path_dependencies") != ["requirements.txt"]
        or source_manifest.get("implementation_source_inventory_sha256")
        != SOURCE_IMPLEMENTATION_INVENTORY_SHA256
        or source_manifest.get("predecessor_archive_sha256")
        != RUNTIME_SOURCE_ARCHIVE_FILE_SHA256
        or any(path not in declared for path in (*expected_globals, "requirements.txt"))
    ):
        raise PackageContractError("Source archive manifest drifted.")
    expected_patch = [
        {"path": path, "before_sha256": before, "after_sha256": after}
        for path, before, after in SOURCE_PATCH_BINDINGS
    ]
    if (
        source_manifest.get("source_patch") != expected_patch
        or audit.get("source_patch") != expected_patch
        or audit.get("runtime_source_change")
        != (
            "reporting_identity_authority_accepts_named_global_singleton_"
            "insertion_algorithms_v1"
        )
        or audit.get("scientific_changes")
        != [
            "plateau_trigger_from_fixed_absolute_v1_to_prior_accepted_mean_ratio_v2"
        ]
    ):
        raise PackageContractError("Exact v2 runtime/source-delta audit drifted.")
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            row = declared.get(member.name)
            if (
                row is None
                or member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(f"Unsafe source archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError("Unreadable source archive member.")
            import hashlib

            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != row.get("sha256"):
                raise PackageContractError(f"Source member hash drifted: {member.name}")
            observed.add(member.name)
    if observed != set(declared):
        raise PackageContractError("Source archive exact member closure failed.")

    jobs: list[dict[str, Any]] = []
    if len(expected_ids) != len(REGIME_ROWS):
        raise PackageContractError("Execution/regime row cardinality drifted.")
    for execution_id, (regime_id, nph) in zip(expected_ids, REGIME_ROWS):
        job_path = PACKAGE_DIR / "jobs" / f"{execution_id}.json"
        job, _manifest, protocol, _source_locks = _load_closed_job(job_path)
        if (
            job.get("regime_id") != regime_id
            or job.get("nph") != nph
            or protocol.get("horizon") != TARGET_HORIZON
        ):
            raise PackageContractError(f"Cell identity drifted: {execution_id}")
        jobs.append(job)

    expected_queue = "".join(
        "\t".join(
            (
                str(job["execution_id"]),
                str(job["job_path"]),
                str(job["protocol_path"]),
                str(job["sha256"]),
                str(job["resources"]["request_cpus"]),
                str(job["resources"]["request_memory_mb"]),
                str(job["resources"]["request_disk_mb"]),
                str(job["resources"]["max_runtime_seconds"]),
            )
        )
        + "\n"
        for job in jobs
    )
    if queue_path.read_text(encoding="utf-8") != expected_queue:
        raise PackageContractError("Queue rows drifted from the three jobs.")

    submit = (PACKAGE_DIR / "submit.sub.in").read_text(encoding="utf-8")
    required_fragments = (
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "preserve_relative_paths = True",
        "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
        "leave_in_queue = False",
        "queue execution_id, job_path, protocol_path, job_sha256, request_cpus, request_memory_mb, request_disk_mb, max_runtime_seconds from queue.tsv",
    )
    if any(fragment not in submit for fragment in required_fragments) or "max_materialize" in submit.lower():
        raise PackageContractError("Ordinary-cluster submit template drifted.")

    preflights: list[dict[str, Any]] = []
    if deep:
        for execution_id in expected_ids:
            preflights.append(
                _subprocess_worker(
                    "--preflight", PACKAGE_DIR / "jobs" / f"{execution_id}.json"
                )
            )
    smoke = None
    if smoke_one_round:
        smoke = _subprocess_worker(
            "--smoke-one-round",
            PACKAGE_DIR / "jobs" / f"{expected_ids[0]}.json",
        )
    return digested(
        {
            "schema": "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_package_validation_v2",
            "status": "passed",
            "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "row_count": 3,
            "execution_ids": expected_ids,
            "source_archive_member_count": len(declared),
            "global_source_count": len(expected_globals),
            "deep_preflight_count": len(preflights),
            "real_one_round_smoke": smoke,
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--smoke-one-round", action="store_true")
    args = parser.parse_args()
    try:
        payload = validate_package(
            deep=args.deep, smoke_one_round=args.smoke_one_round
        )
    except (OSError, PackageContractError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
