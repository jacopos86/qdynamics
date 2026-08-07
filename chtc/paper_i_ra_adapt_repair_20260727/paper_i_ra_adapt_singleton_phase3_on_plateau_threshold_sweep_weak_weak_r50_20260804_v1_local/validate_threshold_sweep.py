#!/usr/bin/env python3
"""Validate the inert three-package CHTC threshold calibration closure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materialize_threshold_sweep import (  # noqa: E402
    EXPECTED_DERIVATIVE_HASHES,
    FANOUT_MANIFEST_PATH,
    LOCAL_ANCHOR_FAILURE_PATH,
    PLAN_PATH,
    REMOTE_IMAGE,
    SOURCE_PACKAGE,
    SOURCE_PROTOCOL_RELATIVE,
    SOURCE_EXECUTION_ID,
    SOURCE_FILE_RELATIVE,
    SOURCE_THRESHOLD,
    SweepError,
    _scalar_differences,
    canonical_json_bytes,
    digested,
    load_json,
    package_dir,
    sha256_file,
    verify_self_digest,
    write_json,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SweepError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SweepError(f"{label} must be a list.")
    return value


def _bound_file(
    root: Path,
    raw: Any,
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    row = _mapping(raw, label=f"{label} binding")
    path = root / str(row.get("path", ""))
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SweepError(f"{label} escaped package.") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("size_bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise SweepError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != row.get("canonical_sha256"):
        raise SweepError(f"{label} canonical binding drifted.")
    return path, payload


def _preflight(package: Path, job: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(package / "run_cell.py"),
            "--preflight",
            "--job",
            str(job),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "off",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SweepError(
            f"Isolated preflight failed for {package.name}: "
            f"{completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout.splitlines()[-1])
    if payload.get("status") != "passed":
        raise SweepError(f"Preflight did not pass for {package.name}.")
    verify_self_digest(payload, label=f"{package.name} preflight")
    return payload


def validate_package(threshold: float) -> dict[str, Any]:
    package = package_dir(threshold)
    if (package / "authorizations").exists():
        raise SweepError("Inert package contains an authorization overlay.")
    forbidden = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    ]
    if forbidden:
        raise SweepError(f"Unbound package bytecode exists: {forbidden}")
    manifest = load_json(package / "package_manifest.json", label="manifest")
    verify_self_digest(manifest, label="manifest")
    expected = EXPECTED_DERIVATIVE_HASHES[threshold]
    if (
        manifest.get("status") != "passed_inert_one_row"
        or manifest.get("row_count") != 1
        or manifest.get("execution_ids") != [SOURCE_EXECUTION_ID]
        or manifest.get("threshold") != threshold
        or manifest.get("execution_target") != "chtc"
        or manifest.get("remote_image") != REMOTE_IMAGE
        or manifest.get("fanout_execution_blocked_pending_chtc_anchor")
        != (threshold != SOURCE_THRESHOLD)
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
    ):
        raise SweepError(f"Inert manifest drifted for threshold {threshold}.")
    for index, raw in enumerate(
        _sequence(manifest.get("control_files"), label="control files")
    ):
        _bound_file(package, raw, label=f"control file {index}")
    control_path, _ = _bound_file(
        package,
        next(
            row
            for row in manifest["control_files"]
            if row["path"] == "package_control.json"
        ),
        label="package control",
    )
    control = load_json(control_path, label="package control")
    verify_self_digest(control, label="package control")
    if (
        control.get("threshold") != threshold
        or control.get("execution_target") != "chtc"
        or control.get("route_contract_sha256")
        != expected["route_contract_sha256"]
        or control.get("parent_route_contract_sha256")
        != expected["parent_route_contract_sha256"]
    ):
        raise SweepError("Package control hashes drifted.")
    archive_path, _ = _bound_file(
        package, manifest.get("source_archive"), label="source archive"
    )
    _, archive_manifest = _bound_file(
        package,
        manifest.get("source_archive_manifest"),
        label="source archive manifest",
        canonical=True,
    )
    assert archive_manifest is not None
    rows = _sequence(archive_manifest.get("members"), label="source members")
    by_path = {
        str(row["path"]): row for row in rows if isinstance(row, Mapping)
    }
    patch_rows = archive_manifest.get("source_patch")
    if not isinstance(patch_rows, list):
        raise SweepError("Source patch rows are malformed.")
    changed = {
        str(row["path"])
        for row in patch_rows
        if isinstance(row, Mapping)
    }
    expected_changed = (
        set()
        if threshold == SOURCE_THRESHOLD
        else {SOURCE_FILE_RELATIVE.as_posix()}
    )
    if (
        len(rows) != 165
        or len(by_path) != 165
        or by_path[SOURCE_FILE_RELATIVE.as_posix()]["sha256"]
        != expected["source_member_sha256"]
        or changed != expected_changed
        or archive_manifest.get("implementation_source_inventory_sha256")
        != expected["implementation_inventory_sha256"]
    ):
        raise SweepError("Source archive/member delta drifted.")
    observed_tar: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name in observed_tar or not member.isfile():
                raise SweepError("Unsafe/duplicate tar member.")
            observed_tar.add(member.name)
    if observed_tar != set(by_path):
        raise SweepError("Tar membership differs from its manifest.")
    _, audit = _bound_file(
        package,
        manifest.get("source_lock_audit"),
        label="source-lock audit",
        canonical=True,
    )
    assert audit is not None
    if (
        audit.get("status") != "passed"
        or audit.get("non_swept_settings_diff") != []
        or set(audit.get("source_archive_changed_paths", []))
        != expected_changed
        or set(audit.get("implementation_inventory_changed_paths", []))
        != expected_changed
    ):
        raise SweepError("Source-lock equality audit drifted.")
    jobs = _sequence(manifest.get("jobs"), label="jobs")
    if len(jobs) != 1:
        raise SweepError("Package does not contain exactly one job.")
    job_path, job = _bound_file(
        package, jobs[0], label="job", canonical=True
    )
    assert job is not None
    if (
        job.get("threshold", job.get("plateau_prior_mean_decrease_ratio_threshold"))
        != threshold
        or job.get("execution_target") != "chtc"
        or job.get("resources")
        != {
            "request_cpus": 4,
            "request_memory_mb": 24576,
            "request_disk_mb": 40960,
            "max_runtime_seconds": 259200,
        }
    ):
        raise SweepError("Job settings/resources drifted.")
    source_protocol = load_json(
        SOURCE_PACKAGE / SOURCE_PROTOCOL_RELATIVE,
        label="source protocol",
    )
    target_protocol = load_json(
        package / str(job["protocol_path"]),
        label="target protocol",
    )
    operational_paths = {
        ("bundle_id",),
        ("bundle_manifest_sha256",),
        ("bundle_materialization", "bundle_id"),
        ("bundle_materialization", "bundle_manifest_sha256"),
        ("bundle_materialization", "sha256"),
        ("sha256",),
    }
    threshold_paths = {
        ("bundle_materialization", "source_lock_refs_sha256"),
        ("bundle_materialization", "source_locks_sha256"),
        ("lineage_authority", "parent_contract_sha256"),
        ("route_contract", "lineage_authority", "parent_contract_sha256"),
        (
            "route_contract",
            "semantic_invariants",
            "plateau_prior_mean_decrease_ratio_threshold",
        ),
        ("route_contract", "sha256"),
        ("source_locks", "implementation_source_inventory_sha256"),
        ("source_locks", "source_locks_manifest_sha256"),
    }
    observed_protocol_paths = {
        path
        for path, _before, _after in _scalar_differences(
            source_protocol, target_protocol
        )
    }
    expected_protocol_paths = operational_paths | (
        set() if threshold == SOURCE_THRESHOLD else threshold_paths
    )
    if observed_protocol_paths != expected_protocol_paths:
        raise SweepError(
            f"Protocol diff allowlist drifted: {sorted(observed_protocol_paths)}"
        )
    preflight = _preflight(package, job_path)
    return {
        "threshold": threshold,
        "package": package.name,
        "package_manifest_sha256": manifest["sha256"],
        "source_archive_sha256": sha256_file(archive_path),
        "route_contract_sha256": expected["route_contract_sha256"],
        "preflight_sha256": preflight["sha256"],
        "protocol_difference_paths": [
            list(path) for path in sorted(observed_protocol_paths)
        ],
        "status": "passed",
    }


def validate() -> dict[str, Any]:
    local_failure = load_json(
        LOCAL_ANCHOR_FAILURE_PATH, label="local anchor failure"
    )
    verify_self_digest(local_failure, label="local anchor failure")
    plan = load_json(PLAN_PATH, label="sensitivity plan")
    verify_self_digest(plan, label="sensitivity plan")
    fanout = load_json(FANOUT_MANIFEST_PATH, label="fanout manifest")
    verify_self_digest(fanout, label="fanout manifest")
    if (
        local_failure.get("status")
        != "diagnostic_invalid_environment_divergence"
        or local_failure.get("local_variants_launched") is not False
        or plan.get("status") != "anchor_pending"
        or plan.get("anchor", {}).get("status") != "pending_chtc_execution"
        or fanout.get("status")
        != "passed_inert_two_rows_awaiting_chtc_anchor"
        or fanout.get("anchor_comparison_required_before_activation") is not True
        or fanout.get("execution_authorized") is not False
    ):
        raise SweepError("Sweep-level anchor/fan-out gate drifted.")
    rows = [validate_package(value) for value in (1.0e-4, 1.0e-5, 1.0e-6)]
    return digested(
        {
            "schema": "source_locked_sensitivity_three_package_validation_v1",
            "status": "passed",
            "local_anchor_status": local_failure["status"],
            "chtc_anchor_status": "pending",
            "variant_activation_status": "blocked_pending_chtc_anchor",
            "remote_image": dict(REMOTE_IMAGE),
            "rows": rows,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = validate()
        if args.output is not None:
            write_json(args.output.resolve(), payload)
    except (OSError, SweepError, ValueError, KeyError, StopIteration) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
