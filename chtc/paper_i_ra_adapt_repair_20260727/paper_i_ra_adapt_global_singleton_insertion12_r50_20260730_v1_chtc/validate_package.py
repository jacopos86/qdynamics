#!/usr/bin/env python3
"""Read-only validation for the inert global-singleton package."""

from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    CALIBRATION_RECEIPT_NAME,
    CONTROL_FILES,
    DIRECT_EXECUTION_COUNT,
    EXECUTION_PLAN_NAME,
    GENERATED_FILES,
    JOB_SCHEMA,
    MANIFEST_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_NAME,
    QUEUE_NAME,
    SMOKE_RECEIPT_NAME,
    SOURCE_ARCHIVE_MANIFEST_NAME,
    SOURCE_ARCHIVE_NAME,
    PackageContractError,
    canonical_json_bytes,
    direct_execution_rows,
    load_json,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    validate_calibration_receipt,
    validate_materialization_authority,
    validate_smoke_receipt,
    verify_self_digest,
)


def _verify_binding(
    binding: Mapping[str, Any],
    *,
    base: Path,
    label: str,
) -> Path:
    relative = safe_relative_path(binding.get("path"), label=label)
    path = base / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != binding.get("sha256")
        or path.stat().st_size
        != int(binding.get("size_bytes", -1))
    ):
        raise PackageContractError(f"{label} binding drifted.")
    payload = load_json(path, label=label)
    if (
        binding.get("canonical_sha256") is not None
        and payload.get("sha256")
        != binding.get("canonical_sha256")
    ):
        raise PackageContractError(
            f"{label} canonical binding drifted."
        )
    return path


def _validate_archive(manifest: Mapping[str, Any]) -> None:
    verify_self_digest(manifest, label="source archive manifest")
    archive = manifest.get("archive")
    rows = manifest.get("members")
    if (
        not isinstance(archive, Mapping)
        or archive.get("path") != SOURCE_ARCHIVE_NAME
        or not isinstance(rows, list)
        or int(manifest.get("member_count", -1)) != len(rows)
    ):
        raise PackageContractError(
            "Source archive manifest drifted."
        )
    archive_path = PACKAGE_DIR / SOURCE_ARCHIVE_NAME
    if (
        sha256_file(archive_path) != archive.get("sha256")
        or archive_path.stat().st_size
        != int(archive.get("size_bytes", -1))
    ):
        raise PackageContractError("Source archive bytes drifted.")
    declared = {
        safe_relative_path(
            row.get("path"), label="archive member"
        ).as_posix(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(declared) != len(rows):
        raise PackageContractError(
            "Source archive member set duplicates."
        )
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive_file:
        for member in archive_file:
            relative = safe_relative_path(
                member.name, label="tar member"
            ).as_posix()
            if (
                relative in observed
                or relative not in declared
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise PackageContractError(
                    f"Unsafe source archive member: {relative}"
                )
            stream = archive_file.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"Unreadable source archive member: {relative}"
                )
            digest = hashlib.sha256()
            size = 0
            for block in iter(
                lambda: stream.read(1024 * 1024), b""
            ):
                digest.update(block)
                size += len(block)
            row = declared[relative]
            if (
                digest.hexdigest() != row.get("sha256")
                or size != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(
                    f"Source archive member drifted: {relative}"
                )
            observed.add(relative)
    if observed != set(declared):
        raise PackageContractError(
            "Source archive member closure failed."
        )


def main() -> int:
    repo_root = repo_root_from_script(__file__)
    authority = validate_materialization_authority(repo_root)
    if (PACKAGE_DIR / "authority").exists():
        raise PackageContractError(
            "The inert package contains forbidden authority."
        )
    manifest = load_json(
        PACKAGE_DIR / PACKAGE_MANIFEST_NAME,
        label="package manifest",
    )
    verify_self_digest(manifest, label="package manifest")
    smoke = load_json(
        PACKAGE_DIR / SMOKE_RECEIPT_NAME,
        label="two-round smoke receipt",
    )
    validate_smoke_receipt(smoke)
    calibration = load_json(
        PACKAGE_DIR / CALIBRATION_RECEIPT_NAME,
        label="open-plateau calibration",
    )
    validate_calibration_receipt(calibration)
    plan = load_json(
        PACKAGE_DIR / EXECUTION_PLAN_NAME,
        label="execution plan",
    )
    verify_self_digest(plan, label="execution plan")
    archive_manifest = load_json(
        PACKAGE_DIR / SOURCE_ARCHIVE_MANIFEST_NAME,
        label="source archive manifest",
    )
    _validate_archive(archive_manifest)

    false_fields = (
        "execution_authorized",
        "submission_authorized",
        "remote_stage",
        "condor_submit",
    )
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("status") != "passed"
        or manifest.get("direct_execution_count")
        != DIRECT_EXECUTION_COUNT
        or manifest.get("insertion_policy_count") != 2
        or manifest.get("resource_status")
        != "provisional_not_demonstrated"
        or manifest.get("authority_overlay_present") is not False
        or manifest.get("submission_state") != "not_submitted"
        or manifest.get("submitted") is not False
        or any(
            manifest.get(field) is not False
            for field in false_fields
        )
        or any(plan.get(field) is not False for field in false_fields)
        or plan.get("submission_state") != "not_submitted"
        or plan.get("submitted") is not False
        or plan.get("resource_status")
        != "provisional_not_demonstrated"
    ):
        raise PackageContractError("Inert package envelope drifted.")

    _verify_binding(
        manifest["source_archive"]["manifest"],
        base=PACKAGE_DIR,
        label="source archive manifest",
    )
    _verify_binding(
        manifest["execution_plan"],
        base=PACKAGE_DIR,
        label="execution plan",
    )
    _verify_binding(
        manifest["smoke_receipt"],
        base=PACKAGE_DIR,
        label="smoke receipt",
    )
    _verify_binding(
        manifest["open_plateau_calibration"],
        base=PACKAGE_DIR,
        label="open-plateau calibration",
    )
    if (
        manifest["source_archive"]["sha256"]
        != sha256_file(PACKAGE_DIR / SOURCE_ARCHIVE_NAME)
        or manifest["materialization_receipt"]
        != authority["final_binding"]
        or manifest["source_lock_delta_receipt"]
        != authority["source_lock_delta_binding"]
        or manifest["bundle_bindings"]
        != authority["bundle_bindings"]
        or manifest["cross_arm_equality_audit"]
        != authority["equality_audit"]
        or plan.get("cross_arm_equality_sha256")
        != authority["equality_audit"]["sha256"]
        or plan.get("smoke_receipt_sha256") != smoke["sha256"]
        or plan.get("open_plateau_calibration_sha256")
        != calibration["sha256"]
        or plan.get("source_lock_delta_sha256")
        != authority["source_lock_delta"]["sha256"]
    ):
        raise PackageContractError(
            "Package provenance binding drifted."
        )

    expected_rows = list(direct_execution_rows())
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(expected_rows):
        raise PackageContractError("Package job count drifted.")
    expected_ids = [row["execution_id"] for row in expected_rows]
    for row, binding in zip(expected_rows, jobs):
        path = _verify_binding(
            binding,
            base=PACKAGE_DIR,
            label=f"{row['execution_id']} job",
        )
        job = load_json(path, label=f"{row['execution_id']} job")
        verify_self_digest(job, label=f"{row['execution_id']} job")
        if (
            job.get("schema") != JOB_SCHEMA
            or job.get("execution_id") != row["execution_id"]
            or any(
                job.get(key) != value
                for key, value in row.items()
            )
            or job.get("protocol")
            != authority["protocol_bindings"][
                row["execution_id"]
            ]
            or job.get("expected_artifact_destinations")
            != authority["artifact_destinations_by_execution_id"][
                row["execution_id"]
            ]
            or job.get("resource_status")
            != (
                "provisional_not_demonstrated_by_"
                "bounded_calibration"
            )
            or job.get("execution_authorized") is not False
            or job.get("submission_authorized") is not False
            or job.get("submission_state") != "not_submitted"
            or job.get("submitted") is not False
        ):
            raise PackageContractError(
                f"Package job drifted: {row['execution_id']}"
            )

    queue_rows = (
        PACKAGE_DIR / QUEUE_NAME
    ).read_text(encoding="utf-8").splitlines()
    if (
        len(queue_rows) != DIRECT_EXECUTION_COUNT
        or [line.split("\t")[0] for line in queue_rows]
        != expected_ids
        or any(len(line.split("\t")) != 5 for line in queue_rows)
        or manifest["queue"]["sha256"]
        != sha256_file(PACKAGE_DIR / QUEUE_NAME)
        or manifest["queue"].get("neutral_item_names") is not True
    ):
        raise PackageContractError("Package queue drifted.")
    submit = (PACKAGE_DIR / "submit.sub").read_text(
        encoding="utf-8"
    )
    if (
        "queue execution_id,cpus,memory_mb,disk_mb,"
        "max_runtime_seconds from queue.tsv"
        not in submit
        or "request_cpus = $(cpus)" not in submit
        or "request_memory = $(memory_mb)MB" not in submit
        or "request_disk = $(disk_mb)MB" not in submit
        or "queue execution_id,request_" in submit
        or "$(request_memory_mb)" in submit
        or "$(request_disk_mb)" in submit
    ):
        raise PackageContractError(
            "Submit template does not use neutral queue items."
        )

    control_names = [
        row["path"] for row in manifest["control_plane"]
    ]
    if control_names != list(CONTROL_FILES):
        raise PackageContractError(
            "Control-plane allowlist drifted."
        )
    for row in manifest["control_plane"]:
        path = PACKAGE_DIR / str(row["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != row["sha256"]
            or path.stat().st_size
            != int(row["size_bytes"])
        ):
            raise PackageContractError(
                f"Control-plane file drifted: {row['path']}"
            )

    expected_files = {
        *(PACKAGE_DIR / name for name in CONTROL_FILES),
        *(PACKAGE_DIR / name for name in GENERATED_FILES),
        *(
            PACKAGE_DIR / "jobs" / f"{execution_id}.json"
            for execution_id in expected_ids
        ),
    }
    observed_files = {
        path
        for path in PACKAGE_DIR.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise PackageContractError(
            "Package recursive file allowlist drifted."
        )
    if any(
        path.is_symlink() for path in PACKAGE_DIR.rglob("*")
    ):
        raise PackageContractError(
            "Package contains a forbidden symlink."
        )

    print(
        canonical_json_bytes(
            {
                "status": "passed",
                "package_id": PACKAGE_ID,
                "package_manifest_sha256": manifest["sha256"],
                "execution_plan_sha256": plan["sha256"],
                "source_archive_sha256": manifest[
                    "source_archive"
                ]["sha256"],
                "smoke_receipt_sha256": smoke["sha256"],
                "open_plateau_calibration_sha256": (
                    calibration["sha256"]
                ),
                "cross_arm_equality_sha256": authority[
                    "equality_audit"
                ]["sha256"],
                "direct_execution_count": (
                    DIRECT_EXECUTION_COUNT
                ),
                "resource_status": (
                    "provisional_not_demonstrated"
                ),
                "execution_authorized": False,
                "submission_authorized": False,
                "remote_stage": False,
                "condor_submit": False,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
