#!/usr/bin/env python3
"""Build the inert source-locked 12-cell global-singleton package."""

from __future__ import annotations

import gzip
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ARCHIVE_SCHEMA,
    BUNDLE_ID,
    CALIBRATION_RECEIPT_NAME,
    CAMPAIGN_ID,
    CONTROL_FILES,
    DIRECT_EXECUTION_COUNT,
    EXECUTION_PLAN_NAME,
    EXECUTION_TARGET,
    HORIZON,
    JOB_SCHEMA,
    MANIFEST_SCHEMA,
    MATERIALIZER_RELATIVE_PATH,
    PACKAGE_ID,
    PACKAGE_MANIFEST_NAME,
    PLAN_SCHEMA,
    QUEUE_NAME,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    ROUTE_IDS,
    RUN_CLASS,
    SMOKE_RECEIPT_NAME,
    SOURCE_ARCHIVE_MANIFEST_NAME,
    SOURCE_ARCHIVE_NAME,
    V13_FINAL_RECEIPT_RELATIVE_PATH,
    PackageContractError,
    canonical_json_bytes,
    digested,
    direct_execution_rows,
    load_json,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    validate_calibration_receipt,
    validate_materialization_authority,
    validate_smoke_receipt,
)


def _exclusive_write(
    path: Path, data: bytes, *, executable: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PackageContractError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            temporary.chmod(0o755)
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _exclusive_write(path, canonical_json_bytes(payload) + b"\n")


def _binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = load_json(path, label=f"{path.name} binding")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "canonical_sha256": str(payload["sha256"]),
        "size_bytes": path.stat().st_size,
    }


def _source_members(
    *,
    repo_root: Path,
    authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    paths: set[str] = set()
    declared_sha256: dict[str, str] = {}
    implementation = authority["implementation_inventory"]
    files = implementation.get("files")
    if not isinstance(files, list):
        raise PackageContractError(
            "Implementation inventory has no file list."
        )
    for row in files:
        if not isinstance(row, Mapping):
            raise PackageContractError(
                "Implementation inventory row is malformed."
            )
        relative = safe_relative_path(
            row.get("path"), label="implementation source"
        ).as_posix()
        paths.add(relative)
        declared_sha256[relative] = str(row.get("sha256", ""))

    source_locks = authority["source_locks"]
    global_sources = source_locks.get("global_sources")
    if not isinstance(global_sources, Mapping):
        raise PackageContractError(
            "Source lock has no global sources."
        )
    for row in global_sources.values():
        if not isinstance(row, Mapping):
            raise PackageContractError(
                "Global source lock is malformed."
            )
        relative = safe_relative_path(
            row.get("path"), label="global source"
        ).as_posix()
        paths.add(relative)
        declared_sha256[relative] = str(row.get("sha256", ""))

    materialization_root = Path(authority["materialization_root"])
    for path in sorted(materialization_root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            paths.add(path.relative_to(repo_root).as_posix())
    paths.update(
        {
            MATERIALIZER_RELATIVE_PATH,
            V13_FINAL_RECEIPT_RELATIVE_PATH,
            "requirements.txt",
        }
    )

    members: list[dict[str, Any]] = []
    for relative in sorted(paths):
        source = repo_root / relative
        if not source.is_file() or source.is_symlink():
            raise PackageContractError(
                f"Source archive member missing or unsafe: {relative}"
            )
        if (
            relative in declared_sha256
            and sha256_file(source) != declared_sha256[relative]
        ):
            raise PackageContractError(
                f"Declared source lock drifted: {relative}"
            )
        members.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    return members


def _write_archive(
    *,
    repo_root: Path,
    destination: Path,
    members: list[dict[str, Any]],
) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.exists() or destination.is_symlink():
        raise PackageContractError(
            f"Refusing to overwrite source archive: {destination}"
        )
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    mode="w",
                    fileobj=compressed,
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for row in members:
                        relative = str(row["path"])
                        source = repo_root / relative
                        if (
                            sha256_file(source) != row["sha256"]
                            or source.stat().st_size
                            != int(row["size_bytes"])
                        ):
                            raise PackageContractError(
                                f"Archive input drifted: {relative}"
                            )
                        info = tarfile.TarInfo(relative)
                        info.size = source.stat().st_size
                        info.mode = (
                            0o755
                            if source.stat().st_mode & 0o111
                            else 0o644
                        )
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        with source.open("rb") as stream:
                            archive.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    repo_root = repo_root_from_script(__file__)
    if (PACKAGE_DIR / "authority").exists():
        raise PackageContractError(
            "The inert package must not contain authority."
        )
    for name in (
        SOURCE_ARCHIVE_NAME,
        SOURCE_ARCHIVE_MANIFEST_NAME,
        EXECUTION_PLAN_NAME,
        QUEUE_NAME,
        PACKAGE_MANIFEST_NAME,
    ):
        candidate = PACKAGE_DIR / name
        if candidate.exists() or candidate.is_symlink():
            raise PackageContractError(
                f"Refusing to overwrite generated file: {candidate}"
            )
    jobs_dir = PACKAGE_DIR / "jobs"
    if jobs_dir.exists() or jobs_dir.is_symlink():
        raise PackageContractError(
            f"Refusing to overwrite jobs: {jobs_dir}"
        )

    smoke_path = PACKAGE_DIR / SMOKE_RECEIPT_NAME
    calibration_path = PACKAGE_DIR / CALIBRATION_RECEIPT_NAME
    smoke = load_json(smoke_path, label="two-round smoke receipt")
    calibration = load_json(
        calibration_path, label="open-plateau calibration"
    )
    validate_smoke_receipt(smoke)
    validate_calibration_receipt(calibration)
    authority = validate_materialization_authority(repo_root)
    members = _source_members(
        repo_root=repo_root, authority=authority
    )
    archive_path = PACKAGE_DIR / SOURCE_ARCHIVE_NAME
    _write_archive(
        repo_root=repo_root,
        destination=archive_path,
        members=members,
    )
    archive_sha256 = sha256_file(archive_path)
    archive_manifest = digested(
        {
            "schema": ARCHIVE_SCHEMA,
            "package_id": PACKAGE_ID,
            "status": "passed",
            "archive": {
                "path": SOURCE_ARCHIVE_NAME,
                "sha256": archive_sha256,
                "size_bytes": archive_path.stat().st_size,
            },
            "member_count": len(members),
            "members": members,
            "deterministic_archive": {
                "gzip_mtime": 0,
                "tar_member_mtime": 0,
                "uid": 0,
                "gid": 0,
                "ordered_by_path": True,
            },
        }
    )
    archive_manifest_path = (
        PACKAGE_DIR / SOURCE_ARCHIVE_MANIFEST_NAME
    )
    _write_json(archive_manifest_path, archive_manifest)

    rows = list(direct_execution_rows())
    plan = digested(
        {
            "schema": PLAN_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "direct_execution_count": len(rows),
            "execution_ids": [row["execution_id"] for row in rows],
            "routes": list(ROUTE_IDS),
            "insertion_policy_count": len(ROUTE_IDS),
            "horizon": HORIZON,
            "source_archive_sha256": archive_sha256,
            "smoke_receipt_sha256": smoke["sha256"],
            "open_plateau_calibration_sha256": (
                calibration["sha256"]
            ),
            "source_lock_delta_sha256": authority[
                "source_lock_delta"
            ]["sha256"],
            "cross_arm_equality_sha256": authority[
                "equality_audit"
            ]["sha256"],
            "resource_status": "provisional_not_demonstrated",
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    plan_path = PACKAGE_DIR / EXECUTION_PLAN_NAME
    _write_json(plan_path, plan)

    jobs_dir.mkdir()
    for row in rows:
        protocol = authority["protocol_bindings"][
            row["execution_id"]
        ]
        job = digested(
            {
                "schema": JOB_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "bundle_id": BUNDLE_ID,
                **row,
                "horizon": HORIZON,
                "protocol": dict(protocol),
                "expected_artifact_destinations": authority[
                    "artifact_destinations_by_execution_id"
                ][row["execution_id"]],
                "source_archive": {
                    "path": SOURCE_ARCHIVE_NAME,
                    "sha256": archive_sha256,
                },
                "execution_plan_sha256": plan["sha256"],
                "remote_image": {
                    "path": REMOTE_IMAGE_PATH,
                    "sha256": REMOTE_IMAGE_SHA256,
                },
                "resource_status": (
                    "provisional_not_demonstrated_by_"
                    "bounded_calibration"
                ),
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_state": "not_submitted",
                "submitted": False,
            }
        )
        _write_json(
            jobs_dir / f"{row['execution_id']}.json", job
        )

    queue = "".join(
        f"{row['execution_id']}\t"
        f"{row['resources']['request_cpus']}\t"
        f"{row['resources']['request_memory_mb']}\t"
        f"{row['resources']['request_disk_mb']}\t"
        f"{row['resources']['max_runtime_seconds']}\n"
        for row in rows
    )
    _exclusive_write(
        PACKAGE_DIR / QUEUE_NAME, queue.encode("utf-8")
    )

    controls = []
    for name in CONTROL_FILES:
        path = PACKAGE_DIR / name
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Control-plane file missing or unsafe: {name}"
            )
        controls.append(
            {
                "path": name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    jobs = [
        {
            "path": path.relative_to(PACKAGE_DIR).as_posix(),
            "sha256": sha256_file(path),
            "canonical_sha256": str(
                load_json(
                    path, label=f"{path.stem} job"
                )["sha256"]
            ),
            "size_bytes": path.stat().st_size,
        }
        for row in rows
        for path in (
            jobs_dir / f"{row['execution_id']}.json",
        )
    ]
    manifest = digested(
        {
            "schema": MANIFEST_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "status": "passed",
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "direct_execution_count": DIRECT_EXECUTION_COUNT,
            "insertion_policy_count": len(ROUTE_IDS),
            "control_plane": controls,
            "source_archive": {
                "path": SOURCE_ARCHIVE_NAME,
                "sha256": archive_sha256,
                "size_bytes": archive_path.stat().st_size,
                "manifest": _binding(
                    archive_manifest_path,
                    relative_to=PACKAGE_DIR,
                ),
            },
            "execution_plan": _binding(
                plan_path, relative_to=PACKAGE_DIR
            ),
            "smoke_receipt": _binding(
                smoke_path, relative_to=PACKAGE_DIR
            ),
            "open_plateau_calibration": _binding(
                calibration_path, relative_to=PACKAGE_DIR
            ),
            "jobs": jobs,
            "queue": {
                "path": QUEUE_NAME,
                "sha256": sha256_file(
                    PACKAGE_DIR / QUEUE_NAME
                ),
                "size_bytes": (
                    PACKAGE_DIR / QUEUE_NAME
                ).stat().st_size,
                "item_names": [
                    "execution_id",
                    "cpus",
                    "memory_mb",
                    "disk_mb",
                    "max_runtime_seconds",
                ],
                "neutral_item_names": True,
            },
            "materialization_receipt": authority[
                "final_binding"
            ],
            "source_lock_delta_receipt": authority[
                "source_lock_delta_binding"
            ],
            "bundle_bindings": authority["bundle_bindings"],
            "cross_arm_equality_audit": authority[
                "equality_audit"
            ],
            "resource_status": "provisional_not_demonstrated",
            "resource_calibration_scope": (
                "bounded_depth1_open_domain_not_depth50_v1"
            ),
            "authority_overlay_present": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
            "explicit_future_user_authorization_required": True,
        }
    )
    manifest_path = PACKAGE_DIR / PACKAGE_MANIFEST_NAME
    _write_json(manifest_path, manifest)
    print(
        canonical_json_bytes(
            {
                "status": "passed",
                "package_id": PACKAGE_ID,
                "manifest_sha256": manifest["sha256"],
                "source_archive_sha256": archive_sha256,
                "smoke_receipt_sha256": smoke["sha256"],
                "open_plateau_calibration_sha256": (
                    calibration["sha256"]
                ),
                "direct_execution_count": DIRECT_EXECUTION_COUNT,
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
