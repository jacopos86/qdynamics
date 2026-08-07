#!/usr/bin/env python3
"""Materialize the inert three-row 128-GiB accepted-resume package."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from resume_contract import (  # noqa: E402
    CAMPAIGN_ID,
    CONTROL_FILES,
    GENERATED_PATHS,
    IMPLEMENTATION_REPAIR_ID,
    INPUTS_RELATIVE,
    PACKAGE_ID,
    PACKAGE_RELATIVE,
    PACKAGE_SCHEMA,
    ROUTE_CONTRACT_SHA256,
    ROUTE_PROFILE,
    RESUME_LOADER_AFTER_SHA256,
    RESUME_LOADER_BEFORE_SHA256,
    RESUME_LOADER_PATCH_PATH,
    RUN_CLASS,
    SOURCE_CLUSTER_ID,
    SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
    SOURCE_PACKAGE_RELATIVE,
    TARGET_HORIZON,
    ResumeContractError,
    canonical_json_bytes,
    digested,
    expected_jobs,
    file_binding,
    json_binding,
    repo_root_from_script,
)


def _exclusive_write(path: Path, data: bytes, *, created: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ResumeContractError(f"Refusing to overwrite: {path}")
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


def materialize() -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    if PACKAGE_DIR != repo_root / PACKAGE_RELATIVE:
        raise ResumeContractError("Package directory identity drifted.")
    for name in GENERATED_PATHS:
        path = PACKAGE_DIR / name
        if path.exists() or path.is_symlink():
            raise ResumeContractError(f"Refusing to overwrite: {path}")
    created: list[Path] = []
    try:
        jobs = expected_jobs(repo_root, hash_archives=True)
        job_bindings: list[dict[str, Any]] = []
        queue_lines: list[str] = []
        resume_cells: dict[str, Any] = {}
        for job in jobs:
            identifier = str(job["execution_id"])
            job_path = PACKAGE_DIR / "jobs" / f"{identifier}.json"
            _write_json(job_path, job, created=created)
            binding = json_binding(job_path, relative_to=PACKAGE_DIR)
            job_bindings.append(
                {"execution_id": identifier, **binding}
            )
            resources = job["resources"]
            resume = job["resume_input"]
            queue_lines.append(
                "\t".join(
                    (
                        identifier,
                        job_path.relative_to(repo_root).as_posix(),
                        binding["sha256"],
                        str(resume["archive"]["path"]),
                        str(resume["archive"]["sha256"]),
                        str(resources["request_cpus"]),
                        str(resources["request_memory_mb"]),
                        str(resources["request_disk_mb"]),
                        str(resources["max_runtime_seconds"]),
                    )
                )
                + "\n"
            )
            resume_cells[identifier] = resume

        resume_manifest = digested(
            {
                "schema": (
                    "paper_i_ra_adapt_historical_mean_global_singleton_"
                    "resume128gb_inputs_manifest_v1"
                ),
                "status": "passed",
                "package_id": PACKAGE_ID,
                "source_cluster_id": SOURCE_CLUSTER_ID,
                "input_root": INPUTS_RELATIVE.as_posix(),
                "cell_count": len(jobs),
                "cells": resume_cells,
                "local_storage_reuse": "hardlinks_to_validated_snapshots",
                "archive_bytes_duplicated_locally": False,
            }
        )
        _write_json(
            PACKAGE_DIR / "resume_inputs_manifest.json",
            resume_manifest,
            created=created,
        )
        plan = digested(
            {
                "schema": (
                    "paper_i_ra_adapt_historical_mean_global_singleton_"
                    "resume128gb_execution_plan_v1"
                ),
                "status": "passed_inert",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "run_class": RUN_CLASS,
                "execution_target": "chtc",
                "execution_ids": [job["execution_id"] for job in jobs],
                "row_count": len(jobs),
                "source_cluster_id": SOURCE_CLUSTER_ID,
                "source_held_jobs_preserved": True,
                "target_horizon": TARGET_HORIZON,
                "route_profile": ROUTE_PROFILE,
                "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "scientific_protocol_changed": False,
                "scientific_settings_changed": [],
                "implementation_repair": {
                    "repair_id": IMPLEMENTATION_REPAIR_ID,
                    "path": RESUME_LOADER_PATCH_PATH,
                    "before_sha256": RESUME_LOADER_BEFORE_SHA256,
                    "after_sha256": RESUME_LOADER_AFTER_SHA256,
                    "scientific_protocol_changed": False,
                    "scientific_settings_changed": [],
                },
                "resume_policy": "accepted_state_resume",
                "request_memory_mb": 131_072,
                "request_disk_mb": 81_920,
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_ready": False,
                "submitted": False,
            }
        )
        _write_json(
            PACKAGE_DIR / "execution_plan.json", plan, created=created
        )
        queue_path = PACKAGE_DIR / "queue.tsv"
        _exclusive_write(
            queue_path, "".join(queue_lines).encode("utf-8"), created=created
        )
        controls = [
            file_binding(PACKAGE_DIR / name, relative_to=PACKAGE_DIR)
            for name in CONTROL_FILES
        ]
        manifest = digested(
            {
                "schema": PACKAGE_SCHEMA,
                "status": "passed_inert_three_authenticated_resumes",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "run_class": RUN_CLASS,
                "row_count": len(jobs),
                "execution_ids": [job["execution_id"] for job in jobs],
                "source_package": {
                    "path": SOURCE_PACKAGE_RELATIVE.as_posix(),
                    "manifest_sha256": SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
                    "manifest_file_sha256": SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
                },
                "control_files": controls,
                "jobs": job_bindings,
                "resume_inputs_manifest": json_binding(
                    PACKAGE_DIR / "resume_inputs_manifest.json",
                    relative_to=PACKAGE_DIR,
                ),
                "execution_plan": json_binding(
                    PACKAGE_DIR / "execution_plan.json",
                    relative_to=PACKAGE_DIR,
                ),
                "queue": file_binding(queue_path, relative_to=PACKAGE_DIR),
                "scientific_protocol_changed": False,
                "scientific_settings_changed": [],
                "implementation_repair": {
                    "repair_id": IMPLEMENTATION_REPAIR_ID,
                    "path": RESUME_LOADER_PATCH_PATH,
                    "before_sha256": RESUME_LOADER_BEFORE_SHA256,
                    "after_sha256": RESUME_LOADER_AFTER_SHA256,
                    "scientific_protocol_changed": False,
                    "scientific_settings_changed": [],
                },
                "source_held_jobs_preserved": True,
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_ready": False,
                "submitted": False,
            }
        )
        _write_json(
            PACKAGE_DIR / "package_manifest.json",
            manifest,
            created=created,
        )
        return {
            "status": manifest["status"],
            "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "row_count": len(jobs),
        }
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        jobs_dir = PACKAGE_DIR / "jobs"
        if jobs_dir.is_dir() and not jobs_dir.is_symlink():
            try:
                jobs_dir.rmdir()
            except OSError:
                pass
        raise


if __name__ == "__main__":
    print(canonical_json_bytes(materialize()).decode("ascii"))
