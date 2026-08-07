#!/usr/bin/env python3
"""Materialize the inert exact-prefix r70 continuation package."""

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

from package_contract import (  # noqa: E402
    CAMPAIGN_ID,
    CONTROL_FILES,
    GENERATED_PATHS,
    HORIZON_CHANGED_PATHS,
    LOADER_PACKAGE_MANIFEST_CANONICAL_SHA256,
    LOADER_PACKAGE_MANIFEST_FILE_SHA256,
    LOADER_PACKAGE_RELATIVE,
    PACKAGE_ID,
    PACKAGE_RELATIVE,
    PACKAGE_SCHEMA,
    RESOURCE_ENVELOPE,
    ROUTE_CONTRACT_SHA256,
    ROUTE_PROFILE,
    RUN_CLASS,
    SCIENTIFIC_SETTINGS_CHANGED,
    SNAPSHOT_ROOT_RELATIVE,
    SOURCE_CLUSTER_ID,
    SOURCE_HORIZON,
    SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
    SOURCE_PACKAGE_RELATIVE,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_materialization,
    file_binding,
    implementation_repair,
    json_binding,
    repo_root_from_script,
)


VISIBLE_ADAPTER_RELATIVE = Path(
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "historical_mean_global_singleton_live_page7_authenticated_k49_45_31_"
    "v3_adapter.json"
)
VISIBLE_ADAPTER_FILE_SHA256 = (
    "f0ed9bc155167372e0b749079a818e534bce2d83229234c17f9338e6836a0920"
)


def _exclusive_write(path: Path, data: bytes, *, created: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PackageContractError(f"Refusing to overwrite: {path}")
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


def _source_binding(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return json_binding(path, relative_to=repo_root)


def materialize() -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    if PACKAGE_DIR != repo_root / PACKAGE_RELATIVE:
        raise PackageContractError("Package directory identity drifted.")
    for name in GENERATED_PATHS:
        path = PACKAGE_DIR / name
        if path.exists() or path.is_symlink():
            raise PackageContractError(f"Refusing to overwrite: {path}")
    visible_adapter = repo_root / VISIBLE_ADAPTER_RELATIVE
    if (
        not visible_adapter.is_file()
        or visible_adapter.is_symlink()
        or file_binding(visible_adapter, relative_to=repo_root)["sha256"]
        != VISIBLE_ADAPTER_FILE_SHA256
    ):
        raise PackageContractError("Visible page-7 source adapter drifted.")
    created: list[Path] = []
    try:
        rows = expected_materialization(repo_root, hash_archives=True)
        job_bindings: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        resume_cells: dict[str, Any] = {}
        queue_lines: list[str] = []
        for row in rows:
            identifier = str(row["execution_id"])
            protocol_path = PACKAGE_DIR / "protocols" / f"{identifier}.json"
            _write_json(
                protocol_path, row["derived_protocol"], created=created
            )
            derived_binding = json_binding(
                protocol_path, relative_to=repo_root
            )
            source_protocol_binding = _source_binding(
                row["source_protocol_path"], repo_root=repo_root
            )
            source_job_binding = _source_binding(
                row["source_job_path"], repo_root=repo_root
            )
            predecessor_binding = _source_binding(
                row["predecessor_path"], repo_root=repo_root
            )
            resume = row["resume_input"]
            job = digested(
                {
                    "schema": (
                        "paper_i_ra_adapt_historical_mean_global_singleton_"
                        "r70_resume256gb_job_v1"
                    ),
                    "package_id": PACKAGE_ID,
                    "campaign_id": CAMPAIGN_ID,
                    "execution_id": identifier,
                    "base_execution_id": row["spec"]["base_execution_id"],
                    "predecessor_execution_id": row["spec"][
                        "predecessor_execution_id"
                    ],
                    "execution_mode": "authenticated_exact_prefix_resume_to_70",
                    "run_class": RUN_CLASS,
                    "execution_target": "chtc",
                    "regime_id": row["spec"]["regime_id"],
                    "nph": 7,
                    "route_profile": ROUTE_PROFILE,
                    "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                    "source_horizon": SOURCE_HORIZON,
                    "target_horizon": TARGET_HORIZON,
                    "source_cluster_id": SOURCE_CLUSTER_ID,
                    "source_proc_id": row["spec"]["proc_id"],
                    "source_held_job_preserved": True,
                    "source_package": {
                        "path": SOURCE_PACKAGE_RELATIVE.as_posix(),
                        "manifest_sha256": (
                            SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256
                        ),
                        "manifest_file_sha256": (
                            SOURCE_PACKAGE_MANIFEST_FILE_SHA256
                        ),
                    },
                    "loader_package": {
                        "path": LOADER_PACKAGE_RELATIVE.as_posix(),
                        "manifest_sha256": (
                            LOADER_PACKAGE_MANIFEST_CANONICAL_SHA256
                        ),
                        "manifest_file_sha256": (
                            LOADER_PACKAGE_MANIFEST_FILE_SHA256
                        ),
                    },
                    "source_job": source_job_binding,
                    "predecessor_job": predecessor_binding,
                    "source_protocol": source_protocol_binding,
                    "derived_protocol": derived_binding,
                    "source_protocol_sha256": row["source_protocol"]["sha256"],
                    "derived_protocol_sha256": row["derived_protocol"]["sha256"],
                    "resume_input": resume,
                    "resources": dict(RESOURCE_ENVELOPE),
                    "implementation_repair": implementation_repair(),
                    "scientific_protocol_changed": True,
                    "scientific_settings_changed": list(
                        SCIENTIFIC_SETTINGS_CHANGED
                    ),
                    "only_scientific_change": (
                        "maximum_controller_rounds_50_to_70"
                    ),
                    "changed_protocol_paths": list(HORIZON_CHANGED_PATHS),
                    "non_swept_settings_diff": [],
                    "operational_changes": [
                        "resume_prefix_to_latest_authenticated_live_triplet",
                        "request_memory_to_262144_mb",
                        "request_disk_to_102400_mb",
                        "staging_backed_large_file_transfer",
                        "one_job_at_a_time_activation",
                    ],
                    "expected_output_root": f"runs/{identifier}",
                    "execution_authorized": False,
                    "submission_authorized": False,
                    "submitted": False,
                }
            )
            job_path = PACKAGE_DIR / "jobs" / f"{identifier}.json"
            _write_json(job_path, job, created=created)
            job_binding = json_binding(job_path, relative_to=PACKAGE_DIR)
            job_bindings.append(
                {"execution_id": identifier, **job_binding}
            )
            resume_cells[identifier] = resume
            queue_lines.append(
                "\t".join(
                    (
                        identifier,
                        job_path.relative_to(repo_root).as_posix(),
                        job_binding["sha256"],
                        resume["local_archive"]["path"],
                        resume["local_archive"]["sha256"],
                        str(RESOURCE_ENVELOPE["request_cpus"]),
                        str(RESOURCE_ENVELOPE["request_memory_mb"]),
                        str(RESOURCE_ENVELOPE["request_disk_mb"]),
                        str(RESOURCE_ENVELOPE["max_runtime_seconds"]),
                    )
                )
                + "\n"
            )
            audit_rows.append(
                {
                    "execution_id": identifier,
                    "regime_id": row["spec"]["regime_id"],
                    "resume_controller_round": row["spec"][
                        "resume_controller_round"
                    ],
                    "source_protocol": source_protocol_binding,
                    "derived_protocol": derived_binding,
                    "source_protocol_sha256": row["source_protocol"]["sha256"],
                    "derived_protocol_sha256": row["derived_protocol"]["sha256"],
                    "changed_protocol_paths": list(HORIZON_CHANGED_PATHS),
                    "non_swept_settings_diff": [],
                    "authenticated_projection": resume[
                        "authenticated_projection"
                    ],
                    "validation_receipt": resume["validation"],
                    "archive": resume["local_archive"],
                }
            )

        resume_manifest = digested(
            {
                "schema": (
                    "paper_i_ra_adapt_historical_mean_global_singleton_"
                    "r70_exact_prefix_inputs_manifest_v1"
                ),
                "status": "passed",
                "package_id": PACKAGE_ID,
                "source_cluster_id": SOURCE_CLUSTER_ID,
                "snapshot_root": SNAPSHOT_ROOT_RELATIVE.as_posix(),
                "cell_count": len(rows),
                "resume_controller_rounds": [
                    row["spec"]["resume_controller_round"] for row in rows
                ],
                "cells": resume_cells,
                "archive_bytes_duplicated_locally": False,
            }
        )
        _write_json(
            PACKAGE_DIR / "resume_inputs_manifest.json",
            resume_manifest,
            created=created,
        )
        audit = digested(
            {
                "schema": (
                    "paper_i_ra_adapt_historical_mean_global_singleton_"
                    "r70_source_lock_audit_v1"
                ),
                "status": "passed",
                "package_id": PACKAGE_ID,
                "visible_source_adapter": file_binding(
                    visible_adapter, relative_to=repo_root
                ),
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "only_scientific_change": (
                    "maximum_controller_rounds_50_to_70"
                ),
                "changed_protocol_paths": list(HORIZON_CHANGED_PATHS),
                "non_swept_settings_diff": [],
                "route_profile": ROUTE_PROFILE,
                "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "implementation_repair": implementation_repair(),
                "rows": audit_rows,
                "paper_evidence_adopted": False,
            }
        )
        _write_json(
            PACKAGE_DIR / "source_lock_audit.json", audit, created=created
        )
        plan = digested(
            {
                "schema": (
                    "paper_i_ra_adapt_historical_mean_global_singleton_"
                    "r70_execution_plan_v1"
                ),
                "status": "passed_inert",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "run_class": RUN_CLASS,
                "execution_target": "chtc",
                "execution_ids": [row["execution_id"] for row in rows],
                "row_count": len(rows),
                "activation_order": [
                    "weak_strong",
                    "intermediate_strong",
                    "strong_strong_u8",
                ],
                "activation_cardinality": 1,
                "source_cluster_id": SOURCE_CLUSTER_ID,
                "source_held_jobs_preserved": True,
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "only_scientific_change": (
                    "maximum_controller_rounds_50_to_70"
                ),
                "changed_protocol_paths": list(HORIZON_CHANGED_PATHS),
                "non_swept_settings_diff": [],
                "resources": dict(RESOURCE_ENVELOPE),
                "large_file_transport": "one_row_osdf_staging_v1",
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_ready": False,
                "submitted": False,
            }
        )
        _write_json(PACKAGE_DIR / "execution_plan.json", plan, created=created)
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
                "status": "passed_inert_three_authenticated_r70_resumes",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "run_class": RUN_CLASS,
                "row_count": len(rows),
                "execution_ids": [row["execution_id"] for row in rows],
                "control_files": controls,
                "jobs": job_bindings,
                "resume_inputs_manifest": json_binding(
                    PACKAGE_DIR / "resume_inputs_manifest.json",
                    relative_to=PACKAGE_DIR,
                ),
                "source_lock_audit": json_binding(
                    PACKAGE_DIR / "source_lock_audit.json",
                    relative_to=PACKAGE_DIR,
                ),
                "execution_plan": json_binding(
                    PACKAGE_DIR / "execution_plan.json",
                    relative_to=PACKAGE_DIR,
                ),
                "queue": file_binding(queue_path, relative_to=PACKAGE_DIR),
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "only_scientific_change": (
                    "maximum_controller_rounds_50_to_70"
                ),
                "changed_protocol_paths": list(HORIZON_CHANGED_PATHS),
                "non_swept_settings_diff": [],
                "implementation_repair": implementation_repair(),
                "source_held_jobs_preserved": True,
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_ready": False,
                "submitted": False,
            }
        )
        _write_json(
            PACKAGE_DIR / "package_manifest.json", manifest, created=created
        )
        return {
            "status": manifest["status"],
            "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "row_count": len(rows),
            "resume_controller_rounds": [
                row["spec"]["resume_controller_round"] for row in rows
            ],
        }
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        for directory in (PACKAGE_DIR / "jobs", PACKAGE_DIR / "protocols"):
            if directory.is_dir() and not directory.is_symlink():
                try:
                    directory.rmdir()
                except OSError:
                    pass
        raise


if __name__ == "__main__":
    print(canonical_json_bytes(materialize()).decode("ascii"))
