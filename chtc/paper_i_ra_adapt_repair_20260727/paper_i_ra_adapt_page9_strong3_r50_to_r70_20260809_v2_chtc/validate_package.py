#!/usr/bin/env python3
"""Validate the inert Page-9 strong-sector continuation package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from package_contract import (  # noqa: E402
    PACKAGE_ID,
    REGIMES,
    RESOURCE_ENVELOPE,
    ROUTE_CONTRACT_SHA256,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    continuation_execution_id,
    digested,
    expected_execution_ids,
    load_json,
    verify_self_digest,
)
from run_cell import _load_job, _validate_materialization, preflight  # noqa: E402


def validate(*, worker_preflight: bool, resume_root: Path | None) -> dict:
    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("package_id") != PACKAGE_ID
        or manifest.get("status") != "passed_inert_blocked_1_of_3_resume_inputs"
        or manifest.get("row_count") != 3
        or manifest.get("execution_ids") != list(expected_execution_ids())
        or manifest.get("blocked_execution_ids")
        != [continuation_execution_id("strong_strong_u8")]
        or manifest.get("source_horizon") != SOURCE_HORIZON
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submitted") is not False
        or manifest.get("remote_stage") is not False
        or manifest.get("condor_submit") is not False
    ):
        raise PackageContractError("Inert package state drifted.")
    queue_rows = [line.split("\t") for line in (PACKAGE_DIR / "queue.tsv").read_text().splitlines()]
    if (
        len(queue_rows) != 3
        or any(len(row) != 9 for row in queue_rows)
        or [row[0] for row in queue_rows] != list(expected_execution_ids())
        or [row[4] for row in queue_rows]
        != [
            "remote_archive_preserved_materialization_pending",
            "remote_archive_preserved_materialization_pending",
            "blocked_predecessor_terminal_missing",
        ]
        or any(
            row[5:]
            != [
                str(RESOURCE_ENVELOPE["request_cpus"]),
                str(RESOURCE_ENVELOPE["request_memory_mb"]),
                str(RESOURCE_ENVELOPE["request_disk_mb"]),
                str(RESOURCE_ENVELOPE["max_runtime_seconds"]),
            ]
            for row in queue_rows
        )
    ):
        raise PackageContractError("Package queue drifted.")
    template = (PACKAGE_DIR / "submit.sub.in").read_text(encoding="utf-8")
    output = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"
    if (
        f"transfer_output_files = {output}" not in template
        or f'transfer_output_remaps = "{output}=@@REMOTE_OUTPUT_ROOT@@/' not in template
        or "output_destination" in template
        or "periodic_release = False" not in template
        or "preserve_relative_paths = False" not in template
        or (PACKAGE_DIR / "submit.sub").exists()
    ):
        raise PackageContractError("Inert submit template drifted.")

    jobs = []
    worker_receipts = []
    materialized = []
    for regime in REGIMES:
        execution_id = continuation_execution_id(regime)
        job_path = PACKAGE_DIR / "jobs" / f"{execution_id}.json"
        job, _manifest, protocol = _load_job(job_path)
        if (
            job["route_contract_sha256"] != ROUTE_CONTRACT_SHA256
            or protocol["route_contract"]["sha256"] != ROUTE_CONTRACT_SHA256
            or int(protocol["horizon"]) != TARGET_HORIZON
        ):
            raise PackageContractError(f"Route identity drifted: {execution_id}")
        jobs.append(job)
        if worker_preflight:
            worker_receipts.append(
                preflight(
                    job_path=job_path,
                    base_package_dir=(PACKAGE_DIR.parents[2] / manifest["base_package"]["path"]),
                    resume_manifest=None,
                    resume_archive=None,
                )
            )
        if resume_root is not None:
            cell = resume_root / execution_id
            resume_manifest = cell / "resume_materialization.json"
            resume_archive = cell / "resume_input.tar.gz"
            if not resume_manifest.is_file() or not resume_archive.is_file():
                raise PackageContractError(
                    f"Submission readiness blocked: missing {execution_id} resume input."
                )
            materialized.append(
                _validate_materialization(
                    job=job,
                    manifest_path=resume_manifest,
                    archive_path=resume_archive,
                )["sha256"]
            )
    audit = load_json(PACKAGE_DIR / "source_lock_audit.json", label="source lock audit")
    verify_self_digest(audit, label="source lock audit")
    if (
        audit.get("rows") is None
        or len(audit["rows"]) != 3
        or any(row.get("non_horizon_route_diff") != [] for row in audit["rows"])
    ):
        raise PackageContractError("Source-lock identity audit drifted.")
    return digested(
        {
            "schema": "paper_i_page9_strong3_r70_validation_receipt_v2",
            "status": "passed_ready" if resume_root is not None else "passed_inert_one_blocked",
            "package_id": PACKAGE_ID,
            "row_count": len(jobs),
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "worker_preflight_count": len(worker_receipts),
            "resume_materialization_count": len(materialized),
            "blocked_execution_ids": manifest["blocked_execution_ids"] if resume_root is None else [],
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": resume_root is not None,
            "submitted": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-preflight", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--resume-root", type=Path)
    args = parser.parse_args()
    try:
        if args.require_ready and args.resume_root is None:
            raise PackageContractError("--require-ready requires --resume-root.")
        if args.resume_root is not None and not args.require_ready:
            raise PackageContractError("--resume-root is valid only with --require-ready.")
        result = validate(
            worker_preflight=args.worker_preflight,
            resume_root=(None if args.resume_root is None else args.resume_root.resolve()),
        )
    except (OSError, ValueError, PackageContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
