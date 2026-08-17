#!/usr/bin/env python3
"""Validate the inert matched singleton-12 package without executing science."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import *  # noqa: E402,F403
import run_cell as worker  # noqa: E402


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _verify_binding(raw: Any, *, label: str, canonical: bool) -> tuple[Path, dict[str, Any] | None]:
    row = _mapping(raw, label=f"{label} binding")
    path = PACKAGE_DIR / safe_relative_path(row.get("path"), label=f"{label} path")
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped the package.") from exc
    if (
        path.is_symlink()
        or not path.is_file()
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


def validate(*, worker_preflight: bool = False) -> dict[str, Any]:
    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_matched_singleton12"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("row_count") != 12
        or manifest.get("execution_ids") != list(expected_execution_ids())
        or manifest.get("methods") != list(METHODS)
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("paper_adoption_authorized") is not False
        or manifest.get("paper_evidence_adoption_authorized") is not False
        or manifest.get("operational_checkpoint_overlay", {}).get(
            "execution_source_policy"
        )
        != EXECUTION_SOURCE_POLICY
        or manifest.get("operational_checkpoint_overlay", {}).get(
            "sealed_resume_reader_sha256"
        )
        != SEALED_RESUME_READER_SHA256
        or manifest.get("operational_checkpoint_overlay", {}).get(
            "ambient_resume_overlay"
        )
        is not False
        or manifest.get("operational_checkpoint_overlay", {}).get(
            "checkpoint_usage"
        )
        != CHECKPOINT_USAGE
        or manifest.get("operational_checkpoint_overlay", {}).get(
            "checkpoint_resume_authorized"
        )
        is not False
        or manifest.get("operational_checkpoint_overlay", {}).get(
            "parity_canary_scope"
        )
        != PARITY_CANARY_SCOPE
        or manifest.get("operational_checkpoint_overlay", {}).get(
            "multi_round_compact_tail_resume_validated"
        )
        is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Package manifest identity drifted.")

    for key in (
        "bundle_manifest",
        "bundle_expected_artifacts",
        "bundle_source_locks",
        "bundle_validation_report",
        "source_archive_manifest",
        "execution_plan",
    ):
        _verify_binding(manifest.get(key), label=key, canonical=True)
    _verify_binding(manifest.get("source_archive"), label="source archive", canonical=False)
    _verify_binding(manifest.get("queue"), label="queue", canonical=False)
    for index, row in enumerate(manifest.get("control_files", [])):
        _verify_binding(row, label=f"control file {index}", canonical=False)

    _source_path, source_manifest = _verify_binding(
        manifest.get("source_archive_manifest"),
        label="source archive manifest",
        canonical=True,
    )
    _plan_path, execution_plan = _verify_binding(
        manifest.get("execution_plan"), label="execution plan", canonical=True
    )
    assert source_manifest is not None and execution_plan is not None
    if (
        source_manifest.get("archive_construction_no_ambient_repo_imports")
        is not True
        or source_manifest.get("execution_source_policy")
        != EXECUTION_SOURCE_POLICY
        or source_manifest.get("post_extraction_overlay_count") != 1
        or source_manifest.get("sealed_resume_reader", {}).get("sha256")
        != SEALED_RESUME_READER_SHA256
        or source_manifest.get("sealed_resume_reader", {}).get(
            "ambient_resume_overlay"
        )
        is not False
        or execution_plan.get("execution_source_policy")
        != EXECUTION_SOURCE_POLICY
        or execution_plan.get("fresh_start_only") is not True
        or execution_plan.get("checkpoint_usage") != CHECKPOINT_USAGE
        or execution_plan.get("checkpoint_resume_authorized") is not False
        or execution_plan.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
    ):
        raise PackageContractError("Checkpoint execution authority drifted.")

    protocol_rows = manifest.get("protocols")
    job_rows = manifest.get("jobs")
    if not isinstance(protocol_rows, list) or not isinstance(job_rows, list):
        raise PackageContractError("Protocol/job closures are absent.")
    if len(protocol_rows) != 12 or len(job_rows) != 12:
        raise PackageContractError("Protocol/job closure cardinality drifted.")
    methods: Counter[str] = Counter()
    jobs: dict[str, Path] = {}
    for index, row in enumerate(protocol_rows):
        _path, payload = _verify_binding(row, label=f"protocol {index}", canonical=True)
        assert payload is not None
        execution = str(row.get("execution_id", ""))
        method = str(row.get("method", ""))
        if execution not in expected_execution_ids() or method not in METHODS:
            raise PackageContractError("Protocol identity drifted.")
        regime = next(
            regime
            for regime, nph in REGIME_ROWS
            if execution == execution_id(regime, nph, method)
        )
        if method == "ra_singleton_plateau" and payload.get("sha256") != RA_PROTOCOL_SHA256_BY_REGIME[regime]:
            raise PackageContractError(f"RA protocol drifted for {regime}.")
        if method == "append_singleton" and (
            payload.get("algorithm_id") != APPEND_ALGORITHM_ID
            or payload.get("adapter_id") != APPEND_ADAPTER_ID
            or payload.get("selector_identity") != APPEND_SELECTOR_ID
            or payload.get("selector_scope") != APPEND_SELECTOR_SCOPE
            or payload.get("request", {}).get("observation", {}).get("checkpoint", {}).get("keep_history_tail") != 1
        ):
            raise PackageContractError(f"Append protocol drifted for {regime}.")
        methods[method] += 1
    for index, row in enumerate(job_rows):
        path, payload = _verify_binding(row, label=f"job {index}", canonical=True)
        assert payload is not None
        execution = str(payload.get("execution_id", ""))
        if execution in jobs or execution not in expected_execution_ids():
            raise PackageContractError("Job execution closure drifted.")
        if (
            payload.get("fresh_start_contract", {}).get("fresh_start_only")
            is not True
            or payload.get("paper_adoption_authorized") is not False
            or payload.get("paper_evidence_adoption_authorized") is not False
            or payload.get("fresh_start_contract", {}).get(
                "checkpoint_resume_authorized"
            )
            is not False
            or payload.get("checkpoint_observation", {}).get("usage")
            != CHECKPOINT_USAGE
            or payload.get("checkpoint_observation", {}).get("resume_consumable")
            is not False
        ):
            raise PackageContractError("Job checkpoint authority drifted.")
        jobs[execution] = path
    if set(jobs) != set(expected_execution_ids()) or methods != Counter(
        {"ra_singleton_plateau": 6, "append_singleton": 6}
    ):
        raise PackageContractError("Matched method closure drifted.")
    _validation_path, validation = _verify_binding(
        manifest.get("bundle_validation_report"),
        label="bundle validation report",
        canonical=True,
    )
    assert validation is not None
    pool_rows = validation.get("matched_pool_receipts")
    if (
        not isinstance(pool_rows, list)
        or len(pool_rows) != 6
        or any(
            not isinstance(row, Mapping)
            or not str(row.get("problem_request_sha256", ""))
            or len(str(row.get("problem_request_sha256", ""))) != 64
            for row in pool_rows
        )
    ):
        raise PackageContractError("Matched problem identity receipts drifted.")
    if (
        validation.get("checkpoint_usage") != CHECKPOINT_USAGE
        or validation.get("fresh_start_only") is not True
        or validation.get("checkpoint_resume_authorized") is not False
        or validation.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
    ):
        raise PackageContractError("Matched checkpoint validation drifted.")

    preflights: list[dict[str, Any]] = []
    if worker_preflight:
        for execution in manifest["execution_order"]:
            preflight = worker.preflight(jobs[execution])
            if preflight.get("status") != "passed" or preflight.get("scientific_execution_performed") is not False:
                raise PackageContractError(f"Worker preflight failed for {execution}.")
            preflights.append(preflight)
    return digested(
        {
            "schema": "paper_i_page12_matched_singleton12_package_validation_v1",
            "status": "passed",
            "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "row_count": 12,
            "method_counts": dict(methods),
            "worker_preflight_count": len(preflights),
            "scientific_execution_performed": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-preflight", action="store_true")
    args = parser.parse_args()
    try:
        print(canonical_json_bytes(validate(worker_preflight=args.worker_preflight)).decode("utf-8"))
        return 0
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
