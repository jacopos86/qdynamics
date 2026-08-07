#!/usr/bin/env python3
"""Validate the weak-only authorized-pending-staging activation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ACTIVATION_DIR = Path(__file__).resolve().parent
if str(ACTIVATION_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVATION_DIR))
sys.dont_write_bytecode = True

from activation_contract import (  # noqa: E402
    ACTIVATION_ID,
    ACTIVATION_RELATIVE,
    ACTIVATION_SCHEMA,
    BATCH_NAME,
    CAMPAIGN_ID,
    CONTROL_FILES,
    OUTPUT_DESTINATION,
    OUTPUT_URI_TEMPLATE,
    PACKAGE_ID,
    PACKAGE_RELATIVE,
    QUEUE_VARIABLES,
    STAGING_ROOT,
    WEAK_ARCHIVE_BASENAME,
    WEAK_ARCHIVE_SHA256,
    WEAK_ARCHIVE_SIZE_BYTES,
    WEAK_EXECUTION_ID,
    WEAK_INPUT_URI,
    ActivationContractError,
    authorization_payload,
    canonical_json_bytes,
    canonical_sha256,
    file_binding,
    load_json,
    render_submit,
    repo_root_from_script,
    sha256_file,
    validate_package,
    validate_submit_text,
    verify_self_digest,
    weak_job,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActivationContractError(f"{label} must be a mapping.")
    return value


def validate() -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    if ACTIVATION_DIR != repo_root / ACTIVATION_RELATIVE:
        raise ActivationContractError("Activation directory identity drifted.")
    package_validation = validate_package(repo_root)
    package_manifest = load_json(
        repo_root / PACKAGE_RELATIVE / "package_manifest.json",
        label="package manifest",
    )
    verify_self_digest(package_manifest, label="package manifest")
    job_path, job = weak_job(repo_root)
    controls = [
        file_binding(ACTIVATION_DIR / name, relative_to=ACTIVATION_DIR)
        for name in CONTROL_FILES
    ]
    control_sha = canonical_sha256({"controls": controls})
    submit_path = ACTIVATION_DIR / "submit.sub"
    submit_text = submit_path.read_text(encoding="utf-8")
    validate_submit_text(submit_text)
    if submit_text != render_submit():
        raise ActivationContractError("Rendered submit descriptor drifted.")
    completed = subprocess.run(
        ["bash", "-n", str(ACTIVATION_DIR / "execute_resume_job.sh")],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise ActivationContractError(
            f"Worker shell syntax failed: {completed.stderr}"
        )
    authorization_path = (
        ACTIVATION_DIR / "authorizations" / f"{WEAK_EXECUTION_ID}.json"
    )
    authorization = load_json(
        authorization_path, label="execution authorization"
    )
    verify_self_digest(authorization, label="execution authorization")
    authorized_utc = authorization.get("authorized_utc")
    if not isinstance(authorized_utc, str) or not authorized_utc.strip():
        raise ActivationContractError("Authorization timestamp is absent.")
    expected_authorization = authorization_payload(
        job=job,
        package_manifest=package_manifest,
        control_sha256=control_sha,
        authorized_utc=authorized_utc,
    )
    if set(authorization) != set(expected_authorization) | {"sha256"} or any(
        authorization.get(key) != value
        for key, value in expected_authorization.items()
    ):
        raise ActivationContractError("Execution authorization drifted.")
    queue_path = ACTIVATION_DIR / "queue.tsv"
    rows = queue_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        raise ActivationContractError("Activation must contain one queue row.")
    fields = rows[0].split("\t")
    expected_fields = [
        WEAK_EXECUTION_ID,
        job_path.relative_to(repo_root).as_posix(),
        sha256_file(job_path),
        authorization_path.relative_to(repo_root).as_posix(),
        sha256_file(authorization_path),
        WEAK_INPUT_URI,
        WEAK_ARCHIVE_BASENAME,
        WEAK_ARCHIVE_SHA256,
        "4",
        "262144",
        "102400",
        "259200",
    ]
    if len(fields) != len(QUEUE_VARIABLES) or fields != expected_fields:
        raise ActivationContractError("Activation queue row drifted.")
    manifest = load_json(
        ACTIVATION_DIR / "activation_manifest.json",
        label="activation manifest",
    )
    verify_self_digest(manifest, label="activation manifest")
    staging = _mapping(manifest.get("staging"), label="staging contract")
    if (
        manifest.get("schema") != ACTIVATION_SCHEMA
        or manifest.get("status") != "passed_authorized_pending_staging"
        or manifest.get("activation_id") != ACTIVATION_ID
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("batch_name") != BATCH_NAME
        or manifest.get("direct_execution_count") != 1
        or manifest.get("execution_id") != WEAK_EXECUTION_ID
        or manifest.get("regime_id") != "weak_strong"
        or manifest.get("resume_controller_round") != 49
        or manifest.get("target_horizon") != 70
        or manifest.get("resources") != job.get("resources")
        or manifest.get("control_plane") != controls
        or manifest.get("activation_control_plane_sha256") != control_sha
        or manifest.get("ordinary_cluster") is not True
        or manifest.get("bounded_factory") is not False
        or manifest.get("intentional_hold") is not False
        or manifest.get("source_held_jobs_preserved") is not True
        or manifest.get("source_held_job_removal_authorized") is not False
        or manifest.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or manifest.get("non_swept_settings_diff") != []
        or staging.get("path") != STAGING_ROOT
        or staging.get("expected_quota_gb") != 100
        or staging.get("expected_item_limit") != 1000
        or staging.get("provisioned") is not False
        or staging.get("input_uri") != WEAK_INPUT_URI
        or staging.get("input_basename") != WEAK_ARCHIVE_BASENAME
        or staging.get("input_sha256") != WEAK_ARCHIVE_SHA256
        or staging.get("input_size_bytes") != WEAK_ARCHIVE_SIZE_BYTES
        or staging.get("input_uploaded") is not False
        or staging.get("output_uri_template") != OUTPUT_URI_TEMPLATE
        or staging.get("output_destination") != OUTPUT_DESTINATION
        or staging.get("output_exists") is not False
        or manifest.get("execution_authorized") is not True
        or manifest.get("submission_authorized") is not True
        or manifest.get("submission_ready") is not False
        or manifest.get("submission_state") != "authorized_pending_staging"
        or manifest.get("remote_stage") is not False
        or manifest.get("condor_submit") is not False
        or manifest.get("submitted") is not False
        or manifest.get("paper_evidence_adopted") is not False
    ):
        raise ActivationContractError("Activation manifest closure drifted.")
    return {
        "status": "passed_authorized_pending_staging",
        "activation_id": ACTIVATION_ID,
        "activation_manifest_sha256": manifest["sha256"],
        "package_manifest_sha256": package_manifest["sha256"],
        "package_source_preflight_count": package_validation[
            "source_preflight_count"
        ],
        "direct_execution_count": 1,
        "execution_id": WEAK_EXECUTION_ID,
        "resume_controller_round": 49,
        "target_horizon": 70,
        "request_memory_mb": 262_144,
        "request_disk_mb": 102_400,
        "staging_input_uri": WEAK_INPUT_URI,
        "staging_output_uri_template": OUTPUT_URI_TEMPLATE,
        "intentional_hold": False,
        "source_held_jobs_preserved": True,
        "submission_ready": False,
        "submitted": False,
    }


def main() -> int:
    try:
        payload = validate()
    except (OSError, ValueError, json.JSONDecodeError, ActivationContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
