#!/usr/bin/env python3
"""Validate and close the inert stationary-core package.

The default mode is read-only.  ``--write-p4`` is the sole supported
post-seal mutation: after validating an externally produced, one-round
packaged-dispatch result, it atomically publishes the declared P4 and package
preauthorization overlays.  It never creates submission authorization and
never invokes HTCondor.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    CAMPAIGN_ID,
    CORE_FINAL_COPY_RELATIVE,
    DECLARED_OVERLAY_FILES,
    ED_REGIME_NAME_BY_ID,
    EXECUTION_PLAN_SCHEMA,
    EXPECTED_ARTIFACT_ROLES,
    JOB_SPEC_SCHEMA,
    MUTABLE_RUNTIME_DIRECTORIES,
    P2_RECEIPT_RELATIVE,
    P2_RECEIPT_SCHEMA,
    P3_RECEIPT_RELATIVE,
    P4_RECEIPT_RELATIVE,
    P4_RECEIPT_SCHEMA,
    P4_SMOKE_RESULT_SCHEMA,
    P4_SMOKE_SPEC_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    PACKAGE_PREAUTHORIZATION_RELATIVE,
    PACKAGE_PREAUTHORIZATION_SCHEMA,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    RUN_CLASS,
    RUNTIME_RELATIVE_ROOT,
    SOURCE_ARCHIVE_MANIFEST_SCHEMA,
    SUBMISSION_AUTHORIZATION_RELATIVE,
    USER_SELECTION_COPY_RELATIVE,
    PackageContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    control_plane_receipt,
    digested,
    direct_execution_ids,
    direct_execution_rows,
    expected_artifact_path,
    load_json_object,
    safe_relative_path,
    sha256_file,
    validate_core_authority,
    validate_p3_receipt,
    validate_submission_authorization,
    validate_user_selection_authority,
    verify_exact_set,
    verify_self_digest,
)


def sha256_file_from_json_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload) + b"\n").hexdigest()


def _binding(path: Path, relative: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PackageContractError(f"Package member is unsafe: {relative}")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "executable": bool(path.stat().st_mode & 0o111),
    }


def _assert_exact_package_tree(
    *,
    manifest: Mapping[str, Any],
    require_p4: bool,
    require_authorization: bool,
    allow_partial_p4: bool,
) -> None:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise PackageContractError("Package manifest has no file inventory.")
    declared: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                f"Package inventory row {index} is invalid."
            )
        relative = safe_relative_path(
            raw.get("path"), label=f"package inventory row {index}"
        ).as_posix()
        if relative in declared:
            raise PackageContractError(
                f"Package inventory duplicates {relative}."
            )
        declared[relative] = raw
        observed = _binding(PACKAGE_DIR / relative, relative)
        if observed != raw:
            raise PackageContractError(
                f"Package member binding drifted: {relative}"
            )

    overlays = set()
    p4_path = PACKAGE_DIR / P4_RECEIPT_RELATIVE
    preauth_path = PACKAGE_DIR / PACKAGE_PREAUTHORIZATION_RELATIVE
    authorization_path = PACKAGE_DIR / SUBMISSION_AUTHORIZATION_RELATIVE
    if require_p4:
        if not p4_path.is_file() or not preauth_path.is_file():
            raise PackageContractError("Required P4/preauthorization is absent.")
        overlays.update({P4_RECEIPT_RELATIVE, PACKAGE_PREAUTHORIZATION_RELATIVE})
    elif allow_partial_p4:
        if not p4_path.is_file() or preauth_path.exists():
            raise PackageContractError(
                "Recoverable partial P4 state must contain only P4."
            )
        overlays.add(P4_RECEIPT_RELATIVE)
    elif p4_path.exists() or preauth_path.exists():
        raise PackageContractError(
            "P4 overlays exist but this validation did not require them."
        )
    if require_authorization:
        if not authorization_path.is_file():
            raise PackageContractError(
                "Required submission authorization is absent."
            )
        overlays.add(SUBMISSION_AUTHORIZATION_RELATIVE)
    elif authorization_path.exists() or authorization_path.is_symlink():
        raise PackageContractError(
            "Submission authorization must be absent before explicit approval."
        )

    expected_files = set(declared) | {"package_manifest.json"} | overlays
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in PACKAGE_DIR.rglob("*"):
        relative = path.relative_to(PACKAGE_DIR).as_posix()
        if path.is_symlink():
            raise PackageContractError(
                f"Package contains a forbidden symlink: {relative}"
            )
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_directories.add(relative)
        else:
            raise PackageContractError(
                f"Package contains a non-regular entry: {relative}"
            )
    verify_exact_set(
        observed_files, sorted(expected_files), label="recursive package files"
    )
    expected_directories = {
        "authority",
        "jobs",
        *MUTABLE_RUNTIME_DIRECTORIES,
    }
    verify_exact_set(
        observed_directories,
        sorted(expected_directories),
        label="recursive package directories",
    )


def _safe_validate_archive(
    *,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    verify_self_digest(source_manifest, label="source archive manifest")
    if (
        source_manifest.get("schema") != SOURCE_ARCHIVE_MANIFEST_SCHEMA
        or source_manifest.get("package_id") != PACKAGE_ID
    ):
        raise PackageContractError("Source archive manifest identity drifted.")
    archive_binding = source_manifest.get("archive")
    members = source_manifest.get("members")
    if not isinstance(archive_binding, Mapping) or not isinstance(members, list):
        raise PackageContractError("Source archive manifest is incomplete.")
    archive_path = PACKAGE_DIR / "source_locked.tar.gz"
    if (
        archive_binding.get("path") != "source_locked.tar.gz"
        or archive_binding.get("sha256") != sha256_file(archive_path)
        or int(archive_binding.get("size_bytes", -1))
        != archive_path.stat().st_size
        or int(source_manifest.get("member_count", -1)) != len(members)
    ):
        raise PackageContractError("Source archive binding drifted.")
    declared: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(members):
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                f"Source archive member {index} is invalid."
            )
        relative = safe_relative_path(
            raw.get("path"), label=f"archive member {index}"
        ).as_posix()
        if relative in declared:
            raise PackageContractError(
                f"Source archive duplicates {relative}."
            )
        declared[relative] = raw

    temporary = tempfile.TemporaryDirectory(
        prefix="paper_i_stationary_core_archive_"
    )
    # Keep extracted authority bytes alive for the caller's downstream P2/P3
    # and copied-authority comparisons.  The TemporaryDirectory object is
    # retained in the private return field and cleans itself when the
    # validation frame releases that mapping.
    with nullcontext(temporary.name) as raw_extract:
        extract_root = Path(raw_extract)
        observed: set[str] = set()
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                relative = safe_relative_path(
                    member.name, label="tar member"
                ).as_posix()
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or relative in observed
                    or relative not in declared
                ):
                    raise PackageContractError(
                        f"Unsafe or undeclared archive member: {relative}"
                    )
                destination = extract_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise PackageContractError(
                        f"Archive member has no bytes: {relative}"
                    )
                with destination.open("xb") as stream:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                binding = declared[relative]
                if (
                    sha256_file(destination) != binding.get("sha256")
                    or destination.stat().st_size
                    != int(binding.get("size_bytes", -1))
                ):
                    raise PackageContractError(
                        f"Extracted archive member drifted: {relative}"
                    )
                observed.add(relative)
        verify_exact_set(
            observed, sorted(declared), label="source archive members"
        )
        authority = validate_core_authority(extract_root)
        user = validate_user_selection_authority(extract_root)
    return {
        "authority": authority,
        "user_selection": user,
        "archive": dict(archive_binding),
        "_temporary_directory": temporary,
    }


def _validate_plan_and_jobs(
    *,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    verify_self_digest(plan, label="execution plan")
    rows = list(direct_execution_rows())
    expected_ids = list(direct_execution_ids())
    expected_g11_ids = [
        row["execution_id"]
        for row in rows
        if row["g11_bounded_replay_diagnostic"]["selected"]
    ]
    raw_executions = plan.get("direct_executions")
    if (
        plan.get("schema") != EXECUTION_PLAN_SCHEMA
        or plan.get("package_id") != PACKAGE_ID
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("run_class") != RUN_CLASS
        or plan.get("runtime_output_root") != RUNTIME_RELATIVE_ROOT
        or int(plan.get("direct_execution_count", -1)) != 48
        or plan.get("execution_ids") != expected_ids
        or int(
            plan.get("g11_bounded_replay_diagnostic_count", -1)
        )
        != 12
        or plan.get(
            "g11_bounded_replay_diagnostic_execution_ids"
        )
        != expected_g11_ids
        or not isinstance(raw_executions, list)
        or len(raw_executions) != 48
        or int(plan.get("shared_execution_count", -1)) != 0
        or plan.get("append_dedupe_active") is not False
        or plan.get("execution_authorized") is not False
        or plan.get("submission_authorized") is not False
        or plan.get("submission_state")
        != "awaiting_explicit_user_authorization"
        or plan.get("remote_image")
        != {
            "path": REMOTE_IMAGE_PATH,
            "sha256": REMOTE_IMAGE_SHA256,
            "byte_verification_state": "pending_remote_pre_submit",
            "verification_must_pass_before_condor_submit": True,
        }
        or manifest.get("execution_plan_sha256") != plan["sha256"]
    ):
        raise PackageContractError("Execution-plan matrix/state drifted.")
    queue_rows = (
        (PACKAGE_DIR / "queue.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if len(queue_rows) != 48:
        raise PackageContractError("Queue must contain exactly 48 rows.")
    queue_by_id: dict[str, list[str]] = {}
    for line in queue_rows:
        fields = line.split("\t")
        if len(fields) != 7 or fields[0] in queue_by_id:
            raise PackageContractError("Queue row shape/identity drifted.")
        queue_by_id[fields[0]] = fields
    if set(queue_by_id) != set(expected_ids):
        raise PackageContractError("Queue execution ids drifted.")

    execution_by_id = {
        str(raw["execution_id"]): raw
        for raw in raw_executions
        if isinstance(raw, Mapping)
    }
    if len(execution_by_id) != 48:
        raise PackageContractError("Execution plan duplicates a cell.")
    for expected in rows:
        execution_id = str(expected["execution_id"])
        planned = execution_by_id.get(execution_id)
        if planned != {
            **expected,
            "job_spec_path": f"jobs/{execution_id}.json",
        }:
            raise PackageContractError(
                f"Planned cell drifted: {execution_id}"
            )
        job_path = PACKAGE_DIR / f"jobs/{execution_id}.json"
        job = load_json_object(job_path, label=f"job {execution_id}")
        verify_self_digest(job, label=f"job {execution_id}")
        if (
            job.get("schema") != JOB_SPEC_SCHEMA
            or job.get("package_id") != PACKAGE_ID
            or job.get("campaign_id") != CAMPAIGN_ID
            or job.get("execution_plan_sha256") != plan["sha256"]
            or job.get("execution_id") != execution_id
            or job.get("cell_id") != execution_id
            or job.get("resources") != expected["resources"]
            or job.get("artifact_paths")
            != {
                role: (
                    f"{job['core_bundle_root']}/"
                    f"{expected_artifact_path(execution_id, role)}"
                )
                for role in EXPECTED_ARTIFACT_ROLES
            }
            or job.get("execution_authorized") is not False
            or job.get("submission_authorized") is not False
            or job.get("submission_state")
            != "awaiting_explicit_user_authorization"
        ):
            raise PackageContractError(f"Job spec drifted: {execution_id}")
        queue = queue_by_id[execution_id]
        resources = expected["resources"]
        if queue != [
            execution_id,
            f"jobs/{execution_id}.json",
            sha256_file(job_path),
            str(plan["source_archive"]["sha256"]),
            str(resources["request_cpus"]),
            str(resources["request_memory_mb"]),
            str(resources["request_disk_mb"]),
        ]:
            raise PackageContractError(f"Queue binding drifted: {execution_id}")


def _validate_submit_surface() -> None:
    text = (PACKAGE_DIR / "submit.sub").read_text(encoding="utf-8")
    required = (
        "when_to_transfer_output = ON_EXIT",
        "queue execution_id,job_spec,job_spec_sha256,"
        "source_archive_sha256,cpus,memory_mb,disk_mb from "
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "stationary_core_full48_r50_20260728_v7_chtc/queue.tsv",
        "request_cpus = $(cpus)",
        "request_memory = $(memory_mb)MB",
        "request_disk = $(disk_mb)MB",
        SUBMISSION_AUTHORIZATION_RELATIVE,
        "$(Cluster).$(Process)",
        "$(execution_id)",
        "stationary_core_full48_r50_20260728_v7_chtc_runtime",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise PackageContractError(
            f"Submit description lacks required contracts: {missing}"
        )
    forbidden = (
        "should_transfer_files = NO",
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "queue 18",
        "$(NumJobStarts)",
    )
    present = [fragment for fragment in forbidden if fragment in text]
    if present:
        raise PackageContractError(
            f"Submit description contains forbidden contracts: {present}"
        )


def _validate_p4_verified_ed_reference(
    *,
    smoke: Mapping[str, Any],
    smoke_job: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> str:
    """Rebind the P4 parsed ED receipt to the source-locked authority."""

    receipt = smoke.get("verified_same_cutoff_ed_reference")
    if not isinstance(receipt, Mapping):
        raise PackageContractError(
            "P4 lacks a verified same-cutoff ED source receipt."
        )
    digest = verify_self_digest(
        receipt, label="P4 verified same-cutoff ED source receipt"
    )
    source_lock = authority["source_lock_cells"].get(
        smoke_job.get("source_lock_id")
    )
    if not isinstance(source_lock, Mapping):
        raise PackageContractError("P4 source-lock identity drifted.")
    resolver = source_lock.get("resolver_trace")
    exact = (
        resolver.get("same_cutoff_ed_reference")
        if isinstance(resolver, Mapping)
        else None
    )
    global_metadata = authority["global_source_locks"].get(
        "ed_cutoff_reference"
    )
    global_binding = authority["global_source_files"].get(
        "ed_cutoff_reference"
    )
    projection_sha = receipt.get("cell_projection_sha256")
    if (
        not isinstance(exact, Mapping)
        or not isinstance(global_metadata, Mapping)
        or not isinstance(global_binding, Mapping)
        or receipt.get("schema")
        != (
            "paper_i_ra_adapt_stationary_core_"
            "verified_same_cutoff_ed_reference_v1"
        )
        or receipt.get("path") != exact.get("path")
        or receipt.get("path") != global_metadata.get("path")
        or receipt.get("path") != global_binding.get("path")
        or receipt.get("file_sha256") != exact.get("sha256")
        or receipt.get("file_sha256")
        != global_metadata.get("sha256")
        or receipt.get("file_sha256") != global_binding.get("sha256")
        or int(receipt.get("file_size_bytes", -1))
        != int(global_binding.get("size_bytes", -2))
        or receipt.get("source_payload_schema")
        != "paper_i_hh_ed_cutoff_reference_six_regime_v1"
        or receipt.get("regime_id") != smoke_job.get("regime_id")
        or receipt.get("regime_name")
        != ED_REGIME_NAME_BY_ID.get(str(smoke_job.get("regime_id")))
        or receipt.get("regime_name") != exact.get("regime_name")
        or int(receipt.get("n_ph_work", -1))
        != int(smoke_job.get("nph", -2))
        or int(receipt.get("n_ph_reference", -1))
        != int(smoke_job.get("nph", -2))
        or int(receipt.get("n_ph_reference", -1))
        != int(exact.get("nph", -2))
        or receipt.get("E_ED") != exact.get("E_ED")
        or receipt.get("reference_role")
        != "same_cutoff_reporting_reference"
        or receipt.get("controller_decision_influence") is not False
        or receipt.get("status") != "passed"
        or not isinstance(projection_sha, str)
        or len(projection_sha) != 64
        or any(character not in "0123456789abcdef" for character in projection_sha)
        or smoke.get("verified_same_cutoff_ed_reference_sha256")
        != digest
    ):
        raise PackageContractError(
            "P4 verified same-cutoff ED source receipt drifted."
        )
    return digest


def _validate_p4_smoke_payload(
    *,
    smoke: Mapping[str, Any],
    spec: Mapping[str, Any],
    smoke_job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> str:
    """Revalidate the complete retained P4 dispatch and all five payloads."""

    smoke_digest = verify_self_digest(smoke, label="P4 smoke result")
    trusted_execution = smoke.get(
        "trusted_execution_source_dataflow_receipt"
    )
    try:
        from pipelines.static_adapt.ra_adapt.exact_reference_isolation import (
            validate_study1_trusted_execution_receipt,
        )

        validated_trusted_execution = (
            validate_study1_trusted_execution_receipt(
                trusted_execution,
                reverify_source=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"P4 trusted source/dataflow receipt failed: {exc}"
        ) from exc
    _validate_p4_verified_ed_reference(
        smoke=smoke,
        smoke_job=smoke_job,
        authority=authority,
    )
    artifacts = smoke.get("artifact_bindings")
    if (
        smoke.get("schema") != P4_SMOKE_RESULT_SCHEMA
        or smoke.get("package_id") != PACKAGE_ID
        or smoke.get("campaign_id") != CAMPAIGN_ID
        or smoke.get("source_execution_id")
        != spec.get("source_execution_id")
        or smoke.get("p4_smoke_spec_sha256") != spec["sha256"]
        or smoke.get("package_manifest_sha256") != manifest["sha256"]
        or smoke.get("execution_plan_sha256") != plan["sha256"]
        or smoke.get("source_archive_sha256")
        != plan["source_archive"]["sha256"]
        or smoke.get("status") != "passed"
        or smoke.get("bounded_dispatch_passed") is not True
        or smoke.get("source_locked_archive_validated") is not True
        or int(smoke.get("maximum_controller_rounds", -1)) != 1
        or smoke.get("run_class") != "smoke"
        or smoke.get("paper_facing_result_allowed") is not False
        or smoke.get("execution_authorized") is not False
        or smoke.get("submission_authorized") is not False
        or smoke.get("submission_state") != "not_submitted"
        or validated_trusted_execution != trusted_execution
        or smoke.get(
            "trusted_execution_source_dataflow_receipt_sha256"
        )
        != validated_trusted_execution["sha256"]
        or not isinstance(artifacts, list)
    ):
        raise PackageContractError("P4 smoke result contract drifted.")
    artifact_roles = [
        str(raw.get("role", ""))
        for raw in artifacts
        if isinstance(raw, Mapping)
    ]
    if artifact_roles != list(EXPECTED_ARTIFACT_ROLES):
        raise PackageContractError(
            "P4 smoke must retain each of the five artifact roles once "
            "in canonical order."
        )
    local_paths = {
        "execution_manifest": "execution_manifest.json",
        "checkpoint": "checkpoint.json",
        "estimator_ledger": "estimator_ledger.json",
        "result": "result.json",
        "summary": "summary.json",
    }
    artifacts_by_role: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(artifacts):
        retained_artifact = (
            raw.get("canonical_payload")
            if isinstance(raw, Mapping)
            else None
        )
        exact_file_text = (
            raw.get("exact_file_text")
            if isinstance(raw, Mapping)
            else None
        )
        exact_file_bytes = (
            exact_file_text.encode("utf-8")
            if isinstance(exact_file_text, str)
            else b""
        )
        try:
            reparsed = (
                json.loads(exact_file_text)
                if isinstance(exact_file_text, str)
                else None
            )
        except json.JSONDecodeError:
            reparsed = None
        role = raw.get("role") if isinstance(raw, Mapping) else None
        if (
            not isinstance(raw, Mapping)
            or role not in EXPECTED_ARTIFACT_ROLES
            or raw.get("path") != local_paths.get(str(role))
            or not isinstance(retained_artifact, Mapping)
            or raw.get("retention")
            != "embedded_exact_utf8_json_bytes_v1"
            or reparsed != retained_artifact
            or raw.get("sha256")
            != hashlib.sha256(exact_file_bytes).hexdigest()
            or int(raw.get("size_bytes", -1)) != len(exact_file_bytes)
            or raw.get("declared_canonical_path")
            != smoke_job["artifact_paths"].get(role)
            or raw.get("mapping_kind")
            != "bounded_smoke_shadow_not_fulfillment_v1"
        ):
            raise PackageContractError(
                f"P4 artifact binding {index} is invalid."
            )
        artifacts_by_role[str(role)] = raw
    _validate_p4_artifact_semantics(
        artifacts_by_role=artifacts_by_role,
        smoke_job=smoke_job,
    )
    return smoke_digest


def _v9_embedded_summary_matches_typed_summary(
    embedded: Any,
    typed: Mapping[str, Any],
) -> bool:
    """Compare the two summary projections immutable v9 actually emits."""

    if not isinstance(embedded, Mapping):
        return False

    def omit_none(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): omit_none(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, (list, tuple)):
            return [omit_none(item) for item in value]
        return value

    return embedded == omit_none(
        {
            key: value
            for key, value in typed.items()
            if key != "schema"
        }
    )


def _validate_p4_artifact_semantics(
    *,
    artifacts_by_role: Mapping[str, Mapping[str, Any]],
    smoke_job: Mapping[str, Any],
) -> None:
    """Validate role schemas, identities, and retained cross-bindings."""

    payloads = {
        role: raw["canonical_payload"]
        for role, raw in artifacts_by_role.items()
    }
    execution = payloads["execution_manifest"]
    checkpoint = payloads["checkpoint"]
    ledger = payloads["estimator_ledger"]
    result = payloads["result"]
    summary = payloads["summary"]
    expected_protocol_sha256 = smoke_job["protocol"]["canonical_sha256"]

    execution_digest = verify_self_digest(
        execution, label="retained P4 execution manifest"
    )
    output_payloads = execution.get("output_payloads")
    expected_output_bindings = {
        role: {
            "sha256": artifacts_by_role[role]["sha256"],
            "size_bytes": artifacts_by_role[role]["size_bytes"],
        }
        for role in EXPECTED_ARTIFACT_ROLES
        if role != "execution_manifest"
    }
    if (
        execution.get("schema")
        != "paper_i_ra_adapt_stationary_core_execution_manifest_v1"
        or execution.get("package_id") != PACKAGE_ID
        or execution.get("campaign_id") != CAMPAIGN_ID
        or execution.get("execution_id") != smoke_job["execution_id"]
        or execution.get("cell_id") != smoke_job["cell_id"]
        or execution.get("execution_entrypoint") != "run_ra_adapt"
        or execution.get("protocol_sha256") != expected_protocol_sha256
        or execution.get("job_spec_sha256") != smoke_job["sha256"]
        or int(execution.get("maximum_controller_rounds_override", -1))
        != 1
        or execution.get("run_class") != "smoke"
        or execution.get("paper_facing_result_allowed") is not False
        or execution.get("status") != "passed"
        or not isinstance(execution.get("completed_utc"), str)
        or not execution["completed_utc"]
        or output_payloads != expected_output_bindings
        or execution_digest
        != execution.get("sha256")
    ):
        raise PackageContractError(
            "Retained P4 execution-manifest semantics drifted."
        )

    protocol = result.get("protocol")
    run = result.get("run")
    policy = result.get("policy")
    scientific = result.get("scientific_receipts")
    numerical = result.get("numerical_physical_integrity")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol, run, policy, scientific, numerical)
    ):
        raise PackageContractError(
            "Retained P4 result lacks typed contract mappings."
        )
    assert isinstance(protocol, Mapping)
    assert isinstance(run, Mapping)
    assert isinstance(policy, Mapping)
    assert isinstance(scientific, Mapping)
    assert isinstance(numerical, Mapping)
    verify_self_digest(protocol, label="retained P4 resolved protocol")
    problem = protocol.get("problem")
    materialization = protocol.get("bundle_materialization")
    trajectory = run.get("accepted_trajectory")
    transitions = run.get("accepted_transitions")
    replay = run.get("scientific_replay")
    stop = run.get("stop")
    observation = run.get("observation")
    if (
        result.get("schema") != "paper_i_ra_adapt_result_v1"
        or result.get("selector_identity")
        != "ra_adapt_staged_phase_i_ii_iii_funnel_v1"
        or protocol.get("schema")
        != "paper_i_ra_adapt_resolved_protocol_v1"
        or protocol.get("sha256") != expected_protocol_sha256
        or protocol.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or protocol.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
        or protocol.get("candidate_representation")
        != smoke_job["candidate_representation"]
        or str(protocol.get("optimizer", "")).lower() != "powell"
        or int(protocol.get("optimizer_maxiter", -1)) != 200
        or int(protocol.get("horizon", -1)) != 50
        or protocol.get("seeds") != {"adapt": 7, "transpiler": 7}
        or protocol.get("execution_authorized") is not False
        or not isinstance(problem, Mapping)
        or problem.get("problem_key") != "hh"
        or int(problem.get("num_sites", -1)) != 2
        or int(problem.get("n_ph_max", -1)) != int(smoke_job["nph"])
        or not isinstance(materialization, Mapping)
        or materialization.get("cell_id") != smoke_job["cell_id"]
        or materialization.get("protocol_schema")
        != "paper_i_ra_adapt_resolved_protocol_v1"
        or result.get("parent_inventory")
        != protocol.get("parent_inventory")
        or result.get("executable_pool")
        != protocol.get("executable_pool")
        or policy
        != {
            "active_gradient_policy": "stationary_source_response_v1",
            "resource_weighting_scope": "late_resource_weighting_v1",
            "active_gradient_indices_acquired": [],
            "active_gradient_charge": 0,
        }
        or not isinstance(trajectory, list)
        or len(trajectory) != 1
        or not isinstance(transitions, list)
        or len(transitions) != 1
        or not isinstance(replay, list)
        or len(replay) != 1
        or run.get("final_state") != trajectory[-1]
        or run.get("problem") != problem
        or not isinstance(stop, Mapping)
        or int(stop.get("completed_controller_rounds", -1)) != 1
        or int(stop.get("accepted_operator_count", -1)) != 1
        or stop.get("primary_reason") != "maximum_controller_rounds"
        or not isinstance(observation, Mapping)
        or not _v9_embedded_summary_matches_typed_summary(
            run.get("paper_i_summary"),
            summary,
        )
        or scientific.get("policy") != policy
        or scientific.get("numerical_physical_integrity") != numerical
        or scientific.get("numerical_physical_integrity_sha256")
        != canonical_sha256(numerical)
    ):
        raise PackageContractError(
            "Retained P4 RA result/protocol semantics drifted."
        )

    route_contract = protocol.get("route_contract")
    route_contract_sha256 = (
        verify_self_digest(
            route_contract, label="retained P4 route contract"
        )
        if isinstance(route_contract, Mapping)
        else None
    )
    checkpoint_settings = checkpoint.get("settings")
    checkpoint_adapt = checkpoint.get("adapt_vqe")
    if (
        checkpoint.get("schema_version")
        != "static_adapt_current_checkpoint_v1"
        or checkpoint.get("no_credentials_serialized") is not True
        or not isinstance(checkpoint_settings, Mapping)
        or checkpoint_settings.get("problem") != "hh"
        or int(checkpoint_settings.get("L", -1)) != 2
        or int(checkpoint_settings.get("n_ph_max", -1))
        != int(smoke_job["nph"])
        or str(
            checkpoint_settings.get("adapt_inner_optimizer", "")
        ).lower()
        != "powell"
        or checkpoint_settings.get("sr_route_profile_contract_sha256")
        != route_contract_sha256
        or not isinstance(checkpoint_adapt, Mapping)
        or checkpoint_adapt.get("success") is not False
        or int(checkpoint_adapt.get("ansatz_depth", -1)) != 1
        or int(checkpoint_adapt.get("history_count", -1)) != 1
        or checkpoint_adapt.get("sr_route_profile_contract_sha256")
        != route_contract_sha256
    ):
        raise PackageContractError(
            "Retained P4 checkpoint semantics drifted."
        )

    ledger_accounting = ledger.get("accounting")
    full_ledger = ledger.get("ledger")
    if (
        ledger.get("schema") != "paper_i_estimator_call_ledger_sidecar_v2"
        or ledger.get("adapt_success") is not True
        or not isinstance(ledger_accounting, Mapping)
        or ledger_accounting.get("complete") is not True
        or not isinstance(full_ledger, Mapping)
        or full_ledger.get("schema") != "estimator_call_ledger_v1"
    ):
        raise PackageContractError(
            "Retained P4 estimator-ledger semantics drifted."
        )

    summary_provenance = summary.get("provenance")
    accepted_error_trace = summary.get("accepted_error_trace")
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or int(summary.get("available_controller_rounds", -1)) != 1
        or not isinstance(accepted_error_trace, list)
        or len(accepted_error_trace) != 1
        or not isinstance(summary.get("canonical_all_work"), Mapping)
        or not isinstance(summary_provenance, Mapping)
        or summary_provenance.get("problem_key") != "hh"
        or summary_provenance.get("problem_family") != "hh"
        or summary_provenance.get("route_family") != "ra_adapt"
        or summary_provenance.get("route_contract_sha256")
        != route_contract_sha256
        or summary_provenance.get("candidate_representation")
        != smoke_job["candidate_representation"]
        or str(summary_provenance.get("optimizer", "")).lower()
        != "powell"
        or int(summary_provenance.get("optimizer_maxiter", -1)) != 200
        or int(summary_provenance.get("seed", -1)) != 7
    ):
        raise PackageContractError(
            "Retained P4 Paper-I summary semantics drifted."
        )

    observation_rows = observation.get("artifacts")
    expected_observation = {
        "accepted_state_checkpoint": artifacts_by_role["checkpoint"],
        "estimator_ledger": artifacts_by_role["estimator_ledger"],
    }
    if not isinstance(observation_rows, list) or len(observation_rows) != 2:
        raise PackageContractError(
            "Retained P4 observation artifact closure drifted."
        )
    seen_kinds: set[str] = set()
    for raw in observation_rows:
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                "Retained P4 observation row is invalid."
            )
        kind = str(raw.get("kind", ""))
        binding = expected_observation.get(kind)
        if (
            binding is None
            or kind in seen_kinds
            or raw.get("sha256") != binding["sha256"]
            or int(raw.get("size_bytes", -1))
            != int(binding["size_bytes"])
            or Path(str(raw.get("path", ""))).name
            != binding["path"]
            or (
                kind == "accepted_state_checkpoint"
                and int(raw.get("every_controller_rounds", -1)) != 1
            )
        ):
            raise PackageContractError(
                "Retained P4 observation binding drifted."
            )
        seen_kinds.add(kind)
    if seen_kinds != set(expected_observation):
        raise PackageContractError(
            "Retained P4 observation roles are incomplete."
        )

def validate_package(
    *,
    require_p4: bool,
    require_authorization: bool,
    allow_partial_p4: bool = False,
) -> dict[str, Any]:
    manifest = load_json_object(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("run_class") != RUN_CLASS
        or manifest.get("runtime_output_root") != RUNTIME_RELATIVE_ROOT
        or int(manifest.get("direct_execution_count", -1)) != 48
        or int(manifest.get("shared_execution_count", -1)) != 0
        or manifest.get("append_dedupe_active") is not False
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("remote_stage") is not False
        or manifest.get("condor_submit") is not False
        or manifest.get("remote_image")
        != {
            "path": REMOTE_IMAGE_PATH,
            "sha256": REMOTE_IMAGE_SHA256,
            "byte_verification_state": "pending_remote_pre_submit",
            "verification_must_pass_before_condor_submit": True,
        }
    ):
        raise PackageContractError("Package manifest identity/state drifted.")
    if set(manifest.get("declared_post_seal_overlays", {})) != set(
        DECLARED_OVERLAY_FILES
    ):
        raise PackageContractError("Declared package overlays drifted.")
    _assert_exact_package_tree(
        manifest=manifest,
        require_p4=require_p4,
        require_authorization=require_authorization,
        allow_partial_p4=allow_partial_p4,
    )
    observed_control = control_plane_receipt(PACKAGE_DIR)
    sealed_control = load_json_object(
        PACKAGE_DIR / "control_plane_receipt.json",
        label="sealed control-plane receipt",
    )
    if observed_control != sealed_control:
        raise PackageContractError("Sealed control-plane bytes drifted.")

    source_manifest = load_json_object(
        PACKAGE_DIR / "source_archive_manifest.json",
        label="source archive manifest",
    )
    archive = _safe_validate_archive(source_manifest=source_manifest)
    authority = archive["authority"]
    user_selection = archive["user_selection"]
    if (
        (PACKAGE_DIR / CORE_FINAL_COPY_RELATIVE).read_bytes()
        != Path(
            authority["core_root"],
            "final_publication_receipt.json",
        ).read_bytes()
    ):
        raise PackageContractError("Copied core final receipt drifted.")
    if (
        sha256_file(PACKAGE_DIR / USER_SELECTION_COPY_RELATIVE)
        != user_selection["binding"]["sha256"]
    ):
        raise PackageContractError("Copied user-selection authority drifted.")

    p2 = load_json_object(PACKAGE_DIR / P2_RECEIPT_RELATIVE, label="P2 receipt")
    verify_self_digest(p2, label="P2 receipt")
    if (
        p2.get("schema") != P2_RECEIPT_SCHEMA
        or p2.get("package_id") != PACKAGE_ID
        or p2.get("status") != "passed"
        or p2.get("p2_passed") is not True
        or int(p2.get("direct_cell_count", -1)) != 48
        or p2.get("execution_authorized") is not False
        or p2.get("submission_authorized") is not False
    ):
        raise PackageContractError("P2 receipt drifted.")
    p3 = load_json_object(PACKAGE_DIR / P3_RECEIPT_RELATIVE, label="P3 receipt")
    validate_p3_receipt(
        p3,
        receipt_file_sha256=sha256_file(PACKAGE_DIR / P3_RECEIPT_RELATIVE),
        authority=authority,
        control_plane=sealed_control,
    )
    p2_pool_proof = p2.get("six_regime_pool_construction_proof")
    if (
        not isinstance(p2_pool_proof, Mapping)
        or verify_self_digest(
            p2_pool_proof,
            label="P2 six-regime pool/construction proof",
        )
        != p2.get("six_regime_pool_construction_proof_sha256")
        or p2_pool_proof != p3.get("p2_pool_construction_proof")
        or p2.get("six_regime_pool_construction_proof_sha256")
        != p3.get("p2_pool_construction_proof_sha256")
        or p2.get("p3_receipt_sha256") != p3["sha256"]
        or p2.get("core_final_receipt")
        != authority["final_receipt_binding"]
        or p2.get("implementation_source_inventory_sha256")
        != authority["implementation_inventory_sha256"]
        or p2.get("global_source_files")
        != {
            key: dict(value)
            for key, value in sorted(
                authority["global_source_files"].items()
            )
        }
        or p2.get("user_selection_authority")
        != user_selection["binding"]
    ):
        raise PackageContractError(
            "P2 six-regime pool/construction authority drifted."
        )
    plan = load_json_object(
        PACKAGE_DIR / "execution_plan.json", label="execution plan"
    )
    _validate_plan_and_jobs(manifest=manifest, plan=plan)
    _validate_submit_surface()
    smoke_spec = load_json_object(
        PACKAGE_DIR / "p4_smoke_spec.json", label="P4 smoke spec"
    )
    verify_self_digest(smoke_spec, label="P4 smoke spec")
    p4_execution_id = (
        "core__strong_weak_u8__nph3__ra_macro_append_only"
    )
    if (
        smoke_spec.get("schema") != P4_SMOKE_SPEC_SCHEMA
        or smoke_spec.get("package_id") != PACKAGE_ID
        or smoke_spec.get("campaign_id") != CAMPAIGN_ID
        or smoke_spec.get("source_execution_id") != p4_execution_id
        or smoke_spec.get("source_job_spec_path")
        != f"jobs/{p4_execution_id}.json"
        or int(smoke_spec.get("maximum_controller_rounds", -1)) != 1
        or smoke_spec.get("run_class") != "smoke"
        or smoke_spec.get("purpose")
        != "bounded_packaged_dispatch_and_verification_only_v1"
        or smoke_spec.get("paper_facing_result_allowed") is not False
        or smoke_spec.get("submission_authorized") is not False
        or manifest.get("p4_smoke_spec_sha256") != smoke_spec["sha256"]
    ):
        raise PackageContractError("P4 smoke spec drifted.")

    result: dict[str, Any] = {
        "status": "passed",
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "package_manifest_sha256": manifest["sha256"],
        "execution_plan_sha256": plan["sha256"],
        "source_archive_sha256": archive["archive"]["sha256"],
        "control_plane_sha256": sealed_control["sha256"],
        "direct_execution_count": 48,
        "append_dedupe_active": False,
        "p4_required": require_p4,
        "submission_authorization_required": require_authorization,
        "remote_stage": False,
        "condor_submit": False,
    }
    if require_p4:
        p4 = load_json_object(PACKAGE_DIR / P4_RECEIPT_RELATIVE, label="P4 receipt")
        preauth = load_json_object(
            PACKAGE_DIR / PACKAGE_PREAUTHORIZATION_RELATIVE,
            label="package preauthorization receipt",
        )
        verify_self_digest(p4, label="P4 receipt")
        verify_self_digest(preauth, label="package preauthorization receipt")
        retained_smoke = p4.get("smoke_result")
        retained_payload = (
            retained_smoke.get("canonical_payload")
            if isinstance(retained_smoke, Mapping)
            else None
        )
        smoke_job = load_json_object(
            PACKAGE_DIR / str(smoke_spec["source_job_spec_path"]),
            label="P4 source job",
        )
        retained_digest = (
            _validate_p4_smoke_payload(
                smoke=retained_payload,
                spec=smoke_spec,
                smoke_job=smoke_job,
                manifest=manifest,
                plan=plan,
                authority=authority,
            )
            if isinstance(retained_payload, Mapping)
            else None
        )
        expected_next_action = (
            "show_the_final_validated_package_and_request_explicit_"
            "chtc_submission_authorization_before_condor_submit"
        )
        if (
            p4.get("schema") != P4_RECEIPT_SCHEMA
            or p4.get("package_id") != PACKAGE_ID
            or p4.get("campaign_id") != CAMPAIGN_ID
            or p4.get("package_manifest_sha256") != manifest["sha256"]
            or p4.get("execution_plan_sha256") != plan["sha256"]
            or p4.get("source_archive_sha256")
            != archive["archive"]["sha256"]
            or p4.get("p4_smoke_spec_sha256") != smoke_spec["sha256"]
            or p4.get("status") != "passed"
            or p4.get("p4_passed") is not True
            or p4.get("bounded_dispatch_passed") is not True
            or p4.get("source_locked_archive_validated") is not True
            or int(p4.get("maximum_controller_rounds", -1)) != 1
            or p4.get("paper_facing_result_allowed") is not False
            or p4.get("execution_authorized") is not False
            or p4.get("submission_authorized") is not False
            or p4.get("submission_state") != "not_submitted"
            or not isinstance(retained_smoke, Mapping)
            or not isinstance(retained_payload, Mapping)
            or retained_smoke.get("retention")
            != "embedded_complete_canonical_payload_v1"
            or retained_digest != retained_smoke.get("canonical_sha256")
            or retained_smoke.get("file_sha256")
            != sha256_file_from_json_payload(retained_payload)
            or p4.get("artifact_bindings")
            != retained_payload.get("artifact_bindings")
            or preauth.get("schema") != PACKAGE_PREAUTHORIZATION_SCHEMA
            or preauth.get("package_id") != PACKAGE_ID
            or preauth.get("campaign_id") != CAMPAIGN_ID
            or preauth.get("status") != "passed"
            or preauth.get("package_validated") is not True
            or int(preauth.get("direct_execution_count", -1)) != 48
            or preauth.get("package_manifest_sha256") != manifest["sha256"]
            or preauth.get("execution_plan_sha256") != plan["sha256"]
            or preauth.get("source_archive_sha256")
            != archive["archive"]["sha256"]
            or preauth.get("package_control_plane_sha256")
            != sealed_control["sha256"]
            or preauth.get("p2_receipt_sha256") != p2["sha256"]
            or preauth.get("p3_receipt_sha256") != p3["sha256"]
            or preauth.get("p4_receipt_sha256") != p4["sha256"]
            or preauth.get("remote_image")
            != {
                "path": REMOTE_IMAGE_PATH,
                "sha256": REMOTE_IMAGE_SHA256,
                "byte_verification_state": "pending_remote_pre_submit",
                "verification_must_pass_before_condor_submit": True,
            }
            or preauth.get("execution_authorized") is not False
            or preauth.get("submission_authorized") is not False
            or preauth.get("remote_stage") is not False
            or preauth.get("condor_submit") is not False
            or preauth.get("submission_authorization_overlay")
            != {
                "path": SUBMISSION_AUTHORIZATION_RELATIVE,
                "present": False,
                "required_before_condor_submit": True,
            }
            or preauth.get("submission_authorization_overlay_present")
            is not False
            or preauth.get("submission_state")
            != "awaiting_explicit_user_authorization"
            or preauth.get("next_action") != expected_next_action
        ):
            raise PackageContractError("P4/preauthorization overlay drifted.")
        result["p4_receipt_sha256"] = p4["sha256"]
        result["package_preauthorization_sha256"] = preauth["sha256"]
    if require_authorization:
        p4 = load_json_object(PACKAGE_DIR / P4_RECEIPT_RELATIVE, label="P4 receipt")
        authorization = load_json_object(
            PACKAGE_DIR / SUBMISSION_AUTHORIZATION_RELATIVE,
            label="submission authorization",
        )
        result["submission_authorization_sha256"] = (
            validate_submission_authorization(
                authorization,
                package_manifest=manifest,
                execution_plan=plan,
                p4_receipt=p4,
            )
        )
    return result


def close_p4(*, smoke_result_path: Path) -> dict[str, Any]:
    p4_path = PACKAGE_DIR / P4_RECEIPT_RELATIVE
    preauth_path = PACKAGE_DIR / PACKAGE_PREAUTHORIZATION_RELATIVE
    p4_exists = p4_path.exists()
    preauth_exists = preauth_path.exists()
    if preauth_exists and not p4_exists:
        raise PackageContractError(
            "Unrecoverable overlay order: preauthorization exists without P4."
        )
    if p4_exists and preauth_exists:
        final = validate_package(
            require_p4=True, require_authorization=False
        )
        retained = load_json_object(p4_path, label="existing P4 receipt").get(
            "smoke_result"
        )
        supplied = load_json_object(
            smoke_result_path, label="P4 smoke result"
        )
        if (
            not isinstance(retained, Mapping)
            or retained.get("canonical_payload") != supplied
        ):
            raise PackageContractError(
                "Completed P4 closure does not match supplied smoke evidence."
            )
        return final
    # A valid P4-only state is an intentional recoverable transaction prefix.
    base = validate_package(
        require_p4=False,
        require_authorization=False,
        allow_partial_p4=p4_exists,
    )
    smoke = load_json_object(smoke_result_path, label="P4 smoke result")
    canonical_smoke_file_sha = sha256_file_from_json_payload(smoke)
    if sha256_file(smoke_result_path) != canonical_smoke_file_sha:
        raise PackageContractError(
            "P4 smoke result must use canonical JSON plus one newline."
        )
    spec = load_json_object(
        PACKAGE_DIR / "p4_smoke_spec.json", label="P4 smoke spec"
    )
    smoke_job = load_json_object(
        PACKAGE_DIR / str(spec["source_job_spec_path"]),
        label="P4 source job",
    )
    manifest = load_json_object(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    plan = load_json_object(
        PACKAGE_DIR / "execution_plan.json", label="execution plan"
    )
    source_manifest = load_json_object(
        PACKAGE_DIR / "source_archive_manifest.json",
        label="source archive manifest",
    )
    authority = _safe_validate_archive(
        source_manifest=source_manifest
    )["authority"]
    smoke_digest = _validate_p4_smoke_payload(
        smoke=smoke,
        spec=spec,
        smoke_job=smoke_job,
        manifest=manifest,
        plan=plan,
        authority=authority,
    )
    artifacts = smoke["artifact_bindings"]
    p4 = digested(
        {
            "schema": P4_RECEIPT_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "package_manifest_sha256": manifest["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "source_archive_sha256": plan["source_archive"]["sha256"],
            "p4_smoke_spec_sha256": spec["sha256"],
            "smoke_result": {
                "retention": "embedded_complete_canonical_payload_v1",
                "canonical_sha256": smoke_digest,
                "file_sha256": canonical_smoke_file_sha,
                "canonical_payload": dict(smoke),
            },
            "artifact_bindings": artifacts,
            "bounded_dispatch_passed": True,
            "source_locked_archive_validated": True,
            "maximum_controller_rounds": 1,
            "paper_facing_result_allowed": False,
            "status": "passed",
            "p4_passed": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
        }
    )
    if p4_exists:
        existing_p4 = load_json_object(p4_path, label="existing P4 receipt")
        verify_self_digest(existing_p4, label="existing P4 receipt")
        retained = existing_p4.get("smoke_result")
        if (
            existing_p4.get("schema") != P4_RECEIPT_SCHEMA
            or existing_p4.get("package_manifest_sha256")
            != manifest["sha256"]
            or existing_p4.get("execution_plan_sha256") != plan["sha256"]
            or not isinstance(retained, Mapping)
            or retained.get("canonical_payload") != smoke
            or retained.get("canonical_sha256") != smoke_digest
            or retained.get("file_sha256") != canonical_smoke_file_sha
        ):
            raise PackageContractError(
                "Existing P4 transaction prefix does not match supplied evidence."
            )
        p4 = existing_p4
    else:
        atomic_write_json(p4_path, p4)
    p2 = load_json_object(PACKAGE_DIR / P2_RECEIPT_RELATIVE, label="P2 receipt")
    p3 = load_json_object(PACKAGE_DIR / P3_RECEIPT_RELATIVE, label="P3 receipt")
    control = load_json_object(
        PACKAGE_DIR / "control_plane_receipt.json",
        label="control plane receipt",
    )
    preauth = digested(
        {
            "schema": PACKAGE_PREAUTHORIZATION_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "package_validated": True,
            "direct_execution_count": 48,
            "package_manifest_sha256": manifest["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "source_archive_sha256": plan["source_archive"]["sha256"],
            "package_control_plane_sha256": control["sha256"],
            "p2_receipt_sha256": p2["sha256"],
            "p3_receipt_sha256": p3["sha256"],
            "p4_receipt_sha256": p4["sha256"],
            "remote_image": {
                "path": REMOTE_IMAGE_PATH,
                "sha256": REMOTE_IMAGE_SHA256,
                "byte_verification_state": "pending_remote_pre_submit",
                "verification_must_pass_before_condor_submit": True,
            },
            "execution_authorized": False,
            "submission_authorized": False,
            "remote_stage": False,
            "condor_submit": False,
            "submission_authorization_overlay": {
                "path": SUBMISSION_AUTHORIZATION_RELATIVE,
                "present": False,
                "required_before_condor_submit": True,
            },
            "submission_authorization_overlay_present": False,
            "submission_state": "awaiting_explicit_user_authorization",
            "next_action": (
                "show_the_final_validated_package_and_request_explicit_"
                "chtc_submission_authorization_before_condor_submit"
            ),
        }
    )
    atomic_write_json(preauth_path, preauth)
    final = validate_package(
        require_p4=True, require_authorization=False
    )
    return {**base, **final}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-p4", action="store_true")
    parser.add_argument("--require-authorization", action="store_true")
    parser.add_argument("--write-p4", action="store_true")
    parser.add_argument("--p4-smoke-result", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.write_p4:
            if args.p4_smoke_result is None:
                raise PackageContractError(
                    "--write-p4 requires --p4-smoke-result."
                )
            result = close_p4(
                smoke_result_path=args.p4_smoke_result.resolve()
            )
        else:
            if args.p4_smoke_result is not None:
                raise PackageContractError(
                    "--p4-smoke-result is valid only with --write-p4."
                )
            result = validate_package(
                require_p4=args.require_p4
                or args.require_authorization,
                require_authorization=args.require_authorization,
            )
        print(canonical_json_bytes(result).decode("utf-8"))
        return 0
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
