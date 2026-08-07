#!/usr/bin/env python3
"""Validate 18 fetched job archives and close the 20-cell Study-1 matrix.

The validator is evidence-only.  It does not promote results, edit either v7
bundle, or materialize the two measured-policy Append duplicates.  Those two
logical cells close only through receipts produced by ``link_shared_append``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from link_shared_append import build_shared_append_receipts  # noqa: E402
from build_attempt_selection import (  # noqa: E402
    ATTEMPT_DISPOSITIONS,
    attempt_archive_name,
)
from package_contract import (  # noqa: E402
    ATTEMPT_SELECTION_SCHEMA,
    COMPLETION_MATRIX_SCHEMA,
    EXPECTED_ARTIFACT_ROLES,
    FETCH_VALIDATION_SCHEMA,
    PACKAGE_ID,
    REMOTE_IMAGE_SHA256,
    V7_RELATIVE_ROOT,
    WORKER_RECEIPT_SCHEMA,
    PackageContractError,
    atomic_write_json,
    digested,
    direct_execution_ids,
    load_json_object,
    logical_cell_keys,
    safe_relative_path,
    sha256_file,
    require_sha256,
    validate_v7_authority,
    verify_exact_key_set,
    verify_self_digest,
)
from validate_package import validate_package  # noqa: E402
from objective_gates import (  # noqa: E402
    validate_cell_objective_gates,
    validate_objective_gate_matrix,
)


def _job_expected_members(job: Mapping[str, Any]) -> set[str]:
    paths = {
        safe_relative_path(
            job["artifact_paths"][role],
            label=f"{job['execution_id']} {role}",
        ).as_posix()
        for role in EXPECTED_ARTIFACT_ROLES
    }
    paths.add(
        safe_relative_path(
            job["worker_receipt_path"],
            label=f"{job['execution_id']} worker receipt",
        ).as_posix()
    )
    if len(paths) != 6:
        raise PackageContractError(
            f"Job does not have six distinct narrow outputs: "
            f"{job['execution_id']}"
        )
    return paths


def _load_attempt_selection(
    *,
    path: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    payload = load_json_object(path, label="attempt-selection receipt")
    verify_self_digest(payload, label="attempt-selection receipt")
    if (
        payload.get("schema") != ATTEMPT_SELECTION_SCHEMA
        or payload.get("package_id") != PACKAGE_ID
        or payload.get("execution_plan_sha256") != plan["sha256"]
        or payload.get("selection_policy")
        != "explicit_identity_never_mtime_or_lexical_latest_v1"
        or int(payload.get("direct_execution_count", -1)) != 18
    ):
        raise PackageContractError(
            "Attempt-selection control-plane authority drifted."
        )
    raw_rows = payload.get("selections")
    if not isinstance(raw_rows, list):
        raise PackageContractError(
            "Attempt-selection receipt has no ordered selections."
        )
    selections: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                f"Attempt selection[{index}] is not an object."
            )
        verify_exact_key_set(
            raw,
            (
                "execution_id",
                "cluster_id",
                "proc_id",
                "archive_name",
                "disposition",
                "archive_present",
                "archive_sha256",
                "archive_size_bytes",
            ),
            label=f"attempt selection[{index}]",
        )
        execution_id = str(raw.get("execution_id", ""))
        if execution_id in selections:
            raise PackageContractError(
                f"Duplicate selected attempt: {execution_id}"
            )
        cluster_id = raw.get("cluster_id")
        proc_id = raw.get("proc_id")
        if (
            isinstance(cluster_id, bool)
            or isinstance(proc_id, bool)
            or not isinstance(cluster_id, int)
            or not isinstance(proc_id, int)
            or cluster_id < 0
            or proc_id < 0
            or raw.get("archive_name")
            != attempt_archive_name(
                execution_id,
                cluster_id=cluster_id,
                proc_id=proc_id,
            )
            or raw.get("disposition") not in ATTEMPT_DISPOSITIONS
            or not isinstance(raw.get("archive_present"), bool)
        ):
            raise PackageContractError(
                f"Attempt identity drifted for {execution_id}."
            )
        if raw["archive_present"]:
            require_sha256(
                raw.get("archive_sha256"),
                label=f"{execution_id} selected archive SHA-256",
            )
            size = raw.get("archive_size_bytes")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise PackageContractError(
                    f"Selected archive size drifted for {execution_id}."
                )
        elif (
            raw.get("archive_sha256") is not None
            or raw.get("archive_size_bytes") is not None
        ):
            raise PackageContractError(
                f"Missing selected attempt has archive bytes: {execution_id}."
            )
        selections[execution_id] = raw
    verify_exact_key_set(
        selections,
        direct_execution_ids(),
        label="attempt-selection execution IDs",
    )
    if list(selections) != list(direct_execution_ids()):
        raise PackageContractError(
            "Attempt selections are not in canonical direct-execution order."
        )
    expected_status = (
        "ready"
        if all(
            row["disposition"] == "validate"
            and row["archive_present"] is True
            for row in selections.values()
        )
        else "incomplete"
    )
    if payload.get("status") != expected_status:
        raise PackageContractError(
            "Attempt-selection aggregate status drifted."
        )
    return payload, selections


def _extract_selected_job_archives(
    *,
    fetched_dir: Path,
    extracted_root: Path,
    jobs: Mapping[str, Mapping[str, Any]],
    selections: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Extract only explicitly selected attempts; preserve every other file."""

    statuses: dict[str, dict[str, Any]] = {}
    for execution_id in direct_execution_ids():
        job = jobs[execution_id]
        selection = selections[execution_id]
        disposition = str(selection["disposition"])
        selected_attempt = {
            "cluster_id": int(selection["cluster_id"]),
            "proc_id": int(selection["proc_id"]),
            "archive_name": str(selection["archive_name"]),
            "selection_disposition": disposition,
        }
        if disposition in {"blocked", "superseded"}:
            statuses[execution_id] = {
                "status": disposition,
                "selected_attempt": selected_attempt,
                "detail": (
                    "explicit_operator_disposition_in_attempt_selection"
                ),
            }
            continue
        if selection["archive_present"] is not True:
            statuses[execution_id] = {
                "status": "missing",
                "selected_attempt": selected_attempt,
                "detail": "selected_attempt_archive_not_present_at_selection",
            }
            continue

        archive_path = fetched_dir / str(selection["archive_name"])
        if not archive_path.is_file() or archive_path.is_symlink():
            statuses[execution_id] = {
                "status": "failed",
                "selected_attempt": selected_attempt,
                "detail": "selected_attempt_archive_unavailable_or_unsafe",
            }
            continue
        actual_sha256 = sha256_file(archive_path)
        actual_size = archive_path.stat().st_size
        if (
            actual_sha256 != selection["archive_sha256"]
            or actual_size != selection["archive_size_bytes"]
        ):
            statuses[execution_id] = {
                "status": "failed",
                "selected_attempt": selected_attempt,
                "detail": "selected_attempt_archive_binding_drifted",
                "observed_archive_sha256": actual_sha256,
                "observed_archive_size_bytes": actual_size,
            }
            continue
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                members = archive.getmembers()
                observed_names: set[str] = set()
                for member in members:
                    name = safe_relative_path(
                        member.name, label="fetched archive member"
                    ).as_posix()
                    if (
                        name in observed_names
                        or not member.isfile()
                        or member.issym()
                        or member.islnk()
                    ):
                        raise PackageContractError(
                            f"Unsafe/duplicate fetched member: {member.name}"
                        )
                    observed_names.add(name)
                if observed_names != _job_expected_members(job):
                    raise PackageContractError(
                        "Selected archive does not contain the exact six "
                        f"narrow outputs for {execution_id}."
                    )
                for member in members:
                    destination = extracted_root / member.name
                    if destination.exists():
                        raise PackageContractError(
                            f"Fetched artifact path collision: {member.name}"
                        )
                for member in members:
                    relative = safe_relative_path(
                        member.name, label="fetched extraction member"
                    )
                    target = extracted_root.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise PackageContractError(
                            f"Cannot read fetched member: {member.name}"
                        )
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
        except (OSError, PackageContractError, tarfile.TarError) as exc:
            statuses[execution_id] = {
                "status": "failed",
                "selected_attempt": selected_attempt,
                "detail": f"selected_attempt_extraction_failed:{exc}",
            }
            continue
        statuses[execution_id] = {
            "status": "candidate",
            "selected_attempt": selected_attempt,
            "transfer_archive": {
                "archive_path": archive_path.name,
                "archive_sha256": actual_sha256,
                "archive_size_bytes": actual_size,
            },
        }
    return statuses


def _validate_one_job(
    *,
    extracted_root: Path,
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    objective_authority: Mapping[str, Any],
) -> dict[str, Any]:
    execution_id = str(job["execution_id"])
    receipt_path = extracted_root / safe_relative_path(
        job["worker_receipt_path"],
        label=f"{execution_id} worker receipt",
    )
    receipt = load_json_object(
        receipt_path, label=f"{execution_id} worker receipt"
    )
    verify_self_digest(receipt, label=f"{execution_id} worker receipt")
    if (
        receipt.get("schema") != WORKER_RECEIPT_SCHEMA
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("execution_id") != execution_id
        or receipt.get("status") != "completed"
        or receipt.get("mode") != "execute"
        or receipt.get("execution_plan_sha256") != plan["sha256"]
        or receipt.get("job_spec_sha256") != job["sha256"]
        or receipt.get("authorization_sha256") != authorization["sha256"]
        or receipt.get("package_control_plane_sha256")
        != plan["package_control_plane"]["sha256"]
        or receipt.get("source_archive_sha256")
        != job["source_archive"]["sha256"]
        or receipt.get("remote_image_sha256") != REMOTE_IMAGE_SHA256
        or receipt.get("immutable_bundle_hashes_before")
        != receipt.get("immutable_bundle_hashes_after")
    ):
        raise PackageContractError(
            f"Worker receipt authority/state drifted: {execution_id}"
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PackageContractError(
            f"Worker receipt has no artifacts: {execution_id}"
        )
    verify_exact_key_set(
        artifacts,
        EXPECTED_ARTIFACT_ROLES,
        label=f"{execution_id} artifact roles",
    )
    validated_artifacts: dict[str, Any] = {}
    artifact_payloads: dict[str, Mapping[str, Any]] = {}
    for role in EXPECTED_ARTIFACT_ROLES:
        row = artifacts[role]
        if not isinstance(row, Mapping):
            raise PackageContractError(
                f"Invalid worker artifact row: {execution_id}:{role}"
            )
        expected_path = job["artifact_paths"][role]
        path = extracted_root / safe_relative_path(
            expected_path, label=f"{execution_id} {role}"
        )
        if (
            row.get("path") != expected_path
            or row.get("status") != "produced"
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != row.get("sha256")
            or path.stat().st_size != int(row.get("size_bytes", -1))
        ):
            raise PackageContractError(
                f"Fetched artifact drifted: {execution_id}:{role}"
            )
        payload = load_json_object(
            path, label=f"{execution_id} {role}"
        )
        artifact_payloads[role] = payload
        if role == "execution_manifest":
            verify_self_digest(
                payload, label=f"{execution_id} execution manifest"
            )
            authority = payload.get("package_authority")
            if (
                payload.get("schema")
                != "paper_i_ra_adapt_study1_execution_manifest_v1"
                or payload.get("execution_state") != "completed"
                or payload.get("exit_status") != 0
                or not isinstance(authority, Mapping)
                or authority.get("package_id") != PACKAGE_ID
                or authority.get("execution_plan_sha256")
                != plan["sha256"]
                or authority.get("job_spec_sha256") != job["sha256"]
                or authority.get("authorization_sha256")
                != authorization["sha256"]
                or authority.get("package_control_plane_sha256")
                != plan["package_control_plane"]["sha256"]
                or payload["sha256"] != row.get("canonical_sha256")
            ):
                raise PackageContractError(
                    f"Execution manifest authority drifted: {execution_id}"
                )
        elif role == "result":
            result_protocol = payload.get("protocol")
            if (
                not isinstance(result_protocol, Mapping)
                or result_protocol.get("sha256")
                != job["protocol"]["canonical_sha256"]
            ):
                raise PackageContractError(
                    f"Result protocol binding drifted: {execution_id}"
                )
        elif role == "summary":
            expected_schema = (
                "paper_i_append_run_summary_v1"
                if job["execution_entrypoint"] == "run_append_adapt"
                else "paper_i_run_summary_v1"
            )
            if payload.get("schema") != expected_schema:
                raise PackageContractError(
                    f"Typed summary schema drifted: {execution_id}"
                )
        validated_artifacts[role] = {
            "path": expected_path,
            "sha256": row["sha256"],
            "size_bytes": int(row["size_bytes"]),
        }
    replay_diagnostic = receipt.get("g11_replay_diagnostic")
    if not isinstance(replay_diagnostic, Mapping):
        raise PackageContractError(
            f"Worker lacks a G11 replay diagnostic: {execution_id}"
        )
    objective_gates = validate_cell_objective_gates(
        job=job,
        protocol=artifact_payloads["result"]["protocol"],
        checkpoint=artifact_payloads["checkpoint"],
        checkpoint_file_sha256=artifacts["checkpoint"]["sha256"],
        checkpoint_size_bytes=int(
            artifacts["checkpoint"]["size_bytes"]
        ),
        ledger=artifact_payloads["estimator_ledger"],
        ledger_file_sha256=artifacts["estimator_ledger"]["sha256"],
        ledger_size_bytes=int(
            artifacts["estimator_ledger"]["size_bytes"]
        ),
        result=artifact_payloads["result"],
        summary=artifact_payloads["summary"],
        objective_authority=objective_authority,
        replay_diagnostic=replay_diagnostic,
    )
    if receipt.get("objective_gates") != objective_gates:
        raise PackageContractError(
            f"Worker objective-gate receipt drifted: {execution_id}"
        )
    return {
        "execution_id": execution_id,
        "worker_receipt_path": job["worker_receipt_path"],
        "worker_receipt_canonical_sha256": receipt["sha256"],
        "worker_receipt_file_sha256": sha256_file(receipt_path),
        "artifacts": validated_artifacts,
        "objective_gates": objective_gates,
        "_result_payload": artifact_payloads["result"],
        "status": "done",
    }


def validate_fetched(
    *,
    package_dir: Path,
    fetched_dir: Path,
    attempt_selection_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise PackageContractError(
            f"Refusing to overwrite validation output: {output_dir}"
        )
    if not fetched_dir.is_dir():
        raise PackageContractError(
            f"Fetched directory is unavailable: {fetched_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=str(output_dir.parent),
        )
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="paper_i_study1_fetched_"
        ) as fetched_raw, tempfile.TemporaryDirectory(
            prefix="paper_i_study1_source_"
        ) as source_raw:
            fetched_root = Path(fetched_raw)
            source_root = Path(source_raw)
            package_receipt = validate_package(
                package_dir=package_dir,
                image_path=None,
                extracted_root=source_root,
            )
            plan = load_json_object(
                package_dir / "execution_plan.json",
                label="execution plan",
            )
            verify_self_digest(plan, label="execution plan")
            authorization = load_json_object(
                package_dir
                / "authority/execution_authorization_receipt.json",
                label="authorization receipt",
            )
            verify_self_digest(
                authorization, label="authorization receipt"
            )
            authority = validate_v7_authority(
                source_root,
                v7_root=source_root / V7_RELATIVE_ROOT,
                final_receipt_path=(
                    package_dir
                    / "authority/v7_final_materialization_receipt.json"
                ),
                objective_gate_authority_path=(
                    package_dir
                    / "authority/"
                    "study1_objective_gate_authority_receipt.json"
                ),
            )
            dedupe = authority["dedupe_contract"]
            jobs = {
                execution_id: load_json_object(
                    package_dir / "jobs" / f"{execution_id}.json",
                    label=f"{execution_id} job spec",
                )
                for execution_id in direct_execution_ids()
            }
            for execution_id, job in jobs.items():
                verify_self_digest(job, label=f"{execution_id} job spec")

            (
                attempt_selection,
                selected_attempts,
            ) = _load_attempt_selection(
                path=attempt_selection_path,
                plan=plan,
            )
            attempt_statuses = _extract_selected_job_archives(
                fetched_dir=fetched_dir,
                extracted_root=fetched_root,
                jobs=jobs,
                selections=selected_attempts,
            )
            direct_receipts: list[dict[str, Any]] = []
            for execution_id in direct_execution_ids():
                attempt_status = attempt_statuses[execution_id]
                if attempt_status["status"] != "candidate":
                    continue
                try:
                    validated = _validate_one_job(
                        extracted_root=fetched_root,
                        job=jobs[execution_id],
                        plan=plan,
                        authorization=authorization,
                        objective_authority=authority[
                            "objective_gate_authority"
                        ],
                    )
                except (
                    KeyError,
                    OSError,
                    PackageContractError,
                    TypeError,
                    ValueError,
                ) as exc:
                    attempt_statuses[execution_id] = {
                        **attempt_status,
                        "status": "failed",
                        "detail": (
                            "selected_attempt_evidence_validation_failed:"
                            f"{type(exc).__name__}:{exc}"
                        ),
                    }
                    continue
                validated["transfer_archive"] = attempt_status[
                    "transfer_archive"
                ]
                validated["selected_attempt"] = attempt_status[
                    "selected_attempt"
                ]
                direct_receipts.append(validated)
                attempt_statuses[execution_id] = {
                    **attempt_status,
                    "status": "done",
                    "worker_receipt_canonical_sha256": validated[
                        "worker_receipt_canonical_sha256"
                    ],
                }

            all_direct_done = all(
                attempt_statuses[execution_id]["status"] == "done"
                for execution_id in direct_execution_ids()
            )
            shared_receipts: list[dict[str, Any]] = []
            shared_failure: str | None = None
            if all_direct_done:
                try:
                    shared_receipts = build_shared_append_receipts(
                        source_root=source_root,
                        fetched_root=fetched_root,
                        output_dir=temp_output,
                        plan=plan,
                        dedupe=dedupe,
                    )
                except (
                    KeyError,
                    OSError,
                    PackageContractError,
                    TypeError,
                    ValueError,
                ) as exc:
                    shared_failure = (
                        "shared_append_equivalence_validation_failed:"
                        f"{type(exc).__name__}:{exc}"
                    )
            shared_by_key = {
                row["reference_logical_key"]: row
                for row in shared_receipts
            }
            direct_by_logical = {
                f"{jobs[row['execution_id']]['bundle_id']}::"
                f"{jobs[row['execution_id']]['cell_id']}": row
                for row in direct_receipts
            }
            completion_matrix: list[dict[str, Any]] = []
            plan_logical = {
                row["logical_key"]: row
                for row in plan["logical_cells"]
            }
            for logical_key in logical_cell_keys():
                plan_row = plan_logical[logical_key]
                if plan_row["direct_execution_required"]:
                    direct = direct_by_logical.get(logical_key)
                    execution_id = str(plan_row["canonical_execution_id"])
                    attempt_status = attempt_statuses[execution_id]
                    row = {
                        "logical_key": logical_key,
                        "bundle_id": plan_row["bundle_id"],
                        "cell_id": plan_row["cell_id"],
                        "fulfillment_kind": plan_row[
                            "execution_fulfillment"
                        ]["fulfillment_kind"],
                        "status": attempt_status["status"],
                        "execution_id": execution_id,
                        "selected_attempt": attempt_status[
                            "selected_attempt"
                        ],
                    }
                    if direct is not None:
                        row["worker_receipt_canonical_sha256"] = direct[
                            "worker_receipt_canonical_sha256"
                        ]
                    else:
                        row["status_detail"] = attempt_status.get("detail")
                    completion_matrix.append(row)
                else:
                    shared = shared_by_key.get(logical_key)
                    canonical_execution_id = str(
                        plan_row["canonical_execution_id"]
                    )
                    canonical_status = attempt_statuses[
                        canonical_execution_id
                    ]["status"]
                    row = {
                        "logical_key": logical_key,
                        "bundle_id": plan_row["bundle_id"],
                        "cell_id": plan_row["cell_id"],
                        "fulfillment_kind": (
                            "shared_result_reference_v1"
                        ),
                        "status": (
                            "done"
                            if shared is not None
                            else "failed"
                            if shared_failure is not None
                            else canonical_status
                        ),
                        "canonical_execution_id": canonical_execution_id,
                    }
                    if shared is not None:
                        row["equivalence_receipt_canonical_sha256"] = shared[
                            "canonical_sha256"
                        ]
                    elif shared_failure is not None:
                        row["status_detail"] = shared_failure
                    else:
                        row["status_detail"] = (
                            "canonical_shared_execution_not_done"
                        )
                    completion_matrix.append(row)
            if (
                len(completion_matrix) != 20
                or sum(
                    row["fulfillment_kind"]
                    == "shared_result_reference_v1"
                    for row in completion_matrix
                )
                != 2
            ):
                raise PackageContractError(
                    "The exact 20-cell completion matrix shape drifted."
                )
            permitted_states = {
                "done",
                "failed",
                "missing",
                "blocked",
                "superseded",
            }
            if any(
                row["status"] not in permitted_states
                for row in completion_matrix
            ):
                raise PackageContractError(
                    "The completion matrix contains an unknown state."
                )
            completion_states = {
                row["logical_key"]: str(row["status"])
                for row in completion_matrix
            }
            matrix_complete = all(
                state == "done" for state in completion_states.values()
            )
            objective_gate_matrix: dict[str, Any] | None = None
            if matrix_complete:
                objective_gate_matrix = validate_objective_gate_matrix(
                    plan=plan,
                    jobs=jobs,
                    cell_records=[
                        {
                            **row,
                            "result_payload": row["_result_payload"],
                        }
                        for row in direct_receipts
                    ],
                    shared_receipts=shared_receipts,
                    objective_authority=authority[
                        "objective_gate_authority"
                    ],
                    completion_states=completion_states,
                )
                atomic_write_json(
                    temp_output / "objective_gate_matrix.json",
                    objective_gate_matrix,
                )

            completion_payload = digested(
                {
                    "schema": COMPLETION_MATRIX_SCHEMA,
                    "package_id": PACKAGE_ID,
                    "execution_plan_sha256": plan["sha256"],
                    "attempt_selection_sha256": attempt_selection["sha256"],
                    "logical_cell_count": 20,
                    "direct_execution_count": 18,
                    "shared_reference_count": 2,
                    "state_vocabulary": sorted(permitted_states),
                    "rows": completion_matrix,
                    "status": (
                        "complete" if matrix_complete else "incomplete"
                    ),
                }
            )
            atomic_write_json(
                temp_output / "completion_matrix.json",
                completion_payload,
            )
            public_direct_receipts = [
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                }
                for row in direct_receipts
            ]
            validation_receipt = digested(
                {
                    "schema": FETCH_VALIDATION_SCHEMA,
                    "package_id": PACKAGE_ID,
                    "status": (
                        "passed" if matrix_complete else "incomplete"
                    ),
                    "package_manifest_sha256": package_receipt[
                        "package_manifest_sha256"
                    ],
                    "execution_plan_sha256": plan["sha256"],
                    "authorization_sha256": authorization["sha256"],
                    "v7_final_receipt_sha256": authority[
                        "final_receipt"
                    ]["sha256"],
                    "study1_objective_gate_authority_sha256": authority[
                        "objective_gate_authority"
                    ]["sha256"],
                    "study1_dedupe_sha256": dedupe["sha256"],
                    "package_control_plane_sha256": plan[
                        "package_control_plane"
                    ]["sha256"],
                    "source_archive_sha256": package_receipt[
                        "source_archive_sha256"
                    ],
                    "remote_image_sha256": REMOTE_IMAGE_SHA256,
                    "logical_cell_count": 20,
                    "direct_execution_count": 18,
                    "shared_reference_count": 2,
                    "attempt_selection": {
                        "path": attempt_selection_path.name,
                        "canonical_sha256": attempt_selection["sha256"],
                        "file_sha256": sha256_file(
                            attempt_selection_path
                        ),
                    },
                    "direct_attempt_statuses": [
                        {
                            "execution_id": execution_id,
                            **attempt_statuses[execution_id],
                        }
                        for execution_id in direct_execution_ids()
                    ],
                    "direct_receipts": public_direct_receipts,
                    "shared_append_equivalence_receipts": shared_receipts,
                    "completion_matrix": {
                        "path": "completion_matrix.json",
                        "canonical_sha256": completion_payload["sha256"],
                        "file_sha256": sha256_file(
                            temp_output / "completion_matrix.json"
                        ),
                    },
                    "objective_gate_matrix": (
                        None
                        if objective_gate_matrix is None
                        else {
                            "path": "objective_gate_matrix.json",
                            "canonical_sha256": (
                                objective_gate_matrix["sha256"]
                            ),
                            "file_sha256": sha256_file(
                                temp_output / "objective_gate_matrix.json"
                            ),
                        }
                    ),
                    "measured_append_physical_files_materialized": False,
                    "evidence_promotion_performed": False,
                    "bundle_mutation_performed": False,
                }
            )
            atomic_write_json(
                temp_output / "fetched_validation_receipt.json",
                validation_receipt,
            )
        os.replace(temp_output, output_dir)
        return validation_receipt
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir", type=Path, default=PACKAGE_DIR
    )
    parser.add_argument("--fetched-dir", type=Path, required=True)
    parser.add_argument(
        "--attempt-selection", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = validate_fetched(
            package_dir=args.package_dir.resolve(),
            fetched_dir=args.fetched_dir.resolve(),
            attempt_selection_path=args.attempt_selection.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt.get("status") == "passed" else 3
    except (PackageContractError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
