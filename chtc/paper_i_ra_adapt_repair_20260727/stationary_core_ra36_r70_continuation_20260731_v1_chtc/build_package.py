#!/usr/bin/env python3
"""Materialize the inert, source-locked stationary-core RA r70 package."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

import ijson


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    AUTHORIZATION_SCHEMA,
    CAMPAIGN_ID,
    CELL_COUNT,
    COLLISION_CLUSTER_ID,
    COLLISION_PROC_IDS,
    COLLISION_QUEUE_RELATIVE,
    COLLISION_STATE_SNAPSHOT_RELATIVE,
    COLLISION_STATUS_NAME,
    COLLISION_STATUS_SCHEMA,
    COLLISION_SUBMISSION_RECEIPT_RELATIVE,
    CONTROL_FILES,
    EXECUTION_PLAN_NAME,
    EXECUTION_PLAN_SCHEMA,
    EXECUTION_TARGET,
    FRESH_COUNT,
    GENERATED_FILES,
    JOB_SCHEMA,
    MAX_RUNTIME_SECONDS,
    PACKAGE_ID,
    PACKAGE_MANIFEST_NAME,
    PACKAGE_MANIFEST_SCHEMA,
    QUEUE_NAME,
    RESOURCE_WEIGHTING_SCOPE,
    RESUME_COUNT,
    RESUME_INPUTS_NAME,
    RESUME_INPUTS_SCHEMA,
    RUN_CLASS,
    SOURCE_ARCHIVES_NAME,
    SOURCE_ARCHIVES_SCHEMA,
    SOURCE_FAMILIES,
    SOURCE_HORIZON,
    SOURCE_LOCK_AUDIT_NAME,
    SOURCE_REPORT_RELATIVE,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    collision_map,
    digested,
    file_binding,
    load_json,
    planned_rows,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    validate_source_protocol,
    verify_self_digest,
)


class _HashingReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        block = self.source.read(size)
        self.digest.update(block)
        self.size += len(block)
        return block

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()


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


def _exclusive_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PackageContractError(
            f"Refusing to overwrite copied source lock: {destination}"
        )
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with source.open("rb") as input_stream:
            with temporary.open("xb") as output_stream:
                shutil.copyfileobj(
                    input_stream, output_stream, length=1024 * 1024
                )
                output_stream.flush()
                os.fsync(output_stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _copy_source_archives(
    repo_root: Path,
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family_id, raw in sorted(SOURCE_FAMILIES.items()):
        source_root = repo_root / str(raw["source_package_root"])
        packaged_root = PACKAGE_DIR / str(raw["packaged_root"])
        source_archive = source_root / "source_locked.tar.gz"
        source_manifest = source_root / "source_archive_manifest.json"
        manifest = load_json(
            source_manifest,
            label=f"{family_id} source archive manifest",
        )
        verify_self_digest(
            manifest,
            label=f"{family_id} source archive manifest",
        )
        archive_binding = manifest.get("archive")
        if (
            not isinstance(archive_binding, Mapping)
            or archive_binding.get("sha256")
            != sha256_file(source_archive)
            or int(archive_binding.get("size_bytes", -1))
            != source_archive.stat().st_size
        ):
            raise PackageContractError(
                f"{family_id} source archive binding drifted."
            )
        packaged_archive = packaged_root / "source_locked.tar.gz"
        packaged_manifest = (
            packaged_root / "source_archive_manifest.json"
        )
        _exclusive_copy(source_archive, packaged_archive)
        _exclusive_copy(source_manifest, packaged_manifest)
        families[family_id] = {
            "source_package_root": str(raw["source_package_root"]),
            "original_archive": file_binding(
                source_archive, repo_root=repo_root
            ),
            "original_manifest": file_binding(
                source_manifest, repo_root=repo_root
            ),
            "packaged_archive": {
                "path": packaged_archive.relative_to(
                    PACKAGE_DIR
                ).as_posix(),
                "sha256": sha256_file(packaged_archive),
                "size_bytes": packaged_archive.stat().st_size,
            },
            "packaged_manifest": {
                "path": packaged_manifest.relative_to(
                    PACKAGE_DIR
                ).as_posix(),
                "sha256": sha256_file(packaged_manifest),
                "size_bytes": packaged_manifest.stat().st_size,
                "canonical_sha256": manifest["sha256"],
            },
            "member_count": int(manifest.get("member_count", -1)),
            "exact_copy": True,
        }
    return digested(
        {
            "schema": SOURCE_ARCHIVES_SCHEMA,
            "package_id": PACKAGE_ID,
            "status": "passed",
            "family_count": len(families),
            "families": families,
        }
    )


def _checkpoint_member(row: Mapping[str, Any]) -> str:
    if row["route_id"].endswith("_always"):
        return (
            "worker_outputs/runs/"
            f"{row['source_execution_id']}/checkpoints/current.json"
        )
    return "worker_outputs/checkpoint.json"


def _stream_member_to_file(
    *,
    archive_path: Path,
    member_name: str,
    destination: Path,
) -> tuple[str, int]:
    found = False
    digest = hashlib.sha256()
    size = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name != member_name:
                continue
            if (
                found
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise PackageContractError(
                    f"Unsafe duplicate attempt member: {member_name}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"Unreadable attempt member: {member_name}"
                )
            with destination.open("xb") as output:
                for block in iter(
                    lambda: stream.read(1024 * 1024), b""
                ):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                output.flush()
                os.fsync(output.fileno())
            if size != member.size:
                raise PackageContractError(
                    f"Checkpoint member size drifted: {member_name}"
                )
            found = True
            break
    if not found:
        raise PackageContractError(
            f"Attempt lacks checkpoint member: {member_name}"
        )
    return digest.hexdigest(), size


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "active_prefix_checkpoint_count": 0
    }
    ledger: dict[str, Any] = {}
    resume: dict[str, Any] = {}
    scalar_events = {
        "boolean",
        "integer",
        "double",
        "number",
        "null",
        "string",
    }
    with checkpoint_path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream):
            if (
                prefix == "adapt_vqe.active_prefix_checkpoints.item"
                and event == "start_map"
            ):
                metadata["active_prefix_checkpoint_count"] += 1
            elif prefix == "schema_version" and event in scalar_events:
                metadata["schema_version"] = value
            elif prefix == "checkpoint.depth" and event in scalar_events:
                metadata["checkpoint_depth"] = int(value)
            elif (
                prefix == "adapt_vqe.history_count"
                and event in scalar_events
            ):
                metadata["history_count"] = int(value)
            elif (
                prefix == "adapt_vqe.history_checkpoint_complete"
                and event in scalar_events
            ):
                metadata["history_checkpoint_complete"] = bool(value)
            elif (
                prefix == "adapt_vqe.strict_replay.passed"
                and event in scalar_events
            ):
                metadata["strict_replay_passed"] = bool(value)
            elif (
                prefix == "adapt_vqe.route_profile"
                and event in scalar_events
            ):
                metadata["route_profile"] = str(value)
            elif (
                prefix
                == "adapt_vqe.sr_route_profile_contract_sha256"
                and event in scalar_events
            ):
                metadata["route_contract_sha256"] = str(value)
            elif (
                prefix.startswith(
                    "adapt_vqe.estimator_call_ledger_checkpoint."
                )
                and event in scalar_events
            ):
                ledger[prefix.rsplit(".", 1)[-1]] = value
            elif (
                prefix.startswith(
                    "adapt_vqe.verified_singleton_resume_sidecar."
                )
                and event in scalar_events
            ):
                resume[prefix.rsplit(".", 1)[-1]] = value
    metadata["estimator_call_ledger_checkpoint"] = ledger
    metadata["verified_singleton_resume_sidecar"] = resume
    return metadata


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = int(size)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_resume_archive(
    *,
    attempt_path: Path,
    checkpoint_path: Path,
    checkpoint_member: str,
    metadata: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    ledger_pointer = metadata["estimator_call_ledger_checkpoint"]
    resume_pointer = metadata["verified_singleton_resume_sidecar"]
    if not isinstance(ledger_pointer, Mapping) or not isinstance(
        resume_pointer, Mapping
    ):
        raise PackageContractError(
            "Checkpoint pointer payloads are malformed."
        )
    pointer_rows = (
        ("estimator_ledger_checkpoint", ledger_pointer),
        ("verified_resume_sidecar", resume_pointer),
    )
    checkpoint_parent = PurePosixPath(checkpoint_member).parent
    source_members = {
        (
            checkpoint_parent
            / safe_relative_path(
                pointer["path"], label=f"{role} pointer path"
            )
        ).as_posix(): (role, pointer)
        for role, pointer in pointer_rows
    }
    if len(source_members) != 2:
        raise PackageContractError(
            "Checkpoint sidecar pointers are not distinct."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    bindings: list[dict[str, Any]] = []
    checkpoint_name = checkpoint_path.name
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    mode="w",
                    fileobj=compressed,
                    format=tarfile.PAX_FORMAT,
                ) as output:
                    checkpoint_sha256 = sha256_file(checkpoint_path)
                    with checkpoint_path.open("rb") as stream:
                        output.addfile(
                            _tar_info(
                                f"checkpoint/{checkpoint_name}",
                                checkpoint_path.stat().st_size,
                            ),
                            stream,
                        )
                    bindings.append(
                        {
                            "role": "checkpoint",
                            "path": f"checkpoint/{checkpoint_name}",
                            "sha256": checkpoint_sha256,
                            "size_bytes": checkpoint_path.stat().st_size,
                            "source_member": checkpoint_member,
                        }
                    )
                    found: set[str] = set()
                    with tarfile.open(
                        attempt_path, "r:gz"
                    ) as source_archive:
                        for member in source_archive:
                            expected = source_members.get(member.name)
                            if expected is None:
                                continue
                            role, pointer = expected
                            if (
                                member.name in found
                                or not member.isfile()
                                or member.issym()
                                or member.islnk()
                            ):
                                raise PackageContractError(
                                    "Unsafe duplicate checkpoint sidecar: "
                                    f"{member.name}"
                                )
                            source = source_archive.extractfile(member)
                            if source is None:
                                raise PackageContractError(
                                    "Unreadable checkpoint sidecar: "
                                    f"{member.name}"
                                )
                            hashing = _HashingReader(source)
                            packaged_name = (
                                f"checkpoint/{PurePosixPath(member.name).name}"
                            )
                            output.addfile(
                                _tar_info(packaged_name, member.size),
                                hashing,
                            )
                            if (
                                hashing.size != member.size
                                or hashing.sha256 != pointer.get("sha256")
                            ):
                                raise PackageContractError(
                                    f"{role} sidecar digest drifted."
                                )
                            bindings.append(
                                {
                                    "role": role,
                                    "path": packaged_name,
                                    "sha256": hashing.sha256,
                                    "size_bytes": hashing.size,
                                    "source_member": member.name,
                                }
                            )
                            found.add(member.name)
                            if found == set(source_members):
                                break
                    if found != set(source_members):
                        raise PackageContractError(
                            "Attempt lacks a pointer-closed sidecar."
                        )
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "archive": {
            "path": destination.relative_to(PACKAGE_DIR).as_posix(),
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        },
        "checkpoint_path": f"checkpoint/{checkpoint_name}",
        "checkpoint_sha256": bindings[0]["sha256"],
        "member_count": len(bindings),
        "members": bindings,
        "pointer_closed": True,
        "superseded_sidecars_retained": False,
    }


def _materialize_resume_inputs(
    *,
    repo_root: Path,
    rows: tuple[dict[str, Any], ...],
    protocol_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    resume_rows = [
        row
        for row in rows
        if row["execution_mode"]
        == "authenticated_resume_50_to_70"
    ]
    with tempfile.TemporaryDirectory(
        prefix="paper-i-ra-r70-checkpoint."
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        for index, row in enumerate(resume_rows, start=1):
            execution_id = str(row["execution_id"])
            source = row["resume_source"]
            attempt = Path(str(source["attempt_resolved_path"]))
            if (
                not attempt.is_file()
                or attempt.is_symlink()
                or attempt.stat().st_size
                != int(source["attempt_size_bytes"])
                or sha256_file(attempt) != source["attempt_sha256"]
            ):
                raise PackageContractError(
                    f"Selected attempt drifted for {execution_id}."
                )
            checkpoint_member = _checkpoint_member(row)
            checkpoint_path = (
                temporary_root / f"{index:02d}.checkpoint.json"
            )
            checkpoint_sha256, checkpoint_size = (
                _stream_member_to_file(
                    archive_path=attempt,
                    member_name=checkpoint_member,
                    destination=checkpoint_path,
                )
            )
            metadata = _checkpoint_metadata(checkpoint_path)
            protocol = protocol_bindings[execution_id]
            if (
                metadata.get("schema_version")
                != "static_adapt_current_checkpoint_v1"
                or int(metadata.get("checkpoint_depth", -1))
                != SOURCE_HORIZON
                or int(metadata.get("history_count", -1))
                != SOURCE_HORIZON
                or int(
                    metadata.get(
                        "active_prefix_checkpoint_count", -1
                    )
                )
                != SOURCE_HORIZON
                or metadata.get("history_checkpoint_complete")
                is not True
                or metadata.get("strict_replay_passed") is not True
                or metadata.get("route_profile")
                != protocol["route_profile"]
                or metadata.get("route_contract_sha256")
                != protocol["route_contract_sha256"]
            ):
                raise PackageContractError(
                    f"Checkpoint authentication metadata drifted for "
                    f"{execution_id}."
                )
            ledger_pointer = metadata[
                "estimator_call_ledger_checkpoint"
            ]
            resume_pointer = metadata[
                "verified_singleton_resume_sidecar"
            ]
            if (
                ledger_pointer.get("status") != "complete"
                or int(
                    ledger_pointer.get("checkpoint_depth", -1)
                )
                != SOURCE_HORIZON
                or resume_pointer.get("status") != "complete"
                or resume_pointer.get("enabled") is not True
            ):
                raise PackageContractError(
                    f"Checkpoint pointer closure is incomplete for "
                    f"{execution_id}."
                )
            archive_path = (
                PACKAGE_DIR
                / "resume_inputs"
                / f"{execution_id}.tar.gz"
            )
            compact = _write_resume_archive(
                attempt_path=attempt,
                checkpoint_path=checkpoint_path,
                checkpoint_member=checkpoint_member,
                metadata=metadata,
                destination=archive_path,
            )
            cells[execution_id] = {
                "source_attempt": {
                    "path": str(source["attempt_report_path"]),
                    "resolved_path": attempt.as_posix(),
                    "sha256": str(source["attempt_sha256"]),
                    "size_bytes": int(
                        source["attempt_size_bytes"]
                    ),
                    "source_package_id": str(
                        source["source_package_id"]
                    ),
                    "source_receipt_index": int(
                        source["source_receipt_index"]
                    ),
                },
                "source_checkpoint_member": checkpoint_member,
                "source_checkpoint_sha256": checkpoint_sha256,
                "source_checkpoint_size_bytes": checkpoint_size,
                "authentication": {
                    key: metadata[key]
                    for key in (
                        "schema_version",
                        "checkpoint_depth",
                        "history_count",
                        "history_checkpoint_complete",
                        "active_prefix_checkpoint_count",
                        "strict_replay_passed",
                        "route_profile",
                        "route_contract_sha256",
                    )
                },
                **compact,
            }
            print(
                f"[{index}/{len(resume_rows)}] compacted {execution_id}",
                flush=True,
            )
            checkpoint_path.unlink()
    if len(cells) != RESUME_COUNT:
        raise PackageContractError(
            "Resume input materialization did not close 27 cells."
        )
    return digested(
        {
            "schema": RESUME_INPUTS_SCHEMA,
            "package_id": PACKAGE_ID,
            "status": "passed",
            "resume_cell_count": len(cells),
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "closure": (
                "checkpoint_plus_terminal_estimator_ledger_plus_"
                "verified_resume_sidecar_v1"
            ),
            "cells": cells,
        }
    )


def _derived_request(
    *,
    source_request: Mapping[str, Any],
    execution_id: str,
) -> dict[str, Any]:
    request = copy.deepcopy(dict(source_request))
    execution = request.get("execution")
    observation = request.get("observation")
    if not isinstance(execution, dict) or not isinstance(
        observation, dict
    ):
        raise PackageContractError(
            f"Source request is incomplete for {execution_id}."
        )
    stop = execution.get("stop")
    checkpoint = observation.get("checkpoint")
    ledger = observation.get("estimator_ledger")
    if (
        not isinstance(stop, dict)
        or not isinstance(checkpoint, dict)
        or not isinstance(ledger, dict)
    ):
        raise PackageContractError(
            f"Source request mechanics are incomplete for {execution_id}."
        )
    stop["maximum_controller_rounds"] = TARGET_HORIZON
    checkpoint["path"] = (
        f"runs/{execution_id}/checkpoints/current.json"
    )
    ledger["path"] = (
        f"runs/{execution_id}/result/estimator_ledger.json"
    )
    return request


def _source_lock_audit(
    *,
    repo_root: Path,
    rows: tuple[dict[str, Any], ...],
    protocol_bindings: Mapping[str, Mapping[str, Any]],
    resume_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    planned: list[dict[str, Any]] = []
    for row in rows:
        execution_id = str(row["execution_id"])
        protocol = protocol_bindings[execution_id]
        resume = (
            resume_inputs["cells"].get(execution_id)
            if isinstance(resume_inputs.get("cells"), Mapping)
            else None
        )
        planned.append(
            {
                "execution_id": execution_id,
                "base_execution_id": row["base_execution_id"],
                "source_protocol": {
                    key: protocol[key]
                    for key in (
                        "path",
                        "sha256",
                        "canonical_sha256",
                        "size_bytes",
                        "route_profile",
                        "route_contract_sha256",
                    )
                },
                "source_family": row["source_family"],
                "source_variable_value": SOURCE_HORIZON,
                "target_variable_value": TARGET_HORIZON,
                "settings_hash": hashlib.sha256(
                    canonical_json_bytes(
                        _derived_request(
                            source_request=protocol["request"],
                            execution_id=execution_id,
                        )
                    )
                ).hexdigest(),
                "changed_fields_vs_source": [
                    "maximum_controller_rounds"
                ],
                "changed_serialized_paths_vs_source": [
                    "protocol.horizon",
                    (
                        "request.execution.stop."
                        "maximum_controller_rounds"
                    ),
                    (
                        "stopping_rule."
                        "maximum_controller_rounds"
                    ),
                ],
                "execution_mechanics_changed": [
                    (
                        "accepted_state_resume_checkpoint"
                        if row["execution_mode"]
                        == "authenticated_resume_50_to_70"
                        else "fresh_start"
                    ),
                    "checkpoint_output_path",
                    "estimator_ledger_output_path",
                ],
                "non_swept_settings_diff": [],
                "fields_added_by_current_defaults": [],
                "unresolved_source_fields": [],
                "anchor": {
                    "kind": (
                        "authenticated_round50_checkpoint"
                        if resume is not None
                        else "unavailable_live_round50_predecessor"
                    ),
                    "anchor_reproduces_source": (
                        resume is not None
                    ),
                    "operator_sequence_match": (
                        True if resume is not None else None
                    ),
                    "non_swept_settings_diff": [],
                    "checkpoint_sha256": (
                        resume["checkpoint_sha256"]
                        if resume is not None
                        else None
                    ),
                },
                "status": (
                    "passed_authenticated_resume_anchor"
                    if resume is not None
                    else "blocked_live_r50_predecessor"
                ),
            }
        )
    return digested(
        {
            "schema": "source_locked_sensitivity_audit_v1",
            "package_id": PACKAGE_ID,
            "source": {
                "table_label": (
                    "stationary-core diagnostic 48-cell report"
                ),
                "method": "RA-ADAPT",
                "source_json": SOURCE_REPORT_RELATIVE,
                "source_sha256": sha256_file(
                    repo_root / SOURCE_REPORT_RELATIVE
                ),
                "runner_mode": "exact_source_archive_direct_request",
                "source_variable_value": SOURCE_HORIZON,
            },
            "sweep": {
                "run_class": RUN_CLASS,
                "variable": "maximum_controller_rounds",
                "grid": [TARGET_HORIZON],
                "runner_mode": (
                    "exact_source_archive_direct_request"
                ),
                "wrapper_used": False,
                "wrapper_kind": None,
                "baseline_materialization_status": "complete",
                "unresolved_source_fields": [],
                "fields_added_by_current_defaults": [],
                "settings_changed": [
                    "maximum_controller_rounds"
                ],
            },
            "planned_rows": planned,
            "anchor": {
                "authenticated_resume_anchor_count": RESUME_COUNT,
                "blocked_fresh_anchor_count": FRESH_COUNT,
                "all_available_resume_anchors_close": True,
                "all_fresh_rows_blocked": True,
            },
            "status": (
                "blocked_live_r50_predecessors_"
                "with_27_authenticated_resume_anchors"
            ),
        }
    )


def _collision_status(repo_root: Path) -> dict[str, Any]:
    mapped = collision_map(repo_root)
    return digested(
        {
            "schema": COLLISION_STATUS_SCHEMA,
            "package_id": PACKAGE_ID,
            "status": "blocked",
            "blocking": True,
            "reason": (
                "nine_fresh_r70_always_rows_overlap_live_exact_r50_"
                "predecessors"
            ),
            "cluster_id": COLLISION_CLUSTER_ID,
            "proc_ids": list(COLLISION_PROC_IDS),
            "rows": [
                {
                    "base_execution_id": execution_id,
                    **mapped[execution_id],
                }
                for execution_id in sorted(
                    mapped, key=lambda item: mapped[item]["proc_id"]
                )
            ],
            "bindings": {
                "queue": file_binding(
                    repo_root / COLLISION_QUEUE_RELATIVE,
                    repo_root=repo_root,
                ),
                "submission_receipt": file_binding(
                    repo_root
                    / COLLISION_SUBMISSION_RECEIPT_RELATIVE,
                    repo_root=repo_root,
                ),
                "local_state_snapshot": file_binding(
                    repo_root / COLLISION_STATE_SNAPSHOT_RELATIVE,
                    repo_root=repo_root,
                ),
            },
            "external_state_revalidation_required": True,
            "submit_descriptor_present": False,
            "may_submit": False,
            "may_supersede_predecessors": False,
            "may_remove_predecessors": False,
        }
    )


def main() -> int:
    repo_root = repo_root_from_script(__file__)
    if PACKAGE_DIR.relative_to(repo_root).as_posix() != (
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "stationary_core_ra36_r70_continuation_20260731_v1_chtc"
    ):
        raise PackageContractError("Package path drifted.")
    if (PACKAGE_DIR / "authority").exists():
        raise PackageContractError(
            "The inert package must not contain authority."
        )
    if (PACKAGE_DIR / "submit.sub").exists():
        raise PackageContractError(
            "Collision-blocked package must not contain submit.sub."
        )
    for relative in CONTROL_FILES:
        path = PACKAGE_DIR / relative
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Control file is missing: {relative}"
            )
    for relative in GENERATED_FILES:
        path = PACKAGE_DIR / relative
        if path.exists() or path.is_symlink():
            raise PackageContractError(
                f"Refusing to overwrite generated file: {path}"
            )
    for directory in ("jobs", "resume_inputs", "source_archives"):
        path = PACKAGE_DIR / directory
        if path.exists() or path.is_symlink():
            raise PackageContractError(
                f"Refusing to overwrite generated directory: {path}"
            )

    provenance_path = repo_root / SOURCE_REPORT_RELATIVE
    provenance = load_json(
        provenance_path, label="stationary-core provenance"
    )
    rows = planned_rows(
        repo_root=repo_root, provenance=provenance
    )
    protocol_bindings = {
        str(row["execution_id"]): validate_source_protocol(
            repo_root=repo_root, row=row
        )
        for row in rows
    }
    source_archives = _copy_source_archives(repo_root)
    _write_json(
        PACKAGE_DIR / SOURCE_ARCHIVES_NAME, source_archives
    )

    resume_inputs = _materialize_resume_inputs(
        repo_root=repo_root,
        rows=rows,
        protocol_bindings=protocol_bindings,
    )
    _write_json(
        PACKAGE_DIR / RESUME_INPUTS_NAME, resume_inputs
    )
    audit = _source_lock_audit(
        repo_root=repo_root,
        rows=rows,
        protocol_bindings=protocol_bindings,
        resume_inputs=resume_inputs,
    )
    _write_json(PACKAGE_DIR / SOURCE_LOCK_AUDIT_NAME, audit)
    collision = _collision_status(repo_root)
    _write_json(PACKAGE_DIR / COLLISION_STATUS_NAME, collision)

    plan = digested(
        {
            "schema": EXECUTION_PLAN_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "cell_count": CELL_COUNT,
            "authenticated_resume_count": RESUME_COUNT,
            "fresh_count": FRESH_COUNT,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "only_scientific_change": (
                "maximum_controller_rounds_50_to_70"
            ),
            "source_report": file_binding(
                provenance_path, repo_root=repo_root
            ),
            "source_archives_sha256": source_archives["sha256"],
            "resume_inputs_sha256": resume_inputs["sha256"],
            "source_lock_audit_sha256": audit["sha256"],
            "collision_status_sha256": collision["sha256"],
            "execution_ids": [
                row["execution_id"] for row in rows
            ],
            "submission_blockers": [
                "live_r50_predecessors_9397758_0_through_8",
                "fresh_row_source_value_anchor_unavailable",
                "r70_resource_envelopes_not_demonstrated",
                "execution_authorization_absent",
                "submission_descriptor_intentionally_absent",
            ],
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(PACKAGE_DIR / EXECUTION_PLAN_NAME, plan)

    jobs_dir = PACKAGE_DIR / "jobs"
    jobs_dir.mkdir()
    job_bindings: list[dict[str, Any]] = []
    for row in rows:
        execution_id = str(row["execution_id"])
        family = source_archives["families"][
            row["source_family"]
        ]
        resume = resume_inputs["cells"].get(execution_id)
        job = digested(
            {
                "schema": JOB_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                **row,
                "source_protocol": dict(
                    protocol_bindings[execution_id]
                ),
                "source_archive": dict(
                    family["packaged_archive"]
                ),
                "source_archive_manifest": dict(
                    family["packaged_manifest"]
                ),
                "resume_input": (
                    None
                    if resume is None
                    else {
                        "archive": dict(resume["archive"]),
                        "checkpoint_path": resume[
                            "checkpoint_path"
                        ],
                        "checkpoint_sha256": resume[
                            "checkpoint_sha256"
                        ],
                        "member_count": resume["member_count"],
                        "members": list(resume["members"]),
                        "pointer_closed": True,
                    }
                ),
                "source_lock_delta": {
                    "variable": "maximum_controller_rounds",
                    "from": SOURCE_HORIZON,
                    "to": TARGET_HORIZON,
                    "changed_fields_vs_source": [
                        "maximum_controller_rounds"
                    ],
                    "non_swept_settings_diff": [],
                },
                "expected_output_root": f"runs/{execution_id}",
                "execution_plan_sha256": plan["sha256"],
                "source_lock_audit_sha256": audit["sha256"],
                "collision_status_sha256": collision["sha256"],
                "authorization_schema": AUTHORIZATION_SCHEMA,
                "global_submission_blocked": True,
                "submission_ready": False,
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            }
        )
        job_path = jobs_dir / f"{execution_id}.json"
        _write_json(job_path, job)
        job_bindings.append(
            {
                "execution_id": execution_id,
                "path": job_path.relative_to(
                    PACKAGE_DIR
                ).as_posix(),
                "sha256": sha256_file(job_path),
                "canonical_sha256": job["sha256"],
                "size_bytes": job_path.stat().st_size,
            }
        )

    queue_lines = [
        "\t".join(
            (
                str(row["execution_id"]),
                str(row["execution_mode"]),
                str(row["collision_status"]),
                str(row["source_family"]),
                str(row["resources"]["request_cpus"]),
                str(row["resources"]["request_memory_mb"]),
                str(row["resources"]["request_disk_mb"]),
                str(MAX_RUNTIME_SECONDS),
            )
        )
        for row in rows
    ]
    _exclusive_write(
        PACKAGE_DIR / QUEUE_NAME,
        ("\n".join(queue_lines) + "\n").encode("utf-8"),
    )

    control_bindings = [
        {
            "path": relative,
            "sha256": sha256_file(PACKAGE_DIR / relative),
            "size_bytes": (PACKAGE_DIR / relative).stat().st_size,
        }
        for relative in CONTROL_FILES
    ]
    manifest = digested(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "status": "passed_inert_collision_blocked",
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "cell_count": CELL_COUNT,
            "authenticated_resume_count": RESUME_COUNT,
            "fresh_count": FRESH_COUNT,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
            "resource_weighting_scope": (
                RESOURCE_WEIGHTING_SCOPE
            ),
            "source_archives": {
                "path": SOURCE_ARCHIVES_NAME,
                "sha256": source_archives["sha256"],
                "file_sha256": sha256_file(
                    PACKAGE_DIR / SOURCE_ARCHIVES_NAME
                ),
            },
            "resume_inputs": {
                "path": RESUME_INPUTS_NAME,
                "sha256": resume_inputs["sha256"],
                "file_sha256": sha256_file(
                    PACKAGE_DIR / RESUME_INPUTS_NAME
                ),
            },
            "source_lock_audit": {
                "path": SOURCE_LOCK_AUDIT_NAME,
                "sha256": audit["sha256"],
                "file_sha256": sha256_file(
                    PACKAGE_DIR / SOURCE_LOCK_AUDIT_NAME
                ),
            },
            "collision_status": {
                "path": COLLISION_STATUS_NAME,
                "sha256": collision["sha256"],
                "file_sha256": sha256_file(
                    PACKAGE_DIR / COLLISION_STATUS_NAME
                ),
            },
            "execution_plan": {
                "path": EXECUTION_PLAN_NAME,
                "sha256": plan["sha256"],
                "file_sha256": sha256_file(
                    PACKAGE_DIR / EXECUTION_PLAN_NAME
                ),
            },
            "queue": {
                "path": QUEUE_NAME,
                "sha256": sha256_file(
                    PACKAGE_DIR / QUEUE_NAME
                ),
                "row_count": CELL_COUNT,
                "kind": "inert_planning_queue_not_condor_queue",
            },
            "control_files": control_bindings,
            "jobs": job_bindings,
            "submit_descriptor_present": False,
            "authority_overlay_present": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(PACKAGE_DIR / PACKAGE_MANIFEST_NAME, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "package_id": PACKAGE_ID,
                "cell_count": CELL_COUNT,
                "authenticated_resume_count": RESUME_COUNT,
                "fresh_count": FRESH_COUNT,
                "package_manifest_sha256": manifest["sha256"],
                "submission_ready": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
