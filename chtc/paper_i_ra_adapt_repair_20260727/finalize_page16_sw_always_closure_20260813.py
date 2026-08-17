#!/usr/bin/env python3
"""Authenticate and locally seal the fixed Page-16 SW-always closure.

This helper is intentionally inert with respect to CHTC.  It never opens a
network connection and never executes scheduler commands.  ``--preflight``
only authenticates the already-fetched archive and prints the exact remote-
materialization exclusion plan for an independently authenticated session.
``--finalize`` is reserved for locally sealing independently captured evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Any, BinaryIO, Mapping
import uuid


HELPER_PATH = Path(__file__).resolve()
SOURCE_REPO_ROOT = HELPER_PATH.parents[2]
REPAIR_RELATIVE = Path("chtc/paper_i_ra_adapt_repair_20260727")
PACKAGE_RELATIVE = REPAIR_RELATIVE / (
    "paper_i_ra_adapt_page16_insertion_comparators_weak50_strong30_"
    "20260812_v1_chtc"
)
EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_weak_u8__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced"
)
JOB_PATH = SOURCE_REPO_ROOT / PACKAGE_RELATIVE / "jobs" / f"{EXECUTION_ID}.json"
ARCHIVE_RELATIVE = REPAIR_RELATIVE / (
    "retrieved_page16_insertion_comparators_20260812/"
    "strong_weak_u8_always__9647386__1.tar.gz"
)
RECEIPT_RELATIVE = REPAIR_RELATIVE / (
    "paper_i_ra_adapt_page16_cluster9647386_sw_always_"
    "remote_materialization_exclusion_receipt_20260813.json"
)
REMOTE_ARCHIVE_PATH = (
    "osdf:///chtc/staging/j/jsstrobel/"
    "paper_i_ra_adapt_page16_insertion_comparators_20260812_v1/outputs/"
    "transfer/"
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_weak_u8__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced__9647386__1.tar.gz"
)
REMOTE_IDENTITY_NAME = "page16_sw_always_remote_archive_identity.json"
BEFORE_RETIREMENT_NAME = "page16_cluster9647386_before_retirement.json"
REMOVAL_ATTEMPTS_NAME = (
    "page16_cluster9647386_acknowledged_removal_attempts.json"
)
AFTER_EXCLUSION_NAME = (
    "page16_cluster9647386_after_materialization_exclusion.json"
)
HISTORY_NAME = "page16_cluster9647386_history.json"
CONTINUATION_ADAPTER_RELATIVE = REPAIR_RELATIVE / (
    "continue_local_page16_insertion_comparators_k30_to_k50_20260813.py"
)
CONTINUATION_ADAPTER_PATH = SOURCE_REPO_ROOT / CONTINUATION_ADAPTER_RELATIVE
EXPECTED_CONTINUATION_ADAPTER_SHA256 = (
    "56c50f046759d4299d768cb609f08fce8c79e3190aaadb6609afdde4f5452e07"
)
CLUSTER_ID = 9_647_386
PROC_ID = 1
TARGET_HORIZON = 50
LATENT_PROC_IDS = list(range(2, 11))
EXPECTED_JOB_CANONICAL_SHA256 = (
    "598d2b615af58ad1551178920c5363a98ad5094d156e401510e1d1728ae8e0e1"
)
WORKER_SCHEMA = "paper_i_ra_adapt_page16_macro_phase23_qiskit_worker_receipt_v1"
MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_execution_manifest_v1"
)
CLOSURE_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_page16_sw_always_"
    "remote_materialization_exclusion_receipt_v2"
)
CLOSURE_RECEIPT_STATUS = (
    "passed_sw_always_k50_closed_remote_materialization_excluded"
)
REMOVAL_COMMAND = "condor_rm 9647386"
REMOVAL_ACKNOWLEDGEMENT = (
    "All jobs in cluster 9647386 have been marked for removal"
)
AUTHENTICATION = {
    "authenticated_remote_query": True,
    "kind": "interactive_ssh_duo_condor_q_snapshot_v1",
    "source_host": "ap2001.chtc.wisc.edu",
}
QUEUE_COMMAND = (
    "condor_q 9647386 -json -attributes "
    "ClusterId,ProcId,JobStatus,NumJobStarts"
)
FACTORY_COMMAND = (
    "condor_q -factory 9647386 -json -attributes "
    "ClusterId,TotalSubmitProcs,JobMaterializeLimit,JobMaterializeMaxIdle,"
    "JobMaterializeNextProcId,JobMaterializePaused"
)
HISTORY_COMMAND = (
    "condor_history 9647386 -limit 20 -json -attributes "
    "ClusterId,ProcId,JobStatus,ExitCode,NumJobStarts,CompletionDate"
)


class ClosureError(RuntimeError):
    """Raised when local closure evidence fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ClosureError(f"{label} is absent or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClosureError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ClosureError(f"{label} must be a JSON object: {path}")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    actual = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if claimed != actual:
        raise ClosureError(f"{label} self digest drifted")


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    return {
        **unsigned,
        "sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def _safe_member_name(name: str) -> str:
    raw = name
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if (
        not raw
        or raw == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in raw
    ):
        raise ClosureError(f"unsafe archive member: {name}")
    return path.as_posix()


def _hash_stream(stream: BinaryIO) -> tuple[str, int, bytes | None]:
    digest = hashlib.sha256()
    size = 0
    captured: bytes | None = b""
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
        if captured is not None:
            if size <= 4 * 1024 * 1024:
                captured += block
            else:
                captured = None
    return digest.hexdigest(), size, captured


def _captured_json(
    observed: Mapping[str, Mapping[str, Any]],
    relative: str,
    *,
    label: str,
) -> dict[str, Any]:
    row = observed.get(relative)
    if not isinstance(row, Mapping) or row.get("captured") is None:
        raise ClosureError(f"{label} is absent or unexpectedly large")
    try:
        value = json.loads(row["captured"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClosureError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ClosureError(f"{label} must be a JSON object")
    return value


def _load_fixed_job() -> dict[str, Any]:
    job = _load_object(JOB_PATH, label="fixed SW-always job")
    _verify_self_digest(job, label="fixed SW-always job")
    if (
        job.get("sha256") != EXPECTED_JOB_CANONICAL_SHA256
        or job.get("execution_id") != EXECUTION_ID
        or job.get("regime_id") != "strong_weak_u8"
        or job.get("comparator_policy") != "always_commutation_reduced"
        or job.get("typed_insertion_kind") != "always_commutation_reduced"
        or job.get("runtime_insertion_mode") != "full_commutation_reduced"
        or job.get("target_horizon") != TARGET_HORIZON
    ):
        raise ClosureError("fixed SW-always job identity drifted")
    return job


def _authenticate_archive(workspace_root: Path) -> dict[str, Any]:
    archive_path = workspace_root / ARCHIVE_RELATIVE
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ClosureError(f"fixed fetched archive is absent or unsafe: {archive_path}")
    archive_size = archive_path.stat().st_size
    archive_sha256 = _sha256_file(archive_path)
    job = _load_fixed_job()

    observed: dict[str, dict[str, Any]] = {}
    directories: set[str] = set()
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                root_name = member.name
                while root_name.startswith("./"):
                    root_name = root_name[2:]
                if member.isdir() and root_name in {"", "."}:
                    continue
                relative = _safe_member_name(member.name)
                if relative in observed or relative in directories:
                    raise ClosureError(f"duplicate archive member: {relative}")
                if member.issym() or member.islnk():
                    raise ClosureError(f"linked archive member is forbidden: {relative}")
                if member.isdir():
                    directories.add(relative)
                    continue
                if not member.isfile():
                    raise ClosureError(f"unsafe archive member type: {relative}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ClosureError(f"unreadable archive member: {relative}")
                digest, size, captured = _hash_stream(stream)
                observed[relative] = {
                    "sha256": digest,
                    "size_bytes": size,
                    "captured": captured,
                }
    except (tarfile.TarError, OSError) as exc:
        raise ClosureError(f"fixed fetched archive is not a readable gzip tar: {archive_path}") from exc

    roots = {"worker_exit_status.txt", "worker_receipt.json"}
    if not roots.issubset(observed):
        raise ClosureError("archive lacks fixed worker root receipts")
    exit_payload = observed["worker_exit_status.txt"]["captured"]
    if exit_payload is None or exit_payload.strip() != b"0":
        raise ClosureError("worker exit status is nonzero or unreadable")

    worker = _captured_json(observed, "worker_receipt.json", label="worker receipt")
    _verify_self_digest(worker, label="worker receipt")
    if (
        worker.get("schema") != WORKER_SCHEMA
        or worker.get("status") != "passed"
        or worker.get("package_id") != job.get("package_id")
        or worker.get("campaign_id") != job.get("campaign_id")
        or worker.get("execution_id") != EXECUTION_ID
        or worker.get("job_spec_sha256") != job.get("sha256")
        or worker.get("controller_rounds_completed") != TARGET_HORIZON
        or worker.get("fresh_start") is not True
    ):
        raise ClosureError("worker receipt identity drifted")

    raw_artifacts = worker.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ClosureError("worker artifact inventory is absent")
    declared: dict[str, Mapping[str, Any]] = {}
    for row in raw_artifacts:
        if not isinstance(row, Mapping):
            raise ClosureError("worker artifact row is invalid")
        relative = _safe_member_name(str(row.get("path", "")))
        if relative in declared or relative in roots:
            raise ClosureError(f"duplicate or reserved worker artifact: {relative}")
        if type(row.get("size_bytes")) is not int or not isinstance(row.get("sha256"), str):
            raise ClosureError(f"worker artifact binding is incomplete: {relative}")
        declared[relative] = row
    if set(observed) != roots | set(declared):
        raise ClosureError("archive contains missing or unbound files")
    for relative, row in declared.items():
        actual = observed[relative]
        if (
            row.get("sha256") != actual["sha256"]
            or row.get("size_bytes") != actual["size_bytes"]
        ):
            raise ClosureError(f"worker artifact binding drifted: {relative}")

    expected_artifacts = job.get("expected_run_artifacts")
    if not isinstance(expected_artifacts, Mapping):
        raise ClosureError("fixed job expected artifact inventory is absent")
    expected_paths: dict[str, str] = {}
    for role in ("checkpoint", "estimator_ledger", "execution_manifest", "result", "summary"):
        row = expected_artifacts.get(role)
        if (
            not isinstance(row, Mapping)
            or row.get("required") is not True
            or row.get("direct_file_required") is not True
            or not isinstance(row.get("path"), str)
        ):
            raise ClosureError(f"fixed expected artifact contract drifted: {role}")
        expected_paths[role] = _safe_member_name(row["path"])

    checkpoint_path = PurePosixPath(expected_paths["checkpoint"])
    sidecar_paths: dict[str, str] = {}
    for role in (
        "estimator_call_ledger_checkpoint",
        "verified_singleton_resume",
    ):
        prefix = f"{checkpoint_path.stem}.{role}."
        matches = [
            relative
            for relative in declared
            if PurePosixPath(relative).parent == checkpoint_path.parent
            and PurePosixPath(relative).name.startswith(prefix)
            and PurePosixPath(relative).name.endswith(".json")
        ]
        if len(matches) != 1:
            raise ClosureError(
                f"worker checkpoint sidecar inventory drifted: {role}"
            )
        relative = matches[0]
        actual_sha256 = observed[relative]["sha256"]
        expected_relative = checkpoint_path.with_name(
            f"{checkpoint_path.stem}.{role}.{actual_sha256[:16]}.json"
        ).as_posix()
        if relative != expected_relative:
            raise ClosureError(
                f"worker checkpoint sidecar content address drifted: {role}"
            )
        sidecar_paths[role] = relative

    allowed_paths = set(expected_paths.values()) | set(sidecar_paths.values())
    if set(declared) != allowed_paths:
        raise ClosureError("worker artifact inventory differs from the fixed job")

    manifest_path = expected_paths["execution_manifest"]
    manifest = _captured_json(observed, manifest_path, label="execution manifest")
    _verify_self_digest(manifest, label="execution manifest")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("package_id") != job.get("package_id")
        or manifest.get("campaign_id") != job.get("campaign_id")
        or manifest.get("execution_id") != EXECUTION_ID
        or manifest.get("job_spec_sha256") != job.get("sha256")
        or manifest.get("authorization_sha256") != worker.get("authorization_sha256")
        or manifest.get("protocol_sha256") != job.get("protocol_sha256")
        or manifest.get("route_contract_sha256") != job.get("route_contract_sha256")
        or manifest.get("comparator_policy") != job.get("comparator_policy")
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("controller_rounds_completed") != TARGET_HORIZON
        or manifest.get("fresh_start") is not True
        or manifest.get("source_checkpoint_consumed") is not False
        or worker.get("execution_manifest_sha256") != manifest.get("sha256")
    ):
        raise ClosureError("sealed execution-manifest identity drifted")

    output_payloads = manifest.get("output_payloads")
    if not isinstance(output_payloads, Mapping):
        raise ClosureError("execution manifest output payload inventory is absent")
    for role in ("checkpoint", "estimator_ledger", "result", "summary"):
        row = output_payloads.get(role)
        relative = expected_paths[role]
        actual = observed[relative]
        if (
            not isinstance(row, Mapping)
            or row.get("path") != relative
            or row.get("sha256") != actual["sha256"]
            or row.get("size_bytes") != actual["size_bytes"]
        ):
            raise ClosureError(f"execution-manifest output binding drifted: {role}")
    if set(output_payloads) != {"checkpoint", "estimator_ledger", "result", "summary"}:
        raise ClosureError("execution manifest contains unexpected output payload roles")

    return {
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size,
        "artifact_count": len(declared),
        "worker": worker,
        "manifest": manifest,
        "job": job,
        "summary": {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size,
            "artifact_count": len(declared),
            "all_declared_payload_hashes_verified": True,
            "execution_manifest_canonical_sha256": manifest["sha256"],
            "unbound_file_count": 0,
            "worker_exit_status": 0,
            "worker_receipt_canonical_sha256": worker["sha256"],
        },
    }


def _authenticate_remote_identity(
    evidence_dir: Path,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
) -> dict[str, Any]:
    identity = _load_object(
        evidence_dir / REMOTE_IDENTITY_NAME,
        label="remote/local archive identity evidence",
    )
    _verify_self_digest(identity, label="remote/local archive identity evidence")
    if (
        identity.get("schema") != "paper_i_page16_sw_always_remote_archive_identity_v1"
        or identity.get("status")
        != "passed_remote_local_size_sha256_match_after_atomic_rename"
        or identity.get("remote_path") != REMOTE_ARCHIVE_PATH
        or identity.get("local_path") != ARCHIVE_RELATIVE.as_posix()
        or type(identity.get("remote_size_bytes")) is not int
        or identity.get("remote_size_bytes") != archive_size_bytes
        or type(identity.get("local_size_bytes")) is not int
        or identity.get("local_size_bytes") != archive_size_bytes
        or identity.get("remote_sha256") != archive_sha256
        or identity.get("local_sha256") != archive_sha256
        or identity.get("gzip_integrity_passed") is not True
        or identity.get("tar_readability_passed") is not True
        or identity.get("atomic_local_rename_completed") is not True
        or identity.get("remote_state")
        != "preserved_after_exact_size_sha256_verified_fetch"
        or not isinstance(identity.get("captured_at_utc"), str)
    ):
        raise ClosureError("remote/local archive identity evidence drifted")
    return identity


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ClosureError(f"{label} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ClosureError(f"{label} is not an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ClosureError(f"{label} is not UTC")
    return parsed


def _authenticated_snapshot(
    evidence_dir: Path,
    filename: str,
    *,
    label: str,
    schema: str,
    status: str,
) -> tuple[dict[str, Any], datetime]:
    value = _load_object(evidence_dir / filename, label=label)
    _verify_self_digest(value, label=label)
    if (
        value.get("schema") != schema
        or value.get("status") != status
        or value.get("authentication") != AUTHENTICATION
        or type(value.get("cluster_id")) is not int
        or value.get("cluster_id") != CLUSTER_ID
    ):
        raise ClosureError(f"{label} identity drifted")
    captured = _parse_utc(value.get("captured_at_utc"), label=f"{label} capture")
    return value, captured


def _query_rows(
    snapshot: Mapping[str, Any],
    key: str,
    *,
    command: str,
    label: str,
) -> list[Mapping[str, Any]]:
    query = snapshot.get(key)
    if not isinstance(query, Mapping) or query.get("command") != command:
        raise ClosureError(f"{label} query command drifted")
    rows = query.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ClosureError(f"{label} query rows are invalid")
    return rows


def _history_by_proc(
    rows: list[Mapping[str, Any]],
    *,
    label: str,
) -> dict[int, Mapping[str, Any]]:
    by_proc: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        values = (
            row.get("ClusterId"),
            row.get("ProcId"),
            row.get("JobStatus"),
            row.get("ExitCode"),
            row.get("NumJobStarts"),
            row.get("CompletionDate"),
        )
        if any(type(value) is not int for value in values):
            raise ClosureError(f"{label} contains a non-integer history field")
        proc_id = row["ProcId"]
        if (
            row["ClusterId"] != CLUSTER_ID
            or proc_id not in (0, PROC_ID)
            or row["JobStatus"] != 4
            or row["ExitCode"] != 0
            or row["NumJobStarts"] < 1
            or row["CompletionDate"] <= 0
            or proc_id in by_proc
        ):
            raise ClosureError(f"{label} completed-proc identity drifted")
        by_proc[proc_id] = row
    if sorted(by_proc) != [0, PROC_ID]:
        raise ClosureError(f"{label} must contain exactly completed procs 0 and 1")
    return by_proc


def _authenticate_materialization_exclusion_evidence(
    evidence_dir: Path,
    *,
    remote_identity: Mapping[str, Any],
) -> dict[str, Any]:
    before, before_time = _authenticated_snapshot(
        evidence_dir,
        BEFORE_RETIREMENT_NAME,
        label="before-retirement snapshot",
        schema="paper_i_page16_sw_always_factory_before_retirement_snapshot_v1",
        status="passed_authenticated_paused_factory_before_retirement",
    )
    attempts, attempts_time = _authenticated_snapshot(
        evidence_dir,
        REMOVAL_ATTEMPTS_NAME,
        label="acknowledged removal-attempt evidence",
        schema="paper_i_page16_sw_always_acknowledged_removal_attempts_v1",
        status="passed_authenticated_at_least_one_acknowledged_condor_rm",
    )
    after, after_time = _authenticated_snapshot(
        evidence_dir,
        AFTER_EXCLUSION_NAME,
        label="after-exclusion snapshot",
        schema=(
            "paper_i_page16_sw_always_remote_materialization_exclusion_"
            "after_snapshot_v2"
        ),
        status=(
            "passed_authenticated_remote_materialization_excluded_"
            "after_removal_attempt"
        ),
    )
    history, history_time = _authenticated_snapshot(
        evidence_dir,
        HISTORY_NAME,
        label="terminal history snapshot",
        schema="paper_i_page16_sw_always_cluster_history_snapshot_v2",
        status="passed_authenticated_only_procs_0_1_completed",
    )
    remote_time = _parse_utc(
        remote_identity.get("captured_at_utc"),
        label="remote/local archive identity capture",
    )
    if not remote_time <= before_time < attempts_time <= after_time <= history_time:
        raise ClosureError("closure evidence capture order drifted")

    attempt_rows = attempts.get("attempts")
    if not isinstance(attempt_rows, list) or not attempt_rows:
        raise ClosureError("acknowledged removal attempt inventory is absent")
    for row in attempt_rows:
        if not isinstance(row, Mapping):
            raise ClosureError("acknowledged removal attempt drifted")
        started = _parse_utc(
            row.get("started_at_utc"), label="removal attempt start"
        )
        acknowledged = _parse_utc(
            row.get("acknowledged_at_utc"),
            label="removal attempt acknowledgement",
        )
        if (
            row.get("command") != REMOVAL_COMMAND
            or type(row.get("exit_code")) is not int
            or row.get("exit_code") != 0
            or row.get("acknowledged") is not True
            or row.get("acknowledgement_text") != REMOVAL_ACKNOWLEDGEMENT
            or not before_time < started <= acknowledged <= attempts_time
        ):
            raise ClosureError("acknowledged removal attempt drifted")

    before_queue = _query_rows(
        before,
        "queue_query",
        command=QUEUE_COMMAND,
        label="before-retirement queue",
    )
    if before_queue:
        raise ClosureError("before-retirement queue must be empty")
    before_factory = _query_rows(
        before,
        "factory_query",
        command=FACTORY_COMMAND,
        label="before-retirement factory",
    )
    if len(before_factory) != 1:
        raise ClosureError("before-retirement factory snapshot must contain one factory")
    factory = before_factory[0]
    expected_factory = {
        "ClusterId": CLUSTER_ID,
        "TotalSubmitProcs": 11,
        "JobMaterializeLimit": 1,
        "JobMaterializeMaxIdle": 0,
        "JobMaterializeNextProcId": 2,
        "JobMaterializePaused": 1,
    }
    if any(
        type(factory.get(key)) is not int or factory.get(key) != expected
        for key, expected in expected_factory.items()
    ):
        raise ClosureError("before-retirement paused factory identity drifted")
    before_history = _history_by_proc(
        _query_rows(
            before,
            "history_query",
            command=HISTORY_COMMAND,
            label="before-retirement history",
        ),
        label="before-retirement history",
    )

    after_queue = _query_rows(
        after,
        "queue_query",
        command=QUEUE_COMMAND,
        label="after-exclusion queue",
    )
    after_factory = _query_rows(
        after,
        "factory_query",
        command=FACTORY_COMMAND,
        label="after-exclusion factory",
    )
    if after_queue:
        raise ClosureError("after-exclusion queue must be empty")
    if not after_factory:
        outcome = "factory_absent_after_acknowledged_removal"
        after_factory_projection = {
            "factory_present": False,
            "factory_materialization_paused": None,
            "job_materialize_limit": None,
            "job_materialize_max_idle": None,
            "job_materialize_next_proc_id": None,
        }
    elif len(after_factory) == 1:
        retained = after_factory[0]
        expected_retained = {
            "ClusterId": CLUSTER_ID,
            "TotalSubmitProcs": 11,
            "JobMaterializeLimit": 2,
            "JobMaterializeMaxIdle": 0,
            "JobMaterializeNextProcId": 2,
            "JobMaterializePaused": 1,
        }
        if any(
            type(retained.get(key)) is not int
            or retained.get(key) != expected
            for key, expected in expected_retained.items()
        ):
            raise ClosureError(
                "retained factory materialization exclusion drifted"
            )
        outcome = (
            "factory_retained_paused_at_completed_prefix_"
            "after_acknowledged_removal"
        )
        after_factory_projection = {
            "factory_present": True,
            "factory_materialization_paused": True,
            "job_materialize_limit": 2,
            "job_materialize_max_idle": 0,
            "job_materialize_next_proc_id": 2,
        }
    else:
        raise ClosureError("after-exclusion factory inventory drifted")
    after_history = _history_by_proc(
        _query_rows(
            after,
            "history_query",
            command=HISTORY_COMMAND,
            label="after-exclusion history",
        ),
        label="after-exclusion history",
    )

    queried_proc_ids = history.get("queried_proc_ids")
    if (
        not isinstance(queried_proc_ids, list)
        or any(type(proc_id) is not int for proc_id in queried_proc_ids)
        or queried_proc_ids != list(range(11))
    ):
        raise ClosureError("terminal history did not explicitly query procs 0 through 10")
    terminal_history = _history_by_proc(
        _query_rows(
            history,
            "history_query",
            command=HISTORY_COMMAND,
            label="terminal history",
        ),
        label="terminal history",
    )
    selected_fields = (
        "ClusterId",
        "ProcId",
        "JobStatus",
        "ExitCode",
        "NumJobStarts",
        "CompletionDate",
    )
    for proc_id in (0, PROC_ID):
        if any(
            before_history[proc_id].get(key) != after_history[proc_id].get(key)
            or after_history[proc_id].get(key)
            != terminal_history[proc_id].get(key)
            for key in selected_fields
        ):
            raise ClosureError("completed history changed across exclusion")
    if any(
        datetime.fromtimestamp(row["CompletionDate"], tz=timezone.utc) > before_time
        for row in terminal_history.values()
    ):
        raise ClosureError("before-retirement snapshot predates a claimed completion")
    return {
        "before": before,
        "attempts": attempts,
        "after": after,
        "history": history,
        "before_history": before_history,
        "after_history": after_history,
        "terminal_history": terminal_history,
        "outcome": outcome,
        "after_factory_projection": after_factory_projection,
    }


def _remote_materialization_exclusion_plan() -> dict[str, Any]:
    return {
        "target_cluster_id": CLUSTER_ID,
        "removal_command": REMOVAL_COMMAND,
        "latent_proc_ids_that_must_never_materialize": LATENT_PROC_IDS,
        "helper_executes_commands": False,
        "requires_interactive_ssh_duo": True,
        "authenticated_source_host": "ap2001.chtc.wisc.edu",
        "required_sequence": [
            {
                "phase": "before_removal_attempt",
                "commands": [
                    QUEUE_COMMAND,
                    FACTORY_COMMAND,
                    HISTORY_COMMAND,
                ],
            },
            {"phase": "attempt_removal", "commands": [REMOVAL_COMMAND]},
            {
                "phase": "after_materialization_exclusion",
                "commands": [
                    QUEUE_COMMAND,
                    FACTORY_COMMAND,
                    HISTORY_COMMAND,
                ],
            },
        ],
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "authentication": AUTHENTICATION,
        "required_files": [
            {
                "capture_order": 0,
                "filename": REMOTE_IDENTITY_NAME,
                "schema": "paper_i_page16_sw_always_remote_archive_identity_v1",
                "status": (
                    "passed_remote_local_size_sha256_match_after_atomic_rename"
                ),
            },
            {
                "capture_order": 1,
                "filename": BEFORE_RETIREMENT_NAME,
                "schema": (
                    "paper_i_page16_sw_always_factory_before_retirement_"
                    "snapshot_v1"
                ),
                "status": "passed_authenticated_paused_factory_before_retirement",
            },
            {
                "capture_order": 2,
                "filename": REMOVAL_ATTEMPTS_NAME,
                "schema": (
                    "paper_i_page16_sw_always_acknowledged_removal_attempts_v1"
                ),
                "status": (
                    "passed_authenticated_at_least_one_acknowledged_condor_rm"
                ),
            },
            {
                "capture_order": 3,
                "filename": AFTER_EXCLUSION_NAME,
                "schema": (
                    "paper_i_page16_sw_always_remote_materialization_"
                    "exclusion_after_snapshot_v2"
                ),
                "status": (
                    "passed_authenticated_remote_materialization_excluded_"
                    "after_removal_attempt"
                ),
            },
            {
                "capture_order": 4,
                "filename": HISTORY_NAME,
                "schema": "paper_i_page16_sw_always_cluster_history_snapshot_v2",
                "status": "passed_authenticated_only_procs_0_1_completed",
            },
        ],
        "publication_rule": (
            "archive_and_all_evidence_first_atomic_strict_receipt_last"
        ),
    }


def _preflight(workspace_root: Path, evidence_dir: Path) -> dict[str, Any]:
    archive = _authenticate_archive(workspace_root)
    _authenticate_remote_identity(
        evidence_dir,
        archive_sha256=archive["archive_sha256"],
        archive_size_bytes=archive["archive_size_bytes"],
    )
    return {
        "schema": "paper_i_page16_sw_always_local_closure_preflight_v2",
        "status": (
            "passed_archive_ready_for_user_mediated_"
            "remote_materialization_exclusion"
        ),
        "archive_closure": archive["summary"],
        "remote_materialization_exclusion_plan": (
            _remote_materialization_exclusion_plan()
        ),
        "evidence_contract": _evidence_contract(),
        "writes_performed": False,
        "scheduler_mutation_performed": False,
    }


def _strict_consumer_validate(
    receipt_path: Path,
    *,
    workspace_root: Path,
    worker: Mapping[str, Any],
    job: Mapping[str, Any],
) -> None:
    if (
        not CONTINUATION_ADAPTER_PATH.is_file()
        or CONTINUATION_ADAPTER_PATH.is_symlink()
        or _sha256_file(CONTINUATION_ADAPTER_PATH)
        != EXPECTED_CONTINUATION_ADAPTER_SHA256
    ):
        raise ClosureError("pinned strict continuation consumer bytes drifted")
    module_name = "paper_i_page16_sw_always_pinned_strict_closure_consumer"
    spec = importlib.util.spec_from_file_location(module_name, CONTINUATION_ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise ClosureError("pinned strict continuation consumer cannot be loaded")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = adapter
    try:
        spec.loader.exec_module(adapter)
        original_root = adapter.REPO_ROOT
        original_receipt = adapter.SW_ALWAYS_CLOSURE_RECEIPT_PATH
        adapter.REPO_ROOT = workspace_root
        adapter.SW_ALWAYS_CLOSURE_RECEIPT_PATH = receipt_path
        try:
            adapter._authenticate_sw_always_closure(worker, job=job)
        finally:
            adapter.REPO_ROOT = original_root
            adapter.SW_ALWAYS_CLOSURE_RECEIPT_PATH = original_receipt
    except Exception as exc:
        if isinstance(exc, ClosureError):
            raise
        raise ClosureError(f"pinned strict continuation consumer rejected receipt: {exc}") from exc


def _atomic_publish_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    workspace_root: Path,
    worker: Mapping[str, Any],
    job: Mapping[str, Any],
) -> None:
    if path.exists() or path.is_symlink():
        raise ClosureError(f"fixed closure receipt already exists; refusing overwrite: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ClosureError(f"fixed closure receipt parent is absent or unsafe: {path.parent}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        _strict_consumer_validate(
            temporary,
            workspace_root=workspace_root,
            worker=worker,
            job=job,
        )
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _finalize(workspace_root: Path, evidence_dir: Path) -> dict[str, Any]:
    archive = _authenticate_archive(workspace_root)
    remote_identity = _authenticate_remote_identity(
        evidence_dir,
        archive_sha256=archive["archive_sha256"],
        archive_size_bytes=archive["archive_size_bytes"],
    )
    evidence = _authenticate_materialization_exclusion_evidence(
        evidence_dir,
        remote_identity=remote_identity,
    )
    worker = archive["worker"]
    manifest = archive["manifest"]
    job = archive["job"]
    proc_history = evidence["terminal_history"][PROC_ID]
    receipt = _digested(
        {
            "schema": CLOSURE_RECEIPT_SCHEMA,
            "status": CLOSURE_RECEIPT_STATUS,
            "created_at_utc": evidence["history"]["captured_at_utc"],
            "completed_remote_cell": {
                "regime_id": "strong_weak_u8",
                "comparator_policy": "always_commutation_reduced",
                "typed_insertion_kind": job["typed_insertion_kind"],
                "runtime_insertion_mode": job["runtime_insertion_mode"],
                "execution_id": EXECUTION_ID,
                "cluster_id": CLUSTER_ID,
                "proc_id": PROC_ID,
                "controller_rounds_completed": TARGET_HORIZON,
                "archive": {
                    "path": ARCHIVE_RELATIVE.as_posix(),
                    "remote_path": REMOTE_ARCHIVE_PATH,
                    "remote_size_bytes": archive["archive_size_bytes"],
                    "local_size_bytes": archive["archive_size_bytes"],
                    "size_bytes": archive["archive_size_bytes"],
                    "remote_sha256": archive["archive_sha256"],
                    "local_sha256": archive["archive_sha256"],
                    "sha256": archive["archive_sha256"],
                },
                "worker_receipt": {
                    "path_inside_archive": "worker_receipt.json",
                    "canonical_sha256": worker["sha256"],
                    "schema": worker["schema"],
                    "status": "passed",
                },
                "execution_manifest": {
                    "path_inside_archive": (
                        f"runs/{EXECUTION_ID}/execution_manifest.json"
                    ),
                    "canonical_sha256": manifest["sha256"],
                },
                "history": {
                    "cluster_id": CLUSTER_ID,
                    "proc_id": PROC_ID,
                    "job_status": proc_history["JobStatus"],
                    "exit_code": proc_history["ExitCode"],
                    "num_job_starts": proc_history["NumJobStarts"],
                    "completion_date_epoch": proc_history["CompletionDate"],
                },
                "authenticated_full_sealed_closure": True,
            },
            "remote_materialization_exclusion": {
                "outcome": evidence["outcome"],
                "removal_command": REMOVAL_COMMAND,
                "removal_attempts_authenticated": True,
                "before_snapshot": {
                    "job_materialize_paused": 1,
                    "job_materialize_next_proc_id": 2,
                    "materialized_proc_ids": [],
                    "history_completed_proc_ids": [0, 1],
                },
                "after_snapshot": {
                    "cluster_present_in_queue": False,
                    **evidence["after_factory_projection"],
                    "history_completed_proc_ids": [0, 1],
                },
                "latent_proc_ids_never_materialized": LATENT_PROC_IDS,
                "queue_cluster_absent": True,
                "remote_materialization_excluded": True,
            },
            "authentication": AUTHENTICATION,
            "closure_evidence": {
                "remote_archive_identity": {
                    "path": REMOTE_IDENTITY_NAME,
                    "canonical_sha256": remote_identity["sha256"],
                },
                "before_removal_attempt": {
                    "path": BEFORE_RETIREMENT_NAME,
                    "canonical_sha256": evidence["before"]["sha256"],
                },
                "acknowledged_removal_attempts": {
                    "path": REMOVAL_ATTEMPTS_NAME,
                    "canonical_sha256": evidence["attempts"]["sha256"],
                },
                "after_materialization_exclusion": {
                    "path": AFTER_EXCLUSION_NAME,
                    "canonical_sha256": evidence["after"]["sha256"],
                },
                "terminal_history": {
                    "path": HISTORY_NAME,
                    "canonical_sha256": evidence["history"]["sha256"],
                },
            },
            "local_finalizer": {
                "path": CONTINUATION_ADAPTER_RELATIVE.parent.joinpath(
                    HELPER_PATH.name
                ).as_posix(),
                "strict_consumer_path": CONTINUATION_ADAPTER_RELATIVE.as_posix(),
                "strict_consumer_file_sha256": EXPECTED_CONTINUATION_ADAPTER_SHA256,
                "network_or_scheduler_commands_executed": False,
            },
            "scientific_execution_performed_by_action": False,
        }
    )
    receipt_path = workspace_root / RECEIPT_RELATIVE
    _atomic_publish_receipt(
        receipt_path,
        receipt,
        workspace_root=workspace_root,
        worker=worker,
        job=job,
    )
    return {
        "schema": "paper_i_page16_sw_always_local_closure_finalization_v2",
        "status": "passed_strict_receipt_atomically_published",
        "receipt_path": RECEIPT_RELATIVE.as_posix(),
        "receipt_canonical_sha256": receipt["sha256"],
        "archive_closure": archive["summary"],
        "scheduler_mutation_performed": False,
        "scientific_execution_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=SOURCE_REPO_ROOT,
        help="local workspace containing the fixed already-fetched archive",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=SOURCE_REPO_ROOT / REPAIR_RELATIVE / "page16_sw_always_closure_evidence",
        help="directory of independently captured authenticated JSON evidence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.preflight:
            result = _preflight(args.workspace_root.resolve(), args.evidence_dir.resolve())
        else:
            result = _finalize(args.workspace_root.resolve(), args.evidence_dir.resolve())
    except ClosureError as exc:
        print(f"closure error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
