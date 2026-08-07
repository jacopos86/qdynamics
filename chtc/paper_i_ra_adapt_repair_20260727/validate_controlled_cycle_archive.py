#!/usr/bin/env python3
"""Authenticate one fetched cluster-9397758/9397760 worker archive.

This utility is deliberately local-only.  It never connects to CHTC and never
changes scheduler state.  Validation streams the tar payload without
extracting it.  Receipt emission is optional and refuses to overwrite an
existing path so that sealed submission/release receipts remain immutable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, BinaryIO, Mapping


BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]
SUPPORTED_CLUSTER_IDS = frozenset({9397758, 9397760})
COMPLETION_RECEIPT_SCHEMA = (
    "paper_i_checkpoint_retention_controlled_cycle_"
    "retrieval_completion_receipt_v1"
)
ATTEMPT_SCHEMAS = frozenset(
    {
        "paper_i_ra_always_factorial_worker_attempt_v1",
        "paper_i_ra_global_singleton_insertion12_worker_attempt_v1",
    }
)
AUTHORITY_MEMBERS = (
    "authority/job.json",
    "authority/execution_authorization.json",
    "authority/activation_manifest.json",
)
ATTEMPT_RECEIPT_MEMBER = "worker_attempt_receipt.json"
REQUIRED_WORKER_PATHS = frozenset(
    {"attempt_identity.tsv", "result.json", "worker_exit_status.txt"}
)
SMALL_MEMBER_LIMIT_BYTES = 16 * 1024 * 1024
SHA256_HEX_LENGTH = 64


class ControlledCycleArchiveError(ValueError):
    """Raised when a fetched attempt or completion receipt is unsafe."""


@dataclass(frozen=True)
class ExpectedAttempt:
    """Exact local and scheduler identity expected for one attempt."""

    execution_id: str
    cluster_id: int
    proc_id: int
    job_path: Path
    authorization_path: Path
    activation_manifest_path: Path
    source_archive_sha256: str
    image_sha256: str


@dataclass(frozen=True)
class RemoteCycleObservation:
    """Explicit remote observations used only when sealing a receipt."""

    receipt_created_utc: str
    retrieved_utc: str
    owner: str
    host: str
    schedd: str
    remote_root: str
    remote_archive_path: str
    remote_archive_sha256: str
    remote_archive_size_bytes: int
    release_target: str
    released_utc: str
    release_exit_code: int
    quota_observed_utc: str
    quota_home_used_gib: float
    quota_home_soft_limit_gib: float
    quota_home_hard_limit_gib: float


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the repository's canonical JSON projection."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy *payload* and append its canonical SHA-256 self digest."""

    result = dict(payload)
    if "sha256" in result:
        raise ControlledCycleArchiveError(
            "Self-digest input already contains sha256."
        )
    result["sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    """Verify a top-level canonical ``sha256`` field."""

    observed = payload.get("sha256")
    projection = dict(payload)
    projection.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(projection)).hexdigest()
    if observed != expected:
        raise ControlledCycleArchiveError(f"{label} self digest drifted.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if not _valid_sha256(value):
        raise ControlledCycleArchiveError(
            f"{label} is not a lowercase SHA-256 digest."
        )
    return str(value)


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ControlledCycleArchiveError(f"{label} is not a valid integer.")
    return value


def _require_plain_file(path: Path, *, label: str) -> Path:
    candidate = path.absolute()
    if not candidate.is_file() or candidate.is_symlink():
        raise ControlledCycleArchiveError(
            f"{label} is missing, not regular, or a symlink: {path}"
        )
    return candidate


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlledCycleArchiveError(
                f"JSON object contains duplicate key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ControlledCycleArchiveError(f"JSON contains non-finite value: {value}")


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlledCycleArchiveError(f"Malformed {label} JSON.") from exc
    if not isinstance(parsed, dict):
        raise ControlledCycleArchiveError(f"{label} is not a JSON object.")
    return parsed


def _load_json_file(path: Path, *, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    candidate = _require_plain_file(path, label=label)
    if candidate.stat().st_size > SMALL_MEMBER_LIMIT_BYTES:
        raise ControlledCycleArchiveError(f"{label} is unexpectedly large.")
    payload = candidate.read_bytes()
    parsed = _load_json_bytes(payload, label=label)
    verify_self_digest(parsed, label=label)
    return candidate, payload, parsed


def _safe_member_name(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ControlledCycleArchiveError(f"Unsafe tar member name: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
        or path.as_posix() != value
    ):
        raise ControlledCycleArchiveError(f"Unsafe tar member name: {value!r}")
    return value


def _display_path(path: Path) -> str:
    candidate = path.absolute()
    try:
        return candidate.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()


def _consume_member(
    stream: BinaryIO,
    *,
    expected_size: int,
    capture: bool,
    label: str,
) -> tuple[str, int, bytes | None]:
    if capture and expected_size > SMALL_MEMBER_LIMIT_BYTES:
        raise ControlledCycleArchiveError(f"{label} is unexpectedly large.")
    digest = hashlib.sha256()
    size = 0
    chunks: list[bytes] | None = [] if capture else None
    while block := stream.read(8 * 1024 * 1024):
        digest.update(block)
        size += len(block)
        if chunks is not None:
            chunks.append(block)
    if size != expected_size:
        raise ControlledCycleArchiveError(
            f"{label} size differs from its tar header."
        )
    return digest.hexdigest(), size, b"".join(chunks) if chunks is not None else None


def _validate_expected(expected: ExpectedAttempt) -> None:
    if not expected.execution_id or any(
        character in expected.execution_id for character in ("\x00", "\n", "\r")
    ):
        raise ControlledCycleArchiveError("Execution ID is unsafe.")
    if expected.cluster_id not in SUPPORTED_CLUSTER_IDS:
        raise ControlledCycleArchiveError(
            f"Unsupported controlled-cycle cluster: {expected.cluster_id}"
        )
    _require_int(expected.proc_id, label="proc_id")
    _require_sha256(
        expected.source_archive_sha256,
        label="expected source archive SHA-256",
    )
    _require_sha256(expected.image_sha256, label="expected image SHA-256")


def _binding(
    path: Path,
    payload: bytes,
    parsed: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_sha256": str(parsed["sha256"]),
        "size_bytes": len(payload),
    }


def _validate_authority_relations(
    *,
    expected: ExpectedAttempt,
    job: Mapping[str, Any],
    job_file_sha256: str,
    authorization: Mapping[str, Any],
    authorization_file_sha256: str,
    activation: Mapping[str, Any],
) -> Mapping[str, Any]:
    if job.get("execution_id") != expected.execution_id:
        raise ControlledCycleArchiveError("Job execution binding drifted.")
    remote_image = job.get("remote_image")
    if (
        not isinstance(remote_image, Mapping)
        or remote_image.get("sha256") != expected.image_sha256
    ):
        raise ControlledCycleArchiveError("Job image binding drifted.")
    if (
        authorization.get("execution_id") != expected.execution_id
        or authorization.get("job_file_sha256") != job_file_sha256
        or authorization.get("job_sha256") != job.get("sha256")
        or authorization.get("source_archive_sha256")
        != expected.source_archive_sha256
        or authorization.get("remote_image_sha256") != expected.image_sha256
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
    ):
        raise ControlledCycleArchiveError(
            "Execution authorization relation closure failed."
        )
    if (
        authorization.get("activation_id") != activation.get("activation_id")
        or authorization.get("activation_control_plane_sha256")
        != activation.get("activation_control_plane_sha256")
    ):
        raise ControlledCycleArchiveError(
            "Authorization-to-activation binding drifted."
        )
    activation_image = activation.get("remote_image")
    if (
        activation.get("source_archive_sha256")
        != expected.source_archive_sha256
        or not isinstance(activation_image, Mapping)
        or activation_image.get("sha256") != expected.image_sha256
    ):
        raise ControlledCycleArchiveError(
            "Activation source/image binding drifted."
        )
    executions = activation.get("executions")
    if not isinstance(executions, list):
        raise ControlledCycleArchiveError("Activation executions are malformed.")
    matching = [
        row
        for row in executions
        if isinstance(row, Mapping)
        and row.get("execution_id") == expected.execution_id
    ]
    if len(matching) != 1:
        raise ControlledCycleArchiveError(
            "Activation does not contain exactly one expected execution."
        )
    execution = matching[0]
    job_binding = execution.get("job")
    authorization_binding = execution.get("authorization")
    if (
        not isinstance(job_binding, Mapping)
        or job_binding.get("sha256") != job_file_sha256
        or job_binding.get("canonical_sha256") != job.get("sha256")
        or not isinstance(authorization_binding, Mapping)
        or authorization_binding.get("sha256") != authorization_file_sha256
        or authorization_binding.get("canonical_sha256")
        != authorization.get("sha256")
    ):
        raise ControlledCycleArchiveError(
            "Activation job/authorization relation closure failed."
        )
    return execution


def validate_attempt_archive(
    archive_path: Path,
    expected: ExpectedAttempt,
) -> dict[str, Any]:
    """Stream-validate one fetched worker archive without extracting it."""

    _validate_expected(expected)
    archive_file = _require_plain_file(archive_path, label="fetched archive")
    job_path, job_bytes, job = _load_json_file(
        expected.job_path, label="expected job"
    )
    authorization_path, authorization_bytes, authorization = _load_json_file(
        expected.authorization_path,
        label="expected execution authorization",
    )
    activation_path, activation_bytes, activation = _load_json_file(
        expected.activation_manifest_path,
        label="expected activation manifest",
    )
    job_file_sha256 = hashlib.sha256(job_bytes).hexdigest()
    authorization_file_sha256 = hashlib.sha256(
        authorization_bytes
    ).hexdigest()
    activation_file_sha256 = hashlib.sha256(activation_bytes).hexdigest()
    _validate_authority_relations(
        expected=expected,
        job=job,
        job_file_sha256=job_file_sha256,
        authorization=authorization,
        authorization_file_sha256=authorization_file_sha256,
        activation=activation,
    )

    member_names: list[str] = []
    observed_worker_files: dict[str, dict[str, Any]] = {}
    captured_worker_files: dict[str, bytes] = {}
    authority_payloads: dict[str, bytes] = {}
    attempt_receipt_bytes: bytes | None = None
    try:
        with archive_file.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as decompressed:
                with tarfile.open(fileobj=decompressed, mode="r|") as archive:
                    for member in archive:
                        member_name = _safe_member_name(member.name)
                        if member_name in member_names:
                            raise ControlledCycleArchiveError(
                                f"Duplicate tar member: {member_name}"
                            )
                        member_names.append(member_name)
                        if member.type not in {
                            tarfile.REGTYPE,
                            tarfile.AREGTYPE,
                        }:
                            raise ControlledCycleArchiveError(
                                f"Non-regular tar member: {member_name}"
                            )
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise ControlledCycleArchiveError(
                                f"Unreadable tar member: {member_name}"
                            )
                        is_worker = member_name.startswith("worker_outputs/")
                        worker_relative = (
                            member_name.removeprefix("worker_outputs/")
                            if is_worker
                            else None
                        )
                        capture = (
                            member_name in AUTHORITY_MEMBERS
                            or member_name == ATTEMPT_RECEIPT_MEMBER
                            or worker_relative
                            in {"attempt_identity.tsv", "worker_exit_status.txt"}
                        )
                        digest, size, payload = _consume_member(
                            stream,
                            expected_size=member.size,
                            capture=capture,
                            label=f"tar member {member_name}",
                        )
                        if is_worker:
                            if not worker_relative:
                                raise ControlledCycleArchiveError(
                                    "Empty worker member path."
                                )
                            _safe_member_name(worker_relative)
                            observed_worker_files[worker_relative] = {
                                "sha256": digest,
                                "size_bytes": size,
                            }
                            if payload is not None:
                                captured_worker_files[worker_relative] = payload
                        elif member_name in AUTHORITY_MEMBERS:
                            assert payload is not None
                            authority_payloads[member_name] = payload
                        elif member_name == ATTEMPT_RECEIPT_MEMBER:
                            assert payload is not None
                            attempt_receipt_bytes = payload
                        else:
                            raise ControlledCycleArchiveError(
                                f"Unexpected archive member: {member_name}"
                            )
                while trailing := decompressed.read(8 * 1024 * 1024):
                    if trailing.strip(b"\0"):
                        raise ControlledCycleArchiveError(
                            "Archive contains non-zero trailing tar payload."
                        )
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as exc:
        raise ControlledCycleArchiveError(
            f"Fetched archive is not a complete gzip tar: {archive_path}"
        ) from exc

    if set(authority_payloads) != set(AUTHORITY_MEMBERS):
        raise ControlledCycleArchiveError("Archive authority closure is incomplete.")
    if attempt_receipt_bytes is None:
        raise ControlledCycleArchiveError("Worker attempt receipt is missing.")
    archived_expected_payloads = {
        "authority/job.json": job_bytes,
        "authority/execution_authorization.json": authorization_bytes,
        "authority/activation_manifest.json": activation_bytes,
    }
    for name, expected_payload in archived_expected_payloads.items():
        if authority_payloads[name] != expected_payload:
            raise ControlledCycleArchiveError(
                f"Archived authority bytes do not match {name}."
            )

    attempt_receipt = _load_json_bytes(
        attempt_receipt_bytes,
        label="worker attempt receipt",
    )
    verify_self_digest(attempt_receipt, label="worker attempt receipt")
    if attempt_receipt.get("schema") not in ATTEMPT_SCHEMAS:
        raise ControlledCycleArchiveError("Unknown worker attempt receipt schema.")
    attempt_ordinal = _require_int(
        attempt_receipt.get("attempt_ordinal"),
        label="attempt ordinal",
        minimum=1,
    )
    worker_exit_status = _require_int(
        attempt_receipt.get("worker_exit_status"),
        label="worker exit status",
    )
    if (
        attempt_receipt.get("execution_id") != expected.execution_id
        or attempt_receipt.get("cluster_id") != expected.cluster_id
        or attempt_receipt.get("proc_id") != expected.proc_id
    ):
        raise ControlledCycleArchiveError(
            "Worker attempt execution/cluster/proc identity drifted."
        )
    if worker_exit_status != 0:
        raise ControlledCycleArchiveError("Worker did not exit successfully.")
    if (
        attempt_receipt.get("job_file_sha256") != job_file_sha256
        or attempt_receipt.get("authorization_file_sha256")
        != authorization_file_sha256
        or attempt_receipt.get("activation_manifest_file_sha256")
        != activation_file_sha256
        or attempt_receipt.get("source_archive_sha256")
        != expected.source_archive_sha256
        or attempt_receipt.get("image_sha256") != expected.image_sha256
    ):
        raise ControlledCycleArchiveError(
            "Worker attempt authority/source/image binding drifted."
        )

    worker_rows = attempt_receipt.get("worker_files")
    if not isinstance(worker_rows, list):
        raise ControlledCycleArchiveError("Worker inventory is malformed.")
    declared_worker_files: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(worker_rows):
        if not isinstance(row, Mapping):
            raise ControlledCycleArchiveError(
                f"Worker inventory row {index} is malformed."
            )
        path_value = row.get("path")
        if not isinstance(path_value, str):
            raise ControlledCycleArchiveError(
                f"Worker inventory row {index} has no path."
            )
        worker_path = _safe_member_name(path_value)
        if worker_path in declared_worker_files:
            raise ControlledCycleArchiveError(
                f"Duplicate worker inventory path: {worker_path}"
            )
        declared_worker_files[worker_path] = {
            "sha256": _require_sha256(
                row.get("sha256"),
                label=f"worker inventory SHA-256 for {worker_path}",
            ),
            "size_bytes": _require_int(
                row.get("size_bytes"),
                label=f"worker inventory size for {worker_path}",
            ),
        }
    if set(declared_worker_files) != set(observed_worker_files):
        raise ControlledCycleArchiveError(
            "Worker inventory does not close over every archived worker file."
        )
    if not REQUIRED_WORKER_PATHS.issubset(declared_worker_files):
        raise ControlledCycleArchiveError(
            "Worker inventory is missing a required completion artifact."
        )
    for worker_path, declared in declared_worker_files.items():
        if declared != observed_worker_files[worker_path]:
            raise ControlledCycleArchiveError(
                f"Worker inventory hash/size mismatch: {worker_path}"
            )
    expected_member_names = {
        f"worker_outputs/{worker_path}"
        for worker_path in declared_worker_files
    } | set(AUTHORITY_MEMBERS) | {ATTEMPT_RECEIPT_MEMBER}
    if set(member_names) != expected_member_names:
        raise ControlledCycleArchiveError("Archive member closure drifted.")

    expected_attempt_identity = (
        f"{expected.execution_id}\t{expected.cluster_id}\t"
        f"{expected.proc_id}\t{attempt_ordinal}\n"
    ).encode("utf-8")
    if captured_worker_files.get("attempt_identity.tsv") != expected_attempt_identity:
        raise ControlledCycleArchiveError("Worker attempt marker drifted.")
    if captured_worker_files.get("worker_exit_status.txt") != b"0\n":
        raise ControlledCycleArchiveError("Worker exit-status sidecar drifted.")

    archive_sha256 = _sha256_file(archive_file)
    return {
        "status": "passed",
        "execution_id": expected.execution_id,
        "cluster_id": expected.cluster_id,
        "proc_id": expected.proc_id,
        "attempt_ordinal": attempt_ordinal,
        "archive": {
            "path": _display_path(archive_file),
            "sha256": archive_sha256,
            "size_bytes": archive_file.stat().st_size,
        },
        "worker_attempt_receipt": {
            "schema": str(attempt_receipt["schema"]),
            "canonical_sha256": str(attempt_receipt["sha256"]),
            "file_sha256": hashlib.sha256(attempt_receipt_bytes).hexdigest(),
            "worker_exit_status": worker_exit_status,
        },
        "member_validation": {
            "gzip_and_full_tar_scan_passed": True,
            "safe_unique_regular_only_member_closure_passed": True,
            "worker_inventory_hash_size_closure_passed": True,
            "authority_byte_identity_passed": True,
            "member_count": len(member_names),
            "worker_file_count": len(declared_worker_files),
        },
        "bindings": {
            "job": _binding(job_path, job_bytes, job),
            "authorization": _binding(
                authorization_path,
                authorization_bytes,
                authorization,
            ),
            "activation_manifest": _binding(
                activation_path,
                activation_bytes,
                activation,
            ),
            "source_archive_sha256": expected.source_archive_sha256,
            "image_sha256": expected.image_sha256,
        },
    }


def _validate_utc(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ControlledCycleArchiveError(f"{label} must be UTC with a Z suffix.")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ControlledCycleArchiveError(f"{label} is not RFC-3339.") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ControlledCycleArchiveError(f"{label} is not UTC.")
    return value


def _nonempty(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ControlledCycleArchiveError(f"{label} is empty or unsafe.")
    return value


def _quota_value(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlledCycleArchiveError(f"{label} is not numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ControlledCycleArchiveError(f"{label} is invalid.")
    return result


def _load_submission_receipt(
    path: Path,
    *,
    expected: ExpectedAttempt,
    validation: Mapping[str, Any],
    remote: RemoteCycleObservation,
) -> tuple[Path, bytes, dict[str, Any]]:
    receipt_path, receipt_bytes, receipt = _load_json_file(
        path, label="submission receipt"
    )
    activation = validation["bindings"]["activation_manifest"]
    source_sha256 = validation["bindings"]["source_archive_sha256"]
    if (
        receipt.get("status") != "passed"
        or receipt.get("cluster_id") != expected.cluster_id
        or receipt.get("owner") != remote.owner
        or receipt.get("submit_host") != remote.host
        or receipt.get("schedd") != remote.schedd
        or receipt.get("remote_root") != remote.remote_root
    ):
        raise ControlledCycleArchiveError(
            "Submission receipt remote identity closure failed."
        )
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ControlledCycleArchiveError("Submission receipt bindings are malformed.")
    activation_binding = bindings.get("activation_manifest")
    source_binding = bindings.get("source_archive")
    if (
        not isinstance(activation_binding, Mapping)
        or activation_binding.get("sha256") != activation["sha256"]
        or activation_binding.get("canonical_sha256")
        != activation["canonical_sha256"]
        or not isinstance(source_binding, Mapping)
        or source_binding.get("sha256") != source_sha256
    ):
        raise ControlledCycleArchiveError(
            "Submission receipt activation/source binding drifted."
        )
    lifecycle = receipt.get("lifecycle")
    initial_state = receipt.get("initial_state")
    initial_proc_ids = (
        initial_state.get("proc_ids")
        if isinstance(initial_state, Mapping)
        else None
    )
    if (
        not isinstance(lifecycle, Mapping)
        or lifecycle.get("mode")
        != "ordinary_held_exact_proc_release_v1"
        or lifecycle.get("release_scope") != "exact_cluster_proc_only"
        or lifecycle.get("one_proc_per_quota_cycle") is not True
        or not isinstance(initial_state, Mapping)
        or not isinstance(initial_proc_ids, list)
        or expected.proc_id not in initial_proc_ids
    ):
        raise ControlledCycleArchiveError(
            "Submission receipt controlled-cycle lifecycle drifted."
        )
    return receipt_path, receipt_bytes, receipt


def build_completion_receipt(
    *,
    validation: Mapping[str, Any],
    expected: ExpectedAttempt,
    submission_receipt_path: Path,
    remote: RemoteCycleObservation,
) -> dict[str, Any]:
    """Build a new self-digested retrieval/completion receipt."""

    if validation.get("status") != "passed":
        raise ControlledCycleArchiveError(
            "A passing archive validation is required for receipt emission."
        )
    if (
        validation.get("execution_id") != expected.execution_id
        or validation.get("cluster_id") != expected.cluster_id
        or validation.get("proc_id") != expected.proc_id
    ):
        raise ControlledCycleArchiveError(
            "Validation identity does not match the expected attempt."
        )
    archive_binding = validation.get("archive")
    if not isinstance(archive_binding, Mapping):
        raise ControlledCycleArchiveError("Validation archive binding is malformed.")
    _validate_utc(remote.receipt_created_utc, label="receipt_created_utc")
    _validate_utc(remote.retrieved_utc, label="retrieved_utc")
    _validate_utc(remote.released_utc, label="released_utc")
    _validate_utc(remote.quota_observed_utc, label="quota_observed_utc")
    owner = _nonempty(remote.owner, label="remote owner")
    host = _nonempty(remote.host, label="remote host")
    schedd = _nonempty(remote.schedd, label="remote schedd")
    remote_root = _nonempty(remote.remote_root, label="remote root")
    remote_archive_path = _nonempty(
        remote.remote_archive_path,
        label="remote archive path",
    )
    remote_archive_sha256 = _require_sha256(
        remote.remote_archive_sha256,
        label="remote archive SHA-256",
    )
    remote_archive_size = _require_int(
        remote.remote_archive_size_bytes,
        label="remote archive size",
        minimum=1,
    )
    if (
        remote_archive_sha256 != archive_binding.get("sha256")
        or remote_archive_size != archive_binding.get("size_bytes")
    ):
        raise ControlledCycleArchiveError(
            "Remote and local archive hash/size bindings differ."
        )
    expected_target = f"{expected.cluster_id}.{expected.proc_id}"
    if remote.release_target != expected_target or remote.release_exit_code != 0:
        raise ControlledCycleArchiveError(
            "Release target or release exit status is not exact."
        )
    used = _quota_value(
        remote.quota_home_used_gib,
        label="quota home used GiB",
    )
    soft = _quota_value(
        remote.quota_home_soft_limit_gib,
        label="quota home soft limit GiB",
    )
    hard = _quota_value(
        remote.quota_home_hard_limit_gib,
        label="quota home hard limit GiB",
    )
    if soft <= 0 or hard < soft or used > hard:
        raise ControlledCycleArchiveError("Quota limits are inconsistent.")
    soft_headroom = float(Decimal(str(soft)) - Decimal(str(used)))
    submission_path, submission_bytes, submission = _load_submission_receipt(
        submission_receipt_path,
        expected=expected,
        validation=validation,
        remote=remote,
    )
    return digested(
        {
            "schema": COMPLETION_RECEIPT_SCHEMA,
            "receipt_created_utc": remote.receipt_created_utc,
            "status": "passed",
            "completion_classification": (
                "worker_exit_zero_archive_fully_authenticated"
            ),
            "execution": {
                "execution_id": expected.execution_id,
                "cluster_id": expected.cluster_id,
                "proc_id": expected.proc_id,
                "attempt_ordinal": validation["attempt_ordinal"],
            },
            "retrieval": {
                "retrieved_utc": remote.retrieved_utc,
                "remote_archive_path": remote_archive_path,
                "remote_archive_sha256": remote_archive_sha256,
                "remote_archive_size_bytes": remote_archive_size,
                "local_archive": dict(archive_binding),
                "remote_local_hash_size_match": True,
            },
            "remote_identity": {
                "owner": owner,
                "host": host,
                "schedd": schedd,
                "remote_root": remote_root,
            },
            "release": {
                "target": remote.release_target,
                "command": f"condor_release {remote.release_target}",
                "scope": "exact_cluster_proc_only",
                "released_utc": remote.released_utc,
                "exit_code": remote.release_exit_code,
            },
            "quota_after_retrieval": {
                "observed_utc": remote.quota_observed_utc,
                "home_used_gib": used,
                "home_soft_limit_gib": soft,
                "home_hard_limit_gib": hard,
                "soft_limit_headroom_gib": soft_headroom,
            },
            "bindings": {
                "submission_receipt": _binding(
                    submission_path,
                    submission_bytes,
                    submission,
                ),
                **dict(validation["bindings"]),
            },
            "archive_validation": {
                "worker_attempt_receipt": dict(
                    validation["worker_attempt_receipt"]
                ),
                **dict(validation["member_validation"]),
            },
            "paper_evidence_adopted": False,
        }
    )


def write_new_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically create, but never replace, a completion receipt."""

    verify_self_digest(receipt, label="completion receipt")
    output = path.absolute()
    if output.exists() or output.is_symlink():
        raise ControlledCycleArchiveError(
            f"Refusing to overwrite existing receipt: {path}"
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ControlledCycleArchiveError("Receipt parent is missing or unsafe.")
    payload = canonical_json_bytes(receipt) + b"\n"
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ControlledCycleArchiveError(
            f"Receipt temporary path already exists: {temporary}"
        )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _decimal_float(value: str, *, label: str) -> float:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"{label} is not numeric") from exc
    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError(f"{label} is invalid")
    return float(parsed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--proc-id", type=int, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--receipt-created-utc")
    parser.add_argument("--retrieved-utc")
    parser.add_argument("--submission-receipt", type=Path)
    parser.add_argument("--remote-owner")
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-schedd")
    parser.add_argument("--remote-root")
    parser.add_argument("--remote-archive-path")
    parser.add_argument("--remote-archive-sha256")
    parser.add_argument("--remote-archive-size-bytes", type=int)
    parser.add_argument("--release-target")
    parser.add_argument("--released-utc")
    parser.add_argument("--release-exit-code", type=int)
    parser.add_argument("--quota-observed-utc")
    parser.add_argument(
        "--quota-home-used-gib",
        type=lambda value: _decimal_float(value, label="quota home used GiB"),
    )
    parser.add_argument(
        "--quota-home-soft-limit-gib",
        type=lambda value: _decimal_float(
            value, label="quota home soft limit GiB"
        ),
    )
    parser.add_argument(
        "--quota-home-hard-limit-gib",
        type=lambda value: _decimal_float(
            value, label="quota home hard limit GiB"
        ),
    )
    return parser.parse_args()


def _required_receipt_argument(
    args: argparse.Namespace,
    name: str,
) -> Any:
    value = getattr(args, name)
    if value is None:
        raise ControlledCycleArchiveError(
            f"--{name.replace('_', '-')} is required with --receipt-output."
        )
    return value


def main() -> int:
    args = _parse_args()
    expected = ExpectedAttempt(
        execution_id=args.execution_id,
        cluster_id=args.cluster_id,
        proc_id=args.proc_id,
        job_path=args.job,
        authorization_path=args.authorization,
        activation_manifest_path=args.activation_manifest,
        source_archive_sha256=args.source_archive_sha256,
        image_sha256=args.image_sha256,
    )
    validation = validate_attempt_archive(args.archive, expected)
    output: Mapping[str, Any] = validation
    if args.receipt_output is not None:
        remote = RemoteCycleObservation(
            receipt_created_utc=_required_receipt_argument(
                args, "receipt_created_utc"
            ),
            retrieved_utc=_required_receipt_argument(args, "retrieved_utc"),
            owner=_required_receipt_argument(args, "remote_owner"),
            host=_required_receipt_argument(args, "remote_host"),
            schedd=_required_receipt_argument(args, "remote_schedd"),
            remote_root=_required_receipt_argument(args, "remote_root"),
            remote_archive_path=_required_receipt_argument(
                args, "remote_archive_path"
            ),
            remote_archive_sha256=_required_receipt_argument(
                args, "remote_archive_sha256"
            ),
            remote_archive_size_bytes=_required_receipt_argument(
                args, "remote_archive_size_bytes"
            ),
            release_target=_required_receipt_argument(args, "release_target"),
            released_utc=_required_receipt_argument(args, "released_utc"),
            release_exit_code=_required_receipt_argument(
                args, "release_exit_code"
            ),
            quota_observed_utc=_required_receipt_argument(
                args, "quota_observed_utc"
            ),
            quota_home_used_gib=_required_receipt_argument(
                args, "quota_home_used_gib"
            ),
            quota_home_soft_limit_gib=_required_receipt_argument(
                args, "quota_home_soft_limit_gib"
            ),
            quota_home_hard_limit_gib=_required_receipt_argument(
                args, "quota_home_hard_limit_gib"
            ),
        )
        receipt = build_completion_receipt(
            validation=validation,
            expected=expected,
            submission_receipt_path=_required_receipt_argument(
                args, "submission_receipt"
            ),
            remote=remote,
        )
        write_new_receipt(args.receipt_output, receipt)
        output = {
            "status": "passed",
            "receipt_output": _display_path(args.receipt_output),
            "receipt_sha256": receipt["sha256"],
            "archive_sha256": validation["archive"]["sha256"],
        }
    print(canonical_json_bytes(output).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
