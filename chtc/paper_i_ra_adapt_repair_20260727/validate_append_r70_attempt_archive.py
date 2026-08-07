#!/usr/bin/env python3
"""Authenticate one fetched Append-r70 cluster-9398375 attempt archive.

This utility is local-only.  It streams the archive without extracting it,
binds the attempt to the immutable Append package, held activation, and exact
proc/job row, and optionally creates a new retrieval-authentication receipt.
It does not replay the scientific calculation or authenticate scheduler
terminal history, release, rename, or remote-cleanup operations.
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

CLUSTER_ID = 9_398_375
PACKAGE_ID = (
    "paper_i_append_adapt_stationary_core12_r70_fresh_"
    "20260731_v1_chtc"
)
CAMPAIGN_ID = "paper_i_append_adapt_stationary_core_r70_fresh_v1"
ACTIVATION_ID = f"{PACKAGE_ID}_activation_ordinary_held_v1"
BATCH_NAME = (
    "paper-i-append-adapt-stationary-core12-r70-fresh-"
    "20260731-v1-ordinary-held-v1"
)
REMOTE_ROOT = "/home/jsstrobel/Holstein_phase3_optuna_chtc"
REMOTE_OWNER = "jsstrobel"
REMOTE_HOST = "ap2001.chtc.wisc.edu"
REMOTE_SCHEDD = REMOTE_HOST

PACKAGE_MANIFEST_FILE_SHA256 = (
    "334fb630c061d205d61554dca4b3e4f734edf5af394eda1ddb9c994869efefee"
)
PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "eea38b59e60d727281dc3bdaf6d2efa7880f3f49375ce49e61134fbb35a566ea"
)
AUTHORIZATION_FILE_SHA256 = (
    "4662e0b4732559b8efe26c1c27a9be3f94b3649751b6100dd186b5b55ca8d979"
)
AUTHORIZATION_CANONICAL_SHA256 = (
    "6e3f7378fdbe352e29ba60c119ae94b3a8351a4687ea5346ec6d5368e1d8fbd4"
)
ACTIVATION_FILE_SHA256 = (
    "77c813fafc0173af32802a8dac055f0e38bb4d38c196065d37c9776d07a5f9b9"
)
ACTIVATION_CANONICAL_SHA256 = (
    "543ff80423e3348800d975bab139ccac53fc87192ef3065e3e9e0fbebd183ce2"
)
SUBMISSION_RECEIPT_FILE_SHA256 = (
    "bec08c95d2fdcda2c558dd73d6e86148850693895080a4a75203b13fef45dc3d"
)
SUBMISSION_RECEIPT_CANONICAL_SHA256 = (
    "8c0847eb85e6e332f4bca90aef8ddba2ed4bffb81bfbf56649c25810b75d9b9d"
)
SOURCE_ARCHIVE_SHA256 = (
    "1f949b0cc8b61dca63911832e8dc8bb32614174755ac476827956bb0812accee"
)
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)

PACKAGE_SCHEMA = "paper_i_append_adapt_stationary_core_r70_fresh_package_v1"
JOB_SCHEMA = "paper_i_append_adapt_stationary_core_r70_fresh_job_v1"
AUTHORIZATION_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_execution_authorization_v1"
)
ACTIVATION_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_held_activation_v1"
)
SUBMISSION_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_ordinary_held_"
    "submission_receipt_v1"
)
ATTEMPT_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_worker_attempt_v1"
)
WORKER_RECEIPT_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_worker_receipt_v1"
)
EXECUTION_MANIFEST_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_execution_manifest_v1"
)
CHECKPOINT_SCHEMA = "paper_i_append_adapt_reconstruction_checkpoint_v1"
SUMMARY_SCHEMA = "paper_i_append_run_summary_v1"
RETRIEVAL_RECEIPT_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_"
    "retrieval_authentication_receipt_v1"
)

AUTHORITY_MEMBERS = (
    "authority/job.json",
    "authority/execution_authorization.json",
    "authority/activation_manifest.json",
)
ATTEMPT_RECEIPT_MEMBER = "worker_attempt_receipt.json"
ARTIFACT_ROLES = (
    "checkpoint",
    "estimator_ledger",
    "execution_manifest",
    "result",
    "summary",
)
ARTIFACT_FILENAMES = {role: f"{role}.json" for role in ARTIFACT_ROLES}
REQUIRED_WORKER_PATHS = frozenset(
    {
        "attempt_identity.tsv",
        "worker_exit_status.txt",
        "worker_receipt.json",
        *(f"payload/{name}" for name in ARTIFACT_FILENAMES.values()),
    }
)
CAPTURED_WORKER_PATHS = frozenset(
    {
        "attempt_identity.tsv",
        "worker_exit_status.txt",
        "worker_receipt.json",
        "payload/checkpoint.json",
        "payload/execution_manifest.json",
        "payload/summary.json",
    }
)
SMALL_MEMBER_LIMIT_BYTES = 16 * 1024 * 1024
SHA256_HEX_LENGTH = 64


class AppendAttemptArchiveError(ValueError):
    """Raised when an Append-r70 archive or its provenance is unsafe."""


class _BoundedDecompressedReader:
    """Count every decompressed byte and fail before crossing a fixed cap."""

    def __init__(self, stream: BinaryIO, *, limit_bytes: int) -> None:
        self._stream = stream
        self.limit_bytes = _require_int(
            limit_bytes,
            label="decompressed archive limit",
            minimum=1,
        )
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self.limit_bytes - self.bytes_read
        probe_bytes = remaining + 1
        request_bytes = (
            min(probe_bytes, 8 * 1024 * 1024)
            if size is None or size < 0
            else min(size, probe_bytes)
        )
        block = self._stream.read(request_bytes)
        self.bytes_read += len(block)
        if self.bytes_read > self.limit_bytes:
            raise AppendAttemptArchiveError(
                "Archive exceeds the activated row decompressed size budget."
            )
        return block

    def readinto(self, buffer: bytearray) -> int:
        block = self.read(len(buffer))
        buffer[: len(block)] = block
        return len(block)

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.bytes_read


@dataclass(frozen=True)
class ExpectedAppendAttempt:
    """Exact local authority and scheduler identity for one attempt."""

    execution_id: str
    cluster_id: int
    proc_id: int
    package_manifest_path: Path
    job_path: Path
    authorization_path: Path
    activation_manifest_path: Path


@dataclass(frozen=True)
class RemoteRetrievalObservation:
    """Observed remote facts required for a retrieval receipt."""

    receipt_created_utc: str
    remote_identity_observed_utc: str
    retrieved_utc: str
    remote_archive_sha256: str
    remote_archive_size_bytes: int
    quota_observed_utc: str
    quota_home_used_gib: float
    quota_home_soft_limit_gib: float
    quota_home_hard_limit_gib: float


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "sha256" in result:
        raise AppendAttemptArchiveError("Self-digest input already has sha256.")
    result["sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    observed = payload.get("sha256")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise AppendAttemptArchiveError(f"{label} self digest drifted.")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if not _valid_sha256(value):
        raise AppendAttemptArchiveError(
            f"{label} is not a lowercase SHA-256 digest."
        )
    return str(value)


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AppendAttemptArchiveError(f"{label} is not a valid integer.")
    return value


def _require_plain_file(path: Path, *, label: str) -> Path:
    candidate = path.absolute()
    if not candidate.is_file() or candidate.is_symlink():
        raise AppendAttemptArchiveError(
            f"{label} is missing, not regular, or a symlink: {path}"
        )
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AppendAttemptArchiveError(
                f"JSON object contains duplicate key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AppendAttemptArchiveError(f"JSON contains non-finite value: {value}")


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AppendAttemptArchiveError(f"Malformed {label} JSON.") from exc
    if not isinstance(parsed, dict):
        raise AppendAttemptArchiveError(f"{label} is not a JSON object.")
    return parsed


def _load_json_file(
    path: Path,
    *,
    label: str,
    expected_file_sha256: str | None = None,
    expected_canonical_sha256: str | None = None,
) -> tuple[Path, bytes, dict[str, Any]]:
    candidate = _require_plain_file(path, label=label)
    if candidate.stat().st_size > SMALL_MEMBER_LIMIT_BYTES:
        raise AppendAttemptArchiveError(f"{label} is unexpectedly large.")
    payload = candidate.read_bytes()
    file_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise AppendAttemptArchiveError(f"{label} file digest drifted.")
    parsed = _load_json_bytes(payload, label=label)
    verify_self_digest(parsed, label=label)
    if (
        expected_canonical_sha256 is not None
        and parsed.get("sha256") != expected_canonical_sha256
    ):
        raise AppendAttemptArchiveError(f"{label} canonical digest drifted.")
    return candidate, payload, parsed


def _safe_member_name(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise AppendAttemptArchiveError(f"Unsafe tar member name: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
        or path.as_posix() != value
    ):
        raise AppendAttemptArchiveError(f"Unsafe tar member name: {value!r}")
    return value


def _consume_member(
    stream: BinaryIO,
    *,
    expected_size: int,
    capture: bool,
    label: str,
) -> tuple[str, int, bytes | None]:
    if capture and expected_size > SMALL_MEMBER_LIMIT_BYTES:
        raise AppendAttemptArchiveError(f"{label} is unexpectedly large.")
    digest = hashlib.sha256()
    size = 0
    chunks: list[bytes] | None = [] if capture else None
    while block := stream.read(8 * 1024 * 1024):
        digest.update(block)
        size += len(block)
        if chunks is not None:
            chunks.append(block)
    if size != expected_size:
        raise AppendAttemptArchiveError(
            f"{label} size differs from its tar header."
        )
    payload = b"".join(chunks) if chunks is not None else None
    return digest.hexdigest(), size, payload


def _bounded_uncompressed_total(
    *,
    current_bytes: int,
    member_bytes: int,
    limit_bytes: int,
) -> int:
    """Return the next total or reject a tar header beyond row disk scope."""

    current = _require_int(
        current_bytes, label="current uncompressed archive size"
    )
    member = _require_int(member_bytes, label="tar member size")
    limit = _require_int(
        limit_bytes, label="activated row uncompressed archive limit", minimum=1
    )
    if current > limit or member > limit - current:
        raise AppendAttemptArchiveError(
            "Archive exceeds the activated row uncompressed size budget."
        )
    return current + member


def _display_path(path: Path) -> str:
    candidate = path.absolute()
    try:
        return candidate.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()


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


def _expected_archive_name(expected: ExpectedAppendAttempt) -> str:
    return (
        f"{expected.execution_id}__cluster_{expected.cluster_id}__"
        f"proc_{expected.proc_id}.tar.gz"
    )


def _expected_remote_archive_path(expected: ExpectedAppendAttempt) -> str:
    return (
        f"{REMOTE_ROOT}/chtc/paper_i_ra_adapt_repair_20260727/"
        f"{PACKAGE_ID}_runtime/fetched/{_expected_archive_name(expected)}"
    )


def _validate_local_authority(
    expected: ExpectedAppendAttempt,
) -> dict[str, Any]:
    if expected.cluster_id != CLUSTER_ID:
        raise AppendAttemptArchiveError(
            f"Unsupported Append-r70 cluster: {expected.cluster_id}"
        )
    _require_int(expected.proc_id, label="proc_id")
    if not expected.execution_id or any(
        character in expected.execution_id for character in ("\x00", "\n", "\r")
    ):
        raise AppendAttemptArchiveError("Execution ID is unsafe.")

    package_path, package_bytes, package = _load_json_file(
        expected.package_manifest_path,
        label="package manifest",
        expected_file_sha256=PACKAGE_MANIFEST_FILE_SHA256,
        expected_canonical_sha256=PACKAGE_MANIFEST_CANONICAL_SHA256,
    )
    job_path, job_bytes, job = _load_json_file(
        expected.job_path,
        label="job",
    )
    authorization_path, authorization_bytes, authorization = _load_json_file(
        expected.authorization_path,
        label="execution authorization",
        expected_file_sha256=AUTHORIZATION_FILE_SHA256,
        expected_canonical_sha256=AUTHORIZATION_CANONICAL_SHA256,
    )
    activation_path, activation_bytes, activation = _load_json_file(
        expected.activation_manifest_path,
        label="activation manifest",
        expected_file_sha256=ACTIVATION_FILE_SHA256,
        expected_canonical_sha256=ACTIVATION_CANONICAL_SHA256,
    )

    source_binding = package.get("source_archive")
    if (
        package.get("schema") != PACKAGE_SCHEMA
        or package.get("package_id") != PACKAGE_ID
        or package.get("campaign_id") != CAMPAIGN_ID
        or package.get("direct_execution_count") != 12
        or not isinstance(source_binding, Mapping)
        or source_binding.get("sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise AppendAttemptArchiveError("Package authority relation drifted.")

    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "passed"
        or authorization.get("activation_id") != ACTIVATION_ID
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("package_manifest_sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or authorization.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or authorization.get("remote_image_sha256") != IMAGE_SHA256
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
    ):
        raise AppendAttemptArchiveError(
            "Execution authorization relation closure failed."
        )
    authorized_ids = authorization.get("authorized_execution_ids")
    if (
        not isinstance(authorized_ids, list)
        or expected.execution_id not in authorized_ids
    ):
        raise AppendAttemptArchiveError("Execution is not authorized.")

    image_binding = activation.get("remote_image")
    sealed_package = activation.get("sealed_package")
    if (
        activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("activation_id") != ACTIVATION_ID
        or activation.get("package_id") != PACKAGE_ID
        or activation.get("campaign_id") != CAMPAIGN_ID
        or activation.get("batch_name") != BATCH_NAME
        or activation.get("operational_mode")
        != "ordinary_held_exact_proc_release_v1"
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not True
        or not isinstance(image_binding, Mapping)
        or image_binding.get("sha256") != IMAGE_SHA256
        or not isinstance(sealed_package, Mapping)
        or sealed_package.get("manifest_canonical_sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or sealed_package.get("manifest_file_sha256")
        != PACKAGE_MANIFEST_FILE_SHA256
        or sealed_package.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise AppendAttemptArchiveError("Activation authority relation drifted.")
    activation_authorization = activation.get("execution_authorization")
    if (
        not isinstance(activation_authorization, Mapping)
        or activation_authorization.get("sha256")
        != hashlib.sha256(authorization_bytes).hexdigest()
        or activation_authorization.get("canonical_sha256")
        != AUTHORIZATION_CANONICAL_SHA256
    ):
        raise AppendAttemptArchiveError(
            "Activation authorization binding drifted."
        )

    executions = activation.get("executions")
    if not isinstance(executions, list):
        raise AppendAttemptArchiveError("Activation executions are malformed.")
    matching = [
        row
        for row in executions
        if isinstance(row, Mapping)
        and row.get("execution_id") == expected.execution_id
    ]
    if len(matching) != 1:
        raise AppendAttemptArchiveError(
            "Activation does not contain exactly one expected execution."
        )
    execution = matching[0]
    job_binding = execution.get("job")
    activation_resources = execution.get("resources")
    job_resources = job.get("resources")
    job_file_sha256 = hashlib.sha256(job_bytes).hexdigest()
    if (
        execution.get("queue_index") != expected.proc_id
        or not isinstance(job_binding, Mapping)
        or job_binding.get("sha256") != job_file_sha256
        or job_binding.get("canonical_sha256") != job.get("sha256")
        or job_binding.get("size_bytes") != len(job_bytes)
        or not isinstance(activation_resources, Mapping)
        or not isinstance(job_resources, Mapping)
        or dict(activation_resources)
        != {
            "request_cpus": job_resources.get("request_cpus"),
            "request_memory_mb": job_resources.get("request_memory_mb"),
            "request_disk_mb": job_resources.get("request_disk_mb"),
            "max_runtime_seconds": job_resources.get("max_runtime_seconds"),
        }
    ):
        raise AppendAttemptArchiveError("Proc-to-job activation binding drifted.")
    request_disk_mb = _require_int(
        activation_resources.get("request_disk_mb"),
        label="activated row request_disk_mb",
        minimum=1,
    )

    fresh = job.get("fresh_start_contract")
    source_archive = job.get("source_archive")
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("execution_id") != expected.execution_id
        or job.get("cell_id") != expected.execution_id
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_entrypoint") != "run_append_adapt"
        or job.get("horizon") != {"source": 50, "target": 70}
        or not isinstance(fresh, Mapping)
        or fresh.get("kind") != "fresh_start"
        or fresh.get("controller_round_origin") != 0
        or fresh.get("resume_claimed") is not False
        or fresh.get("source_checkpoint_consumed") is not False
        or fresh.get("source_result_consumed") is not False
        or not isinstance(source_archive, Mapping)
        or source_archive.get("sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise AppendAttemptArchiveError("Append job scientific binding drifted.")
    protocol_sha256 = _require_sha256(
        job.get("derived_protocol_sha256"),
        label="derived protocol SHA-256",
    )

    return {
        "package": (package_path, package_bytes, package),
        "job": (job_path, job_bytes, job),
        "authorization": (
            authorization_path,
            authorization_bytes,
            authorization,
        ),
        "activation": (activation_path, activation_bytes, activation),
        "execution": execution,
        "protocol_sha256": protocol_sha256,
        "uncompressed_archive_limit_bytes": request_disk_mb * 1024 * 1024,
    }


def _parse_inventory(
    rows: Any,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise AppendAttemptArchiveError(f"{label} is malformed.")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise AppendAttemptArchiveError(f"{label} row {index} is malformed.")
        path = _safe_member_name(str(row["path"]))
        if path in result:
            raise AppendAttemptArchiveError(f"Duplicate {label} path: {path}")
        result[path] = {
            "sha256": _require_sha256(
                row.get("sha256"), label=f"{label} SHA-256 for {path}"
            ),
            "size_bytes": _require_int(
                row.get("size_bytes"), label=f"{label} size for {path}"
            ),
        }
    return result


def _validate_worker_payloads(
    *,
    expected: ExpectedAppendAttempt,
    authority: Mapping[str, Any],
    observed_worker_files: Mapping[str, Mapping[str, Any]],
    captured_worker_files: Mapping[str, bytes],
) -> dict[str, Any]:
    job_path, job_bytes, job = authority["job"]
    del job_path
    authorization_path, authorization_bytes, authorization = authority[
        "authorization"
    ]
    del authorization_path
    protocol_sha256 = str(authority["protocol_sha256"])

    worker_receipt_bytes = captured_worker_files["worker_receipt.json"]
    worker_receipt = _load_json_bytes(
        worker_receipt_bytes, label="Append worker receipt"
    )
    verify_self_digest(worker_receipt, label="Append worker receipt")
    if (
        worker_receipt.get("schema") != WORKER_RECEIPT_SCHEMA
        or worker_receipt.get("status") != "passed"
        or worker_receipt.get("package_id") != PACKAGE_ID
        or worker_receipt.get("campaign_id") != CAMPAIGN_ID
        or worker_receipt.get("execution_id") != expected.execution_id
        or worker_receipt.get("job_spec_sha256") != job.get("sha256")
        or worker_receipt.get("authorization_sha256")
        != authorization.get("sha256")
        or worker_receipt.get("derived_protocol_sha256") != protocol_sha256
        or worker_receipt.get("fresh_start") is not True
        or worker_receipt.get("resume_claimed") is not False
    ):
        raise AppendAttemptArchiveError("Append worker receipt binding drifted.")

    artifact_rows = worker_receipt.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise AppendAttemptArchiveError("Worker artifact inventory is malformed.")
    artifacts: dict[str, Mapping[str, Any]] = {}
    for row in artifact_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("role"), str):
            raise AppendAttemptArchiveError("Worker artifact row is malformed.")
        role = str(row["role"])
        if role in artifacts:
            raise AppendAttemptArchiveError(f"Duplicate worker artifact role: {role}")
        artifacts[role] = row
    if set(artifacts) != set(ARTIFACT_ROLES):
        raise AppendAttemptArchiveError("Worker artifact-role closure drifted.")
    job_artifact_paths = job.get("artifact_paths")
    if not isinstance(job_artifact_paths, Mapping):
        raise AppendAttemptArchiveError("Job artifact paths are malformed.")
    for role, filename in ARTIFACT_FILENAMES.items():
        row = artifacts[role]
        worker_path = f"payload/{filename}"
        observed = observed_worker_files[worker_path]
        if (
            row.get("path") != filename
            or row.get("declared_canonical_path") != job_artifact_paths.get(role)
            or row.get("sha256") != observed.get("sha256")
            or row.get("size_bytes") != observed.get("size_bytes")
        ):
            raise AppendAttemptArchiveError(
                f"Worker artifact binding drifted for {role}."
            )

    execution_manifest_bytes = captured_worker_files[
        "payload/execution_manifest.json"
    ]
    execution_manifest = _load_json_bytes(
        execution_manifest_bytes, label="Append execution manifest"
    )
    verify_self_digest(execution_manifest, label="Append execution manifest")
    if (
        execution_manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
        or execution_manifest.get("status") != "passed"
        or execution_manifest.get("package_id") != PACKAGE_ID
        or execution_manifest.get("campaign_id") != CAMPAIGN_ID
        or execution_manifest.get("execution_id") != expected.execution_id
        or execution_manifest.get("source_execution_id")
        != job.get("source_execution_id")
        or execution_manifest.get("job_spec_sha256") != job.get("sha256")
        or execution_manifest.get("authorization_sha256")
        != authorization.get("sha256")
        or execution_manifest.get("source_protocol_sha256")
        != job.get("source_protocol", {}).get("canonical_sha256")
        or execution_manifest.get("derived_protocol_sha256") != protocol_sha256
        or execution_manifest.get("source_horizon") != 50
        or execution_manifest.get("target_horizon") != 70
        or execution_manifest.get("controller_round_origin") != 0
        or execution_manifest.get("fresh_start") is not True
        or execution_manifest.get("source_checkpoint_consumed") is not False
        or execution_manifest.get("source_result_consumed") is not False
        or execution_manifest.get("resume_claimed") is not False
    ):
        raise AppendAttemptArchiveError("Execution manifest binding drifted.")
    output_payloads = execution_manifest.get("output_payloads")
    expected_preliminary_roles = set(ARTIFACT_ROLES) - {"execution_manifest"}
    if not isinstance(output_payloads, Mapping) or set(output_payloads) != (
        expected_preliminary_roles
    ):
        raise AppendAttemptArchiveError(
            "Execution-manifest output closure drifted."
        )
    for role in expected_preliminary_roles:
        row = output_payloads[role]
        observed = observed_worker_files[f"payload/{ARTIFACT_FILENAMES[role]}"]
        if not isinstance(row, Mapping) or dict(row) != dict(observed):
            raise AppendAttemptArchiveError(
                f"Execution-manifest payload binding drifted for {role}."
            )

    checkpoint = _load_json_bytes(
        captured_worker_files["payload/checkpoint.json"],
        label="Append reconstruction checkpoint",
    )
    verify_self_digest(checkpoint, label="Append reconstruction checkpoint")
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("execution_id") != expected.execution_id
        or checkpoint.get("protocol_sha256") != protocol_sha256
        or checkpoint.get("controller_rounds_completed") != 70
        or checkpoint.get("continuation_boundary")
        != "authenticated_reconstruction_only_v1"
        or checkpoint.get("public_resume_execution_supported") is not False
        or checkpoint.get("reconstruction_fields_complete") is not True
        or checkpoint.get("fresh_start_execution") is not True
        or checkpoint.get("source_checkpoint_consumed") is not False
        or checkpoint.get("source_result_consumed") is not False
        or checkpoint.get("resume_claimed") is not False
    ):
        raise AppendAttemptArchiveError("Checkpoint completion binding drifted.")

    summary = _load_json_bytes(
        captured_worker_files["payload/summary.json"],
        label="Append Paper-I summary",
    )
    history = summary.get("accepted_history")
    labels = summary.get("accepted_operator_labels")
    identities = summary.get("accepted_generator_identities")
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("protocol_sha256") != protocol_sha256
        or summary.get("protocol_horizon") != 70
        or summary.get("controller_rounds_completed") != 70
        or summary.get("stop_reason") != "maximum_controller_rounds"
        or not isinstance(history, list)
        or len(history) != 70
        or not isinstance(labels, list)
        or len(labels) != 70
        or not isinstance(identities, list)
        or len(identities) != 70
    ):
        raise AppendAttemptArchiveError("Append summary horizon closure drifted.")

    return {
        "worker_receipt": {
            "schema": WORKER_RECEIPT_SCHEMA,
            "canonical_sha256": str(worker_receipt["sha256"]),
            "file_sha256": hashlib.sha256(worker_receipt_bytes).hexdigest(),
        },
        "execution_manifest": {
            "schema": EXECUTION_MANIFEST_SCHEMA,
            "canonical_sha256": str(execution_manifest["sha256"]),
            "file_sha256": hashlib.sha256(execution_manifest_bytes).hexdigest(),
            "target_horizon": 70,
            "fresh_start": True,
        },
        "summary": {
            "schema": SUMMARY_SCHEMA,
            "file_sha256": hashlib.sha256(
                captured_worker_files["payload/summary.json"]
            ).hexdigest(),
            "controller_rounds_completed": 70,
            "stop_reason": "maximum_controller_rounds",
        },
        "job_file_sha256": hashlib.sha256(job_bytes).hexdigest(),
        "authorization_file_sha256": hashlib.sha256(
            authorization_bytes
        ).hexdigest(),
    }


def validate_append_attempt_archive(
    archive_path: Path,
    expected: ExpectedAppendAttempt,
) -> dict[str, Any]:
    """Stream-validate one fetched Append-r70 archive without extraction."""

    authority = _validate_local_authority(expected)
    archive_file = _require_plain_file(archive_path, label="fetched archive")
    if archive_file.name != _expected_archive_name(expected):
        raise AppendAttemptArchiveError("Fetched archive filename drifted.")

    member_names: set[str] = set()
    authority_payloads: dict[str, bytes] = {}
    observed_worker_files: dict[str, dict[str, Any]] = {}
    captured_worker_files: dict[str, bytes] = {}
    attempt_receipt_bytes: bytes | None = None
    total_member_payload_bytes = 0
    uncompressed_limit_bytes = int(
        authority["uncompressed_archive_limit_bytes"]
    )
    decompressed: _BoundedDecompressedReader | None = None
    try:
        with archive_file.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as gzip_stream:
                decompressed = _BoundedDecompressedReader(
                    gzip_stream,
                    limit_bytes=uncompressed_limit_bytes,
                )
                with tarfile.open(fileobj=decompressed, mode="r|") as archive:
                    for member in archive:
                        member_name = _safe_member_name(member.name)
                        if member_name in member_names:
                            raise AppendAttemptArchiveError(
                                f"Duplicate tar member: {member_name}"
                            )
                        member_names.add(member_name)
                        if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                            raise AppendAttemptArchiveError(
                                f"Non-regular tar member: {member_name}"
                            )
                        total_member_payload_bytes = _bounded_uncompressed_total(
                            current_bytes=total_member_payload_bytes,
                            member_bytes=member.size,
                            limit_bytes=uncompressed_limit_bytes,
                        )
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise AppendAttemptArchiveError(
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
                            or worker_relative in CAPTURED_WORKER_PATHS
                        )
                        digest, size, payload = _consume_member(
                            stream,
                            expected_size=member.size,
                            capture=capture,
                            label=f"tar member {member_name}",
                        )
                        if is_worker:
                            if not worker_relative:
                                raise AppendAttemptArchiveError(
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
                            raise AppendAttemptArchiveError(
                                f"Unexpected archive member: {member_name}"
                            )
                while trailing := decompressed.read(8 * 1024 * 1024):
                    if trailing.strip(b"\0"):
                        raise AppendAttemptArchiveError(
                            "Archive contains non-zero trailing tar payload."
                        )
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as exc:
        raise AppendAttemptArchiveError(
            f"Fetched archive is not a complete gzip tar: {archive_path}"
        ) from exc

    if set(authority_payloads) != set(AUTHORITY_MEMBERS):
        raise AppendAttemptArchiveError("Archive authority closure is incomplete.")
    if attempt_receipt_bytes is None:
        raise AppendAttemptArchiveError("Worker attempt receipt is missing.")
    if set(observed_worker_files) != set(REQUIRED_WORKER_PATHS):
        raise AppendAttemptArchiveError("Worker output member closure drifted.")
    if set(captured_worker_files) != set(CAPTURED_WORKER_PATHS):
        raise AppendAttemptArchiveError("Captured worker closure drifted.")

    expected_authority_payloads = {
        "authority/job.json": authority["job"][1],
        "authority/execution_authorization.json": authority["authorization"][1],
        "authority/activation_manifest.json": authority["activation"][1],
    }
    for name, expected_payload in expected_authority_payloads.items():
        if authority_payloads[name] != expected_payload:
            raise AppendAttemptArchiveError(
                f"Archived authority bytes do not match {name}."
            )

    attempt_receipt = _load_json_bytes(
        attempt_receipt_bytes, label="Append worker attempt receipt"
    )
    verify_self_digest(attempt_receipt, label="Append worker attempt receipt")
    attempt_ordinal = _require_int(
        attempt_receipt.get("attempt_ordinal"), label="attempt ordinal", minimum=1
    )
    attempt_cluster_id = _require_int(
        attempt_receipt.get("cluster_id"), label="attempt cluster ID"
    )
    attempt_proc_id = _require_int(
        attempt_receipt.get("proc_id"), label="attempt proc ID"
    )
    worker_exit_status = _require_int(
        attempt_receipt.get("worker_exit_status"),
        label="worker exit status",
    )
    if (
        attempt_receipt.get("schema") != ATTEMPT_SCHEMA
        or attempt_receipt.get("execution_id") != expected.execution_id
        or attempt_cluster_id != expected.cluster_id
        or attempt_proc_id != expected.proc_id
        or worker_exit_status != 0
        or attempt_receipt.get("job_file_sha256")
        != hashlib.sha256(authority["job"][1]).hexdigest()
        or attempt_receipt.get("authorization_file_sha256")
        != AUTHORIZATION_FILE_SHA256
        or attempt_receipt.get("activation_manifest_file_sha256")
        != ACTIVATION_FILE_SHA256
        or attempt_receipt.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
        or attempt_receipt.get("image_sha256") != IMAGE_SHA256
    ):
        raise AppendAttemptArchiveError("Worker attempt binding drifted.")
    declared_worker_files = _parse_inventory(
        attempt_receipt.get("worker_files"), label="worker inventory"
    )
    if declared_worker_files != observed_worker_files:
        raise AppendAttemptArchiveError(
            "Worker inventory hash/size closure drifted."
        )
    expected_member_names = {
        *(f"worker_outputs/{path}" for path in REQUIRED_WORKER_PATHS),
        *AUTHORITY_MEMBERS,
        ATTEMPT_RECEIPT_MEMBER,
    }
    if member_names != expected_member_names:
        raise AppendAttemptArchiveError("Archive member closure drifted.")

    expected_marker = (
        f"{expected.execution_id}\t{expected.cluster_id}\t"
        f"{expected.proc_id}\t{attempt_ordinal}\n"
    ).encode("utf-8")
    if captured_worker_files["attempt_identity.tsv"] != expected_marker:
        raise AppendAttemptArchiveError("Worker attempt marker drifted.")
    if captured_worker_files["worker_exit_status.txt"] != b"0\n":
        raise AppendAttemptArchiveError("Worker exit-status sidecar drifted.")

    output_validation = _validate_worker_payloads(
        expected=expected,
        authority=authority,
        observed_worker_files=observed_worker_files,
        captured_worker_files=captured_worker_files,
    )
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
        "attempt_receipt": {
            "schema": ATTEMPT_SCHEMA,
            "canonical_sha256": str(attempt_receipt["sha256"]),
            "file_sha256": hashlib.sha256(attempt_receipt_bytes).hexdigest(),
            "worker_exit_status": 0,
        },
        "worker_outputs": output_validation,
        "member_validation": {
            "gzip_and_full_tar_scan_passed": True,
            "safe_unique_regular_only_member_closure_passed": True,
            "worker_inventory_hash_size_closure_passed": True,
            "authority_byte_identity_passed": True,
            "worker_declared_fresh70_crosslink_checks_passed": True,
            "member_count": len(member_names),
            "worker_file_count": len(observed_worker_files),
            "total_member_payload_bytes": total_member_payload_bytes,
            "total_uncompressed_bytes": (
                decompressed.bytes_read if decompressed is not None else 0
            ),
            "activated_row_uncompressed_limit_bytes": (
                uncompressed_limit_bytes
            ),
        },
        "validation_scope": {
            "transport_integrity": "fully_streamed_and_hash_closed",
            "authority_provenance": "sealed_byte_identity_closed",
            "worker_inventory": "exact_hash_size_member_closure",
            "scientific_payload_semantics": (
                "worker_declared_crosslinks_checked_not_semantically_replayed"
            ),
            "scheduler_terminal_history": "not_validated",
        },
        "bindings": {
            "package_manifest": _binding(*authority["package"]),
            "job": _binding(*authority["job"]),
            "authorization": _binding(*authority["authorization"]),
            "activation_manifest": _binding(*authority["activation"]),
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "image_sha256": IMAGE_SHA256,
        },
    }


def _parse_utc(value: str, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AppendAttemptArchiveError(f"{label} must be UTC with a Z suffix.")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise AppendAttemptArchiveError(f"{label} is not RFC-3339.") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise AppendAttemptArchiveError(f"{label} is not UTC.")
    return parsed


def _quota_value(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AppendAttemptArchiveError(f"{label} is not numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise AppendAttemptArchiveError(f"{label} is invalid.")
    return result


def _load_submission_receipt(
    path: Path,
    *,
    expected: ExpectedAppendAttempt,
) -> tuple[Path, bytes, dict[str, Any]]:
    receipt_path, receipt_bytes, receipt = _load_json_file(
        path,
        label="Append submission receipt",
        expected_file_sha256=SUBMISSION_RECEIPT_FILE_SHA256,
        expected_canonical_sha256=SUBMISSION_RECEIPT_CANONICAL_SHA256,
    )
    bindings = receipt.get("bindings")
    activation_binding = (
        bindings.get("activation_manifest") if isinstance(bindings, Mapping) else None
    )
    authorization_binding = (
        bindings.get("execution_authorization")
        if isinstance(bindings, Mapping)
        else None
    )
    package_binding = (
        bindings.get("package_manifest") if isinstance(bindings, Mapping) else None
    )
    source_binding = (
        bindings.get("source_archive") if isinstance(bindings, Mapping) else None
    )
    lifecycle = receipt.get("lifecycle")
    initial_state = receipt.get("initial_state")
    proc_ids = (
        initial_state.get("proc_ids")
        if isinstance(initial_state, Mapping)
        else None
    )
    if (
        receipt.get("schema") != SUBMISSION_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("cluster_id") != CLUSTER_ID
        or receipt.get("activation_id") != ACTIVATION_ID
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("batch_name") != BATCH_NAME
        or receipt.get("owner") != REMOTE_OWNER
        or receipt.get("submit_host") != REMOTE_HOST
        or receipt.get("schedd") != REMOTE_SCHEDD
        or receipt.get("remote_root") != REMOTE_ROOT
        or not isinstance(activation_binding, Mapping)
        or activation_binding.get("sha256") != ACTIVATION_FILE_SHA256
        or activation_binding.get("canonical_sha256")
        != ACTIVATION_CANONICAL_SHA256
        or not isinstance(authorization_binding, Mapping)
        or authorization_binding.get("sha256") != AUTHORIZATION_FILE_SHA256
        or authorization_binding.get("canonical_sha256")
        != AUTHORIZATION_CANONICAL_SHA256
        or not isinstance(package_binding, Mapping)
        or package_binding.get("sha256") != PACKAGE_MANIFEST_FILE_SHA256
        or package_binding.get("canonical_sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or not isinstance(source_binding, Mapping)
        or source_binding.get("sha256") != SOURCE_ARCHIVE_SHA256
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("mode") != "ordinary_held_exact_proc_release_v1"
        or lifecycle.get("release_scope") != "exact_cluster_proc_only"
        or lifecycle.get("one_proc_per_quota_cycle") is not True
        or not isinstance(proc_ids, list)
        or expected.proc_id not in proc_ids
    ):
        raise AppendAttemptArchiveError(
            "Append submission receipt relation closure failed."
        )
    return receipt_path, receipt_bytes, receipt


def build_retrieval_receipt(
    *,
    validation: Mapping[str, Any],
    expected: ExpectedAppendAttempt,
    submission_receipt_path: Path,
    remote: RemoteRetrievalObservation,
) -> dict[str, Any]:
    """Build a receipt for archive identity/integrity, not science replay."""

    if (
        validation.get("status") != "passed"
        or validation.get("execution_id") != expected.execution_id
        or validation.get("cluster_id") != expected.cluster_id
        or validation.get("proc_id") != expected.proc_id
    ):
        raise AppendAttemptArchiveError(
            "Passing validation for the exact attempt is required."
        )
    archive_binding = validation.get("archive")
    if not isinstance(archive_binding, Mapping):
        raise AppendAttemptArchiveError("Validation archive binding is malformed.")
    timestamps = {
        "remote_identity": _parse_utc(
            remote.remote_identity_observed_utc,
            label="remote_identity_observed_utc",
        ),
        "retrieved": _parse_utc(remote.retrieved_utc, label="retrieved_utc"),
        "quota": _parse_utc(
            remote.quota_observed_utc, label="quota_observed_utc"
        ),
        "created": _parse_utc(
            remote.receipt_created_utc, label="receipt_created_utc"
        ),
    }
    if list(timestamps.values()) != sorted(timestamps.values()):
        raise AppendAttemptArchiveError("Retrieval timestamps are out of order.")
    remote_sha256 = _require_sha256(
        remote.remote_archive_sha256, label="remote archive SHA-256"
    )
    remote_size = _require_int(
        remote.remote_archive_size_bytes,
        label="remote archive size",
        minimum=1,
    )
    if (
        remote_sha256 != archive_binding.get("sha256")
        or remote_size != archive_binding.get("size_bytes")
    ):
        raise AppendAttemptArchiveError(
            "Remote and local archive hash/size bindings differ."
        )

    used = _quota_value(remote.quota_home_used_gib, label="quota home used GiB")
    soft = _quota_value(
        remote.quota_home_soft_limit_gib, label="quota home soft limit GiB"
    )
    hard = _quota_value(
        remote.quota_home_hard_limit_gib, label="quota home hard limit GiB"
    )
    if soft <= 0 or hard < soft or used > hard:
        raise AppendAttemptArchiveError("Quota limits are inconsistent.")
    soft_headroom = float(Decimal(str(soft)) - Decimal(str(used)))
    hard_headroom = float(Decimal(str(hard)) - Decimal(str(used)))
    submission_path, submission_bytes, submission = _load_submission_receipt(
        submission_receipt_path,
        expected=expected,
    )
    return digested(
        {
            "schema": RETRIEVAL_RECEIPT_SCHEMA,
            "receipt_created_utc": remote.receipt_created_utc,
            "status": "passed",
            "retrieval_classification": (
                "remote_local_identity_matched_authority_and_inventory_closed"
            ),
            "receipt_scope": {
                "transport_integrity_authenticated": True,
                "sealed_authority_provenance_authenticated": True,
                "worker_inventory_authenticated": True,
                "worker_exit_zero_declared": True,
                "scientific_payload_semantics": (
                    "worker_declared_not_semantically_replayed"
                ),
                "scheduler_terminal_history": "not_authenticated",
                "release_operation": "not_authenticated",
                "remote_cleanup_state": "not_authenticated",
            },
            "execution": {
                "execution_id": expected.execution_id,
                "cluster_id": expected.cluster_id,
                "proc_id": expected.proc_id,
                "attempt_ordinal": validation["attempt_ordinal"],
                "source_horizon": 50,
                "target_horizon": 70,
                "fresh_start": True,
                "resume_claimed": False,
            },
            "retrieval": {
                "remote_identity_observed_utc": (
                    remote.remote_identity_observed_utc
                ),
                "retrieved_utc": remote.retrieved_utc,
                "remote_archive_path": _expected_remote_archive_path(expected),
                "remote_archive_sha256": remote_sha256,
                "remote_archive_size_bytes": remote_size,
                "local_archive": dict(archive_binding),
                "remote_local_hash_size_match": True,
                "expected_final_basename_match": True,
            },
            "remote_identity": {
                "owner": REMOTE_OWNER,
                "host": REMOTE_HOST,
                "schedd": REMOTE_SCHEDD,
                "remote_root": REMOTE_ROOT,
            },
            "operator_supplied_quota_observation": {
                "observed_utc": remote.quota_observed_utc,
                "home_used_gib": used,
                "home_soft_limit_gib": soft,
                "home_hard_limit_gib": hard,
                "soft_limit_headroom_gib": soft_headroom,
                "hard_limit_headroom_gib": hard_headroom,
            },
            "bindings": {
                "submission_receipt": _binding(
                    submission_path, submission_bytes, submission
                ),
                **dict(validation["bindings"]),
            },
            "archive_validation": {
                "attempt_receipt": dict(validation["attempt_receipt"]),
                "worker_outputs": dict(validation["worker_outputs"]),
                "validation_scope": dict(validation["validation_scope"]),
                **dict(validation["member_validation"]),
            },
            "paper_evidence_adopted": False,
        }
    )


def write_new_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically create, but never replace, a retrieval receipt."""

    verify_self_digest(receipt, label="retrieval receipt")
    output = path.absolute()
    if output.exists() or output.is_symlink():
        raise AppendAttemptArchiveError(
            f"Refusing to overwrite existing receipt: {path}"
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise AppendAttemptArchiveError("Receipt parent is missing or unsafe.")
    payload = canonical_json_bytes(receipt) + b"\n"
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise AppendAttemptArchiveError(
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
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--submission-receipt", type=Path)
    parser.add_argument("--receipt-created-utc")
    parser.add_argument("--remote-identity-observed-utc")
    parser.add_argument("--retrieved-utc")
    parser.add_argument("--remote-archive-sha256")
    parser.add_argument("--remote-archive-size-bytes", type=int)
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


def _required_receipt_argument(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is None:
        raise AppendAttemptArchiveError(
            f"--{name.replace('_', '-')} is required with --receipt-output."
        )
    return value


def main() -> int:
    args = _parse_args()
    expected = ExpectedAppendAttempt(
        execution_id=args.execution_id,
        cluster_id=args.cluster_id,
        proc_id=args.proc_id,
        package_manifest_path=args.package_manifest,
        job_path=args.job,
        authorization_path=args.authorization,
        activation_manifest_path=args.activation_manifest,
    )
    validation = validate_append_attempt_archive(args.archive, expected)
    output: Mapping[str, Any] = validation
    if args.receipt_output is not None:
        remote = RemoteRetrievalObservation(
            receipt_created_utc=_required_receipt_argument(
                args, "receipt_created_utc"
            ),
            remote_identity_observed_utc=_required_receipt_argument(
                args, "remote_identity_observed_utc"
            ),
            retrieved_utc=_required_receipt_argument(args, "retrieved_utc"),
            remote_archive_sha256=_required_receipt_argument(
                args, "remote_archive_sha256"
            ),
            remote_archive_size_bytes=_required_receipt_argument(
                args, "remote_archive_size_bytes"
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
        receipt = build_retrieval_receipt(
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
