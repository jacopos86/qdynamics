#!/usr/bin/env python3
"""Authenticate one completed cluster-9400779 nph3-v3 attempt archive.

This helper is deliberately local-only.  It does not connect to CHTC and it
does not mutate scheduler state.  The compressed archive and every tar member
are hashed while streaming; giant result, checkpoint, and ledger payloads are
never extracted or retained in memory.  Only the small authority and receipt
sidecars needed for relation validation are captured.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
from typing import Any, BinaryIO, Mapping


BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]
PACKAGE_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_"
    "r50_20260802_v3_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_v3"
)
ACTIVATION_ID = f"{PACKAGE_ID}_activation_ordinary_v1"
BATCH_NAME = (
    "paper-i-ra-historical-mean-global-singleton-plateau3-nph3-r50-"
    "20260802-v3-finalizer-repair"
)
CLUSTER_ID = 9400779
TARGET_HORIZON = 50
SOURCE_ARCHIVE_SHA256 = (
    "7e7fa374f629ce684035d318176f354b24cfdf7cf4ac9548be921c790bf57d01"
)
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)

JOB_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "job_v2"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "execution_authorization_v2"
)
ACTIVATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "ordinary_activation_v2"
)
ATTEMPT_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "worker_attempt_v2"
)
WORKER_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "worker_receipt_v2"
)
EXECUTION_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "execution_manifest_v2"
)
SUBMISSION_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "submission_receipt_v1"
)
ARCHIVE_VALIDATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "attempt_archive_validation_v1"
)
RETRIEVAL_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "retrieval_completion_receipt_v1"
)

ACTIVATION_FILE_SHA256 = (
    "bc38c762c380dc75adea3e0d5f2b0cc3e995c2b8a401e25d03a361464496673e"
)
ACTIVATION_CANONICAL_SHA256 = (
    "97d6070ebbc2180051536c3bdfc1e979df60819e287c8e6449257e71fac906b1"
)
SUBMISSION_RECEIPT_FILE_SHA256 = (
    "7809539e40ab898a141b24b91f29c984c02ad937cff6388149260a07c9fd0de8"
)
PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "dd756ffa8fa0b1d9b21f906d2587a664ff49743f4eb80c4f1c787c0989cf4f23"
)
SUBMIT_DESCRIPTOR_SHA256 = (
    "aa518afcbb37c5f6f621884cdaeff13b678bc6960d1de2cd22672f44bf6d83ff"
)

PACKAGE_DIR = BASE / PACKAGE_ID
ACTIVATION_DIR = BASE / ACTIVATION_ID
DEFAULT_SUBMISSION_RECEIPT = BASE / f"{PACKAGE_ID}_runtime/submission_receipt.json"
AUTHORITY_MEMBERS = (
    "authority/job.json",
    "authority/execution_authorization.json",
    "authority/activation_manifest.json",
)
ATTEMPT_RECEIPT_MEMBER = "worker_attempt_receipt.json"
CAPTURED_WORKER_MEMBERS = frozenset(
    {
        "attempt_identity.tsv",
        "worker_exit_status.txt",
        "worker_receipt.json",
        "artifacts/execution_manifest.json",
    }
)
REQUIRED_WORKER_MEMBERS = frozenset(
    {
        "attempt_identity.tsv",
        "worker_exit_status.txt",
        "worker_receipt.json",
        "artifacts/checkpoint.json",
        "artifacts/estimator_ledger.json",
        "artifacts/execution_manifest.json",
        "artifacts/result.json",
    }
)
SMALL_MEMBER_LIMIT_BYTES = 16 * 1024 * 1024
STREAM_BLOCK_BYTES = 8 * 1024 * 1024
SHA256_HEX_LENGTH = 64


class Nph3AttemptArchiveError(ValueError):
    """Raised when a cluster-9400779 attempt cannot be authenticated."""


@dataclass(frozen=True)
class ExecutionAuthority:
    proc_id: int
    job_file_sha256: str
    job_canonical_sha256: str
    authorization_file_sha256: str
    authorization_canonical_sha256: str


EXECUTION_AUTHORITIES: Mapping[str, ExecutionAuthority] = {
    (
        "historical_mean_global_singleton_v3_nph3_r50__weak_weak__nph3__"
        "ra_global_singleton_plateau"
    ): ExecutionAuthority(
        proc_id=0,
        job_file_sha256=(
            "a655704ca3fcb302598e1e0104912423150e4c74aefcb1a49c56ac7bf38878ba"
        ),
        job_canonical_sha256=(
            "5f6183e329c1350f4a062e58e6a42367e9ee5e88c6a1a5d2e8aad95c814f8544"
        ),
        authorization_file_sha256=(
            "31df2a26ab39eb1d5e88c335aa54d625e80e94dbf4a0c5c67643767c0581ecba"
        ),
        authorization_canonical_sha256=(
            "5ee5eb9fdccf2d600a46783af510e919f41f2d722df653c6b62b248d0aeec5d9"
        ),
    ),
    (
        "historical_mean_global_singleton_v3_nph3_r50__intermediate_weak__"
        "nph3__ra_global_singleton_plateau"
    ): ExecutionAuthority(
        proc_id=1,
        job_file_sha256=(
            "1f4b76b8ebfe379d9b9809c96430938cb81db221d1fd2aecfe2453a696e7da1b"
        ),
        job_canonical_sha256=(
            "7b29a2372208e449d23ecce05c1134c345109af1d1e7e099eaf66f9be36f5798"
        ),
        authorization_file_sha256=(
            "6c8464e161da99e815cc84bf295e195ae7f26d2e2d56e116e1559bf923d175fd"
        ),
        authorization_canonical_sha256=(
            "7b091e82542c1c0043dbce230be29a322c4fae26e1bcdbac751d20ade23fa93d"
        ),
    ),
    (
        "historical_mean_global_singleton_v3_nph3_r50__strong_weak_u8__"
        "nph3__ra_global_singleton_plateau"
    ): ExecutionAuthority(
        proc_id=2,
        job_file_sha256=(
            "0caf0067664b798d25ed737d990017f9b598ea4fc463abf9cd88bd2a1065ee07"
        ),
        job_canonical_sha256=(
            "6a5193ee97ed5cd33f84348cae21a65b8657cd064cca82b2a8cebcf87f92d600"
        ),
        authorization_file_sha256=(
            "c47b019751cb8017f95d4fcb8860ec45f48f107871b7cc824e961c387b9990a2"
        ),
        authorization_canonical_sha256=(
            "0bc48a0ee74174456aa0d2f7abfe367712b62482f651fdbc2173f99b6823417c"
        ),
    ),
}


@dataclass(frozen=True)
class ExpectedAttempt:
    """Exact execution and proc identity for one cluster-9400779 result."""

    execution_id: str
    proc_id: int


@dataclass(frozen=True)
class RemoteArchiveObservation:
    """Remote hash/size metadata supplied after immutable retrieval."""

    receipt_created_utc: str
    remote_observed_utc: str
    retrieved_utc: str
    remote_host: str
    remote_archive_path: str
    remote_archive_sha256: str
    remote_archive_size_bytes: int


@dataclass(frozen=True)
class _JsonDocument:
    path: Path
    payload: bytes
    parsed: dict[str, Any]


class _HashingReader:
    """Count and hash every compressed byte read from a binary stream."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def read(self, size: int = -1) -> bytes:
        payload = self._stream.read(size)
        if payload:
            self._digest.update(payload)
            self.size_bytes += len(payload)
        return payload

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the canonical JSON representation used by this campaign."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a mapping and append its canonical SHA-256 self digest."""

    result = dict(payload)
    if "sha256" in result:
        raise Nph3AttemptArchiveError("Self-digest input already contains sha256.")
    result["sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    observed = payload.get("sha256")
    projection = dict(payload)
    projection.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(projection)).hexdigest()
    if observed != expected:
        raise Nph3AttemptArchiveError(f"{label} self digest drifted.")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if not _valid_sha256(value):
        raise Nph3AttemptArchiveError(
            f"{label} is not a lowercase SHA-256 digest."
        )
    return str(value)


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Nph3AttemptArchiveError(f"{label} is not a valid integer.")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise Nph3AttemptArchiveError(f"{label} field closure drifted.")


def _require_plain_file(path: Path, *, label: str) -> Path:
    candidate = path.absolute()
    try:
        details = candidate.lstat()
    except OSError as exc:
        raise Nph3AttemptArchiveError(f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(details.st_mode) or candidate.is_symlink():
        raise Nph3AttemptArchiveError(
            f"{label} is missing, not regular, or a symlink: {path}"
        )
    return candidate


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Nph3AttemptArchiveError(
                f"JSON object contains duplicate key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise Nph3AttemptArchiveError(f"JSON contains non-finite value: {value}")


def _load_json_bytes(
    payload: bytes,
    *,
    label: str,
    require_canonical_file: bool = False,
) -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Nph3AttemptArchiveError(f"Malformed {label} JSON.") from exc
    if not isinstance(parsed, dict):
        raise Nph3AttemptArchiveError(f"{label} is not a JSON object.")
    if require_canonical_file and payload != canonical_json_bytes(parsed) + b"\n":
        raise Nph3AttemptArchiveError(f"{label} file encoding drifted.")
    return parsed


def _load_exact_json(
    path: Path,
    *,
    label: str,
    file_sha256: str,
    canonical_sha256: str | None,
) -> _JsonDocument:
    candidate = _require_plain_file(path, label=label)
    if candidate.stat().st_size > SMALL_MEMBER_LIMIT_BYTES:
        raise Nph3AttemptArchiveError(f"{label} is unexpectedly large.")
    payload = candidate.read_bytes()
    if hashlib.sha256(payload).hexdigest() != file_sha256:
        raise Nph3AttemptArchiveError(f"{label} exact file digest drifted.")
    parsed = _load_json_bytes(payload, label=label)
    if canonical_sha256 is not None:
        verify_self_digest(parsed, label=label)
        if parsed.get("sha256") != canonical_sha256:
            raise Nph3AttemptArchiveError(f"{label} canonical digest drifted.")
    return _JsonDocument(candidate, payload, parsed)


def _safe_member_name(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise Nph3AttemptArchiveError(f"Unsafe tar member name: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
        or path.as_posix() != value
    ):
        raise Nph3AttemptArchiveError(f"Unsafe tar member name: {value!r}")
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
) -> tuple[dict[str, Any], bytes | None]:
    if capture and expected_size > SMALL_MEMBER_LIMIT_BYTES:
        raise Nph3AttemptArchiveError(f"{label} is unexpectedly large.")
    digest = hashlib.sha256()
    size = 0
    chunks: list[bytes] | None = [] if capture else None
    while block := stream.read(STREAM_BLOCK_BYTES):
        digest.update(block)
        size += len(block)
        if chunks is not None:
            chunks.append(block)
    if size != expected_size:
        raise Nph3AttemptArchiveError(f"{label} size differs from its tar header.")
    return (
        {"sha256": digest.hexdigest(), "size_bytes": size},
        b"".join(chunks) if chunks is not None else None,
    )


def _authority_paths(execution_id: str) -> tuple[Path, Path, Path]:
    return (
        PACKAGE_DIR / f"jobs/{execution_id}.json",
        ACTIVATION_DIR / f"authorizations/{execution_id}.json",
        ACTIVATION_DIR / "activation_manifest.json",
    )


def _validate_expected(expected: ExpectedAttempt) -> ExecutionAuthority:
    authority = EXECUTION_AUTHORITIES.get(expected.execution_id)
    if authority is None:
        raise Nph3AttemptArchiveError("Execution is outside the sealed nph3-v3 set.")
    _require_int(expected.proc_id, label="proc_id")
    if expected.proc_id != authority.proc_id:
        raise Nph3AttemptArchiveError(
            "Proc id does not match the sealed activation queue index."
        )
    return authority


def _load_authorities(
    expected: ExpectedAttempt,
    authority: ExecutionAuthority,
) -> tuple[_JsonDocument, _JsonDocument, _JsonDocument]:
    job_path, authorization_path, activation_path = _authority_paths(
        expected.execution_id
    )
    job = _load_exact_json(
        job_path,
        label="sealed job",
        file_sha256=authority.job_file_sha256,
        canonical_sha256=authority.job_canonical_sha256,
    )
    authorization = _load_exact_json(
        authorization_path,
        label="sealed execution authorization",
        file_sha256=authority.authorization_file_sha256,
        canonical_sha256=authority.authorization_canonical_sha256,
    )
    activation = _load_exact_json(
        activation_path,
        label="sealed activation manifest",
        file_sha256=ACTIVATION_FILE_SHA256,
        canonical_sha256=ACTIVATION_CANONICAL_SHA256,
    )
    return job, authorization, activation


def _single_matching_row(
    rows: Any,
    *,
    execution_id: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(rows, list):
        raise Nph3AttemptArchiveError(f"{label} is malformed.")
    matching = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("execution_id") == execution_id
    ]
    if len(matching) != 1:
        raise Nph3AttemptArchiveError(
            f"{label} does not contain exactly one expected execution."
        )
    return matching[0]


def _validate_authority_relations(
    *,
    expected: ExpectedAttempt,
    authority: ExecutionAuthority,
    job: _JsonDocument,
    authorization: _JsonDocument,
    activation: _JsonDocument,
) -> None:
    job_payload = job.parsed
    auth_payload = authorization.parsed
    activation_payload = activation.parsed
    if (
        job_payload.get("schema") != JOB_SCHEMA
        or job_payload.get("package_id") != PACKAGE_ID
        or job_payload.get("campaign_id") != CAMPAIGN_ID
        or job_payload.get("execution_id") != expected.execution_id
        or job_payload.get("execution_mode") != "fresh_0_to_50"
        or job_payload.get("target_horizon") != TARGET_HORIZON
        or job_payload.get("source_horizon") != TARGET_HORIZON
        or job_payload.get("nph") != 3
        or job_payload.get("fresh_start_contract")
        != {
            "kind": "fresh_start",
            "resume_archive": None,
            "source_checkpoint": None,
        }
        or job_payload.get("execution_authorized") is not False
        or job_payload.get("submission_authorized") is not False
        or job_payload.get("submitted") is not False
    ):
        raise Nph3AttemptArchiveError("Sealed job semantic closure drifted.")
    if (
        auth_payload.get("schema") != AUTHORIZATION_SCHEMA
        or auth_payload.get("package_id") != PACKAGE_ID
        or auth_payload.get("campaign_id") != CAMPAIGN_ID
        or auth_payload.get("activation_id") != ACTIVATION_ID
        or auth_payload.get("execution_id") != expected.execution_id
        or auth_payload.get("job_spec_sha256")
        != authority.job_canonical_sha256
        or auth_payload.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or auth_payload.get("remote_image_path") != IMAGE_PATH
        or auth_payload.get("remote_image_sha256") != IMAGE_SHA256
        or auth_payload.get("execution_authorized") is not True
        or auth_payload.get("submission_authorized") is not True
        or auth_payload.get("paper_evidence_adoption_authorized") is not False
    ):
        raise Nph3AttemptArchiveError(
            "Execution authorization semantic closure drifted."
        )
    remote_image = activation_payload.get("remote_image")
    sealed_package = activation_payload.get("sealed_package")
    source_binding = (
        sealed_package.get("source_archive")
        if isinstance(sealed_package, Mapping)
        else None
    )
    if (
        activation_payload.get("schema") != ACTIVATION_SCHEMA
        or activation_payload.get("activation_id") != ACTIVATION_ID
        or activation_payload.get("package_id") != PACKAGE_ID
        or activation_payload.get("campaign_id") != CAMPAIGN_ID
        or activation_payload.get("batch_name") != BATCH_NAME
        or activation_payload.get("direct_execution_count") != 3
        or activation_payload.get("execution_authorized") is not True
        or activation_payload.get("submission_authorized") is not True
        or activation_payload.get("submitted") is not False
        or not isinstance(remote_image, Mapping)
        or remote_image.get("path") != IMAGE_PATH
        or remote_image.get("sha256") != IMAGE_SHA256
        or not isinstance(source_binding, Mapping)
        or source_binding.get("sha256") != SOURCE_ARCHIVE_SHA256
        or auth_payload.get("activation_control_plane_sha256")
        != activation_payload.get("activation_control_plane_sha256")
    ):
        raise Nph3AttemptArchiveError("Activation semantic closure drifted.")

    execution_row = _single_matching_row(
        activation_payload.get("executions"),
        execution_id=expected.execution_id,
        label="Activation execution rows",
    )
    job_binding = execution_row.get("job")
    if (
        execution_row.get("queue_index") != expected.proc_id
        or not isinstance(job_binding, Mapping)
        or job_binding.get("sha256") != authority.job_file_sha256
        or job_binding.get("canonical_sha256")
        != authority.job_canonical_sha256
        or job_binding.get("size_bytes") != len(job.payload)
    ):
        raise Nph3AttemptArchiveError("Activation job binding drifted.")
    authorization_row = _single_matching_row(
        activation_payload.get("execution_authorizations"),
        execution_id=expected.execution_id,
        label="Activation authorization rows",
    )
    if (
        authorization_row.get("sha256") != authority.authorization_file_sha256
        or authorization_row.get("canonical_sha256")
        != authority.authorization_canonical_sha256
        or authorization_row.get("size_bytes") != len(authorization.payload)
    ):
        raise Nph3AttemptArchiveError("Activation authorization binding drifted.")


def _inventory_from_rows(rows: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise Nph3AttemptArchiveError(f"{label} is malformed.")
    inventory: dict[str, dict[str, Any]] = {}
    ordered_paths: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Nph3AttemptArchiveError(f"{label} row {index} is malformed.")
        _require_exact_keys(
            row,
            {"path", "sha256", "size_bytes"},
            label=f"{label} row {index}",
        )
        raw_path = row.get("path")
        if not isinstance(raw_path, str):
            raise Nph3AttemptArchiveError(f"{label} row {index} has no path.")
        path = _safe_member_name(raw_path)
        if path in inventory:
            raise Nph3AttemptArchiveError(f"Duplicate {label} path: {path}")
        ordered_paths.append(path)
        inventory[path] = {
            "sha256": _require_sha256(
                row.get("sha256"), label=f"{label} SHA-256 for {path}"
            ),
            "size_bytes": _require_int(
                row.get("size_bytes"), label=f"{label} size for {path}"
            ),
        }
    if ordered_paths != sorted(ordered_paths):
        raise Nph3AttemptArchiveError(f"{label} path ordering drifted.")
    return inventory


def _inventory_from_mapping(
    rows: Any, *, label: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, Mapping):
        raise Nph3AttemptArchiveError(f"{label} is malformed.")
    inventory: dict[str, dict[str, Any]] = {}
    for raw_path, raw_binding in rows.items():
        if not isinstance(raw_path, str) or not isinstance(raw_binding, Mapping):
            raise Nph3AttemptArchiveError(f"{label} row is malformed.")
        path = _safe_member_name(raw_path)
        if PurePosixPath(path).name != path:
            raise Nph3AttemptArchiveError(f"{label} path is not a flat artifact.")
        _require_exact_keys(
            raw_binding,
            {"sha256", "size_bytes"},
            label=f"{label} binding for {path}",
        )
        inventory[path] = {
            "sha256": _require_sha256(
                raw_binding.get("sha256"), label=f"{label} SHA-256 for {path}"
            ),
            "size_bytes": _require_int(
                raw_binding.get("size_bytes"), label=f"{label} size for {path}"
            ),
        }
    return inventory


def _scan_archive(
    archive_path: Path,
) -> tuple[
    dict[str, Any],
    set[str],
    dict[str, dict[str, Any]],
    dict[str, bytes],
    dict[str, bytes],
    bytes,
]:
    archive_file = _require_plain_file(archive_path, label="fetched archive")
    initial_stat = archive_file.stat()
    member_names: set[str] = set()
    observed_worker_files: dict[str, dict[str, Any]] = {}
    captured_worker_files: dict[str, bytes] = {}
    authority_payloads: dict[str, bytes] = {}
    attempt_receipt_bytes: bytes | None = None
    try:
        with archive_file.open("rb", buffering=0) as raw:
            tracker = _HashingReader(raw)
            with gzip.GzipFile(fileobj=tracker, mode="rb") as decompressed:
                with tarfile.open(fileobj=decompressed, mode="r|") as archive:
                    for member in archive:
                        member_name = _safe_member_name(member.name)
                        if member_name in member_names:
                            raise Nph3AttemptArchiveError(
                                f"Duplicate tar member: {member_name}"
                            )
                        member_names.add(member_name)
                        if member.type not in {
                            tarfile.REGTYPE,
                            tarfile.AREGTYPE,
                        }:
                            raise Nph3AttemptArchiveError(
                                f"Non-regular tar member: {member_name}"
                            )
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise Nph3AttemptArchiveError(
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
                            or worker_relative in CAPTURED_WORKER_MEMBERS
                        )
                        binding, payload = _consume_member(
                            stream,
                            expected_size=member.size,
                            capture=capture,
                            label=f"tar member {member_name}",
                        )
                        if is_worker:
                            if not worker_relative:
                                raise Nph3AttemptArchiveError(
                                    "Empty worker member path."
                                )
                            _safe_member_name(worker_relative)
                            observed_worker_files[worker_relative] = binding
                            if payload is not None:
                                captured_worker_files[worker_relative] = payload
                        elif member_name in AUTHORITY_MEMBERS:
                            assert payload is not None
                            authority_payloads[member_name] = payload
                        elif member_name == ATTEMPT_RECEIPT_MEMBER:
                            assert payload is not None
                            attempt_receipt_bytes = payload
                        else:
                            raise Nph3AttemptArchiveError(
                                f"Unexpected archive member: {member_name}"
                            )
                while trailing := decompressed.read(STREAM_BLOCK_BYTES):
                    if trailing.strip(b"\0"):
                        raise Nph3AttemptArchiveError(
                            "Archive contains non-zero trailing tar payload."
                        )
            while tracker.read(STREAM_BLOCK_BYTES):
                pass
            archive_binding = {
                "path": _display_path(archive_file),
                "sha256": tracker.sha256,
                "size_bytes": tracker.size_bytes,
            }
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as exc:
        raise Nph3AttemptArchiveError(
            f"Fetched archive is not a complete gzip tar: {archive_path}"
        ) from exc
    final_stat = archive_file.stat()
    if (
        initial_stat.st_dev != final_stat.st_dev
        or initial_stat.st_ino != final_stat.st_ino
        or initial_stat.st_size != final_stat.st_size
        or initial_stat.st_mtime_ns != final_stat.st_mtime_ns
        or archive_binding["size_bytes"] != final_stat.st_size
    ):
        raise Nph3AttemptArchiveError("Fetched archive changed during validation.")
    if attempt_receipt_bytes is None:
        raise Nph3AttemptArchiveError("Worker attempt receipt is missing.")
    return (
        archive_binding,
        member_names,
        observed_worker_files,
        captured_worker_files,
        authority_payloads,
        attempt_receipt_bytes,
    )


def _binding(document: _JsonDocument) -> dict[str, Any]:
    return {
        "path": _display_path(document.path),
        "sha256": hashlib.sha256(document.payload).hexdigest(),
        "canonical_sha256": str(document.parsed["sha256"]),
        "size_bytes": len(document.payload),
    }


def validate_attempt_archive(
    archive_path: Path,
    expected: ExpectedAttempt,
) -> dict[str, Any]:
    """Stream-authenticate one completed 50-round worker archive."""

    authority = _validate_expected(expected)
    job, authorization, activation = _load_authorities(expected, authority)
    _validate_authority_relations(
        expected=expected,
        authority=authority,
        job=job,
        authorization=authorization,
        activation=activation,
    )
    (
        archive_binding,
        member_names,
        observed_worker_files,
        captured_worker_files,
        authority_payloads,
        attempt_receipt_bytes,
    ) = _scan_archive(archive_path)

    expected_authority_payloads = {
        "authority/job.json": job.payload,
        "authority/execution_authorization.json": authorization.payload,
        "authority/activation_manifest.json": activation.payload,
    }
    if set(authority_payloads) != set(expected_authority_payloads):
        raise Nph3AttemptArchiveError("Archive authority closure is incomplete.")
    for name, expected_payload in expected_authority_payloads.items():
        if authority_payloads[name] != expected_payload:
            raise Nph3AttemptArchiveError(
                f"Archived authority bytes do not match {name}."
            )

    attempt_receipt = _load_json_bytes(
        attempt_receipt_bytes,
        label="worker attempt receipt",
        require_canonical_file=True,
    )
    _require_exact_keys(
        attempt_receipt,
        {
            "schema",
            "execution_id",
            "cluster_id",
            "proc_id",
            "attempt_ordinal",
            "worker_exit_status",
            "job_file_sha256",
            "authorization_file_sha256",
            "activation_manifest_file_sha256",
            "source_archive_sha256",
            "image_sha256",
            "worker_files",
            "sha256",
        },
        label="worker attempt receipt",
    )
    verify_self_digest(attempt_receipt, label="worker attempt receipt")
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
        attempt_receipt.get("schema") != ATTEMPT_SCHEMA
        or attempt_receipt.get("execution_id") != expected.execution_id
        or attempt_receipt.get("cluster_id") != CLUSTER_ID
        or attempt_receipt.get("proc_id") != expected.proc_id
    ):
        raise Nph3AttemptArchiveError(
            "Worker attempt execution/cluster/proc identity drifted."
        )
    if worker_exit_status != 0:
        raise Nph3AttemptArchiveError("Worker did not exit successfully.")
    if (
        attempt_receipt.get("job_file_sha256") != authority.job_file_sha256
        or attempt_receipt.get("authorization_file_sha256")
        != authority.authorization_file_sha256
        or attempt_receipt.get("activation_manifest_file_sha256")
        != ACTIVATION_FILE_SHA256
        or attempt_receipt.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
        or attempt_receipt.get("image_sha256") != IMAGE_SHA256
    ):
        raise Nph3AttemptArchiveError(
            "Worker attempt authority/source/image binding drifted."
        )

    declared_worker_files = _inventory_from_rows(
        attempt_receipt.get("worker_files"), label="worker inventory"
    )
    if set(declared_worker_files) != set(observed_worker_files):
        raise Nph3AttemptArchiveError(
            "Worker inventory does not close over every archived worker file."
        )
    if not REQUIRED_WORKER_MEMBERS.issubset(declared_worker_files):
        raise Nph3AttemptArchiveError(
            "Worker inventory is missing a required completion artifact."
        )
    for path, declared in declared_worker_files.items():
        if declared != observed_worker_files[path]:
            raise Nph3AttemptArchiveError(
                f"Worker inventory hash/size mismatch: {path}"
            )
    expected_member_names = {
        f"worker_outputs/{path}" for path in declared_worker_files
    } | set(AUTHORITY_MEMBERS) | {ATTEMPT_RECEIPT_MEMBER}
    if member_names != expected_member_names:
        raise Nph3AttemptArchiveError("Archive member closure drifted.")

    expected_marker = (
        f"{expected.execution_id}\t{CLUSTER_ID}\t{expected.proc_id}\t"
        f"{attempt_ordinal}\n"
    ).encode("utf-8")
    if captured_worker_files.get("attempt_identity.tsv") != expected_marker:
        raise Nph3AttemptArchiveError("Worker attempt marker drifted.")
    if captured_worker_files.get("worker_exit_status.txt") != b"0\n":
        raise Nph3AttemptArchiveError("Worker exit-status sidecar drifted.")

    execution_manifest_bytes = captured_worker_files.get(
        "artifacts/execution_manifest.json"
    )
    worker_receipt_bytes = captured_worker_files.get("worker_receipt.json")
    if execution_manifest_bytes is None or worker_receipt_bytes is None:
        raise Nph3AttemptArchiveError("Worker completion sidecars are missing.")
    execution_manifest = _load_json_bytes(
        execution_manifest_bytes,
        label="execution manifest",
        require_canonical_file=True,
    )
    worker_receipt = _load_json_bytes(
        worker_receipt_bytes,
        label="worker receipt",
        require_canonical_file=True,
    )
    _require_exact_keys(
        execution_manifest,
        {
            "schema",
            "status",
            "package_id",
            "campaign_id",
            "execution_id",
            "job_spec_sha256",
            "authorization_sha256",
            "protocol_sha256",
            "target_horizon",
            "controller_rounds_completed",
            "fresh_start",
            "source_checkpoint_consumed",
            "output_payloads",
            "sha256",
        },
        label="execution manifest",
    )
    _require_exact_keys(
        worker_receipt,
        {
            "schema",
            "status",
            "package_id",
            "campaign_id",
            "execution_id",
            "job_spec_sha256",
            "authorization_sha256",
            "execution_manifest_sha256",
            "controller_rounds_completed",
            "fresh_start",
            "artifacts",
            "sha256",
        },
        label="worker receipt",
    )
    verify_self_digest(execution_manifest, label="execution manifest")
    verify_self_digest(worker_receipt, label="worker receipt")
    if (
        execution_manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
        or execution_manifest.get("status") != "passed"
        or execution_manifest.get("package_id") != PACKAGE_ID
        or execution_manifest.get("campaign_id") != CAMPAIGN_ID
        or execution_manifest.get("execution_id") != expected.execution_id
        or execution_manifest.get("job_spec_sha256")
        != authority.job_canonical_sha256
        or execution_manifest.get("authorization_sha256")
        != authority.authorization_canonical_sha256
        or execution_manifest.get("protocol_sha256")
        != job.parsed.get("protocol_sha256")
        or execution_manifest.get("target_horizon") != TARGET_HORIZON
        or execution_manifest.get("controller_rounds_completed")
        != TARGET_HORIZON
        or execution_manifest.get("fresh_start") is not True
        or execution_manifest.get("source_checkpoint_consumed") is not False
    ):
        raise Nph3AttemptArchiveError(
            "Execution manifest does not prove 50-round fresh success."
        )
    if (
        worker_receipt.get("schema") != WORKER_RECEIPT_SCHEMA
        or worker_receipt.get("status") != "passed"
        or worker_receipt.get("package_id") != PACKAGE_ID
        or worker_receipt.get("campaign_id") != CAMPAIGN_ID
        or worker_receipt.get("execution_id") != expected.execution_id
        or worker_receipt.get("job_spec_sha256")
        != authority.job_canonical_sha256
        or worker_receipt.get("authorization_sha256")
        != authority.authorization_canonical_sha256
        or worker_receipt.get("execution_manifest_sha256")
        != execution_manifest.get("sha256")
        or worker_receipt.get("controller_rounds_completed") != TARGET_HORIZON
        or worker_receipt.get("fresh_start") is not True
    ):
        raise Nph3AttemptArchiveError(
            "Worker receipt does not prove 50-round fresh success."
        )

    observed_artifacts = {
        path.removeprefix("artifacts/"): binding
        for path, binding in observed_worker_files.items()
        if path.startswith("artifacts/")
    }
    if any("/" in path for path in observed_artifacts):
        raise Nph3AttemptArchiveError("Nested worker artifact path drifted.")
    receipt_artifacts = _inventory_from_rows(
        worker_receipt.get("artifacts"), label="worker receipt artifact inventory"
    )
    if receipt_artifacts != observed_artifacts:
        raise Nph3AttemptArchiveError(
            "Worker receipt artifact hash/size closure failed."
        )
    manifest_outputs = _inventory_from_mapping(
        execution_manifest.get("output_payloads"),
        label="execution manifest output inventory",
    )
    expected_manifest_outputs = dict(observed_artifacts)
    expected_manifest_outputs.pop("execution_manifest.json", None)
    if manifest_outputs != expected_manifest_outputs:
        raise Nph3AttemptArchiveError(
            "Execution manifest output hash/size closure failed."
        )
    if (
        observed_artifacts["execution_manifest.json"]["sha256"]
        != hashlib.sha256(execution_manifest_bytes).hexdigest()
    ):
        raise Nph3AttemptArchiveError("Execution manifest file binding drifted.")

    return digested(
        {
            "schema": ARCHIVE_VALIDATION_SCHEMA,
            "status": "passed",
            "execution_id": expected.execution_id,
            "cluster_id": CLUSTER_ID,
            "proc_id": expected.proc_id,
            "attempt_ordinal": attempt_ordinal,
            "controller_rounds_completed": TARGET_HORIZON,
            "archive": archive_binding,
            "worker_attempt_receipt": {
                "schema": ATTEMPT_SCHEMA,
                "canonical_sha256": attempt_receipt["sha256"],
                "file_sha256": hashlib.sha256(attempt_receipt_bytes).hexdigest(),
                "worker_exit_status": worker_exit_status,
            },
            "worker_receipt": {
                "schema": WORKER_RECEIPT_SCHEMA,
                "canonical_sha256": worker_receipt["sha256"],
                "file_sha256": hashlib.sha256(worker_receipt_bytes).hexdigest(),
                "controller_rounds_completed": TARGET_HORIZON,
            },
            "execution_manifest": {
                "schema": EXECUTION_MANIFEST_SCHEMA,
                "canonical_sha256": execution_manifest["sha256"],
                "file_sha256": hashlib.sha256(
                    execution_manifest_bytes
                ).hexdigest(),
                "controller_rounds_completed": TARGET_HORIZON,
            },
            "member_validation": {
                "gzip_and_full_tar_scan_passed": True,
                "compressed_hash_size_stream_closure_passed": True,
                "safe_unique_regular_only_member_closure_passed": True,
                "worker_inventory_hash_size_closure_passed": True,
                "nested_artifact_inventory_closure_passed": True,
                "authority_byte_identity_passed": True,
                "fifty_round_success_closure_passed": True,
                "member_count": len(member_names),
                "worker_file_count": len(declared_worker_files),
            },
            "bindings": {
                "job": _binding(job),
                "authorization": _binding(authorization),
                "activation_manifest": _binding(activation),
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "image_sha256": IMAGE_SHA256,
            },
        }
    )


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Nph3AttemptArchiveError(f"{label} must be UTC with a Z suffix.")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise Nph3AttemptArchiveError(f"{label} is not RFC-3339.") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise Nph3AttemptArchiveError(f"{label} is not UTC.")
    return parsed


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise Nph3AttemptArchiveError(f"{label} is empty or unsafe.")
    return value


def _load_submission_receipt(path: Path) -> _JsonDocument:
    receipt = _load_exact_json(
        path,
        label="cluster-9400779 submission receipt",
        file_sha256=SUBMISSION_RECEIPT_FILE_SHA256,
        canonical_sha256=None,
    )
    payload = receipt.parsed
    package = payload.get("package")
    activation = payload.get("activation")
    remote_image = payload.get("remote_image")
    preflight = payload.get("preflight")
    if (
        payload.get("schema") != SUBMISSION_RECEIPT_SCHEMA
        or payload.get("cluster_id") != CLUSTER_ID
        or payload.get("batch_name") != BATCH_NAME
        or payload.get("direct_job_count") != 3
        or not isinstance(package, Mapping)
        or package.get("id") != PACKAGE_ID
        or package.get("manifest_canonical_sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or package.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or not isinstance(activation, Mapping)
        or activation.get("id") != ACTIVATION_ID
        or activation.get("manifest_canonical_sha256")
        != ACTIVATION_CANONICAL_SHA256
        or activation.get("submit_descriptor_sha256")
        != SUBMIT_DESCRIPTOR_SHA256
        or not isinstance(remote_image, Mapping)
        or remote_image.get("path") != IMAGE_PATH
        or remote_image.get("sha256") != IMAGE_SHA256
        or not isinstance(preflight, Mapping)
        or preflight.get("package_validation") != "passed"
        or preflight.get("activation_validation") != "passed"
        or preflight.get("submit_lifecycle_validation") != "passed"
        or preflight.get("condor_dry_run_job_count") != 3
        or preflight.get("batch_collision_count") != 0
        or preflight.get("fetched_output_collision_count") != 0
        or payload.get("paper_evidence_adopted") is not False
    ):
        raise Nph3AttemptArchiveError("Submission receipt authority drifted.")
    return receipt


def build_retrieval_receipt(
    *,
    validation: Mapping[str, Any],
    expected: ExpectedAttempt,
    remote: RemoteArchiveObservation,
    submission_receipt_path: Path = DEFAULT_SUBMISSION_RECEIPT,
) -> dict[str, Any]:
    """Bind remote hash/size observations into an immutable local receipt."""

    verify_self_digest(validation, label="archive validation")
    if (
        validation.get("schema") != ARCHIVE_VALIDATION_SCHEMA
        or validation.get("status") != "passed"
        or validation.get("execution_id") != expected.execution_id
        or validation.get("cluster_id") != CLUSTER_ID
        or validation.get("proc_id") != expected.proc_id
        or validation.get("controller_rounds_completed") != TARGET_HORIZON
    ):
        raise Nph3AttemptArchiveError(
            "Archive validation does not match the expected 50-round attempt."
        )
    _validate_expected(expected)
    archive_binding = validation.get("archive")
    if not isinstance(archive_binding, Mapping):
        raise Nph3AttemptArchiveError("Validation archive binding is malformed.")

    created = _parse_utc(remote.receipt_created_utc, label="receipt_created_utc")
    observed = _parse_utc(remote.remote_observed_utc, label="remote_observed_utc")
    retrieved = _parse_utc(remote.retrieved_utc, label="retrieved_utc")
    if observed > retrieved or retrieved > created:
        raise Nph3AttemptArchiveError("Retrieval timestamps are out of order.")
    remote_host = _nonempty(remote.remote_host, label="remote host")
    remote_path = _nonempty(remote.remote_archive_path, label="remote archive path")
    pure_remote_path = PurePosixPath(remote_path)
    expected_name = (
        f"{expected.execution_id}__{CLUSTER_ID}__{expected.proc_id}.tar.gz"
    )
    if (
        not pure_remote_path.is_absolute()
        or ".." in pure_remote_path.parts
        or pure_remote_path.name != expected_name
    ):
        raise Nph3AttemptArchiveError("Remote archive path identity drifted.")
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
        raise Nph3AttemptArchiveError(
            "Remote and local archive hash/size bindings differ."
        )

    submission = _load_submission_receipt(submission_receipt_path)
    return digested(
        {
            "schema": RETRIEVAL_RECEIPT_SCHEMA,
            "receipt_created_utc": remote.receipt_created_utc,
            "status": "passed",
            "completion_classification": (
                "worker_exit_zero_50_round_archive_fully_authenticated"
            ),
            "campaign_id": CAMPAIGN_ID,
            "package_id": PACKAGE_ID,
            "activation_id": ACTIVATION_ID,
            "execution": {
                "execution_id": expected.execution_id,
                "cluster_id": CLUSTER_ID,
                "proc_id": expected.proc_id,
                "attempt_ordinal": validation["attempt_ordinal"],
                "controller_rounds_completed": TARGET_HORIZON,
            },
            "retrieval": {
                "remote_observed_utc": remote.remote_observed_utc,
                "retrieved_utc": remote.retrieved_utc,
                "remote_host": remote_host,
                "remote_archive_path": remote_path,
                "remote_archive_sha256": remote_sha256,
                "remote_archive_size_bytes": remote_size,
                "local_archive": dict(archive_binding),
                "remote_local_hash_size_match": True,
            },
            "bindings": {
                "submission_receipt": {
                    "path": _display_path(submission.path),
                    "sha256": SUBMISSION_RECEIPT_FILE_SHA256,
                    "size_bytes": len(submission.payload),
                    "schema": SUBMISSION_RECEIPT_SCHEMA,
                },
                **dict(validation["bindings"]),
            },
            "archive_validation": {
                "canonical_sha256": validation["sha256"],
                "worker_attempt_receipt": dict(
                    validation["worker_attempt_receipt"]
                ),
                "worker_receipt": dict(validation["worker_receipt"]),
                "execution_manifest": dict(validation["execution_manifest"]),
                **dict(validation["member_validation"]),
            },
            "paper_evidence_adopted": False,
        }
    )


def write_new_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically create, but never replace, one retrieval receipt."""

    verify_self_digest(receipt, label="retrieval receipt")
    if receipt.get("schema") != RETRIEVAL_RECEIPT_SCHEMA:
        raise Nph3AttemptArchiveError("Retrieval receipt schema drifted.")
    output = path.absolute()
    if output.exists() or output.is_symlink():
        raise Nph3AttemptArchiveError(
            f"Refusing to overwrite existing receipt: {path}"
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise Nph3AttemptArchiveError("Receipt parent is missing or unsafe.")
    payload = canonical_json_bytes(receipt) + b"\n"
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise Nph3AttemptArchiveError(
            f"Receipt temporary path already exists: {temporary}"
        )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise Nph3AttemptArchiveError(
                f"Refusing to overwrite existing receipt: {path}"
            ) from exc
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--execution-id", choices=tuple(EXECUTION_AUTHORITIES), required=True
    )
    parser.add_argument("--proc-id", type=int, required=True)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument(
        "--submission-receipt", type=Path, default=DEFAULT_SUBMISSION_RECEIPT
    )
    parser.add_argument("--receipt-created-utc")
    parser.add_argument("--remote-observed-utc")
    parser.add_argument("--retrieved-utc")
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-archive-path")
    parser.add_argument("--remote-archive-sha256")
    parser.add_argument("--remote-archive-size-bytes", type=int)
    return parser.parse_args()


def _required_receipt_argument(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is None:
        raise Nph3AttemptArchiveError(
            f"--{name.replace('_', '-')} is required with --receipt-output."
        )
    return value


def main() -> int:
    args = _parse_args()
    expected = ExpectedAttempt(args.execution_id, args.proc_id)
    validation = validate_attempt_archive(args.archive, expected)
    output: Mapping[str, Any] = validation
    if args.receipt_output is not None:
        remote = RemoteArchiveObservation(
            receipt_created_utc=_required_receipt_argument(
                args, "receipt_created_utc"
            ),
            remote_observed_utc=_required_receipt_argument(
                args, "remote_observed_utc"
            ),
            retrieved_utc=_required_receipt_argument(args, "retrieved_utc"),
            remote_host=_required_receipt_argument(args, "remote_host"),
            remote_archive_path=_required_receipt_argument(
                args, "remote_archive_path"
            ),
            remote_archive_sha256=_required_receipt_argument(
                args, "remote_archive_sha256"
            ),
            remote_archive_size_bytes=_required_receipt_argument(
                args, "remote_archive_size_bytes"
            ),
        )
        receipt = build_retrieval_receipt(
            validation=validation,
            expected=expected,
            remote=remote,
            submission_receipt_path=args.submission_receipt,
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
