"""Offline acquisition coordination for completed prospective seed jobs.

The coordinator consumes already-captured scheduler evidence and durable local
submission evidence.  It does not invoke Condor, SSH, Duo, or a submission
command.  Its state-changing boundary is intentionally narrow: after another
component has packaged and guarded-fetched one immutable remote archive, this
module validates and safely extracts the exact five-file seed payload and calls
the lane's existing typed seed-ingestion function.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Callable, Mapping, Sequence

from pipelines.time_dynamics.guarded_archive_transfer import (
    ArchiveTransferSpec,
    DEFAULT_FREE_FLOOR_BYTES,
    TransferRecoveryError,
    enforce_local_storage_floor,
    read_finalization_receipt,
    verify_final_archive,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HH,
    LANE_HUBBARD,
    SourceEpochIdentity,
)
from pipelines.time_dynamics.combined_scheduler_snapshot import (
    CombinedSnapshotError,
    validate_combined_snapshot,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    ProspectiveExecutionError,
    SourceEpochReceipt,
)
from pipelines.time_dynamics.prospective_hh_seed_ingestion import (
    HHSeedIngestionError,
    HH_SEED_INCREMENT_RECEIPT_V1,
    PRESERVATION_MARKER,
    RAW_FILENAMES,
    ingest_hh_seed_delta,
)
from pipelines.time_dynamics.prospective_hh_seed_worker import (
    validate_canonical_seed_request as validate_hh_seed_request,
)
from pipelines.time_dynamics.prospective_hubbard_seed_ingestion import (
    HUBBARD_SEED_INCREMENT_RECEIPT_V1,
    HubbardSeedIngestionError,
    ingest_hubbard_seed_delta,
)
from pipelines.time_dynamics.prospective_hubbard_seed_recovery import (
    load_hubbard_durable_submission_intent,
)
from pipelines.time_dynamics.prospective_hubbard_seed_worker import (
    validate_canonical_seed_request as validate_hubbard_seed_request,
)
from pipelines.time_dynamics.prospective_submission_recovery import (
    EXPECTED_HH_SEED_SLUGS,
    PROVISIONAL_SUBMISSION_RESULT_V1,
    SubmissionRecoveryError,
    load_durable_submission_intent,
)


PLAN_V1 = "paper_ii_prospective_seed_delta_acquisition_plan_v1"
PLAN_ITEM_V1 = "paper_ii_prospective_seed_delta_plan_item_v1"
ARCHIVE_IDENTITY_V1 = "paper_ii_seed_remote_archive_identity_v1"
ACQUISITION_RECEIPT_V1 = "paper_ii_seed_delta_acquisition_receipt_v1"
REMOTE_PACKAGING_RECIPE_V1 = "paper_ii_seed_remote_packaging_recipe_v1"
LOCAL_TRANSFER_RECIPE_V1 = "paper_ii_seed_guarded_transfer_recipe_v1"
SCHEDULER_DISCOVERY_V1 = "paper_ii_seed_scheduler_delta_discovery_v1"

_SUBMISSION_RECEIPT_V1 = "time_dynamics_submission_receipt_v1"
_PACKAGE_MANIFEST_V1 = "paper_ii_bootstrap_package_manifest_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_REMOTE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
_EXPECTED_INVENTORY = tuple(sorted(RAW_FILENAMES))
_SUCCESS_STATUS = 4
_HELD_STATUS = 5
_NONTERMINAL_STATUSES = frozenset({1, 2, 6, 7})
_TERMINAL_FAILURE_STATUSES = frozenset({3})
_RAM_GUARD_ENV = "PAPER_II_SEED_DELTA_RAM_GUARD_LIMIT_MB"


class SeedDeltaAcquisitionError(RuntimeError):
    """A scheduler delta or verified archive cannot be admitted safely."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, *, label: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise SeedDeltaAcquisitionError(f"{label} must be a lowercase SHA-256")
    return text


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError(f"{label} must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or result < minimum:
        raise SeedDeltaAcquisitionError(f"{label} must be >= {minimum}")
    return result


def _relative_path(value: Any, *, label: str) -> str:
    text = str(value or "")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise SeedDeltaAcquisitionError(
            f"{label} must be a canonical repository-relative path"
        )
    return text


def _absolute_remote(value: Any, *, label: str) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts or str(path) != text:
        raise SeedDeltaAcquisitionError(
            f"{label} must be a canonical absolute POSIX path"
        )
    return text


def _repo_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        lexical = candidate.absolute()
    else:
        if ".." in candidate.parts:
            raise SeedDeltaAcquisitionError(f"{label} contains path traversal")
        lexical = (root / candidate).absolute()
    try:
        relative = lexical.relative_to(root.absolute())
    except ValueError as exc:
        raise SeedDeltaAcquisitionError(
            f"{label} must remain inside repository root"
        ) from exc
    cursor = root.absolute()
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise SeedDeltaAcquisitionError(f"{label} traverses a symlink")
    try:
        resolved = lexical.resolve(strict=must_exist)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SeedDeltaAcquisitionError(
            f"{label} must remain inside repository root"
        ) from exc
    return resolved


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise SeedDeltaAcquisitionError(f"{label} is not a nofollow file")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except SeedDeltaAcquisitionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SeedDeltaAcquisitionError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise SeedDeltaAcquisitionError(f"{label} must be a JSON object")
    return dict(payload)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


@dataclass(frozen=True, order=True)
class SeedSubmissionBinding:
    """Exact local, scheduler, source, and remote identity for one seed job."""

    lane_id: str
    seed_id: str
    seed_slug: str
    unit_id: str
    action_id: str
    execution_request_sha256: str
    source_epoch_id: str
    source_epoch_identity_sha256: str
    job_batch_name: str
    cluster_id: int
    process_id: int
    remote_workspace: str
    execution_request_path: str
    execution_request_file_sha256: str
    source_epoch_receipt_path: str
    source_epoch_receipt_file_sha256: str
    submission_receipt_path: str
    submission_receipt_file_sha256: str
    intent_path: str
    intent_file_sha256: str
    provisional_result_path: str
    provisional_result_file_sha256: str

    def __post_init__(self) -> None:
        if self.lane_id not in {LANE_HH, LANE_HUBBARD}:
            raise ValueError(f"unsupported prospective seed lane: {self.lane_id}")
        if (
            not self.seed_id.startswith(f"{self.lane_id}::seed::")
            or self.seed_id.rsplit("::", 1)[-1] != self.seed_slug
            or not self.seed_slug
            or not self.unit_id
            or not self.action_id
            or not self.job_batch_name
            or not self.source_epoch_id
        ):
            raise ValueError("seed submission textual identity is invalid")
        if int(self.cluster_id) < 0 or int(self.process_id) != 0:
            raise ValueError("seed submission must bind one scheduler process 0")
        _sha(self.execution_request_sha256, label="execution request digest")
        _sha(
            self.source_epoch_identity_sha256,
            label="source epoch identity digest",
        )
        for label, value in (
            ("execution request file digest", self.execution_request_file_sha256),
            (
                "source epoch receipt file digest",
                self.source_epoch_receipt_file_sha256,
            ),
            ("submission receipt file digest", self.submission_receipt_file_sha256),
            ("intent file digest", self.intent_file_sha256),
            (
                "provisional result file digest",
                self.provisional_result_file_sha256,
            ),
        ):
            _sha(value, label=label)
        _absolute_remote(self.remote_workspace, label="remote workspace")
        for label, value in (
            ("execution request path", self.execution_request_path),
            ("source epoch receipt path", self.source_epoch_receipt_path),
            ("submission receipt path", self.submission_receipt_path),
            ("intent path", self.intent_path),
            ("provisional result path", self.provisional_result_path),
        ):
            _relative_path(value, label=label)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "SeedSubmissionBinding":
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise SeedDeltaAcquisitionError(
                "seed submission binding fields are incomplete or unexpected"
            )
        try:
            return cls(**{name: payload[name] for name in cls.__dataclass_fields__})
        except (TypeError, ValueError) as exc:
            raise SeedDeltaAcquisitionError(
                "seed submission binding is invalid"
            ) from exc


@dataclass(frozen=True, order=True)
class CoveredSeedIncrement:
    lane_id: str
    cluster_id: int
    process_id: int
    action_id: str
    execution_request_sha256: str
    source_epoch_identity_sha256: str
    receipt_path: str
    receipt_file_sha256: str

    def __post_init__(self) -> None:
        if self.lane_id not in {LANE_HH, LANE_HUBBARD}:
            raise ValueError("covered seed increment lane is invalid")
        if int(self.cluster_id) < 0 or int(self.process_id) != 0:
            raise ValueError("covered seed increment identity is invalid")
        _sha(self.execution_request_sha256, label="covered request digest")
        _sha(
            self.source_epoch_identity_sha256,
            label="covered source epoch digest",
        )
        _sha(self.receipt_file_sha256, label="covered receipt file digest")
        _relative_path(self.receipt_path, label="covered receipt path")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "CoveredSeedIncrement":
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise SeedDeltaAcquisitionError(
                "covered seed increment fields are incomplete or unexpected"
            )
        try:
            return cls(**{name: payload[name] for name in cls.__dataclass_fields__})
        except (TypeError, ValueError) as exc:
            raise SeedDeltaAcquisitionError(
                "covered seed increment binding is invalid"
            ) from exc


@dataclass(frozen=True)
class SeedSchedulerObservation:
    lane_id: str
    cluster_id: int
    process_id: int
    unit_id: str
    disposition: str
    job_status: int | None
    exit_code: int | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "cluster_id": self.cluster_id,
            "process_id": self.process_id,
            "unit_id": self.unit_id,
            "disposition": self.disposition,
            "job_status": self.job_status,
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True)
class SeedDeltaDiscovery:
    deltas: tuple[SeedSubmissionBinding, ...]
    observations: tuple[SeedSchedulerObservation, ...]


def _row_scheduler_identity(row: Mapping[str, Any]) -> tuple[int, int]:
    try:
        cluster = int(row["ClusterId"])
        process = int(row["ProcId"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError(
            "scheduler row has no valid cluster/process identity"
        ) from exc
    if cluster < 0 or process < 0:
        raise SeedDeltaAcquisitionError("scheduler identity must be nonnegative")
    return cluster, process


def _validate_scheduler_row_identity(
    row: Mapping[str, Any], binding: SeedSubmissionBinding
) -> None:
    observed = {
        "JobBatchName": row.get("JobBatchName"),
        "AcquisitionUnitId": row.get("AcquisitionUnitId"),
        "AcquisitionLaneId": row.get("AcquisitionLaneId"),
        "PaperIIActionId": row.get("PaperIIActionId"),
        "PaperIIExecutionRequestSha256": row.get(
            "PaperIIExecutionRequestSha256"
        ),
        "PaperIISourceEpochId": row.get("PaperIISourceEpochId"),
    }
    expected = {
        "JobBatchName": binding.job_batch_name,
        "AcquisitionUnitId": binding.unit_id,
        "AcquisitionLaneId": binding.lane_id,
        "PaperIIActionId": binding.action_id,
        "PaperIIExecutionRequestSha256": binding.execution_request_sha256,
        "PaperIISourceEpochId": binding.source_epoch_id,
    }
    optional_if_absent = {"AcquisitionUnitId", "AcquisitionLaneId"}
    changed = [
        field
        for field in expected
        if observed[field] != expected[field]
        and not (
            field in optional_if_absent and observed[field] in (None, "")
        )
    ]
    if changed:
        raise SeedDeltaAcquisitionError(
            "scheduler row crosses the exact seed submission identity: "
            + ", ".join(changed)
        )


def _row_state(row: Mapping[str, Any]) -> tuple[str, int, int | None]:
    try:
        status = int(row["JobStatus"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError("scheduler row JobStatus is invalid") from exc
    exit_raw = row.get("ExitCode")
    try:
        exit_code = None if exit_raw is None else int(exit_raw)
    except (TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError("scheduler row ExitCode is invalid") from exc
    if status == _SUCCESS_STATUS and exit_code == 0:
        return "new_success", status, exit_code
    if status == _SUCCESS_STATUS or status in _TERMINAL_FAILURE_STATUSES:
        return "terminal_failure", status, exit_code
    if status == _HELD_STATUS:
        return "held", status, exit_code
    if status in _NONTERMINAL_STATUSES:
        return "nonterminal", status, exit_code
    raise SeedDeltaAcquisitionError(f"scheduler row JobStatus is unsupported: {status}")


def discover_seed_deltas(
    snapshot: Mapping[str, Any],
    *,
    submissions: Sequence[SeedSubmissionBinding],
    covered: Sequence[CoveredSeedIncrement],
) -> SeedDeltaDiscovery:
    """Return newly successful seed jobs without serializing independent lanes."""

    if snapshot.get("schema") != "paper_ii_combined_scoped_scheduler_snapshot_v1":
        raise SeedDeltaAcquisitionError("combined scheduler snapshot schema is invalid")
    submission_by_key: dict[tuple[int, int], SeedSubmissionBinding] = {}
    for binding in submissions:
        key = (binding.cluster_id, binding.process_id)
        if key in submission_by_key:
            raise SeedDeltaAcquisitionError("duplicate submitted scheduler identity")
        submission_by_key[key] = binding

    coverage_by_key: dict[tuple[int, int], CoveredSeedIncrement] = {}
    for receipt in covered:
        key = (receipt.cluster_id, receipt.process_id)
        if key in coverage_by_key:
            raise SeedDeltaAcquisitionError("duplicate covered scheduler identity")
        coverage_by_key[key] = receipt

    rows_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for lane_field in ("hh_rows", "hubbard_rows"):
        rows = snapshot.get(lane_field)
        if not isinstance(rows, list):
            raise SeedDeltaAcquisitionError(
                f"combined scheduler snapshot {lane_field} is invalid"
            )
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise SeedDeltaAcquisitionError("scheduler row must be an object")
            key = _row_scheduler_identity(raw)
            if key in rows_by_key:
                raise SeedDeltaAcquisitionError(
                    f"duplicate scheduler identity {key[0]}.{key[1]}"
                )
            rows_by_key[key] = dict(raw)

    observations: list[SeedSchedulerObservation] = []
    deltas: list[SeedSubmissionBinding] = []
    for binding in sorted(submissions, key=lambda item: (item.lane_id, item.cluster_id)):
        key = (binding.cluster_id, binding.process_id)
        row = rows_by_key.get(key)
        receipt = coverage_by_key.get(key)
        if receipt is not None and (
            receipt.lane_id != binding.lane_id
            or receipt.action_id != binding.action_id
            or receipt.execution_request_sha256
            != binding.execution_request_sha256
            or receipt.source_epoch_identity_sha256
            != binding.source_epoch_identity_sha256
        ):
            raise SeedDeltaAcquisitionError(
                "covered increment crosses the exact seed submission identity"
            )
        if row is None:
            disposition = (
                "covered_without_current_row" if receipt is not None else "not_observed"
            )
            observations.append(
                SeedSchedulerObservation(
                    lane_id=binding.lane_id,
                    cluster_id=binding.cluster_id,
                    process_id=binding.process_id,
                    unit_id=binding.unit_id,
                    disposition=disposition,
                    job_status=None,
                    exit_code=None,
                )
            )
            continue
        _validate_scheduler_row_identity(row, binding)
        disposition, job_status, exit_code = _row_state(row)
        if receipt is not None:
            if disposition != "new_success":
                raise SeedDeltaAcquisitionError(
                    "durable completed increment conflicts with scheduler state"
                )
            disposition = "covered_success"
        elif disposition == "new_success":
            deltas.append(binding)
        observations.append(
            SeedSchedulerObservation(
                lane_id=binding.lane_id,
                cluster_id=binding.cluster_id,
                process_id=binding.process_id,
                unit_id=binding.unit_id,
                disposition=disposition,
                job_status=job_status,
                exit_code=exit_code,
            )
        )
    return SeedDeltaDiscovery(
        deltas=tuple(deltas), observations=tuple(observations)
    )


def _verify_bound_file(
    root: Path,
    binding: Any,
    *,
    label: str,
    require_bytes: bool,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, Mapping):
        raise SeedDeltaAcquisitionError(f"{label} binding is missing")
    path_text = _relative_path(binding.get("path"), label=f"{label} path")
    expected_sha = _sha(binding.get("sha256"), label=f"{label} digest")
    path = _repo_path(root, path_text, label=label, must_exist=True)
    if not path.is_file() or path.is_symlink() or _file_sha(path) != expected_sha:
        raise SeedDeltaAcquisitionError(f"{label} local digest changed")
    result: dict[str, Any] = {"path": path_text, "sha256": expected_sha}
    if require_bytes:
        expected_bytes = _positive_int(
            binding.get("bytes"), label=f"{label} byte count"
        )
        if path.stat(follow_symlinks=False).st_size != expected_bytes:
            raise SeedDeltaAcquisitionError(f"{label} local byte count changed")
        result["bytes"] = expected_bytes
    return path, result


def _verify_payload_binding(
    root: Path, binding: Any, *, label: str
) -> tuple[Path, dict[str, str]]:
    """Verify a binding that separates canonical payload and file digests."""

    if not isinstance(binding, Mapping):
        raise SeedDeltaAcquisitionError(f"{label} binding is missing")
    path_text = _relative_path(binding.get("path"), label=f"{label} path")
    payload_sha = _sha(binding.get("sha256"), label=f"{label} payload digest")
    file_sha = _sha(binding.get("file_sha256"), label=f"{label} file digest")
    path = _repo_path(root, path_text, label=label, must_exist=True)
    if not path.is_file() or path.is_symlink() or _file_sha(path) != file_sha:
        raise SeedDeltaAcquisitionError(f"{label} file digest changed")
    return path, {"path": path_text, "sha256": payload_sha, "file_sha256": file_sha}


def _validate_self_digest(
    payload: Mapping[str, Any], *, key: str, label: str
) -> str:
    unsigned = dict(payload)
    declared = _sha(unsigned.pop(key, None), label=f"{label} self digest")
    if declared != _canonical_sha(unsigned):
        raise SeedDeltaAcquisitionError(f"{label} self digest mismatch")
    return declared


def _parse_time(value: Any, *, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SeedDeltaAcquisitionError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SeedDeltaAcquisitionError(f"{label} must include a timezone")
    return parsed


def _load_provisional_evidence(
    root: Path,
    *,
    path: str,
    intent_path: str,
    intent_sha256: str,
    lane_id: str,
    unit_id: str,
    action_id: str,
    request_sha256: str,
    job_batch_name: str,
    package_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    source = _repo_path(
        root, path, label="provisional submission result", must_exist=True
    )
    payload = _read_object(source, label="provisional submission result")
    declared = _validate_self_digest(
        payload,
        key="submission_result_sha256",
        label="provisional submission result",
    )
    try:
        cluster_id = int(payload.get("cluster_id"))
        process_ids = [int(value) for value in payload.get("expected_process_ids")]
    except (TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError(
            "provisional scheduler identity is invalid"
        ) from exc
    stdout = str(payload.get("condor_submit_stdout") or "")
    if (
        payload.get("schema") != PROVISIONAL_SUBMISSION_RESULT_V1
        or payload.get("status") != "submitted"
        or payload.get("lane_id") != lane_id
        or payload.get("unit_id") != unit_id
        or payload.get("action_ids") != [action_id]
        or payload.get("job_batch_name") != job_batch_name
        or payload.get("intent_path") != intent_path
        or payload.get("intent_sha256") != intent_sha256
        or payload.get("package") != dict(package_binding)
        or process_ids != [0]
        or cluster_id < 0
        or payload.get("condor_submit_returncode") != 0
        or f"cluster {cluster_id}." not in stdout
        or payload.get("scheduler_acceptance_query_performed") is not False
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("lifecycle_validator_status") != "pass"
        or payload.get("dry_run_status") != "pass"
    ):
        raise SeedDeltaAcquisitionError(
            "provisional submission evidence identity diverged"
        )
    # The pre-submit snapshot is historical authorization evidence.  Its bytes
    # remain bound here without retroactively applying a newer snapshot schema.
    pre_path = _relative_path(
        payload.get("pre_submit_snapshot_path"),
        label="pre-submit snapshot path",
    )
    pre = _repo_path(root, pre_path, label="pre-submit snapshot", must_exist=True)
    pre_payload = _read_object(pre, label="pre-submit snapshot")
    pre_sha = _sha(
        payload.get("pre_submit_snapshot_sha256"),
        label="pre-submit snapshot digest",
    )
    pre_unsigned = dict(pre_payload)
    observed_pre_sha = _sha(
        pre_unsigned.pop("snapshot_sha256", None),
        label="pre-submit snapshot self digest",
    )
    if observed_pre_sha != pre_sha or observed_pre_sha != _canonical_sha(pre_unsigned):
        raise SeedDeltaAcquisitionError("pre-submit snapshot binding changed")
    if payload.get("submit_descriptor"):
        descriptor, _ = _verify_bound_file(
            root,
            payload["submit_descriptor"],
            label="submit descriptor",
            require_bytes=False,
        )
        if descriptor.is_symlink():  # pragma: no cover - guarded above
            raise SeedDeltaAcquisitionError("submit descriptor is a symlink")
    return payload, declared


def _load_recovery_snapshot_row(
    root: Path,
    *,
    binding: Mapping[str, Any],
    query_name: str,
    row_sha256: str,
    cluster_id: int,
    process_id: int,
) -> None:
    if not isinstance(binding, Mapping):
        raise SeedDeltaAcquisitionError("recovery scheduler snapshot binding is missing")
    path_text = _relative_path(
        binding.get("path"), label="recovery scheduler snapshot path"
    )
    payload_sha = _sha(
        binding.get("sha256"), label="recovery scheduler snapshot payload digest"
    )
    path = _repo_path(
        root, path_text, label="recovery scheduler snapshot", must_exist=True
    )
    expected_file_sha = binding.get("file_sha256")
    if expected_file_sha is not None and _file_sha(path) != _sha(
        expected_file_sha, label="recovery snapshot file digest"
    ):
        raise SeedDeltaAcquisitionError("recovery snapshot file digest changed")
    snapshot = _read_object(path, label="recovery scheduler snapshot")
    declared = _sha(
        snapshot.get("snapshot_sha256"), label="recovery snapshot self digest"
    )
    unsigned = dict(snapshot)
    unsigned.pop("snapshot_sha256", None)
    if declared != payload_sha:
        raise SeedDeltaAcquisitionError("recovery snapshot digest binding changed")
    if declared != _canonical_sha(unsigned):
        raise SeedDeltaAcquisitionError("recovery snapshot self digest mismatch")
    queries = snapshot.get("queries")
    query = queries.get(query_name) if isinstance(queries, Mapping) else None
    rows = query.get("rows") if isinstance(query, Mapping) else None
    if not isinstance(rows, list):
        raise SeedDeltaAcquisitionError("recovery snapshot query is missing")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and int(row.get("ClusterId", -1)) == cluster_id
        and int(row.get("ProcId", -1)) == process_id
    ]
    if len(matches) != 1 or _canonical_sha(dict(matches[0])) != row_sha256:
        raise SeedDeltaAcquisitionError("recovery scheduler row binding changed")


def _source_epoch_from_request(
    request: ExecutionRequest, *, expected_lane: str
) -> SourceEpochIdentity:
    raw = request.inputs.get("source_epoch")
    if not isinstance(raw, Mapping):
        raise SeedDeltaAcquisitionError("seed request has no source epoch")
    try:
        epoch = SourceEpochIdentity(
            lane_id=str(raw.get("lane_id") or ""),
            source_epoch_id=str(raw.get("source_epoch_id") or ""),
            source_revision=raw.get("source_revision"),
            source_archive_sha256=raw.get("source_archive_sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError("seed request source epoch is invalid") from exc
    if (
        epoch.lane_id != expected_lane
        or raw.get("source_epoch_identity_sha256") != epoch.identity_sha256
    ):
        raise SeedDeltaAcquisitionError("seed request source epoch identity changed")
    return epoch


def _load_seed_package_contract(
    root: Path,
    *,
    lane_id: str,
    intent_unit: Any,
) -> tuple[
    ExecutionRequest,
    SourceEpochIdentity,
    SourceEpochReceipt,
    Path,
    dict[str, Any],
]:
    manifest_path, manifest_binding = _verify_bound_file(
        root,
        intent_unit.payload.get("package_manifest"),
        label="seed package manifest",
        require_bytes=False,
    )
    manifest = _read_object(manifest_path, label="seed package manifest")
    _validate_self_digest(
        manifest,
        key="bootstrap_package_manifest_sha256",
        label="seed package manifest",
    )
    request_raw = manifest.get("execution_request")
    if not isinstance(request_raw, Mapping):
        raise SeedDeltaAcquisitionError("seed package has no execution request")
    try:
        request = ExecutionRequest.from_mapping(request_raw)
        seed = (
            validate_hh_seed_request(request)
            if lane_id == LANE_HH
            else validate_hubbard_seed_request(request)
        )
    except (ProspectiveExecutionError, TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError(
            "seed package request is not canonical"
        ) from exc
    epoch = _source_epoch_from_request(request, expected_lane=lane_id)
    source_path, source_binding = _verify_bound_file(
        root,
        manifest.get("source_epoch_receipt"),
        label="source epoch receipt",
        require_bytes=False,
    )
    try:
        source = SourceEpochReceipt.load(
            source_path, repository_root=root, expected_epoch=epoch
        )
    except (OSError, ProspectiveExecutionError, ValueError) as exc:
        raise SeedDeltaAcquisitionError(
            "source epoch receipt does not bind the seed request"
        ) from exc
    package_path, package_binding = _verify_bound_file(
        root,
        intent_unit.payload.get("package"),
        label="seed action package",
        require_bytes=True,
    )
    del package_path
    expected_batch = (
        f"paper-ii-seed-{lane_id.replace('_', '-')}-{request.digest[:16]}"
    )
    if (
        manifest.get("schema") != _PACKAGE_MANIFEST_V1
        or manifest.get("status") != "prepared_local_only"
        or manifest.get("lane_id") != lane_id
        or manifest.get("action_id") != request.action_id
        or manifest.get("execution_request_sha256") != request.digest
        or manifest.get("source_epoch_identity_sha256") != epoch.identity_sha256
        or manifest.get("job_batch_name") != expected_batch
        or manifest.get("qiskit_compilation") is not False
        or manifest.get("production_submission_enabled") is not False
        or manifest.get("package")
        != {"path": package_binding["path"], "sha256": package_binding["sha256"]}
        or manifest.get("package_bytes") != package_binding["bytes"]
        or intent_unit.action_ids != (request.action_id,)
        or intent_unit.payload.get("execution_request_sha256") != request.digest
        or intent_unit.payload.get("source_epoch_identity_sha256")
        != epoch.identity_sha256
        or intent_unit.payload.get("job_batch_name") != expected_batch
        or intent_unit.payload.get("remote_raw_evidence_policy") != "preserve"
    ):
        raise SeedDeltaAcquisitionError(
            "seed intent/package/request/source identity diverged"
        )
    prepared = "prepared_seed_actions" if lane_id == LANE_HH else "prepared_seed_action"
    request_path = Path(source.receipt_path).parent / prepared
    if lane_id == LANE_HH:
        request_path = request_path / seed.seed_id.rsplit("::", 1)[-1]
    request_path = request_path / "execution_request.json"
    local_request = _repo_path(
        root,
        request_path,
        label="canonical seed execution request",
        must_exist=True,
    )
    try:
        request_from_file = ExecutionRequest.from_mapping(
            _read_object(local_request, label="canonical seed execution request")
        )
    except (ProspectiveExecutionError, TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError(
            "canonical seed execution request is invalid"
        ) from exc
    if request_from_file.to_json_dict() != request.to_json_dict():
        raise SeedDeltaAcquisitionError(
            "canonical and packaged execution requests diverged"
        )
    return request, epoch, source, local_request, manifest_binding


def _validate_canonical_submission_receipt(
    root: Path,
    *,
    receipt_path: Path,
    lane_id: str,
    unit: Any,
    intent_path: str,
    intent_file_sha256: str,
    intent_sha256: str,
    request: ExecutionRequest,
    epoch: SourceEpochIdentity,
    provisional: Mapping[str, Any],
    provisional_path: str,
    provisional_file_sha256: str,
    provisional_self_sha256: str,
    package_manifest_binding: Mapping[str, Any],
) -> tuple[int, str]:
    receipt = _read_object(receipt_path, label="canonical submission receipt")
    remote = receipt.get("remote")
    recovery = remote.get("recovery_evidence") if isinstance(remote, Mapping) else None
    snapshot_binding = recovery.get("snapshot") if isinstance(recovery, Mapping) else None
    provisional_binding = (
        recovery.get("provisional_submission_result")
        if isinstance(recovery, Mapping)
        else None
    )
    try:
        cluster_id = int(remote.get("cluster_id"))
        process_ids = [int(value) for value in remote.get("process_ids")]
    except (AttributeError, TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError(
            "canonical submission receipt scheduler identity is invalid"
        ) from exc
    expected_path = (
        Path("output")
        / lane_id
        / "acquisition_supervisor/submission_receipts"
        / f"{unit.unit_id}.json"
    )
    if receipt_path.relative_to(root).as_posix() != expected_path.as_posix():
        raise SeedDeltaAcquisitionError(
            "canonical submission receipt is at a noncanonical path"
        )
    if (
        receipt.get("schema") != _SUBMISSION_RECEIPT_V1
        or receipt.get("lane_id") != lane_id
        or receipt.get("unit_id") != unit.unit_id
        or receipt.get("action_ids") != [request.action_id]
        or receipt.get("intent_sha256") != intent_sha256
        or receipt.get("recovered_from_scheduler") is not True
        or not isinstance(remote, Mapping)
        or cluster_id != int(provisional.get("cluster_id"))
        or process_ids != [0]
        or remote.get("unit_id") != unit.unit_id
        or remote.get("job_batch_name") != provisional.get("job_batch_name")
        or remote.get("action_ids") != [request.action_id]
        or remote.get("scheduler_identity") != f"{cluster_id}.0"
        or not isinstance(recovery, Mapping)
        or recovery.get("scheduler_confirmation") is not True
        or recovery.get("query") not in {"queue", "history"}
        or provisional_binding
        != {
            "path": provisional_path,
            "sha256": provisional_self_sha256,
            "file_sha256": provisional_file_sha256,
        }
        or not isinstance(snapshot_binding, Mapping)
    ):
        raise SeedDeltaAcquisitionError(
            "canonical submission receipt identity diverged"
        )
    if lane_id == LANE_HUBBARD and (
        remote.get("source_epoch_identity_sha256") != epoch.identity_sha256
        or remote.get("source_epoch_id") != epoch.source_epoch_id
        or recovery.get("package_manifest") != dict(package_manifest_binding)
    ):
        raise SeedDeltaAcquisitionError(
            "Hubbard canonical recovery source/package identity diverged"
        )
    row_sha = _sha(
        recovery.get("scheduler_row_sha256"),
        label="recovery scheduler row digest",
    )
    _load_recovery_snapshot_row(
        root,
        binding=snapshot_binding,
        query_name=str(recovery["query"]),
        row_sha256=row_sha,
        cluster_id=cluster_id,
        process_id=0,
    )
    _parse_time(receipt.get("recorded_at"), label="submission receipt recorded_at")
    # Rebind the intent bytes explicitly because Hubbard's public loader returns
    # the typed intent rather than its file digest.
    intent_source = _repo_path(root, intent_path, label="submission intent", must_exist=True)
    if _file_sha(intent_source) != intent_file_sha256:
        raise SeedDeltaAcquisitionError("submission intent file digest changed")
    return cluster_id, _file_sha(receipt_path)


def load_hh_submission_binding(
    path: str | Path, *, repository_root: str | Path
) -> SeedSubmissionBinding:
    """Load one of the six canonical recovered HH seed submissions."""

    root = Path(repository_root).resolve(strict=True)
    receipt_path = _repo_path(
        root, path, label="HH canonical submission receipt", must_exist=True
    )
    receipt = _read_object(receipt_path, label="HH canonical submission receipt")
    remote = receipt.get("remote")
    recovery = remote.get("recovery_evidence") if isinstance(remote, Mapping) else None
    provisional_binding = (
        recovery.get("provisional_submission_result")
        if isinstance(recovery, Mapping)
        else None
    )
    provisional_path, normalized_provisional = _verify_payload_binding(
        root,
        provisional_binding,
        label="HH provisional submission result",
    )
    provisional_raw = _read_object(
        provisional_path, label="HH provisional submission result"
    )
    intent_path = _relative_path(
        provisional_raw.get("intent_path"), label="HH intent path"
    )
    try:
        intent = load_durable_submission_intent(intent_path, repository_root=root)
    except SubmissionRecoveryError as exc:
        raise SeedDeltaAcquisitionError(str(exc)) from exc
    request, epoch, source, request_path, manifest_binding = (
        _load_seed_package_contract(root, lane_id=LANE_HH, intent_unit=intent.unit)
    )
    provisional, provisional_self_sha = _load_provisional_evidence(
        root,
        path=provisional_path.relative_to(root).as_posix(),
        intent_path=intent.path,
        intent_sha256=intent.intent_sha256,
        lane_id=LANE_HH,
        unit_id=intent.unit.unit_id,
        action_id=request.action_id,
        request_sha256=request.digest,
        job_batch_name=str(intent.unit.payload["job_batch_name"]),
        package_binding=intent.unit.payload["package"],
    )
    if normalized_provisional.get("sha256") != provisional_self_sha:
        raise SeedDeltaAcquisitionError(
            "HH receipt provisional payload digest changed"
        )
    cluster_id, receipt_file_sha = _validate_canonical_submission_receipt(
        root,
        receipt_path=receipt_path,
        lane_id=LANE_HH,
        unit=intent.unit,
        intent_path=intent.path,
        intent_file_sha256=intent.file_sha256,
        intent_sha256=intent.intent_sha256,
        request=request,
        epoch=epoch,
        provisional=provisional,
        provisional_path=provisional_path.relative_to(root).as_posix(),
        provisional_file_sha256=_file_sha(provisional_path),
        provisional_self_sha256=provisional_self_sha,
        package_manifest_binding=manifest_binding,
    )
    seed_slug = request.inputs["seed_id"].rsplit("::", 1)[-1]
    return SeedSubmissionBinding(
        lane_id=LANE_HH,
        seed_id=str(request.inputs["seed_id"]),
        seed_slug=seed_slug,
        unit_id=intent.unit.unit_id,
        action_id=request.action_id,
        execution_request_sha256=request.digest,
        source_epoch_id=epoch.source_epoch_id,
        source_epoch_identity_sha256=epoch.identity_sha256,
        job_batch_name=str(provisional["job_batch_name"]),
        cluster_id=cluster_id,
        process_id=0,
        remote_workspace=_absolute_remote(
            intent.unit.payload.get("remote_workspace"), label="HH remote workspace"
        ),
        execution_request_path=request_path.relative_to(root).as_posix(),
        execution_request_file_sha256=_file_sha(request_path),
        source_epoch_receipt_path=source.receipt_path,
        source_epoch_receipt_file_sha256=_file_sha(root / source.receipt_path),
        submission_receipt_path=receipt_path.relative_to(root).as_posix(),
        submission_receipt_file_sha256=receipt_file_sha,
        intent_path=intent.path,
        intent_file_sha256=intent.file_sha256,
        provisional_result_path=provisional_path.relative_to(root).as_posix(),
        provisional_result_file_sha256=_file_sha(provisional_path),
    )


def load_hubbard_submission_binding(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    provisional_result_path: str | Path,
    canonical_receipt_path: str | Path,
) -> SeedSubmissionBinding:
    """Load the intent/provisional/canonical Hubbard recovery chain."""

    root = Path(repository_root).resolve(strict=True)
    try:
        intent = load_hubbard_durable_submission_intent(
            intent_path, repository_root=root
        )
    except SubmissionRecoveryError as exc:
        raise SeedDeltaAcquisitionError(str(exc)) from exc
    unit = intent.base.unit
    request, epoch, source, request_path, manifest_binding = (
        _load_seed_package_contract(root, lane_id=LANE_HUBBARD, intent_unit=unit)
    )
    if request.to_json_dict() != intent.request.to_json_dict() or epoch != intent.source_epoch:
        raise SeedDeltaAcquisitionError("Hubbard recovery/package request diverged")
    provisional_path = _repo_path(
        root,
        provisional_result_path,
        label="Hubbard provisional submission result",
        must_exist=True,
    )
    provisional, provisional_self_sha = _load_provisional_evidence(
        root,
        path=provisional_path.relative_to(root).as_posix(),
        intent_path=intent.base.path,
        intent_sha256=intent.base.intent_sha256,
        lane_id=LANE_HUBBARD,
        unit_id=unit.unit_id,
        action_id=request.action_id,
        request_sha256=request.digest,
        job_batch_name=str(unit.payload["job_batch_name"]),
        package_binding=unit.payload["package"],
    )
    receipt_path = _repo_path(
        root,
        canonical_receipt_path,
        label="Hubbard canonical recovery receipt",
        must_exist=True,
    )
    cluster_id, receipt_file_sha = _validate_canonical_submission_receipt(
        root,
        receipt_path=receipt_path,
        lane_id=LANE_HUBBARD,
        unit=unit,
        intent_path=intent.base.path,
        intent_file_sha256=intent.base.file_sha256,
        intent_sha256=intent.base.intent_sha256,
        request=request,
        epoch=epoch,
        provisional=provisional,
        provisional_path=provisional_path.relative_to(root).as_posix(),
        provisional_file_sha256=_file_sha(provisional_path),
        provisional_self_sha256=provisional_self_sha,
        package_manifest_binding={
            "path": intent.package_manifest_path,
            "sha256": intent.package_manifest_sha256,
        },
    )
    seed_slug = request.inputs["seed_id"].rsplit("::", 1)[-1]
    return SeedSubmissionBinding(
        lane_id=LANE_HUBBARD,
        seed_id=str(request.inputs["seed_id"]),
        seed_slug=seed_slug,
        unit_id=unit.unit_id,
        action_id=request.action_id,
        execution_request_sha256=request.digest,
        source_epoch_id=epoch.source_epoch_id,
        source_epoch_identity_sha256=epoch.identity_sha256,
        job_batch_name=str(provisional["job_batch_name"]),
        cluster_id=cluster_id,
        process_id=0,
        remote_workspace=_absolute_remote(
            unit.payload.get("remote_workspace"), label="Hubbard remote workspace"
        ),
        execution_request_path=request_path.relative_to(root).as_posix(),
        execution_request_file_sha256=_file_sha(request_path),
        source_epoch_receipt_path=source.receipt_path,
        source_epoch_receipt_file_sha256=_file_sha(root / source.receipt_path),
        submission_receipt_path=receipt_path.relative_to(root).as_posix(),
        submission_receipt_file_sha256=receipt_file_sha,
        intent_path=intent.base.path,
        intent_file_sha256=intent.base.file_sha256,
        provisional_result_path=provisional_path.relative_to(root).as_posix(),
        provisional_result_file_sha256=_file_sha(provisional_path),
    )


def _load_covered_seed_increment(
    path: str | Path,
    *,
    repository_root: Path,
    expected_lane: str,
) -> CoveredSeedIncrement:
    """Reload a receipt and its raw/canonical evidence without promoting anew."""

    root = repository_root
    receipt_path = _repo_path(
        root,
        path,
        label=f"{expected_lane} seed increment receipt",
        must_exist=True,
    )
    payload = _read_object(
        receipt_path, label=f"{expected_lane} seed increment receipt"
    )
    schema = (
        HH_SEED_INCREMENT_RECEIPT_V1
        if expected_lane == LANE_HH
        else HUBBARD_SEED_INCREMENT_RECEIPT_V1
    )
    if payload.get("schema") != schema:
        raise SeedDeltaAcquisitionError("seed increment receipt schema changed")
    unsigned = dict(payload)
    declared = _sha(
        unsigned.pop("increment_receipt_sha256", None),
        label="seed increment receipt self digest",
    )
    if declared != _canonical_sha(unsigned):
        raise SeedDeltaAcquisitionError("seed increment receipt self digest mismatch")
    execution = payload.get("execution_request")
    source = payload.get("source_epoch_receipt")
    raw = payload.get("raw_evidence")
    seed = payload.get("seed")
    try:
        cluster_id = int(payload.get("cluster_id"))
        process_ids = [int(value) for value in payload.get("process_ids")]
    except (TypeError, ValueError) as exc:
        raise SeedDeltaAcquisitionError(
            "seed increment scheduler identity is invalid"
        ) from exc
    relative = receipt_path.relative_to(root).as_posix()
    expected_prefix = (
        f"output/{expected_lane}/attempts/cluster_{cluster_id}/increments/"
    )
    if (
        payload.get("status") != "complete"
        or payload.get("lane_id") != expected_lane
        or process_ids != [0]
        or not isinstance(execution, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(raw, Mapping)
        or not isinstance(seed, Mapping)
        or raw.get("remote_raw_preserved") is not True
        or payload.get("canonical_pair_promoted_atomically") is not True
        or payload.get("qiskit_compilation_used") is not False
        or not relative.startswith(expected_prefix)
    ):
        raise SeedDeltaAcquisitionError(
            "seed increment receipt identity or preservation diverged"
        )
    request_sha = _sha(
        execution.get("execution_request_sha256"),
        label="covered execution request digest",
    )
    source_sha = _sha(
        source.get("source_epoch_identity_sha256"),
        label="covered source epoch digest",
    )
    action_id = str(payload.get("action_id") or "")
    if not action_id:
        raise SeedDeltaAcquisitionError("covered seed action identity is empty")
    raw_root = _relative_path(raw.get("root"), label="covered raw root")
    request_path = _relative_path(
        execution.get("path"), label="covered execution request path"
    )
    source_path = _relative_path(
        source.get("path"), label="covered source receipt path"
    )
    # Resolve every bound path before replay so a symlinked parent cannot turn a
    # durable receipt into apparent coverage.
    for label, bound in (
        ("covered raw root", raw_root),
        ("covered execution request", request_path),
        ("covered source receipt", source_path),
    ):
        _repo_path(root, bound, label=label, must_exist=True)
    increment_root = receipt_path.parent.parent.relative_to(root).as_posix()
    kwargs = {
        "repository_root": root,
        "raw_output_root": raw_root,
        "execution_request": request_path,
        "source_epoch_receipt": source_path,
        "expected_action_id": action_id,
        "expected_execution_request_sha256": request_sha,
        "expected_source_epoch_identity_sha256": source_sha,
        "cluster_id": cluster_id,
        "process_id": 0,
        "increment_root": increment_root,
    }
    try:
        replayed = (
            ingest_hh_seed_delta(**kwargs)
            if expected_lane == LANE_HH
            else ingest_hubbard_seed_delta(**kwargs)
        )
    except (HHSeedIngestionError, HubbardSeedIngestionError) as exc:
        raise SeedDeltaAcquisitionError(
            "covered seed raw/canonical evidence failed typed replay"
        ) from exc
    qualification = seed.get("qualification_receipt")
    if (
        replayed.increment_receipt_path != relative
        or replayed.increment_receipt_sha256 != _file_sha(receipt_path)
        or not isinstance(qualification, Mapping)
        or seed.get("artifact") != replayed.seed_receipt.artifact.to_json_dict()
        or qualification.get("path") != replayed.seed_receipt.receipt_path
        or qualification.get("sha256") != replayed.seed_receipt.receipt_sha256
    ):
        raise SeedDeltaAcquisitionError(
            "covered seed receipt no longer binds its raw/canonical evidence"
        )
    return CoveredSeedIncrement(
        lane_id=expected_lane,
        cluster_id=cluster_id,
        process_id=0,
        action_id=action_id,
        execution_request_sha256=request_sha,
        source_epoch_identity_sha256=source_sha,
        receipt_path=relative,
        receipt_file_sha256=_file_sha(receipt_path),
    )


def load_covered_seed_increments(
    *, repository_root: str | Path
) -> tuple[CoveredSeedIncrement, ...]:
    """Load durable process coverage only after full typed evidence replay."""

    root = Path(repository_root).resolve(strict=True)
    results: list[CoveredSeedIncrement] = []
    seen: set[tuple[int, int]] = set()
    for lane_id in (LANE_HH, LANE_HUBBARD):
        attempts = root / "output" / lane_id / "attempts"
        for lexical in sorted(
            attempts.glob("cluster_*/increments/*/evidence/increment_receipt.json")
        ):
            covered = _load_covered_seed_increment(
                lexical, repository_root=root, expected_lane=lane_id
            )
            key = (covered.cluster_id, covered.process_id)
            if key in seen:
                raise SeedDeltaAcquisitionError(
                    "duplicate covered scheduler identity"
                )
            seen.add(key)
            results.append(covered)
    return tuple(sorted(results))


def _output_token(binding: SeedSubmissionBinding) -> str:
    lane = binding.lane_id.replace("_", "-")
    token = f"{lane}-{binding.execution_request_sha256[:20]}"
    if _SAFE_REMOTE_COMPONENT.fullmatch(token) is None:
        raise SeedDeltaAcquisitionError("derived remote output token is unsafe")
    return token


def build_delta_plan_item(
    repository_root: str | Path, binding: SeedSubmissionBinding
) -> dict[str, Any]:
    """Materialize one typed packaging/transfer/ingestion recipe."""

    root = Path(repository_root).resolve(strict=True)
    token = _output_token(binding)
    workspace = PurePosixPath(
        _absolute_remote(binding.remote_workspace, label="remote workspace")
    )
    raw_subtree = workspace / "raw_outputs" / "paper_ii_seed" / token
    archive = (
        workspace / "transfer_archives" / "paper_ii_seed" / f"{token}.tar.gz"
    )
    attempt = (
        Path("output")
        / binding.lane_id
        / "attempts"
        / f"cluster_{binding.cluster_id}"
    )
    archive_rel = attempt / "archives" / f"proc0_seed_{token}.tar.gz"
    increment = attempt / "increments" / "proc0"
    raw_root = increment / "raw" / "paper_ii_seed"
    final_path = archive_rel.as_posix()
    packaging = {
        "schema": REMOTE_PACKAGING_RECIPE_V1,
        "raw_subtree": str(raw_subtree),
        "archive_path": str(archive),
        "archive_temporary_path": f"{archive}.building.part",
        "archive_format": "tar.gz",
        "ordered_archive_members": list(_EXPECTED_INVENTORY),
        "expected_inventory": list(_EXPECTED_INVENTORY),
        "expected_member_count": len(_EXPECTED_INVENTORY),
        "member_type": "regular_file_nofollow",
        "source_confinement_root": str(raw_subtree),
        "forbid_path_escape": True,
        "forbid_symlinks": True,
        "preserve_source": True,
        "remote_raw_evidence_policy": "preserve",
        "publication": "fsync_then_atomic_rename_without_source_mutation",
        "identity_collection": {
            "required_fields": [
                "archive_bytes",
                "archive_sha256",
                "tar_member_count",
                "gzip_integrity",
                "tar_integrity",
                "members",
            ],
            "archive_identity_adapter": (
                "pipelines.time_dynamics.chtc_acquisition_operations:"
                "AuthenticatedControlMaster.archive_identity"
            ),
            "member_identity": "nofollow_size_sha256_for_each_expected_member",
        },
    }
    local_transfer = {
        "schema": LOCAL_TRANSFER_RECIPE_V1,
        "adapter": (
            "pipelines.time_dynamics.chtc_acquisition_operations:"
            "CHTCAcquisitionOperations.fetch_verified_archive"
        ),
        "archive_transfer_key": [
            binding.lane_id,
            binding.cluster_id,
            [binding.process_id],
        ],
        "remote_path": str(archive),
        "final_path": final_path,
        "part_path": f"{final_path}.part",
        "quarantine_root": (archive_rel.parent / "quarantine").as_posix(),
        "recovery_receipt_path": (
            archive_rel.parent / f"{archive_rel.name}.transfer_recovery_receipt.json"
        ).as_posix(),
        "finalization_intent_path": (
            archive_rel.parent
            / f"{archive_rel.name}.transfer_finalization_intent.json"
        ).as_posix(),
        "finalization_receipt_path": (
            archive_rel.parent
            / f"{archive_rel.name}.transfer_finalization_receipt.json"
        ).as_posix(),
        "expected_identity_source": ARCHIVE_IDENTITY_V1,
        "expected_member_count": len(_EXPECTED_INVENTORY),
        "free_floor_bytes": DEFAULT_FREE_FLOOR_BYTES,
        "ram_guard_limit_mb": 8000,
        "one_archive_at_a_time": True,
        "remote_prefix_proof_required_for_resume": True,
        "remote_raw_preserved": True,
    }
    ingestion = {
        "handler": (
            "pipelines.time_dynamics.prospective_hh_seed_ingestion:"
            "ingest_hh_seed_delta"
            if binding.lane_id == LANE_HH
            else "pipelines.time_dynamics.prospective_hubbard_seed_ingestion:"
            "ingest_hubbard_seed_delta"
        ),
        "lane_id": binding.lane_id,
        "seed_id": binding.seed_id,
        "unit_id": binding.unit_id,
        "job_batch_name": binding.job_batch_name,
        "source_epoch_id": binding.source_epoch_id,
        "increment_root": increment.as_posix(),
        "raw_output_root": raw_root.as_posix(),
        "execution_request": binding.execution_request_path,
        "source_epoch_receipt": binding.source_epoch_receipt_path,
        "expected_action_id": binding.action_id,
        "expected_execution_request_sha256": binding.execution_request_sha256,
        "expected_source_epoch_identity_sha256": (
            binding.source_epoch_identity_sha256
        ),
        "cluster_id": binding.cluster_id,
        "process_id": binding.process_id,
        "safe_extraction": "exact_inventory_manual_nofollow_atomic_directory_rename",
    }
    core: dict[str, Any] = {
        "schema": PLAN_ITEM_V1,
        "lane_id": binding.lane_id,
        "seed_id": binding.seed_id,
        "seed_slug": binding.seed_slug,
        "scheduler_identity": {
            "cluster_id": binding.cluster_id,
            "process_id": binding.process_id,
        },
        "submission_identity": {
            "unit_id": binding.unit_id,
            "job_batch_name": binding.job_batch_name,
            "action_id": binding.action_id,
            "execution_request_sha256": binding.execution_request_sha256,
            "source_epoch_id": binding.source_epoch_id,
            "source_epoch_identity_sha256": (
                binding.source_epoch_identity_sha256
            ),
        },
        "evidence": {
            "execution_request": {
                "path": binding.execution_request_path,
                "sha256": binding.execution_request_file_sha256,
            },
            "source_epoch_receipt": {
                "path": binding.source_epoch_receipt_path,
                "sha256": binding.source_epoch_receipt_file_sha256,
            },
            "submission_receipt": {
                "path": binding.submission_receipt_path,
                "sha256": binding.submission_receipt_file_sha256,
            },
            "submission_intent": {
                "path": binding.intent_path,
                "sha256": binding.intent_file_sha256,
            },
            "provisional_submission_result": {
                "path": binding.provisional_result_path,
                "sha256": binding.provisional_result_file_sha256,
            },
        },
        "remote_packaging": packaging,
        "local_transfer": local_transfer,
        "ingestion": ingestion,
        "qiskit_compilation": False,
        "submission_authorized": False,
        "production_submission_enabled": False,
    }
    # Touching no path here is deliberate: a plan can precede every local
    # archive directory and remains a pure, deterministic value.
    del root
    return {**core, "plan_item_sha256": _canonical_sha(core)}


def assemble_seed_delta_acquisition_plan(
    *,
    repository_root: str | Path,
    snapshot_path: str | Path,
    submissions: Sequence[SeedSubmissionBinding],
    covered: Sequence[CoveredSeedIncrement],
) -> dict[str, Any]:
    """Assemble one immutable plan from typed, already-loaded evidence."""

    root = Path(repository_root).resolve(strict=True)
    snapshot_source = _repo_path(
        root,
        snapshot_path,
        label="combined scheduler snapshot",
        must_exist=True,
    )
    snapshot = _read_object(snapshot_source, label="combined scheduler snapshot")
    try:
        snapshot_sha = validate_combined_snapshot(snapshot)
    except CombinedSnapshotError as exc:
        raise SeedDeltaAcquisitionError(str(exc)) from exc
    hh = tuple(row for row in submissions if row.lane_id == LANE_HH)
    hubbard = tuple(row for row in submissions if row.lane_id == LANE_HUBBARD)
    if (
        len(submissions) != 7
        or len(hh) != 6
        or {row.seed_slug for row in hh} != set(EXPECTED_HH_SEED_SLUGS)
        or len(hubbard) != 1
        or hubbard[0].seed_slug != "l4__u4__half_filled"
        or len({row.cluster_id for row in submissions}) != 7
        or len({row.unit_id for row in submissions}) != 7
        or len({row.action_id for row in submissions}) != 7
        or len({row.execution_request_sha256 for row in submissions}) != 7
    ):
        raise SeedDeltaAcquisitionError(
            "seed submission evidence must cover six HH seeds and one Hubbard-L4 seed"
        )
    discovery = discover_seed_deltas(
        snapshot, submissions=submissions, covered=covered
    )
    items = [build_delta_plan_item(root, row) for row in discovery.deltas]
    core: dict[str, Any] = {
        "schema": PLAN_V1,
        "status": "planned_offline_only",
        "snapshot": {
            "path": snapshot_source.relative_to(root).as_posix(),
            "sha256": snapshot_sha,
            "file_sha256": _file_sha(snapshot_source),
        },
        "submission_count": len(submissions),
        "hh_submission_count": len(hh),
        "hubbard_submission_count": len(hubbard),
        "submission_bindings": [
            row.to_json_dict() for row in sorted(submissions)
        ],
        "covered_increment_count": len(covered),
        "covered_increment_bindings": [
            row.to_json_dict() for row in sorted(covered)
        ],
        "delta_count": len(items),
        "observations": [
            row.to_json_dict() for row in discovery.observations
        ],
        "deltas": items,
        "storage_floor_bytes": DEFAULT_FREE_FLOOR_BYTES,
        "ram_guard_limit_mb": 8000,
        "one_archive_at_a_time": True,
        "remote_raw_evidence_policy": "preserve",
        "qiskit_compilation": False,
        "scheduler_contact_performed": False,
        "ssh_contact_performed": False,
        "submission_authorized": False,
        "production_submission_enabled": False,
    }
    return {**core, "plan_sha256": _canonical_sha(core)}


def _atomic_write_immutable_json(
    path: Path, payload: Mapping[str, Any], *, label: str
) -> str:
    content = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise SeedDeltaAcquisitionError(
                f"refusing to overwrite divergent immutable {label}: {path}"
            )
        return "unchanged"
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise SeedDeltaAcquisitionError(
                        f"immutable {label} write made no progress"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.read_bytes() != content:
                raise SeedDeltaAcquisitionError(
                    f"concurrent divergent immutable {label}: {path}"
                )
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def write_seed_delta_plan(
    path: str | Path,
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> str:
    root = Path(repository_root).resolve(strict=True)
    validate_seed_delta_plan(plan, repository_root=root, verify_evidence=True)
    target = _repo_path(
        root, path, label="seed delta plan output", must_exist=False
    )
    return _atomic_write_immutable_json(target, plan, label="seed delta plan")


def validate_seed_delta_plan(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
    verify_evidence: bool = True,
) -> str:
    """Validate a plan and, by default, every bound local evidence file."""

    root = Path(repository_root).resolve(strict=True)
    unsigned = dict(plan)
    declared = _sha(unsigned.pop("plan_sha256", None), label="plan digest")
    if declared != _canonical_sha(unsigned):
        raise SeedDeltaAcquisitionError("seed delta plan self digest mismatch")
    deltas = plan.get("deltas")
    observations = plan.get("observations")
    submission_rows = plan.get("submission_bindings")
    covered_rows = plan.get("covered_increment_bindings")
    if (
        plan.get("schema") != PLAN_V1
        or plan.get("status") != "planned_offline_only"
        or plan.get("submission_count") != 7
        or plan.get("hh_submission_count") != 6
        or plan.get("hubbard_submission_count") != 1
        or not isinstance(deltas, list)
        or plan.get("delta_count") != len(deltas)
        or not isinstance(observations, list)
        or len(observations) != 7
        or not isinstance(submission_rows, list)
        or len(submission_rows) != 7
        or not isinstance(covered_rows, list)
        or plan.get("covered_increment_count") != len(covered_rows)
        or plan.get("storage_floor_bytes") != DEFAULT_FREE_FLOOR_BYTES
        or plan.get("ram_guard_limit_mb") != 8000
        or plan.get("one_archive_at_a_time") is not True
        or plan.get("remote_raw_evidence_policy") != "preserve"
        or plan.get("qiskit_compilation") is not False
        or plan.get("scheduler_contact_performed") is not False
        or plan.get("ssh_contact_performed") is not False
        or plan.get("submission_authorized") is not False
        or plan.get("production_submission_enabled") is not False
    ):
        raise SeedDeltaAcquisitionError("seed delta plan policy fields diverged")
    if not all(isinstance(row, Mapping) for row in submission_rows):
        raise SeedDeltaAcquisitionError(
            "seed submission binding is not an object"
        )
    if not all(isinstance(row, Mapping) for row in covered_rows):
        raise SeedDeltaAcquisitionError(
            "covered increment binding is not an object"
        )
    submissions = tuple(
        SeedSubmissionBinding.from_json_dict(row) for row in submission_rows
    )
    covered = tuple(
        CoveredSeedIncrement.from_json_dict(row) for row in covered_rows
    )
    hh = tuple(row for row in submissions if row.lane_id == LANE_HH)
    hubbard = tuple(row for row in submissions if row.lane_id == LANE_HUBBARD)
    submission_keys = {(row.cluster_id, row.process_id) for row in submissions}
    covered_keys = {(row.cluster_id, row.process_id) for row in covered}
    if (
        tuple(sorted(submissions)) != submissions
        or tuple(sorted(covered)) != covered
        or len(submission_keys) != 7
        or len(covered_keys) != len(covered)
        or not covered_keys.issubset(submission_keys)
        or len(hh) != 6
        or {row.seed_slug for row in hh} != set(EXPECTED_HH_SEED_SLUGS)
        or len(hubbard) != 1
        or hubbard[0].seed_slug != "l4__u4__half_filled"
        or len({row.unit_id for row in submissions}) != 7
        or len({row.action_id for row in submissions}) != 7
        or len({row.execution_request_sha256 for row in submissions}) != 7
    ):
        raise SeedDeltaAcquisitionError(
            "seed delta plan submission/coverage binding set diverged"
        )
    snapshot_binding = plan.get("snapshot")
    if not isinstance(snapshot_binding, Mapping):
        raise SeedDeltaAcquisitionError("plan snapshot binding is missing")
    snapshot_path = _relative_path(
        snapshot_binding.get("path"), label="plan snapshot path"
    )
    snapshot_sha = _sha(
        snapshot_binding.get("sha256"), label="plan snapshot payload digest"
    )
    snapshot_file_sha = _sha(
        snapshot_binding.get("file_sha256"), label="plan snapshot file digest"
    )
    source = _repo_path(root, snapshot_path, label="plan snapshot", must_exist=True)
    if _file_sha(source) != snapshot_file_sha:
        raise SeedDeltaAcquisitionError("plan snapshot file digest changed")
    snapshot = _read_object(source, label="plan snapshot")
    try:
        observed = validate_combined_snapshot(snapshot)
    except CombinedSnapshotError as exc:
        raise SeedDeltaAcquisitionError(str(exc)) from exc
    if observed != snapshot_sha:
        raise SeedDeltaAcquisitionError("plan snapshot payload digest changed")

    if verify_evidence:
        for binding in submissions:
            for label, path, digest in (
                (
                    "execution request",
                    binding.execution_request_path,
                    binding.execution_request_file_sha256,
                ),
                (
                    "source epoch receipt",
                    binding.source_epoch_receipt_path,
                    binding.source_epoch_receipt_file_sha256,
                ),
                (
                    "submission receipt",
                    binding.submission_receipt_path,
                    binding.submission_receipt_file_sha256,
                ),
                ("submission intent", binding.intent_path, binding.intent_file_sha256),
                (
                    "provisional submission result",
                    binding.provisional_result_path,
                    binding.provisional_result_file_sha256,
                ),
            ):
                _verify_bound_file(
                    root,
                    {"path": path, "sha256": digest},
                    label=f"{binding.seed_id} {label}",
                    require_bytes=False,
                )
        for binding in covered:
            reloaded = _load_covered_seed_increment(
                binding.receipt_path,
                repository_root=root,
                expected_lane=binding.lane_id,
            )
            if reloaded != binding:
                raise SeedDeltaAcquisitionError(
                    "covered seed increment binding changed after replay"
                )

    discovery = discover_seed_deltas(
        snapshot, submissions=submissions, covered=covered
    )
    expected_observations = [
        row.to_json_dict() for row in discovery.observations
    ]
    expected_items = [
        build_delta_plan_item(root, binding) for binding in discovery.deltas
    ]
    if observations != expected_observations or deltas != expected_items:
        raise SeedDeltaAcquisitionError(
            "seed delta plan no longer matches the bound scheduler snapshot"
        )
    for raw_item, binding in zip(deltas, discovery.deltas):
        if not isinstance(raw_item, Mapping):
            raise SeedDeltaAcquisitionError("seed delta plan item is invalid")
        _validate_plan_item(
            raw_item,
            submission_binding=binding,
            repository_root=root,
        )
    return declared


def load_seed_delta_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repo_path(root, path, label="seed delta plan", must_exist=True)
    payload = _read_object(source, label="seed delta plan")
    validate_seed_delta_plan(payload, repository_root=root, verify_evidence=True)
    return payload


def build_seed_delta_acquisition_plan(
    *,
    repository_root: str | Path,
    snapshot_path: str | Path,
    hh_submission_receipts: Sequence[str | Path],
    hubbard_intent_path: str | Path,
    hubbard_provisional_result_path: str | Path,
    hubbard_canonical_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Load all canonical evidence, plan deltas, and publish immutably."""

    root = Path(repository_root).resolve(strict=True)
    if len(hh_submission_receipts) != 6:
        raise SeedDeltaAcquisitionError(
            "plan requires exactly six HH canonical submission receipts"
        )
    hh = tuple(
        load_hh_submission_binding(path, repository_root=root)
        for path in hh_submission_receipts
    )
    hubbard = load_hubbard_submission_binding(
        repository_root=root,
        intent_path=hubbard_intent_path,
        provisional_result_path=hubbard_provisional_result_path,
        canonical_receipt_path=hubbard_canonical_receipt_path,
    )
    plan = assemble_seed_delta_acquisition_plan(
        repository_root=root,
        snapshot_path=snapshot_path,
        submissions=(*hh, hubbard),
        covered=load_covered_seed_increments(repository_root=root),
    )
    write_seed_delta_plan(output_path, plan, repository_root=root)
    return plan


def _validate_plan_item(
    item: Mapping[str, Any],
    *,
    submission_binding: SeedSubmissionBinding | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    payload = dict(item)
    declared = _sha(payload.pop("plan_item_sha256", None), label="plan item digest")
    if declared != _canonical_sha(payload) or payload.get("schema") != PLAN_ITEM_V1:
        raise SeedDeltaAcquisitionError("plan item self digest mismatch")
    if (
        payload.get("lane_id") not in {LANE_HH, LANE_HUBBARD}
        or payload.get("qiskit_compilation") is not False
        or payload.get("submission_authorized") is not False
        or payload.get("production_submission_enabled") is not False
    ):
        raise SeedDeltaAcquisitionError("plan item policy fields are invalid")
    packaging = payload.get("remote_packaging")
    transfer = payload.get("local_transfer")
    ingestion = payload.get("ingestion")
    scheduler = payload.get("scheduler_identity")
    submission = payload.get("submission_identity")
    evidence = payload.get("evidence")
    if not all(
        isinstance(value, Mapping)
        for value in (
            packaging,
            transfer,
            ingestion,
            scheduler,
            submission,
            evidence,
        )
    ):
        raise SeedDeltaAcquisitionError("plan item recipes are incomplete")
    lane_id = str(payload.get("lane_id") or "")
    cluster_id = _positive_int(
        scheduler.get("cluster_id"), label="plan item cluster id", allow_zero=True
    )
    process_id = _positive_int(
        scheduler.get("process_id"), label="plan item process id", allow_zero=True
    )
    expected_handler = (
        "pipelines.time_dynamics.prospective_hh_seed_ingestion:"
        "ingest_hh_seed_delta"
        if lane_id == LANE_HH
        else "pipelines.time_dynamics.prospective_hubbard_seed_ingestion:"
        "ingest_hubbard_seed_delta"
    )
    if (
        packaging.get("schema") != REMOTE_PACKAGING_RECIPE_V1
        or packaging.get("expected_inventory") != list(_EXPECTED_INVENTORY)
        or packaging.get("ordered_archive_members") != list(_EXPECTED_INVENTORY)
        or packaging.get("expected_member_count") != len(_EXPECTED_INVENTORY)
        or packaging.get("preserve_source") is not True
        or packaging.get("forbid_path_escape") is not True
        or packaging.get("forbid_symlinks") is not True
        or transfer.get("schema") != LOCAL_TRANSFER_RECIPE_V1
        or transfer.get("expected_member_count") != len(_EXPECTED_INVENTORY)
        or transfer.get("free_floor_bytes") != DEFAULT_FREE_FLOOR_BYTES
        or transfer.get("ram_guard_limit_mb") != 8000
        or transfer.get("one_archive_at_a_time") is not True
        or transfer.get("part_path") != f"{transfer.get('final_path')}.part"
        or process_id != 0
        or ingestion.get("cluster_id") != cluster_id
        or ingestion.get("process_id") != process_id
        or ingestion.get("handler") != expected_handler
        or ingestion.get("lane_id") != lane_id
        or ingestion.get("seed_id") != payload.get("seed_id")
        or transfer.get("archive_transfer_key")
        != [lane_id, cluster_id, [process_id]]
        or submission.get("unit_id") != ingestion.get("unit_id")
        or submission.get("job_batch_name") != ingestion.get("job_batch_name")
        or submission.get("action_id") != ingestion.get("expected_action_id")
        or submission.get("execution_request_sha256")
        != ingestion.get("expected_execution_request_sha256")
        or submission.get("source_epoch_identity_sha256")
        != ingestion.get("expected_source_epoch_identity_sha256")
        or submission.get("source_epoch_id") != ingestion.get("source_epoch_id")
        or not str(submission.get("unit_id") or "")
        or not str(submission.get("job_batch_name") or "")
        or not str(submission.get("source_epoch_id") or "")
        or payload.get("seed_id")
        != f"{lane_id}::seed::{payload.get('seed_slug')}"
    ):
        raise SeedDeltaAcquisitionError("plan item recipe policy diverged")
    _sha(
        submission.get("execution_request_sha256"),
        label="plan item submission request digest",
    )
    _sha(
        submission.get("source_epoch_identity_sha256"),
        label="plan item submission source digest",
    )
    _absolute_remote(packaging.get("raw_subtree"), label="remote raw subtree")
    remote_path = _absolute_remote(
        packaging.get("archive_path"), label="remote archive path"
    )
    if not remote_path.endswith(".tar.gz") or transfer.get("remote_path") != remote_path:
        raise SeedDeltaAcquisitionError("plan item remote archive binding diverged")
    for field in (
        "final_path",
        "part_path",
        "quarantine_root",
        "recovery_receipt_path",
        "finalization_intent_path",
        "finalization_receipt_path",
    ):
        _relative_path(transfer.get(field), label=f"transfer {field}")
    for field in (
        "increment_root",
        "raw_output_root",
        "execution_request",
        "source_epoch_receipt",
    ):
        _relative_path(ingestion.get(field), label=f"ingestion {field}")
    if submission_binding is not None:
        if repository_root is None:
            raise SeedDeltaAcquisitionError(
                "plan item submission validation requires repository root"
            )
        expected = build_delta_plan_item(repository_root, submission_binding)
        if dict(item) != expected:
            raise SeedDeltaAcquisitionError(
                "plan item does not exactly join its submission/ingestion evidence"
            )
    return dict(item)


def validate_remote_archive_identity(
    plan_item: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate remote size/SHA/gzip/tar/member identity evidence."""

    item = _validate_plan_item(plan_item)
    payload = dict(identity)
    declared = _sha(
        payload.pop("archive_identity_sha256", None),
        label="remote archive identity digest",
    )
    if declared != _canonical_sha(payload):
        raise SeedDeltaAcquisitionError("remote archive identity self digest mismatch")
    packaging = item["remote_packaging"]
    members = payload.get("members")
    if (
        payload.get("schema") != ARCHIVE_IDENTITY_V1
        or payload.get("plan_item_sha256") != item["plan_item_sha256"]
        or payload.get("remote_path") != packaging["archive_path"]
        or payload.get("source_raw_subtree") != packaging["raw_subtree"]
        or payload.get("gzip_integrity") is not True
        or payload.get("tar_integrity") is not True
        or payload.get("remote_raw_preserved") is not True
        or payload.get("tar_member_count") != len(_EXPECTED_INVENTORY)
        or not isinstance(members, Mapping)
        or sorted(members) != list(_EXPECTED_INVENTORY)
    ):
        raise SeedDeltaAcquisitionError(
            "remote archive identity or exact inventory diverged"
        )
    archive_bytes = _positive_int(payload.get("archive_bytes"), label="archive bytes")
    archive_sha = _sha(payload.get("archive_sha256"), label="archive digest")
    normalized_members: dict[str, dict[str, Any]] = {}
    for name in _EXPECTED_INVENTORY:
        row = members[name]
        if not isinstance(row, Mapping) or row.get("type") != "regular":
            raise SeedDeltaAcquisitionError(
                "remote archive inventory requires regular files"
            )
        size = _positive_int(
            row.get("bytes"), label=f"remote member {name} bytes", allow_zero=True
        )
        digest = _sha(row.get("sha256"), label=f"remote member {name} digest")
        normalized_members[name] = {
            "bytes": size,
            "sha256": digest,
            "type": "regular",
        }
    if normalized_members["REMOTE_RAW_PRESERVED"]["bytes"] != len(
        PRESERVATION_MARKER
    ):
        raise SeedDeltaAcquisitionError("remote preservation marker size diverged")
    return {
        **payload,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "members": normalized_members,
        "archive_identity_sha256": declared,
    }


def build_archive_transfer_spec(
    *,
    repository_root: str | Path,
    plan_item: Mapping[str, Any],
    remote_identity: Mapping[str, Any],
) -> ArchiveTransferSpec:
    """Resolve a plan template into the existing guarded-transfer type."""

    root = Path(repository_root).resolve(strict=True)
    item = _validate_plan_item(plan_item)
    identity = validate_remote_archive_identity(item, remote_identity)
    transfer = item["local_transfer"]

    def path(field: str) -> Path:
        return _repo_path(
            root,
            transfer[field],
            label=f"archive transfer {field}",
            must_exist=False,
        )

    return ArchiveTransferSpec(
        remote_path=identity["remote_path"],
        local_final_path=path("final_path"),
        expected_bytes=identity["archive_bytes"],
        expected_sha256=identity["archive_sha256"],
        expected_member_count=len(_EXPECTED_INVENTORY),
        quarantine_root=path("quarantine_root"),
        recovery_receipt_path=path("recovery_receipt_path"),
        finalization_receipt_path=path("finalization_receipt_path"),
        finalization_intent_path=path("finalization_intent_path"),
        extraction_reserve_bytes=sum(
            row["bytes"] for row in identity["members"].values()
        ),
        local_confinement_root=path("final_path").parent,
    )


def _archive_member_inventory(
    archive_path: Path,
    *,
    expected: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or sorted(names) != list(
                _EXPECTED_INVENTORY
            ):
                raise SeedDeltaAcquisitionError(
                    "verified archive inventory is not the exact five files"
                )
            result: dict[str, dict[str, Any]] = {}
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or len(path.parts) != 1
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise SeedDeltaAcquisitionError(
                        "verified archive members must be confined regular files"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise SeedDeltaAcquisitionError(
                        f"verified archive member is unreadable: {member.name}"
                    )
                digest = hashlib.sha256()
                size = 0
                marker = bytearray() if member.name == "REMOTE_RAW_PRESERVED" else None
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    size += len(chunk)
                    digest.update(chunk)
                    if marker is not None:
                        marker.extend(chunk)
                row = {
                    "bytes": size,
                    "sha256": digest.hexdigest(),
                    "type": "regular",
                }
                if row != dict(expected[member.name]):
                    raise SeedDeltaAcquisitionError(
                        f"verified archive member digest changed: {member.name}"
                    )
                if marker is not None and bytes(marker) != PRESERVATION_MARKER:
                    raise SeedDeltaAcquisitionError(
                        "verified archive remote preservation marker is invalid"
                    )
                result[member.name] = row
    except SeedDeltaAcquisitionError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise SeedDeltaAcquisitionError(
            f"verified archive gzip/tar validation failed: {exc}"
        ) from exc
    return {name: result[name] for name in sorted(result)}


def validate_verified_archive(
    spec: ArchiveTransferSpec,
    plan_item: Mapping[str, Any],
    remote_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Reverify the guarded final archive and its exact safe inventory."""

    item = _validate_plan_item(plan_item)
    identity = validate_remote_archive_identity(item, remote_identity)
    expected_spec = build_archive_transfer_spec(
        repository_root=_infer_repository_root(spec, item),
        plan_item=item,
        remote_identity=identity,
    )
    if spec != expected_spec:
        raise SeedDeltaAcquisitionError("archive transfer spec diverges from plan")
    try:
        archive = dict(verify_final_archive(spec))
        receipt = dict(read_finalization_receipt(spec))
    except TransferRecoveryError as exc:
        raise SeedDeltaAcquisitionError(str(exc)) from exc
    if (
        archive.get("archive_bytes") != identity["archive_bytes"]
        or archive.get("archive_sha256") != identity["archive_sha256"]
        or archive.get("tar_member_count") != len(_EXPECTED_INVENTORY)
        or receipt.get("remote_preservation_required") is not True
        or receipt.get("archive_sha256") != identity["archive_sha256"]
    ):
        raise SeedDeltaAcquisitionError("verified archive binding mismatch")
    inventory = _archive_member_inventory(
        spec.local_final_path, expected=identity["members"]
    )
    return {
        **archive,
        "members": inventory,
        "archive_identity_sha256": identity["archive_identity_sha256"],
        "finalization_receipt_sha256": _file_sha(
            spec.effective_finalization_receipt_path
        ),
        "remote_raw_preserved": True,
    }


def _infer_repository_root(
    spec: ArchiveTransferSpec, item: Mapping[str, Any]
) -> Path:
    relative = Path(item["local_transfer"]["final_path"])
    absolute = spec.local_final_path.absolute()
    if len(relative.parts) > len(absolute.parts):
        raise SeedDeltaAcquisitionError("cannot infer archive repository root")
    root = absolute
    for _ in relative.parts:
        root = root.parent
    if (root / relative).absolute() != absolute:
        raise SeedDeltaAcquisitionError("archive repository root binding diverged")
    return root.resolve(strict=True)


def _local_inventory(
    raw_root: Path, expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    try:
        children = tuple(raw_root.iterdir())
    except OSError as exc:
        raise SeedDeltaAcquisitionError("extracted raw root is unreadable") from exc
    if sorted(path.name for path in children) != list(_EXPECTED_INVENTORY):
        raise SeedDeltaAcquisitionError("extracted raw inventory is not exact")
    result: dict[str, dict[str, Any]] = {}
    for path in children:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise SeedDeltaAcquisitionError("extracted raw member is not regular")
        row = {
            "bytes": metadata.st_size,
            "sha256": _file_sha(path),
            "type": "regular",
        }
        if row != dict(expected[path.name]):
            raise SeedDeltaAcquisitionError(
                f"extracted raw member digest changed: {path.name}"
            )
        result[path.name] = row
    if (raw_root / "REMOTE_RAW_PRESERVED").read_bytes() != PRESERVATION_MARKER:
        raise SeedDeltaAcquisitionError("extracted preservation marker is invalid")
    return {name: result[name] for name in sorted(result)}


def _ensure_nofollow_directory(path: Path, *, root: Path) -> None:
    relative = path.absolute().relative_to(root.absolute())
    cursor = root.absolute()
    for component in relative.parts:
        cursor = cursor / component
        if cursor.exists():
            metadata = cursor.lstat()
            if cursor.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise SeedDeltaAcquisitionError(
                    f"extraction path is not a nofollow directory: {cursor}"
                )
        else:
            cursor.mkdir(mode=0o700)
            descriptor = os.open(cursor.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def extract_verified_seed_archive(
    spec: ArchiveTransferSpec,
    plan_item: Mapping[str, Any],
    remote_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Safely extract one guarded archive with idempotent exact replay."""

    item = _validate_plan_item(plan_item)
    identity = validate_remote_archive_identity(item, remote_identity)
    validation = validate_verified_archive(spec, item, identity)
    root = _infer_repository_root(spec, item)
    raw_root = _repo_path(
        root,
        item["ingestion"]["raw_output_root"],
        label="raw extraction root",
        must_exist=False,
    )
    parent = raw_root.parent
    _ensure_nofollow_directory(parent, root=root)
    lock_path = spec.local_final_path.parent / ".seed_delta_extraction.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    temporary: Path | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        enforce_local_storage_floor(
            spec,
            required_write_bytes=0,
            extraction_reserve_bytes=sum(
                row["bytes"] for row in identity["members"].values()
            ),
            free_floor_bytes=DEFAULT_FREE_FLOOR_BYTES,
        )
        if raw_root.exists():
            if raw_root.is_symlink() or not raw_root.is_dir():
                raise SeedDeltaAcquisitionError(
                    "existing raw extraction root is not a nofollow directory"
                )
            inventory = _local_inventory(raw_root, identity["members"])
            return {
                "status": "verified",
                "raw_output_root": raw_root.relative_to(root).as_posix(),
                "members": inventory,
                "archive_sha256": validation["archive_sha256"],
            }
        temporary = Path(
            tempfile.mkdtemp(prefix=".paper_ii_seed_extract.", dir=parent)
        )
        with tarfile.open(spec.local_final_path, mode="r:gz") as archive:
            member_by_name = {member.name: member for member in archive.getmembers()}
            for name in _EXPECTED_INVENTORY:
                member = member_by_name[name]
                source = archive.extractfile(member)
                if source is None:
                    raise SeedDeltaAcquisitionError(
                        f"archive member is unreadable during extraction: {name}"
                    )
                target = temporary / name
                output = os.open(
                    target,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        view = memoryview(chunk)
                        while view:
                            written = os.write(output, view)
                            if written <= 0:
                                raise SeedDeltaAcquisitionError(
                                    "safe extraction write made no progress"
                                )
                            view = view[written:]
                    os.fsync(output)
                finally:
                    os.close(output)
        inventory = _local_inventory(temporary, identity["members"])
        directory = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        try:
            os.rename(temporary, raw_root)
            temporary = None
        except FileExistsError:
            if raw_root.is_symlink() or not raw_root.is_dir():
                raise SeedDeltaAcquisitionError(
                    "concurrent extraction produced an unsafe raw root"
                )
            inventory = _local_inventory(raw_root, identity["members"])
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return {
            "status": "verified",
            "raw_output_root": raw_root.relative_to(root).as_posix(),
            "members": inventory,
            "archive_sha256": validation["archive_sha256"],
        }
    except TransferRecoveryError as exc:
        raise SeedDeltaAcquisitionError(str(exc)) from exc
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _plan_item_by_sha(
    plan: Mapping[str, Any], item_sha256: str
) -> dict[str, Any]:
    expected = _sha(item_sha256, label="requested plan item digest")
    matches = [
        dict(item)
        for item in plan.get("deltas", [])
        if isinstance(item, Mapping)
        and item.get("plan_item_sha256") == expected
    ]
    if len(matches) != 1:
        raise SeedDeltaAcquisitionError(
            "plan must contain exactly one requested delta item"
        )
    return _validate_plan_item(matches[0])


def _load_remote_identity(
    root: Path, path: str | Path, *, item: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    source = _repo_path(
        root, path, label="remote archive identity", must_exist=True
    )
    payload = _read_object(source, label="remote archive identity")
    return source, validate_remote_archive_identity(item, payload)


def _ingest_verified_seed_archive_under_guard(
    *,
    repository_root: str | Path,
    plan_path: str | Path,
    plan_item_sha256: str,
    remote_identity_path: str | Path,
) -> dict[str, Any]:
    """Private worker boundary; callable only beneath the 8000-MB RAM guard."""

    if os.environ.get(_RAM_GUARD_ENV) != "8000":
        raise SeedDeltaAcquisitionError(
            "seed archive ingestion requires the 8000-MB RAM guard"
        )

    root = Path(repository_root).resolve(strict=True)
    plan_source = _repo_path(root, plan_path, label="seed delta plan", must_exist=True)
    plan = load_seed_delta_plan(plan_source, repository_root=root)
    item = _plan_item_by_sha(plan, plan_item_sha256)
    identity_source, identity = _load_remote_identity(
        root, remote_identity_path, item=item
    )
    spec = build_archive_transfer_spec(
        repository_root=root, plan_item=item, remote_identity=identity
    )
    archive = validate_verified_archive(spec, item, identity)
    extraction = extract_verified_seed_archive(spec, item, identity)
    ingestion = item["ingestion"]
    kwargs = {
        "repository_root": root,
        "raw_output_root": ingestion["raw_output_root"],
        "execution_request": ingestion["execution_request"],
        "source_epoch_receipt": ingestion["source_epoch_receipt"],
        "expected_action_id": ingestion["expected_action_id"],
        "expected_execution_request_sha256": ingestion[
            "expected_execution_request_sha256"
        ],
        "expected_source_epoch_identity_sha256": ingestion[
            "expected_source_epoch_identity_sha256"
        ],
        "cluster_id": ingestion["cluster_id"],
        "process_id": ingestion["process_id"],
        "increment_root": ingestion["increment_root"],
    }
    if item["lane_id"] == LANE_HH:
        result = ingest_hh_seed_delta(**kwargs)
        collection_status = result.collection_status
    elif item["lane_id"] == LANE_HUBBARD:
        result = ingest_hubbard_seed_delta(**kwargs)
        collection_status = "complete"
    else:  # pragma: no cover - plan validation excludes this
        raise SeedDeltaAcquisitionError("unsupported ingestion lane")
    expected_increment = (
        Path(ingestion["increment_root"]) / "evidence/increment_receipt.json"
    ).as_posix()
    if result.increment_receipt_path != expected_increment:
        raise SeedDeltaAcquisitionError(
            "lane ingestion returned a noncanonical increment receipt"
        )
    increment_path = _repo_path(
        root,
        result.increment_receipt_path,
        label="seed increment receipt",
        must_exist=True,
    )
    if _file_sha(increment_path) != result.increment_receipt_sha256:
        raise SeedDeltaAcquisitionError("seed increment receipt changed after ingest")
    stable = {
        "schema": ACQUISITION_RECEIPT_V1,
        "status": "verified_and_ingested",
        "lane_id": item["lane_id"],
        "seed_id": item["seed_id"],
        "scheduler_identity": dict(item["scheduler_identity"]),
        "plan": {
            "path": plan_source.relative_to(root).as_posix(),
            "sha256": plan["plan_sha256"],
            "file_sha256": _file_sha(plan_source),
            "plan_item_sha256": item["plan_item_sha256"],
        },
        "remote_archive_identity": {
            "path": identity_source.relative_to(root).as_posix(),
            "sha256": identity["archive_identity_sha256"],
            "file_sha256": _file_sha(identity_source),
        },
        "verified_archive": {
            "path": spec.local_final_path.relative_to(root).as_posix(),
            "bytes": archive["archive_bytes"],
            "sha256": archive["archive_sha256"],
            "member_count": archive["tar_member_count"],
            "finalization_receipt": {
                "path": spec.effective_finalization_receipt_path.relative_to(
                    root
                ).as_posix(),
                "sha256": archive["finalization_receipt_sha256"],
            },
            "remote_preserved": True,
        },
        "extraction": extraction,
        "increment_receipt": {
            "path": result.increment_receipt_path,
            "sha256": result.increment_receipt_sha256,
        },
        "seed_qualification_receipt": {
            "path": result.seed_receipt.receipt_path,
            "sha256": result.seed_receipt.receipt_sha256,
        },
        "collection_status": collection_status,
        "ram_guard_limit_mb": 8000,
        "storage_floor_bytes": DEFAULT_FREE_FLOOR_BYTES,
        "remote_raw_evidence_policy": "preserve",
        "qiskit_compilation": False,
        "scheduler_contact_performed": False,
        "ssh_contact_performed": False,
        "submission_authorized": False,
    }
    receipt_path = (
        _repo_path(
            root,
            ingestion["increment_root"],
            label="seed increment root",
            must_exist=True,
        )
        / "evidence/seed_delta_acquisition_receipt.json"
    )
    if receipt_path.exists():
        existing = _read_object(receipt_path, label="seed delta acquisition receipt")
        unsigned = dict(existing)
        declared = _sha(
            unsigned.pop("acquisition_receipt_sha256", None),
            label="seed delta acquisition receipt digest",
        )
        unsigned.pop("recorded_at", None)
        if declared != _canonical_sha(
            {key: value for key, value in existing.items() if key != "acquisition_receipt_sha256"}
        ) or unsigned != stable:
            raise SeedDeltaAcquisitionError(
                "existing seed delta acquisition receipt diverges"
            )
        return existing
    payload = {**stable, "recorded_at": _utc_now()}
    payload["acquisition_receipt_sha256"] = _canonical_sha(payload)
    _atomic_write_immutable_json(
        receipt_path, payload, label="seed delta acquisition receipt"
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--repository-root", "--repo-root", default=".")
    plan.add_argument("--snapshot", required=True)
    plan.add_argument("--hh-receipt", action="append", required=True)
    plan.add_argument("--hubbard-intent", required=True)
    plan.add_argument("--hubbard-provisional", required=True)
    plan.add_argument("--hubbard-receipt", required=True)
    plan.add_argument("--output", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--repository-root", "--repo-root", default=".")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--item-sha256")
    validate.add_argument("--remote-identity")

    for name in ("ingest-verified-archive", "replay", "_ingest-worker"):
        command = commands.add_parser(
            name, help=argparse.SUPPRESS if name == "_ingest-worker" else None
        )
        command.add_argument("--repository-root", "--repo-root", default=".")
        command.add_argument("--plan", required=True)
        command.add_argument("--item-sha256", required=True)
        command.add_argument("--remote-identity", required=True)
    return parser


def _run_ram_guarded(args: argparse.Namespace) -> int:
    root = Path(args.repository_root).resolve(strict=True)
    guard = root / "pipelines/shell/ram_guard.py"
    if not guard.is_file():
        raise SeedDeltaAcquisitionError("RAM guard is missing")
    argv = [
        sys.executable,
        str(guard),
        "--limit-mb",
        "8000",
        "--",
        sys.executable,
        "-m",
        "pipelines.time_dynamics.runners.acquire_prospective_seed_delta",
        "_ingest-worker",
        "--repository-root",
        str(root),
        "--plan",
        str(args.plan),
        "--item-sha256",
        str(args.item_sha256),
        "--remote-identity",
        str(args.remote_identity),
    ]
    environment = os.environ.copy()
    environment[_RAM_GUARD_ENV] = "8000"
    completed = subprocess.run(
        argv,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            plan = build_seed_delta_acquisition_plan(
                repository_root=args.repository_root,
                snapshot_path=args.snapshot,
                hh_submission_receipts=args.hh_receipt,
                hubbard_intent_path=args.hubbard_intent,
                hubbard_provisional_result_path=args.hubbard_provisional,
                hubbard_canonical_receipt_path=args.hubbard_receipt,
                output_path=args.output,
            )
            summary = {
                "status": "planned_offline_only",
                "plan": str(args.output),
                "plan_sha256": plan["plan_sha256"],
                "delta_count": plan["delta_count"],
                "scheduler_contact_performed": False,
                "ssh_contact_performed": False,
                "submission_authorized": False,
            }
        elif args.command == "validate":
            root = Path(args.repository_root).resolve(strict=True)
            plan_source = _repo_path(
                root, args.plan, label="seed delta plan", must_exist=True
            )
            plan = load_seed_delta_plan(plan_source, repository_root=root)
            summary = {
                "status": "valid",
                "plan": plan_source.relative_to(root).as_posix(),
                "plan_sha256": plan["plan_sha256"],
                "delta_count": plan["delta_count"],
            }
            if (args.item_sha256 is None) != (args.remote_identity is None):
                raise SeedDeltaAcquisitionError(
                    "validate requires both item digest and remote identity"
                )
            if args.item_sha256 is not None:
                item = _plan_item_by_sha(plan, args.item_sha256)
                _, identity = _load_remote_identity(
                    root, args.remote_identity, item=item
                )
                spec = build_archive_transfer_spec(
                    repository_root=root,
                    plan_item=item,
                    remote_identity=identity,
                )
                archive = validate_verified_archive(spec, item, identity)
                summary["archive_sha256"] = archive["archive_sha256"]
                summary["archive_status"] = "verified"
        elif args.command in {"ingest-verified-archive", "replay"}:
            return _run_ram_guarded(args)
        elif args.command == "_ingest-worker":
            receipt = _ingest_verified_seed_archive_under_guard(
                repository_root=args.repository_root,
                plan_path=args.plan,
                plan_item_sha256=args.item_sha256,
                remote_identity_path=args.remote_identity,
            )
            summary = {
                "status": "verified_and_ingested",
                "lane_id": receipt["lane_id"],
                "seed_id": receipt["seed_id"],
                "increment_receipt": receipt["increment_receipt"],
                "acquisition_receipt_sha256": receipt[
                    "acquisition_receipt_sha256"
                ],
                "remote_raw_preserved": True,
                "ram_guard_limit_mb": 8000,
            }
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except SeedDeltaAcquisitionError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            summary,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


__all__ = [
    "ACQUISITION_RECEIPT_V1",
    "ARCHIVE_IDENTITY_V1",
    "CoveredSeedIncrement",
    "PLAN_ITEM_V1",
    "PLAN_V1",
    "SeedDeltaAcquisitionError",
    "SeedDeltaDiscovery",
    "SeedSchedulerObservation",
    "SeedSubmissionBinding",
    "assemble_seed_delta_acquisition_plan",
    "build_archive_transfer_spec",
    "build_delta_plan_item",
    "build_seed_delta_acquisition_plan",
    "discover_seed_deltas",
    "extract_verified_seed_archive",
    "load_covered_seed_increments",
    "load_hh_submission_binding",
    "load_hubbard_submission_binding",
    "load_seed_delta_plan",
    "main",
    "validate_remote_archive_identity",
    "validate_seed_delta_plan",
    "validate_verified_archive",
    "write_seed_delta_plan",
]


if __name__ == "__main__":  # pragma: no cover - runner owns CLI coverage
    raise SystemExit(main())
