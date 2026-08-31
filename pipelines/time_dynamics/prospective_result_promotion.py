"""Offline promotion of verified prospective Paper-II CHTC results.

This module is deliberately downstream of transport.  It never contacts SSH,
Condor, or a submit command.  A promotion starts from an inert, digest-bound
package, its canonical submission receipt, one later combined scheduler
snapshot proving ``JobStatus == 4`` and ``ExitCode == 0``, and a result archive
already finalized by :mod:`guarded_archive_transfer`.

The publication boundary is receipt-last.  Raw files are extracted into an
attempt-scoped immutable directory under the repository's 8-GiB RAM guard,
validated with the producing worker, and copied to their canonical content
addresses without replacement.  Only after every scientific file validates is
the canonical worker completion receipt published.  A durable promotion intent
makes a crash between those operations replayable without overwriting evidence.
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
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.combined_scheduler_snapshot import (
    COMBINED_SCOPED_SNAPSHOT_V1,
    validate_combined_snapshot,
)
from pipelines.time_dynamics.guarded_archive_transfer import (
    ArchiveTransferSpec,
    DEFAULT_FREE_FLOOR_BYTES,
    TransferRecoveryError,
    enforce_local_storage_floor,
    read_finalization_receipt,
    verify_final_archive,
)
from pipelines.time_dynamics.prospective_avqds_scout_chtc_package import (
    validate_frozen_submitted_avqds_prefix_package,
)
from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    AVQDSPrefixCompletionReceipt,
    load_frozen_submitted_work_order as load_frozen_avqds_prefix_work_order,
    validate_raw_run as validate_avqds_prefix_raw_run,
)
from pipelines.time_dynamics.prospective_avqds_trajectory_chtc_package import (
    validate_avqds_trajectory_package,
)
from pipelines.time_dynamics.prospective_avqds_trajectory_worker import (
    AVQDSTrajectoryCompletionReceipt,
    load_work_order as load_avqds_trajectory_work_order,
    validate_raw_run as validate_avqds_trajectory_raw_run,
)
from pipelines.time_dynamics.prospective_bootstrap_activation import (
    BootstrapActivationError,
    load_frozen_submitted_intent,
    load_submission_receipt,
)
from pipelines.time_dynamics.prospective_exact_reference_chtc_package import (
    PreparedExactReferenceActionPackage,
)
from pipelines.time_dynamics.prospective_exact_reference_worker import (
    load_work_order as load_exact_reference_work_order,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExactReferenceReceipt,
    ExecutionRequest,
    ProspectiveExecutionError,
    SeedQualificationReceipt,
    SourceEpochReceipt,
)


MODE_EXACT_REFERENCE = "exact_reference"
MODE_AVQDS_PREFIX = "avqds_prefix"
MODE_AVQDS_TRAJECTORY = "avqds_trajectory"
MODES = frozenset(
    {MODE_EXACT_REFERENCE, MODE_AVQDS_PREFIX, MODE_AVQDS_TRAJECTORY}
)

PROMOTION_PLAN_V1 = "paper_ii_prospective_result_promotion_plan_v1"
REMOTE_ARCHIVE_IDENTITY_V1 = "paper_ii_prospective_result_archive_identity_v1"
PROMOTION_INTENT_V1 = "paper_ii_prospective_result_promotion_intent_v1"
PROMOTION_RECEIPT_V1 = "paper_ii_prospective_result_promotion_receipt_v1"

_SUBMISSION_RECEIPT_V1 = "time_dynamics_submission_receipt_v1"
_SUCCESS_STATUS = 4
_PRESERVATION_MARKER = (
    b"raw evidence preserved; no remote cleanup authorized\n"
)
_BASE_SOURCE_MARKER = (
    b"pristine base source epoch verified before executor overlay\n"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.-]+")
_CHTC_REMOTE_HOME = PurePosixPath("/home/jsstrobel")

_INVENTORIES: Mapping[str, tuple[str, ...]] = {
    MODE_EXACT_REFERENCE: tuple(
        sorted(
            {
                "REMOTE_RAW_PRESERVED",
                "exact_reference_receipt.json",
                "exact_reference_work_order.json",
                "execution_request.json",
                "manifest.json",
                "reference_data.json",
                "seed.json",
                "seed_qualification_receipt.json",
                "source_epoch_receipt.json",
                "worker_stderr.log",
                "worker_stdout.json",
            }
        )
    ),
    MODE_AVQDS_PREFIX: tuple(
        sorted(
            {
                "BASE_SOURCE_VERIFIED_BEFORE_OVERLAY",
                "REMOTE_RAW_PRESERVED",
                "completion_receipt.json",
                "executor_overlay_identity.json",
                "run.json",
                "worker_stderr.log",
                "worker_stdout.json",
            }
        )
    ),
    MODE_AVQDS_TRAJECTORY: tuple(
        sorted(
            {
                "REMOTE_RAW_PRESERVED",
                "completion_receipt.json",
                "run.json",
                "worker_stderr.log",
                "worker_stdout.json",
            }
        )
    ),
}

_RAW_FAMILIES = {
    MODE_EXACT_REFERENCE: "paper_ii_exact_reference",
    MODE_AVQDS_PREFIX: "paper_ii_avqds",
    MODE_AVQDS_TRAJECTORY: "paper_ii_avqds_trajectory",
}


class ResultPromotionError(RuntimeError):
    """Promotion evidence is stale, unsafe, or scientifically invalid."""


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


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(value: Any, *, label: str) -> str:
    token = str(value or "")
    if _SHA256.fullmatch(token) is None:
        raise ResultPromotionError(f"{label} must be a lowercase SHA-256")
    return token


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ResultPromotionError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ResultPromotionError(f"{label} must be an integer") from exc
    if result < minimum:
        raise ResultPromotionError(f"{label} must be >= {minimum}")
    return result


def _aware_time(value: Any, *, label: str) -> dt.datetime:
    try:
        result = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResultPromotionError(f"{label} must be ISO-8601") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ResultPromotionError(f"{label} must include a timezone")
    return result


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _relative(value: Any, *, label: str) -> str:
    token = str(value or "")
    path = Path(token)
    if (
        not token
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != token
    ):
        raise ResultPromotionError(
            f"{label} must be a canonical repository-relative path"
        )
    return token


def _home_relative_remote(value: Any, *, label: str) -> str:
    token = str(value or "")
    path = PurePosixPath(token)
    if (
        not token
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != token
        or any(_SAFE_TOKEN.fullmatch(part) is None for part in path.parts)
    ):
        raise ResultPromotionError(
            f"{label} must be a canonical safe path relative to the remote home"
        )
    return token


def _absolute_remote(value: Any, *, label: str) -> str:
    token = str(value or "")
    path = PurePosixPath(token)
    if not path.is_absolute() or ".." in path.parts or str(path) != token:
        raise ResultPromotionError(
            f"{label} must be a canonical absolute POSIX path"
        )
    return token


def _repo_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        lexical = raw.absolute()
    else:
        if ".." in raw.parts:
            raise ResultPromotionError(f"{label} contains traversal")
        lexical = (root / raw).absolute()
    try:
        relative = lexical.relative_to(root.absolute())
    except ValueError as exc:
        raise ResultPromotionError(f"{label} is outside repository") from exc
    cursor = root.absolute()
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ResultPromotionError(f"{label} traverses a symlink")
    try:
        resolved = lexical.resolve(strict=must_exist)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ResultPromotionError(f"{label} is outside repository") from exc
    return resolved


def _relative_to_root(root: Path, path: Path, *, label: str) -> str:
    try:
        result = path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise ResultPromotionError(f"{label} is outside repository") from exc
    return _relative(result, label=label)


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise ResultPromotionError(f"{label} is not a nofollow regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except ResultPromotionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultPromotionError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ResultPromotionError(f"{label} must be a JSON object")
    return dict(value)


def _verify_file_binding(
    root: Path,
    value: Any,
    *,
    label: str,
    require_bytes: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ResultPromotionError(f"{label} binding is missing")
    relative = _relative(value.get("path"), label=f"{label} path")
    digest = _sha(value.get("sha256"), label=f"{label} digest")
    path = _repo_path(root, relative, label=label, must_exist=True)
    if not path.is_file() or _file_sha(path) != digest:
        raise ResultPromotionError(f"{label} file digest changed")
    result: dict[str, Any] = {"path": relative, "sha256": digest}
    if require_bytes:
        size = _integer(value.get("bytes"), label=f"{label} bytes", minimum=1)
        if path.stat().st_size != size:
            raise ResultPromotionError(f"{label} byte count changed")
        result["bytes"] = size
    return path, result


def _verify_logical_file_binding(
    root: Path,
    value: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, str]]:
    """Verify evidence with distinct logical-payload and serialized-file digests."""

    if not isinstance(value, Mapping):
        raise ResultPromotionError(f"{label} binding is missing")
    relative = _relative(value.get("path"), label=f"{label} path")
    logical_digest = _sha(value.get("sha256"), label=f"{label} logical digest")
    file_digest = _sha(value.get("file_sha256"), label=f"{label} file digest")
    path = _repo_path(root, relative, label=label, must_exist=True)
    if not path.is_file() or _file_sha(path) != file_digest:
        raise ResultPromotionError(f"{label} file digest changed")
    return path, {
        "path": relative,
        "sha256": logical_digest,
        "file_sha256": file_digest,
    }


def _write_immutable(path: Path, content: bytes) -> None:
    """Publish one file atomically and never replace a different inode."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise ResultPromotionError(
                f"refusing to overwrite immutable promotion evidence: {path}"
            )
        return
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publish", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise ResultPromotionError(
                    f"concurrent immutable publication conflicts at {path}"
                )
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, _pretty_bytes(value))


@dataclass(frozen=True)
class _PackageContext:
    mode: str
    request: ExecutionRequest
    order: Any
    source_receipt: SourceEpochReceipt
    seed_receipt: SeedQualificationReceipt
    preflight_path: str
    preflight_sha256: str
    package_path: str
    package_sha256: str
    package_bytes: int
    package_manifest_path: str
    package_manifest_sha256: str
    descriptor_path: str
    descriptor_sha256: str
    work_order_path: str
    work_order_file_sha256: str
    work_order_sha256: str


@dataclass(frozen=True)
class _SubmissionContext:
    receipt_path: str
    receipt_file_sha256: str
    intent_path: str
    intent_sha256: str
    intent_file_sha256: str
    unit_id: str
    job_batch_name: str
    cluster_id: int
    process_id: int
    remote_workdir: str


@dataclass(frozen=True)
class _SuccessContext:
    snapshot_path: str
    snapshot_file_sha256: str
    snapshot_sha256: str
    captured_at: str
    query: str
    scheduler_row: Mapping[str, Any]
    scheduler_row_sha256: str


def _load_package_context(
    root: Path,
    *,
    mode: str,
    preflight_path: str | Path,
    work_order_path: str | Path,
) -> _PackageContext:
    if mode not in MODES:
        raise ResultPromotionError(f"unsupported result mode: {mode}")
    work_path = _repo_path(
        root, work_order_path, label="work order", must_exist=True
    )
    preflight = _repo_path(
        root, preflight_path, label="package preflight", must_exist=True
    )
    try:
        if mode == MODE_EXACT_REFERENCE:
            prepared = PreparedExactReferenceActionPackage.load(
                preflight, repository_root=root
            )
            order, source, seed = load_exact_reference_work_order(
                work_path, repository_root=root
            )
            if (
                prepared.work_order.path
                != _relative_to_root(root, work_path, label="work order")
                or prepared.work_order.sha256 != _file_sha(work_path)
                or prepared.work_order_sha256 != order.work_order_sha256
                or prepared.request_sha256 != order.request.digest
            ):
                raise ResultPromotionError(
                    "exact-reference package/work-order binding diverged"
                )
            preflight_file = prepared.preflight_receipt
            package = prepared.package
            manifest = prepared.package_manifest
            descriptor = prepared.submit_descriptor
            package_bytes = prepared.package_bytes
        elif mode == MODE_AVQDS_PREFIX:
            prepared = validate_frozen_submitted_avqds_prefix_package(
                preflight, repository_root=root
            )
            order, source, seed = load_frozen_avqds_prefix_work_order(
                work_path, repository_root=root
            )
            if (
                prepared.work_order_sha256 != order.work_order_sha256
                or prepared.execution_request_sha256 != order.request.digest
            ):
                raise ResultPromotionError(
                    "AVQDS-prefix package/work-order binding diverged"
                )
            preflight_file = prepared.preflight
            package = prepared.package
            manifest = prepared.package_manifest
            descriptor = prepared.descriptor
            package_bytes = (root / package.path).stat().st_size
        else:
            prepared = validate_avqds_trajectory_package(
                preflight, repository_root=root
            )
            order, source, seed = load_avqds_trajectory_work_order(
                work_path, repository_root=root
            )
            if (
                prepared.work_order_sha256 != order.work_order_sha256
                or prepared.execution_request_sha256 != order.request.digest
            ):
                raise ResultPromotionError(
                    "scientific AVQDS package/work-order binding diverged"
                )
            preflight_file = prepared.preflight
            package = prepared.package
            manifest = prepared.package_manifest
            descriptor = prepared.descriptor
            package_bytes = (root / package.path).stat().st_size
    except ResultPromotionError:
        raise
    except (ValueError, ProspectiveExecutionError, OSError) as exc:
        raise ResultPromotionError(str(exc)) from exc
    if (
        preflight_file.path
        != _relative_to_root(root, preflight, label="package preflight")
        or preflight_file.sha256 != _file_sha(preflight)
        or (root / package.path).stat().st_size != package_bytes
    ):
        raise ResultPromotionError("package preflight file binding diverged")
    return _PackageContext(
        mode=mode,
        request=order.request,
        order=order,
        source_receipt=source,
        seed_receipt=seed,
        preflight_path=preflight_file.path,
        preflight_sha256=preflight_file.sha256,
        package_path=package.path,
        package_sha256=package.sha256,
        package_bytes=package_bytes,
        package_manifest_path=manifest.path,
        package_manifest_sha256=manifest.sha256,
        descriptor_path=descriptor.path,
        descriptor_sha256=descriptor.sha256,
        work_order_path=_relative_to_root(root, work_path, label="work order"),
        work_order_file_sha256=_file_sha(work_path),
        work_order_sha256=order.work_order_sha256,
    )


def _load_submission_context(
    root: Path,
    *,
    submission_receipt_path: str | Path,
    package: _PackageContext,
) -> _SubmissionContext:
    source = _repo_path(
        root,
        submission_receipt_path,
        label="submission receipt",
        must_exist=True,
    )
    try:
        receipt = load_submission_receipt(source, repository_root=root)
    except (BootstrapActivationError, ProspectiveExecutionError, OSError, ValueError) as exc:
        raise ResultPromotionError(
            f"canonical bootstrap submission receipt is invalid: {exc}"
        ) from exc
    remote = receipt.get("remote")
    evidence = remote.get("recovery_evidence") if isinstance(remote, Mapping) else None
    intent_row = evidence.get("intent") if isinstance(evidence, Mapping) else None
    if (
        receipt.get("schema") != _SUBMISSION_RECEIPT_V1
        or receipt.get("lane_id") != package.request.lane_id
        or receipt.get("action_ids") != [package.request.action_id]
        or receipt.get("recovered_from_scheduler") is not True
        or not isinstance(remote, Mapping)
        or not isinstance(intent_row, Mapping)
    ):
        raise ResultPromotionError("submission receipt identity is incomplete")
    intent_path, intent_binding = _verify_logical_file_binding(
        root, intent_row, label="submission intent"
    )
    try:
        intent = load_frozen_submitted_intent(
            intent_path, repository_root=root
        )
        intent_payload = _read_object(intent_path, label="submission intent")
        unit = intent_payload["unit"]
        request = intent.request
        intent_sha = intent.intent_sha256
        intent_file_sha = intent.file_sha256
    except (BootstrapActivationError, ProspectiveExecutionError, OSError, ValueError) as exc:
        raise ResultPromotionError(
            f"canonical bootstrap submission intent is invalid: {exc}"
        ) from exc
    body = unit.get("payload")
    if not isinstance(body, Mapping):
        raise ResultPromotionError("submission intent unit payload is missing")
    package_row = body.get("package")
    preflight_row = body.get("package_preflight")
    descriptor_row = body.get("submit_descriptor_binding")
    same_request = request.to_json_dict() == package.request.to_json_dict()
    if package.mode == MODE_AVQDS_TRAJECTORY:
        same_request = (
            request.digest == package.request.digest
            and request.action_id == package.request.action_id
            and request.lane_id == package.request.lane_id
            and request.kind == package.request.kind
            and request.binding_id == package.request.binding_id
            and request.handler == package.request.handler
            and request.declared_outputs == package.request.declared_outputs
            and request.exact_reference_blind
            == package.request.exact_reference_blind
            and request.execution_id == package.request.execution_id
            and request.attempt == package.request.attempt
            and request.planner_sequence == package.request.planner_sequence
            and request.requested_memory_gib
            == package.request.requested_memory_gib
            and request.arm == package.request.arm
        )
    if (
        not same_request
        or body.get("execution_request_sha256") != package.request.digest
        or body.get("source_epoch_identity_sha256")
        != package.source_receipt.source_epoch.identity_sha256
        or body.get("source_epoch_id")
        != package.source_receipt.source_epoch.source_epoch_id
        or not isinstance(package_row, Mapping)
        or package_row.get("path") != package.package_path
        or package_row.get("sha256") != package.package_sha256
        or int(package_row.get("bytes", -1)) != package.package_bytes
        or not isinstance(preflight_row, Mapping)
        or preflight_row.get("path") != package.preflight_path
        or preflight_row.get("sha256") != package.preflight_sha256
        or not isinstance(descriptor_row, Mapping)
        or descriptor_row.get("path") != package.descriptor_path
        or descriptor_row.get("sha256") != package.descriptor_sha256
        or body.get("remote_raw_evidence_policy") not in (None, "preserve")
    ):
        raise ResultPromotionError(
            "submission lifecycle does not bind the exact package/request/source"
        )
    if (
        intent_binding["sha256"] != intent_sha
        or intent_binding["file_sha256"] != intent_file_sha
        or receipt.get("intent_sha256") != intent_sha
        or remote.get("unit_id") != unit.get("unit_id")
        or remote.get("job_batch_name") != body.get("job_batch_name")
        or remote.get("action_ids") != [package.request.action_id]
        or remote.get("process_ids") != [0]
        or remote.get("source_epoch_identity_sha256")
        != package.source_receipt.source_epoch.identity_sha256
        or remote.get("source_epoch_id")
        != package.source_receipt.source_epoch.source_epoch_id
        or remote.get("remote_raw_evidence_policy") != "preserve"
    ):
        raise ResultPromotionError("submission receipt diverged from its intent")
    cluster = _integer(remote.get("cluster_id"), label="cluster id")
    process = _integer(remote["process_ids"][0], label="process id")
    if process != 0 or remote.get("scheduler_identity") != f"{cluster}.0":
        raise ResultPromotionError("submission receipt is not one process 0")
    remote_workdir = _home_relative_remote(
        body.get("remote_workdir"), label="remote workdir"
    )
    return _SubmissionContext(
        receipt_path=_relative_to_root(root, source, label="submission receipt"),
        receipt_file_sha256=_file_sha(source),
        intent_path=_relative_to_root(root, intent_path, label="submission intent"),
        intent_sha256=intent_sha,
        intent_file_sha256=intent_file_sha,
        unit_id=str(unit.get("unit_id") or ""),
        job_batch_name=str(body.get("job_batch_name") or ""),
        cluster_id=cluster,
        process_id=process,
        remote_workdir=remote_workdir,
    )


def _load_success_context(
    root: Path,
    *,
    success_snapshot_path: str | Path,
    package: _PackageContext,
    submission: _SubmissionContext,
) -> _SuccessContext:
    source = _repo_path(
        root,
        success_snapshot_path,
        label="success scheduler snapshot",
        must_exist=True,
    )
    snapshot = _read_object(source, label="success scheduler snapshot")
    try:
        snapshot_sha = validate_combined_snapshot(snapshot)
    except ValueError as exc:
        raise ResultPromotionError(str(exc)) from exc
    queries = snapshot.get("queries")
    if (
        snapshot.get("schema") != COMBINED_SCOPED_SNAPSHOT_V1
        or snapshot.get("single_combined_snapshot") is not True
        or snapshot.get("polling_performed") is not False
        or not isinstance(queries, Mapping)
    ):
        raise ResultPromotionError("success evidence is not one combined snapshot")
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for name in ("queue", "history"):
        query = queries.get(name)
        rows = query.get("rows") if isinstance(query, Mapping) else None
        if not isinstance(rows, list) or query.get("returncode") != 0:
            raise ResultPromotionError(f"success snapshot {name} query failed")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ResultPromotionError("scheduler row must be an object")
            try:
                identity = (int(raw["ClusterId"]), int(raw["ProcId"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ResultPromotionError(
                    "scheduler row has no cluster/process identity"
                ) from exc
            if identity == (submission.cluster_id, submission.process_id):
                matches.append((name, dict(raw)))
    if len(matches) != 1:
        raise ResultPromotionError(
            "success snapshot must contain exactly one submitted cluster/process"
        )
    query_name, row = matches[0]
    expected = {
        "JobBatchName": submission.job_batch_name,
        "PaperIIActionId": package.request.action_id,
        "PaperIIExecutionRequestSha256": package.request.digest,
        "PaperIISourceEpochId": (
            package.source_receipt.source_epoch.source_epoch_id
        ),
    }
    changed = [name for name, value in expected.items() if row.get(name) != value]
    if changed:
        raise ResultPromotionError(
            "successful scheduler row crosses package identity: "
            + ", ".join(changed)
        )
    if (
        "AcquisitionUnitId" in row
        and row.get("AcquisitionUnitId") != submission.unit_id
    ):
        raise ResultPromotionError(
            "successful scheduler row has a mismatched acquisition-unit identity"
        )
    acquisition_lane = row.get("AcquisitionLaneId")
    paper_lane = row.get("PaperIILaneId")
    if (
        "AcquisitionLaneId" in row
        and acquisition_lane != package.request.lane_id
    ) or (
        "PaperIILaneId" in row and paper_lane != package.request.lane_id
    ):
        raise ResultPromotionError(
            "successful scheduler row has a mismatched acquisition-lane identity"
        )
    if (
        _integer(row.get("JobStatus"), label="successful JobStatus")
        != _SUCCESS_STATUS
        or _integer(row.get("ExitCode"), label="successful ExitCode") != 0
        or query_name != "history"
        or _integer(row.get("CompletionDate"), label="CompletionDate", minimum=1)
        <= 0
    ):
        raise ResultPromotionError(
            "scheduler evidence does not prove terminal zero-exit history"
        )
    if package.mode == MODE_AVQDS_TRAJECTORY:
        trajectory_expected = {
            "PaperIICellId": package.order.bindings.cell_id,
            "PaperIIBindingsSha256": package.order.bindings.digest,
            "PaperIISourceEpochIdentitySha256": (
                package.order.bindings.source_epoch_identity_sha256
            ),
            "PaperIIAVQDSExecutorOverlayRevision": (
                package.order.executor_overlay.overlay_revision
            ),
            "PaperIIAVQDSExecutorOverlaySha256": (
                package.order.executor_overlay.identity_sha256
            ),
            "PaperIICompositeSourceSha256": (
                package.order.executor_overlay.composite_source_identity_sha256
            ),
            "PaperIIHorizon": package.request.inputs["horizon"],
        }
        changed = [
            name
            for name, value in trajectory_expected.items()
            if row.get(name) != value
        ]
        if changed or int(row.get("PaperIILadderIndex", -1)) != int(
            package.request.inputs["ladder_index"]
        ):
            raise ResultPromotionError(
                "scientific AVQDS scheduler identity diverged: "
                + ", ".join(changed or ["PaperIILadderIndex"])
            )
    captured_at = str(snapshot.get("captured_at") or "")
    _aware_time(captured_at, label="success snapshot captured_at")
    return _SuccessContext(
        snapshot_path=_relative_to_root(root, source, label="success snapshot"),
        snapshot_file_sha256=_file_sha(source),
        snapshot_sha256=snapshot_sha,
        captured_at=captured_at,
        query=query_name,
        scheduler_row=row,
        scheduler_row_sha256=_canonical_sha(dict(row)),
    )


def _raw_token(package: _PackageContext) -> str:
    if package.mode == MODE_AVQDS_TRAJECTORY:
        token = package.request.digest[:20]
    else:
        token = (
            f"{package.request.lane_id}-{package.request.digest[:20]}"
        ).replace("_", "-")
    if _SAFE_TOKEN.fullmatch(token) is None:
        raise ResultPromotionError("derived result token is unsafe")
    return token


def _plan_core(
    *,
    package: _PackageContext,
    submission: _SubmissionContext,
    success: _SuccessContext,
) -> dict[str, Any]:
    token = _raw_token(package)
    remote_workdir = _CHTC_REMOTE_HOME / submission.remote_workdir
    raw_subtree = (
        remote_workdir
        / "raw_outputs"
        / _RAW_FAMILIES[package.mode]
        / token
    )
    remote_archive = (
        remote_workdir
        / "transfer_archives"
        / "prospective_results"
        / f"{token}.tar.gz"
    )
    increment = (
        Path("output")
        / package.request.lane_id
        / "attempts"
        / f"cluster_{submission.cluster_id}"
        / "increments"
        / f"proc0_{package.mode}_{package.request.digest[:16]}"
    )
    archive = increment / "archives" / f"{token}.tar.gz"
    raw_root = increment / "raw" / _RAW_FAMILIES[package.mode]
    evidence = increment / "evidence"
    inventory = list(_INVENTORIES[package.mode])
    return {
        "schema": PROMOTION_PLAN_V1,
        "status": "ready_for_verified_local_archive",
        "mode": package.mode,
        "lane_id": package.request.lane_id,
        "cell_id": package.request.inputs.get("cell_id"),
        "action_id": package.request.action_id,
        "execution_request_sha256": package.request.digest,
        "work_order_sha256": package.work_order_sha256,
        "source_epoch_id": package.source_receipt.source_epoch.source_epoch_id,
        "source_epoch_identity_sha256": (
            package.source_receipt.source_epoch.identity_sha256
        ),
        "scheduler_identity": {
            "cluster_id": submission.cluster_id,
            "process_id": submission.process_id,
            "job_batch_name": submission.job_batch_name,
        },
        "package_evidence": {
            "preflight": {
                "path": package.preflight_path,
                "sha256": package.preflight_sha256,
            },
            "package": {
                "path": package.package_path,
                "sha256": package.package_sha256,
                "bytes": package.package_bytes,
            },
            "package_manifest": {
                "path": package.package_manifest_path,
                "sha256": package.package_manifest_sha256,
            },
            "submit_descriptor": {
                "path": package.descriptor_path,
                "sha256": package.descriptor_sha256,
            },
            "work_order": {
                "path": package.work_order_path,
                "sha256": package.work_order_file_sha256,
            },
        },
        "submission_evidence": {
            "receipt": {
                "path": submission.receipt_path,
                "sha256": submission.receipt_file_sha256,
            },
            "intent": {
                "path": submission.intent_path,
                "sha256": submission.intent_sha256,
                "file_sha256": submission.intent_file_sha256,
            },
            "unit_id": submission.unit_id,
        },
        "success_evidence": {
            "snapshot": {
                "path": success.snapshot_path,
                "sha256": success.snapshot_file_sha256,
                "snapshot_sha256": success.snapshot_sha256,
            },
            "captured_at": success.captured_at,
            "query": success.query,
            "scheduler_row_sha256": success.scheduler_row_sha256,
        },
        "remote_packaging": {
            "raw_subtree": str(raw_subtree),
            "archive_path": str(remote_archive),
            "archive_temporary_path": f"{remote_archive}.building.part",
            "expected_inventory": inventory,
            "expected_member_count": len(inventory),
            "regular_files_only": True,
            "forbid_links": True,
            "preserve_source": True,
            "remote_raw_evidence_policy": "preserve",
        },
        "local_transfer": {
            "final_path": archive.as_posix(),
            "part_path": f"{archive.as_posix()}.part",
            "quarantine_root": (archive.parent / "quarantine").as_posix(),
            "recovery_receipt_path": (
                archive.parent / f"{archive.name}.transfer_recovery_receipt.json"
            ).as_posix(),
            "finalization_intent_path": (
                archive.parent / f"{archive.name}.transfer_finalization_intent.json"
            ).as_posix(),
            "finalization_receipt_path": (
                archive.parent / f"{archive.name}.transfer_finalization_receipt.json"
            ).as_posix(),
            "free_floor_bytes": DEFAULT_FREE_FLOOR_BYTES,
            "ram_guard_limit_mb": 8000,
            "remote_prefix_proof_required_for_resume": True,
            "one_archive_at_a_time": True,
        },
        "promotion": {
            "increment_root": increment.as_posix(),
            "raw_output_root": raw_root.as_posix(),
            "intent_path": (evidence / "promotion_intent.json").as_posix(),
            "receipt_path": (evidence / "promotion_receipt.json").as_posix(),
            "receipt_last_publication": True,
            "immutable_no_overwrite": True,
        },
        "scheduler_contacted": False,
        "submission_authorized": False,
        "production_submission_enabled": False,
        "remote_raw_evidence_policy": "preserve",
        "remote_cleanup_authorized": False,
        "qiskit_compilation": False,
    }


def build_promotion_plan(
    *,
    repository_root: str | Path,
    mode: str,
    package_preflight_path: str | Path,
    work_order_path: str | Path,
    submission_receipt_path: str | Path,
    success_snapshot_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build one immutable local promotion plan from durable evidence only."""

    root = Path(repository_root).resolve(strict=True)
    package = _load_package_context(
        root,
        mode=mode,
        preflight_path=package_preflight_path,
        work_order_path=work_order_path,
    )
    submission = _load_submission_context(
        root,
        submission_receipt_path=submission_receipt_path,
        package=package,
    )
    success = _load_success_context(
        root,
        success_snapshot_path=success_snapshot_path,
        package=package,
        submission=submission,
    )
    core = _plan_core(package=package, submission=submission, success=success)
    plan = {**core, "promotion_plan_sha256": _canonical_sha(core)}
    output = _repo_path(
        root, output_path, label="promotion plan output", must_exist=False
    )
    _write_immutable_json(output, plan)
    return load_promotion_plan(output, repository_root=root)


def _validate_plan_shape(payload: Mapping[str, Any]) -> None:
    core = dict(payload)
    declared = _sha(
        core.pop("promotion_plan_sha256", None), label="promotion plan digest"
    )
    if declared != _canonical_sha(core) or core.get("schema") != PROMOTION_PLAN_V1:
        raise ResultPromotionError("promotion plan self digest mismatch")
    mode = str(core.get("mode") or "")
    inventory = _INVENTORIES.get(mode)
    packaging = core.get("remote_packaging")
    transfer = core.get("local_transfer")
    promotion = core.get("promotion")
    if (
        inventory is None
        or core.get("status") != "ready_for_verified_local_archive"
        or core.get("scheduler_contacted") is not False
        or core.get("submission_authorized") is not False
        or core.get("production_submission_enabled") is not False
        or core.get("remote_raw_evidence_policy") != "preserve"
        or core.get("remote_cleanup_authorized") is not False
        or core.get("qiskit_compilation") is not False
        or not all(
            isinstance(row, Mapping)
            for row in (packaging, transfer, promotion)
        )
    ):
        raise ResultPromotionError("promotion plan policy fields diverged")
    if (
        packaging.get("expected_inventory") != list(inventory)
        or packaging.get("expected_member_count") != len(inventory)
        or packaging.get("preserve_source") is not True
        or packaging.get("remote_raw_evidence_policy") != "preserve"
        or packaging.get("regular_files_only") is not True
        or packaging.get("forbid_links") is not True
        or transfer.get("free_floor_bytes") != DEFAULT_FREE_FLOOR_BYTES
        or transfer.get("ram_guard_limit_mb") != 8000
        or transfer.get("one_archive_at_a_time") is not True
        or transfer.get("part_path") != f"{transfer.get('final_path')}.part"
        or promotion.get("receipt_last_publication") is not True
        or promotion.get("immutable_no_overwrite") is not True
    ):
        raise ResultPromotionError("promotion plan transfer policy diverged")
    remote_archive = _absolute_remote(
        packaging.get("archive_path"), label="remote archive"
    )
    _absolute_remote(packaging.get("raw_subtree"), label="remote raw subtree")
    if not remote_archive.endswith(".tar.gz"):
        raise ResultPromotionError("remote result archive is not tar.gz")
    for name in (
        "final_path",
        "part_path",
        "quarantine_root",
        "recovery_receipt_path",
        "finalization_intent_path",
        "finalization_receipt_path",
    ):
        _relative(transfer.get(name), label=f"transfer {name}")
    for name in (
        "increment_root",
        "raw_output_root",
        "intent_path",
        "receipt_path",
    ):
        _relative(promotion.get(name), label=f"promotion {name}")


def load_promotion_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Reload a plan and reauthenticate every package/scheduler dependency."""

    root = Path(repository_root).resolve(strict=True)
    source = _repo_path(root, path, label="promotion plan", must_exist=True)
    payload = _read_object(source, label="promotion plan")
    _validate_plan_shape(payload)
    package_rows = payload.get("package_evidence")
    submission_rows = payload.get("submission_evidence")
    success_rows = payload.get("success_evidence")
    if not all(
        isinstance(row, Mapping)
        for row in (package_rows, submission_rows, success_rows)
    ):
        raise ResultPromotionError("promotion plan evidence bindings are missing")
    preflight_path, _ = _verify_file_binding(
        root, package_rows.get("preflight"), label="package preflight"
    )
    work_order_path, _ = _verify_file_binding(
        root, package_rows.get("work_order"), label="work order"
    )
    receipt_path, _ = _verify_file_binding(
        root, submission_rows.get("receipt"), label="submission receipt"
    )
    snapshot_row = success_rows.get("snapshot")
    snapshot_path, _ = _verify_file_binding(
        root, snapshot_row, label="success snapshot"
    )
    package = _load_package_context(
        root,
        mode=str(payload["mode"]),
        preflight_path=preflight_path,
        work_order_path=work_order_path,
    )
    submission = _load_submission_context(
        root, submission_receipt_path=receipt_path, package=package
    )
    success = _load_success_context(
        root,
        success_snapshot_path=snapshot_path,
        package=package,
        submission=submission,
    )
    expected = _plan_core(package=package, submission=submission, success=success)
    if payload != {**expected, "promotion_plan_sha256": _canonical_sha(expected)}:
        raise ResultPromotionError("promotion plan changed from its source evidence")
    return dict(payload)


def validate_remote_archive_identity(
    plan: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a remote archive identity with exact per-member digests."""

    _validate_plan_shape(plan)
    payload = dict(identity)
    declared = _sha(
        payload.pop("archive_identity_sha256", None),
        label="archive identity digest",
    )
    if declared != _canonical_sha(payload):
        raise ResultPromotionError("remote archive identity self digest mismatch")
    packaging = plan["remote_packaging"]
    expected_names = _INVENTORIES[str(plan["mode"])]
    members = payload.get("members")
    if (
        payload.get("schema") != REMOTE_ARCHIVE_IDENTITY_V1
        or payload.get("promotion_plan_sha256")
        != plan.get("promotion_plan_sha256")
        or payload.get("remote_path") != packaging["archive_path"]
        or payload.get("source_raw_subtree") != packaging["raw_subtree"]
        or payload.get("gzip_integrity") is not True
        or payload.get("tar_integrity") is not True
        or payload.get("remote_raw_preserved") is not True
        or payload.get("remote_cleanup_performed") is not False
        or payload.get("tar_member_count") != len(expected_names)
        or not isinstance(members, Mapping)
        or tuple(sorted(members)) != expected_names
    ):
        raise ResultPromotionError("remote archive identity/inventory diverged")
    archive_bytes = _integer(
        payload.get("archive_bytes"), label="archive bytes", minimum=1
    )
    archive_sha = _sha(payload.get("archive_sha256"), label="archive digest")
    normalized: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        row = members[name]
        if not isinstance(row, Mapping) or row.get("type") != "regular":
            raise ResultPromotionError("archive members must be regular files")
        normalized[name] = {
            "bytes": _integer(
                row.get("bytes"), label=f"member {name} bytes", minimum=0
            ),
            "sha256": _sha(
                row.get("sha256"), label=f"member {name} digest"
            ),
            "type": "regular",
        }
    if (
        normalized["REMOTE_RAW_PRESERVED"]["bytes"]
        != len(_PRESERVATION_MARKER)
        or (
            str(plan["mode"]) == MODE_AVQDS_PREFIX
            and normalized["BASE_SOURCE_VERIFIED_BEFORE_OVERLAY"]["bytes"]
            != len(_BASE_SOURCE_MARKER)
        )
    ):
        raise ResultPromotionError("remote preservation marker size diverged")
    return {
        **payload,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "members": normalized,
        "archive_identity_sha256": declared,
    }


def build_archive_transfer_spec(
    *,
    repository_root: str | Path,
    plan: Mapping[str, Any],
    remote_identity: Mapping[str, Any],
) -> ArchiveTransferSpec:
    root = Path(repository_root).resolve(strict=True)
    _validate_plan_shape(plan)
    identity = validate_remote_archive_identity(plan, remote_identity)
    transfer = plan["local_transfer"]

    def path(name: str) -> Path:
        return _repo_path(
            root,
            transfer[name],
            label=f"archive transfer {name}",
            must_exist=False,
        )

    return ArchiveTransferSpec(
        remote_path=identity["remote_path"],
        local_final_path=path("final_path"),
        expected_bytes=identity["archive_bytes"],
        expected_sha256=identity["archive_sha256"],
        expected_member_count=len(_INVENTORIES[str(plan["mode"])]),
        quarantine_root=path("quarantine_root"),
        recovery_receipt_path=path("recovery_receipt_path"),
        finalization_receipt_path=path("finalization_receipt_path"),
        finalization_intent_path=path("finalization_intent_path"),
        extraction_reserve_bytes=sum(
            int(row["bytes"]) for row in identity["members"].values()
        ),
        local_confinement_root=path("final_path").parent,
    )


def _archive_inventory(
    archive_path: Path,
    *,
    expected: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or tuple(sorted(names)) != tuple(
                sorted(expected)
            ):
                raise ResultPromotionError("archive exact inventory diverged")
            for member in members:
                posix = PurePosixPath(member.name)
                if (
                    posix.is_absolute()
                    or ".." in posix.parts
                    or len(posix.parts) != 1
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise ResultPromotionError(
                        "archive members must be confined regular files"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise ResultPromotionError(
                        f"archive member is unreadable: {member.name}"
                    )
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    size += len(chunk)
                    digest.update(chunk)
                row = {
                    "bytes": size,
                    "sha256": digest.hexdigest(),
                    "type": "regular",
                }
                if row != dict(expected[member.name]):
                    raise ResultPromotionError(
                        f"archive member digest changed: {member.name}"
                    )
                result[member.name] = row
    except ResultPromotionError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ResultPromotionError(f"archive gzip/tar validation failed: {exc}") from exc
    return {name: result[name] for name in sorted(result)}


def validate_finalized_archive(
    *,
    repository_root: str | Path,
    plan: Mapping[str, Any],
    remote_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate final size/SHA/gzip/tar and the guarded-transfer receipt."""

    root = Path(repository_root).resolve(strict=True)
    identity = validate_remote_archive_identity(plan, remote_identity)
    spec = build_archive_transfer_spec(
        repository_root=root, plan=plan, remote_identity=identity
    )
    try:
        archive = dict(verify_final_archive(spec))
        finalization = dict(read_finalization_receipt(spec))
    except TransferRecoveryError as exc:
        raise ResultPromotionError(str(exc)) from exc
    if (
        archive.get("archive_bytes") != identity["archive_bytes"]
        or archive.get("archive_sha256") != identity["archive_sha256"]
        or archive.get("tar_member_count") != len(identity["members"])
        or finalization.get("archive_sha256") != identity["archive_sha256"]
        or finalization.get("archive_bytes") != identity["archive_bytes"]
        or finalization.get("remote_preservation_required") is not True
        or finalization.get("gzip_integrity") is not True
        or finalization.get("tar_integrity") is not True
    ):
        raise ResultPromotionError("guarded finalization binding diverged")
    inventory = _archive_inventory(
        spec.local_final_path, expected=identity["members"]
    )
    return {
        **archive,
        "members": inventory,
        "archive_identity_sha256": identity["archive_identity_sha256"],
        "finalization_receipt": {
            "path": _relative_to_root(
                root,
                spec.effective_finalization_receipt_path,
                label="finalization receipt",
            ),
            "sha256": _file_sha(spec.effective_finalization_receipt_path),
        },
        "remote_raw_preserved": True,
    }


def _ensure_directory(path: Path, *, root: Path) -> None:
    relative = path.absolute().relative_to(root.absolute())
    cursor = root.absolute()
    for component in relative.parts:
        cursor = cursor / component
        if cursor.exists():
            metadata = cursor.lstat()
            if cursor.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise ResultPromotionError(
                    f"promotion directory is unsafe: {cursor}"
                )
        else:
            cursor.mkdir(mode=0o700)
            descriptor = os.open(cursor.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _local_inventory(
    raw_root: Path, expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ResultPromotionError("raw extraction root is unsafe")
    children = tuple(raw_root.iterdir())
    if tuple(sorted(path.name for path in children)) != tuple(sorted(expected)):
        raise ResultPromotionError("extracted exact inventory diverged")
    result: dict[str, dict[str, Any]] = {}
    for path in children:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ResultPromotionError("extracted member is not a regular file")
        row = {
            "bytes": metadata.st_size,
            "sha256": _file_sha(path),
            "type": "regular",
        }
        if row != dict(expected[path.name]):
            raise ResultPromotionError(
                f"extracted member digest changed: {path.name}"
            )
        result[path.name] = row
    if (raw_root / "REMOTE_RAW_PRESERVED").read_bytes() != _PRESERVATION_MARKER:
        raise ResultPromotionError("remote preservation marker content diverged")
    if (
        "BASE_SOURCE_VERIFIED_BEFORE_OVERLAY" in expected
        and (raw_root / "BASE_SOURCE_VERIFIED_BEFORE_OVERLAY").read_bytes()
        != _BASE_SOURCE_MARKER
    ):
        raise ResultPromotionError("base-source verification marker diverged")
    return {name: result[name] for name in sorted(result)}


def _extract_verified_archive(
    *,
    repository_root: Path,
    plan: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Child-process extraction implementation used below the RAM guard."""

    root = repository_root.resolve(strict=True)
    validation = validate_finalized_archive(
        repository_root=root, plan=plan, remote_identity=identity
    )
    spec = build_archive_transfer_spec(
        repository_root=root, plan=plan, remote_identity=identity
    )
    raw_root = _repo_path(
        root,
        plan["promotion"]["raw_output_root"],
        label="raw output root",
        must_exist=False,
    )
    parent = raw_root.parent
    _ensure_directory(parent, root=root)
    lock = parent / ".result_extraction.lock"
    descriptor = os.open(
        lock,
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
                int(row["bytes"]) for row in identity["members"].values()
            ),
            free_floor_bytes=DEFAULT_FREE_FLOOR_BYTES,
        )
        if raw_root.exists():
            inventory = _local_inventory(raw_root, identity["members"])
            return {
                "status": "verified_extracted",
                "raw_output_root": raw_root.relative_to(root).as_posix(),
                "members": inventory,
                "archive_sha256": validation["archive_sha256"],
            }
        temporary = Path(
            tempfile.mkdtemp(prefix=".prospective_result_extract.", dir=parent)
        )
        with tarfile.open(spec.local_final_path, mode="r:gz") as archive:
            by_name = {member.name: member for member in archive.getmembers()}
            for name in _INVENTORIES[str(plan["mode"])]:
                member = by_name[name]
                source = archive.extractfile(member)
                if source is None:
                    raise ResultPromotionError(
                        f"archive member vanished during extraction: {name}"
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
                                raise ResultPromotionError(
                                    "safe extraction write made no progress"
                                )
                            view = view[written:]
                    os.fsync(output)
                finally:
                    os.close(output)
        inventory = _local_inventory(temporary, identity["members"])
        temporary_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(temporary_descriptor)
        finally:
            os.close(temporary_descriptor)
        try:
            os.rename(temporary, raw_root)
            temporary = None
        except FileExistsError:
            inventory = _local_inventory(raw_root, identity["members"])
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return {
            "status": "verified_extracted",
            "raw_output_root": raw_root.relative_to(root).as_posix(),
            "members": inventory,
            "archive_sha256": validation["archive_sha256"],
        }
    except TransferRecoveryError as exc:
        raise ResultPromotionError(str(exc)) from exc
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _extract_under_ram_guard(
    *,
    repository_root: Path,
    plan_path: Path,
    identity_path: Path,
) -> dict[str, Any]:
    guard = repository_root / "pipelines/shell/ram_guard.py"
    argv = [
        sys.executable,
        str(guard),
        "--limit-mb",
        "8000",
        "--",
        sys.executable,
        "-m",
        "pipelines.time_dynamics.prospective_result_promotion",
        "_extract",
        "--repo-root",
        str(repository_root),
        "--plan",
        _relative_to_root(repository_root, plan_path, label="promotion plan"),
        "--remote-identity",
        _relative_to_root(
            repository_root, identity_path, label="remote archive identity"
        ),
    ]
    completed = subprocess.run(
        argv,
        cwd=repository_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ResultPromotionError(
            f"RAM-guarded result extraction failed: {detail[:1000]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ResultPromotionError(
            "RAM-guarded extraction returned unreadable evidence"
        ) from exc
    if not isinstance(value, dict) or value.get("status") != "verified_extracted":
        raise ResultPromotionError("RAM-guarded extraction did not verify output")
    return value


def _copy_dependency(root: Path, mirror: Path, relative: str) -> None:
    source = _repo_path(root, relative, label="validation dependency", must_exist=True)
    target = mirror / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(target, source.read_bytes())


def _validate_worker_outputs(
    root: Path,
    package: _PackageContext,
    raw_root: Path,
) -> tuple[dict[str, str], tuple[tuple[Path, str], ...]]:
    """Validate raw science before exposing any canonical completion receipt."""

    mode = package.mode
    order = package.order
    seed = package.seed_receipt
    if mode == MODE_EXACT_REFERENCE:
        request_payload = _read_object(
            raw_root / "execution_request.json", label="raw execution request"
        )
        if request_payload != package.request.to_json_dict():
            raise ResultPromotionError("raw exact execution request diverged")
        if (
            (raw_root / "exact_reference_work_order.json").read_bytes()
            != (root / package.work_order_path).read_bytes()
            or (raw_root / "source_epoch_receipt.json").read_bytes()
            != (root / package.source_receipt.receipt_path).read_bytes()
            or (raw_root / "seed_qualification_receipt.json").read_bytes()
            != (root / seed.receipt_path).read_bytes()
            or (raw_root / "seed.json").read_bytes()
            != (root / seed.artifact.path).read_bytes()
        ):
            raise ResultPromotionError("raw exact package inputs diverged")
        manifest_path = str(package.request.inputs["artifact_path"])
        data_path = (Path(manifest_path).with_name("reference_data.json")).as_posix()
        receipt_path = (
            Path(manifest_path).with_name("exact_reference_receipt.json")
        ).as_posix()
        mapping = (
            (raw_root / "reference_data.json", data_path),
            (raw_root / "manifest.json", manifest_path),
            (raw_root / "exact_reference_receipt.json", receipt_path),
        )
        with tempfile.TemporaryDirectory(
            prefix=".exact_result_validation.", dir=raw_root.parent
        ) as temporary:
            mirror = Path(temporary)
            _copy_dependency(root, mirror, seed.receipt_path)
            _copy_dependency(root, mirror, seed.artifact.path)
            for source, destination in mapping:
                target = mirror / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_immutable(target, source.read_bytes())
            try:
                ExactReferenceReceipt.load(
                    mirror / receipt_path,
                    repository_root=mirror,
                    expected_reference=order.reference,
                    expected_seed_identity=order.seed,
                    expected_seed_receipt=seed,
                    expected_source_epoch=package.source_receipt.source_epoch,
                )
            except ProspectiveExecutionError as exc:
                raise ResultPromotionError(str(exc)) from exc
        canonical = tuple(mapping)
    elif mode == MODE_AVQDS_PREFIX:
        overlay = _read_object(
            raw_root / "executor_overlay_identity.json",
            label="raw AVQDS executor overlay",
        )
        if overlay != order.executor_overlay.to_json_dict():
            raise ResultPromotionError("raw AVQDS executor overlay diverged")
        try:
            validate_avqds_prefix_raw_run(
                raw_root / "run.json",
                order=order,
                seed_receipt=seed,
                source_epoch=package.source_receipt.source_epoch,
            )
        except ProspectiveExecutionError as exc:
            raise ResultPromotionError(str(exc)) from exc
        mapping = (
            (raw_root / "run.json", order.run_path),
            (raw_root / "completion_receipt.json", order.completion_receipt_path),
        )
        with tempfile.TemporaryDirectory(
            prefix=".avqds_prefix_validation.", dir=raw_root.parent
        ) as temporary:
            mirror = Path(temporary)
            for source, destination in mapping:
                target = mirror / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_immutable(target, source.read_bytes())
            try:
                AVQDSPrefixCompletionReceipt.load(
                    mirror / order.completion_receipt_path,
                    repository_root=mirror,
                    expected_order=order,
                    expected_seed_receipt=seed,
                    expected_source_epoch=package.source_receipt.source_epoch,
                )
            except ProspectiveExecutionError as exc:
                raise ResultPromotionError(str(exc)) from exc
        canonical = tuple(mapping)
    else:
        try:
            validate_avqds_trajectory_raw_run(
                raw_root / "run.json", order=order, seed_receipt=seed
            )
        except ProspectiveExecutionError as exc:
            raise ResultPromotionError(str(exc)) from exc
        mapping = (
            (raw_root / "run.json", order.run_path),
            (raw_root / "completion_receipt.json", order.completion_receipt_path),
        )
        with tempfile.TemporaryDirectory(
            prefix=".avqds_trajectory_validation.", dir=raw_root.parent
        ) as temporary:
            mirror = Path(temporary)
            for source, destination in mapping:
                target = mirror / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_immutable(target, source.read_bytes())
            try:
                AVQDSTrajectoryCompletionReceipt.load(
                    mirror / order.completion_receipt_path,
                    repository_root=mirror,
                    expected_order=order,
                    expected_seed_receipt=seed,
                )
            except ProspectiveExecutionError as exc:
                raise ResultPromotionError(str(exc)) from exc
        canonical = tuple(mapping)
    digests = {
        destination: _file_sha(source) for source, destination in canonical
    }
    return digests, canonical


def _validate_canonical_outputs(root: Path, package: _PackageContext) -> Any:
    order = package.order
    try:
        if package.mode == MODE_EXACT_REFERENCE:
            receipt_path = Path(str(package.request.inputs["artifact_path"])).with_name(
                "exact_reference_receipt.json"
            )
            return ExactReferenceReceipt.load(
                root / receipt_path,
                repository_root=root,
                expected_reference=order.reference,
                expected_seed_identity=order.seed,
                expected_seed_receipt=package.seed_receipt,
                expected_source_epoch=package.source_receipt.source_epoch,
            )
        if package.mode == MODE_AVQDS_PREFIX:
            return AVQDSPrefixCompletionReceipt.load(
                root / order.completion_receipt_path,
                repository_root=root,
                expected_order=order,
                expected_seed_receipt=package.seed_receipt,
                expected_source_epoch=package.source_receipt.source_epoch,
            )
        return AVQDSTrajectoryCompletionReceipt.load(
            root / order.completion_receipt_path,
            repository_root=root,
            expected_order=order,
            expected_seed_receipt=package.seed_receipt,
        )
    except ProspectiveExecutionError as exc:
        raise ResultPromotionError(str(exc)) from exc


def _receipt_core_matches(
    existing: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    left = dict(existing)
    right = dict(expected)
    left.pop("recorded_at", None)
    right.pop("recorded_at", None)
    left.pop("promotion_receipt_sha256", None)
    right.pop("promotion_receipt_sha256", None)
    return left == right


def _validate_recorded_canonical_artifacts(
    root: Path, receipt: Mapping[str, Any]
) -> None:
    """Recheck immutable public bytes before accepting a receipt replay."""

    rows = receipt.get("canonical_artifacts")
    if not isinstance(rows, list) or not rows:
        raise ResultPromotionError("promotion receipt has no canonical artifacts")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _path, binding = _verify_file_binding(
            root, row, label=f"canonical artifact {index}"
        )
        if binding["path"] in seen:
            raise ResultPromotionError("promotion receipt repeats a canonical artifact")
        seen.add(binding["path"])


def promote_result(
    *,
    repository_root: str | Path,
    plan_path: str | Path,
    remote_identity_path: str | Path,
) -> dict[str, Any]:
    """Extract, scientifically validate, and receipt-last publish one result."""

    root = Path(repository_root).resolve(strict=True)
    plan_source = _repo_path(root, plan_path, label="promotion plan", must_exist=True)
    identity_source = _repo_path(
        root,
        remote_identity_path,
        label="remote archive identity",
        must_exist=True,
    )
    plan = load_promotion_plan(plan_source, repository_root=root)
    identity = validate_remote_archive_identity(
        plan, _read_object(identity_source, label="remote archive identity")
    )
    archive_validation = validate_finalized_archive(
        repository_root=root, plan=plan, remote_identity=identity
    )
    package = _load_package_context(
        root,
        mode=str(plan["mode"]),
        preflight_path=plan["package_evidence"]["preflight"]["path"],
        work_order_path=plan["package_evidence"]["work_order"]["path"],
    )
    receipt_path = _repo_path(
        root,
        plan["promotion"]["receipt_path"],
        label="promotion receipt",
        must_exist=False,
    )
    extracted = _extract_under_ram_guard(
        repository_root=root,
        plan_path=plan_source,
        identity_path=identity_source,
    )
    raw_root = _repo_path(
        root,
        extracted["raw_output_root"],
        label="extracted raw output",
        must_exist=True,
    )
    local_inventory = _local_inventory(raw_root, identity["members"])
    canonical_digests, canonical = _validate_worker_outputs(
        root, package, raw_root
    )
    canonical_bindings = [
        {"path": destination, "sha256": canonical_digests[destination]}
        for _source, destination in canonical
    ]
    receipt_core = {
        "schema": PROMOTION_RECEIPT_V1,
        "status": "promoted_completed_unscored",
        "mode": plan["mode"],
        "lane_id": plan["lane_id"],
        "cell_id": plan["cell_id"],
        "action_id": plan["action_id"],
        "execution_request_sha256": plan["execution_request_sha256"],
        "work_order_sha256": plan["work_order_sha256"],
        "source_epoch_identity_sha256": plan[
            "source_epoch_identity_sha256"
        ],
        "scheduler_identity": dict(plan["scheduler_identity"]),
        "scheduler_success": dict(plan["success_evidence"]),
        "promotion_plan": {
            "path": _relative_to_root(root, plan_source, label="promotion plan"),
            "sha256": _file_sha(plan_source),
            "promotion_plan_sha256": plan["promotion_plan_sha256"],
        },
        "remote_archive_identity": {
            "path": _relative_to_root(
                root, identity_source, label="remote archive identity"
            ),
            "sha256": _file_sha(identity_source),
            "archive_identity_sha256": identity["archive_identity_sha256"],
        },
        "archive": {
            "path": plan["local_transfer"]["final_path"],
            "bytes": archive_validation["archive_bytes"],
            "sha256": archive_validation["archive_sha256"],
            "gzip_integrity": True,
            "tar_integrity": True,
            "tar_member_count": archive_validation["tar_member_count"],
            "finalization_receipt": archive_validation[
                "finalization_receipt"
            ],
        },
        "raw_output_root": plan["promotion"]["raw_output_root"],
        "raw_inventory": local_inventory,
        "canonical_artifacts": canonical_bindings,
        "scientific_status": "completed_unscored",
        "remote_raw_evidence_policy": "preserve",
        "remote_raw_preserved": True,
        "remote_cleanup_authorized": False,
        "production_submission_enabled": False,
        "qiskit_compilation": False,
    }
    if receipt_path.exists():
        existing = _read_object(receipt_path, label="promotion receipt")
        declared = _sha(
            existing.get("promotion_receipt_sha256"),
            label="promotion receipt digest",
        )
        unsigned = dict(existing)
        unsigned.pop("promotion_receipt_sha256", None)
        if (
            declared != _canonical_sha(unsigned)
            or not _receipt_core_matches(existing, receipt_core)
        ):
            raise ResultPromotionError("existing promotion receipt conflicts")
        _validate_recorded_canonical_artifacts(root, existing)
        _validate_canonical_outputs(root, package)
        return existing

    intent_core = {
        "schema": PROMOTION_INTENT_V1,
        "status": "validated_ready_for_receipt_last_publication",
        "promotion_plan_sha256": plan["promotion_plan_sha256"],
        "archive_identity_sha256": identity["archive_identity_sha256"],
        "archive_sha256": identity["archive_sha256"],
        "raw_output_root": plan["promotion"]["raw_output_root"],
        "canonical_artifacts": canonical_bindings,
        "completion_receipt_published_last": True,
        "remote_raw_preserved": True,
        "remote_cleanup_authorized": False,
    }
    intent = {**intent_core, "promotion_intent_sha256": _canonical_sha(intent_core)}
    intent_path = _repo_path(
        root,
        plan["promotion"]["intent_path"],
        label="promotion intent",
        must_exist=False,
    )
    _write_immutable_json(intent_path, intent)

    # The worker completion receipt is always the final mapping entry.  Its
    # presence is therefore the public commit marker for the canonical group.
    for source, destination in canonical:
        target = _repo_path(
            root, destination, label="canonical result artifact", must_exist=False
        )
        _write_immutable(target, source.read_bytes())
    _validate_canonical_outputs(root, package)
    receipt_unsigned = {**receipt_core, "recorded_at": _utc_now()}
    receipt = {
        **receipt_unsigned,
        "promotion_receipt_sha256": _canonical_sha(receipt_unsigned),
    }
    _write_immutable_json(receipt_path, receipt)
    return receipt


def validate_promotion_inputs(
    *,
    repository_root: str | Path,
    plan_path: str | Path,
    remote_identity_path: str | Path,
) -> dict[str, Any]:
    """Validate plan, scheduler evidence, final archive, and member identities."""

    root = Path(repository_root).resolve(strict=True)
    plan = load_promotion_plan(plan_path, repository_root=root)
    identity_path = _repo_path(
        root,
        remote_identity_path,
        label="remote archive identity",
        must_exist=True,
    )
    identity = validate_remote_archive_identity(
        plan, _read_object(identity_path, label="remote archive identity")
    )
    archive = validate_finalized_archive(
        repository_root=root, plan=plan, remote_identity=identity
    )
    return {
        "status": "validated_ready_for_promotion",
        "mode": plan["mode"],
        "action_id": plan["action_id"],
        "execution_request_sha256": plan["execution_request_sha256"],
        "promotion_plan_sha256": plan["promotion_plan_sha256"],
        "archive_identity_sha256": identity["archive_identity_sha256"],
        "archive_sha256": archive["archive_sha256"],
        "tar_member_count": archive["tar_member_count"],
        "scheduler_contacted": False,
        "submission_authorized": False,
        "production_submission_enabled": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--repo-root", default=".")
    plan.add_argument("--mode", choices=sorted(MODES), required=True)
    plan.add_argument("--package-preflight", required=True)
    plan.add_argument("--work-order", required=True)
    plan.add_argument("--submission-receipt", required=True)
    plan.add_argument("--success-snapshot", required=True)
    plan.add_argument("--output", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--repo-root", default=".")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--remote-identity", required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--repo-root", default=".")
    promote.add_argument("--plan", required=True)
    promote.add_argument("--remote-identity", required=True)
    extract = commands.add_parser("_extract", help=argparse.SUPPRESS)
    extract.add_argument("--repo-root", default=".")
    extract.add_argument("--plan", required=True)
    extract.add_argument("--remote-identity", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repo_root).resolve(strict=True)
    if args.command == "plan":
        result = build_promotion_plan(
            repository_root=root,
            mode=args.mode,
            package_preflight_path=args.package_preflight,
            work_order_path=args.work_order,
            submission_receipt_path=args.submission_receipt,
            success_snapshot_path=args.success_snapshot,
            output_path=args.output,
        )
        summary = {
            "status": result["status"],
            "mode": result["mode"],
            "action_id": result["action_id"],
            "promotion_plan_sha256": result["promotion_plan_sha256"],
            "scheduler_contacted": False,
            "submission_authorized": False,
            "production_submission_enabled": False,
        }
    elif args.command == "validate":
        summary = validate_promotion_inputs(
            repository_root=root,
            plan_path=args.plan,
            remote_identity_path=args.remote_identity,
        )
    elif args.command == "promote":
        result = promote_result(
            repository_root=root,
            plan_path=args.plan,
            remote_identity_path=args.remote_identity,
        )
        summary = {
            "status": result["status"],
            "mode": result["mode"],
            "action_id": result["action_id"],
            "promotion_receipt_sha256": result["promotion_receipt_sha256"],
            "remote_raw_preserved": True,
            "production_submission_enabled": False,
        }
    else:
        plan = load_promotion_plan(args.plan, repository_root=root)
        identity_path = _repo_path(
            root,
            args.remote_identity,
            label="remote archive identity",
            must_exist=True,
        )
        identity = validate_remote_archive_identity(
            plan,
            _read_object(identity_path, label="remote archive identity"),
        )
        summary = _extract_verified_archive(
            repository_root=root, plan=plan, identity=identity
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "MODE_AVQDS_PREFIX",
    "MODE_AVQDS_TRAJECTORY",
    "MODE_EXACT_REFERENCE",
    "PROMOTION_INTENT_V1",
    "PROMOTION_PLAN_V1",
    "PROMOTION_RECEIPT_V1",
    "REMOTE_ARCHIVE_IDENTITY_V1",
    "ResultPromotionError",
    "build_archive_transfer_spec",
    "build_promotion_plan",
    "load_promotion_plan",
    "main",
    "promote_result",
    "validate_finalized_archive",
    "validate_promotion_inputs",
    "validate_remote_archive_identity",
]


if __name__ == "__main__":
    raise SystemExit(main())
