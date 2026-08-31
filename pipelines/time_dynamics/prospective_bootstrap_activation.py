"""Local-only exact-reference/AVQDS bootstrap activation coordinator.

The coordinator is intentionally narrower than the production acquisition
supervisor.  It prepares the two independent post-seed bootstrap arms, seals
their submission units and intents, and records transport/scheduler evidence
supplied by an authenticated operator.  No function opens SSH, invokes Condor,
or enables a registry lane.
"""

from __future__ import annotations

from contextvars import ContextVar
import datetime as dt
import fcntl
from functools import wraps
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
from typing import Any, Callable, Mapping, ParamSpec, Sequence, TypeVar

from pipelines.time_dynamics.acquisition_supervisor import (
    SUBMISSION_INTENT_SCHEMA_V1,
    SUBMISSION_RECEIPT_SCHEMA_V1,
    SubmissionUnit,
)
from pipelines.time_dynamics.combined_scheduler_snapshot import (
    ABSENCE_PROOF_V1,
    COMBINED_SCOPED_SNAPSHOT_V1,
    RECONCILIATION_CONTEXT_V1,
    validate_combined_snapshot,
)
from pipelines.time_dynamics.prospective_avqds_scout_chtc_package import (
    AVQDSCHTCResources,
    PreparedAVQDSPrefixPackage,
    avqds_scout_submission_unit_id,
    materialize_avqds_prefix_package,
    validate_avqds_prefix_package,
    validate_frozen_submitted_avqds_prefix_package,
)
from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    build_work_order as build_avqds_work_order,
    write_work_order as write_avqds_work_order,
)
from pipelines.time_dynamics.prospective_bootstrap_provenance import (
    BootstrapPackageManifest,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HH,
    LANE_HUBBARD,
    LANE_SIAM,
    BootstrapEvidence,
    ExactReferenceIdentity,
    SeedIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_exact_reference_chtc_package import (
    PreparedExactReferenceActionPackage,
    exact_reference_submission_unit_id,
    materialize_exact_reference_action_package,
)
from pipelines.time_dynamics.prospective_exact_reference_worker import (
    build_work_order as build_exact_work_order,
    write_work_order as write_exact_work_order,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    ProspectiveExecutionError,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    executor_bindings,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    SeedCHTCResources,
    load_runtime_image_identity,
)


ACTIVATION_MANIFEST_V1 = "paper_ii_bootstrap_dual_arm_activation_manifest_v1"
UPLOAD_VERIFICATION_V1 = "paper_ii_bootstrap_upload_verification_v1"
DRY_RUN_RESULT_V1 = "paper_ii_bootstrap_condor_dry_run_result_v1"
SUBMISSION_AUTHORIZATION_V1 = "paper_ii_bootstrap_submission_authorization_v1"
PRE_SUBMIT_EXECUTION_CLAIM_V1 = "paper_ii_bootstrap_pre_submit_execution_claim_v1"
PRE_SUBMIT_CLAIM_CONSUMPTION_V1 = (
    "paper_ii_bootstrap_pre_submit_claim_consumption_v1"
)
PRE_SUBMIT_CLAIM_RESOLUTION_V1 = (
    "paper_ii_bootstrap_pre_submit_claim_resolution_v1"
)
PROVISIONAL_SUBMISSION_RESULT_V1 = (
    "paper_ii_bootstrap_provisional_submission_result_v1"
)
EXACT_REFERENCE_TERMINAL_FAILURE_V1 = (
    "paper_ii_exact_reference_terminal_failure_evidence_v1"
)
AVQDS_SCOUT_TERMINAL_FAILURE_V1 = (
    "paper_ii_avqds_bootstrap_terminal_failure_evidence_v1"
)

_SUPPORTED_LANES = frozenset({LANE_HH, LANE_HUBBARD})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_SUBMIT_STDOUT = re.compile(
    r"\ASubmitting job\(s\)\.\s*1 job\(s\) submitted to cluster (\d+)\.\s*\Z"
)
_FROZEN_SUBMITTED_EVIDENCE = ContextVar(
    "paper_ii_frozen_submitted_evidence", default=False
)
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _classify_condor_submit_result(
    returncode: int, stdout: str
) -> tuple[str, int | None, str]:
    """Classify local submit evidence without erasing contradictions."""

    code = int(returncode)
    text = str(stdout)
    match = _SUBMIT_STDOUT.fullmatch(text)
    cluster_id = int(match.group(1)) if match is not None else None
    if code == 0 and cluster_id is not None:
        return (
            "submitted",
            cluster_id,
            "condor_submit_zero_exit_and_cluster_id_stdout",
        )
    if code != 0 and cluster_id is not None:
        return (
            "indeterminate",
            cluster_id,
            "condor_submit_nonzero_exit_with_cluster_id_stdout",
        )
    return "failed", None, "condor_submit_not_accepted"


def _frozen_submitted_evidence_only(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Confine historical package validation to immutable recovery evidence."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        token = _FROZEN_SUBMITTED_EVIDENCE.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            _FROZEN_SUBMITTED_EVIDENCE.reset(token)

    return wrapped


class BootstrapActivationError(ValueError):
    """Activation evidence is stale, ambiguous, or noncanonical."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
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


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BootstrapActivationError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapActivationError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise BootstrapActivationError(f"{label} must be a JSON object")
    return value


def _repository_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    candidate = Path(value)
    target = candidate if candidate.is_absolute() else root / candidate
    target = target.resolve(strict=must_exist)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise BootstrapActivationError(
            f"{label} must remain inside repository root"
        ) from exc
    return target


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        value = path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise BootstrapActivationError(
            f"{label} must remain inside repository root"
        ) from exc
    if not value or ".." in Path(value).parts:
        raise BootstrapActivationError(f"{label} is not repository-relative")
    return value


def _atomic_write_immutable(path: Path, payload: Mapping[str, Any]) -> str:
    content = _pretty_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != content
            ):
                raise BootstrapActivationError(
                    "refusing to overwrite immutable activation evidence: "
                    f"{path}"
                )
            status = "unchanged"
        else:
            status = "created"
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return status


def _sha(value: Any, *, label: str) -> str:
    result = str(value or "").strip().lower()
    if _SHA256.fullmatch(result) is None:
        raise BootstrapActivationError(f"{label} must be a lowercase SHA-256")
    return result


def _aware_time(value: Any, *, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise BootstrapActivationError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise BootstrapActivationError(f"{label} must include a timezone")
    return parsed


def _timestamp(value: str | None) -> str:
    if value is None:
        return dt.datetime.now(dt.timezone.utc).isoformat()
    parsed = _aware_time(value, label="recorded_at")
    return parsed.isoformat()


def _sealed(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    if key in result:
        raise BootstrapActivationError(f"payload already contains {key}")
    result[key] = _digest(result)
    return result


def _verify_seal(
    payload: Mapping[str, Any], *, key: str, label: str
) -> str:
    core = dict(payload)
    declared = core.pop(key, None)
    actual = _digest(core)
    if declared != actual:
        raise BootstrapActivationError(f"{label} self digest mismatch")
    return actual


def _file_binding(root: Path, path: Path, *, include_bytes: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _relative(root, path, label="artifact"),
        "sha256": _file_sha(path),
    }
    if include_bytes:
        result["bytes"] = path.stat().st_size
    return result


def _verified_binding(
    root: Path,
    value: Any,
    *,
    label: str,
    require_bytes: bool,
    allow_empty_bytes: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise BootstrapActivationError(f"{label} binding is missing")
    path = _repository_path(
        root,
        str(value.get("path") or ""),
        label=label,
        must_exist=True,
    )
    digest = _sha(value.get("sha256"), label=f"{label} digest")
    result: dict[str, Any] = {
        "path": _relative(root, path, label=label),
        "sha256": digest,
    }
    if not path.is_file() or _file_sha(path) != digest:
        raise BootstrapActivationError(f"{label} digest changed")
    if require_bytes:
        try:
            size = int(value.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise BootstrapActivationError(f"{label} byte count is invalid") from exc
        if (
            size < 0
            or (size == 0 and not allow_empty_bytes)
            or path.stat().st_size != size
        ):
            raise BootstrapActivationError(f"{label} byte count changed")
        result["bytes"] = size
    return path, result


def _verified_evidence_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise BootstrapActivationError(f"{label} binding is missing")
    path = _repository_path(
        root,
        str(value.get("path") or ""),
        label=label,
        must_exist=True,
    )
    file_digest = _sha(value.get("file_sha256"), label=f"{label} file digest")
    _sha(value.get("sha256"), label=f"{label} logical digest")
    if not path.is_file() or _file_sha(path) != file_digest:
        raise BootstrapActivationError(f"{label} file digest changed")
    return path


def _safe_remote_path(value: str, *, label: str) -> str:
    path = PurePosixPath(str(value).strip())
    if (
        not str(path)
        or path.is_absolute()
        or ".." in path.parts
        or any(_SAFE_ID.fullmatch(part) is None for part in path.parts)
    ):
        raise BootstrapActivationError(
            f"{label} must be a safe path relative to the remote home"
        )
    return path.as_posix()


def _write_execution_request(path: Path, request: ExecutionRequest) -> None:
    _atomic_write_immutable(path, request.to_json_dict())
    reloaded = ExecutionRequest.from_mapping(
        _read_object(path, label="execution request")
    )
    if reloaded != request:
        raise BootstrapActivationError("execution request changed after writing")


def _lane_snapshot_key(lane_id: str) -> str:
    if lane_id == LANE_HH:
        return "hh"
    if lane_id == LANE_HUBBARD:
        return "hubbard"
    raise BootstrapActivationError("bootstrap activation crosses into SIAM")


@dataclass(frozen=True)
class BootstrapSubmissionIntent:
    path: str
    file_sha256: str
    intent_sha256: str
    recorded_at: str
    lane_id: str
    unit: SubmissionUnit
    request: ExecutionRequest
    package_manifest: BootstrapPackageManifest
    source_epoch: SourceEpochReceipt
    package_kind: str


@dataclass(frozen=True)
class ValidatedSnapshotEvidence:
    path: Path
    payload: dict[str, Any]
    binding: dict[str, Any]
    prior_payload: dict[str, Any] | None


def _state_path(root: Path, lane_id: str, category: str, unit_id: str) -> Path:
    if lane_id not in _SUPPORTED_LANES or _SAFE_ID.fullmatch(unit_id) is None:
        raise BootstrapActivationError("submission state identity is unsafe")
    return (
        root
        / "output"
        / lane_id
        / "acquisition_supervisor"
        / category
        / f"{unit_id}.json"
    )


def _intent_core(lane_id: str, unit: SubmissionUnit) -> dict[str, Any]:
    return {
        "schema": SUBMISSION_INTENT_SCHEMA_V1,
        "lane_id": lane_id,
        "unit": unit.to_json_dict(),
    }


def _ensure_submission_intent(
    root: Path,
    *,
    lane_id: str,
    unit: SubmissionUnit,
    recorded_at: str | None,
) -> BootstrapSubmissionIntent:
    path = _state_path(root, lane_id, "submission_intents", unit.unit_id)
    core = _intent_core(lane_id, unit)
    if path.exists():
        existing = _read_object(path, label="submission intent")
        if existing.get("intent_sha256") != _digest(core):
            raise BootstrapActivationError(
                "submission unit changed after its intent was recorded"
            )
    else:
        payload = {
            **core,
            "intent_sha256": _digest(core),
            "recorded_at": _timestamp(recorded_at),
        }
        _atomic_write_immutable(path, payload)
    return load_submission_intent(path, repository_root=root)


def _load_submission_intent(
    path: str | Path,
    *,
    repository_root: str | Path,
    frozen_submitted: bool,
) -> BootstrapSubmissionIntent:
    """Reload one exact/scout acquisition-supervisor submission intent."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="submission intent", must_exist=True
    )
    payload = _read_object(source, label="submission intent")
    unit_row = payload.get("unit")
    if (
        payload.get("schema") != SUBMISSION_INTENT_SCHEMA_V1
        or payload.get("lane_id") not in _SUPPORTED_LANES
        or not isinstance(unit_row, Mapping)
    ):
        raise BootstrapActivationError("submission intent is not a bootstrap action")
    try:
        unit = SubmissionUnit(
            unit_id=str(unit_row.get("unit_id") or ""),
            action_ids=tuple(str(value) for value in unit_row.get("action_ids") or ()),
            payload=dict(unit_row.get("payload") or {}),
        )
    except (TypeError, ValueError) as exc:
        raise BootstrapActivationError("submission intent unit is invalid") from exc
    lane_id = str(payload["lane_id"])
    core = _intent_core(lane_id, unit)
    intent_sha = _sha(payload.get("intent_sha256"), label="intent digest")
    if intent_sha != _digest(core):
        raise BootstrapActivationError("submission intent self digest mismatch")
    recorded_at = str(payload.get("recorded_at") or "")
    _aware_time(recorded_at, label="intent recorded_at")
    expected_path = _state_path(root, lane_id, "submission_intents", unit.unit_id)
    if source != expected_path.resolve(strict=True):
        raise BootstrapActivationError("submission intent is not at its canonical path")
    body = unit.payload
    request_path, request_binding = _verified_binding(
        root, body.get("execution_request"), label="execution request", require_bytes=False
    )
    request = ExecutionRequest.from_mapping(
        _read_object(request_path, label="execution request")
    )
    manifest_path, manifest_binding = _verified_binding(
        root,
        body.get("package_manifest"),
        label="package manifest",
        require_bytes=False,
    )
    package_manifest = BootstrapPackageManifest.load(
        manifest_path, repository_root=root
    )
    source_path, source_binding = _verified_binding(
        root,
        body.get("source_epoch_receipt"),
        label="source epoch receipt",
        require_bytes=False,
    )
    source_epoch = SourceEpochReceipt.load(
        source_path, repository_root=root
    )
    package_path, package_binding = _verified_binding(
        root, body.get("package"), label="bootstrap package", require_bytes=True
    )
    descriptor_path, descriptor_binding = _verified_binding(
        root,
        body.get("submit_descriptor_binding"),
        label="submit descriptor",
        require_bytes=False,
    )
    archive_path, archive_binding = _verified_binding(
        root,
        body.get("runtime_source_archive"),
        label="runtime source archive",
        require_bytes=True,
    )
    image_identity_path, image_identity_binding = _verified_binding(
        root,
        body.get("runtime_image_identity"),
        label="runtime image identity",
        require_bytes=False,
    )
    image = load_runtime_image_identity(
        image_identity_path, repository_root=root
    )
    package_kind = str(body.get("package_kind") or "")
    preflight_path, preflight_binding = _verified_binding(
        root,
        body.get("package_preflight"),
        label="package preflight",
        require_bytes=False,
    )
    prepared_runtime_source_archive = None
    prepared_runtime_source_archive_bytes = None
    if package_kind == "exact_reference":
        prepared_exact = PreparedExactReferenceActionPackage.load(
            preflight_path, repository_root=root
        )
        prepared_request_sha = prepared_exact.request_sha256
        prepared_package = prepared_exact.package
        prepared_manifest = prepared_exact.package_manifest
        prepared_descriptor = prepared_exact.submit_descriptor
        operational_generation = prepared_exact.operational_generation
    elif package_kind == "avqds_scout":
        scout_validator = (
            validate_frozen_submitted_avqds_prefix_package
            if frozen_submitted
            else validate_avqds_prefix_package
        )
        prepared_scout = scout_validator(
            preflight_path, repository_root=root
        )
        prepared_request_sha = prepared_scout.execution_request_sha256
        prepared_package = prepared_scout.package
        prepared_manifest = prepared_scout.package_manifest
        prepared_descriptor = prepared_scout.descriptor
        prepared_runtime_source_archive = prepared_scout.runtime_source_archive
        prepared_runtime_source_archive_bytes = (
            prepared_scout.runtime_source_archive_bytes
        )
        operational_generation = prepared_scout.operational_generation
    else:
        raise BootstrapActivationError("submission intent package kind is invalid")
    expected_kind = (
        "build_exact_reference"
        if package_kind == "exact_reference"
        else "screen_avqds"
    )
    scope = body.get("submission_scope")
    expected_seed_id = str(request.inputs.get("seed_id") or "")
    if not expected_seed_id and request.kind == "screen_avqds":
        expected_seed_id = build_identity_registry().seed_for_cell(
            str(request.inputs.get("cell_id") or "")
        ).seed_id
    remote_workdir = _safe_remote_path(
        str(body.get("remote_workdir") or ""), label="remote workdir"
    )
    remote_package = _safe_remote_path(
        str(body.get("remote_package_path") or ""), label="remote package path"
    )
    if PurePosixPath(remote_package).parent.as_posix() != remote_workdir:
        raise BootstrapActivationError("remote package is outside its workdir")
    if (
        lane_id != request.lane_id
        or len(unit.action_ids) != 1
        or unit.action_ids[0] != request.action_id
        or request.kind != expected_kind
        or body.get("execution_request_sha256") != request.digest
        or request_binding["sha256"] != _file_sha(request_path)
        or package_manifest.request != request
        or package_manifest.source_epoch_receipt.path != source_binding["path"]
        or package_manifest.source_epoch_receipt.sha256 != source_binding["sha256"]
        or source_epoch.lane_id != lane_id
        or request.inputs.get("source_epoch")
        != source_epoch.source_epoch.to_json_dict()
        or manifest_binding["sha256"] != package_manifest.manifest_sha256
        or package_manifest.package.path != package_binding["path"]
        or package_manifest.package.sha256 != package_binding["sha256"]
        or package_manifest.package_bytes != package_binding["bytes"]
        or prepared_request_sha != request.digest
        or prepared_package.path != package_binding["path"]
        or prepared_package.sha256 != package_binding["sha256"]
        or prepared_manifest.path != manifest_binding["path"]
        or prepared_manifest.sha256 != manifest_binding["sha256"]
        or prepared_descriptor.path != descriptor_binding["path"]
        or prepared_descriptor.sha256 != descriptor_binding["sha256"]
        or (
            prepared_runtime_source_archive is not None
            and (
                prepared_runtime_source_archive.path != archive_binding["path"]
                or prepared_runtime_source_archive.sha256
                != archive_binding["sha256"]
                or prepared_runtime_source_archive_bytes
                != archive_binding["bytes"]
            )
        )
        or body.get("submit_descriptor") != descriptor_binding["path"]
        or body.get("job_batch_name") != package_manifest.job_batch_name
        or body.get("expected_process_count") != 1
        or scope
        != {"action_ids": [request.action_id], "process_ids": [0]}
        or body.get("lane_id") != lane_id
        or body.get("cell_id") != request.inputs.get("cell_id")
        or body.get("seed_id") != expected_seed_id
        or body.get("seed_identity_sha256")
        != request.inputs.get("seed_identity_sha256")
        or body.get("seed_artifact_sha256")
        != request.inputs.get("seed_artifact_sha256")
        or body.get("reference_id") != request.inputs.get("reference_id")
        or body.get("reference_identity_sha256")
        != request.inputs.get("reference_identity_sha256")
        or body.get("source_epoch_identity_sha256")
        != source_epoch.source_epoch.identity_sha256
        or body.get("source_epoch_id")
        != source_epoch.source_epoch.source_epoch_id
        or body.get("runtime_source_archive") != archive_binding
        or body.get("runtime_image_identity") != image_identity_binding
        or body.get("runtime_image")
        != {"path": image.path, "sha256": image.sha256, "bytes": image.bytes}
        or PurePosixPath(remote_package).name != package_path.name
        or body.get("remote_raw_evidence_policy") != "preserve"
        or body.get("qiskit_compilation") is not False
        or body.get("production_submission_enabled") is not False
    ):
        raise BootstrapActivationError(
            "submission intent request/package/source identity diverged"
        )
    if archive_path.stat().st_size != archive_binding["bytes"]:
        raise AssertionError("verified archive binding changed unexpectedly")
    if preflight_binding["sha256"] != _file_sha(preflight_path):
        raise AssertionError("verified package preflight changed unexpectedly")
    if package_kind == "exact_reference":
        expected_unit_id = exact_reference_submission_unit_id(
            request, operational_generation=operational_generation
        )
        failure_row = body.get("terminal_failure_evidence")
        predecessor_row = body.get("predecessor_intent")
        if (
            unit.unit_id != expected_unit_id
            or int(body.get("operational_generation", 1))
            != operational_generation
        ):
            raise BootstrapActivationError(
                "exact-reference operational generation diverged"
            )
        if operational_generation == 1:
            if failure_row is not None or predecessor_row is not None:
                raise BootstrapActivationError(
                    "generation-one exact intent carries successor evidence"
                )
        else:
            if (
                operational_generation != 2
                or not isinstance(failure_row, Mapping)
                or not isinstance(predecessor_row, Mapping)
                or body.get("automatic_retry_authorized") is not False
                or body.get("scientific_identity_changed") is not False
                or body.get("successor_reason")
                != "verified_terminal_execution_binding_plumbing_failure"
            ):
                raise BootstrapActivationError(
                    "successor exact intent lacks terminal-failure authority"
                )
            failure_path = _verified_evidence_path(
                root,
                failure_row,
                label="terminal exact-reference failure evidence",
            )
            failure = load_exact_reference_terminal_failure_evidence(
                failure_path, repository_root=root
            )
            predecessor_path = _verified_evidence_path(
                root, predecessor_row, label="predecessor submission intent"
            )
            predecessor = load_submission_intent(
                predecessor_path, repository_root=root
            )
            if (
                predecessor.package_kind != "exact_reference"
                or predecessor.request != request
                or int(
                    predecessor.unit.payload.get("operational_generation", 1)
                )
                != 1
                or failure.get("predecessor_intent")
                != _intent_binding(predecessor)
                or failure_row.get("sha256")
                != failure.get("terminal_failure_evidence_sha256")
                or dict(predecessor_row) != _intent_binding(predecessor)
            ):
                raise BootstrapActivationError(
                    "successor exact intent crosses predecessor identity"
                )
    elif package_kind == "avqds_scout":
        expected_unit_id = avqds_scout_submission_unit_id(
            request, operational_generation=operational_generation
        )
        failure_row = body.get("terminal_failure_evidence")
        predecessor_row = body.get("predecessor_intent")
        if (
            unit.unit_id != expected_unit_id
            or int(body.get("operational_generation", 1))
            != operational_generation
        ):
            raise BootstrapActivationError(
                "AVQDS-scout operational generation diverged"
            )
        if operational_generation == 1:
            if failure_row is not None or predecessor_row is not None:
                raise BootstrapActivationError(
                    "generation-one AVQDS intent carries successor evidence"
                )
        else:
            if (
                operational_generation != 2
                or not isinstance(failure_row, Mapping)
                or not isinstance(predecessor_row, Mapping)
                or body.get("automatic_retry_authorized") is not False
                or body.get("scientific_identity_changed") is not False
                or body.get("canary_only") is not True
                or body.get("successor_reason")
                != "verified_terminal_post_run_contract_validation_failure"
            ):
                raise BootstrapActivationError(
                    "successor AVQDS intent lacks terminal-failure authority"
                )
            failure_path = _verified_evidence_path(
                root,
                failure_row,
                label="terminal AVQDS-scout failure evidence",
            )
            failure = load_avqds_scout_terminal_failure_evidence(
                failure_path, repository_root=root
            )
            predecessor_path = _verified_evidence_path(
                root, predecessor_row, label="predecessor submission intent"
            )
            predecessor = load_frozen_submitted_intent(
                predecessor_path, repository_root=root
            )
            predecessor_prepared = validate_frozen_submitted_avqds_prefix_package(
                root / predecessor.unit.payload["package_preflight"]["path"],
                repository_root=root,
            )
            if (
                predecessor.package_kind != "avqds_scout"
                or predecessor.request != request
                or int(
                    predecessor.unit.payload.get("operational_generation", 1)
                )
                != 1
                or _avqds_failure_intent_binding(
                    failure.get("submission", {}).get("intent", {})
                )
                != _intent_binding(predecessor)
                or failure_row.get("sha256")
                != failure.get("failure_receipt_sha256")
                or dict(predecessor_row) != _intent_binding(predecessor)
                or prepared_scout.resources != predecessor_prepared.resources
                or prepared_scout.resources
                != _avqds_failure_recommended_resources(failure)
            ):
                raise BootstrapActivationError(
                    "successor AVQDS intent crosses predecessor identity or resources"
                )
    return BootstrapSubmissionIntent(
        path=_relative(root, source, label="submission intent"),
        file_sha256=_file_sha(source),
        intent_sha256=intent_sha,
        recorded_at=recorded_at,
        lane_id=lane_id,
        unit=unit,
        request=request,
        package_manifest=package_manifest,
        source_epoch=source_epoch,
        package_kind=package_kind,
    )


def load_submission_intent(
    path: str | Path, *, repository_root: str | Path
) -> BootstrapSubmissionIntent:
    """Reload an activation intent against the current package source bytes."""

    return _load_submission_intent(
        path,
        repository_root=repository_root,
        frozen_submitted=_FROZEN_SUBMITTED_EVIDENCE.get(),
    )


def load_frozen_submitted_intent(
    path: str | Path, *, repository_root: str | Path
) -> BootstrapSubmissionIntent:
    """Reload an already-submitted intent from its immutable source revisions."""

    return _load_submission_intent(
        path,
        repository_root=repository_root,
        frozen_submitted=True,
    )


def snapshot_intent_binding(intent: BootstrapSubmissionIntent) -> dict[str, Any]:
    """Return the exact intent row consumed by the combined snapshot builder."""

    return {
        "job_batch_name": intent.package_manifest.job_batch_name,
        "acquisition_unit_id": intent.unit.unit_id,
        "paper_ii_action_id": intent.request.action_id,
        "paper_ii_execution_request_sha256": intent.request.digest,
        "intent_sha256": intent.intent_sha256,
        "recorded_at": intent.recorded_at,
    }


def _unit_for_package(
    root: Path,
    *,
    request_path: Path,
    request: ExecutionRequest,
    reference: ExactReferenceIdentity,
    seed: SeedIdentity,
    source: SourceEpochReceipt,
    source_archive: Path,
    runtime_image_identity_path: Path,
    package_kind: str,
    package: Path,
    package_manifest: BootstrapPackageManifest,
    descriptor: Path,
    preflight: Path,
    remote_root: str,
    concurrent_peer_action_id: str,
) -> SubmissionUnit:
    image = load_runtime_image_identity(
        runtime_image_identity_path, repository_root=root
    )
    unit_id = f"bootstrap-{package_kind.replace('_', '-')}-{request.digest[:16]}"
    if _SAFE_ID.fullmatch(unit_id) is None:
        raise BootstrapActivationError("derived submission unit ID is unsafe")
    remote_workdir = _safe_remote_path(
        (
            PurePosixPath(remote_root)
            / request.lane_id
            / reference.identity_sha256[:16]
            / package_kind
        ).as_posix(),
        label="remote workdir",
    )
    remote_package = (
        PurePosixPath(remote_workdir) / package.name
    ).as_posix()
    payload = {
        "package_kind": package_kind,
        "lane_id": request.lane_id,
        "cell_id": reference.cell_id,
        "seed_id": seed.seed_id,
        "seed_identity_sha256": seed.identity_sha256,
        "seed_artifact_sha256": request.inputs["seed_artifact_sha256"],
        "reference_id": reference.reference_id,
        "reference_identity_sha256": reference.identity_sha256,
        "execution_request": _file_binding(
            root, request_path, include_bytes=False
        ),
        "execution_request_sha256": request.digest,
        "package_manifest": _file_binding(
            root,
            root / package_manifest.manifest_path,
            include_bytes=False,
        ),
        "package": _file_binding(root, package, include_bytes=True),
        "package_preflight": _file_binding(root, preflight, include_bytes=False),
        "submit_descriptor": _relative(root, descriptor, label="submit descriptor"),
        "submit_descriptor_binding": _file_binding(
            root, descriptor, include_bytes=False
        ),
        "source_epoch_receipt": _file_binding(
            root, root / source.receipt_path, include_bytes=False
        ),
        "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
        "source_epoch_id": source.source_epoch.source_epoch_id,
        "runtime_source_archive": _file_binding(
            root, source_archive, include_bytes=True
        ),
        "runtime_image_identity": _file_binding(
            root, runtime_image_identity_path, include_bytes=False
        ),
        "runtime_image": {
            "path": image.path,
            "sha256": image.sha256,
            "bytes": image.bytes,
        },
        "job_batch_name": package_manifest.job_batch_name,
        "expected_process_count": 1,
        "submission_scope": {
            "action_ids": [request.action_id],
            "process_ids": [0],
        },
        "remote_workdir": remote_workdir,
        "remote_package_path": remote_package,
        "concurrent_group_id": (
            f"seed-cell::{seed.identity_sha256[:16]}::"
            f"{reference.identity_sha256[:16]}"
        ),
        "concurrent_peer_action_id": concurrent_peer_action_id,
        "remote_raw_evidence_policy": "preserve",
        "qiskit_compilation": False,
        "production_submission_enabled": False,
    }
    return SubmissionUnit(
        unit_id=unit_id,
        action_ids=(request.action_id,),
        payload=payload,
    )


def _load_source_receipts(
    root: Path, paths: Sequence[str | Path]
) -> dict[str, SourceEpochReceipt]:
    result: dict[str, SourceEpochReceipt] = {}
    for path in paths:
        receipt = SourceEpochReceipt.load(path, repository_root=root)
        if receipt.lane_id not in _SUPPORTED_LANES:
            raise BootstrapActivationError("source receipt crosses into SIAM")
        if receipt.lane_id in result:
            raise BootstrapActivationError("source receipt lane is repeated")
        result[receipt.lane_id] = receipt
    if not result:
        raise BootstrapActivationError("no HH/Hubbard source receipts were supplied")
    return result


def _load_seed_receipts(
    root: Path,
    paths: Sequence[str | Path],
    *,
    sources: Mapping[str, SourceEpochReceipt],
) -> dict[str, SeedQualificationReceipt]:
    registry = build_identity_registry()
    seed_by_id = {row.seed_id: row for row in registry.seeds}
    result: dict[str, SeedQualificationReceipt] = {}
    for path in paths:
        source = _repository_path(root, path, label="seed receipt", must_exist=True)
        payload = _read_object(source, label="seed receipt")
        seed_id = str(payload.get("seed_id") or "")
        seed = seed_by_id.get(seed_id)
        if seed is None or seed.lane_id not in _SUPPORTED_LANES:
            raise BootstrapActivationError("seed receipt is not a prospective HH/Hubbard seed")
        epoch = sources.get(seed.lane_id)
        if epoch is None:
            raise BootstrapActivationError("seed has no same-lane source epoch receipt")
        receipt = SeedQualificationReceipt.load(
            source,
            repository_root=root,
            expected_seed=seed,
            expected_source_epoch=epoch.source_epoch,
        )
        if receipt.seed_id in result:
            raise BootstrapActivationError("seed receipt identity is repeated")
        result[receipt.seed_id] = receipt
    if not result:
        raise BootstrapActivationError("no qualified seed receipts were supplied")
    return result


def prepare_bootstrap_activation(
    *,
    repository_root: str | Path,
    output_dir: str | Path,
    seed_receipt_paths: Sequence[str | Path],
    source_epoch_receipt_paths: Sequence[str | Path],
    exact_source_archive_paths: Mapping[str, str | Path],
    avqds_source_archive_paths: Mapping[str, str | Path],
    runtime_image_identity_path: str | Path,
    remote_root: str = "Holstein_phase3_optuna_chtc/paper_ii_bootstrap_v1",
    exact_resources: SeedCHTCResources | None = None,
    scout_resources: AVQDSCHTCResources | None = None,
    free_bytes: int | None = None,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Prepare both bootstrap arms for every supplied qualified seed/cell."""

    root = Path(repository_root).resolve(strict=True)
    output = _repository_path(
        root, output_dir, label="activation output", must_exist=False
    )
    remote = _safe_remote_path(remote_root, label="remote root")
    runtime_identity = _repository_path(
        root,
        runtime_image_identity_path,
        label="runtime image identity",
        must_exist=True,
    )
    load_runtime_image_identity(runtime_identity, repository_root=root)
    sources = _load_source_receipts(root, source_epoch_receipt_paths)
    seeds = _load_seed_receipts(root, seed_receipt_paths, sources=sources)
    exact_archives: dict[str, Path] = {}
    avqds_archives: dict[str, Path] = {}
    for lane_id, source in sources.items():
        exact_value = exact_source_archive_paths.get(lane_id)
        avqds_value = avqds_source_archive_paths.get(lane_id)
        if exact_value is None or avqds_value is None:
            raise BootstrapActivationError(
                f"distinct exact/AVQDS source archives are missing for lane {lane_id}"
            )
        exact_archives[lane_id] = _repository_path(
            root,
            exact_value,
            label=f"{lane_id} exact source archive",
            must_exist=True,
        )
        avqds_archives[lane_id] = _repository_path(
            root,
            avqds_value,
            label=f"{lane_id} AVQDS source archive",
            must_exist=False,
        )
        if (
            not exact_archives[lane_id].is_file()
            or (
                avqds_archives[lane_id].exists()
                and not avqds_archives[lane_id].is_file()
            )
        ):
            raise BootstrapActivationError("source archive is not a regular file")
        if (
            exact_archives[lane_id] == avqds_archives[lane_id]
            or (
                avqds_archives[lane_id].exists()
                and _file_sha(exact_archives[lane_id])
                == _file_sha(avqds_archives[lane_id])
            )
        ):
            raise BootstrapActivationError(
                "exact and AVQDS arms must bind independent source archives"
            )
        if source.source_epoch.source_archive_sha256 is None:
            raise BootstrapActivationError("source epoch has no archive digest")
    if (
        set(exact_source_archive_paths) != set(sources)
        or set(avqds_source_archive_paths) != set(sources)
    ):
        raise BootstrapActivationError(
            "exact/AVQDS source archive lanes diverged from source receipts"
        )

    registry = build_identity_registry()
    registry_seeds = {row.seed_id: row for row in registry.seeds}
    references = tuple(
        sorted(
            (
                row
                for row in registry.references
                if row.seed_id in seeds and row.lane_id in _SUPPORTED_LANES
            ),
            key=lambda row: row.reference_id,
        )
    )
    if not references:
        raise BootstrapActivationError("qualified seeds open no prospective cells")
    evidence = BootstrapEvidence(
        seed_artifact_sha256={
            seed_id: receipt.artifact.sha256
            for seed_id, receipt in seeds.items()
        },
        source_epochs={
            lane_id: receipt.source_epoch for lane_id, receipt in sources.items()
        },
    )
    plan = plan_bootstrap_actions(
        registry,
        evidence=evidence,
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    ready = tuple(plan.ready_actions)
    expected_action_count = 2 * len(references)
    selected = tuple(
        row
        for row in ready
        if row.kind in {"build_exact_reference", "screen_avqds"}
        and (
            row.inputs.get("seed_id") in seeds
            or registry.seed_for_cell(str(row.inputs.get("cell_id") or "")).seed_id
            in seeds
        )
    )
    if len(selected) != expected_action_count:
        raise BootstrapActivationError(
            "frozen bootstrap planner did not open exact+AVQDS for every eligible cell"
        )

    exact_resource_row = exact_resources or SeedCHTCResources(1, 16, 8, 43_200)
    scout_resource_row = scout_resources or AVQDSCHTCResources()
    pairs: list[dict[str, Any]] = []
    for reference in references:
        seed = registry_seeds[reference.seed_id]
        seed_receipt = seeds[seed.seed_id]
        source = sources[reference.lane_id]
        matches_exact = [
            row
            for row in selected
            if row.kind == "build_exact_reference"
            and row.subject_id == reference.reference_id
        ]
        matches_scout = [
            row
            for row in selected
            if row.kind == "screen_avqds" and row.subject_id == reference.cell_id
        ]
        if len(matches_exact) != 1 or len(matches_scout) != 1:
            raise BootstrapActivationError(
                "eligible cell does not have exactly one ready exact/scout pair"
            )
        exact_action = matches_exact[0]
        scout_action = matches_scout[0]
        if (
            exact_action.prerequisites != (seed.seed_id,)
            or scout_action.prerequisites != (seed.seed_id,)
            or exact_action.lane_id != scout_action.lane_id
            or exact_action.lane_id != reference.lane_id
            or LANE_SIAM in exact_action.prerequisites
            or LANE_SIAM in scout_action.prerequisites
        ):
            raise BootstrapActivationError(
                "bootstrap pair is serialized or crosses lanes"
            )
        exact_request = resolve_bootstrap_action(exact_action)
        scout_request = resolve_bootstrap_action(scout_action)
        cell_root = (
            output
            / reference.lane_id
            / f"cell-{reference.identity_sha256[:16]}"
        )
        exact_root = cell_root / "exact_reference"
        scout_root = cell_root / "avqds_scout"
        exact_request_path = exact_root / "execution_request.json"
        scout_request_path = scout_root / "execution_request.json"
        _write_execution_request(exact_request_path, exact_request)
        _write_execution_request(scout_request_path, scout_request)

        exact_order = build_exact_work_order(
            exact_request,
            source_epoch_receipt=source,
            seed_receipt=seed_receipt,
        )
        exact_order_path = exact_root / "work_order.json"
        write_exact_work_order(exact_order_path, exact_order)
        exact_package_root = exact_root / "package"
        exact_prepared = materialize_exact_reference_action_package(
            repository_root=root,
            output_dir=exact_package_root,
            request_path=exact_request_path,
            work_order_path=exact_order_path,
            source_epoch_receipt=source,
            source_archive_path=exact_archives[reference.lane_id],
            runtime_image_identity=load_runtime_image_identity(
                runtime_identity, repository_root=root
            ),
            resources=exact_resource_row,
            free_bytes=free_bytes,
        )
        exact_manifest = BootstrapPackageManifest.load(
            root / exact_prepared.package_manifest.path,
            repository_root=root,
        )

        scout_order = build_avqds_work_order(
            scout_request,
            source_epoch_receipt=source,
            seed_receipt=seed_receipt,
            repository_root=root,
        )
        scout_order_path = scout_root / "work_order.json"
        write_avqds_work_order(scout_order_path, scout_order)
        scout_package_root = scout_root / "package"
        if scout_package_root.exists():
            scout_prepared = validate_avqds_prefix_package(
                scout_package_root / "local_preflight_receipt.json",
                repository_root=root,
            )
            supplied_archive = avqds_archives[reference.lane_id]
            supplied_binding = {
                "path": _relative(
                    root, supplied_archive, label="supplied AVQDS runtime archive"
                ),
                "sha256": _file_sha(supplied_archive)
                if supplied_archive.is_file()
                else None,
                "bytes": supplied_archive.stat().st_size
                if supplied_archive.is_file()
                else None,
            }
            if (
                scout_prepared.execution_request_sha256 != scout_request.digest
                or scout_prepared.work_order_sha256 != scout_order.work_order_sha256
                or supplied_binding
                != {
                    "path": scout_prepared.runtime_source_archive.path,
                    "sha256": scout_prepared.runtime_source_archive.sha256,
                    "bytes": scout_prepared.runtime_source_archive_bytes,
                }
            ):
                raise BootstrapActivationError(
                    "existing AVQDS scout package has another identity or runtime source archive"
                )
        else:
            scout_prepared = materialize_avqds_prefix_package(
                work_order_path=scout_order_path,
                runtime_source_archive_path=avqds_archives[reference.lane_id],
                runtime_image_identity_path=runtime_identity,
                output_dir=scout_package_root,
                repository_root=root,
                resources=scout_resource_row,
                free_bytes=free_bytes,
            )
        if _file_sha(exact_archives[reference.lane_id]) == _file_sha(
            avqds_archives[reference.lane_id]
        ):
            raise BootstrapActivationError(
                "exact and AVQDS runtime archives unexpectedly share one digest"
            )
        scout_manifest = BootstrapPackageManifest.load(
            root / scout_prepared.package_manifest.path,
            repository_root=root,
        )

        exact_unit = _unit_for_package(
            root,
            request_path=exact_request_path,
            request=exact_request,
            reference=reference,
            seed=seed,
            source=source,
            source_archive=exact_archives[reference.lane_id],
            runtime_image_identity_path=runtime_identity,
            package_kind="exact_reference",
            package=root / exact_prepared.package.path,
            package_manifest=exact_manifest,
            descriptor=root / exact_prepared.submit_descriptor.path,
            preflight=root / exact_prepared.preflight_receipt.path,
            remote_root=remote,
            concurrent_peer_action_id=scout_request.action_id,
        )
        scout_unit = _unit_for_package(
            root,
            request_path=scout_request_path,
            request=scout_request,
            reference=reference,
            seed=seed,
            source=source,
            source_archive=avqds_archives[reference.lane_id],
            runtime_image_identity_path=runtime_identity,
            package_kind="avqds_scout",
            package=root / scout_prepared.package.path,
            package_manifest=scout_manifest,
            descriptor=root / scout_prepared.descriptor.path,
            preflight=root / scout_prepared.preflight.path,
            remote_root=remote,
            concurrent_peer_action_id=exact_request.action_id,
        )
        exact_intent = _ensure_submission_intent(
            root,
            lane_id=reference.lane_id,
            unit=exact_unit,
            recorded_at=recorded_at,
        )
        scout_intent = _ensure_submission_intent(
            root,
            lane_id=reference.lane_id,
            unit=scout_unit,
            recorded_at=recorded_at,
        )
        pairs.append(
            {
                "lane_id": reference.lane_id,
                "cell_id": reference.cell_id,
                "reference_id": reference.reference_id,
                "reference_identity_sha256": reference.identity_sha256,
                "seed_id": seed.seed_id,
                "seed_identity_sha256": seed.identity_sha256,
                "concurrency": {
                    "mode": "independent_after_same_qualified_seed",
                    "join_required_before_launch": False,
                    "siam_dependency": False,
                    "apm_dependency": False,
                    "arms": ["exact_reference", "avqds_scout"],
                },
                "exact_reference": {
                    "action_id": exact_request.action_id,
                    "execution_request_sha256": exact_request.digest,
                    "intent": {
                        "path": exact_intent.path,
                        "sha256": exact_intent.file_sha256,
                    },
                    "package": exact_unit.payload["package"],
                    "package_manifest": exact_unit.payload["package_manifest"],
                },
                "avqds_scout": {
                    "action_id": scout_request.action_id,
                    "execution_request_sha256": scout_request.digest,
                    "intent": {
                        "path": scout_intent.path,
                        "sha256": scout_intent.file_sha256,
                    },
                    "package": scout_unit.payload["package"],
                    "package_manifest": scout_unit.payload["package_manifest"],
                },
            }
        )

    manifest_core = {
        "schema": ACTIVATION_MANIFEST_V1,
        "status": "prepared_local_only",
        "identity_registry_sha256": registry.digest,
        "bootstrap_plan_sha256": _digest(plan.to_json_dict()),
        "qualified_seed_receipts": [
            {
                "seed_id": receipt.seed_id,
                "path": receipt.receipt_path,
                "sha256": receipt.receipt_sha256,
            }
            for receipt in sorted(seeds.values(), key=lambda row: row.seed_id)
        ],
        "source_epoch_receipts": [
            {
                "lane_id": receipt.lane_id,
                "path": receipt.receipt_path,
                "sha256": receipt.receipt_sha256,
            }
            for receipt in sorted(sources.values(), key=lambda row: row.lane_id)
        ],
        "pairs": pairs,
        "pair_count": len(pairs),
        "action_count": 2 * len(pairs),
        "concurrency_contract": {
            "exact_and_avqds_launch_independently": True,
            "join_only_after_completion": True,
            "siam_dependency": False,
            "apm_selected": False,
        },
        "delegated_lanes": dict(plan.delegated_lanes),
        "remote_root": remote,
        "scheduler_contacted": False,
        "submission_authorized": False,
        "remote_raw_evidence_policy": "preserve",
        "qiskit_compilation": False,
        "production_submission_enabled": False,
    }
    manifest = _sealed(manifest_core, "activation_manifest_sha256")
    manifest_path = output / "activation_manifest.json"
    _atomic_write_immutable(manifest_path, manifest)
    return load_activation_manifest(manifest_path, repository_root=root)


def load_activation_manifest(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Reload a prepared dual-arm manifest and every bound intent/package."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="activation manifest", must_exist=True
    )
    payload = _read_object(source, label="activation manifest")
    _verify_seal(
        payload, key="activation_manifest_sha256", label="activation manifest"
    )
    pairs = payload.get("pairs")
    if (
        payload.get("schema") != ACTIVATION_MANIFEST_V1
        or payload.get("status") != "prepared_local_only"
        or not isinstance(pairs, list)
        or payload.get("pair_count") != len(pairs)
        or payload.get("action_count") != 2 * len(pairs)
        or not pairs
        or payload.get("concurrency_contract")
        != {
            "exact_and_avqds_launch_independently": True,
            "join_only_after_completion": True,
            "siam_dependency": False,
            "apm_selected": False,
        }
        or payload.get("scheduler_contacted") is not False
        or payload.get("submission_authorized") is not False
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("qiskit_compilation") is not False
        or payload.get("production_submission_enabled") is not False
    ):
        raise BootstrapActivationError("activation manifest is active or incomplete")
    action_ids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise BootstrapActivationError("activation pair is not an object")
        concurrency = pair.get("concurrency")
        if (
            concurrency
            != {
                "mode": "independent_after_same_qualified_seed",
                "join_required_before_launch": False,
                "siam_dependency": False,
                "apm_dependency": False,
                "arms": ["exact_reference", "avqds_scout"],
            }
        ):
            raise BootstrapActivationError("activation pair is not concurrent")
        intents: list[BootstrapSubmissionIntent] = []
        for arm in ("exact_reference", "avqds_scout"):
            row = pair.get(arm)
            if not isinstance(row, Mapping):
                raise BootstrapActivationError("activation pair arm is missing")
            intent_row = row.get("intent")
            if not isinstance(intent_row, Mapping):
                raise BootstrapActivationError("activation pair intent is missing")
            intent = load_submission_intent(
                str(intent_row.get("path") or ""), repository_root=root
            )
            if (
                intent.file_sha256 != intent_row.get("sha256")
                or intent.package_kind != arm
                or intent.request.action_id != row.get("action_id")
                or intent.request.digest != row.get("execution_request_sha256")
                or intent.request.lane_id != pair.get("lane_id")
                or intent.request.inputs.get("cell_id") != pair.get("cell_id")
                or intent.request.inputs.get("reference_id")
                != pair.get("reference_id")
                or intent.unit.payload.get("seed_id") != pair.get("seed_id")
                or intent.unit.payload.get("package") != row.get("package")
                or intent.unit.payload.get("package_manifest")
                != row.get("package_manifest")
            ):
                raise BootstrapActivationError(
                    "activation manifest arm binding changed"
                )
            if intent.request.action_id in action_ids:
                raise BootstrapActivationError("activation action is repeated")
            action_ids.add(intent.request.action_id)
            intents.append(intent)
        if (
            intents[0].unit.payload.get("concurrent_peer_action_id")
            != intents[1].request.action_id
            or intents[1].unit.payload.get("concurrent_peer_action_id")
            != intents[0].request.action_id
        ):
            raise BootstrapActivationError("activation peers are not bidirectionally bound")
    return payload


def _intent_evidence_path(
    root: Path, intent: BootstrapSubmissionIntent, category: str
) -> Path:
    return _state_path(root, intent.lane_id, category, intent.unit.unit_id)


def _intent_transition_lock_path(
    root: Path, intent: BootstrapSubmissionIntent
) -> Path:
    return _intent_evidence_path(
        root, intent, "intent_transition_locks"
    ).with_suffix(".lock")


def _ensure_nofollow_directory(path: Path, *, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise BootstrapActivationError(
            "intent transition lock must remain inside repository root"
        ) from exc
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            try:
                cursor.mkdir(mode=0o700)
            except FileExistsError:
                pass
            metadata = cursor.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise BootstrapActivationError(
                f"intent transition lock parent is not a nofollow directory: {cursor}"
            )


class _IntentTransitionLock:
    """Crash-releasing advisory lock for one intent's state transitions."""

    def __init__(self, root: Path, intent: BootstrapSubmissionIntent):
        self.root = root
        self.path = _intent_transition_lock_path(root, intent)
        self._parent_descriptor: int | None = None
        self._descriptor: int | None = None

    def __enter__(self) -> "_IntentTransitionLock":
        if self.root != self.root.resolve(strict=True):
            raise BootstrapActivationError("repository root is not canonical")
        _ensure_nofollow_directory(self.path.parent, root=self.root)
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise BootstrapActivationError(
                "intent transition locking requires nofollow directory support"
            )
        directory_flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        lock_flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            parent_descriptor = os.open(self.path.parent, directory_flags)
            descriptor = os.open(
                self.path.name,
                lock_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            if "parent_descriptor" in locals():
                os.close(parent_descriptor)
            raise BootstrapActivationError(
                f"cannot safely open intent transition lock: {self.path}"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            linked = os.stat(
                self.path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            ):
                raise BootstrapActivationError(
                    "intent transition lock is not one linked regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            linked_after_lock = os.stat(
                self.path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(linked_after_lock.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (linked_after_lock.st_dev, linked_after_lock.st_ino)
            ):
                raise BootstrapActivationError(
                    "intent transition lock changed while being acquired"
                )
        except Exception:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                os.close(parent_descriptor)
            raise
        self._parent_descriptor = parent_descriptor
        self._descriptor = descriptor
        return self

    def __exit__(self, *_exc: Any) -> None:
        descriptor = self._descriptor
        parent_descriptor = self._parent_descriptor
        self._descriptor = None
        self._parent_descriptor = None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _generational_state_path(
    root: Path,
    intent: BootstrapSubmissionIntent,
    *,
    category: str,
    generation: int,
    predecessor_resolution_sha256: str | None,
) -> Path:
    if generation < 1:
        raise BootstrapActivationError("claim generation must be positive")
    if generation == 1:
        if predecessor_resolution_sha256 is not None:
            raise BootstrapActivationError(
                "generation-one claim cannot have a predecessor resolution"
            )
        identity = intent.unit.unit_id
    else:
        digest = _sha(
            predecessor_resolution_sha256,
            label="predecessor claim-resolution digest",
        )
        identity = f"{intent.unit.unit_id}--g{generation}-{digest[:16]}"
    return _state_path(root, intent.lane_id, category, identity)


def _claim_resolution_path(
    root: Path,
    intent: BootstrapSubmissionIntent,
    *,
    claim_generation: int,
    claim_sha256: str,
) -> Path:
    digest = _sha(claim_sha256, label="resolved claim digest")
    identity = f"{intent.unit.unit_id}--g{claim_generation}-{digest[:16]}"
    return _state_path(
        root, intent.lane_id, "pre_submit_claim_resolutions", identity
    )


def _claim_scoped_state_path(
    root: Path,
    intent: BootstrapSubmissionIntent,
    *,
    category: str,
    claim: Mapping[str, Any],
) -> Path:
    """Return the immutable state path owned by one claim generation."""

    try:
        generation = int(claim.get("claim_generation"))
    except (TypeError, ValueError) as exc:
        raise BootstrapActivationError("claim generation is invalid") from exc
    predecessor = claim.get("predecessor_resolution")
    predecessor_sha = (
        str(predecessor.get("sha256") or "")
        if isinstance(predecessor, Mapping)
        else None
    )
    return _generational_state_path(
        root,
        intent,
        category=category,
        generation=generation,
        predecessor_resolution_sha256=predecessor_sha,
    )


def _intent_binding(intent: BootstrapSubmissionIntent) -> dict[str, Any]:
    return {
        "path": intent.path,
        "sha256": intent.intent_sha256,
        "file_sha256": intent.file_sha256,
    }


def _logical_evidence_binding(
    root: Path, path: Path, *, logical_sha256: str
) -> dict[str, Any]:
    return {
        "path": _relative(root, path, label="activation evidence"),
        "sha256": logical_sha256,
        "file_sha256": _file_sha(path),
    }


def load_pre_submit_execution_claim(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(root, path, label="pre-submit claim", must_exist=True)
    payload = _read_object(source, label="pre-submit claim")
    _verify_seal(payload, key="pre_submit_claim_sha256", label="pre-submit claim")
    intent_row = payload.get("intent")
    snapshot_row = payload.get("verified_absent_snapshot")
    if not isinstance(intent_row, Mapping) or not isinstance(snapshot_row, Mapping):
        raise BootstrapActivationError("pre-submit claim bindings are incomplete")
    intent = load_submission_intent(
        str(intent_row.get("path") or ""), repository_root=root
    )
    snapshot = _load_snapshot_binding(
        root, snapshot_row, label="pre-submit claim snapshot"
    )
    outcome = _snapshot_outcome(intent, snapshot.payload)
    prior_outcome = (
        _snapshot_outcome(intent, snapshot.prior_payload)
        if isinstance(snapshot.prior_payload, Mapping)
        else None
    )
    upload_row = payload.get("upload_verification")
    dry_row = payload.get("dry_run_result")
    if not isinstance(upload_row, Mapping) or not isinstance(dry_row, Mapping):
        raise BootstrapActivationError("pre-submit claim preflights are incomplete")
    upload_path = _verified_evidence_path(
        root, upload_row, label="claimed upload verification"
    )
    dry_path = _verified_evidence_path(
        root, dry_row, label="claimed dry-run result"
    )
    upload = load_upload_verification(upload_path, repository_root=root)
    dry = load_dry_run_result(dry_path, repository_root=root)
    try:
        generation = int(payload.get("claim_generation"))
    except (TypeError, ValueError) as exc:
        raise BootstrapActivationError("claim generation is invalid") from exc
    predecessor_row = payload.get("predecessor_resolution")
    predecessor_sha = (
        str(predecessor_row.get("sha256") or "")
        if isinstance(predecessor_row, Mapping)
        else None
    )
    expected = _generational_state_path(
        root,
        intent,
        category="pre_submit_claims",
        generation=generation,
        predecessor_resolution_sha256=predecessor_sha,
    )
    predecessor: Mapping[str, Any] | None = None
    if generation == 1:
        if predecessor_row is not None:
            raise BootstrapActivationError(
                "generation-one claim has a predecessor resolution"
            )
    else:
        if not isinstance(predecessor_row, Mapping):
            raise BootstrapActivationError(
                "successor claim has no predecessor resolution"
            )
        predecessor_path = _verified_evidence_path(
            root, predecessor_row, label="predecessor claim resolution"
        )
        predecessor = load_pre_submit_claim_resolution(
            predecessor_path, repository_root=root
        )
    initial_absence_chain = (
        isinstance(prior_outcome, Mapping)
        and prior_outcome.get("status") == "indeterminate"
        and prior_outcome.get("indeterminate_reason")
        == "first_snapshot_for_exact_intent_identity"
        and isinstance(outcome.get("absence_proof"), Mapping)
        and outcome["absence_proof"].get("prior_snapshot_sha256")
        == snapshot.prior_payload.get("snapshot_sha256")
        and outcome["absence_proof"].get("prior_snapshot_status")
        == "indeterminate"
    )
    rejected_submit_absence_chain = (
        predecessor is not None
        and predecessor.get("status")
        == "verified_absent_after_rejected_submit"
        and isinstance(prior_outcome, Mapping)
        and prior_outcome.get("status") == "verified_absent"
        and prior_outcome.get("match_count") == 0
    )
    if (
        source != expected.resolve(strict=True)
        or payload.get("schema") != PRE_SUBMIT_EXECUTION_CLAIM_V1
        or payload.get("status") != "claimed_for_single_external_submit"
        or dict(intent_row) != _intent_binding(intent)
        or payload.get("lane_id") != intent.lane_id
        or payload.get("unit_id") != intent.unit.unit_id
        or payload.get("action_id") != intent.request.action_id
        or payload.get("execution_request_sha256") != intent.request.digest
        or payload.get("claim_generation") != generation
        or payload.get("claim_path")
        != _relative(root, expected, label="pre-submit claim")
        or outcome.get("status") != "verified_absent"
        or payload.get("absence_proof") != outcome.get("absence_proof")
        or not (initial_absence_chain or rejected_submit_absence_chain)
        or upload_row.get("sha256") != upload.get("upload_verification_sha256")
        or dry_row.get("sha256") != dry.get("dry_run_result_sha256")
        or dry.get("status") != "pass"
        or payload.get("single_external_submit_may_be_attempted") is not True
        or payload.get("unknown_outcome_requires_later_reconciliation") is not True
        or payload.get("automatic_retry_authorized") is not False
        or payload.get("claimed_at") != snapshot.payload.get("captured_at")
        or (
            predecessor is not None
            and (
                predecessor_row.get("sha256")
                != predecessor.get("pre_submit_claim_resolution_sha256")
                or predecessor.get("successor_claim_generation") != generation
                or predecessor.get("later_verified_absent_snapshot")
                != dict(snapshot_row)
            )
        )
    ):
        raise BootstrapActivationError("pre-submit claim identity diverged")
    _aware_time(payload.get("claimed_at"), label="claim claimed_at")
    return payload


def load_pre_submit_claim_consumption(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="pre-submit claim consumption", must_exist=True
    )
    payload = _read_object(source, label="pre-submit claim consumption")
    _verify_seal(
        payload,
        key="pre_submit_claim_consumption_sha256",
        label="pre-submit claim consumption",
    )
    claim_row = payload.get("pre_submit_claim")
    if not isinstance(claim_row, Mapping):
        raise BootstrapActivationError("claim consumption has no claim binding")
    claim_path = _verified_evidence_path(
        root, claim_row, label="consumed pre-submit claim"
    )
    claim = load_pre_submit_execution_claim(claim_path, repository_root=root)
    intent_row = claim.get("intent")
    assert isinstance(intent_row, Mapping)
    intent = load_submission_intent(
        str(intent_row.get("path") or ""), repository_root=root
    )
    expected = _claim_scoped_state_path(
        root,
        intent,
        category="pre_submit_claim_consumptions",
        claim=claim,
    )
    if (
        source != expected.resolve(strict=True)
        or payload.get("schema") != PRE_SUBMIT_CLAIM_CONSUMPTION_V1
        or payload.get("status") != "consumed_by_recorded_submit_result"
        or claim_row.get("sha256") != claim.get("pre_submit_claim_sha256")
        or payload.get("lane_id") != intent.lane_id
        or payload.get("unit_id") != intent.unit.unit_id
        or payload.get("action_id") != intent.request.action_id
        or payload.get("execution_request_sha256") != intent.request.digest
        or payload.get("claim_generation") != claim.get("claim_generation")
        or _SHA256.fullmatch(str(payload.get("submission_attempt_sha256") or ""))
        is None
        or _SHA256.fullmatch(str(payload.get("stdout_sha256") or "")) is None
        or _SHA256.fullmatch(str(payload.get("stderr_sha256") or "")) is None
        or payload.get("claim_consumed_before_result_publication") is not True
        or payload.get("automatic_retry_authorized") is not False
    ):
        raise BootstrapActivationError("claim consumption identity diverged")
    _aware_time(payload.get("submitted_at"), label="claim consumption submitted_at")
    return payload


def record_upload_verification(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    remote_bytes: int,
    remote_sha256: str,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Record an external transport's exact remote size/SHA verification."""

    root = Path(repository_root).resolve(strict=True)
    intent = load_submission_intent(intent_path, repository_root=root)
    output = _intent_evidence_path(root, intent, "upload_verifications")
    if output.exists():
        return load_upload_verification(output, repository_root=root)
    package = dict(intent.unit.payload["package"])
    size = int(remote_bytes)
    digest = _sha(remote_sha256, label="remote package digest")
    if size != package["bytes"] or digest != package["sha256"]:
        raise BootstrapActivationError(
            "remote upload does not exactly match the local package"
        )
    core = {
        "schema": UPLOAD_VERIFICATION_V1,
        "status": "verified",
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_id": intent.request.action_id,
        "execution_request_sha256": intent.request.digest,
        "intent": _intent_binding(intent),
        "local_package": package,
        "remote_package": {
            "path": intent.unit.payload["remote_package_path"],
            "bytes": size,
            "sha256": digest,
        },
        "remote_raw_evidence_policy": "preserve",
        "transport_contact_performed_by_coordinator": False,
        "recorded_at": _timestamp(recorded_at),
    }
    _atomic_write_immutable(output, _sealed(core, "upload_verification_sha256"))
    return load_upload_verification(output, repository_root=root)


def load_upload_verification(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(root, path, label="upload verification", must_exist=True)
    payload = _read_object(source, label="upload verification")
    _verify_seal(payload, key="upload_verification_sha256", label="upload verification")
    intent_row = payload.get("intent")
    if not isinstance(intent_row, Mapping):
        raise BootstrapActivationError("upload verification has no intent")
    intent = load_submission_intent(
        str(intent_row.get("path") or ""), repository_root=root
    )
    expected = _intent_evidence_path(root, intent, "upload_verifications")
    package = intent.unit.payload["package"]
    remote = payload.get("remote_package")
    if (
        source != expected.resolve(strict=True)
        or payload.get("schema") != UPLOAD_VERIFICATION_V1
        or payload.get("status") != "verified"
        or dict(intent_row) != _intent_binding(intent)
        or payload.get("lane_id") != intent.lane_id
        or payload.get("unit_id") != intent.unit.unit_id
        or payload.get("action_id") != intent.request.action_id
        or payload.get("execution_request_sha256") != intent.request.digest
        or payload.get("local_package") != package
        or not isinstance(remote, Mapping)
        or remote.get("path") != intent.unit.payload["remote_package_path"]
        or remote.get("bytes") != package["bytes"]
        or remote.get("sha256") != package["sha256"]
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("transport_contact_performed_by_coordinator") is not False
    ):
        raise BootstrapActivationError("upload verification identity diverged")
    _aware_time(payload.get("recorded_at"), label="upload recorded_at")
    return payload


def record_dry_run_result(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    returncode: int,
    dry_run_ad_path: str | Path,
    remote_submit_descriptor_path: str | Path,
    stdout: str = "",
    stderr: str = "",
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Record one transport-executed ``condor_submit -dry-run`` result."""

    root = Path(repository_root).resolve(strict=True)
    intent = load_submission_intent(intent_path, repository_root=root)
    upload_path = _intent_evidence_path(root, intent, "upload_verifications")
    upload = load_upload_verification(upload_path, repository_root=root)
    output = _intent_evidence_path(root, intent, "dry_run_results")
    ad_path = _repository_path(
        root, dry_run_ad_path, label="preserved dry-run ClassAd", must_exist=True
    )
    remote_descriptor_path = _repository_path(
        root,
        remote_submit_descriptor_path,
        label="preserved remote submit descriptor",
        must_exist=True,
    )
    local_descriptor_path = _repository_path(
        root,
        str(intent.unit.payload["submit_descriptor"]),
        label="package submit descriptor",
        must_exist=True,
    )
    if not ad_path.is_file() or ad_path.stat().st_size <= 0:
        raise BootstrapActivationError("preserved dry-run ClassAd is empty")
    if (
        not remote_descriptor_path.is_file()
        or remote_descriptor_path.read_bytes() != local_descriptor_path.read_bytes()
    ):
        raise BootstrapActivationError(
            "remote submit descriptor does not match the final package descriptor"
        )
    code = int(returncode)
    status = "pass" if code == 0 else "failed"
    existing = (
        load_dry_run_result(output, repository_root=root)
        if output.exists()
        else None
    )
    recorded_time = (
        str(existing["recorded_at"])
        if existing is not None and recorded_at is None
        else _timestamp(recorded_at)
    )
    core = {
        "schema": DRY_RUN_RESULT_V1,
        "status": status,
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_id": intent.request.action_id,
        "execution_request_sha256": intent.request.digest,
        "intent": _intent_binding(intent),
        "upload_verification": {
            "path": _relative(root, upload_path, label="upload verification"),
            "sha256": upload["upload_verification_sha256"],
            "file_sha256": _file_sha(upload_path),
        },
        "submit_descriptor": intent.unit.payload["submit_descriptor_binding"],
        "remote_submit_descriptor": _file_binding(
            root, remote_descriptor_path, include_bytes=True
        ),
        "package_preflight": intent.unit.payload["package_preflight"],
        "dry_run_ad": _file_binding(root, ad_path, include_bytes=True),
        "argv": [
            "condor_submit",
            "-dry-run",
            (
                PurePosixPath(intent.unit.payload["remote_workdir"])
                / "dryrun.ad"
            ).as_posix(),
            (
                PurePosixPath(intent.unit.payload["remote_workdir"])
                / Path(intent.unit.payload["submit_descriptor"]).name
            ).as_posix(),
        ],
        "returncode": code,
        "stdout": str(stdout),
        "stderr": str(stderr),
        "single_dry_run_invocation": True,
        "scheduler_submission_performed": False,
        "recorded_at": recorded_time,
    }
    candidate = _sealed(core, "dry_run_result_sha256")
    if existing is not None:
        if existing != candidate:
            raise BootstrapActivationError(
                "existing dry-run result binds different preserved artifacts"
            )
        return existing
    _atomic_write_immutable(output, candidate)
    return load_dry_run_result(output, repository_root=root)


def load_dry_run_result(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(root, path, label="dry-run result", must_exist=True)
    payload = _read_object(source, label="dry-run result")
    _verify_seal(payload, key="dry_run_result_sha256", label="dry-run result")
    intent_row = payload.get("intent")
    upload_row = payload.get("upload_verification")
    if not isinstance(intent_row, Mapping) or not isinstance(upload_row, Mapping):
        raise BootstrapActivationError("dry-run result bindings are incomplete")
    intent = load_submission_intent(
        str(intent_row.get("path") or ""), repository_root=root
    )
    upload_path = _verified_evidence_path(
        root, upload_row, label="upload verification"
    )
    upload = load_upload_verification(upload_path, repository_root=root)
    ad_path, ad_binding = _verified_binding(
        root,
        payload.get("dry_run_ad"),
        label="preserved dry-run ClassAd",
        require_bytes=True,
    )
    remote_descriptor_path, remote_descriptor_binding = _verified_binding(
        root,
        payload.get("remote_submit_descriptor"),
        label="preserved remote submit descriptor",
        require_bytes=True,
    )
    package_preflight_path, package_preflight_binding = _verified_binding(
        root,
        payload.get("package_preflight"),
        label="package preflight",
        require_bytes=False,
    )
    local_descriptor_path, local_descriptor_binding = _verified_binding(
        root,
        payload.get("submit_descriptor"),
        label="package submit descriptor",
        require_bytes=False,
    )
    expected = _intent_evidence_path(root, intent, "dry_run_results")
    code = int(payload.get("returncode"))
    status = "pass" if code == 0 else "failed"
    if (
        source != expected.resolve(strict=True)
        or payload.get("schema") != DRY_RUN_RESULT_V1
        or payload.get("status") != status
        or dict(intent_row) != _intent_binding(intent)
        or payload.get("lane_id") != intent.lane_id
        or payload.get("unit_id") != intent.unit.unit_id
        or payload.get("action_id") != intent.request.action_id
        or payload.get("execution_request_sha256") != intent.request.digest
        or upload_row.get("sha256") != upload["upload_verification_sha256"]
        or local_descriptor_binding
        != intent.unit.payload["submit_descriptor_binding"]
        or package_preflight_binding != intent.unit.payload["package_preflight"]
        or remote_descriptor_binding["sha256"]
        != local_descriptor_binding["sha256"]
        or remote_descriptor_binding["bytes"]
        != local_descriptor_path.stat().st_size
        or remote_descriptor_path.read_bytes() != local_descriptor_path.read_bytes()
        or ad_binding["bytes"] <= 0
        or payload.get("single_dry_run_invocation") is not True
        or payload.get("scheduler_submission_performed") is not False
    ):
        raise BootstrapActivationError("dry-run result identity diverged")
    if not ad_path.is_file() or not package_preflight_path.is_file():
        raise AssertionError("verified dry-run evidence changed unexpectedly")
    _aware_time(payload.get("recorded_at"), label="dry-run recorded_at")
    return payload


def _load_snapshot(
    root: Path,
    path: str | Path,
    *,
    queue_stdout_path: str | Path,
    history_stdout_path: str | Path,
    prior_snapshot_path: str | Path | None = None,
    prior_queue_stdout_path: str | Path | None = None,
    prior_history_stdout_path: str | Path | None = None,
) -> ValidatedSnapshotEvidence:
    source = _repository_path(
        root, path, label="combined scheduler snapshot", must_exist=True
    )
    queue_path = _repository_path(
        root, queue_stdout_path, label="preserved queue stdout", must_exist=True
    )
    history_path = _repository_path(
        root,
        history_stdout_path,
        label="preserved history stdout",
        must_exist=True,
    )
    payload = _read_object(source, label="combined scheduler snapshot")
    try:
        validate_combined_snapshot(
            payload,
            queue_stdout=queue_path,
            history_stdout=history_path,
        )
    except ValueError as exc:
        raise BootstrapActivationError(str(exc)) from exc
    context = payload.get("reconciliation_context")
    summarized_prior = (
        context.get("prior_snapshot") if isinstance(context, Mapping) else None
    )
    prior_arguments = (
        prior_snapshot_path,
        prior_queue_stdout_path,
        prior_history_stdout_path,
    )
    prior_payload: dict[str, Any] | None = None
    prior_evidence: dict[str, Any] | None = None
    if summarized_prior is None:
        if any(value is not None for value in prior_arguments):
            raise BootstrapActivationError(
                "snapshot has no referenced prior snapshot evidence"
            )
    else:
        if any(value is None for value in prior_arguments):
            raise BootstrapActivationError(
                "referenced prior snapshot evidence and raw streams are required"
            )
        assert prior_snapshot_path is not None
        assert prior_queue_stdout_path is not None
        assert prior_history_stdout_path is not None
        prior_source = _repository_path(
            root,
            prior_snapshot_path,
            label="referenced prior scheduler snapshot",
            must_exist=True,
        )
        prior_queue = _repository_path(
            root,
            prior_queue_stdout_path,
            label="preserved prior queue stdout",
            must_exist=True,
        )
        prior_history = _repository_path(
            root,
            prior_history_stdout_path,
            label="preserved prior history stdout",
            must_exist=True,
        )
        prior_payload = _read_object(
            prior_source, label="referenced prior scheduler snapshot"
        )
        try:
            validate_combined_snapshot(
                prior_payload,
                queue_stdout=prior_queue,
                history_stdout=prior_history,
            )
        except ValueError as exc:
            raise BootstrapActivationError(str(exc)) from exc
        current_intents = (
            context.get("intents") if isinstance(context, Mapping) else None
        )
        prior_context = prior_payload.get("reconciliation_context")
        prior_intents = (
            prior_context.get("intents")
            if isinstance(prior_context, Mapping)
            else None
        )
        prior_outcomes = prior_payload.get("reconciliation")
        if (
            not isinstance(summarized_prior, Mapping)
            or not isinstance(current_intents, Mapping)
            or not isinstance(prior_intents, Mapping)
            or not isinstance(prior_outcomes, Mapping)
        ):
            raise BootstrapActivationError(
                "referenced prior snapshot reconciliation is incomplete"
            )
        expected_outcomes: dict[str, Any] = {}
        for batch, current_intent in current_intents.items():
            prior_intent = prior_intents.get(batch)
            prior_outcome = prior_outcomes.get(batch)
            if (
                isinstance(current_intent, Mapping)
                and isinstance(prior_intent, Mapping)
                and isinstance(prior_outcome, Mapping)
                and prior_intent.get("intent_identity_sha256")
                == current_intent.get("intent_identity_sha256")
                and prior_outcome.get("intent_identity_sha256")
                == current_intent.get("intent_identity_sha256")
            ):
                expected_outcomes[str(batch)] = {
                    "status": prior_outcome.get("status"),
                    "intent_identity_sha256": current_intent.get(
                        "intent_identity_sha256"
                    ),
                }
        expected_prior = {
            "snapshot_sha256": prior_payload.get("snapshot_sha256"),
            "captured_at": prior_payload.get("captured_at"),
            "captured_at_unix_ns": prior_payload.get("captured_at_unix_ns"),
            "intent_outcomes": expected_outcomes,
        }
        if dict(summarized_prior) != expected_prior:
            raise BootstrapActivationError(
                "referenced prior snapshot artifact does not match its binding"
            )
        prior_evidence = {
            "path": _relative(root, prior_source, label="prior snapshot"),
            "sha256": prior_payload["snapshot_sha256"],
            "file_sha256": _file_sha(prior_source),
            "captured_at": prior_payload["captured_at"],
            "captured_at_unix_ns": prior_payload["captured_at_unix_ns"],
            "queue_stdout": _file_binding(root, prior_queue, include_bytes=True),
            "history_stdout": _file_binding(
                root, prior_history, include_bytes=True
            ),
        }
    binding = {
        "path": _relative(root, source, label="scheduler snapshot"),
        "sha256": payload["snapshot_sha256"],
        "file_sha256": _file_sha(source),
        "captured_at": payload["captured_at"],
        "captured_at_unix_ns": payload["captured_at_unix_ns"],
        "queue_stdout": _file_binding(root, queue_path, include_bytes=True),
        "history_stdout": _file_binding(root, history_path, include_bytes=True),
        "prior_snapshot": prior_evidence,
    }
    return ValidatedSnapshotEvidence(source, payload, binding, prior_payload)


def _load_snapshot_binding(
    root: Path, value: Any, *, label: str
) -> ValidatedSnapshotEvidence:
    if not isinstance(value, Mapping):
        raise BootstrapActivationError(f"{label} binding is missing")
    snapshot_path = _verified_evidence_path(root, value, label=label)
    queue_path, _queue = _verified_binding(
        root,
        value.get("queue_stdout"),
        label=f"{label} queue stdout",
        require_bytes=True,
        allow_empty_bytes=True,
    )
    history_path, _history = _verified_binding(
        root,
        value.get("history_stdout"),
        label=f"{label} history stdout",
        require_bytes=True,
        allow_empty_bytes=True,
    )
    prior = value.get("prior_snapshot")
    prior_path: Path | None = None
    prior_queue: Path | None = None
    prior_history: Path | None = None
    if prior is not None:
        if not isinstance(prior, Mapping):
            raise BootstrapActivationError(f"{label} prior binding is invalid")
        prior_path = _verified_evidence_path(
            root, prior, label=f"{label} prior snapshot"
        )
        prior_queue, _prior_queue = _verified_binding(
            root,
            prior.get("queue_stdout"),
            label=f"{label} prior queue stdout",
            require_bytes=True,
            allow_empty_bytes=True,
        )
        prior_history, _prior_history = _verified_binding(
            root,
            prior.get("history_stdout"),
            label=f"{label} prior history stdout",
            require_bytes=True,
            allow_empty_bytes=True,
        )
    loaded = _load_snapshot(
        root,
        snapshot_path,
        queue_stdout_path=queue_path,
        history_stdout_path=history_path,
        prior_snapshot_path=prior_path,
        prior_queue_stdout_path=prior_queue,
        prior_history_stdout_path=prior_history,
    )
    if loaded.binding != dict(value):
        raise BootstrapActivationError(f"{label} evidence binding changed")
    return loaded


def _snapshot_outcome(
    intent: BootstrapSubmissionIntent,
    snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    context = snapshot.get("reconciliation_context")
    outcomes = snapshot.get("reconciliation")
    scope = snapshot.get("scope")
    batch = intent.package_manifest.job_batch_name
    if (
        snapshot.get("schema") != COMBINED_SCOPED_SNAPSHOT_V1
        or not isinstance(context, Mapping)
        or context.get("schema") != RECONCILIATION_CONTEXT_V1
        or not isinstance(outcomes, Mapping)
        or not isinstance(scope, Mapping)
    ):
        raise BootstrapActivationError("snapshot lacks intent-first reconciliation")
    intents = context.get("intents")
    bound = intents.get(batch) if isinstance(intents, Mapping) else None
    expected_core = snapshot_intent_binding(intent)
    expected_identity = _digest(expected_core)
    outcome = outcomes.get(batch)
    lane_scope = (
        scope.get("lanes", {}).get(_lane_snapshot_key(intent.lane_id), {})
        if isinstance(scope.get("lanes"), Mapping)
        else {}
    )
    if (
        not isinstance(bound, Mapping)
        or dict(bound) != {**expected_core, "intent_identity_sha256": expected_identity}
        or not isinstance(outcome, Mapping)
        or outcome.get("intent_identity_sha256") != expected_identity
        or batch not in lane_scope.get("job_batch_names", [])
        or _aware_time(snapshot.get("captured_at"), label="snapshot captured_at")
        <= _aware_time(intent.recorded_at, label="intent recorded_at")
    ):
        raise BootstrapActivationError("snapshot is not bound to the exact intent")
    return outcome


def _exact_found_match(
    intent: BootstrapSubmissionIntent,
    outcome: Mapping[str, Any],
) -> Mapping[str, Any]:
    matches = outcome.get("matches")
    if outcome.get("status") != "found" or not isinstance(matches, list):
        raise BootstrapActivationError("snapshot outcome is not found")
    if len(matches) != 1 or outcome.get("match_count") != 1:
        raise BootstrapActivationError("found snapshot contains duplicate identities")
    row = matches[0]
    if not isinstance(row, Mapping):
        raise BootstrapActivationError("found scheduler identity is malformed")
    lane = row.get("acquisition_lane_id")
    unit = row.get("acquisition_unit_id")
    try:
        cluster = int(row.get("cluster_id"))
        process = int(row.get("proc_id"))
        status = int(row.get("job_status"))
    except (TypeError, ValueError) as exc:
        raise BootstrapActivationError("found scheduler identity is invalid") from exc
    if (
        cluster < 0
        or process != 0
        or status not in {1, 2, 3, 4, 5, 6, 7}
        or lane not in (None, "", intent.lane_id)
        or unit not in (None, "", intent.unit.unit_id)
        or row.get("paper_ii_action_id") != intent.request.action_id
        or row.get("paper_ii_execution_request_sha256") != intent.request.digest
        or row.get("paper_ii_source_epoch_id")
        != intent.source_epoch.source_epoch.source_epoch_id
    ):
        raise BootstrapActivationError("found scheduler identity crosses unit/lane/action")
    return row


def _terminal_history_row(
    history_path: Path,
    *,
    intent: BootstrapSubmissionIntent,
    cluster_id: int,
) -> dict[str, Any]:
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapActivationError(
            "terminal-failure history stream is unreadable"
        ) from exc
    if not isinstance(payload, list):
        raise BootstrapActivationError(
            "terminal-failure history stream is not a JSON list"
        )
    matches = [
        row
        for row in payload
        if isinstance(row, Mapping)
        and row.get("ClusterId") == cluster_id
        and row.get("ProcId") == 0
        and row.get("PaperIIActionId") == intent.request.action_id
        and row.get("PaperIIExecutionRequestSha256") == intent.request.digest
        and row.get("PaperIISourceEpochId")
        == intent.source_epoch.source_epoch.source_epoch_id
    ]
    if len(matches) != 1:
        raise BootstrapActivationError(
            "terminal-failure history has duplicate or missing exact identity"
        )
    row = matches[0]
    try:
        status = int(row.get("JobStatus"))
        exit_code = int(row.get("ExitCode"))
    except (TypeError, ValueError) as exc:
        raise BootstrapActivationError(
            "terminal-failure history lacks typed terminal status"
        ) from exc
    if status != 4 or exit_code == 0:
        raise BootstrapActivationError(
            "only a terminal nonzero exact-reference predecessor is recoverable"
        )
    return {
        "cluster_id": cluster_id,
        "proc_id": 0,
        "job_status": status,
        "exit_code": exit_code,
        "paper_ii_action_id": intent.request.action_id,
        "paper_ii_execution_request_sha256": intent.request.digest,
        "paper_ii_source_epoch_id": intent.source_epoch.source_epoch.source_epoch_id,
    }


def _terminal_archive_diagnostics(
    archive: Path,
    *,
    intent: BootstrapSubmissionIntent,
    cluster_id: int,
    exit_code: int,
) -> list[dict[str, Any]]:
    reference_prefix = str(
        intent.request.inputs.get("reference_identity_sha256") or ""
    )[:16]
    if len(reference_prefix) != 16:
        raise BootstrapActivationError(
            "terminal exact-reference identity prefix is invalid"
        )
    try:
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            names = [row.name for row in members]
            if len(names) != len(set(names)):
                raise BootstrapActivationError(
                    "terminal diagnostic archive repeats a member"
                )
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not member.isfile()
                ):
                    raise BootstrapActivationError(
                        "terminal diagnostic archive has an unsafe member"
                    )
            cell_token = f"/{reference_prefix}/exact_reference/"
            stderr_rows = [
                row
                for row in members
                if cell_token in f"/{row.name}"
                and row.name.endswith("/worker_stderr.log")
            ]
            event_rows = [
                row
                for row in members
                if cell_token in f"/{row.name}"
                and row.name.endswith(f".{cluster_id}.0.log")
            ]
            if len(stderr_rows) != 1 or len(event_rows) != 1:
                raise BootstrapActivationError(
                    "terminal diagnostic archive lacks one exact stderr/event log"
                )
            result: list[dict[str, Any]] = []
            for kind, member in (
                ("worker_stderr", stderr_rows[0]),
                ("condor_event_log", event_rows[0]),
            ):
                source = handle.extractfile(member)
                if source is None:
                    raise BootstrapActivationError(
                        "terminal diagnostic member is unreadable"
                    )
                content = source.read()
                if len(content) != member.size or len(content) > 4 * 1024 * 1024:
                    raise BootstrapActivationError(
                        "terminal diagnostic member size is invalid"
                    )
                text = content.decode("utf-8", errors="strict")
                if kind == "worker_stderr":
                    if (
                        "ExecutionRequest.from_mapping" not in text
                        or "execution request resolver binding mismatch" not in text
                    ):
                        raise BootstrapActivationError(
                            "terminal stderr is not the admitted binding-plumbing failure"
                        )
                elif (
                    f"({cluster_id}.000.000)" not in text
                    or f"return value {exit_code}" not in text
                ):
                    raise BootstrapActivationError(
                        "terminal Condor event log disagrees with scheduler history"
                    )
                result.append(
                    {
                        "kind": kind,
                        "path": member.name,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            return result
    except (OSError, tarfile.TarError, UnicodeDecodeError) as exc:
        if isinstance(exc, BootstrapActivationError):
            raise
        raise BootstrapActivationError(
            "terminal diagnostic archive is invalid"
        ) from exc


def record_exact_reference_terminal_failure_evidence(
    *,
    repository_root: str | Path,
    predecessor_intent_path: str | Path,
    provisional_result_path: str | Path,
    submission_receipt_path: str | Path,
    snapshot_path: str | Path,
    queue_stdout_path: str | Path,
    history_stdout_path: str | Path,
    diagnostic_archive_path: str | Path,
) -> dict[str, Any]:
    """Seal one known generation-one resolver-plumbing terminal failure."""

    root = Path(repository_root).resolve(strict=True)
    intent = load_submission_intent(
        predecessor_intent_path, repository_root=root
    )
    if (
        intent.package_kind != "exact_reference"
        or int(intent.unit.payload.get("operational_generation", 1)) != 1
    ):
        raise BootstrapActivationError(
            "terminal-failure predecessor is not generation-one exact work"
        )
    result_path = _repository_path(
        root, provisional_result_path, label="predecessor submit result", must_exist=True
    )
    result = load_provisional_submit_result(result_path, repository_root=root)
    receipt_path = _repository_path(
        root,
        submission_receipt_path,
        label="predecessor submission receipt",
        must_exist=True,
    )
    receipt = load_submission_receipt(receipt_path, repository_root=root)
    if (
        result.get("status") != "submitted"
        or result.get("intent") != _intent_binding(intent)
        or receipt.get("unit_id") != intent.unit.unit_id
        or receipt.get("intent_sha256") != intent.intent_sha256
    ):
        raise BootstrapActivationError(
            "terminal-failure predecessor was not one accepted submission"
        )
    snapshot = _load_snapshot(
        root,
        snapshot_path,
        queue_stdout_path=queue_stdout_path,
        history_stdout_path=history_stdout_path,
    )
    found = _exact_found_match(intent, _snapshot_outcome(intent, snapshot.payload))
    cluster_id = int(result.get("cluster_id"))
    if (
        int(found.get("cluster_id")) != cluster_id
        or receipt.get("remote", {}).get("cluster_id") != cluster_id
    ):
        raise BootstrapActivationError(
            "terminal-failure scheduler cluster crosses predecessor submission"
        )
    history_path = _repository_path(
        root,
        history_stdout_path,
        label="terminal-failure history stdout",
        must_exist=True,
    )
    terminal = _terminal_history_row(
        history_path, intent=intent, cluster_id=cluster_id
    )
    archive = _repository_path(
        root,
        diagnostic_archive_path,
        label="terminal diagnostic archive",
        must_exist=True,
    )
    diagnostics = _terminal_archive_diagnostics(
        archive,
        intent=intent,
        cluster_id=cluster_id,
        exit_code=int(terminal["exit_code"]),
    )
    output = _intent_evidence_path(
        root, intent, "terminal_failure_evidence"
    )
    core = {
        "schema": EXACT_REFERENCE_TERMINAL_FAILURE_V1,
        "status": "verified_terminal_execution_binding_plumbing_failure",
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_id": intent.request.action_id,
        "execution_request_sha256": intent.request.digest,
        "operational_generation": 1,
        "predecessor_intent": _intent_binding(intent),
        "provisional_submission_result": _logical_evidence_binding(
            root,
            result_path,
            logical_sha256=result["submission_result_sha256"],
        ),
        "submission_receipt": {
            "path": _relative(root, receipt_path, label="submission receipt"),
            "sha256": _file_sha(receipt_path),
            "file_sha256": _file_sha(receipt_path),
        },
        "terminal_snapshot": snapshot.binding,
        "scheduler_match": dict(found),
        "scheduler_match_sha256": _digest(dict(found)),
        "terminal_history_row": terminal,
        "diagnostic_archive": _file_binding(root, archive, include_bytes=True),
        "diagnostic_members": diagnostics,
        "failure_class": "execution_request_resolver_binding_mismatch",
        "scientific_computation_started": False,
        "scientific_identity_changed": False,
        "successor_operational_generation": 2,
        "automatic_retry_authorized": False,
        "maximum_successor_submissions": 1,
        "remote_raw_evidence_policy": "preserve",
        "recorded_at": snapshot.payload["captured_at"],
    }
    _atomic_write_immutable(
        output, _sealed(core, "terminal_failure_evidence_sha256")
    )
    return load_exact_reference_terminal_failure_evidence(
        output, repository_root=root
    )


def load_exact_reference_terminal_failure_evidence(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="terminal exact-reference failure", must_exist=True
    )
    payload = _read_object(source, label="terminal exact-reference failure")
    _verify_seal(
        payload,
        key="terminal_failure_evidence_sha256",
        label="terminal exact-reference failure",
    )
    intent_row = payload.get("predecessor_intent")
    result_row = payload.get("provisional_submission_result")
    receipt_row = payload.get("submission_receipt")
    snapshot_row = payload.get("terminal_snapshot")
    archive_row = payload.get("diagnostic_archive")
    if not all(
        isinstance(row, Mapping)
        for row in (intent_row, result_row, receipt_row, snapshot_row, archive_row)
    ):
        raise BootstrapActivationError(
            "terminal exact-reference failure bindings are incomplete"
        )
    intent_path = _verified_evidence_path(
        root, intent_row, label="terminal predecessor intent"
    )
    intent = load_submission_intent(intent_path, repository_root=root)
    expected = _intent_evidence_path(root, intent, "terminal_failure_evidence")
    result_path = _verified_evidence_path(
        root, result_row, label="terminal predecessor result"
    )
    result = load_provisional_submit_result(result_path, repository_root=root)
    receipt_path = _verified_evidence_path(
        root, receipt_row, label="terminal predecessor receipt"
    )
    receipt = load_submission_receipt(receipt_path, repository_root=root)
    snapshot = _load_snapshot_binding(
        root, snapshot_row, label="terminal scheduler snapshot"
    )
    found = _exact_found_match(intent, _snapshot_outcome(intent, snapshot.payload))
    cluster_id = int(result.get("cluster_id"))
    history_path = root / snapshot.binding["history_stdout"]["path"]
    terminal = _terminal_history_row(
        history_path, intent=intent, cluster_id=cluster_id
    )
    archive_path, archive_binding = _verified_binding(
        root,
        archive_row,
        label="terminal diagnostic archive",
        require_bytes=True,
    )
    diagnostics = _terminal_archive_diagnostics(
        archive_path,
        intent=intent,
        cluster_id=cluster_id,
        exit_code=int(terminal["exit_code"]),
    )
    if (
        source != expected.resolve(strict=True)
        or payload.get("schema") != EXACT_REFERENCE_TERMINAL_FAILURE_V1
        or payload.get("status")
        != "verified_terminal_execution_binding_plumbing_failure"
        or intent.package_kind != "exact_reference"
        or int(intent.unit.payload.get("operational_generation", 1)) != 1
        or dict(intent_row) != _intent_binding(intent)
        or result.get("status") != "submitted"
        or result.get("intent") != _intent_binding(intent)
        or result_row.get("sha256") != result.get("submission_result_sha256")
        or receipt.get("unit_id") != intent.unit.unit_id
        or receipt.get("intent_sha256") != intent.intent_sha256
        or receipt.get("remote", {}).get("cluster_id") != cluster_id
        or int(found.get("cluster_id")) != cluster_id
        or payload.get("scheduler_match") != dict(found)
        or payload.get("scheduler_match_sha256") != _digest(dict(found))
        or payload.get("terminal_history_row") != terminal
        or payload.get("diagnostic_archive") != archive_binding
        or payload.get("diagnostic_members") != diagnostics
        or payload.get("failure_class")
        != "execution_request_resolver_binding_mismatch"
        or payload.get("scientific_computation_started") is not False
        or payload.get("scientific_identity_changed") is not False
        or payload.get("operational_generation") != 1
        or payload.get("successor_operational_generation") != 2
        or payload.get("automatic_retry_authorized") is not False
        or payload.get("maximum_successor_submissions") != 1
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("recorded_at") != snapshot.payload.get("captured_at")
    ):
        raise BootstrapActivationError(
            "terminal exact-reference failure identity diverged"
        )
    return payload


def prepare_exact_reference_terminal_failure_successor(
    *,
    repository_root: str | Path,
    predecessor_intent_path: str | Path,
    terminal_failure_evidence_path: str | Path,
    repaired_package_preflight_path: str | Path,
    remote_root: str,
    recorded_at: str | None = None,
) -> BootstrapSubmissionIntent:
    """Prepare the only admitted generation-two exact plumbing successor."""

    root = Path(repository_root).resolve(strict=True)
    predecessor = load_submission_intent(
        predecessor_intent_path, repository_root=root
    )
    failure_path = _repository_path(
        root,
        terminal_failure_evidence_path,
        label="terminal exact-reference failure evidence",
        must_exist=True,
    )
    failure = load_exact_reference_terminal_failure_evidence(
        failure_path, repository_root=root
    )
    if (
        predecessor.package_kind != "exact_reference"
        or int(predecessor.unit.payload.get("operational_generation", 1)) != 1
        or failure.get("predecessor_intent") != _intent_binding(predecessor)
        or failure.get("successor_operational_generation") != 2
    ):
        raise BootstrapActivationError(
            "terminal successor does not follow one generation-one exact failure"
        )
    prepared = PreparedExactReferenceActionPackage.load(
        repaired_package_preflight_path, repository_root=root
    )
    if (
        prepared.operational_generation != 2
        or prepared.request_sha256 != predecessor.request.digest
        or prepared.source_epoch != predecessor.source_epoch
    ):
        raise BootstrapActivationError(
            "repaired package changes scientific identity or operational generation"
        )
    manifest = BootstrapPackageManifest.load(
        root / prepared.package_manifest.path, repository_root=root
    )
    if (
        manifest.request != predecessor.request
        or manifest.job_batch_name == predecessor.package_manifest.job_batch_name
    ):
        raise BootstrapActivationError(
            "successor package did not advance its JobBatchName"
        )
    registry = build_identity_registry()
    references = tuple(
        row
        for row in registry.references
        if row.reference_id
        == predecessor.request.inputs.get("reference_id")
    )
    if len(references) != 1:
        raise BootstrapActivationError(
            "terminal successor has no canonical exact reference"
        )
    reference = references[0]
    seeds = tuple(row for row in registry.seeds if row.seed_id == reference.seed_id)
    if len(seeds) != 1:
        raise BootstrapActivationError(
            "terminal successor has no canonical seed"
        )
    seed = seeds[0]
    request_path = root / prepared.request.path
    source_archive = root / prepared.source_archive.path
    image_identity_path = root / prepared.runtime_image.identity_path
    base_unit = _unit_for_package(
        root,
        request_path=request_path,
        request=predecessor.request,
        reference=reference,
        seed=seed,
        source=predecessor.source_epoch,
        source_archive=source_archive,
        runtime_image_identity_path=image_identity_path,
        package_kind="exact_reference",
        package=root / prepared.package.path,
        package_manifest=manifest,
        descriptor=root / prepared.submit_descriptor.path,
        preflight=root / prepared.preflight_receipt.path,
        remote_root=_safe_remote_path(remote_root, label="successor remote root"),
        concurrent_peer_action_id=str(
            predecessor.unit.payload.get("concurrent_peer_action_id") or ""
        ),
    )
    payload = dict(base_unit.payload)
    predecessor_binding = _intent_binding(predecessor)
    failure_binding = _logical_evidence_binding(
        root,
        failure_path,
        logical_sha256=failure["terminal_failure_evidence_sha256"],
    )
    remote_workdir = (
        PurePosixPath(str(payload["remote_workdir"])).parent
        / "exact_reference_g2"
    ).as_posix()
    payload.update(
        {
            "remote_workdir": remote_workdir,
            "remote_package_path": (
                PurePosixPath(remote_workdir) / Path(prepared.package.path).name
            ).as_posix(),
            "operational_generation": 2,
            "predecessor_intent": predecessor_binding,
            "terminal_failure_evidence": failure_binding,
            "successor_reason": (
                "verified_terminal_execution_binding_plumbing_failure"
            ),
            "scientific_identity_changed": False,
            "automatic_retry_authorized": False,
        }
    )
    unit = SubmissionUnit(
        unit_id=exact_reference_submission_unit_id(
            predecessor.request, operational_generation=2
        ),
        action_ids=(predecessor.request.action_id,),
        payload=payload,
    )
    descriptor = (root / prepared.submit_descriptor.path).read_text(
        encoding="utf-8"
    )
    if (
        f'+AcquisitionUnitId = "{unit.unit_id}"' not in descriptor
        or f'+JobBatchName = "{manifest.job_batch_name}"' not in descriptor
        or "+PaperIIOperationalGeneration = 2" not in descriptor
    ):
        raise BootstrapActivationError(
            "successor descriptor does not expose its operational identity"
        )
    return _ensure_submission_intent(
        root,
        lane_id=predecessor.lane_id,
        unit=unit,
        recorded_at=recorded_at,
    )


def _avqds_failure_recommended_resources(
    payload: Mapping[str, Any],
) -> AVQDSCHTCResources:
    retry = payload.get("retry_disposition")
    resources = retry.get("recommended_resources") if isinstance(retry, Mapping) else None
    if not isinstance(resources, Mapping):
        raise BootstrapActivationError(
            "terminal AVQDS failure has no recommended resources"
        )
    try:
        memory_mib = int(resources["memory_mib"])
        disk_kib = int(resources["disk_kib"])
        if memory_mib % 1024 or disk_kib % (1024 * 1024):
            raise ValueError("resources are not whole GiB")
        return AVQDSCHTCResources(
            request_cpus=int(resources["cpus"]),
            request_memory_gib=memory_mib // 1024,
            request_disk_gib=disk_kib // (1024 * 1024),
            max_runtime_seconds=int(resources["walltime_seconds"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BootstrapActivationError(
            "terminal AVQDS failure resources are invalid"
        ) from exc


def _avqds_failure_intent_binding(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(payload.get("path") or ""),
        "sha256": str(payload.get("intent_sha256") or ""),
        "file_sha256": str(payload.get("file_sha256") or ""),
    }


def load_avqds_scout_terminal_failure_evidence(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Authenticate one retained non-resource AVQDS validator failure."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="terminal AVQDS-scout failure", must_exist=True
    )
    payload = _read_object(source, label="terminal AVQDS-scout failure")
    _verify_seal(
        payload,
        key="failure_receipt_sha256",
        label="terminal AVQDS-scout failure",
    )
    submission = payload.get("submission")
    failure = payload.get("failure")
    diagnosis = payload.get("diagnosis")
    retry = payload.get("retry_disposition")
    observed = payload.get("observed_execution")
    scheduler = payload.get("scheduler")
    archive_row = payload.get("archive")
    request_row = payload.get("execution_request")
    work_order_row = payload.get("work_order")
    remote = payload.get("remote_preservation")
    raw_recovery = payload.get("raw_run_recovery")
    if not all(
        isinstance(row, Mapping)
        for row in (
            submission,
            failure,
            diagnosis,
            retry,
            observed,
            scheduler,
            archive_row,
            request_row,
            work_order_row,
            remote,
            raw_recovery,
        )
    ):
        raise BootstrapActivationError(
            "terminal AVQDS-scout failure evidence is incomplete"
        )
    intent_row = submission.get("intent")
    if not isinstance(intent_row, Mapping):
        raise BootstrapActivationError(
            "terminal AVQDS-scout failure has no predecessor intent"
        )
    normalized_intent_row = _avqds_failure_intent_binding(intent_row)
    intent_path = _verified_evidence_path(
        root,
        normalized_intent_row,
        label="terminal AVQDS predecessor intent",
    )
    predecessor = load_frozen_submitted_intent(
        intent_path, repository_root=root
    )
    request_path, request_binding = _verified_binding(
        root,
        request_row,
        label="terminal AVQDS execution request",
        require_bytes=False,
    )
    request = ExecutionRequest.from_mapping(
        _read_object(request_path, label="terminal AVQDS execution request")
    )
    work_path, work_binding = _verified_binding(
        root,
        work_order_row,
        label="terminal AVQDS work order",
        require_bytes=False,
    )
    work_payload = _read_object(work_path, label="terminal AVQDS work order")
    _verify_seal(
        work_payload,
        key="work_order_sha256",
        label="terminal AVQDS work order",
    )
    archive_path, archive_binding = _verified_binding(
        root,
        archive_row,
        label="terminal AVQDS diagnostic archive",
        require_bytes=True,
    )
    try:
        with tarfile.open(archive_path, mode="r:gz") as handle:
            members = handle.getmembers()
            if not members or any(
                PurePosixPath(member.name).is_absolute()
                or ".." in PurePosixPath(member.name).parts
                for member in members
            ):
                raise BootstrapActivationError(
                    "terminal AVQDS diagnostic archive is unsafe"
                )
    except (OSError, tarfile.TarError) as exc:
        if isinstance(exc, BootstrapActivationError):
            raise
        raise BootstrapActivationError(
            "terminal AVQDS diagnostic archive is invalid"
        ) from exc
    resources = _avqds_failure_recommended_resources(payload)
    observed_resources = AVQDSCHTCResources(
        request_cpus=int(observed.get("requested_cpus", 0)),
        request_memory_gib=int(observed.get("requested_memory_mib", 0)) // 1024,
        request_disk_gib=int(observed.get("requested_disk_kib", 0))
        // (1024 * 1024),
        max_runtime_seconds=resources.max_runtime_seconds,
    )
    if (
        payload.get("schema") != AVQDS_SCOUT_TERMINAL_FAILURE_V1
        or payload.get("status")
        != "TERMINAL_POST_RUN_VALIDATION_FAILURE_RETAINED_NO_SCIENTIFIC_OUTPUT"
        or predecessor.package_kind != "avqds_scout"
        or predecessor.request != request
        or normalized_intent_row != _intent_binding(predecessor)
        or payload.get("lane_id") != request.lane_id
        or payload.get("action_id") != request.action_id
        or payload.get("execution_request_sha256") != request.digest
        or payload.get("source_epoch_identity_sha256")
        != predecessor.source_epoch.source_epoch.identity_sha256
        or dict(request_row) != request_binding
        or work_order_row.get("sha256") != work_binding["sha256"]
        or work_order_row.get("work_order_sha256")
        != work_payload.get("work_order_sha256")
        or work_payload.get("execution_request") != request.to_json_dict()
        or work_order_row.get("executor_overlay_revision")
        != work_payload.get("executor_overlay", {}).get("overlay_revision")
        or payload.get("archive", {}).get("sha256") != archive_binding["sha256"]
        or payload.get("archive", {}).get("bytes") != archive_binding["bytes"]
        or archive_row.get("gzip_integrity_passed") is not True
        or archive_row.get("tar_integrity_passed") is not True
        or archive_row.get("remote_raw_evidence_preserved") is not True
        or failure.get("classification")
        != "post_run_contract_validation_failure"
        or failure.get("failure_phase")
        != "post_execution_raw_run_validation_before_atomic_promotion"
        or failure.get("non_resource_failure") is not True
        or failure.get("resource_limit_exceeded") is not False
        or failure.get("trajectory_accuracy_admitted") is not False
        or failure.get("usable_scientific_output_present") is not False
        or diagnosis.get("root_cause_confirmed") is not True
        or diagnosis.get("scientific_contract_change_required") is not False
        or retry.get("automatic_retry_authorized") is not False
        or retry.get("peer_avqds_submissions_authorized") is not False
        or retry.get("recommended_canary_only") is not True
        or retry.get("requires_digest_verified_regression_fix") is not True
        or retry.get("requires_new_executor_overlay_and_package_epoch") is not True
        or retry.get("scientific_policy_change_authorized") is not False
        or observed_resources != resources
        or observed.get("scientific_computation_completed") is not True
        or scheduler.get("job_status") != 4
        or scheduler.get("exit_code") != 1
        or remote.get("remote_mutation_performed") is not False
        or remote.get("remote_raw_evidence_preserved") is not True
        or raw_recovery.get("recovery_status")
        != "irrecoverable_after_condor_scratch_reclamation"
        or raw_recovery.get("failure_evidence_package_included_part") is not False
    ):
        raise BootstrapActivationError(
            "terminal AVQDS-scout failure identity diverged"
        )
    return payload


def prepare_avqds_scout_terminal_failure_successor(
    *,
    repository_root: str | Path,
    predecessor_intent_path: str | Path,
    terminal_failure_evidence_path: str | Path,
    repaired_package_preflight_path: str | Path,
    remote_root: str,
    recorded_at: str | None = None,
) -> BootstrapSubmissionIntent:
    """Prepare one inert generation-two AVQDS canary with unchanged science."""

    root = Path(repository_root).resolve(strict=True)
    predecessor = load_frozen_submitted_intent(
        predecessor_intent_path, repository_root=root
    )
    failure_path = _repository_path(
        root,
        terminal_failure_evidence_path,
        label="terminal AVQDS failure evidence",
        must_exist=True,
    )
    failure = load_avqds_scout_terminal_failure_evidence(
        failure_path, repository_root=root
    )
    predecessor_prepared = validate_frozen_submitted_avqds_prefix_package(
        root / predecessor.unit.payload["package_preflight"]["path"],
        repository_root=root,
    )
    prepared = validate_avqds_prefix_package(
        repaired_package_preflight_path, repository_root=root
    )
    if (
        predecessor.package_kind != "avqds_scout"
        or int(predecessor.unit.payload.get("operational_generation", 1)) != 1
        or _avqds_failure_intent_binding(
            failure.get("submission", {}).get("intent", {})
        )
        != _intent_binding(predecessor)
        or prepared.operational_generation != 2
        or prepared.execution_request_sha256 != predecessor.request.digest
        or prepared.resources != predecessor_prepared.resources
        or prepared.resources != _avqds_failure_recommended_resources(failure)
    ):
        raise BootstrapActivationError(
            "AVQDS successor changes science, provenance, or resources"
        )
    manifest = BootstrapPackageManifest.load(
        root / prepared.package_manifest.path, repository_root=root
    )
    if (
        manifest.request != predecessor.request
        or manifest.source_epoch_receipt
        != predecessor.package_manifest.source_epoch_receipt
        or manifest.job_batch_name == predecessor.package_manifest.job_batch_name
    ):
        raise BootstrapActivationError(
            "AVQDS successor package did not advance only operational identity"
        )
    registry = build_identity_registry()
    references = tuple(
        row
        for row in registry.references
        if row.reference_id == predecessor.request.inputs.get("reference_id")
    )
    if len(references) != 1:
        raise BootstrapActivationError(
            "AVQDS successor has no canonical exact-reference identity"
        )
    reference = references[0]
    seeds = tuple(row for row in registry.seeds if row.seed_id == reference.seed_id)
    if len(seeds) != 1:
        raise BootstrapActivationError("AVQDS successor has no canonical seed")
    seed = seeds[0]
    image_identity_path = root / predecessor.unit.payload["runtime_image_identity"][
        "path"
    ]
    preflight_payload = _read_object(
        root / prepared.preflight.path, label="successor AVQDS package preflight"
    )
    image = load_runtime_image_identity(
        image_identity_path, repository_root=root
    )
    if preflight_payload.get("runtime_image_identity") != {
        "path": image.identity_path,
        "sha256": image.identity_sha256,
    }:
        raise BootstrapActivationError("AVQDS successor changes runtime image")
    request_path = _repository_path(
        root,
        str(failure["execution_request"]["path"]),
        label="successor AVQDS execution request",
        must_exist=True,
    )
    base_unit = _unit_for_package(
        root,
        request_path=request_path,
        request=predecessor.request,
        reference=reference,
        seed=seed,
        source=predecessor.source_epoch,
        source_archive=root / prepared.runtime_source_archive.path,
        runtime_image_identity_path=image_identity_path,
        package_kind="avqds_scout",
        package=root / prepared.package.path,
        package_manifest=manifest,
        descriptor=root / prepared.descriptor.path,
        preflight=root / prepared.preflight.path,
        remote_root=_safe_remote_path(remote_root, label="successor remote root"),
        concurrent_peer_action_id=str(
            predecessor.unit.payload.get("concurrent_peer_action_id") or ""
        ),
    )
    payload = dict(base_unit.payload)
    remote_workdir = (
        PurePosixPath(str(payload["remote_workdir"])).parent
        / "avqds_scout_g2"
    ).as_posix()
    payload.update(
        {
            "remote_workdir": remote_workdir,
            "remote_package_path": (
                PurePosixPath(remote_workdir) / Path(prepared.package.path).name
            ).as_posix(),
            "operational_generation": 2,
            "predecessor_intent": _intent_binding(predecessor),
            "terminal_failure_evidence": _logical_evidence_binding(
                root,
                failure_path,
                logical_sha256=failure["failure_receipt_sha256"],
            ),
            "successor_reason": (
                "verified_terminal_post_run_contract_validation_failure"
            ),
            "scientific_identity_changed": False,
            "canary_only": True,
            "automatic_retry_authorized": False,
        }
    )
    unit = SubmissionUnit(
        unit_id=avqds_scout_submission_unit_id(
            predecessor.request, operational_generation=2
        ),
        action_ids=(predecessor.request.action_id,),
        payload=payload,
    )
    descriptor = (root / prepared.descriptor.path).read_text(encoding="utf-8")
    if (
        f'+AcquisitionUnitId = "{unit.unit_id}"' not in descriptor
        or f'+JobBatchName = "{manifest.job_batch_name}"' not in descriptor
        or "+PaperIIOperationalGeneration = 2" not in descriptor
    ):
        raise BootstrapActivationError(
            "AVQDS successor descriptor lacks its operational identity"
        )
    return _ensure_submission_intent(
        root,
        lane_id=predecessor.lane_id,
        unit=unit,
        recorded_at=recorded_at,
    )


def load_pre_submit_claim_resolution(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(
        root, path, label="pre-submit claim resolution", must_exist=True
    )
    payload = _read_object(source, label="pre-submit claim resolution")
    _verify_seal(
        payload,
        key="pre_submit_claim_resolution_sha256",
        label="pre-submit claim resolution",
    )
    intent_row = payload.get("intent")
    claim_row = payload.get("predecessor_claim")
    later_row = payload.get("later_verified_absent_snapshot")
    if not all(isinstance(row, Mapping) for row in (intent_row, claim_row, later_row)):
        raise BootstrapActivationError("claim resolution bindings are incomplete")
    assert isinstance(intent_row, Mapping)
    assert isinstance(claim_row, Mapping)
    assert isinstance(later_row, Mapping)
    intent = load_submission_intent(
        str(intent_row.get("path") or ""), repository_root=root
    )
    claim_path = _verified_evidence_path(
        root, claim_row, label="resolved pre-submit claim"
    )
    claim = load_pre_submit_execution_claim(claim_path, repository_root=root)
    later = _load_snapshot_binding(
        root, later_row, label="later post-claim snapshot"
    )
    later_outcome = _snapshot_outcome(intent, later.payload)
    try:
        generation = int(claim.get("claim_generation"))
    except (TypeError, ValueError) as exc:
        raise BootstrapActivationError("resolved claim generation is invalid") from exc
    expected = _claim_resolution_path(
        root,
        intent,
        claim_generation=generation,
        claim_sha256=str(claim.get("pre_submit_claim_sha256") or ""),
    )
    claim_time = _aware_time(claim.get("claimed_at"), label="claim claimed_at")
    later_time = _aware_time(
        later.payload.get("captured_at"), label="later post-claim captured_at"
    )
    common_diverged = (
        source != expected.resolve(strict=True)
        or payload.get("schema") != PRE_SUBMIT_CLAIM_RESOLUTION_V1
        or dict(intent_row) != _intent_binding(intent)
        or claim_row.get("sha256") != claim.get("pre_submit_claim_sha256")
        or payload.get("lane_id") != intent.lane_id
        or payload.get("unit_id") != intent.unit.unit_id
        or payload.get("action_id") != intent.request.action_id
        or payload.get("execution_request_sha256") != intent.request.digest
        or payload.get("predecessor_claim_generation") != generation
        or payload.get("successor_claim_generation") != generation + 1
        or payload.get("resolution_path")
        != _relative(root, expected, label="claim resolution")
        or later.binding != dict(later_row)
        or later_outcome.get("status") != "verified_absent"
        or later_outcome.get("match_count") != 0
        or later_outcome.get("matches") != []
        or payload.get("absence_proof") != later_outcome.get("absence_proof")
        or payload.get("automatic_retry_authorized") is not False
        or payload.get("resolved_at") != later.payload.get("captured_at")
    )
    if common_diverged:
        raise BootstrapActivationError("claim resolution identity diverged")

    status = payload.get("status")
    if status == "verified_absent_after_unresolved_claim":
        first_row = payload.get("first_post_claim_snapshot")
        if not isinstance(first_row, Mapping):
            raise BootstrapActivationError(
                "unresolved-claim resolution lacks its first snapshot"
            )
        first = _load_snapshot_binding(
            root, first_row, label="first post-claim snapshot"
        )
        first_outcome = _snapshot_outcome(intent, first.payload)
        first_time = _aware_time(
            first.payload.get("captured_at"), label="first post-claim captured_at"
        )
        if (
            first.binding != dict(first_row)
            or first.binding.get("prior_snapshot") is not None
            or first_outcome.get("status") != "indeterminate"
            or first_outcome.get("match_count") != 0
            or first_outcome.get("indeterminate_reason")
            != "first_snapshot_for_exact_intent_identity"
            or later_outcome.get("absence_proof", {}).get(
                "prior_snapshot_sha256"
            )
            != first.payload.get("snapshot_sha256")
            or not (claim_time < first_time < later_time)
            or payload.get("no_submit_result_observed") is not True
            or payload.get("no_submission_receipt_observed") is not True
        ):
            raise BootstrapActivationError("claim resolution identity diverged")
    elif status == "verified_absent_after_rejected_submit":
        result_row = payload.get("failed_submission_result")
        consumption_row = payload.get("predecessor_claim_consumption")
        if not isinstance(result_row, Mapping) or not isinstance(
            consumption_row, Mapping
        ):
            raise BootstrapActivationError(
                "rejected-submit resolution lacks failure evidence"
            )
        result_path = _verified_evidence_path(
            root, result_row, label="failed provisional submit result"
        )
        result = load_provisional_submit_result(result_path, repository_root=root)
        consumption_path = _verified_evidence_path(
            root, consumption_row, label="rejected-submit claim consumption"
        )
        consumption = load_pre_submit_claim_consumption(
            consumption_path, repository_root=root
        )
        submitted_time = _aware_time(
            result.get("submitted_at"), label="failed submit submitted_at"
        )
        prior_payload = later.prior_payload
        if (
            generation != 1
            or result.get("status") != "failed"
            or result.get("cluster_id") is not None
            or result.get("acceptance_evidence") != "condor_submit_not_accepted"
            or int(result.get("condor_submit_returncode")) == 0
            or result.get("claim_generation") != generation
            or result.get("intent") != _intent_binding(intent)
            or result_row.get("sha256") != result.get("submission_result_sha256")
            or consumption_row.get("sha256")
            != consumption.get("pre_submit_claim_consumption_sha256")
            or result.get("pre_submit_claim_consumption")
            != dict(consumption_row)
            or consumption.get("pre_submit_claim", {}).get("sha256")
            != claim.get("pre_submit_claim_sha256")
            or not isinstance(prior_payload, Mapping)
            or later_outcome.get("absence_proof", {}).get(
                "prior_snapshot_sha256"
            )
            != prior_payload.get("snapshot_sha256")
            or not (claim_time < submitted_time < later_time)
            or payload.get("submit_rejection_was_definitive") is not True
            or payload.get("no_scheduler_acceptance_observed") is not True
            or payload.get("maximum_successor_submit_attempts") != 1
        ):
            raise BootstrapActivationError("claim resolution identity diverged")
    else:
        raise BootstrapActivationError("unsupported claim resolution status")
    return payload


def _write_submission_receipt(
    root: Path,
    *,
    intent: BootstrapSubmissionIntent,
    snapshot_evidence: ValidatedSnapshotEvidence,
    match: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = snapshot_evidence.payload
    output = _intent_evidence_path(root, intent, "submission_receipts")
    cluster = int(match["cluster_id"])
    process = int(match["proc_id"])
    remote = {
        "cluster_id": cluster,
        "process_ids": [process],
        "unit_id": intent.unit.unit_id,
        "job_batch_name": intent.package_manifest.job_batch_name,
        "action_ids": [intent.request.action_id],
        "scheduler_identity": f"{cluster}.{process}",
        "source_epoch_identity_sha256": (
            intent.source_epoch.source_epoch.identity_sha256
        ),
        "source_epoch_id": intent.source_epoch.source_epoch.source_epoch_id,
        "remote_raw_evidence_policy": "preserve",
        "recovery_evidence": {
            "intent": _intent_binding(intent),
            "snapshot": snapshot_evidence.binding,
            "scheduler_match_sha256": _digest(dict(match)),
            "scheduler_acceptance_query_performed_by_coordinator": False,
        },
    }
    receipt = {
        "schema": SUBMISSION_RECEIPT_SCHEMA_V1,
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_ids": [intent.request.action_id],
        "intent_sha256": intent.intent_sha256,
        "recovered_from_scheduler": True,
        "remote": remote,
        "recorded_at": snapshot["captured_at"],
    }
    _atomic_write_immutable(output, receipt)
    return load_submission_receipt(output, repository_root=root)


def _publish_claim_authorization(
    root: Path,
    *,
    intent: BootstrapSubmissionIntent,
    snapshot_evidence: ValidatedSnapshotEvidence,
    absence_proof: Mapping[str, Any],
    upload_binding: Mapping[str, Any],
    dry_binding: Mapping[str, Any],
    generation: int,
    predecessor_resolution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    predecessor_sha = (
        str(predecessor_resolution.get("sha256") or "")
        if predecessor_resolution is not None
        else None
    )
    claim_path = _generational_state_path(
        root,
        intent,
        category="pre_submit_claims",
        generation=generation,
        predecessor_resolution_sha256=predecessor_sha,
    )
    authorization_path = _generational_state_path(
        root,
        intent,
        category="submission_authorizations",
        generation=generation,
        predecessor_resolution_sha256=predecessor_sha,
    )
    claim_core = {
        "schema": PRE_SUBMIT_EXECUTION_CLAIM_V1,
        "status": "claimed_for_single_external_submit",
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_id": intent.request.action_id,
        "execution_request_sha256": intent.request.digest,
        "intent": _intent_binding(intent),
        "claim_generation": generation,
        "claim_path": _relative(root, claim_path, label="pre-submit claim"),
        "predecessor_resolution": (
            dict(predecessor_resolution)
            if predecessor_resolution is not None
            else None
        ),
        "upload_verification": dict(upload_binding),
        "dry_run_result": dict(dry_binding),
        "verified_absent_snapshot": snapshot_evidence.binding,
        "absence_proof": dict(absence_proof),
        "single_external_submit_may_be_attempted": True,
        "unknown_outcome_requires_later_reconciliation": True,
        "automatic_retry_authorized": False,
        "claimed_at": snapshot_evidence.payload["captured_at"],
    }
    claim = _sealed(claim_core, "pre_submit_claim_sha256")
    if _atomic_write_immutable(claim_path, claim) != "created":
        raise BootstrapActivationError(
            "existing pre-submit claim requires later scheduler reconciliation"
        )
    claim_binding = _logical_evidence_binding(
        root, claim_path, logical_sha256=claim["pre_submit_claim_sha256"]
    )
    authorization_core = {
        "schema": SUBMISSION_AUTHORIZATION_V1,
        "status": "authorized_single_submit",
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_id": intent.request.action_id,
        "execution_request_sha256": intent.request.digest,
        "intent": _intent_binding(intent),
        "claim_generation": generation,
        "authorization_path": _relative(
            root, authorization_path, label="submission authorization"
        ),
        "claim_resolution": (
            dict(predecessor_resolution)
            if predecessor_resolution is not None
            else None
        ),
        "upload_verification": dict(upload_binding),
        "dry_run_result": dict(dry_binding),
        "verified_absent_snapshot": snapshot_evidence.binding,
        "pre_submit_claim": claim_binding,
        "absence_proof": dict(absence_proof),
        "single_condor_submit_authorized": True,
        "automatic_retry_authorized": False,
        "production_submission_enabled": False,
        "recorded_at": snapshot_evidence.payload["captured_at"],
    }
    _atomic_write_immutable(
        authorization_path,
        _sealed(authorization_core, "submission_authorization_sha256"),
    )
    return load_submission_authorization(authorization_path, repository_root=root)


def authorize_from_snapshot(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    snapshot_path: str | Path,
    queue_stdout_path: str | Path,
    history_stdout_path: str | Path,
    prior_snapshot_path: str | Path | None = None,
    prior_queue_stdout_path: str | Path | None = None,
    prior_history_stdout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authorize one submit only after later absence, or recover a found job."""

    root = Path(repository_root).resolve(strict=True)
    intent = load_frozen_submitted_intent(intent_path, repository_root=root)
    with _IntentTransitionLock(root, intent):
        return _authorize_from_snapshot_locked(
            root=root,
            intent=intent,
            snapshot_path=snapshot_path,
            queue_stdout_path=queue_stdout_path,
            history_stdout_path=history_stdout_path,
            prior_snapshot_path=prior_snapshot_path,
            prior_queue_stdout_path=prior_queue_stdout_path,
            prior_history_stdout_path=prior_history_stdout_path,
        )


def _authorize_from_snapshot_locked(
    *,
    root: Path,
    intent: BootstrapSubmissionIntent,
    snapshot_path: str | Path,
    queue_stdout_path: str | Path,
    history_stdout_path: str | Path,
    prior_snapshot_path: str | Path | None,
    prior_queue_stdout_path: str | Path | None,
    prior_history_stdout_path: str | Path | None,
) -> dict[str, Any]:
    snapshot_evidence = _load_snapshot(
        root,
        snapshot_path,
        queue_stdout_path=queue_stdout_path,
        history_stdout_path=history_stdout_path,
        prior_snapshot_path=prior_snapshot_path,
        prior_queue_stdout_path=prior_queue_stdout_path,
        prior_history_stdout_path=prior_history_stdout_path,
    )
    snapshot = snapshot_evidence.payload
    outcome = _snapshot_outcome(intent, snapshot)
    if outcome.get("status") == "found":
        receipt = _write_submission_receipt(
            root,
            intent=intent,
            snapshot_evidence=snapshot_evidence,
            match=_exact_found_match(intent, outcome),
        )
        return {
            "status": "found_recovered_without_resubmission",
            "submission_authorized": False,
            "submission_receipt": _relative(
                root,
                _intent_evidence_path(root, intent, "submission_receipts"),
                label="submission receipt",
            ),
            "cluster_id": receipt["remote"]["cluster_id"],
        }
    if outcome.get("status") != "verified_absent":
        raise BootstrapActivationError(
            "first or indeterminate intent snapshot cannot authorize submission"
        )
    live_intent = _load_submission_intent(
        intent.path,
        repository_root=root,
        frozen_submitted=False,
    )
    if _intent_binding(live_intent) != _intent_binding(intent):
        raise BootstrapActivationError(
            "live submission intent changed before authorization"
        )
    intent = live_intent
    outcome = _snapshot_outcome(intent, snapshot)
    proof = outcome.get("absence_proof")
    prior_payload = snapshot_evidence.prior_payload
    prior_outcome = (
        _snapshot_outcome(intent, prior_payload)
        if isinstance(prior_payload, Mapping)
        else None
    )
    if (
        outcome.get("match_count") != 0
        or outcome.get("matches") != []
        or not isinstance(proof, Mapping)
        or proof.get("schema") != ABSENCE_PROOF_V1
        or not isinstance(prior_outcome, Mapping)
        or prior_outcome.get("status") != "indeterminate"
        or prior_outcome.get("indeterminate_reason")
        != "first_snapshot_for_exact_intent_identity"
        or proof.get("prior_snapshot_sha256")
        != prior_payload.get("snapshot_sha256")
        or proof.get("prior_snapshot_status") != "indeterminate"
    ):
        raise BootstrapActivationError("verified absence proof is incomplete")
    result_path = _intent_evidence_path(root, intent, "submission_results")
    receipt_path = _intent_evidence_path(root, intent, "submission_receipts")
    if result_path.exists() or receipt_path.exists():
        raise BootstrapActivationError(
            "existing provisional result or receipt blocks authorization"
        )
    upload_path = _intent_evidence_path(root, intent, "upload_verifications")
    dry_path = _intent_evidence_path(root, intent, "dry_run_results")
    upload = load_upload_verification(upload_path, repository_root=root)
    dry = load_dry_run_result(dry_path, repository_root=root)
    if dry.get("status") != "pass":
        raise BootstrapActivationError("failed dry-run blocks submission authorization")
    captured = _aware_time(snapshot["captured_at"], label="snapshot captured_at")
    if (
        _aware_time(upload["recorded_at"], label="upload recorded_at") > captured
        or _aware_time(dry["recorded_at"], label="dry-run recorded_at") > captured
    ):
        raise BootstrapActivationError(
            "authorization snapshot predates upload/dry-run evidence"
        )
    upload_binding = _logical_evidence_binding(
        root,
        upload_path,
        logical_sha256=upload["upload_verification_sha256"],
    )
    dry_binding = _logical_evidence_binding(
        root, dry_path, logical_sha256=dry["dry_run_result_sha256"]
    )
    return _publish_claim_authorization(
        root,
        intent=intent,
        snapshot_evidence=snapshot_evidence,
        absence_proof=proof,
        upload_binding=upload_binding,
        dry_binding=dry_binding,
        generation=1,
        predecessor_resolution=None,
    )


def load_submission_authorization(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(root, path, label="submission authorization", must_exist=True)
    payload = _read_object(source, label="submission authorization")
    _verify_seal(
        payload,
        key="submission_authorization_sha256",
        label="submission authorization",
    )
    intent_row = payload.get("intent")
    snapshot_row = payload.get("verified_absent_snapshot")
    if not isinstance(intent_row, Mapping) or not isinstance(snapshot_row, Mapping):
        raise BootstrapActivationError("submission authorization is incomplete")
    intent = load_submission_intent(
        str(intent_row.get("path") or ""), repository_root=root
    )
    snapshot_evidence = _load_snapshot_binding(
        root, snapshot_row, label="verified-absent snapshot"
    )
    snapshot = snapshot_evidence.payload
    outcome = _snapshot_outcome(intent, snapshot)
    try:
        generation = int(payload.get("claim_generation"))
    except (TypeError, ValueError) as exc:
        raise BootstrapActivationError("authorization generation is invalid") from exc
    resolution_row = payload.get("claim_resolution")
    resolution_sha = (
        str(resolution_row.get("sha256") or "")
        if isinstance(resolution_row, Mapping)
        else None
    )
    expected = _generational_state_path(
        root,
        intent,
        category="submission_authorizations",
        generation=generation,
        predecessor_resolution_sha256=resolution_sha,
    )
    resolution: Mapping[str, Any] | None = None
    if generation == 1:
        if resolution_row is not None:
            raise BootstrapActivationError(
                "generation-one authorization has a claim resolution"
            )
    else:
        if not isinstance(resolution_row, Mapping):
            raise BootstrapActivationError(
                "successor authorization has no claim resolution"
            )
        resolution_path = _verified_evidence_path(
            root, resolution_row, label="authorization claim resolution"
        )
        resolution = load_pre_submit_claim_resolution(
            resolution_path, repository_root=root
        )
    if (
        source != expected.resolve(strict=True)
        or payload.get("schema") != SUBMISSION_AUTHORIZATION_V1
        or payload.get("status") != "authorized_single_submit"
        or dict(intent_row) != _intent_binding(intent)
        or payload.get("lane_id") != intent.lane_id
        or payload.get("unit_id") != intent.unit.unit_id
        or payload.get("action_id") != intent.request.action_id
        or payload.get("execution_request_sha256") != intent.request.digest
        or payload.get("claim_generation") != generation
        or payload.get("authorization_path")
        != _relative(root, expected, label="submission authorization")
        or dict(snapshot_row) != snapshot_evidence.binding
        or outcome.get("status") != "verified_absent"
        or payload.get("absence_proof") != outcome.get("absence_proof")
        or payload.get("single_condor_submit_authorized") is not True
        or payload.get("automatic_retry_authorized") is not False
        or payload.get("production_submission_enabled") is not False
    ):
        raise BootstrapActivationError("submission authorization identity diverged")
    upload_row = payload.get("upload_verification")
    dry_row = payload.get("dry_run_result")
    claim_row = payload.get("pre_submit_claim")
    if (
        not isinstance(upload_row, Mapping)
        or not isinstance(dry_row, Mapping)
        or not isinstance(claim_row, Mapping)
    ):
        raise BootstrapActivationError("authorization preflight bindings are missing")
    upload_path = _verified_evidence_path(
        root, upload_row, label="upload verification"
    )
    dry_path = _verified_evidence_path(
        root, dry_row, label="dry-run result"
    )
    upload = load_upload_verification(upload_path, repository_root=root)
    dry = load_dry_run_result(dry_path, repository_root=root)
    claim_path = _verified_evidence_path(
        root, claim_row, label="pre-submit execution claim"
    )
    claim = load_pre_submit_execution_claim(claim_path, repository_root=root)
    if (
        upload_row.get("sha256") != upload.get("upload_verification_sha256")
        or dry_row.get("sha256") != dry.get("dry_run_result_sha256")
        or dry.get("status") != "pass"
        or claim_row.get("sha256") != claim.get("pre_submit_claim_sha256")
        or claim.get("verified_absent_snapshot") != snapshot_evidence.binding
        or claim.get("upload_verification") != dict(upload_row)
        or claim.get("dry_run_result") != dict(dry_row)
        or claim.get("claim_generation") != generation
        or claim.get("predecessor_resolution")
        != (dict(resolution_row) if resolution_row is not None else None)
        or (
            resolution is not None
            and resolution_row.get("sha256")
            != resolution.get("pre_submit_claim_resolution_sha256")
        )
    ):
        raise BootstrapActivationError("authorization preflight changed")
    return payload


def recover_claim_after_absence(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    predecessor_claim_path: str | Path,
    first_snapshot_path: str | Path,
    first_queue_stdout_path: str | Path,
    first_history_stdout_path: str | Path,
    later_snapshot_path: str | Path,
    later_queue_stdout_path: str | Path,
    later_history_stdout_path: str | Path,
) -> dict[str, Any]:
    """Resolve an unconsumed claim through two post-claim absence captures."""

    root = Path(repository_root).resolve(strict=True)
    intent = load_submission_intent(intent_path, repository_root=root)
    with _IntentTransitionLock(root, intent):
        return _recover_claim_after_absence_locked(
            root=root,
            intent=intent,
            predecessor_claim_path=predecessor_claim_path,
            first_snapshot_path=first_snapshot_path,
            first_queue_stdout_path=first_queue_stdout_path,
            first_history_stdout_path=first_history_stdout_path,
            later_snapshot_path=later_snapshot_path,
            later_queue_stdout_path=later_queue_stdout_path,
            later_history_stdout_path=later_history_stdout_path,
        )


def authorize_retry_after_rejected_submit(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    failed_result_path: str | Path,
    later_snapshot_path: str | Path,
    queue_stdout_path: str | Path,
    history_stdout_path: str | Path,
    prior_snapshot_path: str | Path,
    prior_queue_stdout_path: str | Path,
    prior_history_stdout_path: str | Path,
) -> dict[str, Any]:
    """Authorize one g2 retry after a definitive local submit rejection.

    A nonzero ``condor_submit`` result proves that generation one did not
    create a cluster.  A later digest-verified absence snapshot is still
    required so the retry remains compatible with intent-first
    reconciliation.  This transition never changes the scientific intent and
    admits at most one successor attempt.
    """

    root = Path(repository_root).resolve(strict=True)
    intent = load_submission_intent(intent_path, repository_root=root)
    with _IntentTransitionLock(root, intent):
        result_source = _repository_path(
            root,
            failed_result_path,
            label="failed provisional submit result",
            must_exist=True,
        )
        result = load_provisional_submit_result(
            result_source, repository_root=root
        )
        consumption_row = result.get("pre_submit_claim_consumption")
        if not isinstance(consumption_row, Mapping):
            raise BootstrapActivationError(
                "failed submit result has no claim consumption"
            )
        consumption_path = _verified_evidence_path(
            root, consumption_row, label="failed submit claim consumption"
        )
        consumption = load_pre_submit_claim_consumption(
            consumption_path, repository_root=root
        )
        claim_row = consumption.get("pre_submit_claim")
        if not isinstance(claim_row, Mapping):
            raise BootstrapActivationError(
                "failed submit consumption has no predecessor claim"
            )
        claim_path = _verified_evidence_path(
            root, claim_row, label="failed submit predecessor claim"
        )
        claim = load_pre_submit_execution_claim(claim_path, repository_root=root)
        generation = int(claim["claim_generation"])
        if (
            generation != 1
            or result.get("intent") != _intent_binding(intent)
            or result.get("status") != "failed"
            or result.get("cluster_id") is not None
            or result.get("acceptance_evidence") != "condor_submit_not_accepted"
            or int(result.get("condor_submit_returncode")) == 0
            or result.get("claim_generation") != generation
            or consumption.get("claim_generation") != generation
        ):
            raise BootstrapActivationError(
                "only one definitive generation-one submit rejection may retry"
            )
        receipt_path = _intent_evidence_path(root, intent, "submission_receipts")
        if receipt_path.exists():
            raise BootstrapActivationError(
                "canonical receipt blocks rejected-submit retry"
            )
        resolution_path = _claim_resolution_path(
            root,
            intent,
            claim_generation=generation,
            claim_sha256=claim["pre_submit_claim_sha256"],
        )
        if resolution_path.exists():
            resolution = load_pre_submit_claim_resolution(
                resolution_path, repository_root=root
            )
            if resolution.get("status") != "verified_absent_after_rejected_submit":
                raise BootstrapActivationError(
                    "predecessor claim already has another resolution"
                )
        snapshot = _load_snapshot(
            root,
            later_snapshot_path,
            queue_stdout_path=queue_stdout_path,
            history_stdout_path=history_stdout_path,
            prior_snapshot_path=prior_snapshot_path,
            prior_queue_stdout_path=prior_queue_stdout_path,
            prior_history_stdout_path=prior_history_stdout_path,
        )
        outcome = _snapshot_outcome(intent, snapshot.payload)
        proof = outcome.get("absence_proof")
        submitted_time = _aware_time(
            result.get("submitted_at"), label="failed submit submitted_at"
        )
        captured_time = _aware_time(
            snapshot.payload.get("captured_at"), label="retry snapshot captured_at"
        )
        if (
            outcome.get("status") != "verified_absent"
            or outcome.get("match_count") != 0
            or outcome.get("matches") != []
            or not isinstance(proof, Mapping)
            or proof.get("schema") != ABSENCE_PROOF_V1
            or not submitted_time < captured_time
        ):
            raise BootstrapActivationError(
                "rejected submit lacks later verified-absence evidence"
            )
        claim_binding = _logical_evidence_binding(
            root, claim_path, logical_sha256=claim["pre_submit_claim_sha256"]
        )
        result_binding = _logical_evidence_binding(
            root,
            result_source,
            logical_sha256=result["submission_result_sha256"],
        )
        resolution_core = {
            "schema": PRE_SUBMIT_CLAIM_RESOLUTION_V1,
            "status": "verified_absent_after_rejected_submit",
            "lane_id": intent.lane_id,
            "unit_id": intent.unit.unit_id,
            "action_id": intent.request.action_id,
            "execution_request_sha256": intent.request.digest,
            "intent": _intent_binding(intent),
            "predecessor_claim": claim_binding,
            "predecessor_claim_generation": generation,
            "successor_claim_generation": generation + 1,
            "resolution_path": _relative(
                root, resolution_path, label="claim resolution"
            ),
            "failed_submission_result": result_binding,
            "predecessor_claim_consumption": dict(consumption_row),
            "later_verified_absent_snapshot": snapshot.binding,
            "absence_proof": dict(proof),
            "submit_rejection_was_definitive": True,
            "no_scheduler_acceptance_observed": True,
            "maximum_successor_submit_attempts": 1,
            "automatic_retry_authorized": False,
            "resolved_at": snapshot.payload["captured_at"],
        }
        _atomic_write_immutable(
            resolution_path,
            _sealed(resolution_core, "pre_submit_claim_resolution_sha256"),
        )
        resolution = load_pre_submit_claim_resolution(
            resolution_path, repository_root=root
        )
        resolution_binding = _logical_evidence_binding(
            root,
            resolution_path,
            logical_sha256=resolution["pre_submit_claim_resolution_sha256"],
        )
        return _publish_claim_authorization(
            root,
            intent=intent,
            snapshot_evidence=snapshot,
            absence_proof=proof,
            upload_binding=claim["upload_verification"],
            dry_binding=claim["dry_run_result"],
            generation=generation + 1,
            predecessor_resolution=resolution_binding,
        )


def _recover_claim_after_absence_locked(
    *,
    root: Path,
    intent: BootstrapSubmissionIntent,
    predecessor_claim_path: str | Path,
    first_snapshot_path: str | Path,
    first_queue_stdout_path: str | Path,
    first_history_stdout_path: str | Path,
    later_snapshot_path: str | Path,
    later_queue_stdout_path: str | Path,
    later_history_stdout_path: str | Path,
) -> dict[str, Any]:
    claim_path = _repository_path(
        root,
        predecessor_claim_path,
        label="predecessor pre-submit claim",
        must_exist=True,
    )
    claim = load_pre_submit_execution_claim(claim_path, repository_root=root)
    if claim.get("intent") != _intent_binding(intent):
        raise BootstrapActivationError("predecessor claim crosses intent identity")
    if Path(first_snapshot_path) == Path(later_snapshot_path):
        raise BootstrapActivationError(
            "two post-claim snapshots are required for claim recovery"
        )
    result_path = _claim_scoped_state_path(
        root, intent, category="submission_results", claim=claim
    )
    receipt_path = _intent_evidence_path(root, intent, "submission_receipts")
    consumption_path = _claim_scoped_state_path(
        root,
        intent,
        category="pre_submit_claim_consumptions",
        claim=claim,
    )
    if result_path.exists() or receipt_path.exists() or consumption_path.exists():
        raise BootstrapActivationError(
            "submitted or recovered claim cannot be absence-resolved"
        )
    first = _load_snapshot(
        root,
        first_snapshot_path,
        queue_stdout_path=first_queue_stdout_path,
        history_stdout_path=first_history_stdout_path,
    )
    later = _load_snapshot(
        root,
        later_snapshot_path,
        queue_stdout_path=later_queue_stdout_path,
        history_stdout_path=later_history_stdout_path,
        prior_snapshot_path=first_snapshot_path,
        prior_queue_stdout_path=first_queue_stdout_path,
        prior_history_stdout_path=first_history_stdout_path,
    )
    claim_time = _aware_time(claim.get("claimed_at"), label="claim claimed_at")
    first_time = _aware_time(
        first.payload.get("captured_at"), label="first post-claim captured_at"
    )
    later_time = _aware_time(
        later.payload.get("captured_at"), label="later post-claim captured_at"
    )
    if not (claim_time < first_time < later_time):
        raise BootstrapActivationError(
            "both reconciliation snapshots must be strictly after the prior claim"
        )
    first_outcome = _snapshot_outcome(intent, first.payload)
    later_outcome = _snapshot_outcome(intent, later.payload)
    proof = later_outcome.get("absence_proof")
    if (
        first.binding.get("prior_snapshot") is not None
        or first_outcome.get("status") != "indeterminate"
        or first_outcome.get("match_count") != 0
        or first_outcome.get("matches") != []
        or first_outcome.get("indeterminate_reason")
        != "first_snapshot_for_exact_intent_identity"
        or later_outcome.get("status") != "verified_absent"
        or later_outcome.get("match_count") != 0
        or later_outcome.get("matches") != []
        or not isinstance(proof, Mapping)
        or proof.get("schema") != ABSENCE_PROOF_V1
        or proof.get("prior_snapshot_sha256")
        != first.payload.get("snapshot_sha256")
        or proof.get("prior_snapshot_status") != "indeterminate"
    ):
        raise BootstrapActivationError(
            "two post-claim same-intent absence observations are incomplete"
        )
    generation = int(claim["claim_generation"])
    resolution_path = _claim_resolution_path(
        root,
        intent,
        claim_generation=generation,
        claim_sha256=claim["pre_submit_claim_sha256"],
    )
    claim_binding = _logical_evidence_binding(
        root,
        claim_path,
        logical_sha256=claim["pre_submit_claim_sha256"],
    )
    resolution_core = {
        "schema": PRE_SUBMIT_CLAIM_RESOLUTION_V1,
        "status": "verified_absent_after_unresolved_claim",
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_id": intent.request.action_id,
        "execution_request_sha256": intent.request.digest,
        "intent": _intent_binding(intent),
        "predecessor_claim": claim_binding,
        "predecessor_claim_generation": generation,
        "successor_claim_generation": generation + 1,
        "resolution_path": _relative(
            root, resolution_path, label="claim resolution"
        ),
        "first_post_claim_snapshot": first.binding,
        "later_verified_absent_snapshot": later.binding,
        "absence_proof": dict(proof),
        "no_submit_result_observed": True,
        "no_submission_receipt_observed": True,
        "automatic_retry_authorized": False,
        "resolved_at": later.payload["captured_at"],
    }
    resolution_payload = _sealed(
        resolution_core, "pre_submit_claim_resolution_sha256"
    )
    _atomic_write_immutable(resolution_path, resolution_payload)
    resolution = load_pre_submit_claim_resolution(
        resolution_path, repository_root=root
    )
    resolution_binding = _logical_evidence_binding(
        root,
        resolution_path,
        logical_sha256=resolution["pre_submit_claim_resolution_sha256"],
    )
    return _publish_claim_authorization(
        root,
        intent=intent,
        snapshot_evidence=later,
        absence_proof=proof,
        upload_binding=claim["upload_verification"],
        dry_binding=claim["dry_run_result"],
        generation=generation + 1,
        predecessor_resolution=resolution_binding,
    )


def record_provisional_submit_result(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    authorization_path: str | Path,
    returncode: int,
    stdout: str,
    stderr: str = "",
    submitted_at: str | None = None,
) -> dict[str, Any]:
    """Record exactly one external ``condor_submit`` result without querying."""

    root = Path(repository_root).resolve(strict=True)
    intent = load_submission_intent(intent_path, repository_root=root)
    with _IntentTransitionLock(root, intent):
        return _record_provisional_submit_result_locked(
            root=root,
            intent=intent,
            authorization_path=authorization_path,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            submitted_at=submitted_at,
        )


def _record_provisional_submit_result_locked(
    *,
    root: Path,
    intent: BootstrapSubmissionIntent,
    authorization_path: str | Path,
    returncode: int,
    stdout: str,
    stderr: str,
    submitted_at: str | None,
) -> dict[str, Any]:
    authorization_source = _repository_path(
        root,
        authorization_path,
        label="submission authorization",
        must_exist=True,
    )
    authorization = load_submission_authorization(
        authorization_source, repository_root=root
    )
    if authorization.get("intent") != _intent_binding(intent):
        raise BootstrapActivationError("submission authorization crosses intent")
    claim_row = authorization.get("pre_submit_claim")
    if not isinstance(claim_row, Mapping):
        raise BootstrapActivationError("submission authorization has no claim")
    claim_path = _verified_evidence_path(
        root, claim_row, label="authorized pre-submit claim"
    )
    claim = load_pre_submit_execution_claim(claim_path, repository_root=root)
    resolution_path = _claim_resolution_path(
        root,
        intent,
        claim_generation=int(claim["claim_generation"]),
        claim_sha256=claim["pre_submit_claim_sha256"],
    )
    if resolution_path.exists():
        load_pre_submit_claim_resolution(resolution_path, repository_root=root)
        raise BootstrapActivationError(
            "resolved predecessor claim is not active"
        )
    output = _claim_scoped_state_path(
        root, intent, category="submission_results", claim=claim
    )
    if output.exists():
        existing = load_provisional_submit_result(output, repository_root=root)
        if (
            existing.get("submission_authorization", {}).get("path")
            != _relative(root, authorization_source, label="submission authorization")
            or existing.get("condor_submit_returncode") != int(returncode)
            or existing.get("condor_submit_stdout") != str(stdout)
            or existing.get("condor_submit_stderr") != str(stderr)
            or (
                submitted_at is not None
                and existing.get("submitted_at") != _timestamp(submitted_at)
            )
        ):
            raise BootstrapActivationError(
                "existing claim was consumed by a different submit result"
            )
        return existing
    receipt_path = _intent_evidence_path(root, intent, "submission_receipts")
    if receipt_path.exists():
        raise BootstrapActivationError(
            "canonical receipt already resolved the pre-submit claim"
        )
    code = int(returncode)
    text = str(stdout)
    error_text = str(stderr)
    status, cluster_id, acceptance_evidence = _classify_condor_submit_result(
        code, text
    )
    consumption_path = _claim_scoped_state_path(
        root,
        intent,
        category="pre_submit_claim_consumptions",
        claim=claim,
    )
    existing_consumption = (
        load_pre_submit_claim_consumption(consumption_path, repository_root=root)
        if consumption_path.exists()
        else None
    )
    submitted_time = (
        str(existing_consumption["submitted_at"])
        if existing_consumption is not None and submitted_at is None
        else _timestamp(submitted_at)
    )
    authorization_time = _aware_time(
        authorization["recorded_at"], label="authorization recorded_at"
    )
    if _aware_time(submitted_time, label="submitted_at") <= authorization_time:
        raise BootstrapActivationError("submit result does not follow authorization")
    attempt_core = {
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_id": intent.request.action_id,
        "execution_request_sha256": intent.request.digest,
        "claim_generation": claim["claim_generation"],
        "pre_submit_claim_sha256": claim["pre_submit_claim_sha256"],
        "condor_submit_returncode": code,
        "condor_submit_stdout": text,
        "condor_submit_stderr": error_text,
        "cluster_id": cluster_id,
        "submitted_at": submitted_time,
    }
    attempt_sha256 = _digest(attempt_core)
    claim_binding = _logical_evidence_binding(
        root, claim_path, logical_sha256=claim["pre_submit_claim_sha256"]
    )
    consumption_core = {
        "schema": PRE_SUBMIT_CLAIM_CONSUMPTION_V1,
        "status": "consumed_by_recorded_submit_result",
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_id": intent.request.action_id,
        "execution_request_sha256": intent.request.digest,
        "claim_generation": claim["claim_generation"],
        "pre_submit_claim": claim_binding,
        "submission_attempt_sha256": attempt_sha256,
        "condor_submit_returncode": code,
        "cluster_id": cluster_id,
        "stdout_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(error_text.encode("utf-8")).hexdigest(),
        "submitted_at": submitted_time,
        "claim_consumed_before_result_publication": True,
        "automatic_retry_authorized": False,
    }
    consumption_payload = _sealed(
        consumption_core, "pre_submit_claim_consumption_sha256"
    )
    if (
        existing_consumption is not None
        and existing_consumption.get("submission_attempt_sha256")
        != attempt_sha256
    ):
        raise BootstrapActivationError(
            "existing claim was consumed by a different submit result"
        )
    _atomic_write_immutable(consumption_path, consumption_payload)
    consumption = load_pre_submit_claim_consumption(
        consumption_path, repository_root=root
    )
    if consumption.get("submission_attempt_sha256") != attempt_sha256:
        raise BootstrapActivationError(
            "existing claim was consumed by a different submit result"
        )
    consumption_binding = _logical_evidence_binding(
        root,
        consumption_path,
        logical_sha256=consumption["pre_submit_claim_consumption_sha256"],
    )
    core = {
        "schema": PROVISIONAL_SUBMISSION_RESULT_V1,
        "status": status,
        "lane_id": intent.lane_id,
        "unit_id": intent.unit.unit_id,
        "action_ids": [intent.request.action_id],
        "cluster_id": cluster_id,
        "expected_process_ids": [0],
        "job_batch_name": intent.package_manifest.job_batch_name,
        "intent": _intent_binding(intent),
        "intent_sha256": intent.intent_sha256,
        "execution_request_sha256": intent.request.digest,
        "package": intent.unit.payload["package"],
        "submit_descriptor": intent.unit.payload["submit_descriptor_binding"],
        "submission_authorization": {
            "path": _relative(
                root, authorization_source, label="submission authorization"
            ),
            "sha256": authorization["submission_authorization_sha256"],
            "file_sha256": _file_sha(authorization_source),
        },
        "pre_submit_claim_consumption": consumption_binding,
        "claim_generation": claim["claim_generation"],
        "submission_attempt_sha256": attempt_sha256,
        "condor_submit_returncode": code,
        "condor_submit_stdout": text,
        "condor_submit_stderr": error_text,
        "single_condor_submit_invocation": True,
        "scheduler_acceptance_query_performed": False,
        "acceptance_evidence": acceptance_evidence,
        "automatic_retry_authorized": False,
        "remote_raw_evidence_policy": "preserve",
        "submitted_at": submitted_time,
    }
    _atomic_write_immutable(output, _sealed(core, "submission_result_sha256"))
    return load_provisional_submit_result(output, repository_root=root)


def load_provisional_submit_result(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(root, path, label="provisional submit result", must_exist=True)
    payload = _read_object(source, label="provisional submit result")
    _verify_seal(
        payload, key="submission_result_sha256", label="provisional submit result"
    )
    intent_row = payload.get("intent")
    authorization_row = payload.get("submission_authorization")
    consumption_row = payload.get("pre_submit_claim_consumption")
    if (
        not isinstance(intent_row, Mapping)
        or not isinstance(authorization_row, Mapping)
        or not isinstance(consumption_row, Mapping)
    ):
        raise BootstrapActivationError("provisional submit bindings are incomplete")
    intent = load_submission_intent(
        str(intent_row.get("path") or ""), repository_root=root
    )
    authorization_path = _verified_evidence_path(
        root, authorization_row, label="submission authorization"
    )
    authorization = load_submission_authorization(
        authorization_path, repository_root=root
    )
    consumption_path = _verified_evidence_path(
        root, consumption_row, label="pre-submit claim consumption"
    )
    consumption = load_pre_submit_claim_consumption(
        consumption_path, repository_root=root
    )
    consumed_claim_row = consumption.get("pre_submit_claim")
    if not isinstance(consumed_claim_row, Mapping):
        raise BootstrapActivationError(
            "provisional result consumption has no claim binding"
        )
    consumed_claim_path = _verified_evidence_path(
        root, consumed_claim_row, label="provisional result consumed claim"
    )
    consumed_claim = load_pre_submit_execution_claim(
        consumed_claim_path, repository_root=root
    )
    expected = _claim_scoped_state_path(
        root,
        intent,
        category="submission_results",
        claim=consumed_claim,
    )
    code = int(payload.get("condor_submit_returncode"))
    stdout = str(payload.get("condor_submit_stdout") or "")
    expected_status, expected_cluster, expected_acceptance = (
        _classify_condor_submit_result(code, stdout)
    )
    if (
        source != expected.resolve(strict=True)
        or payload.get("schema") != PROVISIONAL_SUBMISSION_RESULT_V1
        or payload.get("status") != expected_status
        or dict(intent_row) != _intent_binding(intent)
        or payload.get("lane_id") != intent.lane_id
        or payload.get("unit_id") != intent.unit.unit_id
        or payload.get("action_ids") != [intent.request.action_id]
        or payload.get("cluster_id") != expected_cluster
        or payload.get("expected_process_ids") != [0]
        or payload.get("job_batch_name") != intent.package_manifest.job_batch_name
        or payload.get("intent_sha256") != intent.intent_sha256
        or payload.get("execution_request_sha256") != intent.request.digest
        or payload.get("claim_generation") != authorization.get("claim_generation")
        or payload.get("claim_generation") != consumption.get("claim_generation")
        or payload.get("package") != intent.unit.payload["package"]
        or payload.get("submit_descriptor")
        != intent.unit.payload["submit_descriptor_binding"]
        or authorization_row.get("sha256")
        != authorization["submission_authorization_sha256"]
        or consumption_row.get("sha256")
        != consumption["pre_submit_claim_consumption_sha256"]
        or payload.get("submission_attempt_sha256")
        != consumption.get("submission_attempt_sha256")
        or consumption.get("condor_submit_returncode") != code
        or consumption.get("cluster_id") != expected_cluster
        or consumption.get("stdout_sha256")
        != hashlib.sha256(stdout.encode("utf-8")).hexdigest()
        or consumption.get("stderr_sha256")
        != hashlib.sha256(
            str(payload.get("condor_submit_stderr") or "").encode("utf-8")
        ).hexdigest()
        or payload.get("single_condor_submit_invocation") is not True
        or payload.get("scheduler_acceptance_query_performed") is not False
        or payload.get("acceptance_evidence") != expected_acceptance
        or payload.get("automatic_retry_authorized") is not False
        or payload.get("remote_raw_evidence_policy") != "preserve"
    ):
        raise BootstrapActivationError("provisional submit identity diverged")
    submitted_time = _aware_time(payload.get("submitted_at"), label="submitted_at")
    if submitted_time <= _aware_time(
        authorization["recorded_at"], label="authorization recorded_at"
    ):
        raise BootstrapActivationError("submit result predates authorization")
    if str(consumption.get("submitted_at")) != submitted_time.isoformat():
        raise BootstrapActivationError("claim consumption timestamp diverged")
    return payload


@_frozen_submitted_evidence_only
def recover_submission_receipt(
    *,
    repository_root: str | Path,
    intent_path: str | Path,
    provisional_result_path: str | Path,
    later_snapshot_path: str | Path,
    queue_stdout_path: str | Path,
    history_stdout_path: str | Path,
    prior_snapshot_path: str | Path | None = None,
    prior_queue_stdout_path: str | Path | None = None,
    prior_history_stdout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recover a canonical supervisor receipt from one later found identity."""

    root = Path(repository_root).resolve(strict=True)
    intent = load_frozen_submitted_intent(intent_path, repository_root=root)
    result = load_provisional_submit_result(
        provisional_result_path, repository_root=root
    )
    if result.get("status") != "submitted":
        raise BootstrapActivationError("failed submit result cannot be recovered")
    snapshot_evidence = _load_snapshot(
        root,
        later_snapshot_path,
        queue_stdout_path=queue_stdout_path,
        history_stdout_path=history_stdout_path,
        prior_snapshot_path=prior_snapshot_path,
        prior_queue_stdout_path=prior_queue_stdout_path,
        prior_history_stdout_path=prior_history_stdout_path,
    )
    snapshot = snapshot_evidence.payload
    if _aware_time(snapshot["captured_at"], label="snapshot captured_at") <= _aware_time(
        result["submitted_at"], label="submitted_at"
    ):
        raise BootstrapActivationError("recovery snapshot is not later than submission")
    outcome = _snapshot_outcome(intent, snapshot)
    match = _exact_found_match(intent, outcome)
    if int(match["cluster_id"]) != int(result["cluster_id"]):
        raise BootstrapActivationError("later found identity has another cluster")
    return _write_submission_receipt(
        root,
        intent=intent,
        snapshot_evidence=snapshot_evidence,
        match=match,
    )


@_frozen_submitted_evidence_only
def load_submission_receipt(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(root, path, label="submission receipt", must_exist=True)
    payload = _read_object(source, label="submission receipt")
    remote = payload.get("remote")
    evidence = remote.get("recovery_evidence") if isinstance(remote, Mapping) else None
    intent_row = evidence.get("intent") if isinstance(evidence, Mapping) else None
    snapshot_row = evidence.get("snapshot") if isinstance(evidence, Mapping) else None
    if not isinstance(intent_row, Mapping) or not isinstance(snapshot_row, Mapping):
        raise BootstrapActivationError("submission receipt evidence is incomplete")
    intent = load_frozen_submitted_intent(
        str(intent_row.get("path") or ""), repository_root=root
    )
    snapshot_evidence = _load_snapshot_binding(
        root, snapshot_row, label="recovery snapshot"
    )
    snapshot = snapshot_evidence.payload
    match = _exact_found_match(intent, _snapshot_outcome(intent, snapshot))
    expected = _intent_evidence_path(root, intent, "submission_receipts")
    if (
        source != expected.resolve(strict=True)
        or payload.get("schema") != SUBMISSION_RECEIPT_SCHEMA_V1
        or payload.get("lane_id") != intent.lane_id
        or payload.get("unit_id") != intent.unit.unit_id
        or payload.get("action_ids") != [intent.request.action_id]
        or payload.get("intent_sha256") != intent.intent_sha256
        or payload.get("recovered_from_scheduler") is not True
        or not isinstance(remote, Mapping)
        or remote.get("cluster_id") != int(match["cluster_id"])
        or remote.get("process_ids") != [0]
        or remote.get("unit_id") != intent.unit.unit_id
        or remote.get("job_batch_name") != intent.package_manifest.job_batch_name
        or remote.get("action_ids") != [intent.request.action_id]
        or remote.get("scheduler_identity") != f"{int(match['cluster_id'])}.0"
        or remote.get("source_epoch_identity_sha256")
        != intent.source_epoch.source_epoch.identity_sha256
        or remote.get("source_epoch_id")
        != intent.source_epoch.source_epoch.source_epoch_id
        or remote.get("remote_raw_evidence_policy") != "preserve"
        or dict(intent_row) != _intent_binding(intent)
        or dict(snapshot_row) != snapshot_evidence.binding
        or evidence.get("scheduler_match_sha256") != _digest(dict(match))
        or evidence.get("scheduler_acceptance_query_performed_by_coordinator")
        is not False
        or payload.get("recorded_at") != snapshot.get("captured_at")
    ):
        raise BootstrapActivationError("submission receipt identity diverged")
    return payload


def validate_artifact(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Validate one coordinator artifact by its typed schema."""

    root = Path(repository_root).resolve(strict=True)
    source = _repository_path(root, path, label="activation artifact", must_exist=True)
    schema = _read_object(source, label="activation artifact").get("schema")
    if schema == ACTIVATION_MANIFEST_V1:
        payload = load_activation_manifest(source, repository_root=root)
    elif schema == SUBMISSION_INTENT_SCHEMA_V1:
        intent = load_submission_intent(source, repository_root=root)
        return {
            "schema": schema,
            "status": "validated",
            "path": intent.path,
            "lane_id": intent.lane_id,
            "unit_id": intent.unit.unit_id,
            "action_id": intent.request.action_id,
        }
    elif schema == UPLOAD_VERIFICATION_V1:
        payload = load_upload_verification(source, repository_root=root)
    elif schema == DRY_RUN_RESULT_V1:
        payload = load_dry_run_result(source, repository_root=root)
    elif schema == SUBMISSION_AUTHORIZATION_V1:
        payload = load_submission_authorization(source, repository_root=root)
    elif schema == PRE_SUBMIT_EXECUTION_CLAIM_V1:
        payload = load_pre_submit_execution_claim(source, repository_root=root)
    elif schema == PRE_SUBMIT_CLAIM_CONSUMPTION_V1:
        payload = load_pre_submit_claim_consumption(source, repository_root=root)
    elif schema == PRE_SUBMIT_CLAIM_RESOLUTION_V1:
        payload = load_pre_submit_claim_resolution(source, repository_root=root)
    elif schema == PROVISIONAL_SUBMISSION_RESULT_V1:
        payload = load_provisional_submit_result(source, repository_root=root)
    elif schema == EXACT_REFERENCE_TERMINAL_FAILURE_V1:
        payload = load_exact_reference_terminal_failure_evidence(
            source, repository_root=root
        )
    elif schema == AVQDS_SCOUT_TERMINAL_FAILURE_V1:
        payload = load_avqds_scout_terminal_failure_evidence(
            source, repository_root=root
        )
    elif schema == SUBMISSION_RECEIPT_SCHEMA_V1:
        payload = load_submission_receipt(source, repository_root=root)
    else:
        raise BootstrapActivationError("unsupported activation artifact schema")
    return {
        "schema": schema,
        "status": "validated",
        "path": _relative(root, source, label="activation artifact"),
        "identity_sha256": _file_sha(source),
        "artifact_status": payload.get("status"),
    }


__all__ = [
    "ACTIVATION_MANIFEST_V1",
    "AVQDS_SCOUT_TERMINAL_FAILURE_V1",
    "BootstrapActivationError",
    "BootstrapSubmissionIntent",
    "DRY_RUN_RESULT_V1",
    "EXACT_REFERENCE_TERMINAL_FAILURE_V1",
    "PRE_SUBMIT_CLAIM_CONSUMPTION_V1",
    "PRE_SUBMIT_CLAIM_RESOLUTION_V1",
    "PRE_SUBMIT_EXECUTION_CLAIM_V1",
    "PROVISIONAL_SUBMISSION_RESULT_V1",
    "SUBMISSION_AUTHORIZATION_V1",
    "UPLOAD_VERIFICATION_V1",
    "authorize_retry_after_rejected_submit",
    "authorize_from_snapshot",
    "load_activation_manifest",
    "load_avqds_scout_terminal_failure_evidence",
    "load_dry_run_result",
    "load_exact_reference_terminal_failure_evidence",
    "load_frozen_submitted_intent",
    "load_provisional_submit_result",
    "load_pre_submit_claim_consumption",
    "load_pre_submit_claim_resolution",
    "load_pre_submit_execution_claim",
    "load_submission_authorization",
    "load_submission_intent",
    "load_submission_receipt",
    "load_upload_verification",
    "prepare_bootstrap_activation",
    "prepare_avqds_scout_terminal_failure_successor",
    "prepare_exact_reference_terminal_failure_successor",
    "record_exact_reference_terminal_failure_evidence",
    "record_dry_run_result",
    "record_provisional_submit_result",
    "record_upload_verification",
    "recover_submission_receipt",
    "recover_claim_after_absence",
    "snapshot_intent_binding",
    "validate_artifact",
]
