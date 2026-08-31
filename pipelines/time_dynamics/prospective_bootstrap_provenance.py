"""Local provenance gates for prospective Paper-II bootstrap packages.

This module does not activate a lane or contact a scheduler.  It freezes an
exact committed Git tree into the source-epoch contracts already consumed by
``SourceEpochReceipt.load``.  Bootstrap package and pre-submit contracts are
added below as a separate boundary from the post-bootstrap scientific audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.acquisition_supervisor import CampaignRegistry
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HH,
    LANE_HUBBARD,
    SourceEpochIdentity,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    SourceEpochReceipt,
    VerifiedFile,
)


_FULL_GIT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_SAFE_LANE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_SAFE_BATCH_NAME = re.compile(r"[A-Za-z0-9_.:-]+")
_BOOTSTRAP_KINDS = frozenset(
    {"build_seed", "build_exact_reference", "screen_avqds"}
)
_BOOTSTRAP_LANES = frozenset({LANE_HH, LANE_HUBBARD})

BOOTSTRAP_PACKAGE_MANIFEST_V1 = "paper_ii_bootstrap_package_manifest_v1"
BOOTSTRAP_PRE_SUBMIT_AUDIT_V1 = "paper_ii_bootstrap_pre_submit_audit_v1"
BOOTSTRAP_INTENT_FIRST_REQUIREMENTS = MappingProxyType(
    {
        "submission_intent_required_before_scheduler_contact": True,
        "first_snapshot_outcome": "indeterminate",
        "later_submission_requires_reconciliation": "verified_absent",
        "found_recovers_without_resubmission": True,
        "indeterminate_blocks_submission": True,
    }
)


class BootstrapProvenanceError(ValueError):
    """A bootstrap provenance artifact is missing, stale, or ambiguous."""


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


def _repository_path(root: Path, value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    target = candidate if candidate.is_absolute() else root / candidate
    target = target.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise BootstrapProvenanceError(
            f"{label} is outside repository root: {value}"
        ) from exc
    return target


def _relative(root: Path, path: Path, *, label: str) -> str:
    try:
        relative = path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise BootstrapProvenanceError(
            f"{label} is outside repository root: {path}"
        ) from exc
    if not relative.parts or ".." in relative.parts:
        raise BootstrapProvenanceError(f"{label} is not repository-relative")
    return relative.as_posix()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise BootstrapProvenanceError(
                f"provenance identity collision: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BootstrapProvenanceError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapProvenanceError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise BootstrapProvenanceError(f"{label} must be a JSON object")
    return value


def _self_digest(payload: Mapping[str, Any], *, key: str, label: str) -> str:
    unsigned = dict(payload)
    declared = unsigned.pop(key, None)
    actual = _canonical_sha(unsigned)
    if declared != actual:
        raise BootstrapProvenanceError(f"{label} self digest mismatch")
    return actual


def _verified_file(value: Any, *, label: str) -> VerifiedFile:
    if not isinstance(value, Mapping):
        raise BootstrapProvenanceError(f"{label} binding is missing")
    try:
        return VerifiedFile(
            path=str(value.get("path") or ""),
            sha256=str(value.get("sha256") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise BootstrapProvenanceError(f"{label} binding is invalid") from exc


def _reload_source_epoch(
    source: SourceEpochReceipt, *, root: Path
) -> SourceEpochReceipt:
    current = SourceEpochReceipt.load(
        root / source.receipt_path,
        repository_root=root,
        expected_epoch=source.source_epoch,
    )
    if current != source:
        raise BootstrapProvenanceError("source epoch changed after validation")
    return current


def _bootstrap_request(value: ExecutionRequest) -> ExecutionRequest:
    try:
        current = ExecutionRequest.from_mapping(value.to_json_dict())
    except (TypeError, ValueError) as exc:
        raise BootstrapProvenanceError(
            "bootstrap execution request is not canonical"
        ) from exc
    if current != value or value.kind not in _BOOTSTRAP_KINDS:
        raise BootstrapProvenanceError(
            "package request is not a supported bootstrap action"
        )
    if value.inputs.get("source_epoch") is None:
        raise BootstrapProvenanceError("bootstrap request has no source epoch")
    if value.to_json_dict().get("qiskit_compilation_requested") is not False:
        raise BootstrapProvenanceError("bootstrap request enabled Qiskit")
    return current


def _archive_inventory(path: Path) -> tuple[tuple[str, ...], int]:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise BootstrapProvenanceError(
            f"bootstrap package is not a readable gzip/tar archive: {path}"
        ) from exc
    names: list[str] = []
    for member in members:
        name = str(member.name)
        candidate = Path(name)
        if (
            not name
            or candidate.is_absolute()
            or ".." in candidate.parts
            or not (member.isfile() or member.isdir())
        ):
            raise BootstrapProvenanceError(
                f"bootstrap package contains an unsafe member: {name!r}"
            )
        names.append(candidate.as_posix())
    if not names or len(names) != len(set(names)):
        raise BootstrapProvenanceError(
            "bootstrap package is empty or repeats a member"
        )
    return tuple(names), len(members)


def _source_tree_entries(listing: bytes) -> tuple[dict[str, str], ...]:
    """Decode ``git ls-tree -z`` without consulting the working tree."""

    entries: list[dict[str, str]] = []
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        try:
            header, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id = header.decode("ascii").split(" ", 2)
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise BootstrapProvenanceError(
                "source tree contains an unsupported Git entry"
            ) from exc
        candidate = Path(path)
        if (
            not path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != path
            or kind not in {"blob", "commit"}
            or not re.fullmatch(r"[0-7]{6}", mode)
            or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", object_id)
        ):
            raise BootstrapProvenanceError(
                f"source tree contains an unsafe Git entry: {path!r}"
            )
        entries.append(
            {
                "path": path,
                "mode": mode,
                "type": kind,
                "git_object": object_id,
            }
        )
    if not entries or len({row["path"] for row in entries}) != len(entries):
        raise BootstrapProvenanceError("source tree is empty or repeats a path")
    return tuple(entries)


@dataclass(frozen=True)
class BootstrapPackageManifest:
    """Exact local package bytes for one source-bound bootstrap request."""

    request: ExecutionRequest
    source_epoch_receipt: VerifiedFile
    package: VerifiedFile
    package_bytes: int
    member_count: int
    ordered_members: tuple[str, ...]
    submit_descriptor: str
    job_batch_name: str
    manifest_path: str
    manifest_sha256: str
    manifest_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "BootstrapPackageManifest":
        root = Path(repository_root).resolve(strict=True)
        source = _repository_path(root, path, label="bootstrap package manifest")
        payload = _read_object(source, label="bootstrap package manifest")
        self_sha = _self_digest(
            payload,
            key="bootstrap_package_manifest_sha256",
            label="bootstrap package manifest",
        )
        if (
            payload.get("schema") != BOOTSTRAP_PACKAGE_MANIFEST_V1
            or payload.get("status") != "prepared_local_only"
            or payload.get("qiskit_compilation") is not False
            or payload.get("production_submission_enabled") is not False
        ):
            raise BootstrapProvenanceError(
                "bootstrap package manifest is not local-only"
            )
        request_row = payload.get("execution_request")
        if not isinstance(request_row, Mapping):
            raise BootstrapProvenanceError(
                "bootstrap package manifest has no execution request"
            )
        try:
            request = _bootstrap_request(ExecutionRequest.from_mapping(request_row))
        except (TypeError, ValueError) as exc:
            raise BootstrapProvenanceError(
                "bootstrap package request is invalid"
            ) from exc
        source_file = _verified_file(
            payload.get("source_epoch_receipt"), label="source epoch receipt"
        )
        package_file = _verified_file(payload.get("package"), label="package")
        source_file.verify(root)
        package_file.verify(root)
        source_receipt = SourceEpochReceipt.load(
            root / source_file.path, repository_root=root
        )
        if (
            request.lane_id != source_receipt.lane_id
            or request.inputs.get("source_epoch")
            != source_receipt.source_epoch.to_json_dict()
            or payload.get("source_epoch_identity_sha256")
            != source_receipt.source_epoch.identity_sha256
        ):
            raise BootstrapProvenanceError(
                "bootstrap package crosses source epochs"
            )
        package_path = root / package_file.path
        members, member_count = _archive_inventory(package_path)
        try:
            package_bytes = int(payload.get("package_bytes"))
            declared_member_count = int(payload.get("member_count"))
        except (TypeError, ValueError) as exc:
            raise BootstrapProvenanceError(
                "bootstrap package size/member count is invalid"
            ) from exc
        declared_members = payload.get("ordered_members")
        submit_descriptor = str(payload.get("submit_descriptor") or "")
        job_batch_name = str(payload.get("job_batch_name") or "")
        if (
            payload.get("lane_id") != request.lane_id
            or payload.get("action_id") != request.action_id
            or payload.get("execution_id") != request.execution_id
            or payload.get("execution_request_sha256") != request.digest
            or package_bytes != package_path.stat().st_size
            or declared_member_count != member_count
            or declared_members != list(members)
            or payload.get("ordered_members_sha256") != _canonical_sha(list(members))
            or submit_descriptor not in members
            or not _SAFE_BATCH_NAME.fullmatch(job_batch_name)
        ):
            raise BootstrapProvenanceError(
                "bootstrap package manifest does not reproduce its request/bytes"
            )
        return cls(
            request=request,
            source_epoch_receipt=source_file,
            package=package_file,
            package_bytes=package_bytes,
            member_count=member_count,
            ordered_members=members,
            submit_descriptor=submit_descriptor,
            job_batch_name=job_batch_name,
            manifest_path=_relative(root, source, label="bootstrap package manifest"),
            manifest_sha256=_file_sha(source),
            manifest_self_sha256=self_sha,
        )


def write_bootstrap_package_manifest(
    path: str | Path,
    *,
    request: ExecutionRequest,
    source_epoch_receipt: SourceEpochReceipt,
    package_path: str | Path,
    submit_descriptor: str,
    job_batch_name: str,
    repository_root: str | Path,
) -> BootstrapPackageManifest:
    """Bind one archive to its exact source epoch and bootstrap request."""

    root = Path(repository_root).resolve(strict=True)
    current_request = _bootstrap_request(request)
    source = _reload_source_epoch(source_epoch_receipt, root=root)
    if (
        current_request.lane_id != source.lane_id
        or current_request.inputs.get("source_epoch")
        != source.source_epoch.to_json_dict()
    ):
        raise BootstrapProvenanceError("bootstrap request/source epoch mismatch")
    package = _repository_path(root, package_path, label="bootstrap package")
    if not package.is_file():
        raise BootstrapProvenanceError(f"bootstrap package is missing: {package}")
    members, member_count = _archive_inventory(package)
    descriptor = str(submit_descriptor).strip()
    batch_name = str(job_batch_name).strip()
    if descriptor not in members:
        raise BootstrapProvenanceError(
            "submit descriptor is not present in the bootstrap package"
        )
    if not _SAFE_BATCH_NAME.fullmatch(batch_name):
        raise BootstrapProvenanceError("job batch name is not a stable ClassAd value")
    target = _repository_path(root, path, label="bootstrap package manifest")
    payload = {
        "schema": BOOTSTRAP_PACKAGE_MANIFEST_V1,
        "status": "prepared_local_only",
        "lane_id": current_request.lane_id,
        "action_id": current_request.action_id,
        "execution_id": current_request.execution_id,
        "execution_request_sha256": current_request.digest,
        "execution_request": current_request.to_json_dict(),
        "source_epoch_receipt": {
            "path": source.receipt_path,
            "sha256": source.receipt_sha256,
        },
        "source_epoch_identity_sha256": source.source_epoch.identity_sha256,
        "package": {
            "path": _relative(root, package, label="bootstrap package"),
            "sha256": _file_sha(package),
        },
        "package_bytes": package.stat().st_size,
        "archive_format": "tar.gz",
        "member_count": member_count,
        "ordered_members": list(members),
        "ordered_members_sha256": _canonical_sha(list(members)),
        "submit_descriptor": descriptor,
        "job_batch_name": batch_name,
        "qiskit_compilation": False,
        "production_submission_enabled": False,
    }
    payload["bootstrap_package_manifest_sha256"] = _canonical_sha(payload)
    _atomic_write_json(target, payload)
    return BootstrapPackageManifest.load(target, repository_root=root)


@dataclass(frozen=True)
class BootstrapPreSubmitAudit:
    """Passed local package audit; explicitly not a scheduler authorization."""

    status: str
    lane_id: str
    action_id: str
    execution_id: str
    execution_request_sha256: str
    execution_request: ExecutionRequest
    source_epoch_receipt: VerifiedFile
    package_manifest: VerifiedFile
    registry: VerifiedFile
    registry_id: str
    registry_sha256: str
    lane_blocked_by: tuple[str, ...]
    intent_first: Mapping[str, Any]
    production_submission_enabled: bool
    scheduler_submission_authorized: bool
    warnings: tuple[str, ...]
    receipt_path: str
    receipt_sha256: str
    receipt_self_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "BootstrapPreSubmitAudit":
        root = Path(repository_root).resolve(strict=True)
        source = _repository_path(root, path, label="bootstrap pre-submit audit")
        payload = _read_object(source, label="bootstrap pre-submit audit")
        self_sha = _self_digest(
            payload,
            key="bootstrap_pre_submit_audit_sha256",
            label="bootstrap pre-submit audit",
        )
        request_row = payload.get("execution_request")
        warnings = payload.get("warnings")
        blockers = payload.get("lane_blocked_by")
        if (
            payload.get("schema") != BOOTSTRAP_PRE_SUBMIT_AUDIT_V1
            or payload.get("status") != "passed_local_only"
            or payload.get("errors") != []
            or payload.get("qiskit_compilation") is not False
            or payload.get("production_submission_enabled") is not False
            or payload.get("scheduler_submission_authorized") is not False
            or payload.get("intent_first")
            != dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS)
            or not isinstance(request_row, Mapping)
            or not isinstance(warnings, list)
            or any(not isinstance(value, str) for value in warnings)
            or not isinstance(blockers, list)
            or any(not isinstance(value, str) or not value for value in blockers)
        ):
            raise BootstrapProvenanceError(
                "bootstrap pre-submit audit is not a passed local-only contract"
            )
        try:
            request = _bootstrap_request(ExecutionRequest.from_mapping(request_row))
        except (TypeError, ValueError) as exc:
            raise BootstrapProvenanceError(
                "bootstrap audit request is invalid"
            ) from exc
        source_file = _verified_file(
            payload.get("source_epoch_receipt"), label="source epoch receipt"
        )
        package_file = _verified_file(
            payload.get("package_manifest"), label="package manifest"
        )
        registry_file = _verified_file(payload.get("registry"), label="registry")
        for file in (source_file, package_file, registry_file):
            file.verify(root)
        source_receipt = SourceEpochReceipt.load(
            root / source_file.path, repository_root=root
        )
        package = BootstrapPackageManifest.load(
            root / package_file.path, repository_root=root
        )
        registry = CampaignRegistry.read(
            root / registry_file.path, repository_root=root
        )
        try:
            lane = registry.lane(request.lane_id)
        except KeyError as exc:
            raise BootstrapProvenanceError(
                "bootstrap audit lane is absent from the registry"
            ) from exc
        if (
            package.request != request
            or package.source_epoch_receipt != source_file
            or source_receipt.lane_id != request.lane_id
            or payload.get("lane_id") != request.lane_id
            or payload.get("action_id") != request.action_id
            or payload.get("execution_id") != request.execution_id
            or payload.get("execution_request_sha256") != request.digest
            or payload.get("registry_id") != registry.registry_id
            or payload.get("registry_sha256") != registry.digest
            or tuple(blockers) != lane.blocked_by
            or registry.production_operations is not None
            or lane.submission_enabled
        ):
            raise BootstrapProvenanceError(
                "bootstrap audit no longer binds a disabled current registry"
            )
        return cls(
            status="passed_local_only",
            lane_id=request.lane_id,
            action_id=request.action_id,
            execution_id=request.execution_id,
            execution_request_sha256=request.digest,
            execution_request=request,
            source_epoch_receipt=source_file,
            package_manifest=package_file,
            registry=registry_file,
            registry_id=registry.registry_id,
            registry_sha256=registry.digest,
            lane_blocked_by=tuple(blockers),
            intent_first=BOOTSTRAP_INTENT_FIRST_REQUIREMENTS,
            production_submission_enabled=False,
            scheduler_submission_authorized=False,
            warnings=tuple(warnings),
            receipt_path=_relative(root, source, label="bootstrap pre-submit audit"),
            receipt_sha256=_file_sha(source),
            receipt_self_sha256=self_sha,
        )


def write_bootstrap_pre_submit_audit(
    path: str | Path,
    *,
    request: ExecutionRequest,
    source_epoch_receipt: SourceEpochReceipt,
    package_manifest: BootstrapPackageManifest,
    registry_path: str | Path,
    repository_root: str | Path,
    warnings: Sequence[str] = (),
) -> BootstrapPreSubmitAudit:
    """Audit bootstrap bytes while proving the control plane is still inert."""

    root = Path(repository_root).resolve(strict=True)
    current_request = _bootstrap_request(request)
    source = _reload_source_epoch(source_epoch_receipt, root=root)
    current_package = BootstrapPackageManifest.load(
        root / package_manifest.manifest_path, repository_root=root
    )
    if current_package != package_manifest:
        raise BootstrapProvenanceError("bootstrap package changed after validation")
    registry_source = _repository_path(root, registry_path, label="registry")
    registry = CampaignRegistry.read(registry_source, repository_root=root)
    try:
        lane = registry.lane(current_request.lane_id)
    except KeyError as exc:
        raise BootstrapProvenanceError(
            "bootstrap lane is absent from the current registry"
        ) from exc
    if registry.production_operations is not None or lane.submission_enabled:
        raise BootstrapProvenanceError(
            "bootstrap local-only audit requires production to remain disabled"
        )
    if (
        current_package.request != current_request
        or current_package.source_epoch_receipt
        != VerifiedFile(source.receipt_path, source.receipt_sha256)
    ):
        raise BootstrapProvenanceError(
            "bootstrap audit request/package/source binding mismatch"
        )
    if any(not isinstance(value, str) for value in warnings):
        raise BootstrapProvenanceError("bootstrap audit warning is invalid")
    warning_rows = tuple(sorted(set(warnings)))
    if any(not value or value != value.strip() for value in warning_rows):
        raise BootstrapProvenanceError("bootstrap audit warning is invalid")
    target = _repository_path(root, path, label="bootstrap pre-submit audit")
    payload = {
        "schema": BOOTSTRAP_PRE_SUBMIT_AUDIT_V1,
        "status": "passed_local_only",
        "lane_id": current_request.lane_id,
        "action_id": current_request.action_id,
        "execution_id": current_request.execution_id,
        "execution_request_sha256": current_request.digest,
        "execution_request": current_request.to_json_dict(),
        "source_epoch_receipt": {
            "path": source.receipt_path,
            "sha256": source.receipt_sha256,
        },
        "package_manifest": {
            "path": current_package.manifest_path,
            "sha256": current_package.manifest_sha256,
        },
        "registry": {
            "path": _relative(root, registry_source, label="registry"),
            "sha256": _file_sha(registry_source),
        },
        "registry_id": registry.registry_id,
        "registry_sha256": registry.digest,
        "lane_blocked_by": list(lane.blocked_by),
        "intent_first": dict(BOOTSTRAP_INTENT_FIRST_REQUIREMENTS),
        "qiskit_compilation": False,
        "production_submission_enabled": False,
        "scheduler_submission_authorized": False,
        "errors": [],
        "warnings": list(warning_rows),
    }
    payload["bootstrap_pre_submit_audit_sha256"] = _canonical_sha(payload)
    _atomic_write_json(target, payload)
    return BootstrapPreSubmitAudit.load(target, repository_root=root)


def _git_output(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise BootstrapProvenanceError(f"cannot execute Git: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BootstrapProvenanceError(
            f"Git {' '.join(arguments)} failed: {detail[:1000]}"
        )
    return completed.stdout


def _git_archive_sha256(root: Path, revision: str) -> str:
    try:
        process = subprocess.Popen(
            ("git", "archive", "--format=tar", revision),
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise BootstrapProvenanceError(f"cannot execute git archive: {exc}") from exc
    assert process.stdout is not None
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1 << 20), b""):
        digest.update(chunk)
    stderr = b"" if process.stderr is None else process.stderr.read()
    returncode = process.wait()
    if returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise BootstrapProvenanceError(f"git archive failed: {detail[:1000]}")
    return digest.hexdigest()


def write_source_epoch_contracts(
    *,
    repository_root: str | Path,
    lane_id: str,
    source_revision: str,
    output_dir: str | Path,
    source_epoch_id: str | None = None,
) -> SourceEpochReceipt:
    """Freeze one exact committed Git tree into loader-compatible contracts."""

    root = Path(repository_root).resolve(strict=True)
    lane = str(lane_id).strip()
    revision = str(source_revision).strip().lower()
    if not _SAFE_LANE_ID.fullmatch(lane):
        raise BootstrapProvenanceError("lane_id is not a stable identifier")
    if lane not in _BOOTSTRAP_LANES:
        raise BootstrapProvenanceError(
            "source epochs are limited to prospective HH/Hubbard lanes"
        )
    if not _FULL_GIT_ID.fullmatch(revision):
        raise BootstrapProvenanceError(
            "source_revision must be a full 40- or 64-character commit identity"
        )
    resolved = _git_output(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    resolved_revision = resolved.decode("ascii", errors="strict").strip().lower()
    if resolved_revision != revision:
        raise BootstrapProvenanceError(
            "source_revision does not resolve to that exact commit"
        )
    tree_object = _git_output(
        root, "rev-parse", "--verify", f"{revision}^{{tree}}"
    ).decode("ascii", errors="strict").strip().lower()
    listing = _git_output(root, "ls-tree", "-r", "-z", "--full-tree", revision)
    if not listing:
        raise BootstrapProvenanceError("source revision contains no tracked files")
    entries = _source_tree_entries(listing)
    source_tree_sha256 = _canonical_sha(list(entries))
    source_archive_sha256 = _git_archive_sha256(root, revision)
    epoch_id = str(source_epoch_id or "").strip()
    if not epoch_id:
        lane_token = lane.replace("_", "-")
        epoch_id = (
            f"{lane_token}-source-{revision[:12]}-{source_tree_sha256[:12]}"
        )
    epoch = SourceEpochIdentity(
        lane_id=lane,
        source_epoch_id=epoch_id,
        source_revision=revision,
        source_archive_sha256=source_archive_sha256,
    )
    output = _repository_path(root, output_dir, label="source epoch output")
    manifest_path = output / "source_tree_manifest.json"
    receipt_path = output / "source_epoch_receipt.json"
    manifest = {
        "schema": SOURCE_TREE_MANIFEST_V1,
        "lane_id": lane,
        "source_epoch_identity_sha256": epoch.identity_sha256,
        "source_revision": revision,
        "source_archive_sha256": source_archive_sha256,
        "source_tree_sha256": source_tree_sha256,
        "git_tree_object": tree_object,
        "git_ls_tree_sha256": hashlib.sha256(listing).hexdigest(),
        "tracked_entry_count": len(entries),
        "entries": list(entries),
        "provenance_kind": "git_archive",
    }
    manifest["source_tree_manifest_sha256"] = _canonical_sha(manifest)
    _atomic_write_json(manifest_path, manifest)
    receipt = {
        "schema": SOURCE_EPOCH_RECEIPT_V1,
        "source_epoch": epoch.to_json_dict(),
        "source_tree_manifest": {
            "path": _relative(root, manifest_path, label="source tree manifest"),
            "sha256": _file_sha(manifest_path),
        },
        "source_tree_sha256": source_tree_sha256,
    }
    receipt["source_epoch_receipt_sha256"] = _canonical_sha(receipt)
    _atomic_write_json(receipt_path, receipt)
    return SourceEpochReceipt.load(
        receipt_path,
        repository_root=root,
        expected_epoch=epoch,
    )


def _load_execution_request(root: Path, path: str | Path) -> ExecutionRequest:
    source = _repository_path(root, path, label="execution request")
    payload = _read_object(source, label="execution request")
    try:
        return _bootstrap_request(ExecutionRequest.from_mapping(payload))
    except (TypeError, ValueError) as exc:
        raise BootstrapProvenanceError(
            f"execution request is invalid: {source}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)
    source = subcommands.add_parser("source-epoch")
    source.add_argument("--repo-root", default=".")
    source.add_argument("--lane-id", required=True)
    source.add_argument("--source-revision", required=True)
    source.add_argument("--output-dir", required=True)
    source.add_argument("--source-epoch-id")
    package = subcommands.add_parser("package-manifest")
    package.add_argument("--repo-root", default=".")
    package.add_argument("--request", required=True)
    package.add_argument("--source-epoch-receipt", required=True)
    package.add_argument("--package", required=True)
    package.add_argument("--submit-descriptor", required=True)
    package.add_argument("--job-batch-name", required=True)
    package.add_argument("--output", required=True)
    audit = subcommands.add_parser("audit")
    audit.add_argument("--repo-root", default=".")
    audit.add_argument("--request", required=True)
    audit.add_argument("--source-epoch-receipt", required=True)
    audit.add_argument("--package-manifest", required=True)
    audit.add_argument("--registry", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--warning", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.repo_root).resolve(strict=True)
    if args.command == "source-epoch":
        artifact = write_source_epoch_contracts(
            repository_root=root,
            lane_id=args.lane_id,
            source_revision=args.source_revision,
            output_dir=args.output_dir,
            source_epoch_id=args.source_epoch_id,
        )
        summary = {
            "ok": True,
            "artifact_kind": "source_epoch_receipt",
            "lane_id": artifact.lane_id,
            "source_epoch_id": artifact.source_epoch.source_epoch_id,
            "source_revision": artifact.source_epoch.source_revision,
            "source_tree_sha256": artifact.source_tree_sha256,
            "artifact": artifact.receipt_path,
            "artifact_sha256": artifact.receipt_sha256,
            "production_submission_enabled": False,
        }
    elif args.command == "package-manifest":
        request = _load_execution_request(root, args.request)
        source = SourceEpochReceipt.load(
            _repository_path(
                root,
                args.source_epoch_receipt,
                label="source epoch receipt",
            ),
            repository_root=root,
        )
        artifact = write_bootstrap_package_manifest(
            args.output,
            request=request,
            source_epoch_receipt=source,
            package_path=args.package,
            submit_descriptor=args.submit_descriptor,
            job_batch_name=args.job_batch_name,
            repository_root=root,
        )
        summary = {
            "ok": True,
            "artifact_kind": "bootstrap_package_manifest",
            "lane_id": artifact.request.lane_id,
            "action_id": artifact.request.action_id,
            "execution_request_sha256": artifact.request.digest,
            "artifact": artifact.manifest_path,
            "artifact_sha256": artifact.manifest_sha256,
            "package": artifact.package.path,
            "package_sha256": artifact.package.sha256,
            "package_bytes": artifact.package_bytes,
            "production_submission_enabled": False,
        }
    elif args.command == "audit":
        request = _load_execution_request(root, args.request)
        source = SourceEpochReceipt.load(
            _repository_path(
                root,
                args.source_epoch_receipt,
                label="source epoch receipt",
            ),
            repository_root=root,
        )
        package = BootstrapPackageManifest.load(
            _repository_path(
                root,
                args.package_manifest,
                label="bootstrap package manifest",
            ),
            repository_root=root,
        )
        artifact = write_bootstrap_pre_submit_audit(
            args.output,
            request=request,
            source_epoch_receipt=source,
            package_manifest=package,
            registry_path=args.registry,
            repository_root=root,
            warnings=args.warning,
        )
        summary = {
            "ok": True,
            "artifact_kind": "bootstrap_pre_submit_audit",
            "lane_id": artifact.lane_id,
            "action_id": artifact.action_id,
            "execution_request_sha256": artifact.execution_request_sha256,
            "artifact": artifact.receipt_path,
            "artifact_sha256": artifact.receipt_sha256,
            "production_submission_enabled": False,
            "scheduler_submission_authorized": False,
        }
    else:
        raise AssertionError(f"unsupported command {args.command!r}")
    print(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return 0


__all__ = [
    "BOOTSTRAP_INTENT_FIRST_REQUIREMENTS",
    "BOOTSTRAP_PACKAGE_MANIFEST_V1",
    "BOOTSTRAP_PRE_SUBMIT_AUDIT_V1",
    "BootstrapPackageManifest",
    "BootstrapPreSubmitAudit",
    "BootstrapProvenanceError",
    "main",
    "write_bootstrap_package_manifest",
    "write_bootstrap_pre_submit_audit",
    "write_source_epoch_contracts",
]


if __name__ == "__main__":
    raise SystemExit(main())
