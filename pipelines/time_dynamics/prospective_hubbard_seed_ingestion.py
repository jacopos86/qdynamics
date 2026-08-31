"""Typed ingestion of the sole prospective Paper-II Hubbard-L4 seed delta.

The returned CHTC directory is admitted only when its five-file inventory is
exact, its preservation marker is intact, and its seed artifact/receipt pair
reloads against the exact canonical execution request and source epoch.  The
pair is promoted by one no-overwrite directory rename and covered by an
immutable, process-scoped increment receipt.

No function in this module contacts a scheduler, deletes remote evidence, or
authorizes downstream execution.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HUBBARD,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    ProspectiveExecutionError,
    SeedQualificationReceipt,
    SourceEpochReceipt,
)
from pipelines.time_dynamics.prospective_hh_seed_ingestion import (
    HHSeedIngestionError,
    PRESERVATION_MARKER,
    RAW_FILENAMES,
    _atomic_promote_pair,
    _canonical_sha,
    _file_sha,
    _raw_inventory,
    _read_object,
    _regular_file,
    _repository_path,
    _validate_raw_seed,
    _validate_worker_stdout,
)
from pipelines.time_dynamics.prospective_hubbard_seed_worker import (
    validate_canonical_seed_request,
)


HUBBARD_SEED_INCREMENT_RECEIPT_V1 = (
    "paper_ii_hubbard_l4_seed_increment_receipt_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HubbardSeedIngestionError(RuntimeError):
    """A Hubbard-L4 seed delta cannot be admitted safely."""


def _translate(operation: Any, *args: Any, **kwargs: Any) -> Any:
    """Reuse the audited HH nofollow/atomic primitives with lane-local errors."""

    try:
        return operation(*args, **kwargs)
    except HHSeedIngestionError as exc:
        raise HubbardSeedIngestionError(str(exc)) from exc


def _path(
    root: Path, value: str | Path, *, must_exist: bool, label: str
) -> Path:
    return _translate(
        _repository_path,
        root,
        value,
        must_exist=must_exist,
        label=label,
    )


def _object(path: Path, *, label: str) -> dict[str, Any]:
    return _translate(_read_object, path, label=label)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _canonical_seed() -> SeedIdentity:
    rows = tuple(
        row
        for row in build_identity_registry().seeds
        if row.lane_id == LANE_HUBBARD
    )
    if len(rows) != 1:
        raise HubbardSeedIngestionError(
            "canonical matrix does not contain exactly one Hubbard-L4 seed"
        )
    return rows[0]


def _execution_request(path: Path) -> tuple[ExecutionRequest, SeedIdentity]:
    try:
        request = ExecutionRequest.from_mapping(
            _object(path, label="Hubbard execution request")
        )
        seed = validate_canonical_seed_request(request)
    except (ProspectiveExecutionError, TypeError, ValueError) as exc:
        raise HubbardSeedIngestionError(
            "execution request is not the canonical Hubbard-L4 seed"
        ) from exc
    if seed != _canonical_seed():
        raise HubbardSeedIngestionError("Hubbard seed identity registry changed")
    return request, seed


def _request_source_epoch(request: ExecutionRequest) -> SourceEpochIdentity:
    raw = request.inputs.get("source_epoch")
    if not isinstance(raw, Mapping):
        raise HubbardSeedIngestionError("Hubbard request has no source epoch")
    try:
        epoch = SourceEpochIdentity(
            lane_id=str(raw.get("lane_id") or ""),
            source_epoch_id=str(raw.get("source_epoch_id") or ""),
            source_revision=raw.get("source_revision"),
            source_archive_sha256=raw.get("source_archive_sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise HubbardSeedIngestionError("Hubbard source epoch is invalid") from exc
    if (
        epoch.lane_id != LANE_HUBBARD
        or raw.get("source_epoch_identity_sha256") != epoch.identity_sha256
    ):
        raise HubbardSeedIngestionError("Hubbard source epoch identity diverged")
    return epoch


def _source_receipt(
    path: Path, *, root: Path, expected_epoch: SourceEpochIdentity
) -> SourceEpochReceipt:
    try:
        receipt = SourceEpochReceipt.load(
            path, repository_root=root, expected_epoch=expected_epoch
        )
    except (OSError, ProspectiveExecutionError, ValueError) as exc:
        raise HubbardSeedIngestionError(
            "source epoch receipt does not bind the Hubbard request"
        ) from exc
    if receipt.lane_id != LANE_HUBBARD:
        raise HubbardSeedIngestionError("Hubbard source epoch crosses lanes")
    return receipt


def _validate_increment_root(
    root: Path, increment_root: Path, *, cluster_id: int
) -> None:
    relative = increment_root.relative_to(root)
    expected = (
        "output",
        LANE_HUBBARD,
        "attempts",
        f"cluster_{cluster_id}",
        "increments",
        "proc0",
    )
    if relative.parts != expected:
        raise HubbardSeedIngestionError(
            "increment root is not the declared Hubbard cluster increment"
        )


def _prior_increment_receipts(
    *, root: Path, seed_id: str
) -> tuple[tuple[Path, dict[str, Any]], ...]:
    attempts = root / "output" / LANE_HUBBARD / "attempts"
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(
        attempts.glob("cluster_*/increments/*/evidence/increment_receipt.json")
    ):
        payload = _object(path, label="prior Hubbard seed increment receipt")
        seed = payload.get("seed")
        if (
            payload.get("schema") == HUBBARD_SEED_INCREMENT_RECEIPT_V1
            and isinstance(seed, Mapping)
            and seed.get("seed_id") == seed_id
        ):
            found.append((path, payload))
    return tuple(found)


def _write_increment_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    document = dict(payload)
    declared = document.pop("increment_receipt_sha256", None)
    if (
        declared != _canonical_sha(document)
        or document.get("schema") != HUBBARD_SEED_INCREMENT_RECEIPT_V1
    ):
        raise HubbardSeedIngestionError("increment receipt self digest is invalid")
    content = json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if path.exists():
        _translate(_regular_file, path, label="Hubbard increment receipt")
        if path.read_bytes() != content:
            raise HubbardSeedIngestionError(
                "refusing to overwrite divergent Hubbard increment receipt"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publish", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise HubbardSeedIngestionError(
                        "Hubbard increment receipt write made no progress"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.read_bytes() != content:
                raise HubbardSeedIngestionError(
                    "concurrent divergent Hubbard increment receipt"
                )
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _build_increment_receipt(
    *,
    root: Path,
    raw_root: Path,
    raw_inventory: Mapping[str, Mapping[str, Any]],
    request_path: Path,
    request: ExecutionRequest,
    source_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
    cluster_id: int,
    process_id: int,
    recovered_existing_promotion: bool,
) -> dict[str, Any]:
    files = {name: dict(raw_inventory[name]) for name in sorted(raw_inventory)}
    payload: dict[str, Any] = {
        "schema": HUBBARD_SEED_INCREMENT_RECEIPT_V1,
        "status": "complete",
        "lane_id": LANE_HUBBARD,
        "cluster_id": cluster_id,
        "process_ids": [process_id],
        "action_id": request.action_id,
        "action_ids": [request.action_id],
        "execution_id": request.execution_id,
        "execution_request": {
            "path": request_path.relative_to(root).as_posix(),
            "sha256": _file_sha(request_path),
            "execution_request_sha256": request.digest,
        },
        "source_epoch_receipt": {
            "path": source_receipt.receipt_path,
            "sha256": source_receipt.receipt_sha256,
            "source_epoch_identity_sha256": source_receipt.source_epoch.identity_sha256,
        },
        "seed": {
            "seed_id": seed_receipt.seed_id,
            "seed_identity_sha256": seed_receipt.seed_identity_sha256,
            "artifact": seed_receipt.artifact.to_json_dict(),
            "qualification_receipt": {
                "path": seed_receipt.receipt_path,
                "sha256": seed_receipt.receipt_sha256,
                "seed_receipt_sha256": seed_receipt.receipt_self_sha256,
            },
            "energy_error": seed_receipt.energy_error,
            "energy_gate": 1.0e-6,
            "exact_state_infidelity": seed_receipt.exact_state_infidelity,
            "restarts": seed_receipt.restarts,
            "optimizer_execution_id": seed_receipt.optimizer_execution_id,
        },
        "raw_evidence": {
            "root": raw_root.relative_to(root).as_posix(),
            "files": files,
            "inventory_sha256": _canonical_sha(files),
            "remote_raw_preserved": True,
        },
        "one_seed_collection": {
            "status": "complete",
            "seed_id": seed_receipt.seed_id,
            "seed_receipt_sha256": seed_receipt.receipt_sha256,
        },
        "canonical_pair_promoted_atomically": True,
        "recovered_existing_promotion": recovered_existing_promotion,
        "qiskit_compilation_used": False,
        "recorded_at": _utc_now(),
    }
    payload["increment_receipt_sha256"] = _canonical_sha(payload)
    return payload


@dataclass(frozen=True)
class HubbardSeedIngestionResult:
    seed_receipt: SeedQualificationReceipt
    increment_receipt_path: str
    increment_receipt_sha256: str
    recovered_existing_promotion: bool


def _reload_existing_increment(
    *,
    path: Path,
    root: Path,
    raw_root: Path,
    raw_inventory: Mapping[str, Mapping[str, Any]],
    request: ExecutionRequest,
    seed: SeedIdentity,
    source_epoch: SourceEpochIdentity,
    cluster_id: int,
    process_id: int,
) -> HubbardSeedIngestionResult:
    payload = _object(path, label="Hubbard seed increment receipt")
    unsigned = dict(payload)
    declared = unsigned.pop("increment_receipt_sha256", None)
    execution = payload.get("execution_request")
    source = payload.get("source_epoch_receipt")
    seed_binding = payload.get("seed")
    raw = payload.get("raw_evidence")
    collection = payload.get("one_seed_collection")
    files = {name: dict(raw_inventory[name]) for name in sorted(raw_inventory)}
    if (
        payload.get("schema") != HUBBARD_SEED_INCREMENT_RECEIPT_V1
        or declared != _canonical_sha(unsigned)
        or payload.get("status") != "complete"
        or payload.get("lane_id") != LANE_HUBBARD
        or payload.get("cluster_id") != cluster_id
        or payload.get("process_ids") != [process_id]
        or payload.get("action_id") != request.action_id
        or not isinstance(execution, Mapping)
        or execution.get("execution_request_sha256") != request.digest
        or not isinstance(source, Mapping)
        or source.get("source_epoch_identity_sha256") != source_epoch.identity_sha256
        or not isinstance(seed_binding, Mapping)
        or seed_binding.get("seed_id") != seed.seed_id
        or not isinstance(raw, Mapping)
        or raw.get("root") != raw_root.relative_to(root).as_posix()
        or raw.get("files") != files
        or not isinstance(collection, Mapping)
        or collection.get("status") != "complete"
    ):
        raise HubbardSeedIngestionError(
            "existing Hubbard increment receipt diverges from input"
        )
    receipt_path = root / Path(seed.artifact_path).with_name(
        "seed_qualification_receipt.json"
    )
    try:
        receipt = SeedQualificationReceipt.load(
            receipt_path,
            repository_root=root,
            expected_seed=seed,
            expected_source_epoch=source_epoch,
        )
    except (OSError, ProspectiveExecutionError, ValueError) as exc:
        raise HubbardSeedIngestionError(
            "existing Hubbard canonical evidence is invalid"
        ) from exc
    if (
        seed_binding.get("artifact") != receipt.artifact.to_json_dict()
        or not isinstance(seed_binding.get("qualification_receipt"), Mapping)
        or seed_binding["qualification_receipt"].get("sha256")
        != receipt.receipt_sha256
    ):
        raise HubbardSeedIngestionError("existing Hubbard seed binding diverges")
    return HubbardSeedIngestionResult(
        seed_receipt=receipt,
        increment_receipt_path=path.relative_to(root).as_posix(),
        increment_receipt_sha256=_file_sha(path),
        recovered_existing_promotion=bool(
            payload.get("recovered_existing_promotion")
        ),
    )


def ingest_hubbard_seed_delta(
    *,
    repository_root: str | Path,
    raw_output_root: str | Path,
    execution_request: str | Path,
    source_epoch_receipt: str | Path,
    expected_action_id: str,
    expected_execution_request_sha256: str,
    expected_source_epoch_identity_sha256: str,
    cluster_id: int,
    process_id: int,
    increment_root: str | Path,
) -> HubbardSeedIngestionResult:
    """Validate and ingest one successful Hubbard-L4 seed process delta."""

    if int(cluster_id) < 0 or int(process_id) != 0:
        raise HubbardSeedIngestionError(
            "Hubbard seed delta must be one cluster process 0"
        )
    root = Path(repository_root).resolve(strict=True)
    increment = _path(
        root, increment_root, must_exist=False, label="Hubbard increment root"
    )
    _validate_increment_root(root, increment, cluster_id=int(cluster_id))
    raw_root = _path(
        root, raw_output_root, must_exist=True, label="Hubbard raw output root"
    )
    try:
        raw_root.relative_to(increment)
    except ValueError as exc:
        raise HubbardSeedIngestionError(
            "Hubbard raw output root is outside its increment"
        ) from exc
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise HubbardSeedIngestionError(
            "Hubbard raw output root is not a nofollow directory"
        )
    inventory = _translate(_raw_inventory, raw_root)
    if set(inventory) != RAW_FILENAMES:
        raise HubbardSeedIngestionError("Hubbard raw inventory changed")
    request_path = _path(
        root,
        execution_request,
        must_exist=True,
        label="Hubbard execution request",
    )
    request, seed = _execution_request(request_path)
    epoch = _request_source_epoch(request)
    if (
        request.action_id != str(expected_action_id)
        or request.digest != str(expected_execution_request_sha256)
        or epoch.identity_sha256 != str(expected_source_epoch_identity_sha256)
        or not _SHA256.fullmatch(str(expected_execution_request_sha256))
        or not _SHA256.fullmatch(str(expected_source_epoch_identity_sha256))
    ):
        raise HubbardSeedIngestionError(
            "Hubbard execution request diverges from action/request/source identity"
        )
    source_path = _path(
        root,
        source_epoch_receipt,
        must_exist=True,
        label="Hubbard source epoch receipt",
    )
    source = _source_receipt(source_path, root=root, expected_epoch=epoch)
    raw_receipt = _translate(
        _validate_raw_seed,
        raw_root=raw_root,
        seed=seed,
        source_epoch=epoch,
    )
    _translate(
        _validate_worker_stdout,
        _object(raw_root / "worker_stdout.json", label="Hubbard worker stdout"),
        seed=seed,
        receipt=raw_receipt,
    )
    if raw_receipt.energy_error > 1.0e-6 or raw_receipt.restarts != 4:
        raise HubbardSeedIngestionError(
            "Hubbard seed fails the hard energy/restart qualification gate"
        )
    increment_receipt_path = increment / "evidence/increment_receipt.json"
    if increment_receipt_path.exists():
        return _reload_existing_increment(
            path=increment_receipt_path,
            root=root,
            raw_root=raw_root,
            raw_inventory=inventory,
            request=request,
            seed=seed,
            source_epoch=epoch,
            cluster_id=int(cluster_id),
            process_id=int(process_id),
        )
    prior = _prior_increment_receipts(root=root, seed_id=seed.seed_id)
    if prior:
        raise HubbardSeedIngestionError(
            f"Hubbard seed already covered by another increment: {prior[0][0]}"
        )
    target_dir = (root / seed.artifact_path).parent
    recovered = _translate(
        _atomic_promote_pair,
        raw_artifact=raw_root / "seed.json",
        raw_receipt=raw_root / "seed_qualification_receipt.json",
        target_dir=target_dir,
    )
    try:
        promoted = SeedQualificationReceipt.load(
            target_dir / "seed_qualification_receipt.json",
            repository_root=root,
            expected_seed=seed,
            expected_source_epoch=epoch,
        )
    except (OSError, ProspectiveExecutionError, ValueError) as exc:
        raise HubbardSeedIngestionError(
            "promoted Hubbard seed failed typed reload"
        ) from exc
    if (
        promoted.artifact.sha256 != raw_receipt.artifact.sha256
        or promoted.receipt_self_sha256 != raw_receipt.receipt_self_sha256
    ):
        raise HubbardSeedIngestionError(
            "promoted Hubbard seed evidence changed after validation"
        )
    receipt = _build_increment_receipt(
        root=root,
        raw_root=raw_root,
        raw_inventory=inventory,
        request_path=request_path,
        request=request,
        source_receipt=source,
        seed_receipt=promoted,
        cluster_id=int(cluster_id),
        process_id=int(process_id),
        recovered_existing_promotion=recovered,
    )
    _write_increment_receipt(increment_receipt_path, receipt)
    return HubbardSeedIngestionResult(
        seed_receipt=promoted,
        increment_receipt_path=increment_receipt_path.relative_to(root).as_posix(),
        increment_receipt_sha256=_file_sha(increment_receipt_path),
        recovered_existing_promotion=recovered,
    )


__all__ = [
    "HUBBARD_SEED_INCREMENT_RECEIPT_V1",
    "HubbardSeedIngestionError",
    "HubbardSeedIngestionResult",
    "PRESERVATION_MARKER",
    "RAW_FILENAMES",
    "ingest_hubbard_seed_delta",
]
