"""Typed ingestion for completed prospective Paper-II HH seed jobs.

The CHTC seed packages return a deliberately small raw-evidence directory.  This
module authenticates that directory against one canonical execution request and
source epoch, atomically promotes the two scientific files as one directory,
and writes an immutable process-scoped increment receipt.  It never contacts a
scheduler and never treats submission or worker stdout as scientific evidence
without reloading the typed seed receipt.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping

from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HH,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExecutionRequest,
    ProspectiveExecutionError,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    validate_seed_receipt_collection,
)
from pipelines.time_dynamics.prospective_hh_seed_worker import (
    validate_canonical_seed_request,
)


HH_SEED_INCREMENT_RECEIPT_V1 = "paper_ii_hh_seed_increment_receipt_v1"
PRESERVATION_MARKER = b"raw evidence preserved; no remote cleanup authorized\n"
RAW_FILENAMES = frozenset(
    {
        "REMOTE_RAW_PRESERVED",
        "seed.json",
        "seed_qualification_receipt.json",
        "worker_stderr.log",
        "worker_stdout.json",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HHSeedIngestionError(RuntimeError):
    """A prospective HH seed delta cannot be admitted safely."""


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


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat()


def _repository_path(
    root: Path,
    value: str | Path,
    *,
    must_exist: bool,
    label: str,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        lexical = path.absolute()
        relative = lexical.relative_to(root)
        cursor = root
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise HHSeedIngestionError(f"{label} traverses a symlink")
        resolved = path.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise HHSeedIngestionError(f"{label} is outside the repository") from exc
    return resolved


def _regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HHSeedIngestionError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise HHSeedIngestionError(f"{label} is not a regular nofollow file: {path}")


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HHSeedIngestionError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise HHSeedIngestionError(f"{label} is not a JSON object: {path}")
    return dict(payload)


def _execution_request(path: Path) -> tuple[ExecutionRequest, SeedIdentity]:
    try:
        request = ExecutionRequest.from_mapping(
            _read_object(path, label="execution request")
        )
        seed = validate_canonical_seed_request(request)
    except (ProspectiveExecutionError, TypeError, ValueError) as exc:
        raise HHSeedIngestionError("execution request is not a canonical HH seed") from exc
    return request, seed


def _request_source_epoch(request: ExecutionRequest) -> SourceEpochIdentity:
    raw = request.inputs.get("source_epoch")
    if not isinstance(raw, Mapping):
        raise HHSeedIngestionError("execution request has no source epoch")
    try:
        epoch = SourceEpochIdentity(
            lane_id=str(raw.get("lane_id") or ""),
            source_epoch_id=str(raw.get("source_epoch_id") or ""),
            source_revision=raw.get("source_revision"),
            source_archive_sha256=raw.get("source_archive_sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise HHSeedIngestionError("execution request source epoch is invalid") from exc
    if (
        epoch.lane_id != LANE_HH
        or raw.get("source_epoch_identity_sha256") != epoch.identity_sha256
    ):
        raise HHSeedIngestionError("execution request source epoch diverged")
    return epoch


def _source_receipt(
    path: Path,
    *,
    root: Path,
    expected_epoch: SourceEpochIdentity,
) -> SourceEpochReceipt:
    try:
        receipt = SourceEpochReceipt.load(
            path,
            repository_root=root,
            expected_epoch=expected_epoch,
        )
    except (OSError, ProspectiveExecutionError, ValueError) as exc:
        raise HHSeedIngestionError("source epoch receipt does not match the request") from exc
    if receipt.lane_id != LANE_HH:
        raise HHSeedIngestionError("source epoch receipt crosses lanes")
    return receipt


def _validate_increment_root(
    root: Path,
    increment_root: Path,
    *,
    cluster_id: int,
) -> None:
    relative = increment_root.relative_to(root)
    expected_prefix = (
        "output",
        "paper_ii_hh_product_v1",
        "attempts",
        f"cluster_{cluster_id}",
        "increments",
    )
    if len(relative.parts) < 6 or relative.parts[:5] != expected_prefix:
        raise HHSeedIngestionError(
            "increment root is not the declared HH cluster increment directory"
        )


def _raw_inventory(raw_root: Path) -> dict[str, dict[str, Any]]:
    try:
        children = tuple(raw_root.iterdir())
    except OSError as exc:
        raise HHSeedIngestionError("raw output directory is unreadable") from exc
    names = {path.name for path in children}
    if len(children) != len(names) or names != RAW_FILENAMES:
        missing = sorted(RAW_FILENAMES - names)
        extra = sorted(names - RAW_FILENAMES)
        raise HHSeedIngestionError(
            f"raw output inventory is not exact: missing={missing}, extra={extra}"
        )
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(children, key=lambda item: item.name):
        _regular_file(path, label=f"raw output {path.name}")
        inventory[path.name] = {
            "bytes": path.stat(follow_symlinks=False).st_size,
            "sha256": _file_sha(path),
        }
    if (raw_root / "REMOTE_RAW_PRESERVED").read_bytes() != PRESERVATION_MARKER:
        raise HHSeedIngestionError("remote raw preservation marker is invalid")
    return inventory


def _write_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise HHSeedIngestionError("atomic promotion write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _target_pair(
    target_dir: Path,
    *,
    artifact_bytes: bytes,
    receipt_bytes: bytes,
) -> bool:
    """Validate an existing target pair; return True when it is identical."""

    if not os.path.lexists(target_dir):
        return False
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise HHSeedIngestionError("canonical seed target is not a nofollow directory")
    children = tuple(target_dir.iterdir())
    names = {path.name for path in children}
    if len(children) != 2 or names != {"seed.json", "seed_qualification_receipt.json"}:
        raise HHSeedIngestionError("canonical seed target is partial or has extra files")
    for path in children:
        _regular_file(path, label=f"canonical seed {path.name}")
    if (
        (target_dir / "seed.json").read_bytes() != artifact_bytes
        or (target_dir / "seed_qualification_receipt.json").read_bytes()
        != receipt_bytes
    ):
        raise HHSeedIngestionError("canonical seed target contains divergent evidence")
    return True


def _atomic_promote_pair(
    *,
    raw_artifact: Path,
    raw_receipt: Path,
    target_dir: Path,
) -> bool:
    """Promote the pair by one directory rename, never replacing a target.

    Returns ``True`` only when an identical previously promoted pair was
    recovered.  A partial or divergent target always blocks.
    """

    artifact_bytes = raw_artifact.read_bytes()
    receipt_bytes = raw_receipt.read_bytes()
    if _target_pair(
        target_dir,
        artifact_bytes=artifact_bytes,
        receipt_bytes=receipt_bytes,
    ):
        return True
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.parent.is_symlink():
        raise HHSeedIngestionError("canonical seed parent is a symlink")
    lock_path = target_dir.parent / f".{target_dir.name}.ingestion.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    try:
        if _target_pair(
            target_dir,
            artifact_bytes=artifact_bytes,
            receipt_bytes=receipt_bytes,
        ):
            return True
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target_dir.name}.promote-", dir=target_dir.parent
            )
        )
        published = False
        try:
            _write_bytes(staging / "seed.json", artifact_bytes)
            _write_bytes(staging / "seed_qualification_receipt.json", receipt_bytes)
            _fsync_directory(staging)
            try:
                os.rename(staging, target_dir)
                published = True
                _fsync_directory(target_dir.parent)
                return False
            except OSError:
                if _target_pair(
                    target_dir,
                    artifact_bytes=artifact_bytes,
                    receipt_bytes=receipt_bytes,
                ):
                    return True
                raise
        except OSError:
            raise
        finally:
            if not published and staging.exists():
                shutil.rmtree(staging)
    except OSError as exc:
        raise HHSeedIngestionError("canonical seed pair promotion failed") from exc
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _validate_raw_seed(
    *,
    raw_root: Path,
    seed: SeedIdentity,
    source_epoch: SourceEpochIdentity,
) -> SeedQualificationReceipt:
    """Reload raw evidence in an isolated repository-shaped staging root."""

    with tempfile.TemporaryDirectory(prefix="hh-seed-validate-", dir=raw_root.parent) as tmp:
        staging_root = Path(tmp)
        artifact_path = staging_root / seed.artifact_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw_root / "seed.json", artifact_path)
        receipt_path = artifact_path.with_name("seed_qualification_receipt.json")
        shutil.copyfile(raw_root / "seed_qualification_receipt.json", receipt_path)
        try:
            receipt = SeedQualificationReceipt.load(
                receipt_path,
                repository_root=staging_root,
                expected_seed=seed,
                expected_source_epoch=source_epoch,
            )
        except (OSError, ProspectiveExecutionError, ValueError) as exc:
            raise HHSeedIngestionError("raw seed artifact/receipt pair is invalid") from exc
    return receipt


def _validate_worker_stdout(
    payload: Mapping[str, Any],
    *,
    seed: SeedIdentity,
    receipt: SeedQualificationReceipt,
) -> None:
    expected_receipt_path = str(
        Path(seed.artifact_path).with_name("seed_qualification_receipt.json")
    )
    try:
        valid = (
            payload.get("status") == "qualified"
            and payload.get("seed_id") == seed.seed_id
            and payload.get("seed_receipt") == expected_receipt_path
            and payload.get("seed_receipt_sha256") == receipt.receipt_sha256
            and float(payload.get("energy_error")) == receipt.energy_error
            and float(payload.get("exact_state_infidelity"))
            == receipt.exact_state_infidelity
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise HHSeedIngestionError("worker stdout diverges from typed seed evidence")


def _canonical_hh_seeds() -> tuple[SeedIdentity, ...]:
    rows = tuple(
        row for row in build_identity_registry().seeds if row.lane_id == LANE_HH
    )
    if len(rows) != 6 or len({row.seed_id for row in rows}) != 6:
        raise HHSeedIngestionError("canonical HH seed registry is not exactly six")
    return rows


def validate_hh_seed_collection(
    *,
    repository_root: str | Path,
    source_epoch_receipt: str | Path,
) -> tuple[SeedQualificationReceipt, ...]:
    """Load and validate the complete canonical six-seed HH collection."""

    root = Path(repository_root).resolve(strict=True)
    seeds = _canonical_hh_seeds()
    source_path = _repository_path(
        root,
        source_epoch_receipt,
        must_exist=True,
        label="source epoch receipt",
    )
    try:
        source = SourceEpochReceipt.load(source_path, repository_root=root)
    except (OSError, ProspectiveExecutionError, ValueError) as exc:
        raise HHSeedIngestionError("six-seed collection source is invalid") from exc
    if source.lane_id != LANE_HH:
        raise HHSeedIngestionError("six-seed collection source crosses lanes")
    receipts: list[SeedQualificationReceipt] = []
    for seed in seeds:
        path = root / Path(seed.artifact_path).with_name(
            "seed_qualification_receipt.json"
        )
        if not path.is_file():
            raise HHSeedIngestionError(
                f"six-seed collection is incomplete: missing {seed.seed_id}"
            )
        try:
            receipts.append(
                SeedQualificationReceipt.load(
                    path,
                    repository_root=root,
                    expected_seed=seed,
                    expected_source_epoch=source.source_epoch,
                )
            )
        except (OSError, ProspectiveExecutionError, ValueError) as exc:
            raise HHSeedIngestionError(
                f"six-seed collection contains invalid evidence for {seed.seed_id}"
            ) from exc
    try:
        validate_seed_receipt_collection(receipts, expected_seeds=seeds)
    except ProspectiveExecutionError as exc:
        raise HHSeedIngestionError("six-seed collection failed qualification") from exc
    return tuple(sorted(receipts, key=lambda row: row.seed_id))


def _collection_status(
    *,
    root: Path,
    source_receipt: SourceEpochReceipt,
) -> dict[str, Any]:
    seeds = _canonical_hh_seeds()
    present: list[SeedQualificationReceipt] = []
    missing: list[str] = []
    for seed in seeds:
        target = root / Path(seed.artifact_path).with_name(
            "seed_qualification_receipt.json"
        )
        target_dir = target.parent
        if not os.path.lexists(target_dir):
            missing.append(seed.seed_id)
            continue
        if target_dir.is_symlink() or not target_dir.is_dir():
            raise HHSeedIngestionError(
                f"canonical collection target is invalid for {seed.seed_id}"
            )
        names = {path.name for path in target_dir.iterdir()}
        if names != {"seed.json", "seed_qualification_receipt.json"}:
            raise HHSeedIngestionError(
                f"canonical collection target is partial for {seed.seed_id}"
            )
        try:
            present.append(
                SeedQualificationReceipt.load(
                    target,
                    repository_root=root,
                    expected_seed=seed,
                    expected_source_epoch=source_receipt.source_epoch,
                )
            )
        except (OSError, ProspectiveExecutionError, ValueError) as exc:
            raise HHSeedIngestionError(
                f"canonical collection evidence is invalid for {seed.seed_id}"
            ) from exc
    if len({row.artifact.sha256 for row in present}) != len(present) or len(
        {row.optimizer_execution_id for row in present}
    ) != len(present):
        raise HHSeedIngestionError("available HH seeds are not independent")
    if missing:
        return {
            "status": "pending",
            "present_seed_ids": sorted(row.seed_id for row in present),
            "missing_seed_ids": sorted(missing),
            "seed_receipt_sha256": sorted(row.receipt_sha256 for row in present),
        }
    receipts = validate_hh_seed_collection(
        repository_root=root,
        source_epoch_receipt=source_receipt.receipt_path,
    )
    return {
        "status": "complete",
        "present_seed_ids": [row.seed_id for row in receipts],
        "missing_seed_ids": [],
        "seed_receipt_sha256": [row.receipt_sha256 for row in receipts],
    }


def _prior_seed_increment_receipts(
    *, root: Path, seed_id: str
) -> tuple[tuple[Path, dict[str, Any]], ...]:
    attempts = root / "output/paper_ii_hh_product_v1/attempts"
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(
        attempts.glob("cluster_*/increments/*/evidence/increment_receipt.json")
    ):
        payload = _read_object(path, label="prior HH seed increment receipt")
        seed = payload.get("seed")
        if payload.get("schema") == HH_SEED_INCREMENT_RECEIPT_V1 and isinstance(
            seed, Mapping
        ) and seed.get("seed_id") == seed_id:
            found.append((path, payload))
    return tuple(found)


def build_increment_receipt(
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
    collection: Mapping[str, Any],
    recovered_existing_promotion: bool,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Build one self-digested process-scoped seed increment receipt."""

    payload: dict[str, Any] = {
        "schema": HH_SEED_INCREMENT_RECEIPT_V1,
        "status": "complete",
        "lane_id": LANE_HH,
        "cluster_id": int(cluster_id),
        "process_ids": [int(process_id)],
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
            "exact_state_infidelity": seed_receipt.exact_state_infidelity,
        },
        "raw_evidence": {
            "root": raw_root.relative_to(root).as_posix(),
            "files": {
                name: dict(raw_inventory[name]) for name in sorted(raw_inventory)
            },
            "inventory_sha256": _canonical_sha(
                {name: dict(raw_inventory[name]) for name in sorted(raw_inventory)}
            ),
            "remote_raw_preserved": True,
        },
        "canonical_pair_promoted_atomically": True,
        "recovered_existing_promotion": bool(recovered_existing_promotion),
        "six_seed_collection": dict(collection),
        "qiskit_compilation_used": False,
        "recorded_at": str(recorded_at or _utc_now()),
    }
    payload["increment_receipt_sha256"] = _canonical_sha(payload)
    return payload


def write_increment_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable receipt; an identical receipt is idempotent."""

    document = dict(payload)
    declared = document.pop("increment_receipt_sha256", None)
    if (
        declared != _canonical_sha(document)
        or document.get("schema") != HH_SEED_INCREMENT_RECEIPT_V1
    ):
        raise HHSeedIngestionError("increment receipt self digest is invalid")
    content = json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if path.exists():
        _regular_file(path, label="increment receipt")
        if path.read_bytes() != content:
            raise HHSeedIngestionError("refusing to overwrite divergent increment receipt")
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
                    raise HHSeedIngestionError(
                        "increment receipt write made no progress"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.read_bytes() != content:
                raise HHSeedIngestionError(
                    "concurrent divergent increment receipt publication"
                )
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class HHSeedIngestionResult:
    seed_receipt: SeedQualificationReceipt
    increment_receipt_path: str
    increment_receipt_sha256: str
    collection_status: str
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
) -> HHSeedIngestionResult:
    payload = _read_object(path, label="HH seed increment receipt")
    unsigned = dict(payload)
    declared = unsigned.pop("increment_receipt_sha256", None)
    execution = payload.get("execution_request")
    source = payload.get("source_epoch_receipt")
    seed_binding = payload.get("seed")
    raw = payload.get("raw_evidence")
    collection = payload.get("six_seed_collection")
    if (
        payload.get("schema") != HH_SEED_INCREMENT_RECEIPT_V1
        or declared != _canonical_sha(unsigned)
        or payload.get("status") != "complete"
        or payload.get("lane_id") != LANE_HH
        or payload.get("cluster_id") != cluster_id
        or payload.get("process_ids") != [process_id]
        or payload.get("action_id") != request.action_id
        or not isinstance(execution, Mapping)
        or execution.get("execution_request_sha256")
        != request.digest
        or not isinstance(source, Mapping)
        or source.get("source_epoch_identity_sha256")
        != source_epoch.identity_sha256
        or not isinstance(seed_binding, Mapping)
        or seed_binding.get("seed_id") != seed.seed_id
        or not isinstance(raw, Mapping)
        or raw.get("root")
        != raw_root.relative_to(root).as_posix()
        or raw.get("files")
        != {name: dict(raw_inventory[name]) for name in sorted(raw_inventory)}
        or not isinstance(collection, Mapping)
    ):
        raise HHSeedIngestionError("existing increment receipt diverges from input")
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
        raise HHSeedIngestionError("existing increment canonical evidence is invalid") from exc
    if (
        seed_binding.get("artifact") != receipt.artifact.to_json_dict()
        or not isinstance(seed_binding.get("qualification_receipt"), Mapping)
        or seed_binding["qualification_receipt"].get("sha256")
        != receipt.receipt_sha256
    ):
        raise HHSeedIngestionError("existing increment seed binding diverges")
    return HHSeedIngestionResult(
        seed_receipt=receipt,
        increment_receipt_path=path.relative_to(root).as_posix(),
        increment_receipt_sha256=_file_sha(path),
        collection_status=str(collection.get("status")),
        recovered_existing_promotion=bool(
            payload.get("recovered_existing_promotion")
        ),
    )


def ingest_hh_seed_delta(
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
) -> HHSeedIngestionResult:
    """Validate and ingest one successful canonical HH seed process delta."""

    if int(cluster_id) < 0 or int(process_id) != 0:
        raise HHSeedIngestionError("HH seed delta must be one cluster process 0")
    root = Path(repository_root).resolve(strict=True)
    increment = _repository_path(
        root, increment_root, must_exist=False, label="increment root"
    )
    _validate_increment_root(root, increment, cluster_id=int(cluster_id))
    raw_root = _repository_path(
        root, raw_output_root, must_exist=True, label="raw output root"
    )
    try:
        raw_root.relative_to(increment)
    except ValueError as exc:
        raise HHSeedIngestionError("raw output root is outside its increment") from exc
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise HHSeedIngestionError("raw output root is not a nofollow directory")
    inventory = _raw_inventory(raw_root)
    request_path = _repository_path(
        root, execution_request, must_exist=True, label="execution request"
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
        raise HHSeedIngestionError(
            "execution request diverges from the expected action/source identity"
        )
    source_path = _repository_path(
        root,
        source_epoch_receipt,
        must_exist=True,
        label="source epoch receipt",
    )
    source = _source_receipt(source_path, root=root, expected_epoch=epoch)
    raw_receipt = _validate_raw_seed(
        raw_root=raw_root,
        seed=seed,
        source_epoch=epoch,
    )
    _validate_worker_stdout(
        _read_object(raw_root / "worker_stdout.json", label="worker stdout"),
        seed=seed,
        receipt=raw_receipt,
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
    prior = _prior_seed_increment_receipts(root=root, seed_id=seed.seed_id)
    if prior:
        raise HHSeedIngestionError(
            f"seed already covered by another increment receipt: {prior[0][0]}"
        )
    target_dir = (root / seed.artifact_path).parent
    recovered = _atomic_promote_pair(
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
        raise HHSeedIngestionError("promoted seed evidence failed typed reload") from exc
    if (
        promoted.artifact.sha256 != raw_receipt.artifact.sha256
        or promoted.receipt_self_sha256 != raw_receipt.receipt_self_sha256
    ):
        raise HHSeedIngestionError("promoted seed evidence changed after validation")
    collection = _collection_status(root=root, source_receipt=source)
    receipt = build_increment_receipt(
        root=root,
        raw_root=raw_root,
        raw_inventory=inventory,
        request_path=request_path,
        request=request,
        source_receipt=source,
        seed_receipt=promoted,
        cluster_id=int(cluster_id),
        process_id=int(process_id),
        collection=collection,
        recovered_existing_promotion=recovered,
    )
    write_increment_receipt(increment_receipt_path, receipt)
    return HHSeedIngestionResult(
        seed_receipt=promoted,
        increment_receipt_path=increment_receipt_path.relative_to(root).as_posix(),
        increment_receipt_sha256=_file_sha(increment_receipt_path),
        collection_status=str(collection["status"]),
        recovered_existing_promotion=recovered,
    )


__all__ = [
    "HH_SEED_INCREMENT_RECEIPT_V1",
    "HHSeedIngestionError",
    "HHSeedIngestionResult",
    "PRESERVATION_MARKER",
    "RAW_FILENAMES",
    "build_increment_receipt",
    "ingest_hh_seed_delta",
    "validate_hh_seed_collection",
    "write_increment_receipt",
]
