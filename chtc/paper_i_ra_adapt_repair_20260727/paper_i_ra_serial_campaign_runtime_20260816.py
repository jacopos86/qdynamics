"""Small fail-closed runtime primitives for serial local Paper-I campaigns.

The scientific runners own their protocol and result semantics.  This module
only supplies deterministic immutable-artifact, capacity, and lock behavior so
that the canary and overnight supervisors do not acquire subtly different
operational rules.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping

import psutil


class RuntimeContractError(RuntimeError):
    """Raised when an immutable campaign artifact or runtime gate drifts."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    body = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_digested(
    path: Path,
    *,
    schema: str,
    error_type: type[Exception] = RuntimeContractError,
) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise error_type(f"Cannot read immutable artifact: {path}") from exc
    if not isinstance(loaded, dict):
        raise error_type(f"Immutable artifact is not an object: {path}")
    payload = dict(loaded)
    observed = payload.pop("sha256", None)
    if payload.get("schema") != schema or observed != canonical_sha256(payload):
        raise error_type(f"Invalid digested artifact: {path}")
    payload["sha256"] = observed
    return payload


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def artifact_binding(path: Path, root: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeContractError(f"Artifact escapes campaign root: {path}") from exc
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def publish_or_validate_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    schema: str,
    error_type: type[Exception] = RuntimeContractError,
) -> None:
    if path.exists() or path.is_symlink():
        if load_digested(path, schema=schema, error_type=error_type) != dict(payload):
            raise error_type(f"Existing immutable JSON differs: {path}")
        return
    write_json_exclusive(path, payload)


def publish_or_validate_text(
    path: Path,
    body: str,
    *,
    error_type: type[Exception] = RuntimeContractError,
) -> None:
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.read_text(encoding="utf-8") != body:
            raise error_type(f"Existing immutable text differs: {path}")
        return
    write_text_exclusive(path, body)


def prepare_authority_directory(
    authority_dir: Path,
    *,
    files: Mapping[str, Mapping[str, Any]],
    error_type: type[Exception] = RuntimeContractError,
) -> None:
    """Publish several immutable authority files as one directory rename."""

    if authority_dir.exists() or authority_dir.is_symlink():
        raise error_type(f"Authority path already exists: {authority_dir}")
    authority_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{authority_dir.name}.", dir=authority_dir.parent)
    )
    try:
        for name, payload in files.items():
            if Path(name).name != name:
                raise error_type(f"Authority filename is not flat: {name}")
            write_json_exclusive(staging / name, payload)
        os.rename(staging, authority_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def capacity_snapshot(
    memory: int,
    disk: int,
    *,
    launch_memory_bytes: int,
    launch_disk_bytes: int,
) -> dict[str, Any]:
    return {
        "available_memory_bytes": int(memory),
        "free_disk_bytes": int(disk),
        "launch_available_memory_bytes": int(launch_memory_bytes),
        "launch_free_disk_bytes": int(launch_disk_bytes),
        "launch_ready": memory >= launch_memory_bytes and disk >= launch_disk_bytes,
    }


def wait_for_capacity(
    *,
    repo_root: Path,
    launch_memory_bytes: int,
    launch_disk_bytes: int,
    maximum_wait_seconds: float,
    poll_seconds: float = 10.0,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    memory_supplier: Callable[[], int] | None = None,
    disk_supplier: Callable[[], int] | None = None,
) -> dict[str, Any]:
    memory_supplier = memory_supplier or (
        lambda: int(psutil.virtual_memory().available)
    )
    disk_supplier = disk_supplier or (
        lambda: int(shutil.disk_usage(repo_root).free)
    )
    started = clock()
    while True:
        snapshot = capacity_snapshot(
            memory_supplier(),
            disk_supplier(),
            launch_memory_bytes=launch_memory_bytes,
            launch_disk_bytes=launch_disk_bytes,
        )
        elapsed = max(0.0, clock() - started)
        if snapshot["launch_ready"]:
            return {**snapshot, "elapsed_wait_seconds": elapsed, "status": "ready"}
        if elapsed >= maximum_wait_seconds:
            return {
                **snapshot,
                "elapsed_wait_seconds": elapsed,
                "status": "blocked_capacity",
            }
        sleeper(min(poll_seconds, max(0.0, maximum_wait_seconds - elapsed)))


@contextmanager
def exclusive_campaign_lock(
    path: Path,
    *,
    label: str,
    error_type: type[Exception] = RuntimeContractError,
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise error_type(f"{label} lock is already held.") from exc
        yield
    finally:
        os.close(descriptor)
