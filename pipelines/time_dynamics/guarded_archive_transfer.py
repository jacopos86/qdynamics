"""Fail-closed recovery and finalization for incremental archive transfers.

The acquisition supervisor treats ``.part`` files as untrusted bytes.  A
partial file may be resumed only after a caller proves that its complete local
contents are the exact prefix of the immutable remote archive.  Otherwise the
partial is moved to a non-overwriting quarantine location and a durable receipt
is written before a fresh transfer can start.

This module owns only local state and verification.  The CHTC adapter supplies
the remote-prefix digest through a narrow callback, so tests never need network
access and no local helper can silently initiate SSH or Duo authentication.
"""

from __future__ import annotations

import datetime as _dt
import contextlib
import errno
import gzip
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tarfile
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping


TRANSFER_RECOVERY_RECEIPT_V1 = "guarded_archive_transfer_recovery_v1"
TRANSFER_QUARANTINE_INTENT_V1 = "guarded_archive_transfer_quarantine_intent_v1"
TRANSFER_FINALIZATION_RECEIPT_V1 = "guarded_archive_transfer_finalization_v1"
TRANSFER_FINALIZATION_INTENT_V1 = "guarded_archive_transfer_finalization_intent_v1"
DEFAULT_FREE_FLOOR_BYTES = 10 * 1024**3


class TransferRecoveryError(RuntimeError):
    """A transfer target cannot be resumed or finalized safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    token = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return token


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    spec: ArchiveTransferSpec | None = None,
) -> None:
    """Atomically publish one immutable JSON receipt without replacement.

    The final pathname is never visible until the complete JSON bytes have
    been written and fsynced.  ``link(2)`` is the publication primitive, so a
    concurrent writer receives ``FileExistsError`` and cannot overwrite the
    earlier evidence.
    """

    if spec is not None:
        _write_json_exclusive_confined(spec, path, payload)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publish", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        try:
            content = (
                json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise TransferRecoveryError("receipt write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class ArchiveTransferSpec:
    """Immutable local/remote identity for one archive transfer."""

    remote_path: str
    local_final_path: Path
    expected_bytes: int
    expected_sha256: str
    expected_member_count: int
    quarantine_root: Path | None = None
    recovery_receipt_path: Path | None = None
    finalization_receipt_path: Path | None = None
    finalization_intent_path: Path | None = None
    extraction_reserve_bytes: int = 0
    local_confinement_root: Path | None = None

    def __post_init__(self) -> None:
        remote = PurePosixPath(str(self.remote_path))
        if (
            not remote.is_absolute()
            or ".." in remote.parts
            or str(remote) != str(self.remote_path)
        ):
            raise ValueError("remote archive path must be absolute and canonical")
        final = Path(self.local_final_path)
        if not final.is_absolute() or final.name.endswith(".part"):
            raise ValueError("local final archive path must be absolute and final")
        if int(self.expected_bytes) <= 0:
            raise ValueError("expected archive size must be positive")
        if int(self.expected_member_count) <= 0:
            raise ValueError("expected archive member count must be positive")
        if int(self.extraction_reserve_bytes) < 0:
            raise ValueError("extraction reserve must be nonnegative")
        object.__setattr__(
            self,
            "expected_sha256",
            _require_sha256(self.expected_sha256, label="expected archive digest"),
        )
        for label, value in (
            ("quarantine_root", self.quarantine_root),
            ("recovery_receipt_path", self.recovery_receipt_path),
            ("finalization_receipt_path", self.finalization_receipt_path),
            ("finalization_intent_path", self.finalization_intent_path),
            ("local_confinement_root", self.local_confinement_root),
        ):
            if value is not None and not Path(value).is_absolute():
                raise ValueError(f"{label} must be absolute when supplied")

    @property
    def part_path(self) -> Path:
        return Path(f"{self.local_final_path}.part")

    @property
    def effective_quarantine_root(self) -> Path:
        return self.quarantine_root or self.local_final_path.parent / "quarantine"

    @property
    def effective_recovery_receipt_path(self) -> Path:
        return self.recovery_receipt_path or (
            self.local_final_path.parent / "transfer_recovery_receipt.json"
        )

    @property
    def effective_finalization_receipt_path(self) -> Path:
        return self.finalization_receipt_path or (
            self.local_final_path.parent / "transfer_finalization_receipt.json"
        )

    @property
    def effective_finalization_intent_path(self) -> Path:
        return self.finalization_intent_path or (
            self.local_final_path.parent / "transfer_finalization_intent.json"
        )


_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


def _require_nofollow_support() -> None:
    if not _O_NOFOLLOW or not _O_DIRECTORY:
        raise TransferRecoveryError(
            "path confinement requires O_NOFOLLOW and O_DIRECTORY"
        )


def _durably_create_directory_tree(path: Path) -> None:
    """Create a missing directory chain and fsync every new parent entry."""

    _require_nofollow_support()
    target = path.absolute()
    missing: list[str] = []
    cursor = target
    while True:
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            if cursor == cursor.parent:
                raise TransferRecoveryError(
                    f"no existing ancestor for confinement root: {target}"
                )
            missing.append(cursor.name)
            cursor = cursor.parent
            continue
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise TransferRecoveryError(
                f"confinement ancestor is not a nofollow directory: {cursor}"
            )
        break

    flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC
    descriptor = os.open(cursor, flags)
    try:
        for component in reversed(missing):
            try:
                os.mkdir(component, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            # Persist the new directory entry in its parent before relying on
            # the child as the root of immutable transfer evidence.
            os.fsync(descriptor)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.fsync(next_descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    finally:
        os.close(descriptor)


def _confinement_root(spec: ArchiveTransferSpec) -> Path:
    """Return the trusted lexical root used for descriptor-relative traversal."""

    root = Path(spec.local_confinement_root or spec.local_final_path.parent).absolute()
    if spec.local_confinement_root is None:
        _durably_create_directory_tree(root)
    try:
        if root.is_symlink() or root.resolve(strict=True) != root:
            raise TransferRecoveryError("archive path confinement root is not canonical")
    except OSError as exc:
        raise TransferRecoveryError(
            f"archive path confinement root is unavailable: {exc}"
        ) from exc
    return root


@contextlib.contextmanager
def _secure_parent_directory(
    spec: ArchiveTransferSpec,
    path: Path,
    *,
    create: bool,
):
    """Pin ``path.parent`` below the registered root without following symlinks."""

    _require_nofollow_support()
    root = _confinement_root(spec)
    lexical = Path(path).absolute()
    try:
        relative_parent = lexical.parent.relative_to(root)
    except ValueError as exc:
        raise TransferRecoveryError(
            f"archive path escapes confinement root: {lexical}"
        ) from exc
    flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise TransferRecoveryError(
            f"archive path confinement root cannot be opened: {exc}"
        ) from exc
    try:
        for component in relative_parent.parts:
            if component in {"", ".", ".."}:
                raise TransferRecoveryError("archive path is not canonical")
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise TransferRecoveryError(
                    f"archive path confinement rejected {lexical.parent}: {exc}"
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor
    finally:
        os.close(descriptor)


def _open_existing_regular(
    spec: ArchiveTransferSpec,
    path: Path,
    *,
    writable: bool = False,
) -> int | None:
    flags = (os.O_RDWR if writable else os.O_RDONLY) | _O_NOFOLLOW | _O_CLOEXEC
    try:
        with _secure_parent_directory(spec, path, create=False) as parent:
            try:
                descriptor = os.open(path.name, flags, dir_fd=parent)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise TransferRecoveryError(
                    f"archive path is not a nofollow regular file: {path}: {exc}"
                ) from exc
    except FileNotFoundError:
        return None
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise TransferRecoveryError(f"archive path is not a regular file: {path}")
    return descriptor


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _read_fd_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _archive_member_count_fd(descriptor: int) -> int:
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as handle:
            while handle.read(1 << 20):
                pass
    # dup(2) shares the open-file description and therefore its offset.
    # Rewind before the independent tar reader and again for the caller.
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as raw:
        with tarfile.open(fileobj=raw, mode="r:gz") as archive:
            count = sum(1 for _ in archive)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return count


def _same_inode(metadata: os.stat_result, other: os.stat_result) -> bool:
    return (metadata.st_dev, metadata.st_ino) == (other.st_dev, other.st_ino)


def _same_regular_name(parent: int, name: str, metadata: os.stat_result) -> bool:
    try:
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(current.st_mode) and _same_inode(current, metadata)


def _write_json_exclusive_confined(
    spec: ArchiveTransferSpec,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    content = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with _secure_parent_directory(spec, path, create=True) as parent:
        temporary_name = f".{path.name}.{secrets.token_hex(12)}.publish"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC,
            0o600,
            dir_fd=parent,
        )
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise TransferRecoveryError("receipt write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
            os.fsync(parent)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=parent)
                os.fsync(parent)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class TransferRecoveryDecision:
    status: str
    resume_offset: int
    part_sha256: str | None
    quarantine_path: Path | None
    receipt_path: Path | None


def _nonoverwriting_quarantine_path(
    spec: ArchiveTransferSpec,
    digest: str,
    *,
    source_name: str | None = None,
) -> Path:
    root = spec.effective_quarantine_root
    stem = f"{source_name or spec.part_path.name}.{digest[:16]}"
    return root / stem


def _move_to_quarantine(
    source: Path,
    spec: ArchiveTransferSpec,
    digest: str,
) -> Path:
    """Move by exclusive hard-link allocation so a concurrent attempt cannot win."""

    source_descriptor = _open_existing_regular(spec, source, writable=True)
    if source_descriptor is None:
        raise TransferRecoveryError(f"quarantine source disappeared: {source}")
    try:
        source_metadata = os.fstat(source_descriptor)
        if _sha256_fd(source_descriptor) != digest:
            raise TransferRecoveryError("quarantine source changed before move")
        # External or recovered partials may never have passed through the
        # chunk writer.  Make their bytes durable before publishing any
        # durable move metadata.
        os.fsync(source_descriptor)
        _write_quarantine_intent(
            source, spec, digest, source_metadata=source_metadata
        )

        base = _nonoverwriting_quarantine_path(
            spec, digest, source_name=source.name
        )
        with _secure_parent_directory(spec, source, create=False) as source_parent:
            if not _same_regular_name(
                source_parent, source.name, source_metadata
            ):
                raise TransferRecoveryError(
                    "quarantine source identity changed before publication"
                )
            with _secure_parent_directory(spec, base, create=True) as target_parent:
                ordinal = 0
                while True:
                    candidate = (
                        base
                        if ordinal == 0
                        else base.with_name(f"{base.name}.{ordinal:04d}")
                    )
                    try:
                        os.link(
                            source.name,
                            candidate.name,
                            src_dir_fd=source_parent,
                            dst_dir_fd=target_parent,
                            follow_symlinks=False,
                        )
                        copied = False
                        break
                    except FileExistsError:
                        ordinal += 1
                        continue
                    except OSError as exc:
                        if exc.errno != errno.EXDEV:
                            raise
                        copied = True
                        temporary_name = (
                            f".{candidate.name}.{secrets.token_hex(12)}.quarantine"
                        )
                        temporary_descriptor = os.open(
                            temporary_name,
                            os.O_RDWR
                            | os.O_CREAT
                            | os.O_EXCL
                            | _O_NOFOLLOW
                            | _O_CLOEXEC,
                            0o600,
                            dir_fd=target_parent,
                        )
                        try:
                            os.lseek(source_descriptor, 0, os.SEEK_SET)
                            while True:
                                chunk = os.read(source_descriptor, 1 << 20)
                                if not chunk:
                                    break
                                view = memoryview(chunk)
                                while view:
                                    written = os.write(temporary_descriptor, view)
                                    if written <= 0:
                                        raise TransferRecoveryError(
                                            "quarantine copy made no progress"
                                        )
                                    view = view[written:]
                            os.fsync(temporary_descriptor)
                            if _sha256_fd(temporary_descriptor) != digest:
                                raise TransferRecoveryError(
                                    "cross-filesystem quarantine copy digest mismatch"
                                )
                            try:
                                os.link(
                                    temporary_name,
                                    candidate.name,
                                    src_dir_fd=target_parent,
                                    dst_dir_fd=target_parent,
                                    follow_symlinks=False,
                                )
                            except FileExistsError:
                                ordinal += 1
                                continue
                            break
                        finally:
                            os.close(temporary_descriptor)
                            try:
                                os.unlink(temporary_name, dir_fd=target_parent)
                            except FileNotFoundError:
                                pass

                candidate_descriptor = os.open(
                    candidate.name,
                    os.O_RDWR | _O_NOFOLLOW | _O_CLOEXEC,
                    dir_fd=target_parent,
                )
                try:
                    candidate_metadata = os.fstat(candidate_descriptor)
                    if (
                        not stat.S_ISREG(candidate_metadata.st_mode)
                        or _sha256_fd(candidate_descriptor) != digest
                        or (
                            not copied
                            and not _same_inode(
                                candidate_metadata, source_metadata
                            )
                        )
                    ):
                        raise TransferRecoveryError(
                            "quarantine candidate identity mismatch"
                        )
                    # This fsync is required even on the same-filesystem
                    # hard-link path: the source may have been created outside
                    # this guarded transfer.
                    os.fsync(candidate_descriptor)
                finally:
                    os.close(candidate_descriptor)
                os.fsync(target_parent)
                if not _same_regular_name(
                    source_parent, source.name, source_metadata
                ):
                    raise TransferRecoveryError(
                        "quarantine source identity changed before unlink"
                    )
                os.unlink(source.name, dir_fd=source_parent)
                os.fsync(source_parent)
                return candidate
    finally:
        os.close(source_descriptor)


def _write_quarantine_intent(
    source: Path,
    spec: ArchiveTransferSpec,
    digest: str,
    *,
    source_metadata: os.stat_result | None = None,
) -> Path:
    """Publish durable provenance before unsafe bytes can leave their source path."""

    payload = {
        "schema": TRANSFER_QUARANTINE_INTENT_V1,
        "status": "quarantine_required",
        "recorded_at": _utc_now(),
        "remote_path": spec.remote_path,
        "source_path": str(source),
        "source_kind": (
            "final" if source == spec.local_final_path else "partial"
        ),
        "source_bytes": int(
            (source_metadata or os.stat(source, follow_symlinks=False)).st_size
        ),
        "source_sha256": _require_sha256(digest, label="quarantine source"),
        "quarantine_root": str(spec.effective_quarantine_root),
    }
    payload["intent_sha256"] = _canonical_payload_sha256(payload)
    base = (
        spec.local_final_path.parent
        / "quarantine_intents"
        / "quarantine_intent.json"
    )
    ordinal = 0
    while True:
        path = (
            base
            if ordinal == 0
            else base.with_name(
                f"{base.stem}.attempt_{ordinal:04d}{base.suffix}"
            )
        )
        try:
            _write_json_exclusive(path, payload, spec=spec)
            return path
        except FileExistsError:
            ordinal += 1


def _write_recovery_receipt(
    spec: ArchiveTransferSpec,
    *,
    status: str,
    part_bytes: int,
    part_sha256: str,
    remote_prefix_sha256: str | None,
    quarantine_path: Path | None,
    detail: str,
    source_path: Path | None = None,
    source_kind: str = "partial",
) -> Path:
    receipt = {
        "schema": TRANSFER_RECOVERY_RECEIPT_V1,
        "status": status,
        "recorded_at": _utc_now(),
        "remote": {
            "path": spec.remote_path,
            "expected_bytes": int(spec.expected_bytes),
            "expected_sha256": spec.expected_sha256,
            "expected_member_count": int(spec.expected_member_count),
        },
        "local": {
            "final_path": str(spec.local_final_path),
            "part_path": str(spec.part_path),
            "source_path": str(source_path or spec.part_path),
            "source_kind": source_kind,
            "part_bytes": int(part_bytes),
            "part_sha256": part_sha256,
        },
        "prefix_proof": {
            "length": int(part_bytes),
            "remote_prefix_sha256": remote_prefix_sha256,
            "matches": remote_prefix_sha256 == part_sha256,
        },
        "quarantine_path": (
            None if quarantine_path is None else str(quarantine_path)
        ),
        "detail": str(detail),
    }
    receipt["recovery_receipt_sha256"] = _canonical_payload_sha256(receipt)
    base = spec.effective_recovery_receipt_path
    ordinal = 0
    while True:
        path = (
            base
            if ordinal == 0
            else base.with_name(
                f"{base.stem}.attempt_{ordinal:04d}{base.suffix}"
            )
        )
        try:
            _write_json_exclusive(path, receipt, spec=spec)
            return path
        except FileExistsError:
            ordinal += 1


def prepare_partial_transfer(
    spec: ArchiveTransferSpec,
    *,
    remote_prefix_sha256: Callable[[int], str],
) -> TransferRecoveryDecision:
    """Return a safe resume offset or quarantine an unproved partial.

    ``remote_prefix_sha256`` must compute SHA-256 over exactly the first
    ``length`` bytes of the immutable remote archive.  Failure to return a
    valid digest is treated as absence of proof, never as permission to resume.
    """

    final = spec.local_final_path
    part = spec.part_path
    final_descriptor = _open_existing_regular(spec, final)
    if final_descriptor is not None:
        try:
            final_metadata = os.fstat(final_descriptor)
            size = int(final_metadata.st_size)
            digest = _sha256_fd(final_descriptor)
        finally:
            os.close(final_descriptor)
        if size != spec.expected_bytes or digest != spec.expected_sha256:
            quarantine = _move_to_quarantine(final, spec, digest)
            receipt_path = _write_recovery_receipt(
                spec,
                status="quarantined_invalid_final",
                part_bytes=size,
                part_sha256=digest,
                remote_prefix_sha256=(
                    spec.expected_sha256 if size == spec.expected_bytes else None
                ),
                quarantine_path=quarantine,
                detail="existing final archive failed its immutable size/digest binding",
                source_path=final,
                source_kind="final",
            )
            return TransferRecoveryDecision(
                status="restart_after_quarantine",
                resume_offset=0,
                part_sha256=digest,
                quarantine_path=quarantine,
                receipt_path=receipt_path,
            )
        part_descriptor = _open_existing_regular(spec, part)
        if part_descriptor is not None:
            try:
                part_metadata = os.fstat(part_descriptor)
            finally:
                os.close(part_descriptor)
            if not _same_inode(final_metadata, part_metadata):
                raise TransferRecoveryError(
                    "verified final archive and partial archive coexist"
                )
        return TransferRecoveryDecision(
            status="final_present_requires_recovery",
            resume_offset=size,
            part_sha256=digest,
            quarantine_path=None,
            receipt_path=None,
        )
    part_descriptor = _open_existing_regular(spec, part)
    if part_descriptor is None:
        return TransferRecoveryDecision(
            status="start_empty",
            resume_offset=0,
            part_sha256=None,
            quarantine_path=None,
            receipt_path=None,
        )
    try:
        part_metadata = os.fstat(part_descriptor)
        size = int(part_metadata.st_size)
        digest = _sha256_fd(part_descriptor)
        remote_digest: str | None = None
        detail = ""
        if size <= spec.expected_bytes:
            try:
                remote_digest = _require_sha256(
                    remote_prefix_sha256(size), label="remote prefix digest"
                )
            except Exception as exc:  # proof failure never authorizes a resume
                detail = f"remote prefix proof unavailable: {exc}"
        else:
            detail = "partial archive exceeds the registered remote size"

        current_metadata = os.fstat(part_descriptor)
        current_digest = _sha256_fd(part_descriptor)
        with _secure_parent_directory(spec, part, create=False) as parent:
            name_is_current = _same_regular_name(
                parent, part.name, part_metadata
            )
        if (
            not _same_inode(part_metadata, current_metadata)
            or int(current_metadata.st_size) != size
            or current_digest != digest
            or not name_is_current
        ):
            raise TransferRecoveryError(
                "partial archive changed during remote-prefix proof"
            )
    finally:
        os.close(part_descriptor)

    if remote_digest == digest:
        receipt_path = _write_recovery_receipt(
            spec,
            status="prefix_verified",
            part_bytes=size,
            part_sha256=digest,
            remote_prefix_sha256=remote_digest,
            quarantine_path=None,
            detail="resume authorized from exact verified remote prefix",
        )
        return TransferRecoveryDecision(
            status="resume_verified_prefix",
            resume_offset=size,
            part_sha256=digest,
            quarantine_path=None,
            receipt_path=receipt_path,
        )

    quarantine = _move_to_quarantine(part, spec, digest)
    receipt_path = _write_recovery_receipt(
        spec,
        status="quarantined_unverified_partial",
        part_bytes=size,
        part_sha256=digest,
        remote_prefix_sha256=remote_digest,
        quarantine_path=quarantine,
        detail=detail or "local bytes do not match the exact remote prefix",
    )
    return TransferRecoveryDecision(
        status="restart_after_quarantine",
        resume_offset=0,
        part_sha256=digest,
        quarantine_path=quarantine,
        receipt_path=receipt_path,
    )


def finalize_archive_transfer(spec: ArchiveTransferSpec) -> Mapping[str, Any]:
    """Verify a complete ``.part`` and publish it without overwriting evidence."""

    part = spec.part_path
    final = spec.local_final_path
    final_descriptor = _open_existing_regular(spec, final, writable=True)
    if final_descriptor is not None:
        try:
            final_metadata = os.fstat(final_descriptor)
            size = int(final_metadata.st_size)
            digest = _sha256_fd(final_descriptor)
            if size == spec.expected_bytes and digest == spec.expected_sha256:
                try:
                    member_count = _archive_member_count_fd(final_descriptor)
                except (OSError, EOFError, tarfile.TarError, zlib.error) as exc:
                    member_count = -1
                    archive_error: BaseException | None = exc
                else:
                    archive_error = None
            else:
                member_count = -1
                archive_error = None
        finally:
            os.close(final_descriptor)
        if size != spec.expected_bytes or digest != spec.expected_sha256:
            quarantine = _move_to_quarantine(final, spec, digest)
            _write_recovery_receipt(
                spec,
                status="quarantined_invalid_final",
                part_bytes=size,
                part_sha256=digest,
                remote_prefix_sha256=(
                    spec.expected_sha256 if size == spec.expected_bytes else None
                ),
                quarantine_path=quarantine,
                detail="existing final archive failed its immutable size/digest binding",
                source_path=final,
                source_kind="final",
            )
            raise TransferRecoveryError("existing final archive binding mismatch")
        if archive_error is not None:
            quarantine = _move_to_quarantine(final, spec, digest)
            _write_recovery_receipt(
                spec,
                status="quarantined_invalid_final_archive",
                part_bytes=size,
                part_sha256=digest,
                remote_prefix_sha256=spec.expected_sha256,
                quarantine_path=quarantine,
                detail=(
                    "existing final archive integrity check failed: "
                    f"{archive_error}"
                ),
                source_path=final,
                source_kind="final",
            )
            raise TransferRecoveryError(
                "existing final archive integrity check failed: "
                f"{archive_error}"
            ) from archive_error
        if member_count != spec.expected_member_count:
            quarantine = _move_to_quarantine(final, spec, digest)
            _write_recovery_receipt(
                spec,
                status="quarantined_final_member_count_mismatch",
                part_bytes=size,
                part_sha256=digest,
                remote_prefix_sha256=spec.expected_sha256,
                quarantine_path=quarantine,
                detail=(
                    f"expected {spec.expected_member_count} archive members, "
                    f"observed {member_count}"
                ),
                source_path=final,
                source_kind="final",
            )
            raise TransferRecoveryError("final archive member-count binding mismatch")
        intent_path = spec.effective_finalization_intent_path
        part_descriptor = _open_existing_regular(spec, part, writable=True)
        if part_descriptor is not None:
            try:
                part_metadata = os.fstat(part_descriptor)
                if not _same_inode(final_metadata, part_metadata):
                    raise TransferRecoveryError(
                        "verified final archive and partial archive coexist"
                    )
                intent = _read_finalization_intent(intent_path, spec)
                os.fsync(part_descriptor)
                with _secure_parent_directory(
                    spec, final, create=False
                ) as parent:
                    if not _same_regular_name(
                        parent, final.name, final_metadata
                    ) or not _same_regular_name(
                        parent, part.name, part_metadata
                    ):
                        raise TransferRecoveryError(
                            "finalization link identity changed during replay"
                        )
                    os.unlink(part.name, dir_fd=parent)
                    os.fsync(parent)
            finally:
                os.close(part_descriptor)
        else:
            intent_descriptor = _open_existing_regular(spec, intent_path)
            if intent_descriptor is None:
                raise TransferRecoveryError(
                    "verified final archive has neither receipt nor finalization intent"
                )
            os.close(intent_descriptor)
            intent = _read_finalization_intent(intent_path, spec)

        receipt_path = spec.effective_finalization_receipt_path
        receipt_descriptor = _open_existing_regular(spec, receipt_path)
        if receipt_descriptor is not None:
            os.close(receipt_descriptor)
            return _read_finalization_receipt(receipt_path, spec)
        if not intent:
            raise TransferRecoveryError(
                "verified final archive has neither receipt nor finalization intent"
            )
        return _write_finalization_receipt(
            spec,
            member_count=member_count,
            recovered_after_rename=True,
            intent=intent,
        )
    part_descriptor = _open_existing_regular(spec, part, writable=True)
    if part_descriptor is None:
        raise TransferRecoveryError(f"complete partial archive is missing: {part}")
    try:
        part_metadata = os.fstat(part_descriptor)
        size = int(part_metadata.st_size)
        digest = _sha256_fd(part_descriptor)
        if size != spec.expected_bytes or digest != spec.expected_sha256:
            os.close(part_descriptor)
            part_descriptor = -1
            quarantine = _move_to_quarantine(part, spec, digest)
            _write_recovery_receipt(
                spec,
                status="quarantined_invalid_complete_partial",
                part_bytes=size,
                part_sha256=digest,
                remote_prefix_sha256=(
                    spec.expected_sha256 if size == spec.expected_bytes else None
                ),
                quarantine_path=quarantine,
                detail="complete partial archive failed immutable size/digest binding",
            )
            raise TransferRecoveryError("complete partial archive binding mismatch")
        try:
            member_count = _archive_member_count_fd(part_descriptor)
        except (OSError, EOFError, tarfile.TarError, zlib.error) as exc:
            os.close(part_descriptor)
            part_descriptor = -1
            quarantine = _move_to_quarantine(part, spec, digest)
            _write_recovery_receipt(
                spec,
                status="quarantined_invalid_archive",
                part_bytes=size,
                part_sha256=digest,
                remote_prefix_sha256=spec.expected_sha256,
                quarantine_path=quarantine,
                detail=f"archive integrity check failed: {exc}",
            )
            raise TransferRecoveryError(
                f"archive integrity check failed: {exc}"
            ) from exc
        if member_count != spec.expected_member_count:
            os.close(part_descriptor)
            part_descriptor = -1
            quarantine = _move_to_quarantine(part, spec, digest)
            _write_recovery_receipt(
                spec,
                status="quarantined_member_count_mismatch",
                part_bytes=size,
                part_sha256=digest,
                remote_prefix_sha256=spec.expected_sha256,
                quarantine_path=quarantine,
                detail=(
                    f"expected {spec.expected_member_count} archive members, "
                    f"observed {member_count}"
                ),
            )
            raise TransferRecoveryError("archive member-count binding mismatch")

        # A recovered complete part may have been written by a previous
        # process and therefore has no write-path fsync guarantee.  Make the
        # verified inode durable before publishing the intent.
        os.fsync(part_descriptor)

        receipt_path = spec.effective_finalization_receipt_path
        receipt_descriptor = _open_existing_regular(spec, receipt_path)
        if receipt_descriptor is not None:
            os.close(receipt_descriptor)
            raise TransferRecoveryError(
                f"transfer finalization receipt already exists: {receipt_path}"
            )
        intent_path = spec.effective_finalization_intent_path
        intent = {
            "schema": TRANSFER_FINALIZATION_INTENT_V1,
            "status": "verified_part_ready_for_atomic_rename",
            "remote_path": spec.remote_path,
            "local_part_path": str(part),
            "local_final_path": str(final),
            "archive_bytes": size,
            "archive_sha256": digest,
            "tar_member_count": member_count,
            "remote_preservation_required": True,
        }
        intent["intent_sha256"] = _canonical_payload_sha256(intent)
        intent_descriptor = _open_existing_regular(spec, intent_path)
        if intent_descriptor is not None:
            os.close(intent_descriptor)
            intent = dict(_read_finalization_intent(intent_path, spec))
        else:
            try:
                _write_json_exclusive(intent_path, intent, spec=spec)
            except FileExistsError:
                intent = dict(_read_finalization_intent(intent_path, spec))

        with _secure_parent_directory(spec, final, create=True) as parent:
            if not _same_regular_name(parent, part.name, part_metadata):
                raise TransferRecoveryError(
                    "complete partial identity changed before finalization"
                )
            staging_name = f".{final.name}.{secrets.token_hex(12)}.verified"
            try:
                os.link(
                    part.name,
                    staging_name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:  # cryptographically improbable
                raise TransferRecoveryError(
                    "finalization staging identity collision"
                ) from exc
            try:
                staging_descriptor = os.open(
                    staging_name,
                    os.O_RDWR | _O_NOFOLLOW | _O_CLOEXEC,
                    dir_fd=parent,
                )
                try:
                    staging_metadata = os.fstat(staging_descriptor)
                    if not _same_inode(staging_metadata, part_metadata):
                        raise TransferRecoveryError(
                            "partial changed while pinning finalization staging"
                        )
                    os.fsync(staging_descriptor)
                finally:
                    os.close(staging_descriptor)
                try:
                    os.link(
                        staging_name,
                        final.name,
                        src_dir_fd=parent,
                        dst_dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileExistsError as exc:
                    raise TransferRecoveryError(
                        "final archive already exists; refusing overwrite"
                    ) from exc
                final_descriptor = os.open(
                    final.name,
                    os.O_RDWR | _O_NOFOLLOW | _O_CLOEXEC,
                    dir_fd=parent,
                )
                try:
                    if not _same_inode(
                        os.fstat(final_descriptor), part_metadata
                    ):
                        raise TransferRecoveryError(
                            "published final archive identity mismatch"
                        )
                    os.fsync(final_descriptor)
                finally:
                    os.close(final_descriptor)
                os.fsync(parent)
                if not _same_regular_name(parent, part.name, part_metadata):
                    raise TransferRecoveryError(
                        "partial changed before finalization unlink"
                    )
                os.unlink(part.name, dir_fd=parent)
                os.fsync(parent)
            finally:
                try:
                    os.unlink(staging_name, dir_fd=parent)
                    os.fsync(parent)
                except FileNotFoundError:
                    pass
    finally:
        if part_descriptor >= 0:
            os.close(part_descriptor)

    verified = verify_final_archive(spec)
    if (
        verified["archive_bytes"] != size
        or verified["archive_sha256"] != digest
        or verified["tar_member_count"] != member_count
    ):
        raise TransferRecoveryError("archive changed during atomic finalization")
    return _write_finalization_receipt(
        spec,
        member_count=member_count,
        recovered_after_rename=False,
        intent=intent,
    )


def _tar_member_count(path: Path) -> int:
    with tarfile.open(path, "r:gz") as archive:
        return sum(1 for _ in archive)


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_finalization_intent(
    path: Path, spec: ArchiveTransferSpec
) -> Mapping[str, Any]:
    payload = _read_json_mapping(
        path, label="finalization intent", spec=spec
    )
    unsigned = dict(payload)
    declared = unsigned.pop("intent_sha256", None)
    expected_keys = {
        "schema",
        "status",
        "remote_path",
        "local_part_path",
        "local_final_path",
        "archive_bytes",
        "archive_sha256",
        "tar_member_count",
        "remote_preservation_required",
        "intent_sha256",
    }
    try:
        matches = (
            set(payload) == expected_keys
            and payload.get("schema") == TRANSFER_FINALIZATION_INTENT_V1
            and payload.get("status") == "verified_part_ready_for_atomic_rename"
            and payload.get("remote_path") == spec.remote_path
            and payload.get("local_part_path") == str(spec.part_path)
            and payload.get("local_final_path") == str(spec.local_final_path)
            and int(payload.get("archive_bytes", -1)) == spec.expected_bytes
            and payload.get("archive_sha256") == spec.expected_sha256
            and int(payload.get("tar_member_count", -1))
            == spec.expected_member_count
            and payload.get("remote_preservation_required") is True
            and declared == _canonical_payload_sha256(unsigned)
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise TransferRecoveryError("finalization intent binding mismatch")
    return payload


def _read_finalization_receipt(
    path: Path, spec: ArchiveTransferSpec
) -> Mapping[str, Any]:
    payload = _read_json_mapping(
        path, label="finalization receipt", spec=spec
    )
    unsigned = dict(payload)
    declared = unsigned.pop("receipt_sha256", None)
    intent_path = spec.effective_finalization_intent_path
    try:
        intent = _read_finalization_intent(intent_path, spec)
    except TransferRecoveryError as exc:
        raise TransferRecoveryError(
            "finalization receipt has no immutable intent"
        ) from exc
    expected_keys = {
        "schema",
        "status",
        "recorded_at",
        "remote_path",
        "local_final_path",
        "archive_bytes",
        "archive_sha256",
        "gzip_integrity",
        "tar_integrity",
        "tar_member_count",
        "remote_preservation_required",
        "recovered_after_atomic_rename",
        "finalization_intent",
        "receipt_sha256",
    }
    try:
        matches = (
            set(payload) == expected_keys
            and payload.get("schema") == TRANSFER_FINALIZATION_RECEIPT_V1
            and payload.get("status") == "verified_and_finalized"
            and isinstance(payload.get("recorded_at"), str)
            and bool(payload.get("recorded_at"))
            and payload.get("remote_path") == spec.remote_path
            and payload.get("local_final_path") == str(spec.local_final_path)
            and int(payload.get("archive_bytes", -1)) == spec.expected_bytes
            and payload.get("archive_sha256") == spec.expected_sha256
            and payload.get("gzip_integrity") is True
            and payload.get("tar_integrity") is True
            and int(payload.get("tar_member_count", -1))
            == spec.expected_member_count
            and payload.get("remote_preservation_required") is True
            and isinstance(payload.get("recovered_after_atomic_rename"), bool)
            and payload.get("finalization_intent")
            == {
                "path": str(intent_path),
                "sha256": str(intent["intent_sha256"]),
            }
            and declared == _canonical_payload_sha256(unsigned)
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise TransferRecoveryError("finalization receipt binding mismatch")
    return payload


def read_finalization_receipt(spec: ArchiveTransferSpec) -> Mapping[str, Any]:
    """Reload and authenticate the canonical finalization receipt from disk."""

    return _read_finalization_receipt(
        spec.effective_finalization_receipt_path, spec
    )


def _read_json_mapping(
    path: Path,
    *,
    label: str,
    spec: ArchiveTransferSpec | None = None,
) -> Mapping[str, Any]:
    try:
        if spec is None:
            raw = path.read_bytes()
        else:
            descriptor = _open_existing_regular(spec, path)
            if descriptor is None:
                raise TransferRecoveryError(f"{label} is missing: {path}")
            try:
                raw = _read_fd_bytes(descriptor)
            finally:
                os.close(descriptor)
        payload = json.loads(raw.decode("utf-8"))
    except TransferRecoveryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransferRecoveryError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TransferRecoveryError(f"{label} must be a JSON object")
    return payload


def _write_finalization_receipt(
    spec: ArchiveTransferSpec,
    *,
    member_count: int,
    recovered_after_rename: bool,
    intent: Mapping[str, Any],
) -> Mapping[str, Any]:
    verified = verify_final_archive(spec)
    final = spec.local_final_path
    receipt = {
        "schema": TRANSFER_FINALIZATION_RECEIPT_V1,
        "status": "verified_and_finalized",
        "recorded_at": _utc_now(),
        "remote_path": spec.remote_path,
        "local_final_path": str(final),
        "archive_bytes": verified["archive_bytes"],
        "archive_sha256": verified["archive_sha256"],
        "gzip_integrity": True,
        "tar_integrity": True,
        "tar_member_count": member_count,
        "remote_preservation_required": True,
        "recovered_after_atomic_rename": recovered_after_rename,
        "finalization_intent": {
            "path": str(spec.effective_finalization_intent_path),
            "sha256": str(intent["intent_sha256"]),
        },
    }
    receipt["receipt_sha256"] = _canonical_payload_sha256(receipt)
    receipt_path = spec.effective_finalization_receipt_path
    try:
        _write_json_exclusive(receipt_path, receipt, spec=spec)
        return receipt
    except FileExistsError:
        return _read_finalization_receipt(receipt_path, spec)


def verify_final_archive(spec: ArchiveTransferSpec) -> Mapping[str, Any]:
    """Verify the registered final archive through one pinned no-follow FD."""

    descriptor = _open_existing_regular(spec, spec.local_final_path)
    if descriptor is None:
        raise TransferRecoveryError(
            f"final archive is missing: {spec.local_final_path}"
        )
    try:
        metadata = os.fstat(descriptor)
        size = int(metadata.st_size)
        digest = _sha256_fd(descriptor)
        if size != spec.expected_bytes or digest != spec.expected_sha256:
            raise TransferRecoveryError("final archive binding mismatch")
        try:
            member_count = _archive_member_count_fd(descriptor)
        except (OSError, EOFError, tarfile.TarError, zlib.error) as exc:
            raise TransferRecoveryError(
                f"final archive integrity check failed: {exc}"
            ) from exc
        if member_count != spec.expected_member_count:
            raise TransferRecoveryError(
                "final archive member-count binding mismatch"
            )
        with _secure_parent_directory(
            spec, spec.local_final_path, create=False
        ) as parent:
            if not _same_regular_name(
                parent, spec.local_final_path.name, metadata
            ):
                raise TransferRecoveryError(
                    "final archive identity changed during verification"
                )
        return {
            "archive_bytes": size,
            "archive_sha256": digest,
            "tar_member_count": member_count,
            "gzip_integrity": True,
            "tar_integrity": True,
        }
    finally:
        os.close(descriptor)


def verified_file_sha256(
    spec: ArchiveTransferSpec, path: Path
) -> str:
    """Hash one registered local artifact through a pinned no-follow FD."""

    descriptor = _open_existing_regular(spec, path)
    if descriptor is None:
        raise TransferRecoveryError(f"verified file is missing: {path}")
    try:
        metadata = os.fstat(descriptor)
        digest = _sha256_fd(descriptor)
        with _secure_parent_directory(spec, path, create=False) as parent:
            if not _same_regular_name(parent, path.name, metadata):
                raise TransferRecoveryError(
                    "verified file identity changed while hashing"
                )
        return digest
    finally:
        os.close(descriptor)


def append_transfer_chunk(
    spec: ArchiveTransferSpec,
    *,
    expected_offset: int,
    chunk: bytes,
    chunk_sha256: str,
) -> int:
    """Append one binary-safe chunk at an exact offset and fsync it."""

    if not isinstance(chunk, bytes) or not chunk:
        raise TransferRecoveryError("transfer chunks must be nonempty bytes")
    if hashlib.sha256(chunk).hexdigest() != _require_sha256(
        chunk_sha256, label="chunk digest"
    ):
        raise TransferRecoveryError("transfer chunk digest mismatch")
    part = spec.part_path
    with _secure_parent_directory(spec, part, create=True) as parent:
        created = False
        flags = os.O_WRONLY | os.O_APPEND | _O_NOFOLLOW | _O_CLOEXEC
        try:
            descriptor = os.open(part.name, flags, dir_fd=parent)
        except FileNotFoundError:
            try:
                descriptor = os.open(
                    part.name,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=parent,
                )
                created = True
            except OSError as exc:
                raise TransferRecoveryError(
                    f"partial archive nofollow creation failed: {exc}"
                ) from exc
        except OSError as exc:
            raise TransferRecoveryError(
                f"partial archive is not a nofollow regular file: {exc}"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise TransferRecoveryError(
                    "partial archive is not a regular file"
                )
            actual_offset = int(metadata.st_size)
            if actual_offset != int(expected_offset):
                raise TransferRecoveryError(
                    "transfer offset mismatch: "
                    f"expected {expected_offset}, got {actual_offset}"
                )
            if actual_offset + len(chunk) > spec.expected_bytes:
                raise TransferRecoveryError(
                    "transfer chunk exceeds registered archive size"
                )
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise TransferRecoveryError(
                        "binary chunk append made no progress"
                    )
                view = view[written:]
            os.fsync(descriptor)
            final_metadata = os.fstat(descriptor)
            final_offset = int(final_metadata.st_size)
            if (
                final_offset != actual_offset + len(chunk)
                or not _same_regular_name(parent, part.name, metadata)
            ):
                raise TransferRecoveryError(
                    "binary chunk append identity or length mismatch"
                )
            if created:
                os.fsync(parent)
            return final_offset
        finally:
            os.close(descriptor)


def enforce_local_storage_floor(
    spec: ArchiveTransferSpec,
    *,
    required_write_bytes: int,
    extraction_reserve_bytes: int = 0,
    free_floor_bytes: int = DEFAULT_FREE_FLOOR_BYTES,
) -> None:
    """Fail before a write/extraction would cross the registered free floor."""

    if (
        int(required_write_bytes) < 0
        or int(extraction_reserve_bytes) < 0
        or int(free_floor_bytes) < 0
    ):
        raise ValueError("storage guards must be nonnegative")
    reserve = max(
        int(extraction_reserve_bytes), int(spec.extraction_reserve_bytes)
    )
    probe = spec.local_final_path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    if free - int(required_write_bytes) - reserve < int(free_floor_bytes):
        raise TransferRecoveryError("local free-space floor would be violated")


def stream_archive_transfer(
    spec: ArchiveTransferSpec,
    *,
    remote_prefix_sha256: Callable[[int], str],
    remote_chunk: Callable[[int, int], bytes],
    chunk_bytes: int = 1 << 20,
    free_floor_bytes: int = DEFAULT_FREE_FLOOR_BYTES,
    extraction_reserve_bytes: int = 0,
) -> Mapping[str, Any]:
    """Resume, stream, and finalize one immutable archive without remote deletion."""

    if chunk_bytes <= 0 or chunk_bytes > 8 * (1 << 20):
        raise ValueError("transfer chunk size must be in (0, 8 MiB]")
    if free_floor_bytes < 0 or extraction_reserve_bytes < 0:
        raise ValueError("storage guards must be nonnegative")
    extraction_reserve_bytes = max(
        int(extraction_reserve_bytes), int(spec.extraction_reserve_bytes)
    )
    decision = prepare_partial_transfer(
        spec, remote_prefix_sha256=remote_prefix_sha256
    )
    if decision.status == "final_present_requires_recovery":
        return finalize_archive_transfer(spec)
    offset = decision.resume_offset

    enforce_local_storage_floor(
        spec,
        required_write_bytes=spec.expected_bytes - offset,
        extraction_reserve_bytes=extraction_reserve_bytes,
        free_floor_bytes=free_floor_bytes,
    )
    while offset < spec.expected_bytes:
        length = min(int(chunk_bytes), spec.expected_bytes - offset)
        enforce_local_storage_floor(
            spec,
            required_write_bytes=length,
            extraction_reserve_bytes=extraction_reserve_bytes,
            free_floor_bytes=free_floor_bytes,
        )
        chunk = remote_chunk(offset, length)
        if len(chunk) != length:
            raise TransferRecoveryError(
                f"remote chunk length mismatch at offset {offset}"
            )
        offset = append_transfer_chunk(
            spec,
            expected_offset=offset,
            chunk=chunk,
            chunk_sha256=hashlib.sha256(chunk).hexdigest(),
        )
    return finalize_archive_transfer(spec)


__all__ = [
    "ArchiveTransferSpec",
    "DEFAULT_FREE_FLOOR_BYTES",
    "TRANSFER_FINALIZATION_RECEIPT_V1",
    "TRANSFER_FINALIZATION_INTENT_V1",
    "TRANSFER_QUARANTINE_INTENT_V1",
    "TRANSFER_RECOVERY_RECEIPT_V1",
    "TransferRecoveryDecision",
    "TransferRecoveryError",
    "append_transfer_chunk",
    "enforce_local_storage_floor",
    "finalize_archive_transfer",
    "prepare_partial_transfer",
    "read_finalization_receipt",
    "stream_archive_transfer",
    "verify_final_archive",
    "verified_file_sha256",
]
