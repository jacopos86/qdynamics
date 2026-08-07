#!/usr/bin/env python3
"""Build one authenticated worker-attempt archive without stale ledger snapshots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import Any, Mapping, Sequence


LEDGER_SIDECAR_RE = re.compile(
    r"^checkpoint\.estimator_call_ledger_checkpoint\."
    r"(?P<prefix>[0-9a-f]{16})\.json$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TransferArchiveError(ValueError):
    """Raised when worker output cannot be compacted without evidence loss."""


def _safe_relative(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
    ):
        raise TransferArchiveError(f"Unsafe {label}: {value}")
    return path


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TransferArchiveError(f"{label} must be a mapping.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_ledger_pointer(
    checkpoint_path: Path,
) -> tuple[str, str] | None:
    try:
        raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransferArchiveError(
            "checkpoint.json cannot authenticate its ledger sidecar."
        ) from exc
    checkpoint = _mapping(raw, label="checkpoint.json")
    pointers: list[tuple[str, str]] = []
    for owner_name in ("checkpoint", "adapt_vqe"):
        owner = checkpoint.get(owner_name)
        if owner is None:
            continue
        owner_map = _mapping(owner, label=f"checkpoint.json {owner_name}")
        raw_pointer = owner_map.get("estimator_call_ledger_checkpoint")
        if raw_pointer is None:
            continue
        pointer = _mapping(
            raw_pointer,
            label=f"checkpoint.json {owner_name} ledger pointer",
        )
        pointer_path = str(pointer.get("path", ""))
        pointer_sha256 = str(pointer.get("sha256", ""))
        match = LEDGER_SIDECAR_RE.fullmatch(pointer_path)
        if (
            match is None
            or SHA256_RE.fullmatch(pointer_sha256) is None
            or match.group("prefix") != pointer_sha256[:16]
        ):
            raise TransferArchiveError(
                "checkpoint.json ledger pointer is malformed."
            )
        pointers.append((pointer_path, pointer_sha256))
    if not pointers:
        return None
    if len(set(pointers)) != 1:
        raise TransferArchiveError(
            "checkpoint.json ledger pointers disagree."
        )
    return pointers[0]


def _retained_worker_files(worker_root: Path) -> tuple[list[Path], str | None]:
    if (
        worker_root.as_posix() != "worker_outputs"
        or not worker_root.is_dir()
        or worker_root.is_symlink()
    ):
        raise TransferArchiveError(
            "Worker root must be the fixed worker_outputs directory."
        )
    files: list[Path] = []
    ledger_sidecars: dict[str, Path] = {}
    for entry in sorted(worker_root.rglob("*")):
        if entry.is_symlink():
            raise TransferArchiveError(
                f"Worker output contains a symlink: {entry}"
            )
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise TransferArchiveError(
                f"Worker output is not a regular file: {entry}"
            )
        relative = entry.relative_to(worker_root)
        if len(relative.parts) == 1 and LEDGER_SIDECAR_RE.fullmatch(
            relative.name
        ):
            ledger_sidecars[relative.name] = entry
        else:
            files.append(entry)

    checkpoint_path = worker_root / "checkpoint.json"
    pointer = (
        _checkpoint_ledger_pointer(checkpoint_path)
        if checkpoint_path.is_file() and not checkpoint_path.is_symlink()
        else None
    )
    if ledger_sidecars and pointer is None:
        raise TransferArchiveError(
            "Ledger checkpoint sidecars exist without an authenticated pointer."
        )
    if pointer is None:
        return files, None

    pointer_path, pointer_sha256 = pointer
    retained = ledger_sidecars.get(pointer_path)
    if retained is None:
        raise TransferArchiveError(
            "Checkpoint-referenced ledger sidecar is unavailable."
        )
    if _sha256_file(retained) != pointer_sha256:
        raise TransferArchiveError(
            "Checkpoint-referenced ledger sidecar SHA-256 drifted."
        )
    files.append(retained)
    files.sort()
    return files, pointer_path


def _tar_info(path: Path, *, name: str) -> tarfile.TarInfo:
    stat = path.stat()
    info = tarfile.TarInfo(name)
    info.size = stat.st_size
    info.mode = 0o755 if stat.st_mode & 0o111 else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def build_transfer_archive(
    *,
    worker_root: Path,
    job_spec: Path,
    output_archive: Path,
) -> dict[str, Any]:
    job_relative = _safe_relative(job_spec.as_posix(), label="job-spec path")
    if not job_spec.is_file() or job_spec.is_symlink():
        raise TransferArchiveError("Job spec is unavailable or unsafe.")
    if output_archive.exists() or output_archive.is_symlink():
        raise TransferArchiveError("Output archive already exists.")
    if not output_archive.parent.is_dir() or output_archive.parent.is_symlink():
        raise TransferArchiveError("Output archive parent is unavailable/unsafe.")

    retained_files, retained_ledger = _retained_worker_files(worker_root)
    temporary = output_archive.with_name(f".{output_archive.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise TransferArchiveError("Stale transfer-archive temporary exists.")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    mode="w",
                    fileobj=compressed,
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    root_info = tarfile.TarInfo("worker_outputs")
                    root_info.type = tarfile.DIRTYPE
                    root_info.mode = 0o755
                    root_info.uid = 0
                    root_info.gid = 0
                    root_info.uname = ""
                    root_info.gname = ""
                    root_info.mtime = 0
                    archive.addfile(root_info)
                    for source in retained_files:
                        relative = source.relative_to(worker_root)
                        name = (
                            PurePosixPath("worker_outputs")
                            / PurePosixPath(relative.as_posix())
                        ).as_posix()
                        with source.open("rb") as stream:
                            archive.addfile(
                                _tar_info(source, name=name),
                                stream,
                            )
                    with job_spec.open("rb") as stream:
                        archive.addfile(
                            _tar_info(
                                job_spec,
                                name=job_relative.as_posix(),
                            ),
                            stream,
                        )
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, output_archive)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "status": "passed",
        "retained_worker_file_count": len(retained_files),
        "retained_ledger_sidecar": retained_ledger,
        "output_archive": output_archive.as_posix(),
        "output_archive_sha256": _sha256_file(output_archive),
        "output_archive_size_bytes": output_archive.stat().st_size,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_transfer_archive(
            worker_root=args.worker_root,
            job_spec=args.job_spec,
            output_archive=args.output_archive,
        )
    except (OSError, TransferArchiveError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
