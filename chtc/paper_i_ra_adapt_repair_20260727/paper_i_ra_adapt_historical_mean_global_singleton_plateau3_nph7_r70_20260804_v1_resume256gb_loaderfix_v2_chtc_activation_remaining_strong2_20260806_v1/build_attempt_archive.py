#!/usr/bin/env python3
"""Build an authenticated failure-safe r70 continuation attempt archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any


ATTEMPT_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_r70_"
    "worker_attempt_v1"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EXECUTION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class AttemptArchiveError(ValueError):
    """Raised when a failure-safe attempt archive cannot be built."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tar_info(*, name: str, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _safe_relative(path: Path, *, base: Path) -> PurePosixPath:
    relative = PurePosixPath(path.relative_to(base).as_posix())
    if (
        not relative.parts
        or "." in relative.parts
        or ".." in relative.parts
        or any(not part for part in relative.parts)
    ):
        raise AttemptArchiveError(f"Unsafe worker member: {relative}")
    return relative


def _worker_files(
    root: Path, *, execution_id: str
) -> tuple[list[Path], list[Path]]:
    if root.as_posix() != "worker_outputs" or not root.is_dir() or root.is_symlink():
        raise AttemptArchiveError("Worker root identity drifted.")
    imported_resume = root / f".{execution_id}.resume_work_v1" / "resume_input"
    files: list[Path] = []
    excluded: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AttemptArchiveError(f"Worker symlink is forbidden: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AttemptArchiveError(f"Unsafe worker member: {path}")
        try:
            path.relative_to(imported_resume)
        except ValueError:
            files.append(path)
        else:
            excluded.append(path)
    return files, excluded


def _binding(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": _safe_relative(path, base=root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _checkpoint_triplets(
    root: Path, *, worker_files: list[Path]
) -> list[dict[str, Any]]:
    triplets: list[dict[str, Any]] = []
    available = set(worker_files)
    for checkpoint in sorted(
        path for path in worker_files if path.name == "checkpoint.json"
    ):
        directory = checkpoint.parent
        ledgers = sorted(
            path
            for path in available
            if path.parent == directory
            and path.match(
                "checkpoint.estimator_call_ledger_checkpoint.*.json"
            )
        )
        sidecars = sorted(
            path
            for path in available
            if path.parent == directory
            and path.match("checkpoint.verified_singleton_resume.*.json")
        )
        if len(ledgers) != 1 or len(sidecars) != 1:
            raise AttemptArchiveError(
                f"Checkpoint sidecar closure is incomplete: {checkpoint}"
            )
        triplets.append(
            {
                "checkpoint": _binding(checkpoint, root=root),
                "estimator_ledger_checkpoint": _binding(
                    ledgers[0], root=root
                ),
                "verified_resume_sidecar": _binding(sidecars[0], root=root),
                "pointer_closed_by_sibling_identity": True,
            }
        )
    return triplets


def _validate_args(args: argparse.Namespace) -> None:
    if not _EXECUTION_ID.fullmatch(args.execution_id):
        raise AttemptArchiveError("Execution id is unsafe.")
    if args.job.name != f"{args.execution_id}.json":
        raise AttemptArchiveError("Job path does not match execution id.")
    if args.cluster_id < 0 or args.proc_id < 0 or args.attempt_ordinal < 1:
        raise AttemptArchiveError("Attempt identity is invalid.")
    for label, value in (
        ("source archive", args.source_archive_sha256),
        ("resume archive", args.resume_archive_sha256),
        ("image", args.image_sha256),
    ):
        if not _HEX64.fullmatch(value):
            raise AttemptArchiveError(f"{label} digest is invalid.")


def build_archive(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output = args.output_archive
    if output.exists() or output.is_symlink():
        raise AttemptArchiveError("Attempt archive already exists.")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise AttemptArchiveError("Attempt archive parent is unsafe.")
    external = (
        ("job.json", args.job),
        ("execution_authorization.json", args.authorization),
        ("activation_manifest.json", args.activation_manifest),
    )
    for label, path in external:
        if not path.is_file() or path.is_symlink():
            raise AttemptArchiveError(f"Unsafe {label} input.")
    worker_files, excluded = _worker_files(
        args.worker_root, execution_id=args.execution_id
    )
    bindings = [_binding(path, root=args.worker_root) for path in worker_files]
    triplets = _checkpoint_triplets(
        args.worker_root, worker_files=worker_files
    )
    receipt: dict[str, Any] = {
        "schema": ATTEMPT_SCHEMA,
        "execution_id": args.execution_id,
        "cluster_id": args.cluster_id,
        "proc_id": args.proc_id,
        "attempt_ordinal": args.attempt_ordinal,
        "worker_exit_status": args.worker_exit_status,
        "job_file_sha256": sha256_file(args.job),
        "authorization_file_sha256": sha256_file(args.authorization),
        "activation_manifest_file_sha256": sha256_file(
            args.activation_manifest
        ),
        "source_archive_sha256": args.source_archive_sha256,
        "resume_archive_sha256": args.resume_archive_sha256,
        "image_sha256": args.image_sha256,
        "imported_resume_input_excluded": True,
        "excluded_imported_resume_file_count": len(excluded),
        "excluded_imported_resume_bytes": sum(
            path.stat().st_size for path in excluded
        ),
        "source_resume_archive_retained_separately_in_staging": True,
        "worker_files": bindings,
        "resumable_checkpoint_triplets": triplets,
        "resumable_checkpoint_triplet_count": len(triplets),
        "failure_safe_checkpoint_transfer": bool(triplets),
    }
    receipt["sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT
                ) as archive:
                    for path, binding in zip(
                        worker_files, bindings, strict=True
                    ):
                        with path.open("rb") as stream:
                            archive.addfile(
                                _tar_info(
                                    name=(
                                        PurePosixPath("worker_outputs")
                                        / binding["path"]
                                    ).as_posix(),
                                    size=path.stat().st_size,
                                    mode=(
                                        0o755
                                        if path.stat().st_mode & 0o111
                                        else 0o644
                                    ),
                                ),
                                stream,
                            )
                    for name, path in external:
                        with path.open("rb") as stream:
                            archive.addfile(
                                _tar_info(
                                    name=f"authority/{name}",
                                    size=path.stat().st_size,
                                ),
                                stream,
                            )
                    archive.addfile(
                        _tar_info(
                            name="worker_attempt_receipt.json",
                            size=len(receipt_bytes),
                        ),
                        fileobj=io.BytesIO(receipt_bytes),
                    )
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, output)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "status": "passed",
        "output_archive": output.as_posix(),
        "output_archive_sha256": sha256_file(output),
        "output_archive_size_bytes": output.stat().st_size,
        "worker_attempt_receipt_sha256": receipt["sha256"],
        "resumable_checkpoint_triplet_count": len(triplets),
        "excluded_imported_resume_file_count": len(excluded),
        "excluded_imported_resume_bytes": sum(
            path.stat().st_size for path in excluded
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--proc-id", type=int, required=True)
    parser.add_argument("--attempt-ordinal", type=int, required=True)
    parser.add_argument("--worker-exit-status", type=int, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--resume-archive-sha256", required=True)
    parser.add_argument("--image-sha256", required=True)
    args = parser.parse_args()
    print(canonical_json_bytes(build_archive(args)).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
