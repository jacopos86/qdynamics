#!/usr/bin/env python3
"""Build one authenticated deterministic RA-ADAPT continuation attempt archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any


ATTEMPT_SCHEMA = "paper_i_ra_adapt_ss_singleton_plateau_r70_worker_attempt_v1"


class AttemptArchiveError(ValueError):
    """Raised when a worker attempt cannot be archived safely."""


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
        while block := stream.read(1024 * 1024):
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


def _worker_files(root: Path) -> list[Path]:
    if root.as_posix() != "worker_outputs" or not root.is_dir() or root.is_symlink():
        raise AttemptArchiveError("Worker root identity drifted.")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AttemptArchiveError(f"Worker symlink is forbidden: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AttemptArchiveError(f"Unsafe worker member: {path}")
        files.append(path)
    return files


def build_archive(args: argparse.Namespace) -> dict[str, Any]:
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

    worker_files = _worker_files(args.worker_root)
    bindings = [
        {
            "path": _safe_relative(path, base=args.worker_root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in worker_files
    ]
    receipt: dict[str, Any] = {
        "schema": ATTEMPT_SCHEMA,
        "execution_id": args.execution_id,
        "cluster_id": args.cluster_id,
        "proc_id": args.proc_id,
        "attempt_ordinal": args.attempt_ordinal,
        "worker_exit_status": args.worker_exit_status,
        "job_file_sha256": sha256_file(args.job),
        "authorization_file_sha256": sha256_file(args.authorization),
        "activation_manifest_file_sha256": sha256_file(args.activation_manifest),
        "source_archive_sha256": args.source_archive_sha256,
        "resume_archive_sha256": args.resume_archive_sha256,
        "image_sha256": args.image_sha256,
        "worker_files": bindings,
    }
    receipt["sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"

    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT) as archive:
                    for path, binding in zip(worker_files, bindings, strict=True):
                        with path.open("rb") as stream:
                            archive.addfile(
                                _tar_info(
                                    name=(PurePosixPath("worker_outputs") / binding["path"]).as_posix(),
                                    size=path.stat().st_size,
                                    mode=0o755 if path.stat().st_mode & 0o111 else 0o644,
                                ),
                                stream,
                            )
                    for name, path in external:
                        with path.open("rb") as stream:
                            archive.addfile(_tar_info(name=f"authority/{name}", size=path.stat().st_size), stream)
                    archive.addfile(
                        _tar_info(name="worker_attempt_receipt.json", size=len(receipt_bytes)),
                        fileobj=io.BytesIO(receipt_bytes),
                    )
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, output)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "status": "passed",
        "output_archive": output.as_posix(),
        "output_archive_sha256": sha256_file(output),
        "output_archive_size_bytes": output.stat().st_size,
        "worker_attempt_receipt_sha256": receipt["sha256"],
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
