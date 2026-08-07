#!/usr/bin/env python3
"""Validate one streamed live-checkpoint archive without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


CHUNK_SIZE = 1024 * 1024
SHA_PATTERN = re.compile(rb'"sha256"\s*:\s*"([0-9a-f]{64})"')
DEPTH_PATTERN = re.compile(rb'"history_count"\s*:\s*(\d+)')
POINTER_PATTERNS = {
    "ledger": re.compile(
        rb'"path"\s*:\s*"(checkpoint\.estimator_call_ledger_checkpoint\.'
        rb'[0-9a-f]{16}\.json)"'
    ),
    "resume": re.compile(
        rb'"path"\s*:\s*"(checkpoint\.verified_singleton_resume\.'
        rb'[0-9a-f]{16}\.json)"'
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_checkpoint(stream) -> tuple[str, dict[str, dict[str, str]], int]:
    digest = hashlib.sha256()
    tail = b""
    pointers: dict[str, dict[str, str]] = {}
    pending: dict[str, bytearray] = {}
    depth = -1
    first_nonspace: int | None = None
    last_nonspace: int | None = None

    while True:
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        stripped = chunk.lstrip()
        if first_nonspace is None and stripped:
            first_nonspace = stripped[0]
        trailing = chunk.rstrip()
        if trailing:
            last_nonspace = trailing[-1]

        window = tail + chunk
        for match in DEPTH_PATTERN.finditer(window):
            depth = max(depth, int(match.group(1)))
        for kind, pattern in POINTER_PATTERNS.items():
            if kind not in pointers and kind not in pending:
                match = pattern.search(window)
                if match:
                    pointers[kind] = {"path": match.group(1).decode("ascii")}
                    pending[kind] = bytearray(window[match.end() :])
        for kind in tuple(pending):
            match = SHA_PATTERN.search(pending[kind])
            if match:
                pointers[kind]["sha256"] = match.group(1).decode("ascii")
                del pending[kind]
            else:
                pending[kind].extend(chunk)
                if len(pending[kind]) > CHUNK_SIZE:
                    raise RuntimeError(f"No SHA-256 found near {kind} pointer")
        tail = window[-4096:]

    if first_nonspace != ord("{") or last_nonspace != ord("}"):
        raise RuntimeError("checkpoint.json is not a complete JSON object")
    for kind in POINTER_PATTERNS:
        if kind not in pointers or "sha256" not in pointers[kind]:
            raise RuntimeError(f"Missing complete {kind} pointer")
    if depth < 0:
        raise RuntimeError("Missing checkpoint history_count")
    return digest.hexdigest(), pointers, depth


def validate(path: Path) -> dict[str, object]:
    members: dict[str, dict[str, object]] = {}
    pointers: dict[str, dict[str, str]] | None = None
    depth: int | None = None

    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                raise RuntimeError(f"Unexpected non-file archive member: {member.name}")
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Cannot read archive member: {member.name}")
            if member.name == "checkpoint.json":
                member_sha, pointers, depth = _scan_checkpoint(stream)
            else:
                digest = hashlib.sha256()
                first_nonspace: int | None = None
                last_nonspace: int | None = None
                while True:
                    chunk = stream.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)
                    stripped = chunk.lstrip()
                    if first_nonspace is None and stripped:
                        first_nonspace = stripped[0]
                    trailing = chunk.rstrip()
                    if trailing:
                        last_nonspace = trailing[-1]
                if first_nonspace != ord("{") or last_nonspace != ord("}"):
                    raise RuntimeError(f"Incomplete JSON sidecar: {member.name}")
                member_sha = digest.hexdigest()
            members[member.name] = {
                "size_bytes": member.size,
                "sha256": member_sha,
            }

    if set(members) != {
        "checkpoint.json",
        pointers["ledger"]["path"] if pointers else "",
        pointers["resume"]["path"] if pointers else "",
    }:
        raise RuntimeError("Archive members do not exactly match checkpoint pointers")
    assert pointers is not None
    for kind in ("ledger", "resume"):
        name = pointers[kind]["path"]
        actual_sha = str(members[name]["sha256"])
        if actual_sha != pointers[kind]["sha256"]:
            raise RuntimeError(f"{kind} sidecar SHA-256 does not match pointer")
        filename_prefix = name.rsplit(".", 2)[-2]
        if not actual_sha.startswith(filename_prefix):
            raise RuntimeError(f"{kind} sidecar filename digest prefix is invalid")

    final_archive_path = path.with_suffix("") if path.suffix == ".part" else path
    return {
        "schema": "paper_i_live_checkpoint_snapshot_validation_v1",
        "archive": str(final_archive_path),
        "archive_size_bytes": path.stat().st_size,
        "archive_sha256": _sha256_file(path),
        "checkpoint_depth": depth,
        "pointers": pointers,
        "members": members,
        "validation": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.archive), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
