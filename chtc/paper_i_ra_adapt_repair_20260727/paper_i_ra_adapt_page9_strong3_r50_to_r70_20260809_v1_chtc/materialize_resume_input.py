#!/usr/bin/env python3
"""Stream one authenticated Page-9 attempt into a three-member resume archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from package_contract import (  # noqa: E402
    JOB_SCHEMA,
    PACKAGE_ID,
    RESUME_MATERIALIZATION_SCHEMA,
    ROUTE_CONTRACT_SHA256,
    ROUTE_PROFILE,
    SOURCE_HORIZON,
    VISIBLE_ADAPTER_SCHEMA,
    VISIBLE_PAGE_ID,
    PackageContractError,
    canonical_json_bytes,
    digested,
    file_binding,
    load_json,
    prefix_projection,
    sha256_file,
    validate_resume_archive,
    verify_self_digest,
)


class _DigestReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        block = self.source.read(size)
        if block:
            self.digest.update(block)
            self.size += len(block)
        return block


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_job(path: Path) -> dict[str, Any]:
    job = load_json(path, label="continuation job")
    verify_self_digest(job, label="continuation job")
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("source_horizon") != SOURCE_HORIZON
        or job.get("target_horizon") != 70
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("route_profile") != ROUTE_PROFILE
        or job.get("accepted_state_resume_required") is not True
        or job.get("triplet_pointer_closure_required") is not True
    ):
        raise PackageContractError("Continuation job identity drifted.")
    return job


def _triplet_from_receipt(
    receipt: Mapping[str, Any], *, execution_id: str
) -> list[dict[str, Any]]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise PackageContractError("Source worker receipt lacks artifacts.")
    rows: dict[str, dict[str, Any]] = {}
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            continue
        source = str(raw.get("path", ""))
        if not source.startswith(f"runs/{execution_id}/checkpoints/current"):
            continue
        name = Path(source).name
        if name == "current.json":
            role = "checkpoint"
        elif ".estimator_call_ledger_checkpoint." in name:
            role = "estimator_ledger_checkpoint"
        elif ".verified_singleton_resume." in name:
            role = "verified_resume_sidecar"
        else:
            continue
        if role in rows:
            raise PackageContractError(f"Duplicate resume role: {role}")
        sha = str(raw.get("sha256", ""))
        if role != "checkpoint" and f".{sha[:16]}.json" not in name:
            raise PackageContractError("Content-addressed sidecar name drifted.")
        rows[role] = {
            "role": role,
            "source_path": source,
            "archive_member": f"./{source}",
            "materialized_path": name,
            "sha256": sha,
            "size_bytes": int(raw.get("size_bytes", -1)),
        }
    if set(rows) != {
        "checkpoint",
        "estimator_ledger_checkpoint",
        "verified_resume_sidecar",
    }:
        raise PackageContractError("Source receipt does not close the triplet.")
    return [rows[role] for role in sorted(rows)]


def _complete_blocked_source(
    *,
    job: Mapping[str, Any],
    adapter_path: Path,
    receipt_path: Path,
    summary_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    adapter = load_json(adapter_path, label="completed Page-9 adapter")
    verify_self_digest(adapter, label="completed Page-9 adapter")
    if (
        adapter.get("schema") != VISIBLE_ADAPTER_SCHEMA
        or adapter.get("page_id") != VISIBLE_PAGE_ID
        or adapter.get("route_profile") != ROUTE_PROFILE
        or adapter.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
    ):
        raise PackageContractError("Completed Page-9 adapter identity drifted.")
    cells = adapter.get("cells")
    matches = [
        row
        for row in cells if isinstance(cells, list) and isinstance(row, Mapping)
        and row.get("regime_id") == job.get("regime_id")
    ] if isinstance(cells, list) else []
    if len(matches) != 1:
        raise PackageContractError("Completed adapter row is not unique.")
    route = matches[0].get("phase3_qiskit_no_lanes")
    if not isinstance(route, Mapping) or route.get("status") != "complete":
        raise PackageContractError("Blocked predecessor is not complete.")
    bindings = route.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise PackageContractError("Completed source bindings are absent.")
    remote = bindings.get("remote_full_archive")
    if not isinstance(remote, Mapping):
        raise PackageContractError("Completed remote archive binding is absent.")
    receipt = load_json(receipt_path, label="completed worker receipt")
    verify_self_digest(receipt, label="completed worker receipt")
    receipt_expected = bindings.get("worker_receipt")
    summary_expected = bindings.get("summary")
    if (
        not isinstance(receipt_expected, Mapping)
        or sha256_file(receipt_path) != receipt_expected.get("sha256")
        or receipt_path.stat().st_size
        != int(receipt_expected.get("size_bytes", -1))
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != job.get("source_execution_id")
        or receipt.get("job_spec_sha256")
        != job.get("source_job", {}).get("canonical_sha256")
        or receipt.get("controller_rounds_completed") != SOURCE_HORIZON
    ):
        raise PackageContractError("Completed worker receipt drifted.")
    if (
        not isinstance(summary_expected, Mapping)
        or sha256_file(summary_path) != summary_expected.get("sha256")
        or summary_path.stat().st_size
        != int(summary_expected.get("size_bytes", -1))
    ):
        raise PackageContractError("Completed summary bytes drifted.")
    summary = load_json(summary_path, label="completed summary")
    return (
        {
            "path": remote["preserved_location"],
            "sha256": remote["sha256"],
            "size_bytes": int(remote["size_bytes"]),
        },
        _triplet_from_receipt(
            receipt, execution_id=str(job["source_execution_id"])
        ),
        prefix_projection(summary),
        {
            "adapter": {
                "path": adapter_path.as_posix(),
                "sha256": sha256_file(adapter_path),
                "size_bytes": adapter_path.stat().st_size,
                "canonical_sha256": adapter["sha256"],
            },
            "worker_receipt": {
                "path": receipt_path.as_posix(),
                "sha256": sha256_file(receipt_path),
                "size_bytes": receipt_path.stat().st_size,
                "canonical_sha256": receipt["sha256"],
            },
            "summary": {
                "path": summary_path.as_posix(),
                "sha256": sha256_file(summary_path),
                "size_bytes": summary_path.stat().st_size,
            },
        },
    )


def _normalize_member(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name


def materialize(
    *,
    job_path: Path,
    source_archive: Path,
    output_dir: Path,
    completed_adapter: Path | None,
    source_worker_receipt: Path | None,
    source_summary: Path | None,
) -> dict[str, Any]:
    job = _load_job(job_path)
    source = job.get("resume_source")
    if not isinstance(source, Mapping):
        raise PackageContractError("Job resume source is absent.")
    completion_bindings: dict[str, Any] | None = None
    if source.get("state") == "remote_archive_preserved_materialization_pending":
        remote = dict(source["remote_full_archive"])
        triplet = [dict(row) for row in source["resume_triplet"]]
        anchor_path = PACKAGE_DIR / str(source["prefix_anchor"]["path"])
        anchor = load_json(anchor_path, label="source prefix anchor")
        verify_self_digest(anchor, label="source prefix anchor")
    elif source.get("state") == "blocked_predecessor_terminal_missing":
        if any(
            path is None
            for path in (completed_adapter, source_worker_receipt, source_summary)
        ):
            raise PackageContractError(
                "Blocked strong--strong requires its completed adapter, terminal "
                "worker receipt, and terminal summary."
            )
        assert completed_adapter is not None
        assert source_worker_receipt is not None
        assert source_summary is not None
        remote, triplet, anchor, completion_bindings = _complete_blocked_source(
            job=job,
            adapter_path=completed_adapter.resolve(),
            receipt_path=source_worker_receipt.resolve(),
            summary_path=source_summary.resolve(),
        )
    else:
        raise PackageContractError("Unknown resume-source state.")
    source_archive = source_archive.resolve()
    if (
        source_archive.as_posix() != str(remote["path"])
        or not source_archive.is_file()
        or source_archive.is_symlink()
        or source_archive.stat().st_size != int(remote["size_bytes"])
        or sha256_file(source_archive) != remote["sha256"]
    ):
        raise PackageContractError("Authenticated source archive binding drifted.")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent)
    )
    try:
        archive_path = temporary / "resume_input.tar.gz"
        by_source = {_normalize_member(str(row["archive_member"])): row for row in triplet}
        observed: set[str] = set()
        with archive_path.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with tarfile.open(mode="w", fileobj=gz, format=tarfile.PAX_FORMAT) as output:
                    with tarfile.open(source_archive, mode="r|gz") as incoming:
                        for member in incoming:
                            normalized = _normalize_member(member.name)
                            row = by_source.get(normalized)
                            if row is None:
                                continue
                            if (
                                normalized in observed
                                or not member.isfile()
                                or member.issym()
                                or member.islnk()
                                or member.size != int(row["size_bytes"])
                            ):
                                raise PackageContractError(
                                    f"Unsafe source resume member: {member.name}"
                                )
                            stream = incoming.extractfile(member)
                            if stream is None:
                                raise PackageContractError(
                                    f"Unreadable source member: {member.name}"
                                )
                            reader = _DigestReader(stream)
                            info = tarfile.TarInfo(
                                f"resume/{row['materialized_path']}"
                            )
                            info.size = member.size
                            info.mode = 0o644
                            info.uid = info.gid = 0
                            info.uname = info.gname = ""
                            info.mtime = 0
                            output.addfile(info, reader)
                            if (
                                reader.size != member.size
                                or reader.digest.hexdigest() != row["sha256"]
                            ):
                                raise PackageContractError(
                                    f"Streamed resume member drifted: {member.name}"
                                )
                            observed.add(normalized)
            raw.flush()
            os.fsync(raw.fileno())
        if observed != set(by_source):
            missing = sorted(set(by_source) - observed)
            raise PackageContractError(f"Resume source archive is incomplete: {missing}")
        members = [
            {
                "role": row["role"],
                "path": f"resume/{row['materialized_path']}",
                "sha256": row["sha256"],
                "size_bytes": int(row["size_bytes"]),
            }
            for row in triplet
        ]
        manifest = digested(
            {
                "schema": RESUME_MATERIALIZATION_SCHEMA,
                "status": "passed_pointer_closed_triplet",
                "package_id": PACKAGE_ID,
                "execution_id": job["execution_id"],
                "source_execution_id": job["source_execution_id"],
                "source_job_sha256": job["source_job"]["canonical_sha256"],
                "source_archive": dict(remote),
                "source_completion_bindings": completion_bindings,
                "resume_round": SOURCE_HORIZON,
                "member_count": 3,
                "members": members,
                "pointer_closed": True,
                "pointer_closure_validator": "ijson_exact_checkpoint_pointers_v1",
                "prefix_anchor": anchor,
                "archive": {
                    "path": "resume_input.tar.gz",
                    "sha256": sha256_file(archive_path),
                    "size_bytes": archive_path.stat().st_size,
                },
            }
        )
        validate_resume_archive(
            archive_path, manifest, expected_round=SOURCE_HORIZON
        )
        _write_json(temporary / "resume_materialization.json", manifest)
        os.rename(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--completed-adapter", type=Path)
    parser.add_argument("--source-worker-receipt", type=Path)
    parser.add_argument("--source-summary", type=Path)
    args = parser.parse_args()
    try:
        result = materialize(
            job_path=args.job.resolve(),
            source_archive=args.source_archive,
            output_dir=args.output_dir.resolve(),
            completed_adapter=args.completed_adapter,
            source_worker_receipt=args.source_worker_receipt,
            source_summary=args.source_summary,
        )
    except (OSError, ValueError, PackageContractError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
