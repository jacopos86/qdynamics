#!/usr/bin/env python3
"""Build one authenticated deterministic Phase-III-on-plateau attempt archive."""

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
    "paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_1em6_"
    "worker_attempt_v2"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EXECUTION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_ESTIMATOR_CHECKPOINT_SIDECAR = re.compile(
    r"^checkpoint\.estimator_call_ledger_checkpoint\.([0-9a-f]{16})\.json$"
)
_SINGLETON_RESUME_SIDECAR = re.compile(
    r"^checkpoint\.verified_singleton_resume\.([0-9a-f]{16})\.json$"
)


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


def _verify_self_digest(payload: Any, *, label: str) -> str:
    if not isinstance(payload, dict):
        raise AttemptArchiveError(f"{label} must be an object.")
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    observed = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if payload.get("sha256") != observed:
        raise AttemptArchiveError(f"{label} self-digest drifted.")
    return observed


def _load_self_digested_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptArchiveError(f"{label} is unreadable.") from exc
    _verify_self_digest(payload, label=label)
    return payload


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


def _validate_args(args: argparse.Namespace) -> None:
    if not _EXECUTION_ID.fullmatch(args.execution_id):
        raise AttemptArchiveError("Execution id is unsafe.")
    if args.job.name != f"{args.execution_id}.json":
        raise AttemptArchiveError("Job path does not match execution id.")
    if args.cluster_id < 0 or args.proc_id < 0 or args.attempt_ordinal < 1:
        raise AttemptArchiveError("Attempt identity is invalid.")
    if not _HEX64.fullmatch(args.source_archive_sha256):
        raise AttemptArchiveError("Source archive digest is invalid.")
    if not _HEX64.fullmatch(args.image_sha256):
        raise AttemptArchiveError("Image digest is invalid.")


def _validate_success_artifacts(
    *, worker_root: Path, worker_files: list[Path], args: argparse.Namespace
) -> None:
    required = {
        "artifacts/checkpoint.json",
        "artifacts/estimator_ledger.json",
        "artifacts/execution_manifest.json",
        "artifacts/paper_i_summary.json",
        "artifacts/result.json",
        "worker_receipt.json",
    }
    observed = {
        _safe_relative(path, base=worker_root).as_posix()
        for path in worker_files
    }
    missing = sorted(required - observed)
    if missing:
        raise AttemptArchiveError(
            f"Successful worker artifact closure is incomplete: {missing}"
        )
    artifact_files = {
        relative.removeprefix("artifacts/"): path
        for path in worker_files
        if (relative := _safe_relative(path, base=worker_root).as_posix()).startswith(
            "artifacts/"
        )
    }
    required_names = {
        "checkpoint.json",
        "estimator_ledger.json",
        "execution_manifest.json",
        "paper_i_summary.json",
        "result.json",
    }
    dynamic = set(artifact_files) - required_names
    estimator_names = {
        name for name in dynamic if _ESTIMATOR_CHECKPOINT_SIDECAR.fullmatch(name)
    }
    resume_names = {
        name for name in dynamic if _SINGLETON_RESUME_SIDECAR.fullmatch(name)
    }
    if (
        dynamic != estimator_names | resume_names
        or len(estimator_names) != 1
        or len(resume_names) != 1
    ):
        raise AttemptArchiveError(
            "Successful worker checkpoint sidecar closure drifted."
        )

    try:
        checkpoint = json.loads(
            artifact_files["checkpoint.json"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptArchiveError("Successful checkpoint is unreadable.") from exc
    if not isinstance(checkpoint, dict):
        raise AttemptArchiveError("Successful checkpoint must be an object.")
    pointer_rows: set[tuple[str, str]] = set()
    stack: list[Any] = [checkpoint]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            path = current.get("path")
            digest = current.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                pointer_rows.add((path, digest))
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    expected_schemas = {
        **{
            name: "paper_i_estimator_call_ledger_checkpoint_sidecar_v2"
            for name in estimator_names
        },
        **{
            name: "static_adapt_signed_active_prefix_resume_sidecar_v2"
            for name in resume_names
        },
    }
    for name, schema in expected_schemas.items():
        path = artifact_files[name]
        digest = sha256_file(path)
        match = (
            _ESTIMATOR_CHECKPOINT_SIDECAR.fullmatch(name)
            or _SINGLETON_RESUME_SIDECAR.fullmatch(name)
        )
        assert match is not None
        if match.group(1) != digest[:16] or (name, digest) not in pointer_rows:
            raise AttemptArchiveError(
                f"Successful checkpoint sidecar binding drifted: {name}"
            )
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AttemptArchiveError(
                f"Successful checkpoint sidecar is unreadable: {name}"
            ) from exc
        if not isinstance(sidecar, dict) or sidecar.get("schema") != schema:
            raise AttemptArchiveError(
                f"Successful checkpoint sidecar schema drifted: {name}"
            )
    dynamic_pointer_rows = {
        (path, digest)
        for path, digest in pointer_rows
        if _ESTIMATOR_CHECKPOINT_SIDECAR.fullmatch(path)
        or _SINGLETON_RESUME_SIDECAR.fullmatch(path)
    }
    expected_pointer_rows = {
        (name, sha256_file(artifact_files[name])) for name in dynamic
    }
    if dynamic_pointer_rows != expected_pointer_rows:
        raise AttemptArchiveError(
            "Successful checkpoint sidecar pointer closure drifted."
        )

    try:
        execution_manifest = json.loads(
            artifact_files["execution_manifest.json"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptArchiveError(
            "Successful execution manifest is unreadable."
        ) from exc
    expected_payloads = {
        name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(artifact_files.items())
        if name != "execution_manifest.json"
    }
    if (
        not isinstance(execution_manifest, dict)
        or execution_manifest.get("schema")
        != (
            "paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_sweep_"
            "weak_weak_r50_execution_manifest_v1"
        )
        or execution_manifest.get("status") != "passed"
        or execution_manifest.get("worker_owned_live_progress") is not True
        or execution_manifest.get("same_filesystem_atomic_success_publication")
        is not True
        or execution_manifest.get("output_payloads") != expected_payloads
    ):
        raise AttemptArchiveError(
            "Successful execution manifest payload closure drifted."
        )
    _verify_self_digest(
        execution_manifest,
        label="Successful execution manifest",
    )

    job = _load_self_digested_json(args.job, label="Job authority")
    authorization = _load_self_digested_json(
        args.authorization,
        label="Execution authorization",
    )
    activation = _load_self_digested_json(
        args.activation_manifest,
        label="Activation manifest",
    )
    execution_rows = [
        row
        for row in activation.get("executions", [])
        if isinstance(row, dict)
        and row.get("execution_id") == args.execution_id
    ]
    authorization_rows = [
        row
        for row in activation.get("execution_authorizations", [])
        if isinstance(row, dict)
        and row.get("execution_id") == args.execution_id
    ]
    sealed_package = activation.get("sealed_package")
    sealed_manifest = (
        sealed_package.get("manifest")
        if isinstance(sealed_package, dict)
        else None
    )
    sealed_archive = (
        sealed_package.get("source_archive")
        if isinstance(sealed_package, dict)
        else None
    )
    remote_image = activation.get("remote_image")
    execution_job_binding = (
        execution_rows[0].get("job")
        if len(execution_rows) == 1
        else None
    )
    if (
        job.get("execution_id") != args.execution_id
        or authorization.get("execution_id") != args.execution_id
        or activation.get("package_id") != job.get("package_id")
        or authorization.get("package_id") != job.get("package_id")
        or activation.get("campaign_id") != job.get("campaign_id")
        or authorization.get("campaign_id") != job.get("campaign_id")
        or authorization.get("job_spec_sha256") != job.get("sha256")
        or authorization.get("protocol_sha256") != job.get("protocol_sha256")
        or authorization.get("protocol_file_sha256")
        != job.get("protocol_file_sha256")
        or not isinstance(sealed_manifest, dict)
        or sealed_manifest.get("canonical_sha256")
        != authorization.get("package_manifest_sha256")
        or not isinstance(sealed_archive, dict)
        or sealed_archive.get("sha256")
        != authorization.get("source_archive_sha256")
        or sealed_archive.get("sha256") != args.source_archive_sha256
        or not isinstance(remote_image, dict)
        or remote_image.get("sha256")
        != authorization.get("remote_image_sha256")
        or remote_image.get("sha256") != args.image_sha256
        or len(execution_rows) != 1
        or not isinstance(execution_job_binding, dict)
        or execution_job_binding.get("canonical_sha256") != job.get("sha256")
        or len(authorization_rows) != 1
        or authorization_rows[0].get("canonical_sha256")
        != authorization.get("sha256")
    ):
        raise AttemptArchiveError(
            "Successful job/authorization/activation authority closure drifted."
        )
    if (
        execution_manifest.get("package_id") != job.get("package_id")
        or execution_manifest.get("campaign_id") != job.get("campaign_id")
        or execution_manifest.get("execution_id") != args.execution_id
        or execution_manifest.get("job_spec_sha256") != job.get("sha256")
        or execution_manifest.get("authorization_sha256")
        != authorization.get("sha256")
        or execution_manifest.get("protocol_sha256")
        != job.get("protocol_sha256")
        or execution_manifest.get("target_horizon")
        != job.get("target_horizon")
        or execution_manifest.get("controller_rounds_completed")
        != job.get("target_horizon")
        or execution_manifest.get("fresh_start") is not True
        or execution_manifest.get("source_checkpoint_consumed") is not False
    ):
        raise AttemptArchiveError(
            "Successful execution manifest authority closure drifted."
        )

    try:
        worker_receipt = json.loads(
            (worker_root / "worker_receipt.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptArchiveError(
            "Successful worker receipt is unreadable."
        ) from exc
    _verify_self_digest(worker_receipt, label="Successful worker receipt")
    expected_artifacts = [
        {
            "path": name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(artifact_files.items())
    ]
    if (
        worker_receipt.get("schema")
        != (
            "paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_sweep_"
            "weak_weak_r50_worker_receipt_v1"
        )
        or worker_receipt.get("status") != "passed"
        or worker_receipt.get("package_id") != job.get("package_id")
        or worker_receipt.get("campaign_id") != job.get("campaign_id")
        or worker_receipt.get("execution_id") != args.execution_id
        or worker_receipt.get("job_spec_sha256") != job.get("sha256")
        or worker_receipt.get("authorization_sha256")
        != authorization.get("sha256")
        or worker_receipt.get("execution_manifest_sha256")
        != execution_manifest["sha256"]
        or worker_receipt.get("controller_rounds_completed")
        != job.get("target_horizon")
        or worker_receipt.get("fresh_start") is not True
        or worker_receipt.get("artifacts") != expected_artifacts
    ):
        raise AttemptArchiveError(
            "Successful worker receipt artifact closure drifted."
        )
    try:
        ledger = json.loads(
            (worker_root / "artifacts/estimator_ledger.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptArchiveError(
            "Successful estimator ledger is unreadable."
        ) from exc
    if not isinstance(ledger, dict):
        raise AttemptArchiveError(
            "Successful estimator ledger must be an object."
        )
    accounting = ledger.get("accounting")
    if not isinstance(accounting, dict):
        raise AttemptArchiveError(
            "Successful estimator accounting is absent."
        )
    components = accounting.get("components")
    s_alg = accounting.get("S_alg")
    if (
        ledger.get("schema") != "paper_i_estimator_call_ledger_sidecar_v2"
        or ledger.get("adapt_success") is not True
        or ledger.get("adapt_error") is not None
        or accounting.get("complete") is not True
        or accounting.get("exact_blockers") != []
        or not isinstance(components, dict)
        or set(components) != {"N_H_outer", "N_H_refit", "N_grad", "N_metric"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in components.values()
        )
        or isinstance(s_alg, bool)
        or not isinstance(s_alg, int)
        or s_alg != sum(components.values())
    ):
        raise AttemptArchiveError(
            "Successful estimator ledger is not closed."
        )


def _failure_evidence_state(
    *, worker_root: Path, worker_files: list[Path]
) -> str:
    relatives = {
        _safe_relative(path, base=worker_root).as_posix()
        for path in worker_files
    }
    if any(path.startswith("artifacts.in_progress/") for path in relatives):
        return "in_progress_science_preserved_unvalidated_v2"
    if any(path.startswith("artifacts/") for path in relatives):
        return "published_science_preserved_after_late_failure_v2"
    return "failed_before_science_payload_publication_v2"


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

    worker_files = _worker_files(args.worker_root)
    if args.worker_exit_status == 0:
        _validate_success_artifacts(
            worker_root=args.worker_root,
            worker_files=worker_files,
            args=args,
        )
        science_evidence_state = "success_payload_closed_v2"
    else:
        science_evidence_state = _failure_evidence_state(
            worker_root=args.worker_root,
            worker_files=worker_files,
        )
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
        "image_sha256": args.image_sha256,
        "science_evidence_state": science_evidence_state,
        "worker_files": bindings,
    }
    receipt["sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    if len(worker_files) != len(bindings):
        raise AttemptArchiveError("Worker file binding cardinality drifted.")

    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(
                    mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT
                ) as archive:
                    for path, binding in zip(worker_files, bindings):
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
    parser.add_argument("--image-sha256", required=True)
    args = parser.parse_args()
    result = build_archive(args)
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
