#!/usr/bin/env python3
"""Authorize and run the sealed v3 macro-only campaign on the local host.

The sealed package is intentionally left untouched.  This adapter changes only
the execution target: it validates the exact v3 package/source/protocol bytes,
mints a local-only authority overlay, and delegates scientific execution back
to the sealed worker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


LOCAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_DIR.parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "cap24_tau1em4_r50_20260810_v3_chtc"
)
PACKAGE_MANIFEST_SHA256 = (
    "1ec606f6162a6a5c83b8f618112cb36ed271d2d233543c8c70d81f5098e3f7fb"
)
SOURCE_ARCHIVE_SHA256 = (
    "b70723e52058275ab31ef654f638d9510f0b81707f731d0b750aa395e539027c"
)
ROUTE_CONTRACT_SHA256 = (
    "1b2f7254a96a27a7f2a262f1b4bc19c886b421a9cbaa5e24c95e354a02f2cf45"
)
LOCAL_REQUEST_SCHEMA = "paper_i_macro_phase0_local_activation_request_v1"
LOCAL_PREFLIGHT_SCHEMA = "paper_i_macro_phase0_local_host_preflight_v1"
LOCAL_AUTHORIZATION_SCHEMA = "paper_i_macro_phase0_local_authorization_v1"
LOCAL_ACTIVATION_SCHEMA = "paper_i_macro_phase0_local_activation_manifest_v1"
LOCAL_SERIAL_SCHEMA = "paper_i_macro_phase0_local_serial_manifest_v1"
LOCAL_STATUS_SCHEMA = "paper_i_macro_phase0_local_serial_status_v1"


def _load_worker() -> Any:
    package_text = PACKAGE_DIR.as_posix()
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    spec = importlib.util.spec_from_file_location(
        "paper_i_macro_phase0_v3_worker",
        PACKAGE_DIR / "run_cell.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the sealed v3 worker.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json_exclusive(worker: Any, path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(worker.canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_atomic(worker: Any, path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"Stale status temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(worker.canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_digested(worker: Any, path: Path, *, label: str) -> dict[str, Any]:
    payload = worker.load_json(path, label=label)
    worker.verify_self_digest(payload, label=label)
    return payload


def _binding(worker: Any, path: Path, *, root: Path) -> dict[str, Any]:
    payload = _load_digested(worker, path, label=path.name)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": worker.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "canonical_sha256": payload["sha256"],
    }


def _queue_rows(worker: Any) -> list[dict[str, Any]]:
    queue_path = PACKAGE_DIR / "queue.tsv"
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(queue_path.read_text(encoding="utf-8").splitlines()):
        fields = line.split("\t")
        if len(fields) != 8:
            raise RuntimeError(f"Malformed queue row {index}.")
        (
            execution_id,
            job_path,
            protocol_path,
            job_sha256,
            request_cpus,
            memory_mb,
            disk_mb,
            max_runtime_seconds,
        ) = fields
        absolute_job = PACKAGE_DIR / worker.safe_relative_path(
            job_path, label=f"queue job {index}"
        )
        if worker.sha256_file(absolute_job) != job_sha256:
            raise RuntimeError(f"Queue job binding drifted at row {index}.")
        rows.append(
            {
                "execution_id": execution_id,
                "job_path": job_path,
                "protocol_path": protocol_path,
                "job_sha256": job_sha256,
                "request_cpus": int(request_cpus),
                "memory_mb": int(memory_mb),
                "disk_mb": int(disk_mb),
                "max_runtime_seconds": int(max_runtime_seconds),
            }
        )
    if [row["execution_id"] for row in rows] != list(
        worker.expected_execution_ids()
    ):
        raise RuntimeError("Queue execution order drifted.")
    return rows


def _closed_manifest(worker: Any) -> dict[str, Any]:
    manifest = _load_digested(
        worker, PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    source = manifest.get("source_archive")
    if (
        manifest.get("sha256") != PACKAGE_MANIFEST_SHA256
        or manifest.get("execution_target") != "chtc"
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
        or not isinstance(source, Mapping)
        or source.get("sha256") != SOURCE_ARCHIVE_SHA256
        or manifest.get("child_route_contract_sha256") != ROUTE_CONTRACT_SHA256
    ):
        raise RuntimeError("Sealed v3 package identity drifted.")
    if worker.sha256_file(PACKAGE_DIR / str(source["path"])) != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Sealed source archive bytes drifted.")
    _queue_rows(worker)
    return manifest


def _physical_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def prepare_activation(*, activation_dir: Path) -> dict[str, Any]:
    worker = _load_worker()
    manifest = _closed_manifest(worker)
    rows = _queue_rows(worker)
    if activation_dir.exists() or activation_dir.is_symlink():
        raise FileExistsError(f"Activation destination exists: {activation_dir}")
    activation_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{activation_dir.name}.build-", dir=activation_dir.parent
        )
    )
    try:
        request = worker.digested(
            {
                "schema": LOCAL_REQUEST_SCHEMA,
                "status": "authorized_local_execution",
                "source_package_id": manifest["package_id"],
                "source_campaign_id": manifest["campaign_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "requested_execution_ids": [row["execution_id"] for row in rows],
                "scope": "prepare_six_cell_local_serial_execution_v1",
                "authorization_kind": "explicit_user_local_execution_authority",
                "explicit_user_authority_recorded": True,
                "execution_target": "local_mac_serial",
                "source_package_execution_target": "chtc",
                "execution_target_change_only": True,
                "scientific_settings_changed": False,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
            }
        )
        request_path = temporary / "activation_request.json"
        _write_json_exclusive(worker, request_path, request)

        preflight_rows = [
            worker.preflight(PACKAGE_DIR / row["job_path"]) for row in rows
        ]
        preflight = worker.digested(
            {
                "schema": LOCAL_PREFLIGHT_SCHEMA,
                "status": "passed_local_host_preflight",
                "source_package_id": manifest["package_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "execution_target": "local_mac_serial",
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "physical_memory_bytes": _physical_memory_bytes(),
                "sealed_source_preflight_count": len(preflight_rows),
                "sealed_source_preflights": preflight_rows,
                "scientific_execution_performed": False,
            }
        )
        preflight_path = temporary / "host_preflight.json"
        _write_json_exclusive(worker, preflight_path, preflight)
        request_binding = _binding(worker, request_path, root=temporary)
        preflight_binding = _binding(worker, preflight_path, root=temporary)

        authorizations: list[dict[str, Any]] = []
        for row in rows:
            job, _package, _protocol, _locks = worker._load_closed_job(
                PACKAGE_DIR / row["job_path"]
            )
            authority = worker.digested(
                {
                    "schema": LOCAL_AUTHORIZATION_SCHEMA,
                    "status": "authorized_local_cell_execution",
                    "source_package_id": manifest["package_id"],
                    "source_campaign_id": manifest["campaign_id"],
                    "execution_id": row["execution_id"],
                    "job_spec_sha256": job["sha256"],
                    "protocol_sha256": job["protocol_sha256"],
                    "package_manifest_sha256": manifest["sha256"],
                    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                    "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                    "activation_request": request_binding,
                    "host_preflight": preflight_binding,
                    "scope": "single_cell_local_execution_only",
                    "authorization_kind": "explicit_user_local_execution_authority",
                    "execution_target": "local_mac_serial",
                    "execution_target_change_only": True,
                    "scientific_settings_changed": False,
                    "execution_authorized": True,
                    "submission_authorized": False,
                    "paper_evidence_adoption_authorized": False,
                    "submitted": False,
                }
            )
            authority_path = temporary / "authorizations" / f"{row['execution_id']}.json"
            _write_json_exclusive(worker, authority_path, authority)
            authorizations.append(
                {"execution_id": row["execution_id"], **_binding(worker, authority_path, root=temporary)}
            )

        activation = worker.digested(
            {
                "schema": LOCAL_ACTIVATION_SCHEMA,
                "status": "passed_local_activation_prepared",
                "source_package_id": manifest["package_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "activation_request": request_binding,
                "host_preflight": preflight_binding,
                "authorizations": authorizations,
                "authorization_count": len(authorizations),
                "execution_target": "local_mac_serial",
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
            }
        )
        _write_json_exclusive(worker, temporary / "activation_manifest.json", activation)
        os.rename(temporary, activation_dir)
        return activation
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_local_authorization(
    worker: Any,
    path: Path,
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _load_digested(worker, path, label="local execution authorization")
    if path.parent.name != "authorizations":
        raise RuntimeError("Local authority is outside an activation directory.")
    activation_root = path.parent.parent

    def bound_payload(raw: Any, *, label: str, expected: str) -> dict[str, Any]:
        if not isinstance(raw, Mapping) or raw.get("path") != expected:
            raise RuntimeError(f"{label} binding path drifted.")
        target = activation_root / expected
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != int(raw.get("size_bytes", -1))
            or worker.sha256_file(target) != raw.get("sha256")
        ):
            raise RuntimeError(f"{label} byte binding drifted.")
        payload = _load_digested(worker, target, label=label)
        if payload["sha256"] != raw.get("canonical_sha256"):
            raise RuntimeError(f"{label} canonical binding drifted.")
        return payload

    request = bound_payload(
        authority.get("activation_request"),
        label="local activation request",
        expected="activation_request.json",
    )
    preflight = bound_payload(
        authority.get("host_preflight"),
        label="local host preflight",
        expected="host_preflight.json",
    )
    source = manifest.get("source_archive")
    expected_ids = list(worker.expected_execution_ids())
    if (
        authority.get("schema") != LOCAL_AUTHORIZATION_SCHEMA
        or authority.get("status") != "authorized_local_cell_execution"
        or authority.get("source_package_id") != manifest.get("package_id")
        or authority.get("source_campaign_id") != manifest.get("campaign_id")
        or authority.get("execution_id") != job.get("execution_id")
        or authority.get("job_spec_sha256") != job.get("sha256")
        or authority.get("protocol_sha256") != job.get("protocol_sha256")
        or authority.get("package_manifest_sha256") != manifest.get("sha256")
        or authority.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or authority.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or authority.get("scope") != "single_cell_local_execution_only"
        or authority.get("authorization_kind")
        != "explicit_user_local_execution_authority"
        or authority.get("execution_target") != "local_mac_serial"
        or authority.get("execution_target_change_only") is not True
        or authority.get("scientific_settings_changed") is not False
        or authority.get("execution_authorized") is not True
        or authority.get("submission_authorized") is not False
        or authority.get("paper_evidence_adoption_authorized") is not False
        or authority.get("submitted") is not False
        or not isinstance(source, Mapping)
        or source.get("sha256") != SOURCE_ARCHIVE_SHA256
        or request.get("schema") != LOCAL_REQUEST_SCHEMA
        or request.get("package_manifest_sha256") != manifest.get("sha256")
        or request.get("requested_execution_ids") != expected_ids
        or request.get("scope") != "prepare_six_cell_local_serial_execution_v1"
        or request.get("execution_authorized") is not True
        or request.get("submission_authorized") is not False
        or preflight.get("schema") != LOCAL_PREFLIGHT_SCHEMA
        or preflight.get("status") != "passed_local_host_preflight"
        or preflight.get("package_manifest_sha256") != manifest.get("sha256")
        or preflight.get("sealed_source_preflight_count") != 6
        or [row.get("execution_id") for row in preflight.get("sealed_source_preflights", [])]
        != expected_ids
        or preflight.get("scientific_execution_performed") is not False
    ):
        raise RuntimeError("Local execution authorization drifted.")
    return authority


def run_local_cell(
    *, job_path: Path, authorization_path: Path, output_dir: Path, receipt_path: Path
) -> dict[str, Any]:
    worker = _load_worker()

    def validator(
        path: Path,
        *,
        job: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        return _validate_local_authorization(
            worker, path, job=job, manifest=manifest
        )

    worker._validate_authorization = validator
    return worker.run_cell(
        job_path=job_path,
        authorization_path=authorization_path,
        output_dir=output_dir,
        receipt_path=receipt_path,
    )


def run_serial(*, activation_dir: Path, runtime_dir: Path) -> int:
    worker = _load_worker()
    manifest = _closed_manifest(worker)
    rows = _queue_rows(worker)
    activation = _load_digested(
        worker, activation_dir / "activation_manifest.json", label="local activation"
    )
    if (
        activation.get("schema") != LOCAL_ACTIVATION_SCHEMA
        or activation.get("package_manifest_sha256") != manifest.get("sha256")
        or activation.get("authorization_count") != 6
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not False
    ):
        raise RuntimeError("Local activation manifest drifted.")
    if runtime_dir.exists() or runtime_dir.is_symlink():
        raise FileExistsError(f"Runtime destination exists: {runtime_dir}")
    for name in ("runs", "worker_receipts", "logs", "in_progress"):
        (runtime_dir / name).mkdir(parents=True, exist_ok=False)
    serial_manifest = worker.digested(
        {
            "schema": LOCAL_SERIAL_SCHEMA,
            "status": "authorized_pending_execution",
            "run_class": "candidate",
            "source_package_id": manifest["package_id"],
            "package_manifest_sha256": manifest["sha256"],
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "activation_manifest_sha256": activation["sha256"],
            "execution_target": "local_mac_serial",
            "execution_ids": [row["execution_id"] for row in rows],
            "target_horizon": 50,
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _write_json_exclusive(worker, runtime_dir / "serial_manifest.json", serial_manifest)
    completed: list[str] = []
    for row in rows:
        execution_id = row["execution_id"]
        status = worker.digested(
            {
                "schema": LOCAL_STATUS_SCHEMA,
                "status": "running",
                "serial_manifest_sha256": serial_manifest["sha256"],
                "current_execution_id": execution_id,
                "completed_execution_ids": completed,
                "remaining_execution_ids": [
                    candidate["execution_id"]
                    for candidate in rows
                    if candidate["execution_id"] not in completed
                    and candidate["execution_id"] != execution_id
                ],
            }
        )
        _write_json_atomic(worker, runtime_dir / "serial_status.json", status)
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "run-cell",
            "--job",
            str(PACKAGE_DIR / row["job_path"]),
            "--authorization",
            str(activation_dir / "authorizations" / f"{execution_id}.json"),
            "--output-dir",
            str(runtime_dir / "runs" / execution_id),
            "--receipt",
            str(runtime_dir / "worker_receipts" / f"{execution_id}.json"),
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "STATIC_ADAPT_HH_POOL_CACHE": "off",
                "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
                "TMPDIR": str(runtime_dir / "in_progress"),
            }
        )
        with (runtime_dir / "logs" / f"{execution_id}.out").open("xb") as stdout, (
            runtime_dir / "logs" / f"{execution_id}.err"
        ).open("xb") as stderr:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if result.returncode != 0:
            failed = worker.digested(
                {
                    "schema": LOCAL_STATUS_SCHEMA,
                    "status": "failed",
                    "serial_manifest_sha256": serial_manifest["sha256"],
                    "failed_execution_id": execution_id,
                    "exit_code": result.returncode,
                    "completed_execution_ids": completed,
                }
            )
            _write_json_atomic(worker, runtime_dir / "serial_status.json", failed)
            return result.returncode
        completed.append(execution_id)
    finished = worker.digested(
        {
            "schema": LOCAL_STATUS_SCHEMA,
            "status": "passed",
            "serial_manifest_sha256": serial_manifest["sha256"],
            "completed_execution_ids": completed,
            "completed_count": len(completed),
        }
    )
    _write_json_atomic(worker, runtime_dir / "serial_status.json", finished)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--activation-dir", type=Path, required=True)
    cell = subparsers.add_parser("run-cell")
    cell.add_argument("--job", type=Path, required=True)
    cell.add_argument("--authorization", type=Path, required=True)
    cell.add_argument("--output-dir", type=Path, required=True)
    cell.add_argument("--receipt", type=Path, required=True)
    serial = subparsers.add_parser("run-serial")
    serial.add_argument("--activation-dir", type=Path, required=True)
    serial.add_argument("--runtime-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            payload = prepare_activation(activation_dir=args.activation_dir.resolve())
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "run-cell":
            payload = run_local_cell(
                job_path=args.job.resolve(),
                authorization_path=args.authorization.resolve(),
                output_dir=args.output_dir.resolve(),
                receipt_path=args.receipt.resolve(),
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
        return run_serial(
            activation_dir=args.activation_dir.resolve(),
            runtime_dir=args.runtime_dir.resolve(),
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
