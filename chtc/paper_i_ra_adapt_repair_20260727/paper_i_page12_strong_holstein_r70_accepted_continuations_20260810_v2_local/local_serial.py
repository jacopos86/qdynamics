#!/usr/bin/env python3
"""Gate and serially run the three local Page-12 continuations.

This entrypoint is deliberately local-only and fixed at one scientific child.
It cannot submit work, and it refuses to start until the independent Page-13
campaign has six authenticated round-50 closures and its report watcher has
refreshed all six cells.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from package_contract import (  # noqa: E402
    ACTIVATION_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    RESOURCE_ENVELOPE,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    load_json,
    sha256_file,
    verify_self_digest,
)
from pipelines.reporting import (  # noqa: E402
    append_paper_i_macro_phase0_proxy_no_lanes_page13 as page13,
)
from pipelines.reporting import (  # noqa: E402
    watch_paper_i_macro_phase0_proxy_no_lanes_page13 as page13_watch,
)


DEFAULT_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page12_global_singleton_gradient_phase0_strong_"
    "r50_to_r70_serial_20260810_v2"
)
SERIAL_STATUS_SCHEMA = "paper_i_page12_strong_r70_local_serial_status_v2"
SERIAL_MANIFEST_SCHEMA = "paper_i_page12_strong_r70_local_serial_manifest_v2"
MAX_CONCURRENCY = 1
LOCAL_MIN_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
LOCAL_MIN_FREE_DISK_BYTES = 48 * 1024**3


def _write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    if exclusive:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise PackageContractError(f"Stale status temporary: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_page13_completion_gate() -> dict[str, Any]:
    """Authenticate both Page-13 science closure and final report refresh."""

    expected = page13_watch.expected_execution_ids()
    serial = page13.load(page13_watch.SERIAL_STATUS_PATH)
    page13.verify_self_digest(serial, label="Page-13 source serial status")
    completed = page13_watch.validated_completed_execution_ids(serial)
    watcher = page13.load(page13_watch.WATCH_STATUS_PATH)
    page13.verify_self_digest(watcher, label="Page-13 final watcher status")
    validated = tuple(
        str(row) for row in watcher.get("validated_completed_execution_ids", [])
    )
    refreshed = tuple(
        str(row) for row in watcher.get("refreshed_completed_execution_ids", [])
    )
    if (
        serial.get("status") != "passed"
        or serial.get("running_execution_ids", []) != []
        or serial.get("current_execution_id") is not None
        or completed != expected
        or watcher.get("status") != "passed_all_six_round50_cells_refreshed"
        or validated != expected
        or refreshed != expected
        or watcher.get("last_error") is not None
    ):
        raise PackageContractError("Page-13 completion gate is not closed.")

    jobs = page13._jobs()
    jobs_by_execution_id = {
        str(job["execution_id"]): (regime, job)
        for regime, (_job_path, job) in jobs.items()
    }
    if set(jobs_by_execution_id) != set(expected):
        raise PackageContractError("Page-13 closure identity drifted.")
    exact = page13.exact_references()
    closed: list[str] = []
    for execution_id in expected:
        regime, _job = jobs_by_execution_id[execution_id]
        route = page13._completed_route(execution_id, exact=exact[regime])
        if route is None or route.get("latest", {}).get("k") != 50:
            raise PackageContractError(
                f"Page-13 round-50 closure is absent: {execution_id}"
            )
        closed.append(execution_id)
    if tuple(closed) != expected:
        raise PackageContractError("Page-13 closure order drifted.")
    return digested(
        {
            "schema": "paper_i_page13_completion_gate_receipt_v2",
            "status": "passed_all_six_authenticated_and_refreshed",
            "execution_ids": list(expected),
            "serial_status_sha256": serial["sha256"],
            "watcher_status_sha256": watcher["sha256"],
            "round50_closure_count": 6,
        }
    )


def _available_memory_bytes() -> int | None:
    try:
        completed = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = completed.stdout.splitlines()
        page_size = 4096
        if lines and "page size of" in lines[0]:
            page_size = int(lines[0].split("page size of", 1)[1].split("bytes", 1)[0])
        values: dict[str, int] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            values[key.strip()] = int(raw.strip().rstrip("."))
        pages = sum(
            values.get(key, 0)
            for key in (
                "Pages free",
                "Pages inactive",
                "Pages speculative",
                "Pages purgeable",
            )
        )
        return pages * page_size
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def capacity_receipt(runtime_parent: Path) -> dict[str, Any]:
    available_memory = _available_memory_bytes()
    free_disk = shutil.disk_usage(runtime_parent).free
    blockers: list[str] = []
    if available_memory is None:
        blockers.append("available_memory_unavailable")
    elif available_memory < LOCAL_MIN_AVAILABLE_MEMORY_BYTES:
        blockers.append("available_or_reclaimable_memory_below_local_guard")
    if free_disk < LOCAL_MIN_FREE_DISK_BYTES:
        blockers.append("free_disk_below_local_guard")
    return digested(
        {
            "schema": "paper_i_page12_strong_r70_local_capacity_receipt_v2",
            "status": "passed" if not blockers else "blocked",
            "available_or_reclaimable_memory_bytes": available_memory,
            "free_disk_bytes": free_disk,
            "local_required_available_memory_bytes": (
                LOCAL_MIN_AVAILABLE_MEMORY_BYTES
            ),
            "local_required_free_disk_bytes": LOCAL_MIN_FREE_DISK_BYTES,
            "chtc_resource_envelope_provenance_only": dict(RESOURCE_ENVELOPE),
            "blockers": blockers,
            "scientific_execution_performed": False,
        }
    )


def _load_package() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="package")
    verify_self_digest(manifest, label="package")
    activation = load_json(
        PACKAGE_DIR / "activation/activation_manifest.json", label="activation"
    )
    verify_self_digest(activation, label="activation")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("execution_target") != "local_mac_serial"
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
        or activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("package_manifest_sha256") != manifest.get("sha256")
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not False
        or activation.get("submitted") is not False
        or activation.get("authorization_count") != 3
    ):
        raise PackageContractError("Local package/activation identity drifted.")
    return manifest, activation


def _overlapping_worker_commands() -> list[str]:
    try:
        output = subprocess.run(
            ["ps", "-axo", "command="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageContractError("Cannot audit local worker overlap.") from exc
    runner = (PACKAGE_DIR / "run_cell.py").resolve().as_posix()
    return [line for line in output.splitlines() if runner in line and " --run " in line]


def _validate_completed_cell(
    runtime_dir: Path,
    *,
    execution_id: str,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = runtime_dir / "runs" / execution_id
    manifest_path = run_root / "execution_manifest.json"
    receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    manifest = load_json(manifest_path, label=f"{execution_id} manifest")
    receipt = load_json(receipt_path, label=f"{execution_id} receipt")
    verify_self_digest(manifest, label=f"{execution_id} manifest")
    verify_self_digest(receipt, label=f"{execution_id} receipt")
    prefix = manifest.get("accepted_prefix_preservation")
    if (
        manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("controller_rounds_completed") != TARGET_HORIZON
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("source_checkpoint_sha256")
        != job.get("checkpoint_sha256")
        or not isinstance(prefix, Mapping)
        or prefix.get("status") != "passed"
        or prefix.get("source_round") != 50
        or prefix.get("source_checkpoint_sha256")
        != job.get("checkpoint_sha256")
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("controller_rounds_completed") != TARGET_HORIZON
        or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
    ):
        raise PackageContractError(
            f"Post-run accepted-prefix closure failed: {execution_id}"
        )
    return digested(
        {
            "schema": "paper_i_page12_strong_r70_local_cell_closure_v2",
            "status": "passed",
            "execution_id": execution_id,
            "execution_manifest_sha256": manifest["sha256"],
            "worker_receipt_sha256": receipt["sha256"],
            "accepted_prefix_preserved": True,
            "controller_rounds_completed": TARGET_HORIZON,
        }
    )


def preflight(runtime_dir: Path) -> dict[str, Any]:
    manifest, activation = _load_package()
    gate = validate_page13_completion_gate()
    capacity = capacity_receipt(runtime_dir.parent)
    overlap = _overlapping_worker_commands()
    collisions = runtime_dir.exists() or runtime_dir.is_symlink()
    return digested(
        {
            "schema": "paper_i_page12_strong_r70_local_serial_preflight_v2",
            "status": (
                "passed"
                if capacity["status"] == "passed" and not overlap and not collisions
                else "blocked"
            ),
            "package_manifest_sha256": manifest["sha256"],
            "activation_manifest_sha256": activation["sha256"],
            "page13_gate_sha256": gate["sha256"],
            "capacity": capacity,
            "overlapping_worker_commands": overlap,
            "runtime_collision": collisions,
            "max_concurrency": MAX_CONCURRENCY,
            "target_horizon": TARGET_HORIZON,
            "scientific_execution_performed": False,
        }
    )


def run_serial(runtime_dir: Path) -> int:
    check = preflight(runtime_dir)
    if check["status"] != "passed":
        raise PackageContractError(
            "Local continuation preflight is blocked; no science was started."
        )
    lock_path = runtime_dir.with_name(f".{runtime_dir.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PackageContractError("A local serial launcher already owns the lock.") from exc
        if runtime_dir.exists() or _overlapping_worker_commands():
            raise PackageContractError("Runtime collision appeared after lock acquisition.")
        for name in ("runs", "worker_receipts", "logs", "in_progress"):
            (runtime_dir / name).mkdir(parents=True, exist_ok=False)
        serial_manifest = digested(
            {
                "schema": SERIAL_MANIFEST_SCHEMA,
                "status": "authorized_pending_execution",
                "package_id": PACKAGE_ID,
                "package_manifest_sha256": check["package_manifest_sha256"],
                "page13_gate_sha256": check["page13_gate_sha256"],
                "execution_ids": list(expected_execution_ids()),
                "target_horizon": TARGET_HORIZON,
                "max_concurrency": MAX_CONCURRENCY,
                "execution_authorized": True,
                "submission_authorized": False,
            }
        )
        _write_json(
            runtime_dir / "serial_manifest.json", serial_manifest, exclusive=True
        )
        completed_ids: list[str] = []
        manifest, activation = _load_package()
        auth_by_id = {
            str(row["execution_id"]): row
            for row in activation["authorizations"]
        }
        jobs_by_id = {
            str(row["execution_id"]): row for row in manifest["jobs"]
        }
        for execution_id in expected_execution_ids():
            status = digested(
                {
                    "schema": SERIAL_STATUS_SCHEMA,
                    "status": "running",
                    "serial_manifest_sha256": serial_manifest["sha256"],
                    "completed_execution_ids": completed_ids,
                    "running_execution_ids": [execution_id],
                    "pending_execution_ids": [
                        row
                        for row in expected_execution_ids()
                        if row not in completed_ids and row != execution_id
                    ],
                    "max_concurrency": MAX_CONCURRENCY,
                }
            )
            _write_json(runtime_dir / "serial_status.json", status, exclusive=False)
            job_path = PACKAGE_DIR / str(jobs_by_id[execution_id]["path"])
            auth_path = PACKAGE_DIR / str(auth_by_id[execution_id]["path"])
            command = [
                sys.executable,
                "-B",
                str(PACKAGE_DIR / "run_cell.py"),
                "--job",
                str(job_path),
                "--run",
                "--execution-authorization",
                str(auth_path),
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
            with (runtime_dir / "logs" / f"{execution_id}.out").open(
                "xb"
            ) as stdout, (runtime_dir / "logs" / f"{execution_id}.err").open(
                "xb"
            ) as stderr:
                child = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if child.returncode != 0:
                failed = digested(
                    {
                        "schema": SERIAL_STATUS_SCHEMA,
                        "status": "failed",
                        "serial_manifest_sha256": serial_manifest["sha256"],
                        "completed_execution_ids": completed_ids,
                        "failed_execution_id": execution_id,
                        "exit_code": child.returncode,
                        "running_execution_ids": [],
                        "max_concurrency": MAX_CONCURRENCY,
                    }
                )
                _write_json(runtime_dir / "serial_status.json", failed, exclusive=False)
                return child.returncode
            _validate_completed_cell(
                runtime_dir,
                execution_id=execution_id,
                job=load_json(job_path, label=f"{execution_id} job"),
            )
            completed_ids.append(execution_id)
        passed = digested(
            {
                "schema": SERIAL_STATUS_SCHEMA,
                "status": "passed",
                "serial_manifest_sha256": serial_manifest["sha256"],
                "completed_execution_ids": completed_ids,
                "completed_count": 3,
                "running_execution_ids": [],
                "pending_execution_ids": [],
                "max_concurrency": MAX_CONCURRENCY,
            }
        )
        _write_json(runtime_dir / "serial_status.json", passed, exclusive=False)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    try:
        if args.preflight:
            payload = preflight(args.runtime_dir.resolve())
            print(canonical_json_bytes(payload).decode("utf-8"))
            return 0 if payload["status"] == "passed" else 3
        return run_serial(args.runtime_dir.resolve())
    except (OSError, PackageContractError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
