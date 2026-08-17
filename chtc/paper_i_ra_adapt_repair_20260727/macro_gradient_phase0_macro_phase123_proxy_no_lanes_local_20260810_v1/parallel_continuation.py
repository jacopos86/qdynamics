#!/usr/bin/env python3
"""Adopt the active local campaign and keep exactly two cells in flight."""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


LOCAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_DIR.parents[2]
RUNNER_PATH = LOCAL_DIR / "local_runner.py"
ACTIVATION_DIR = LOCAL_DIR / "activation"
RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_macro_gradient_phase0_macro_phase123_proxy_no_lanes_r50_"
    "serial_20260810_v1"
)
LOCK_PATH = RUNTIME_DIR / "parallel_continuation.lock"
SUPERVISOR_STATUS_PATH = RUNTIME_DIR / "parallel_continuation_status.json"
MAX_CONCURRENCY = 2
SUPERVISOR_SCHEMA = "paper_i_macro_phase0_local_parallel_supervisor_status_v1"


class SupervisorError(RuntimeError):
    pass


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "paper_i_macro_phase0_local_runner_for_parallel", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise SupervisorError("Unable to load the local execution adapter.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_status(runner: Any, worker: Any, value: Mapping[str, Any]) -> None:
    payload = worker.digested(dict(value))
    runner._write_json_atomic(worker, SUPERVISOR_STATUS_PATH, payload)
    artifact_root = os.environ.get("REMOTE_ARTIFACT_DIR")
    if artifact_root:
        artifact_path = Path(artifact_root) / "parallel_continuation_status.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        runner._write_json_atomic(worker, artifact_path, payload)


def _closed_execution(
    runner: Any,
    worker: Any,
    execution_id: str,
) -> bool:
    run_root = RUNTIME_DIR / "runs" / execution_id
    manifest_path = run_root / "execution_manifest.json"
    receipt_path = RUNTIME_DIR / "worker_receipts" / f"{execution_id}.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        return False
    manifest = runner._load_digested(
        worker, manifest_path, label=f"{execution_id} execution manifest"
    )
    receipt = runner._load_digested(
        worker, receipt_path, label=f"{execution_id} worker receipt"
    )
    if (
        manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("controller_rounds_completed") != 50
        or manifest.get("target_horizon") != 50
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("controller_rounds_completed") != 50
    ):
        raise SupervisorError(f"Round-50 closure drifted: {execution_id}")
    summary_path = run_root / "summary/summary.json"
    summary = worker.load_json(summary_path, label=f"{execution_id} summary")
    trace = summary.get("accepted_error_trace")
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or not isinstance(trace, list)
        or [row.get("controller_round") for row in trace] != list(range(1, 51))
    ):
        raise SupervisorError(f"Round-50 summary drifted: {execution_id}")
    return True


def _live_run_cell_pids(execution_ids: tuple[str, ...]) -> dict[str, int]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command=", "-ww"],
        check=True,
        capture_output=True,
        text=True,
    )
    matches: dict[str, list[int]] = {execution_id: [] for execution_id in execution_ids}
    for raw in result.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        pid_text, _, command = raw.partition(" ")
        if "local_runner.py run-cell" not in command:
            continue
        for execution_id in execution_ids:
            if f"/{execution_id}.json" in command:
                matches[execution_id].append(int(pid_text))
    duplicates = {
        execution_id: pids for execution_id, pids in matches.items() if len(pids) > 1
    }
    if duplicates:
        raise SupervisorError(f"Duplicate live cell processes: {duplicates}")
    return {
        execution_id: pids[0]
        for execution_id, pids in matches.items()
        if pids
    }


def _serial_parent_pids() -> list[int]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command=", "-ww"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        int(raw.strip().partition(" ")[0])
        for raw in result.stdout.splitlines()
        if "local_runner.py run-serial" in raw
        and RUNTIME_DIR.name in raw
    ]


def _launch_cell(
    runner: Any,
    row: Mapping[str, Any],
) -> subprocess.Popen[bytes]:
    execution_id = str(row["execution_id"])
    output_dir = RUNTIME_DIR / "runs" / execution_id
    receipt_path = RUNTIME_DIR / "worker_receipts" / f"{execution_id}.json"
    stdout_path = RUNTIME_DIR / "logs" / f"{execution_id}.out"
    stderr_path = RUNTIME_DIR / "logs" / f"{execution_id}.err"
    collisions = [
        path
        for path in (output_dir, receipt_path, stdout_path, stderr_path)
        if path.exists() or path.is_symlink()
    ]
    if collisions:
        raise SupervisorError(
            "Refusing to overwrite a prior cell attempt: "
            + ", ".join(str(path) for path in collisions)
        )
    command = [
        sys.executable,
        "-B",
        str(RUNNER_PATH),
        "run-cell",
        "--job",
        str(runner.PACKAGE_DIR / str(row["job_path"])),
        "--authorization",
        str(ACTIVATION_DIR / "authorizations" / f"{execution_id}.json"),
        "--output-dir",
        str(output_dir),
        "--receipt",
        str(receipt_path),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "off",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "TMPDIR": str(RUNTIME_DIR / "in_progress"),
        }
    )
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        return subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )


def _contiguous_prefix(
    execution_ids: tuple[str, ...], closed: set[str]
) -> tuple[str, ...]:
    result: list[str] = []
    for execution_id in execution_ids:
        if execution_id not in closed:
            break
        result.append(execution_id)
    return tuple(result)


def _publish_serial_status(
    runner: Any,
    worker: Any,
    *,
    serial_manifest_sha256: str,
    execution_ids: tuple[str, ...],
    closed: set[str],
    live: Mapping[str, int],
    status: str,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed_prefix = _contiguous_prefix(execution_ids, closed)
    running_ids = [row for row in execution_ids if row in live and row not in closed]
    remaining_ids = [
        row for row in execution_ids if row not in closed and row not in live
    ]
    unsigned: dict[str, Any] = {
        "schema": runner.LOCAL_STATUS_SCHEMA,
        "status": status,
        "execution_mode": "local_parallel_two_regimes_v1",
        "maximum_concurrency": MAX_CONCURRENCY,
        "serial_manifest_sha256": serial_manifest_sha256,
        "completed_execution_ids": list(completed_prefix),
        "published_completed_execution_ids": [
            row for row in execution_ids if row in closed
        ],
        "running_execution_ids": running_ids,
        "running_pids": {row: int(live[row]) for row in running_ids},
        "current_execution_id": running_ids[0] if running_ids else None,
        "remaining_execution_ids": remaining_ids,
    }
    if failure is not None:
        unsigned["failure"] = copy.deepcopy(dict(failure))
    payload = worker.digested(unsigned)
    runner._write_json_atomic(worker, RUNTIME_DIR / "serial_status.json", payload)
    return payload


def supervise(*, poll_seconds: float, once: bool = False) -> int:
    runner = _load_runner()
    worker = runner._load_worker()
    manifest = runner._closed_manifest(worker)
    rows = runner._queue_rows(worker)
    execution_ids = tuple(str(row["execution_id"]) for row in rows)
    rows_by_id = {str(row["execution_id"]): row for row in rows}
    serial_manifest = runner._load_digested(
        worker, RUNTIME_DIR / "serial_manifest.json", label="local serial manifest"
    )
    if (
        serial_manifest.get("package_manifest_sha256") != manifest.get("sha256")
        or tuple(serial_manifest.get("execution_ids", [])) != execution_ids
        or serial_manifest.get("target_horizon") != 50
    ):
        raise SupervisorError("Local serial manifest drifted.")
    if _serial_parent_pids():
        raise SupervisorError("The obsolete one-lane parent scheduler is still running.")

    launched: dict[str, subprocess.Popen[bytes]] = {}
    missing_polls: dict[str, int] = {}
    while True:
        closed = {
            execution_id
            for execution_id in execution_ids
            if _closed_execution(runner, worker, execution_id)
        }
        live = _live_run_cell_pids(execution_ids)
        for execution_id, process in list(launched.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            launched.pop(execution_id)
            if returncode != 0 and execution_id not in closed:
                failure = {
                    "execution_id": execution_id,
                    "reason": "launched_cell_exit_nonzero",
                    "returncode": returncode,
                }
                _publish_serial_status(
                    runner,
                    worker,
                    serial_manifest_sha256=serial_manifest["sha256"],
                    execution_ids=execution_ids,
                    closed=closed,
                    live=live,
                    status="failed",
                    failure=failure,
                )
                _write_status(
                    runner,
                    worker,
                    {
                        "schema": SUPERVISOR_SCHEMA,
                        "status": "failed",
                        "failure": failure,
                    },
                )
                return 2
        for execution_id in execution_ids:
            if execution_id in closed or execution_id in live:
                missing_polls.pop(execution_id, None)
                continue
            output_dir = RUNTIME_DIR / "runs" / execution_id
            receipt = RUNTIME_DIR / "worker_receipts" / f"{execution_id}.json"
            if output_dir.exists() or receipt.exists():
                missing_polls[execution_id] = missing_polls.get(execution_id, 0) + 1
                if missing_polls[execution_id] >= 3:
                    raise SupervisorError(
                        f"Incomplete published output without a live process: {execution_id}"
                    )

        if len(closed) == len(execution_ids):
            serial_status = _publish_serial_status(
                runner,
                worker,
                serial_manifest_sha256=serial_manifest["sha256"],
                execution_ids=execution_ids,
                closed=closed,
                live={},
                status="passed",
            )
            _write_status(
                runner,
                worker,
                {
                    "schema": SUPERVISOR_SCHEMA,
                    "status": "passed_all_six_round50_cells",
                    "serial_status_sha256": serial_status["sha256"],
                    "completed_execution_ids": list(execution_ids),
                },
            )
            return 0

        available_slots = MAX_CONCURRENCY - len(
            [row for row in live if row not in closed]
        )
        if available_slots < 0:
            raise SupervisorError("More than two scientific cells are running.")
        for execution_id in execution_ids:
            if available_slots == 0:
                break
            if execution_id in closed or execution_id in live:
                continue
            try:
                process = _launch_cell(runner, rows_by_id[execution_id])
            except SupervisorError as exc:
                failure = {
                    "execution_id": execution_id,
                    "reason": "refused_unsafe_or_ambiguous_relaunch",
                    "detail": str(exc),
                }
                _publish_serial_status(
                    runner,
                    worker,
                    serial_manifest_sha256=serial_manifest["sha256"],
                    execution_ids=execution_ids,
                    closed=closed,
                    live=live,
                    status="failed",
                    failure=failure,
                )
                _write_status(
                    runner,
                    worker,
                    {
                        "schema": SUPERVISOR_SCHEMA,
                        "status": "failed",
                        "failure": failure,
                    },
                )
                return 2
            launched[execution_id] = process
            live[execution_id] = process.pid
            available_slots -= 1
            print(
                json.dumps(
                    {
                        "event": "launched_cell",
                        "execution_id": execution_id,
                        "pid": process.pid,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        serial_status = _publish_serial_status(
            runner,
            worker,
            serial_manifest_sha256=serial_manifest["sha256"],
            execution_ids=execution_ids,
            closed=closed,
            live=live,
            status="running",
        )
        _write_status(
            runner,
            worker,
            {
                "schema": SUPERVISOR_SCHEMA,
                "status": "running_two_regimes",
                "maximum_concurrency": MAX_CONCURRENCY,
                "serial_status_sha256": serial_status["sha256"],
                "completed_execution_ids": [
                    row for row in execution_ids if row in closed
                ],
                "running_execution_ids": [
                    row for row in execution_ids if row in live and row not in closed
                ],
                "running_pids": {
                    row: live[row]
                    for row in execution_ids
                    if row in live and row not in closed
                },
            },
        )
        if once:
            return 0
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1.0:
        raise SystemExit("--poll-seconds must be at least 1")
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("The two-regime supervisor is already running.", file=sys.stderr)
            return 3
        lock_stream.seek(0)
        lock_stream.truncate()
        lock_stream.write(f"{os.getpid()}\n")
        lock_stream.flush()
        try:
            return supervise(poll_seconds=args.poll_seconds, once=args.once)
        except (OSError, ValueError, SupervisorError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
