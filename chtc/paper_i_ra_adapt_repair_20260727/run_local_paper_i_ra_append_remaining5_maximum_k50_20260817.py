#!/usr/bin/env python3
"""Run the five remaining append-only all-phase-adaptive Paper-I cells.

Weak--weak is deliberately excluded: its preserved accepted prefix is report
evidence, not an execution authority for this fresh opt-in V2 campaign.  Each
cell closes at either fifty accepted rounds or the authenticated Phase-III
natural terminal.  A failed cell is preserved and does not prevent later
pristine cells from running.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import psutil


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    paper_i_ra_serial_campaign_runtime_20260816 as serial_runtime,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_paper_i_ra_all6_adaptive_shortlist_append_then_plateau_maximum_k50_20260817
    as maximum_runner,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    RAAdaptOperationalControls,
)
from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt  # noqa: E402
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    CheckpointObservation,
    EstimatorLedgerObservation,
    FreshStart,
    SRObservationPolicy,
)


RunnerError = maximum_runner.RunnerError
CellSpec = maximum_runner.CellSpec
TARGET_HORIZON = 50
CAMPAIGN_ID = (
    "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_20260817_v3"
)
MAXIMUM_COMPLETION_KIND = maximum_runner.MAXIMUM_COMPLETION_KIND
NATURAL_COMPLETION_KIND = maximum_runner.NATURAL_COMPLETION_KIND

AUTHORITY_DIR = RUNNER_PATH.parent / f"{CAMPAIGN_ID}_authority"
PLAN_PATH = AUTHORITY_DIR / "plan.json"
AUTHORIZATION_PATH = AUTHORITY_DIR / "authorization.json"
RUNTIME_ROOT = REPO_ROOT / "output/local_runs" / CAMPAIGN_ID
RUNS_ROOT = RUNTIME_ROOT / "runs"
STAGING_ROOT = RUNTIME_ROOT / "in_progress"
WORKER_ROOT = RUNTIME_ROOT / "worker_receipts"
GUARD_ROOT = RUNTIME_ROOT / "guard_receipts"
LOG_ROOT = RUNTIME_ROOT / "cell_logs"
STATUS_PATH = RUNTIME_ROOT / "status.json"
LOCK_PATH = RUNTIME_ROOT / "campaign.lock"

PLAN_SCHEMA = "paper_i_ra_append_remaining5_maximum_k50_plan_v1"
AUTH_SCHEMA = "paper_i_ra_append_remaining5_maximum_k50_authorization_v1"
MANIFEST_SCHEMA = "paper_i_ra_append_remaining5_maximum_k50_manifest_v1"
WORKER_SCHEMA = "paper_i_ra_append_remaining5_maximum_k50_worker_v1"
GUARD_SCHEMA = "paper_i_ra_append_remaining5_maximum_k50_guard_v1"
STATUS_SCHEMA = "paper_i_ra_append_remaining5_maximum_k50_status_v1"

EXPECTED_ENV = dict(maximum_runner.EXPECTED_ENV)
CHILD_TOKEN_ENV = "PAPER_I_RA_APPEND_REMAINING5_CHILD_TOKEN"
MAXIMUM_CONCURRENCY = 2
PAIR_MEMORY_BYTES = 12 * 1024**3
PAIR_DISK_BYTES = 20 * 1024**3
SINGLE_MEMORY_BYTES = 5 * 1024**3
SINGLE_DISK_BYTES = 10 * 1024**3
INITIAL_DISK_BYTES = 31 * 1024**3
AVAILABLE_MEMORY_FLOOR_BYTES = 2 * 1024**3
FREE_DISK_FLOOR_BYTES = 2 * 1024**3
NPH3_RSS_LIMIT_BYTES = 8 * 1024**3
NPH7_RSS_LIMIT_BYTES = 10 * 1024**3
POLL_SECONDS = 1.0

_REGIME_ORDER = (
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
CELL_SPECS = tuple(
    cell
    for regime_id in _REGIME_ORDER
    for cell in maximum_runner.CELL_SPECS
    if cell.block == "append" and cell.regime_id == regime_id
)
if len(CELL_SPECS) != 5:
    raise RuntimeError("The frozen maximum-k50 route does not expose five append cells.")


digested = serial_runtime.digested
canonical_sha256 = serial_runtime.canonical_sha256
sha256_file = serial_runtime.sha256_file
write_json_exclusive = serial_runtime.write_json_exclusive
file_binding = serial_runtime.file_binding


def _source_inventory() -> dict[str, Any]:
    return maximum_runner._load_core_module().semantic_closure_source_implementation_inventory()


def _protocol_binding(cell: CellSpec) -> dict[str, Any]:
    return maximum_runner._protocol_binding(cell)


def _protocol_inventory_sha(binding: Mapping[str, Any]) -> str:
    direct = binding.get("source_implementation_inventory_sha256")
    if isinstance(direct, str):
        return direct
    locks = binding.get("source_locks")
    if isinstance(locks, Mapping):
        value = locks.get("implementation_source_inventory_sha256")
        if isinstance(value, str):
            return value
    raise RunnerError("Protocol binding omits its implementation inventory.")


def _cell(execution_id: str) -> CellSpec:
    matches = [cell for cell in CELL_SPECS if cell.execution_id == execution_id]
    if len(matches) != 1:
        raise RunnerError("Unknown remaining-five execution ID.")
    return matches[0]


def _paths(cell: CellSpec) -> tuple[Path, Path, Path, Path, Path]:
    return (
        RUNS_ROOT / cell.execution_id,
        STAGING_ROOT / cell.execution_id,
        WORKER_ROOT / f"{cell.execution_id}.json",
        GUARD_ROOT / f"{cell.execution_id}.json",
        LOG_ROOT / f"{cell.execution_id}.log",
    )


def build_plan(*, created_at: str | None = None) -> dict[str, Any]:
    inventory = _source_inventory()
    bindings = [_protocol_binding(cell) for cell in CELL_SPECS]
    if {_protocol_inventory_sha(row) for row in bindings} != {inventory["sha256"]}:
        raise RunnerError("Remaining-five protocols do not share one source inventory.")
    return digested(
        {
            "schema": PLAN_SCHEMA,
            "created_at": created_at or serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "scope": "five_remaining_regimes_append_only_v1",
            "maximum_controller_rounds": TARGET_HORIZON,
            "allowed_cell_completions": [
                MAXIMUM_COMPLETION_KIND,
                NATURAL_COMPLETION_KIND,
            ],
            "phase3_no_positive_policy": "typed_natural_terminal_v1",
            "insertion_policies": ["append_only"],
            "excluded_existing_regime": "weak_weak",
            "execution_ids": [cell.execution_id for cell in CELL_SPECS],
            "cells": [asdict(cell) for cell in CELL_SPECS],
            "protocol_bindings": bindings,
            "source_implementation_inventory_sha256": inventory["sha256"],
            "source_implementation_file_count": inventory["source_count"],
            "runner": file_binding(RUNNER_PATH),
            "maximum_runner_dependency": file_binding(
                Path(maximum_runner.__file__).resolve()
            ),
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "serial_capacity_fallback_authorized": True,
            "capacity": {
                "initial_free_disk_bytes": INITIAL_DISK_BYTES,
                "pair_available_memory_bytes": PAIR_MEMORY_BYTES,
                "pair_free_disk_bytes": PAIR_DISK_BYTES,
                "single_available_memory_bytes": SINGLE_MEMORY_BYTES,
                "single_free_disk_bytes": SINGLE_DISK_BYTES,
                "runtime_available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
                "runtime_free_disk_floor_bytes": FREE_DISK_FLOOR_BYTES,
                "nph3_rss_limit_bytes": NPH3_RSS_LIMIT_BYTES,
                "nph7_rss_limit_bytes": NPH7_RSS_LIMIT_BYTES,
            },
            "runtime_environment": dict(EXPECTED_ENV),
            "execution_authorized": False,
            "archive_rotation_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def prepare_plan() -> dict[str, Any]:
    plan = build_plan()
    serial_runtime.prepare_authority_directory(
        AUTHORITY_DIR,
        files={"plan.json": plan},
        error_type=RunnerError,
    )
    return plan


def _load_digested(path: Path, schema: str) -> dict[str, Any]:
    return serial_runtime.load_digested(path, schema=schema, error_type=RunnerError)


def validate_plan(*, recompute_protocols: bool) -> dict[str, Any]:
    plan = _load_digested(PLAN_PATH, PLAN_SCHEMA)
    expected = build_plan(created_at=str(plan.get("created_at")))
    if not recompute_protocols:
        expected["protocol_bindings"] = plan.get("protocol_bindings")
        body = dict(expected)
        body.pop("sha256", None)
        expected["sha256"] = canonical_sha256(body)
    if plan != expected:
        raise RunnerError("Remaining-five plan drifted.")
    return plan


def authorize() -> dict[str, Any]:
    plan = validate_plan(recompute_protocols=True)
    if AUTHORIZATION_PATH.exists() or AUTHORIZATION_PATH.is_symlink():
        raise RunnerError("Remaining-five authorization already exists.")
    receipt = digested(
        {
            "schema": AUTH_SCHEMA,
            "created_at": serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "authorization_basis": "explicit_user_request_run_other_five_regimes_v1",
            "plan_sha256": plan["sha256"],
            "runner_sha256": plan["runner"]["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "execution_ids": plan["execution_ids"],
            "execution_authorized": True,
            "archive_rotation_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(AUTHORIZATION_PATH, receipt)
    return receipt


def validate_authority(*, recompute_protocols: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = validate_plan(recompute_protocols=recompute_protocols)
    receipt = _load_digested(AUTHORIZATION_PATH, AUTH_SCHEMA)
    if any(
        (
            receipt.get("campaign_id") != CAMPAIGN_ID,
            receipt.get("authorization_basis")
            != "explicit_user_request_run_other_five_regimes_v1",
            receipt.get("plan_sha256") != plan["sha256"],
            receipt.get("runner_sha256") != plan["runner"]["sha256"],
            receipt.get("source_implementation_inventory_sha256")
            != plan["source_implementation_inventory_sha256"],
            receipt.get("execution_ids") != plan["execution_ids"],
            receipt.get("execution_authorized") is not True,
            receipt.get("archive_rotation_authorized") is not False,
            receipt.get("submission_authorized") is not False,
            receipt.get("paper_adoption_authorized") is not False,
            receipt.get("paper_evidence_adoption_authorized") is not False,
        )
    ):
        raise RunnerError("Remaining-five authorization drifted.")
    return plan, receipt


def _assert_environment() -> None:
    drift = {key: (os.environ.get(key), value) for key, value in EXPECTED_ENV.items() if os.environ.get(key) != value}
    if drift:
        raise RunnerError(f"Runtime environment drifted: {drift}")


def _child_token(authorization_sha256: str, cell: CellSpec) -> str:
    return canonical_sha256(
        {
            "campaign_id": CAMPAIGN_ID,
            "authorization_sha256": authorization_sha256,
            "execution_id": cell.execution_id,
        }
    )


def _artifact_binding(path: Path, root: Path) -> dict[str, Any]:
    return serial_runtime.artifact_binding(path, root)


def run_child(execution_id: str) -> int:
    cell = _cell(execution_id)
    plan, authorization = validate_authority(recompute_protocols=True)
    _assert_environment()
    if os.environ.get(CHILD_TOKEN_ENV) != _child_token(authorization["sha256"], cell):
        raise RunnerError("Child capability is invalid.")
    run_dir, staging, worker_path, guard_path, _log_path = _paths(cell)
    if any(path.exists() or path.is_symlink() for path in (run_dir, staging, worker_path)):
        raise RunnerError("Cell output is not pristine.")
    if guard_path.exists() or guard_path.is_symlink():
        raise RunnerError("Detached guard evidence is present.")
    staging.mkdir(parents=True)
    checkpoint_path = staging / "checkpoints/current.json"
    ledger_path = staging / "result/estimator_ledger.json"
    observation = SRObservationPolicy(
        checkpoint=CheckpointObservation(
            path=checkpoint_path,
            every_controller_rounds=1,
            keep_history_tail=TARGET_HORIZON,
        ),
        estimator_ledger=EstimatorLedgerObservation(path=ledger_path),
        resource_rounds=tuple(range(1, TARGET_HORIZON + 1)),
    )
    problem, protocol = maximum_runner._materialize_cell(cell)
    expected_binding = next(
        row for row in plan["protocol_bindings"] if row["execution_id"] == execution_id
    )
    if _protocol_binding(cell) != expected_binding:
        raise RunnerError("Cell protocol drifted immediately before science.")
    result = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_HORIZON,
            resume=FreshStart(),
            observation=observation,
        ),
    )
    result_mapping = result.to_dict()
    summary_mapping = (
        result.run.paper_i_summary.to_dict()
        if result.run.paper_i_summary is not None
        else None
    )
    completion = maximum_runner.validate_cell_completion(
        cell,
        result=result_mapping,
        summary=summary_mapping,
        checkpoint_path=checkpoint_path,
    )
    current_plan, current_authorization = validate_authority(recompute_protocols=True)
    if current_plan != plan or current_authorization != authorization or _protocol_binding(cell) != expected_binding:
        raise RunnerError("Source or authority drifted during science.")
    result_path = staging / "result/result.json"
    summary_path = staging / "summary/summary.json"
    write_json_exclusive(result_path, result_mapping)
    if summary_mapping is not None:
        write_json_exclusive(summary_path, summary_mapping)
    artifacts = {
        "checkpoint": _artifact_binding(checkpoint_path, staging),
        "estimator_ledger": _artifact_binding(ledger_path, staging),
        "result": _artifact_binding(result_path, staging),
        **(
            {"summary": _artifact_binding(summary_path, staging)}
            if summary_mapping is not None
            else {}
        ),
    }
    manifest = digested(
        {
            "schema": MANIFEST_SCHEMA,
            "status": "passed_authenticated_completion",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": execution_id,
            "cell": asdict(cell),
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "protocol_binding": expected_binding,
            "maximum_controller_rounds": TARGET_HORIZON,
            "accepted_controller_rounds": completion["accepted_controller_rounds"],
            "cell_completion": completion,
            "artifacts": artifacts,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(staging / "execution_manifest.json", manifest)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, run_dir)
    worker = digested(
        {
            "schema": WORKER_SCHEMA,
            "status": "passed_authenticated_completion",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": execution_id,
            "manifest_sha256": manifest["sha256"],
            "artifact_inventory": [
                _artifact_binding(path, RUNTIME_ROOT)
                for path in sorted(run_dir.rglob("*"))
                if path.is_file()
            ],
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(worker_path, worker)
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"Required evidence is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerError(f"Evidence is not an object: {path}")
    return value


def _validate_signed(payload: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    row = dict(payload)
    observed = row.pop("sha256", None)
    if row.get("schema") != schema or observed != canonical_sha256(row):
        raise RunnerError(f"{schema} digest drifted.")
    return dict(payload)


def load_closed_cell(cell: CellSpec) -> dict[str, Any]:
    run_dir, staging, worker_path, guard_path, log_path = _paths(cell)
    if staging.exists() or staging.is_symlink():
        raise RunnerError(f"Cell has preserved partial staging: {cell.execution_id}")
    worker = _validate_signed(_load_json(worker_path), schema=WORKER_SCHEMA)
    guard = _validate_signed(_load_json(guard_path), schema=GUARD_SCHEMA)
    manifest = _validate_signed(
        _load_json(run_dir / "execution_manifest.json"), schema=MANIFEST_SCHEMA
    )
    if (
        worker.get("status") != "passed_authenticated_completion"
        or guard.get("status") != "passed"
        or worker.get("execution_id") != cell.execution_id
        or guard.get("execution_id") != cell.execution_id
        or manifest.get("execution_id") != cell.execution_id
        or worker.get("manifest_sha256") != manifest["sha256"]
        or guard.get("worker_receipt_sha256") != worker["sha256"]
        or guard.get("log_file_binding") != file_binding(log_path)
    ):
        raise RunnerError(f"Closed cell evidence drifted: {cell.execution_id}")
    result = _load_json(run_dir / "result/result.json")
    summary_path = run_dir / "summary/summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else None
    completion = maximum_runner.validate_cell_completion(
        cell,
        result=result,
        summary=summary,
        checkpoint_path=run_dir / "checkpoints/current.json",
    )
    if manifest.get("cell_completion") != completion:
        raise RunnerError(f"Cell completion drifted: {cell.execution_id}")
    return {"cell": cell, "completion": completion, "manifest": manifest, "worker": worker, "guard": guard}


def classify_cell(cell: CellSpec) -> str:
    run_dir, staging, worker_path, guard_path, log_path = _paths(cell)
    states = [path.exists() or path.is_symlink() for path in (run_dir, staging, worker_path, guard_path, log_path)]
    if states == [False, False, False, False, False]:
        return "pristine"
    if states == [True, False, True, True, True]:
        load_closed_cell(cell)
        return "closed"
    return "partial"


def choose_concurrency(*, available_memory_bytes: int, free_disk_bytes: int) -> int:
    if available_memory_bytes >= PAIR_MEMORY_BYTES and free_disk_bytes >= PAIR_DISK_BYTES:
        return 2
    return 1


def _total_rss(process: psutil.Process) -> int:
    try:
        processes = (process, *process.children(recursive=True))
    except psutil.Error:
        processes = (process,)
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except psutil.Error:
            pass
    return total


def _terminate(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _status(status: str, **extra: Any) -> None:
    payload = digested(
        {
            "schema": STATUS_SCHEMA,
            "updated_at": serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "status": status,
            **extra,
        }
    )
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS_PATH)


def _launch(cell: CellSpec, authorization: Mapping[str, Any]) -> dict[str, Any]:
    run_dir, staging, worker_path, guard_path, log_path = _paths(cell)
    del run_dir, staging, worker_path
    if guard_path.exists() or log_path.exists():
        raise RunnerError("Launch evidence is not pristine.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("xb")
    env = os.environ.copy()
    env.update(EXPECTED_ENV)
    env[CHILD_TOKEN_ENV] = _child_token(str(authorization["sha256"]), cell)
    process = subprocess.Popen(
        [sys.executable, "-u", "-B", str(RUNNER_PATH), "--child", cell.execution_id],
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        "cell": cell,
        "process": process,
        "psutil": psutil.Process(process.pid),
        "log_handle": log_handle,
        "started_at": serial_runtime.utc_now(),
        "peak_rss_bytes": 0,
        "minimum_available_memory_bytes": psutil.virtual_memory().available,
        "minimum_free_disk_bytes": shutil.disk_usage(REPO_ROOT).free,
        "stop_reason": None,
    }


def _finish(active: dict[str, Any]) -> bool:
    cell: CellSpec = active["cell"]
    process: subprocess.Popen[Any] = active["process"]
    active["log_handle"].flush()
    active["log_handle"].close()
    returncode = int(process.returncode)
    worker_sha: str | None = None
    passed = False
    error: str | None = None
    try:
        worker_path = _paths(cell)[2]
        if returncode == 0:
            worker = _validate_signed(_load_json(worker_path), schema=WORKER_SCHEMA)
            worker_sha = str(worker["sha256"])
            passed = True
    except Exception as exc:  # preserved in signed guard/status evidence
        error = f"{type(exc).__name__}: {exc}"
    guard = digested(
        {
            "schema": GUARD_SCHEMA,
            "status": "passed" if passed else "failed",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "started_at": active["started_at"],
            "finished_at": serial_runtime.utc_now(),
            "returncode": returncode,
            "stop_reason": active["stop_reason"],
            "peak_rss_bytes": active["peak_rss_bytes"],
            "rss_limit_bytes": NPH7_RSS_LIMIT_BYTES if cell.nph == 7 else NPH3_RSS_LIMIT_BYTES,
            "minimum_available_memory_bytes": active["minimum_available_memory_bytes"],
            "minimum_free_disk_bytes": active["minimum_free_disk_bytes"],
            "worker_receipt_sha256": worker_sha,
            "validation_error": error,
            "log_file_binding": file_binding(_paths(cell)[4]),
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(_paths(cell)[3], guard)
    if passed:
        load_closed_cell(cell)
    return passed


def _acquire_lock():
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RunnerError("Remaining-five campaign lock is held.") from exc
    return handle


def run_campaign() -> int:
    plan, authorization = validate_authority(recompute_protocols=True)
    _assert_environment()
    lock = _acquire_lock()
    try:
        states = {cell.execution_id: classify_cell(cell) for cell in CELL_SPECS}
        pending = [cell for cell in CELL_SPECS if states[cell.execution_id] == "pristine"]
        failures = [cell.execution_id for cell in CELL_SPECS if states[cell.execution_id] == "partial"]
        closed = [cell.execution_id for cell in CELL_SPECS if states[cell.execution_id] == "closed"]
        free_disk = shutil.disk_usage(REPO_ROOT).free
        if not closed and free_disk < INITIAL_DISK_BYTES:
            raise RunnerError("Initial 31-GiB campaign capacity gate failed.")
        concurrency = choose_concurrency(
            available_memory_bytes=psutil.virtual_memory().available,
            free_disk_bytes=free_disk,
        )
        active: dict[str, dict[str, Any]] = {}
        _status("running", concurrency=concurrency, pending=[c.execution_id for c in pending], closed=closed, failures=failures)
        while pending or active:
            while pending and len(active) < concurrency:
                if psutil.virtual_memory().available < SINGLE_MEMORY_BYTES or shutil.disk_usage(REPO_ROOT).free < SINGLE_DISK_BYTES:
                    break
                cell = pending.pop(0)
                active[cell.execution_id] = _launch(cell, authorization)
            if not active:
                if pending:
                    _status("blocked_capacity", pending=[c.execution_id for c in pending], closed=closed, failures=failures)
                    return 2
                break
            time.sleep(POLL_SECONDS)
            for execution_id, item in list(active.items()):
                process: subprocess.Popen[Any] = item["process"]
                cell: CellSpec = item["cell"]
                rss = _total_rss(item["psutil"])
                available = int(psutil.virtual_memory().available)
                disk = int(shutil.disk_usage(REPO_ROOT).free)
                item["peak_rss_bytes"] = max(item["peak_rss_bytes"], rss)
                item["minimum_available_memory_bytes"] = min(item["minimum_available_memory_bytes"], available)
                item["minimum_free_disk_bytes"] = min(item["minimum_free_disk_bytes"], disk)
                limit = NPH7_RSS_LIMIT_BYTES if cell.nph == 7 else NPH3_RSS_LIMIT_BYTES
                if process.poll() is None and (rss > limit or available < AVAILABLE_MEMORY_FLOOR_BYTES or disk < FREE_DISK_FLOOR_BYTES):
                    item["stop_reason"] = "rss_limit" if rss > limit else ("memory_floor" if available < AVAILABLE_MEMORY_FLOOR_BYTES else "disk_floor")
                    _terminate(process)
                if process.poll() is not None:
                    passed = _finish(item)
                    if passed:
                        closed.append(execution_id)
                    else:
                        failures.append(execution_id)
                    del active[execution_id]
                    _status("running", concurrency=concurrency, active=list(active), pending=[c.execution_id for c in pending], closed=closed, failures=failures)
        validate_authority(recompute_protocols=True)
        final_status = "passed" if not failures and len(closed) == len(CELL_SPECS) else "completed_with_failures"
        _status(final_status, closed=closed, failures=failures, plan_sha256=plan["sha256"], authorization_sha256=authorization["sha256"])
        return 0 if final_status == "passed" else 1
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def preflight() -> dict[str, Any]:
    inventory = _source_inventory()
    bindings = [_protocol_binding(cell) for cell in CELL_SPECS]
    return {
        "campaign_id": CAMPAIGN_ID,
        "cell_count": len(CELL_SPECS),
        "execution_ids": [cell.execution_id for cell in CELL_SPECS],
        "regimes": [cell.regime_id for cell in CELL_SPECS],
        "insertion_policies": sorted({cell.insertion_policy for cell in CELL_SPECS}),
        "route_variants": sorted({cell.route_variant for cell in CELL_SPECS}),
        "source_implementation_inventory_sha256": inventory["sha256"],
        "protocol_sha256": [row["protocol_sha256"] for row in bindings],
        "plan_present": PLAN_PATH.is_file(),
        "authorization_present": AUTHORIZATION_PATH.is_file(),
        "runtime_present": RUNTIME_ROOT.exists(),
        "scientific_execution_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--prepare-plan", action="store_true")
    actions.add_argument("--authorize", action="store_true")
    actions.add_argument("--run-campaign", action="store_true")
    actions.add_argument("--child")
    args = parser.parse_args(argv)
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    if args.prepare_plan:
        print(json.dumps(prepare_plan(), indent=2, sort_keys=True))
        return 0
    if args.authorize:
        print(json.dumps(authorize(), indent=2, sort_keys=True))
        return 0
    if args.run_campaign:
        return run_campaign()
    return run_child(str(args.child))


if __name__ == "__main__":
    raise SystemExit(main())
