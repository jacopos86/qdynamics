#!/usr/bin/env python3
"""Run the source-locked native Phase-0 eight-arm plateau canary."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import psutil

_BOOTSTRAP_RUNNER_PATH = Path(__file__).resolve()
_BOOTSTRAP_REPO_ROOT = _BOOTSTRAP_RUNNER_PATH.parents[2]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from pipelines.static_adapt.ra_adapt.contracts import RAAdaptOperationalControls
from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
    PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
    PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
    build_paper_i_ra_strong_weak_nph3_problem,
    build_paper_i_ra_strong_weak_plateau_k5_request,
    materialize_paper_i_ra_semantic_protocol,
    preflight_paper_i_ra_strong_weak_plateau_k5,
    semantic_closure_route_identity,
    semantic_closure_source_implementation_inventory,
)
from pipelines.static_adapt.sr_snake.contracts import (
    CheckpointObservation,
    EstimatorLedgerObservation,
    FreshStart,
    SRObservationPolicy,
)


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[2]
CAMPAIGN_ID = "paper_i_native_phase0_plateau_eight_arm_k5_20260816_v1"
AUTHORITY_DIR = RUNNER_PATH.parent / f"{CAMPAIGN_ID}_authority"
PLAN_PATH = AUTHORITY_DIR / "plan.json"
AUTHORIZATION_PATH = AUTHORITY_DIR / "authorization.json"
RUNTIME_ROOT = REPO_ROOT / "output/local_runs" / CAMPAIGN_ID
RUNS_ROOT = RUNTIME_ROOT / "runs"
STAGING_ROOT = RUNTIME_ROOT / "in_progress"
RECEIPTS_ROOT = RUNTIME_ROOT / "worker_receipts"
GUARD_ROOT = RUNTIME_ROOT / "guard_receipts"
STATUS_PATH = RUNTIME_ROOT / "status.json"
LOCK_PATH = RUNTIME_ROOT / "campaign.lock"
REPORT_JSON = RUNTIME_ROOT / "comparison.json"
REPORT_CSV = RUNTIME_ROOT / "comparison.csv"
REPORT_MD = RUNTIME_ROOT / "comparison.md"
TERMINAL_PATH = RUNTIME_ROOT / "terminal_matrix_receipt.json"

TARGET_HORIZON = 5
SHORTLIST_CAP = 24
PHASE_I_MAXIMUM = 24
PHASE_II_MAXIMUM = 12
LAUNCH_AVAILABLE_MEMORY_BYTES = 5 * 1024**3
LAUNCH_FREE_DISK_BYTES = 10 * 1024**3
CAPACITY_WAIT_SECONDS = 5 * 60
CHILD_RSS_LIMIT_BYTES = 8 * 1024**3
AVAILABLE_MEMORY_FLOOR_BYTES = 2 * 1024**3
FREE_DISK_FLOOR_BYTES = 2 * 1024**3
POLL_SECONDS = 1.0
CAPACITY_POLL_SECONDS = 10.0
CHILD_TOKEN_ENV = "PAPER_I_NATIVE_PHASE0_EIGHT_ARM_CHILD_TOKEN"

PLAN_SCHEMA = "paper_i_native_phase0_eight_arm_plateau_plan_v1"
AUTH_SCHEMA = "paper_i_native_phase0_eight_arm_plateau_authorization_v1"
MANIFEST_SCHEMA = "paper_i_native_phase0_eight_arm_execution_manifest_v1"
WORKER_SCHEMA = "paper_i_native_phase0_eight_arm_worker_receipt_v1"
GUARD_SCHEMA = "paper_i_native_phase0_eight_arm_guard_receipt_v1"
REPORT_SCHEMA = "paper_i_native_phase0_eight_arm_comparison_v1"
TERMINAL_SCHEMA = "paper_i_native_phase0_eight_arm_terminal_matrix_v1"
STATUS_SCHEMA = "paper_i_native_phase0_eight_arm_status_v1"

EXPECTED_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "CUDA_VISIBLE_DEVICES": "",
    "STATIC_ADAPT_HH_POOL_CACHE": "off",
    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
}


class RunnerError(RuntimeError):
    """Raised when the canary cannot preserve its closed contract."""


@dataclass(frozen=True, slots=True)
class CellSpec:
    ordinal: int
    placement: str
    score: str
    cardinality: str
    route_variant: str
    execution_id: str
    insertion_policy: str = "plateau_commutation"
    horizon: int = TARGET_HORIZON
    regime_id: str = "strong_weak_u8"
    nph: int = 3


def _cell(
    ordinal: int,
    placement: str,
    score: str,
    cardinality: str,
    route_variant: str,
) -> CellSpec:
    return CellSpec(
        ordinal=ordinal,
        placement=placement,
        score=score,
        cardinality=cardinality,
        route_variant=route_variant,
        execution_id=(
            f"native_phase0_plateau_k5__strong_weak_u8__nph3__"
            f"{placement}__{score}__{cardinality}"
        ),
    )


CELL_SPECS = (
    _cell(1, "generator_first", "gradient", "fixed24", PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2),
    _cell(2, "position_aware", "gradient", "fixed24", PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1),
    _cell(3, "generator_first", "gradient", "adaptive", PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2),
    _cell(4, "position_aware", "gradient", "adaptive", PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1),
    _cell(5, "generator_first", "proxy", "fixed24", PAPER_I_RA_PHASE0_PROXY_FIXED24_V2),
    _cell(6, "position_aware", "proxy", "fixed24", PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1),
    _cell(7, "generator_first", "proxy", "adaptive", PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2),
    _cell(8, "position_aware", "proxy", "adaptive", PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_status(payload: Mapping[str, Any]) -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_name(f".{STATUS_PATH.name}.{os.getpid()}.tmp")
    body = digested(
        {
            "schema": STATUS_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "updated_at": utc_now(),
            **dict(payload),
        }
    )
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(body, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, STATUS_PATH)


def load_digested(path: Path, *, schema: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    observed = payload.pop("sha256", None)
    if payload.get("schema") != schema or observed != canonical_sha256(payload):
        raise RunnerError(f"Invalid digested artifact: {path}")
    payload["sha256"] = observed
    return payload


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _protocol_binding(cell: CellSpec) -> dict[str, Any]:
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    request = build_paper_i_ra_strong_weak_plateau_k5_request(cell.route_variant)
    preflight = preflight_paper_i_ra_strong_weak_plateau_k5(problem, request)
    if preflight.execution_authorized is not False:
        raise RunnerError("Inert semantic preflight unexpectedly authorized execution.")
    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
    identity = semantic_closure_route_identity(cell.route_variant)
    if protocol.route_contract is None or protocol.bundle_materialization is None:
        raise RunnerError("Semantic materialization is incomplete.")
    native = protocol.route_contract["native_semantic_contract"]
    execution = protocol.route_contract["execution_settings"]
    if (
        protocol.horizon != TARGET_HORIZON
        or protocol.optimizer != "powell"
        or protocol.optimizer_maxiter != 200
        or protocol.seeds != {"adapt": 7, "transpiler": 7}
        or execution.get("adapt_inner_optimizer") != "POWELL"
        or execution.get("adapt_maxiter") != 200
        or execution.get("adapt_scipy_maxfev") != 0
        or execution.get("phase1_shortlist_size") != PHASE_I_MAXIMUM
        or execution.get("phase2_shortlist_size") != PHASE_II_MAXIMUM
        or native.get("optimizer_options")
        != {"xtol": 1.0e-4, "ftol": 1.0e-8, "maxfev": None}
        or native.get("phase_shortlist_maxima")
        != {"phase_i": PHASE_I_MAXIMUM, "phase_ii": PHASE_II_MAXIMUM}
        or native.get("phase_frontier_ratios")
        != {"phase_ii": 0.9, "phase_iii": 0.9}
    ):
        raise RunnerError("Optimizer or downstream shortlist contract drifted.")
    return {
        "execution_id": cell.execution_id,
        "route_variant": cell.route_variant,
        "algorithm_id": identity.algorithm_id,
        "route_id": identity.route_id,
        "semantic_implementation_version": identity.semantic_implementation_version,
        "route_contract_sha256": protocol.route_contract["sha256"],
        "protocol_sha256": protocol.sha256,
        "bundle_id": protocol.bundle_id,
        "bundle_manifest_sha256": protocol.bundle_manifest_sha256,
        "materialization_receipt_sha256": protocol.bundle_materialization.sha256,
        "source_locks": dict(protocol.source_locks),
        "execution_authorized_in_serialized_protocol": protocol.execution_authorized,
    }


def build_plan() -> dict[str, Any]:
    inventory = semantic_closure_source_implementation_inventory()
    protocols = [_protocol_binding(cell) for cell in CELL_SPECS]
    inventory_hashes = {
        row["source_locks"]["implementation_source_inventory_sha256"]
        for row in protocols
    }
    if inventory_hashes != {inventory["sha256"]}:
        raise RunnerError("Eight cells do not share one source inventory.")
    return digested(
        {
            "schema": PLAN_SCHEMA,
            "created_at": utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "run_class": "local_diagnostic_non_adopted",
            "problem": {"regime_id": "strong_weak_u8", "nph": 3},
            "insertion_policy": "plateau_commutation",
            "target_horizon": TARGET_HORIZON,
            "fixed_serial_order": [cell.execution_id for cell in CELL_SPECS],
            "cells": [asdict(cell) for cell in CELL_SPECS],
            "protocol_bindings": protocols,
            "source_implementation_inventory_sha256": inventory["sha256"],
            "source_implementation_file_count": inventory["source_count"],
            "runner": file_binding(RUNNER_PATH),
            "optimizer": {
                "name": "powell",
                "xtol": 1.0e-4,
                "ftol": 1.0e-8,
                "maxiter": 200,
                "maxfev": None,
            },
            "seeds": {"adapt": 7, "transpiler": 7},
            "frontier_ratios": {"phase_ii": 0.9, "phase_iii": 0.9},
            "shortlist_maxima": {"phase_i": 24, "phase_ii": 12},
            "maximum_concurrency": 1,
            "capacity": {
                "maximum_wait_seconds": CAPACITY_WAIT_SECONDS,
                "launch_available_memory_bytes": LAUNCH_AVAILABLE_MEMORY_BYTES,
                "launch_free_disk_bytes": LAUNCH_FREE_DISK_BYTES,
                "child_rss_limit_bytes": CHILD_RSS_LIMIT_BYTES,
                "runtime_available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
                "runtime_free_disk_floor_bytes": FREE_DISK_FLOOR_BYTES,
            },
            "runtime_environment": dict(EXPECTED_ENV),
            "execution_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def prepare_plan() -> dict[str, Any]:
    if AUTHORITY_DIR.exists() or AUTHORITY_DIR.is_symlink():
        raise RunnerError(f"Authority path already exists: {AUTHORITY_DIR}")
    plan = build_plan()
    AUTHORITY_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{AUTHORITY_DIR.name}.", dir=AUTHORITY_DIR.parent))
    try:
        write_json_exclusive(staging / "plan.json", plan)
        os.rename(staging, AUTHORITY_DIR)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return plan


def validate_plan(*, recompute_protocols: bool) -> dict[str, Any]:
    plan = load_digested(PLAN_PATH, schema=PLAN_SCHEMA)
    inventory = semantic_closure_source_implementation_inventory()
    if (
        plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("cells") != [asdict(cell) for cell in CELL_SPECS]
        or plan.get("fixed_serial_order") != [cell.execution_id for cell in CELL_SPECS]
        or plan.get("source_implementation_inventory_sha256") != inventory["sha256"]
        or plan.get("runner") != file_binding(RUNNER_PATH)
        or plan.get("maximum_concurrency") != 1
        or plan.get("execution_authorized") is not False
        or plan.get("paper_adoption_authorized") is not False
        or plan.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError("Eight-arm plan drifted.")
    if recompute_protocols and plan.get("protocol_bindings") != [
        _protocol_binding(cell) for cell in CELL_SPECS
    ]:
        raise RunnerError("Eight-arm protocol bindings drifted.")
    return plan


def authorize() -> dict[str, Any]:
    plan = validate_plan(recompute_protocols=True)
    if AUTHORIZATION_PATH.exists() or AUTHORIZATION_PATH.is_symlink():
        raise RunnerError("Authorization receipt already exists.")
    receipt = digested(
        {
            "schema": AUTH_SCHEMA,
            "created_at": utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "authorization_basis": "explicit_current_user_implementation_and_run_request",
            "plan_sha256": plan["sha256"],
            "runner_sha256": plan["runner"]["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "execution_ids": [cell.execution_id for cell in CELL_SPECS],
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(AUTHORIZATION_PATH, receipt)
    return receipt


def validate_authority(*, recompute_protocols: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = validate_plan(recompute_protocols=recompute_protocols)
    authorization = load_digested(AUTHORIZATION_PATH, schema=AUTH_SCHEMA)
    if (
        authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("plan_sha256") != plan["sha256"]
        or authorization.get("runner_sha256") != plan["runner"]["sha256"]
        or authorization.get("source_implementation_inventory_sha256")
        != plan["source_implementation_inventory_sha256"]
        or authorization.get("execution_ids") != [cell.execution_id for cell in CELL_SPECS]
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("paper_adoption_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError("Eight-arm authorization drifted.")
    return plan, authorization


def capacity_snapshot(memory: int, disk: int) -> dict[str, Any]:
    ready = memory >= LAUNCH_AVAILABLE_MEMORY_BYTES and disk >= LAUNCH_FREE_DISK_BYTES
    return {
        "available_memory_bytes": int(memory),
        "free_disk_bytes": int(disk),
        "launch_available_memory_bytes": LAUNCH_AVAILABLE_MEMORY_BYTES,
        "launch_free_disk_bytes": LAUNCH_FREE_DISK_BYTES,
        "launch_ready": ready,
    }


def wait_for_capacity(
    *,
    maximum_wait_seconds: float = CAPACITY_WAIT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    memory_supplier: Callable[[], int] = lambda: int(psutil.virtual_memory().available),
    disk_supplier: Callable[[], int] = lambda: int(shutil.disk_usage(REPO_ROOT).free),
) -> dict[str, Any]:
    started = clock()
    while True:
        snapshot = capacity_snapshot(memory_supplier(), disk_supplier())
        elapsed = clock() - started
        if snapshot["launch_ready"]:
            return {**snapshot, "elapsed_wait_seconds": elapsed, "status": "ready"}
        if elapsed >= maximum_wait_seconds:
            return {**snapshot, "elapsed_wait_seconds": elapsed, "status": "blocked_capacity"}
        sleeper(CAPACITY_POLL_SECONDS)


def assert_environment() -> None:
    drift = {
        key: {"expected": value, "observed": os.environ.get(key)}
        for key, value in EXPECTED_ENV.items()
        if os.environ.get(key) != value
    }
    if drift:
        raise RunnerError(f"Numerical environment drifted: {drift}")


def cell_paths(cell: CellSpec) -> tuple[Path, Path, Path, Path]:
    return (
        RUNS_ROOT / cell.execution_id,
        STAGING_ROOT / cell.execution_id,
        RECEIPTS_ROOT / f"{cell.execution_id}.json",
        GUARD_ROOT / f"{cell.execution_id}.json",
    )


def child_token(authorization_sha256: str, cell: CellSpec) -> str:
    return canonical_sha256(
        {
            "campaign_id": CAMPAIGN_ID,
            "authorization_sha256": authorization_sha256,
            "execution_id": cell.execution_id,
            "route_variant": cell.route_variant,
        }
    )


def _cell_by_execution_id(execution_id: str) -> CellSpec:
    matches = [cell for cell in CELL_SPECS if cell.execution_id == execution_id]
    if len(matches) != 1:
        raise RunnerError("Unknown eight-arm execution ID.")
    return matches[0]


def _artifact_binding(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def run_child(execution_id: str) -> int:
    cell = _cell_by_execution_id(execution_id)
    plan, authorization = validate_authority()
    assert_environment()
    if os.environ.get(CHILD_TOKEN_ENV) != child_token(authorization["sha256"], cell):
        raise RunnerError("Child capability is invalid.")
    run_dir, staging, receipt_path, _guard_path = cell_paths(cell)
    if any(path.exists() or path.is_symlink() for path in (run_dir, staging, receipt_path)):
        raise RunnerError("Cell output is not pristine.")
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
    problem = build_paper_i_ra_strong_weak_nph3_problem()
    request = build_paper_i_ra_strong_weak_plateau_k5_request(cell.route_variant)
    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
    expected_binding = next(
        row for row in plan["protocol_bindings"] if row["execution_id"] == execution_id
    )
    if _protocol_binding(cell) != expected_binding:
        raise RunnerError("Cell protocol drifted immediately before execution.")
    result = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_HORIZON,
            resume=FreshStart(),
            observation=observation,
        ),
    )
    rounds = len(result.run.accepted_trajectory)
    if rounds != TARGET_HORIZON or result.run.paper_i_summary is None:
        raise RunnerError(f"Cell ended at {rounds}; exact k=5 is required.")
    result_path = staging / "result/result.json"
    summary_path = staging / "summary/summary.json"
    write_json_exclusive(result_path, result.to_dict())
    write_json_exclusive(summary_path, result.run.paper_i_summary.to_dict())
    artifacts = {
        role: _artifact_binding(path, staging)
        for role, path in {
            "checkpoint": checkpoint_path,
            "estimator_ledger": ledger_path,
            "result": result_path,
            "summary": summary_path,
        }.items()
    }
    manifest = digested(
        {
            "schema": MANIFEST_SCHEMA,
            "status": "passed_k5",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": execution_id,
            "cell": asdict(cell),
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "protocol_binding": expected_binding,
            "controller_rounds_completed": rounds,
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
            "status": "passed_k5",
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
    write_json_exclusive(receipt_path, worker)
    return 0


def total_rss(process: psutil.Process) -> int:
    total = 0
    for candidate in (process, *process.children(recursive=True)):
        try:
            total += int(candidate.memory_info().rss)
        except psutil.Error:
            pass
    return total


def monitor_cell(cell: CellSpec, authorization: Mapping[str, Any]) -> dict[str, Any]:
    environment = {**os.environ, **EXPECTED_ENV}
    environment[CHILD_TOKEN_ENV] = child_token(authorization["sha256"], cell)
    command = [sys.executable, "-u", "-B", str(RUNNER_PATH), "--child", cell.execution_id]
    child = subprocess.Popen(command, cwd=REPO_ROOT, env=environment)
    process = psutil.Process(child.pid)
    started = time.monotonic()
    peak_rss = 0
    minimum_memory = int(psutil.virtual_memory().available)
    minimum_disk = int(shutil.disk_usage(REPO_ROOT).free)
    stop_reason: str | None = None
    while child.poll() is None:
        rss = total_rss(process)
        memory = int(psutil.virtual_memory().available)
        disk = int(shutil.disk_usage(REPO_ROOT).free)
        peak_rss = max(peak_rss, rss)
        minimum_memory = min(minimum_memory, memory)
        minimum_disk = min(minimum_disk, disk)
        if rss > CHILD_RSS_LIMIT_BYTES:
            stop_reason = "rss_limit_breached"
        elif memory < AVAILABLE_MEMORY_FLOOR_BYTES:
            stop_reason = "available_memory_floor_breached"
        elif disk < FREE_DISK_FLOOR_BYTES:
            stop_reason = "free_disk_floor_breached"
        write_status(
            {
                "status": "running_cell",
                "cell_index": cell.ordinal,
                "cell_count": len(CELL_SPECS),
                "execution_id": cell.execution_id,
                "child_pid": child.pid,
                "elapsed_seconds": time.monotonic() - started,
                "current_rss_bytes": rss,
                "peak_rss_bytes": peak_rss,
                "available_memory_bytes": memory,
                "free_disk_bytes": disk,
                "stop_reason": stop_reason,
            }
        )
        if stop_reason is not None:
            child.terminate()
            break
        time.sleep(POLL_SECONDS)
    try:
        returncode = child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        returncode = child.wait(timeout=30)
    worker_path = cell_paths(cell)[2]
    worker_sha = (
        load_digested(worker_path, schema=WORKER_SCHEMA)["sha256"]
        if returncode == 0 and worker_path.is_file()
        else None
    )
    guard = digested(
        {
            "schema": GUARD_SCHEMA,
            "status": "passed" if returncode == 0 and stop_reason is None else "failed",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "returncode": returncode,
            "stop_reason": stop_reason,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss,
            "minimum_available_memory_bytes": minimum_memory,
            "minimum_free_disk_bytes": minimum_disk,
            "worker_receipt_sha256": worker_sha,
        }
    )
    write_json_exclusive(cell_paths(cell)[3], guard)
    if guard["status"] != "passed":
        raise RunnerError(
            f"Cell {cell.execution_id} failed rc={returncode}, reason={stop_reason}."
        )
    return guard


def _phase_counts(round_receipt: Mapping[str, Any]) -> tuple[int, int, int, int]:
    phases = round_receipt["scored_insertion_position_population"]["phases"]
    phase_i_input = int(phases[0]["population_count"])
    phase_ii_input = int(phases[1]["population_count"])
    phase_iii_input = int(phases[2]["population_count"])
    return phase_i_input, phase_ii_input, phase_ii_input, phase_iii_input


def _require_phase_maxima(
    *,
    phase_i_retained: int,
    phase_ii_retained: int,
) -> None:
    if not 0 < phase_i_retained <= PHASE_I_MAXIMUM:
        raise RunnerError(
            "Observed Phase-I retention violates the separately bound maximum 24."
        )
    if not 0 < phase_ii_retained <= PHASE_II_MAXIMUM:
        raise RunnerError(
            "Observed Phase-II retention violates the separately bound maximum 12."
        )


def report_rows(cell: CellSpec, result: Mapping[str, Any], summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    accepted = result["scientific_receipts"]["accepted_round_receipts"]
    replay = result["run"]["scientific_replay"]
    errors = summary["accepted_error_trace"]
    requested = {int(row["controller_round"]): row for row in summary["requested_rounds"]}
    transitions = {int(row["controller_round"]): row for row in result["run"]["accepted_transitions"]}
    rows: list[dict[str, Any]] = []
    for ordinal, (round_receipt, replay_row, error_row) in enumerate(
        zip(accepted, replay, errors, strict=True), start=1
    ):
        phase0 = round_receipt["ra_gradient_phase0_shortlist"]
        phase_i_input, phase_i_retained, phase_ii_input, phase_ii_retained = _phase_counts(round_receipt)
        _require_phase_maxima(
            phase_i_retained=phase_i_retained,
            phase_ii_retained=phase_ii_retained,
        )
        plateau = round_receipt.get("insertion_commutation_plateau", {})
        prefix = requested[ordinal]
        resources = prefix.get("resources") or {}
        transition = transitions[ordinal]
        if int(transition["controller_round"]) != ordinal:
            raise RunnerError("Accepted transition order drifted.")
        work = prefix["algorithmic_work"]
        rows.append(
            {
                "execution_id": cell.execution_id,
                "ordinal": cell.ordinal,
                "placement": cell.placement,
                "score": cell.score,
                "cardinality": cell.cardinality,
                "controller_round": ordinal,
                "energy": float(error_row["accepted_energy"]),
                "absolute_delta_e": float(error_row["absolute_energy_error"]),
                "plateau_state": "open" if plateau.get("domain_open") is True else "closed",
                "phase0_population_count": int(phase0["input_candidate_count"]),
                "phase0_retained_count": int(phase0["retained_candidate_count"]),
                "phase_i_input_count": phase_i_input,
                "phase_i_retained_count": phase_i_retained,
                "phase_ii_input_count": phase_ii_input,
                "phase_ii_retained_count": phase_ii_retained,
                "selected_generator": str(replay_row["generator_id"]),
                "selected_operator": str(replay_row["selected_operator"]),
                "selected_position": int(replay_row["selected_position"]),
                "s_alg": int(work["s_alg"]),
                "n_h_outer": int(work["components"]["n_h_outer"]),
                "n_h_refit": int(work["components"]["n_h_refit"]),
                "n_grad": int(work["components"]["n_grad"]),
                "n_metric": int(work["components"]["n_metric"]),
                "n2q": int(resources["compiled_two_qubit_count"]),
                "d2q": int(resources["compiled_two_qubit_depth"]),
                "dc": int(resources["compiled_total_depth"]),
                "checkpoint_sha256": str(replay_row["checkpoint"]["checkpoint_sha256"]),
            }
        )
    return rows


def load_closed_cell(
    cell: CellSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> tuple[CellSpec, Mapping[str, Any], Mapping[str, Any], str, str]:
    run_dir, staging, worker_path, guard_path = cell_paths(cell)
    if staging.exists() or staging.is_symlink():
        raise RunnerError(f"Cell has a preserved partial attempt: {cell.execution_id}")
    manifest_path = run_dir / "execution_manifest.json"
    manifest = load_digested(manifest_path, schema=MANIFEST_SCHEMA)
    worker = load_digested(worker_path, schema=WORKER_SCHEMA)
    guard = load_digested(guard_path, schema=GUARD_SCHEMA)
    expected_binding = next(
        row
        for row in plan["protocol_bindings"]
        if row["execution_id"] == cell.execution_id
    )
    if (
        manifest.get("status") != "passed_k5"
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("execution_id") != cell.execution_id
        or manifest.get("cell") != asdict(cell)
        or manifest.get("plan_sha256") != plan["sha256"]
        or manifest.get("authorization_sha256") != authorization["sha256"]
        or manifest.get("source_implementation_inventory_sha256")
        != plan["source_implementation_inventory_sha256"]
        or manifest.get("protocol_binding") != expected_binding
        or manifest.get("controller_rounds_completed") != TARGET_HORIZON
        or manifest.get("submission_authorized") is not False
        or manifest.get("paper_adoption_authorized") is not False
        or manifest.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError(f"Cell manifest drifted: {cell.execution_id}")
    for binding in manifest.get("artifacts", {}).values():
        artifact = run_dir / str(binding["path"])
        if binding != _artifact_binding(artifact, run_dir):
            raise RunnerError(f"Cell artifact drifted: {artifact}")
    expected_inventory = [
        _artifact_binding(path, RUNTIME_ROOT)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    ]
    if (
        worker.get("status") != "passed_k5"
        or worker.get("campaign_id") != CAMPAIGN_ID
        or worker.get("execution_id") != cell.execution_id
        or worker.get("manifest_sha256") != manifest["sha256"]
        or worker.get("artifact_inventory") != expected_inventory
        or worker.get("submission_authorized") is not False
        or worker.get("paper_adoption_authorized") is not False
        or worker.get("paper_evidence_adoption_authorized") is not False
        or guard.get("status") != "passed"
        or guard.get("campaign_id") != CAMPAIGN_ID
        or guard.get("execution_id") != cell.execution_id
        or guard.get("returncode") != 0
        or guard.get("stop_reason") is not None
        or guard.get("worker_receipt_sha256") != worker["sha256"]
    ):
        raise RunnerError(f"Cell worker/guard closure drifted: {cell.execution_id}")
    with (run_dir / "result/result.json").open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    with (run_dir / "summary/summary.json").open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if len(report_rows(cell, result, summary)) != TARGET_HORIZON:
        raise RunnerError(f"Cell report does not contain exact k=5: {cell.execution_id}")
    return cell, result, summary, worker["sha256"], guard["sha256"]


def factorial_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    terminals = [row for row in rows if int(row["controller_round"]) == TARGET_HORIZON]
    metrics = ("energy", "absolute_delta_e", "s_alg", "n2q", "d2q", "dc")
    signs = {
        "placement": lambda row: 1 if row["placement"] == "position_aware" else -1,
        "score": lambda row: 1 if row["score"] == "proxy" else -1,
        "cardinality": lambda row: 1 if row["cardinality"] == "adaptive" else -1,
    }
    placement_activated = any(row["plateau_state"] == "open" for row in rows)
    effects: dict[str, Any] = {}
    factors = tuple(signs)
    terms = [
        (factor, (factor,)) for factor in factors
    ] + [
        ("placement:score", ("placement", "score")),
        ("placement:cardinality", ("placement", "cardinality")),
        ("score:cardinality", ("score", "cardinality")),
        ("placement:score:cardinality", factors),
    ]
    for label, term in terms:
        if "placement" in term and not placement_activated:
            effects[label] = {"status": "not_activated", "metrics": None}
            continue
        effects[label] = {
            "status": "estimated",
            "metrics": {
                metric: math.fsum(
                    math.prod(signs[factor](row) for factor in term) * float(row[metric])
                    for row in terminals
                ) / 4.0
                for metric in metrics
            },
        }
    return effects


def build_comparison(
    cells: Sequence[tuple[CellSpec, Mapping[str, Any], Mapping[str, Any], str, str]],
) -> tuple[dict[str, Any], str, str]:
    rows = [
        row
        for cell, result, summary, _worker_sha, _guard_sha in cells
        for row in report_rows(cell, result, summary)
    ]
    placement_activated = any(row["plateau_state"] == "open" for row in rows)
    agreement: list[dict[str, Any]] = []
    by_key = {
        (row["placement"], row["score"], row["cardinality"], row["controller_round"]): row
        for row in rows
    }
    for score in ("gradient", "proxy"):
        for cardinality in ("fixed24", "adaptive"):
            for round_index in range(1, TARGET_HORIZON + 1):
                generator = by_key[("generator_first", score, cardinality, round_index)]
                position = by_key[("position_aware", score, cardinality, round_index)]
                agreement.append(
                    {
                        "score": score,
                        "cardinality": cardinality,
                        "controller_round": round_index,
                        "status": "agree" if (
                            generator["selected_generator"], generator["selected_position"]
                        ) == (
                            position["selected_generator"], position["selected_position"]
                        ) else "diverge",
                    }
                )
    payload = digested(
        {
            "schema": REPORT_SCHEMA,
            "status": "passed_eight_k5",
            "campaign_id": CAMPAIGN_ID,
            "placement_factor_status": "activated" if placement_activated else "not_activated",
            "rows": rows,
            "factorial_effects_at_k5": factorial_effects(rows),
            "selected_record_agreement": agreement,
            "cell_receipts": [
                {
                    "execution_id": cell.execution_id,
                    "worker_receipt_sha256": worker_sha,
                    "guard_receipt_sha256": guard_sha,
                }
                for cell, _result, _summary, worker_sha, guard_sha in cells
            ],
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    columns = list(rows[0])
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    markdown = [
        "# Native Phase-0 eight-arm plateau canary",
        "",
        "Diagnostic only; no manuscript or evidence adoption.",
        "",
        f"Placement factor: **{payload['placement_factor_status']}**.",
        "",
        "| arm | k | E | |ΔE| | plateau | P0 in/keep | PI in/keep | PII in/keep | selected | S_alg | N2q | D2q | Dc |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        arm = f"{row['placement']}/{row['score']}/{row['cardinality']}"
        selected = f"{row['selected_generator']}@{row['selected_position']}"
        markdown.append(
            f"| {arm} | {row['controller_round']} | {row['energy']:.12g} | "
            f"{row['absolute_delta_e']:.4e} | {row['plateau_state']} | "
            f"{row['phase0_population_count']}/{row['phase0_retained_count']} | "
            f"{row['phase_i_input_count']}/{row['phase_i_retained_count']} | "
            f"{row['phase_ii_input_count']}/{row['phase_ii_retained_count']} | "
            f"{selected} | {row['s_alg']} | {row['n2q']} | {row['d2q']} | {row['dc']} |"
        )
    markdown.extend(["", "Factorial effects and categorical agreements are authoritative in `comparison.json`.", ""])
    return payload, csv_stream.getvalue(), "\n".join(markdown)


def _publish_or_validate_json(path: Path, payload: Mapping[str, Any], *, schema: str) -> None:
    if path.exists() or path.is_symlink():
        if load_digested(path, schema=schema) != dict(payload):
            raise RunnerError(f"Existing immutable JSON differs: {path}")
        return
    write_json_exclusive(path, payload)


def _publish_or_validate_text(path: Path, body: str) -> None:
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.read_text(encoding="utf-8") != body:
            raise RunnerError(f"Existing immutable text differs: {path}")
        return
    write_text_exclusive(path, body)


def validate_terminal_matrix(
    *,
    plan: Mapping[str, Any] | None = None,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if plan is None or authorization is None:
        plan, authorization = validate_authority(recompute_protocols=True)
    closed = [
        load_closed_cell(cell, plan=plan, authorization=authorization)
        for cell in CELL_SPECS
    ]
    comparison, csv_text, markdown = build_comparison(closed)
    if load_digested(REPORT_JSON, schema=REPORT_SCHEMA) != comparison:
        raise RunnerError("Terminal comparison JSON failed recomputation.")
    if REPORT_CSV.read_text(encoding="utf-8") != csv_text:
        raise RunnerError("Terminal comparison CSV failed recomputation.")
    if REPORT_MD.read_text(encoding="utf-8") != markdown:
        raise RunnerError("Terminal comparison Markdown failed recomputation.")
    terminal = load_digested(TERMINAL_PATH, schema=TERMINAL_SCHEMA)
    expected = {
        "schema": TERMINAL_SCHEMA,
        "status": "passed_eight_k5",
        "campaign_id": CAMPAIGN_ID,
        "plan_sha256": plan["sha256"],
        "authorization_sha256": authorization["sha256"],
        "source_implementation_inventory_sha256": plan[
            "source_implementation_inventory_sha256"
        ],
        "fixed_serial_order": [cell.execution_id for cell in CELL_SPECS],
        "comparison_sha256": comparison["sha256"],
        "comparison_csv_sha256": sha256_file(REPORT_CSV),
        "comparison_markdown_sha256": sha256_file(REPORT_MD),
        "controller_rounds_completed_by_cell": {
            cell.execution_id: TARGET_HORIZON for cell in CELL_SPECS
        },
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    if {key: terminal.get(key) for key in expected} != expected:
        raise RunnerError("Terminal matrix receipt failed deep validation.")
    return terminal


def run_campaign() -> int:
    plan, authorization = validate_authority(recompute_protocols=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    lock_descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError("Eight-arm campaign lock is already held.") from exc
        if TERMINAL_PATH.is_file():
            validate_terminal_matrix(plan=plan, authorization=authorization)
            return 0
        completed: list[tuple[CellSpec, Mapping[str, Any], Mapping[str, Any], str, str]] = []
        for cell in CELL_SPECS:
            run_dir, staging, worker_path, guard_path = cell_paths(cell)
            state = tuple(
                path.exists() or path.is_symlink()
                for path in (run_dir, staging, worker_path, guard_path)
            )
            if state == (True, False, True, True):
                completed.append(
                    load_closed_cell(cell, plan=plan, authorization=authorization)
                )
                continue
            if any(state):
                raise RunnerError(f"Cell has partial output: {cell.execution_id}")
            capacity = wait_for_capacity()
            if capacity["status"] != "ready":
                write_status({**capacity, "status": "blocked_capacity", "execution_id": cell.execution_id})
                return 2
            guard = monitor_cell(cell, authorization)
            completed.append(
                load_closed_cell(cell, plan=plan, authorization=authorization)
            )
        comparison, csv_text, markdown = build_comparison(completed)
        _publish_or_validate_json(REPORT_JSON, comparison, schema=REPORT_SCHEMA)
        _publish_or_validate_text(REPORT_CSV, csv_text)
        _publish_or_validate_text(REPORT_MD, markdown)
        terminal = digested(
            {
                "schema": TERMINAL_SCHEMA,
                "status": "passed_eight_k5",
                "campaign_id": CAMPAIGN_ID,
                "plan_sha256": plan["sha256"],
                "authorization_sha256": authorization["sha256"],
                "source_implementation_inventory_sha256": plan[
                    "source_implementation_inventory_sha256"
                ],
                "fixed_serial_order": [cell.execution_id for cell in CELL_SPECS],
                "comparison_sha256": comparison["sha256"],
                "comparison_csv_sha256": sha256_file(REPORT_CSV),
                "comparison_markdown_sha256": sha256_file(REPORT_MD),
                "controller_rounds_completed_by_cell": {
                    cell.execution_id: TARGET_HORIZON for cell in CELL_SPECS
                },
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        write_status({"status": "passed_eight_k5", "terminal_sha256": terminal["sha256"]})
        write_json_exclusive(TERMINAL_PATH, terminal)
        validate_terminal_matrix(plan=plan, authorization=authorization)
        return 0
    finally:
        os.close(lock_descriptor)


def preflight() -> dict[str, Any]:
    plan = build_plan() if not PLAN_PATH.is_file() else validate_plan(recompute_protocols=True)
    return {
        "campaign_id": CAMPAIGN_ID,
        "cell_count": len(CELL_SPECS),
        "fixed_serial_order": [cell.execution_id for cell in CELL_SPECS],
        "source_implementation_inventory_sha256": plan[
            "source_implementation_inventory_sha256"
        ],
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
