#!/usr/bin/env python3
"""Run one diagnostic strong--weak position-aware Phase-0 RA canary to k=15."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import psutil

from position_aware_phase0_canary_20260816 import (
    POSITION_PHASE0_INSERTION_SCOPE,
    POSITION_PHASE0_POLICY,
    POSITION_PHASE0_SCHEMA,
    install_position_aware_phase0_overlay,
)


RUNNER_PATH = Path(__file__).resolve()
REPAIR_ROOT = RUNNER_PATH.parent
REPO_ROOT = RUNNER_PATH.parents[2]
OVERLAY_PATH = REPAIR_ROOT / "position_aware_phase0_canary_20260816.py"
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc"
)
SOURCE_EXECUTION_ID = (
    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
    "strong_weak_u8__nph3__ra_global_singleton_gradient_phase0_"
    "phase123_qiskit_phase23_always_commutation_reduced"
)
EXECUTION_ID = (
    "position_aware_phase0__strong_weak_u8__nph3__"
    "ra_always_commutation_reduced__k15"
)
SOURCE_JOB = PACKAGE_DIR / "jobs" / f"{SOURCE_EXECUTION_ID}.json"
AUTHORITY_DIR = REPAIR_ROOT / (
    "paper_i_position_aware_phase0_sw_always_k15_20260816_v1_authority"
)
PLAN_PATH = AUTHORITY_DIR / "plan.json"
AUTHORIZATION_PATH = AUTHORITY_DIR / "authorization.json"
RUNTIME_ROOT = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_position_aware_phase0_sw_always_k15_20260816_v1"
)
RUN_DIR = RUNTIME_ROOT / "runs" / EXECUTION_ID
IN_PROGRESS_DIR = RUNTIME_ROOT / "in_progress" / EXECUTION_ID
WORKER_RECEIPT_PATH = RUNTIME_ROOT / "receipts" / f"{EXECUTION_ID}.json"
GUARD_RECEIPT_PATH = RUNTIME_ROOT / "guard_receipts" / f"{EXECUTION_ID}.json"
TERMINAL_PATH = RUNTIME_ROOT / "terminal_receipt.json"
STATUS_PATH = RUNTIME_ROOT / "status.json"
LOCK_PATH = RUNTIME_ROOT / "campaign.lock"

REMOTE_JOB_ID = (
    "holstein-paper-i-position-aware-phase0-strong-weak-always-k15-20260816-v1"
)
TARGET_HORIZON = 15
SHORTLIST_SIZE = 24
RSS_LIMIT_BYTES = 8 * 1024**3
AVAILABLE_MEMORY_FLOOR_BYTES = 2 * 1024**3
MIN_LAUNCH_AVAILABLE_MEMORY_BYTES = 5 * 1024**3 // 2
RUNTIME_FREE_DISK_FLOOR_BYTES = 2 * 1024**3
MIN_LAUNCH_FREE_DISK_BYTES = 8 * 1024**3
POLL_SECONDS = 1.0
STATUS_SECONDS = 10.0
CHILD_TOKEN_ENV = "PAPER_I_POSITION_AWARE_PHASE0_CANARY_CHILD_TOKEN"
PLAN_SCHEMA = "paper_i_position_aware_phase0_canary_plan_v1"
AUTHORIZATION_SCHEMA = "paper_i_position_aware_phase0_canary_authorization_v1"
PREFLIGHT_SCHEMA = "paper_i_position_aware_phase0_canary_preflight_v1"
EXECUTION_MANIFEST_SCHEMA = (
    "paper_i_position_aware_phase0_canary_execution_manifest_v1"
)
WORKER_RECEIPT_SCHEMA = "paper_i_position_aware_phase0_canary_worker_receipt_v1"
GUARD_RECEIPT_SCHEMA = "paper_i_position_aware_phase0_canary_guard_receipt_v1"
TERMINAL_SCHEMA = "paper_i_position_aware_phase0_canary_terminal_receipt_v1"
STATUS_SCHEMA = "paper_i_position_aware_phase0_canary_status_v1"
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
}


class CanaryError(RuntimeError):
    """Raised when the diagnostic canary fails closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "sha256" in result:
        raise CanaryError("Self-digested payload already contains sha256.")
    result["sha256"] = canonical_sha256(result)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CanaryError(f"Required regular file is absent: {path}")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise CanaryError(f"JSON payload is not a mapping: {path}")
    return payload


def load_digested(path: Path, *, schema: str) -> dict[str, Any]:
    payload = load_json(path)
    observed = payload.pop("sha256", None)
    if payload.get("schema") != schema or observed != canonical_sha256(payload):
        raise CanaryError(f"Self-digested authority drifted: {path}")
    payload["sha256"] = observed
    return payload


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(payload) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(canonical_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def import_worker() -> Any:
    path = PACKAGE_DIR / "run_cell.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_position_phase0_parent_worker",
        path,
    )
    if spec is None or spec.loader is None:
        raise CanaryError("Could not load the source-locked parent worker.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_bindings() -> dict[str, Any]:
    job = load_json(SOURCE_JOB)
    protocol_relative = Path(str(job["protocol_path"]))
    protocol_path = PACKAGE_DIR / protocol_relative
    source_archive = PACKAGE_DIR / "source/source_locked.tar.gz"
    if (
        job.get("execution_id") != SOURCE_EXECUTION_ID
        or job.get("regime_id") != "strong_weak_u8"
        or job.get("nph") != 3
        or job.get("comparator_policy") != "always_commutation_reduced"
        or job.get("runtime_insertion_mode") != "full_commutation_reduced"
        or job.get("typed_insertion_kind") != "always_commutation_reduced"
        or job.get("target_horizon") != 50
        or job.get("protocol_file_sha256") != sha256_file(protocol_path)
        or job.get("sha256")
        != canonical_sha256({key: value for key, value in job.items() if key != "sha256"})
    ):
        raise CanaryError("Parent strong--weak always-open job identity drifted.")
    protocol = load_json(protocol_path)
    if (
        protocol.get("sha256") != job.get("protocol_sha256")
        or protocol.get("horizon") != 50
        or protocol.get("request", {}).get("method", {}).get("insertion", {}).get("kind")
        != "always_commutation_reduced"
        or protocol.get("request", {}).get("execution", {}).get("stop", {}).get(
            "maximum_controller_rounds"
        )
        != 50
    ):
        raise CanaryError("Parent strong--weak always-open protocol drifted.")
    return {
        "job": {**file_binding(SOURCE_JOB), "canonical_sha256": job["sha256"]},
        "protocol": {
            **file_binding(protocol_path),
            "canonical_sha256": protocol["sha256"],
        },
        "source_archive": file_binding(source_archive),
        "parent_route_contract_sha256": str(job["route_contract_sha256"]),
        "parent_protocol_sha256": str(job["protocol_sha256"]),
    }


def build_plan() -> dict[str, Any]:
    return digested(
        {
            "schema": PLAN_SCHEMA,
            "created_at": utc_now(),
            "run_class": "diagnostic",
            "campaign_id": (
                "paper_i_position_aware_phase0_sw_always_k15_20260816_v1"
            ),
            "execution_id": EXECUTION_ID,
            "method": "RA-ADAPT always-open commutation-reduced insertion",
            "regime_id": "strong_weak_u8",
            "nph": 3,
            "target_horizon": TARGET_HORIZON,
            "phase0_shortlist_size": SHORTLIST_SIZE,
            "phase0_policy": POSITION_PHASE0_POLICY,
            "phase0_schema": POSITION_PHASE0_SCHEMA,
            "phase0_insertion_position_scope": POSITION_PHASE0_INSERTION_SCOPE,
            "source": source_bindings(),
            "runner": file_binding(RUNNER_PATH),
            "overlay": file_binding(OVERLAY_PATH),
            "wrapper_used": True,
            "wrapper_kind": "extracted_source_runtime_monkeypatch_v1",
            "settings_reused": [
                "strong_weak_u8_hubbard_holstein_problem_and_same_cutoff_reference",
                "nph3_global_guarded_singleton_pool",
                "always_commutation_reduced_insertion_domain",
                "stationary_source_response_v1",
                "all_phase_resource_weighting_v1",
                "powell_maxiter_200",
                "adapt_and_transpiler_seed_7",
                "phase0_shortlist_cap_24",
                "phase1_phase2_phase3_qiskit_selector",
            ],
            "settings_changed": [
                "phase0_selection_unit:generator_to_commutation_reduced_generator_position_record",
                "phase0_gradient_chart:append_endpoint_to_actual_insertion_position",
                "maximum_controller_rounds:50_to_15",
                "execution_wrapper:sealed_parent_to_diagnostic_runtime_overlay",
            ],
            "unresolved_source_fields": [],
            "fresh_start": True,
            "maximum_concurrency": 1,
            "execution_target": "local_mac_guarded_serial",
            "runtime_environment": dict(EXPECTED_ENV),
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
            "execution_authorized": False,
        }
    )


def materialize_authority() -> dict[str, Any]:
    if AUTHORITY_DIR.exists() or AUTHORITY_DIR.is_symlink():
        raise CanaryError(f"Authority already exists: {AUTHORITY_DIR}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{AUTHORITY_DIR.name}.", dir=AUTHORITY_DIR.parent)
    )
    try:
        plan = build_plan()
        write_json_exclusive(temporary / "plan.json", plan)
        authorization = digested(
            {
                "schema": AUTHORIZATION_SCHEMA,
                "created_at": utc_now(),
                "authorization_kind": "explicit_current_user_diagnostic_execution",
                "scope": "one_local_strong_weak_position_phase0_always_open_k15_cell",
                "execution_id": EXECUTION_ID,
                "plan_sha256": plan["sha256"],
                "runner_sha256": plan["runner"]["sha256"],
                "overlay_sha256": plan["overlay"]["sha256"],
                "target_horizon": TARGET_HORIZON,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        write_json_exclusive(temporary / "authorization.json", authorization)
        os.rename(temporary, AUTHORITY_DIR)
        return authorization
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_digested(PLAN_PATH, schema=PLAN_SCHEMA)
    authorization = load_digested(
        AUTHORIZATION_PATH,
        schema=AUTHORIZATION_SCHEMA,
    )
    current_source = source_bindings()
    if (
        plan.get("execution_id") != EXECUTION_ID
        or plan.get("target_horizon") != TARGET_HORIZON
        or plan.get("phase0_shortlist_size") != SHORTLIST_SIZE
        or plan.get("phase0_policy") != POSITION_PHASE0_POLICY
        or plan.get("phase0_insertion_position_scope")
        != POSITION_PHASE0_INSERTION_SCOPE
        or plan.get("source") != current_source
        or plan.get("runner") != file_binding(RUNNER_PATH)
        or plan.get("overlay") != file_binding(OVERLAY_PATH)
        or plan.get("execution_authorized") is not False
        or plan.get("submission_authorized") is not False
        or plan.get("paper_adoption_authorized") is not False
        or plan.get("paper_evidence_adoption_authorized") is not False
        or authorization.get("scope")
        != "one_local_strong_weak_position_phase0_always_open_k15_cell"
        or authorization.get("execution_id") != EXECUTION_ID
        or authorization.get("plan_sha256") != plan["sha256"]
        or authorization.get("runner_sha256") != plan["runner"]["sha256"]
        or authorization.get("overlay_sha256") != plan["overlay"]["sha256"]
        or authorization.get("target_horizon") != TARGET_HORIZON
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("paper_adoption_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise CanaryError("Position-aware canary authority chain drifted.")
    return plan, authorization


def assert_environment() -> None:
    drift = {
        key: {"expected": value, "observed": os.environ.get(key)}
        for key, value in EXPECTED_ENV.items()
        if os.environ.get(key) != value
    }
    if drift:
        raise CanaryError(f"Numerical thread environment drifted: {drift}")


def scientific_overlap() -> list[dict[str, Any]]:
    own_ancestry: set[int] = set()
    process = psutil.Process(os.getpid())
    while process is not None:
        own_ancestry.add(process.pid)
        try:
            process = process.parent()
        except psutil.Error:
            break
    overlaps: list[dict[str, Any]] = []
    for candidate in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(candidate.info["pid"])
            argv = [str(value) for value in candidate.info.get("cmdline") or []]
        except (psutil.Error, TypeError, ValueError):
            continue
        if pid in own_ancestry or not argv or "python" not in Path(argv[0]).name.lower():
            continue
        command = " ".join(argv)
        if any(
            marker in command
            for marker in (
                "run_local_page12_weak_holstein_priority6_20260815.py",
                "run_local_paper_i_weak12_priority_then_matched_unique6_20260815.py",
                "run_local_paper_i_page12_matched_singleton12_r50_20260815.py",
                "run_local_page12_strong_holstein_sector5_20260814.py",
                "run_local_page12_weak_append_only_first2_k50_20260816.py",
                "run_cell.py --run",
                "--child-cell",
            )
        ):
            overlaps.append({"pid": pid, "command": command})
    return overlaps


def preflight() -> dict[str, Any]:
    plan, authorization = validate_authority()
    available = int(psutil.virtual_memory().available)
    free_disk = int(shutil.disk_usage(REPO_ROOT).free)
    overlap = scientific_overlap()
    output_state = (
        "complete"
        if TERMINAL_PATH.is_file()
        else "in_progress"
        if IN_PROGRESS_DIR.exists()
        else "absent"
    )
    ready = bool(
        not overlap
        and output_state in {"absent", "complete"}
        and available >= MIN_LAUNCH_AVAILABLE_MEMORY_BYTES
        and free_disk >= MIN_LAUNCH_FREE_DISK_BYTES
    )
    return digested(
        {
            "schema": PREFLIGHT_SCHEMA,
            "observed_at": utc_now(),
            "status": "passed" if ready else "blocked",
            "execution_id": EXECUTION_ID,
            "target_horizon": TARGET_HORIZON,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "available_memory_bytes": available,
            "minimum_launch_available_memory_bytes": (
                MIN_LAUNCH_AVAILABLE_MEMORY_BYTES
            ),
            "free_disk_bytes": free_disk,
            "minimum_launch_free_disk_bytes": MIN_LAUNCH_FREE_DISK_BYTES,
            "scientific_overlap": overlap,
            "output_state": output_state,
            "run_ready": ready,
            "scientific_execution_performed": False,
        }
    )


def child_token(authorization_sha256: str) -> str:
    return hashlib.sha256(
        f"{authorization_sha256}:{sha256_file(RUNNER_PATH)}:{EXECUTION_ID}".encode()
    ).hexdigest()


def _write_child_outputs(
    *,
    worker: Any,
    result: Any,
    rounds: int,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    staging: Path,
) -> dict[str, Any]:
    write_json_exclusive(staging / "result/result.json", result.to_dict())
    if result.run.paper_i_summary is None:
        raise CanaryError("Canary execution omitted its Paper-I summary.")
    write_json_exclusive(
        staging / "summary/summary.json",
        result.run.paper_i_summary.to_dict(),
    )
    overlay_receipt = digested(
        {
            "schema": "paper_i_position_aware_phase0_route_overlay_v1",
            "execution_id": EXECUTION_ID,
            "parent_execution_id": SOURCE_EXECUTION_ID,
            "parent_protocol_sha256": plan["source"]["parent_protocol_sha256"],
            "parent_route_contract_sha256": plan["source"][
                "parent_route_contract_sha256"
            ],
            "runner_sha256": plan["runner"]["sha256"],
            "overlay_sha256": plan["overlay"]["sha256"],
            "phase0_policy": POSITION_PHASE0_POLICY,
            "phase0_insertion_position_scope": POSITION_PHASE0_INSERTION_SCOPE,
            "target_horizon": TARGET_HORIZON,
            "wrapper_used": True,
            "run_class": "diagnostic",
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(staging / "route_overlay.json", overlay_receipt)
    artifacts = {}
    for role, relative in {
        "checkpoint": "checkpoints/current.json",
        "estimator_ledger": "result/estimator_ledger.json",
        "result": "result/result.json",
        "summary": "summary/summary.json",
        "route_overlay": "route_overlay.json",
    }.items():
        path = staging / relative
        if not path.is_file():
            raise CanaryError(f"Canary output role is absent: {role}")
        artifacts[role] = {
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = digested(
        {
            "schema": EXECUTION_MANIFEST_SCHEMA,
            "status": "passed",
            "execution_id": EXECUTION_ID,
            "parent_execution_id": SOURCE_EXECUTION_ID,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "parent_protocol_sha256": plan["source"]["parent_protocol_sha256"],
            "position_phase0_policy": POSITION_PHASE0_POLICY,
            "target_horizon": TARGET_HORIZON,
            "controller_rounds_completed": int(rounds),
            "fresh_start": True,
            "run_class": "diagnostic",
            "artifacts": artifacts,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(staging / "execution_manifest.json", manifest)
    return manifest


def run_child() -> int:
    plan, authorization = validate_authority()
    assert_environment()
    if os.environ.get(CHILD_TOKEN_ENV) != child_token(authorization["sha256"]):
        raise CanaryError("Canary child capability is absent or invalid.")
    if RUN_DIR.exists() or RUN_DIR.is_symlink() or IN_PROGRESS_DIR.exists():
        raise CanaryError("Canary child destination is not pristine.")
    IN_PROGRESS_DIR.parent.mkdir(parents=True, exist_ok=True)
    IN_PROGRESS_DIR.mkdir()
    worker = import_worker()
    job, _manifest, protocol, problem, temporary = worker._prepare(SOURCE_JOB)
    restore_overlay = None
    try:
        if (
            job.get("execution_id") != SOURCE_EXECUTION_ID
            or protocol.request.method.insertion.kind
            != "always_commutation_reduced"
            or int(protocol.horizon) != 50
        ):
            raise CanaryError("Prepared parent protocol drifted.")
        restore_overlay = install_position_aware_phase0_overlay()
        source_root = Path(temporary.name) / "source"
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result, rounds = worker._execute(
                protocol=protocol,
                problem=problem,
                staging=IN_PROGRESS_DIR,
                maximum_rounds=TARGET_HORIZON,
            )
        finally:
            os.chdir(original)
        if rounds != TARGET_HORIZON:
            raise CanaryError(
                f"Canary stopped at {rounds} accepted rounds instead of {TARGET_HORIZON}."
            )
        execution_manifest = _write_child_outputs(
            worker=worker,
            result=result,
            rounds=rounds,
            plan=plan,
            authorization=authorization,
            staging=IN_PROGRESS_DIR,
        )
        RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
        os.rename(IN_PROGRESS_DIR, RUN_DIR)
        receipt = digested(
            {
                "schema": WORKER_RECEIPT_SCHEMA,
                "status": "passed",
                "execution_id": EXECUTION_ID,
                "plan_sha256": plan["sha256"],
                "authorization_sha256": authorization["sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "controller_rounds_completed": rounds,
                "artifacts": [
                    {
                        "path": path.relative_to(RUNTIME_ROOT).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(RUN_DIR.rglob("*"))
                    if path.is_file()
                ],
                "run_class": "diagnostic",
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        write_json_exclusive(WORKER_RECEIPT_PATH, receipt)
        return 0
    finally:
        if restore_overlay is not None:
            restore_overlay()
        temporary.cleanup()


def total_rss(process: psutil.Process) -> int:
    total = 0
    for candidate in [process, *process.children(recursive=True)]:
        try:
            total += int(candidate.memory_info().rss)
        except psutil.Error:
            pass
    return total


def terminal_is_valid() -> bool:
    if not TERMINAL_PATH.is_file():
        return False
    try:
        terminal = load_digested(TERMINAL_PATH, schema=TERMINAL_SCHEMA)
        worker_receipt = load_digested(
            WORKER_RECEIPT_PATH,
            schema=WORKER_RECEIPT_SCHEMA,
        )
        guard = load_digested(GUARD_RECEIPT_PATH, schema=GUARD_RECEIPT_SCHEMA)
    except (OSError, ValueError, CanaryError):
        return False
    return bool(
        terminal.get("status") == "passed_k15"
        and terminal.get("execution_id") == EXECUTION_ID
        and terminal.get("worker_receipt_sha256") == worker_receipt["sha256"]
        and terminal.get("guard_receipt_sha256") == guard["sha256"]
        and terminal.get("paper_adoption_authorized") is False
        and terminal.get("paper_evidence_adoption_authorized") is False
    )


def run_supervisor() -> int:
    plan, authorization = validate_authority()
    assert_environment()
    if os.environ.get("REMOTE_JOB_ID") != REMOTE_JOB_ID:
        raise CanaryError("Scientific launch must come from the fixed remote-runner job.")
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise CanaryError("The position-aware canary already owns its lock.") from exc
    try:
        if terminal_is_valid():
            return 0
        preflight_payload = preflight()
        if not preflight_payload["run_ready"]:
            write_json_atomic(
                STATUS_PATH,
                digested(
                    {
                        "schema": STATUS_SCHEMA,
                        "status": "blocked_preflight",
                        "updated_at": utc_now(),
                        "execution_id": EXECUTION_ID,
                        "preflight": preflight_payload,
                    }
                ),
            )
            raise CanaryError("Position-aware canary preflight is blocked.")
        token = child_token(authorization["sha256"])
        environment = dict(os.environ)
        environment[CHILD_TOKEN_ENV] = token
        command = [
            sys.executable,
            "-u",
            "-B",
            str(RUNNER_PATH),
            "--child",
        ]
        child = subprocess.Popen(command, cwd=REPO_ROOT, env=environment)
        process = psutil.Process(child.pid)
        started = time.monotonic()
        last_status = 0.0
        peak_rss = 0
        minimum_available = int(psutil.virtual_memory().available)
        minimum_disk = int(shutil.disk_usage(REPO_ROOT).free)
        stop_reason: str | None = None
        try:
            while child.poll() is None:
                rss = total_rss(process)
                available = int(psutil.virtual_memory().available)
                free_disk = int(shutil.disk_usage(REPO_ROOT).free)
                peak_rss = max(peak_rss, rss)
                minimum_available = min(minimum_available, available)
                minimum_disk = min(minimum_disk, free_disk)
                if rss > RSS_LIMIT_BYTES:
                    stop_reason = "rss_limit_breached"
                elif available < AVAILABLE_MEMORY_FLOOR_BYTES:
                    stop_reason = "available_memory_floor_breached"
                elif free_disk < RUNTIME_FREE_DISK_FLOOR_BYTES:
                    stop_reason = "free_disk_floor_breached"
                now = time.monotonic()
                if now - last_status >= STATUS_SECONDS:
                    write_json_atomic(
                        STATUS_PATH,
                        digested(
                            {
                                "schema": STATUS_SCHEMA,
                                "status": "running_position_aware_phase0_k15",
                                "updated_at": utc_now(),
                                "execution_id": EXECUTION_ID,
                                "child_pid": child.pid,
                                "elapsed_seconds": now - started,
                                "current_rss_bytes": rss,
                                "peak_rss_bytes": peak_rss,
                                "available_memory_bytes": available,
                                "minimum_available_memory_bytes": minimum_available,
                                "free_disk_bytes": free_disk,
                                "minimum_free_disk_bytes": minimum_disk,
                                "stop_reason": stop_reason,
                                "remote_run_id": os.environ.get("REMOTE_RUN_ID"),
                            }
                        ),
                    )
                    last_status = now
                if stop_reason is not None:
                    child.terminate()
                    break
                time.sleep(POLL_SECONDS)
        except BaseException:
            if child.poll() is None:
                child.terminate()
            raise
        try:
            returncode = child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            returncode = child.wait(timeout=30)
        elapsed = time.monotonic() - started
        if returncode != 0 or stop_reason is not None:
            write_json_atomic(
                STATUS_PATH,
                digested(
                    {
                        "schema": STATUS_SCHEMA,
                        "status": "failed_or_guard_stopped",
                        "updated_at": utc_now(),
                        "execution_id": EXECUTION_ID,
                        "returncode": returncode,
                        "stop_reason": stop_reason,
                        "elapsed_seconds": elapsed,
                        "peak_rss_bytes": peak_rss,
                        "minimum_available_memory_bytes": minimum_available,
                        "minimum_free_disk_bytes": minimum_disk,
                    }
                ),
            )
            raise CanaryError(
                f"Position-aware canary failed rc={returncode}, reason={stop_reason}."
            )
        worker_receipt = load_digested(
            WORKER_RECEIPT_PATH,
            schema=WORKER_RECEIPT_SCHEMA,
        )
        guard = digested(
            {
                "schema": GUARD_RECEIPT_SCHEMA,
                "status": "passed",
                "execution_id": EXECUTION_ID,
                "child_returncode": returncode,
                "elapsed_seconds": elapsed,
                "peak_rss_bytes": peak_rss,
                "minimum_available_memory_bytes": minimum_available,
                "minimum_free_disk_bytes": minimum_disk,
                "rss_limit_bytes": RSS_LIMIT_BYTES,
                "available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
                "runtime_free_disk_floor_bytes": RUNTIME_FREE_DISK_FLOOR_BYTES,
                "stop_reason": None,
            }
        )
        write_json_exclusive(GUARD_RECEIPT_PATH, guard)
        terminal = digested(
            {
                "schema": TERMINAL_SCHEMA,
                "status": "passed_k15",
                "execution_id": EXECUTION_ID,
                "plan_sha256": plan["sha256"],
                "authorization_sha256": authorization["sha256"],
                "worker_receipt_sha256": worker_receipt["sha256"],
                "guard_receipt_sha256": guard["sha256"],
                "controller_rounds_completed": TARGET_HORIZON,
                "phase0_policy": POSITION_PHASE0_POLICY,
                "run_class": "diagnostic",
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        write_json_exclusive(TERMINAL_PATH, terminal)
        write_json_atomic(
            STATUS_PATH,
            digested(
                {
                    "schema": STATUS_SCHEMA,
                    "status": "passed_k15",
                    "updated_at": utc_now(),
                    "execution_id": EXECUTION_ID,
                    "terminal_sha256": terminal["sha256"],
                    "elapsed_seconds": elapsed,
                    "peak_rss_bytes": peak_rss,
                    "minimum_available_memory_bytes": minimum_available,
                    "minimum_free_disk_bytes": minimum_disk,
                }
            ),
        )
        return 0
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize-authority", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.materialize_authority:
        print(json.dumps(materialize_authority(), sort_keys=True))
        return 0
    if args.preflight:
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if args.child:
        return run_child()
    return run_supervisor()


if __name__ == "__main__":
    raise SystemExit(main())
