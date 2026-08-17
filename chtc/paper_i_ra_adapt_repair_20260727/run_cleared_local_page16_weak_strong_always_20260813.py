#!/usr/bin/env python3
"""Supervise the exact local Page-16 weak-strong always-open k30 cell.

This adapter never runs the other wave-2 member.  It waits for authenticated
CHTC closure and exclusion of further remote strong-weak materialization,
then for local wave 5 to close, before acquiring the existing serialized-wave
lock.  Scientific execution remains delegated to the pinned v2 k30 runner and
its existing activation.  The resulting native receipt and plateau gate are
also authenticated through the pinned k30-to-k50 continuation adapter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()
REPAIR_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_local_page16_insertion_comparators_20260812.py"
)
EXPECTED_RUNNER_SHA256 = (
    "bd9d61fb98b48911c3da04faf8b6c38eb391b1a02ab3362e22ef02316a414c4e"
)
CONTINUATION_ADAPTER_PATH = SCRIPT_PATH.with_name(
    "continue_local_page16_insertion_comparators_k30_to_k50_20260813.py"
)
EXPECTED_CONTINUATION_ADAPTER_SHA256 = (
    "56c50f046759d4299d768cb609f08fce8c79e3190aaadb6609afdde4f5452e07"
)
ACTIVATION_DIR = SCRIPT_PATH.with_name(
    "paper_i_ra_adapt_page16_insertion_comparators_k30_"
    "20260812_v2_local_activation"
)
RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page16_insertion_comparators_k30_20260812_v2"
)
EXPECTED_ACTIVATION_FILE_SHA256 = (
    "7e138a7dcc898f596555bf0839ec987893dc4784c0a7f1d1f01f57504a9f79eb"
)
EXPECTED_ACTIVATION_CANONICAL_SHA256 = (
    "e4813a5bcb8e154a621326badc7c922e1e78fccdc3891c8350cb0d36799ebc78"
)
EXPECTED_RUNTIME_CANONICAL_SHA256 = (
    "ce892e8f31658df18cbe0ff65317bfc798b7de461e2bf2373f5b56fc2184577e"
)
SW_CLOSURE_RECEIPT_PATH = SCRIPT_PATH.with_name(
    "paper_i_ra_adapt_page16_cluster9647386_sw_always_"
    "remote_materialization_exclusion_receipt_20260813.json"
)
STATUS_PATH = RUNTIME_DIR / (
    "status/wave_2_weak_strong_always_remote_sw_exclusion.json"
)
STANDARD_WAVE2_STATUS_PATH = RUNTIME_DIR / "status/wave_2.json"
WAVE5_STATUS_PATH = RUNTIME_DIR / "status/wave_5.json"
LOCK_PATH = RUNTIME_DIR / "wave_supervisor.lock"
POLL_SECONDS = 20
WAIT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
STATUS_SCHEMA = (
    "paper_i_page16_weak_strong_always_remote_sw_exclusion_status_v1"
)
PREFLIGHT_SCHEMA = (
    "paper_i_page16_weak_strong_always_remote_sw_exclusion_preflight_v1"
)

TARGET_EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_strong__"
    "nph7__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced"
)
EXCLUDED_REMOTE_EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_weak_u8__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced"
)
EXPECTED_TARGET_JOB_SHA256 = (
    "9d6ddafed245ff15c23568355d8d4ce1bdc3828443c69e082c9d160aadb13eec"
)
EXPECTED_TARGET_PROTOCOL_SHA256 = (
    "57e4043b01b21d6971a43b4e0a12985045ab7f74457228d39aa9e8e0fdbf62e3"
)
EXPECTED_TARGET_ROUTE_CONTRACT_SHA256 = (
    "9b9d6bdbb9edb6128e2f0973dd740b44d0daa00d55ecd910fd587f091ae81338"
)
EXPECTED_TARGET_SOURCE_LOCKS_SHA256 = (
    "fc4bdd4c1d1419ffa669c7ea619a456330790e60a6166dbf5a36ca304076df71"
)
EXPECTED_TARGET_AUTHORIZATION_SHA256 = (
    "35f6e758a63b431c90bd691b840c8f1c50afa371c8a594fb64a7216f5e176d67"
)
EXPECTED_EXCLUDED_REMOTE_JOB_SHA256 = (
    "598d2b615af58ad1551178920c5363a98ad5094d156e401510e1d1728ae8e0e1"
)


class SingleCellError(RuntimeError):
    """The exact single-cell authorization or closure did not validate."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    return {**unsigned, "sha256": _canonical_sha256(unsigned)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned(path: Path, expected_sha256: str, name: str) -> Any:
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != expected_sha256
    ):
        raise SingleCellError(f"Pinned source is absent, unsafe, or drifted: {path}")
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SingleCellError(f"Pinned source cannot be imported: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner() -> Any:
    return _load_pinned(
        RUNNER_PATH,
        EXPECTED_RUNNER_SHA256,
        "paper_i_page16_ws_always_pinned_k30_runner",
    )


def _load_continuation_adapter() -> Any:
    return _load_pinned(
        CONTINUATION_ADAPTER_PATH,
        EXPECTED_CONTINUATION_ADAPTER_SHA256,
        "paper_i_page16_ws_always_pinned_k50_continuation_adapter",
    )


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SingleCellError(f"{label} is absent or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SingleCellError(f"{label} must be a JSON object.")
    expected = value.get("sha256")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if expected != _canonical_sha256(unsigned):
        raise SingleCellError(f"{label} self-digest drifted.")
    return value


def _fixed_context() -> SimpleNamespace:
    runner = _load_runner()
    continuation = _load_continuation_adapter()
    if (
        runner.MAX_CONCURRENCY != 1
        or tuple(runner.WAVES[1])
        != (EXCLUDED_REMOTE_EXECUTION_ID, TARGET_EXECUTION_ID)
        or runner.REGIME_BY_EXECUTION_ID.get(TARGET_EXECUTION_ID)
        != "weak_strong"
        or runner.NPH_BY_EXECUTION_ID.get(TARGET_EXECUTION_ID) != 7
        or TARGET_EXECUTION_ID not in continuation.CONDITIONAL_EXECUTION_IDS
        or continuation.SW_ALWAYS_CHTC_EXECUTION_ID
        != EXCLUDED_REMOTE_EXECUTION_ID
        or EXCLUDED_REMOTE_EXECUTION_ID
        not in continuation.TERMINAL_CHTC_EXECUTION_IDS
        or TARGET_EXECUTION_ID in continuation.TERMINAL_CHTC_EXECUTION_IDS
        or continuation.SW_ALWAYS_CLOSURE_RECEIPT_PATH
        != SW_CLOSURE_RECEIPT_PATH
        or continuation.SW_ALWAYS_CLOSURE_RECEIPT_STATUS
        != "passed_sw_always_k50_closed_remote_materialization_excluded"
    ):
        raise SingleCellError("Exact weak-strong/SW-exclusion inventory drifted.")
    activation_path = ACTIVATION_DIR / "activation_manifest.json"
    if (
        ACTIVATION_DIR.is_symlink()
        or not ACTIVATION_DIR.is_dir()
        or not activation_path.is_file()
        or activation_path.is_symlink()
        or _sha256_file(activation_path) != EXPECTED_ACTIVATION_FILE_SHA256
    ):
        raise SingleCellError("Pinned v2 activation bytes drifted.")
    package_path = runner.PACKAGE_DIR.as_posix()
    sys.path[:] = [row for row in sys.path if row != package_path]
    sys.path.insert(0, package_path)
    worker = runner._load_worker()
    manifest, rows = runner._closed_package(worker)
    activation = runner._validate_activation(
        worker,
        ACTIVATION_DIR,
        manifest=manifest,
    )
    if activation.get("sha256") != EXPECTED_ACTIVATION_CANONICAL_SHA256:
        raise SingleCellError("Pinned v2 activation identity drifted.")
    runtime = _load_digested(
        RUNTIME_DIR / "runtime_manifest.json",
        label="pinned v2 runtime manifest",
    )
    expected_runtime = runner._runtime_manifest(worker, activation=activation)
    if (
        runtime != expected_runtime
        or runtime.get("sha256") != EXPECTED_RUNTIME_CANONICAL_SHA256
        or runtime.get("maximum_concurrency") != 1
    ):
        raise SingleCellError("Pinned v2 runtime identity drifted.")
    jobs = continuation._job_by_id(worker)
    target_job = jobs[TARGET_EXECUTION_ID]
    excluded_job = jobs[EXCLUDED_REMOTE_EXECUTION_ID]
    target_authority = runner._authorization_for_cell(
        worker,
        ACTIVATION_DIR,
        activation=activation,
        manifest=manifest,
        job=target_job,
    )
    if (
        target_job.get("sha256") != EXPECTED_TARGET_JOB_SHA256
        or target_job.get("protocol_sha256")
        != EXPECTED_TARGET_PROTOCOL_SHA256
        or target_job.get("route_contract_sha256")
        != EXPECTED_TARGET_ROUTE_CONTRACT_SHA256
        or target_job.get("source_locks_sha256")
        != EXPECTED_TARGET_SOURCE_LOCKS_SHA256
        or target_authority.get("sha256")
        != EXPECTED_TARGET_AUTHORIZATION_SHA256
        or target_job.get("comparator_policy")
        != "always_commutation_reduced"
        or target_job.get("typed_insertion_kind")
        != "always_commutation_reduced"
        or target_job.get("runtime_insertion_mode")
        != "full_commutation_reduced"
        or int(target_job.get("target_horizon", -1)) != 30
        or excluded_job.get("sha256")
        != EXPECTED_EXCLUDED_REMOTE_JOB_SHA256
        or excluded_job.get("comparator_policy")
        != "always_commutation_reduced"
        or int(excluded_job.get("target_horizon", -1)) != 50
    ):
        raise SingleCellError("Pinned wave-2 job contract drifted.")
    return SimpleNamespace(
        runner=runner,
        continuation=continuation,
        worker=worker,
        manifest=manifest,
        rows=rows,
        activation=activation,
        runtime=runtime,
        jobs=jobs,
        target_authority=target_authority,
    )


def _authenticate_sw_exclusion(context: SimpleNamespace) -> dict[str, Any]:
    receipt = context.continuation._authenticate_sw_always_closure(
        context.worker,
        job=context.jobs[EXCLUDED_REMOTE_EXECUTION_ID],
    )
    if (
        receipt.get("execution_id") != EXCLUDED_REMOTE_EXECUTION_ID
        or receipt.get("controller_rounds_completed") != 50
        or receipt.get("authenticated_full_sealed_closure") is not True
        or receipt.get("remote_materialization_exclusion_authenticated")
        is not True
        or receipt.get("continuation_required") is not False
        or receipt.get("local_rerun_authorized") is not False
    ):
        raise SingleCellError("Authenticated SW terminal exclusion drifted.")
    return receipt


def _wave5_terminal_state(context: SimpleNamespace) -> dict[str, Any]:
    if not WAVE5_STATUS_PATH.exists() and not WAVE5_STATUS_PATH.is_symlink():
        return {"terminal": False, "status": "absent"}
    value = _load_digested(WAVE5_STATUS_PATH, label="wave 5 status")
    expected_ids = list(context.runner.WAVES[4])
    state = str(value.get("status", ""))
    if (
        value.get("schema") != context.runner.LOCAL_STATUS_SCHEMA
        or value.get("wave") != 5
        or value.get("execution_ids") != expected_ids
        or value.get("runtime_manifest_sha256")
        != context.runtime.get("sha256")
        or value.get("maximum_concurrency") != 1
        or value.get("local_operational_target_horizon") != 30
        or value.get("round50_continuation_executed") is not False
    ):
        raise SingleCellError("Wave 5 status identity drifted.")
    if state in {"failed", "interrupted"}:
        raise SingleCellError(f"Wave 5 ended in terminal {state} state.")
    terminal = state in {"passed", "passed_already_complete"}
    if terminal and (
        value.get("completed_execution_ids") != expected_ids
        or value.get("running_execution_ids") != []
        or value.get("running_pids") != {}
    ):
        raise SingleCellError("Wave 5 passed closure drifted.")
    return {
        "terminal": terminal,
        "status": state,
        "sha256": value["sha256"],
    }


def _target_closure_state(context: SimpleNamespace) -> dict[str, Any]:
    closed = context.runner._closed_cell(
        context.worker,
        RUNTIME_DIR,
        TARGET_EXECUTION_ID,
    )
    if not closed:
        return {"closed": False, "decision": None}
    decision = context.continuation._closed_k30_decision(
        context.worker,
        execution_id=TARGET_EXECUTION_ID,
        job=context.jobs[TARGET_EXECUTION_ID],
    )
    if (
        decision is None
        or decision.get("execution_id") != TARGET_EXECUTION_ID
        or decision.get("extension_decision")
        not in {"eligible_for_authenticated_resume_to_k50", "stop_at_k30"}
    ):
        raise SingleCellError(
            "Weak-strong closure is not consumable by the 9-local/3-CHTC continuation."
        )
    return {"closed": True, "decision": decision}


def _cell_paths(execution_id: str) -> tuple[Path, Path, Path]:
    return (
        RUNTIME_DIR / "runs" / execution_id,
        RUNTIME_DIR / "worker_receipts" / f"{execution_id}.json",
        RUNTIME_DIR / "plateau_gates" / f"{execution_id}.json",
    )


def _excluded_remote_paths_absent() -> bool:
    return not any(
        path.exists() or path.is_symlink()
        for path in _cell_paths(EXCLUDED_REMOTE_EXECUTION_ID)
    )


def _standard_wave2_status_absent() -> bool:
    return not (
        STANDARD_WAVE2_STATUS_PATH.exists()
        or STANDARD_WAVE2_STATUS_PATH.is_symlink()
    )


def _target_logs_absent() -> bool:
    return not any(
        path.exists() or path.is_symlink()
        for path in (
            RUNTIME_DIR / "logs" / f"{TARGET_EXECUTION_ID}.out",
            RUNTIME_DIR / "logs" / f"{TARGET_EXECUTION_ID}.err",
        )
    )


def _local_scientific_overlap() -> list[str]:
    return list(_load_runner()._overlapping_scientific_commands())


def _wave_lock_available() -> bool:
    if not LOCK_PATH.exists() and not LOCK_PATH.is_symlink():
        return True
    if LOCK_PATH.is_symlink() or not LOCK_PATH.is_file():
        raise SingleCellError("Existing wave-supervisor lock path is unsafe.")
    with LOCK_PATH.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return True


def _runner_preflight() -> dict[str, Any]:
    runner = _load_runner()
    value = runner.inert_preflight(
        activation_dir=ACTIVATION_DIR,
        runtime_dir=RUNTIME_DIR,
    )
    if (
        value.get("status") != "passed_inert_preflight"
        or value.get("activation_status") != "validated"
        or value.get("local_adapter_sha256") != EXPECTED_RUNNER_SHA256
        or value.get("maximum_concurrency", 1) != 1
        or value.get("scientific_execution_performed") is not False
        or value.get("submission_performed") is not False
    ):
        raise SingleCellError("Pinned k30 runner preflight drifted.")
    return value


def preflight(*, lock_held: bool = False) -> dict[str, Any]:
    context = _fixed_context()
    runner_preflight = _runner_preflight()
    receipt: dict[str, Any] | None = None
    receipt_status = "absent"
    if SW_CLOSURE_RECEIPT_PATH.exists() or SW_CLOSURE_RECEIPT_PATH.is_symlink():
        receipt = _authenticate_sw_exclusion(context)
        receipt_status = "authenticated"
    wave5 = _wave5_terminal_state(context)
    target = _target_closure_state(context)
    excluded_absent = _excluded_remote_paths_absent()
    standard_wave2_absent = _standard_wave2_status_absent()
    target_logs_absent = _target_logs_absent()
    overlap = _local_scientific_overlap()
    lock_available = True if lock_held else _wave_lock_available()
    capacity_ready = runner_preflight.get("capacity_ready") is True
    source_ready = runner_preflight.get("run_ready") is True

    if receipt is None:
        status = (
            "waiting_for_authenticated_sw_always_closure_and_remote_"
            "materialization_exclusion"
        )
    elif not wave5["terminal"]:
        status = "waiting_for_terminal_wave_5"
    elif not excluded_absent:
        status = "blocked_excluded_remote_sw_local_paths_present"
    elif not standard_wave2_absent:
        status = "blocked_ambiguous_standard_wave_2_status_present"
    elif not target["closed"] and not target_logs_absent:
        status = "blocked_ambiguous_prior_weak_strong_attempt_logs_present"
    elif not capacity_ready or not source_ready:
        status = "waiting_for_local_capacity"
    elif overlap or not lock_available:
        status = "waiting_for_local_exclusivity"
    elif target["closed"]:
        status = "passed_target_already_closed_and_continuation_compatible"
    else:
        status = "passed_ready_for_exact_single_cell_launch"
    run_ready = status in {
        "passed_ready_for_exact_single_cell_launch",
        "passed_target_already_closed_and_continuation_compatible",
    }
    return _digested(
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": status,
            "runner_sha256": EXPECTED_RUNNER_SHA256,
            "continuation_adapter_sha256": (
                EXPECTED_CONTINUATION_ADAPTER_SHA256
            ),
            "activation_manifest_sha256": context.activation.get(
                "sha256", EXPECTED_ACTIVATION_CANONICAL_SHA256
            ),
            "runtime_manifest_sha256": context.runtime["sha256"],
            "target_execution_id": TARGET_EXECUTION_ID,
            "excluded_remote_execution_id": EXCLUDED_REMOTE_EXECUTION_ID,
            "target_job_spec_sha256": EXPECTED_TARGET_JOB_SHA256,
            "target_protocol_sha256": EXPECTED_TARGET_PROTOCOL_SHA256,
            "target_route_contract_sha256": (
                EXPECTED_TARGET_ROUTE_CONTRACT_SHA256
            ),
            "target_source_locks_sha256": (
                EXPECTED_TARGET_SOURCE_LOCKS_SHA256
            ),
            "target_authorization_sha256": (
                EXPECTED_TARGET_AUTHORIZATION_SHA256
            ),
            "sw_closure_receipt_path": SW_CLOSURE_RECEIPT_PATH.as_posix(),
            "sw_closure_receipt_status": receipt_status,
            "sw_terminal_receipt_sha256": (
                None if receipt is None else receipt["sha256"]
            ),
            "wave5_terminal": wave5["terminal"],
            "wave5_status": wave5["status"],
            "target_already_closed": target["closed"],
            "target_decision_sha256": (
                None
                if target["decision"] is None
                else target["decision"]["sha256"]
            ),
            "excluded_remote_local_paths_absent": excluded_absent,
            "standard_wave2_status_absent": standard_wave2_absent,
            "target_attempt_logs_absent": target_logs_absent,
            "local_scientific_overlap": overlap,
            "wave_supervisor_lock_available": lock_available,
            "capacity_ready": capacity_ready,
            "maximum_concurrency": context.runner.MAX_CONCURRENCY,
            "run_ready": run_ready,
            "scientific_execution_performed": False,
            "submission_performed": False,
        }
    )


def _write_status(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        **value,
        "schema": STATUS_SCHEMA,
        "launcher_sha256": _sha256_file(SCRIPT_PATH),
        "updated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    payload = _digested(unsigned)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STATUS_PATH.is_symlink():
        raise SingleCellError("Single-cell status path is an unsafe symlink.")
    temporary = STATUS_PATH.with_name(f".{STATUS_PATH.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise SingleCellError(f"Stale status temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, STATUS_PATH)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def _child_command() -> list[str]:
    return [
        sys.executable,
        "-B",
        RUNNER_PATH.as_posix(),
        "--activation-dir",
        ACTIVATION_DIR.as_posix(),
        "--runtime-dir",
        RUNTIME_DIR.as_posix(),
        "--run-cell",
        TARGET_EXECUTION_ID,
    ]


def _terminate_child(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def _wait_until_ready() -> dict[str, Any]:
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        state = preflight()
        if state["run_ready"]:
            return state
        _write_status(
            {
                "status": state["status"],
                "target_execution_id": TARGET_EXECUTION_ID,
                "excluded_remote_execution_id": EXCLUDED_REMOTE_EXECUTION_ID,
                "preflight_sha256": state["sha256"],
                "running_execution_ids": [],
                "completed_execution_ids": [],
                "maximum_concurrency": 1,
                "scientific_execution_performed_by_status": False,
                "submission_performed": False,
            }
        )
        time.sleep(POLL_SECONDS)
    raise SingleCellError("Timed out waiting for exact single-cell launch gates.")


def supervise() -> dict[str, Any]:
    _wait_until_ready()
    if LOCK_PATH.is_symlink():
        raise SingleCellError("Wave-supervisor lock is an unsafe symlink.")
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(
                lock_stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise SingleCellError(
                "Another Page-16 wave supervisor owns the serialization lock."
            ) from exc
        state = preflight(lock_held=True)
        if not state["run_ready"]:
            raise SingleCellError(
                "Exact single-cell gates changed before serialized execution."
            )
        context = _fixed_context()
        sw_receipt = _authenticate_sw_exclusion(context)
        wave5 = _wave5_terminal_state(context)
        target_before = _target_closure_state(context)
        if not wave5["terminal"]:
            raise SingleCellError("Wave 5 ceased to be terminal before execution.")
        if not _excluded_remote_paths_absent():
            raise SingleCellError("Excluded SW local output appeared before execution.")
        if not _standard_wave2_status_absent():
            raise SingleCellError("Ambiguous standard wave-2 status appeared.")
        overlap = _local_scientific_overlap()
        if overlap:
            raise SingleCellError(
                "Another local scientific worker is active: " + " | ".join(overlap)
            )
        if target_before["closed"]:
            return _write_status(
                {
                    "status": "passed_already_complete_exact_single_cell",
                    "target_execution_id": TARGET_EXECUTION_ID,
                    "excluded_remote_execution_id": EXCLUDED_REMOTE_EXECUTION_ID,
                    "runtime_manifest_sha256": context.runtime["sha256"],
                    "sw_terminal_receipt_sha256": sw_receipt["sha256"],
                    "wave5_status_sha256": wave5["sha256"],
                    "target_decision": target_before["decision"],
                    "running_execution_ids": [],
                    "completed_execution_ids": [TARGET_EXECUTION_ID],
                    "maximum_concurrency": 1,
                    "scientific_execution_performed_by_action": False,
                    "scientific_execution_performed_by_status": False,
                    "submission_performed": False,
                }
            )

        stdout_path = RUNTIME_DIR / "logs" / f"{TARGET_EXECUTION_ID}.out"
        stderr_path = RUNTIME_DIR / "logs" / f"{TARGET_EXECUTION_ID}.err"
        if any(
            path.exists() or path.is_symlink()
            for path in (stdout_path, stderr_path, *_cell_paths(TARGET_EXECUTION_ID))
        ):
            raise SingleCellError("Refusing ambiguous prior weak-strong attempt.")
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
                "TMPDIR": (RUNTIME_DIR / "in_progress").as_posix(),
                context.runner.LOCAL_CHILD_TOKEN_ENV: (
                    f"{context.runtime['sha256']}:wave-2"
                ),
            }
        )
        child: subprocess.Popen[bytes] | None = None
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                child = subprocess.Popen(
                    _child_command(),
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=False,
                )
                _write_status(
                    {
                        "status": "running_exact_weak_strong_always_k30",
                        "target_execution_id": TARGET_EXECUTION_ID,
                        "excluded_remote_execution_id": (
                            EXCLUDED_REMOTE_EXECUTION_ID
                        ),
                        "runtime_manifest_sha256": context.runtime["sha256"],
                        "sw_terminal_receipt_sha256": sw_receipt["sha256"],
                        "wave5_status_sha256": wave5["sha256"],
                        "running_execution_ids": [TARGET_EXECUTION_ID],
                        "running_pid": child.pid,
                        "completed_execution_ids": [],
                        "maximum_concurrency": 1,
                        "scientific_execution_performed_by_status": False,
                        "submission_performed": False,
                    }
                )
                returncode = child.wait()
        except BaseException as exc:
            if child is not None:
                _terminate_child(child)
            _write_status(
                {
                    "status": "interrupted_exact_weak_strong_always_k30",
                    "target_execution_id": TARGET_EXECUTION_ID,
                    "excluded_remote_execution_id": EXCLUDED_REMOTE_EXECUTION_ID,
                    "running_execution_ids": [],
                    "completed_execution_ids": [],
                    "failure_type": type(exc).__name__,
                    "maximum_concurrency": 1,
                    "scientific_execution_performed_by_status": False,
                    "submission_performed": False,
                }
            )
            raise
        if returncode != 0:
            _write_status(
                {
                    "status": "failed_exact_weak_strong_always_k30",
                    "target_execution_id": TARGET_EXECUTION_ID,
                    "excluded_remote_execution_id": EXCLUDED_REMOTE_EXECUTION_ID,
                    "child_returncode": returncode,
                    "running_execution_ids": [],
                    "completed_execution_ids": [],
                    "maximum_concurrency": 1,
                    "scientific_execution_performed_by_status": False,
                    "submission_performed": False,
                }
            )
            raise SingleCellError(
                f"Exact weak-strong child failed with exit code {returncode}."
            )
        try:
            target_after = _target_closure_state(_fixed_context())
            if not target_after["closed"] or target_after["decision"] is None:
                raise SingleCellError(
                    "Exact weak-strong child did not close "
                    "continuation-compatible k30 output."
                )
            if not _excluded_remote_paths_absent():
                raise SingleCellError(
                    "Excluded SW local output appeared during execution."
                )
            if not _standard_wave2_status_absent():
                raise SingleCellError(
                    "Standard wave-2 status was written unexpectedly."
                )
        except BaseException as exc:
            _write_status(
                {
                    "status": "failed_post_child_k30_closure_validation",
                    "target_execution_id": TARGET_EXECUTION_ID,
                    "excluded_remote_execution_id": EXCLUDED_REMOTE_EXECUTION_ID,
                    "child_returncode": returncode,
                    "running_execution_ids": [],
                    "completed_execution_ids": [],
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "maximum_concurrency": 1,
                    "scientific_execution_performed_by_status": False,
                    "submission_performed": False,
                }
            )
            raise
        return _write_status(
            {
                "status": "passed_exact_weak_strong_always_k30",
                "target_execution_id": TARGET_EXECUTION_ID,
                "excluded_remote_execution_id": EXCLUDED_REMOTE_EXECUTION_ID,
                "runtime_manifest_sha256": context.runtime["sha256"],
                "sw_terminal_receipt_sha256": sw_receipt["sha256"],
                "wave5_status_sha256": wave5["sha256"],
                "target_decision": target_after["decision"],
                "running_execution_ids": [],
                "completed_execution_ids": [TARGET_EXECUTION_ID],
                "maximum_concurrency": 1,
                "scientific_execution_performed_by_action": True,
                "scientific_execution_performed_by_status": False,
                "submission_performed": False,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Supervise only the Page-16 weak-strong always-open local k30 cell"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        value = preflight() if args.preflight else supervise()
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        SingleCellError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        # Imported pinned modules expose their own exception types only after
        # import; preserve those failures without permitting partial progress.
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(_canonical_json_bytes(value).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
