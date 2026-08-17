#!/usr/bin/env python3
"""Wait for all Page-16 k30 decisions and serialize eligible k50 resumes.

This local-only supervisor is inert until all nine conditional cells have
authenticated, runner-native k30 closure.  It also validates three CHTC k50
always-open terminals, including the ``9647386.1`` remote-materialization
exclusion closure.  A
fresh, short-lived authenticated CHTC no-overlap clearance is required before
the first eligible continuation.  Scientific children then run strictly one
at a time through the pinned continuation adapter.
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
import signal
import subprocess
import sys
import time
from typing import Any, Mapping


SUPERVISOR_PATH = Path(__file__).resolve()
REPO_ROOT = SUPERVISOR_PATH.parents[2]
ADAPTER_PATH = SUPERVISOR_PATH.with_name(
    "continue_local_page16_insertion_comparators_k30_to_k50_20260813.py"
)
EXPECTED_ADAPTER_SHA256 = (
    "56c50f046759d4299d768cb609f08fce8c79e3190aaadb6609afdde4f5452e07"
)
ACTIVATION_DIR = SUPERVISOR_PATH.with_name(
    "paper_i_ra_adapt_page16_insertion_comparators_k30_to_k50_"
    "20260813_v2_local_activation"
)
RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page16_insertion_comparators_k30_to_k50_20260813_v2"
)
REMOTE_CLEARANCE_DIR = SUPERVISOR_PATH.with_name(
    "paper_i_ra_adapt_page16_insertion_comparators_k30_to_k50_"
    "20260813_v2_remote_overlap_clearances"
)
MACRO_TERMINAL_RECEIPT_PATH = SUPERVISOR_PATH.with_name(
    "paper_i_ra_adapt_page16_macro_k30_k50_terminal_clearance_20260813.json"
)
POLL_SECONDS = 20
WAIT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
REMOTE_CLEARANCE_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_no_remote_overlap_clearance_v2"
)
REMOTE_CLEARANCE_AUTHENTICATION_KIND = (
    "interactive_ssh_duo_condor_q_snapshot_v1"
)
REMOTE_CLEARANCE_MAX_WINDOW_SECONDS = 15 * 60
STATUS_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_supervisor_status_v2"
)
PREFLIGHT_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_supervisor_preflight_v2"
)
MACRO_TERMINAL_SCHEMA = (
    "paper_i_page16_insertion_comparator_macro_k30_k50_terminal_clearance_v1"
)
SUPERVISOR_LOCK = RUNTIME_DIR.with_name(f".{RUNTIME_DIR.name}.supervisor.lock")


class SupervisorError(RuntimeError):
    pass


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_adapter() -> Any:
    if (
        not ADAPTER_PATH.is_file()
        or ADAPTER_PATH.is_symlink()
        or _sha256_file(ADAPTER_PATH) != EXPECTED_ADAPTER_SHA256
    ):
        raise SupervisorError("Pinned continuation adapter is absent or unsafe.")
    name = "paper_i_page16_k50_continuation_adapter_for_supervisor"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise SupervisorError("Continuation adapter cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_activation(adapter: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    activation, bundle = adapter._validate_activation(
        adapter.k30._load_worker(), ACTIVATION_DIR
    )
    return activation, bundle


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SupervisorError(f"{label} is absent or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupervisorError(f"{label} must be a JSON object.")
    expected = value.get("sha256")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if expected != _canonical_sha256(unsigned):
        raise SupervisorError(f"{label} self-digest drifted.")
    return value


def _write_status(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        **value,
        "schema": STATUS_SCHEMA,
        "adapter_sha256": _sha256_file(ADAPTER_PATH),
        "updated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    payload = {**unsigned, "sha256": _canonical_sha256(unsigned)}
    status_path = RUNTIME_DIR / "status/supervisor.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_name(f".{status_path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise SupervisorError(f"Stale supervisor status temporary: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, status_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def _utc_datetime(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SupervisorError(f"{label} is not ISO-8601.") from exc
    if parsed.tzinfo is None:
        raise SupervisorError(f"{label} must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


def _sha256_text(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_remote_overlap_clearance(
    value: Mapping[str, Any],
    *,
    execution_ids: tuple[str, ...],
    adapter_sha256: str,
    activation_manifest_sha256: str,
    k30_runtime_manifest_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    observed_at = _utc_datetime(
        value.get("observed_at_utc"), label="clearance observed_at_utc"
    )
    valid_until = _utc_datetime(
        value.get("valid_until_utc"), label="clearance valid_until_utc"
    )
    current = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    if (
        value.get("schema") != REMOTE_CLEARANCE_SCHEMA
        or value.get("status")
        != "passed_authenticated_no_remote_overlap_clearance"
        or value.get("execution_ids") != list(execution_ids)
        or value.get("adapter_sha256") != adapter_sha256
        or value.get("activation_manifest_sha256")
        != activation_manifest_sha256
        or value.get("k30_runtime_manifest_sha256")
        != k30_runtime_manifest_sha256
        or value.get("authentication_kind")
        != REMOTE_CLEARANCE_AUTHENTICATION_KIND
        or value.get("authenticated_remote_query") is not True
        or value.get("scheduler") != "chtc_condor"
        or not _sha256_text(value.get("scheduler_snapshot_sha256"))
        or value.get("remote_active_execution_ids") != []
        or value.get("overlapping_execution_ids") != []
        or value.get("remote_factories_frozen") is not True
        or value.get("no_remote_overlap") is not True
        or value.get("scientific_execution_performed") is not False
        or observed_at > current
        or current > valid_until
        or valid_until <= observed_at
        or (valid_until - observed_at).total_seconds()
        > REMOTE_CLEARANCE_MAX_WINDOW_SECONDS
    ):
        raise SupervisorError("Authenticated remote-overlap clearance drifted.")
    return dict(value)


def _require_all_decisions(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    adapter = _load_adapter()
    decisions = snapshot.get("decisions")
    eligible = snapshot.get("eligible_execution_ids")
    stopped = snapshot.get("stop_at_k30_execution_ids")
    pending = snapshot.get("pending_execution_ids")
    if (
        snapshot.get("status") != "passed_all_k30_decisions_closed"
        or snapshot.get("all_decisions_closed") is not True
        or snapshot.get("closed_decision_count")
        != len(adapter.CONDITIONAL_EXECUTION_IDS)
        or not isinstance(decisions, list)
        or [row.get("execution_id") for row in decisions]
        != list(adapter.CONDITIONAL_EXECUTION_IDS)
        or not isinstance(eligible, list)
        or not isinstance(stopped, list)
        or pending != []
        or set(eligible).intersection(stopped)
        or set(eligible).union(stopped)
        != set(adapter.CONDITIONAL_EXECUTION_IDS)
    ):
        raise SupervisorError(
            "Cannot continue before all nine k30 decisions close exactly."
        )
    return dict(snapshot)


def _require_all_terminals(status: Mapping[str, Any]) -> dict[str, Any]:
    adapter = _load_adapter()
    receipts = status.get("authenticated_terminal_receipts")
    if (
        status.get("schema") != adapter.TERMINAL_STATUS_SCHEMA
        or status.get("status")
        != "passed_all_three_authenticated_chtc_k50_terminals"
        or status.get("all_terminal_cells_authenticated") is not True
        or status.get("authenticated_terminal_count")
        != len(adapter.TERMINAL_CHTC_EXECUTION_IDS)
        or status.get("terminal_chtc_k50_execution_ids")
        != list(adapter.TERMINAL_CHTC_EXECUTION_IDS)
        or not isinstance(receipts, list)
        or [row.get("execution_id") for row in receipts]
        != list(adapter.TERMINAL_CHTC_EXECUTION_IDS)
        or status.get("pending_execution_ids") != []
        or status.get("validation_errors") != {}
        or status.get("scientific_execution_performed") is not False
    ):
        raise SupervisorError(
            "Cannot continue before all three authenticated CHTC k50 terminals close exactly."
        )
    return dict(status)


def _require_remote_clearance(
    *,
    runtime: Mapping[str, Any],
    activation: Mapping[str, Any],
    execution_id: str,
) -> dict[str, Any]:
    eligible = tuple(str(row) for row in runtime["eligible_execution_ids"])
    if execution_id not in eligible:
        raise SupervisorError("Remote clearance requested outside eligible cells.")
    clearance = _load_digested(
        REMOTE_CLEARANCE_DIR / f"{execution_id}.json",
        label="authenticated k50 remote-overlap clearance",
    )
    return _validate_remote_overlap_clearance(
        clearance,
        execution_ids=(execution_id,),
        adapter_sha256=str(runtime["adapter_sha256"]),
        activation_manifest_sha256=str(activation["sha256"]),
        k30_runtime_manifest_sha256=str(runtime["k30_runtime_manifest_sha256"]),
    )


def _wait_for_remote_clearance(
    *,
    runtime: Mapping[str, Any],
    activation: Mapping[str, Any],
    execution_id: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            return _require_remote_clearance(
                runtime=runtime,
                activation=activation,
                execution_id=execution_id,
            )
        except SupervisorError as exc:
            last_error = str(exc)
            _write_status(
                {
                    "status": "waiting_for_authenticated_remote_clearance",
                    "eligible_execution_ids": runtime[
                        "eligible_execution_ids"
                    ],
                    "stop_at_k30_execution_ids": runtime[
                        "stop_at_k30_execution_ids"
                    ],
                    "next_execution_id": execution_id,
                    "clearance_path": (
                        REMOTE_CLEARANCE_DIR / f"{execution_id}.json"
                    ).as_posix(),
                    "last_clearance_error": last_error,
                    "running_execution_ids": [],
                    "maximum_concurrency": 1,
                }
            )
            time.sleep(POLL_SECONDS)
    raise SupervisorError(
        "Timed out waiting for authenticated remote-overlap clearance: "
        f"{execution_id}; last error: {last_error}"
    )


def preflight() -> dict[str, Any]:
    adapter = _load_adapter()
    activation_status = "absent"
    activation: dict[str, Any] | None = None
    if ACTIVATION_DIR.exists() or ACTIVATION_DIR.is_symlink():
        activation, _bundle = _validate_activation(adapter)
        activation_status = "validated"
    snapshot = adapter.decision_snapshot()
    terminal_status = adapter.terminal_chtc_status()
    runtime_status = "absent"
    if RUNTIME_DIR.exists() or RUNTIME_DIR.is_symlink():
        adapter._validate_runtime(
            adapter.k30._load_worker(),
            activation_dir=ACTIVATION_DIR,
            runtime_dir=RUNTIME_DIR,
        )
        runtime_status = "validated"
    return adapter._digested(
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": (
                "passed_waiting_for_authenticated_hybrid_inputs"
                if not terminal_status["all_terminal_cells_authenticated"]
                else (
                    "passed_waiting_for_all_k30_decisions"
                    if not snapshot["all_decisions_closed"]
                    else (
                        "passed_ready_to_prepare_activation"
                        if activation_status == "absent"
                        else "passed_hybrid_inputs_closed"
                    )
                )
            ),
            "adapter_sha256": _sha256_file(ADAPTER_PATH),
            "activation_status": activation_status,
            "runtime_status": runtime_status,
            "decision_status": snapshot,
            "terminal_chtc_status": terminal_status,
            "all_decisions_closed": snapshot["all_decisions_closed"],
            "all_terminal_cells_authenticated": terminal_status[
                "all_terminal_cells_authenticated"
            ],
            "campaign_inputs_closed": (
                snapshot["all_decisions_closed"]
                and terminal_status["all_terminal_cells_authenticated"]
            ),
            "eligible_execution_ids": snapshot["eligible_execution_ids"],
            "stop_at_k30_execution_ids": snapshot[
                "stop_at_k30_execution_ids"
            ],
            "terminal_chtc_k50_execution_ids": list(
                adapter.TERMINAL_CHTC_EXECUTION_IDS
            ),
            "remote_overlap_clearance_required_before_each_child": True,
            "remote_overlap_clearance_directory": REMOTE_CLEARANCE_DIR.as_posix(),
            "maximum_concurrency": adapter.MAX_CONCURRENCY,
            "scientific_execution_performed": False,
            "submission_performed": False,
        }
    )


def _wait_for_campaign_inputs(
    adapter: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
    decision_cache: dict[str, dict[str, Any]] = {}
    terminal_cache: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        snapshot = adapter.decision_snapshot(cached=decision_cache)
        terminal_status = adapter.terminal_chtc_status(cached=terminal_cache)
        if (
            snapshot["all_decisions_closed"]
            and terminal_status["all_terminal_cells_authenticated"]
        ):
            # Reauthenticate every closed gate and byte binding once at the
            # transition from waiting to executable campaign state.  The cache
            # only reduces repeated hashing while other cells remain open.
            final_snapshot = _require_all_decisions(
                adapter.decision_snapshot(cached={})
            )
            final_terminals = _require_all_terminals(
                adapter.terminal_chtc_status(cached={})
            )
            return final_snapshot, final_terminals
        if RUNTIME_DIR.exists() and (RUNTIME_DIR / "status").is_dir():
            _write_status(
                {
                    "status": "waiting_for_hybrid_k30_k50_inputs",
                    "closed_decision_count": snapshot["closed_decision_count"],
                    "pending_execution_ids": snapshot["pending_execution_ids"],
                    "eligible_execution_ids": snapshot[
                        "eligible_execution_ids"
                    ],
                    "stop_at_k30_execution_ids": snapshot[
                        "stop_at_k30_execution_ids"
                    ],
                    "terminal_chtc_status": terminal_status,
                    "running_execution_ids": [],
                    "completed_execution_ids": [],
                    "maximum_concurrency": adapter.MAX_CONCURRENCY,
                }
            )
        time.sleep(POLL_SECONDS)
    raise SupervisorError(
        "Timed out waiting for all nine local k30 decisions and three "
        "authenticated CHTC k50 terminals."
    )


def _child_command(adapter: Any, execution_id: str) -> list[str]:
    return [
        sys.executable,
        "-B",
        ADAPTER_PATH.as_posix(),
        "--activation-dir",
        ACTIVATION_DIR.as_posix(),
        "--runtime-dir",
        RUNTIME_DIR.as_posix(),
        "--run-cell",
        execution_id,
    ]


def _terminate_child(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is not None:
        return
    child.send_signal(signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def _emit_macro_terminal_receipt(
    adapter: Any,
    *,
    runtime: Mapping[str, Any],
    activation: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    terminal_status: Mapping[str, Any],
    path: Path = MACRO_TERMINAL_RECEIPT_PATH,
) -> dict[str, Any]:
    closed_snapshot = _require_all_decisions(snapshot)
    closed_terminals = _require_all_terminals(terminal_status)
    eligible = list(closed_snapshot["eligible_execution_ids"])
    stopped = list(closed_snapshot["stop_at_k30_execution_ids"])
    if (
        runtime.get("decision_status_sha256") != closed_snapshot.get("sha256")
        or runtime.get("conditional_execution_ids")
        != list(adapter.CONDITIONAL_EXECUTION_IDS)
        or runtime.get("eligible_execution_ids") != eligible
        or runtime.get("stop_at_k30_execution_ids") != stopped
        or runtime.get("terminal_chtc_k50_execution_ids")
        != list(adapter.TERMINAL_CHTC_EXECUTION_IDS)
        or activation.get("terminal_chtc_k50_execution_ids")
        != list(adapter.TERMINAL_CHTC_EXECUTION_IDS)
    ):
        raise SupervisorError("Macro terminal inventory or provenance drifted.")
    unclosed = [
        execution_id
        for execution_id in eligible
        if not adapter.closed_continuation_cell(
            runtime_dir=RUNTIME_DIR,
            execution_id=execution_id,
        )
    ]
    if unclosed:
        raise SupervisorError(
            "Cannot emit macro terminal receipt before every eligible k50 "
            "continuation closes: " + ", ".join(unclosed)
        )
    payload = adapter._digested(
        {
            "schema": MACRO_TERMINAL_SCHEMA,
            "status": "passed_all_required_macro_k30_k50_work_terminal",
            "adapter_sha256": _sha256_file(ADAPTER_PATH),
            "activation_manifest_sha256": activation["sha256"],
            "runtime_manifest_sha256": runtime["sha256"],
            "k30_runtime_manifest_sha256": runtime[
                "k30_runtime_manifest_sha256"
            ],
            "decision_status_sha256": closed_snapshot["sha256"],
            "terminal_chtc_status_sha256": closed_terminals["sha256"],
            "conditional_execution_ids": list(
                adapter.CONDITIONAL_EXECUTION_IDS
            ),
            "terminal_chtc_k50_execution_ids": list(
                adapter.TERMINAL_CHTC_EXECUTION_IDS
            ),
            "eligible_k50_continuation_execution_ids": eligible,
            "stop_at_k30_execution_ids": stopped,
            "closed_k50_continuation_execution_ids": eligible,
            "all_k30_cells_closed": True,
            "all_extension_required_cells_closed_at_k50": True,
            "remaining_macro_execution_ids": [],
            "active_macro_execution_ids": [],
            "scientific_execution_performed_by_receipt": False,
        }
    )
    if path.exists() or path.is_symlink():
        observed = _load_digested(path, label="Page12 macro-terminal receipt")
        if observed != payload:
            raise SupervisorError("Existing Page12 macro-terminal receipt drifted.")
        return observed
    adapter._write_json(path, payload, exclusive=True)
    return payload


def run_supervisor() -> int:
    adapter = _load_adapter()
    SUPERVISOR_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with SUPERVISOR_LOCK.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorError("Another continuation supervisor owns the lock.") from exc
        snapshot, terminal_status = _wait_for_campaign_inputs(adapter)
        if not ACTIVATION_DIR.exists() and not ACTIVATION_DIR.is_symlink():
            adapter.prepare_activation(activation_dir=ACTIVATION_DIR)
        activation, _bundle = _validate_activation(adapter)
        if not RUNTIME_DIR.exists():
            runtime = adapter.initialize_runtime(
                activation_dir=ACTIVATION_DIR,
                runtime_dir=RUNTIME_DIR,
                snapshot=snapshot,
            )
        else:
            runtime, _activation, _bundle = adapter._validate_runtime(
                adapter.k30._load_worker(),
                activation_dir=ACTIVATION_DIR,
                runtime_dir=RUNTIME_DIR,
            )
        if runtime.get("decision_status_sha256") != snapshot.get("sha256"):
            raise SupervisorError(
                "Runtime decision snapshot does not match the final "
                "reauthenticated nine-cell closure."
            )
        eligible = list(runtime["eligible_execution_ids"])
        completed = [
            execution_id
            for execution_id in eligible
            if adapter.closed_continuation_cell(
                runtime_dir=RUNTIME_DIR,
                execution_id=execution_id,
            )
        ]
        worker = adapter.k30._load_worker()
        jobs = adapter._job_by_id(worker)
        for execution_id in eligible:
            if execution_id in completed:
                continue
            capacity = adapter.capacity_receipt(runtime_dir=RUNTIME_DIR)
            if capacity["status"] != "passed":
                raise SupervisorError(
                    "Host capacity fell below the continuation guard: "
                    + ", ".join(capacity["blockers"])
                )
            source_decision = adapter._closed_k30_decision(
                worker,
                execution_id=execution_id,
                job=jobs[execution_id],
            )
            authority, _runtime, _bundle = adapter._resume_authorization(
                worker,
                activation_dir=ACTIVATION_DIR,
                runtime_dir=RUNTIME_DIR,
                execution_id=execution_id,
            )
            if (
                source_decision is None
                or source_decision.get("extension_decision")
                != "eligible_for_authenticated_resume_to_k50"
                or source_decision.get("k30_execution_manifest_sha256")
                != authority.get("k30_execution_manifest_sha256")
                or source_decision.get("k30_worker_receipt_sha256")
                != authority.get("k30_worker_receipt_sha256")
                or source_decision.get("k30_plateau_gate_sha256")
                != authority.get("k30_plateau_gate_sha256")
                or source_decision.get("resume_checkpoint")
                != authority.get("resume_checkpoint")
                or source_decision.get("resume_checkpoint_siblings")
                != authority.get("resume_checkpoint_siblings")
            ):
                raise SupervisorError(
                    f"Live k30 source reauthentication drifted: {execution_id}"
                )
            # Authenticate remote non-overlap last so the short-lived receipt
            # remains fresh after potentially expensive local artifact hashing.
            _wait_for_remote_clearance(
                runtime=runtime,
                activation=activation,
                execution_id=execution_id,
            )
            capacity = adapter.capacity_receipt(runtime_dir=RUNTIME_DIR)
            if capacity["status"] != "passed":
                raise SupervisorError(
                    "Host capacity fell below the continuation guard after "
                    "remote clearance: " + ", ".join(capacity["blockers"])
                )
            _write_status(
                {
                    "status": "running_eligible_continuation",
                    "closed_decision_count": len(
                        adapter.CONDITIONAL_EXECUTION_IDS
                    ),
                    "eligible_execution_ids": eligible,
                    "stop_at_k30_execution_ids": runtime[
                        "stop_at_k30_execution_ids"
                    ],
                    "completed_execution_ids": completed,
                    "running_execution_ids": [execution_id],
                    "maximum_concurrency": adapter.MAX_CONCURRENCY,
                }
            )
            environment = dict(os.environ)
            environment.update(
                {
                    adapter.LOCAL_CHILD_TOKEN_ENV: (
                        f"{runtime['sha256']}:{execution_id}"
                    ),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "STATIC_ADAPT_HH_POOL_CACHE": "off",
                    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
                    "TMPDIR": (RUNTIME_DIR / "in_progress").as_posix(),
                }
            )
            stdout_path = RUNTIME_DIR / "logs" / f"{execution_id}.out"
            stderr_path = RUNTIME_DIR / "logs" / f"{execution_id}.err"
            child: subprocess.Popen[bytes] | None = None
            try:
                with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                    child = subprocess.Popen(
                        _child_command(adapter, execution_id),
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    return_code = child.wait()
            except BaseException:
                if child is not None:
                    _terminate_child(child)
                raise
            if return_code != 0:
                _write_status(
                    {
                        "status": "failed",
                        "failed_execution_id": execution_id,
                        "exit_code": return_code,
                        "completed_execution_ids": completed,
                        "running_execution_ids": [],
                        "maximum_concurrency": adapter.MAX_CONCURRENCY,
                    }
                )
                return return_code
            if not adapter.closed_continuation_cell(
                runtime_dir=RUNTIME_DIR,
                execution_id=execution_id,
            ):
                raise SupervisorError(f"Continuation did not close: {execution_id}")
            completed.append(execution_id)
        final_snapshot = _require_all_decisions(
            adapter.decision_snapshot(cached={})
        )
        final_terminals = _require_all_terminals(
            adapter.terminal_chtc_status(cached={})
        )
        macro_terminal_receipt = _emit_macro_terminal_receipt(
            adapter,
            runtime=runtime,
            activation=activation,
            snapshot=final_snapshot,
            terminal_status=final_terminals,
        )
        _write_status(
            {
                "status": "passed_all_conditional_decisions_resolved",
                "closed_decision_count": len(
                    adapter.CONDITIONAL_EXECUTION_IDS
                ),
                "eligible_execution_ids": eligible,
                "stop_at_k30_execution_ids": runtime[
                    "stop_at_k30_execution_ids"
                ],
                "completed_execution_ids": completed,
                "running_execution_ids": [],
                "terminal_chtc_k50_execution_ids": list(
                    adapter.TERMINAL_CHTC_EXECUTION_IDS
                ),
                "maximum_concurrency": adapter.MAX_CONCURRENCY,
                "macro_terminal_receipt_path": (
                    MACRO_TERMINAL_RECEIPT_PATH.as_posix()
                ),
                "macro_terminal_receipt_sha256": macro_terminal_receipt[
                    "sha256"
                ],
            }
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Supervise conditional Page-16 k30 to k50 continuations"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    try:
        if args.preflight:
            print(_canonical_json_bytes(preflight()).decode("utf-8"))
            return 0
        return run_supervisor()
    except (SupervisorError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
