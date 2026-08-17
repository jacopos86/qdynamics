#!/usr/bin/env python3
"""Run the user-priority weak-Holstein 12, then the unique matched strong six.

The existing matched-12 activation remains immutable.  Its six weak-sector
plateau/Append cells are executed first and later recognized by its normal
restart logic, so they are never duplicated.  Between those two tranches this
supervisor runs the separately authorized six weak-sector RA insertion-policy
cells.  Finally it resumes the ordinary matched runner, which executes only
the six still-missing strong-Holstein cells and closes its original terminal.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPAIR_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
MATCHED_RUNNER_PATH = REPAIR_ROOT / (
    "run_local_paper_i_page12_matched_singleton12_r50_20260815.py"
)
WEAK_RA6_RUNNER_PATH = REPAIR_ROOT / (
    "run_local_page12_weak_holstein_priority6_20260815.py"
)
ORCHESTRATOR_STATUS = REPO_ROOT / (
    "output/local_runs/paper_i_weak12_priority_then_matched_unique6_"
    "20260815_v1/status.json"
)

WEAK_REGIMES = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
)

# A prior attempt entered with 5.65 GiB available and later crossed the
# matched runner's immutable 2-GiB emergency floor.  Keep that floor intact,
# but use a 7-GiB high-water mark before starting or retrying priority cells.
# This is launch hysteresis, not a change to scientific or guard semantics.
MEMORY_RETRY_AVAILABLE_BYTES = 7 * 1024**3
MEMORY_RETRY_POLL_SECONDS = 30


class PriorityScheduleError(RuntimeError):
    """Raised when the deduplicated priority schedule cannot proceed safely."""


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PriorityScheduleError(f"Unable to import {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PriorityScheduleError(f"Expected JSON object: {path}")
    return value


def _write_status(runner: Any, status: str, **fields: Any) -> None:
    payload = runner._digested(
        {
            "schema": "paper_i_weak12_then_matched_unique6_status_v1",
            "status": status,
            "schedule": [
                "weak_holstein_ra_plateau_three",
                "weak_holstein_ra_append_only_three",
                "weak_holstein_ra_always_open_three",
                "weak_holstein_conventional_append_three",
                "matched_suite_unique_strong_holstein_six",
            ],
            "maximum_concurrency": 1,
            "duplicate_execution_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
            **fields,
        }
    )
    runner._write_json_atomic(ORCHESTRATOR_STATUS, payload)


def _matched_execution_ids(runner: Any, method: str) -> tuple[str, ...]:
    manifest = runner._package_manifest()
    rows = runner._cell_rows(manifest)
    by_key = {
        (str(row["regime"]), int(row["n_ph"]), str(row["method"])): str(
            row["execution_id"]
        )
        for row in rows
    }
    return tuple(by_key[(regime, nph, method)] for regime, nph in WEAK_REGIMES)


def _retryable_memory_guard(runner: Any, exc: BaseException) -> bool:
    return isinstance(exc, runner.MatchedSingleton12Error) and str(exc).endswith(
        "available_memory_floor_breached."
    )


def _wait_for_memory_headroom(
    runner: Any,
    *,
    phase: str,
    execution_id: str,
    retry_count: int,
    last_failure: str | None,
) -> dict[str, Any]:
    """Persistently wait for launch hysteresis without changing the guard."""

    while True:
        capacity = runner._capacity(runner.DEFAULT_RUNTIME_DIR)
        available = int(capacity["available_memory_bytes"])
        if available >= MEMORY_RETRY_AVAILABLE_BYTES:
            return capacity
        _write_status(
            runner,
            "waiting_for_memory_headroom",
            priority_phase=phase,
            current_execution_id=execution_id,
            retry_count=retry_count,
            last_retryable_failure=last_failure,
            available_memory_bytes=available,
            minimum_retry_available_memory_bytes=MEMORY_RETRY_AVAILABLE_BYTES,
            poll_seconds=MEMORY_RETRY_POLL_SECONDS,
        )
        time.sleep(MEMORY_RETRY_POLL_SECONDS)


def _run_guarded_child_with_memory_retry(
    runner: Any,
    *,
    cell: Mapping[str, Any],
    activation: Mapping[str, Any],
    handoff: Mapping[str, Any],
    phase: str,
    completed_execution_ids: Sequence[str],
    recorded_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Retry only recoverable memory-floor stops; all other failures escape."""

    execution_id = str(cell["execution_id"])
    retry_count = 0
    last_failure = None
    while True:
        launch_capacity = _wait_for_memory_headroom(
            runner,
            phase=phase,
            execution_id=execution_id,
            retry_count=retry_count,
            last_failure=last_failure,
        )
        runtime_check = None
        if recorded_runtime is not None:
            runtime_check = runner._runtime_check(recorded_runtime, cell)
            runner._write_json_atomic(
                runner.DEFAULT_RUNTIME_DIR / "status/campaign.json",
                runner._status(
                    "priority_running_serial_cell",
                    priority_phase=phase,
                    current_execution_id=execution_id,
                    completed_execution_ids=list(completed_execution_ids),
                    retry_count=retry_count,
                    launch_available_memory_bytes=int(
                        launch_capacity["available_memory_bytes"]
                    ),
                    runtime_check_sha256=runtime_check["sha256"],
                ),
            )
        try:
            return runner._run_guarded_child(
                cell=cell, activation=activation, handoff=handoff
            )
        except BaseException as exc:
            if not _retryable_memory_guard(runner, exc):
                raise
            retry_count += 1
            last_failure = str(exc)
            _write_status(
                runner,
                "retrying_after_memory_guard",
                priority_phase=phase,
                current_execution_id=execution_id,
                completed_execution_ids=list(completed_execution_ids),
                retry_count=retry_count,
                last_retryable_failure=last_failure,
                minimum_retry_available_memory_bytes=MEMORY_RETRY_AVAILABLE_BYTES,
            )


def _run_selected_matched_cells(
    runner: Any,
    execution_ids: Sequence[str],
    *,
    phase: str,
) -> list[str]:
    activation, plan, authorization, recorded_runtime = runner._validate_activation()
    handoff = runner._validate_handoff(activation)
    runtime = runner._ensure_runtime(activation, handoff)
    archive = runner._archive_module()
    cells = {str(row["execution_id"]): row for row in plan["cells"]}
    pair_cells = {
        (str(row["regime"]), str(row["method"])): row for row in plan["cells"]
    }
    if any(execution_id not in cells for execution_id in execution_ids):
        raise PriorityScheduleError(f"{phase} escaped the matched activation.")

    completed: list[str] = []
    lock_path = runner.DEFAULT_RUNTIME_DIR / "campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PriorityScheduleError(
                "The matched campaign lock is already owned."
            ) from exc
        try:
            for execution_id in execution_ids:
                cell = cells[execution_id]
                existing = runner._existing_archive_closure(
                    module=archive,
                    cell=cell,
                    activation=activation,
                    authorization=authorization,
                    handoff=handoff,
                    runtime=runtime,
                )
                if existing is not None:
                    completed.append(execution_id)
                    continue
                direct = runner.DEFAULT_RUNTIME_DIR / "runs" / execution_id
                if direct.exists() or direct.is_symlink():
                    raise PriorityScheduleError(
                        f"Partial matched cell requires inspection: {execution_id}"
                    )
                _run_guarded_child_with_memory_retry(
                    runner,
                    cell=cell,
                    activation=activation,
                    handoff=handoff,
                    phase=phase,
                    completed_execution_ids=completed,
                    recorded_runtime=recorded_runtime,
                )
                if cell["method"] == "append_singleton":
                    runner._publish_pair_parity(
                        append_cell=cell,
                        ra_cell=pair_cells[
                            (str(cell["regime"]), "ra_singleton_plateau")
                        ],
                    )
                closure = runner._archive_and_rotate(
                    module=archive,
                    cell=cell,
                    activation=activation,
                    authorization=authorization,
                    handoff=handoff,
                    runtime=runtime,
                )
                completed.append(execution_id)
                runner._write_json_atomic(
                    runner.DEFAULT_RUNTIME_DIR / "status/campaign.json",
                    runner._status(
                        "priority_cell_passed_archived_pending_remaining",
                        priority_phase=phase,
                        current_execution_id=None,
                        completed_execution_ids=completed,
                        archive_closure_sha256=closure["sha256"],
                    ),
                )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return completed


def _run_weak_ra6() -> None:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
            "-B",
            WEAK_RA6_RUNNER_PATH.as_posix(),
            "--run-campaign",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise PriorityScheduleError(
            f"Weak-Holstein RA insertion-policy runner failed: {completed.returncode}."
        )


def preflight() -> dict[str, Any]:
    """Validate both authorities, exact order, locks, and deduplication inertly."""

    runner = _load_module(MATCHED_RUNNER_PATH, "paper_i_priority_matched_preflight")
    weak_runner = _load_module(
        WEAK_RA6_RUNNER_PATH, "paper_i_priority_weak_ra6_preflight"
    )
    weak = weak_runner.inert_preflight(
        planning_dir=weak_runner.DEFAULT_PLANNING_DIR,
        activation_dir=weak_runner.DEFAULT_ACTIVATION_DIR,
        runtime_dir=weak_runner.DEFAULT_RUNTIME_DIR,
    )
    if weak.get("run_ready") is not True:
        raise PriorityScheduleError("Weak RA six-cell preflight is not ready.")
    handoff = _load_json(runner.DEFAULT_HANDOFF_RECEIPT)
    with runner.DEFAULT_HANDOFF_LOCK.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PriorityScheduleError("The matched handoff lock is active.") from exc
        os.set_inheritable(lock.fileno(), True)
        os.environ.update(runner.REQUIRED_NUMERICAL_ENVIRONMENT)
        os.environ[runner.HANDOFF_RECEIPT_ENV] = (
            runner.DEFAULT_HANDOFF_RECEIPT.resolve().as_posix()
        )
        os.environ[runner.HANDOFF_TOKEN_ENV] = hashlib.sha256(
            f"{handoff['sha256']}:matched-singleton12-target-launch-v1".encode(
                "utf-8"
            )
        ).hexdigest()
        os.environ[runner.HANDOFF_LOCK_FD_ENV] = str(lock.fileno())
        activation, plan, _authorization, _runtime = runner._validate_activation()
        runner._validate_handoff(activation)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    plateau_ids = _matched_execution_ids(runner, "ra_singleton_plateau")
    append_ids = _matched_execution_ids(runner, "append_singleton")
    all_ids = (
        *plateau_ids,
        *weak_runner.TARGET_EXECUTION_IDS,
        *append_ids,
        *tuple(
            str(row["execution_id"])
            for row in plan["cells"]
            if int(row["n_ph"]) == 7
        ),
    )
    if len(all_ids) != 18 or len(set(all_ids)) != 18:
        raise PriorityScheduleError("The priority schedule is not exactly 18 unique cells.")
    return runner._digested(
        {
            "schema": "paper_i_weak12_then_matched_unique6_preflight_v1",
            "status": "passed_inert_exact_18_unique_cells",
            "weak_plateau_execution_ids": list(plateau_ids),
            "weak_ra_insertion_execution_ids": list(
                weak_runner.TARGET_EXECUTION_IDS
            ),
            "weak_conventional_append_execution_ids": list(append_ids),
            "unique_matched_strong_execution_ids": [
                str(row["execution_id"])
                for row in plan["cells"]
                if int(row["n_ph"]) == 7
            ],
            "execution_count": 18,
            "duplicate_execution_authorized": False,
            "maximum_concurrency": 1,
            "scientific_execution_performed": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def run() -> dict[str, Any]:
    runner = _load_module(MATCHED_RUNNER_PATH, "paper_i_priority_matched_runner")
    weak_runner = _load_module(WEAK_RA6_RUNNER_PATH, "paper_i_priority_weak_ra6")
    if weak_runner.DEFAULT_RUNTIME_DIR == runner.DEFAULT_RUNTIME_DIR:
        raise PriorityScheduleError("Priority runtimes unexpectedly collide.")

    handoff = _load_json(runner.DEFAULT_HANDOFF_RECEIPT)
    runner._load_digested(
        runner.DEFAULT_HANDOFF_RECEIPT, label="priority handoff receipt"
    )
    lock_path = runner.DEFAULT_HANDOFF_LOCK
    with lock_path.open("a+", encoding="utf-8") as handoff_lock:
        try:
            fcntl.flock(
                handoff_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise PriorityScheduleError(
                "The original matched handoff still owns its lock."
            ) from exc
        os.set_inheritable(handoff_lock.fileno(), True)
        os.environ.update(runner.REQUIRED_NUMERICAL_ENVIRONMENT)
        os.environ[runner.HANDOFF_RECEIPT_ENV] = (
            runner.DEFAULT_HANDOFF_RECEIPT.resolve().as_posix()
        )
        os.environ[runner.HANDOFF_TOKEN_ENV] = hashlib.sha256(
            f"{handoff['sha256']}:matched-singleton12-target-launch-v1".encode(
                "utf-8"
            )
        ).hexdigest()
        os.environ[runner.HANDOFF_LOCK_FD_ENV] = str(handoff_lock.fileno())

        plateau_ids = _matched_execution_ids(runner, "ra_singleton_plateau")
        append_ids = _matched_execution_ids(runner, "append_singleton")
        _write_status(
            runner,
            "running_weak_holstein_ra_plateau_three",
            current_execution_ids=list(plateau_ids),
        )
        _run_selected_matched_cells(
            runner, plateau_ids, phase="weak_holstein_ra_plateau_three"
        )

        _write_status(
            runner,
            "running_weak_holstein_ra_insertion_six",
            current_execution_ids=list(weak_runner.TARGET_EXECUTION_IDS),
        )
        _run_weak_ra6()

        _write_status(
            runner,
            "running_weak_holstein_conventional_append_three",
            current_execution_ids=list(append_ids),
        )
        _run_selected_matched_cells(
            runner,
            append_ids,
            phase="weak_holstein_conventional_append_three",
        )

        _write_status(
            runner,
            "running_matched_unique_strong_holstein_six",
            current_execution_ids=[],
        )
        terminal = runner.run_campaign()
        _write_status(
            runner,
            "passed_weak12_then_matched_unique6",
            current_execution_ids=[],
            matched_terminal_receipt_sha256=terminal["sha256"],
        )
        return terminal


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run weak-Holstein priority 12, then unique matched strong six"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        payload = preflight() if args.preflight else run()
    except (OSError, ValueError, PriorityScheduleError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
