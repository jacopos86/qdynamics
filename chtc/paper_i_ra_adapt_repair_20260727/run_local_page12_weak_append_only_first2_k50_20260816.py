#!/usr/bin/env python3
"""Run only weak--weak and intermediate--weak RA append-only to k=50.

This is a narrow priority supervisor over the already authorized six-cell
weak-Holstein activation.  It publishes ordinary six-cell runtime closures
for the selected cells, then exits without starting strong--weak or either
always-insertion route.  A later full campaign therefore skips these cells.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
BASE_RUNNER_PATH = SCRIPT_PATH.with_name(
    "run_local_page12_weak_holstein_priority6_20260815.py"
)
SUBSET_RECEIPT_NAME = "weak_weak_intermediate_weak_append_only_k50.json"


class PrioritySubsetError(RuntimeError):
    """Raised when the exact two-cell priority subset cannot run safely."""


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "paper_i_weak_append_only_first2_base", BASE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise PrioritySubsetError(f"Unable to import {BASE_RUNNER_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target_ids(runner: Any) -> tuple[str, str]:
    selected = tuple(runner.TARGET_CELLS[:2])
    if (
        len(selected) != 2
        or tuple(cell.regime_id for cell in selected)
        != ("weak_weak", "intermediate_weak")
        or any(cell.policy != "append_only" for cell in selected)
        or runner.TARGET_HORIZON != 50
    ):
        raise PrioritySubsetError("Base authority no longer starts with the exact k50 subset.")
    return tuple(cell.execution_id for cell in selected)


def preflight() -> dict[str, Any]:
    runner = _load_runner()
    targets = _target_ids(runner)
    base = runner.inert_preflight(
        planning_dir=runner.DEFAULT_PLANNING_DIR,
        activation_dir=runner.DEFAULT_ACTIVATION_DIR,
        runtime_dir=runner.DEFAULT_RUNTIME_DIR,
    )
    if base.get("run_ready") is not True:
        raise PrioritySubsetError(
            "Base weak-sector activation is not currently launch-ready: "
            + json.dumps(base, sort_keys=True, separators=(",", ":"))
        )
    return runner._digested(
        {
            "schema": "paper_i_page12_weak_append_only_first2_k50_preflight_v1",
            "status": "passed_inert_exact_two_cells",
            "execution_ids": list(targets),
            "target_horizon": 50,
            "maximum_concurrency": 1,
            "base_preflight_sha256": base["sha256"],
            "scientific_execution_performed": False,
            "submission_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _subset_receipt(
    runner: Any,
    *,
    targets: tuple[str, str],
    activation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    completed_at_utc: str,
) -> dict[str, Any]:
    return runner._digested(
        {
            "schema": "paper_i_page12_weak_append_only_first2_k50_receipt_v1",
            "status": "passed_exact_two_cells_pending_later_campaign",
            "completed_at_utc": completed_at_utc,
            "execution_ids": list(targets),
            "target_horizon": 50,
            "activation_manifest_sha256": activation["sha256"],
            "runtime_manifest_sha256": runtime["sha256"],
            "cells": [
                runner._terminal_cell_binding(runner.DEFAULT_RUNTIME_DIR, execution_id)
                for execution_id in targets
            ],
            "maximum_concurrency": 1,
            "remaining_six_cell_campaign_authorized": True,
            "remaining_cells_executed_by_this_supervisor": False,
            "submission_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def run() -> dict[str, Any]:
    runner = _load_runner()
    targets = _target_ids(runner)
    worker = runner._load_worker()
    manifest, _rows = runner._closed_inputs(worker)
    activation, plan, authorization, _prior = runner._validate_activation(
        runner.DEFAULT_ACTIVATION_DIR,
        manifest=manifest,
        require_closed_rows=False,
    )
    runtime = runner._ensure_runtime(
        runner.DEFAULT_RUNTIME_DIR, activation=activation
    )
    receipt_path = (
        runner.DEFAULT_RUNTIME_DIR / "priority_subset_receipts" / SUBSET_RECEIPT_NAME
    )
    lock_path = runner.DEFAULT_RUNTIME_DIR / "campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PrioritySubsetError("Another weak-sector supervisor owns the lock.") from exc

        if receipt_path.exists() or receipt_path.is_symlink():
            observed = runner._load_digested(
                receipt_path, label="two-cell priority receipt"
            )
            expected = _subset_receipt(
                runner,
                targets=targets,
                activation=activation,
                runtime=runtime,
                completed_at_utc=str(observed.get("completed_at_utc")),
            )
            if observed != expected:
                raise PrioritySubsetError("Existing two-cell receipt drifted.")
            return observed

        fresh = runner._validate_activation(
            runner.DEFAULT_ACTIVATION_DIR,
            manifest=manifest,
            require_closed_rows=True,
        )
        if fresh[:3] != (activation, plan, authorization):
            raise PrioritySubsetError("Activation drifted before subset launch.")
        overlap = runner._scientific_overlap()
        if overlap:
            raise PrioritySubsetError(
                "Another local scientific worker is active: " + " | ".join(overlap)
            )

        completed = [
            execution_id
            for execution_id in runner.TARGET_EXECUTION_IDS
            if runner._closed_cell(runner.DEFAULT_RUNTIME_DIR, execution_id)
        ]
        for execution_id in targets:
            if execution_id in completed:
                continue
            attempt = runner.DEFAULT_RUNTIME_DIR / "in_progress" / execution_id
            if attempt.exists() or attempt.is_symlink():
                raise PrioritySubsetError(
                    f"Preserved attempt requires inspection: {execution_id}"
                )
            try:
                runner._guarded_cell(
                    execution_id=execution_id,
                    activation_dir=runner.DEFAULT_ACTIVATION_DIR,
                    runtime_dir=runner.DEFAULT_RUNTIME_DIR,
                    runtime=runtime,
                    completed=completed,
                )
            except BaseException as exc:
                runner._write_json_atomic(
                    runner.DEFAULT_RUNTIME_DIR / "status/campaign.json",
                    runner._status_payload(
                        runtime=runtime,
                        status="priority_subset_failed_or_guard_stopped",
                        completed=completed,
                        current_execution_id=execution_id,
                        failure={
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    ),
                )
                raise
            completed.append(execution_id)
            runner._write_json_atomic(
                runner.DEFAULT_RUNTIME_DIR / "status/campaign.json",
                runner._status_payload(
                    runtime=runtime,
                    status="priority_subset_cell_passed_pending_second",
                    completed=completed,
                    current_execution_id=None,
                ),
            )

        receipt = _subset_receipt(
            runner,
            targets=targets,
            activation=activation,
            runtime=runtime,
            completed_at_utc=runner._utc_now(),
        )
        runner._write_json_atomic_noreplace(receipt_path, receipt)
        runner._write_json_atomic(
            runner.DEFAULT_RUNTIME_DIR / "status/campaign.json",
            runner._status_payload(
                runtime=runtime,
                status="priority_subset_passed_two_append_only_k50",
                completed=completed,
                current_execution_id=None,
            ),
        )
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run exactly two weak-Holstein RA append-only k50 cells"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        payload = preflight() if args.preflight else run()
    except (OSError, ValueError, PrioritySubsetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
