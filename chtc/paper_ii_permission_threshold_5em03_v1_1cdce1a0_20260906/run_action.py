#!/usr/bin/env python3
"""Run one immutable Paper-II corrected-exchange action.

Same completion contract as ``paper_ii_final_estimator_telemetry_v1``: a
trajectory, an exact logical estimator-work receipt, and a fresh terminal
FakeMarrakesh cost.  The difference is the search: these eleven cells run the
single q-compiled route under ``paper_declared_exchange_v1``, whose declared
insertion frontier is what makes the exchange face reachable at all.  The
AVQDS comparators are not re-run here; they are the ones already produced by
``paper_ii_final_estimator_telemetry_v1`` and are unaffected by the defect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, Mapping

from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
    complete_estimator_work,
)


CAMPAIGN_ID = "paper_ii_permission_threshold_5em03_v1"
EXPECTED_ACTION_COUNT = 1
EXPECTED_CODE_REVISION = "1cdce1a0cc38cf8f9a60cca86fe592358ce36d3a"
EXPECTED_SOURCE_SHA256 = (
    "5e79c267b85190c99a377b973f5b9f30e0971b7ca6d53a8a57c9d402024dc702"
)
EXPECTED_TELEMETRY_CONTRACT = "paper_ii_exact_logical_estimator_telemetry_20260903"
EXPECTED_ROUTE_BUNDLE = "paper_ii_exchange_q_compiled_v1"
EXPECTED_INSERTION_FRONTIER = "8"
PACKAGE_DIR = Path(__file__).resolve().parent
ACTIONS_JSONL = PACKAGE_DIR / "input/actions.jsonl"
ROUTE_BY_ARM = {"exchange_q_compiled": "q_compiled"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def archive_action(action_id: str, action_dir: Path) -> Path:
    archive_dir = Path("archives")
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{action_id}.tar.gz"
    if not action_dir.exists():
        write_json(
            action_dir / "missing_action_directory.json",
            {
                "schema": "paper_ii_exact_estimator_missing_action_directory_v1",
                "action_id": action_id,
            },
        )
    with tarfile.open(target, "w:gz", compresslevel=6) as handle:
        handle.add(action_dir, arcname=str(action_dir))
    write_json(
        archive_dir / f"{action_id}.archive.json",
        {
            "schema": "paper_ii_final_estimator_telemetry_archive_receipt_v1",
            "action_id": action_id,
            "archive": str(target),
            "archive_sha256": sha256(target),
            "source_package_sha256": EXPECTED_SOURCE_SHA256,
            "code_revision": EXPECTED_CODE_REVISION,
        },
    )
    return target


def load_action(action_id: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in ACTIONS_JSONL.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != EXPECTED_ACTION_COUNT:
        raise RuntimeError(f"expected {EXPECTED_ACTION_COUNT} actions; found {len(rows)}")
    if len({str(row["action_id"]) for row in rows}) != EXPECTED_ACTION_COUNT:
        raise RuntimeError("action ids are not unique")
    record = next((row for row in rows if row["action_id"] == action_id), None)
    if record is None:
        raise RuntimeError(f"unknown action_id {action_id}")
    return dict(record)


def validate_action(record: Mapping[str, Any]) -> tuple[str, Path]:
    action_id = str(record["action_id"])
    if record.get("method") != "apm":
        raise RuntimeError("this campaign runs APM actions only")
    arm = str(record.get("arm") or "")
    if arm not in ROUTE_BY_ARM:
        raise RuntimeError(f"unknown arm {arm!r}")
    if record.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("action campaign drift")
    if record.get("code_revision") != EXPECTED_CODE_REVISION:
        raise RuntimeError("action code revision drift")
    if record.get("source_package_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("action source-package digest drift")
    if record.get("estimator_telemetry_contract") != EXPECTED_TELEMETRY_CONTRACT:
        raise RuntimeError("estimator telemetry contract drift")
    if record.get("precursor") is not False:
        raise RuntimeError("precursor action is forbidden")
    if record.get("exchange_route_bundle") != EXPECTED_ROUTE_BUNDLE:
        raise RuntimeError("selector route drift")
    runner_argv = list(record.get("runner_argv") or ())
    if not runner_argv or "--resume-from-run-json" in runner_argv:
        raise RuntimeError("runner argv is empty or resumes a precursor")
    pairs = set(zip(runner_argv, runner_argv[1:]))
    if ("--exchange-route-bundle", EXPECTED_ROUTE_BUNDLE) not in pairs:
        raise RuntimeError("runner argv does not carry the compiled-Q route bundle")
    if ("--exchange-insertion-frontier", EXPECTED_INSERTION_FRONTIER) not in pairs:
        raise RuntimeError("runner argv does not carry the declared exchange frontier")
    trajectory = Path(str(record["trajectory_json"]))
    expected_parent = Path("raw_outputs") / str(record["campaign_id"]) / action_id
    if trajectory != expected_parent / "run.json":
        raise RuntimeError(f"unexpected trajectory path: {trajectory}")
    return ROUTE_BY_ARM[arm], trajectory


def exact_estimator_receipt(
    *,
    action_id: str,
    route: str,
    trajectory: Path,
) -> tuple[dict[str, Any], bool]:
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    work = complete_estimator_work([payload], route=route)
    exact = bool(
        work.get("logical_total_estimator_work_status") == "exact"
        and work.get("candidate_acquisition_status") == "exact"
        and work.get("certification_geometry_status") in {"exact", "not_applicable"}
        and work.get("refit_estimator_status") in {"exact", "not_applicable"}
        and work.get("selection_round_ledger_complete") is True
    )
    receipt = {
        "schema": "paper_ii_final_estimator_telemetry_work_receipt_v1",
        "action_id": action_id,
        "route": route,
        "status": "EXACT" if exact else "INCOMPLETE",
        "trajectory_json": str(trajectory),
        "trajectory_sha256": sha256(trajectory),
        "telemetry_contract": EXPECTED_TELEMETRY_CONTRACT,
        "estimator_work": work,
        "physical_shots": None,
        "physical_shots_status": "unavailable_statevector_no_shot_ledger",
    }
    return receipt, exact


def terminal_cost(action_id: str, trajectory: Path) -> tuple[dict[str, Any], int]:
    cost_json = trajectory.with_name("terminal_qiskit_cost.json")
    command = [
        sys.executable,
        "-m",
        "pipelines.time_dynamics.diagnostics.ap_terminal_qiskit_cost",
        "--trajectory-json",
        str(trajectory),
        "--output-cost-table-json",
        str(cost_json),
        "--label",
        action_id,
        "--backend-name",
        "FakeMarrakesh",
        "--seed-transpiler",
        "7",
        "--optimization-level",
        "1",
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not cost_json.is_file():
        return (
            {
                "terminal_cost_json": None,
                "terminal_cost_sha256": None,
                "terminal_cost_status": "FAILED",
                "terminal_cost_command": command,
                "terminal_cost_returncode": int(completed.returncode),
            },
            6,
        )
    return (
        {
            "terminal_cost_json": str(cost_json),
            "terminal_cost_sha256": sha256(cost_json),
            "terminal_cost_status": "SUCCESS",
        },
        0,
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: run_action.py ACTION_ID")
    action_id = args[0]
    action_dir = Path("raw_outputs") / CAMPAIGN_ID / action_id
    failure: dict[str, Any] | None = None
    return_code = 5
    try:
        record = load_action(action_id)
        route, trajectory = validate_action(record)
        action_dir = trajectory.parent
        action_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            str(record["runner_module"]),
            *list(record["runner_argv"]),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failure = {
                "stage": "trajectory",
                "returncode": int(completed.returncode),
                "command": command,
            }
            return_code = int(completed.returncode)
        elif not trajectory.is_file() or trajectory.stat().st_size == 0:
            failure = {
                "stage": "trajectory_validation",
                "reason": "missing_or_empty_run_json",
            }
            return_code = 3
        else:
            work_receipt, work_exact = exact_estimator_receipt(
                action_id=action_id,
                route=route,
                trajectory=trajectory,
            )
            work_path = action_dir / "exact_estimator_work.json"
            write_json(work_path, work_receipt)
            cost_receipt, cost_code = terminal_cost(action_id, trajectory)
            if not work_exact:
                failure = {
                    "stage": "estimator_telemetry_validation",
                    "reason": "logical_estimator_work_is_not_exact",
                    "work_receipt": str(work_path),
                    "work_receipt_sha256": sha256(work_path),
                }
                return_code = 4
            elif cost_code != 0:
                failure = {"stage": "terminal_cost", **cost_receipt}
                return_code = cost_code
            else:
                write_json(
                    action_dir / "completion_receipt.json",
                    {
                        "schema": "paper_ii_final_estimator_telemetry_completion_v1",
                        "status": "SUCCESS_EXACT",
                        "action_id": action_id,
                        "campaign_id": record["campaign_id"],
                        "cell_id": record["cell_id"],
                        "method": record["method"],
                        "arm": record["arm"],
                        "route": route,
                        "exchange_route_bundle": record.get("exchange_route_bundle"),
                        "code_revision": EXPECTED_CODE_REVISION,
                        "source_package_sha256": EXPECTED_SOURCE_SHA256,
                        "matched_cell_lock_sha256": record["matched_cell_lock_sha256"],
                        "telemetry_contract": EXPECTED_TELEMETRY_CONTRACT,
                        "comparator": record.get("comparator"),
                        "trajectory_json": str(trajectory),
                        "trajectory_sha256": sha256(trajectory),
                        "estimator_work_json": str(work_path),
                        "estimator_work_sha256": sha256(work_path),
                        **cost_receipt,
                    },
                )
                return_code = 0
    except Exception as exc:
        failure = {
            "stage": "wrapper_exception",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return_code = 5
    finally:
        if failure is not None:
            write_json(
                action_dir / "failure_receipt.json",
                {
                    "schema": "paper_ii_final_estimator_telemetry_failure_v1",
                    "action_id": action_id,
                    "code_revision": EXPECTED_CODE_REVISION,
                    "source_package_sha256": EXPECTED_SOURCE_SHA256,
                    "failure": failure,
                },
            )
        archive_action(action_id, action_dir)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
