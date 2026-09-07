#!/usr/bin/env python3
"""Run one provenance-bound Paper-II selector retry and archive its evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any


EXPECTED_ACTION_COUNT = 1
EXPECTED_CODE_REVISION = "4bca80b01f3d4e20e74f21b21ec36d23374b3392"
EXPECTED_SOURCE_SHA256 = (
    "8f12755668b793062650801eab5755b81ff64c20e851f15e9b0ec18813e047d3"
)
EXPECTED_PARENT_CLUSTER_ID = 9718529
EXPECTED_RETRY_WAVE = 4
PACKAGE_DIR = Path(__file__).resolve().parent
ACTIONS_JSONL = PACKAGE_DIR / "input/actions.jsonl"
RETRY_PARENT_FAILURES = PACKAGE_DIR / "input/retry_parent_failures.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def retry_provenance(record: dict[str, Any], parent_failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "retry_wave": EXPECTED_RETRY_WAVE,
        "parent_cluster_id": EXPECTED_PARENT_CLUSTER_ID,
        "parent_proc_id": int(record["retry_parent_proc_id"]),
        "parent_stderr_sha256": str(parent_failure["stderr_sha256"]),
        "parent_final_error": str(parent_failure["final_error"]),
        "failure_class": "GammaMonotonicityError",
        "repair_policy": (
            "reject only the invalid Gamma candidate from the shared compiled "
            "Q/Gamma population"
        ),
    }


def archive_action(
    action_id: str,
    action_dir: Path,
    provenance: dict[str, Any],
) -> Path:
    archive_dir = Path("archives")
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{action_id}.tar.gz"
    if action_dir.exists():
        with tarfile.open(target, "w:gz", compresslevel=6) as handle:
            handle.add(action_dir, arcname=str(action_dir))
    else:
        placeholder = action_dir / "missing_action_directory.json"
        write_json(
            placeholder,
            {"schema": "paper_ii_selector_missing_action_directory_v1", "action_id": action_id},
        )
        with tarfile.open(target, "w:gz", compresslevel=6) as handle:
            handle.add(action_dir, arcname=str(action_dir))
    write_json(
        archive_dir / f"{action_id}.archive.json",
        {
            "schema": "paper_ii_final_selector_gamma_candidate_retry_archive_v1",
            "action_id": action_id,
            "archive": str(target),
            "archive_sha256": sha256(target),
            "source_package_sha256": EXPECTED_SOURCE_SHA256,
            "code_revision": EXPECTED_CODE_REVISION,
            "retry_provenance": provenance,
        },
    )
    return target


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: run_action.py ACTION_ID")
    action_id = args[0]
    rows = [json.loads(line) for line in ACTIONS_JSONL.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != EXPECTED_ACTION_COUNT:
        raise SystemExit(f"expected {EXPECTED_ACTION_COUNT} actions; found {len(rows)}")
    if len({str(row["action_id"]) for row in rows}) != EXPECTED_ACTION_COUNT:
        raise SystemExit("action ids are not unique")
    parent = json.loads(RETRY_PARENT_FAILURES.read_text(encoding="utf-8"))
    if parent.get("schema") != "paper_ii_final_selector_retry_parent_failures_v1":
        raise SystemExit("retry-parent schema drift")
    if int(parent.get("retry_wave", -1)) != EXPECTED_RETRY_WAVE:
        raise SystemExit("retry wave drift")
    if int(parent.get("parent", {}).get("cluster_id", -1)) != EXPECTED_PARENT_CLUSTER_ID:
        raise SystemExit("parent cluster drift")
    parent_failures = {
        str(item["action_id"]): item for item in parent.get("failures", ())
    }
    if len(parent_failures) != EXPECTED_ACTION_COUNT:
        raise SystemExit("retry-parent failure count drift")
    record = next((row for row in rows if row["action_id"] == action_id), None)
    if record is None:
        raise SystemExit(f"unknown action_id {action_id}")
    if record.get("method") != "apm" or record.get("comparator", {}).get("action") is not None:
        raise SystemExit("package contains a forbidden comparator action")
    if record.get("code_revision") != EXPECTED_CODE_REVISION:
        raise SystemExit("action code revision drift")
    if record.get("source_package_sha256") != EXPECTED_SOURCE_SHA256:
        raise SystemExit("action source-package digest drift")
    parent_failure = parent_failures.get(action_id)
    if parent_failure is None:
        raise SystemExit("action was not a terminal parent failure")
    if int(record.get("retry_parent_cluster_id", -1)) != EXPECTED_PARENT_CLUSTER_ID:
        raise SystemExit("action parent-cluster binding drift")
    if int(record.get("retry_parent_proc_id", -1)) != int(parent_failure["proc_id"]):
        raise SystemExit("action parent-proc binding drift")
    if str(record.get("retry_parent_stderr_sha256")) != str(parent_failure["stderr_sha256"]):
        raise SystemExit("action parent-stderr binding drift")
    if (
        int(parent_failure.get("job_status", -1)),
        int(parent_failure.get("exit_code", -1)),
        bool(parent_failure.get("exit_by_signal")),
    ) != (4, 2, False):
        raise SystemExit("parent action was not a verified terminal exit-2 failure")
    provenance = retry_provenance(record, parent_failure)
    runner_argv = list(record.get("runner_argv") or ())
    if not runner_argv:
        raise SystemExit("empty runner argv")
    trajectory = Path(str(record["trajectory_json"]))
    expected_parent = Path("raw_outputs") / str(record["campaign_id"]) / action_id
    if trajectory != expected_parent / "run.json":
        raise SystemExit(f"unexpected trajectory path: {trajectory}")
    expected_parent.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-m", str(record["runner_module"]), *runner_argv]
    failure: dict[str, Any] | None = None
    try:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failure = {
                "stage": "trajectory",
                "returncode": int(completed.returncode),
                "command": command,
            }
            return_code = int(completed.returncode)
        elif not trajectory.is_file() or trajectory.stat().st_size == 0:
            failure = {"stage": "trajectory_validation", "reason": "missing_or_empty_run_json"}
            return_code = 3
        else:
            cost_json = trajectory.with_name("terminal_qiskit_cost.json")
            cost_command = [
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
            cost_completed = subprocess.run(cost_command, check=False)
            if cost_completed.returncode != 0 or not cost_json.is_file():
                cost_failure = {
                    "schema": "paper_ii_optional_terminal_cost_failure_v1",
                    "action_id": action_id,
                    "returncode": int(cost_completed.returncode),
                    "command": cost_command,
                    "trajectory_json": str(trajectory),
                }
                write_json(expected_parent / "terminal_qiskit_cost_failure.json", cost_failure)
                cost_receipt = {
                    "terminal_cost_json": None,
                    "terminal_cost_sha256": None,
                    "status": "TRAJECTORY_SUCCESS_COST_FAILED",
                }
            else:
                cost_receipt = {
                    "terminal_cost_json": str(cost_json),
                    "terminal_cost_sha256": sha256(cost_json),
                    "status": "SUCCESS",
                }
            write_json(
                expected_parent / "completion_receipt.json",
                {
                    "schema": "paper_ii_final_selector_gamma_candidate_retry_completion_v1",
                    "action_id": action_id,
                    "campaign_id": record["campaign_id"],
                    "exchange_route_bundle": record["exchange_route_bundle"],
                    "exchange_route_contract_sha256": record[
                        "exchange_route_contract_sha256"
                    ],
                    "code_revision": EXPECTED_CODE_REVISION,
                    "source_package_sha256": EXPECTED_SOURCE_SHA256,
                    "matched_cell_lock_sha256": record["matched_cell_lock_sha256"],
                    "trajectory_json": str(trajectory),
                    "trajectory_sha256": sha256(trajectory),
                    "retry_provenance": provenance,
                    **cost_receipt,
                },
            )
            return_code = 0
    except Exception as exc:  # preserve a transferable receipt for unexpected failures
        failure = {"stage": "wrapper_exception", "type": type(exc).__name__, "message": str(exc)}
        return_code = 5
    finally:
        if failure is not None:
            write_json(
                expected_parent / "failure_receipt.json",
                {
                    "schema": "paper_ii_final_selector_gamma_candidate_retry_failure_v1",
                    "action_id": action_id,
                    "code_revision": EXPECTED_CODE_REVISION,
                    "source_package_sha256": EXPECTED_SOURCE_SHA256,
                    "retry_provenance": provenance,
                    "failure": failure,
                },
            )
        archive_action(action_id, expected_parent, provenance)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
