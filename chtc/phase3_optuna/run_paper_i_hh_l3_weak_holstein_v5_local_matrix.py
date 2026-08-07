#!/usr/bin/env python3
"""Exact local rerun launcher for the L=3 weak-Holstein SNAKE v5 matrix.

This is intentionally concrete: it pins the v5 record file and runs the three
SNAKE regimes two at a time with the settings in that TSV. The per-cell runner
writes resolved_command.json before each scientific run, so the exact executed
CLI and environment overlay are preserved beside each output.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BATCH_ID = "paper_i_hh_l3_weak_holstein_schur_lift_sectorfix_20260626_v5_local"
RECORDS_PATH = (
    Path("chtc/phase3_optuna/input")
    / BATCH_ID
    / "paper_i_hh_l3_weak_holstein_schur_lift_sectorfix_records.tsv"
)
OUTPUT_ROOT = Path("raw_outputs") / BATCH_ID
CELL_RUNNER = Path("chtc/phase3_optuna/run_paper_i_hh_l3_weak_holstein_cell.py")
MAX_PARALLEL = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rows() -> list[dict[str, str]]:
    with RECORDS_PATH.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_state(
    *,
    active: list[tuple[str, subprocess.Popen[bytes], Path]],
    done: list[dict[str, Any]],
    pending: list[dict[str, str]],
) -> None:
    write_json(
        OUTPUT_ROOT / "supervisor_state.json",
        {
            "schema": "paper_i_hh_l3_local_supervisor_v1",
            "batch_id": BATCH_ID,
            "records_path": str(RECORDS_PATH),
            "output_root": str(OUTPUT_ROOT),
            "cell_runner": str(CELL_RUNNER),
            "max_parallel": int(MAX_PARALLEL),
            "updated_utc": utc_now(),
            "active": [
                {"record_id": record_id, "pid": int(proc.pid), "output_root": str(output_root)}
                for record_id, proc, output_root in active
            ],
            "pending": [str(row["record_id"]) for row in pending],
            "done": list(done),
        },
    )


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    pending = list(rows)
    active: list[tuple[str, subprocess.Popen[bytes], Path]] = []
    done: list[dict[str, Any]] = []
    log_path = OUTPUT_ROOT / "supervisor.log"

    with log_path.open("a", encoding="utf-8") as log:
        print(utc_now(), "START", BATCH_ID, "rows", len(rows), flush=True, file=log)
        write_state(active=active, done=done, pending=pending)
        while pending or active:
            while pending and len(active) < MAX_PARALLEL:
                row = pending.pop(0)
                record_id = str(row["record_id"])
                cell_output_root = OUTPUT_ROOT / record_id
                cell_output_root.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(CELL_RUNNER),
                    record_id,
                    str(RECORDS_PATH),
                    str(cell_output_root),
                ]
                print(utc_now(), "LAUNCH", record_id, "out", cell_output_root, flush=True, file=log)
                proc = subprocess.Popen(cmd, cwd=".")
                active.append((record_id, proc, cell_output_root))
                write_state(active=active, done=done, pending=pending)

            time.sleep(30)
            still_active: list[tuple[str, subprocess.Popen[bytes], Path]] = []
            for record_id, proc, cell_output_root in active:
                returncode = proc.poll()
                if returncode is None:
                    still_active.append((record_id, proc, cell_output_root))
                    continue
                print(utc_now(), "DONE", record_id, "rc", returncode, flush=True, file=log)
                done.append(
                    {
                        "record_id": record_id,
                        "returncode": int(returncode),
                        "output_root": str(cell_output_root),
                        "finished_utc": utc_now(),
                    }
                )
            active = still_active
            write_state(active=active, done=done, pending=pending)

        print(utc_now(), "COMPLETE", BATCH_ID, flush=True, file=log)

    return 0 if all(int(row["returncode"]) == 0 for row in done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
