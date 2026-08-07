from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def progress(current_path: Path, record_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "paper_iv_h2o_chtc_progress_v1",
        "timestamp_utc": utc_now(),
        "record_id": record_id,
        "current_json_exists": current_path.is_file(),
    }
    if current_path.is_file():
        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
            adapt = current.get("adapt_vqe", {})
            payload.update(
                {
                    "ansatz_depth": adapt.get("ansatz_depth"),
                    "energy": adapt.get("energy"),
                    "abs_delta_e": adapt.get("abs_delta_e"),
                    "stop_reason": adapt.get("stop_reason"),
                }
            )
        except Exception as exc:
            payload["read_error"] = str(exc)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    command_path = args.command_json.resolve()
    command = json.loads(command_path.read_text(encoding="utf-8"))
    record_id = command["record_id"]
    outdir = root / "raw_outputs" / record_id
    logdir = root / "logs" / record_id
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    current_path = outdir / "current.json"
    result_path = outdir / "result.json"
    stdout_path = outdir / "stdout.log"
    stderr_path = outdir / "stderr.log"
    status_path = outdir / "shell_status.json"
    audit_path = outdir / "source_locked_continuation_audit.json"
    checkpoint_path = root / "runtime_inputs" / "h2o_depth12_current.json"
    write_json(outdir / "submit_manifest.json", command)

    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(root / "runtime_source"),
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "memory",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    heartbeat_seconds = max(60, int(os.environ.get("H2O_HEARTBEAT_SEC", "300")))
    received_signal: int | None = None
    child: subprocess.Popen[str] | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        child = subprocess.Popen(
            command["argv"],
            cwd=root / "runtime_source",
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        print(
            "H2O_PROGRESS " + json.dumps(progress(current_path, record_id), sort_keys=True),
            flush=True,
        )
        next_heartbeat = time.monotonic() + heartbeat_seconds
        while child.poll() is None:
            time.sleep(5)
            if time.monotonic() >= next_heartbeat:
                print(
                    "H2O_PROGRESS "
                    + json.dumps(progress(current_path, record_id), sort_keys=True),
                    flush=True,
                )
                next_heartbeat = time.monotonic() + heartbeat_seconds
        returncode = int(child.returncode or 0)

    validation_status = "not_run"
    if returncode == 0 and result_path.is_file():
        validation = subprocess.run(
            [
                sys.executable,
                str(args.validator.resolve()),
                "--command-json",
                str(command_path),
                "--result-json",
                str(result_path),
                "--checkpoint-json",
                str(checkpoint_path),
                "--audit-json",
                str(audit_path),
            ],
            cwd=root,
            check=False,
            text=True,
        )
        validation_status = "pass" if validation.returncode == 0 else "blocked"
        if validation.returncode != 0:
            returncode = validation.returncode

    write_json(
        status_path,
        {
            "schema": "paper_iv_h2o_chtc_shell_status_v1",
            "timestamp_utc": utc_now(),
            "record_id": record_id,
            "state": "interrupted" if received_signal else ("completed" if returncode == 0 else "failed"),
            "returncode": returncode,
            "received_signal": received_signal,
            "validation_status": validation_status,
            "progress": progress(current_path, record_id),
        },
    )
    print("H2O_PROGRESS " + json.dumps(progress(current_path, record_id), sort_keys=True), flush=True)
    if received_signal:
        raise SystemExit(128 + received_signal)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
