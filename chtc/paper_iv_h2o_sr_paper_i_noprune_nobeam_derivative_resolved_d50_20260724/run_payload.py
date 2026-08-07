from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, gzip.open(
        destination, "wb", compresslevel=1
    ) as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)


def build_deliverables(
    *,
    root: Path,
    outdir: Path,
    record_id: str,
    command: dict[str, Any],
    checkpoint_only: bool = False,
) -> dict[str, Any]:
    deliver = root / "deliverables" / record_id
    deliver.mkdir(parents=True, exist_ok=True)
    write_json(deliver / "submit_manifest.json", command)
    copied = []
    for name in ("shell_status.json", "runtime_preflight.json", "result_validation.json"):
        source = outdir / name
        if source.is_file():
            shutil.copy2(source, deliver / name)
            copied.append(name)
    compressed = []
    compressed_names = (
        ("current.json",)
        if checkpoint_only
        else (
            "current.json",
            "result.json",
            "estimator_call_ledger.json",
            "stdout.log",
            "stderr.log",
        )
    )
    for name in compressed_names:
        source = outdir / name
        if source.is_file():
            target_name = f"{name}.gz"
            gzip_file(source, deliver / target_name)
            compressed.append(target_name)
    files = {}
    for path in sorted(deliver.iterdir()):
        if path.is_file() and path.name != "deliverables_manifest.json":
            files[path.name] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "schema": (
            "paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_"
            "deliverables_manifest_v1"
        ),
        "timestamp_utc": utc_now(),
        "record_id": record_id,
        "copied_files": copied,
        "compressed_files": compressed,
        "files": files,
        "continuation_checkpoint": (
            "current.json.gz" if "current.json.gz" in compressed else None
        ),
        "checkpoint_only": bool(checkpoint_only),
    }
    write_json(deliver / "deliverables_manifest.json", manifest)
    return manifest


def latest_progress(stdout_path: Path, record_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": (
            "paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_"
            "progress_v1"
        ),
        "timestamp_utc": utc_now(),
        "record_id": record_id,
    }
    if not stdout_path.is_file():
        return payload
    with stdout_path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - 4 * 1024 * 1024))
        tail = handle.read().decode("utf-8", errors="replace")
    for line in reversed(tail.splitlines()):
        if "hardcoded_adapt_iteration_done" not in line:
            continue
        try:
            event = json.loads(line[line.index("{") :])
        except (ValueError, json.JSONDecodeError):
            continue
        payload.update(
            {
                "depth": event.get("depth"),
                "energy": event.get("energy"),
                "abs_delta_e": event.get("abs_delta_e"),
                "selected_label": event.get("selected_label"),
            }
        )
        break
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    command_path = args.command_json.resolve()
    command = json.loads(command_path.read_text(encoding="utf-8"))
    record_id = str(command["record_id"])
    outdir = root / "raw_outputs" / record_id
    outdir.mkdir(parents=True, exist_ok=True)
    stdout_path = outdir / "stdout.log"
    stderr_path = outdir / "stderr.log"
    result_path = outdir / "result.json"
    status_path = outdir / "shell_status.json"
    write_json(outdir / "submit_manifest.json", command)

    preflight = subprocess.run(
        [
            sys.executable,
            str(args.preflight.resolve()),
            "--root",
            str(root),
            "--command-json",
            str(command_path),
            "--output-json",
            str(outdir / "runtime_preflight.json"),
        ],
        cwd=root / "runtime_source",
        check=False,
        text=True,
    )
    if preflight.returncode != 0:
        write_json(
            status_path,
            {
                "schema": (
                    "paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_"
                    "resolved_shell_status_v1"
                ),
                "timestamp_utc": utc_now(),
                "record_id": record_id,
                "state": "preflight_failed",
                "returncode": preflight.returncode,
            },
        )
        build_deliverables(
            root=root, outdir=outdir, record_id=record_id, command=command
        )
        raise SystemExit(preflight.returncode)

    (root / "runtime_cache/candidate_records").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(root / "runtime_source"),
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": str(
                root / "runtime_cache/candidate_records"
            ),
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
        print("H2O_PROGRESS " + json.dumps(latest_progress(stdout_path, record_id)), flush=True)
        next_heartbeat = time.monotonic() + heartbeat_seconds
        while child.poll() is None:
            time.sleep(5)
            if time.monotonic() >= next_heartbeat:
                print(
                    "H2O_PROGRESS "
                    + json.dumps(latest_progress(stdout_path, record_id), sort_keys=True),
                    flush=True,
                )
                build_deliverables(
                    root=root,
                    outdir=outdir,
                    record_id=record_id,
                    command=command,
                    checkpoint_only=True,
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
                "--output-json",
                str(outdir / "result_validation.json"),
            ],
            cwd=root,
            check=False,
            text=True,
        )
        validation_status = "pass" if validation.returncode == 0 else "blocked"
        if validation.returncode != 0:
            returncode = validation.returncode

    state = "interrupted" if received_signal else ("completed" if returncode == 0 else "failed")
    write_json(
        status_path,
        {
            "schema": "paper_iv_h2o_sr_derivative_resolved_shell_status_v1",
            "timestamp_utc": utc_now(),
            "record_id": record_id,
            "state": state,
            "returncode": returncode,
            "received_signal": received_signal,
            "validation_status": validation_status,
            "progress": latest_progress(stdout_path, record_id),
        },
    )
    deliverables = build_deliverables(
        root=root, outdir=outdir, record_id=record_id, command=command
    )
    print(
        "H2O_PROGRESS "
        + json.dumps(
            {
                **latest_progress(stdout_path, record_id),
                "deliverables": deliverables,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if received_signal:
        raise SystemExit(128 + received_signal)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
