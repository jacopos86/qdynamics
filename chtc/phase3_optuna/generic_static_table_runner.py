#!/usr/bin/env python3
"""One-record generic static Table-I runner with live heartbeat files."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.runtime_heartbeat import (  # noqa: E402
    LiveHeartbeatRecorder,
    parse_ai_log_line,
)

_DEFAULT_RECORDS_PATH = Path("chtc/phase3_optuna/input/generic_static_table_records.tsv")
_ENV_PREFIX_FIELDS = (
    "phase3_",
    "hardware_resolution_",
    "static_route_",
    "benchmark_value_noise_",
    "benchmark_decision_noise_",
)
_DIRECT_ENV_FIELDS = {
    "phase2_novelty_mode": "PHASE3_POLICY_PHASE2_NOVELTY_MODE",
    "fixed_inner_optimizer": "PHASE3_POLICY_INNER_OPTIMIZER",
    "same_cutoff_exact_gs_energy": "GENERIC_STATIC_TABLE_SAME_CUTOFF_EXACT_GS_ENERGY",
    "exact_reference_energy": "GENERIC_STATIC_TABLE_EXACT_REFERENCE_ENERGY",
    "exact_reference_n_ph_max": "GENERIC_STATIC_TABLE_EXACT_REFERENCE_N_PH_MAX",
    "primary_energy_metric": "GENERIC_STATIC_TABLE_PRIMARY_ENERGY_METRIC",
    "same_cutoff_error_role": "GENERIC_STATIC_TABLE_SAME_CUTOFF_ERROR_ROLE",
}


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_record(records_path: str | Path, record_id: str) -> dict[str, str]:
    rows = csv.DictReader(Path(records_path).read_text(encoding="utf-8").splitlines(), delimiter="\t")
    for row in rows:
        if str(row.get("record_id", "")) == str(record_id):
            return {str(k): "" if v is None else str(v) for k, v in row.items()}
    raise SystemExit(f"record_id {record_id!r} not found in {records_path}")


def env_overlay_from_record(row: Mapping[str, str]) -> dict[str, str]:
    """Return child environment values implied by a generated Table-I TSV row."""

    overlay: dict[str, str] = {}
    suite_profile = str(row.get("suite_profile", "")).strip()
    if suite_profile:
        overlay["TABLE_I_STATIC_SUITE_PROFILE"] = suite_profile
    energy_stop_target = str(row.get("energy_stop_target", "")).strip()
    if energy_stop_target:
        overlay["GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET"] = energy_stop_target
    first_hit_thresholds = str(row.get("first_hit_thresholds", "")).strip()
    if first_hit_thresholds:
        overlay["GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS"] = first_hit_thresholds
    for field_key, env_name in _DIRECT_ENV_FIELDS.items():
        value = str(row.get(field_key, "") or "").strip()
        if value:
            overlay[env_name] = value
    for field, raw_value in row.items():
        field_key = str(field)
        if not field_key.startswith(_ENV_PREFIX_FIELDS):
            continue
        value = str(raw_value or "").strip()
        if value == "":
            continue
        overlay["GENERIC_STATIC_TABLE_" + field_key.upper()] = value
    return overlay


def _clear_stale_env(env: dict[str, str], row: Mapping[str, str]) -> None:
    env.pop("GENERIC_STATIC_TABLE_STATIC_ROUTE_ID", None)
    env.pop("STATIC_ROUTE_ID", None)
    env.pop("GENERIC_STATIC_TABLE_PHASE3_POLICY_JSON", None)
    for env_name in _DIRECT_ENV_FIELDS.values():
        env.pop(env_name, None)
    for field in row:
        field_key = str(field)
        names: list[str] = []
        if field_key == "suite_profile":
            names.append("TABLE_I_STATIC_SUITE_PROFILE")
        elif field_key == "energy_stop_target":
            names.append("GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET")
        elif field_key == "first_hit_thresholds":
            names.append("GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS")
        elif field_key.startswith(_ENV_PREFIX_FIELDS):
            names.extend(["GENERIC_STATIC_TABLE_" + field_key.upper(), field_key.upper()])
        for name in names:
            env.pop(name, None)


def command_for_record(row: Mapping[str, str], out_root: str | Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "pipelines.exact_bench.generic_static_benchmark",
        "--run-single",
        "--family",
        str(row["family"]),
        "--case-id",
        str(row["case_id"]),
        "--algorithm-id",
        str(row["algorithm_id"]),
        "--output-dir",
        str(Path(out_root) / "result"),
    ]


def run_command_with_heartbeat(
    command: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None,
    stdout_path: str | Path,
    stderr_path: str | Path,
    heartbeat_path: str | Path,
    heartbeat_events_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    echo_stdout: bool = True,
) -> int:
    """Run a child process, tee stdout, and update heartbeat from ``AI_LOG`` lines."""

    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    recorder = LiveHeartbeatRecorder(
        heartbeat_path=heartbeat_path,
        event_jsonl_path=heartbeat_events_path,
        metadata=dict(metadata or {}),
    )
    started = time.perf_counter()
    proc: subprocess.Popen[str] | None = None
    returncode: int | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
        try:
            proc = subprocess.Popen(
                [str(x) for x in command],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
                env=(dict(env) if env is not None else None),
                text=True,
                bufsize=1,
            )
            recorder.mark_started(pid=int(proc.pid), command=[str(x) for x in command])
            stream = getattr(proc, "stdout", None)
            if stream is not None:
                for line in stream:
                    stdout_fh.write(line)
                    stdout_fh.flush()
                    if echo_stdout:
                        print(line, end="", flush=True)
                    payload = parse_ai_log_line(line)
                    if payload is not None:
                        recorder.update_from_ai_log(
                            payload,
                            elapsed_s=float(time.perf_counter() - started),
                            pid=int(proc.pid),
                        )
            returncode = int(proc.wait())
            return int(returncode)
        except BaseException:
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            raise
        finally:
            elapsed = float(time.perf_counter() - started)
            if returncode is None and proc is not None:
                polled = proc.poll()
                if polled is not None:
                    returncode = int(polled)
            status = "completed" if returncode == 0 else ("failed" if returncode is not None else "interrupted")
            recorder.mark_finished(status=status, returncode=returncode, elapsed_s=elapsed)


def run_record(
    *,
    record_id: str,
    records_path: str | Path,
    out_root: str | Path,
    cwd: str | Path = REPO_ROOT,
) -> int:
    row = load_record(records_path, record_id)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = command_for_record(row, out)
    env = os.environ.copy()
    _clear_stale_env(env, row)
    overlay = env_overlay_from_record(row)
    env.update(overlay)
    _json_write(out / "record.json", row)
    _json_write(out / "effective_env_overlay.json", overlay)
    (out / "command.sh").write_text(shlex.join([str(x) for x in command]) + "\n", encoding="utf-8")
    print("RUN", shlex.join([str(x) for x in command]), flush=True)
    return run_command_with_heartbeat(
        command,
        cwd=cwd,
        env=env,
        stdout_path=logs / "stdout.log",
        stderr_path=logs / "stderr.log",
        heartbeat_path=out / "heartbeat.json",
        heartbeat_events_path=out / "heartbeat_events.jsonl",
        metadata={
            "record_id": str(record_id),
            "family": row.get("family"),
            "case_id": row.get("case_id"),
            "algorithm_id": row.get("algorithm_id"),
            "suite_profile": row.get("suite_profile"),
            "runner": "generic_static_table_runner_v1",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one generic static Table-I record with live heartbeat output.")
    parser.add_argument("record_id")
    parser.add_argument("records_path", nargs="?", default=None)
    parser.add_argument("out_root", nargs="?", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records_path = Path(
        args.records_path
        or os.environ.get("GENERIC_STATIC_TABLE_RECORDS_PATH", "")
        or _DEFAULT_RECORDS_PATH
    )
    out_root = Path(args.out_root or Path("raw_outputs/generic_static_table") / str(args.record_id))
    return int(run_record(record_id=str(args.record_id), records_path=records_path, out_root=out_root))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
