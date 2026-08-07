#!/usr/bin/env python3
"""Run one command while recording and enforcing local memory safety."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def _process_table() -> dict[int, tuple[int, int]]:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,rss="),
        check=True,
        capture_output=True,
        text=True,
    )
    table: dict[int, tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            table[int(fields[0])] = (int(fields[1]), int(fields[2]))
    return table


def _tree_rss_kib(root_pid: int) -> int:
    table = _process_table()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in table.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(table.get(pid, (0, 0))[1] for pid in descendants)


def _free_percentage() -> int:
    completed = subprocess.run(
        ("memory_pressure", "-Q"),
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"free percentage:\s*(\d+)%", completed.stdout)
    if match is None:
        raise RuntimeError("could not parse memory_pressure output")
    return int(match.group(1))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(
    command: list[str],
    *,
    log_path: Path,
    summary_path: Path,
    poll_seconds: float,
    report_seconds: float,
    maximum_rss_gib: float,
    minimum_free_percentage: int,
) -> int:
    """Run ``command`` and stop only for an operational memory hazard."""

    if not command:
        raise ValueError("a monitored command is required")
    if poll_seconds <= 0.0 or report_seconds <= 0.0:
        raise ValueError("poll and report intervals must be positive")
    if maximum_rss_gib <= 0.0:
        raise ValueError("maximum_rss_gib must be positive")
    if not 0 < minimum_free_percentage < 100:
        raise ValueError("minimum_free_percentage must lie between 0 and 100")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    peak_rss_kib = 0
    minimum_observed_free = 100
    safety_reason: str | None = None
    last_report = -report_seconds

    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("elapsed_seconds", "process_tree_rss_gib", "system_free_percent")
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            rss_kib = _tree_rss_kib(process.pid)
            free_percentage = _free_percentage()
            peak_rss_kib = max(peak_rss_kib, rss_kib)
            minimum_observed_free = min(
                minimum_observed_free,
                free_percentage,
            )
            rss_gib = rss_kib / 1024.0**2
            writer.writerow((f"{elapsed:.3f}", f"{rss_gib:.6f}", free_percentage))
            handle.flush()
            if elapsed - last_report >= report_seconds:
                print(
                    "[memory-monitor] "
                    f"elapsed={elapsed:.1f}s rss={rss_gib:.3f}GiB "
                    f"system_free={free_percentage}%",
                    flush=True,
                )
                last_report = elapsed
            if rss_gib > maximum_rss_gib:
                safety_reason = (
                    f"process-tree RSS exceeded {maximum_rss_gib:.3f} GiB"
                )
            elif free_percentage < minimum_free_percentage:
                safety_reason = (
                    "system free memory fell below "
                    f"{minimum_free_percentage}%"
                )
            if safety_reason is not None:
                print(f"[memory-monitor] safety stop: {safety_reason}", flush=True)
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10.0)
                break
            time.sleep(poll_seconds)

    return_code = int(process.wait())
    elapsed = time.monotonic() - started
    status = "memory_safety_stop" if safety_reason is not None else "complete"
    _write_json(
        summary_path,
        {
            "command": command,
            "elapsed_seconds": elapsed,
            "maximum_rss_gib": maximum_rss_gib,
            "minimum_free_percentage": minimum_free_percentage,
            "minimum_observed_system_free_percentage": minimum_observed_free,
            "peak_process_tree_rss_gib": peak_rss_kib / 1024.0**2,
            "return_code": return_code,
            "safety_reason": safety_reason,
            "status": status,
        },
    )
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--report-seconds", type=float, default=10.0)
    parser.add_argument("--maximum-rss-gib", type=float, default=4.0)
    parser.add_argument("--minimum-free-percentage", type=int, default=15)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    return run(
        command,
        log_path=arguments.log,
        summary_path=arguments.summary,
        poll_seconds=arguments.poll_seconds,
        report_seconds=arguments.report_seconds,
        maximum_rss_gib=arguments.maximum_rss_gib,
        minimum_free_percentage=arguments.minimum_free_percentage,
    )


if __name__ == "__main__":
    raise SystemExit(main())
