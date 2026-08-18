"""RAM guard: run a command, kill its process tree before it exceeds a limit.

Implements the enforcement step of `agent_guidance/shared/memory-budget.md`
(hard 10 GB aggregate ceiling for agent work on this machine; default
single-tree limit 8000 MB).

Usage:
    python3 pipelines/shell/ram_guard.py [--limit-mb 8000] [--interval 2] -- <command...>

Exit code: the command's exit code, or 137 if killed by the guard.
Stdlib only; polls resident set size of the spawned process group.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

HARD_CAP_MB = 10000


def tree_rss_mb(root_pid: int) -> float:
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.SubprocessError:
        return 0.0
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, kb = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb
    total = 0
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mb", type=float, default=8000.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given (use: ram_guard.py [opts] -- cmd ...)")
    limit = min(float(args.limit_mb), float(HARD_CAP_MB))

    proc = subprocess.Popen(command, start_new_session=True)
    peak = 0.0
    try:
        while True:
            code = proc.poll()
            if code is not None:
                print(
                    f"[ram_guard] exit={code} peak={peak:.0f}MB "
                    f"limit={limit:.0f}MB",
                    file=sys.stderr,
                )
                return code
            usage = tree_rss_mb(proc.pid)
            peak = max(peak, usage)
            if usage > limit:
                print(
                    f"[ram_guard] KILLING process tree: {usage:.0f}MB > "
                    f"{limit:.0f}MB limit (memory-budget.md)",
                    file=sys.stderr,
                )
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait()
                return 137
            time.sleep(args.interval)
    except KeyboardInterrupt:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
