"""Execute one local selector diagnostic with bounded, owned-process cleanup."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, BinaryIO, Mapping, Sequence

import psutil

from pipelines.time_dynamics.diagnostics.ap_runtime_benchmark import (
    BLAS_THREAD_ENV_VARS,
    BenchmarkCase,
)


_STOP_GRACE_SECONDS = 3.0
_POLL_SECONDS = 0.05


def _numerical_child(guard_pid: int) -> tuple[int, int] | None:
    """Only ram_guard's direct child that owns a new session is numerical work."""

    try:
        for child in psutil.Process(guard_pid).children(recursive=False):
            try:
                pgid = os.getpgid(child.pid)
                if pgid == child.pid and os.getsid(child.pid) == child.pid:
                    return child.pid, pgid
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue
    except (ProcessLookupError, psutil.NoSuchProcess):
        pass
    return None


def _stop_owned(guard: subprocess.Popen, child_pgid: int | None) -> dict[str, Any]:
    cleanup: dict[str, Any] = {
        "guard_sigint_sent": False, "group_sigkill_sent": False,
        "guard_sigkill_sent": False,
    }
    if guard.poll() is None:
        try:
            guard.send_signal(signal.SIGINT)
            cleanup["guard_sigint_sent"] = True
        except ProcessLookupError:
            pass
        try:
            guard.wait(timeout=_STOP_GRACE_SECONDS)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            pass
    # The group can outlive both its leader and the guard. Never substitute
    # the caller's or guard's group when discovery found no numerical child.
    if child_pgid is not None and child_pgid not in {os.getpgrp(), guard.pid}:
        try:
            os.killpg(child_pgid, signal.SIGKILL)
            cleanup["group_sigkill_sent"] = True
        except ProcessLookupError:
            pass
    if guard.poll() is None:
        guard.kill()
        cleanup["guard_sigkill_sent"] = True
    guard.wait(timeout=_STOP_GRACE_SECONDS)
    return cleanup


def _execute_guarded(
    command: Sequence[str], *, repo_root: Path, env: Mapping[str, str],
    stdout: BinaryIO, stderr: BinaryIO, timeout_seconds: float,
) -> dict[str, Any]:
    """Run a guard command; no shell, captured pipes, or unrelated-process scan."""

    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "failed", "returncode": None, "error": None,
        "guard_pid": None, "child_pid": None, "child_pgid": None,
    }
    guard = None
    try:
        guard = subprocess.Popen(
            list(command), cwd=repo_root, env=dict(env), stdout=stdout,
            stderr=stderr, start_new_session=True,
        )
        result["guard_pid"] = guard.pid
        while guard.poll() is None:
            child = _numerical_child(guard.pid)
            if child is not None:
                result["child_pid"], result["child_pgid"] = child
            if time.monotonic() - started >= timeout_seconds:
                result["status"] = "timed_out"
                break
            time.sleep(_POLL_SECONDS)
        else:
            result["status"] = "completed" if guard.returncode == 0 else "failed"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if guard is not None:
            # An interruption during interpreter startup can precede the
            # guard's child creation. Give that bounded startup a chance to
            # expose the owned group before asking the guard to stop it.
            discovery_deadline = time.monotonic() + 0.5
            while result["child_pgid"] is None and guard.poll() is None:
                child = _numerical_child(guard.pid)
                if child is not None:
                    result["child_pid"], result["child_pgid"] = child
                    break
                if time.monotonic() >= discovery_deadline:
                    break
                time.sleep(_POLL_SECONDS)
            try:
                result["cleanup"] = _stop_owned(guard, result["child_pgid"])
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if result["status"] == "completed":
                    result["status"] = "failed"
            result["returncode"] = guard.poll()
        result["wall_seconds"] = time.monotonic() - started
    return result


def execute_case(
    *, case_dir: Path, case: dict, python: str, repo_root: Path,
    timeout_seconds: float = 600, limit_mb: int = 3000,
) -> dict:
    """Execute a fresh BenchmarkCase; interruption is recorded, then re-raised.

    Existing case artifacts are never replaced. Precondition failures raise
    before reservation; every reserved case receives an execution receipt.
    """

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not 0 < limit_mb <= 17000:
        raise ValueError("limit_mb must be positive and at most 17000")
    parsed = BenchmarkCase.from_json_dict(case)
    if parsed.blas_threads != 1:
        raise ValueError("local selector cases require blas_threads=1")
    case_bytes = (json.dumps(case, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    case_dir, repo_root = Path(case_dir).resolve(), Path(repo_root).resolve()
    paths = {name: case_dir / name for name in (
        "case.json", "stdout.log", "stderr.log", "measurement.json", "execution.json",
    )}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite case artifact: {path}")
    if parsed.output_json:
        output = Path(parsed.output_json)
        if not output.is_absolute():
            output = repo_root / output
        if output.exists():
            raise FileExistsError(f"refusing to overwrite runner output: {output}")
    case_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is also the reservation when two callers race.
    with paths["case.json"].open("xb") as stream:
        stream.write(case_bytes)
    started = time.monotonic()
    env = dict(os.environ)
    env.update({name: "1" for name in BLAS_THREAD_ENV_VARS})
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        str(python), str(repo_root / "pipelines/shell/ram_guard.py"),
        "--limit-mb", str(limit_mb), "--interval", "0.25", "--",
        str(python), "-m", "pipelines.time_dynamics.diagnostics.ap_runtime_benchmark",
        "--single-case-json", str(paths["case.json"]),
        "--output-json", str(paths["measurement.json"]),
    ]
    receipt: dict[str, Any] = {
        "schema": "paper_ii_selector_local_execution_v1",
        "case_id": parsed.case_id, "case_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "repo_root": str(repo_root),
        "timeout_seconds": timeout_seconds, "limit_mb": limit_mb,
        "blas_environment": {name: env[name] for name in BLAS_THREAD_ENV_VARS},
        "paths": {name: str(path) for name, path in paths.items()},
        "status": "failed", "returncode": None, "error": None,
    }
    try:
        with paths["stdout.log"].open("xb") as stdout, paths["stderr.log"].open("xb") as stderr:
            receipt.update(_execute_guarded(
                command, repo_root=repo_root, env=env, stdout=stdout, stderr=stderr,
                timeout_seconds=timeout_seconds,
            ))
        if receipt["status"] == "completed":
            measurement = json.loads(paths["measurement.json"].read_text())
            if measurement.get("ok") is not True:
                receipt.update(status="failed", error="benchmark measurement did not report ok=true")
    except KeyboardInterrupt:
        receipt["status"] = "interrupted"
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        receipt["ok"] = receipt["status"] == "completed"
        receipt["measurement_exists"] = paths["measurement.json"].is_file()
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with paths["execution.json"].open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    if receipt["status"] == "interrupted":
        raise KeyboardInterrupt
    return receipt


__all__ = ["execute_case"]
