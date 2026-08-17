#!/usr/bin/env python3
"""Wait for Page-16 wave 1, then execute waves 2--5 in order.

The scientific work remains delegated to the source-locked local adapter.  This
supervisor is only a persistent scheduler: it pins the adapter bytes, waits for
the active wave lock to be released, re-runs the inert preflight before every
wave, and refuses failed, interrupted, or ambiguous state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RUNNER = SCRIPT_PATH.with_name(
    "run_local_page16_insertion_comparators_20260812.py"
)
EXPECTED_RUNNER_SHA256 = (
    "bd9d61fb98b48911c3da04faf8b6c38eb391b1a02ab3362e22ef02316a414c4e"
)
ACTIVATION_DIR = SCRIPT_PATH.with_name(
    "paper_i_ra_adapt_page16_insertion_comparators_k30_"
    "20260812_v2_local_activation"
)
RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page16_insertion_comparators_k30_20260812_v2"
)
REMOTE_CLEARANCE_DIR = SCRIPT_PATH.with_name(
    "paper_i_ra_adapt_page16_insertion_comparators_k30_"
    "20260812_v2_remote_overlap_clearances"
)
STATUS_PATH = RUNTIME_DIR / "status/waves_2_5_supervisor.json"
LOCK_PATH = RUNTIME_DIR / "wave_supervisor.lock"
POLL_SECONDS = 20
WAIT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
STATUS_SCHEMA = (
    "paper_i_page16_insertion_comparator_waves_2_5_supervisor_status_v2"
)
REMOTE_CLEARANCE_SCHEMA = (
    "paper_i_page16_insertion_comparator_no_remote_overlap_clearance_v1"
)
REMOTE_CLEARANCE_AUTHENTICATION_KIND = (
    "interactive_ssh_duo_condor_q_snapshot_v1"
)
REMOTE_CLEARANCE_MAX_WINDOW_SECONDS = 15 * 60
LOCAL_ACTIVATION_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_activation_manifest_v2"
)
LOCAL_RUNTIME_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_runtime_manifest_v2"
)


class SupervisorError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _utc_datetime(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SupervisorError(f"{label} is not an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise SupervisorError(f"{label} must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


def _sha256_text(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _validate_remote_overlap_clearance(
    value: Mapping[str, Any],
    *,
    wave_number: int,
    execution_ids: tuple[str, ...],
    runner_sha256: str,
    activation_manifest_sha256: str,
    runtime_manifest_sha256: str,
    activation_dir: Path,
    runtime_dir: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate one short-lived authenticated CHTC overlap clearance."""

    observed_at = _utc_datetime(
        value.get("observed_at_utc"), label="clearance observed_at_utc"
    )
    valid_until = _utc_datetime(
        value.get("valid_until_utc"), label="clearance valid_until_utc"
    )
    current = (
        datetime.now(timezone.utc)
        if now is None
        else now.astimezone(timezone.utc)
    )
    if (
        value.get("schema") != REMOTE_CLEARANCE_SCHEMA
        or value.get("status")
        != "passed_authenticated_no_remote_overlap_clearance"
        or value.get("wave") != wave_number
        or value.get("execution_ids") != list(execution_ids)
        or value.get("runner_sha256") != runner_sha256
        or value.get("activation_manifest_sha256")
        != activation_manifest_sha256
        or value.get("runtime_manifest_sha256") != runtime_manifest_sha256
        or value.get("activation_dir") != activation_dir.as_posix()
        or value.get("runtime_dir") != runtime_dir.as_posix()
        or value.get("authentication_kind")
        != REMOTE_CLEARANCE_AUTHENTICATION_KIND
        or value.get("authenticated_remote_query") is not True
        or value.get("scheduler") != "chtc_condor"
        or not _sha256_text(value.get("scheduler_snapshot_sha256"))
        or value.get("remote_active_execution_ids") != []
        or value.get("overlapping_execution_ids") != []
        or value.get("remote_factories_frozen") is not True
        or value.get("no_remote_overlap") is not True
        or value.get("scientific_execution_performed") is not False
        or observed_at > current
        or current > valid_until
        or valid_until <= observed_at
        or (valid_until - observed_at).total_seconds()
        > REMOTE_CLEARANCE_MAX_WINDOW_SECONDS
    ):
        raise SupervisorError(
            f"Wave {wave_number} remote-overlap clearance drifted."
        )
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SupervisorError(f"{label} is absent or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupervisorError(f"{label} must be a JSON object.")
    supplied = value.get("sha256")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise SupervisorError(f"{label} self-digest drifted.")
    return value


def _write_status(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.update(
        {
            "schema": STATUS_SCHEMA,
            "runner_sha256": EXPECTED_RUNNER_SHA256,
            "updated_at_utc": _utc_now(),
        }
    )
    payload = {**unsigned, "sha256": _canonical_sha256(unsigned)}
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_name(f".{STATUS_PATH.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise SupervisorError(f"Stale status temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, STATUS_PATH)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def _verify_fixed_inputs() -> None:
    if not RUNNER.is_file() or RUNNER.is_symlink():
        raise SupervisorError("Pinned local runner is absent or unsafe.")
    observed = _sha256_file(RUNNER)
    if observed != EXPECTED_RUNNER_SHA256:
        raise SupervisorError(
            "Pinned local runner drifted: "
            f"expected {EXPECTED_RUNNER_SHA256}, observed {observed}."
        )
    if not ACTIVATION_DIR.is_dir() or ACTIVATION_DIR.is_symlink():
        raise SupervisorError("Validated local activation is absent or unsafe.")


def _require_remote_overlap_clearance(
    wave_number: int,
) -> dict[str, Any]:
    if wave_number not in {2, 3, 4, 5}:
        raise SupervisorError("Remote-overlap clearance is scoped to waves 2--5.")
    _verify_fixed_inputs()
    activation = _load_digested(
        ACTIVATION_DIR / "activation_manifest.json",
        label="v2 local activation manifest",
    )
    runtime = _load_digested(
        RUNTIME_DIR / "runtime_manifest.json",
        label="v2 local runtime manifest",
    )
    waves = activation.get("waves")
    if (
        activation.get("schema") != LOCAL_ACTIVATION_SCHEMA
        or not isinstance(waves, list)
        or len(waves) != 5
        or runtime.get("schema") != LOCAL_RUNTIME_SCHEMA
        or runtime.get("activation_manifest_sha256") != activation.get("sha256")
    ):
        raise SupervisorError(
            "V2 activation/runtime drifted before remote-overlap clearance."
        )
    execution_rows = waves[wave_number - 1]
    if (
        not isinstance(execution_rows, list)
        or len(execution_rows) != 2
        or not all(isinstance(row, str) and row for row in execution_rows)
    ):
        raise SupervisorError(
            f"Wave {wave_number} execution inventory is malformed."
        )
    clearance = _load_digested(
        REMOTE_CLEARANCE_DIR / f"wave_{wave_number}.json",
        label=f"wave {wave_number} authenticated remote-overlap clearance",
    )
    return _validate_remote_overlap_clearance(
        clearance,
        wave_number=wave_number,
        execution_ids=tuple(execution_rows),
        runner_sha256=EXPECTED_RUNNER_SHA256,
        activation_manifest_sha256=str(activation["sha256"]),
        runtime_manifest_sha256=str(runtime["sha256"]),
        activation_dir=ACTIVATION_DIR,
        runtime_dir=RUNTIME_DIR,
    )


def _runner_command(*extra: str) -> list[str]:
    return [
        sys.executable,
        "-B",
        RUNNER.as_posix(),
        "--activation-dir",
        ACTIVATION_DIR.as_posix(),
        "--runtime-dir",
        RUNTIME_DIR.as_posix(),
        *extra,
    ]


def _runner_preflight() -> dict[str, Any]:
    _verify_fixed_inputs()
    result = subprocess.run(
        _runner_command("--preflight"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SupervisorError(
            "Local runner preflight failed: " + result.stderr.strip()
        )
    value = json.loads(result.stdout)
    if (
        not isinstance(value, dict)
        or value.get("status") != "passed_inert_preflight"
        or value.get("activation_status") != "validated"
        or value.get("capacity_ready") is not True
        or value.get("run_ready") is not True
        or value.get("local_adapter_sha256") != EXPECTED_RUNNER_SHA256
        or value.get("scientific_execution_performed") is not False
        or value.get("submission_performed") is not False
    ):
        raise SupervisorError("Local runner preflight did not close run-ready.")
    return value


def _wave_status(wave_number: int) -> dict[str, Any] | None:
    path = RUNTIME_DIR / f"status/wave_{wave_number}.json"
    if not path.exists() and not path.is_symlink():
        return None
    value = _load_digested(path, label=f"wave {wave_number} status")
    if value.get("wave") != wave_number:
        raise SupervisorError(f"Wave {wave_number} status identity drifted.")
    return value


def _wave_lock_available() -> bool:
    if not LOCK_PATH.exists():
        return True
    with LOCK_PATH.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return True


def _wait_for_wave_1() -> dict[str, Any]:
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = _wave_status(1)
        if status is not None:
            state = str(status.get("status", ""))
            if state in {"failed", "interrupted"}:
                raise SupervisorError(f"Wave 1 ended in {state} state.")
            if state in {"passed", "passed_already_complete"}:
                if _wave_lock_available():
                    return status
        _write_status(
            {
                "status": "waiting_for_wave_1",
                "completed_waves": [],
                "next_wave": 2,
                "round50_continuation_executed": False,
            }
        )
        time.sleep(POLL_SECONDS)
    raise SupervisorError("Timed out waiting for wave 1 to close.")


def _run_wave(wave_number: int) -> dict[str, Any]:
    _require_remote_overlap_clearance(wave_number)
    _runner_preflight()
    result = subprocess.run(
        _runner_command("--run-wave", str(wave_number)),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SupervisorError(
            f"Wave {wave_number} failed: {result.stderr.strip()}"
        )
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or value.get("status") not in {
        "passed",
        "passed_already_complete",
    }:
        raise SupervisorError(f"Wave {wave_number} did not close passed.")
    return value


def preflight() -> dict[str, Any]:
    runner = _runner_preflight()
    wave_1 = _wave_status(1)
    return {
        "schema": STATUS_SCHEMA,
        "status": "passed_inert_supervisor_preflight",
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "activation_status": runner["activation_status"],
        "capacity_ready": runner["capacity_ready"],
        "run_ready": runner["run_ready"],
        "wave_1_status": None if wave_1 is None else wave_1.get("status"),
        "waves_to_execute": [2, 3, 4, 5],
        "remote_overlap_clearance_required_before_each_wave": True,
        "remote_overlap_clearance_directory": REMOTE_CLEARANCE_DIR.as_posix(),
        "maximum_concurrency": 1,
        "scientific_execution_performed": False,
        "submission_performed": False,
    }


def supervise() -> dict[str, Any]:
    _runner_preflight()
    _wait_for_wave_1()
    completed = [1]
    for wave_number in range(2, 6):
        _write_status(
            {
                "status": "running_wave",
                "completed_waves": completed,
                "next_wave": wave_number,
                "round50_continuation_executed": False,
            }
        )
        _run_wave(wave_number)
        completed.append(wave_number)
    return _write_status(
        {
            "status": "passed_all_five_k30_waves",
            "completed_waves": completed,
            "next_wave": None,
            "round50_continuation_executed": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Persistent supervisor for Page-16 local k30 waves 2--5"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        payload = preflight() if args.preflight else supervise()
    except (OSError, ValueError, json.JSONDecodeError, SupervisorError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
