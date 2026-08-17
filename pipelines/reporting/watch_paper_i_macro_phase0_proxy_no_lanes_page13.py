#!/usr/bin/env python3
"""Refresh Page 13 whenever another local macro-only cell reaches round 50."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (
    append_paper_i_macro_phase0_proxy_no_lanes_page13 as page13,
)


RUNTIME_DIR = page13.RUNTIME_DIR
SERIAL_STATUS_PATH = RUNTIME_DIR / "serial_status.json"
SERIAL_MANIFEST_PATH = RUNTIME_DIR / "serial_manifest.json"
WATCH_STATUS_PATH = RUNTIME_DIR / "page13_auto_refresh_status.json"
LOCK_PATH = RUNTIME_DIR / "page13_auto_refresh.lock"
UPDATER_PATH = Path(page13.__file__).resolve()
SCHEMA = "paper_i_macro_phase0_page13_auto_refresh_status_v1"


class WatchError(ValueError):
    pass


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return page13._canonical_sha256(value)


def _write_status(value: Mapping[str, Any]) -> None:
    payload = copy.deepcopy(dict(value))
    payload["sha256"] = _canonical_sha256(payload)
    page13._atomic_json(WATCH_STATUS_PATH, payload)
    artifact_root = os.environ.get("REMOTE_ARTIFACT_DIR")
    if artifact_root:
        artifact_path = Path(artifact_root) / "page13_auto_refresh_status.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        page13._atomic_json(artifact_path, payload)


def _load_previous_status() -> dict[str, Any] | None:
    if not WATCH_STATUS_PATH.is_file():
        return None
    value = page13.load(WATCH_STATUS_PATH)
    claimed = value.pop("sha256", None)
    if claimed != _canonical_sha256(value):
        raise WatchError("watch status self digest drifted")
    value["sha256"] = claimed
    return value


def expected_execution_ids() -> tuple[str, ...]:
    manifest = page13.load(SERIAL_MANIFEST_PATH)
    page13.verify_self_digest(manifest, label="local serial manifest")
    rows = manifest.get("execution_ids")
    if (
        manifest.get("status") != "authorized_pending_execution"
        or manifest.get("target_horizon") != 50
        or not isinstance(rows, list)
        or len(rows) != 6
        or len(set(rows)) != 6
    ):
        raise WatchError("local serial manifest identity drifted")
    return tuple(str(row) for row in rows)


def validated_completed_execution_ids(
    serial_status: Mapping[str, Any],
) -> tuple[str, ...]:
    expected = expected_execution_ids()
    raw = serial_status.get(
        "published_completed_execution_ids",
        serial_status.get("completed_execution_ids"),
    )
    if not isinstance(raw, list):
        raise WatchError("serial status has no completed-execution list")
    claimed = tuple(str(row) for row in raw)
    if len(claimed) != len(set(claimed)) or not set(claimed).issubset(expected):
        raise WatchError("serial completion set is duplicated or unauthorized")
    ordered_claimed = tuple(row for row in expected if row in set(claimed))
    jobs = page13._jobs()
    exact = page13.exact_references()
    jobs_by_id = {
        str(job["execution_id"]): regime
        for regime, (_, job) in jobs.items()
    }
    for execution_id in ordered_claimed:
        regime = jobs_by_id.get(execution_id)
        if regime is None:
            raise WatchError(f"completed execution is not authorized: {execution_id}")
        closed = page13._completed_route(execution_id, exact=exact[regime])
        if closed is None or closed.get("latest", {}).get("k") != 50:
            raise WatchError(f"round-50 closure is not published: {execution_id}")
    return ordered_claimed


def _run_updater() -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
    completed = subprocess.run(
        [sys.executable, "-B", str(UPDATER_PATH)],
        cwd=page13.REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise WatchError("Page-13 updater returned non-JSON output") from exc
    pdf = result.get("pdf")
    if (
        result.get("page_count") != 13
        or result.get("status") != "updated_existing_report_in_place"
        or not isinstance(pdf, Mapping)
        or not isinstance(pdf.get("sha256"), str)
        or len(pdf["sha256"]) != 64
        or int(pdf.get("size_bytes", -1)) <= 0
    ):
        raise WatchError("Page-13 updater returned an unexpected result")
    return result


def _state(
    *,
    serial_status: Mapping[str, Any],
    completed: tuple[str, ...],
    refreshed: tuple[str, ...],
    status: str,
    refresh_result: Mapping[str, Any] | None,
    last_error: str | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "target_horizon": 50,
        "serial_status": {
            "status": serial_status.get("status"),
            "current_execution_id": serial_status.get("current_execution_id"),
            "completed_execution_ids": list(completed),
        },
        "validated_completed_execution_ids": list(completed),
        "refreshed_completed_execution_ids": list(refreshed),
        "page13_pdf": page13.binding(page13.TARGET_PDF),
        "page13_provenance": page13.binding(page13.TARGET_PROVENANCE),
        "last_refresh_result": (
            None if refresh_result is None else copy.deepcopy(dict(refresh_result))
        ),
        "last_error": last_error,
    }


def watch(*, poll_seconds: float, once: bool) -> int:
    expected = expected_execution_ids()
    previous = _load_previous_status()
    refreshed = tuple(
        str(row)
        for row in (
            []
            if previous is None
            else previous.get("refreshed_completed_execution_ids", [])
        )
    )
    while True:
        serial_status = page13.load(SERIAL_STATUS_PATH)
        page13.verify_self_digest(serial_status, label="local serial status")
        completed = validated_completed_execution_ids(serial_status)
        if not set(refreshed).issubset(completed):
            raise WatchError(
                "validated serial completion set regressed behind the last refresh"
            )
        refresh_result: dict[str, Any] | None = None
        last_error: str | None = None
        needs_refresh = len(completed) > len(refreshed)
        if not needs_refresh and previous is not None:
            previous_pdf = previous.get("page13_pdf")
            if isinstance(previous_pdf, Mapping):
                needs_refresh = previous_pdf.get("sha256") != page13.sha256(
                    page13.TARGET_PDF
                )
        if needs_refresh:
            try:
                refresh_result = _run_updater()
                refreshed = completed
                print(
                    json.dumps(
                        {
                            "event": "page13_refreshed",
                            "completed_count": len(completed),
                            "completed_execution_ids": list(completed),
                            "pdf_sha256": refresh_result["pdf"]["sha256"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                last_error = str(exc)
                print(
                    json.dumps(
                        {"event": "page13_refresh_failed", "error": last_error},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        serial_state = str(serial_status.get("status"))
        if serial_state == "passed" and completed == expected and refreshed == expected:
            watcher_state = "passed_all_six_round50_cells_refreshed"
        elif serial_state == "failed":
            watcher_state = "source_serial_run_failed"
        elif last_error is not None:
            watcher_state = "refresh_retry_pending"
        else:
            watcher_state = "watching_for_next_round50_completion"
        payload = _state(
            serial_status=serial_status,
            completed=completed,
            refreshed=refreshed,
            status=watcher_state,
            refresh_result=refresh_result,
            last_error=last_error,
        )
        _write_status(payload)
        previous = payload
        if watcher_state == "passed_all_six_round50_cells_refreshed":
            return 0
        if watcher_state == "source_serial_run_failed":
            return 2
        if once:
            return 0 if last_error is None else 1
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1.0:
        raise SystemExit("--poll-seconds must be at least 1")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Page-13 completion watcher is already running.", file=sys.stderr)
            return 3
        lock_stream.seek(0)
        lock_stream.truncate()
        lock_stream.write(f"{os.getpid()}\n")
        lock_stream.flush()
        return watch(poll_seconds=args.poll_seconds, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
