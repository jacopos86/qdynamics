#!/usr/bin/env python3
"""Looping controller for non-invasive Path A HH Optuna launches."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.hh_path_a_tmux_wrapper import DEFAULT_MANIFEST_PATH, ensure_path_a_run

DEFAULT_STATE_PATH = REPO_ROOT / "artifacts" / "agent_runs" / "hh_path_a_autopilot_v1" / "state.json"


# Built Math: slug(x) = sanitize(x) over [a-zA-Z0-9._-].
def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "path_a"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _tmux_bin() -> str:
    tmux = shutil.which("tmux")
    if tmux is None:
        raise RuntimeError("tmux is required for Path A autopilot.")
    return str(tmux)


def _session_prefix_from_manifest(manifest_path: Path) -> str:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    prefix = str(payload.get("tmux_session_prefix") or "hh_path_a")
    return _safe_slug(prefix)


def _list_tmux_sessions(prefix: str) -> list[str]:
    proc = subprocess.run(
        [_tmux_bin(), "list-sessions", "-F", "#{session_name}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    joined = "\n".join(x for x in (proc.stdout, proc.stderr) if x).lower()
    if proc.returncode != 0 and "no server running" not in joined:
        raise RuntimeError(f"tmux list-sessions failed: {joined.strip()}")
    sessions = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    wanted_prefix = f"{_safe_slug(prefix)}_"
    return [name for name in sessions if name.startswith(wanted_prefix)]


def _active_run_sessions(prefix: str) -> list[str]:
    return [name for name in _list_tmux_sessions(prefix) if "_autopilot_" not in name]


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_autopilot_once(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    tag_prefix: str = "hh_path_a_autorun",
    state_path: Path = DEFAULT_STATE_PATH,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    session_prefix = _session_prefix_from_manifest(manifest_path)
    active_sessions = _active_run_sessions(session_prefix)
    payload: dict[str, Any] = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_path_a_autopilot_v1",
        "manifest_path": str(manifest_path),
        "tag_prefix": str(tag_prefix),
        "active_sessions": active_sessions,
    }
    if active_sessions:
        payload["status"] = "waiting_for_active_run"
        _write_state(Path(state_path), payload)
        return payload

    launch_tag = f"{_safe_slug(tag_prefix)}_{_timestamp_slug()}"
    launch_result = ensure_path_a_run(tag=launch_tag, manifest_path=manifest_path)
    payload["status"] = str(launch_result.status)
    payload["launch_tag"] = str(launch_result.resolved_tag)
    payload["launch_result"] = {
        "requested_tag": launch_result.requested_tag,
        "resolved_tag": launch_result.resolved_tag,
        "session_name": launch_result.session_name,
        "status": launch_result.status,
        "output_dir": launch_result.output_dir,
        "logs_dir": launch_result.logs_dir,
        "ledger_path": launch_result.ledger_path,
        "manifest_path": launch_result.manifest_path,
        "command_path": launch_result.command_path,
        "optuna_command": list(launch_result.optuna_command),
    }
    _write_state(Path(state_path), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--tag-prefix", type=str, default="hh_path_a_autorun")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--poll-seconds", type=float, default=900.0)
    parser.add_argument("--once", action="store_true", help="Run one controller tick and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    while True:
        payload = run_autopilot_once(
            manifest_path=Path(args.manifest_path),
            tag_prefix=str(args.tag_prefix),
            state_path=Path(args.state_path),
        )
        print(json.dumps(payload, indent=2))
        if bool(args.once):
            return
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()
