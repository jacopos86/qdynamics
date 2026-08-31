#!/usr/bin/env python3
"""Looping controller for non-invasive Path B proposal-cycle bundles."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.pareto_offline.proposal_cycle import run_proposal_cycle

DEFAULT_MANIFEST_PATH = Path(__file__).with_name("proposal_cycle_manifest.json")
DEFAULT_STATE_PATH = REPO_ROOT / "artifacts" / "agent_runs" / "hh_path_b_autopilot_v1" / "state.json"
DEFAULT_PATH_A_RUN_ROOT = REPO_ROOT / "artifacts" / "agent_runs"
DEFAULT_PATH_A_MAX_CASES = 6


# Built Math: slug(x) = sanitize(x) over [a-zA-Z0-9._-].
def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "path_b"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_autopilot_once(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    tag_prefix: str = "hh_path_b_proposal_cycle",
    state_path: Path = DEFAULT_STATE_PATH,
    path_a_run_root: Path = DEFAULT_PATH_A_RUN_ROOT,
    path_a_max_cases: int = DEFAULT_PATH_A_MAX_CASES,
) -> dict[str, Any]:
    run_tag = f"{_safe_slug(tag_prefix)}_{_timestamp_slug()}"
    summary = run_proposal_cycle(
        tag=run_tag,
        manifest_path=manifest_path,
        include_path_a_discovery=True,
        path_a_run_root=path_a_run_root,
        path_a_max_cases=int(path_a_max_cases),
    )
    payload: dict[str, Any] = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_path_b_autopilot_v1",
        "manifest_path": str(Path(manifest_path)),
        "path_a_run_root": str(Path(path_a_run_root)),
        "path_a_max_cases": int(path_a_max_cases),
        "status": "cycled",
        "run_tag": run_tag,
        "summary": summary,
    }
    _write_state(Path(state_path), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--tag-prefix", type=str, default="hh_path_b_proposal_cycle")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--path-a-run-root", type=Path, default=DEFAULT_PATH_A_RUN_ROOT)
    parser.add_argument("--path-a-max-cases", type=int, default=DEFAULT_PATH_A_MAX_CASES)
    parser.add_argument("--poll-seconds", type=float, default=1800.0)
    parser.add_argument("--once", action="store_true", help="Run one proposal-cycle tick and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    while True:
        payload = run_autopilot_once(
            manifest_path=Path(args.manifest_path),
            tag_prefix=str(args.tag_prefix),
            state_path=Path(args.state_path),
            path_a_run_root=Path(args.path_a_run_root),
            path_a_max_cases=int(args.path_a_max_cases),
        )
        print(json.dumps(payload, indent=2))
        if bool(args.once):
            return
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()
