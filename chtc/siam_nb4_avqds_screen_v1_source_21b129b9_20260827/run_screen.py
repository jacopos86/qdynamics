#!/usr/bin/env python3
"""Run the single exact-blind SIAM Nb=4 AVQDS screen action."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from pipelines.time_dynamics.paper_ii_runs import (
    RUNNER_PATH,
    build_siam_nb4_screen_run,
)


PACKAGE = Path("chtc/siam_nb4_avqds_screen_v1_source_21b129b9_20260827")
ACTION_ID = "screen_avqds_nb4_t1_cut1e-3"
SOURCE_REVISION = "21b129b9c4a1743c58b221f5770eada17a3a19cc"
SEED_PATH = PACKAGE / "input/seed.json"
SEED_SHA256 = "c664f90c8f6f284a593a9347596f2f3c19b7dec29055977a0bbbe5c22b9dad72"
ACTION_PATH = PACKAGE / "input/action.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f"usage: run_screen.py {ACTION_ID}")
    if not SEED_PATH.is_file() or _sha256(SEED_PATH) != SEED_SHA256:
        raise SystemExit("SIAM seed is missing or fails its SHA-256 lock")

    action = json.loads(ACTION_PATH.read_text(encoding="utf-8"))
    if action.get("action_id") != ACTION_ID:
        raise SystemExit("SIAM action identity mismatch")
    if action.get("seed_sha256") != SEED_SHA256:
        raise SystemExit("SIAM action does not bind the verified seed")
    if action.get("source_revision") != SOURCE_REVISION:
        raise SystemExit("SIAM action does not bind the source revision")
    if action.get("exact_reference_blind") is not True:
        raise SystemExit("SIAM action is not declared exact-reference blind")
    if action.get("compile_terminal_cost") is not False:
        raise SystemExit("SIAM screen must not request Qiskit compilation")

    output_dir = Path("raw_outputs") / ACTION_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = output_dir / "run.json"
    receipt = output_dir / "action_receipt.json"
    command = build_siam_nb4_screen_run(
        seed_path=str(SEED_PATH),
        arm="avqds",
        activation_cut=1.0e-3,
        output_json=str(trajectory),
        code_revision=SOURCE_REVISION,
    )
    argv = tuple(command.argv())
    argv_sha256 = hashlib.sha256(
        json.dumps(
            list(argv), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    if argv_sha256 != action.get("expected_runner_argv_sha256"):
        raise SystemExit("resolved SIAM runner command fails its digest lock")
    if command.run_id != action.get("expected_run_id"):
        raise SystemExit("resolved SIAM runner command fails its run-id lock")
    prohibited = {
        "--exact-reference",
        "--exact-reference-json",
        "--reference-json",
        "--seed-reference-json",
    }
    if prohibited.intersection(argv):
        raise SystemExit("raw SIAM production command is not exact-reference blind")
    if "--record-statevector" not in argv:
        raise SystemExit("raw SIAM production command must retain checkpoint states")

    base_receipt: dict[str, object] = {
        "schema": "siam_nb4_chtc_action_receipt_v1",
        "action_id": ACTION_ID,
        "run_id": command.run_id,
        "source_revision": SOURCE_REVISION,
        "seed_path": str(SEED_PATH),
        "seed_sha256": SEED_SHA256,
        "arm": "avqds",
        "activation_cut": 1.0e-3,
        "segment": "screen_t0_t1_101",
        "exact_reference_blind": True,
        "qiskit_compilation_requested": False,
        "runner_argv_sha256": argv_sha256,
        "runner_argv": list(argv),
        "trajectory_json": str(trajectory),
    }
    _write_json(receipt, {**base_receipt, "status": "RUNNING"})
    process_argv = [sys.executable, RUNNER_PATH, *argv]
    try:
        subprocess.run(process_argv, check=True)
        if not trajectory.is_file() or trajectory.stat().st_size == 0:
            raise RuntimeError("runner returned without a nonempty trajectory")
        payload = json.loads(trajectory.read_text(encoding="utf-8"))
        run_lock = dict(payload.get("run_lock") or {})
        summary = dict(payload.get("summary") or {})
        resume_state = dict(payload.get("resume_state") or {})
        rows = list(payload.get("trajectory") or [])
        checks = {
            "source_revision": run_lock.get("code_revision") == SOURCE_REVISION,
            "uncured_local_subdivisions": (
                summary.get("uncured_local_subdivision_count") == 0
            ),
            "resume_schema": resume_state.get("schema") == "paper_ii_resume_state_v2",
            "trajectory_points": len(rows) == 101,
            "terminal_time": bool(rows) and float(rows[-1].get("time")) == 1.0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"completed trajectory failed boundary checks: {checks}")
        _write_json(
            receipt,
            {
                **base_receipt,
                "status": "COMPLETED",
                "checks": checks,
                "trajectory_bytes": trajectory.stat().st_size,
                "trajectory_sha256": _sha256(trajectory),
            },
        )
    except BaseException as exc:
        _write_json(
            receipt,
            {
                **base_receipt,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "trajectory_present": trajectory.is_file(),
            },
        )
        raise


if __name__ == "__main__":
    main()
