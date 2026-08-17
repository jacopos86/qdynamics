#!/usr/bin/env python3
"""Run one Page-16 k30 wave after its authenticated CHTC clearance.

This narrow adapter exists so independently cleared waves can run while an
unrelated member of wave 2 is still active on CHTC.  Scientific execution is
still performed by the pinned v2 runner through its existing supervisor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SUPERVISOR_PATH = SCRIPT_PATH.with_name(
    "supervise_local_page16_insertion_comparator_waves_20260812.py"
)
EXPECTED_SUPERVISOR_SHA256 = (
    "bdacd23e4a8c09db9e2036454032afd0362a083f8aad986b69892c0116a74a11"
)
ALLOWED_WAVES = (3, 4, 5)


class ClearedWaveError(RuntimeError):
    """Raised when the exact-wave launch contract is not closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_supervisor() -> Any:
    if not SUPERVISOR_PATH.is_file() or SUPERVISOR_PATH.is_symlink():
        raise ClearedWaveError("Pinned wave supervisor is absent or unsafe.")
    observed = _sha256_file(SUPERVISOR_PATH)
    if observed != EXPECTED_SUPERVISOR_SHA256:
        raise ClearedWaveError(
            "Pinned wave supervisor drifted: "
            f"expected {EXPECTED_SUPERVISOR_SHA256}, observed {observed}."
        )
    name = "paper_i_page16_pinned_exact_wave_supervisor"
    spec = importlib.util.spec_from_file_location(name, SUPERVISOR_PATH)
    if spec is None or spec.loader is None:
        raise ClearedWaveError("Cannot import the pinned wave supervisor.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight(wave: int) -> dict[str, Any]:
    if wave not in ALLOWED_WAVES:
        raise ClearedWaveError("Only independently runnable waves 3, 4, and 5 are allowed.")
    supervisor = _load_supervisor()
    clearance = supervisor._require_remote_overlap_clearance(wave)
    runner = supervisor._runner_preflight()
    return {
        "schema": "paper_i_page16_insertion_comparator_exact_cleared_wave_preflight_v1",
        "status": "passed_inert_exact_wave_preflight",
        "wave": wave,
        "supervisor_sha256": EXPECTED_SUPERVISOR_SHA256,
        "runner_sha256": runner["local_adapter_sha256"],
        "clearance_sha256": clearance["sha256"],
        "clearance_valid_until_utc": clearance["valid_until_utc"],
        "capacity_ready": runner["capacity_ready"],
        "run_ready": runner["run_ready"],
        "scientific_execution_performed": False,
        "submission_performed": False,
    }


def run(wave: int) -> dict[str, Any]:
    preflight(wave)
    supervisor = _load_supervisor()
    return supervisor._run_wave(wave)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one independently cleared Page-16 local k30 wave"
    )
    parser.add_argument("--wave", type=int, choices=ALLOWED_WAVES, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        payload = preflight(args.wave) if args.preflight else run(args.wave)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ClearedWaveError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # The pinned supervisor exposes its own fail-closed exception type only
        # after import, so preserve that exact failure without broad recovery.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
