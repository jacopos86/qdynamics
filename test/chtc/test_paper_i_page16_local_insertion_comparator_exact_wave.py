from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_cleared_local_page16_insertion_comparator_wave_20260813.py"
)


def _load_launcher():
    name = "paper_i_page16_exact_cleared_wave_launcher_test"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_wave_launcher_pins_supervisor_and_excludes_blocked_wave2() -> None:
    launcher = _load_launcher()
    assert launcher.ALLOWED_WAVES == (3, 4, 5)
    assert (
        launcher._sha256_file(launcher.SUPERVISOR_PATH)
        == launcher.EXPECTED_SUPERVISOR_SHA256
    )
    with pytest.raises(launcher.ClearedWaveError, match="waves 3, 4, and 5"):
        launcher.preflight(2)


def test_exact_wave_preflight_delegates_both_clearance_and_runner_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()

    class Supervisor:
        @staticmethod
        def _require_remote_overlap_clearance(wave: int):
            assert wave == 4
            return {"sha256": "a" * 64, "valid_until_utc": "2026-08-13T04:00:00Z"}

        @staticmethod
        def _runner_preflight():
            return {
                "local_adapter_sha256": "b" * 64,
                "capacity_ready": True,
                "run_ready": True,
            }

    monkeypatch.setattr(launcher, "_load_supervisor", lambda: Supervisor)
    value = launcher.preflight(4)
    assert value["status"] == "passed_inert_exact_wave_preflight"
    assert value["wave"] == 4
    assert value["scientific_execution_performed"] is False
    assert value["submission_performed"] is False
