from __future__ import annotations

import importlib.util
import fcntl
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_cleared_local_page16_weak_strong_always_20260813.py"
)


def _load_launcher():
    name = "paper_i_page16_weak_strong_always_single_cell_test"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_exactly_scopes_weak_strong_always_and_pins_live_inputs() -> None:
    launcher = _load_launcher()
    runner = launcher._load_runner()
    continuation = launcher._load_continuation_adapter()

    assert launcher.TARGET_EXECUTION_ID == runner.WAVES[1][1]
    assert launcher.EXCLUDED_REMOTE_EXECUTION_ID == runner.WAVES[1][0]
    assert launcher.TARGET_EXECUTION_ID in continuation.CONDITIONAL_EXECUTION_IDS
    assert (
        launcher.EXCLUDED_REMOTE_EXECUTION_ID
        == continuation.SW_ALWAYS_CHTC_EXECUTION_ID
    )
    assert launcher.TARGET_EXECUTION_ID != launcher.EXCLUDED_REMOTE_EXECUTION_ID
    assert runner.REGIME_BY_EXECUTION_ID[launcher.TARGET_EXECUTION_ID] == (
        "weak_strong"
    )
    assert runner.NPH_BY_EXECUTION_ID[launcher.TARGET_EXECUTION_ID] == 7
    assert runner.MAX_CONCURRENCY == 1
    assert launcher.EXPECTED_TARGET_JOB_SHA256 == (
        "9d6ddafed245ff15c23568355d8d4ce1bdc3828443c69e082c9d160aadb13eec"
    )
    assert launcher.EXPECTED_TARGET_PROTOCOL_SHA256 == (
        "57e4043b01b21d6971a43b4e0a12985045ab7f74457228d39aa9e8e0fdbf62e3"
    )
    assert launcher.EXPECTED_TARGET_ROUTE_CONTRACT_SHA256 == (
        "9b9d6bdbb9edb6128e2f0973dd740b44d0daa00d55ecd910fd587f091ae81338"
    )
    assert launcher.EXPECTED_TARGET_SOURCE_LOCKS_SHA256 == (
        "fc4bdd4c1d1419ffa669c7ea619a456330790e60a6166dbf5a36ca304076df71"
    )
    assert launcher._sha256_file(launcher.RUNNER_PATH) == (
        launcher.EXPECTED_RUNNER_SHA256
    )
    assert launcher._sha256_file(launcher.CONTINUATION_ADAPTER_PATH) == (
        launcher.EXPECTED_CONTINUATION_ADAPTER_SHA256
    )


def test_preflight_waits_for_exact_sw_exclusion_receipt_without_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    context = SimpleNamespace(
        runner=SimpleNamespace(MAX_CONCURRENCY=1),
        activation={"sha256": "a" * 64},
        runtime={"sha256": "r" * 64},
    )
    monkeypatch.setattr(launcher, "_fixed_context", lambda: context)
    monkeypatch.setattr(
        launcher,
        "SW_CLOSURE_RECEIPT_PATH",
        tmp_path / "absent.json",
    )
    monkeypatch.setattr(
        launcher,
        "_wave5_terminal_state",
        lambda _context: {"terminal": False, "status": "absent"},
    )
    monkeypatch.setattr(
        launcher,
        "_target_closure_state",
        lambda _context: {"closed": False, "decision": None},
    )
    monkeypatch.setattr(launcher, "_excluded_remote_paths_absent", lambda: True)
    monkeypatch.setattr(launcher, "_standard_wave2_status_absent", lambda: True)
    monkeypatch.setattr(launcher, "_local_scientific_overlap", lambda: [])
    monkeypatch.setattr(launcher, "_wave_lock_available", lambda: True)
    monkeypatch.setattr(
        launcher,
        "_runner_preflight",
        lambda: {"capacity_ready": True, "run_ready": True},
    )

    value = launcher.preflight()

    assert value["status"] == (
        "waiting_for_authenticated_sw_always_closure_and_remote_"
        "materialization_exclusion"
    )
    assert value["run_ready"] is False
    assert value["target_execution_id"] == launcher.TARGET_EXECUTION_ID
    assert value["excluded_remote_execution_id"] == (
        launcher.EXCLUDED_REMOTE_EXECUTION_ID
    )
    assert value["scientific_execution_performed"] is False
    assert value["submission_performed"] is False
    assert not launcher.STATUS_PATH.exists()


def test_ready_preflight_requires_wave5_lock_overlap_and_exclusion_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    context = SimpleNamespace(
        runner=SimpleNamespace(MAX_CONCURRENCY=1),
        activation={"sha256": "a" * 64},
        runtime={"sha256": "r" * 64},
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "_fixed_context", lambda: context)
    monkeypatch.setattr(launcher, "SW_CLOSURE_RECEIPT_PATH", receipt_path)
    monkeypatch.setattr(
        launcher,
        "_authenticate_sw_exclusion",
        lambda _context: {
            "sha256": "s" * 64,
            "remote_materialization_exclusion_authenticated": True,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_wave5_terminal_state",
        lambda _context: {"terminal": False, "status": "absent"},
    )
    monkeypatch.setattr(
        launcher,
        "_target_closure_state",
        lambda _context: {"closed": False, "decision": None},
    )
    monkeypatch.setattr(launcher, "_excluded_remote_paths_absent", lambda: True)
    monkeypatch.setattr(launcher, "_standard_wave2_status_absent", lambda: True)
    monkeypatch.setattr(launcher, "_local_scientific_overlap", lambda: [])
    monkeypatch.setattr(launcher, "_wave_lock_available", lambda: True)
    monkeypatch.setattr(
        launcher,
        "_runner_preflight",
        lambda: {"capacity_ready": True, "run_ready": True},
    )

    waiting = launcher.preflight()
    assert waiting["status"] == "waiting_for_terminal_wave_5"
    assert waiting["run_ready"] is False

    monkeypatch.setattr(
        launcher,
        "_wave5_terminal_state",
        lambda _context: {"terminal": True, "status": "passed"},
    )
    ready = launcher.preflight()
    assert ready["status"] == "passed_ready_for_exact_single_cell_launch"
    assert ready["run_ready"] is True

    monkeypatch.setattr(
        launcher,
        "_local_scientific_overlap",
        lambda: ["123 other-scientific-worker"],
    )
    blocked = launcher.preflight()
    assert blocked["status"] == "waiting_for_local_exclusivity"
    assert blocked["run_ready"] is False
    assert blocked["local_scientific_overlap"] == [
        "123 other-scientific-worker"
    ]


def test_child_command_can_name_only_the_ws_cell() -> None:
    launcher = _load_launcher()

    command = launcher._child_command()

    assert "--run-cell" in command
    assert command[-1] == launcher.TARGET_EXECUTION_ID
    assert "--run-wave" not in command
    assert launcher.EXCLUDED_REMOTE_EXECUTION_ID not in command


def test_supervisor_serializes_one_child_and_publishes_continuation_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    runtime = tmp_path / "runtime"
    for name in ("logs", "status", "in_progress"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    protected_statuses = {
        runtime / f"status/wave_{wave}.json": f"protected-wave-{wave}\n"
        for wave in (3, 4, 5)
    }
    for path, text in protected_statuses.items():
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(launcher, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(launcher, "LOCK_PATH", runtime / "wave_supervisor.lock")
    monkeypatch.setattr(
        launcher,
        "STATUS_PATH",
        runtime / "status/wave_2_weak_strong_always_remote_sw_exclusion.json",
    )
    monkeypatch.setattr(
        launcher,
        "STANDARD_WAVE2_STATUS_PATH",
        runtime / "status/wave_2.json",
    )
    context = SimpleNamespace(
        runner=SimpleNamespace(LOCAL_CHILD_TOKEN_ENV="EXACT_CHILD_TOKEN"),
        runtime={"sha256": "r" * 64},
    )
    monkeypatch.setattr(launcher, "_wait_until_ready", lambda: {"run_ready": True})
    monkeypatch.setattr(
        launcher,
        "preflight",
        lambda **_kwargs: {"run_ready": True},
    )
    monkeypatch.setattr(launcher, "_fixed_context", lambda: context)
    monkeypatch.setattr(
        launcher,
        "_authenticate_sw_exclusion",
        lambda _context: {
            "sha256": "s" * 64,
            "remote_materialization_exclusion_authenticated": True,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_wave5_terminal_state",
        lambda _context: {
            "terminal": True,
            "status": "passed",
            "sha256": "w" * 64,
        },
    )
    decision = {
        "execution_id": launcher.TARGET_EXECUTION_ID,
        "extension_decision": "stop_at_k30",
        "sha256": "d" * 64,
    }
    closure_states = iter(
        (
            {"closed": False, "decision": None},
            {"closed": True, "decision": decision},
        )
    )
    monkeypatch.setattr(
        launcher,
        "_target_closure_state",
        lambda _context: next(closure_states),
    )
    monkeypatch.setattr(launcher, "_excluded_remote_paths_absent", lambda: True)
    monkeypatch.setattr(launcher, "_standard_wave2_status_absent", lambda: True)
    monkeypatch.setattr(launcher, "_local_scientific_overlap", lambda: [])
    observed: dict[str, object] = {}

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["environment"] = kwargs["env"]
            with launcher.LOCK_PATH.open("a+", encoding="utf-8") as contender:
                try:
                    fcntl.flock(
                        contender.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    observed["existing_wave_lock_held"] = True
                else:
                    observed["existing_wave_lock_held"] = False
                    fcntl.flock(contender.fileno(), fcntl.LOCK_UN)

        @staticmethod
        def wait(timeout=None):
            assert timeout is None
            return 0

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Child)

    result = launcher.supervise()

    assert result["status"] == "passed_exact_weak_strong_always_k30"
    assert result["completed_execution_ids"] == [launcher.TARGET_EXECUTION_ID]
    assert result["target_decision"] == decision
    assert observed["command"][-1] == launcher.TARGET_EXECUTION_ID
    assert "--run-wave" not in observed["command"]
    assert observed["environment"]["EXACT_CHILD_TOKEN"] == (
        f"{'r' * 64}:wave-2"
    )
    assert observed["existing_wave_lock_held"] is True
    assert not launcher.STANDARD_WAVE2_STATUS_PATH.exists()
    assert not any(
        path.exists() for path in launcher._cell_paths(
            launcher.EXCLUDED_REMOTE_EXECUTION_ID
        )
    )
    assert {
        path: path.read_text(encoding="utf-8")
        for path in protected_statuses
    } == protected_statuses
