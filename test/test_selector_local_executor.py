"""Subprocess ownership tests; no scientific runner is launched."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import pytest

from pipelines.time_dynamics.diagnostics import selector_local_executor as executor


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    for relative in ("pipelines", "pipelines/time_dynamics", "pipelines/time_dynamics/diagnostics"):
        folder = repo / relative
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text("")
    (repo / "pipelines/shell").mkdir()
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "pipelines/shell/ram_guard.py",
        repo / "pipelines/shell/ram_guard.py",
    )
    (repo / "pipelines/time_dynamics/diagnostics/ap_runtime_benchmark.py").write_text('''
import argparse, json, os, subprocess, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--single-case-json")
p.add_argument("--output-json")
a = p.parse_args()
case = json.loads(Path(a.single_case_json).read_text())
mode = case["runner_argv"][0]
print("fake stdout", flush=True)
print("fake stderr", file=sys.stderr, flush=True)
if mode == "sleep":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    Path(case["metadata"]["pids"]).write_text(json.dumps([os.getpid(), child.pid]))
    time.sleep(120)
elif mode == "crash":
    raise SystemExit(7)
elif mode == "malformed":
    Path(a.output_json).write_text("not json")
else:
    time.sleep(0.15)
    if case.get("output_json"):
        Path(case["output_json"]).write_text("{}")
    Path(a.output_json).write_text(json.dumps({"ok": True, "env": dict(os.environ)}))
''')
    return repo


def _case(tmp_path, mode="success"):
    return {
        "case_id": mode, "runner_argv": [mode], "blas_threads": 1,
        "metadata": {"pids": str(tmp_path / "owned_pids.json")},
    }


def _run(tmp_path, fake_repo, case=None, **kwargs):
    return executor.execute_case(
        case_dir=tmp_path / "case", case=case or _case(tmp_path),
        python=sys.executable, repo_root=fake_repo, timeout_seconds=kwargs.pop("timeout_seconds", 5),
        limit_mb=128, **kwargs,
    )


def _assert_stopped(pids):
    def live(pid):
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    deadline = time.monotonic() + 3
    while any(live(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not [pid for pid in pids if live(pid)]


def test_success_records_case_environment_logs_and_refuses_overwrite(tmp_path, fake_repo):
    case = _case(tmp_path)
    receipt = _run(tmp_path, fake_repo, case)
    directory = tmp_path / "case"
    assert receipt["status"] == "completed" and receipt["ok"]
    assert receipt["returncode"] == 0 and receipt["wall_seconds"] > 0
    assert receipt["guard_pid"] != receipt["child_pgid"]
    assert receipt["child_pid"] == receipt["child_pgid"]
    assert json.loads((directory / "case.json").read_text()) == case
    assert json.loads((directory / "execution.json").read_text()) == receipt
    measured = json.loads((directory / "measurement.json").read_text())
    assert all(measured["env"][name] == "1" for name in executor.BLAS_THREAD_ENV_VARS)
    assert "fake stdout" in (directory / "stdout.log").read_text()
    assert "fake stderr" in (directory / "stderr.log").read_text()
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        _run(tmp_path, fake_repo, case)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("mode", ("crash", "malformed"))
def test_failed_case_still_has_receipt(tmp_path, fake_repo, mode):
    receipt = _run(tmp_path, fake_repo, _case(tmp_path, mode))
    assert receipt["status"] == "failed" and not receipt["ok"]
    assert receipt["returncode"] == (7 if mode == "crash" else 0)
    assert json.loads((tmp_path / "case/execution.json").read_text()) == receipt


def test_launch_failure_still_has_receipt(tmp_path, fake_repo):
    receipt = executor.execute_case(
        case_dir=tmp_path / "case", case=_case(tmp_path),
        python=str(tmp_path / "missing-python"), repo_root=fake_repo,
    )
    assert receipt["status"] == "failed" and receipt["returncode"] is None
    assert "FileNotFoundError" in receipt["error"]
    assert json.loads((tmp_path / "case/execution.json").read_text()) == receipt


def test_timeout_stops_guard_child_and_grandchild_only(tmp_path, fake_repo):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        receipt = _run(tmp_path, fake_repo, _case(tmp_path, "sleep"), timeout_seconds=0.6)
        assert receipt["status"] == "timed_out" and not receipt["ok"]
        assert receipt["cleanup"]["guard_sigint_sent"]
        pids = json.loads((tmp_path / "owned_pids.json").read_text())
        assert receipt["child_pid"] == pids[0]
        _assert_stopped([receipt["guard_pid"], *pids])
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()


def test_parent_interrupt_cleans_owned_work_and_records_before_reraise(tmp_path, fake_repo, monkeypatch):
    original = executor.time.sleep
    interrupted = False

    def interrupt_when_child_ready(seconds):
        nonlocal interrupted
        if not interrupted and (tmp_path / "owned_pids.json").exists():
            interrupted = True
            raise KeyboardInterrupt
        original(seconds)

    monkeypatch.setattr(executor.time, "sleep", interrupt_when_child_ready)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, fake_repo, _case(tmp_path, "sleep"))
    receipt = json.loads((tmp_path / "case/execution.json").read_text())
    assert receipt["status"] == "interrupted"
    _assert_stopped([receipt["guard_pid"], *json.loads((tmp_path / "owned_pids.json").read_text())])


def test_unresponsive_guard_uses_only_discovered_owned_group(tmp_path, monkeypatch):
    fake_guard = tmp_path / "unresponsive_guard.py"
    fake_guard.write_text('''
import json, os, signal, subprocess, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
open(sys.argv[1], "w").write(json.dumps([os.getpid(), child.pid]))
time.sleep(120)
''')
    monkeypatch.setattr(executor, "_STOP_GRACE_SECONDS", 0.2)
    with (tmp_path / "stdout").open("wb") as stdout, (tmp_path / "stderr").open("wb") as stderr:
        receipt = executor._execute_guarded(
            [sys.executable, str(fake_guard), str(tmp_path / "pids")],
            repo_root=tmp_path, env=os.environ, stdout=stdout, stderr=stderr,
            timeout_seconds=0.5,
        )
    assert receipt["status"] == "timed_out"
    assert receipt["cleanup"]["group_sigkill_sent"]
    assert receipt["cleanup"]["guard_sigkill_sent"]
    _assert_stopped(json.loads((tmp_path / "pids").read_text()))
