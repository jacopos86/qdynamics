from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "macro_gradient_phase0_macro_phase123_proxy_no_lanes_local_20260810_v1/"
    "parallel_continuation.py"
)


def _load_supervisor() -> Any:
    spec = importlib.util.spec_from_file_location(
        "paper_i_macro_phase0_parallel_continuation_for_test",
        SUPERVISOR_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


supervisor = _load_supervisor()


def test_contiguous_prefix_stops_before_out_of_order_closures() -> None:
    execution_ids = ("cell-a", "cell-b", "cell-c", "cell-d")

    assert supervisor._contiguous_prefix(
        execution_ids,
        {"cell-a", "cell-c", "cell-d"},
    ) == ("cell-a",)
    assert supervisor._contiguous_prefix(
        execution_ids,
        {"cell-a", "cell-b", "cell-d"},
    ) == ("cell-a", "cell-b")
    assert supervisor._contiguous_prefix(
        execution_ids,
        set(execution_ids),
    ) == execution_ids


def test_live_run_cell_pids_parses_only_authorized_cell_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_table = "\n".join(
        (
            "  101 /usr/bin/python3 -B /work/local_runner.py run-cell "
            "--job /sealed/jobs/cell-a.json",
            "202 /usr/bin/python3 -B /work/local_runner.py run-cell "
            "--authorization /activation/cell-b.json",
            "303 /usr/bin/python3 -B /work/local_runner.py run-serial "
            "--job /sealed/jobs/cell-a.json",
            "404 /usr/bin/python3 unrelated.py /sealed/jobs/cell-b.json",
            "505 /usr/bin/python3 -B /work/local_runner.py run-cell "
            "--job /sealed/jobs/not-authorized.json",
        )
    )

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        assert command == ["ps", "-axo", "pid=,command=", "-ww"]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return SimpleNamespace(stdout=process_table)

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)

    assert supervisor._live_run_cell_pids(("cell-a", "cell-b")) == {
        "cell-a": 101,
        "cell-b": 202,
    }


def test_live_run_cell_pids_rejects_duplicate_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_table = "\n".join(
        (
            "101 /usr/bin/python3 /work/local_runner.py run-cell "
            "--job /sealed/jobs/cell-a.json",
            "202 /usr/bin/python3 /work/local_runner.py run-cell "
            "--authorization /activation/cell-a.json",
        )
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=process_table,
        ),
    )

    with pytest.raises(
        supervisor.SupervisorError,
        match=r"Duplicate live cell processes: .*cell-a.*101.*202",
    ):
        supervisor._live_run_cell_pids(("cell-a", "cell-b"))


def test_publish_serial_status_keeps_prefix_and_out_of_order_publications_distinct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    execution_ids = (
        "cell-a",
        "cell-b",
        "cell-c",
        "cell-d",
        "cell-e",
        "cell-f",
    )
    writes: list[tuple[Path, dict[str, Any]]] = []

    class Worker:
        @staticmethod
        def digested(value: dict[str, Any]) -> dict[str, Any]:
            return {**value, "sha256": "status-digest"}

    runner = SimpleNamespace(
        LOCAL_STATUS_SCHEMA="local-status-v1",
        _write_json_atomic=lambda _worker, path, payload: writes.append(
            (path, dict(payload))
        ),
    )
    monkeypatch.setattr(supervisor, "RUNTIME_DIR", tmp_path)

    payload = supervisor._publish_serial_status(
        runner,
        Worker(),
        serial_manifest_sha256="serial-manifest-digest",
        execution_ids=execution_ids,
        closed={"cell-a", "cell-c"},
        live={"cell-b": 202, "cell-c": 303, "cell-d": 404},
        status="running",
    )

    assert writes == [(tmp_path / "serial_status.json", payload)]
    assert payload["completed_execution_ids"] == ["cell-a"]
    assert payload["published_completed_execution_ids"] == [
        "cell-a",
        "cell-c",
    ]
    assert payload["running_execution_ids"] == ["cell-b", "cell-d"]
    assert payload["running_pids"] == {"cell-b": 202, "cell-d": 404}
    assert payload["current_execution_id"] == "cell-b"
    assert payload["remaining_execution_ids"] == ["cell-e", "cell-f"]
    assert payload["execution_mode"] == "local_parallel_two_regimes_v1"
    assert payload["maximum_concurrency"] == 2


@pytest.mark.parametrize(
    "collision_kind",
    ("output_dir", "receipt", "stdout", "stderr"),
)
def test_launch_cell_refuses_every_prior_attempt_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    collision_kind: str,
) -> None:
    execution_id = "cell-a"
    collisions = {
        "output_dir": tmp_path / "runs" / execution_id,
        "receipt": tmp_path / "worker_receipts" / f"{execution_id}.json",
        "stdout": tmp_path / "logs" / f"{execution_id}.out",
        "stderr": tmp_path / "logs" / f"{execution_id}.err",
    }
    collision = collisions[collision_kind]
    collision.parent.mkdir(parents=True, exist_ok=True)
    if collision_kind == "output_dir":
        collision.mkdir()
    else:
        collision.write_text("prior attempt\n", encoding="utf-8")

    monkeypatch.setattr(supervisor, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "a colliding attempt must not launch a process"
        ),
    )

    with pytest.raises(
        supervisor.SupervisorError,
        match=r"Refusing to overwrite a prior cell attempt",
    ):
        supervisor._launch_cell(
            SimpleNamespace(PACKAGE_DIR=tmp_path / "package"),
            {"execution_id": execution_id, "job_path": "jobs/cell-a.json"},
        )
