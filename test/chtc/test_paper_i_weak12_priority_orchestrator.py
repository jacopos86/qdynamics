from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_paper_i_weak12_priority_then_matched_unique6_20260815.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("paper_i_priority_orchestrator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_memory_wait_requires_high_water_mark(monkeypatch):
    module = _module()
    capacities = iter(
        [
            {"available_memory_bytes": 3 * 1024**3},
            {"available_memory_bytes": 7 * 1024**3},
        ]
    )
    statuses = []

    class Runner:
        DEFAULT_RUNTIME_DIR = Path("/unused")

        @staticmethod
        def _capacity(_path):
            return next(capacities)

    monkeypatch.setattr(module, "_write_status", lambda *_args, **kwargs: statuses.append(kwargs))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module._wait_for_memory_headroom(
        Runner,
        phase="weak_holstein_ra_plateau_three",
        execution_id="cell-a",
        retry_count=0,
        last_failure=None,
    )

    assert result["available_memory_bytes"] == 7 * 1024**3
    assert statuses[0]["current_execution_id"] == "cell-a"
    assert statuses[0]["minimum_retry_available_memory_bytes"] == 7 * 1024**3


def test_only_available_memory_guard_is_retryable():
    module = _module()

    class MatchedError(RuntimeError):
        pass

    class Runner:
        MatchedSingleton12Error = MatchedError

    assert module._retryable_memory_guard(
        Runner,
        MatchedError("Runtime guard stopped cell-a: available_memory_floor_breached."),
    )
    assert not module._retryable_memory_guard(
        Runner,
        MatchedError("Runtime guard stopped cell-a: rss_limit_exceeded."),
    )
    assert not module._retryable_memory_guard(Runner, RuntimeError("available_memory_floor_breached"))


def test_guarded_child_retries_exact_memory_stop(monkeypatch):
    module = _module()
    waits = []
    starts = []

    class MatchedError(RuntimeError):
        pass

    class Runner:
        MatchedSingleton12Error = MatchedError

        @staticmethod
        def _run_guarded_child(**_kwargs):
            starts.append("start")
            if len(starts) == 1:
                raise MatchedError(
                    "Runtime guard stopped cell-a: available_memory_floor_breached."
                )
            return {"status": "passed"}

    monkeypatch.setattr(
        module,
        "_wait_for_memory_headroom",
        lambda *_args, **kwargs: waits.append(kwargs) or {"available_memory_bytes": 8 * 1024**3},
    )
    monkeypatch.setattr(module, "_write_status", lambda *_args, **_kwargs: None)

    result = module._run_guarded_child_with_memory_retry(
        Runner,
        cell={"execution_id": "cell-a"},
        activation={},
        handoff={},
        phase="weak_holstein_ra_plateau_three",
        completed_execution_ids=["cell-0"],
    )

    assert result == {"status": "passed"}
    assert len(starts) == 2
    assert [row["retry_count"] for row in waits] == [0, 1]
    assert "available_memory_floor_breached" in waits[1]["last_failure"]
