from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_paper_i_ra_canary_then_all6_adaptive_k50_20260816.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("paper_i_ra_direct_chain", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_runners(*, inventory: str, canary_rc: int = 0, overnight_rc: int = 0):
    events: list[str] = []
    canary_plan = {
        "sha256": "1" * 64,
        "source_implementation_inventory_sha256": inventory,
    }
    overnight_plan = {
        "sha256": "2" * 64,
        "source_implementation_inventory_sha256": inventory,
    }
    canary_terminal = {
        "sha256": "3" * 64,
        "source_implementation_inventory_sha256": inventory,
    }
    overnight_terminal = {
        "sha256": "4" * 64,
        "source_implementation_inventory_sha256": inventory,
    }
    canary = SimpleNamespace(
        CAMPAIGN_ID="canary",
        validate_authority=lambda **_kwargs: (canary_plan, {"sha256": "5" * 64}),
        run_campaign=lambda: events.append("run_canary") or canary_rc,
        validate_terminal_matrix=lambda: events.append("validate_canary")
        or canary_terminal,
    )
    overnight = SimpleNamespace(
        CAMPAIGN_ID="overnight",
        AUTHORIZATION_PATH=Path("/does/not/exist"),
        validate_conditional_authority=lambda **_kwargs: (
            overnight_plan,
            {"sha256": "6" * 64},
        ),
        authorize_after_canary=lambda **_kwargs: events.append("authorize_overnight")
        or {"sha256": "7" * 64},
        run_campaign=lambda: events.append("run_overnight") or overnight_rc,
        validate_terminal_matrix=lambda: events.append("validate_overnight")
        or overnight_terminal,
    )
    return canary, overnight, events


def test_chain_does_not_authorize_or_start_overnight_when_canary_is_incomplete(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    canary, overnight, events = _fake_runners(inventory="a" * 64, canary_rc=2)
    monkeypatch.setattr(runner, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(runner, "LOCK_PATH", tmp_path / "chain.lock")
    monkeypatch.setattr(runner, "TERMINAL_PATH", tmp_path / "terminal.json")

    assert runner.run_chain(canary_runner=canary, overnight_runner=overnight) == 2
    assert events == ["run_canary"]


def test_chain_is_non_idling_and_orders_validation_authorization_and_overnight(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    canary, overnight, events = _fake_runners(inventory="b" * 64)
    monkeypatch.setattr(runner, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(runner, "LOCK_PATH", tmp_path / "chain.lock")
    monkeypatch.setattr(runner, "TERMINAL_PATH", tmp_path / "terminal.json")

    assert runner.run_chain(canary_runner=canary, overnight_runner=overnight) == 0
    assert events == [
        "run_canary",
        "validate_canary",
        "authorize_overnight",
        "run_overnight",
        "validate_overnight",
        "validate_canary",
        "validate_overnight",
    ]
    first_terminal = runner.validate_terminal_chain(
        canary_runner=canary, overnight_runner=overnight
    )
    assert runner.run_chain(canary_runner=canary, overnight_runner=overnight) == 0
    assert runner.validate_terminal_chain(
        canary_runner=canary, overnight_runner=overnight
    ) == first_terminal


def test_chain_rejects_different_source_inventories_before_any_run(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    canary, overnight, events = _fake_runners(inventory="c" * 64)
    original = overnight.validate_conditional_authority
    overnight.validate_conditional_authority = lambda **kwargs: (
        {
            **original(**kwargs)[0],
            "source_implementation_inventory_sha256": "d" * 64,
        },
        original(**kwargs)[1],
    )
    monkeypatch.setattr(runner, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(runner, "LOCK_PATH", tmp_path / "chain.lock")
    monkeypatch.setattr(runner, "TERMINAL_PATH", tmp_path / "terminal.json")

    with pytest.raises(runner.RunnerError, match="same source inventory"):
        runner.run_chain(canary_runner=canary, overnight_runner=overnight)
    assert events == []
