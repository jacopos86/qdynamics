from __future__ import annotations

import importlib.util
import fcntl
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_paper_i_native_phase0_plateau_eight_arm_k5_20260816.py"
)


def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("paper_i_native_phase0_eight", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_eight_cell_order_is_exact() -> None:
    runner = load_runner()
    observed = [
        (cell.placement, cell.score, cell.cardinality)
        for cell in runner.CELL_SPECS
    ]
    assert observed == [
        ("generator_first", "gradient", "fixed24"),
        ("position_aware", "gradient", "fixed24"),
        ("generator_first", "gradient", "adaptive"),
        ("position_aware", "gradient", "adaptive"),
        ("generator_first", "proxy", "fixed24"),
        ("position_aware", "proxy", "fixed24"),
        ("generator_first", "proxy", "adaptive"),
        ("position_aware", "proxy", "adaptive"),
    ]
    assert [cell.ordinal for cell in runner.CELL_SPECS] == list(range(1, 9))
    assert {cell.insertion_policy for cell in runner.CELL_SPECS} == {
        "plateau_commutation"
    }


def test_capacity_wait_is_bounded_and_reports_blocked() -> None:
    runner = load_runner()
    ticks = iter((0.0, 0.0, 301.0))
    observed = runner.wait_for_capacity(
        maximum_wait_seconds=300.0,
        clock=lambda: next(ticks),
        sleeper=lambda _seconds: None,
        memory_supplier=lambda: runner.LAUNCH_AVAILABLE_MEMORY_BYTES - 1,
        disk_supplier=lambda: runner.LAUNCH_FREE_DISK_BYTES - 1,
    )
    assert observed["status"] == "blocked_capacity"
    assert observed["launch_ready"] is False


def test_plan_and_authorization_are_separate_no_replace_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    authority = tmp_path / "authority"
    monkeypatch.setattr(runner, "AUTHORITY_DIR", authority)
    monkeypatch.setattr(runner, "PLAN_PATH", authority / "plan.json")
    monkeypatch.setattr(runner, "AUTHORIZATION_PATH", authority / "authorization.json")
    plan = runner.digested(
        {
            "schema": runner.PLAN_SCHEMA,
            "campaign_id": runner.CAMPAIGN_ID,
            "runner": {"sha256": "a" * 64},
            "source_implementation_inventory_sha256": "b" * 64,
            "execution_authorized": False,
        }
    )
    monkeypatch.setattr(runner, "build_plan", lambda: plan)
    assert runner.prepare_plan() == plan
    assert runner.PLAN_PATH.is_file()
    assert not runner.AUTHORIZATION_PATH.exists()

    monkeypatch.setattr(
        runner,
        "validate_plan",
        lambda *, recompute_protocols: plan,
    )
    authorization = runner.authorize()
    assert authorization["execution_authorized"] is True
    assert authorization["plan_sha256"] == plan["sha256"]
    with pytest.raises(runner.RunnerError, match="already exists"):
        runner.authorize()


def test_factorial_report_marks_dormant_placement_not_activated() -> None:
    runner = load_runner()
    rows = []
    for cell in runner.CELL_SPECS:
        rows.append(
            {
                "placement": cell.placement,
                "score": cell.score,
                "cardinality": cell.cardinality,
                "controller_round": 5,
                "plateau_state": "closed",
                "energy": float(cell.ordinal),
                "absolute_delta_e": float(cell.ordinal) / 10.0,
                "s_alg": 100 + cell.ordinal,
                "n2q": 10 + cell.ordinal,
                "d2q": 20 + cell.ordinal,
                "dc": 30 + cell.ordinal,
            }
        )
    effects = runner.factorial_effects(rows)
    assert effects["placement"]["status"] == "not_activated"
    assert effects["placement:score"]["status"] == "not_activated"
    assert effects["score"]["status"] == "estimated"
    assert effects["cardinality"]["status"] == "estimated"


def test_digested_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    runner = load_runner()
    path = tmp_path / "receipt.json"
    payload = runner.digested({"schema": "example_v1", "value": 1})
    runner.write_json_exclusive(path, payload)
    assert runner.load_digested(path, schema="example_v1") == payload
    path.write_text(path.read_text().replace('"value": 1', '"value": 2'))
    with pytest.raises(runner.RunnerError, match="Invalid digested artifact"):
        runner.load_digested(path, schema="example_v1")


def test_phase_i_and_phase_ii_maxima_are_validated_separately() -> None:
    runner = load_runner()
    runner._require_phase_maxima(phase_i_retained=24, phase_ii_retained=12)
    with pytest.raises(runner.RunnerError, match="Phase-I"):
        runner._require_phase_maxima(phase_i_retained=25, phase_ii_retained=12)
    with pytest.raises(runner.RunnerError, match="Phase-II"):
        runner._require_phase_maxima(phase_i_retained=24, phase_ii_retained=13)


def test_terminal_matrix_recomputes_all_comparison_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    comparison_path = tmp_path / "comparison.json"
    csv_path = tmp_path / "comparison.csv"
    markdown_path = tmp_path / "comparison.md"
    terminal_path = tmp_path / "terminal.json"
    monkeypatch.setattr(runner, "REPORT_JSON", comparison_path)
    monkeypatch.setattr(runner, "REPORT_CSV", csv_path)
    monkeypatch.setattr(runner, "REPORT_MD", markdown_path)
    monkeypatch.setattr(runner, "TERMINAL_PATH", terminal_path)
    plan = {
        "sha256": "a" * 64,
        "source_implementation_inventory_sha256": "b" * 64,
    }
    authorization = {"sha256": "c" * 64}
    closed = [
        (cell, {}, {}, f"{cell.ordinal:064x}", f"{cell.ordinal + 8:064x}")
        for cell in runner.CELL_SPECS
    ]
    monkeypatch.setattr(
        runner,
        "load_closed_cell",
        lambda cell, **_kwargs: closed[cell.ordinal - 1],
    )
    comparison = runner.digested(
        {"schema": runner.REPORT_SCHEMA, "status": "passed_eight_k5", "rows": []}
    )
    csv_text = "arm,k\n"
    markdown = "# comparison\n"
    monkeypatch.setattr(
        runner,
        "build_comparison",
        lambda _closed: (comparison, csv_text, markdown),
    )
    runner.write_json_exclusive(comparison_path, comparison)
    runner.write_text_exclusive(csv_path, csv_text)
    runner.write_text_exclusive(markdown_path, markdown)
    terminal = runner.digested(
        {
            "schema": runner.TERMINAL_SCHEMA,
            "status": "passed_eight_k5",
            "campaign_id": runner.CAMPAIGN_ID,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "fixed_serial_order": [cell.execution_id for cell in runner.CELL_SPECS],
            "comparison_sha256": comparison["sha256"],
            "comparison_csv_sha256": runner.sha256_file(csv_path),
            "comparison_markdown_sha256": runner.sha256_file(markdown_path),
            "controller_rounds_completed_by_cell": {
                cell.execution_id: runner.TARGET_HORIZON
                for cell in runner.CELL_SPECS
            },
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    runner.write_json_exclusive(terminal_path, terminal)
    assert runner.validate_terminal_matrix(
        plan=plan,
        authorization=authorization,
    ) == terminal
    csv_path.write_text("tampered\n")
    with pytest.raises(runner.RunnerError, match="CSV failed recomputation"):
        runner.validate_terminal_matrix(plan=plan, authorization=authorization)


def test_campaign_lock_and_partial_output_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(runner, "LOCK_PATH", tmp_path / "campaign.lock")
    monkeypatch.setattr(runner, "TERMINAL_PATH", tmp_path / "terminal.json")
    monkeypatch.setattr(
        runner,
        "validate_authority",
        lambda **_kwargs: ({"sha256": "a" * 64}, {"sha256": "b" * 64}),
    )
    descriptor = (tmp_path / "campaign.lock").open("w")
    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(runner.RunnerError, match="lock is already held"):
        runner.run_campaign()
    descriptor.close()

    partial = tmp_path / "partial"
    partial.mkdir()
    monkeypatch.setattr(
        runner,
        "cell_paths",
        lambda _cell: (partial, tmp_path / "stage", tmp_path / "worker", tmp_path / "guard"),
    )
    with pytest.raises(runner.RunnerError, match="partial output"):
        runner.run_campaign()


def test_plan_rejects_live_source_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    plan_path = tmp_path / "plan.json"
    monkeypatch.setattr(runner, "PLAN_PATH", plan_path)
    plan = runner.digested(
        {
            "schema": runner.PLAN_SCHEMA,
            "campaign_id": runner.CAMPAIGN_ID,
            "cells": [runner.asdict(cell) for cell in runner.CELL_SPECS],
            "fixed_serial_order": [cell.execution_id for cell in runner.CELL_SPECS],
            "source_implementation_inventory_sha256": "a" * 64,
            "runner": runner.file_binding(runner.RUNNER_PATH),
            "maximum_concurrency": 1,
            "execution_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    runner.write_json_exclusive(plan_path, plan)
    monkeypatch.setattr(
        runner,
        "semantic_closure_source_implementation_inventory",
        lambda: {"sha256": "b" * 64},
    )
    with pytest.raises(runner.RunnerError, match="plan drifted"):
        runner.validate_plan(recompute_protocols=False)
