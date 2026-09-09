"""Local diagnostic planning and reporting; no campaign is launched here."""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelines.time_dynamics.diagnostics import selector_local_pilot as pilot


REPO = Path(__file__).resolve().parents[1]


def test_plan_has_24_matched_cells_and_does_not_grant_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "file_hash", lambda _path: pilot.SEED_SHA256)
    monkeypatch.setattr(pilot, "source_hashes", lambda _repo: {"source.py": "frozen"})
    plan = pilot.make_plan(REPO, tmp_path, budgets=(256, 1024))
    assert plan["execution_authorized"] is False
    cases = plan["cases"]
    assert len(cases) == len({case["case_id"] for case in cases}) == 24
    assert len({case["output_json"] for case in cases}) == 24
    assert {
        tuple(case["metadata"][axis] for axis in ("policy", "gain", "admission", "budget"))
        for case in cases
    } == set(product(pilot.POLICIES, pilot.GAINS, pilot.ADMISSIONS, (256, 1024)))
    varying = {
        "search_policy", "gain_evaluation", "max_insertion_batch_size",
        "max_deletion_batch_size", "max_joint_patch_evaluations",
        "max_compiled_patches_per_checkpoint", "output_json",
    }
    frozen = None
    for case in cases:
        args = case["metadata"]["effective_arguments"]
        shared = {key: value for key, value in args.items() if key not in varying}
        if frozen is None:
            frozen = shared
        assert shared == frozen
        assert args["replay_candidate_pool_mode"] == "diagnostic_replay_family_pool"
        assert args["require_complete_candidate_pool"] is True
        assert args["record_statevector"] is True
        assert args["artifact_json"] == str(REPO / pilot.SEED)
        assert args["t_final"] == 0.5 and args["num_times"] == 13
        assert args["compile_workers"] == 4 and args["support_patch_scoring_workers"] == 1
        assert args["max_insertion_rounds_per_checkpoint"] == 1
        assert args["exchange_insertion_frontier"] is None
        assert args["interaction_frontier_widths"] is None
        assert args["exchange_route_bundle"] == "paper_ii_exchange_q_compiled_v1"
        assert (args["max_insertion_batch_size"], args["max_deletion_batch_size"]) == (
            (1, 1) if case["metadata"]["admission"] == "singleton" else (4, None)
        )
        assert args["max_joint_patch_evaluations"] == args["max_compiled_patches_per_checkpoint"]
    assert not list(tmp_path.iterdir())


def test_calibration_plan_is_one_explicitly_authorized_batch_cell(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "file_hash", lambda _path: pilot.SEED_SHA256)
    monkeypatch.setattr(pilot, "source_hashes", lambda _repo: {})
    plan = pilot.make_plan(REPO, tmp_path, budgets=(256,), calibration=True, execution_authorized=True)
    assert plan["execution_authorized"] is True
    assert len(plan["cases"]) == 1
    metadata = plan["cases"][0]["metadata"]
    assert (metadata["policy"], metadata["gain"], metadata["max_A"], metadata["max_D"]) == (
        "bounded_exhaustive", "joint_batch", 4, None,
    )


def _search(d, a, count, *, permitted=1, refused=0, reason="joint_work_budget"):
    return {
        "stop_reason": reason,
        "coverage": {
            "evaluated_patches_by_cardinality": [
                {"deletion_size": d, "insertion_size": a, "count": count},
            ],
            "visited_deletion_sets_by_cardinality": [
                {"deletion_size": d, "permitted": permitted, "refused": refused},
            ],
        },
    }


def _call(index, searches, *, kind="exchange", accepted=True):
    return {
        "selector_call_index": index, "search_receipts": searches,
        "accepted": accepted, "patch_kind": kind, "reason": "recorded",
        "committed": {"removed_runtime_indices": [0, 1]},
        "work_guard": {"scored_count": index + 1},
        "compile_guard": {"scored_count": index + 2},
    }


def test_coverage_report_counts_every_recorded_round_and_complete_deletion_sets():
    calls = [
        _call(0, [_search(2, 2, 3, refused=2), _search(0, 1, 4)]),
        _call(1, [_search(2, 2, 5, reason="universe_exhausted")], kind="stay", accepted=False),
    ]
    run = {"trajectory": {"points": [
        {"time": 0.0, "patch_decision": {"patch_kind": "stay", "metadata": {
            "selection_round_work": calls,
            # A final-call copy must not be counted a second time.
            "selector_route": {"levels": [{"search": _search(2, 2, 5)}]},
        }}},
        {"time": 0.1, "patch_decision": None},
    ]}}
    coverage = pilot.search_coverage(run)
    assert coverage["scope"] == "all_recorded_selector_calls_and_passes"
    assert coverage["evaluated_patches"] == [
        {"D": 0, "A": 1, "count": 4}, {"D": 2, "A": 2, "count": 8},
    ]
    assert coverage["stop_reasons"] == {"joint_work_budget": 2, "universe_exhausted": 1}
    permissions = {(row["D"], row["outcome"]): row["count"] for row in coverage["visited_deletion_sets"]}
    assert permissions == {(0, "permitted"): 1, (0, "refused"): 0, (2, "permitted"): 2, (2, "refused"): 2}
    assert [row["kind"] for row in coverage["decisions"]] == ["exchange", "stay"]
    assert [row["call"] for row in coverage["guards"]] == [0, 1]


def test_round_coverage_ledger_matches_actual_small_trajectory_calls(monkeypatch):
    from pipelines.time_dynamics.ap_mclachlan import adaptive_trajectory as adaptive
    from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import run_append_ap_mclachlan_from_runtime_input
    from test.test_ap_mclachlan_exchange_integration import _config
    from test.test_ap_mclachlan_route_parity import _runtime_input

    captured = []
    original = adaptive.select_generalized_exchange_patch

    def observe(**kwargs):
        selected, payload = original(**kwargs)
        captured.append((kwargs["time"], deepcopy(payload)))
        return selected, payload

    monkeypatch.setattr(adaptive, "select_generalized_exchange_patch", observe)
    payload = run_append_ap_mclachlan_from_runtime_input(
        _runtime_input(theta=(0.05, -0.04)), times=(0.0, 0.02, 0.04), integrator_method="rk4",
        pinv_rcond=1e-10, ridge_lambda=1e-7, controller_config=adaptive.AppendControllerConfig(),
        support_patch_config=_config(
            exchange_route_bundle="paper_ii_exchange_q_compiled_v1", search_policy="greedy",
            gain_evaluation="joint_batch", insertion_gate_mode="mclachlan_l2", insertion_l2_cut=0.0,
            max_insertion_batch_size=1, max_deletion_batch_size=1,
            max_insertion_rounds_per_checkpoint=1, max_joint_patch_evaluations=256,
            max_compiled_patches_per_checkpoint=256, compile_workers=1,
        ),
        solve_repair_config=SolveRepairConfig.minimal_profile(), code_revision="round_coverage_test",
    )
    rows = [
        (point["time"], call)
        for point in payload["trajectory"]["points"]
        for call in (point.get("patch_decision") or {}).get("metadata", {}).get("selection_round_work", [])
    ]
    assert len(rows) == len(captured) > 0
    assert any(row["selector_call_index"] == 1 for _time, row in rows)
    for (time, call), (actual_time, actual) in zip(rows, captured):
        assert time == actual_time
        route = actual["selector_route"]
        assert call["search_receipts"] == [level["search"] for level in route["levels"] if level.get("search")]
        assert call["work_guard"] == actual["work_guard"]
        assert call["compile_guard"] == route["compile_guard"]
        assert call["committed"] == actual.get("committed")
        if actual.get("committed"):
            assert call["committed"]["score"] == actual["attempts"][-1]["score"]
            assert call["committed"]["cost_factor_k"] == actual["attempts"][-1]["cost_factor_k"]
            assert "route_score" not in actual["attempts"][-1]
        for receipt in call["search_receipts"]:
            assert sum(row["count"] for row in receipt["coverage"]["evaluated_patches_by_cardinality"]) == receipt["evaluated_patch_count"]
    summary = pilot.search_coverage(payload)
    assert len(summary["decisions"]) == len(rows)
    for reported, (_time, actual) in zip(summary["decisions"], captured):
        assert reported["committed"] == actual.get("committed")


def test_exact_reference_and_terminal_cost_are_acquired_only_by_successful_run_reporting(monkeypatch, tmp_path):
    from pipelines.time_dynamics.diagnostics import ap_terminal_qiskit_cost as terminal
    from pipelines.time_dynamics.diagnostics import paper_ii_four_route_comparison as comparison
    from pipelines.time_dynamics.diagnostics import physical_errors as physical

    exact_calls, cost_calls = [], []

    def exact(**kwargs):
        exact_calls.append(kwargs)
        return SimpleNamespace(to_json_dict=lambda: {"energy_error": 0.0})

    def cost(**kwargs):
        cost_calls.append(kwargs)
        return {"rows": [{"n2q": 2}]}

    monkeypatch.setattr(physical, "physical_errors_from_artifacts", exact)
    monkeypatch.setattr(terminal, "build_ap_terminal_cost_table", cost)
    monkeypatch.setattr(comparison, "complete_estimator_work", lambda _runs, **_kw: {"status": "exact"})
    cases = [pilot.pilot_case(REPO, tmp_path / "runs", policy="greedy", gain="joint_batch", admission="singleton", budget=b) for b in (16, 32)]
    plan = {"cases": cases, "output_root": str(tmp_path)}
    pilot.write_json(tmp_path / "plan.json", plan)
    assert not exact_calls and not cost_calls
    for index, case in enumerate(cases):
        directory = Path(case["output_json"]).parent
        pilot.write_json(directory / "execution.json", {"status": "completed" if index == 0 else "timed_out"})
        pilot.write_json(directory / "measurement.json", {"ok": index == 0})
        pilot.write_json(Path(case["output_json"]), {"trajectory": {"points": []}, "plot_rows": [], "summary": {}})
    report = pilot.report_plan(plan)
    assert len(exact_calls) == len(cost_calls) == 1
    assert exact_calls[0]["run_path"] == Path(cases[0]["output_json"])
    assert "errors" in report["rows"][0]
    assert "errors" not in report["rows"][1]
