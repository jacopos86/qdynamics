"""Contracts for the four-route Paper-II comparison diagnostics."""

from types import SimpleNamespace

from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
    _base_coordinate_id,
    _pair_match,
    build_terminal_compile_plan,
    deletion_gate_outcomes,
    integration_geometry_work,
    structural_churn,
)


def _point(labels, *, accepted, candidates=4, scored=3, reason="test"):
    return {
        "fixed_step": {"support_labels": list(labels)},
        "patch_decision": {
            "accepted": accepted,
            "candidate_count": candidates,
            "scored_count": scored,
            "reason": reason,
        },
    }


def test_integration_geometry_work_keeps_rhs_work_distinct_from_steps() -> None:
    run = {
        "plot_rows": [
            {
                "runtime_parameter_count": 3,
                "parameter_control_rhs_evaluation_count": 0,
                "integration_rhs_evaluation_count": 2,
            },
            {
                "runtime_parameter_count": 1,
                "parameter_control_rhs_evaluation_count": 3,
                "integration_rhs_evaluation_count": 0,
            },
        ],
        "summary": {"rhs_evaluation_count": 5},
    }

    result = integration_geometry_work([run])

    assert result == {
        "rhs_geometry_evaluation_count": 5,
        "logical_integration_geometry_work": 34,
    }


def test_structural_churn_finds_delete_readd_redelete_cycle() -> None:
    run = {
        "initial_state": {"runtime_coordinate_labels": ["a", "b"]},
        "trajectory": {
            "points": [
                _point(["b"], accepted=True, reason="delete"),
                _point(
                    ["b", "a::insr2c0o0::r0::x"],
                    accepted=True,
                    reason="readd",
                ),
                _point(["b"], accepted=True, reason="redelete"),
            ]
        },
    }

    result = structural_churn([run], short_cycle_window=2)

    assert result["added_coordinate_count"] == 1
    assert result["deleted_coordinate_count"] == 2
    assert "gross_churn" not in result
    assert "excess_churn" not in result
    assert result["short_delete_to_readd_count"] == 1
    assert result["short_add_to_redelete_count"] == 1
    assert result["support_changing_checkpoint_count"] == 3


def test_deletion_gate_outcomes_count_actual_post_delete_failures() -> None:
    run = {
        "trajectory": {
            "points": [
                {
                    "patch_decision": {
                        "accepted": True,
                        "patch_kind": "exchange",
                        "metadata": {
                            "attempt_count": 3,
                            "attempts": [
                                {
                                    "removed_runtime_indices": [0],
                                    "reason": "smoothness_eta_above_max",
                                },
                                {
                                    "removed_runtime_indices": [1],
                                    "reason": "compiled_word_identity_mismatch",
                                },
                                {
                                    "removed_runtime_indices": [2],
                                    "reason": "certified",
                                },
                            ],
                        },
                    }
                }
            ]
        }
    }

    result = deletion_gate_outcomes([run], applicable=True)

    assert result["attempt_record_complete"] is True
    assert result["deletion_bearing_finalist_attempt_count"] == 3
    assert result["post_delete_gate_failure_count"] == 1
    assert result["pre_certification_rejection_count"] == 1
    assert result["committed_deletion_patch_count"] == 1


def test_base_identity_removes_instance_and_occurrence_noise() -> None:
    assert (
        _base_coordinate_id("generator::ap_occ2::insr19c4o0::r0::xy")
        == "generator"
    )
    assert _base_coordinate_id("generator::ap_occurrence[3]") == "generator"


def test_pair_match_enforces_burden_and_every_component() -> None:
    policy = SimpleNamespace(
        response_scale_floor=1.0e-12,
        matched_accuracy_factor=2.0,
        matched_component_deadband=0.10,
    )
    left = {
        "comparison_eligible": True,
        "accuracy_burden": 0.8,
        "response_components": {"a": 0.2, "b": 0.5},
    }
    right = {
        "comparison_eligible": True,
        "accuracy_burden": 1.2,
        "response_components": {"a": 0.3, "b": 0.9},
    }
    passing = _pair_match(left, right, policy)
    assert passing["passes"] is True

    right["response_components"]["a"] = 0.41
    failing = _pair_match(left, right, policy)
    assert failing["passes"] is False
    assert failing["worst_component_ratio"] > 2.0


def test_terminal_compile_plan_uses_only_matched_individual_cells(tmp_path) -> None:
    routes = {
        route: {
            "run_sources": [
                {"path": f"/runs/{route}/run.json", "sha256": route * 2}
            ],
            "terminal_runtime_source_artifact": {
                "declared_path": "seed.json",
                "resolved_path": f"/runs/{route}/seed.json",
                "sha256": route * 3,
            },
        }
        for route in (
            "q_compiled",
            "gamma_compiled",
            "repaired_proxy",
            "old_proxy",
            "avqds",
        )
    }
    comparison = {
        "cells": [
            {
                "cell_id": "matched",
                "comparison_tier": "individual_target",
                "routes": routes,
            },
            {
                "cell_id": "guard",
                "comparison_tier": "comparison_guard",
                "routes": routes,
            },
        ]
    }

    plan = build_terminal_compile_plan(comparison, output_root=tmp_path)

    assert plan["job_count"] == 5
    assert {job["cell_id"] for job in plan["jobs"]} == {"matched"}
    assert all(job["output_cost_table_json"].startswith(str(tmp_path)) for job in plan["jobs"])


def test_terminal_compile_plan_can_include_complete_individual_target_cell(tmp_path) -> None:
    routes = {
        route: {
            "accuracy": {"qualifies_individually": True},
            "run_sources": [
                {"path": f"/runs/{route}/run.json", "sha256": route * 2}
            ],
            "terminal_runtime_source_artifact": {
                "declared_path": "seed.json",
                "resolved_path": f"/runs/{route}/seed.json",
                "sha256": route * 3,
            },
        }
        for route in (
            "q_compiled",
            "gamma_compiled",
            "repaired_proxy",
            "old_proxy",
            "avqds",
        )
    }
    comparison = {
        "cells": [
            {
                "cell_id": "complete_target",
                "comparison_tier": "unmatched",
                "all_routes_complete": True,
                "routes": routes,
            }
        ]
    }

    plan = build_terminal_compile_plan(
        comparison,
        output_root=tmp_path,
        include_complete_target=True,
    )

    assert plan["job_count"] == 5
    assert {job["cell_id"] for job in plan["jobs"]} == {"complete_target"}
