"""Contracts for the four-route Paper-II comparison diagnostics."""

from types import SimpleNamespace

from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
    _base_coordinate_id,
    _pair_match,
    build_terminal_compile_plan,
    complete_estimator_work,
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


def test_complete_estimator_work_includes_apm_candidates_and_finalists() -> None:
    run = {
        "initial_state": {"runtime_coordinate_labels": ["a", "b"]},
        "plot_rows": [
            {
                "runtime_parameter_count": 3,
                "parameter_control_rhs_evaluation_count": 0,
                "integration_rhs_evaluation_count": 1,
            }
        ],
        "summary": {"rhs_evaluation_count": 1},
        "trajectory": {
            "points": [
                {
                    "fixed_step": {"support_labels": ["a", "b", "c"]},
                    "patch_decision": {
                        "accepted": True,
                        "patch_kind": "append",
                        "metadata": {
                            "insertion_rounds": 1,
                            "insertions_enabled": True,
                            "work_guard": {
                                "admitted_families": ["singleton_d0"],
                                "scored_count": 3,
                            },
                            "committed": {
                                "removed_runtime_indices": [],
                                "inserted_selection": [["c", 2]],
                            },
                            "attempt_count": 1,
                            "attempts": [
                                {
                                    "removed_runtime_indices": [],
                                    "inserted_selection": [["c", 2]],
                                    "reason": "certified",
                                }
                            ],
                        },
                    },
                }
            ]
        },
    }

    result = complete_estimator_work([run], route="q_compiled")

    assert result["logical_integration_geometry_work"] == 11
    assert result["logical_checkpoint_base_geometry_work"] == 7
    assert result["logical_candidate_acquisition_work"] == 9
    assert result["logical_certification_geometry_work"] == 11
    assert result["logical_total_estimator_work"] == 38
    assert result["candidate_acquisition_status"] == "exact"
    assert result["certification_geometry_status"] == "lower_bound"
    assert result["physical_shots"] is None


def test_complete_estimator_work_counts_avqds_candidate_geometry_as_bound() -> None:
    run = {
        "initial_state": {"runtime_coordinate_labels": ["a", "b"]},
        "plot_rows": [
            {
                "runtime_parameter_count": 3,
                "parameter_control_rhs_evaluation_count": 0,
                "integration_rhs_evaluation_count": 1,
            }
        ],
        "summary": {"rhs_evaluation_count": 1},
        "trajectory": {
            "points": [
                {
                    "fixed_step": {"support_labels": ["a", "b", "c"]},
                    "patch_decision": {
                        "accepted": True,
                        "metadata": {
                            "appended_atom_ids": ["c"],
                            "candidates_scored": 4,
                        },
                    },
                }
            ]
        },
    }

    result = complete_estimator_work([run], route="avqds")

    assert result["candidate_full_geometry_evaluation_count"] == 4
    assert result["logical_candidate_acquisition_work"] == 44
    assert result["candidate_acquisition_status"] == "lower_bound"
    assert result["logical_certification_geometry_work"] is None
    assert result["certification_geometry_status"] == "not_applicable"
    assert result["logical_total_estimator_work"] == 62
    assert result["logical_total_estimator_work_status"] == "lower_bound"


def test_complete_estimator_work_consumes_exact_avqds_round_ledger() -> None:
    round_work = {
        "round_index": 0,
        "base_runtime_parameter_count": 2,
        "candidate_runtime_parameter_count": 3,
        "candidate_pool_count": 2,
        "materialization_failure_count": 0,
        "geometry_evaluation_attempt_count": 2,
        "geometry_evaluation_count": 2,
        "finite_candidate_count": 2,
        "nonfinite_candidate_count": 0,
        "logical_geometry_entry_count": 22,
        "committed": True,
        "complete": True,
    }
    run = {
        "initial_state": {"runtime_coordinate_labels": ["a", "b"]},
        "plot_rows": [
            {
                "runtime_parameter_count": 3,
                "parameter_control_rhs_evaluation_count": 0,
                "integration_rhs_evaluation_count": 1,
            }
        ],
        "summary": {"rhs_evaluation_count": 1},
        "trajectory": {
            "points": [
                {
                    "fixed_step": {"support_labels": ["a", "b", "c"]},
                    "patch_decision": {
                        "accepted": True,
                        "metadata": {
                            "candidate_geometry_rounds": [round_work],
                            "estimator_work": {
                                "schema": "paper_ii_avqds_estimator_work_v1",
                                "selector_round_count": 1,
                                "full_geometry_evaluation_attempt_count": 2,
                                "full_geometry_evaluation_count": 2,
                                "logical_candidate_geometry_entries": 22,
                                "complete": True,
                            },
                        },
                    },
                }
            ]
        },
    }

    result = complete_estimator_work([run], route="avqds")

    assert result["logical_integration_geometry_work"] == 11
    assert result["logical_checkpoint_base_geometry_work"] == 7
    assert result["candidate_full_geometry_evaluation_count"] == 2
    assert result["logical_candidate_acquisition_work"] == 22
    assert result["candidate_acquisition_status"] == "exact"
    assert result["selector_call_count"] == 1
    assert result["refit_estimator_status"] == "not_applicable"
    assert result["logical_total_estimator_work"] == 40
    assert result["logical_total_estimator_work_status"] == "exact"
    assert result["selection_round_ledger_complete"] is True

    run["trajectory"]["points"][0]["patch_decision"]["metadata"][
        "estimator_work"
    ]["selector_round_count"] = 2
    inconsistent = complete_estimator_work([run], route="avqds")
    assert inconsistent["candidate_acquisition_status"] == "lower_bound"
    assert inconsistent["logical_total_estimator_work_status"] == "lower_bound"


def test_complete_estimator_work_consumes_exact_exchange_ledgers() -> None:
    certification = {
        "schema": "paper_ii_exchange_certification_estimator_work_v1",
        "base_runtime_parameter_count": 2,
        "patched_runtime_parameter_count": 3,
        "refit_requested": True,
        "refit_counter_available": True,
        "refit_objective_evaluation_attempt_count": 2,
        "refit_objective_evaluation_count": 2,
        "logical_refit_entry_count": 8,
        "full_geometry_evaluation_attempt_count": 1,
        "full_geometry_evaluation_count": 1,
        "logical_full_geometry_entry_count": 11,
        "ray_overlap_check_count": 1,
        "velocity_check_count": 1,
        "complete": True,
        "logical_total_estimator_entries": 19,
        "family": "singleton_d0",
        "kind": "insert",
        "reason": "certified",
        "removed_runtime_indices": [],
        "inserted_selection": [["c", 2]],
    }
    selector_pass = {
        "pass_index": 0,
        "pre_runtime_parameter_count": 2,
        "positioned_candidate_column_count": 2,
        "candidate_pool_count": 2,
        "allowed_deletion_atom_count": 0,
        "insertions_enabled": True,
        "logical_shared_cache_entry_count": 9,
        "complete": True,
    }
    selector_work = {
        "schema": "paper_ii_generalized_exchange_estimator_work_v1",
        "selector_pass_count": 1,
        "selector_passes": [selector_pass],
        "positioned_candidate_column_count": 2,
        "logical_shared_cache_entry_count": 9,
        "structural_candidate_record_count": 3,
        "certification_attempt_count": 1,
        "certification_attempts": [certification],
        "refit_objective_evaluation_attempt_count": 2,
        "refit_objective_evaluation_count": 2,
        "logical_refit_entry_count": 8,
        "full_geometry_evaluation_attempt_count": 1,
        "full_geometry_evaluation_count": 1,
        "logical_finalist_geometry_entry_count": 11,
        "ray_overlap_check_count": 1,
        "velocity_check_count": 1,
        "attempt_record_count": 1,
        "attempt_reason_counts": {"certified": 1},
        "serialized_attempt_record_count": 1,
        "serialized_attempt_ledger_complete": True,
        "logical_total_selector_estimator_entries": 28,
        "complete": True,
    }
    run = {
        "initial_state": {"runtime_coordinate_labels": ["a", "b"]},
        "plot_rows": [
            {
                "runtime_parameter_count": 3,
                "parameter_control_rhs_evaluation_count": 0,
                "integration_rhs_evaluation_count": 1,
            }
        ],
        "summary": {"rhs_evaluation_count": 1},
        "trajectory": {
            "points": [
                {
                    "fixed_step": {"support_labels": ["a", "b", "c"]},
                    "patch_decision": {
                        "accepted": True,
                        "metadata": {
                            "selection_round_work_schema": (
                                "paper_ii_exchange_selection_round_work_v1"
                            ),
                            "selector_call_count": 1,
                            "accepted_selector_count": 1,
                            "selection_round_work": [
                                {
                                    "schema": (
                                        "paper_ii_exchange_selection_round_work_v1"
                                    ),
                                    "selector_call_index": 0,
                                    "pre_runtime_parameter_count": 2,
                                    "post_runtime_parameter_count": 3,
                                    "accepted": True,
                                    "patch_kind": "append",
                                    "reason": "accepted",
                                    "estimator_work": selector_work,
                                    "complete": True,
                                }
                            ],
                        },
                    },
                }
            ]
        },
    }

    result = complete_estimator_work([run], route="q_compiled")

    assert result["logical_integration_geometry_work"] == 11
    assert result["logical_checkpoint_base_geometry_work"] == 7
    assert result["logical_candidate_acquisition_work"] == 9
    assert result["candidate_acquisition_status"] == "exact"
    assert result["certification_full_geometry_evaluation_count"] == 1
    assert result["logical_certification_geometry_work"] == 11
    assert result["certification_geometry_status"] == "exact"
    assert result["refit_objective_evaluation_count"] == 2
    assert result["logical_refit_estimator_work"] == 8
    assert result["refit_estimator_status"] == "exact"
    assert result["selector_call_count"] == 1
    assert result["logical_total_estimator_work"] == 46
    assert result["logical_total_estimator_work_status"] == "exact"
    assert result["selection_round_ledger_complete"] is True

    run["trajectory"]["points"][0]["patch_decision"]["metadata"][
        "selector_call_count"
    ] = 2
    inconsistent = complete_estimator_work([run], route="q_compiled")
    assert inconsistent["candidate_acquisition_status"] == "lower_bound"
    assert inconsistent["certification_geometry_status"] == "lower_bound"
    assert inconsistent["refit_estimator_status"] == "lower_bound"
    assert inconsistent["logical_total_estimator_work_status"] == "lower_bound"


def test_incomplete_exact_avqds_receipt_remains_a_lower_bound() -> None:
    round_work = {
        "geometry_evaluation_attempt_count": 2,
        "geometry_evaluation_count": 1,
        "logical_geometry_entry_count": 11,
        "complete": False,
    }
    run = {
        "initial_state": {"runtime_coordinate_labels": ["a", "b"]},
        "plot_rows": [],
        "summary": {"rhs_evaluation_count": 0},
        "trajectory": {
            "points": [
                {
                    "fixed_step": {"support_labels": ["a", "b"]},
                    "patch_decision": {
                        "metadata": {
                            "candidate_geometry_rounds": [round_work],
                            "estimator_work": {
                                "schema": "paper_ii_avqds_estimator_work_v1",
                                "selector_round_count": 1,
                                "full_geometry_evaluation_attempt_count": 2,
                                "full_geometry_evaluation_count": 1,
                                "logical_candidate_geometry_entries": 11,
                                "complete": False,
                            },
                        }
                    },
                }
            ]
        },
    }

    result = complete_estimator_work([run], route="avqds")

    assert result["logical_candidate_acquisition_work"] == 11
    assert result["candidate_acquisition_status"] == "lower_bound"
    assert result["logical_total_estimator_work_status"] == "lower_bound"


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


def test_multi_round_checkpoint_makes_attempt_counts_lower_bounds() -> None:
    run = {
        "trajectory": {
            "points": [
                {
                    "patch_decision": {
                        "accepted": False,
                        "metadata": {
                            "insertion_rounds": 3,
                            "attempt_count": 1,
                            "attempts": [
                                {
                                    "removed_runtime_indices": [0],
                                    "reason": "smoothness_eta_above_max",
                                }
                            ],
                        },
                    }
                }
            ]
        }
    }

    result = deletion_gate_outcomes([run], applicable=True)

    assert result["attempt_record_complete"] is False
    assert result["multi_commit_checkpoint_count"] == 1
    assert result["post_delete_gate_failure_count"] == 1


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
