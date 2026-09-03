"""Contracts for the compact LaTeX four-route report."""

import json

from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
    ROUTE_ORDER,
)
from pipelines.time_dynamics.diagnostics.paper_ii_four_route_report import (
    bind_terminal_costs,
    render_latex,
)


def _route_record(route: str) -> dict:
    return {
        "run_sources": [{"path": f"/runs/{route}/run.json", "sha256": route}],
        "time_step_count": 10,
        "n_theta": 4,
        "accuracy": {
            "mean_infidelity": 1.0e-6,
            "mean_abs_energy_error": 2.0e-4,
            "mean_abs_doublon_error": 3.0e-4,
            "mean_abs_charge_imbalance_error": 4.0e-4,
            "accuracy_burden": 0.5,
            "qualifies_individually": True,
        },
        "estimator_work": {
            "rhs_geometry_evaluation_count": 20,
            "logical_integration_geometry_work": 200,
        },
        "structural_churn": {
            "added_coordinate_count": 3,
            "deleted_coordinate_count": 2,
            "gross_churn": 5,
            "excess_churn": 4,
            "short_delete_to_readd_count": 1,
            "short_add_to_redelete_count": 1,
            "support_changing_checkpoint_count": 4,
            "persisted_scored_not_selected_lower_bound": 17,
        },
    }


def _cost_payload(route: str) -> dict:
    return {
        "schema": "ap_terminal_qiskit_cost_table_v1",
        "compile_defaults": {
            "backend_name": "FakeMarrakesh",
            "requested_backend_name": "FakeMarrakesh",
            "local_fake_only": True,
            "optimization_level": 1,
            "seed_transpiler": 7,
        },
        "compile_result": {
            "N2q": 12,
            "D2q": 8,
            "Dc": 30,
            "requested_backend_name": "FakeMarrakesh",
            "backend_name": "FakeMarrakesh",
            "local_fake_only": True,
            "runtime_lookup_enabled": False,
            "optimization_level": 1,
            "seed_transpiler": 7,
            "terminal_reconstruction_parity": {"passed": True},
            "resolution_audit": [
                {
                    "success": True,
                    "resolved_name": "FakeMarrakesh",
                    "fallback_reason": None,
                }
            ],
            "final_support_layout_sha256": "support",
            "final_theta_sha256": "theta",
        },
        "rows": [{"trajectory_json_sha256": route}],
    }


def test_report_binds_four_fresh_costs_and_renders_four_columns(tmp_path) -> None:
    cell_id = "fastweak__delta_theta_5e-5"
    routes = {route: _route_record(route) for route in ROUTE_ORDER}
    comparison = {
        "generated_utc": "2026-09-03T00:00:00+00:00",
        "cells": [
            {
                "cell_id": cell_id,
                "drive": "fastweak",
                "controller": "delta_theta_5e-5",
                "apm_cut": 1.0e-6,
                "comparison_tier": "individual_target",
                "routes": routes,
            }
        ],
    }
    for route in ROUTE_ORDER:
        path = tmp_path / cell_id / f"{route}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_cost_payload(route)), encoding="utf-8")

    selected = bind_terminal_costs(comparison, terminal_cost_root=tmp_path)
    tex = render_latex(comparison, selected, markdown_name="report.md")

    assert len(selected) == 1
    assert "$Q$ compiled" in tex
    assert "support $\\Gamma$" in tex
    assert "repaired proxy" in tex
    assert "old proxy$^{*}$" in tex
    assert "Logical geometry work" in tex
    assert "Gross churn" in tex
    assert "Delete $\\rightarrow$ re-add" in tex
    assert "Scored but not selected" in tex
    assert "physical shot count" in tex


def test_report_can_show_complete_individually_qualified_unmatched_cell(tmp_path) -> None:
    cell_id = "midres__state_motion_5e-5"
    routes = {route: _route_record(route) for route in ROUTE_ORDER}
    comparison = {
        "generated_utc": "2026-09-03T00:00:00+00:00",
        "cells": [
            {
                "cell_id": cell_id,
                "drive": "midres",
                "controller": "state_motion_5e-5",
                "apm_cut": 3.0e-6,
                "comparison_tier": "unmatched",
                "four_way_complete": True,
                "pairwise_accuracy": {
                    "q_compiled__old_proxy": {
                        "passes": False,
                        "burden_ratio": 2.34,
                    }
                },
                "routes": routes,
            }
        ],
    }
    for route in ROUTE_ORDER:
        path = tmp_path / cell_id / f"{route}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_cost_payload(route)), encoding="utf-8")

    selected = bind_terminal_costs(
        comparison,
        terminal_cost_root=tmp_path,
        include_complete_target=True,
    )
    tex = render_latex(comparison, selected, markdown_name="report.md")

    assert len(selected) == 1
    assert "all four routes pass individually" in tex
    assert "burden ratio 2.34" in tex


def test_report_marks_incomplete_calibration_as_analysis_only(tmp_path) -> None:
    comparison = {"generated_utc": "2026-09-03T00:00:00+00:00"}
    cell = {
        "cell_id": "cell",
        "drive": "midres",
        "controller": "state_motion_5e-5",
        "apm_cut": 3.0e-6,
        "comparison_tier": "individual_target",
        "routes": {route: _route_record(route) for route in ROUTE_ORDER},
        "terminal_costs": {
            route: {"N2q": 12, "D2q": 8, "Dc": 30} for route in ROUTE_ORDER
        },
    }

    tex = render_latex(
        comparison,
        [cell],
        markdown_name="report.md",
        calibration={"status": "INCOMPLETE"},
    )

    assert "Calibration gate: \\texttt{INCOMPLETE}" in tex
    assert "analysis-only comparison" in tex
