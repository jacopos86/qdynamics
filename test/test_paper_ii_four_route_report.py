"""Contracts for the compact LaTeX four-route report."""

import json

from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
    ROUTE_ORDER,
)
from pipelines.time_dynamics.diagnostics.paper_ii_four_route_report import (
    FRESH_ROUTE_ORDER,
    bind_terminal_costs,
    render_latex,
    render_markdown,
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
            "comparison_eligible": True,
        },
        "estimator_work": {
            "rhs_geometry_evaluation_count": 20,
            "logical_integration_geometry_work": 200,
            "logical_checkpoint_base_geometry_work": 50,
            "checkpoint_base_status": "exact",
            "logical_candidate_acquisition_work": 300,
            "candidate_acquisition_status": "lower_bound",
            "logical_certification_geometry_work": (
                None if route == "avqds" else 40
            ),
            "certification_geometry_status": (
                "not_applicable" if route == "avqds" else "lower_bound"
            ),
            "logical_refit_estimator_work": (
                None if route == "avqds" else 10
            ),
            "refit_estimator_status": (
                "not_applicable" if route == "avqds" else "lower_bound"
            ),
            "logical_total_estimator_work": (
                550 if route == "avqds" else 600
            ),
            "logical_total_estimator_work_status": "lower_bound",
            "physical_shots": None,
            "physical_shots_status": "unavailable_statevector_no_shot_ledger",
        },
        "support_motion": {
            "added_coordinate_count": 3,
            "deleted_coordinate_count": 2,
            "short_delete_to_readd_count": 1,
            "short_add_to_redelete_count": 1,
            "support_changing_checkpoint_count": 4,
            "support_event_ledger_complete": True,
        },
        "deletion_gate_outcomes": {
            "applicable": route != "avqds",
            "attempt_record_complete": True,
            "deletion_bearing_finalist_attempt_count": (
                None if route == "avqds" else 7
            ),
            "post_delete_gate_failure_count": (
                None if route == "avqds" else 2
            ),
            "committed_deletion_patch_count": (
                None if route == "avqds" else 5
            ),
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


def test_report_binds_five_fresh_costs_and_renders_five_columns(tmp_path) -> None:
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
                "reportable_current_best": True,
                "all_routes_complete": True,
                "available_routes_clique": True,
                "missing_routes": [],
                "achieved_accuracy_floor": {
                    "route": "old_proxy",
                    "accuracy_burden": 0.5,
                },
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
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    calibration = {
        "status": "INCOMPLETE",
        "not_checked": ["benchmark_point_sameness", "seed_provenance"],
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    markdown = render_markdown(
        comparison,
        selected,
        comparison_path=comparison_path,
        tex_path=tmp_path / "report.tex",
        pdf_path=tmp_path / "report.pdf",
        calibration_path=calibration_path,
        calibration=calibration,
    )

    assert len(selected) == 1
    assert "$\\Gamma(v^\\star)$" in tex
    assert "$\\Gamma(v)$" in tex
    assert "is realized and" in tex
    assert "repaired proxy" in tex
    assert "old proxy$^{*}$" in tex
    assert "AVQDS" in tex
    assert "Interim exact-estimator refresh:" in tex
    assert "Propagation geometry elements" in tex
    assert "Checkpoint geometry elements" in tex
    assert "Candidate acquisition elements" in tex
    assert "Finalist geometry elements" in tex
    assert "Local-refit estimator elements" in tex
    assert "Total logical estimator elements" in tex
    assert "$\\geq$300" in tex
    assert "Gross churn" not in tex
    assert "Failed post-delete gates" in tex
    assert "Finalists tested with deletion" in tex
    assert "No hardware-shot ledger" in tex
    assert "Fresh terminal compilation: 5 displayed route/cell arms" in markdown
    assert "FakeMarrakesh" in markdown
    assert "`benchmark_point_sameness`, `seed_provenance`" in markdown
    assert "This is no-evidence, not a pass" in markdown
    assert "logical_candidate_acquisition_work" in markdown
    assert "logical_refit_estimator_work" in markdown
    assert ">=300" in markdown
    assert "unavailable_statevector_no_shot_ledger" in markdown

    fresh = bind_terminal_costs(
        comparison,
        terminal_cost_root=tmp_path,
        route_order=FRESH_ROUTE_ORDER,
        require_all_routes=True,
    )
    fresh_tex = render_latex(comparison, fresh, markdown_name="report.md")
    assert "Three fresh side-by-side routes" in fresh_tex
    assert "repaired proxy" not in fresh_tex
    assert "old proxy" not in fresh_tex
    assert "AVQDS" in fresh_tex
    assert "Xrrr" in fresh_tex


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
                "reportable_current_best": True,
                "all_routes_complete": True,
                "available_routes_clique": False,
                "missing_routes": [],
                "achieved_accuracy_floor": {
                    "route": "old_proxy",
                    "accuracy_burden": 0.5,
                },
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
    assert "least accurate available route" in tex
    assert "does not pass" in tex


def test_report_keeps_gamma_pending_in_current_best_cell(tmp_path) -> None:
    cell_id = "fastweak__delta_theta_5e-5"
    routes = {
        route: _route_record(route)
        for route in ROUTE_ORDER
        if route != "gamma_compiled"
    }
    comparison = {
        "generated_utc": "2026-09-03T00:00:00+00:00",
        "cells": [
            {
                "cell_id": cell_id,
                "drive": "fastweak",
                "controller": "delta_theta_5e-5",
                "apm_cut": 3.0e-7,
                "comparison_tier": "same_cell_route_best",
                "reportable_current_best": True,
                "all_routes_complete": False,
                "available_routes_clique": True,
                "missing_routes": ["gamma_compiled"],
                "achieved_accuracy_floor": {
                    "route": "q_compiled",
                    "accuracy_burden": 0.5,
                },
                "routes": routes,
            }
        ],
    }
    for route in routes:
        path = tmp_path / cell_id / f"{route}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_cost_payload(route)), encoding="utf-8")

    selected = bind_terminal_costs(comparison, terminal_cost_root=tmp_path)
    tex = render_latex(comparison, selected, markdown_name="report.md")

    assert len(selected) == 1
    assert "\\emph{pending}" in tex
    assert "Gamma" in tex
    assert "AVQDS" in tex


def test_report_marks_incomplete_calibration_as_analysis_only(tmp_path) -> None:
    comparison = {"generated_utc": "2026-09-03T00:00:00+00:00"}
    cell = {
        "cell_id": "cell",
        "drive": "midres",
        "controller": "state_motion_5e-5",
        "apm_cut": 3.0e-6,
        "comparison_tier": "individual_target",
        "all_routes_complete": True,
        "available_routes_clique": True,
        "missing_routes": [],
        "achieved_accuracy_floor": {
            "route": "old_proxy",
            "accuracy_burden": 0.5,
        },
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
