import json
from pathlib import Path

from pipelines.reporting import build_paper_i_three_model_support_maps as support


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_three_model_support_maps_append_completed_snake_rows(tmp_path: Path, monkeypatch):
    root = tmp_path / "raw_outputs" / "routeA_paper_i_three_model_hubbard_l2_three_model_weak_selected_logical_v2"
    benchmark = "hubbard_L2_three_model_weak"
    summary = root / "run" / benchmark / "summary.json"
    result = root / "run" / benchmark / "trial_0004" / benchmark / "json" / "result.json"
    _write_json(
        summary,
        {
            "best_trial_number": 4,
            "best_user_attrs": {
                "objective_score_components": {
                    "terminal_compile_matches_first_hit": True,
                    "qiskit_first_hit_cost_status": "recoverable_terminal_first_hit",
                    "paper_facing_resource_display_reason": "benchmark_target_stop_terminal_compile_is_first_hit",
                    "terminal_compiled_resources": {
                        "count_2q": 56,
                        "depth_2q": 52,
                        "circuit_depth": 219,
                        "shot_cost_proxy": 320,
                    },
                }
            },
        },
    )
    _write_json(
        result,
        {
            "adapt_vqe": {
                "success": True,
                "abs_delta_e": 8.8e-7,
                "exact_state_fidelity": 0.999,
            },
            "paper_i_first_crossing": {
                "reached": True,
                "primary_error_at_crossing": 8.8e-7,
                "history_position_tau": 3,
            },
        },
    )
    cost_source = tmp_path / "cost.json"
    fid_source = tmp_path / "fid.json"
    _write_json(cost_source, {"rows": [{"family": "hubbard", "case_id": benchmark, "algorithm_id": "static_hea_qiskit_vqe"}]})
    _write_json(fid_source, {"rows": [{"family": "hubbard", "case_id": benchmark, "algorithm_id": "static_hea_qiskit_vqe"}]})

    monkeypatch.setattr(
        support,
        "_snake_fidelity_fields",
        lambda **_kwargs: {
            "fidelity": 0.999,
            "infidelity": 0.001,
            "one_minus_fidelity": 0.001,
            "infidelity_source_key": "infidelity_same",
            "infidelity_status": "ok",
            "infidelity_statuses": {"infidelity_same": "ok"},
        },
    )

    out_cost = tmp_path / "out_cost.json"
    out_fid = tmp_path / "out_fid.json"
    summary_payload = support.build_support_maps(
        comparator_cost_source=cost_source,
        comparator_fidelity_source=fid_source,
        snake_roots=(tmp_path / "raw_outputs",),
        output_cost_source=out_cost,
        output_fidelity_source=out_fid,
        threshold=2e-4,
        suite_profile="paper_i_three_model_main_20260525_v1",
    )

    assert summary_payload["snake_row_count"] == 1
    cost_rows = json.loads(out_cost.read_text())["rows"]
    fid_rows = json.loads(out_fid.read_text())["rows"]
    snake_cost = [row for row in cost_rows if row["algorithm_id"] == support.SNAKE_ALGORITHM_ID][0]
    snake_fid = [row for row in fid_rows if row["algorithm_id"] == support.SNAKE_ALGORITHM_ID][0]
    assert snake_cost["compiled_count_2q_total"] == 56.0
    assert snake_cost["compiled_depth_2q_total"] == 52.0
    assert snake_cost["compiled_depth_total"] == 219.0
    assert snake_cost["compiled_resource_source_kind"] == "snake_qiskit_compiled_first_hit_ansatz_circuit"
    assert snake_cost["Snorm"] is None
    assert snake_cost["Snorm_status"] == "missing_controller_measurement_work_summary"
    assert snake_cost["legacy_terminal_compiled_resources_shot_cost_proxy"] == 320.0
    assert snake_cost["delta_e_excess_display"] == 0.0
    assert snake_fid["one_minus_fidelity"] == 0.001
    assert snake_fid["infidelity_status"] == "ok"


def test_three_model_support_maps_recompute_snake_shot_proxy_with_phase0(tmp_path: Path, monkeypatch):
    root = tmp_path / "raw_outputs" / "routeA_paper_i_three_model_spin_boson_l2_nph1_three_model_weak_selected_logical_v4"
    benchmark = "spin_boson_L2_nph1_three_model_weak"
    summary = root / "run" / benchmark / "summary.json"
    result = root / "run" / benchmark / "trial_0002" / benchmark / "json" / "result.json"
    _write_json(
        summary,
        {
            "best_trial_number": 2,
            "best_user_attrs": {
                "objective_score_components": {
                    "terminal_compile_matches_first_hit": True,
                    "terminal_compiled_resources": {
                        "count_2q": 28,
                        "depth_2q": 28,
                        "circuit_depth": 113,
                        "shot_cost_proxy": 8,
                    },
                }
            },
        },
    )
    _write_json(
        result,
        {
            "adapt_vqe": {
                "success": True,
                "abs_delta_e": 8.8e-7,
                "exact_state_fidelity": 0.999,
                "adapt_selected_logical_filter": {
                    "schema": "adapt_selected_logical_pool_filter_v1",
                    "applied": True,
                    "fallback_to_full_pool": False,
                    "pool_size_before": 46,
                    "pool_size_after": 2,
                },
                "controller_measurement_work_summary": {
                    "schema": "controller_measurement_work_proxy_v1",
                    "source": "native_controller_live_decision_work_v1",
                    "source_kind": "native_controller_work",
                    "legacy_fallback_used": False,
                    "by_phase": {
                        "phase1": {"records_with_group_keys": 2},
                        "phase2": {"records_with_group_keys": 2},
                        "phase3": {"records_with_group_keys": 4},
                    },
                },
            },
            "paper_i_first_crossing": {
                "reached": True,
                "primary_error_at_crossing": 8.8e-7,
                "history_position_tau": 1,
            },
        },
    )
    cost_source = tmp_path / "cost.json"
    fid_source = tmp_path / "fid.json"
    _write_json(cost_source, {"rows": []})
    _write_json(fid_source, {"rows": []})
    monkeypatch.setattr(
        support,
        "_snake_fidelity_fields",
        lambda **_kwargs: {
            "fidelity": 0.999,
            "infidelity": 0.001,
            "one_minus_fidelity": 0.001,
            "infidelity_source_key": "infidelity_same",
            "infidelity_status": "ok",
            "infidelity_statuses": {"infidelity_same": "ok"},
        },
    )

    out_cost = tmp_path / "out_cost.json"
    out_fid = tmp_path / "out_fid.json"
    support.build_support_maps(
        comparator_cost_source=cost_source,
        comparator_fidelity_source=fid_source,
        snake_roots=(tmp_path / "raw_outputs",),
        output_cost_source=out_cost,
        output_fidelity_source=out_fid,
        threshold=2e-4,
        suite_profile="paper_i_three_model_main_20260525_v1",
    )

    snake_cost = json.loads(out_cost.read_text())["rows"][0]
    assert snake_cost["Snorm"] == 54.0
    assert snake_cost["Snorm_source"] == "snake_runtime_controller_phase_reconstruction_v1"
    assert snake_cost["Snorm_status"] == "ok"
    assert snake_cost["legacy_terminal_compiled_resources_shot_cost_proxy"] == 8.0
    accounting = snake_cost["selected_logical_phase0_accounting"]
    assert accounting["status"] == "inferred_from_selected_logical_filter"
    assert accounting["gradient_probe_count"] == 46.0
