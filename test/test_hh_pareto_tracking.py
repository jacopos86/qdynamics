from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_pareto_tracking import (
    compute_pareto_frontier,
    extract_staged_hh_pareto_rows,
    write_pareto_tracking,
)
from pipelines.static_adapt.selector_measurement_proxy import (
    CONTROLLER_WORK_SCOPE_VERSION,
    ControllerMeasurementWorkAccumulator,
    controller_proxy_from_history_rows,
    history_controller_measurement_work_stats,
    history_selector_measurement_stats,
    validate_controller_proxy_for_shot_objective,
)


def _physics() -> dict[str, object]:
    return {
        "L": 2,
        "t": 1.0,
        "u": 2.0,
        "dv": 0.0,
        "omega0": 1.0,
        "g_ep": 1.0,
        "n_ph_max": 1,
    }


def test_extract_staged_hh_pareto_rows_accumulates_measurement_and_gate_cost() -> None:
    rows = extract_staged_hh_pareto_rows(
        run_tag="hh_run",
        physics=_physics(),
        warm_payload={
            "ansatz": "hh_hva_ptw",
            "energy": -1.00,
            "exact_filtered_energy": -1.10,
            "num_parameters": 4,
        },
        adapt_payload={
            "continuation_mode": "phase3_v1",
            "pool_type": "paop_full",
            "exact_gs_energy": -1.10,
            "optimal_point": [0.1, 0.2, 0.3, 0.4],
            "history": [
                {
                    "depth": 1,
                    "depth_cumulative": 1,
                    "batch_size": 1,
                    "candidate_family": "paop_full",
                    "selection_mode": "append",
                    "energy_after_opt": -1.04,
                    "delta_abs_current": 0.06,
                    "delta_abs_drop_from_prev": 0.04,
                    "measurement_cache_stats": {
                        "groups_new": 2,
                        "shots_new": 2000.0,
                        "reuse_count_cost": 2.0,
                    },
                    "compile_cost_proxy": {
                        "gate_proxy_total": 5.0,
                        "cx_proxy_total": 3.0,
                        "sq_proxy_total": 4.0,
                        "max_pauli_weight": 2.0,
                    },
                },
                {
                    "depth": 2,
                    "depth_cumulative": 2,
                    "batch_size": 2,
                    "candidate_family": "paop_full",
                    "selection_mode": "batch",
                    "energy_after_opt": -1.08,
                    "delta_abs_current": 0.02,
                    "delta_abs_drop_from_prev": 0.04,
                    "runtime_split_mode": "shortlist_pauli_children_v1",
                    "runtime_split_child_count": 2,
                    "measurement_cache_stats": {
                        "groups_new": 1,
                        "shots_new": 1000.0,
                        "reuse_count_cost": 1.0,
                    },
                    "compile_cost_proxy": {
                        "gate_proxy_total": 7.0,
                        "cx_proxy_total": 4.0,
                        "sq_proxy_total": 6.0,
                        "max_pauli_weight": 3.0,
                    },
                },
            ],
        },
        replay_payload={
            "replay_contract": {"continuation_mode": "phase3_v1"},
            "generator_family": {"resolved": "paop_full"},
            "exact": {"E_exact_sector": -1.10},
            "vqe": {
                "energy": -1.099,
                "abs_delta_e": 0.001,
                "num_parameters": 8,
            },
        },
        recorded_utc="2026-03-20T12:00:00Z",
    )

    assert len(rows) == 4
    warm, adapt_1, adapt_2, replay = rows
    assert warm.stage_kind == "warm_start"
    assert bool(warm.pareto_eligible) is False

    assert adapt_1.stage_kind == "adapt_depth"
    assert adapt_1.num_parameters == 2
    assert adapt_1.measurement_groups_cumulative == 2
    assert adapt_1.compile_gate_proxy_cumulative == 5.0
    assert adapt_1.delta_E_drop_per_new_group == 0.02

    assert adapt_2.num_parameters == 4
    assert adapt_2.measurement_groups_cumulative == 3
    assert adapt_2.measurement_shots_cumulative == 3000.0
    assert adapt_2.compile_gate_proxy_cumulative == 12.0
    assert adapt_2.compile_cx_proxy_cumulative == 7.0
    assert adapt_2.runtime_split_mode == "shortlist_pauli_children_v1"
    assert adapt_2.runtime_split_child_count == 2
    assert bool(adapt_2.pareto_eligible) is True

    assert replay.stage_kind == "conventional_replay"
    assert bool(replay.pareto_eligible) is False


def test_extract_staged_hh_pareto_rows_prefers_native_controller_measurement_work() -> None:
    rows = extract_staged_hh_pareto_rows(
        run_tag="hh_run",
        physics=_physics(),
        warm_payload={"energy": -1.0, "exact_filtered_energy": -1.1},
        adapt_payload={
            "continuation_mode": "phase3_v1",
            "pool_type": "paop_full",
            "exact_gs_energy": -1.1,
            "optimal_point": [0.1],
            "history": [
                {
                    "depth": 1,
                    "depth_cumulative": 1,
                    "batch_size": 1,
                    "energy_after_opt": -1.05,
                    "delta_abs_current": 0.05,
                    "delta_abs_drop_from_prev": 0.05,
                    "measurement_cache_stats": {"groups_new": 99, "shots_new": 9900.0, "reuse_count_cost": 99.0},
                    "selector_measurement_cache_stats": {
                        "source": "native_admitted_selector_pre_commit_v1",
                        "groups_new": 2,
                        "shots_new": 20.0,
                        "reuse_count_cost": 2.0,
                    },
                    "controller_measurement_work_proxy": {
                        "schema": "controller_measurement_work_proxy_v1",
                        "source_kind": "native_controller_work",
                        "groups_new": 5,
                        "total_groups_new": 5,
                        "shots_new": 50.0,
                        "total_shots_new": 50.0,
                        "reuse_count_cost": 5.0,
                        "records_evaluated": 7,
                    },
                    "compile_cost_proxy": {"gate_proxy_total": 4.0, "cx_proxy_total": 2.0, "sq_proxy_total": 4.0},
                }
            ],
        },
        replay_payload={"vqe": {"energy": -1.09, "abs_delta_e": 0.01}, "exact": {"E_exact_sector": -1.1}},
        recorded_utc="2026-03-20T12:00:00Z",
    )

    adapt_1 = rows[1]
    assert adapt_1.measurement_groups_cumulative == 5
    assert adapt_1.measurement_shots_cumulative == pytest.approx(50.0)
    assert adapt_1.selector_shot_proxy_legacy_fallback_used is False
    assert adapt_1.controller_shot_proxy_legacy_fallback_used is False



def test_controller_measurement_work_accumulator_is_shot_depth_and_cache_aware() -> None:
    acc = ControllerMeasurementWorkAccumulator(nominal_shots_per_group=2)

    first = acc.record_event(
        phase="phase1",
        event_kind="probe",
        group_keys=["ze", "zz"],
        nominal_shots_per_group=2,
        records_evaluated=2,
    )
    second = acc.record_event(
        phase="phase1",
        event_kind="probe",
        group_keys=["ze"],
        nominal_shots_per_group=5,
        records_evaluated=1,
    )

    assert first["groups_total"] == 1  # `zz` covers `ze` under QWC basis compression.
    assert first["shots_new"] == pytest.approx(2.0)
    assert second["groups_topup"] == 1
    assert second["shots_new"] == pytest.approx(3.0)
    summary = acc.summary(include_events=False)
    assert summary["total_shots_new"] == pytest.approx(5.0)
    assert summary["records_evaluated"] == pytest.approx(3.0)
    assert summary["work_scope_version"] == CONTROLLER_WORK_SCOPE_VERSION
    assert summary["work_scope_count"] == 1


def test_controller_measurement_work_accumulator_scopes_distinct_controller_surfaces() -> None:
    acc = ControllerMeasurementWorkAccumulator(nominal_shots_per_group=2)

    phase1 = acc.record_event(
        phase="phase1",
        event_kind="phase1_append_probe",
        group_keys=["zz"],
        nominal_shots_per_group=2,
        records_evaluated=1,
        depth=1,
    )
    phase3 = acc.record_event(
        phase="phase3",
        event_kind="phase3_reduced_geometry_rerank",
        group_keys=["zz"],
        nominal_shots_per_group=2,
        records_evaluated=1,
        depth=1,
    )
    batch = acc.record_event(
        phase="phase3",
        event_kind="batch_union_scoring",
        group_keys=["zz"],
        nominal_shots_per_group=2,
        records_evaluated=1,
        depth=1,
    )

    assert phase1["shots_new"] == pytest.approx(2.0)
    assert phase3["shots_new"] == pytest.approx(2.0)
    assert batch["shots_new"] == pytest.approx(2.0)
    assert phase1["work_scope"] != phase3["work_scope"]
    assert phase3["work_scope"] != batch["work_scope"]
    summary = acc.summary(include_events=False)
    assert summary["total_shots_new"] == pytest.approx(6.0)
    assert summary["work_scope_count"] == 3
    assert set(summary["by_phase"]) == {"phase1", "phase3"}
    assert summary["by_phase"]["phase3"]["total_shots_new"] == pytest.approx(4.0)


def test_controller_measurement_work_accumulator_preserves_operator_probe_contract_v2() -> None:
    acc = ControllerMeasurementWorkAccumulator(nominal_shots_per_group=1)

    event = acc.record_event(
        phase="phase1",
        event_kind="phase1_append_probe",
        group_keys=["zz"],
        records_evaluated=2,
        candidate_count=2,
        shortlist_size=1,
        retained_count=1,
        rejected_count=1,
        probe_role="gradient",
        actual_operator_probe_count=2,
        common_parent_candidate_count=5,
        common_expanded_candidate_count=7,
        common_exposure_operator_probe_count=7,
    )

    assert event["operator_probe_event_schema"] == "paper_i_operator_probe_event_v2"
    assert event["work_contract_id"] == "paper_i_hh_operator_probe_contract_v2"
    assert event["operator_probe_charge_basis"] == "logical_estimator_request_pre_grouping_v1"
    assert event["common_exposure_stage"] == "post_common_eligibility_post_expansion_pre_method_filter"
    assert event["common_exposure_policy_id"] == "trajectory_conditioned_full_child_common_exposure_v1"

    summary = acc.summary(include_events=False)
    phase1 = summary["by_phase"]["phase1"]
    assert summary["actual_operator_probe_count"] == 2
    assert summary["common_exposure_operator_probe_count"] == 7
    assert summary["common_parent_candidate_count_total"] == 5
    assert summary["common_expanded_candidate_count_total"] == 7
    assert summary["method_input_candidate_count_total"] == 2
    assert summary["method_shortlist_candidate_count_total"] == 1
    assert phase1["probe_role"] == "gradient"
    assert phase1["common_exposure_operator_probe_count"] == 7
    assert phase1["expansion_policy_id"] == "shortlist_pauli_children_v1"


def test_controller_measurement_work_accumulator_keeps_beam_branch_scopes_separate() -> None:
    acc = ControllerMeasurementWorkAccumulator(nominal_shots_per_group=1)

    left = acc.record_event(
        phase="phase3",
        event_kind="phase3_reduced_geometry_rerank",
        group_keys=["zz"],
        records_evaluated=1,
        depth=2,
        scope_qualifiers={"branch": "left"},
    )
    right = acc.record_event(
        phase="phase3",
        event_kind="phase3_reduced_geometry_rerank",
        group_keys=["zz"],
        records_evaluated=1,
        depth=2,
        scope_qualifiers={"branch": "right"},
    )

    assert left["shots_new"] == pytest.approx(1.0)
    assert right["shots_new"] == pytest.approx(1.0)
    assert left["work_scope"] != right["work_scope"]
    assert acc.summary(include_events=False)["work_scope_count"] == 2


def test_controller_proxy_validation_rejects_legacy_fallback_for_shot_objective() -> None:
    native = validate_controller_proxy_for_shot_objective(
        {
            "schema": "controller_measurement_work_proxy_v1",
            "source_kind": "native_controller_work",
            "legacy_fallback_used": False,
            "events_count": 1,
            "native_row_count": 1,
            "history_row_count": 1,
            "total_shots_new": 10.0,
        }
    )
    legacy = validate_controller_proxy_for_shot_objective(
        {
            "schema": "controller_measurement_work_proxy_v1",
            "source_kind": "legacy_admitted_selector",
            "legacy_fallback_used": True,
            "events_count": 1,
            "legacy_row_count": 1,
            "history_row_count": 1,
            "total_shots_new": 10.0,
        }
    )

    assert native["valid"] is True
    assert native["reason"] == "valid_native_controller_work"
    assert legacy["valid"] is False
    assert legacy["reason"] == "legacy_fallback"
    legacy_row = {
        "selector_measurement_cache_stats": {
            "groups_new": 2,
            "shots_new": 20.0,
            "groups_total": 2,
        }
    }
    legacy_from_history = history_controller_measurement_work_stats(legacy_row)
    assert legacy_from_history is not None
    assert "work_scope_version" not in legacy_from_history
    legacy_run_summary = controller_proxy_from_history_rows([legacy_row])
    assert legacy_run_summary["work_scope_count"] == 0
    assert legacy_run_summary["scoped_row_count"] == 0
    assert "work_scope_version" not in legacy_run_summary
    missing = validate_controller_proxy_for_shot_objective({})
    assert missing["valid"] is False
    assert missing["reason"] == "not_native_controller_work"


def test_selector_measurement_stats_falls_back_when_native_payload_is_malformed() -> None:
    stats = history_selector_measurement_stats(
        {
            "selector_measurement_cache_stats": {},
            "measurement_cache_stats": {"groups_new": 3, "shots_new": 30.0, "reuse_count_cost": 3.0},
        }
    )

    assert stats is not None
    assert stats["source_kind"] == "legacy_history"
    assert stats["groups_new"] == 3
    assert stats["shots_new"] == pytest.approx(30.0)


def test_compute_pareto_frontier_filters_dominated_rows() -> None:
    frontier = compute_pareto_frontier(
        [
            {
                "run_tag": "a",
                "stage_depth": 1,
                "delta_E_abs": 0.05,
                "measurement_groups_cumulative": 2,
                "compile_gate_proxy_cumulative": 10,
                "pareto_eligible": True,
            },
            {
                "run_tag": "a",
                "stage_depth": 2,
                "delta_E_abs": 0.04,
                "measurement_groups_cumulative": 3,
                "compile_gate_proxy_cumulative": 9,
                "pareto_eligible": True,
            },
            {
                "run_tag": "a",
                "stage_depth": 3,
                "delta_E_abs": 0.06,
                "measurement_groups_cumulative": 4,
                "compile_gate_proxy_cumulative": 12,
                "pareto_eligible": True,
            },
        ]
    )

    assert [row["stage_depth"] for row in frontier] == [2, 1]


def test_write_pareto_tracking_replaces_existing_run_tag_rows(tmp_path: Path) -> None:
    run_json = tmp_path / "hh_run.json"
    rows = extract_staged_hh_pareto_rows(
        run_tag="hh_run",
        physics=_physics(),
        warm_payload={"energy": -1.0, "exact_filtered_energy": -1.1},
        adapt_payload={
            "continuation_mode": "phase3_v1",
            "pool_type": "paop_full",
            "exact_gs_energy": -1.1,
            "optimal_point": [0.1],
            "history": [
                {
                    "depth": 1,
                    "depth_cumulative": 1,
                    "batch_size": 1,
                    "energy_after_opt": -1.05,
                    "delta_abs_current": 0.05,
                    "delta_abs_drop_from_prev": 0.05,
                    "measurement_cache_stats": {"groups_new": 1, "shots_new": 1000.0, "reuse_count_cost": 1.0},
                    "compile_cost_proxy": {"gate_proxy_total": 4.0, "cx_proxy_total": 2.0, "sq_proxy_total": 4.0},
                }
            ],
        },
        replay_payload={"vqe": {"energy": -1.09, "abs_delta_e": 0.01}, "exact": {"E_exact_sector": -1.1}},
        recorded_utc="2026-03-20T12:00:00Z",
    )

    out1 = write_pareto_tracking(rows=rows, output_json_path=run_json, run_tag="hh_run")
    assert Path(out1["paths"]["run_rows_json"]).exists()
    assert Path(out1["paths"]["rolling_frontier_json"]).exists()

    trimmed_rows = rows[:2]
    out2 = write_pareto_tracking(rows=trimmed_rows, output_json_path=run_json, run_tag="hh_run")
    ledger_path = Path(out2["paths"]["rolling_ledger_jsonl"])
    ledger_lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(ledger_lines) == len(trimmed_rows)

    frontier_payload = json.loads(Path(out2["paths"]["run_frontier_json"]).read_text(encoding="utf-8"))
    assert frontier_payload["frontier_count"] == 1
