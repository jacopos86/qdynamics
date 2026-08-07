from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.legacy.hh_benchmarks import hh_avqds_benchmark as bench


def _cost_row(method: str, scope: str, twoq: int, depth: int, size: int):
    return bench.overlay.CircuitCostRow(
        method=method,
        order=None,
        scope=scope,
        trotter_steps=None,
        includes_seed_prep=True,
        abstract_size=10,
        abstract_depth=5,
        compiled_count_2q=twoq,
        compiled_depth=depth,
        compiled_size=size,
        compiled_num_qubits=156,
        backend_name="FakeMarrakesh",
        seed_transpiler=7,
        optimization_level=2,
        transpile_status="ok",
        compiled_op_counts={},
        logical_to_physical=[],
    )


def _run_artifact(*, num_times: int = 4, intervals: int = 3) -> dict:
    epoch_records = [
        bench._ScaffoldCompileRecord(
            signature="epoch-a",
            interval_count=2,
            cost=_cost_row(bench.METHOD_KIND, bench.EPOCH_SOURCE_SCOPE, 11, 23, 31),
            raw_rows=[],
        ),
        bench._ScaffoldCompileRecord(
            signature="epoch-b",
            interval_count=1,
            cost=_cost_row(bench.METHOD_KIND, bench.EPOCH_SOURCE_SCOPE, 17, 29, 37),
            raw_rows=[],
        ),
    ]
    hardware_rows = bench._hardware_report_rows(
        final_state_cost=_cost_row(bench.METHOD_KIND, bench.STATE_SCOPE, 17, 29, 37),
        controller_cost=_cost_row("controller", bench.CONTROLLER_SOURCE_SCOPE, 85, 188, 366),
        epoch_records=epoch_records,
        intervals=intervals,
    )
    return {
        "schema_version": bench.RUN_SCHEMA_VERSION,
        "case_id": "hh_l2_t8_anchor_v1",
        "method_id": bench.METHOD_ID,
        "method_kind": bench.METHOD_KIND,
        "source": {"controller_json": "controller.json", "artifact_json": "seed.json", "run_tag": "anchor"},
        "parameter_manifest": {
            "controller_json": "controller.json",
            "seed_artifact_json": "seed.json",
            "drive_enabled": True,
            "t_final": 8.0,
            "num_times": num_times,
            "finite_difference_epsilon": 1.0e-5,
            "regularization_lambda": 1.0e-8,
            "pinv_relative_cutoff": 1.0e-10,
            "append_rhs_residual_ratio_threshold": 1.0e-3,
            "append_min_residual_ratio_gain": 1.0e-5,
            "exact_reference_method": "benchmark_exact",
            "exact_steps_multiplier": 4,
            "decision_mode": "benchmark_rhs_v1",
            "diagnostic_exact_assisted": False,
            "exact_fields_reporting_only": True,
            "compile_backend": "FakeMarrakesh",
            "compile_seed_transpiler": 7,
            "compile_optimization_level": 2,
        },
        "summary": {
            "row_count": num_times,
            "final_energy_total": 0.9,
            "final_energy_total_exact": 1.0,
            "final_abs_energy_total_error": 0.1,
            "mean_abs_energy_total_error": 0.05,
            "max_abs_energy_total_error": 0.1,
            "fidelity_min": 0.97,
            "append_events_total": 1,
            "append_candidate_evaluations_total": 2,
            "unique_scaffold_count": 2,
            "final_logical_block_count": 15,
            "final_runtime_parameter_count": 26,
            "avqds_linear_solve_total": 123,
            "avqds_step_count": intervals,
            "avqds_state_prep_total": 456,
            "rhs_residual_ratio_final": 2.5e-4,
            "rhs_residual_ratio_max": 8.0e-4,
        },
        "hardware_report_rows": hardware_rows,
    }


def test_default_case_and_manifest_contract() -> None:
    case = bench.default_cases()[0]

    assert case.case_id == "hh_l2_t8_anchor_v1"
    assert case.controller_json == bench.overlay.DEFAULT_CONTROLLER_JSON
    assert case.source_artifact_json == bench.fixed.DEFAULT_PARETO_ARTIFACT
    assert case.spec_name == "pareto_lean_l2"
    assert case.loader_mode == "replay_family"
    assert case.generator_family == "match_adapt"
    assert case.fallback_family == "full_meta"
    assert case.append_pool_family == "match_replay"
    assert case.finite_difference_epsilon == pytest.approx(1.0e-5)
    assert case.regularization_lambda == pytest.approx(1.0e-8)
    assert case.pinv_relative_cutoff == pytest.approx(1.0e-10)
    assert case.append_rhs_residual_ratio_threshold == pytest.approx(1.0e-3)
    assert case.append_min_residual_ratio_gain == pytest.approx(1.0e-5)

    manifest = bench._manifest_payload(
        records=[],
        output_dir=Path("out"),
        manifest_json=Path("out/manifest.json"),
        rows_json=Path("out/rows.json"),
        summary_json=Path("out/summary.json"),
        command="unit",
    )
    contract = manifest["method_contract"]
    assert contract["method_id"] == bench.METHOD_ID
    assert contract["method_kind"] == "avqds"
    assert contract["decision_mode"] == "benchmark_rhs_v1"
    assert contract["diagnostic_exact_assisted"] is False
    assert contract["qpu_faithful"] is False
    assert contract["exact_fields_reporting_only"] is True
    assert "local-Hamiltonian RHS" in contract["step_policy"]
    assert "serial sum" in contract["compile_cost_policy"]


def test_row_extraction_semantics_plain_rhs_non_qpu() -> None:
    case = bench.default_cases()[0]
    row = bench._row_from_run_artifact(
        _run_artifact(),
        case=case,
        artifact_run_json=Path("runs/hh_l2_t8_anchor_v1.json"),
        artifact_manifest_json=Path("manifest.json"),
        artifact_rows_json=Path("rows.json"),
        artifact_summary_json=Path("summary.json"),
        preferred_fake_backends=("FakeMarrakesh",),
    )

    assert row["method_id"] == "hh_td_avqds_pareto_lean_l2_rhsv1"
    assert row["method_kind"] == "avqds"
    assert row["decision_mode"] == "benchmark_rhs_v1"
    assert row["diagnostic_exact_assisted"] is False
    assert row["qpu_faithful"] is False
    assert row["exact_fields_reporting_only"] is True
    assert row["append_events_total"] == 1
    assert row["append_candidate_evaluations_total"] == 2
    assert row["unique_scaffold_count"] == 2
    assert row["final_logical_block_count"] == 15
    assert row["final_runtime_parameter_count"] == 26
    assert row["fidelity_min"] == pytest.approx(0.97)
    assert row["avqds_linear_solve_total"] == 123
    assert row["avqds_step_count"] == 3
    assert row["avqds_state_prep_total"] == 456
    assert row["rhs_residual_ratio_final"] == pytest.approx(2.5e-4)
    assert row["rhs_residual_ratio_max"] == pytest.approx(8.0e-4)
    assert row["state_at_time_scope"] == bench.STATE_SCOPE
    assert row["state_at_time_2q"] == 17
    assert row["state_at_time_depth"] == 29
    assert row["full_horizon_scope"] == bench.HORIZON_SCOPE
    assert row["full_horizon_horizon_2q"] == 11 * 2 + 17
    assert row["full_horizon_depth_serial"] == 23 * 2 + 29
    assert row["controller_state_2q"] == 85
    assert row["controller_state_depth"] == 188


def test_append_candidate_pool_fails_closed_when_unavailable() -> None:
    class DummyReplayContext:
        pool_meta = {"candidate_pool_complete": True}
        append_pool_meta = {
            "candidate_pool_complete": False,
            "append_pool_source": "explicit_family_incomplete",
            "incomplete_reason": "unit_test",
        }
        family_pool = ()
        append_family_pool = ()

    with pytest.raises(ValueError, match="complete append candidate pool"):
        bench._append_candidate_pool(DummyReplayContext())


def test_epoch_horizon_budget_uses_adaptive_serial_sum_policy() -> None:
    rows = bench._hardware_report_rows(
        final_state_cost=_cost_row(bench.METHOD_KIND, bench.STATE_SCOPE, 17, 29, 37),
        controller_cost=_cost_row("controller", bench.CONTROLLER_SOURCE_SCOPE, 85, 188, 366),
        epoch_records=[
            bench._ScaffoldCompileRecord(
                signature="a",
                interval_count=2,
                cost=_cost_row(bench.METHOD_KIND, bench.EPOCH_SOURCE_SCOPE, 5, 7, 9),
                raw_rows=[],
            ),
            bench._ScaffoldCompileRecord(
                signature="b",
                interval_count=3,
                cost=_cost_row(bench.METHOD_KIND, bench.EPOCH_SOURCE_SCOPE, 11, 13, 15),
                raw_rows=[],
            ),
        ],
        intervals=5,
    )
    horizon = bench._required_report_row(rows, method=bench.METHOD_KIND, scope=bench.HORIZON_SCOPE)

    assert horizon["intervals"] == 5
    assert horizon["unique_scaffold_count"] == 2
    assert horizon["compile_representative_policy"] == "latest_active_theta_for_scaffold_signature"
    assert horizon["horizon_count_2q"] == 5 * 2 + 11 * 3
    assert horizon["horizon_depth_serial"] == 7 * 2 + 13 * 3
    assert horizon["compiled_count_2q"] == horizon["horizon_count_2q"]
    assert horizon["compiled_depth"] == horizon["horizon_depth_serial"]


def test_single_signature_horizon_has_no_final_scaffold_inconsistency() -> None:
    rows = bench._hardware_report_rows(
        final_state_cost=_cost_row(bench.METHOD_KIND, bench.STATE_SCOPE, 85, 188, 366),
        controller_cost=_cost_row("controller", bench.CONTROLLER_SOURCE_SCOPE, 85, 188, 366),
        epoch_records=[
            bench._ScaffoldCompileRecord(
                signature="only-active-shape",
                interval_count=160,
                cost=_cost_row(bench.METHOD_KIND, bench.EPOCH_SOURCE_SCOPE, 85, 188, 366),
                raw_rows=[],
            ),
        ],
        intervals=160,
    )
    state = bench._required_report_row(rows, method=bench.METHOD_KIND, scope=bench.STATE_SCOPE)
    horizon = bench._required_report_row(rows, method=bench.METHOD_KIND, scope=bench.HORIZON_SCOPE)

    assert horizon["unique_scaffold_count"] == 1
    assert horizon["horizon_count_2q"] == state["compiled_count_2q"] * 160
    assert horizon["horizon_depth_serial"] == state["compiled_depth"] * 160
    assert horizon["epoch_costs"][0]["compiled_count_2q"] == state["compiled_count_2q"]
    assert horizon["epoch_costs"][0]["compiled_depth"] == state["compiled_depth"]


def test_toy_residual_based_append_selection() -> None:
    h_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)

    def prepare_base(theta: np.ndarray) -> np.ndarray:
        del theta
        return np.asarray([1.0, 0.0], dtype=complex)

    def prepare_candidate(theta: np.ndarray) -> np.ndarray:
        angle = float(np.asarray(theta, dtype=float).reshape(-1)[0])
        return np.asarray([np.cos(angle / 2.0), np.sin(angle / 2.0)], dtype=complex)

    base_fit = bench._solve_rhs_tangent_step(
        prepare_state=prepare_base,
        theta_start=np.asarray([], dtype=float),
        hmat_total=h_y,
        dt=0.1,
        finite_difference_epsilon=1.0e-5,
        regularization_lambda=1.0e-8,
        pinv_relative_cutoff=1.0e-10,
    )
    candidate_fit = bench._solve_rhs_tangent_step(
        prepare_state=prepare_candidate,
        theta_start=np.asarray([0.0], dtype=float),
        hmat_total=h_y,
        dt=0.1,
        finite_difference_epsilon=1.0e-5,
        regularization_lambda=1.0e-8,
        pinv_relative_cutoff=1.0e-10,
    )
    selected, gain = bench._select_append_candidate(
        base_fit=base_fit,
        candidate_fits=[
            bench.AppendCandidateTangentFit(
                candidate_pool_index=0,
                candidate_label="toy_ry",
                fit=candidate_fit,
                theta_runtime=candidate_fit.theta_runtime,
                terms=(),
                layout=None,
                executor=None,  # type: ignore[arg-type]
                new_runtime_indices=(0,),
            )
        ],
        min_residual_ratio_gain=1.0e-3,
    )

    assert selected is not None
    assert selected.candidate_label == "toy_ry"
    assert gain > 0.9
    assert base_fit.rhs_residual_ratio == pytest.approx(1.0)
    assert candidate_fit.rhs_residual_ratio < 1.0e-6


def test_near_singular_tangent_system_regularizes_clearly() -> None:
    h_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)

    def prepare_redundant(theta: np.ndarray) -> np.ndarray:
        arr = np.asarray(theta, dtype=float).reshape(-1)
        angle = float(arr[0] + arr[1])
        return np.asarray([np.cos(angle / 2.0), np.sin(angle / 2.0)], dtype=complex)

    result = bench._solve_rhs_tangent_step(
        prepare_state=prepare_redundant,
        theta_start=np.asarray([0.0, 0.0], dtype=float),
        hmat_total=h_y,
        dt=0.1,
        finite_difference_epsilon=1.0e-5,
        regularization_lambda=1.0e-8,
        pinv_relative_cutoff=1.0e-10,
    )

    assert result.linear_solve_status == "regularized_spectral_solve"
    assert result.retained_rank == 1
    assert np.all(np.isfinite(result.theta_runtime))
    assert np.isfinite(result.rhs_residual_ratio)
    assert result.rhs_residual_ratio < 1.0e-6
    assert result.delta_norm > 0.0


def test_exact_reporting_only_invariance_for_row_counters_and_costs() -> None:
    case = bench.default_cases()[0]
    base_payload = _run_artifact()
    alt_payload = deepcopy(base_payload)
    alt_payload["summary"].update(
        {
            "final_energy_total_exact": 2.0,
            "final_abs_energy_total_error": 1.1,
            "mean_abs_energy_total_error": 0.55,
            "max_abs_energy_total_error": 1.1,
            "fidelity_min": 0.42,
        }
    )

    row_a = bench._row_from_run_artifact(
        base_payload,
        case=case,
        artifact_run_json=Path("runs/a.json"),
        artifact_manifest_json=Path("manifest.json"),
        artifact_rows_json=Path("rows.json"),
        artifact_summary_json=Path("summary.json"),
    )
    row_b = bench._row_from_run_artifact(
        alt_payload,
        case=case,
        artifact_run_json=Path("runs/b.json"),
        artifact_manifest_json=Path("manifest.json"),
        artifact_rows_json=Path("rows.json"),
        artifact_summary_json=Path("summary.json"),
    )

    invariant_fields = [
        "append_events_total",
        "append_candidate_evaluations_total",
        "unique_scaffold_count",
        "final_logical_block_count",
        "final_runtime_parameter_count",
        "avqds_linear_solve_total",
        "avqds_step_count",
        "avqds_state_prep_total",
        "rhs_residual_ratio_final",
        "rhs_residual_ratio_max",
        "state_at_time_2q",
        "state_at_time_depth",
        "full_horizon_horizon_2q",
        "full_horizon_depth_serial",
        "controller_state_2q",
        "controller_state_depth",
    ]
    for field in invariant_fields:
        assert row_b[field] == row_a[field]

    assert row_a["exact_fields_reporting_only"] is True
    assert row_b["exact_fields_reporting_only"] is True
    assert row_b["final_energy_total_exact"] != row_a["final_energy_total_exact"]
    assert row_b["final_abs_energy_total_error"] != row_a["final_abs_energy_total_error"]
    assert row_b["fidelity_min"] != row_a["fidelity_min"]
