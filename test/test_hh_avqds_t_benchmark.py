from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.legacy.hh_benchmarks import hh_avqds_t_benchmark as bench


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
            "append_overlap_threshold": 0.9999,
            "append_min_overlap_gain": 1.0e-7,
            "exact_reference_method": "benchmark_exact",
            "exact_steps_multiplier": 4,
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
            "avqdst_linear_solve_total": 123,
            "avqdst_step_count": intervals,
            "avqdst_state_prep_total": 456,
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
    assert case.append_overlap_threshold == pytest.approx(0.9999)
    assert case.append_min_overlap_gain == pytest.approx(1.0e-7)

    manifest = bench._manifest_payload(
        records=[],
        output_dir=Path("out"),
        manifest_json=Path("out/manifest.json"),
        rows_json=Path("out/rows.json"),
        summary_json=Path("out/summary.json"),
        command="unit",
    )
    assert manifest["method_contract"]["method_id"] == bench.METHOD_ID
    assert manifest["method_contract"]["method_kind"] == "avqds_t"
    assert manifest["method_contract"]["decision_mode"] == "exact_v1"
    assert manifest["method_contract"]["diagnostic_exact_assisted"] is True
    assert manifest["method_contract"]["qpu_faithful"] is False
    assert "single finite-difference" in manifest["method_contract"]["step_policy"]
    assert "serial sum" in manifest["method_contract"]["compile_cost_policy"]


def test_row_extraction_semantics_exact_assisted_non_qpu() -> None:
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

    assert row["method_id"] == "hh_td_avqds_t_pareto_lean_l2_exactv1"
    assert row["method_kind"] == "avqds_t"
    assert row["decision_mode"] == "exact_v1"
    assert row["diagnostic_exact_assisted"] is True
    assert row["qpu_faithful"] is False
    assert row["append_events_total"] == 1
    assert row["append_candidate_evaluations_total"] == 2
    assert row["unique_scaffold_count"] == 2
    assert row["final_logical_block_count"] == 15
    assert row["final_runtime_parameter_count"] == 26
    assert row["fidelity_min"] == pytest.approx(0.97)
    assert row["avqdst_linear_solve_total"] == 123
    assert row["avqdst_step_count"] == 3
    assert row["avqdst_state_prep_total"] == 456
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


def test_toy_append_improves_target_fit() -> None:
    target_theta = 0.7
    target = np.asarray([np.cos(target_theta / 2.0), np.sin(target_theta / 2.0)], dtype=complex)

    def prepare_base(theta: np.ndarray) -> np.ndarray:
        del theta
        return np.asarray([1.0, 0.0], dtype=complex)

    def prepare_candidate(theta: np.ndarray) -> np.ndarray:
        angle = float(np.asarray(theta, dtype=float).reshape(-1)[0])
        return np.asarray([np.cos(angle / 2.0), np.sin(angle / 2.0)], dtype=complex)

    base_fit = bench._solve_target_tangent_step(
        prepare_state=prepare_base,
        theta_start=np.asarray([], dtype=float),
        target_state=target,
        finite_difference_epsilon=1.0e-5,
        regularization_lambda=1.0e-8,
        pinv_relative_cutoff=1.0e-10,
    )
    candidate_fit = bench._solve_target_tangent_step(
        prepare_state=prepare_candidate,
        theta_start=np.asarray([0.0], dtype=float),
        target_state=target,
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
        min_overlap_gain=1.0e-4,
    )

    assert selected is not None
    assert selected.candidate_label == "toy_ry"
    assert gain > 1.0e-4
    assert candidate_fit.final_overlap > base_fit.final_overlap
    assert candidate_fit.final_overlap > 0.999


def test_near_singular_tangent_system_regularizes_clearly() -> None:
    target_theta = 0.4
    target = np.asarray([np.cos(target_theta / 2.0), np.sin(target_theta / 2.0)], dtype=complex)

    def prepare_redundant(theta: np.ndarray) -> np.ndarray:
        arr = np.asarray(theta, dtype=float).reshape(-1)
        angle = float(arr[0] + arr[1])
        return np.asarray([np.cos(angle / 2.0), np.sin(angle / 2.0)], dtype=complex)

    result = bench._solve_target_tangent_step(
        prepare_state=prepare_redundant,
        theta_start=np.asarray([0.0, 0.0], dtype=float),
        target_state=target,
        finite_difference_epsilon=1.0e-5,
        regularization_lambda=1.0e-8,
        pinv_relative_cutoff=1.0e-10,
    )

    assert result.linear_solve_status == "regularized_spectral_solve"
    assert result.retained_rank == 1
    assert np.all(np.isfinite(result.theta_runtime))
    assert np.isfinite(result.final_overlap)
    assert result.final_overlap > result.initial_overlap
    assert result.delta_norm > 0.0

