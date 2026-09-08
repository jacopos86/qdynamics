"""Small evolving trajectories exercise all search/gain combinations.

These are software and accounting checks, not scientific route comparisons.
Each variant starts from the same two-qubit state and declared candidate pool.
"""

from itertools import product

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import AppendControllerConfig
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
    complete_estimator_work,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    run_append_ap_mclachlan_from_runtime_input,
)
from test.test_ap_mclachlan_exchange_integration import _config
from test.test_ap_mclachlan_route_parity import _runtime_input


TIMES = (0.0, 0.02, 0.04)
MODES = tuple(product(
    ("greedy", "block_exchange", "bounded_exhaustive"),
    ("singleton_factorized", "joint_batch"),
))


@pytest.mark.parametrize("policy,gain", MODES)
def test_selector_modes_evolve_and_preserve_complete_work_receipts(policy, gain):
    payload = run_append_ap_mclachlan_from_runtime_input(
        _runtime_input(theta=(0.05, -0.04)),
        times=TIMES,
        integrator_method="rk4",
        pinv_rcond=1.0e-10,
        ridge_lambda=1.0e-7,
        controller_config=AppendControllerConfig(),
        support_patch_config=_config(
            exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
            search_policy=policy,
            gain_evaluation=gain,
            insertion_gate_mode="mclachlan_l2",
            debt_policy="drift_ranked",
            max_insertion_batch_size=2,
            max_insertion_rounds_per_checkpoint=1,
            max_joint_patch_evaluations=256,
            max_compiled_patches_per_checkpoint=256,
            compile_workers=1,
        ),
        solve_repair_config=SolveRepairConfig.minimal_profile(),
        code_revision="selector_trajectory_smoke",
    )
    points = payload["trajectory"]["points"]
    assert [point["time"] for point in points] == list(TIMES)
    assert len(payload["plot_rows"]) == 3
    for recorded in (
        payload["run_lock"]["policy"]["guards"],
        payload["trajectory"]["support_patch_config"],
        payload["trajectory"]["metadata"],
    ):
        assert (recorded["search_policy"], recorded["gain_evaluation"]) == (policy, gain)
    assert payload["decision_data_flow"]["uses_exact_reference_for_decision"] is False
    for point in points:
        assert np.isfinite(point["energy_expectation"])
        assert np.all(np.isfinite(point["theta_runtime"]))
        assert len(point["theta_runtime"]) == point["runtime_parameter_count"]
        assert len(point["fixed_step"]["support_labels"]) == point["runtime_parameter_count"]

    intervals = [point["integration_to_next"] for point in points[:-1]]
    assert points[-1]["integration_to_next"] is None
    assert all(interval["integrator_method"] == "rk4" for interval in intervals)
    assert all(interval["rhs_evaluation_count"] >= 4 for interval in intervals)
    assert any(np.linalg.norm(interval["theta_dot"]) > 1.0e-8 for interval in intervals)
    for point, interval in zip(points, intervals):
        np.testing.assert_allclose(
            interval["theta_next"],
            np.asarray(point["theta_runtime"]) + interval["dt"] * np.asarray(interval["theta_dot"]),
            rtol=0.0, atol=1.0e-13,
        )
    np.testing.assert_array_equal(points[-1]["theta_runtime"], intervals[-1]["theta_next"])

    selection_work = validation_work = selection_pairs = validation_pairs = 0
    finalist_work = refit_work = calls = batch_finalists = 0
    for point in points[:-1]:
        metadata = point["patch_decision"]["metadata"]
        route = metadata["selector_route"]
        assert (route["search_policy"], route["gain_evaluation"]) == (policy, gain)
        assert route["bundle"]["bundle_id"] == EXCHANGE_ROUTE_BUNDLE_Q_COMPILED
        rounds = metadata["selection_round_work"]
        assert len(rounds) == metadata["selector_call_count"] > 0
        calls += len(rounds)
        for round_row in rounds:
            work = round_row["estimator_work"]
            assert round_row["complete"] and work["complete"]
            pass_total = 0
            for selector_pass in work["selector_passes"]:
                acquired = selector_pass["structural_acquisition"]
                assert acquired["lazy_candidate_pairs"] and acquired["complete"]
                selected, validated = (acquired["stages"][name] for name in ("selection", "validation"))
                selection_work += selected["logical_entry_count"]
                validation_work += validated["logical_entry_count"]
                selection_pairs += selected["candidate_pair_entry_count"]
                validation_pairs += validated["candidate_pair_entry_count"]
                count = selected["logical_entry_count"] + validated["logical_entry_count"]
                assert count == acquired["logical_total_estimator_entries"]
                assert count == selector_pass["logical_shared_cache_entry_count"]
                pass_total += count
            assert pass_total == work["logical_shared_cache_entry_count"]
            finalist_work += work["logical_finalist_geometry_entry_count"]
            refit_work += work["logical_refit_entry_count"]
            for attempt in work["certification_attempts"]:
                assert attempt["gain_evaluation"] == gain
                assert np.isfinite(attempt["validated_frozen_delta"])
                batch_finalists += len(attempt["inserted_selection"]) > 1

    report = complete_estimator_work([payload], route="q_compiled")
    # Count propagation independently from the selector cache: every RHS
    # evaluates a fresh full geometry on that interval's evolving support.
    evolution_work = sum(
        interval["rhs_evaluation_count"] * (p * (p + 1) // 2 + p + 2)
        for point, interval in zip(points, intervals)
        for p in (point["runtime_parameter_count"],)
    )
    assert report["rhs_geometry_evaluation_count"] == sum(
        interval["rhs_evaluation_count"] for interval in intervals
    ) == payload["summary"]["rhs_evaluation_count"]
    assert report["logical_integration_geometry_work"] == evolution_work > 0
    assert report["selector_call_count"] == calls
    assert report["logical_selection_candidate_acquisition_work"] == selection_work
    assert report["logical_validation_candidate_acquisition_work"] == validation_work
    assert report["logical_candidate_acquisition_work"] == selection_work + validation_work
    assert report["logical_certification_geometry_work"] == finalist_work
    assert report["logical_refit_estimator_work"] == refit_work
    assert report["logical_total_estimator_work"] == (
        report["logical_checkpoint_base_geometry_work"] + evolution_work
        + selection_work + validation_work + finalist_work + refit_work
    )
    assert report["selection_round_ledger_complete"] is True
    assert report["logical_total_estimator_work_status"] == "exact"
    if gain == "singleton_factorized":
        assert selection_pairs == 0
        if batch_finalists:
            assert validation_pairs > 0
    else:
        assert selection_pairs > 0
        assert validation_pairs == 0
