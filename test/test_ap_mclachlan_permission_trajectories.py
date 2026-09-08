"""Tight-budget trajectories exercise permitted exchanges after a refused D."""

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import AppendControllerConfig
from pipelines.time_dynamics.ap_mclachlan.exchange_config import EXCHANGE_ROUTE_BUNDLE_Q_COMPILED
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    run_append_ap_mclachlan_from_runtime_input,
)
from test.test_ap_mclachlan_exchange_integration import _config
from test.test_ap_mclachlan_route_parity import _runtime_input


@pytest.mark.parametrize("policy", ("greedy", "block_exchange", "bounded_exhaustive"))
def test_refused_deletion_preserves_six_slot_coverage_and_singleton_trajectory_parity(policy):
    trajectories = []
    for gain in ("joint_batch", "singleton_factorized"):
        payload = run_append_ap_mclachlan_from_runtime_input(
            _runtime_input(theta=(0.2, 0.0)),
            times=(0.0, 0.02, 0.04), integrator_method="rk4",
            pinv_rcond=1.0e-10, ridge_lambda=1.0e-7,
            controller_config=AppendControllerConfig(),
            support_patch_config=_config(
                exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
                search_policy=policy, gain_evaluation=gain,
                insertion_gate_mode="mclachlan_l2", debt_policy="drift_ranked",
                exchange_permission_ray_distance_max=0.1,
                max_insertion_batch_size=1, max_insertion_rounds_per_checkpoint=1,
                max_joint_patch_evaluations=6,
                max_compiled_patches_per_checkpoint=256, compile_workers=1,
            ),
            solve_repair_config=SolveRepairConfig.minimal_profile(),
            code_revision="permission_budget_trajectory_validation",
        )
        points = payload["trajectory"]["points"]
        assert [point["time"] for point in points] == [0.0, 0.02, 0.04]
        first = points[0]["patch_decision"]["metadata"]
        permission = first["deletion_permission"]
        assert permission["rejected_deletion_set_count"] == 1
        assert permission["permitted_deletion_set_count"] == 1
        search = first["selector_route"]["levels"][0]["search"]
        # Stay, two pure insertions, permitted deletion and its two exchanges.
        # The earlier rejected D must not consume a seventh evaluation slot.
        assert search["evaluated_patch_count"] == 6
        assert search["refused_deletion_count"] == 1
        assert payload["decision_data_flow"]["uses_exact_reference_for_decision"] is False
        for point in points:
            assert np.isfinite(point["energy_expectation"])
            assert np.all(np.isfinite(point["theta_runtime"]))
            assert len(point["theta_runtime"]) == point["runtime_parameter_count"]
        trajectories.append(points)

    # Singleton gain modes share the solve. Compare each complete evolving run,
    # including certified choices, not just the initial frozen geometry.
    for joint, factorized in zip(*trajectories):
        assert joint["runtime_parameter_count"] == factorized["runtime_parameter_count"]
        assert joint["fixed_step"]["support_labels"] == factorized["fixed_step"]["support_labels"]
        np.testing.assert_allclose(joint["theta_runtime"], factorized["theta_runtime"], rtol=0, atol=1e-12)
        assert joint["energy_expectation"] == pytest.approx(factorized["energy_expectation"], abs=1e-12, rel=0)
        assert joint["patch_decision"]["accepted"] == factorized["patch_decision"]["accepted"]
