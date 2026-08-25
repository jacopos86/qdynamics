from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt.accepted_refit import (
    ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
    ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1,
    AcceptedRefitConfig,
    SupportedFSPowellChart,
)
from pipelines.static_adapt.joint_linear_solve import JointLinearSolveConfig
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS,
    PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract,
)














def test_material_window_trust_reuses_full_accepted_refit_source_metric() -> None:
    chart = SupportedFSPowellChart(
        objective=lambda value: float(np.sum(np.asarray(value) ** 2)),
        x0=np.zeros(3, dtype=float),
        lift_to_runtime=lambda value: np.asarray(value, dtype=float),
        coordinate_mode="supported_fs_whitened:logical_shared_reduced_v1",
        active_logical_indices=(0, 1, 2),
        active_runtime_indices=(0, 1, 2),
        active_optimizer_indices=(0, 1, 2),
        reduced_positions_by_logical={0: (0,), 1: (1,), 2: (2,)},
        origin_state=np.asarray([1.0, 0.0], dtype=complex),
        origin_logical_theta=np.zeros(3, dtype=float),
        origin_runtime_theta=np.zeros(3, dtype=float),
        whitened_to_logical_map=np.diag([1.0, 0.5, 1.0 / 3.0]),
        logical_to_whitened_map=np.diag([1.0, 2.0, 3.0]),
        coordinate_registry=("old0", "candidate", "old1"),
        base_telemetry={
            "base_coordinate_kind": "logical_shared_reduced",
            "raw_logical_fs_metric": np.diag([1.0, 4.0, 9.0]).tolist(),
            "supported_metric_whitening_provenance_id": "accepted-refit-proof",
        },
    )
    selector_summary = {
        "joint_linear_solve_policy_effective": (
            "supported_metric_projected_generalized_trust_v1"
        ),
        "joint_batch_context_mode": "active_window_v1",
        "geometry_workspace": {"active_indices": [1]},
        # W(old pre-index 1), then the inserted singleton at pre-position 1.
        "joint_step": [0.2, 0.3],
        "joint_fubini_study_displacement_sq": 0.72,
        "material_window_receipt": {
            "retained_indices": [1],
            "omitted_indices": [0],
        },
    }
    transaction = (
        adapt_pipeline._material_window_full_source_metric_trust_transaction(
            selector_summary=selector_summary,
            pre_parameter_count=2,
            positions_in_commit_order=[1],
            accepted_refit_chart=chart,
            pre_refit_logical_theta=np.zeros(3, dtype=float),
            post_refit_logical_theta=np.asarray([0.1, 0.4, 0.2]),
            radius_before=0.5,
            metric_query_accounting={
                "incremental_quantum_query_charge": 6,
                "metric_element_count_acquired": 6,
            },
        )
    )

    # Original old order is [old0, old1], while the post chart is
    # [old0, candidate, old1]. Omitted old0 is zero-filled.
    assert transaction["predicted_full_post_logical_step"] == pytest.approx(
        [0.0, 0.3, 0.2]
    )
    assert transaction["predicted_source_metric_displacement_sq"] == pytest.approx(
        0.72
    )
    assert transaction["realized_source_metric_displacement_sq"] == pytest.approx(
        1.01
    )
    assert transaction["source_metric_reused_from_accepted_refit"] is True
    assert transaction["incremental_quantum_query_charge"] == 0
    assert transaction["accepted_refit_metric_query_accounting"][
        "metric_element_count_acquired"
    ] == 6
    assert transaction["phase3_prediction_coordinate_scope"] == (
        "candidate_material_W_plus_singleton_v1"
    )
    assert transaction["trust_calibration_metric_scope"] == (
        "full_accepted_refit_source_gram_v1"
    )
