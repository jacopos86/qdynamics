from __future__ import annotations

import numpy as np
import pytest

from pipelines.scaffold import hh_continuation_scoring as scoring
from pipelines.static_adapt.accepted_refit import (
    SupportedFSPowellChart,
    map_phase_order_joint_step_to_supported_fs,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
)
from pipelines.static_adapt.joint_step_warm_start import (
    ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1,
    RouteAJointStepWarmStartConfig,
    guard_supported_fs_full_joint_step_seed,
)


def _workspace(
    *,
    candidate_gain_policy: str,
    active_gradient: float = 1.0,
) -> scoring._BatchFullGeometryWorkspace:
    records = (
        {
            "candidate_label": "useless",
            "candidate_pool_index": 0,
            "position_id": 1,
        },
        {
            "candidate_label": "useful",
            "candidate_pool_index": 1,
            "position_id": 1,
        },
    )
    return scoring._BatchFullGeometryWorkspace(
        records=records,
        record_index={},
        ansatz_depth=1,
        active_indices=(0,),
        active_labels=("active",),
        G_AA=np.eye(1, dtype=float),
        H_AA=np.eye(1, dtype=float),
        G_AB=np.zeros((1, 2), dtype=float),
        H_AB=np.zeros((1, 2), dtype=float),
        G_BB=np.eye(2, dtype=float),
        H_BB=np.eye(2, dtype=float),
        g_A=np.asarray([active_gradient], dtype=float),
        g_B=np.asarray([0.0, 2.0], dtype=float),
        phase2_reported_g_B=np.asarray([0.0, 2.0], dtype=float),
        geometry_mode="full_residual_gram_hessian_v1",
        joint_context_mode="full_ansatz_v1",
        workspace_fingerprint="incremental-gain-test",
        metric_regularization=1.0e-9,
        energy_regularization=1.0e-12,
        joint_linear_solve_policy=(
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        ),
        rank_relative_tolerance=1.0e-9,
        max_gram_condition_number=1.0e12,
        max_fubini_study_step=10.0,
        state_delta_norm=0.0,
        state_consistency_tolerance=1.0e-10,
        phase2_reuse_validation={},
        _subset_cache={},
        phase3_candidate_gain_policy=candidate_gain_policy,
    )


def test_candidate_feature_stamps_configured_incremental_gain_policy() -> None:
    compile_cost = scoring.Phase1CompileCostOracle().estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=0,
    )
    measurement_stats = scoring.MeasurementCacheAudit().estimate(["x"])
    common = {
        "stage_name": "core",
        "candidate_label": "x",
        "candidate_family": "test",
        "candidate_pool_index": 0,
        "position_id": 0,
        "append_position": 0,
        "positions_considered": [0],
        "gradient_signed": 0.5,
        "metric_proxy": 1.0,
        "sigma_hat": 0.0,
        "refit_window_indices": [],
        "compile_cost": compile_cost,
        "measurement_stats": measurement_stats,
        "leakage_penalty": 0.0,
        "stage_gate_open": True,
        "leakage_gate_open": True,
        "trough_probe_triggered": False,
        "trough_detected": False,
        "cfg": scoring.SimpleScoreConfig(),
    }

    corrected = scoring.build_candidate_features(
        **common,
        cheap_score_cfg=scoring.FullScoreConfig(
            phase3_candidate_gain_policy=(
                scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
            )
        ),
    )
    legacy = scoring.build_candidate_features(**common)

    assert corrected.phase3_candidate_gain_policy == (
        scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
    )
    assert legacy.phase3_candidate_gain_policy == (
        scoring.PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1
    )


def test_incremental_gain_subtracts_one_shared_active_only_baseline() -> None:
    workspace = _workspace(
        candidate_gain_policy=(
            scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
        )
    )

    useless = workspace._supported_metric_summary_for_indices((0,))
    useful = workspace._supported_metric_summary_for_indices((1,))

    assert useless["full_joint_gain"] == pytest.approx(0.5)
    assert useless["active_only_gain"] == pytest.approx(0.5)
    assert useless["incremental_candidate_gain_raw"] == pytest.approx(0.0)
    assert useless["joint_gain"] == pytest.approx(0.0)
    assert useful["full_joint_gain"] == pytest.approx(2.5)
    assert useful["active_only_gain"] == pytest.approx(0.5)
    assert useful["incremental_candidate_gain_raw"] == pytest.approx(2.0)
    assert useful["joint_gain"] == pytest.approx(2.0)
    assert useful["active_parameter_relaxation"] == pytest.approx([1.0])
    assert useful["batch_coordinate_step"] == pytest.approx([2.0])

    first_baseline = useless["phase3_candidate_gain_receipt"][
        "active_only_baseline"
    ]
    second_baseline = useful["phase3_candidate_gain_receipt"][
        "active_only_baseline"
    ]
    assert first_baseline["candidate_independent"] is True
    assert first_baseline == second_baseline
    assert first_baseline["classical_quantum_query_charge"] == 0
    assert useless["joint_gain_semantics"] == "incremental_candidate_gain_v1"
    assert useless["phase3_candidate_gain_receipt"][
        "comparison_tolerance"
    ] >= 2.0e-12


def test_batch_subtracts_the_shared_active_baseline_exactly_once() -> None:
    workspace = _workspace(
        candidate_gain_policy=(
            scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
        )
    )

    batch = workspace._supported_metric_summary_for_indices((0, 1))

    assert batch["full_joint_gain"] == pytest.approx(2.5)
    assert batch["active_only_gain"] == pytest.approx(0.5)
    assert batch["joint_gain"] == pytest.approx(2.0)
    assert batch["phase3_candidate_gain_receipt"][
        "incremental_candidate_gain_raw"
    ] == pytest.approx(2.0)
    assert batch["contextual_single_total"] == pytest.approx(2.0)


def test_rank_gated_incremental_summary_retains_policy_receipt() -> None:
    workspace = _workspace(
        candidate_gain_policy=(
            scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
        )
    )
    workspace.G_BB[0, 0] = 0.0

    summary = workspace._supported_metric_summary_for_indices((0,))
    cached_summary = workspace._supported_metric_summary_for_indices((0,))

    assert summary["feasible"] is False
    assert summary["reason"] == "rank_gate"
    assert summary["phase3_candidate_gain_policy"] == (
        scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
    )
    receipt = summary["phase3_candidate_gain_receipt"]
    assert receipt["policy"] == (
        scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
    )
    assert receipt["comparison_feasible"] is False
    assert receipt["selected_gain"] == pytest.approx(0.0)
    assert receipt["active_only_baseline"]["candidate_independent"] is True
    assert receipt["classical_quantum_query_charge"] == 0
    assert cached_summary["phase3_candidate_gain_receipt"][
        "active_only_baseline"
    ] == receipt["active_only_baseline"]


def test_rank_gated_summary_rejects_unknown_gain_policy() -> None:
    workspace = _workspace(candidate_gain_policy="unknown")
    workspace.G_BB[0, 0] = 0.0

    with pytest.raises(
        ValueError,
        match="phase3_candidate_gain_policy must be one of",
    ):
        workspace._supported_metric_summary_for_indices((0,))


def test_active_baseline_receipt_is_independent_of_candidate_order() -> None:
    first_workspace = _workspace(
        candidate_gain_policy=(
            scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
        )
    )
    first = first_workspace._supported_metric_summary_for_indices((0,))
    second = first_workspace._supported_metric_summary_for_indices((1,))

    reverse_workspace = _workspace(
        candidate_gain_policy=(
            scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
        )
    )
    reverse_second = reverse_workspace._supported_metric_summary_for_indices(
        (1,)
    )
    reverse_first = reverse_workspace._supported_metric_summary_for_indices(
        (0,)
    )

    assert first["phase3_candidate_gain_receipt"][
        "active_only_baseline"
    ] == reverse_first["phase3_candidate_gain_receipt"][
        "active_only_baseline"
    ]
    assert second["phase3_candidate_gain_receipt"][
        "active_only_baseline"
    ] == reverse_second["phase3_candidate_gain_receipt"][
        "active_only_baseline"
    ]


def test_legacy_total_gain_and_stationary_limit_remain_explicit() -> None:
    legacy = _workspace(
        candidate_gain_policy=(
            scoring.PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1
        )
    )._supported_metric_summary_for_indices((0,))
    stationary = _workspace(
        candidate_gain_policy=(
            scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
        ),
        active_gradient=0.0,
    )._supported_metric_summary_for_indices((1,))

    assert legacy["joint_gain"] == pytest.approx(0.5)
    assert legacy["phase3_candidate_gain_receipt"]["policy"] == (
        scoring.PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1
    )
    assert stationary["active_only_gain"] == pytest.approx(0.0)
    assert stationary["full_joint_gain"] == pytest.approx(2.0)
    assert stationary["joint_gain"] == pytest.approx(2.0)


def _chart(
    *,
    whitened_to_logical: np.ndarray,
    objective,
) -> SupportedFSPowellChart:
    logical_count, rank = whitened_to_logical.shape
    return SupportedFSPowellChart(
        objective=objective,
        x0=np.zeros(rank, dtype=float),
        lift_to_runtime=lambda x: np.asarray(x, dtype=float),
        coordinate_mode="supported_fs_whitened:test",
        active_logical_indices=tuple(range(logical_count)),
        active_runtime_indices=tuple(range(logical_count)),
        active_optimizer_indices=tuple(range(rank)),
        reduced_positions_by_logical={},
        origin_state=np.asarray([1.0 + 0.0j]),
        origin_logical_theta=np.zeros(logical_count, dtype=float),
        origin_runtime_theta=np.zeros(logical_count, dtype=float),
        whitened_to_logical_map=np.asarray(
            whitened_to_logical, dtype=float
        ),
        logical_to_whitened_map=np.linalg.pinv(
            np.asarray(whitened_to_logical, dtype=float)
        ),
        coordinate_registry=tuple(
            f"theta_{index}" for index in range(logical_count)
        ),
        base_telemetry={},
    )


def test_rank_deficient_joint_step_map_records_supported_projection() -> None:
    chart = _chart(
        whitened_to_logical=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=float,
        ),
        objective=lambda x: float(np.dot(x, x)),
    )

    mapped, receipt = map_phase_order_joint_step_to_supported_fs(
        chart=chart,
        phase_order_joint_step=np.asarray([0.2, 0.7, -0.3]),
        phase3_to_post_logical_permutation=[1, 2, 0],
    )

    assert mapped == pytest.approx([-0.3, 0.2])
    assert receipt["requested_post_logical_step"] == pytest.approx(
        [-0.3, 0.2, 0.7]
    )
    assert receipt["supported_post_logical_step"] == pytest.approx(
        [-0.3, 0.2, 0.0]
    )
    assert receipt["discarded_null_logical_step_norm"] == pytest.approx(0.7)
    assert receipt["source_step_within_supported_chart"] is False
    assert receipt["classical_quantum_query_charge"] == 0


def test_rank_deficient_joint_step_is_not_silently_changed_before_guard() -> None:
    calls: list[np.ndarray] = []

    def objective(value: np.ndarray) -> float:
        calls.append(np.asarray(value, dtype=float).copy())
        return 0.0

    chart = _chart(
        whitened_to_logical=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=float,
        ),
        objective=objective,
    )
    x0, receipt, nfev = guard_supported_fs_full_joint_step_seed(
        objective=objective,
        incumbent_energy=1.0,
        chart=chart,
        config=RouteAJointStepWarmStartConfig(
            mode=ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
        ),
        selected_records=[{"candidate_label": "candidate"}],
        selector_summary={
            "active_parameter_relaxation": [0.2, 0.7],
            "batch_coordinate_step": [-0.3],
            "joint_step": [0.2, 0.7, -0.3],
            "selected_labels": ["candidate"],
        },
        phase3_to_post_logical_permutation=[1, 2, 0],
    )

    assert x0 == pytest.approx([0.0, 0.0])
    assert receipt["status"] == "unavailable"
    assert receipt["reason"] == "joint_step_outside_refit_supported_chart"
    assert receipt["supported_fs_mapping"][
        "source_step_within_supported_chart"
    ] is False
    assert nfev == 0
    assert calls == []


@pytest.mark.parametrize(
    ("proposal_energy", "expected_status", "expected_x"),
    [
        (0.8, "accepted", [0.3, 0.2]),
        (1.2, "rejected", [0.0, 0.0]),
    ],
)
def test_supported_fs_full_joint_guard_reuses_incumbent_and_evaluates_once(
    proposal_energy: float,
    expected_status: str,
    expected_x: list[float],
) -> None:
    calls: list[np.ndarray] = []

    def objective(value: np.ndarray) -> float:
        calls.append(np.asarray(value, dtype=float).copy())
        return float(proposal_energy)

    chart = _chart(
        whitened_to_logical=np.eye(2, dtype=float),
        objective=objective,
    )
    selected_records = [{"candidate_label": "candidate"}]
    selector_summary = {
        "active_parameter_relaxation": [0.2],
        "batch_coordinate_step": [0.3],
        "joint_step": [0.2, 0.3],
        "selected_labels": ["candidate"],
        "applied_predicted_reduction": 0.25,
        "phase3_candidate_gain_receipt": {
            "schema": "phase3_candidate_gain_receipt_v1",
            "policy": (
                scoring.PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
            ),
        },
    }

    x0, receipt, nfev = guard_supported_fs_full_joint_step_seed(
        objective=chart.objective,
        incumbent_energy=1.0,
        chart=chart,
        config=RouteAJointStepWarmStartConfig(
            mode=ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
        ),
        selected_records=selected_records,
        selector_summary=selector_summary,
        phase3_to_post_logical_permutation=[1, 0],
    )

    assert x0 == pytest.approx(expected_x)
    assert receipt["status"] == expected_status
    assert receipt["guard_objective_evals"] == 1
    assert receipt["selection_mutated"] is False
    assert nfev == 1
    assert len(calls) == 1


def test_supported_fs_guard_maps_active_and_ordered_batch_coordinates() -> None:
    calls: list[np.ndarray] = []

    def objective(value: np.ndarray) -> float:
        calls.append(np.asarray(value, dtype=float).copy())
        return 0.5

    chart = _chart(
        whitened_to_logical=np.eye(3, dtype=float),
        objective=objective,
    )
    x0, receipt, nfev = guard_supported_fs_full_joint_step_seed(
        objective=objective,
        incumbent_energy=1.0,
        chart=chart,
        config=RouteAJointStepWarmStartConfig(
            mode=ROUTE_A_JOINT_STEP_WARM_START_EXACT_GUARDED_V1
        ),
        selected_records=[
            {"candidate_label": "first"},
            {"candidate_label": "second"},
        ],
        selector_summary={
            "active_parameter_relaxation": [0.1],
            "batch_coordinate_step": [0.2, 0.3],
            "joint_step": [0.1, 0.2, 0.3],
            "selected_labels": ["first", "second"],
            "applied_predicted_reduction": 0.4,
        },
        phase3_to_post_logical_permutation=[2, 0, 1],
    )

    assert x0 == pytest.approx([0.2, 0.3, 0.1])
    assert receipt["status"] == "accepted"
    assert receipt["supported_fs_mapping"][
        "requested_post_logical_step"
    ] == pytest.approx([0.2, 0.3, 0.1])
    assert receipt["guard_objective_evals"] == 1
    assert nfev == 1
    assert calls[0] == pytest.approx([0.2, 0.3, 0.1])
