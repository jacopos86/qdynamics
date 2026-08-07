from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pipelines.scaffold import hh_continuation_scoring as scoring
from pipelines.static_adapt.engine_support import (
    SelectedOptimizerChart,
    _guard_sr_active_only_step,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
    JointLinearSolveConfig,
    solve_joint_linear_model,
)


_V2 = JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2


def _v2_config() -> JointLinearSolveConfig:
    return JointLinearSolveConfig(
        policy=_V2,
        rank_relative_tolerance=1.0e-6,
        metric_regularization=1.0e-9,
        energy_regularization=1.0e-12,
        max_fubini_study_step=0.25,
    )


def _workspace(
    *,
    gram: np.ndarray,
    hessian: np.ndarray,
    gradient: np.ndarray,
) -> scoring._BatchFullGeometryWorkspace:
    record: dict[str, Any] = {
        "candidate_label": "candidate",
        "candidate_pool_index": 0,
        "position_id": 1,
    }
    return scoring._BatchFullGeometryWorkspace(
        records=(record,),
        record_index={},
        ansatz_depth=1,
        active_indices=(0,),
        active_labels=("theta_0",),
        G_AA=np.asarray(gram[:1, :1], dtype=float),
        H_AA=np.asarray(hessian[:1, :1], dtype=float),
        G_AB=np.asarray(gram[:1, 1:], dtype=float),
        H_AB=np.asarray(hessian[:1, 1:], dtype=float),
        G_BB=np.asarray(gram[1:, 1:], dtype=float),
        H_BB=np.asarray(hessian[1:, 1:], dtype=float),
        g_A=np.asarray(gradient[:1], dtype=float),
        g_B=np.asarray(gradient[1:], dtype=float),
        phase2_reported_g_B=np.asarray(gradient[1:], dtype=float),
        geometry_mode="supported_metric_whitened_full_joint_model_v1",
        joint_context_mode="full_ansatz_v1",
        workspace_fingerprint="unit-test-workspace",
        metric_regularization=1.0e-9,
        energy_regularization=1.0e-12,
        joint_linear_solve_policy=_V2,
        rank_relative_tolerance=1.0e-6,
        max_gram_condition_number=1.0e12,
        max_fubini_study_step=0.25,
        state_delta_norm=0.0,
        state_consistency_tolerance=1.0e-10,
        phase2_reuse_validation={},
        _subset_cache={},
    )


def _angled_active_image_model(
    *, image_scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a null-compatible rank-two model with a 30-degree image angle."""

    scale = float(image_scale)
    active_image = np.asarray(
        [0.5 * scale, np.sqrt(3.0) * 0.5 * scale],
        dtype=float,
    )
    first_retained = np.asarray(
        [active_image[0], np.sqrt(1.0 - active_image[0] ** 2), 0.0],
        dtype=float,
    )
    second_middle = float(
        -first_retained[0]
        * (active_image[1] / np.sqrt(2.0))
        / first_retained[1]
    )
    second_retained = np.asarray(
        [
            active_image[1] / np.sqrt(2.0),
            second_middle,
            np.sqrt(
                1.0
                - (active_image[1] / np.sqrt(2.0)) ** 2
                - second_middle**2
            ),
        ],
        dtype=float,
    )
    discarded = np.cross(first_retained, second_retained)
    eigenvectors = np.column_stack(
        [discarded, first_retained, second_retained]
    )
    gram = eigenvectors @ np.diag([0.0, 1.0, 2.0]) @ eigenvectors.T
    hessian = eigenvectors @ np.diag([0.0, 1.0, -2.0]) @ eigenvectors.T
    return gram, hessian, np.zeros(3, dtype=float)


def test_active_restriction_uses_full_support_physical_active_image() -> None:
    gram = np.ones((2, 2), dtype=float)
    hessian = -gram
    gradient = np.ones(2, dtype=float)
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_v2_config(),
    )
    assert full.feasible is True
    assert full.telemetry["metric_support_rank"] == 1

    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0e-6,
        energy_regularization=1.0e-12,
        metric_regularization=1.0e-9,
        max_fubini_study_step=0.25,
    )
    independently_refactored = solve_joint_linear_model(
        gram=gram[:1, :1],
        hessian=hessian[:1, :1],
        gradient=gradient[:1],
        active_coordinate_count=1,
        config=_v2_config(),
    )

    # The retained joint support is span((1, 1)).  The exact active step
    # (a, 0) has supported representative (a/2, a/2); their difference lies
    # in the certified raw-metric nullspace.  Constraining the canonical
    # representative's batch coordinate to zero would incorrectly erase this
    # physical active direction.
    assert restricted["valid"] is True
    assert restricted["schema"] == "sr_v2_shared_support_active_restriction_v2"
    assert restricted["active_restriction_supported_dimension"] == 1
    assert restricted["active_image_rank"] == 1
    assert restricted["active_image_subspace_rotation_certified"] is True
    assert 0.0 <= restricted["active_image_subspace_rotation_bound"] < 1.0
    assert restricted["predicted_reduction"] > 0.0
    assert restricted["predicted_reduction"] == pytest.approx(
        independently_refactored.predicted_reduction
    )
    assert restricted["active_restriction_independent_metric_factorization"] is False
    assert restricted["full_supported_metric_whitening_provenance_id"] == (
        full.telemetry["supported_metric_whitening_provenance_id"]
    )
    assert restricted["joint_step"][0] != 0.0
    assert restricted["joint_step"][1] == 0.0
    assert restricted["active_parameter_relaxation"][0] != 0.0
    assert restricted["batch_coordinate_step"] == [0.0]

    quotient = scoring._sr_supported_quotient_summary(
        full_solve_result=full,
        shared_support_active_restriction=restricted,
        hessian_cluster_tolerance=float(
            full.telemetry["global_trust_eigenspace_tolerance"]
        ),
    )
    assert quotient["quotient_participation_resolved"] is True
    assert quotient["quotient_active_subspace_dimension"] == 1
    assert quotient["quotient_redundant_certified"] is True
    assert max(quotient["quotient_residual_metric_eigenvalues"]) <= (
        quotient["quotient_resolution_floor"]
    )


def test_near_resolution_angled_active_image_fails_closed() -> None:
    gram, hessian, gradient = _angled_active_image_model(image_scale=5.0e-12)
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_v2_config(),
    )
    assert full.feasible is True
    assert full.telemetry["raw_metric_null_compatibility_certified"] is True

    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0e-6,
        energy_regularization=1.0e-12,
        metric_regularization=1.0e-9,
        max_fubini_study_step=0.25,
    )

    sigma_min = restricted["active_image_minimum_retained_singular_value"]
    resolution = restricted["active_image_resolution"]
    assert sigma_min > resolution
    assert sigma_min <= 2.0 * resolution
    assert restricted["valid"] is False
    assert restricted["reason"] == (
        "physical_active_image_subspace_rotation_unresolved"
    )
    assert restricted["active_image_subspace_rotation_certified"] is False
    assert restricted["active_image_subspace_rotation_bound"] is None


def test_stable_angled_active_image_propagates_wedin_bound() -> None:
    gram, hessian, gradient = _angled_active_image_model(image_scale=1.0e-6)
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_v2_config(),
    )
    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0e-6,
        energy_regularization=1.0e-12,
        metric_regularization=1.0e-9,
        max_fubini_study_step=0.25,
    )

    assert restricted["valid"] is True
    rotation_bound = restricted["active_image_subspace_rotation_bound"]
    assert 0.0 < rotation_bound < 1.0
    assert rotation_bound == pytest.approx(
        restricted["active_image_resolution"]
        / restricted["active_image_singular_gap_lower_bound"]
    )
    assert restricted["active_image_subspace_gain_error_bound"] > 0.0

    quotient = scoring._sr_supported_quotient_summary(
        full_solve_result=full,
        shared_support_active_restriction=restricted,
        hessian_cluster_tolerance=float(
            full.telemetry["global_trust_eigenspace_tolerance"]
        ),
    )
    assert quotient["quotient_participation_resolved"] is True
    assert quotient["quotient_participation"] == pytest.approx(0.5)
    assert 0.0 < quotient["quotient_participation_lower_bound"] < 0.5
    assert quotient["quotient_active_image_subspace_rotation_bound"] == (
        pytest.approx(rotation_bound)
    )


def test_saturated_transport_uncertainty_cannot_certify_redundancy() -> None:
    gram = np.diag([4.0, 1.0])
    hessian = np.diag([1.0, -2.0])
    gradient = np.zeros(2, dtype=float)
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_v2_config(),
    )
    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0e-6,
        energy_regularization=1.0e-12,
        metric_regularization=1.0e-9,
        max_fubini_study_step=0.25,
    )
    saturated = dict(restricted)
    saturated_transport = dict(
        restricted["shared_support_quotient_transport"]
    )
    saturated_transport["active_image_subspace_rotation_bound"] = 0.25
    saturated["shared_support_quotient_transport"] = saturated_transport

    quotient = scoring._sr_supported_quotient_summary(
        full_solve_result=full,
        shared_support_active_restriction=saturated,
        hessian_cluster_tolerance=float(
            full.telemetry["global_trust_eigenspace_tolerance"]
        ),
    )

    assert quotient["quotient_participation_resolved"] is False
    assert quotient["quotient_participation_reason"] == (
        "quotient_resolution_saturated_by_transport_uncertainty"
    )
    assert quotient["quotient_redundant_certified"] is False


def test_mixed_null_active_image_fails_closed_without_null_compatibility() -> None:
    gram = np.ones((2, 2), dtype=float)
    hessian = np.diag([-1.0, 1.0])
    gradient = np.zeros(2, dtype=float)
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_v2_config(),
    )
    assert full.telemetry["raw_metric_support_status"] == "resolved"
    assert full.telemetry["raw_metric_null_compatibility_certified"] is False

    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0e-6,
        energy_regularization=1.0e-12,
        metric_regularization=1.0e-9,
        max_fubini_study_step=0.25,
    )

    assert restricted["valid"] is False
    assert restricted["reason"] == (
        "physical_active_image_requires_raw_metric_null_compatibility"
    )
    assert restricted["raw_metric_null_compatibility_reason"] == (
        full.telemetry["raw_metric_null_compatibility_reason"]
    )


def test_active_restriction_reconstructs_the_derived_v2_ridge_from_telemetry() -> None:
    gram = np.diag([1.0, 1.0e-4])
    hessian = np.eye(2)
    gradient = np.asarray([0.2, -0.1])
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=JointLinearSolveConfig(
            policy=_V2,
            rank_relative_tolerance=1.0,
            metric_regularization=0.5,
            energy_regularization=1.0e-12,
            max_fubini_study_step=10.0,
            global_trust_kkt_residual_accuracy=1.0e-10,
            global_trust_metric_distortion_budget=0.8,
        ),
    )
    assert full.feasible is True
    derived_ridge = float(full.telemetry["metric_whitening_ridge"])
    assert 0.0 < derived_ridge < 0.5

    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0,
        energy_regularization=1.0e-12,
        metric_regularization=0.5,
        max_fubini_study_step=10.0,
    )

    assert restricted["valid"] is True
    assert restricted["full_metric_regularization"] == pytest.approx(
        derived_ridge
    )
    assert restricted["full_configured_legacy_metric_regularization"] == (
        pytest.approx(0.5)
    )
    assert restricted["full_supported_metric_whitening_provenance_id"] == (
        full.telemetry["supported_metric_whitening_provenance_id"]
    )


def test_active_restriction_transports_exact_hard_pair_to_full_joint_coordinates() -> None:
    gram = np.eye(2, dtype=float)
    hessian = np.diag([-1.0, 1.0])
    gradient = np.zeros(2, dtype=float)
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_v2_config(),
    )

    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0e-6,
        energy_regularization=1.0e-12,
        metric_regularization=1.0e-9,
        max_fubini_study_step=0.25,
    )

    nested = restricted["restricted_coordinate_trust_solve"]
    assert nested["hard_case_detected"] is True
    assert len(nested["hard_case_sign_candidates_joint"]) == 2
    assert restricted["valid"] is True
    assert restricted["hard_case_detected"] is True
    assert restricted["hard_case_sign_pair_atomic_required"] is True
    transported_candidates = np.asarray(
        restricted["hard_case_sign_candidates_joint"], dtype=float
    )
    assert sorted(transported_candidates[:, 0].tolist()) == pytest.approx(
        [-0.25, 0.25]
    )
    assert transported_candidates[:, 1] == pytest.approx([0.0, 0.0])
    assert restricted["hard_case_sign_candidate_predicted_reductions"] == (
        pytest.approx([0.03125, 0.03125])
    )
    assert all(
        candidate[1] == 0.0
        for candidate in restricted["hard_case_sign_candidates_joint"]
    )
    transport = restricted["active_restriction_atomic_candidate_transport"]
    assert transport["valid"] is True
    assert transport["required_candidate_count"] == 2
    assert transport["transported_candidate_count"] == 2
    assert transport["transport_provenance_id"] == (
        restricted["hard_case_sign_pair_transport_provenance_id"]
    )
    assert transport["full_supported_metric_whitening_provenance_id"] == (
        full.telemetry["supported_metric_whitening_provenance_id"]
    )

    chart = SelectedOptimizerChart(
        objective=lambda theta: float((float(theta[0]) - 0.2) ** 2),
        x0=np.asarray([0.0], dtype=float),
        lift_to_runtime=lambda theta: np.asarray(theta, dtype=float),
        coordinate_mode="logical_shared",
        active_logical_indices=(0,),
        active_runtime_indices=(0,),
        active_optimizer_indices=(0,),
        reduced_positions_by_logical={0: (0,)},
    )
    guarded_seed, guard, nfev = _guard_sr_active_only_step(
        chart=chart,
        active_parameter_step=restricted["active_parameter_relaxation"],
        retained_joint_step_candidates=restricted[
            "hard_case_sign_candidates_joint"
        ],
        retained_candidate_predicted_reductions=restricted[
            "hard_case_sign_candidate_predicted_reductions"
        ],
        retained_candidate_roles=restricted[
            "hard_case_sign_candidate_point_estimate_roles"
        ],
        candidate_coordinate_count=1,
        candidate_block_zero_tolerance=restricted[
            "active_restriction_batch_zero_tolerance"
        ],
    )
    assert nfev == 3
    assert guard["retained_candidate_count"] == 2
    assert guard["status"] == "accepted"
    assert guarded_seed == pytest.approx([0.25])


def test_mixed_null_hard_pair_maps_to_batch_zero_active_representatives() -> None:
    gram = np.ones((2, 2), dtype=float)
    hessian = -gram
    gradient = np.zeros(2, dtype=float)
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_v2_config(),
    )
    assert full.telemetry["raw_metric_null_compatibility_certified"] is True

    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0e-6,
        energy_regularization=1.0e-12,
        metric_regularization=1.0e-9,
        max_fubini_study_step=0.25,
    )

    assert restricted["valid"] is True
    assert restricted["active_restriction_supported_dimension"] == 1
    assert restricted["hard_case_detected"] is True
    candidates = np.asarray(
        restricted["hard_case_sign_candidates_joint"], dtype=float
    )
    assert candidates.shape == (2, 2)
    assert sorted(candidates[:, 0].tolist()) == pytest.approx([-0.25, 0.25])
    assert candidates[:, 1] == pytest.approx([0.0, 0.0])
    assert restricted["hard_case_sign_candidate_predicted_reductions"] == (
        pytest.approx([0.03125, 0.03125])
    )
    transport = restricted["active_restriction_atomic_candidate_transport"]
    assert transport["schema"] == (
        "sr_v2_active_restriction_atomic_candidate_transport_v2"
    )
    assert transport["valid"] is True
    assert transport["maximum_active_image_step_residual"] <= (
        restricted["active_restriction_image_step_tolerance"]
    )
    assert transport["maximum_metric_null_equivalence_residual"] <= (
        restricted["active_restriction_metric_null_tolerance"]
    )


def test_active_restriction_hard_pair_metadata_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gram = np.eye(2, dtype=float)
    hessian = np.diag([-1.0, 1.0])
    gradient = np.zeros(2, dtype=float)
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_v2_config(),
    )
    original = scoring.solve_joint_linear_model

    def _malformed_nested_solve(**kwargs: Any):  # noqa: ANN202
        result = original(**kwargs)
        telemetry = dict(result.telemetry)
        telemetry["hard_case_detected"] = True
        telemetry["hard_case_sign_candidates_joint"] = [
            telemetry["hard_case_sign_candidates_joint"][0]
        ]
        telemetry["hard_case_sign_candidate_predicted_reductions"] = [
            telemetry["hard_case_sign_candidate_predicted_reductions"][0]
        ]
        telemetry["hard_case_sign_candidate_point_estimate_roles"] = [
            telemetry["hard_case_sign_candidate_point_estimate_roles"][0]
        ]
        return replace(result, telemetry=telemetry)

    monkeypatch.setattr(scoring, "solve_joint_linear_model", _malformed_nested_solve)
    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0e-6,
        energy_regularization=1.0e-12,
        metric_regularization=1.0e-9,
        max_fubini_study_step=0.25,
    )

    assert restricted["valid"] is False
    assert restricted["reason"] == (
        "active_restriction_atomic_candidate_transport_failed"
    )
    transport = restricted["active_restriction_atomic_candidate_transport"]
    assert transport["valid"] is False
    assert transport["required_candidate_count"] == 2
    assert transport["source_candidate_count"] == 1
    assert transport["transported_candidate_count"] == 0
    assert "hard_case_sign_candidates_joint" not in restricted


def test_workspace_marginal_uses_shared_support_and_conservative_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gram = np.diag([4.0, 1.0])
    hessian = np.diag([1.0, -2.0])
    gradient = np.zeros(2, dtype=float)
    workspace = _workspace(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
    )
    calls: list[tuple[np.ndarray, str, float]] = []
    original = scoring.solve_joint_linear_model

    def _recording_solve(**kwargs: Any):  # noqa: ANN202
        config = kwargs["config"]
        calls.append(
            (
                np.asarray(kwargs["gram"], dtype=float).copy(),
                str(config.policy),
                float(config.metric_regularization),
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(scoring, "solve_joint_linear_model", _recording_solve)
    summary = workspace._supported_metric_summary_for_indices((0,))

    assert summary["feasible"] is True
    v2_calls = [row for row in calls if row[1] == _V2]
    assert len(v2_calls) == 3
    assert np.array_equal(v2_calls[0][0], gram)
    assert np.array_equal(v2_calls[1][0], np.eye(1))
    assert v2_calls[1][2] == pytest.approx(0.0)
    assert np.array_equal(v2_calls[2][0], np.asarray([[4.0]]))
    assert v2_calls[2][2] == pytest.approx(1.0e-9)
    assert summary["sr_escape_state_stationarity_summary"][
        "comparison_scope"
    ] == (
        "working_state_active_coordinates_independent_of_singleton_v1"
    )

    active = summary["active_restriction_solve"]
    assert summary["active_restriction_independent_metric_factorization"] is False
    assert summary["active_restriction_shared_support_provenance_id"] == (
        summary["supported_metric_whitening_provenance_id"]
    )
    assert active["full_supported_metric_whitening_provenance_id"] == (
        summary["supported_metric_whitening_provenance_id"]
    )
    assert active["active_image_subspace_rotation_certified"] is True
    assert 0.0 <= active["active_image_subspace_rotation_bound"] < 1.0
    assert len(active["joint_step"]) == 2
    assert len(active["active_parameter_relaxation"]) == 1
    assert active["batch_coordinate_step"] == pytest.approx([0.0], abs=1.0e-12)
    assert active["applied_predicted_reduction"] == pytest.approx(
        summary["active_restricted_trust_gain"]
    )
    assert active["fubini_study_displacement_sq"] == pytest.approx(
        active["joint_fubini_study_displacement_sq"]
    )

    assert summary["marginal_trust_gain_comparison_valid"] is True
    assert summary["full_trust_gain_lower_bound"] <= summary["full_trust_gain"]
    assert summary["full_trust_gain"] <= summary["full_trust_gain_upper_bound"]
    assert (
        summary["active_restricted_trust_gain_lower_bound"]
        <= summary["active_restricted_trust_gain"]
        <= summary["active_restricted_trust_gain_upper_bound"]
    )
    assert summary["marginal_trust_gain_lower_bound"] <= (
        summary["marginal_trust_gain_raw"]
    )
    assert summary["marginal_trust_gain_raw"] <= (
        summary["marginal_trust_gain_upper_bound"]
    )
    assert summary["marginal_trust_gain_numerical_error_bound"] == pytest.approx(
        summary["full_trust_gain_numerical_error_bound"]
        + summary["active_restricted_trust_gain_numerical_error_bound"]
    )
    assert summary["active_restricted_trust_gain_numerical_error_bound"] >= (
        summary["active_restricted_subspace_transport_error_bound"]
    )


def test_workspace_zero_ridge_fallback_preserves_physical_marginal_route() -> None:
    coordinate_map = np.diag([1.0, 1.0e-3])
    gram = np.eye(2, dtype=float)
    hessian = np.eye(2, dtype=float)
    gradient = np.asarray([0.2, -0.1], dtype=float)
    original = _workspace(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
    )._supported_metric_summary_for_indices((0,))
    rescaled = _workspace(
        gram=coordinate_map.T @ gram @ coordinate_map,
        hessian=coordinate_map.T @ hessian @ coordinate_map,
        gradient=coordinate_map.T @ gradient,
    )._supported_metric_summary_for_indices((0,))

    assert original["feasible"] is True
    assert rescaled["feasible"] is True
    assert original[
        "metric_stabilization_zero_ridge_fallback_eligible"
    ] is False
    assert rescaled[
        "metric_stabilization_zero_ridge_fallback_eligible"
    ] is True
    assert rescaled[
        "metric_stabilization_zero_ridge_fallback_attempted"
    ] is True
    assert rescaled[
        "metric_stabilization_zero_ridge_fallback_solver_status"
    ] == "certified"
    np.testing.assert_allclose(
        coordinate_map @ np.asarray(rescaled["joint_step"], dtype=float),
        np.asarray(original["joint_step"], dtype=float),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert rescaled["joint_gain"] == pytest.approx(original["joint_gain"])
    assert rescaled["supported_stationarity_status"] == (
        original["supported_stationarity_status"]
    )
    assert rescaled["supported_inertia_status"] == (
        original["supported_inertia_status"]
    )
    assert rescaled["marginal_trust_gain_comparison_valid"] is True
    assert rescaled["active_restriction_solve"]["valid"] is True
    assert rescaled["quotient_participation_resolved"] is True
    assert rescaled["active_restricted_trust_gain"] == pytest.approx(
        original["active_restricted_trust_gain"]
    )
    assert rescaled["marginal_trust_gain_raw"] == pytest.approx(
        original["marginal_trust_gain_raw"]
    )
    provenance = rescaled["supported_metric_whitening_provenance_id"]
    assert rescaled[
        "active_restriction_shared_support_provenance_id"
    ] == provenance
    assert rescaled["active_restriction_solve"][
        "full_supported_metric_whitening_provenance_id"
    ] == provenance
    assert rescaled["active_restriction_solve"][
        "shared_support_quotient_transport"
    ]["full_supported_metric_whitening_provenance_id"] == provenance


def test_quotient_reuses_v2_transport_without_refactorization_or_pinv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gram = np.diag([4.0, 1.0])
    hessian = np.diag([1.0, -2.0])
    gradient = np.zeros(2, dtype=float)
    full = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_v2_config(),
    )
    restricted = scoring._sr_v2_shared_support_active_restriction(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_count=1,
        full_solve_result=full,
        rank_relative_tolerance=1.0e-6,
        energy_regularization=1.0e-12,
        metric_regularization=1.0e-9,
        max_fubini_study_step=0.25,
    )

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("quotient path attempted an independent factorization")

    monkeypatch.setattr(
        scoring, "factor_supported_metric", _forbidden, raising=False
    )
    monkeypatch.setattr(scoring.np.linalg, "pinv", _forbidden)
    quotient = scoring._sr_supported_quotient_summary(
        full_solve_result=full,
        shared_support_active_restriction=restricted,
        hessian_cluster_tolerance=float(
            full.telemetry["global_trust_eigenspace_tolerance"]
        ),
    )

    provenance = full.telemetry["supported_metric_whitening_provenance_id"]
    assert restricted["active_restriction_supported_dimension"] == 1
    assert restricted["shared_support_quotient_transport"]["schema"] == (
        "sr_v2_shared_support_quotient_transport_v2"
    )
    assert quotient["quotient_geometry_schema"] == (
        "sr_v2_full_support_transported_quotient_v2"
    )
    assert quotient["quotient_participation_resolved"] is True
    assert quotient["quotient_participation"] == pytest.approx(1.0)
    assert 0.0 < quotient["quotient_participation_lower_bound"] <= 1.0
    assert quotient["quotient_redundant_certified"] is False
    assert max(quotient["quotient_residual_metric_eigenvalues"]) == (
        pytest.approx(1.0)
    )
    assert quotient["quotient_independent_metric_factorization"] is False
    assert quotient["quotient_independent_metric_pseudoinverse"] is False
    assert quotient["quotient_shared_support_provenance_id"] == provenance
    assert (
        restricted["shared_support_quotient_transport"]
        ["full_supported_metric_whitening_provenance_id"]
        == provenance
    )


def test_null_hessian_direction_cannot_receive_singleton_saddle_credit() -> None:
    workspace = _workspace(
        gram=np.diag([1.0, 0.0]),
        hessian=np.diag([1.0, -1.0]),
        gradient=np.zeros(2, dtype=float),
    )
    summary = workspace._supported_metric_summary_for_indices((0,))

    assert summary["raw_metric_support_status"] == "resolved"
    assert summary["raw_metric_null_compatibility_certified"] is False
    assert summary["reason"] == "raw_metric_null_hessian_incompatible"
    assert summary["quotient_participation_resolved"] is False
    assert summary["quotient_participation_reason"] == (
        "raw_metric_null_hessian_incompatible"
    )
    assert summary["quotient_participation_lower_bound"] == 0.0
    assert summary["quotient_redundant_certified"] is False
    assert summary["negative_curvature_certified"] is False


def test_compatible_metric_null_candidate_is_quotient_redundant() -> None:
    workspace = _workspace(
        gram=np.diag([1.0, 0.0]),
        hessian=np.diag([1.0, 0.0]),
        gradient=np.zeros(2, dtype=float),
    )
    summary = workspace._supported_metric_summary_for_indices((0,))

    assert summary["raw_metric_null_compatibility_certified"] is True
    assert summary["quotient_participation_resolved"] is True
    assert summary["quotient_participation_lower_bound"] == 0.0
    assert summary["quotient_redundant_certified"] is True
    assert summary["negative_curvature_certified"] is False
