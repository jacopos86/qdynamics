from __future__ import annotations

import numpy as np
import pytest

import pipelines.static_adapt.joint_linear_solve as joint_linear_solve_module
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
    JointLinearSolveConfig,
    factor_supported_metric,
    solve_joint_linear_model,
)


def _config(**overrides: float | str) -> JointLinearSolveConfig:
    values = {
        "policy": JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
        "rank_relative_tolerance": 1e-6,
        "metric_regularization": 1e-12,
        "energy_regularization": 1e-12,
        "max_fubini_study_step": 10.0,
        **overrides,
    }
    return JointLinearSolveConfig(**values)


def test_raw_metric_support_removes_near_null_direction_before_ridge() -> None:
    result = solve_joint_linear_model(
        gram=np.diag([1.0, 1e-14]),
        hessian=np.diag([1.0, 1e-12]),
        gradient=np.asarray([0.0, 1e-4]),
        active_coordinate_count=1,
        config=_config(metric_regularization=1e-2),
    )

    assert result.feasible is True
    assert result.telemetry["metric_retained_mask"] == [False, True]
    assert result.telemetry["metric_support_rank"] == 1
    assert result.joint_step[1] == pytest.approx(0.0, abs=1e-14)
    assert result.predicted_reduction == pytest.approx(0.0, abs=1e-14)


def test_public_factorization_exposes_shared_support_and_both_metric_identities() -> None:
    gram = np.diag([1.0e-14, 4.0])
    factor = factor_supported_metric(
        gram,
        rank_relative_tolerance=1.0e-6,
        metric_regularization=0.5,
    )

    assert factor.feasible is True
    assert factor.retained_mask.tolist() == [False, True]
    assert factor.rank == 1
    regularized_identity = (
        factor.whitening.T
        @ factor.regularized_supported_metric
        @ factor.whitening
    )
    np.testing.assert_allclose(regularized_identity, np.eye(1), atol=1.0e-14)
    np.testing.assert_allclose(
        factor.whitening.T @ gram @ factor.whitening,
        factor.regularized_to_raw_frame.T
        @ factor.regularized_to_raw_frame,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        factor.raw_orthonormalizer.T @ gram @ factor.raw_orthonormalizer,
        np.eye(1),
        atol=1.0e-14,
    )
    assert factor.telemetry()["classical_quantum_query_charge"] == 0


def test_rotated_near_duplicate_direction_cannot_generate_large_gain() -> None:
    epsilon = 1e-12
    gram = np.asarray([[1.0, 1.0 - epsilon], [1.0 - epsilon, 1.0]])
    discarded_direction = np.asarray([1.0, -1.0])
    result = solve_joint_linear_model(
        gram=gram,
        hessian=np.eye(2),
        gradient=1e-4 * discarded_direction,
        active_coordinate_count=1,
        config=_config(metric_regularization=1e-2),
    )

    assert result.feasible is True
    assert result.telemetry["metric_retained_mask"] == [False, True]
    assert np.linalg.norm(result.joint_step) < 1e-12
    assert result.predicted_reduction == pytest.approx(0.0, abs=1e-14)


def test_material_negative_metric_rejects_but_roundoff_negative_is_excluded() -> None:
    rejected = solve_joint_linear_model(
        gram=np.diag([1.0, -1e-3]),
        hessian=np.eye(2),
        gradient=np.ones(2),
        active_coordinate_count=1,
        config=_config(),
    )
    tolerated = solve_joint_linear_model(
        gram=np.diag([1.0, -1e-16]),
        hessian=np.eye(2),
        gradient=np.asarray([1.0, 0.0]),
        active_coordinate_count=1,
        config=_config(),
    )

    assert rejected.feasible is False
    assert rejected.reason == "materially_negative_metric_eigenvalue"
    assert tolerated.feasible is True
    assert tolerated.telemetry["metric_support_rank"] == 1


def test_well_conditioned_whitened_solver_matches_legacy_block_solver() -> None:
    gram = np.asarray([[1.2, 0.1], [0.1, 0.9]])
    hessian = np.asarray([[2.0, 0.2], [0.2, 1.4]])
    gradient = np.asarray([0.3, -0.2])
    common = {
        "rank_relative_tolerance": 1e-12,
        "metric_regularization": 0.0,
        "energy_regularization": 1e-14,
        "max_fubini_study_step": 10.0,
    }

    whitened = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=JointLinearSolveConfig(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
            **common,
        ),
    )
    legacy = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=JointLinearSolveConfig(
            policy=JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1,
            **common,
        ),
    )

    assert whitened.feasible is True
    assert legacy.feasible is True
    assert whitened.joint_step == pytest.approx(legacy.joint_step, rel=1e-10, abs=1e-12)
    assert whitened.predicted_reduction == pytest.approx(
        legacy.predicted_reduction,
        rel=1e-10,
        abs=1e-12,
    )


def test_well_conditioned_supported_solve_matches_direct_full_system() -> None:
    gram = np.asarray(
        [[1.1, 0.1, 0.0], [0.1, 1.3, 0.05], [0.0, 0.05, 0.8]]
    )
    hessian = np.asarray(
        [[2.0, 0.2, 0.1], [0.2, 1.8, -0.1], [0.1, -0.1, 1.5]]
    )
    gradient = np.asarray([0.3, -0.4, 0.2])
    result = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=2,
        config=_config(
            rank_relative_tolerance=1e-12,
            metric_regularization=0.0,
            energy_regularization=0.0,
            max_fubini_study_step=10.0,
        ),
    )

    expected = np.linalg.solve(hessian, gradient)
    assert result.feasible is True
    assert result.trust_lambda == pytest.approx(0.0, abs=1e-14)
    assert result.joint_step == pytest.approx(expected, rel=1e-10, abs=1e-12)
    assert result.telemetry["full_direct_residual"] < 1e-11
    assert result.telemetry["whitened_solve_residual"] < 1e-11


def test_indefinite_whitened_hessian_is_shifted_and_radius_is_enforced() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([-2.0, 1.0]),
        gradient=np.asarray([1.0, 0.5]),
        active_coordinate_count=1,
        config=_config(max_fubini_study_step=0.2),
    )

    assert result.feasible is True
    assert result.trust_lambda > 2.0
    assert result.telemetry["trust_clipped"] is True
    assert result.telemetry["trust_radius_binding"] is True
    assert result.fubini_study_displacement_sq <= 0.2**2 * (1.0 + 1e-10)
    assert result.telemetry["applied_whitened_condition_number"] >= 1.0


def test_global_trust_v2_completes_pure_hard_case_on_boundary() -> None:
    radius = 0.25
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([-1.0, 2.0]),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            max_fubini_study_step=radius,
        ),
    )

    assert result.feasible is True
    assert result.trust_lambda == pytest.approx(1.0, abs=1e-14)
    assert np.linalg.norm(result.joint_step) == pytest.approx(radius, abs=1e-14)
    assert abs(result.joint_step[0]) == pytest.approx(radius, abs=1e-14)
    assert result.joint_step[1] == pytest.approx(0.0, abs=1e-14)
    assert result.predicted_reduction == pytest.approx(0.5 * radius**2)
    assert result.telemetry["hard_case_detected"] is True
    assert result.telemetry["hard_case_boundary_completion"] is True
    assert result.telemetry["hard_case_selected_sign"] == 1
    np.testing.assert_allclose(
        result.telemetry["hard_case_sign_candidates_joint"],
        [[radius, 0.0], [-radius, 0.0]],
        atol=1e-14,
    )
    assert result.telemetry["trust_global_optimality_certified"] is True
    assert result.telemetry["supported_stationarity_status"] == "stationary"
    assert result.telemetry["supported_inertia_label_issued"] is True
    assert result.telemetry["supported_inertia_status"] == "negative"


def test_global_trust_v2_preserves_positive_definite_interior_solution() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([2.0, 4.0]),
        gradient=np.asarray([0.2, -0.4]),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            max_fubini_study_step=1.0,
        ),
    )

    assert result.feasible is True
    assert result.trust_lambda == pytest.approx(0.0, abs=1e-14)
    assert result.joint_step == pytest.approx([0.1, -0.1], abs=1e-14)
    assert result.telemetry["global_trust_solution_case"] == (
        "positive_semidefinite_interior"
    )
    assert result.telemetry["hard_case_detected"] is False
    assert result.telemetry["trust_radius_binding"] is False
    assert result.telemetry["trust_global_optimality_certified"] is True
    assert result.telemetry["supported_stationarity_status"] == (
        "certified_nonstationary"
    )
    assert result.telemetry["supported_inertia_status"] == "psd"


def test_global_trust_v2_solves_indefinite_nonzero_gradient_secular_root() -> None:
    radius = 0.2
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([-2.0, 1.0]),
        gradient=np.asarray([1.0, 0.5]),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            max_fubini_study_step=radius,
        ),
    )

    assert result.feasible is True
    assert result.trust_lambda > 2.0
    assert np.linalg.norm(result.joint_step) == pytest.approx(radius, abs=1e-12)
    assert result.telemetry["hard_case_detected"] is False
    assert result.telemetry["trust_boundary_root_iterations"] > 0
    assert result.telemetry["trust_radius_binding"] is True
    assert result.telemetry["trust_global_optimality_certified"] is True


def test_global_trust_v2_resolves_near_hard_case_as_regular_root() -> None:
    radius = 0.25
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([-1.0, 2.0]),
        gradient=np.asarray([1.0e-7, 0.0]),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            energy_regularization=1.0e-14,
            max_fubini_study_step=radius,
        ),
    )

    assert result.feasible is True
    assert result.trust_lambda > 1.0
    assert result.joint_step[0] == pytest.approx(radius, abs=1e-10)
    assert result.telemetry["minimum_eigenspace_gradient_norm"] == pytest.approx(
        1.0e-7
    )
    assert result.telemetry["hard_case_detected"] is False
    assert result.telemetry["trust_global_optimality_certified"] is True


def test_global_trust_v2_retains_reflection_when_projection_unresolved() -> None:
    radius = 0.25
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([-1.0, 2.0]),
        gradient=np.asarray([1.0e-10, 0.0]),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            energy_regularization=1.0e-9,
            max_fubini_study_step=radius,
        ),
    )

    assert result.feasible is True
    assert result.telemetry["global_trust_solution_case"] == (
        "boundary_secular_root"
    )
    assert result.telemetry["minimum_eigenspace_cluster_schema"] == (
        "propagated_hessian_spectral_gap_cluster_v1"
    )
    assert result.telemetry["minimum_eigenspace_dimension"] == 1
    assert result.telemetry["minimum_eigenspace_boundary_gap"] == pytest.approx(
        3.0
    )
    assert result.telemetry["minimum_eigenspace_gradient_status"] == (
        "unresolved_from_zero"
    )
    assert result.telemetry["hard_case_detected"] is True
    assert result.telemetry["hard_case_classification"] == (
        "unresolved_minimum_projection_reflection_pair"
    )
    assert result.telemetry[
        "hard_case_uncertain_projection_reflection_retained"
    ] is True
    candidates = np.asarray(
        result.telemetry["hard_case_sign_candidates_joint"],
        dtype=float,
    )
    assert candidates.shape == (2, 2)
    np.testing.assert_allclose(candidates[0], result.joint_step, atol=1.0e-14)
    np.testing.assert_allclose(
        candidates[1],
        np.asarray([-result.joint_step[0], result.joint_step[1]]),
        atol=1.0e-14,
    )
    assert result.telemetry["hard_case_selected_sign"] == 1
    assert result.telemetry[
        "hard_case_point_estimate_optimum_candidate_index"
    ] == 0
    assert result.telemetry["hard_case_sign_candidate_point_estimate_roles"] == [
        "regular_point_estimate_global_optimum",
        "uncertainty_reflection_not_point_estimate_optimum",
    ]
    reductions = result.telemetry[
        "hard_case_sign_candidate_predicted_reductions"
    ]
    assert reductions[0] > reductions[1]
    assert result.telemetry["hard_case_reflection_is_point_estimate_optimum"] is False
    assert result.telemetry["hard_case_reflection_preference_resolved"] is False
    assert result.telemetry["trust_global_optimality_certified"] is True


def test_minimum_hessian_cluster_uses_propagated_error_and_spectral_gap() -> None:
    mask, telemetry = (
        joint_linear_solve_module._minimum_hessian_eigenspace_cluster(
            eigenvalues=np.asarray([-1.0, -0.9985, 2.0]),
            propagated_hessian_error=1.0e-3,
            machine_tolerance=1.0e-12,
        )
    )

    assert mask.tolist() == [True, True, False]
    assert telemetry["minimum_eigenspace_dimension"] == 2
    assert telemetry["minimum_eigenspace_separation_threshold"] == pytest.approx(
        2.000000001e-3
    )
    assert telemetry["minimum_eigenspace_boundary_gap"] == pytest.approx(2.9985)


def test_global_trust_v2_certifies_kkt_residuals_in_supported_metric() -> None:
    result = solve_joint_linear_model(
        gram=np.asarray([[2.0, 0.2], [0.2, 0.8]]),
        hessian=np.asarray([[-1.0, 0.3], [0.3, 1.5]]),
        gradient=np.asarray([0.4, -0.2]),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            rank_relative_tolerance=1.0e-12,
            metric_regularization=0.0,
            max_fubini_study_step=0.3,
        ),
    )

    assert result.feasible is True
    telemetry = result.telemetry
    assert telemetry["trust_kkt_stationarity_residual"] <= telemetry[
        "trust_kkt_stationarity_tolerance"
    ]
    assert telemetry["trust_kkt_primal_violation"] <= telemetry[
        "trust_kkt_primal_tolerance"
    ]
    assert telemetry["trust_kkt_dual_violation"] <= telemetry[
        "trust_kkt_dual_tolerance"
    ]
    assert telemetry["trust_kkt_complementarity_residual"] <= telemetry[
        "trust_kkt_complementarity_tolerance"
    ]
    assert telemetry["trust_kkt_psd_violation"] <= telemetry[
        "trust_kkt_psd_tolerance"
    ]
    assert telemetry["trust_kkt_objective_identity_residual"] <= telemetry[
        "trust_kkt_objective_identity_tolerance"
    ]
    assert telemetry["trust_global_optimality_certified"] is True


def test_global_trust_v2_uses_deterministic_degenerate_hard_direction() -> None:
    radius = 0.4
    result = solve_joint_linear_model(
        gram=np.eye(3),
        hessian=np.diag([-2.0, -2.0, 1.0]),
        gradient=np.zeros(3),
        active_coordinate_count=2,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            max_fubini_study_step=radius,
        ),
    )

    assert result.feasible is True
    assert result.telemetry["minimum_eigenspace_dimension"] == 2
    assert result.telemetry["hard_case_deterministic_direction_whitened"] == (
        pytest.approx([1.0, 0.0, 0.0])
    )
    assert result.joint_step == pytest.approx([radius, 0.0, 0.0])
    assert result.predicted_reduction == pytest.approx(radius**2)
    assert result.telemetry[
        "hard_case_sign_candidate_predicted_reductions"
    ] == pytest.approx([radius**2, radius**2])
    assert result.telemetry["trust_global_optimality_certified"] is True


def test_global_trust_v2_degenerate_hard_case_maximizes_candidate_quotient() -> None:
    radius = 0.3
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=-np.eye(2),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            max_fubini_study_step=radius,
        ),
    )

    assert result.feasible is True
    assert result.telemetry["minimum_eigenspace_dimension"] == 2
    assert result.telemetry["hard_case_orientation_policy"] == (
        "raw_joint_candidate_quotient_max_v1"
    )
    # The old canonical coordinate anchor was e_0, which is entirely active.
    # Candidate attribution instead selects e_1, the unique quotient-maximizer.
    assert result.telemetry["hard_case_orientation_direction_whitened"] == (
        pytest.approx([0.0, 1.0], abs=1.0e-14)
    )
    assert result.telemetry[
        "hard_case_orientation_plus_quotient_fraction"
    ] == pytest.approx(1.0)
    assert result.telemetry[
        "hard_case_orientation_minus_quotient_fraction"
    ] == pytest.approx(1.0)
    assert result.telemetry["hard_case_orientation_exact_signs_retained"] is True
    np.testing.assert_allclose(
        result.telemetry["hard_case_sign_candidates_joint"],
        [[0.0, radius], [0.0, -radius]],
        atol=1.0e-14,
    )
    assert result.joint_step == pytest.approx([0.0, radius], abs=1.0e-14)
    assert result.telemetry["trust_global_optimality_certified"] is True


def test_global_trust_v2_rejects_materially_negative_metric() -> None:
    result = solve_joint_linear_model(
        gram=np.diag([1.0, -1.0e-3]),
        hessian=np.eye(2),
        gradient=np.ones(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
        ),
    )

    assert result.feasible is False
    assert result.reason == "materially_negative_metric_eigenvalue"
    assert result.telemetry["joint_linear_solve_policy_effective"] == (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
    )
    assert result.telemetry.get("hard_case_detected", False) is False
    assert result.telemetry["raw_metric_support_status"] == "invalid_geometry"
    assert "supported_inertia_status" not in result.telemetry


def test_global_trust_v2_rejects_material_primitive_hessian_antisymmetry() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.asarray([[1.0, 0.1], [-0.1, 2.0]]),
        gradient=np.asarray([0.2, -0.1]),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            energy_regularization=1.0e-12,
            max_fubini_study_step=0.25,
        ),
    )

    assert result.feasible is False
    assert result.reason == (
        "primitive_hessian_antisymmetry_exceeds_uncertainty"
    )
    assert result.telemetry["primitive_hessian_symmetry_status"] == (
        "invalid_geometry"
    )
    assert result.telemetry[
        "primitive_hessian_antisymmetric_residual_norm"
    ] == pytest.approx(0.1)
    assert result.telemetry[
        "primitive_hessian_antisymmetric_residual_norm"
    ] > result.telemetry["primitive_hessian_antisymmetric_total_bound"]


def test_v1_keeps_historical_primitive_hessian_symmetrization() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.asarray([[1.0, 0.1], [-0.1, 2.0]]),
        gradient=np.asarray([0.2, -0.1]),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
            energy_regularization=1.0e-12,
            max_fubini_study_step=0.25,
        ),
    )

    assert result.feasible is True
    assert "primitive_hessian_symmetry_status" not in result.telemetry


def test_hard_case_quotient_does_not_resurrect_shared_null_modes() -> None:
    result = solve_joint_linear_model(
        gram=np.diag([4.0e-14, 8.0e-14, 1.0]),
        hessian=np.diag([0.0, 0.0, -1.0]),
        gradient=np.zeros(3),
        active_coordinate_count=2,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            energy_regularization=1.0e-12,
            max_fubini_study_step=0.25,
        ),
    )

    assert result.feasible is True
    assert result.telemetry["hard_case_orientation_support_status"] == (
        "resolved"
    )
    assert result.telemetry[
        "hard_case_orientation_shared_metric_provenance_id"
    ] == result.telemetry["supported_metric_whitening_provenance_id"]
    assert result.telemetry[
        "hard_case_orientation_active_metric_eigenvalues"
    ] == pytest.approx([0.0, 0.0], abs=1.0e-28)
    assert result.telemetry[
        "hard_case_orientation_active_projection_rank"
    ] == 0


def test_global_trust_v2_resolves_zero_straddling_mode_by_stable_support_gap() -> None:
    result = solve_joint_linear_model(
        gram=np.diag([1.0, 0.0]),
        hessian=np.diag([1.0, 0.0]),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
        ),
    )

    epsilon_G = result.telemetry["raw_metric_support_epsilon_G"]
    assert result.feasible is True
    assert epsilon_G > 0.0
    assert result.telemetry["raw_metric_minimum_eigenvalue_lower_bound"] < 0.0
    assert result.telemetry["raw_metric_minimum_eigenvalue_upper_bound"] > 0.0
    assert result.telemetry["raw_metric_support_status"] == "resolved"
    assert result.telemetry["metric_support_rank"] == 1
    assert result.telemetry["metric_retained_mask"] == [False, True]
    assert result.telemetry["raw_metric_support_cluster_gap"] == pytest.approx(
        1.0
    )
    assert result.telemetry["raw_metric_support_rotation_bound"] == (
        pytest.approx(epsilon_G / (1.0 - epsilon_G))
    )
    assert result.telemetry[
        "metric_support_selection_is_relative_budget_not_cutoff"
    ] is True


def test_global_trust_v2_support_gap_at_uncertainty_scale_is_unresolved() -> None:
    gram = np.diag([1.0, 1.0 + 4.0e-14])
    common = {
        "gram": gram,
        "hessian": np.eye(2),
        "gradient": np.zeros(2),
        "active_coordinate_count": 1,
    }
    calibration = solve_joint_linear_model(
        **common,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            rank_relative_tolerance=1.0,
            metric_regularization=0.0,
        ),
    )
    epsilon_G = calibration.telemetry["raw_metric_support_epsilon_G"]
    eigenvalues = np.linalg.eigvalsh(gram)
    full_relative_bound = epsilon_G / (eigenvalues[0] - epsilon_G)
    top_relative_bound = epsilon_G / (eigenvalues[1] - epsilon_G)
    eta_G = 0.5 * (full_relative_bound + top_relative_bound)

    result = solve_joint_linear_model(
        **common,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            rank_relative_tolerance=eta_G,
            metric_regularization=0.0,
        ),
    )

    assert result.feasible is False
    assert result.reason == "raw_metric_support_unresolved"
    assert result.telemetry["raw_metric_support_status"] == "unresolved"
    assert "metric_support_rank" not in result.telemetry
    checks = result.telemetry["raw_metric_support_candidate_checks"]
    assert checks[0]["within_eta_G_budget"] is False
    assert checks[1]["within_eta_G_budget"] is True
    assert checks[1]["gap_exceeds_twice_epsilon_G"] is False
    assert "H_w_eigenvalues" not in result.telemetry
    assert "supported_inertia_status" not in result.telemetry


def test_global_trust_v2_all_null_raw_metric_is_support_unresolved() -> None:
    result = solve_joint_linear_model(
        gram=np.zeros((2, 2)),
        hessian=np.zeros((2, 2)),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
        ),
    )

    assert result.feasible is False
    assert result.reason == "raw_metric_support_unresolved"
    assert result.telemetry["raw_metric_support_status"] == "unresolved"
    assert "metric_support_rank" not in result.telemetry
    assert "supported_inertia_status" not in result.telemetry


def test_global_trust_v2_derives_minimum_kkt_ridge_and_ignores_v1_fixed_ridge() -> None:
    gram = np.diag([1.0, 1.0e-4])
    common = {
        "gram": gram,
        "hessian": np.eye(2),
        "gradient": np.asarray([0.2, -0.1]),
        "active_coordinate_count": 1,
    }
    config_values = {
        "policy": JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
        "rank_relative_tolerance": 1.0,
        "energy_regularization": 1.0e-12,
        "max_fubini_study_step": 10.0,
        "global_trust_kkt_residual_accuracy": 1.0e-10,
        "global_trust_metric_distortion_budget": 0.8,
    }
    zero_legacy_ridge = solve_joint_linear_model(
        **common,
        config=JointLinearSolveConfig(
            **config_values,
            metric_regularization=0.0,
        ),
    )
    large_legacy_ridge = solve_joint_linear_model(
        **common,
        config=JointLinearSolveConfig(
            **config_values,
            metric_regularization=0.5,
        ),
    )

    assert zero_legacy_ridge.feasible is True
    assert large_legacy_ridge.feasible is True
    telemetry = zero_legacy_ridge.telemetry
    kappa_target = telemetry["metric_stabilization_target_condition_number"]
    expected_ridge = max(
        0.0,
        (1.0 - kappa_target * 1.0e-4) / (kappa_target - 1.0),
    )
    expected_condition = (1.0 + expected_ridge) / (
        1.0e-4 + expected_ridge
    )
    assert telemetry["metric_stabilization_lambda_G"] == pytest.approx(
        expected_ridge
    )
    assert telemetry["metric_whitening_ridge"] == pytest.approx(expected_ridge)
    assert telemetry["metric_stabilization_raw_supported_condition_number"] == (
        pytest.approx(1.0e4)
    )
    assert telemetry["metric_stabilization_stabilized_condition_number"] == (
        pytest.approx(expected_condition)
    )
    assert telemetry["metric_stabilization_stabilized_condition_number"] == (
        pytest.approx(kappa_target)
    )
    assert telemetry["metric_stabilization_distortion_lower_bound"] == (
        pytest.approx(expected_ridge / (1.0 + expected_ridge))
    )
    assert telemetry["metric_stabilization_distortion_upper_bound"] == (
        pytest.approx(expected_ridge / (1.0e-4 + expected_ridge))
    )
    assert telemetry["metric_stabilization_status"] == "resolved"
    assert telemetry[
        "metric_stabilization_fixed_metric_regularization_applied"
    ] is False
    assert large_legacy_ridge.telemetry["metric_whitening_ridge"] == (
        pytest.approx(expected_ridge)
    )
    assert large_legacy_ridge.telemetry[
        "supported_metric_whitening_provenance_id"
    ] == telemetry["supported_metric_whitening_provenance_id"]
    np.testing.assert_allclose(
        large_legacy_ridge.joint_step,
        zero_legacy_ridge.joint_step,
        rtol=0.0,
        atol=0.0,
    )


def test_global_trust_v2_unridged_congruence_preserves_step_and_certificates() -> None:
    gram = np.asarray([[1.5, 0.2], [0.2, 0.8]])
    hessian = np.asarray([[2.0, -0.1], [-0.1, 1.2]])
    gradient = np.asarray([0.4, -0.3])
    coordinate_map = np.diag([3.0, 0.4])
    config = _config(
        policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
        rank_relative_tolerance=1.0e-10,
        metric_regularization=0.7,
        max_fubini_study_step=10.0,
    )

    original = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=config,
    )
    rescaled = solve_joint_linear_model(
        gram=coordinate_map.T @ gram @ coordinate_map,
        hessian=coordinate_map.T @ hessian @ coordinate_map,
        gradient=coordinate_map.T @ gradient,
        active_coordinate_count=1,
        config=config,
    )

    assert original.feasible is True
    assert rescaled.feasible is True
    assert original.telemetry["metric_stabilization_lambda_G"] == 0.0
    assert rescaled.telemetry["metric_stabilization_lambda_G"] == 0.0
    np.testing.assert_allclose(
        coordinate_map @ rescaled.joint_step,
        original.joint_step,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert rescaled.predicted_reduction == pytest.approx(
        original.predicted_reduction,
        rel=1.0e-10,
        abs=1.0e-12,
    )
    assert rescaled.fubini_study_displacement_sq == pytest.approx(
        original.fubini_study_displacement_sq,
        rel=1.0e-10,
        abs=1.0e-12,
    )
    assert rescaled.telemetry["supported_stationarity_status"] == (
        original.telemetry["supported_stationarity_status"]
    )
    assert rescaled.telemetry["supported_inertia_status"] == (
        original.telemetry["supported_inertia_status"]
    )
    assert rescaled.telemetry["trust_global_optimality_certified"] is True
    assert rescaled.telemetry["supported_stationarity_status"] == (
        original.telemetry["supported_stationarity_status"]
    )
    assert rescaled.telemetry["supported_inertia_status"] == (
        original.telemetry["supported_inertia_status"]
    )


@pytest.mark.parametrize("coordinate_scale", (1.0e-2, 1.0e-4, 1.0e-6))
def test_global_trust_v2_rescaling_uses_certified_zero_ridge_fallback(
    coordinate_scale: float,
) -> None:
    coordinate_map = np.diag([1.0, coordinate_scale])
    gram = np.eye(2)
    hessian = np.eye(2)
    gradient = np.asarray([0.2, -0.1])
    config = _config(
        policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
        rank_relative_tolerance=1.0,
        metric_regularization=0.5,
        global_trust_kkt_residual_accuracy=1.0e-10,
        global_trust_metric_distortion_budget=1.0e-2,
    )

    original = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=config,
    )
    rescaled = solve_joint_linear_model(
        gram=coordinate_map.T @ gram @ coordinate_map,
        hessian=coordinate_map.T @ hessian @ coordinate_map,
        gradient=coordinate_map.T @ gradient,
        active_coordinate_count=1,
        config=config,
    )

    assert original.feasible is True
    assert original.telemetry["metric_stabilization_lambda_G"] == 0.0
    assert rescaled.feasible is True
    assert rescaled.telemetry["raw_metric_support_status"] == "resolved"
    assert rescaled.telemetry["metric_stabilization_status"] == "resolved"
    assert rescaled.telemetry["metric_stabilization_proposed_lambda_G"] > 0.0
    assert rescaled.telemetry[
        "metric_stabilization_proposed_condition_target_met"
    ] is True
    assert rescaled.telemetry[
        "metric_stabilization_proposed_distortion_upper_bound"
    ] > rescaled.telemetry["metric_stabilization_distortion_budget"]
    assert rescaled.telemetry["metric_stabilization_lambda_G"] == 0.0
    assert rescaled.telemetry[
        "metric_stabilization_condition_target_met"
    ] is False
    assert rescaled.telemetry[
        "metric_stabilization_distortion_upper_bound"
    ] == 0.0
    assert rescaled.telemetry[
        "metric_stabilization_zero_ridge_fallback_eligible"
    ] is True
    assert rescaled.telemetry[
        "metric_stabilization_zero_ridge_fallback_attempted"
    ] is True
    assert rescaled.telemetry[
        "metric_stabilization_zero_ridge_fallback_solver_status"
    ] == "certified"
    assert rescaled.telemetry[
        "metric_stabilization_zero_ridge_fallback_solver_certified"
    ] is True
    assert rescaled.telemetry["raw_metric_null_compatibility_certified"] is True
    assert rescaled.telemetry["trust_global_optimality_certified"] is True
    for residual_field, tolerance_field in (
        (
            "trust_kkt_stationarity_residual",
            "trust_kkt_stationarity_tolerance",
        ),
        ("trust_kkt_primal_violation", "trust_kkt_primal_tolerance"),
        ("trust_kkt_dual_violation", "trust_kkt_dual_tolerance"),
        (
            "trust_kkt_complementarity_residual",
            "trust_kkt_complementarity_tolerance",
        ),
        ("trust_kkt_psd_violation", "trust_kkt_psd_tolerance"),
        (
            "trust_kkt_objective_identity_residual",
            "trust_kkt_objective_identity_tolerance",
        ),
    ):
        assert rescaled.telemetry[residual_field] <= rescaled.telemetry[
            tolerance_field
        ]
    np.testing.assert_allclose(
        coordinate_map @ rescaled.joint_step,
        original.joint_step,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert rescaled.predicted_reduction == pytest.approx(
        original.predicted_reduction,
        rel=1.0e-12,
        abs=1.0e-14,
    )
    assert rescaled.fubini_study_displacement_sq == pytest.approx(
        original.fubini_study_displacement_sq,
        rel=1.0e-12,
        abs=1.0e-14,
    )


def test_global_trust_v2_zero_ridge_fallback_rejects_failed_kkt_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        joint_linear_solve_module,
        "_canonical_eigenspace_direction",
        lambda _eigenspace: np.asarray([1.0, 0.0]),
    )
    result = solve_joint_linear_model(
        gram=np.diag([1.0, 1.0e-4]),
        hessian=np.diag([-1.0, 2.0e-4]),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            rank_relative_tolerance=1.0,
            metric_regularization=0.5,
            max_fubini_study_step=0.25,
            global_trust_kkt_residual_accuracy=1.0e-10,
            global_trust_metric_distortion_budget=1.0e-2,
        ),
    )

    assert result.feasible is False
    assert result.reason == "global_trust_kkt_certificate_failed"
    assert result.telemetry[
        "metric_stabilization_zero_ridge_fallback_eligible"
    ] is True
    assert result.telemetry[
        "metric_stabilization_zero_ridge_fallback_attempted"
    ] is True
    assert result.telemetry[
        "metric_stabilization_zero_ridge_fallback_solver_status"
    ] == "rejected"
    assert result.telemetry[
        "metric_stabilization_zero_ridge_fallback_solver_certified"
    ] is False
    assert result.telemetry[
        "metric_stabilization_zero_ridge_fallback_solver_reason"
    ] == "global_trust_kkt_certificate_failed"
    assert result.telemetry["trust_global_optimality_certified"] is False
    assert result.telemetry["supported_inertia_label_issued"] is False
    assert "supported_inertia_status" not in result.telemetry


@pytest.mark.parametrize(
    ("hessian", "gradient", "expected_reason"),
    (
        (
            np.diag([1.0, 1.0e-4, 0.0]),
            np.asarray([0.2, -1.0e-3, 1.0e-4]),
            "raw_metric_null_gradient_incompatible",
        ),
        (
            np.diag([1.0, 1.0e-4, -1.0]),
            np.asarray([0.2, -1.0e-3, 0.0]),
            "raw_metric_null_hessian_incompatible",
        ),
    ),
)
def test_global_trust_v2_zero_ridge_fallback_does_not_bypass_null_gate(
    hessian: np.ndarray,
    gradient: np.ndarray,
    expected_reason: str,
) -> None:
    result = solve_joint_linear_model(
        gram=np.diag([1.0, 1.0e-4, 1.0e-20]),
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            rank_relative_tolerance=1.0,
            metric_regularization=0.5,
            global_trust_kkt_residual_accuracy=1.0e-10,
            global_trust_metric_distortion_budget=1.0e-2,
        ),
    )

    assert result.feasible is False
    assert result.reason == expected_reason
    assert result.telemetry["raw_metric_support_status"] == "resolved"
    assert result.telemetry[
        "metric_stabilization_zero_ridge_fallback_eligible"
    ] is True
    assert result.telemetry[
        "metric_stabilization_zero_ridge_fallback_attempted"
    ] is False
    assert result.telemetry[
        "metric_stabilization_zero_ridge_fallback_solver_status"
    ] == "not_attempted_raw_metric_null_incompatible"
    assert result.telemetry["raw_metric_null_compatibility_certified"] is False
    assert "trust_global_optimality_certified" not in result.telemetry
    assert "supported_inertia_status" not in result.telemetry


def test_global_trust_v2_kkt_accuracy_below_arithmetic_floor_is_unresolved() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.eye(2),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            global_trust_kkt_residual_accuracy=1.0e-16,
        ),
    )

    assert result.feasible is False
    assert result.reason == "kkt_accuracy_below_arithmetic_floor"
    assert result.telemetry["raw_metric_support_status"] == "unresolved"
    assert result.telemetry["metric_stabilization_status"] == "unresolved"
    assert result.telemetry["metric_stabilization_target_condition_number"] <= 1.0
    assert result.telemetry["metric_stabilization_lambda_G"] is None
    assert "supported_inertia_status" not in result.telemetry


def test_global_trust_v2_stationarity_interval_straddling_is_unresolved() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.eye(2),
        gradient=np.asarray([1.0e-6, 0.0]),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            energy_regularization=1.0e-6,
            max_fubini_study_step=1.0,
        ),
    )

    assert result.feasible is True
    assert result.telemetry["supported_gradient_resolution"] == pytest.approx(
        1.0e-6
    )
    assert result.telemetry["supported_gradient_norm_lower_bound"] < 1.0e-6
    assert result.telemetry["supported_gradient_norm_upper_bound"] > 1.0e-6
    assert result.telemetry["supported_stationarity_status"] == "unresolved"
    assert result.telemetry["supported_inertia_status"] == "psd"


def test_global_trust_v2_inertia_interval_straddling_zero_is_unresolved() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([1.0, 1.0e-15]),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
        ),
    )

    assert result.feasible is True
    assert result.telemetry["supported_stationarity_status"] == "stationary"
    assert result.telemetry["supported_hessian_eigenvalue_lower_bounds"][0] < 0.0
    assert result.telemetry["supported_hessian_eigenvalue_upper_bounds"][0] > 0.0
    assert result.telemetry["supported_inertia_status"] == "unresolved"
    assert result.telemetry["supported_hessian_eigenvalue_statuses"][0] == (
        "unresolved"
    )


def test_global_trust_v2_kkt_failure_issues_no_inertia_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        joint_linear_solve_module,
        "_canonical_eigenspace_direction",
        lambda _eigenspace: np.asarray([0.0, 1.0]),
    )
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([-1.0, 2.0]),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            metric_regularization=0.0,
            max_fubini_study_step=0.25,
        ),
    )

    assert result.feasible is False
    assert result.reason == "global_trust_kkt_certificate_failed"
    assert result.telemetry["trust_global_optimality_certified"] is False
    assert result.telemetry["supported_inertia_label_issued"] is False
    assert "supported_inertia_status" not in result.telemetry
    assert "supported_hessian_eigenvalue_statuses" not in result.telemetry


def test_global_trust_v2_rejects_curvature_in_raw_metric_null_direction() -> None:
    config = _config(
        policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
        rank_relative_tolerance=1.0e-6,
        metric_regularization=1.0e-12,
        energy_regularization=1.0e-12,
    )
    result = solve_joint_linear_model(
        gram=np.diag([1.0, 1.0e-14]),
        hessian=np.diag([1.0, -1.0]),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=config,
    )

    assert result.feasible is False
    assert result.reason == "raw_metric_null_hessian_incompatible"
    assert result.joint_step == pytest.approx([0.0, 0.0], abs=0.0)
    assert result.telemetry["metric_retained_mask"] == [False, True]
    assert result.telemetry["raw_metric_discarded_support_dimension"] == 1
    assert result.telemetry["raw_metric_null_gradient_compatible"] is True
    assert result.telemetry["raw_metric_null_hessian_compatible"] is False
    assert result.telemetry["raw_metric_null_null_hessian_norm"] == pytest.approx(
        1.0
    )
    assert result.telemetry["raw_metric_null_hessian_residual_norm"] == (
        pytest.approx(1.0)
    )
    assert result.telemetry["raw_metric_null_compatibility_certified"] is False
    assert result.telemetry.get("hard_case_detected", False) is False


def test_global_trust_v2_rejects_gradient_in_raw_metric_null_direction() -> None:
    result = solve_joint_linear_model(
        gram=np.diag([1.0, 1.0e-14]),
        hessian=np.diag([1.0, 0.0]),
        gradient=np.asarray([0.0, 1.0e-4]),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            rank_relative_tolerance=1.0e-6,
            metric_regularization=1.0e-12,
            energy_regularization=1.0e-12,
        ),
    )

    assert result.feasible is False
    assert result.reason == "raw_metric_null_gradient_incompatible"
    assert result.telemetry["raw_metric_null_gradient_residual_norm"] == (
        pytest.approx(1.0e-4)
    )
    assert result.telemetry["raw_metric_null_gradient_compatible"] is False
    assert result.telemetry["raw_metric_null_hessian_compatible"] is True
    assert result.telemetry["raw_metric_null_compatibility_certified"] is False


def test_global_trust_v2_rejects_support_null_hessian_coupling() -> None:
    result = solve_joint_linear_model(
        gram=np.diag([1.0, 1.0e-14]),
        hessian=np.asarray([[1.0, 0.25], [0.25, 0.0]]),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            rank_relative_tolerance=1.0e-6,
            metric_regularization=1.0e-12,
            energy_regularization=1.0e-12,
        ),
    )

    assert result.feasible is False
    assert result.reason == "raw_metric_null_hessian_incompatible"
    assert result.telemetry[
        "raw_metric_support_null_hessian_coupling_norm"
    ] == pytest.approx(0.25)
    assert result.telemetry["raw_metric_null_null_hessian_norm"] == (
        pytest.approx(0.0)
    )
    assert result.telemetry["raw_metric_null_hessian_compatible"] is False


def test_raw_metric_null_compatibility_gate_does_not_change_v1_policy() -> None:
    result = solve_joint_linear_model(
        gram=np.diag([1.0, 1.0e-14]),
        hessian=np.diag([1.0, -1.0]),
        gradient=np.zeros(2),
        active_coordinate_count=1,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
            rank_relative_tolerance=1.0e-6,
            metric_regularization=1.0e-12,
            energy_regularization=1.0e-12,
        ),
    )

    assert result.feasible is True
    assert result.reason == "supported_metric_whitened_eigh_solve"
    assert "raw_metric_null_compatibility_certified" not in result.telemetry


def test_coordinate_rescaling_preserves_physical_step_and_gain() -> None:
    gram = np.asarray([[1.5, 0.2], [0.2, 0.8]])
    hessian = np.asarray([[2.0, -0.1], [-0.1, 1.2]])
    gradient = np.asarray([0.4, -0.3])
    scale = np.diag([3.0, 0.4])
    config = _config(
        rank_relative_tolerance=1e-12,
        metric_regularization=0.0,
        max_fubini_study_step=10.0,
    )

    original = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=config,
    )
    rescaled = solve_joint_linear_model(
        gram=scale.T @ gram @ scale,
        hessian=scale.T @ hessian @ scale,
        gradient=scale.T @ gradient,
        active_coordinate_count=1,
        config=config,
    )

    assert original.feasible is True
    assert rescaled.feasible is True
    assert scale @ rescaled.joint_step == pytest.approx(
        original.joint_step,
        rel=1e-9,
        abs=1e-11,
    )
    assert rescaled.predicted_reduction == pytest.approx(
        original.predicted_reduction,
        rel=1e-10,
        abs=1e-12,
    )


def test_active_batch_partition_and_query_charge_are_explicit() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(3),
        hessian=np.diag([2.0, 3.0, 4.0]),
        gradient=np.asarray([0.2, -0.3, 0.4]),
        active_coordinate_count=2,
        config=_config(),
    )

    assert result.feasible is True
    assert result.active_parameter_relaxation == pytest.approx(result.joint_step[:2])
    assert result.batch_coordinate_step == pytest.approx(result.joint_step[2:])
    assert result.telemetry["active_coordinate_count"] == 2
    assert result.telemetry["batch_coordinate_count"] == 1
    assert result.telemetry["classical_quantum_query_charge"] == 0


def test_invalid_shapes_and_policy_fail_closed() -> None:
    with pytest.raises(ValueError, match="policy"):
        JointLinearSolveConfig(policy="unknown")
    with pytest.raises(ValueError, match="square"):
        solve_joint_linear_model(
            gram=np.ones((2, 3)),
            hessian=np.eye(2),
            gradient=np.ones(2),
            active_coordinate_count=1,
        )
