from __future__ import annotations

import numpy as np
import pytest

import pipelines.static_adapt.joint_linear_solve as joint_linear_solve_module
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
    JointLinearSolveConfig,
    solve_joint_linear_model,
)


def _config(
    *,
    policy: str = (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
    ),
    radius: float = 0.25,
    metric_regularization: float = 0.0,
) -> JointLinearSolveConfig:
    return JointLinearSolveConfig(
        policy=policy,
        rank_relative_tolerance=1.0e-12,
        metric_regularization=metric_regularization,
        energy_regularization=1.0e-14,
        max_fubini_study_step=radius,
    )


def test_projected_generalized_solver_removes_gram_null_direction() -> None:
    result = solve_joint_linear_model(
        gram=np.diag([0.0, 4.0]),
        hessian=np.diag([2.0, 3.0]),
        gradient=np.asarray([1.0, 0.4]),
        active_coordinate_count=1,
        config=_config(radius=10.0, metric_regularization=0.5),
    )

    assert result.feasible is True
    assert result.telemetry["metric_retained_mask"] == [False, True]
    assert result.telemetry["metric_support_rank"] == 1
    assert result.joint_step[0] == pytest.approx(0.0, abs=1.0e-15)
    assert result.joint_step[1] == pytest.approx(0.4 / 3.0)
    assert result.telemetry["discarded_gradient_norm"] == pytest.approx(1.0)
    assert result.telemetry["metric_regularization_applied"] is False
    assert result.telemetry["metric_regularization_configured_inactive"] == (
        pytest.approx(0.5)
    )


def test_projected_generalized_solver_enforces_fs_bound_and_supported_kkt() -> None:
    gram = np.asarray([[1.2, 0.1], [0.1, 0.9]])
    hessian = np.asarray([[2.0, 0.2], [0.2, 1.4]])
    gradient = np.asarray([0.3, -0.2])
    radius = 0.12

    result = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=_config(radius=radius),
    )

    assert result.feasible is True
    assert result.trust_lambda > 0.0
    assert result.telemetry["trust_clipped"] is True
    assert result.telemetry["trust_radius_binding"] is True
    assert result.fubini_study_displacement_sq == pytest.approx(
        radius**2,
        rel=2.0e-9,
        abs=2.0e-12,
    )
    assert result.telemetry["supported_metric_displacement_sq"] == pytest.approx(
        radius**2,
        rel=2.0e-9,
        abs=2.0e-12,
    )
    assert result.telemetry["supported_generalized_kkt_residual"] < 1.0e-11
    np.testing.assert_allclose(
        (hessian + result.trust_lambda * gram) @ result.joint_step,
        gradient,
        rtol=0.0,
        atol=1.0e-11,
    )


def test_projected_generalized_solver_never_calls_whitening_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden_whitening(*args: object, **kwargs: object) -> object:
        raise AssertionError("projected generalized solve called whitening factor")

    monkeypatch.setattr(
        joint_linear_solve_module,
        "factor_supported_metric",
        _forbidden_whitening,
    )
    result = solve_joint_linear_model(
        gram=np.asarray([[1.0, 0.1], [0.1, 0.8]]),
        hessian=np.asarray([[1.5, 0.2], [0.2, 1.2]]),
        gradient=np.asarray([0.2, -0.1]),
        active_coordinate_count=1,
        config=_config(radius=0.2),
    )

    assert result.feasible is True
    assert result.telemetry["supported_metric_projection_active"] is True
    assert result.telemetry["supported_metric_whitening_active"] is False
    assert result.telemetry["supported_metric_inverse_sqrt_constructed"] is False
    assert result.telemetry["supported_metric_inverse_constructed"] is False


def test_projected_generalized_matches_whitened_solver_without_metric_ridge() -> None:
    gram = np.asarray([[1.2, 0.1], [0.1, 0.9]])
    hessian = np.asarray([[2.0, 0.2], [0.2, 1.4]])
    gradient = np.asarray([0.3, -0.2])
    common = {
        "gram": gram,
        "hessian": hessian,
        "gradient": gradient,
        "active_coordinate_count": 1,
    }

    projected = solve_joint_linear_model(
        **common,
        config=_config(radius=0.12),
    )
    whitened = solve_joint_linear_model(
        **common,
        config=_config(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
            radius=0.12,
        ),
    )

    assert projected.feasible is True
    assert whitened.feasible is True
    np.testing.assert_allclose(
        projected.joint_step,
        whitened.joint_step,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert projected.trust_lambda == pytest.approx(
        whitened.trust_lambda,
        rel=1.0e-11,
        abs=1.0e-12,
    )
    assert projected.predicted_reduction == pytest.approx(
        whitened.predicted_reduction,
        rel=1.0e-11,
        abs=1.0e-12,
    )


def test_projected_generalized_solution_is_coordinate_rescaling_invariant() -> None:
    gram = np.asarray([[1.4, 0.2], [0.2, 0.8]])
    hessian = np.asarray([[2.0, 0.1], [0.1, 1.3]])
    gradient = np.asarray([0.5, -0.3])
    coordinate_map = np.diag([3.0, 0.25])
    config = _config(radius=0.18, metric_regularization=0.1)

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
    np.testing.assert_allclose(
        coordinate_map @ rescaled.joint_step,
        original.joint_step,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert rescaled.trust_lambda == pytest.approx(original.trust_lambda)
    assert rescaled.fubini_study_displacement_sq == pytest.approx(
        original.fubini_study_displacement_sq
    )
    assert rescaled.predicted_reduction == pytest.approx(
        original.predicted_reduction
    )


def test_projected_generalized_solver_fails_closed_on_negative_gram() -> None:
    result = solve_joint_linear_model(
        gram=np.diag([1.0, -1.0e-3]),
        hessian=np.eye(2),
        gradient=np.ones(2),
        active_coordinate_count=1,
        config=_config(radius=0.2),
    )

    assert result.feasible is False
    assert result.reason == "materially_negative_metric_eigenvalue"
    assert result.telemetry["supported_metric_projection_active"] is True
    assert result.telemetry["supported_metric_whitening_active"] is False
    assert result.telemetry["classical_quantum_query_charge"] == 0
