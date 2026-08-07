from __future__ import annotations

import numpy as np
import pytest

import pipelines.static_adapt.ra_adapt.support as support_module
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
    JointLinearSolveConfig,
    factor_supported_metric,
    solve_joint_linear_model,
)
from pipelines.static_adapt.ra_adapt.support import factor_retained_support


def test_canonical_support_wrapper_reuses_existing_factorization_exactly() -> None:
    rng = np.random.default_rng(20260727)
    basis = rng.normal(size=(5, 5))
    gram = basis.T @ basis + np.diag([1.0, 0.5, 0.2, 0.1, 0.05])
    tolerance = 2.0e-10
    ridge = 3.0e-9

    direct = factor_supported_metric(
        gram,
        rank_relative_tolerance=tolerance,
        metric_regularization=ridge,
    )
    canonical = factor_retained_support(
        gram,
        rank_relative_tolerance=tolerance,
        metric_regularization=ridge,
        source_provenance_id="selector-source-gram",
    )

    np.testing.assert_array_equal(
        canonical.factorization.raw_eigenvalues,
        direct.raw_eigenvalues,
    )
    np.testing.assert_array_equal(
        canonical.factorization.retained_mask,
        direct.retained_mask,
    )
    np.testing.assert_array_equal(
        canonical.factorization.retained_eigenvalues,
        direct.retained_eigenvalues,
    )
    np.testing.assert_array_equal(
        canonical.factorization.retained_vectors,
        direct.retained_vectors,
    )
    receipt = canonical.receipt.as_dict()
    assert receipt["support_threshold"] == pytest.approx(
        direct.support_threshold
    )
    assert receipt["negative_eigenvalue_tolerance"] == pytest.approx(
        direct.negative_eigenvalue_tolerance
    )
    assert receipt["factorization_provenance_id"] == direct.provenance_id
    assert receipt["source_provenance_id"] == "selector-source-gram"
    assert receipt["classical_quantum_query_charge"] == 0
    assert len(receipt["retained_eigenpairs"]) == direct.rank


def test_canonical_support_receipt_is_deterministic_and_source_scoped() -> None:
    gram = np.diag([4.0, 1.0, 1.0e-12])
    first = factor_retained_support(
        gram,
        rank_relative_tolerance=1.0e-9,
        metric_regularization=0.0,
        source_provenance_id="selector-window-a",
    )
    repeated = factor_retained_support(
        gram,
        rank_relative_tolerance=1.0e-9,
        metric_regularization=0.0,
        source_provenance_id="selector-window-a",
    )
    other_source = factor_retained_support(
        gram,
        rank_relative_tolerance=1.0e-9,
        metric_regularization=0.0,
        source_provenance_id="accepted-refit-full-ansatz",
    )

    assert first.receipt.retained_mask == (False, True, True)
    assert (
        first.receipt.receipt_provenance_id
        == repeated.receipt.receipt_provenance_id
    )
    assert (
        first.receipt.factorization_provenance_id
        == other_source.receipt.factorization_provenance_id
    )
    assert (
        first.receipt.receipt_provenance_id
        != other_source.receipt.receipt_provenance_id
    )


def _projected_config(*, radius: float) -> JointLinearSolveConfig:
    return JointLinearSolveConfig(
        policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
        rank_relative_tolerance=1.0e-12,
        metric_regularization=0.0,
        energy_regularization=1.0e-6,
        max_fubini_study_step=radius,
    )


def test_projected_solver_reports_curvature_only_as_interior_trust() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([-1.0, 2.0]),
        gradient=np.asarray([0.0, 0.1]),
        active_coordinate_count=1,
        config=_projected_config(radius=1.0),
    )

    assert result.feasible is True
    telemetry = result.telemetry
    assert telemetry["kappa_stabilization_shift"] > 0.0
    assert telemetry["total_metric_multiplier_mu"] == pytest.approx(
        telemetry["kappa_stabilization_shift"]
    )
    assert telemetry["trust_boundary_multiplier_lambda"] == pytest.approx(0.0)
    assert telemetry["trust_boundary_active"] is False
    assert telemetry["trust_regularization_applied"] is False
    assert telemetry["legacy_total_metric_regularization_applied"] is True
    assert result.trust_lambda == pytest.approx(
        telemetry["total_metric_multiplier_mu"]
    )


def test_projected_solver_separates_boundary_increment_from_curvature_shift() -> None:
    result = solve_joint_linear_model(
        gram=np.eye(2),
        hessian=np.diag([-1.0, 2.0]),
        gradient=np.asarray([0.1, 0.1]),
        active_coordinate_count=1,
        config=_projected_config(radius=0.05),
    )

    assert result.feasible is True
    telemetry = result.telemetry
    kappa = float(telemetry["kappa_stabilization_shift"])
    boundary_lambda = float(telemetry["trust_boundary_multiplier_lambda"])
    total_mu = float(telemetry["total_metric_multiplier_mu"])
    assert kappa > 0.0
    assert boundary_lambda > 0.0
    assert total_mu == pytest.approx(kappa + boundary_lambda)
    assert result.trust_lambda == pytest.approx(total_mu)
    assert telemetry["trust_boundary_active"] is True
    assert telemetry["trust_regularization_applied"] is True
    assert telemetry["trust_radius_binding"] is True


def test_projected_solver_calls_canonical_support_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original = support_module.factor_retained_support

    def _recording_factorization(
        gram: np.ndarray,
        **kwargs: object,
    ) -> object:
        calls.append(
            {
                "gram": np.asarray(gram, dtype=float).copy(),
                **kwargs,
            }
        )
        return original(gram, **kwargs)

    monkeypatch.setattr(
        support_module,
        "factor_retained_support",
        _recording_factorization,
    )
    result = solve_joint_linear_model(
        gram=np.diag([3.0, 1.0]),
        hessian=np.diag([1.0, 2.0]),
        gradient=np.asarray([0.1, 0.2]),
        active_coordinate_count=1,
        config=_projected_config(radius=0.25),
    )

    assert result.feasible is True
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["gram"], np.diag([3.0, 1.0]))
    assert calls[0]["metric_regularization"] == 0.0
    receipt = result.telemetry["retained_support_receipt"]
    assert receipt["schema"] == "ra_adapt_retained_support_receipt_v1"
    assert receipt["metric_regularization"] == 0.0
    assert receipt["factorization_provenance_id"] == (
        result.telemetry["supported_metric_projection_provenance_id"]
    )
