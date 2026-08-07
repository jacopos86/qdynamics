from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from paper5.stability.apcm_positive_extension import (
    APCMExtensionSettings,
    SymmetryReducedPositiveExtension,
)
from paper5.stability.adaptive_positive_moment import (
    HIDDEN_RELATIVE_MOMENT_KEYS,
    matrix_state_to_raw_moment_coordinates,
    uncentered_joint_moment_matrix,
)
from paper5.stability.apcm_moment_projection import (
    APCMProjectionError,
    SymmetryReducedAPCMGeometry,
    state_lower_moments,
)
from paper5.stability.moment_hierarchy import (
    IDENTITY,
    MomentKey,
    THIRD_ORDER_HIERARCHY,
)


def _thermal_weyl_moment(key: MomentKey) -> float:
    if key.spin_up != IDENTITY or key.spin_down != IDENTITY:
        return 0.0
    if key.x_power % 2 or key.p_power % 2:
        return 0.0

    def odd_double_factorial(power: int) -> int:
        return int(np.prod(np.arange(1, power, 2))) if power else 1

    return float(
        odd_double_factorial(key.x_power)
        * odd_double_factorial(key.p_power)
    )


def test_dense_logdet_completion_is_rejected_before_solver_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The former path exhausted tens of GB while CVXPY canonicalized it."""

    extension = SymmetryReducedPositiveExtension(
        APCMExtensionSettings(backend="cvxpy_dense")
    )
    lower = {key: 0.0 for key in extension.lower_keys}

    def forbidden_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("unsafe dense CVXPY solve was reached")

    monkeypatch.setattr(cp.Problem, "solve", forbidden_solve)
    with pytest.raises(RuntimeError, match="resource preflight"):
        extension.complete(lower)


def test_default_completion_does_not_construct_dense_cvxpy_graph() -> None:
    extension = SymmetryReducedPositiveExtension()

    assert extension.settings.backend == "clarabel_newton"
    assert extension._problem is None
    assert (
        extension.dense_canonicalization_estimate_bytes
        > extension.settings.maximum_dense_canonicalization_bytes
    )


def test_bounded_completion_accepts_a_strictly_physical_product_state() -> None:
    extension = SymmetryReducedPositiveExtension(
        APCMExtensionSettings(
            logdet_weight=1e-3,
            conic_tolerance=1e-7,
            maximum_iterations=400,
        )
    )
    lower = {
        key: _thermal_weyl_moment(key) for key in extension.lower_keys
    }
    warm = {
        key: _thermal_weyl_moment(key) for key in extension.frontier_keys
    }

    result = extension.complete(lower, warm_frontier=warm)

    assert result.success, (
        result.message,
        result.scaled_minimum_eigenvalue,
        result.kkt_backward_error,
    )
    assert result.scaled_minimum_eigenvalue >= -1e-6
    assert result.moment_matrix.shape == (
        extension.dimension,
        extension.dimension,
    )
    assert extension.complete(lower, warm_frontier=warm) is result


def test_linear_extrema_return_outward_bounds_without_dense_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = SymmetryReducedPositiveExtension(
        APCMExtensionSettings(
            logdet_weight=1e-3,
            conic_tolerance=1e-7,
            maximum_iterations=400,
        )
    )
    lower = {
        key: _thermal_weyl_moment(key) for key in extension.lower_keys
    }
    candidate = next(
        key
        for key in extension.frontier_keys
        if key.spin_up == "I"
        and key.spin_down == "I"
        and key.x_power == 4
        and key.p_power == 0
    )

    def forbidden_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("unsafe CVXPY extrema solve was reached")

    monkeypatch.setattr(cp.Problem, "solve", forbidden_solve)
    extrema = extension.linear_functional_extrema(
        lower,
        {candidate: 1.0},
    )

    assert extrema.success, extrema.message
    assert extrema.outward_lower_bound <= extrema.primal_minimum
    assert extrema.primal_minimum <= extrema.primal_maximum
    assert extrema.primal_maximum <= extrema.outward_upper_bound


def test_infeasible_stage_retraction_avoids_cvxpy_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coupled 40-by-40 stage repair must use the bounded direct backend."""

    moments = {
        key: _thermal_weyl_moment(key)
        for key in THIRD_ORDER_HIERARCHY.moment_keys
    }
    matrix_state = THIRD_ORDER_HIERARCHY.to_matrix_state(
        THIRD_ORDER_HIERARCHY.pack(0.0j, moments)
    )
    raw = matrix_state_to_raw_moment_coordinates(matrix_state)
    hidden = np.asarray(
        [moments[key] for key in HIDDEN_RELATIVE_MOMENT_KEYS],
        dtype=float,
    )
    extension = SymmetryReducedPositiveExtension(
        APCMExtensionSettings(
            logdet_weight=1e-3,
            conic_tolerance=1e-7,
            maximum_iterations=400,
        )
    )
    completion = extension.complete(
        state_lower_moments(raw, hidden),
        warm_frontier={
            key: _thermal_weyl_moment(key)
            for key in extension.frontier_keys
        },
    )
    assert completion.success, completion.message
    geometry = SymmetryReducedAPCMGeometry(extension)
    assert (
        geometry.direct_retraction_estimate_bytes
        < geometry.settings.maximum_direct_workspace_bytes
    )
    raw_trial = raw.copy()
    raw_trial[2] = 1.05
    assert np.linalg.eigvalsh(uncentered_joint_moment_matrix(raw_trial))[0] < 0.0

    def certify_direct_witness(lower, *, warm_frontier=None):  # noqa: ANN001
        """Isolate stage assembly from the separately tested selector."""

        assert warm_frontier is not None
        values = np.asarray(
            [warm_frontier[key] for key in extension.frontier_keys],
            dtype=float,
        )
        return extension._result_from_values(
            lower,
            values,
            iterations=0,
            solver_success=True,
            message="direct-stage-test-witness",
            kkt_error=0.0,
        )

    def forbidden_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("unsafe CVXPY stage solve was reached")

    monkeypatch.setattr(extension, "complete", certify_direct_witness)
    monkeypatch.setattr(cp.Problem, "solve", forbidden_solve)
    retracted = geometry.retract_stage(
        raw_trial,
        hidden,
        completion,
        retained_metric=geometry.retained_metric(raw),
        auxiliary_metric=extension.auxiliary_metric(completion, hidden),
    )

    assert retracted.applied
    assert (
        np.linalg.eigvalsh(
            uncentered_joint_moment_matrix(retracted.raw_coordinates)
        )[0]
        >= -2e-6
    )
    assert retracted.completion.success


def test_projection_preflight_rejects_large_cvxpy_psd_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = SymmetryReducedAPCMGeometry(
        SymmetryReducedPositiveExtension()
    )
    matrix = cp.Variable((17, 17), symmetric=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(matrix)), [matrix >> 0.0])

    def forbidden_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("unsafe CVXPY projection solve was reached")

    monkeypatch.setattr(cp.Problem, "solve", forbidden_solve)
    with pytest.raises(APCMProjectionError, match="resource preflight"):
        geometry._solve_problem(problem)
