from __future__ import annotations

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    gamma_for_support,
    solve_theta_dot,
    supported_inverse,
)


def test_solve_theta_dot_and_gamma_match_diagonal_geometry() -> None:
    K = np.diag([2.0, 4.0])
    f = np.array([2.0, 4.0])

    solve = solve_theta_dot(K, f)

    np.testing.assert_allclose(solve.theta_dot, np.array([1.0, 1.0]))
    assert solve.gamma == pytest.approx(6.0)
    assert solve.inverse.rank == 2


def test_gamma_for_support_uses_same_supported_inverse_convention() -> None:
    K = np.diag([2.0, 4.0])
    f = np.array([2.0, 4.0])

    supported = gamma_for_support(matrix=K, f_vec=f, indices=(1,))

    np.testing.assert_allclose(supported.theta_dot, np.array([1.0]))
    assert supported.gamma == pytest.approx(4.0)


def test_supported_inverse_drops_below_threshold_eigenspace() -> None:
    K = np.diag([2.0, 1.0e-12])
    policy = McLachlanInversePolicy(pinv_rcond=1.0e-6)

    inverse = supported_inverse(K, policy=policy)
    solve = solve_theta_dot(K, np.array([2.0, 1.0]), policy=policy)

    assert inverse.rank == 1
    np.testing.assert_allclose(solve.theta_dot, np.array([1.0, 0.0]), atol=1.0e-12)
    assert solve.gamma == pytest.approx(2.0)


def test_solve_damping_regularizes_retained_eigenspace() -> None:
    K = np.array([[2.0]])
    f = np.array([2.0])
    policy = McLachlanInversePolicy(
        pinv_rcond=0.0,
        ridge_lambda=0.0,
        solve_damping=2.0,
    )

    solve = solve_theta_dot(K, f, policy=policy)

    np.testing.assert_allclose(solve.theta_dot, np.array([0.5]))
    assert solve.gamma == pytest.approx(1.0)
    assert solve.inverse.solve_damping == pytest.approx(2.0)


def test_policy_rejects_negative_regularization_knobs() -> None:
    with pytest.raises(ValueError, match="pinv_rcond"):
        McLachlanInversePolicy(pinv_rcond=-1.0)
    with pytest.raises(ValueError, match="ridge_lambda"):
        McLachlanInversePolicy(ridge_lambda=-1.0)
    with pytest.raises(ValueError, match="solve_damping"):
        McLachlanInversePolicy(solve_damping=-1.0)
