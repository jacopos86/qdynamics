"""Realized-Q and ideal-Gamma ordinates: identities, divergence, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    solve_theta_dot,
)
from pipelines.time_dynamics.ap_mclachlan.objective_ordinates import (
    GAMMA_NESTED_MONOTONICITY_TOL,
    ORDINATE_GAMMA_IDEAL_V1,
    ORDINATE_Q_REALIZED_V1,
    GammaMonotonicityError,
    check_gamma_nested_monotonicity,
    evaluate_gamma_ordinate,
    evaluate_q_ordinate,
    ideal_gamma_from_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.schur_identity import exact_captured_drift


def _random_system(rng, dim, rank=None, ambient=12):
    # K = T^T T, f = T^T b so that f lies in range(K) (the route precondition).
    T = rng.normal(size=(ambient, dim))
    if rank is not None and rank < dim:
        basis = rng.normal(size=(ambient, rank))
        mixing = rng.normal(size=(rank, dim))
        T = basis @ mixing
    b = rng.normal(size=ambient)
    K = T.T @ T
    f = T.T @ b
    return K, f, float(b @ b)


EXACT = McLachlanInversePolicy(pinv_rcond=1.0e-14, ridge_lambda=0.0, solve_damping=0.0)
ACTIVE = McLachlanInversePolicy(pinv_rcond=1.0e-6, ridge_lambda=1.0e-4, solve_damping=1.0e-3)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gamma_equals_exact_captured_drift_and_q_under_exact_policy(seed) -> None:
    rng = np.random.default_rng(seed)
    K, f, nb = _random_system(rng, 6)
    gamma, rank, _thr = ideal_gamma_from_geometry(K, f, rcond=1.0e-12)
    assert gamma == pytest.approx(exact_captured_drift(K, f), rel=1e-9, abs=1e-12)
    assert rank == 6
    assert solve_theta_dot(K, f, policy=EXACT).captured_drift == pytest.approx(gamma, rel=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_q_ordinate_is_bitwise_the_stabilized_captured_drift_difference(seed) -> None:
    rng = np.random.default_rng(seed)
    K, f, nb = _random_system(rng, 7)
    base_idx = list(range(5))
    rec = evaluate_q_ordinate(
        base_G=K[np.ix_(base_idx, base_idx)], base_f=f[base_idx],
        patched_G=K, patched_f=f, norm_b_sq=nb, inverse_policy=ACTIVE, epsilon_norm=1e-12,
    )
    q_base = solve_theta_dot(K[np.ix_(base_idx, base_idx)], f[base_idx], policy=ACTIVE).captured_drift
    q_full = solve_theta_dot(K, f, policy=ACTIVE).captured_drift
    assert rec.raw == q_full - q_base
    assert rec.normalized == rec.raw / (nb + 1e-12)
    assert rec.policy_id == ORDINATE_Q_REALIZED_V1 and rec.finite
    assert rec.base_value == q_base and rec.patched_value == q_full


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_gamma_pure_insertion_is_nested_monotone_on_route_systems(seed) -> None:
    rng = np.random.default_rng(seed)
    K, f, nb = _random_system(rng, 8, rank=5)
    base_idx = list(range(6))
    rec = evaluate_gamma_ordinate(
        base_G=K[np.ix_(base_idx, base_idx)], base_f=f[base_idx],
        patched_G=K, patched_f=f, norm_b_sq=nb, epsilon_norm=1e-12,
        is_pure_insertion=True,
    )
    assert rec.policy_id == ORDINATE_GAMMA_IDEAL_V1
    assert rec.monotonicity_slack is not None
    assert rec.monotonicity_slack >= -GAMMA_NESTED_MONOTONICITY_TOL
    assert rec.normalized == rec.monotonicity_slack


def test_gamma_and_q_diverge_under_stabilization_but_agree_exactly() -> None:
    rng = np.random.default_rng(11)
    K, f, nb = _random_system(rng, 8, rank=5)
    idx = list(range(6))
    q = evaluate_q_ordinate(base_G=K[np.ix_(idx, idx)], base_f=f[idx], patched_G=K, patched_f=f,
                            norm_b_sq=nb, inverse_policy=ACTIVE, epsilon_norm=1e-12)
    g = evaluate_gamma_ordinate(base_G=K[np.ix_(idx, idx)], base_f=f[idx], patched_G=K, patched_f=f,
                                norm_b_sq=nb, epsilon_norm=1e-12)
    assert abs(q.normalized - g.normalized) > 1e-9
    q_exact = evaluate_q_ordinate(base_G=K[np.ix_(idx, idx)], base_f=f[idx], patched_G=K, patched_f=f,
                                  norm_b_sq=nb, inverse_policy=EXACT, epsilon_norm=1e-12)
    assert q_exact.normalized == pytest.approx(g.normalized, rel=1e-6, abs=1e-9)
    assert q.denominator == g.denominator


def test_material_monotonicity_violation_fails_closed() -> None:
    # Diagonal witness: inserting a huge-norm column raises max|eig| and
    # truncates a drive-carrying tiny direction under the relative cutoff.
    K = np.diag([1.0, 1e-12, 1e6])
    x = np.array([1.0, 1e6, 1e-6])
    f = K @ x
    with pytest.raises(GammaMonotonicityError):
        evaluate_gamma_ordinate(
            base_G=K[:2, :2], base_f=f[:2], patched_G=K, patched_f=f,
            norm_b_sq=4.0, epsilon_norm=1e-12, rcond=1e-15, is_pure_insertion=True,
        )
    # Same geometry as a deletion-containing patch: no monotonicity claim.
    rec = evaluate_gamma_ordinate(
        base_G=K[:2, :2], base_f=f[:2], patched_G=K, patched_f=f,
        norm_b_sq=4.0, epsilon_norm=1e-12, rcond=1e-15, is_pure_insertion=False,
    )
    assert rec.monotonicity_slack is None and rec.normalized < 0.0


def test_tiny_violation_is_recorded_not_raised() -> None:
    slack = check_gamma_nested_monotonicity(
        gamma_base=1.0, gamma_patched=1.0 - 1e-13, denominator=1.0, rcond=1e-12
    )
    assert slack == pytest.approx(-1e-13)
    with pytest.raises(GammaMonotonicityError):
        check_gamma_nested_monotonicity(
            gamma_base=1.0, gamma_patched=1.0 - 1e-6, denominator=1.0, rcond=1e-12
        )


def test_gamma_rank_and_threshold_follow_the_named_cutoff() -> None:
    K = np.diag([2.0, 1e-12])
    f = np.array([1.0, 1e-12])
    gamma, rank, thr = ideal_gamma_from_geometry(K, f, rcond=1e-6)
    assert rank == 1 and thr == 2.0e-6 and gamma == pytest.approx(0.5)
    gamma_full, rank_full, _ = ideal_gamma_from_geometry(K, f, rcond=0.0)
    assert rank_full == 2 and gamma_full == pytest.approx(0.5 + 1e-12)
    assert ideal_gamma_from_geometry(np.zeros((0, 0)), np.zeros(0)) == (0.0, 0, 0.0)
    with pytest.raises(ValueError):
        ideal_gamma_from_geometry(K, f, rcond=-1.0)


def test_record_json_is_plain() -> None:
    rng = np.random.default_rng(3)
    K, f, nb = _random_system(rng, 4)
    rec = evaluate_gamma_ordinate(base_G=K[:3, :3], base_f=f[:3], patched_G=K, patched_f=f,
                                  norm_b_sq=nb, epsilon_norm=1e-12, is_pure_insertion=True)
    payload = rec.to_json_dict()
    assert set(payload) >= {"raw", "normalized", "rank", "threshold", "policy_id",
                            "finite", "base_value", "patched_value", "denominator"}
    assert payload["condition_number"] is None
