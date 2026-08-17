import json

import numpy as np
import pytest

from pipelines.time_dynamics.legacy.checkpoint_prune_loss import (
    compute_prune_loss_payload,
)
from pipelines.time_dynamics.ap_mclachlan.support_patch import (
    PATCH_APPEND,
    PATCH_DELETE,
    PATCH_EXCHANGE,
    PATCH_NO_EDIT,
    McLachlanInversePolicy,
    SupportPatch,
    SupportPatchGeometry,
    build_support_patch_after_cache,
    build_support_patch_before_cache,
    legacy_prune_payload_from_score,
    prune_conditioning_diagnostics,
    score_patch_payload,
    score_support_patch,
    schur_novelty,
)


def test_no_edit_patch_score_is_zero() -> None:
    geometry = SupportPatchGeometry(
        K_before=np.array([[2.0, 0.1], [0.1, 3.0]]),
        f_before=np.array([1.0, -0.5]),
        norm_b_sq=2.0,
    )

    score = score_support_patch(
        geometry=geometry,
        patch=SupportPatch(),
        inverse_policy=McLachlanInversePolicy(ridge_lambda=0.0),
    )

    assert score.patch_kind == PATCH_NO_EDIT
    assert score.before_gain == pytest.approx(score.after_gain)
    assert score.signed_delta_gain == pytest.approx(0.0)
    assert score.normalized_score == pytest.approx(0.0)
    assert score.insertion_gain == pytest.approx(0.0)
    assert score.deletion_loss == pytest.approx(0.0)


def test_delete_only_patch_matches_existing_prune_loss_payload() -> None:
    K = np.array([[2.0, 0.2, 0.0], [0.2, 3.0, 0.1], [0.0, 0.1, 1.5]])
    f = np.array([1.0, -0.5, 0.25])
    norm_b_sq = 1.7
    policy = McLachlanInversePolicy(pinv_rcond=1.0e-11, ridge_lambda=0.0)
    removed = (1,)

    geometry = SupportPatchGeometry(K_before=K, f_before=f, norm_b_sq=norm_b_sq)
    score = score_support_patch(
        geometry=geometry,
        patch=SupportPatch(removed_runtime_indices=removed),
        inverse_policy=policy,
    )
    legacy = compute_prune_loss_payload(
        G=None,
        K=K,
        f_vec=f,
        norm_b_sq=norm_b_sq,
        removed_runtime_indices=removed,
        pinv_rcond=policy.pinv_rcond,
        regularization_lambda=policy.ridge_lambda,
        epsilon=policy.epsilon,
    )

    assert score.patch_kind == PATCH_DELETE
    assert score.deletion_loss == pytest.approx(legacy["prune_loss_delta_k_damped"])
    assert score.normalized_score == pytest.approx(-legacy["prune_loss_delta_k_damped"])
    payload = score.to_json_dict()
    assert payload["support_patch_deleted_count"] == 1
    assert payload["support_patch_removed_count"] == 1
    assert legacy_prune_payload_from_score(score)["prune_loss_delta_k_damped"] == pytest.approx(
        legacy["prune_loss_delta_k_damped"]
    )


def test_insert_only_patch_gain_is_positive() -> None:
    geometry = SupportPatchGeometry(
        K_before=np.array([[2.0]]),
        f_before=np.array([1.0]),
        norm_b_sq=2.0,
        K_insert_cross=np.array([[0.0]]),
        K_insert_insert=np.array([[1.0]]),
        f_insert=np.array([1.0]),
    )

    score = score_support_patch(
        geometry=geometry,
        patch=SupportPatch(inserted_count=1, inserted_labels=("candidate_a",)),
    )

    assert score.patch_kind == PATCH_APPEND
    assert score.insertion_gain > 0.0
    assert score.deletion_loss == pytest.approx(0.0)
    assert score.after_indices_before_part == (0,)
    assert score.inserted_labels == ("candidate_a",)
    assert score.schur_novelty is not None
    assert score.schur_novelty.full_rank is True
    assert score.augmented_solve_confirmation is not None
    assert score.augmented_solve_confirmation.confirmed is True
    assert score.augmented_solve_confirmation.rank_retained == 2
    assert score.augmented_solve_confirmation.residual_ratio is not None
    assert np.isfinite(score.augmented_solve_confirmation.residual_ratio)

    payload = score.to_json_dict()
    confirmation = payload["support_patch_augmented_solve_confirmation"]
    assert confirmation["confirmed"] is True
    assert confirmation["support_size"] == 2


def test_schur_novelty_detects_redundant_candidate_direction() -> None:
    policy = McLachlanInversePolicy(pinv_rcond=1.0e-10, ridge_lambda=0.0)

    redundant = schur_novelty(
        K_available=np.array([[1.0]]),
        K_candidate=np.array([[1.0]]),
        K_available_candidate=np.array([[1.0]]),
        inverse_policy=policy,
        candidate_ridge_lambda=0.0,
    )
    novel = schur_novelty(
        K_available=np.array([[1.0]]),
        K_candidate=np.array([[1.0]]),
        K_available_candidate=np.array([[0.0]]),
        inverse_policy=policy,
        candidate_ridge_lambda=0.0,
    )

    assert redundant.rank == 0
    assert redundant.full_rank is False
    assert novel.rank == 1
    assert novel.full_rank is True


def test_exchange_append_child_can_be_redundant_until_active_direction_is_deleted() -> None:
    """Regression for canonical exchange fail-open macro-frontier behavior."""

    policy = McLachlanInversePolicy(pinv_rcond=1.0e-10, ridge_lambda=0.0)
    score_against_full_support = score_support_patch(
        geometry=SupportPatchGeometry(
            K_before=np.eye(2),
            f_before=np.array([1.0, 0.0]),
            norm_b_sq=1.0,
            K_insert_cross=np.array([[1.0], [0.0]]),
            K_insert_insert=np.array([[1.0]]),
            f_insert=np.array([1.0]),
        ),
        patch=SupportPatch(inserted_count=1, inserted_labels=("child_a",)),
        inverse_policy=policy,
    )

    assert score_against_full_support.insertion_gain == pytest.approx(0.0)
    assert score_against_full_support.schur_novelty is not None
    assert score_against_full_support.schur_novelty.rank == 0
    assert score_against_full_support.schur_novelty.full_rank is False

    score_after_deleting_redundant_active_direction = score_support_patch(
        geometry=SupportPatchGeometry(
            K_before=np.array([[1.0]]),
            f_before=np.array([0.0]),
            norm_b_sq=1.0,
            K_insert_cross=np.array([[0.0]]),
            K_insert_insert=np.array([[1.0]]),
            f_insert=np.array([1.0]),
        ),
        patch=SupportPatch(inserted_count=1, inserted_labels=("child_a",)),
        inverse_policy=policy,
    )

    assert (
        score_after_deleting_redundant_active_direction.insertion_gain
        == pytest.approx(1.0)
    )
    assert score_after_deleting_redundant_active_direction.schur_novelty is not None
    assert score_after_deleting_redundant_active_direction.schur_novelty.rank == 1
    assert score_after_deleting_redundant_active_direction.schur_novelty.full_rank is True


def test_exchange_patch_telemetry_is_json_safe() -> None:
    geometry = SupportPatchGeometry(
        K_before=np.diag([2.0, 3.0]),
        f_before=np.array([1.0, 0.5]),
        norm_b_sq=1.0,
        K_insert_cross=np.array([[0.0], [0.0]]),
        K_insert_insert=np.array([[1.0]]),
        f_insert=np.array([1.0]),
    )

    payload = score_patch_payload(
        geometry=geometry,
        patch=SupportPatch(
            removed_runtime_indices=(0,),
            inserted_count=1,
            inserted_labels=("replacement",),
        ),
        cost_terms={"cost_penalty": np.float64(0.25), "counts": np.array([1, 2])},
    )

    assert payload["support_patch_kind"] == PATCH_EXCHANGE
    assert payload["support_patch_after_indices_before_part"] == [1]
    assert payload["support_patch_removed_runtime_indices"] == [0]
    json.dumps(payload)


def test_singular_matrix_uses_pseudoinverse() -> None:
    geometry = SupportPatchGeometry(
        K_before=np.array([[1.0, 1.0], [1.0, 1.0]]),
        f_before=np.array([1.0, 1.0]),
        norm_b_sq=1.0,
    )

    score = score_support_patch(
        geometry=geometry,
        patch=SupportPatch(),
        inverse_policy=McLachlanInversePolicy(ridge_lambda=0.0),
    )

    assert score.rank_before == 1
    assert score.before_gain is not None
    assert np.isfinite(score.before_gain)


def test_invalid_insert_shapes_raise_value_error() -> None:
    geometry = SupportPatchGeometry(
        K_before=np.array([[2.0]]),
        f_before=np.array([1.0]),
        norm_b_sq=1.0,
        K_insert_cross=np.array([[0.0, 0.0]]),
        K_insert_insert=np.array([[1.0]]),
        f_insert=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="K_insert_cross"):
        score_support_patch(geometry=geometry, patch=SupportPatch(inserted_count=1))


def test_out_of_range_removed_indices_are_ignored_in_effective_patch() -> None:
    geometry = SupportPatchGeometry(
        K_before=np.array([[2.0]]),
        f_before=np.array([1.0]),
        norm_b_sq=1.0,
    )

    score = score_support_patch(
        geometry=geometry,
        patch=SupportPatch(removed_runtime_indices=(99,)),
    )

    assert score.patch_kind == PATCH_NO_EDIT
    assert score.removed_runtime_indices == ()
    assert score.after_indices_before_part == (0,)
    assert score.deletion_loss == pytest.approx(0.0)


def test_cost_ranking_does_not_change_raw_score() -> None:
    geometry = SupportPatchGeometry(
        K_before=np.array([[2.0]]),
        f_before=np.array([1.0]),
        norm_b_sq=2.0,
        K_insert_cross=np.array([[0.0]]),
        K_insert_insert=np.array([[1.0]]),
        f_insert=np.array([1.0]),
    )
    patch = SupportPatch(inserted_count=1)

    raw = score_support_patch(geometry=geometry, patch=patch)
    costed = score_support_patch(
        geometry=geometry,
        patch=patch,
        cost_terms={"cost_penalty": 10.0},
        cost_weight=0.1,
    )

    assert costed.normalized_score == pytest.approx(raw.normalized_score)
    assert costed.insertion_gain == pytest.approx(raw.insertion_gain)
    assert costed.rank_score == pytest.approx(raw.normalized_score - 1.0)
    assert costed.rank_score != pytest.approx(raw.rank_score)


def test_prune_conditioning_diagnostics_use_whole_batch_schur() -> None:
    geometry = SupportPatchGeometry(
        K_before=np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        ),
        f_before=np.array([1.0, 0.5, 1.0, 0.5]),
        norm_b_sq=2.0,
    )
    policy = McLachlanInversePolicy(ridge_lambda=0.0, pinv_rcond=1.0e-10)

    diagnostics = prune_conditioning_diagnostics(
        geometry=geometry,
        patch=SupportPatch(removed_runtime_indices=(2, 3)),
        inverse_policy=policy,
    )

    assert diagnostics.available is True
    assert diagnostics.removed_runtime_indices == (2, 3)
    assert diagnostics.schur_candidate_count == 2
    assert diagnostics.schur_rank == 0
    assert diagnostics.schur_rank_deficit_fraction == pytest.approx(1.0)
    assert diagnostics.d_schur >= 1.0
    payload = diagnostics.to_json_dict()
    assert payload["d_conditioning_toxicity"] == pytest.approx(
        diagnostics.d_kappa_rel + diagnostics.d_schur
    )


def test_support_patch_before_cache_preserves_patch_scores() -> None:
    geometry = SupportPatchGeometry(
        K_before=np.array([[2.0, 0.1], [0.1, 1.5]]),
        f_before=np.array([0.75, -0.25]),
        norm_b_sq=1.4,
        K_insert_cross=np.array([[0.05], [0.1]]),
        K_insert_insert=np.array([[1.2]]),
        f_insert=np.array([0.5]),
    )
    policy = McLachlanInversePolicy(pinv_rcond=1.0e-11, ridge_lambda=1.0e-8)
    cache = build_support_patch_before_cache(
        geometry=geometry,
        inverse_policy=policy,
    )

    patch = SupportPatch(inserted_count=1, inserted_labels=("candidate",))
    after_cache = build_support_patch_after_cache(
        geometry=geometry,
        patch=patch,
        inverse_policy=policy,
    )
    uncached = score_support_patch(
        geometry=geometry,
        patch=patch,
        inverse_policy=policy,
    )
    cached = score_support_patch(
        geometry=geometry,
        patch=patch,
        inverse_policy=policy,
        before_cache=cache,
        after_cache=after_cache,
    )

    assert cached.before_gain == pytest.approx(uncached.before_gain)
    assert cached.after_gain == pytest.approx(uncached.after_gain)
    assert cached.normalized_score == pytest.approx(uncached.normalized_score)
    assert cached.rank_before == uncached.rank_before
    assert cached.rank_after == uncached.rank_after

    delete_patch = SupportPatch(removed_runtime_indices=(1,))
    delete_after_cache = build_support_patch_after_cache(
        geometry=geometry,
        patch=delete_patch,
        inverse_policy=policy,
    )
    uncached_conditioning = prune_conditioning_diagnostics(
        geometry=geometry,
        patch=delete_patch,
        inverse_policy=policy,
    )
    cached_conditioning = prune_conditioning_diagnostics(
        geometry=geometry,
        patch=delete_patch,
        inverse_policy=policy,
        before_cache=cache,
        after_cache=delete_after_cache,
    )

    assert cached_conditioning.condition_number_before == pytest.approx(
        uncached_conditioning.condition_number_before
    )
    assert cached_conditioning.condition_number_after == pytest.approx(
        uncached_conditioning.condition_number_after
    )
    assert cached_conditioning.d_kappa_rel == pytest.approx(
        uncached_conditioning.d_kappa_rel
    )
    assert cached_conditioning.d_schur == pytest.approx(uncached_conditioning.d_schur)


# ---------------------------------------------------------------------------
# Realized captured drift (2026-08-15 scoring correction)
#
# Support-patch gains and residuals must use Q = 2 f.theta_dot -
# theta_dot.K.theta_dot for the velocity the policy solve actually returns.
# The historical Gamma = f.theta_dot equals Q only for the exact unridged,
# untruncated, undamped solve; these tests pin the distinction with a policy
# where every regularizer is active.
# ---------------------------------------------------------------------------


def _rank_deficient_tangent_problem(seed: int = 5, dim: int = 16, n: int = 24):
    """Explicit T and b with n > dim, so K is structurally rank-deficient."""

    rng = np.random.default_rng(seed)
    T = rng.standard_normal((dim, n)) + 1j * rng.standard_normal((dim, n))
    b = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    K = np.real(T.conj().T @ T)
    K = 0.5 * (K + K.T)
    f = np.real(T.conj().T @ b)
    return T, b, K, f


_ACTIVE_POLICY = McLachlanInversePolicy(
    pinv_rcond=1.0e-6,
    ridge_lambda=1.0e-4,
    solve_damping=1.0e-3,
)


def test_captured_drift_equals_physical_residual_identity_under_active_policy():
    """||b||^2 - Q must equal the true state-space residual of the solve."""

    from pipelines.time_dynamics.ap_mclachlan.inverse import solve_theta_dot

    T, b, K, f = _rank_deficient_tangent_problem()
    solve = solve_theta_dot(K, f, policy=_ACTIVE_POLICY)
    residual_direct = float(
        np.linalg.norm(T @ solve.theta_dot - b) ** 2
    )
    norm_b_sq = float(np.real(np.vdot(b, b)))
    # Identity: Q = ||b||^2 - ||T theta_dot - b||^2, exact up to rounding.
    assert solve.captured_drift == pytest.approx(
        norm_b_sq - residual_direct, rel=1e-10, abs=1e-10
    )
    # And the biased historical value is measurably different here.
    assert abs(solve.gamma - solve.captured_drift) > 1e-6


def test_score_support_patch_gains_use_captured_drift_not_gamma():
    from pipelines.time_dynamics.ap_mclachlan.inverse import solve_theta_dot

    _T, b, K, f = _rank_deficient_tangent_problem()
    norm_b_sq = float(np.real(np.vdot(b, b)))
    geometry = SupportPatchGeometry(
        K_before=K,
        f_before=f,
        norm_b_sq=norm_b_sq,
    )
    removed = (3, 11)
    score = score_support_patch(
        geometry=geometry,
        patch=SupportPatch(removed_runtime_indices=removed),
        inverse_policy=_ACTIVE_POLICY,
    )
    before_solve = solve_theta_dot(K, f, policy=_ACTIVE_POLICY)
    keep = [i for i in range(f.size) if i not in set(removed)]
    after_solve = solve_theta_dot(
        K[np.ix_(keep, keep)], f[keep], policy=_ACTIVE_POLICY
    )
    assert score.before_gain == pytest.approx(before_solve.captured_drift, rel=1e-12)
    assert score.after_gain == pytest.approx(after_solve.captured_drift, rel=1e-12)
    # Regression: the gains must NOT be the historical gamma values.
    assert abs(score.before_gain - before_solve.gamma) > 1e-6


def test_augmented_confirmation_residual_is_realized_not_gamma_based():
    _T, b, K, f = _rank_deficient_tangent_problem(seed=9, dim=16, n=20)
    norm_b_sq = float(np.real(np.vdot(b, b)))
    n = int(f.size)
    # Append one candidate column taken from a fresh random tangent.
    rng = np.random.default_rng(77)
    K_full = np.zeros((n + 1, n + 1))
    K_full[:n, :n] = K
    cross = rng.standard_normal(n) * 0.3
    K_full[:n, n] = cross
    K_full[n, :n] = cross
    K_full[n, n] = 2.0
    f_full = np.concatenate([f, [0.4]])
    geometry = SupportPatchGeometry(
        K_before=K,
        f_before=f,
        norm_b_sq=norm_b_sq,
        K_insert_cross=K_full[:n, n:].reshape(n, 1),
        K_insert_insert=K_full[n:, n:].reshape(1, 1),
        f_insert=f_full[n:],
    )
    score = score_support_patch(
        geometry=geometry,
        patch=SupportPatch(inserted_count=1, inserted_labels=("cand::r0::x",)),
        inverse_policy=_ACTIVE_POLICY,
    )
    confirmation = score.augmented_solve_confirmation
    assert confirmation is not None and confirmation.confirmed
    from pipelines.time_dynamics.ap_mclachlan.inverse import solve_theta_dot

    aug_solve = solve_theta_dot(K_full, f_full, policy=_ACTIVE_POLICY)
    expected_residual = min(
        max(norm_b_sq - aug_solve.captured_drift, 0.0), norm_b_sq
    )
    assert confirmation.residual_sq == pytest.approx(expected_residual, rel=1e-12)
    gamma_based = min(max(norm_b_sq - aug_solve.gamma, 0.0), norm_b_sq)
    assert abs(confirmation.residual_sq - gamma_based) > 1e-6


def test_captured_drift_reduces_to_gamma_for_exact_solve():
    """With no ridge, damping, or truncation the two definitions coincide."""

    from pipelines.time_dynamics.ap_mclachlan.inverse import solve_theta_dot

    rng = np.random.default_rng(3)
    A = rng.standard_normal((6, 6))
    K = A @ A.T + 6.0 * np.eye(6)  # well conditioned, full rank
    f = rng.standard_normal(6)
    exact_policy = McLachlanInversePolicy(
        pinv_rcond=1.0e-14, ridge_lambda=0.0, solve_damping=0.0
    )
    solve = solve_theta_dot(K, f, policy=exact_policy)
    assert solve.captured_drift == pytest.approx(solve.gamma, rel=1e-9)
