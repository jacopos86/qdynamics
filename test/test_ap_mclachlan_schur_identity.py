"""Pin the reverse-Schur deletion-loss identity.

    Q(J) - Q(R) = r_{D|R}^T S_{D|R}^+ r_{D|R}   (exact, unregularized)

This is the mathematical authority behind Paper II's prune claim, so it is
tested rather than asserted. Its non-negativity is what makes a helpful
deletion impossible under exact projection, and therefore what makes the
realized regularized solve -- not any structural nomination -- the arbiter of
whether a deletion helps.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.schur_identity import (
    conditional_capture,
    exact_captured_drift,
    exact_deletion_loss,
    schur_drive_residual,
    schur_novelty_block,
)


def _random_system(rng, dim, rank=None, ambient=12):
    """A Gram and drive vector built the way the route builds them.

    The identity needs ``f in range(K)``. That is not an extra assumption here:
    the route forms ``K = T^T T`` and ``f = T^T b`` from the tangent matrix
    ``T``, so ``f in range(T^T) = range(K)`` by construction, and the same holds
    for every coordinate subset since ``K_RR = T_R^T T_R`` and
    ``f_R = T_R^T b``. Drawing ``f`` freely instead would put a component of the
    drive outside the range of the Gram, where ``2 f^T theta_dot - theta_dot^T K
    theta_dot`` is unbounded above along null directions and the captured drift
    is not defined at all -- a state the route cannot reach.
    """

    rank = dim if rank is None else rank
    T = rng.normal(size=(ambient, dim))
    if rank < dim:                       # make columns linearly dependent
        basis = rng.normal(size=(ambient, rank))
        mixing = rng.normal(size=(rank, dim))
        T = basis @ mixing
    b = rng.normal(size=ambient)
    K = T.T @ T
    f = T.T @ b
    return 0.5 * (K + K.T), f


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("dim,n_del", [(6, 1), (6, 2), (8, 3)])
def test_deletion_loss_equals_schur_quadratic_form(seed, dim, n_del) -> None:
    rng = np.random.default_rng(seed)
    K, f = _random_system(rng, dim)
    deleted = sorted(rng.choice(dim, size=n_del, replace=False).tolist())
    retained = [i for i in range(dim) if i not in deleted]

    q_full = exact_captured_drift(K, f)
    q_retained = exact_captured_drift(
        K[np.ix_(retained, retained)], f[retained]
    )
    identity = exact_deletion_loss(K, f, retained, deleted)

    assert q_full - q_retained == pytest.approx(identity, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_exact_deletion_can_never_increase_captured_drift(seed) -> None:
    """Non-negativity: no deletion helps under exact projection."""

    rng = np.random.default_rng(seed)
    dim = int(rng.integers(3, 9))
    K, f = _random_system(rng, dim, rank=int(rng.integers(1, dim + 1)))
    n_del = int(rng.integers(1, dim))
    deleted = sorted(rng.choice(dim, size=n_del, replace=False).tolist())
    retained = [i for i in range(dim) if i not in deleted]

    q_full = exact_captured_drift(K, f)
    q_retained = exact_captured_drift(
        K[np.ix_(retained, retained)], f[retained]
    )
    assert q_full - q_retained >= -1e-9
    assert exact_deletion_loss(K, f, retained, deleted) >= 0.0


def test_schur_block_is_positive_semidefinite() -> None:
    rng = np.random.default_rng(11)
    K, _f = _random_system(rng, 7)
    deleted = [1, 4]
    retained = [i for i in range(7) if i not in deleted]
    S = schur_novelty_block(K, retained, deleted)
    eigenvalues = np.linalg.eigvalsh(0.5 * (S + S.T))
    assert eigenvalues.min() >= -1e-9


def test_deleting_a_redundant_coordinate_costs_nothing() -> None:
    """A coordinate in the span of the retained set has zero Schur novelty."""

    rng = np.random.default_rng(5)
    base = rng.normal(size=(6, 3))
    duplicate = np.hstack([base, base[:, :1]])       # column 3 duplicates 0
    K = duplicate.T @ duplicate
    f = duplicate.T @ rng.normal(size=6)
    retained, deleted = [0, 1, 2], [3]
    assert exact_deletion_loss(K, f, retained, deleted) == pytest.approx(0.0, abs=1e-8)
    assert np.linalg.norm(schur_novelty_block(K, retained, deleted)) < 1e-8


def test_deleting_an_orthogonal_forced_coordinate_costs_its_whole_share() -> None:
    """An orthogonal coordinate carries drift no survivor can synthesize."""

    K = np.diag([2.0, 3.0, 5.0])
    f = np.array([1.0, 1.0, 4.0])
    retained, deleted = [0, 1], [2]
    # Orthogonal: S = K_DD, r = f_D, so the loss is f_D^2 / K_DD.
    assert exact_deletion_loss(K, f, retained, deleted) == pytest.approx(16.0 / 5.0)
    assert schur_drive_residual(K, f, retained, deleted) == pytest.approx([4.0])


def test_route_construction_guarantees_the_identity_precondition() -> None:
    """``f in range(K)`` must hold for the full support and every subset.

    This is the precondition the identity rests on, and the route satisfies it
    structurally rather than by luck: `geometry_eval` forms ``K = T^T T`` and
    ``f = T^T b_bar`` from the same tangent matrix, so ``f in range(T^T) =
    range(K)``, and restricting to a coordinate subset restricts ``T`` to those
    columns, preserving the property. If a future change ever formed ``f``
    from a different object than ``K``, the drive would acquire a component
    outside the Gram's range, ``2 f^T x - x^T K x`` would be unbounded above
    along null directions, and captured drift would cease to be defined.
    """

    rng = np.random.default_rng(3)
    for rank in (2, 4, 6):
        T = rng.normal(size=(10, 6))
        if rank < 6:
            T = rng.normal(size=(10, rank)) @ rng.normal(size=(rank, 6))
        b = rng.normal(size=10)
        K = T.T @ T
        f = T.T @ b
        for subset in ([0, 1, 2, 3, 4, 5], [0, 2, 4], [1, 5], [3]):
            K_sub = K[np.ix_(subset, subset)]
            f_sub = f[subset]
            # f in range(K)  <=>  K K^+ f == f
            residual = K_sub @ np.linalg.pinv(K_sub) @ f_sub - f_sub
            assert np.linalg.norm(residual) < 1e-8, (rank, subset)


@pytest.mark.parametrize("seed", [4, 17])
def test_conditional_capture_matches_independent_tangent_projection(seed) -> None:
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(8, 5))
    b = rng.normal(size=8)
    retained, block = (2, 0), (4, 1)
    # Coordinate 3 is deliberately outside both sets. Conditioning must use
    # the requested support, not silently incorporate every column of K.
    retained_tangents = T[:, retained]
    joint_tangents = T[:, retained + block]
    retained_fit = retained_tangents @ np.linalg.lstsq(
        retained_tangents, b, rcond=None
    )[0]
    joint_fit = joint_tangents @ np.linalg.lstsq(
        joint_tangents, b, rcond=None
    )[0]
    expected = np.linalg.norm(b - retained_fit) ** 2 - np.linalg.norm(b - joint_fit) ** 2

    assert conditional_capture(T.T @ T, T.T @ b, retained, block) == pytest.approx(
        expected, rel=1e-10, abs=1e-12
    )


def test_conditional_capture_keeps_interacting_batch_cross_terms() -> None:
    # Retain e1; the new tangents e1+e2 and e1-e2 have opposite residuals.
    # Either captures all of b=e2, so summing singleton gains double-counts it.
    T = np.asarray([[1.0, 1.0, 1.0], [0.0, 1.0, -1.0]])
    b = np.asarray([0.0, 1.0])
    K, f = T.T @ T, T.T @ b

    joint = conditional_capture(K, f, (0,), (1, 2))
    singleton_sum = sum(conditional_capture(K, f, (0,), (i,)) for i in (1, 2))

    assert joint == pytest.approx(1.0)
    assert singleton_sum == pytest.approx(2.0)


def test_conditional_capture_handles_singular_retained_and_added_blocks() -> None:
    # Two copies of e1 are retained and two copies of e2 are added. Both
    # diagonal blocks are singular, but the new captured component is 3*e2.
    T = np.asarray([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    b = np.asarray([2.0, 3.0])

    assert conditional_capture(T.T @ T, T.T @ b, (0, 1), (2, 3)) == pytest.approx(9.0)


def test_conditional_capture_of_empty_block_is_zero() -> None:
    assert conditional_capture(np.eye(2), np.ones(2), (0, 1), ()) == 0.0
    assert conditional_capture(np.empty((0, 0)), np.empty(0), (), ()) == 0.0


def test_conditional_capture_without_retained_coordinates_is_full_capture() -> None:
    # Orthogonal tangents 2*e1 and 3*e2 capture the entire b=(1, 2).
    K = np.diag([4.0, 9.0])
    f = np.asarray([2.0, 6.0])

    assert conditional_capture(K, f, (), (0, 1)) == pytest.approx(5.0)


@pytest.mark.parametrize("forcing", [np.nan, np.inf])
def test_conditional_capture_does_not_hide_nonfinite_raw_result(forcing) -> None:
    result = conditional_capture(np.eye(1), np.asarray([forcing]), (), (0,))

    if np.isnan(forcing):
        assert np.isnan(result)
    else:
        assert np.isposinf(result)
