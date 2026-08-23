"""Exact deletion loss as a Schur-complement quadratic form.

Partition the checkpoint Gram system between the coordinates a candidate
retains, ``R``, and the coordinates it proposes to delete, ``D``::

    K = [[K_RR, K_RD],      f = [f_R,
         [K_DR, K_DD]]           f_D]

The reverse Schur novelty block and its drive residual are

    S_{D|R} = K_DD - K_DR K_RR^+ K_RD
    r_{D|R} = f_D  - K_DR K_RR^+ f_R

``S_{D|R}`` measures how much of the deleted tangent block cannot be
synthesized from the retained tangents; ``r_{D|R}`` measures how much
Hamiltonian forcing remains aligned with that novel part.  Under an exact,
unregularized solve the captured drift lost by deleting ``D`` is exactly

    Q(J) - Q(R) = r_{D|R}^T S_{D|R}^+ r_{D|R}                        (identity)

Two consequences govern this route's prune claim, and neither may drift:

1. The right-hand side is a quadratic form in a positive-semidefinite Schur
   complement, so it is **non-negative**.  Under exact projection a deletion
   can never increase captured drift, and therefore can never lower the
   McLachlan distance ``L^2 = 2(||b||^2 - Q)``.

2. It follows that any *helpful* deletion observed in a run exists only because
   the realized solve is not exact -- ridge regularization, rank truncation,
   damping, or a state refit.  Pruning helps by relieving the conditioning of
   the realized solve, not by improving the variational manifold.

The operational reading is that Schur degeneracy may **nominate** a coordinate
for deletion but can never **authorize** one: only the realized post-deletion
solve establishes that removing it helps.  ``test_ap_mclachlan_schur_identity``
pins the identity numerically so that a future change to the geometry, the
inverse policy, or the scoring cannot quietly break it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

SCHUR_IDENTITY_V1 = "paper_ii_reverse_schur_deletion_loss_v1"


def _blocks(
    K: np.ndarray,
    f: np.ndarray,
    retained: Sequence[int],
    deleted: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = np.asarray(list(retained), dtype=int)
    D = np.asarray(list(deleted), dtype=int)
    K = np.asarray(K, dtype=float)
    f = np.asarray(f, dtype=float).reshape(-1)
    return (
        K[np.ix_(R, R)],
        K[np.ix_(R, D)],
        K[np.ix_(D, D)],
        f[R],
        f[D],
    )


def schur_novelty_block(
    K: np.ndarray, retained: Sequence[int], deleted: Sequence[int]
) -> np.ndarray:
    """``S_{D|R} = K_DD - K_DR K_RR^+ K_RD``."""

    K_RR, K_RD, K_DD, _f_R, _f_D = _blocks(K, np.zeros(K.shape[0]), retained, deleted)
    if K_RR.size == 0:
        return np.asarray(K_DD, dtype=float)
    return np.asarray(K_DD - K_RD.T @ np.linalg.pinv(K_RR) @ K_RD, dtype=float)


def schur_drive_residual(
    K: np.ndarray, f: np.ndarray, retained: Sequence[int], deleted: Sequence[int]
) -> np.ndarray:
    """``r_{D|R} = f_D - K_DR K_RR^+ f_R``."""

    K_RR, K_RD, _K_DD, f_R, f_D = _blocks(K, f, retained, deleted)
    if K_RR.size == 0:
        return np.asarray(f_D, dtype=float)
    return np.asarray(f_D - K_RD.T @ np.linalg.pinv(K_RR) @ f_R, dtype=float)


def exact_deletion_loss(
    K: np.ndarray, f: np.ndarray, retained: Sequence[int], deleted: Sequence[int]
) -> float:
    """``r_{D|R}^T S_{D|R}^+ r_{D|R}`` -- the exact captured drift lost.

    Non-negative by construction, which is the mathematical content of the
    statement that exact projection cannot be improved by deleting coordinates.
    """

    if len(list(deleted)) == 0:
        return 0.0
    S = schur_novelty_block(K, retained, deleted)
    r = schur_drive_residual(K, f, retained, deleted)
    return float(max(0.0, r @ np.linalg.pinv(S) @ r))


def exact_captured_drift(K: np.ndarray, f: np.ndarray) -> float:
    """``Q = f^T K^+ f`` -- captured drift under an exact, unregularized solve.

    With ``theta_dot = K^+ f`` the general expression
    ``Q = 2 f^T theta_dot - theta_dot^T K theta_dot`` collapses to this, because
    ``K K^+ f = f`` on the range of ``K``.
    """

    K = np.asarray(K, dtype=float)
    f = np.asarray(f, dtype=float).reshape(-1)
    if f.size == 0:
        return 0.0
    return float(f @ np.linalg.pinv(K) @ f)


__all__ = [
    "SCHUR_IDENTITY_V1",
    "exact_captured_drift",
    "exact_deletion_loss",
    "schur_drive_residual",
    "schur_novelty_block",
]
