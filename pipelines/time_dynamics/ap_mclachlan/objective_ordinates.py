"""Two objective ordinates for the Paper II generalized-exchange selector.

For a patch ``B = (D, A)`` with patched support ``J^B = (J \\ D) u A``, both
ordinates are normalized captured-drift differences over the SAME
denominator ``||b_bar||^2 + eps_norm``:

    m_Q     = [Q(J^B)     - Q(J)]     / (||b_bar||^2 + eps)     realized
    m_Gamma = [Gamma(J^B) - Gamma(J)] / (||b_bar||^2 + eps)     ideal

``m_Q`` is what the propagated trajectory actually captures: both values come
from ``inverse.solve_theta_dot`` under the live policy (symmetrize -> ridge ->
eigh -> pinv_rcond truncation -> damping), read off as
``McLachlanSolve.captured_drift``, which is evaluated against the RAW
unridged ``G``.  It is therefore bitwise the quantity ``structural_cache``
already memoizes.  ``m_Gamma`` is the variational ceiling
``f^T G^+ f`` on the same raw symmetrized ``G`` with ridge = damping = 0 and
one NAMED relative cutoff, so it answers "what could this support capture"
independently of how the solve is stabilized.  The gap ``m_Gamma - m_Q`` is
stabilizer cost.

Ideal ``Gamma`` through a relative-cutoff pseudo-inverse is NOT nested
monotone: a pure insertion that raises ``max|eig|`` can push a small
drive-carrying eigendirection below ``rcond * max|eig|`` and lose drift the
smaller support kept.  Measured on this repository: benign rounding
violations reach ~6e-14 of ``||b||^2`` while a constructed spectrum shift
reaches 1e-2.  This module therefore FAILS CLOSED on a material violation
instead of clipping; clipping would map "helpful" and "harmless" onto the
same zero, the exact collapse ``exchange_structural`` documents for the
deletion loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.geometry import residual_denominator
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    as_float_matrix,
    as_float_vector,
    solve_theta_dot,
    symmetrize_matrix,
)


ORDINATE_Q_REALIZED_V1 = "paper_ii_ordinate_q_realized_stabilized_v1"
ORDINATE_GAMMA_IDEAL_V1 = "paper_ii_ordinate_gamma_ideal_eigh_v1"

# Named relative cutoff for the ideal projection.  Explicit because numpy's
# ``pinv`` default is a bare 1e-15 that no caller in this lane declares.
DEFAULT_GAMMA_PINV_RCOND = 1.0e-12

# Normalized by ||b_bar||^2 + eps; mirrors ``prune_schur_monotonicity_tol``
# (legacy checkpoint controller) and sits four orders above the measured
# benign floor (~6e-14) and seven below the measured material violation.
GAMMA_NESTED_MONOTONICITY_TOL = 1.0e-9


class GammaMonotonicityError(ValueError):
    """A pure insertion lost ideal captured drift beyond tolerance."""


@dataclass(frozen=True)
class OrdinateRecord:
    """One ordinate evaluation, raw and normalized, with provenance."""

    raw: float
    normalized: float
    rank: int
    threshold: float
    policy_id: str
    finite: bool
    base_value: float
    patched_value: float
    base_rank: int
    denominator: float
    condition_number: float | None = None
    monotonicity_slack: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "raw": float(self.raw),
            "normalized": float(self.normalized),
            "rank": int(self.rank),
            "threshold": float(self.threshold),
            "policy_id": str(self.policy_id),
            "finite": bool(self.finite),
            "base_value": float(self.base_value),
            "patched_value": float(self.patched_value),
            "base_rank": int(self.base_rank),
            "denominator": float(self.denominator),
            "condition_number": (
                None if self.condition_number is None else float(self.condition_number)
            ),
            "monotonicity_slack": (
                None
                if self.monotonicity_slack is None
                else float(self.monotonicity_slack)
            ),
        }


def ideal_gamma_from_geometry(
    G: np.ndarray | Sequence[Sequence[float]],
    f: np.ndarray | Sequence[float],
    *,
    rcond: float = DEFAULT_GAMMA_PINV_RCOND,
) -> tuple[float, int, float]:
    """Return ``(Gamma, retained_rank, threshold)`` for ``f^T G^+ f``.

    ``G`` is symmetrized, eigendecomposed with ``eigh``, and modes with
    ``|lambda| <= rcond * max|lambda|`` are dropped (the same strict retained
    rule ``supported_inverse`` applies, without ridge or damping).  The
    quadratic form is accumulated over retained modes only, so the reported
    rank is exactly the rank that produced the value.
    """

    if float(rcond) < 0.0 or not math.isfinite(float(rcond)):
        raise ValueError("rcond must be finite and non-negative.")
    force = as_float_vector(f, name="f_vec")
    if int(force.size) == 0:
        return 0.0, 0, 0.0
    mat = symmetrize_matrix(as_float_matrix(G, name="matrix"))
    if mat.shape != (int(force.size), int(force.size)):
        raise ValueError(
            "matrix must be square with dimension len(f_vec); "
            f"got matrix {mat.shape}, f_vec {force.shape}."
        )
    try:
        eigenvalues, vectors = np.linalg.eigh(mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ideal Gamma eigendecomposition failed.") from exc
    abs_eigs = np.abs(eigenvalues)
    max_abs = float(np.max(abs_eigs)) if abs_eigs.size else 0.0
    threshold = float(rcond) * max_abs
    retained = abs_eigs > threshold
    projections = vectors.T @ force
    gamma = float(
        np.sum((projections[retained] ** 2) / eigenvalues[retained])
    )
    if not math.isfinite(gamma):
        raise ValueError("ideal Gamma is not finite.")
    return gamma, int(np.count_nonzero(retained)), threshold


def evaluate_q_ordinate(
    *,
    base_G: np.ndarray | Sequence[Sequence[float]],
    base_f: np.ndarray | Sequence[float],
    patched_G: np.ndarray | Sequence[Sequence[float]],
    patched_f: np.ndarray | Sequence[float],
    norm_b_sq: float,
    inverse_policy: McLachlanInversePolicy,
    epsilon_norm: float,
) -> OrdinateRecord:
    """``m_Q`` under the stabilized solve, bitwise the propagation quantity."""

    denominator = float(residual_denominator(float(norm_b_sq), float(epsilon_norm)))
    base = _stabilized_q(base_G, base_f, inverse_policy)
    patched = _stabilized_q(patched_G, patched_f, inverse_policy)
    raw = float(patched.captured_drift - base.captured_drift)
    finite = bool(math.isfinite(raw))
    return OrdinateRecord(
        raw=raw,
        normalized=float(raw / denominator),
        rank=int(patched.inverse.rank),
        threshold=float(patched.inverse.retained_threshold),
        policy_id=ORDINATE_Q_REALIZED_V1,
        finite=finite,
        base_value=float(base.captured_drift),
        patched_value=float(patched.captured_drift),
        base_rank=int(base.inverse.rank),
        denominator=denominator,
        condition_number=(
            None
            if patched.inverse.condition_number is None
            else float(patched.inverse.condition_number)
        ),
    )


def evaluate_gamma_ordinate(
    *,
    base_G: np.ndarray | Sequence[Sequence[float]],
    base_f: np.ndarray | Sequence[float],
    patched_G: np.ndarray | Sequence[Sequence[float]],
    patched_f: np.ndarray | Sequence[float],
    norm_b_sq: float,
    epsilon_norm: float,
    rcond: float = DEFAULT_GAMMA_PINV_RCOND,
    is_pure_insertion: bool = False,
    monotonicity_tol: float = GAMMA_NESTED_MONOTONICITY_TOL,
) -> OrdinateRecord:
    """``m_Gamma`` under ideal projection with one named cutoff.

    When ``is_pure_insertion`` (``D`` empty, so ``J`` is a subset of ``J^B``)
    the normalized slack ``m_Gamma`` must be at least ``-monotonicity_tol``;
    a material violation raises :class:`GammaMonotonicityError` and a
    violation inside the band is recorded on the record, never clipped.
    """

    denominator = float(residual_denominator(float(norm_b_sq), float(epsilon_norm)))
    gamma_base, base_rank, _base_threshold = ideal_gamma_from_geometry(
        base_G, base_f, rcond=rcond
    )
    gamma_patched, rank, threshold = ideal_gamma_from_geometry(
        patched_G, patched_f, rcond=rcond
    )
    raw = float(gamma_patched - gamma_base)
    normalized = float(raw / denominator)
    slack: float | None = None
    if is_pure_insertion:
        slack = check_gamma_nested_monotonicity(
            gamma_base=gamma_base,
            gamma_patched=gamma_patched,
            denominator=denominator,
            rcond=float(rcond),
            tol=float(monotonicity_tol),
        )
    return OrdinateRecord(
        raw=raw,
        normalized=normalized,
        rank=int(rank),
        threshold=float(threshold),
        policy_id=ORDINATE_GAMMA_IDEAL_V1,
        finite=bool(math.isfinite(raw)),
        base_value=float(gamma_base),
        patched_value=float(gamma_patched),
        base_rank=int(base_rank),
        denominator=denominator,
        condition_number=None,
        monotonicity_slack=slack,
    )


def check_gamma_nested_monotonicity(
    *,
    gamma_base: float,
    gamma_patched: float,
    denominator: float,
    rcond: float,
    tol: float = GAMMA_NESTED_MONOTONICITY_TOL,
) -> float:
    """Return the normalized slack; raise when it is materially negative."""

    slack = float((float(gamma_patched) - float(gamma_base)) / float(denominator))
    if slack < -float(tol):
        raise GammaMonotonicityError(
            "ideal Gamma violated nested monotonicity for a pure insertion: "
            f"normalized slack {slack:.3e} < -{float(tol):.3e} at rcond={rcond:.3e}; "
            f"Gamma(J)={gamma_base!r}, Gamma(J^B)={gamma_patched!r}."
        )
    return slack


def _stabilized_q(G, f, policy: McLachlanInversePolicy):
    force = as_float_vector(f, name="f_vec")
    if int(force.size) == 0:
        # Mirror structural_cache's empty-geometry short-circuit.
        from pipelines.time_dynamics.ap_mclachlan.inverse import supported_inverse

        class _Empty:
            captured_drift = 0.0
            inverse = supported_inverse(np.zeros((0, 0), dtype=float), policy=policy)

        return _Empty()
    return solve_theta_dot(G, force, policy=policy)


__all__ = [
    "DEFAULT_GAMMA_PINV_RCOND",
    "GAMMA_NESTED_MONOTONICITY_TOL",
    "GammaMonotonicityError",
    "ORDINATE_GAMMA_IDEAL_V1",
    "ORDINATE_Q_REALIZED_V1",
    "OrdinateRecord",
    "check_gamma_nested_monotonicity",
    "evaluate_gamma_ordinate",
    "evaluate_q_ordinate",
    "ideal_gamma_from_geometry",
]
