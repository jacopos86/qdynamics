"""AVQDS-style adaptive append comparator (Yao et al., PRX Quantum 2, 030307).

This module implements the *decision rule* of adaptive variational quantum
dynamics simulation on the same checkpoint geometry, inverse policy, integrator,
and seed as the deletion-conditioned exchange route, so that a comparison
isolates the structural policy rather than the numerical realization.

The McLachlan distance of the current support is

    L^2(J) = ||b_k||^2 - Q_k(J),

with ``Q_k`` the realized captured drift of :mod:`exchange_structural` (equal to
``f^T G^+ f`` under an exact pseudo-inverse).  The AVQDS rule is:

* while ``L^2 > L2_cut``, append the pool operator whose zero-angle addition
  maximally reduces ``L^2``, appended at the END of the circuit;
* repeat greedily within one checkpoint until the threshold is met or an append
  budget is exhausted;
* never delete, never reposition, and never weight the choice by hardware cost.

Those four points are the only differences from the exchange route's insertion
half; everything else (geometry, regularized inverse, repair, integration) is
shared code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    GeometryEvaluation,
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    solve_theta_dot,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanState,
    state_with_inserted_runtime_coordinates,
)

AVQDS_POLICY_V1 = "avqds_adaptive_append_v1"


def mclachlan_distance_squared(
    evaluation: GeometryEvaluation,
    *,
    inverse_policy: McLachlanInversePolicy,
) -> float:
    """McLachlan distance in the convention of Yao et al., Eq. (8).

    The source defines ``L^2 = 2 var[H] - V M^{-1} V`` at optimal parameter
    velocity.  Here ``||b||^2`` is the energy variance and the realized
    captured drift ``Q`` plays the role of ``V M^{-1} V`` up to the same factor
    of two, so

        ``L^2 = 2 (||b||^2 - Q)``

    reproduces the published quantity and vanishes when the manifold captures
    the drift exactly.  The factor matters: thresholds quoted for AVQDS are in
    this convention, and dropping it silently halves every cut, which is how an
    earlier comparison ran the comparator at half its intended threshold.

    This is an absolute quantity with units of energy squared, unlike this
    route's own normalized residual ratio; the two gates are not interchangeable
    and a run records both.
    """

    K = np.asarray(evaluation.geometry.K, dtype=float)
    f = np.asarray(evaluation.geometry.f, dtype=float).reshape(-1)
    norm_b_sq = float(evaluation.geometry.norm_b_sq)
    if f.size == 0:
        return float(2.0 * norm_b_sq)
    solve = solve_theta_dot(K, f, policy=inverse_policy)
    return float(max(0.0, 2.0 * (norm_b_sq - float(solve.captured_drift))))


@dataclass(frozen=True)
class AVQDSDecision:
    """One checkpoint's AVQDS outcome."""

    appended_atom_ids: tuple[str, ...] = ()
    l2_before: float = 0.0
    l2_after: float = 0.0
    candidates_scored: int = 0
    stop_reason: str = "below_threshold"
    state: APMcLachlanState | None = None
    theta: np.ndarray | None = None
    evaluation: GeometryEvaluation | None = None

    @property
    def accepted(self) -> bool:
        return bool(self.appended_atom_ids)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "policy": AVQDS_POLICY_V1,
            "appended_atom_ids": [str(a) for a in self.appended_atom_ids],
            "l2_before": float(self.l2_before),
            "l2_after": float(self.l2_after),
            "candidates_scored": int(self.candidates_scored),
            "stop_reason": str(self.stop_reason),
        }


def select_avqds_appends(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    evaluation: GeometryEvaluation,
    atoms_by_id: Mapping[str, Any],
    occurrence_label: Callable[[Any, int, int], str],
    inverse_policy: McLachlanInversePolicy,
    l2_cut: float,
    max_appends_per_checkpoint: int | None = None,
) -> AVQDSDecision:
    """Greedy AVQDS appends until the McLachlan distance falls below ``l2_cut``.

    The rule's own stopping conditions are the threshold and the absence of an
    improving candidate; both are algorithmic.  ``max_appends_per_checkpoint``
    is a safety valve, not part of the method, and defaults to unbounded --
    capping it turns "append until satisfied" into "append k per checkpoint",
    which is a different algorithm and was how an earlier comparison
    misrepresented this comparator.
    """

    if float(l2_cut) < 0.0:
        raise ValueError("l2_cut must be non-negative.")

    current_state = state
    current_theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
    current_eval = evaluation
    l2_initial = mclachlan_distance_squared(current_eval, inverse_policy=inverse_policy)
    l2 = l2_initial
    appended: list[str] = []
    scored = 0
    stop_reason = "below_threshold"

    budget = (
        None if max_appends_per_checkpoint is None
        else max(0, int(max_appends_per_checkpoint))
    )
    round_index = -1
    while budget is None or round_index + 1 < budget:
        round_index += 1
        if l2 <= float(l2_cut):
            stop_reason = "below_threshold"
            break
        best = None  # (l2_candidate, atom_id, state, theta, evaluation)
        cut = int(current_state.runtime_parameter_count)
        # Iterate in the pool's frozen order, not sorted order: ties then break
        # the same way they do for the exchange route, so a comparison between
        # the two reflects their rules rather than their candidate ordering.
        for atom_id in atoms_by_id:
            atom = atoms_by_id[atom_id]
            label = occurrence_label(atom, cut, len(appended) + round_index)
            try:
                cand_state, cand_theta = state_with_inserted_runtime_coordinates(
                    current_state,
                    insertions=((cut, atom.term, label),),
                    theta_runtime=current_theta,
                )
                cand_eval = evaluate_mclachlan_geometry(
                    state=cand_state,
                    hamiltonian=hamiltonian,
                    theta_runtime=cand_theta,
                    time=float(time),
                    include_tangent_matrix=True,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            scored += 1
            cand_l2 = mclachlan_distance_squared(
                cand_eval, inverse_policy=inverse_policy
            )
            if not np.isfinite(cand_l2):
                continue
            if best is None or cand_l2 < best[0]:
                best = (cand_l2, str(atom_id), cand_state, cand_theta, cand_eval)
        if best is None:
            stop_reason = "no_finite_candidate"
            break
        if best[0] >= l2:
            stop_reason = "no_improving_candidate"
            break
        l2, atom_id, current_state, current_theta, current_eval = best
        appended.append(atom_id)
    else:
        stop_reason = "append_budget_exhausted" if l2 > float(l2_cut) else "below_threshold"

    return AVQDSDecision(
        appended_atom_ids=tuple(appended),
        l2_before=float(l2_initial),
        l2_after=float(l2),
        candidates_scored=int(scored),
        stop_reason=str(stop_reason),
        state=current_state if appended else None,
        theta=current_theta if appended else None,
        evaluation=current_eval if appended else None,
    )


__all__ = [
    "AVQDS_POLICY_V1",
    "AVQDSDecision",
    "mclachlan_distance_squared",
    "select_avqds_appends",
]
