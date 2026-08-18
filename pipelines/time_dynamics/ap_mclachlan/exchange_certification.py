"""Finalist materialization and commit certification for the exchange selector.

Structural enumeration never builds an ANZATS; only ranked finalists reach
this module.  For one finalist ``(D, I)`` it materializes the commit word
``W_D`` — delete ``D``, insert the plan's children at zero angle at their
deletion-remapped cuts, preserve surviving parameters — optionally applies an
injected local refit, rebuilds the patched geometry and fixed-step solve under
the propagation policy, and evaluates the hard commit gates:

* finite patched solve and retained-solve conditioning bound (every
  finalist); and,
* for **deletion-containing** finalists only, the Fubini--Study ray
  displacement and the phase-aligned state-space velocity smoothness
  ``eta = ||v_patched - v_base||^2 / (||b||^2 + eps) <= tau_vel``.
  These are the route's deletion commit checks; a pure insertion exists to
  change the realized velocity and starts at zero angle on the same ray, so
  the deletion gates are computed for telemetry but not enforced on it.

A failed finalist is rejected whole (patch atomicity) and certification
advances to the next candidate; when none passes, the selector returns stay.
Smoothness is a hard gate here, never a ranking penalty — ranking happened at
the structural stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.fixed_step import (
    FixedMcLachlanStep,
    SolveRepairConfig,
    solve_fixed_mclachlan_step,
    solve_fixed_mclachlan_step_with_repair,
)
from pipelines.time_dynamics.ap_mclachlan.geometry import residual_denominator
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    GeometryEvaluation,
    evaluate_mclachlan_geometry,
    state_space_velocity_from_evaluation,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanState,
    APMcLachlanStateParityError,
    state_with_inserted_runtime_coordinates,
    state_without_runtime_indices,
)


CERTIFICATION_SCHEMA_V1 = "ap_mclachlan_exchange_certification_v1"


def remap_cuts_after_deletion(
    cuts: Sequence[int],
    removed_runtime_indices: Sequence[int],
) -> tuple[int, ...]:
    """Map original-layout cuts onto the post-deletion survivor sequence.

    A cut counts the survivors of the *original* word that precede it, so
    after deleting ``D`` each cut drops by the number of deleted coordinates
    strictly below it.
    """

    removed = sorted(set(int(i) for i in removed_runtime_indices))
    out = []
    for cut in cuts:
        c = int(cut)
        out.append(c - sum(1 for r in removed if r < c))
    return tuple(out)


@dataclass(frozen=True)
class CertificationGates:
    """Hard commit gates; every threshold is explicit."""

    ray_distance_max: float = 5.0e-2
    smoothness_eta_max: float = 1.0e-3
    condition_number_max: float | None = 1.0e12

    def __post_init__(self) -> None:
        if float(self.ray_distance_max) <= 0.0:
            raise ValueError("ray_distance_max must be positive.")
        if float(self.smoothness_eta_max) <= 0.0:
            raise ValueError("smoothness_eta_max must be positive.")
        if (
            self.condition_number_max is not None
            and float(self.condition_number_max) <= 0.0
        ):
            raise ValueError("condition_number_max must be positive when set.")


@dataclass(frozen=True)
class CertificationResult:
    """Outcome of certifying one finalist; telemetry-complete."""

    certified: bool
    reason: str
    ray_distance: float | None = None
    smoothness_eta: float | None = None
    phase_alignment_abs_overlap: float | None = None
    condition_number: float | None = None
    state: APMcLachlanState | None = None
    theta: np.ndarray | None = None
    evaluation: GeometryEvaluation | None = None
    step: FixedMcLachlanStep | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": CERTIFICATION_SCHEMA_V1,
            "certified": bool(self.certified),
            "reason": str(self.reason),
            "ray_distance": self.ray_distance,
            "smoothness_eta": self.smoothness_eta,
            "phase_alignment_abs_overlap": self.phase_alignment_abs_overlap,
            "condition_number": self.condition_number,
        }


def _ray_distance(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    overlap = complex(np.vdot(
        np.asarray(psi_a, dtype=complex).reshape(-1),
        np.asarray(psi_b, dtype=complex).reshape(-1),
    ))
    return float(math.sqrt(max(0.0, 1.0 - min(1.0, abs(overlap) ** 2))))


def certify_finalist(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    removed_runtime_indices: Sequence[int],
    insertions: Sequence[tuple[int, Any, str]],
    inverse_policy: McLachlanInversePolicy,
    gates: CertificationGates,
    refit: Callable[[APMcLachlanState, np.ndarray], tuple[APMcLachlanState, np.ndarray]]
    | None = None,
    solve_repair_config: SolveRepairConfig | None = None,
) -> CertificationResult:
    """Materialize one finalist and run every hard commit gate.

    ``insertions`` carry *original-layout* cuts; they are remapped over the
    deletion set here.  ``refit`` is the optional local refit applied to the
    materialized state before gating (identity when ``None``); it must return
    a state over the same coordinate word.
    """

    removed = tuple(sorted(set(int(i) for i in removed_runtime_indices)))
    try:
        pruned_state, pruned_theta = state_without_runtime_indices(
            state, removed, theta_runtime=theta_runtime
        )
        remapped = remap_cuts_after_deletion(
            [int(cut) for cut, _term, _label in insertions], removed
        )
        entries = tuple(
            (int(new_cut), term, str(label))
            for new_cut, (_old, term, label) in zip(remapped, insertions)
        )
        patched_state, patched_theta = state_with_inserted_runtime_coordinates(
            pruned_state, insertions=entries, theta_runtime=pruned_theta
        )
        if refit is not None:
            patched_state, patched_theta = refit(patched_state, patched_theta)
            patched_theta = np.asarray(patched_theta, dtype=float).reshape(-1)
    except (ValueError, APMcLachlanStateParityError, np.linalg.LinAlgError) as exc:
        return CertificationResult(
            certified=False, reason=f"materialization_failed:{exc}"
        )

    try:
        patched_evaluation = evaluate_mclachlan_geometry(
            state=patched_state,
            hamiltonian=hamiltonian,
            theta_runtime=patched_theta,
            time=float(time),
            include_tangent_matrix=True,
        )
        if solve_repair_config is not None and bool(solve_repair_config.enabled):
            patched_step = solve_fixed_mclachlan_step_with_repair(
                patched_evaluation.geometry,
                inverse_policy=inverse_policy,
                repair_config=solve_repair_config,
                repair_dt=None,
            )
        else:
            patched_step = solve_fixed_mclachlan_step(
                patched_evaluation.geometry,
                inverse_policy=inverse_policy,
            )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return CertificationResult(
            certified=False, reason=f"patched_solve_failed:{exc}"
        )

    theta_dot = np.asarray(patched_step.theta_dot, dtype=float).reshape(-1)
    if not np.all(np.isfinite(theta_dot)):
        return CertificationResult(certified=False, reason="patched_solve_nonfinite")

    condition = patched_step.condition_number
    if (
        gates.condition_number_max is not None
        and condition is not None
        and float(condition) > float(gates.condition_number_max)
    ):
        return CertificationResult(
            certified=False,
            reason="condition_number_above_max",
            condition_number=float(condition),
        )

    deletion_containing = bool(removed)
    ray = _ray_distance(base_evaluation.psi, patched_evaluation.psi)
    if deletion_containing and ray > float(gates.ray_distance_max):
        return CertificationResult(
            certified=False,
            reason="ray_distance_above_max",
            ray_distance=ray,
            condition_number=None if condition is None else float(condition),
        )

    try:
        base_velocity = state_space_velocity_from_evaluation(
            base_evaluation, base_step.theta_dot
        )
        patched_velocity = state_space_velocity_from_evaluation(
            patched_evaluation, patched_step.theta_dot
        )
        overlap = complex(np.vdot(
            np.asarray(base_evaluation.psi, dtype=complex).reshape(-1),
            np.asarray(patched_evaluation.psi, dtype=complex).reshape(-1),
        ))
        magnitude = abs(overlap)
        phase = overlap / magnitude if magnitude > 0.0 else 1.0 + 0.0j
        aligned = np.conj(phase) * patched_velocity
        delta = aligned - base_velocity
        denom = residual_denominator(
            float(base_evaluation.geometry.norm_b_sq),
            float(inverse_policy.epsilon),
        )
        eta = float(max(0.0, float(np.real(np.vdot(delta, delta)))) / denom)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return CertificationResult(
            certified=False,
            reason=f"smoothness_unavailable:{exc}",
            ray_distance=ray,
        )
    if deletion_containing and eta > float(gates.smoothness_eta_max):
        return CertificationResult(
            certified=False,
            reason="smoothness_eta_above_max",
            ray_distance=ray,
            smoothness_eta=eta,
            phase_alignment_abs_overlap=float(magnitude),
        )

    return CertificationResult(
        certified=True,
        reason="certified",
        ray_distance=ray,
        smoothness_eta=eta,
        phase_alignment_abs_overlap=float(magnitude),
        condition_number=None if condition is None else float(condition),
        state=patched_state,
        theta=patched_theta,
        evaluation=patched_evaluation,
        step=patched_step,
    )


__all__ = [
    "CERTIFICATION_SCHEMA_V1",
    "CertificationGates",
    "CertificationResult",
    "certify_finalist",
    "remap_cuts_after_deletion",
]
