"""Measurement-free permission for generalized-exchange deletion sets.

This module answers one question before a deletion-containing patch may enter
the structural search: can the current checkpoint data already certify that
the proposed deletion is both state-local and tangent-redundant?

For runtime rotations ``exp(-i theta_mu c_mu P_mu)``, unitary invariance and
the Fubini--Study triangle inequality give the conservative bound

    d_ray(psi, psi_without_D)
        <= sin(min(pi/2, sum_{mu in D} |theta_mu c_mu|)).

The frozen-checkpoint loss of captured drift is the reverse-Schur identity

    Q(J) - Q(R) = r_{D|R}^T S_{D|R}^+ r_{D|R} >= 0.

Both quantities use angles, generator coefficients, and ``(G, f)`` that the
checkpoint already owns.  No candidate state, overlap circuit, or additional
quantum measurement is required.  Passing this permission is nomination, not
commit authority: the realized patched solve and the temporarily retained
continuity certification still arbitrate the materialized finalist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.schur_identity import (
    exact_deletion_loss,
)


DELETION_PERMISSION_SCHEMA_V1 = "paper_ii_measurement_free_deletion_permission_v1"
DELETION_PERMISSION_ANGLE_RAY = "angle_ray_upper_bound_above_max"
DELETION_PERMISSION_SCHUR_LOSS = "normalized_schur_loss_above_max"


@dataclass(frozen=True)
class DeletionPermissionDecision:
    """One deterministic permission decision for a deletion set."""

    deleted_runtime_indices: tuple[int, ...]
    retained_runtime_indices: tuple[int, ...]
    permitted: bool
    reasons: tuple[str, ...]
    effective_angle_l1: float
    ray_distance_upper_bound: float
    exact_schur_loss: float | None
    normalized_schur_loss: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": DELETION_PERMISSION_SCHEMA_V1,
            "deleted_runtime_indices": [
                int(i) for i in self.deleted_runtime_indices
            ],
            "retained_runtime_indices": [
                int(i) for i in self.retained_runtime_indices
            ],
            "permitted": bool(self.permitted),
            "reasons": [str(reason) for reason in self.reasons],
            "effective_angle_l1": float(self.effective_angle_l1),
            "ray_distance_upper_bound": float(self.ray_distance_upper_bound),
            "exact_schur_loss": (
                None
                if self.exact_schur_loss is None
                else float(self.exact_schur_loss)
            ),
            "normalized_schur_loss": (
                None
                if self.normalized_schur_loss is None
                else float(self.normalized_schur_loss)
            ),
        }


class DeletionPermissionEvaluator:
    """Memoized evaluator over one frozen checkpoint geometry.

    The interface deliberately exposes only ``assess`` and ``summary``.  The
    former is the structural-search seam; the latter is the reproduction and
    measurement-savings ledger emitted with the checkpoint decision.
    """

    def __init__(
        self,
        *,
        gram: np.ndarray,
        force: np.ndarray,
        norm_b_sq: float,
        theta_runtime: np.ndarray,
        rotation_coefficients: np.ndarray,
        ray_distance_max: float,
        normalized_schur_loss_max: float,
        epsilon_norm: float,
    ) -> None:
        K = np.asarray(gram, dtype=float)
        f = np.asarray(force, dtype=float).reshape(-1)
        theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
        coefficients = np.asarray(rotation_coefficients, dtype=float).reshape(-1)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("gram must be square.")
        n = int(K.shape[0])
        if f.size != n or theta.size != n or coefficients.size != n:
            raise ValueError(
                "gram, force, theta_runtime, and rotation_coefficients must "
                "describe the same runtime-coordinate count."
            )
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(f)):
            raise ValueError("gram and force must be finite.")
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("runtime angles and rotation coefficients must be finite.")
        if not np.isfinite(float(norm_b_sq)) or float(norm_b_sq) < 0.0:
            raise ValueError("norm_b_sq must be finite and non-negative.")
        if float(ray_distance_max) <= 0.0:
            raise ValueError("ray_distance_max must be positive.")
        if float(normalized_schur_loss_max) <= 0.0:
            raise ValueError("normalized_schur_loss_max must be positive.")
        if float(epsilon_norm) <= 0.0:
            raise ValueError("epsilon_norm must be positive.")

        self._gram = 0.5 * (K + K.T)
        self._force = f
        self._norm_b_sq = float(norm_b_sq)
        self._theta = theta
        self._coefficients = coefficients
        self._ray_distance_max = float(ray_distance_max)
        self._normalized_schur_loss_max = float(normalized_schur_loss_max)
        self._epsilon_norm = float(epsilon_norm)
        self._memo: dict[tuple[int, ...], DeletionPermissionDecision] = {}

    def assess(
        self, deleted_runtime_indices: Sequence[int]
    ) -> DeletionPermissionDecision:
        """Return the permission decision for one canonicalized deletion set."""

        deleted = tuple(sorted({int(i) for i in deleted_runtime_indices}))
        cached = self._memo.get(deleted)
        if cached is not None:
            return cached
        n = int(self._force.size)
        if any(i < 0 or i >= n for i in deleted):
            raise ValueError("deleted runtime index is outside the checkpoint support.")
        deleted_set = set(deleted)
        retained = tuple(i for i in range(n) if i not in deleted_set)

        effective_angle_l1 = float(
            sum(abs(float(self._theta[i] * self._coefficients[i])) for i in deleted)
        )
        ray_upper = float(math.sin(min(math.pi / 2.0, effective_angle_l1)))
        reasons: list[str] = []
        if ray_upper > self._ray_distance_max:
            reasons.append(DELETION_PERMISSION_ANGLE_RAY)
            # The permission is conjunctive, so a failed state-locality bound
            # makes the more expensive Schur pseudoinverses unnecessary.
            schur_loss = None
            normalized_schur_loss = None
        else:
            schur_loss = float(
                exact_deletion_loss(
                    self._gram,
                    self._force,
                    retained=retained,
                    deleted=deleted,
                )
            )
            normalized_schur_loss = float(
                schur_loss / (self._norm_b_sq + self._epsilon_norm)
            )
            if normalized_schur_loss > self._normalized_schur_loss_max:
                reasons.append(DELETION_PERMISSION_SCHUR_LOSS)
        decision = DeletionPermissionDecision(
            deleted_runtime_indices=deleted,
            retained_runtime_indices=retained,
            permitted=not reasons,
            reasons=tuple(reasons),
            effective_angle_l1=effective_angle_l1,
            ray_distance_upper_bound=ray_upper,
            exact_schur_loss=schur_loss,
            normalized_schur_loss=normalized_schur_loss,
        )
        self._memo[deleted] = decision
        return decision

    def summary(self) -> dict[str, Any]:
        """Return unique-set counts and bounds for decision provenance."""

        decisions = tuple(self._memo.values())
        reason_counts: dict[str, int] = {}
        for decision in decisions:
            for reason in decision.reasons:
                reason_counts[str(reason)] = int(reason_counts.get(str(reason), 0) + 1)
        permitted = tuple(d for d in decisions if d.permitted)
        rejected = tuple(d for d in decisions if not d.permitted)
        return {
            "schema": DELETION_PERMISSION_SCHEMA_V1,
            "ray_distance_max": float(self._ray_distance_max),
            "normalized_schur_loss_max": float(
                self._normalized_schur_loss_max
            ),
            "evaluated_deletion_set_count": int(len(decisions)),
            "permitted_deletion_set_count": int(len(permitted)),
            "rejected_deletion_set_count": int(len(rejected)),
            "schur_evaluated_deletion_set_count": int(
                sum(d.normalized_schur_loss is not None for d in decisions)
            ),
            "schur_skipped_by_angle_count": int(
                sum(d.normalized_schur_loss is None for d in decisions)
            ),
            "rejection_reason_counts": reason_counts,
            "max_permitted_ray_distance_upper_bound": (
                None
                if not permitted
                else float(max(d.ray_distance_upper_bound for d in permitted))
            ),
            "max_permitted_normalized_schur_loss": (
                None
                if not permitted
                else float(
                    max(
                        d.normalized_schur_loss
                        for d in permitted
                        if d.normalized_schur_loss is not None
                    )
                )
            ),
        }


__all__ = [
    "DELETION_PERMISSION_ANGLE_RAY",
    "DELETION_PERMISSION_SCHEMA_V1",
    "DELETION_PERMISSION_SCHUR_LOSS",
    "DeletionPermissionDecision",
    "DeletionPermissionEvaluator",
]
