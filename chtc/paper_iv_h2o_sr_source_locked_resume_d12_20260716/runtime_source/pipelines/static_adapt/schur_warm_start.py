"""Schur warm-start seed proposal helpers for static ADAPT.

This module is intentionally pure and optimizer-agnostic.  It does not select
candidates, mutate ADAPT state, consume RNG, or change optimizer budgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHUR_WARM_START_MODE_OFF = "off"
SCHUR_WARM_START_MODE_APPEND = "append"
SCHUR_WARM_START_MODE_PRUNE = "prune"
SCHUR_WARM_START_MODE_APPEND_PRUNE = "append_prune"
SCHUR_WARM_START_MODES = frozenset(
    {
        SCHUR_WARM_START_MODE_OFF,
        SCHUR_WARM_START_MODE_APPEND,
        SCHUR_WARM_START_MODE_PRUNE,
        SCHUR_WARM_START_MODE_APPEND_PRUNE,
    }
)


@dataclass(frozen=True)
class SchurWarmStartPolicy:
    mode: str = SCHUR_WARM_START_MODE_OFF
    require_full_window: bool = True
    sign_policy: str = "evaluate_both"
    require_valid_metric: bool = True
    require_exact_geometry_revision: bool = True
    guard_abs_tol: float = 1.0e-12
    guard_rel_tol: float = 1.0e-12

    @property
    def append_enabled(self) -> bool:
        return self.mode in {SCHUR_WARM_START_MODE_APPEND, SCHUR_WARM_START_MODE_APPEND_PRUNE}

    @property
    def prune_enabled(self) -> bool:
        return self.mode in {SCHUR_WARM_START_MODE_PRUNE, SCHUR_WARM_START_MODE_APPEND_PRUNE}


def normalize_schur_warm_start_mode(value: Any) -> str:
    mode = str(value if value is not None else SCHUR_WARM_START_MODE_OFF).strip().lower().replace("-", "_")
    if mode not in SCHUR_WARM_START_MODES:
        raise ValueError(
            "adapt_schur_warm_start_mode must be one of "
            f"{sorted(SCHUR_WARM_START_MODES)}."
        )
    return mode


@dataclass(frozen=True)
class SchurTrustStep:
    status: str
    reason: str
    alpha_abs: float = 0.0
    at_trust_boundary: bool = False
    predicted_gain: float = 0.0
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeedProposal:
    name: str
    x0: np.ndarray
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WarmStartResult:
    x0: np.ndarray
    status: str
    reason: str
    chosen_source: str
    telemetry: dict[str, Any] = field(default_factory=dict)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def scalar_schur_trust_step(
    *,
    g_lcb: Any,
    h_eff: Any,
    F_red: Any,
    rho: Any,
    metric_floor: Any,
    reduced_metric_collapse_rel_tol: Any,
    F_raw: Any | None = None,
    zero_step_tol: float = 1.0e-15,
) -> SchurTrustStep:
    """Return the conservative scalar append step magnitude.

    The formula mirrors the selector trust model but exposes the coordinate
    magnitude required by Schur warm-starting.  Collapsed metrics fail closed
    instead of converting a floor-dominated metric into a large boundary step.
    """

    g = _finite_float(g_lcb)
    h = _finite_float(h_eff)
    F = _finite_float(F_red)
    rho_f = _finite_float(rho)
    floor = _finite_float(metric_floor)
    collapse_rel = _finite_float(reduced_metric_collapse_rel_tol)
    F_raw_f = _finite_float(F_raw) if F_raw is not None else None
    telemetry = {
        "g_lcb": g,
        "h_eff": h,
        "F_red": F,
        "F_raw": F_raw_f,
        "rho": rho_f,
        "metric_floor": floor,
        "reduced_metric_collapse_rel_tol": collapse_rel,
    }
    if any(x is None for x in (g, h, F, rho_f, floor, collapse_rel)):
        return SchurTrustStep(status="unavailable", reason="nonfinite_geometry", telemetry=telemetry)
    assert g is not None and h is not None and F is not None and rho_f is not None
    assert floor is not None and collapse_rel is not None
    if rho_f < 0.0 or floor < 0.0 or collapse_rel < 0.0:
        return SchurTrustStep(status="unavailable", reason="nonfinite_geometry", telemetry=telemetry)
    collapse_reference = F_raw_f if F_raw_f is not None else F
    collapse_floor = max(float(floor), float(collapse_rel) * max(float(collapse_reference), float(floor)))
    telemetry["collapse_floor"] = float(collapse_floor)
    if F <= collapse_floor:
        return SchurTrustStep(status="unavailable", reason="metric_collapse", telemetry=telemetry)
    if g <= 0.0:
        return SchurTrustStep(status="unavailable", reason="zero_step", telemetry=telemetry)
    F_safe = max(float(F), float(floor))
    if F_safe <= 0.0:
        return SchurTrustStep(status="unavailable", reason="nonfinite_geometry", telemetry=telemetry)
    alpha_max = float(rho_f) / math.sqrt(F_safe)
    if alpha_max <= float(zero_step_tol):
        return SchurTrustStep(status="unavailable", reason="zero_step", telemetry={**telemetry, "alpha_max": alpha_max})
    if h > 0.0:
        alpha_newton = float(g) / float(h)
        alpha = min(float(alpha_newton), float(alpha_max))
        at_boundary = bool(alpha_newton > alpha_max)
    else:
        alpha = float(alpha_max)
        at_boundary = True
    if abs(alpha) <= float(zero_step_tol):
        return SchurTrustStep(status="unavailable", reason="zero_step", telemetry={**telemetry, "alpha_max": alpha_max})
    predicted_gain = float(float(g) * alpha - 0.5 * max(0.0, float(h)) * alpha * alpha)
    return SchurTrustStep(
        status="available",
        reason="ok",
        alpha_abs=float(alpha),
        at_trust_boundary=bool(at_boundary),
        predicted_gain=float(predicted_gain),
        telemetry={**telemetry, "alpha_max": float(alpha_max)},
    )


def propose_prune_schur_seed(
    *,
    canonical_deleted_x0: np.ndarray,
    survivor_reduced_positions: Sequence[int],
    theta_removed: Sequence[float] | np.ndarray,
    g_survivor: Sequence[float] | np.ndarray | None,
    H_survivor_survivor: Sequence[Sequence[float]] | np.ndarray,
    H_survivor_removed: Sequence[Sequence[float]] | np.ndarray,
    ridge_used: Any,
    compensation_solve: Sequence[float] | np.ndarray | None = None,
) -> tuple[SeedProposal | None, dict[str, Any]]:
    """Construct a prune survivor-compensation seed in reduced coordinates.

    The default formula is the full block conditional displacement
    ``delta_W = -R_WW^{-1} (g_W - H_WJ theta_J)`` with
    ``R_WW = sym(H_WW) + ridge_used I``.  If a caller supplies an already clipped
    or trust-limited ``compensation_solve``, that displacement is used directly
    and this helper does not silently recompute different semantics.
    """

    x0 = np.asarray(canonical_deleted_x0, dtype=float).reshape(-1).copy()
    survivor_positions = [int(i) for i in survivor_reduced_positions]
    telemetry: dict[str, Any] = {
        "schema": "static_adapt_schur_warm_start_v1",
        "stage": "prune_refit",
        "survivor_reduced_positions": [int(i) for i in survivor_positions],
    }
    if len(set(survivor_positions)) != len(survivor_positions):
        return None, {**telemetry, "status": "unsupported", "reason": "unsupported_partial_active_window"}
    if any(pos < 0 or pos >= int(x0.size) for pos in survivor_positions):
        return None, {**telemetry, "status": "unsupported", "reason": "unsupported_partial_active_window"}
    theta_J = np.asarray(theta_removed, dtype=float).reshape(-1)
    H_WW = np.asarray(H_survivor_survivor, dtype=float)
    H_WJ = np.asarray(H_survivor_removed, dtype=float)
    nW = int(len(survivor_positions))
    nJ = int(theta_J.size)
    if H_WW.shape != (nW, nW) or H_WJ.shape != (nW, nJ):
        return None, {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
    ridge = _finite_float(ridge_used)
    if ridge is None or ridge < 0.0:
        return None, {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
    if not (np.all(np.isfinite(theta_J)) and np.all(np.isfinite(H_WW)) and np.all(np.isfinite(H_WJ))):
        return None, {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
    if compensation_solve is not None:
        delta_W = np.asarray(compensation_solve, dtype=float).reshape(-1)
        solve_source = "stored"
        if int(delta_W.size) != nW or not np.all(np.isfinite(delta_W)):
            return None, {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
        rhs_norm = None
    else:
        if g_survivor is None:
            return None, {
                **telemetry,
                "status": "unavailable",
                "reason": "unavailable_missing_survivor_gradient",
            }
        g_W = np.asarray(g_survivor, dtype=float).reshape(-1)
        if int(g_W.size) != nW or not np.all(np.isfinite(g_W)):
            return None, {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
        R = 0.5 * (H_WW + H_WW.T) + float(ridge) * np.eye(nW, dtype=float)
        rhs = g_W - H_WJ @ theta_J
        try:
            delta_W = -np.linalg.solve(R, rhs)
        except np.linalg.LinAlgError:
            return None, {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
        solve_source = "computed_full_block"
        rhs_norm = float(np.linalg.norm(rhs))
    if int(delta_W.size) != nW or not np.all(np.isfinite(delta_W)):
        return None, {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
    trial = np.asarray(x0, dtype=float).copy()
    for local_idx, reduced_pos in enumerate(survivor_positions):
        trial[int(reduced_pos)] = float(trial[int(reduced_pos)] + float(delta_W[int(local_idx)]))
    return SeedProposal(
        name="schur_prune_compensation",
        x0=np.asarray(trial, dtype=float).copy(),
        telemetry={
            "solve_source": str(solve_source),
            "ridge_used": float(ridge),
            "survivor_delta_l2": float(np.linalg.norm(delta_W)),
            "compensation_rhs_norm": rhs_norm,
        },
    ), {
        **telemetry,
        "status": "available",
        "reason": "ok",
        "solve_source": str(solve_source),
        "ridge_used": float(ridge),
        "survivor_delta_l2": float(np.linalg.norm(delta_W)),
        "compensation_rhs_norm": rhs_norm,
    }


def propose_append_schur_seeds(
    *,
    canonical_x0: np.ndarray,
    old_window_reduced_positions: Sequence[int],
    candidate_reduced_position: int,
    schur_window_solve: Sequence[float],
    g_lcb: Any,
    g_signed: Any,
    h_eff: Any,
    F_red: Any,
    F_raw: Any,
    rho: Any,
    metric_floor: Any,
    reduced_metric_collapse_rel_tol: Any,
    batch_size: int = 1,
) -> tuple[list[SeedProposal], dict[str, Any]]:
    """Construct both signed append Schur seed proposals in reduced coordinates.

    This compatibility wrapper is the one-runtime-coordinate special case of
    ``propose_append_schur_seeds_lifted``.
    """

    return propose_append_schur_seeds_lifted(
        canonical_x0=canonical_x0,
        old_window_reduced_position_groups=[[int(i)] for i in old_window_reduced_positions],
        candidate_reduced_positions=[int(candidate_reduced_position)],
        schur_window_solve=schur_window_solve,
        g_lcb=g_lcb,
        g_signed=g_signed,
        h_eff=h_eff,
        F_red=F_red,
        F_raw=F_raw,
        rho=rho,
        metric_floor=metric_floor,
        reduced_metric_collapse_rel_tol=reduced_metric_collapse_rel_tol,
        batch_size=batch_size,
        lift_kind="single_runtime_coordinate_v1",
    )


def propose_append_schur_seeds_lifted(
    *,
    canonical_x0: np.ndarray,
    old_window_reduced_position_groups: Sequence[Sequence[int]],
    candidate_reduced_positions: Sequence[int],
    schur_window_solve: Sequence[float],
    g_lcb: Any,
    g_signed: Any,
    h_eff: Any,
    F_red: Any,
    F_raw: Any,
    rho: Any,
    metric_floor: Any,
    reduced_metric_collapse_rel_tol: Any,
    batch_size: int = 1,
    lift_kind: str = "logical_uniform_runtime_block_v1",
    candidate_zero_tol: float = 1.0e-12,
) -> tuple[list[SeedProposal], dict[str, Any]]:
    """Construct signed append Schur proposals with a logical-to-runtime lift.

    ``schur_window_solve`` is in the logical old-window basis.  Each logical
    displacement is applied to every reduced runtime coordinate in the matching
    full block group.  This embeds logical Schur geometry into a per-Pauli
    runtime parameterization without pretending to have per-runtime Schur
    geometry.
    """

    x0 = np.asarray(canonical_x0, dtype=float).reshape(-1).copy()
    telemetry_base: dict[str, Any] = {
        "lift_kind": str(lift_kind),
        "batch_size": int(batch_size),
    }
    if int(batch_size) != 1:
        return [], {
            **telemetry_base,
            "status": "unsupported",
            "reason": "unsupported_batch_size_gt_1",
        }
    old_groups = [[int(i) for i in group] for group in old_window_reduced_position_groups]
    cand_positions = [int(i) for i in candidate_reduced_positions]
    old_positions_flat = [int(pos) for group in old_groups for pos in group]
    all_positions = [*old_positions_flat, *cand_positions]
    telemetry_base.update(
        {
            "old_window_logical_size": int(len(old_groups)),
            "old_window_runtime_size": int(len(old_positions_flat)),
            "candidate_runtime_count": int(len(cand_positions)),
        }
    )
    if not cand_positions:
        return [], {**telemetry_base, "status": "unsupported", "reason": "unsupported_partial_active_window"}
    if any(len(group) <= 0 for group in old_groups):
        return [], {**telemetry_base, "status": "unsupported", "reason": "unsupported_parameterization_basis"}
    if len(set(all_positions)) != len(all_positions):
        return [], {**telemetry_base, "status": "unsupported", "reason": "unsupported_partial_active_window"}
    if any(pos < 0 or pos >= int(x0.size) for pos in all_positions):
        return [], {**telemetry_base, "status": "unsupported", "reason": "unsupported_partial_active_window"}
    candidate_base = np.asarray([float(x0[int(pos)]) for pos in cand_positions], dtype=float)
    candidate_max_abs = float(np.max(np.abs(candidate_base)) if candidate_base.size else 0.0)
    telemetry_base["canonical_candidate_max_abs"] = float(candidate_max_abs)
    if candidate_max_abs > float(candidate_zero_tol):
        return [], {
            **telemetry_base,
            "status": "unavailable",
            "reason": "candidate_not_initialized_at_zero",
        }
    solve = np.asarray([float(x) for x in schur_window_solve], dtype=float).reshape(-1)
    if int(solve.size) != len(old_groups):
        return [], {**telemetry_base, "status": "unsupported", "reason": "unsupported_partial_active_window"}
    if not np.all(np.isfinite(solve)):
        return [], {**telemetry_base, "status": "unavailable", "reason": "nonfinite_geometry"}
    step = scalar_schur_trust_step(
        g_lcb=g_lcb,
        h_eff=h_eff,
        F_red=F_red,
        F_raw=F_raw,
        rho=rho,
        metric_floor=metric_floor,
        reduced_metric_collapse_rel_tol=reduced_metric_collapse_rel_tol,
    )
    if step.status != "available":
        return [], {**telemetry_base, "status": step.status, "reason": step.reason, **dict(step.telemetry)}
    g_signed_f = _finite_float(g_signed)
    predicted_sign = None
    if g_signed_f is not None and g_signed_f != 0.0:
        predicted_sign = int(-1 if g_signed_f > 0.0 else 1)
    signs = [1, -1]
    if predicted_sign in signs:
        signs = [int(predicted_sign), int(-predicted_sign)]
    proposals: list[SeedProposal] = []
    for sign in signs:
        alpha = float(sign) * float(step.alpha_abs)
        trial = np.asarray(x0, dtype=float).copy()
        old_delta_values: list[float] = []
        for local_idx, group in enumerate(old_groups):
            delta = -float(solve[int(local_idx)]) * alpha
            old_delta_values.extend([float(delta)] * len(group))
            for reduced_pos in group:
                trial[int(reduced_pos)] = float(trial[int(reduced_pos)] + float(delta))
        for reduced_pos in cand_positions:
            trial[int(reduced_pos)] = float(trial[int(reduced_pos)] + alpha)
        proposals.append(
            SeedProposal(
                name=f"schur_append_sign_{int(sign):+d}",
                x0=np.asarray(trial, dtype=float).copy(),
                telemetry={
                    **telemetry_base,
                    "sign": int(sign),
                    "alpha": float(alpha),
                    "alpha_abs": float(step.alpha_abs),
                    "predicted_sign": predicted_sign,
                    "at_trust_boundary": bool(step.at_trust_boundary),
                    "predicted_gain": float(step.predicted_gain),
                    "old_window_delta_l2": float(np.linalg.norm(np.asarray(old_delta_values, dtype=float))),
                    "candidate_delta_l2": float(math.sqrt(len(cand_positions)) * abs(alpha)),
                    "runtime_delta_l2": float(np.linalg.norm(trial - x0)),
                },
            )
        )
    return proposals, {
        **telemetry_base,
        "status": "available",
        "reason": "ok",
        "predicted_sign": predicted_sign,
        "alpha_abs": float(step.alpha_abs),
        "at_trust_boundary": bool(step.at_trust_boundary),
        "predicted_gain": float(step.predicted_gain),
    }


def propose_batch_append_schur_seed(
    *,
    canonical_x0: np.ndarray,
    old_window_reduced_positions: Sequence[int],
    candidate_reduced_positions: Sequence[int],
    schur_window_solves: Sequence[Sequence[float]],
    joint_alpha_abs: Sequence[float],
    g_signed_values: Sequence[Any],
    G_reduced: Sequence[Sequence[float]] | np.ndarray | None = None,
    rho: Any | None = None,
    candidate_keys: Sequence[Any] | None = None,
) -> tuple[list[SeedProposal], dict[str, Any]]:
    """Construct a guarded v1 batch insertion Schur seed.

    This helper intentionally uses the reduced-plane batch geometry already
    computed for admission: a common old window, per-candidate ``R^{-1}b``
    solves, and the joint trust step magnitudes.  It does not claim to be a
    full block-Newton batch Schur solve because the current selector does not
    persist candidate-candidate Hessian off-diagonals.  The returned proposal
    must therefore be accepted only through direct objective guarding.
    """

    x0 = np.asarray(canonical_x0, dtype=float).reshape(-1).copy()
    old_positions = [int(i) for i in old_window_reduced_positions]
    cand_positions = [int(i) for i in candidate_reduced_positions]
    candidate_count = int(len(cand_positions))
    telemetry: dict[str, Any] = {
        "status": "unavailable",
        "reason": "missing_geometry",
        "batch_size": int(candidate_count),
        "batch_model": "linear_window_superposition_diag_candidate_curvature_v1",
        "h_alphaalpha_offdiag_available": False,
        "candidate_curvature_model": "diagonal_h_eff_only",
        "prediction_authority": "seed_proposal_only_guarded_by_objective",
        "joint_alpha_source": "reduced_plane_batch_select.alpha",
        "guard_required": True,
        "old_window_size": int(len(old_positions)),
        "candidate_keys": [str(x) for x in candidate_keys] if candidate_keys is not None else None,
    }
    if candidate_count <= 1:
        return [], {**telemetry, "status": "unsupported", "reason": "unsupported_batch_size_le_1"}
    if len(set(old_positions)) != len(old_positions) or len(set(cand_positions)) != len(cand_positions):
        return [], {**telemetry, "status": "unsupported", "reason": "unsupported_partial_active_window"}
    if set(old_positions).intersection(set(cand_positions)):
        return [], {**telemetry, "status": "unsupported", "reason": "unsupported_partial_active_window"}
    if any(pos < 0 or pos >= int(x0.size) for pos in [*old_positions, *cand_positions]):
        return [], {**telemetry, "status": "unsupported", "reason": "unsupported_partial_active_window"}
    candidate_base = np.asarray([float(x0[int(pos)]) for pos in cand_positions], dtype=float)
    telemetry["canonical_candidate_max_abs"] = float(
        np.max(np.abs(candidate_base)) if candidate_base.size else 0.0
    )
    if candidate_base.size and float(np.max(np.abs(candidate_base))) > 1.0e-12:
        return [], {
            **telemetry,
            "status": "unsupported",
            "reason": "candidate_not_initialized_at_zero",
        }
    if len(schur_window_solves) != candidate_count or len(joint_alpha_abs) != candidate_count or len(g_signed_values) != candidate_count:
        return [], {**telemetry, "status": "unavailable", "reason": "missing_geometry"}
    solves: list[np.ndarray] = []
    for raw_solve in schur_window_solves:
        solve = np.asarray([float(x) for x in raw_solve], dtype=float).reshape(-1)
        if int(solve.size) != len(old_positions):
            return [], {**telemetry, "status": "unsupported", "reason": "unsupported_partial_active_window"}
        if not np.all(np.isfinite(solve)):
            return [], {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
        solves.append(solve)
    alpha_abs = np.asarray([float(x) for x in joint_alpha_abs], dtype=float).reshape(-1)
    if int(alpha_abs.size) != candidate_count or not np.all(np.isfinite(alpha_abs)):
        return [], {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
    if np.any(alpha_abs < 0.0):
        return [], {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
    if not np.any(np.abs(alpha_abs) > 1.0e-15):
        return [], {**telemetry, "status": "unavailable", "reason": "zero_step"}
    signs: list[int] = []
    for value in g_signed_values:
        signed = _finite_float(value)
        if signed is None or signed == 0.0:
            return [], {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
        signs.append(int(-1 if signed > 0.0 else 1))
    alpha_signed = np.asarray(
        [float(signs[idx]) * float(alpha_abs[idx]) for idx in range(candidate_count)],
        dtype=float,
    )
    if G_reduced is not None:
        G = np.asarray(G_reduced, dtype=float)
        if G.shape != (candidate_count, candidate_count) or not np.all(np.isfinite(G)):
            return [], {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
        rho_f = _finite_float(rho)
        if rho_f is None or rho_f < 0.0:
            return [], {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
        metric_norm_sq = float(alpha_signed.T @ G @ alpha_signed)
        telemetry["joint_metric_norm_sq"] = float(metric_norm_sq)
        telemetry["rho_sq"] = float(rho_f * rho_f)
        if metric_norm_sq > float(rho_f * rho_f) * (1.0 + 1.0e-8) + 1.0e-12:
            return [], {**telemetry, "status": "unsupported", "reason": "metric_collapse"}
    old_delta = np.zeros(len(old_positions), dtype=float)
    for idx, solve in enumerate(solves):
        old_delta -= solve * float(alpha_signed[idx])
    trial = np.asarray(x0, dtype=float).copy()
    for local_idx, reduced_pos in enumerate(old_positions):
        trial[int(reduced_pos)] = float(trial[int(reduced_pos)] + float(old_delta[int(local_idx)]))
    for idx, reduced_pos in enumerate(cand_positions):
        trial[int(reduced_pos)] = float(alpha_signed[int(idx)])
    if not np.all(np.isfinite(trial)):
        return [], {**telemetry, "status": "unavailable", "reason": "nonfinite_geometry"}
    proposal_telemetry = {
        **telemetry,
        "status": "available",
        "reason": "ok",
        "alpha_abs": [float(x) for x in alpha_abs.tolist()],
        "alpha_signed": [float(x) for x in alpha_signed.tolist()],
        "predicted_signs": [int(x) for x in signs],
        "old_window_delta_l2": float(np.linalg.norm(old_delta)),
        "candidate_delta_l2": float(np.linalg.norm(alpha_signed)),
        "runtime_delta_l2": float(np.linalg.norm(trial - x0)),
    }
    return [
        SeedProposal(
            name="schur_batch_append_joint_alpha_v1",
            x0=np.asarray(trial, dtype=float).copy(),
            telemetry=proposal_telemetry,
        )
    ], proposal_telemetry


def select_guarded_seed(
    *,
    objective: Callable[[np.ndarray], float],
    incumbent_x0: np.ndarray,
    proposals: Sequence[SeedProposal],
    guard_abs_tol: float = 1.0e-12,
    guard_rel_tol: float = 1.0e-12,
    incumbent_energy: float | None = None,
    max_objective_evals: int | None = None,
) -> WarmStartResult:
    """Choose a proposal only if it materially improves the incumbent seed."""

    incumbent = np.asarray(incumbent_x0, dtype=float).reshape(-1).copy()
    telemetry: dict[str, Any] = {"proposal_count": int(len(proposals)), "evaluations": []}
    eval_budget = None if max_objective_evals is None else int(max_objective_evals)
    eval_count = 0

    def _eval(name: str, x: np.ndarray) -> float | None:
        nonlocal eval_count
        if eval_budget is not None and eval_count >= eval_budget:
            telemetry["budget_exhausted"] = True
            return None
        try:
            value = float(objective(np.asarray(x, dtype=float).copy()))
        except Exception as exc:
            telemetry["evaluations"].append({"name": name, "status": "exception", "exception": exc.__class__.__name__})
            return None
        eval_count += 1
        telemetry["evaluations"].append({"name": name, "status": "ok", "energy": value})
        return value if math.isfinite(value) else None

    E_inc = incumbent_energy if incumbent_energy is not None else _eval("incumbent", incumbent)
    if E_inc is None or not math.isfinite(float(E_inc)):
        return WarmStartResult(
            x0=incumbent.copy(),
            status="error",
            reason="nonfinite_objective",
            chosen_source="incumbent",
            telemetry={**telemetry, "guard_objective_evals": int(eval_count)},
        )
    E_inc = float(E_inc)
    tol = float(guard_abs_tol) + float(guard_rel_tol) * max(1.0, abs(E_inc))
    best: tuple[float, int, SeedProposal] | None = None
    for proposal_idx, proposal in enumerate(proposals):
        prop_x = np.asarray(proposal.x0, dtype=float).reshape(-1)
        if prop_x.shape != incumbent.shape or not np.all(np.isfinite(prop_x)):
            telemetry["evaluations"].append({"name": proposal.name, "status": "shape_or_nonfinite"})
            continue
        E_prop = _eval(proposal.name, prop_x)
        if E_prop is None or not math.isfinite(float(E_prop)):
            continue
        E_prop = float(E_prop)
        if E_prop > E_inc + tol:
            continue
        if E_prop < E_inc - tol:
            if best is None or E_prop < best[0] - tol:
                best = (float(E_prop), int(proposal_idx), proposal)
    telemetry.update({"incumbent_energy": float(E_inc), "guard_tolerance": float(tol), "guard_objective_evals": int(eval_count)})
    if best is None:
        return WarmStartResult(
            x0=incumbent.copy(),
            status="rejected",
            reason="no_material_improvement",
            chosen_source="incumbent",
            telemetry=telemetry,
        )
    E_best, _idx, proposal = best
    return WarmStartResult(
        x0=np.asarray(proposal.x0, dtype=float).reshape(-1).copy(),
        status="accepted",
        reason="accepted_material_improvement",
        chosen_source=str(proposal.name),
        telemetry={**telemetry, "chosen_energy": float(E_best), "chosen_proposal": dict(proposal.telemetry)},
    )
