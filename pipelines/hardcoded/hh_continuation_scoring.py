#!/usr/bin/env python3
"""Scoring and proxy accounting for HH continuation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_continuation_types import (
    CandidateFeatures,
    CompileCostEstimate,
    CurvatureOracle,
    MeasurementCacheStats,
    MeasurementPlan,
    NoveltyOracle,
)
from pipelines.hardcoded.hh_continuation_motifs import motif_bonus_for_generator
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_actions import apply_compiled_pauli


@dataclass(frozen=True)
class SimpleScoreConfig:
    lambda_F: float = 1.0
    lambda_compile: float = 0.05
    lambda_measure: float = 0.02
    lambda_leak: float = 0.0
    z_alpha: float = 0.0
    wD: float = 0.0
    wG: float = 0.0
    wC: float = 0.0
    wc: float = 0.0
    compile_cx_proxy_weight: float = 1.0
    compile_sq_proxy_weight: float = 0.5
    compile_rotation_step_weight: float = 1.0
    compile_position_shift_weight: float = 1.0
    compile_refit_active_weight: float = 1.0
    measure_groups_weight: float = 1.0
    measure_shots_weight: float = 1.0
    measure_reuse_weight: float = 1.0
    opt_dim_cost_scale: float = 1.0
    family_repeat_cost_scale: float = 1.0
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    family_ref: float = 1.0
    lifetime_cost_mode: str = "off"
    score_version: str = "simple_v1"


@dataclass(frozen=True)
class FullScoreConfig:
    z_alpha: float = 0.0
    lambda_F: float = 1.0
    lambda_H: float = 1e-6
    rho: float = 0.25
    eta_L: float = 0.0
    gamma_N: float = 1.0
    wD: float = 0.2
    wG: float = 0.15
    wC: float = 0.15
    wP: float = 0.1
    wc: float = 0.1
    compile_cx_proxy_weight: float = 1.0
    compile_sq_proxy_weight: float = 0.5
    compile_rotation_step_weight: float = 1.0
    compile_position_shift_weight: float = 1.0
    compile_refit_active_weight: float = 1.0
    measure_groups_weight: float = 1.0
    measure_shots_weight: float = 1.0
    measure_reuse_weight: float = 1.0
    opt_dim_cost_scale: float = 1.0
    family_repeat_cost_scale: float = 1.0
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    optdim_ref: float = 1.0
    reuse_ref: float = 1.0
    family_ref: float = 1.0
    novelty_eps: float = 1e-6
    cheap_score_eps: float = 1e-12
    shortlist_fraction: float = 0.2
    shortlist_size: int = 12
    phase2_frontier_ratio: float = 0.9
    phase3_frontier_ratio: float = 0.9
    batch_target_size: int = 2
    batch_size_cap: int = 3
    batch_near_degenerate_ratio: float = 0.9
    batch_rank_rel_tol: float = 1e-6
    batch_additivity_tol: float = 0.25
    duplicate_penalty_weight: float = 0.0
    compat_overlap_weight: float = 0.4
    compat_comm_weight: float = 0.2
    compat_curv_weight: float = 0.2
    compat_sched_weight: float = 0.2
    compat_measure_weight: float = 0.2
    leakage_cap: float = 1e6
    lifetime_cost_mode: str = "off"
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_weight: float = 0.05
    motif_bonus_weight: float = 0.05
    metric_floor: float = 1e-12
    reduced_metric_collapse_rel_tol: float = 1e-8
    ridge_growth_factor: float = 10.0
    ridge_max_steps: int = 12
    score_version: str = "full_v2"


@dataclass(frozen=True)
class _ScaffoldDerivativeContext:
    psi_state: np.ndarray
    hpsi_state: np.ndarray
    refit_window_indices: tuple[int, ...]
    dpsi_window: tuple[np.ndarray, ...]
    tangents_window: tuple[np.ndarray, ...]
    Q_window: np.ndarray
    H_window_hessian: np.ndarray


class Phase1CompileCostOracle:
    """Built-in math expression:
    D_proxy = gate_proxy + shift_span + active_count
    """

    @staticmethod
    def _pauli_weight(label_exyz: str) -> int:
        return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))

    @staticmethod
    def _pauli_xy_count(label_exyz: str) -> int:
        return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y"}))

    @classmethod
    def _cx_proxy_term(cls, label_exyz: str) -> int:
        return int(2 * max(cls._pauli_weight(label_exyz) - 1, 0))

    @classmethod
    def _sq_proxy_term(cls, label_exyz: str) -> int:
        weight = cls._pauli_weight(label_exyz)
        if weight <= 0:
            return 0
        return int(2 * cls._pauli_xy_count(label_exyz) + 1)

    def estimate(
        self,
        *,
        candidate_term_count: int,
        position_id: int,
        append_position: int,
        refit_active_count: int,
        candidate_term: Any | None = None,
    ) -> CompileCostEstimate:
        candidate_labels = _pauli_labels_from_term(candidate_term)
        active_labels = [str(lbl) for lbl in candidate_labels if _pauli_weight_exyz(str(lbl)) > 0]
        if active_labels:
            new_pauli_actions = float(len(active_labels))
            new_rotation_steps = float(len(active_labels))
            cx_proxy_total = float(sum(self._cx_proxy_term(lbl) for lbl in active_labels))
            sq_proxy_total = float(sum(self._sq_proxy_term(lbl) for lbl in active_labels))
            gate_proxy_total = float(cx_proxy_total + 0.5 * sq_proxy_total)
            max_pauli_weight = float(max(self._pauli_weight(lbl) for lbl in active_labels))
        else:
            fallback_count = float(max(1, int(candidate_term_count)))
            new_pauli_actions = fallback_count
            new_rotation_steps = fallback_count
            cx_proxy_total = fallback_count
            sq_proxy_total = fallback_count
            gate_proxy_total = fallback_count
            max_pauli_weight = 0.0
        position_shift_span = float(abs(int(append_position) - int(position_id)))
        refit_active = float(max(0, int(refit_active_count)))
        total = float(gate_proxy_total + position_shift_span + refit_active)
        return CompileCostEstimate(
            new_pauli_actions=new_pauli_actions,
            new_rotation_steps=new_rotation_steps,
            position_shift_span=position_shift_span,
            refit_active_count=refit_active,
            proxy_total=total,
            cx_proxy_total=cx_proxy_total,
            sq_proxy_total=sq_proxy_total,
            gate_proxy_total=gate_proxy_total,
            max_pauli_weight=max_pauli_weight,
        )


class MeasurementCacheAudit:
    """Phase 1 accounting-only grouped reuse tracker."""

    def __init__(
        self,
        nominal_shots_per_group: int = 1,
        *,
        plan_version: str = "phase1_qwc_basis_cover_reuse",
        grouping_mode: str = "qwc_basis_cover_reuse",
    ) -> None:
        self._seen_groups: set[str] = set()
        self._nominal_shots = int(max(1, nominal_shots_per_group))
        self._plan_version = str(plan_version)
        self._grouping_mode = str(grouping_mode)

    def clone(self) -> "MeasurementCacheAudit":
        cloned = MeasurementCacheAudit(
            nominal_shots_per_group=int(self._nominal_shots),
            plan_version=str(self._plan_version),
            grouping_mode=str(self._grouping_mode),
        )
        cloned._seen_groups = set(self._seen_groups)
        return cloned

    def snapshot(self) -> dict[str, Any]:
        return {
            "seen_groups": sorted(str(x) for x in self._seen_groups),
            "nominal_shots_per_group": int(self._nominal_shots),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "MeasurementCacheAudit":
        cloned = cls(
            nominal_shots_per_group=int(snapshot.get("nominal_shots_per_group", 1)),
            plan_version=str(snapshot.get("plan_version", "phase1_qwc_basis_cover_reuse")),
            grouping_mode=str(snapshot.get("grouping_mode", "qwc_basis_cover_reuse")),
        )
        cloned._seen_groups = {
            str(x)
            for x in snapshot.get("seen_groups", [])
            if str(x) != ""
        }
        return cloned

    def plan_for(self, group_keys: Iterable[str]) -> MeasurementPlan:
        unique_keys = _compress_measurement_group_keys([str(k) for k in group_keys if str(k) != ""])
        return MeasurementPlan(
            plan_version=str(self._plan_version),
            group_keys=list(unique_keys),
            nominal_shots_per_group=int(self._nominal_shots),
            grouping_mode=str(self._grouping_mode),
        )

    def estimate(self, group_keys: Iterable[str]) -> MeasurementCacheStats:
        plan = self.plan_for(group_keys)
        unique_keys = list(plan.group_keys)

        groups_total = int(len(unique_keys))
        groups_reused = 0
        seen_keys = list(self._seen_groups)
        for key in unique_keys:
            if any(_measurement_basis_key_covers(str(key), str(seen)) for seen in seen_keys):
                groups_reused += 1
        groups_new = int(groups_total - groups_reused)
        shots_reused = float(groups_reused * self._nominal_shots)
        shots_new = float(groups_new * self._nominal_shots)
        reuse_count_cost = float(groups_new)
        return MeasurementCacheStats(
            groups_total=groups_total,
            groups_reused=int(groups_reused),
            groups_new=int(groups_new),
            shots_reused=shots_reused,
            shots_new=shots_new,
            reuse_count_cost=reuse_count_cost,
        )

    def commit(self, group_keys: Iterable[str]) -> None:
        for key in _compress_measurement_group_keys(group_keys):
            key_s = str(key)
            if key_s == "":
                continue
            if any(_measurement_basis_key_covers(key_s, seen) for seen in self._seen_groups):
                continue
            covered = {seen for seen in self._seen_groups if _measurement_basis_key_covers(seen, key_s)}
            if covered:
                self._seen_groups -= covered
            self._seen_groups.add(key_s)

    def summary(self) -> dict[str, float]:
        return {
            "groups_known": float(len(self._seen_groups)),
            "nominal_shots_per_group": float(self._nominal_shots),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
        }


def _replace_feature(feat: CandidateFeatures, **updates: Any) -> CandidateFeatures:
    return CandidateFeatures(**{**feat.__dict__, **updates})


def _pauli_weight_exyz(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))


def _measurement_basis_key_covers(required_key: str, seen_key: str) -> bool:
    req = str(required_key)
    seen = str(seen_key)
    if len(req) != len(seen):
        return False
    return all((r == "e") or (r == s) for r, s in zip(req, seen))


def _measurement_basis_key_merge(lhs_key: str, rhs_key: str) -> str | None:
    lhs = str(lhs_key)
    rhs = str(rhs_key)
    if len(lhs) != len(rhs):
        return None
    merged: list[str] = []
    for lhs_ch, rhs_ch in zip(lhs, rhs):
        if lhs_ch == "e":
            merged.append(rhs_ch)
            continue
        if rhs_ch in {"e", lhs_ch}:
            merged.append(lhs_ch)
            continue
        return None
    return "".join(merged)


def _compress_measurement_group_keys(group_keys: Iterable[str]) -> list[str]:
    ordered = sorted(
        {str(key) for key in group_keys if str(key) != ""},
        key=lambda key: (-_pauli_weight_exyz(str(key)), str(key)),
    )
    kept: list[str] = []
    for key in ordered:
        if any(_measurement_basis_key_covers(str(key), existing) for existing in kept):
            continue
        kept = [existing for existing in kept if not _measurement_basis_key_covers(existing, str(key))]
        kept.append(str(key))
    return kept


def _measurement_group_keys_from_labels(labels: Sequence[str]) -> list[str]:
    active_labels = sorted(
        {str(lbl) for lbl in labels if _pauli_weight_exyz(str(lbl)) > 0},
        key=lambda lbl: (-_pauli_weight_exyz(lbl), lbl),
    )
    groups: list[str] = []
    for label in active_labels:
        best_idx: int | None = None
        best_key: str | None = None
        best_delta: tuple[int, int] | None = None
        for idx, group_key in enumerate(groups):
            merged = _measurement_basis_key_merge(str(group_key), str(label))
            if merged is None:
                continue
            delta = (_pauli_weight_exyz(merged) - _pauli_weight_exyz(str(group_key)), idx)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = int(idx)
                best_key = str(merged)
        if best_idx is None or best_key is None:
            groups.append(str(label))
        else:
            groups[best_idx] = str(best_key)
    return _compress_measurement_group_keys(groups)


def measurement_group_keys_for_term(term: Any) -> list[str]:
    return _measurement_group_keys_from_labels(_pauli_labels_from_term(term))


def _measurement_group_overlap_score(keys_a: Sequence[str], keys_b: Sequence[str]) -> float:
    groups_a = _compress_measurement_group_keys(keys_a)
    groups_b = _compress_measurement_group_keys(keys_b)
    if not groups_a or not groups_b:
        return 1.0

    def _directional(required_groups: Sequence[str], seen_groups: Sequence[str]) -> float:
        if not required_groups:
            return 1.0
        covered = 0
        for req in required_groups:
            if any(_measurement_basis_key_covers(str(req), str(seen)) for seen in seen_groups):
                covered += 1
        return float(covered / len(required_groups))

    return float(
        0.5 * (
            _directional(groups_a, groups_b)
            + _directional(groups_b, groups_a)
        )
    )


def _effective_gate_proxy_total(
    proxy_cost: Mapping[str, Any],
    cfg: SimpleScoreConfig | FullScoreConfig,
) -> float:
    cx_proxy = float(proxy_cost.get("cx_proxy_total", 0.0))
    sq_proxy = float(proxy_cost.get("sq_proxy_total", 0.0))
    rotation_steps = float(proxy_cost.get("new_rotation_steps", 0.0))
    if cx_proxy > 0.0 or sq_proxy > 0.0:
        return float(cfg.compile_cx_proxy_weight) * cx_proxy + float(cfg.compile_sq_proxy_weight) * sq_proxy
    return float(cfg.compile_rotation_step_weight) * rotation_steps


def _effective_compile_proxy_total(
    proxy_cost: Mapping[str, Any],
    cfg: SimpleScoreConfig | FullScoreConfig,
) -> float:
    gate_proxy = _effective_gate_proxy_total(proxy_cost, cfg)
    position_shift = float(proxy_cost.get("position_shift_span", 0.0))
    refit_active = float(proxy_cost.get("refit_active_count", 0.0))
    return float(
        gate_proxy
        + float(cfg.compile_position_shift_weight) * position_shift
        + float(cfg.compile_refit_active_weight) * refit_active
    )


def _effective_depth_cost(
    proxy_cost: Mapping[str, Any],
    cfg: SimpleScoreConfig | FullScoreConfig,
) -> float:
    gate_proxy = _effective_gate_proxy_total(proxy_cost, cfg)
    position_shift = float(proxy_cost.get("position_shift_span", 0.0))
    return float(
        gate_proxy + float(cfg.compile_position_shift_weight) * position_shift
    )


def simple_v1_score(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> float:
    if not bool(feat.stage_gate_open):
        return float("-inf")
    if not bool(feat.leakage_gate_open):
        return float("-inf")
    if not bool(feat.compile_gate_open):
        return float("-inf")

    compile_proxy = float(feat.compile_cost_total)
    groups_new = float(feat.new_group_cost)
    shots_new = float(feat.new_shot_cost)
    reuse_count_cost = float(feat.reuse_count_cost)
    leakage_penalty = float(feat.leakage_penalty)
    burden_total = (
        float(cfg.lambda_compile) * compile_proxy
        + float(cfg.lambda_measure) * (groups_new + shots_new + reuse_count_cost)
        + float(cfg.lambda_leak) * leakage_penalty
    )
    return float(float(feat.g_abs) / float(1.0 + max(0.0, burden_total)))


def normalize(value: float, ref: float) -> float:
    denom = float(ref)
    if not math.isfinite(denom) or denom <= 0.0:
        return float(max(0.0, value))
    return float(max(0.0, value) / denom)


def trust_region_drop(g_lcb: float, h_eff: float, F: float, rho: float) -> float:
    if float(g_lcb) <= 0.0 or float(F) <= 0.0:
        return 0.0
    h_eff_pos = float(max(0.0, h_eff))
    alpha_max = float(rho) / float(math.sqrt(float(F)))
    if h_eff_pos > 0.0:
        alpha_newton = float(g_lcb) / h_eff_pos
        if alpha_newton <= alpha_max:
            return float(0.5 * float(g_lcb) * float(g_lcb) / h_eff_pos)
    alpha = float(alpha_max)
    return float(float(g_lcb) * alpha - 0.5 * h_eff_pos * alpha * alpha)


def remaining_evaluations_proxy(
    *,
    current_depth: int | None,
    max_depth: int | None,
    mode: str,
) -> float:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return 0.0
    depth_now = 0 if current_depth is None else int(max(0, current_depth))
    depth_cap = depth_now if max_depth is None else int(max(depth_now, max_depth))
    if mode_key == "remaining_depth":
        return float(max(1, depth_cap - depth_now + 1))
    raise ValueError("remaining_evaluations_proxy_mode must be 'none' or 'remaining_depth'")


def family_repeat_cost_from_history(
    *,
    history_rows: Sequence[Mapping[str, Any]],
    candidate_family: str,
) -> float:
    fam = str(candidate_family).strip()
    if fam == "":
        return 0.0
    tail = [row for row in history_rows if isinstance(row, Mapping) and row.get("candidate_family") is not None]
    if not tail:
        return 0.0
    if str(tail[-1].get("candidate_family", "")).strip() != fam:
        return 0.0
    streak = 0
    for row in reversed(tail):
        if str(row.get("candidate_family", "")).strip() != fam:
            break
        streak += 1
    return float(streak)


def lifetime_weight_components(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, float]:
    if str(cfg.lifetime_cost_mode).strip().lower() == "off":
        return {
            "remaining_evaluations_proxy": float(feat.remaining_evaluations_proxy),
            "compiled": 0.0,
            "measurement": 0.0,
            "optimizer_dim": 0.0,
            "total": 0.0,
        }
    rem = float(max(0.0, feat.remaining_evaluations_proxy))
    compiled = rem * normalize(float(feat.depth_cost), float(cfg.depth_ref))
    measurement = rem * (
        normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + normalize(float(feat.reuse_count_cost), float(cfg.reuse_ref))
    )
    optimizer_dim = rem * normalize(float(feat.opt_dim_cost), float(cfg.optdim_ref))
    total = compiled + measurement + optimizer_dim
    return {
        "remaining_evaluations_proxy": float(rem),
        "compiled": float(compiled),
        "measurement": float(measurement),
        "optimizer_dim": float(optimizer_dim),
        "total": float(total),
    }


def _cheap_burden_total(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> float:
    K = (
        1.0
        + float(cfg.wD) * normalize(float(feat.depth_cost), float(cfg.depth_ref))
        + float(cfg.wG) * normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + float(cfg.wC) * normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + float(cfg.wP) * normalize(float(feat.opt_dim_cost), float(cfg.optdim_ref))
        + float(cfg.wc) * normalize(float(feat.reuse_count_cost), float(cfg.reuse_ref))
    )
    lifetime_components = lifetime_weight_components(feat, cfg)
    K = float(K + float(cfg.lifetime_weight) * float(lifetime_components.get("total", 0.0)))
    return float(K)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _phase_confidence_factor(
    g_abs: float,
    sigma_hat: float,
    *,
    z_alpha: float,
    eps: float = 1e-12,
) -> float:
    denom = float(max(abs(float(g_abs)), float(eps)))
    return _clip01(1.0 - float(z_alpha) * float(max(0.0, sigma_hat)) / denom)


def phase2_raw_geometry_score(
    feat: CandidateFeatures,
    *,
    F_raw: float,
    h_raw: float,
    q_window: Sequence[float],
    Q_window: np.ndarray,
    cfg: FullScoreConfig,
) -> dict[str, float]:
    F_safe = float(max(float(F_raw), float(cfg.metric_floor)))
    q_vec = np.asarray(q_window, dtype=float).reshape(-1)
    Q_mat = np.asarray(Q_window, dtype=float)
    overlap_max = 0.0
    if q_vec.size > 0 and Q_mat.size > 0:
        diag = np.diag(Q_mat)
        denom_base = math.sqrt(F_safe)
        for idx, q_val in enumerate(q_vec.tolist()):
            diag_val = float(diag[idx]) if idx < int(diag.size) else 0.0
            denom = denom_base * math.sqrt(max(float(diag_val), float(cfg.metric_floor)))
            if denom <= 0.0:
                continue
            overlap_max = max(overlap_max, abs(float(q_val)) / denom)
    raw_novelty = _clip01(1.0 - overlap_max)
    confidence = _phase_confidence_factor(
        float(feat.g_abs),
        float(feat.sigma_hat),
        z_alpha=float(cfg.z_alpha),
    )
    burden_total = float(_cheap_burden_total(feat, cfg))
    trust_gain = float(trust_region_drop(float(feat.g_lcb), float(max(0.0, h_raw)), F_safe, float(cfg.rho)))
    score = float(trust_gain * confidence * raw_novelty / max(burden_total, float(cfg.cheap_score_eps)))
    return {
        "confidence_factor": float(confidence),
        "phase2_raw_overlap_max": float(overlap_max),
        "phase2_raw_novelty": float(raw_novelty),
        "phase2_raw_trust_gain": float(trust_gain),
        "phase2_burden_total": float(burden_total),
        "phase2_raw_score": float(score),
    }


def phase_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    threshold: float,
    cap: int,
    frontier_ratio: float,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
    score_eps: float = 1e-12,
) -> list[dict[str, Any]]:
    def _record_score(rec: Mapping[str, Any], key: str | None, default: float = float("-inf")) -> float:
        if key is None:
            return 0.0
        raw = rec.get(key, default)
        if raw is None:
            return float(default)
        return float(raw)

    ranked = sorted(
        [dict(rec) for rec in records if float(_record_score(rec, score_key)) >= float(threshold)],
        key=lambda rec: (
            -_record_score(rec, score_key),
            -_record_score(rec, tie_break_score_key),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return []
    cap_eff = int(max(1, min(int(cap), len(ranked))))
    shortlist_size = int(cap_eff)
    frontier_cut = float(max(0.0, min(1.0, frontier_ratio)))
    # Treat 1.0 as "no frontier cut" so callers can make the frontier nonbinding.
    frontier_enabled = bool(0.0 < frontier_cut < 1.0)
    if cap_eff > 1 and frontier_enabled:
        for idx in range(cap_eff - 1):
            s_cur = float(_record_score(ranked[idx], score_key))
            s_next = float(_record_score(ranked[idx + 1], score_key))
            ratio = float((s_next + float(score_eps)) / (s_cur + float(score_eps)))
            if ratio <= frontier_cut:
                shortlist_size = int(idx + 1)
                break
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(ranked[:shortlist_size], start=1):
        updated = dict(rec)
        feat = updated.get("feature")
        if isinstance(feat, CandidateFeatures):
            replacement_kwargs: dict[str, Any] = {
                "shortlist_rank": int(idx),
                "shortlist_size": int(shortlist_size),
            }
            if shortlist_flag is not None and hasattr(feat, str(shortlist_flag)):
                replacement_kwargs[str(shortlist_flag)] = True
            updated["feature"] = _replace_feature(feat, **replacement_kwargs)
        out.append(updated)
    return out


def phase3_cheap_ratio_v1(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, float | str | None]:
    metric_source = float(feat.cheap_metric_proxy)
    if metric_source <= 0.0:
        metric_source = float(feat.metric_proxy)
    cheap_metric_proxy = float(max(0.0, metric_source))
    cheap_burden_total = float(_cheap_burden_total(feat, cfg))
    base_payload = {
        "cheap_score_version": "phase3_cheap_ratio_v1",
        "cheap_metric_proxy": float(cheap_metric_proxy),
        "cheap_benefit_proxy": 0.0,
        "cheap_burden_total": float(cheap_burden_total),
    }
    if not bool(feat.stage_gate_open):
        return {**base_payload, "cheap_score": float("-inf")}
    if not bool(feat.leakage_gate_open):
        return {**base_payload, "cheap_score": float("-inf")}
    if not bool(feat.compile_gate_open):
        return {**base_payload, "cheap_score": float("-inf")}

    g_lcb = float(max(0.0, feat.g_lcb))
    if g_lcb <= 0.0 or cheap_metric_proxy <= 0.0:
        return {**base_payload, "cheap_score": 0.0}

    lambda_F_eff = float(max(float(cfg.lambda_F), float(cfg.cheap_score_eps)))
    cheap_benefit_proxy = float(
        float(g_lcb) * float(g_lcb) / (2.0 * float(lambda_F_eff) * float(cheap_metric_proxy))
    )
    cheap_score = float(
        float(cheap_benefit_proxy)
        / float(float(cheap_burden_total) + float(cfg.cheap_score_eps))
    )
    return {
        **base_payload,
        "cheap_score": float(cheap_score),
        "cheap_benefit_proxy": float(cheap_benefit_proxy),
    }


def full_v2_score(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> tuple[float, str]:
    if (not bool(feat.stage_gate_open)) or (not bool(feat.leakage_gate_open)):
        return float("-inf"), "blocked_stage_or_leakage_gate"
    if not bool(feat.compile_gate_open):
        return float("-inf"), "compile_gate_closed"
    if float(feat.leakage_penalty) > float(cfg.leakage_cap):
        return float("-inf"), "leakage_cap"

    g_lcb = max(float(feat.g_abs) - float(cfg.z_alpha) * float(max(0.0, feat.sigma_hat)), 0.0)
    if g_lcb <= 0.0:
        return 0.0, "nonpositive_gradient"

    F_red_raw = feat.F_red if feat.F_red is not None else feat.F_raw
    if F_red_raw is None:
        if float(feat.F_metric) <= 0.0:
            return 0.0, "nonpositive_metric"
        F_red = float(max(float(feat.F_metric), float(cfg.metric_floor)))
        h_eff = float(max(0.0, feat.h_hat if feat.h_hat is not None else float(cfg.lambda_F) * float(feat.F_metric)))
        novelty = 1.0 if feat.novelty is None else min(max(float(feat.novelty), 0.0), 1.0)
        fallback_mode = "legacy_metric_path"
    else:
        h_eff = float(
            feat.h_eff
            if feat.h_eff is not None
            else (feat.h_hat if feat.h_hat is not None else 0.0)
        )
        F_red = float(max(float(F_red_raw), float(cfg.metric_floor)))
        novelty = 1.0 if feat.novelty is None else float(min(1.0, max(0.0, feat.novelty)))
        if str(feat.curvature_mode).startswith("append_exact_metric_collapse") or novelty <= 0.0:
            return 0.0, "reduced_metric_collapse"
        fallback_mode = (
            "append_exact_reduced_path_ridge_grown"
            if feat.ridge_used is not None and float(feat.ridge_used) > float(max(cfg.lambda_H, 0.0))
            else (
                "append_exact_empty_window"
                if len(feat.refit_window_indices) == 0
                else "append_exact_reduced_path"
            )
        )

    delta_e = trust_region_drop(g_lcb, float(max(0.0, h_eff)), F_red, float(cfg.rho))
    if delta_e <= 0.0:
        return 0.0, fallback_mode

    K = (
        1.0
        + float(cfg.wD) * normalize(float(feat.depth_cost), float(cfg.depth_ref))
        + float(cfg.wG) * normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + float(cfg.wC) * normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + float(cfg.wP) * normalize(float(feat.opt_dim_cost), float(cfg.optdim_ref))
        + float(cfg.wc) * normalize(float(feat.reuse_count_cost), float(cfg.reuse_ref))
    )
    lifetime_components = lifetime_weight_components(feat, cfg)
    K = float(K + float(cfg.lifetime_weight) * float(lifetime_components.get("total", 0.0)))
    score = (
        math.exp(-float(cfg.eta_L) * float(feat.leakage_penalty))
        * (float(novelty) ** float(cfg.gamma_N))
        * float(delta_e)
        / float(K)
    )
    score += float(cfg.motif_bonus_weight) * float(max(0.0, feat.motif_bonus))
    return float(score), str(fallback_mode)


def shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    score_key: str = "simple_score",
    tie_break_score_key: str | None = "simple_score",
) -> list[dict[str, Any]]:
    def _record_score(rec: Mapping[str, Any], key: str | None, default: float = float("-inf")) -> float:
        if key is None:
            return 0.0
        raw = rec.get(key, default)
        if raw is None:
            return float(default)
        return float(raw)

    ranked = sorted(
        [dict(rec) for rec in records],
        key=lambda rec: (
            -_record_score(rec, score_key),
            -_record_score(rec, tie_break_score_key),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return []
    total = int(len(ranked))
    target = int(max(1, min(total, cfg.shortlist_size, math.ceil(float(cfg.shortlist_fraction) * total))))
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(ranked[:target], start=1):
        updated = dict(rec)
        feat = updated.get("feature", None)
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                shortlist_rank=int(idx),
                shortlist_size=int(target),
            )
        out.append(updated)
    return out


def _compiled_for_label(
    *,
    label: str,
    polynomial: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None,
    pauli_action_cache: dict[str, Any] | None,
) -> CompiledPolynomialAction:
    cache = compiled_cache if compiled_cache is not None else {}
    key = str(label)
    compiled = cache.get(key)
    if compiled is None:
        compiled = compile_polynomial_action(
            polynomial,
            tol=1e-12,
            pauli_action_cache=pauli_action_cache,
        )
        cache[key] = compiled
    return compiled


def _tangent_data(
    *,
    psi_state: np.ndarray,
    label: str,
    polynomial: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None,
    pauli_action_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, float]:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    compiled = _compiled_for_label(
        label=str(label),
        polynomial=polynomial,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    apsi = apply_compiled_polynomial(psi, compiled)
    mean = complex(np.vdot(psi, apsi))
    centered = np.asarray(apsi - mean * psi, dtype=complex)
    F = float(max(0.0, np.real(np.vdot(centered, centered))))
    return centered, F


def raw_f_metric_from_state(
    *,
    psi_state: np.ndarray,
    candidate_label: str,
    candidate_term: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> float:
    """Built-in math expression:
    F = ||(A - <A>) psi||^2
    """
    _tangent, F_metric = _tangent_data(
        psi_state=psi_state,
        label=str(candidate_label),
        polynomial=candidate_term.polynomial,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    return float(F_metric)


def _tangent_overlap_matrix(tangents: Sequence[np.ndarray]) -> np.ndarray:
    n = int(len(tangents))
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.real(np.vdot(tangents[i], tangents[j])))
            out[i, j] = val
            out[j, i] = val
    return out


def _executor_for_terms(
    terms: Sequence[Any],
    *,
    pauli_action_cache: dict[str, Any] | None,
) -> CompiledAnsatzExecutor:
    return CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
    )


def _rotation_triplet(vec: np.ndarray, step: Any, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec_arr = np.asarray(vec, dtype=complex).reshape(-1)
    coeff = float(step.coeff_real)
    pvec = apply_compiled_pauli(vec_arr, step.action)
    phi = float(theta) * coeff
    c = math.cos(phi)
    s = math.sin(phi)
    u_vec = c * vec_arr - 1j * s * pvec
    d_vec = -coeff * s * vec_arr - 1j * coeff * c * pvec
    s_vec = -(coeff * coeff) * u_vec
    return np.asarray(u_vec, dtype=complex), np.asarray(d_vec, dtype=complex), np.asarray(s_vec, dtype=complex)


def _horizontal_tangent(psi_state: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    dpsi_vec = np.asarray(dpsi, dtype=complex).reshape(-1)
    overlap = complex(np.vdot(psi, dpsi_vec))
    return np.asarray(dpsi_vec - overlap * psi, dtype=complex)


def _energy_hessian_entry(
    *,
    dpsi_left: np.ndarray,
    dpsi_right: np.ndarray,
    d2psi: np.ndarray,
    hpsi_state: np.ndarray,
    hdpsi_right: np.ndarray,
) -> float:
    return float(
        2.0
        * np.real(
            np.vdot(np.asarray(d2psi, dtype=complex), np.asarray(hpsi_state, dtype=complex))
            + np.vdot(np.asarray(dpsi_left, dtype=complex), np.asarray(hdpsi_right, dtype=complex))
        )
    )


def _propagate_executor_derivatives(
    *,
    executor: CompiledAnsatzExecutor,
    theta: np.ndarray,
    psi_ref: np.ndarray,
    active_indices: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]]]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    active = [int(i) for i in active_indices]
    psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
    n_active = int(len(active))
    dpsi = [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
    d2psi = [[np.zeros_like(psi, dtype=complex) for _ in range(n_active)] for __ in range(n_active)]
    if n_active == 0:
        return executor.prepare_state(theta_vec, psi), dpsi, d2psi

    active_map = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(active)}
    plans = list(getattr(executor, "_plans", []))
    if len(plans) != int(theta_vec.size):
        raise ValueError(f"theta length mismatch: got {theta_vec.size}, expected {len(plans)}.")

    for global_idx, plan in enumerate(plans):
        theta_k = float(theta_vec[global_idx])
        local = active_map.get(int(global_idx), None)
        for step in getattr(plan, "steps", ()):
            old_psi = psi
            old_dpsi = dpsi
            old_d2psi = d2psi

            psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, theta_k)
            psi = psi_u

            next_dpsi: list[np.ndarray] = []
            d_old: list[np.ndarray] = []
            for idx in range(n_active):
                vec_u, vec_d, _vec_s = _rotation_triplet(old_dpsi[idx], step, theta_k)
                next_dpsi.append(vec_u)
                d_old.append(vec_d)
            if local is not None:
                next_dpsi[int(local)] = np.asarray(next_dpsi[int(local)] + psi_d, dtype=complex)

            next_d2psi: list[list[np.ndarray]] = [
                [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
                for __ in range(n_active)
            ]
            for row in range(n_active):
                for col in range(n_active):
                    vec_u, _vec_d, _vec_s = _rotation_triplet(old_d2psi[row][col], step, theta_k)
                    updated = vec_u
                    if local is not None:
                        if row == int(local):
                            updated = np.asarray(updated + d_old[col], dtype=complex)
                        if col == int(local):
                            updated = np.asarray(updated + d_old[row], dtype=complex)
                        if row == int(local) and col == int(local):
                            updated = np.asarray(updated + psi_s, dtype=complex)
                    next_d2psi[row][col] = np.asarray(updated, dtype=complex)

            dpsi = next_dpsi
            d2psi = next_d2psi

    return np.asarray(psi, dtype=complex), dpsi, d2psi


def _propagate_append_candidate(
    *,
    candidate_term: Any,
    psi_state: np.ndarray,
    window_dpsi: Sequence[np.ndarray],
    pauli_action_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    cand_exec = _executor_for_terms([candidate_term], pauli_action_cache=pauli_action_cache)
    plan = list(getattr(cand_exec, "_plans", []))
    if not plan:
        zero = np.zeros_like(np.asarray(psi_state, dtype=complex).reshape(-1), dtype=complex)
        return zero, zero, [np.zeros_like(zero) for _ in window_dpsi]
    steps = list(getattr(plan[0], "steps", ()))
    psi = np.asarray(psi_state, dtype=complex).reshape(-1).copy()
    cand_dpsi = np.zeros_like(psi, dtype=complex)
    cand_d2psi = np.zeros_like(psi, dtype=complex)
    win_dpsi = [np.asarray(vec, dtype=complex).reshape(-1).copy() for vec in window_dpsi]
    cand_win_d2 = [np.zeros_like(psi) for _ in window_dpsi]

    for step in steps:
        old_psi = psi
        old_cand_dpsi = cand_dpsi
        old_cand_d2psi = cand_d2psi
        old_win_dpsi = win_dpsi
        old_cand_win_d2 = cand_win_d2

        psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, 0.0)
        cand_u, cand_d, _cand_s = _rotation_triplet(old_cand_dpsi, step, 0.0)
        cand2_u, _cand2_d, _cand2_s = _rotation_triplet(old_cand_d2psi, step, 0.0)

        psi = psi_u
        cand_dpsi = np.asarray(cand_u + psi_d, dtype=complex)
        cand_d2psi = np.asarray(cand2_u + cand_d + cand_d + psi_s, dtype=complex)

        next_win_dpsi: list[np.ndarray] = []
        next_cand_win_d2: list[np.ndarray] = []
        for idx, win_vec in enumerate(old_win_dpsi):
            win_u, win_d, _win_s = _rotation_triplet(win_vec, step, 0.0)
            cross_u, _cross_d, _cross_s = _rotation_triplet(old_cand_win_d2[idx], step, 0.0)
            next_win_dpsi.append(np.asarray(win_u, dtype=complex))
            next_cand_win_d2.append(np.asarray(cross_u + win_d, dtype=complex))
        win_dpsi = next_win_dpsi
        cand_win_d2 = next_cand_win_d2

    return cand_dpsi, cand_d2psi, cand_win_d2


def _regularized_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    base_ridge: float,
    growth_factor: float,
    max_steps: int,
    require_pd: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    mat = np.asarray(matrix, dtype=float)
    vec = np.asarray(rhs, dtype=float).reshape(-1)
    n = int(mat.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float), float(max(base_ridge, 0.0)), np.zeros((0, 0), dtype=float)
    eye = np.eye(n, dtype=float)
    ridge = float(max(base_ridge, 0.0))
    if ridge == 0.0:
        ridge = 1e-12
    mat_sym = 0.5 * (mat + mat.T)
    for _ in range(int(max(1, max_steps))):
        trial = mat_sym + ridge * eye
        try:
            if require_pd:
                np.linalg.cholesky(trial)
            sol = np.linalg.solve(trial, vec)
            return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)
        except Exception:
            ridge *= float(max(growth_factor, 2.0))
    trial = mat_sym + ridge * eye
    if require_pd:
        np.linalg.cholesky(trial)
    sol = np.linalg.solve(trial, vec)
    return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)


class Phase2NoveltyOracle:
    """Exact ordered-state tangent context for append-only reduced-path scoring."""

    def prepare_scaffold_context(
        self,
        *,
        selected_ops: Sequence[Any],
        theta: np.ndarray,
        psi_ref: np.ndarray,
        psi_state: np.ndarray,
        h_compiled: CompiledPolynomialAction,
        hpsi_state: np.ndarray,
        refit_window_indices: Sequence[int],
        pauli_action_cache: dict[str, Any] | None = None,
    ) -> _ScaffoldDerivativeContext:
        inherited_window = [int(i) for i in refit_window_indices]
        psi_current = np.asarray(psi_state, dtype=complex).reshape(-1)
        hpsi_current = np.asarray(hpsi_state, dtype=complex).reshape(-1)
        if not inherited_window:
            return _ScaffoldDerivativeContext(
                psi_state=psi_current,
                hpsi_state=hpsi_current,
                refit_window_indices=tuple(),
                dpsi_window=tuple(),
                tangents_window=tuple(),
                Q_window=np.zeros((0, 0), dtype=float),
                H_window_hessian=np.zeros((0, 0), dtype=float),
            )

        executor = _executor_for_terms(selected_ops, pauli_action_cache=pauli_action_cache)
        _psi_final, dpsi_window, d2psi_window = _propagate_executor_derivatives(
            executor=executor,
            theta=np.asarray(theta, dtype=float),
            psi_ref=np.asarray(psi_ref, dtype=complex),
            active_indices=inherited_window,
        )
        tangents_window = [
            _horizontal_tangent(psi_current, dpsi_vec)
            for dpsi_vec in dpsi_window
        ]
        q_window = _tangent_overlap_matrix(tangents_window)
        hdpsi_window = [
            apply_compiled_polynomial(np.asarray(dpsi_vec, dtype=complex), h_compiled)
            for dpsi_vec in dpsi_window
        ]
        m = int(len(inherited_window))
        hess = np.zeros((m, m), dtype=float)
        for row in range(m):
            for col in range(m):
                hess[row, col] = _energy_hessian_entry(
                    dpsi_left=dpsi_window[row],
                    dpsi_right=dpsi_window[col],
                    d2psi=d2psi_window[row][col],
                    hpsi_state=hpsi_current,
                    hdpsi_right=hdpsi_window[col],
                )
        hess = 0.5 * (hess + hess.T)
        return _ScaffoldDerivativeContext(
            psi_state=psi_current,
            hpsi_state=hpsi_current,
            refit_window_indices=tuple(inherited_window),
            dpsi_window=tuple(np.asarray(x, dtype=complex) for x in dpsi_window),
            tangents_window=tuple(np.asarray(x, dtype=complex) for x in tangents_window),
            Q_window=np.asarray(q_window, dtype=float),
            H_window_hessian=np.asarray(hess, dtype=float),
        )

    def estimate(
        self,
        *,
        scaffold_context: _ScaffoldDerivativeContext,
        candidate_label: str,
        candidate_term: Any,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
        novelty_eps: float = 1e-6,
    ) -> Mapping[str, Any]:
        del compiled_cache, novelty_eps
        cand_dpsi, cand_d2psi, cand_window_d2 = _propagate_append_candidate(
            candidate_term=candidate_term,
            psi_state=scaffold_context.psi_state,
            window_dpsi=list(scaffold_context.dpsi_window),
            pauli_action_cache=pauli_action_cache,
        )
        cand_tangent = _horizontal_tangent(scaffold_context.psi_state, cand_dpsi)
        q_window = np.asarray(
            [
                float(np.real(np.vdot(tang_j, cand_tangent)))
                for tang_j in scaffold_context.tangents_window
            ],
            dtype=float,
        )
        F_raw = float(max(0.0, np.real(np.vdot(cand_tangent, cand_tangent))))
        return {
            "novelty_mode": "append_exact_tangent_context_v1",
            "candidate_dpsi": np.asarray(cand_dpsi, dtype=complex),
            "candidate_d2psi": np.asarray(cand_d2psi, dtype=complex),
            "candidate_window_d2": [np.asarray(x, dtype=complex) for x in cand_window_d2],
            "candidate_tangent": np.asarray(cand_tangent, dtype=complex),
            "F_raw": float(F_raw),
            "Q_window": np.asarray(scaffold_context.Q_window, dtype=float),
            "q_window": np.asarray(q_window, dtype=float),
        }


class Phase2CurvatureOracle:
    """Exact analytic Hessian blocks for the append-only reduced path."""

    def estimate(
        self,
        *,
        base_feature: CandidateFeatures,
        novelty_info: Mapping[str, Any],
        scaffold_context: _ScaffoldDerivativeContext,
        h_compiled: CompiledPolynomialAction,
        cfg: FullScoreConfig,
        optimizer_memory: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del optimizer_memory
        F_raw = float(max(0.0, novelty_info.get("F_raw", base_feature.F_raw or base_feature.F_metric)))
        q_window = np.asarray(novelty_info.get("q_window", []), dtype=float).reshape(-1)
        Q_window = np.asarray(novelty_info.get("Q_window", scaffold_context.Q_window), dtype=float)
        cand_dpsi = np.asarray(novelty_info.get("candidate_dpsi"), dtype=complex).reshape(-1)
        cand_d2psi = np.asarray(novelty_info.get("candidate_d2psi"), dtype=complex).reshape(-1)
        cand_window_d2 = [
            np.asarray(x, dtype=complex).reshape(-1)
            for x in novelty_info.get("candidate_window_d2", [])
        ]
        hdpsi_candidate = apply_compiled_polynomial(cand_dpsi, h_compiled)
        h_raw = _energy_hessian_entry(
            dpsi_left=cand_dpsi,
            dpsi_right=cand_dpsi,
            d2psi=cand_d2psi,
            hpsi_state=scaffold_context.hpsi_state,
            hdpsi_right=hdpsi_candidate,
        )

        b_mixed = np.zeros(len(scaffold_context.refit_window_indices), dtype=float)
        for idx, dpsi_window in enumerate(scaffold_context.dpsi_window):
            if idx >= len(cand_window_d2):
                break
            b_mixed[idx] = _energy_hessian_entry(
                dpsi_left=dpsi_window,
                dpsi_right=cand_dpsi,
                d2psi=cand_window_d2[idx],
                hpsi_state=scaffold_context.hpsi_state,
                hdpsi_right=hdpsi_candidate,
            )

        H_window = np.asarray(scaffold_context.H_window_hessian, dtype=float)
        if H_window.size == 0:
            h_eff = float(h_raw)
            F_red = float(max(F_raw, float(cfg.metric_floor)))
            novelty = 1.0
            ridge_used = float(max(cfg.lambda_H, 0.0))
            mode = "append_exact_empty_window"
        else:
            minv_b, ridge_used, _M_window = _regularized_solve(
                H_window,
                b_mixed,
                base_ridge=float(max(cfg.lambda_H, 0.0)),
                growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                max_steps=int(max(1, cfg.ridge_max_steps)),
                require_pd=True,
            )
            h_eff = float(h_raw - float(b_mixed.T @ minv_b))
            F_red_exact = float(
                F_raw
                - 2.0 * float(q_window.T @ minv_b)
                + float(minv_b.T @ Q_window @ minv_b)
            )
            F_red = float(max(F_red_exact, float(cfg.metric_floor)))
            q_reduced = np.asarray(q_window - Q_window @ minv_b, dtype=float)
            collapse_floor = max(
                float(cfg.metric_floor),
                float(cfg.reduced_metric_collapse_rel_tol) * float(max(F_raw, float(cfg.metric_floor))),
            )
            metric_collapse = bool(F_red_exact <= collapse_floor)
            if metric_collapse:
                novelty = 0.0
                mode = "append_exact_metric_collapse_v1"
            else:
                qsol, _nov_ridge, _Qreg = _regularized_solve(
                    Q_window,
                    q_reduced,
                    base_ridge=float(max(cfg.novelty_eps, 0.0)),
                    growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                    max_steps=int(max(1, cfg.ridge_max_steps)),
                    require_pd=True,
                )
                novelty_raw = 1.0 - float(q_reduced.T @ qsol) / float(F_red)
                novelty = float(min(1.0, max(0.0, novelty_raw)))
                mode = (
                    "append_exact_window_hessian_ridge_grown_v1"
                    if float(ridge_used) > float(max(cfg.lambda_H, 0.0))
                    else "append_exact_window_hessian_v1"
                )

        return {
            "h_raw": float(h_raw),
            "b_mixed": [float(x) for x in b_mixed.tolist()],
            "H_window_hessian": [[float(x) for x in row] for row in H_window.tolist()],
            "h_eff": float(h_eff),
            "F_red": float(F_red),
            "novelty": float(novelty),
            "ridge_used": float(ridge_used),
            "curvature_mode": str(mode),
        }


def _pauli_labels_from_term(term: Any) -> list[str]:
    labels: list[str] = []
    if term is None or not hasattr(term, "polynomial"):
        return labels
    for poly_term in term.polynomial.return_polynomial():
        labels.append(str(poly_term.pw2strng()))
    return labels


def _support_set(term: Any) -> set[int]:
    support: set[int] = set()
    labels = _pauli_labels_from_term(term)
    for label in labels:
        for idx, ch in enumerate(str(label)):
            if ch != "e":
                support.add(int(idx))
    return support


def _pauli_strings_commute(lhs: str, rhs: str) -> bool:
    anticomm = 0
    for a, b in zip(str(lhs), str(rhs)):
        if a == "e" or b == "e" or a == b:
            continue
        anticomm += 1
    return bool((anticomm % 2) == 0)


def _polynomials_commute(term_a: Any, term_b: Any) -> bool:
    labels_a = _pauli_labels_from_term(term_a)
    labels_b = _pauli_labels_from_term(term_b)
    if not labels_a or not labels_b:
        return True
    for lhs in labels_a:
        for rhs in labels_b:
            if not _pauli_strings_commute(lhs, rhs):
                return False
    return True


def build_full_candidate_features(
    *,
    base_feature: CandidateFeatures,
    candidate_term: Any,
    cfg: FullScoreConfig,
    novelty_oracle: NoveltyOracle,
    curvature_oracle: CurvatureOracle,
    scaffold_context: _ScaffoldDerivativeContext,
    h_compiled: CompiledPolynomialAction,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    optimizer_memory: Mapping[str, Any] | None = None,
    motif_library: Mapping[str, Any] | None = None,
    target_num_sites: int | None = None,
) -> CandidateFeatures:
    novelty_info = novelty_oracle.estimate(
        scaffold_context=scaffold_context,
        candidate_label=str(base_feature.candidate_label),
        candidate_term=candidate_term,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        novelty_eps=float(cfg.novelty_eps),
    )
    curvature_info = curvature_oracle.estimate(
        base_feature=base_feature,
        novelty_info=novelty_info,
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        cfg=cfg,
        optimizer_memory=optimizer_memory,
    )
    raw_geometry = phase2_raw_geometry_score(
        base_feature,
        F_raw=float(max(0.0, novelty_info.get("F_raw", base_feature.F_metric))),
        h_raw=float(curvature_info.get("h_raw", 0.0)),
        q_window=list(novelty_info.get("q_window", [])),
        Q_window=np.asarray(novelty_info.get("Q_window", scaffold_context.Q_window), dtype=float),
        cfg=cfg,
    )
    feat = _replace_feature(
        base_feature,
        novelty=float(curvature_info.get("novelty", 1.0)),
        novelty_mode=str(novelty_info.get("novelty_mode", "append_exact_tangent_context_v1")),
        curvature_mode=str(curvature_info.get("curvature_mode", "append_exact_window_hessian_v1")),
        F_metric=float(max(0.0, novelty_info.get("F_raw", base_feature.F_metric))),
        metric_proxy=float(max(0.0, novelty_info.get("F_raw", base_feature.metric_proxy))),
        F_raw=float(max(0.0, novelty_info.get("F_raw", base_feature.F_raw or base_feature.F_metric))),
        h_eff=float(curvature_info.get("h_eff", 0.0)),
        F_red=float(curvature_info.get("F_red", novelty_info.get("F_raw", 0.0))),
        ridge_used=float(curvature_info.get("ridge_used", max(cfg.lambda_H, 0.0))),
        h_hat=float(curvature_info.get("h_raw", 0.0)),
        b_hat=[float(x) for x in curvature_info.get("b_mixed", [])],
        H_window=[[float(x) for x in row] for row in curvature_info.get("H_window_hessian", [])],
        score_version=str(cfg.score_version),
        confidence_factor=float(raw_geometry.get("confidence_factor", 1.0)),
        phase2_raw_overlap_max=float(raw_geometry.get("phase2_raw_overlap_max", 0.0)),
        phase2_raw_novelty=float(raw_geometry.get("phase2_raw_novelty", 1.0)),
        phase2_raw_trust_gain=float(raw_geometry.get("phase2_raw_trust_gain", 0.0)),
        phase2_raw_score=float(raw_geometry.get("phase2_raw_score", 0.0)),
        phase2_burden_total=float(raw_geometry.get("phase2_burden_total", 1.0)),
        placeholder_hooks={
            **dict(base_feature.placeholder_hooks),
            "novelty_oracle": True,
            "curvature_oracle": True,
            "full_v2_score": True,
        },
    )
    if isinstance(base_feature.generator_metadata, Mapping) and isinstance(motif_library, Mapping):
        motif_bonus, motif_meta = motif_bonus_for_generator(
            generator_metadata=base_feature.generator_metadata,
            motif_library=motif_library,
            target_num_sites=int(max(0, target_num_sites or 0)),
        )
        feat = _replace_feature(
            feat,
            motif_bonus=float(motif_bonus),
            motif_source=(
                str(motif_library.get("source_tag", "payload"))
                if bool(motif_bonus) else str(feat.motif_source)
            ),
            motif_metadata=(dict(motif_meta) if isinstance(motif_meta, Mapping) else feat.motif_metadata),
        )
    feat = _replace_feature(
        feat,
        lifetime_weight_components=dict(lifetime_weight_components(feat, cfg)),
        lifetime_cost_mode=str(cfg.lifetime_cost_mode),
        remaining_evaluations_proxy_mode=str(cfg.remaining_evaluations_proxy_mode),
    )
    score, fallback_mode = full_v2_score(feat, cfg)
    phase_score_components = {
        **dict(feat.phase_score_components),
        "phase2_raw_score": float(feat.phase2_raw_score or 0.0),
        "phase2_raw_trust_gain": float(feat.phase2_raw_trust_gain or 0.0),
        "phase2_raw_novelty": float(feat.phase2_raw_novelty or 0.0),
        "phase3_reduced_score": float(score),
        "phase3_reduced_novelty": float(feat.novelty if feat.novelty is not None else 1.0),
    }
    phase_cost_components = {
        **dict(feat.phase_cost_components),
        "phase2_burden_total": float(feat.phase2_burden_total or 0.0),
        "phase3_burden_total": float(_cheap_burden_total(feat, cfg)),
    }
    return _replace_feature(
        feat,
        full_v2_score=float(score),
        phase3_reduced_novelty=float(feat.novelty if feat.novelty is not None else 1.0),
        phase3_reduced_trust_gain=float(
            trust_region_drop(float(max(0.0, feat.g_lcb)), float(max(0.0, feat.h_eff or 0.0)), float(max(feat.F_red or feat.F_metric, cfg.metric_floor)), float(cfg.rho))
        ),
        phase3_burden_total=float(_cheap_burden_total(feat, cfg)),
        selector_score=float(score),
        selector_burden=float(_cheap_burden_total(feat, cfg)),
        phase_score_components=phase_score_components,
        phase_cost_components=phase_cost_components,
        actual_fallback_mode=str(fallback_mode),
    )


def compatibility_penalty(
    *,
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    cfg: FullScoreConfig,
    psi_state: np.ndarray | None = None,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, float]:
    feat_a = record_a.get("feature")
    feat_b = record_b.get("feature")
    term_a = record_a.get("candidate_term")
    term_b = record_b.get("candidate_term")
    if not isinstance(feat_a, CandidateFeatures) or not isinstance(feat_b, CandidateFeatures):
        return {
            "support_overlap": 0.0,
            "noncommutation": 0.0,
            "cross_curvature": 0.0,
            "schedule": 0.0,
            "measurement_mismatch": 0.0,
            "total": 0.0,
        }

    supp_a = _support_set(term_a)
    supp_b = _support_set(term_b)
    union = len(supp_a | supp_b)
    support_overlap = 0.0 if union == 0 else float(len(supp_a & supp_b) / union)
    noncomm = 0.0 if _polynomials_commute(term_a, term_b) else 1.0

    cross_curv = 0.0
    if psi_state is not None and term_a is not None and term_b is not None:
        try:
            tang_a, F_a = _tangent_data(
                psi_state=np.asarray(psi_state, dtype=complex),
                label=str(feat_a.candidate_label),
                polynomial=term_a.polynomial,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
            )
            tang_b, F_b = _tangent_data(
                psi_state=np.asarray(psi_state, dtype=complex),
                label=str(feat_b.candidate_label),
                polynomial=term_b.polynomial,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
            )
            denom = math.sqrt(max(F_a, 0.0) * max(F_b, 0.0))
            if denom > 0.0:
                cross_curv = float(min(1.0, abs(float(np.real(np.vdot(tang_a, tang_b)))) / denom))
        except Exception:
            cross_curv = float(support_overlap)
    elif feat_a.b_hat is not None and feat_b.b_hat is not None:
        vec_a = np.asarray(feat_a.b_hat, dtype=float)
        vec_b = np.asarray(feat_b.b_hat, dtype=float)
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom > 0.0:
            cross_curv = float(min(1.0, abs(float(vec_a @ vec_b)) / denom))

    win_a = set(int(i) for i in feat_a.refit_window_indices)
    win_b = set(int(i) for i in feat_b.refit_window_indices)
    union_w = len(win_a | win_b)
    schedule = 0.0 if union_w == 0 else float(len(win_a & win_b) / union_w)
    measurement_overlap = _measurement_group_overlap_score(
        measurement_group_keys_for_term(term_a),
        measurement_group_keys_for_term(term_b),
    )
    measurement_mismatch = float(1.0 - measurement_overlap)
    total = (
        float(cfg.compat_overlap_weight) * float(support_overlap)
        + float(cfg.compat_comm_weight) * float(noncomm)
        + float(cfg.compat_curv_weight) * float(cross_curv)
        + float(cfg.compat_sched_weight) * float(schedule)
        + float(cfg.compat_measure_weight) * float(measurement_mismatch)
    )
    return {
        "support_overlap": float(support_overlap),
        "noncommutation": float(noncomm),
        "cross_curvature": float(cross_curv),
        "schedule": float(schedule),
        "measurement_mismatch": float(measurement_mismatch),
        "total": float(total),
    }


class CompatibilityPenaltyOracle:
    def __init__(
        self,
        *,
        cfg: FullScoreConfig,
        psi_state: np.ndarray | None = None,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.psi_state = None if psi_state is None else np.asarray(psi_state, dtype=complex)
        self.compiled_cache = compiled_cache
        self.pauli_action_cache = pauli_action_cache

    def penalty(self, record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> dict[str, float]:
        return compatibility_penalty(
            record_a=record_a,
            record_b=record_b,
            cfg=self.cfg,
            psi_state=self.psi_state,
            compiled_cache=self.compiled_cache,
            pauli_action_cache=self.pauli_action_cache,
        )


def _batch_sort_key(record: Mapping[str, Any], tie_break_score_key: str) -> tuple[float, float, int, int]:
    full_score = record.get("full_v2_score", float("-inf"))
    if full_score is None:
        full_score = float("-inf")
    tie_score = record.get(tie_break_score_key, float("-inf"))
    if tie_score is None:
        tie_score = float("-inf")
    return (
        -float(full_score),
        -float(tie_score),
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
    )


def _solve_joint_trust_region_gain(
    *,
    g_vec: np.ndarray,
    G_mat: np.ndarray,
    H_mat: np.ndarray,
    rho: float,
) -> tuple[float, np.ndarray]:
    g = np.asarray(g_vec, dtype=float).reshape(-1)
    G = 0.5 * (np.asarray(G_mat, dtype=float) + np.asarray(G_mat, dtype=float).T)
    H = 0.5 * (np.asarray(H_mat, dtype=float) + np.asarray(H_mat, dtype=float).T)
    n = int(g.size)
    if n == 0:
        return 0.0, np.zeros(0, dtype=float)
    eye = np.eye(n, dtype=float)
    rho_sq = float(max(0.0, rho)) ** 2

    def _alpha(lam: float) -> np.ndarray:
        trial = H + float(lam) * G + 1e-12 * eye
        sol = np.linalg.solve(trial, g)
        return np.asarray(np.maximum(sol, 0.0), dtype=float)

    def _constraint(alpha: np.ndarray) -> float:
        return float(alpha.T @ G @ alpha)

    alpha0 = _alpha(0.0)
    if _constraint(alpha0) <= rho_sq:
        alpha = alpha0
    else:
        lo = 0.0
        hi = 1.0
        alpha_hi = _alpha(hi)
        while _constraint(alpha_hi) > rho_sq and hi < 1e12:
            lo = hi
            hi *= 2.0
            alpha_hi = _alpha(hi)
        alpha = alpha_hi
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            alpha_mid = _alpha(mid)
            if _constraint(alpha_mid) > rho_sq:
                lo = mid
            else:
                hi = mid
                alpha = alpha_mid
    gain = float(g.T @ alpha - 0.5 * alpha.T @ H @ alpha)
    return float(max(0.0, gain)), np.asarray(alpha, dtype=float)


def _batch_geometry_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared: list[tuple[dict[str, Any], CandidateFeatures, Any]] = []
    common_window = sorted(
        {
            int(idx)
            for rec in records
            for idx in (
                rec.get("feature").refit_window_indices
                if isinstance(rec.get("feature"), CandidateFeatures)
                else []
            )
        }
    )
    scaffold_context = novelty_oracle.prepare_scaffold_context(
        selected_ops=list(selected_ops),
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        h_compiled=h_compiled,
        hpsi_state=apply_compiled_polynomial(np.asarray(psi_state, dtype=complex), h_compiled),
        refit_window_indices=list(common_window),
        pauli_action_cache=pauli_action_cache,
    )
    H_window = np.asarray(scaffold_context.H_window_hessian, dtype=float)
    Q_window = np.asarray(scaffold_context.Q_window, dtype=float)
    for rec in records:
        feat = rec.get("feature")
        candidate_term = rec.get("candidate_term")
        if not isinstance(feat, CandidateFeatures) or candidate_term is None:
            return {"feasible": False, "reason": "invalid_record"}
        novelty_info = novelty_oracle.estimate(
            scaffold_context=scaffold_context,
            candidate_label=str(feat.candidate_label),
            candidate_term=candidate_term,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
            novelty_eps=float(cfg.novelty_eps),
        )
        curvature_info = curvature_oracle.estimate(
            base_feature=feat,
            novelty_info=novelty_info,
            scaffold_context=scaffold_context,
            h_compiled=h_compiled,
            cfg=cfg,
            optimizer_memory=None,
        )
        b_vec = np.asarray(curvature_info.get("b_mixed", []), dtype=float).reshape(-1)
        if H_window.size == 0 or b_vec.size == 0:
            v_vec = np.zeros_like(b_vec, dtype=float)
        else:
            v_vec, _ridge, _trial = _regularized_solve(
                H_window,
                b_vec,
                base_ridge=float(max(cfg.lambda_H, 0.0)),
                growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                max_steps=int(max(1, cfg.ridge_max_steps)),
                require_pd=True,
            )
        prepared.append(
            (
                dict(rec),
                feat,
                {
                    "novelty_info": novelty_info,
                    "curvature_info": curvature_info,
                    "b_vec": np.asarray(b_vec, dtype=float),
                    "v_vec": np.asarray(v_vec, dtype=float),
                    "candidate_tangent": np.asarray(novelty_info.get("candidate_tangent"), dtype=complex).reshape(-1),
                    "q_vec": np.asarray(novelty_info.get("q_window", []), dtype=float).reshape(-1),
                    "h_eff": float(curvature_info.get("h_eff", 0.0)),
                    "g_lcb": float(
                        max(
                            float(feat.g_abs) - float(cfg.z_alpha) * float(max(0.0, feat.sigma_hat)),
                            0.0,
                        )
                    ),
                },
            )
        )
    n = int(len(prepared))
    G = np.zeros((n, n), dtype=float)
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        feat_i = prepared[i][1]
        aux_i = prepared[i][2]
        tau_i = np.asarray(aux_i["candidate_tangent"], dtype=complex)
        q_i = np.asarray(aux_i["q_vec"], dtype=float)
        v_i = np.asarray(aux_i["v_vec"], dtype=float)
        H[i, i] = float(max(0.0, aux_i["h_eff"]))
        for j in range(i, n):
            aux_j = prepared[j][2]
            tau_j = np.asarray(aux_j["candidate_tangent"], dtype=complex)
            q_j = np.asarray(aux_j["q_vec"], dtype=float)
            v_j = np.asarray(aux_j["v_vec"], dtype=float)
            c_ij = float(np.real(np.vdot(tau_i, tau_j)))
            if Q_window.size == 0:
                g_ij = float(c_ij)
            else:
                g_ij = float(c_ij - q_i.T @ v_j - q_j.T @ v_i + v_i.T @ Q_window @ v_j)
            G[i, j] = g_ij
            G[j, i] = g_ij
    trace_G = float(np.trace(G))
    lambda_min = float(np.min(np.linalg.eigvalsh(G))) if n > 0 else 0.0
    rank_floor = float(cfg.batch_rank_rel_tol) * float(trace_G / max(1, n))
    if n > 1 and lambda_min < rank_floor:
        return {
            "feasible": False,
            "reason": "rank_gate",
            "lambda_min": float(lambda_min),
            "rank_floor": float(rank_floor),
            "common_window_indices": [int(x) for x in common_window],
        }
    g_vec = np.asarray([float(item[2]["g_lcb"]) for item in prepared], dtype=float)
    joint_gain, alpha = _solve_joint_trust_region_gain(
        g_vec=g_vec,
        G_mat=G,
        H_mat=H,
        rho=float(cfg.rho),
    )
    contextual_single = [
        float(trust_region_drop(float(g_vec[i]), float(H[i, i]), float(max(G[i, i], cfg.metric_floor)), float(cfg.rho)))
        for i in range(n)
    ]
    single_total = float(sum(contextual_single))
    additivity_defect = float(max(0.0, 1.0 - joint_gain / (single_total + float(cfg.cheap_score_eps))))
    if n > 1 and additivity_defect > float(cfg.batch_additivity_tol):
        return {
            "feasible": False,
            "reason": "additivity_gate",
            "joint_gain": float(joint_gain),
            "contextual_single_total": float(single_total),
            "additivity_defect": float(additivity_defect),
            "common_window_indices": [int(x) for x in common_window],
        }
    mu_tan = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            denom = math.sqrt(max(G[i, i], 0.0) * max(G[j, j], 0.0)) + float(cfg.cheap_score_eps)
            mu_tan = max(mu_tan, abs(float(G[i, j])) / denom)
    return {
        "feasible": True,
        "joint_gain": float(joint_gain),
        "contextual_single_total": float(single_total),
        "additivity_defect": float(additivity_defect),
        "lambda_min": float(lambda_min),
        "rank_floor": float(rank_floor),
        "mu_tan": float(mu_tan),
        "alpha": [float(x) for x in alpha.tolist()],
        "common_window_indices": [int(x) for x in common_window],
        "G": [[float(x) for x in row] for row in G.tolist()],
    }


def reduced_plane_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    tie_break_score_key: str = "simple_score",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked = sorted([dict(rec) for rec in ranked_records], key=lambda rec: _batch_sort_key(rec, tie_break_score_key))
    if not ranked:
        return [], {"selected": False, "reason": "empty_shortlist"}
    top_score = float(ranked[0].get("full_v2_score", float("-inf")))
    shell = [
        dict(rec)
        for rec in ranked
        if float(rec.get("full_v2_score", float("-inf"))) > 0.0
        and float(rec.get("full_v2_score", float("-inf"))) >= float(cfg.batch_near_degenerate_ratio) * float(top_score)
    ]
    if not shell:
        return [dict(ranked[0])], {"selected": False, "reason": "nonpositive_shell"}
    batch = [dict(shell[0])]
    batch_summary = _batch_geometry_summary(
        batch,
        cfg=cfg,
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=novelty_oracle,
        curvature_oracle=curvature_oracle,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    current_gain = float(batch_summary.get("joint_gain", batch[0].get("full_v2_score", 0.0)))
    while len(batch) < int(max(1, cfg.batch_size_cap)) and len(batch) < int(max(1, cfg.batch_target_size)):
        best_candidate: dict[str, Any] | None = None
        best_summary: dict[str, Any] | None = None
        best_marginal = 0.0
        batch_keys = {
            (str(rec.get("candidate_label", rec.get("feature").candidate_label if isinstance(rec.get("feature"), CandidateFeatures) else "")), int(rec.get("position_id", -1)))
            for rec in batch
        }
        for rec in shell[1:]:
            feat = rec.get("feature")
            rec_key = (
                str(rec.get("candidate_label", feat.candidate_label if isinstance(feat, CandidateFeatures) else "")),
                int(rec.get("position_id", -1)),
            )
            if rec_key in batch_keys:
                continue
            trial_batch = [dict(x) for x in batch] + [dict(rec)]
            trial_summary = _batch_geometry_summary(
                trial_batch,
                cfg=cfg,
                selected_ops=selected_ops,
                theta=theta,
                psi_ref=psi_ref,
                psi_state=psi_state,
                h_compiled=h_compiled,
                novelty_oracle=novelty_oracle,
                curvature_oracle=curvature_oracle,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
            )
            if not bool(trial_summary.get("feasible", False)):
                continue
            marginal = float(trial_summary.get("joint_gain", 0.0)) - float(current_gain)
            if marginal > float(best_marginal):
                best_candidate = dict(rec)
                best_summary = dict(trial_summary)
                best_marginal = float(marginal)
        if best_candidate is None or float(best_marginal) <= 0.0:
            break
        batch.append(best_candidate)
        batch_summary = dict(best_summary) if isinstance(best_summary, Mapping) else batch_summary
        current_gain = float(batch_summary.get("joint_gain", current_gain))
    annotated: list[dict[str, Any]] = []
    for rec in batch:
        updated = dict(rec)
        feat = updated.get("feature")
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                compatibility_penalty_total=float(batch_summary.get("additivity_defect", 0.0)),
            )
        updated["compatibility_penalty"] = {
            "total": float(batch_summary.get("additivity_defect", 0.0)),
            "joint_gain": float(batch_summary.get("joint_gain", 0.0)),
            "contextual_single_total": float(batch_summary.get("contextual_single_total", 0.0)),
            "lambda_min": float(batch_summary.get("lambda_min", 0.0)),
            "rank_floor": float(batch_summary.get("rank_floor", 0.0)),
            "mu_tan": float(batch_summary.get("mu_tan", 0.0)),
        }
        annotated.append(updated)
    return annotated, dict(batch_summary)


def greedy_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    compat_oracle: CompatibilityPenaltyOracle,
    cfg: FullScoreConfig,
    tie_break_score_key: str = "simple_score",
) -> tuple[list[dict[str, Any]], float]:
    def _record_score(rec: Mapping[str, Any], key: str, default: float = float("-inf")) -> float:
        raw = rec.get(key, default)
        if raw is None:
            return float(default)
        return float(raw)

    ranked = sorted(
        [dict(rec) for rec in ranked_records],
        key=lambda rec: (
            -_record_score(rec, "full_v2_score"),
            -_record_score(rec, tie_break_score_key),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return [], 0.0

    batch: list[dict[str, Any]] = []
    total_penalty = 0.0
    top_score = float(ranked[0].get("full_v2_score", float("-inf")))
    for rec in ranked:
        if len(batch) >= int(max(1, cfg.batch_size_cap)):
            break
        rec_score = float(rec.get("full_v2_score", float("-inf")))
        if not math.isfinite(rec_score) or rec_score <= 0.0:
            continue
        if batch and rec_score < float(cfg.batch_near_degenerate_ratio) * float(top_score):
            continue
        penalty_total = 0.0
        penalty_breakdown = {
            "support_overlap": 0.0,
            "noncommutation": 0.0,
            "cross_curvature": 0.0,
            "schedule": 0.0,
            "measurement_mismatch": 0.0,
        }
        for existing in batch:
            breakdown = compat_oracle.penalty(rec, existing)
            penalty_total += float(breakdown.get("total", 0.0))
            for key in penalty_breakdown:
                penalty_breakdown[key] += float(breakdown.get(key, 0.0))
        if float(rec_score) - float(penalty_total) <= 0.0 and batch:
            continue
        feat = rec.get("feature")
        updated = dict(rec)
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                compatibility_penalty_total=float(penalty_total),
            )
        updated["compatibility_penalty"] = {
            **penalty_breakdown,
            "total": float(penalty_total),
        }
        batch.append(updated)
        total_penalty += float(penalty_total)
        if len(batch) >= int(max(1, cfg.batch_target_size)):
            break
    return batch if batch else [dict(ranked[0])], float(total_penalty)


def build_candidate_features(
    *,
    stage_name: str,
    candidate_label: str,
    candidate_family: str,
    candidate_pool_index: int,
    position_id: int,
    append_position: int,
    positions_considered: list[int],
    gradient_signed: float,
    metric_proxy: float,
    sigma_hat: float,
    refit_window_indices: list[int],
    compile_cost: CompileCostEstimate,
    measurement_stats: MeasurementCacheStats,
    leakage_penalty: float,
    stage_gate_open: bool,
    leakage_gate_open: bool,
    trough_probe_triggered: bool,
    trough_detected: bool,
    family_repeat_cost: float = 0.0,
    cfg: SimpleScoreConfig,
    cheap_score_cfg: FullScoreConfig | None = None,
    generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    symmetry_mode: str = "none",
    symmetry_mitigation_mode: str = "off",
    motif_metadata: Mapping[str, Any] | None = None,
    motif_bonus: float = 0.0,
    motif_source: str = "none",
    current_depth: int | None = None,
    max_depth: int | None = None,
    lifetime_cost_mode: str = "off",
    remaining_evaluations_proxy_mode: str = "none",
) -> CandidateFeatures:
    """Built-in math expression:
    g_lcb = max(|g| - z_alpha * sigma_hat, 0)
    """
    g_abs = float(abs(float(gradient_signed)))
    g_lcb = max(g_abs - float(cfg.z_alpha) * float(max(0.0, sigma_hat)), 0.0)
    remaining_eval_proxy = remaining_evaluations_proxy(
        current_depth=current_depth,
        max_depth=max_depth,
        mode=str(remaining_evaluations_proxy_mode),
    )
    cost_cfg = cheap_score_cfg if cheap_score_cfg is not None else cfg
    proxy_cost = (
        dict(compile_cost.proxy_baseline)
        if isinstance(compile_cost.proxy_baseline, Mapping)
        else {
            "new_pauli_actions": float(compile_cost.new_pauli_actions),
            "new_rotation_steps": float(compile_cost.new_rotation_steps),
            "position_shift_span": float(compile_cost.position_shift_span),
            "refit_active_count": float(compile_cost.refit_active_count),
            "cx_proxy_total": float(compile_cost.cx_proxy_total),
            "sq_proxy_total": float(compile_cost.sq_proxy_total),
            "gate_proxy_total": float(compile_cost.gate_proxy_total),
            "max_pauli_weight": float(compile_cost.max_pauli_weight),
            "proxy_total": float(compile_cost.proxy_total),
        }
    )
    compile_cost_total = (
        float(compile_cost.penalty_total)
        if compile_cost.penalty_total is not None
        else float(_effective_compile_proxy_total(proxy_cost, cost_cfg))
    )
    depth_cost_value = (
        float(compile_cost.depth_surrogate)
        if compile_cost.depth_surrogate is not None
        else float(_effective_depth_cost(proxy_cost, cost_cfg))
    )
    measurement_groups_cost = float(cost_cfg.measure_groups_weight) * float(measurement_stats.groups_new)
    measurement_shots_cost = float(cost_cfg.measure_shots_weight) * float(measurement_stats.shots_new)
    measurement_reuse_cost = float(cost_cfg.measure_reuse_weight) * float(measurement_stats.reuse_count_cost)
    opt_dim_cost_value = float(cost_cfg.opt_dim_cost_scale) * float(len(refit_window_indices))
    family_repeat_cost_value = float(cost_cfg.family_repeat_cost_scale) * float(family_repeat_cost)
    feat = CandidateFeatures(
        stage_name=str(stage_name),
        candidate_label=str(candidate_label),
        candidate_family=str(candidate_family),
        candidate_pool_index=int(candidate_pool_index),
        position_id=int(position_id),
        append_position=int(append_position),
        positions_considered=[int(x) for x in positions_considered],
        g_signed=float(gradient_signed),
        g_abs=float(g_abs),
        g_lcb=float(g_lcb),
        sigma_hat=float(max(0.0, sigma_hat)),
        F_metric=float(max(0.0, metric_proxy)),
        metric_proxy=float(max(0.0, metric_proxy)),
        novelty=None,
        curvature_mode="lambda_F_metric_proxy_only",
        novelty_mode="none",
        refit_window_indices=[int(i) for i in refit_window_indices],
        compiled_position_cost_proxy={str(k): float(v) for k, v in proxy_cost.items()},
        measurement_cache_stats={
            "groups_total": float(measurement_stats.groups_total),
            "groups_reused": float(measurement_stats.groups_reused),
            "groups_new": float(measurement_stats.groups_new),
            "shots_reused": float(measurement_stats.shots_reused),
            "shots_new": float(measurement_stats.shots_new),
            "reuse_count_cost": float(measurement_stats.reuse_count_cost),
        },
        leakage_penalty=float(max(0.0, leakage_penalty)),
        stage_gate_open=bool(stage_gate_open),
        leakage_gate_open=bool(leakage_gate_open),
        trough_probe_triggered=bool(trough_probe_triggered),
        trough_detected=bool(trough_detected),
        simple_score=None,
        score_version=str(cfg.score_version),
        cheap_score=None,
        cheap_score_version=str(cfg.score_version),
        cheap_metric_proxy=float(max(0.0, metric_proxy)),
        cheap_benefit_proxy=None,
        cheap_burden_total=None,
        depth_cost=float(depth_cost_value),
        new_group_cost=float(measurement_groups_cost),
        new_shot_cost=float(measurement_shots_cost),
        opt_dim_cost=float(opt_dim_cost_value),
        reuse_count_cost=float(measurement_reuse_cost),
        family_repeat_cost=float(family_repeat_cost_value),
        actual_fallback_mode="simple_v1_only",
        generator_id=(
            str(generator_metadata.get("generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("generator_id") is not None
            else None
        ),
        template_id=(
            str(generator_metadata.get("template_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("template_id") is not None
            else None
        ),
        is_macro_generator=bool(generator_metadata.get("is_macro_generator", False)) if isinstance(generator_metadata, Mapping) else False,
        parent_generator_id=(
            str(generator_metadata.get("parent_generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("parent_generator_id") is not None
            else None
        ),
        generator_metadata=(dict(generator_metadata) if isinstance(generator_metadata, Mapping) else None),
        symmetry_spec=(dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None),
        symmetry_mode=str(symmetry_mode),
        symmetry_mitigation_mode=str(symmetry_mitigation_mode),
        motif_metadata=(dict(motif_metadata) if isinstance(motif_metadata, Mapping) else None),
        motif_bonus=float(max(0.0, motif_bonus)),
        motif_source=str(motif_source),
        remaining_evaluations_proxy=float(remaining_eval_proxy),
        remaining_evaluations_proxy_mode=str(remaining_evaluations_proxy_mode),
        lifetime_cost_mode=str(lifetime_cost_mode),
        lifetime_weight_components={
            "remaining_evaluations_proxy": float(remaining_eval_proxy),
        },
        placeholder_hooks={
            "novelty_oracle": False,
            "curvature_oracle": False,
            "full_v2_score": False,
            "qn_spsa_refresh": False,
            "motif_metadata": False,
            "symmetry_metadata": bool(isinstance(symmetry_spec, Mapping)),
            "backend_compile_oracle": bool(str(compile_cost.source_mode) != "proxy"),
        },
        compile_cost_source=str(compile_cost.source_mode),
        compile_cost_total=float(compile_cost_total),
        compile_gate_open=bool(compile_cost.compile_gate_open),
        compile_failure_reason=(
            None if compile_cost.failure_reason is None else str(compile_cost.failure_reason)
        ),
        compiled_position_cost_backend=(
            None
            if str(compile_cost.source_mode) == "proxy"
            else {
                "selected_backend_name": compile_cost.selected_backend_name,
                "selected_resolution_kind": compile_cost.selected_resolution_kind,
                "aggregation_mode": str(compile_cost.aggregation_mode),
                "target_backend_names": [str(x) for x in compile_cost.target_backend_names],
                "successful_target_count": int(compile_cost.successful_target_count),
                "failed_target_count": int(compile_cost.failed_target_count),
                "raw_delta_compiled_count_2q": compile_cost.raw_delta_compiled_count_2q,
                "delta_compiled_count_2q": compile_cost.delta_compiled_count_2q,
                "raw_delta_compiled_depth": compile_cost.raw_delta_compiled_depth,
                "delta_compiled_depth": compile_cost.delta_compiled_depth,
                "raw_delta_compiled_size": compile_cost.raw_delta_compiled_size,
                "delta_compiled_size": compile_cost.delta_compiled_size,
                "delta_compiled_cx_count": compile_cost.delta_compiled_cx_count,
                "delta_compiled_ecr_count": compile_cost.delta_compiled_ecr_count,
                "base_compiled_count_2q": compile_cost.base_compiled_count_2q,
                "base_compiled_depth": compile_cost.base_compiled_depth,
                "base_compiled_size": compile_cost.base_compiled_size,
                "trial_compiled_count_2q": compile_cost.trial_compiled_count_2q,
                "trial_compiled_depth": compile_cost.trial_compiled_depth,
                "trial_compiled_size": compile_cost.trial_compiled_size,
                }
        ),
        phase_score_components={},
        phase_cost_components={},
        controller_snapshot=None,
        window_origin="legacy",
        window_new_indices=[int(i) for i in refit_window_indices],
        window_age_indices=[],
        phase1_shortlisted=False,
        phase2_shortlisted=False,
        phase3_shortlisted=False,
        phase3_duplicate_penalty=0.0,
    )
    score = simple_v1_score(feat, cfg)
    phase_cost_components = {
        "compile_proxy": float(compile_cost_total),
        "compile_cx_proxy_weight": float(cost_cfg.compile_cx_proxy_weight),
        "compile_sq_proxy_weight": float(cost_cfg.compile_sq_proxy_weight),
        "compile_rotation_step_weight": float(cost_cfg.compile_rotation_step_weight),
        "compile_position_shift_weight": float(cost_cfg.compile_position_shift_weight),
        "compile_refit_active_weight": float(cost_cfg.compile_refit_active_weight),
        "measurement_groups_new_raw": float(measurement_stats.groups_new),
        "measurement_shots_new_raw": float(measurement_stats.shots_new),
        "measurement_reuse_cost_raw": float(measurement_stats.reuse_count_cost),
        "measurement_groups_new": float(measurement_groups_cost),
        "measurement_shots_new": float(measurement_shots_cost),
        "measurement_reuse_cost": float(measurement_reuse_cost),
        "opt_dim_cost": float(opt_dim_cost_value),
        "family_repeat_cost": float(family_repeat_cost_value),
        "leakage_penalty": float(max(0.0, leakage_penalty)),
        "burden_total": float(
            float(cfg.lambda_compile) * float(compile_cost_total)
            + float(cfg.lambda_measure)
            * float(
                float(measurement_groups_cost)
                + float(measurement_shots_cost)
                + float(measurement_reuse_cost)
            )
            + float(cfg.lambda_leak) * float(max(0.0, leakage_penalty))
        ),
    }
    phase_score_components = {
        "phase1_gradient_abs": float(g_abs),
        "phase1_score": float(score),
    }
    feat = _replace_feature(
        feat,
        simple_score=float(score),
        cheap_score=float(score),
        cheap_score_version=str(cfg.score_version),
        cheap_metric_proxy=float(max(0.0, metric_proxy)),
        phase_score_components=dict(phase_score_components),
        phase_cost_components=dict(phase_cost_components),
    )
    if cheap_score_cfg is not None:
        feat = _replace_feature(
            feat,
            **phase3_cheap_ratio_v1(feat, cheap_score_cfg),
        )
    return feat
