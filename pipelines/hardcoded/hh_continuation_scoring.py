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
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial,
    compile_polynomial_action,
)


@dataclass(frozen=True)
class SimpleScoreConfig:
    lambda_F: float = 1.0
    lambda_compile: float = 0.05
    lambda_measure: float = 0.02
    lambda_leak: float = 0.0
    z_alpha: float = 0.0
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
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    optdim_ref: float = 1.0
    reuse_ref: float = 1.0
    novelty_eps: float = 1e-6
    shortlist_fraction: float = 0.2
    shortlist_size: int = 12
    batch_target_size: int = 2
    batch_size_cap: int = 3
    batch_near_degenerate_ratio: float = 0.9
    compat_overlap_weight: float = 0.4
    compat_comm_weight: float = 0.2
    compat_curv_weight: float = 0.2
    compat_sched_weight: float = 0.2
    leakage_cap: float = 1e6
    lifetime_cost_mode: str = "off"
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_weight: float = 0.05
    motif_bonus_weight: float = 0.05
    score_version: str = "full_v2"


class Phase1CompileCostOracle:
    """Built-in math expression:
    D_proxy = n_new_pauli + n_rot + shift_span + active_count
    """

    def estimate(
        self,
        *,
        candidate_term_count: int,
        position_id: int,
        append_position: int,
        refit_active_count: int,
    ) -> CompileCostEstimate:
        new_pauli_actions = float(max(1, int(candidate_term_count)))
        new_rotation_steps = float(max(1, int(candidate_term_count)))
        position_shift_span = float(abs(int(append_position) - int(position_id)))
        refit_active = float(max(0, int(refit_active_count)))
        total = float(new_pauli_actions + new_rotation_steps + position_shift_span + refit_active)
        return CompileCostEstimate(
            new_pauli_actions=new_pauli_actions,
            new_rotation_steps=new_rotation_steps,
            position_shift_span=position_shift_span,
            refit_active_count=refit_active,
            proxy_total=total,
        )


class MeasurementCacheAudit:
    """Phase 1 accounting-only grouped reuse tracker."""

    def __init__(
        self,
        nominal_shots_per_group: int = 1,
        *,
        plan_version: str = "phase1_grouped_label_reuse",
        grouping_mode: str = "grouped_label_reuse",
    ) -> None:
        self._seen_groups: set[str] = set()
        self._nominal_shots = int(max(1, nominal_shots_per_group))
        self._plan_version = str(plan_version)
        self._grouping_mode = str(grouping_mode)

    def plan_for(self, group_keys: Iterable[str]) -> MeasurementPlan:
        keys = [str(k) for k in group_keys if str(k) != ""]
        unique_keys: list[str] = []
        for key in keys:
            if key not in unique_keys:
                unique_keys.append(key)
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
        for key in unique_keys:
            if key in self._seen_groups:
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
        for key in group_keys:
            key_s = str(key)
            if key_s != "":
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


def simple_v1_score(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> float:
    if not bool(feat.stage_gate_open):
        return float("-inf")
    if not bool(feat.leakage_gate_open):
        return float("-inf")

    compile_proxy = float(feat.compiled_position_cost_proxy.get("proxy_total", 0.0))
    groups_new = float(feat.measurement_cache_stats.get("groups_new", 0.0))
    shots_new = float(feat.measurement_cache_stats.get("shots_new", 0.0))
    reuse_count_cost = float(feat.measurement_cache_stats.get("reuse_count_cost", 0.0))
    leakage_penalty = float(feat.leakage_penalty)

    score = (
        float(feat.g_abs) + float(cfg.lambda_F) * float(feat.metric_proxy)
        - float(cfg.lambda_compile) * compile_proxy
        - float(cfg.lambda_measure) * (groups_new + shots_new + reuse_count_cost)
        - float(cfg.lambda_leak) * leakage_penalty
    )
    return float(score)


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


def full_v2_score(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> tuple[float, str]:
    if (not bool(feat.stage_gate_open)) or (not bool(feat.leakage_gate_open)):
        return float("-inf"), "blocked_stage_or_leakage_gate"
    if float(feat.leakage_penalty) > float(cfg.leakage_cap):
        return float("-inf"), "leakage_cap"

    g_lcb = max(float(feat.g_abs) - float(cfg.z_alpha) * float(max(0.0, feat.sigma_hat)), 0.0)
    if g_lcb <= 0.0 or float(feat.F_metric) <= 0.0:
        return 0.0, "nonpositive_gradient_or_metric"

    novelty = 1.0 if feat.novelty is None else min(max(float(feat.novelty), 0.0), 1.0)
    fallback_mode = "full"
    h_eff = float(cfg.lambda_F) * float(feat.F_metric)
    if feat.h_hat is None:
        fallback_mode = "lambda_F_metric_only"
    elif feat.b_hat is None or feat.H_window is None:
        h_eff = float(max(0.0, feat.h_hat))
        fallback_mode = "self_curvature_only"
    else:
        try:
            b_vec = np.asarray(feat.b_hat, dtype=float).reshape(-1)
            H_mat = np.asarray(feat.H_window, dtype=float)
            if H_mat.ndim != 2 or H_mat.shape[0] != H_mat.shape[1] or H_mat.shape[0] != b_vec.size:
                raise ValueError("invalid_shape")
            H_reg = H_mat + float(cfg.lambda_H) * np.eye(H_mat.shape[0], dtype=float)
            correction = float(b_vec.T @ np.linalg.solve(H_reg, b_vec))
            h_eff = float(max(0.0, float(feat.h_hat) - correction))
        except Exception:
            h_eff = float(max(0.0, feat.h_hat))
            fallback_mode = "curvature_solve_failed"

    delta_e = trust_region_drop(g_lcb, h_eff, float(feat.F_metric), float(cfg.rho))
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
) -> list[dict[str, Any]]:
    ranked = sorted(
        [dict(rec) for rec in records],
        key=lambda rec: (
            -float(rec.get(score_key, float("-inf"))),
            -float(rec.get("simple_score", float("-inf"))),
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


def _tangent_overlap_matrix(tangents: Sequence[np.ndarray]) -> np.ndarray:
    n = int(len(tangents))
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.real(np.vdot(tangents[i], tangents[j])))
            out[i, j] = val
            out[j, i] = val
    return out


class Phase2NoveltyOracle:
    """Shortlist-only tangent novelty using the current statevector."""

    def estimate(
        self,
        *,
        psi_state: np.ndarray,
        candidate_label: str,
        candidate_term: Any,
        window_terms: Sequence[Any],
        window_labels: Sequence[str],
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
        novelty_eps: float = 1e-6,
    ) -> Mapping[str, Any]:
        candidate_tangent, F_metric = _tangent_data(
            psi_state=psi_state,
            label=str(candidate_label),
            polynomial=candidate_term.polynomial,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
        )
        if F_metric <= 0.0:
            return {
                "novelty": 0.0,
                "novelty_mode": "exact_statevector_tangent_window_proxy",
                "candidate_tangent": candidate_tangent,
                "window_tangents": [],
                "window_overlap": [],
                "window_gram": [],
                "F_metric": float(F_metric),
            }
        if not window_terms:
            return {
                "novelty": 1.0,
                "novelty_mode": "exact_statevector_tangent_window_proxy",
                "candidate_tangent": candidate_tangent,
                "window_tangents": [],
                "window_overlap": [],
                "window_gram": [],
                "F_metric": float(F_metric),
            }
        window_tangents = []
        for lbl, term in zip(window_labels, window_terms):
            tangent_j, _ = _tangent_data(
                psi_state=psi_state,
                label=str(lbl),
                polynomial=term.polynomial,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
            )
            window_tangents.append(tangent_j)
        overlap = np.asarray(
            [float(np.real(np.vdot(tj, candidate_tangent))) for tj in window_tangents],
            dtype=float,
        )
        gram = _tangent_overlap_matrix(window_tangents)
        try:
            novelty_raw = 1.0 - float(
                overlap.T
                @ np.linalg.solve(gram + float(novelty_eps) * np.eye(gram.shape[0], dtype=float), overlap)
                / float(F_metric)
            )
            novelty_val = float(min(1.0, max(0.0, novelty_raw)))
        except Exception:
            novelty_val = 1.0
        return {
            "novelty": float(novelty_val),
            "novelty_mode": "exact_statevector_tangent_window_proxy",
            "candidate_tangent": candidate_tangent,
            "window_tangents": list(window_tangents),
            "window_overlap": [float(x) for x in overlap.tolist()],
            "window_gram": [[float(x) for x in row] for row in gram.tolist()],
            "F_metric": float(F_metric),
        }


class Phase2CurvatureOracle:
    """Shortlist-only Schur-style metric/curvature proxy."""

    def estimate(
        self,
        *,
        base_feature: CandidateFeatures,
        novelty_info: Mapping[str, Any],
        optimizer_memory: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        F_metric = float(max(0.0, novelty_info.get("F_metric", base_feature.F_metric)))
        overlaps = [float(x) for x in novelty_info.get("window_overlap", [])]
        gram_rows = [
            [float(x) for x in row]
            for row in novelty_info.get("window_gram", [])
            if isinstance(row, Sequence)
        ]
        if not overlaps or not gram_rows:
            return {
                "h_hat": float(F_metric),
                "b_hat": None,
                "H_window": None,
                "curvature_mode": "lambda_F_metric_proxy_only",
            }

        H_window = np.asarray(gram_rows, dtype=float)
        if H_window.ndim != 2 or H_window.shape[0] != H_window.shape[1]:
            return {
                "h_hat": float(F_metric),
                "b_hat": None,
                "H_window": None,
                "curvature_mode": "lambda_F_metric_proxy_only",
            }
        if isinstance(optimizer_memory, Mapping):
            raw_diag = list(optimizer_memory.get("preconditioner_diag", []))
            if raw_diag:
                mem_diag = []
                for local_idx, _global_idx in enumerate(base_feature.refit_window_indices):
                    if local_idx < len(raw_diag):
                        denom = max(float(raw_diag[local_idx]), 1e-8)
                        mem_diag.append(1.0 / denom)
                    else:
                        mem_diag.append(0.0)
                if len(mem_diag) == H_window.shape[0]:
                    H_window = H_window + np.diag(np.asarray(mem_diag, dtype=float))
                    mode = "schur_metric_proxy_with_memory_diag"
                else:
                    mode = "schur_metric_proxy"
            else:
                mode = "schur_metric_proxy"
        else:
            mode = "schur_metric_proxy"
        return {
            "h_hat": float(F_metric),
            "b_hat": [float(x) for x in overlaps],
            "H_window": [[float(x) for x in row] for row in H_window.tolist()],
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
    psi_state: np.ndarray,
    candidate_term: Any,
    window_terms: Sequence[Any],
    window_labels: Sequence[str],
    cfg: FullScoreConfig,
    novelty_oracle: NoveltyOracle,
    curvature_oracle: CurvatureOracle,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    optimizer_memory: Mapping[str, Any] | None = None,
    motif_library: Mapping[str, Any] | None = None,
    target_num_sites: int | None = None,
) -> CandidateFeatures:
    novelty_info = novelty_oracle.estimate(
        psi_state=psi_state,
        candidate_label=str(base_feature.candidate_label),
        candidate_term=candidate_term,
        window_terms=list(window_terms),
        window_labels=list(window_labels),
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        novelty_eps=float(cfg.novelty_eps),
    )
    curvature_info = curvature_oracle.estimate(
        base_feature=base_feature,
        novelty_info=novelty_info,
        optimizer_memory=optimizer_memory,
    )
    feat = _replace_feature(
        base_feature,
        novelty=(
            None
            if novelty_info.get("novelty") is None
            else float(novelty_info.get("novelty", 1.0))
        ),
        novelty_mode=str(novelty_info.get("novelty_mode", base_feature.novelty_mode)),
        F_metric=float(max(0.0, novelty_info.get("F_metric", base_feature.F_metric))),
        metric_proxy=float(max(0.0, novelty_info.get("F_metric", base_feature.metric_proxy))),
        h_hat=(
            None
            if curvature_info.get("h_hat") is None
            else float(curvature_info.get("h_hat", 0.0))
        ),
        b_hat=(
            None
            if curvature_info.get("b_hat") is None
            else [float(x) for x in curvature_info.get("b_hat", [])]
        ),
        H_window=(
            None
            if curvature_info.get("H_window") is None
            else [[float(x) for x in row] for row in curvature_info.get("H_window", [])]
        ),
        curvature_mode=str(curvature_info.get("curvature_mode", base_feature.curvature_mode)),
        score_version=str(cfg.score_version),
        motif_bonus=float(base_feature.motif_bonus),
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
    return _replace_feature(
        feat,
        full_v2_score=float(score),
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
        return {"support_overlap": 0.0, "noncommutation": 0.0, "cross_curvature": 0.0, "schedule": 0.0, "total": 0.0}

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
    total = (
        float(cfg.compat_overlap_weight) * float(support_overlap)
        + float(cfg.compat_comm_weight) * float(noncomm)
        + float(cfg.compat_curv_weight) * float(cross_curv)
        + float(cfg.compat_sched_weight) * float(schedule)
    )
    return {
        "support_overlap": float(support_overlap),
        "noncommutation": float(noncomm),
        "cross_curvature": float(cross_curv),
        "schedule": float(schedule),
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


def greedy_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    compat_oracle: CompatibilityPenaltyOracle,
    cfg: FullScoreConfig,
) -> tuple[list[dict[str, Any]], float]:
    ranked = sorted(
        [dict(rec) for rec in ranked_records],
        key=lambda rec: (
            -float(rec.get("full_v2_score", float("-inf"))),
            -float(rec.get("simple_score", float("-inf"))),
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
    cfg: SimpleScoreConfig,
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
        compiled_position_cost_proxy={
            "new_pauli_actions": float(compile_cost.new_pauli_actions),
            "new_rotation_steps": float(compile_cost.new_rotation_steps),
            "position_shift_span": float(compile_cost.position_shift_span),
            "refit_active_count": float(compile_cost.refit_active_count),
            "proxy_total": float(compile_cost.proxy_total),
        },
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
        depth_cost=float(compile_cost.new_rotation_steps + compile_cost.position_shift_span),
        new_group_cost=float(measurement_stats.groups_new),
        new_shot_cost=float(measurement_stats.shots_new),
        opt_dim_cost=float(len(refit_window_indices)),
        reuse_count_cost=float(measurement_stats.reuse_count_cost),
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
        },
    )
    score = simple_v1_score(feat, cfg)
    return _replace_feature(feat, simple_score=float(score))
