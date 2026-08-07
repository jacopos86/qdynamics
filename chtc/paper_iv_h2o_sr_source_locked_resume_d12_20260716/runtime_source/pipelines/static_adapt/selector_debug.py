"""Selector-debug telemetry helpers for static ADAPT.

This module serializes selector debug payloads only. It deliberately does not
own scoring, route identity, beam expansion, pruning, noise, or problem/pool
construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from pipelines.scaffold.hh_continuation_scoring import (
    PHASE2_CANONICAL_RAW_SCORE_FORMULA,
    PHASE2_NOVELTY_COLLECTIVE_SPAN_V1,
    PHASE3_CANONICAL_SCORE_FORMULA,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures


@dataclass(frozen=True)
class _SelectorDebugContext:
    phase2_score_cfg: Any
    phase2_enable_batching: bool
    phase1_score_mode_key: str
    phase2_raw_score_formula: str
    phase3_selector_debug_topk_val: int
    phase3_selector_debug_max_depth_val: int
    phase3_selector_geometry_mode_key: str
    phase3_novelty_ablation_mode_key: str
    phase3_window_relaxation_mode_key: str
    phase3_shadow_legacy_geometry_mode_key: str
    phase3_shadow_legacy_max_depth_val: int
    phase3_parent_collapse_debug_max_depth_val: int
    phase2_batch_target_size_requested: int
    phase2_batch_size_cap_requested: int
    phase2_remaining_evaluations_proxy_mode: str
    phase3_lifetime_cost_mode_key: str
    phase3_hardware_cost_normalization_mode_key: str
    phase3_runtime_split_mode_key: str
    phase3_runtime_split_selection_mode_key: str
    phase3_source_lock_preferred_sequence_enabled: bool
    selector_score_key: str
    primary_selector_score_key: str
    selector_tie_break_score_key: str
    secondary_geometry_score_key: str
    record_sort_key: Callable[[Mapping[str, Any]], tuple[Any, ...]]


def _selector_debug_row(
    record: Mapping[str, Any],
    *,
    context: _SelectorDebugContext,
) -> dict[str, Any]:
    rec = dict(record)
    feat_obj = rec.get("feature")

    def _feat_get(name: str, default: Any = None) -> Any:
        if isinstance(feat_obj, CandidateFeatures):
            return getattr(feat_obj, name, default)
        if isinstance(feat_obj, Mapping):
            return feat_obj.get(name, default)
        return default

    candidate_label_value = rec.get("candidate_label")
    if candidate_label_value in {None, ""}:
        candidate_label_value = _feat_get("candidate_label")
    if candidate_label_value in {None, ""}:
        candidate_label_value = _feat_get("label")

    return {
        "candidate_label": str(candidate_label_value or ""),
        "candidate_pool_index": int(rec.get("candidate_pool_index", -1)),
        "position_id": int(rec.get("position_id", -1)),
        "score_version": str(_feat_get("score_version", "")),
        "curvature_mode": str(_feat_get("curvature_mode", "")),
        "novelty_mode": str(_feat_get("novelty_mode", "")),
        "actual_fallback_mode": str(_feat_get("actual_fallback_mode", "")),
        "representation": str(_feat_get("runtime_split_chosen_representation", "parent")),
        "runtime_split_parent_label": _feat_get("runtime_split_parent_label"),
        "runtime_split_child_labels": [
            str(x) for x in (_feat_get("runtime_split_child_labels", []) or [])
        ],
        "full_v2_score": float(rec.get("full_v2_score", float("-inf"))),
        "phase2_raw_score": float(rec.get("phase2_raw_score", float("-inf"))),
        "cheap_score": float(rec.get("cheap_score", rec.get("simple_score", float("-inf")))),
        "simple_score": float(rec.get("simple_score", float("-inf"))),
        "phase1_score_mode": str(_feat_get("phase1_score_mode", context.phase1_score_mode_key)),
        "phase1_active_score": (
            float(_feat_get("phase1_active_score"))
            if _feat_get("phase1_active_score") is not None
            else None
        ),
        "phase1_legacy_simple_score": (
            float(_feat_get("phase1_legacy_simple_score"))
            if _feat_get("phase1_legacy_simple_score") is not None
            else None
        ),
        "phase1_trust_region_gain": (
            float(_feat_get("phase1_trust_region_gain"))
            if _feat_get("phase1_trust_region_gain") is not None
            else None
        ),
        "phase1_trust_region_score": (
            float(_feat_get("phase1_trust_region_score"))
            if _feat_get("phase1_trust_region_score") is not None
            else None
        ),
        "phase1_rho": (
            float(_feat_get("phase1_rho"))
            if _feat_get("phase1_rho") is not None
            else None
        ),
        "selector_score": (
            float(_feat_get("selector_score"))
            if _feat_get("selector_score") is not None
            else None
        ),
        "phase3_primary_score": (
            float(_feat_get("phase3_primary_score"))
            if _feat_get("phase3_primary_score") is not None
            else None
        ),
        "phase3_tie_break_score": float(_feat_get("phase3_tie_break_score", 0.0) or 0.0),
        "phase3_auxiliary_score_mode": str(
            _feat_get("phase3_auxiliary_score_mode", "tie_break_only")
        ),
        "phase3_canonical_score_formula": str(
            _feat_get("phase3_canonical_score_formula", PHASE3_CANONICAL_SCORE_FORMULA)
        ),
        "selector_geometry_mode": str(_feat_get("selector_geometry_mode", "reduced")),
        "shadow_legacy_geometry_debug": (
            dict((_feat_get("compiled_position_cost_backend", {}) or {}).get("shadow_legacy_geometry_debug", {}))
            if isinstance(_feat_get("compiled_position_cost_backend", {}), Mapping)
            and isinstance((_feat_get("compiled_position_cost_backend", {}) or {}).get("shadow_legacy_geometry_debug"), Mapping)
            else None
        ),
        "g_abs": (
            float(_feat_get("g_abs"))
            if _feat_get("g_abs") is not None
            else None
        ),
        "g_lcb": (
            float(_feat_get("g_lcb"))
            if _feat_get("g_lcb") is not None
            else None
        ),
        "g_lcb_legacy_shot": (
            float(_feat_get("g_lcb_legacy_shot"))
            if _feat_get("g_lcb_legacy_shot") is not None
            else None
        ),
        "g_hw_lcb": (
            float(_feat_get("g_hw_lcb"))
            if _feat_get("g_hw_lcb") is not None
            else None
        ),
        "epsilon_g_shot": (
            float(_feat_get("epsilon_g_shot"))
            if _feat_get("epsilon_g_shot") is not None
            else None
        ),
        "b_g_hw": (
            float(_feat_get("b_g_hw"))
            if _feat_get("b_g_hw") is not None
            else None
        ),
        "b_g_drift": (
            float(_feat_get("b_g_drift"))
            if _feat_get("b_g_drift") is not None
            else None
        ),
        "epsilon_g_res": (
            float(_feat_get("epsilon_g_res"))
            if _feat_get("epsilon_g_res") is not None
            else None
        ),
        "hardware_resolution_mode": str(_feat_get("hardware_resolution_mode", "ideal")),
        "hardware_resolution_source": str(_feat_get("hardware_resolution_source", "legacy_unset")),
        "metric_proxy": (
            float(_feat_get("metric_proxy"))
            if _feat_get("metric_proxy") is not None
            else None
        ),
        "F_raw": (
            float(_feat_get("F_raw"))
            if _feat_get("F_raw") is not None
            else None
        ),
        "F_red": (
            float(_feat_get("F_red"))
            if _feat_get("F_red") is not None
            else None
        ),
        "h_eff": (
            float(_feat_get("h_eff"))
            if _feat_get("h_eff") is not None
            else None
        ),
        "ridge_used": (
            float(_feat_get("ridge_used"))
            if _feat_get("ridge_used") is not None
            else None
        ),
        "selector_burden": (
            float(_feat_get("selector_burden"))
            if _feat_get("selector_burden") is not None
            else None
        ),
        "phase2_raw_trust_gain": (
            float(_feat_get("phase2_raw_trust_gain"))
            if _feat_get("phase2_raw_trust_gain") is not None
            else None
        ),
        "phase3_reduced_trust_gain": (
            float(_feat_get("phase3_reduced_trust_gain"))
            if _feat_get("phase3_reduced_trust_gain") is not None
            else None
        ),
        "phase2_raw_novelty": (
            float(_feat_get("phase2_raw_novelty"))
            if _feat_get("phase2_raw_novelty") is not None
            else None
        ),
        "phase2_novelty_mode": str(
            _feat_get("phase2_novelty_mode", PHASE2_NOVELTY_COLLECTIVE_SPAN_V1)
        ),
        "phase2_novelty_source": str(
            _feat_get("phase2_novelty_source", PHASE2_NOVELTY_COLLECTIVE_SPAN_V1)
        ),
        "phase2_novelty_fallback_reason": _feat_get("phase2_novelty_fallback_reason"),
        "phase2_span_projection_z": (
            float(_feat_get("phase2_span_projection_z"))
            if _feat_get("phase2_span_projection_z") is not None
            else None
        ),
        "phase2_novelty_ridge_used": (
            float(_feat_get("phase2_novelty_ridge_used"))
            if _feat_get("phase2_novelty_ridge_used") is not None
            else None
        ),
        "phase2_raw_F_effective": (
            float(_feat_get("phase2_raw_F_effective"))
            if _feat_get("phase2_raw_F_effective") is not None
            else None
        ),
        "phase2_legacy_pairwise_novelty": (
            float(_feat_get("phase2_legacy_pairwise_novelty"))
            if _feat_get("phase2_legacy_pairwise_novelty") is not None
            else None
        ),
        "phase2_confidence_applied": bool(_feat_get("phase2_confidence_applied", False)),
        "phase2_raw_score_formula": str(
            _feat_get("phase2_raw_score_formula", PHASE2_CANONICAL_RAW_SCORE_FORMULA)
        ),
        "phase3_reduced_novelty": (
            float(_feat_get("phase3_reduced_novelty"))
            if _feat_get("phase3_reduced_novelty") is not None
            else None
        ),
        "phase2_burden_total": (
            float(_feat_get("phase2_burden_total"))
            if _feat_get("phase2_burden_total") is not None
            else None
        ),
        "phase3_burden_total": (
            float(_feat_get("phase3_burden_total"))
            if _feat_get("phase3_burden_total") is not None
            else None
        ),
        "compile_cost_total": (
            float(_feat_get("compile_cost_total"))
            if _feat_get("compile_cost_total") is not None
            else None
        ),
        "depth_cost": (
            float(_feat_get("depth_cost"))
            if _feat_get("depth_cost") is not None
            else None
        ),
        "new_group_cost": (
            float(_feat_get("new_group_cost"))
            if _feat_get("new_group_cost") is not None
            else None
        ),
        "new_shot_cost": (
            float(_feat_get("new_shot_cost"))
            if _feat_get("new_shot_cost") is not None
            else None
        ),
        "opt_dim_cost": (
            float(_feat_get("opt_dim_cost"))
            if _feat_get("opt_dim_cost") is not None
            else None
        ),
        "reuse_count_cost": (
            float(_feat_get("reuse_count_cost"))
            if _feat_get("reuse_count_cost") is not None
            else None
        ),
        "family_repeat_cost": (
            float(_feat_get("family_repeat_cost"))
            if _feat_get("family_repeat_cost") is not None
            else None
        ),
        "motif_bonus": (
            float(_feat_get("motif_bonus"))
            if _feat_get("motif_bonus") is not None
            else None
        ),
        "compatibility_penalty_total": (
            float(_feat_get("compatibility_penalty_total"))
            if _feat_get("compatibility_penalty_total") is not None
            else None
        ),
    }


def _selector_debug_rows(
    rows_raw: Sequence[Mapping[str, Any]] | None,
    *,
    context: _SelectorDebugContext,
    topk: int,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows_raw if isinstance(row, Mapping)] if isinstance(rows_raw, Sequence) else []
    if int(topk) <= 0 or not rows:
        return []
    rows_sorted = sorted(rows, key=context.record_sort_key)[: int(topk)]
    return [_selector_debug_row(row, context=context) for row in rows_sorted]


def _selector_debug_enabled_for_depth(
    context: _SelectorDebugContext,
    *,
    depth_one_based: int,
) -> bool:
    if int(context.phase3_selector_debug_topk_val) <= 0:
        return False
    if int(context.phase3_selector_debug_max_depth_val) <= 0:
        return True
    return int(depth_one_based) <= int(context.phase3_selector_debug_max_depth_val)


def _selector_debug_payload(
    *,
    context: _SelectorDebugContext,
    depth_one_based: int,
    beam_enabled: bool,
    selection_mode_value: str,
    stage_name_value: str,
    selected_feature_row: Mapping[str, Any] | None,
    scored_rows: Sequence[Mapping[str, Any]] | None,
    phase2_rows: Sequence[Mapping[str, Any]] | None,
    phase3_rows: Sequence[Mapping[str, Any]] | None,
    admitted_rows: Sequence[Mapping[str, Any]] | None,
    split_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    phase2_score_cfg = context.phase2_score_cfg
    topk = int(context.phase3_selector_debug_topk_val)
    return {
        "depth": int(depth_one_based),
        "beam_enabled": bool(beam_enabled),
        "selection_mode": str(selection_mode_value),
        "stage_name": str(stage_name_value),
        "score_config": {
            "lambda_H": float(getattr(phase2_score_cfg, "lambda_H", 0.0)),
            "rho": float(getattr(phase2_score_cfg, "rho", 0.0)),
            "gamma_N": float(getattr(phase2_score_cfg, "gamma_N", 0.0)),
            "phase2_frontier_ratio": float(getattr(phase2_score_cfg, "phase2_frontier_ratio", 0.0)),
            "phase3_frontier_ratio": float(getattr(phase2_score_cfg, "phase3_frontier_ratio", 0.0)),
            "batching_enabled": bool(context.phase2_enable_batching),
            "batch_selection_mode": str(getattr(phase2_score_cfg, "batch_selection_mode", "reduced_plane")),
            "batch_target_size": int(getattr(phase2_score_cfg, "batch_target_size", 0)),
            "batch_size_cap": int(getattr(phase2_score_cfg, "batch_size_cap", 0)),
            "batch_near_degenerate_ratio": float(
                getattr(phase2_score_cfg, "batch_near_degenerate_ratio", 0.0)
            ),
            "batch_rank_rel_tol": float(getattr(phase2_score_cfg, "batch_rank_rel_tol", 0.0)),
            "batch_additivity_tol": float(getattr(phase2_score_cfg, "batch_additivity_tol", 0.0)),
            "w_depth": float(getattr(phase2_score_cfg, "wD", 0.0)),
            "w_group": float(getattr(phase2_score_cfg, "wG", 0.0)),
            "w_shot": float(getattr(phase2_score_cfg, "wC", 0.0)),
            "w_optdim": float(getattr(phase2_score_cfg, "wP", 0.0)),
            "w_reuse": float(getattr(phase2_score_cfg, "wc", 0.0)),
            "w_lifetime": float(getattr(phase2_score_cfg, "lifetime_weight", 0.0)),
            "lambda_F": float(getattr(phase2_score_cfg, "lambda_F", 0.0)),
            "score_z_alpha": float(getattr(phase2_score_cfg, "z_alpha", 0.0)),
            "eta_L": float(getattr(phase2_score_cfg, "eta_L", 0.0)),
            "depth_ref": float(getattr(phase2_score_cfg, "depth_ref", 1.0)),
            "group_ref": float(getattr(phase2_score_cfg, "group_ref", 1.0)),
            "shot_ref": float(getattr(phase2_score_cfg, "shot_ref", 1.0)),
            "optdim_ref": float(getattr(phase2_score_cfg, "optdim_ref", 1.0)),
            "reuse_ref": float(getattr(phase2_score_cfg, "reuse_ref", 1.0)),
            "family_ref": float(getattr(phase2_score_cfg, "family_ref", 1.0)),
            "novelty_eps": float(getattr(phase2_score_cfg, "novelty_eps", 0.0)),
            "phase2_novelty_mode": str(
                getattr(phase2_score_cfg, "phase2_novelty_mode", PHASE2_NOVELTY_COLLECTIVE_SPAN_V1)
            ),
            "phase2_selector_gain_mode": str(
                getattr(phase2_score_cfg, "phase2_selector_gain_mode", "trust_region_v1")
            ),
            "phase2_raw_score_formula": context.phase2_raw_score_formula,
            "cheap_score_eps": float(getattr(phase2_score_cfg, "cheap_score_eps", 0.0)),
            "metric_floor": float(getattr(phase2_score_cfg, "metric_floor", 0.0)),
            "reduced_metric_collapse_rel_tol": float(
                getattr(phase2_score_cfg, "reduced_metric_collapse_rel_tol", 0.0)
            ),
            "ridge_growth_factor": float(getattr(phase2_score_cfg, "ridge_growth_factor", 0.0)),
            "ridge_max_steps": int(getattr(phase2_score_cfg, "ridge_max_steps", 0)),
            "geometry_profile": str(getattr(phase2_score_cfg, "geometry_profile", "current_v1")),
            "selector_geometry_mode": str(context.phase3_selector_geometry_mode_key),
            "novelty_ablation_mode": str(context.phase3_novelty_ablation_mode_key),
            "window_relaxation_mode": str(context.phase3_window_relaxation_mode_key),
            "shadow_legacy_geometry_mode": str(context.phase3_shadow_legacy_geometry_mode_key),
            "shadow_legacy_max_depth": int(context.phase3_shadow_legacy_max_depth_val),
            "parent_collapse_debug_max_depth": int(context.phase3_parent_collapse_debug_max_depth_val),
            "selector_score_key": str(context.selector_score_key),
            "primary_selector_score_key": str(context.primary_selector_score_key),
            "selector_tie_break_score_key": str(context.selector_tie_break_score_key),
            "secondary_geometry_score_key": str(context.secondary_geometry_score_key),
            "canonical_score_formula": PHASE3_CANONICAL_SCORE_FORMULA,
            "auxiliary_terms_primary_mode": str(
                getattr(phase2_score_cfg, "auxiliary_score_mode", "tie_break_only")
            ),
            "batch_target_size_requested": int(context.phase2_batch_target_size_requested),
            "batch_target_size_effective": int(phase2_score_cfg.batch_target_size),
            "batch_size_cap_requested": int(context.phase2_batch_size_cap_requested),
            "batch_size_cap_effective": int(phase2_score_cfg.batch_size_cap),
            "leakage_cap": float(getattr(phase2_score_cfg, "leakage_cap", 0.0)),
            "motif_bonus_weight": float(getattr(phase2_score_cfg, "motif_bonus_weight", 0.0)),
            "duplicate_penalty_weight": float(
                getattr(phase2_score_cfg, "duplicate_penalty_weight", 0.0)
            ),
            "compat_overlap_weight": float(
                getattr(phase2_score_cfg, "compat_overlap_weight", 0.0)
            ),
            "compat_comm_weight": float(getattr(phase2_score_cfg, "compat_comm_weight", 0.0)),
            "compat_curv_weight": float(getattr(phase2_score_cfg, "compat_curv_weight", 0.0)),
            "compat_sched_weight": float(getattr(phase2_score_cfg, "compat_sched_weight", 0.0)),
            "compat_measure_weight": float(
                getattr(phase2_score_cfg, "compat_measure_weight", 0.0)
            ),
            "remaining_evaluations_proxy_mode": str(context.phase2_remaining_evaluations_proxy_mode),
            "lifetime_cost_mode": str(context.phase3_lifetime_cost_mode_key),
            "hardware_cost_normalization_mode": str(context.phase3_hardware_cost_normalization_mode_key),
            "runtime_split_mode": str(context.phase3_runtime_split_mode_key),
            "runtime_split_selection_mode": str(
                context.phase3_runtime_split_selection_mode_key
            ),
            "source_lock_preferred_sequence_enabled": bool(
                context.phase3_source_lock_preferred_sequence_enabled
            ),
        },
        "runtime_split_summary": (
            dict(split_summary) if isinstance(split_summary, Mapping) else {}
        ),
        "selected": (
            _selector_debug_row(
                {"feature": dict(selected_feature_row), **dict(selected_feature_row)},
                context=context,
            )
            if isinstance(selected_feature_row, Mapping)
            else None
        ),
        "scored_topk": _selector_debug_rows(
            scored_rows,
            context=context,
            topk=topk,
        ),
        "phase2_shortlist_topk": _selector_debug_rows(
            phase2_rows,
            context=context,
            topk=topk,
        ),
        "phase3_shortlist_topk": _selector_debug_rows(
            phase3_rows,
            context=context,
            topk=topk,
        ),
        "admitted_topk": _selector_debug_rows(
            admitted_rows,
            context=context,
            topk=topk,
        ),
    }
