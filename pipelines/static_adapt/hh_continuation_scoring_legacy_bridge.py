#!/usr/bin/env python3
"""Legacy scorer bridge that can swap proxy-reduced Phase-3 geometry for exact-reduced geometry."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.compiled_polynomial import CompiledPolynomialAction, apply_compiled_polynomial


LEGACY_SCORER_MODULE_NAME = globals().get(
    "LEGACY_SCORER_MODULE_NAME",
    "pipelines.static_adapt._legacy_hh_continuation_scoring_20260322",
)
PHASE3_LEGACY_GEOMETRY_MODE = str(globals().get("PHASE3_LEGACY_GEOMETRY_MODE", "proxy_reduced")).strip().lower()

_LEGACY = importlib.import_module(str(LEGACY_SCORER_MODULE_NAME))
_CURRENT = importlib.import_module("pipelines.scaffold.hh_continuation_scoring")

CandidateFeatures = _LEGACY.CandidateFeatures
CompileCostEstimate = _LEGACY.CompileCostEstimate
CurvatureOracle = _LEGACY.CurvatureOracle
MeasurementCacheAudit = _LEGACY.MeasurementCacheAudit
MeasurementCacheStats = _LEGACY.MeasurementCacheStats
MeasurementPlan = _LEGACY.MeasurementPlan
NoveltyOracle = _LEGACY.NoveltyOracle
SimpleScoreConfig = _LEGACY.SimpleScoreConfig
FullScoreConfig = _LEGACY.FullScoreConfig
Phase1CompileCostOracle = _LEGACY.Phase1CompileCostOracle
Phase2NoveltyOracle = _LEGACY.Phase2NoveltyOracle
Phase2CurvatureOracle = _LEGACY.Phase2CurvatureOracle
CompatibilityPenaltyOracle = _LEGACY.CompatibilityPenaltyOracle
build_candidate_features = _LEGACY.build_candidate_features
compatibility_penalty = _LEGACY.compatibility_penalty
full_v2_score = _LEGACY.full_v2_score
greedy_batch_select = _LEGACY.greedy_batch_select
lifetime_weight_components = _LEGACY.lifetime_weight_components
measurement_group_keys_for_term = _LEGACY.measurement_group_keys_for_term
normalize = _LEGACY.normalize
remaining_evaluations_proxy = _LEGACY.remaining_evaluations_proxy
shortlist_records = _LEGACY.shortlist_records
trust_region_drop = _LEGACY.trust_region_drop

__all__ = [
    "CandidateFeatures",
    "CompileCostEstimate",
    "CurvatureOracle",
    "MeasurementCacheAudit",
    "MeasurementCacheStats",
    "MeasurementPlan",
    "NoveltyOracle",
    "SimpleScoreConfig",
    "FullScoreConfig",
    "Phase1CompileCostOracle",
    "Phase2NoveltyOracle",
    "Phase2CurvatureOracle",
    "CompatibilityPenaltyOracle",
    "build_candidate_features",
    "build_full_candidate_features",
    "compatibility_penalty",
    "full_v2_score",
    "greedy_batch_select",
    "lifetime_weight_components",
    "measurement_group_keys_for_term",
    "normalize",
    "remaining_evaluations_proxy",
    "shortlist_records",
    "trust_region_drop",
]


def _replace_feature(feat: CandidateFeatures, **updates: Any) -> CandidateFeatures:
    return CandidateFeatures(**{**feat.__dict__, **updates})


def _coerce_current_cfg(cfg: FullScoreConfig) -> Any:
    return _CURRENT.FullScoreConfig(
        z_alpha=float(getattr(cfg, "z_alpha", 0.0)),
        lambda_F=float(getattr(cfg, "lambda_F", 1.0)),
        lambda_H=float(getattr(cfg, "lambda_H", 1e-6)),
        rho=float(getattr(cfg, "rho", 0.25)),
        eta_L=float(getattr(cfg, "eta_L", 0.0)),
        gamma_N=float(getattr(cfg, "gamma_N", 1.0)),
        wD=float(getattr(cfg, "wD", 0.2)),
        wG=float(getattr(cfg, "wG", 0.15)),
        wC=float(getattr(cfg, "wC", 0.15)),
        wP=float(getattr(cfg, "wP", 0.1)),
        wc=float(getattr(cfg, "wc", 0.1)),
        depth_ref=float(getattr(cfg, "depth_ref", 1.0)),
        group_ref=float(getattr(cfg, "group_ref", 1.0)),
        shot_ref=float(getattr(cfg, "shot_ref", 1.0)),
        optdim_ref=float(getattr(cfg, "optdim_ref", 1.0)),
        reuse_ref=float(getattr(cfg, "reuse_ref", 1.0)),
        novelty_eps=float(getattr(cfg, "novelty_eps", 1e-6)),
        compat_overlap_weight=float(getattr(cfg, "compat_overlap_weight", 0.4)),
        compat_comm_weight=float(getattr(cfg, "compat_comm_weight", 0.2)),
        compat_curv_weight=float(getattr(cfg, "compat_curv_weight", 0.2)),
        compat_sched_weight=float(getattr(cfg, "compat_sched_weight", 0.2)),
        compat_measure_weight=float(getattr(cfg, "compat_measure_weight", 0.2)),
        leakage_cap=float(getattr(cfg, "leakage_cap", 1e6)),
        lifetime_cost_mode=str(getattr(cfg, "lifetime_cost_mode", "off")),
        remaining_evaluations_proxy_mode=str(getattr(cfg, "remaining_evaluations_proxy_mode", "none")),
        lifetime_weight=float(getattr(cfg, "lifetime_weight", 0.05)),
        motif_bonus_weight=float(getattr(cfg, "motif_bonus_weight", 0.05)),
        phase3_selector_geometry_mode="reduced",
    )


def _extract_exact_reduced_debug(
    *,
    base_feature: CandidateFeatures,
    psi_state: np.ndarray,
    candidate_term: Any,
    cfg: FullScoreConfig,
    pauli_action_cache: dict[str, Any] | None,
    legacy_selected_ops: Sequence[Any] | None,
    legacy_theta: np.ndarray | None,
    legacy_psi_ref: np.ndarray | None,
    legacy_h_compiled: CompiledPolynomialAction | None,
) -> dict[str, Any]:
    if legacy_selected_ops is None or legacy_theta is None or legacy_psi_ref is None or legacy_h_compiled is None:
        raise ValueError("legacy exact-reduced geometry requires selected_ops, theta, psi_ref, and h_compiled context")

    psi_state_arr = np.asarray(psi_state, dtype=complex).reshape(-1)
    hpsi_state = apply_compiled_polynomial(psi_state_arr, legacy_h_compiled)
    exact_cfg = _coerce_current_cfg(cfg)
    novelty_oracle = _CURRENT.Phase2NoveltyOracle()
    scaffold_context = novelty_oracle.prepare_scaffold_context(
        selected_ops=list(legacy_selected_ops),
        theta=np.asarray(legacy_theta, dtype=float),
        psi_ref=np.asarray(legacy_psi_ref, dtype=complex),
        psi_state=psi_state_arr,
        h_compiled=legacy_h_compiled,
        hpsi_state=np.asarray(hpsi_state, dtype=complex),
        refit_window_indices=list(base_feature.refit_window_indices),
        pauli_action_cache=pauli_action_cache,
    )
    novelty_info = novelty_oracle.estimate(
        scaffold_context=scaffold_context,
        candidate_label=str(base_feature.candidate_label),
        candidate_term=candidate_term,
        compiled_cache=None,
        pauli_action_cache=pauli_action_cache,
        novelty_eps=float(exact_cfg.novelty_eps),
    )
    base_payload = dict(base_feature.__dict__)
    base_payload["F_raw"] = None
    base_exact = SimpleNamespace(**base_payload)
    curvature_info = _CURRENT.Phase2CurvatureOracle().estimate(
        base_feature=base_exact,
        novelty_info=novelty_info,
        scaffold_context=scaffold_context,
        h_compiled=legacy_h_compiled,
        cfg=exact_cfg,
        optimizer_memory=None,
    )
    score_payload = dict(base_feature.__dict__)
    score_payload.update(
        {
            "F_raw": float(max(0.0, novelty_info.get("F_raw", getattr(base_feature, "F_metric", 0.0)))),
            "F_red": float(curvature_info.get("F_red", novelty_info.get("F_raw", getattr(base_feature, "F_metric", 0.0)))),
            "h_eff": float(curvature_info.get("h_eff", 0.0)),
            "h_hat": float(curvature_info.get("h_raw", 0.0)),
            "novelty": float(curvature_info.get("novelty", 1.0)),
            "ridge_used": float(curvature_info.get("ridge_used", max(getattr(cfg, "lambda_H", 0.0), 0.0))),
            "curvature_mode": str(curvature_info.get("curvature_mode", "append_exact_window_hessian_v1")),
            "novelty_mode": str(novelty_info.get("novelty_mode", "append_exact_tangent_context_v1")),
        }
    )
    score_feat = SimpleNamespace(**score_payload)
    exact_score, exact_fallback = _CURRENT.full_v2_score(score_feat, exact_cfg)
    return {
        "exact_reduced_full_v2_score": float(exact_score),
        "exact_reduced_fallback_mode": str(exact_fallback),
        "exact_reduced_novelty": float(score_feat.novelty),
        "exact_reduced_F_raw": float(score_feat.F_raw),
        "exact_reduced_F_red": float(score_feat.F_red),
        "exact_reduced_h_raw": float(score_feat.h_hat),
        "exact_reduced_h_eff": float(score_feat.h_eff),
        "exact_reduced_ridge_used": float(score_feat.ridge_used),
        "exact_reduced_curvature_mode": str(score_feat.curvature_mode),
        "exact_reduced_novelty_mode": str(score_feat.novelty_mode),
        "exact_reduced_window_size": int(len(getattr(base_feature, "refit_window_indices", []))),
    }


def _attach_phase3_debug(
    feat: CandidateFeatures,
    *,
    selector_score: float,
    debug_payload: Mapping[str, Any],
    fallback_mode: str,
) -> CandidateFeatures:
    compiled_backend = dict(feat.compiled_position_cost_backend or {})
    compiled_backend["phase3_geometry_debug"] = dict(debug_payload)
    return _replace_feature(
        feat,
        full_v2_score=float(selector_score),
        actual_fallback_mode=str(fallback_mode),
        compiled_position_cost_backend=compiled_backend,
    )


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
    compiled_cache: dict[str, Any] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    optimizer_memory: Mapping[str, Any] | None = None,
    motif_library: Mapping[str, Any] | None = None,
    target_num_sites: int | None = None,
    legacy_selected_ops: Sequence[Any] | None = None,
    legacy_theta: np.ndarray | None = None,
    legacy_psi_ref: np.ndarray | None = None,
    legacy_h_compiled: CompiledPolynomialAction | None = None,
) -> CandidateFeatures:
    legacy_feat = _LEGACY.build_full_candidate_features(
        base_feature=base_feature,
        psi_state=np.asarray(psi_state, dtype=complex),
        candidate_term=candidate_term,
        window_terms=list(window_terms),
        window_labels=[str(x) for x in window_labels],
        cfg=cfg,
        novelty_oracle=novelty_oracle,
        curvature_oracle=curvature_oracle,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        optimizer_memory=optimizer_memory,
        motif_library=motif_library,
        target_num_sites=target_num_sites,
    )
    legacy_score = float(legacy_feat.full_v2_score or 0.0)
    geometry_mode = str(PHASE3_LEGACY_GEOMETRY_MODE).strip().lower()
    debug_payload: dict[str, Any] = {
        "phase3_legacy_geometry_mode": str(geometry_mode),
        "legacy_proxy_full_v2_score": float(legacy_score),
    }
    if geometry_mode != "exact_reduced":
        debug_payload["selector_score"] = float(legacy_score)
        return _attach_phase3_debug(
            legacy_feat,
            selector_score=float(legacy_score),
            debug_payload=debug_payload,
            fallback_mode=str(legacy_feat.actual_fallback_mode),
        )
    try:
        exact_debug = _extract_exact_reduced_debug(
            base_feature=base_feature,
            psi_state=np.asarray(psi_state, dtype=complex),
            candidate_term=candidate_term,
            cfg=cfg,
            pauli_action_cache=pauli_action_cache,
            legacy_selected_ops=legacy_selected_ops,
            legacy_theta=legacy_theta,
            legacy_psi_ref=legacy_psi_ref,
            legacy_h_compiled=legacy_h_compiled,
        )
        debug_payload.update(dict(exact_debug))
        selector_score = float(debug_payload["exact_reduced_full_v2_score"])
        debug_payload["selector_score"] = float(selector_score)
        fallback_mode = str(debug_payload.get("exact_reduced_fallback_mode", "legacy_bridge_exact_reduced"))
        return _attach_phase3_debug(
            legacy_feat,
            selector_score=float(selector_score),
            debug_payload=debug_payload,
            fallback_mode=f"legacy_bridge_exact_reduced::{fallback_mode}",
        )
    except Exception as exc:
        debug_payload["selector_score"] = float(legacy_score)
        debug_payload["exact_reduced_error"] = str(exc)
        return _attach_phase3_debug(
            legacy_feat,
            selector_score=float(legacy_score),
            debug_payload=debug_payload,
            fallback_mode=f"legacy_bridge_exact_reduced_failed::{type(exc).__name__}",
        )
