"""Phase-3 selector/shadow geometry helpers for static ADAPT."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_types import CandidateFeatures

__all__ = [
    "_SHADOW_LEGACY_GEOMETRY_MODES",
    "_SHADOW_LEGACY_BRIDGE_CACHE",
    "_normalize_shadow_legacy_geometry_mode",
    "_load_shadow_legacy_geometry_bridge",
    "_shadow_float_or_none",
    "_shadow_str_or_none",
    "_shadow_legacy_geometry_debug_payload",
    "_attach_shadow_legacy_geometry_debug",
    "_apply_phase3_selector_geometry_override",
    "_phase3_shadow_geometry_active_for_depth",
    "_maybe_apply_phase3_proxy_selector_geometry",
    "_maybe_attach_shadow_legacy_geometry",
    "_apply_runtime_split_family_aggregation_override",
]

_SHADOW_LEGACY_GEOMETRY_MODES = {"off", "proxy_reduced", "exact_reduced"}
_SHADOW_LEGACY_BRIDGE_CACHE: dict[str, Any] = {}


def _normalize_shadow_legacy_geometry_mode(mode: str | None) -> str:
    mode_key = str(mode or "off").strip().lower()
    if mode_key not in _SHADOW_LEGACY_GEOMETRY_MODES:
        raise ValueError(
            "phase3_shadow_legacy_geometry_mode must be one of "
            f"{set(_SHADOW_LEGACY_GEOMETRY_MODES)}."
        )
    return str(mode_key)


def _load_shadow_legacy_geometry_bridge(mode: str) -> Any:
    mode_key = _normalize_shadow_legacy_geometry_mode(mode)
    if mode_key == "off":
        return None
    raise RuntimeError(
        "The shadow legacy Phase-III geometry bridge was author-retired by "
        "the Paper-I RA-ADAPT unification. Historical artifacts retain "
        "their emitted telemetry; active RA-ADAPT uses typed exact ordered "
        "insertion geometry."
    )


def _shadow_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _shadow_str_or_none(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _shadow_legacy_geometry_debug_payload(
    *,
    current_feat: CandidateFeatures,
    shadow_feat: Any,
    shadow_mode: str,
) -> dict[str, Any]:
    shadow_backend = getattr(shadow_feat, "compiled_position_cost_backend", None)
    bridge_debug = (
        dict(shadow_backend.get("phase3_geometry_debug", {}))
        if isinstance(shadow_backend, Mapping)
        and isinstance(shadow_backend.get("phase3_geometry_debug"), Mapping)
        else None
    )
    current_score = _shadow_float_or_none(getattr(current_feat, "full_v2_score", None))
    shadow_score = _shadow_float_or_none(getattr(shadow_feat, "full_v2_score", None))
    return {
        "status": "ok",
        "mode": str(shadow_mode),
        "current_full_v2_score": current_score,
        "shadow_full_v2_score": shadow_score,
        "shadow_minus_current_full_v2_score": (
            float(shadow_score - current_score)
            if current_score is not None and shadow_score is not None
            else None
        ),
        "current_phase2_raw_score": _shadow_float_or_none(getattr(current_feat, "phase2_raw_score", None)),
        "shadow_phase2_raw_score": _shadow_float_or_none(getattr(shadow_feat, "phase2_raw_score", None)),
        "current_novelty": _shadow_float_or_none(getattr(current_feat, "novelty", None)),
        "shadow_novelty": _shadow_float_or_none(getattr(shadow_feat, "novelty", None)),
        "current_novelty_mode": _shadow_str_or_none(getattr(current_feat, "novelty_mode", None)),
        "shadow_novelty_mode": _shadow_str_or_none(getattr(shadow_feat, "novelty_mode", None)),
        "current_curvature_mode": _shadow_str_or_none(getattr(current_feat, "curvature_mode", None)),
        "shadow_curvature_mode": _shadow_str_or_none(getattr(shadow_feat, "curvature_mode", None)),
        "current_F": _shadow_float_or_none(getattr(current_feat, "F", None)),
        "shadow_F": _shadow_float_or_none(getattr(shadow_feat, "F", None)),
        "current_F_raw": _shadow_float_or_none(getattr(current_feat, "F_raw", None)),
        "shadow_F_raw": _shadow_float_or_none(getattr(shadow_feat, "F_raw", None)),
        "current_F_red": _shadow_float_or_none(getattr(current_feat, "F_red", None)),
        "shadow_F_red": _shadow_float_or_none(getattr(shadow_feat, "F_red", None)),
        "current_h_hat": _shadow_float_or_none(getattr(current_feat, "h_hat", None)),
        "shadow_h_hat": _shadow_float_or_none(getattr(shadow_feat, "h_hat", None)),
        "current_h_eff": _shadow_float_or_none(getattr(current_feat, "h_eff", None)),
        "shadow_h_eff": _shadow_float_or_none(getattr(shadow_feat, "h_eff", None)),
        "current_actual_fallback_mode": _shadow_str_or_none(getattr(current_feat, "actual_fallback_mode", None)),
        "shadow_actual_fallback_mode": _shadow_str_or_none(getattr(shadow_feat, "actual_fallback_mode", None)),
        "bridge_debug": bridge_debug,
    }


def _attach_shadow_legacy_geometry_debug(
    current_feat: CandidateFeatures,
    payload: Mapping[str, Any],
) -> CandidateFeatures:
    compiled_backend = dict(current_feat.compiled_position_cost_backend or {})
    compiled_backend["shadow_legacy_geometry_debug"] = dict(payload)
    return CandidateFeatures(
        **{
            **current_feat.__dict__,
            "compiled_position_cost_backend": compiled_backend,
        }
    )


def _apply_phase3_selector_geometry_override(
    current_feat: CandidateFeatures,
    *,
    selector_geometry_mode: str,
    selector_score: float | None,
    source_tag: str,
) -> CandidateFeatures:
    mode_key = str(selector_geometry_mode).strip().lower()
    score_val = _shadow_float_or_none(selector_score)
    if score_val is None:
        return CandidateFeatures(
            **{
                **current_feat.__dict__,
                "selector_geometry_mode": str(mode_key),
            }
        )
    phase_score_components = dict(current_feat.phase_score_components)
    exact_score = _shadow_float_or_none(getattr(current_feat, "full_v2_score", None))
    if exact_score is not None:
        phase_score_components["phase3_exact_reduced_score"] = float(exact_score)
    phase_score_components["phase3_proxy_reduced_score"] = float(score_val)
    phase_score_components["selector_score"] = float(score_val)
    return CandidateFeatures(
        **{
            **current_feat.__dict__,
            "full_v2_score": float(score_val),
            "selector_score": float(score_val),
            "selector_geometry_mode": str(mode_key),
            "phase_score_components": phase_score_components,
            "actual_fallback_mode": f"{source_tag}::{current_feat.actual_fallback_mode}",
        }
    )


def _phase3_shadow_geometry_active_for_depth(
    *,
    enabled: bool,
    max_depth: int,
    depth_one_based: int,
) -> bool:
    if not bool(enabled):
        return False
    if int(max_depth) <= 0:
        return True
    return bool(int(depth_one_based) <= int(max_depth))


def _maybe_apply_phase3_proxy_selector_geometry(
    *,
    feat_full: CandidateFeatures,
    feat_candidate_base: CandidateFeatures,
    candidate_term: Any,
    psi_state_shadow: np.ndarray,
    selected_ops_shadow: Sequence[Any],
    theta_logical_shadow: np.ndarray,
    active_memory_shadow: Mapping[str, Any] | None,
    phase3_proxy_selector_geometry_enabled: bool,
    build_proxy_selector_feature: Callable[..., Any],
) -> CandidateFeatures:
    if not bool(phase3_proxy_selector_geometry_enabled):
        return feat_full
    try:
        proxy_feat = build_proxy_selector_feature(
            feat_candidate_base=feat_candidate_base,
            candidate_term=candidate_term,
            psi_state_shadow=np.asarray(psi_state_shadow, dtype=complex),
            selected_ops_shadow=list(selected_ops_shadow),
            theta_logical_shadow=np.asarray(theta_logical_shadow, dtype=float),
            active_memory_shadow=active_memory_shadow,
        )
    except Exception as exc:
        raise RuntimeError(
            "phase3_selector_geometry_mode='proxy_reduced' failed while computing the proxy-reduced selector score."
        ) from exc
    return _apply_phase3_selector_geometry_override(
        feat_full,
        selector_geometry_mode="proxy_reduced",
        selector_score=getattr(proxy_feat, "full_v2_score", None),
        source_tag="proxy_reduced",
    )


def _maybe_attach_shadow_legacy_geometry(
    *,
    feat_full: CandidateFeatures,
    feat_candidate_base: CandidateFeatures,
    candidate_term: Any,
    psi_state_shadow: np.ndarray,
    selected_ops_shadow: Sequence[Any],
    theta_logical_shadow: np.ndarray,
    active_memory_shadow: Mapping[str, Any] | None,
    depth_one_based: int,
    phase3_shadow_legacy_geometry_enabled: bool,
    phase3_shadow_legacy_max_depth: int,
    phase3_shadow_legacy_geometry_mode: str,
    phase3_proxy_selector_geometry_enabled: bool,
    build_proxy_selector_feature: Callable[..., Any],
    build_shadow_legacy_feature: Callable[..., Any],
) -> CandidateFeatures:
    if not _phase3_shadow_geometry_active_for_depth(
        enabled=phase3_shadow_legacy_geometry_enabled,
        max_depth=int(phase3_shadow_legacy_max_depth),
        depth_one_based=int(depth_one_based),
    ):
        return feat_full
    shadow_mode = str(phase3_shadow_legacy_geometry_mode)
    try:
        if bool(phase3_proxy_selector_geometry_enabled) and shadow_mode == "proxy_reduced":
            shadow_feat = build_proxy_selector_feature(
                feat_candidate_base=feat_candidate_base,
                candidate_term=candidate_term,
                psi_state_shadow=np.asarray(psi_state_shadow, dtype=complex),
                selected_ops_shadow=list(selected_ops_shadow),
                theta_logical_shadow=np.asarray(theta_logical_shadow, dtype=float),
                active_memory_shadow=active_memory_shadow,
            )
        else:
            shadow_feat = build_shadow_legacy_feature(
                feat_candidate_base=feat_candidate_base,
                candidate_term=candidate_term,
                psi_state_shadow=np.asarray(psi_state_shadow, dtype=complex),
                selected_ops_shadow=list(selected_ops_shadow),
                theta_logical_shadow=np.asarray(theta_logical_shadow, dtype=float),
                active_memory_shadow=active_memory_shadow,
            )
        payload = _shadow_legacy_geometry_debug_payload(
            current_feat=feat_full,
            shadow_feat=shadow_feat,
            shadow_mode=shadow_mode,
        )
        payload["depth_1based"] = int(depth_one_based)
        payload["candidate_label"] = str(feat_full.candidate_label)
        return _attach_shadow_legacy_geometry_debug(feat_full, payload)
    except Exception as exc:
        return _attach_shadow_legacy_geometry_debug(
            feat_full,
            {
                "status": "error",
                "mode": shadow_mode,
                "depth_1based": int(depth_one_based),
                "candidate_label": str(feat_full.candidate_label),
                "error_type": str(type(exc).__name__),
                "error": str(exc),
            },
        )


def _apply_runtime_split_family_aggregation_override(
    current_record: Mapping[str, Any],
    *,
    selector_score_key: str,
    family_selector_score: float | None,
    source_tag: str,
) -> dict[str, Any]:
    score_val = _shadow_float_or_none(family_selector_score)
    updated = dict(current_record)
    if score_val is None:
        return updated
    feat_obj = updated.get("feature")
    actual_child_selector_score = _shadow_float_or_none(updated.get(selector_score_key))
    if actual_child_selector_score is None and isinstance(feat_obj, CandidateFeatures):
        actual_child_selector_score = _shadow_float_or_none(
            getattr(feat_obj, selector_score_key, None)
        )
    if isinstance(feat_obj, CandidateFeatures):
        phase_score_components = dict(feat_obj.phase_score_components)
        if actual_child_selector_score is not None:
            phase_score_components["runtime_split_best_child_selector_score"] = float(
                actual_child_selector_score
            )
        phase_score_components["runtime_split_family_aggregate_selector_score"] = float(
            score_val
        )
        feat_updates: dict[str, Any] = {
            "selector_score": float(score_val),
            "phase_score_components": phase_score_components,
            "actual_fallback_mode": f"{source_tag}::{feat_obj.actual_fallback_mode}",
        }
        feat_updates[str(selector_score_key)] = float(score_val)
        updated_feat = CandidateFeatures(
            **{
                **feat_obj.__dict__,
                **feat_updates,
            }
        )
        updated["feature"] = updated_feat
        updated["selector_score"] = float(score_val)
        updated[str(selector_score_key)] = float(score_val)
    else:
        updated["selector_score"] = float(score_val)
        updated[str(selector_score_key)] = float(score_val)
    updated["runtime_split_best_child_selector_score"] = actual_child_selector_score
    updated["runtime_split_family_aggregate_selector_score"] = float(score_val)
    updated["runtime_split_family_aggregate_mode"] = str(source_tag)
    return updated
