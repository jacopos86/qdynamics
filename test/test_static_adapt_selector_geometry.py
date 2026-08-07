from __future__ import annotations

import pytest
import numpy as np

import pipelines.static_adapt.adapt_pipeline as hardcoded_adapt
import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
import pipelines.scaffold.hh_continuation_scoring as scoring
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt import selector_geometry


def _feature(**overrides: object) -> CandidateFeatures:
    base = dict(
        stage_name="phase3",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        g_signed=0.25,
        g_abs=0.25,
        g_lcb=0.25,
        sigma_hat=0.0,
        F_metric=1.0,
        metric_proxy=1.0,
        novelty=0.8,
        curvature_mode="current_curv",
        novelty_mode="current_novelty",
        refit_window_indices=[0],
        compiled_position_cost_proxy={},
        measurement_cache_stats={},
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        simple_score=0.25,
        score_version="test_v1",
        full_v2_score=0.25,
        selector_score=0.25,
        phase_score_components={"existing": 1.0},
        actual_fallback_mode="exact_reduced",
    )
    base.update(overrides)
    return CandidateFeatures(**base)


def test_selector_geometry_helpers_remain_available_through_adapt_pipeline_wrappers() -> None:
    helper_names = (
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
    )
    for name in helper_names:
        assert getattr(adapt_pipeline, name) is getattr(selector_geometry, name)
        assert getattr(hardcoded_adapt, name) is getattr(selector_geometry, name)


def test_selector_geometry_uses_current_scoring_candidate_features_class() -> None:
    assert scoring.CandidateFeatures is CandidateFeatures


def test_shadow_legacy_geometry_mode_normalization_and_cache_identity() -> None:
    assert selector_geometry._normalize_shadow_legacy_geometry_mode(None) == "off"
    assert selector_geometry._normalize_shadow_legacy_geometry_mode(" PROXY_REDUCED ") == "proxy_reduced"
    assert selector_geometry._load_shadow_legacy_geometry_bridge("off") is None
    with pytest.raises(RuntimeError, match="author-retired"):
        selector_geometry._load_shadow_legacy_geometry_bridge(
            "proxy_reduced"
        )

    adapt_pipeline._SHADOW_LEGACY_BRIDGE_CACHE.clear()
    selector_geometry._SHADOW_LEGACY_BRIDGE_CACHE["proxy_reduced"] = object()
    assert "proxy_reduced" in adapt_pipeline._SHADOW_LEGACY_BRIDGE_CACHE
    adapt_pipeline._SHADOW_LEGACY_BRIDGE_CACHE.clear()
    assert selector_geometry._SHADOW_LEGACY_BRIDGE_CACHE == {}

    with pytest.raises(ValueError, match="phase3_shadow_legacy_geometry_mode"):
        selector_geometry._normalize_shadow_legacy_geometry_mode("not-a-mode")


def test_apply_phase3_selector_geometry_override_preserves_none_score_behavior() -> None:
    feat = _feature(full_v2_score=0.25, selector_score=0.25, selector_geometry_mode="reduced")

    updated = selector_geometry._apply_phase3_selector_geometry_override(
        feat,
        selector_geometry_mode=" PROXY_REDUCED ",
        selector_score=None,
        source_tag="proxy_reduced",
    )

    assert updated is not feat
    assert updated.full_v2_score == 0.25
    assert updated.selector_score == 0.25
    assert updated.selector_geometry_mode == "proxy_reduced"
    assert updated.actual_fallback_mode == "exact_reduced"
    assert updated.phase_score_components == {"existing": 1.0}


def test_apply_phase3_selector_geometry_override_updates_score_components() -> None:
    feat = _feature(full_v2_score=0.25, selector_score=0.25, phase_score_components={"existing": 1.0})

    updated = selector_geometry._apply_phase3_selector_geometry_override(
        feat,
        selector_geometry_mode="proxy_reduced",
        selector_score=0.75,
        source_tag="proxy_reduced",
    )

    assert updated is not feat
    assert feat.full_v2_score == 0.25
    assert feat.selector_score == 0.25
    assert updated.full_v2_score == 0.75
    assert updated.selector_score == 0.75
    assert updated.selector_geometry_mode == "proxy_reduced"
    assert updated.actual_fallback_mode == "proxy_reduced::exact_reduced"
    assert updated.phase_score_components == {
        "existing": 1.0,
        "phase3_exact_reduced_score": 0.25,
        "phase3_proxy_reduced_score": 0.75,
        "selector_score": 0.75,
    }


def test_phase3_shadow_geometry_active_for_depth_contract() -> None:
    assert selector_geometry._phase3_shadow_geometry_active_for_depth(
        enabled=False,
        max_depth=0,
        depth_one_based=1,
    ) is False
    assert selector_geometry._phase3_shadow_geometry_active_for_depth(
        enabled=True,
        max_depth=0,
        depth_one_based=100,
    ) is True
    assert selector_geometry._phase3_shadow_geometry_active_for_depth(
        enabled=True,
        max_depth=2,
        depth_one_based=3,
    ) is False


def test_maybe_apply_phase3_proxy_selector_geometry_uses_injected_builder() -> None:
    feat = _feature(full_v2_score=0.25, selector_score=0.25, phase_score_components={"existing": 1.0})
    calls = []

    def build_proxy_selector_feature(**kwargs):
        calls.append(dict(kwargs))
        return _feature(full_v2_score=0.75)

    disabled = selector_geometry._maybe_apply_phase3_proxy_selector_geometry(
        feat_full=feat,
        feat_candidate_base=feat,
        candidate_term=object(),
        psi_state_shadow=np.asarray([1.0 + 0.0j]),
        selected_ops_shadow=[],
        theta_logical_shadow=np.asarray([0.0]),
        active_memory_shadow=None,
        phase3_proxy_selector_geometry_enabled=False,
        build_proxy_selector_feature=build_proxy_selector_feature,
    )

    assert disabled is feat
    assert calls == []

    updated = selector_geometry._maybe_apply_phase3_proxy_selector_geometry(
        feat_full=feat,
        feat_candidate_base=feat,
        candidate_term=object(),
        psi_state_shadow=np.asarray([1.0 + 0.0j]),
        selected_ops_shadow=[],
        theta_logical_shadow=np.asarray([0.0]),
        active_memory_shadow={"source": "unit"},
        phase3_proxy_selector_geometry_enabled=True,
        build_proxy_selector_feature=build_proxy_selector_feature,
    )

    assert updated.full_v2_score == 0.75
    assert updated.selector_score == 0.75
    assert updated.selector_geometry_mode == "proxy_reduced"
    assert calls and calls[-1]["active_memory_shadow"] == {"source": "unit"}


def test_maybe_apply_phase3_proxy_selector_geometry_wraps_builder_error() -> None:
    feat = _feature()

    def build_proxy_selector_feature(**kwargs):
        raise ValueError("bad proxy")

    with pytest.raises(RuntimeError, match="proxy-reduced selector score"):
        selector_geometry._maybe_apply_phase3_proxy_selector_geometry(
            feat_full=feat,
            feat_candidate_base=feat,
            candidate_term=object(),
            psi_state_shadow=np.asarray([1.0 + 0.0j]),
            selected_ops_shadow=[],
            theta_logical_shadow=np.asarray([0.0]),
            active_memory_shadow=None,
            phase3_proxy_selector_geometry_enabled=True,
            build_proxy_selector_feature=build_proxy_selector_feature,
        )


def test_maybe_attach_shadow_legacy_geometry_attaches_debug_or_error_payload() -> None:
    feat = _feature(
        full_v2_score=0.25,
        phase2_raw_score=0.2,
        compiled_position_cost_backend={"existing": True},
    )
    shadow = _feature(full_v2_score=0.5, phase2_raw_score=0.4)
    calls = []

    def build_proxy_selector_feature(**kwargs):
        calls.append("proxy")
        return shadow

    def build_shadow_legacy_feature(**kwargs):
        calls.append("shadow")
        return shadow

    disabled = selector_geometry._maybe_attach_shadow_legacy_geometry(
        feat_full=feat,
        feat_candidate_base=feat,
        candidate_term=object(),
        psi_state_shadow=np.asarray([1.0 + 0.0j]),
        selected_ops_shadow=[],
        theta_logical_shadow=np.asarray([0.0]),
        active_memory_shadow=None,
        depth_one_based=3,
        phase3_shadow_legacy_geometry_enabled=False,
        phase3_shadow_legacy_max_depth=0,
        phase3_shadow_legacy_geometry_mode="proxy_reduced",
        phase3_proxy_selector_geometry_enabled=True,
        build_proxy_selector_feature=build_proxy_selector_feature,
        build_shadow_legacy_feature=build_shadow_legacy_feature,
    )

    assert disabled is feat
    assert calls == []

    updated = selector_geometry._maybe_attach_shadow_legacy_geometry(
        feat_full=feat,
        feat_candidate_base=feat,
        candidate_term=object(),
        psi_state_shadow=np.asarray([1.0 + 0.0j]),
        selected_ops_shadow=[],
        theta_logical_shadow=np.asarray([0.0]),
        active_memory_shadow=None,
        depth_one_based=3,
        phase3_shadow_legacy_geometry_enabled=True,
        phase3_shadow_legacy_max_depth=0,
        phase3_shadow_legacy_geometry_mode="proxy_reduced",
        phase3_proxy_selector_geometry_enabled=True,
        build_proxy_selector_feature=build_proxy_selector_feature,
        build_shadow_legacy_feature=build_shadow_legacy_feature,
    )

    debug = updated.compiled_position_cost_backend["shadow_legacy_geometry_debug"]
    assert calls == ["proxy"]
    assert debug["status"] == "ok"
    assert debug["mode"] == "proxy_reduced"
    assert debug["depth_1based"] == 3
    assert debug["candidate_label"] == "x"
    assert debug["shadow_minus_current_full_v2_score"] == pytest.approx(0.25)

    def failing_shadow_feature(**kwargs):
        raise RuntimeError("shadow failed")

    errored = selector_geometry._maybe_attach_shadow_legacy_geometry(
        feat_full=feat,
        feat_candidate_base=feat,
        candidate_term=object(),
        psi_state_shadow=np.asarray([1.0 + 0.0j]),
        selected_ops_shadow=[],
        theta_logical_shadow=np.asarray([0.0]),
        active_memory_shadow=None,
        depth_one_based=2,
        phase3_shadow_legacy_geometry_enabled=True,
        phase3_shadow_legacy_max_depth=0,
        phase3_shadow_legacy_geometry_mode="exact_reduced",
        phase3_proxy_selector_geometry_enabled=False,
        build_proxy_selector_feature=build_proxy_selector_feature,
        build_shadow_legacy_feature=failing_shadow_feature,
    )
    error_debug = errored.compiled_position_cost_backend["shadow_legacy_geometry_debug"]
    assert error_debug["status"] == "error"
    assert error_debug["mode"] == "exact_reduced"
    assert error_debug["error_type"] == "RuntimeError"


def test_runtime_split_family_aggregation_override_copies_record_and_feature() -> None:
    feat = _feature(selector_score=0.4, phase3_primary_score=0.4, phase_score_components={"existing": 1.0})
    record = {
        "feature": feat,
        "selector_score": 0.4,
        "phase3_primary_score": 0.4,
        "candidate_label": "x",
    }

    updated = selector_geometry._apply_runtime_split_family_aggregation_override(
        record,
        selector_score_key="phase3_primary_score",
        family_selector_score=1.25,
        source_tag="runtime_split_parent_family_sum_top2",
    )

    assert updated is not record
    assert updated["feature"] is not feat
    assert record["selector_score"] == 0.4
    assert record["phase3_primary_score"] == 0.4
    assert feat.selector_score == 0.4
    assert feat.phase3_primary_score == 0.4

    updated_feat = updated["feature"]
    assert isinstance(updated_feat, CandidateFeatures)
    assert updated["selector_score"] == 1.25
    assert updated["phase3_primary_score"] == 1.25
    assert updated["runtime_split_best_child_selector_score"] == 0.4
    assert updated["runtime_split_family_aggregate_selector_score"] == 1.25
    assert updated["runtime_split_family_aggregate_mode"] == "runtime_split_parent_family_sum_top2"
    assert updated_feat.selector_score == 1.25
    assert updated_feat.phase3_primary_score == 1.25
    assert updated_feat.actual_fallback_mode == "runtime_split_parent_family_sum_top2::exact_reduced"
    assert updated_feat.phase_score_components == {
        "existing": 1.0,
        "runtime_split_best_child_selector_score": 0.4,
        "runtime_split_family_aggregate_selector_score": 1.25,
    }


def test_runtime_split_family_aggregation_override_none_score_returns_record_copy() -> None:
    feat = _feature()
    record = {"feature": feat, "selector_score": 0.4}

    updated = selector_geometry._apply_runtime_split_family_aggregation_override(
        record,
        selector_score_key="phase3_primary_score",
        family_selector_score=None,
        source_tag="runtime_split_parent_family_sum_top2",
    )

    assert updated == record
    assert updated is not record
    assert updated["feature"] is feat
