from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.scaffold.hh_continuation_scoring as scoring_mod
from pipelines.scaffold.hh_continuation_scoring import (
    CompatibilityPenaltyOracle,
    FullScoreConfig,
    GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1,
    MeasurementCacheAudit,
    measurement_group_specs_for_term,
    normalize_hardware_cost_feature_family,
    rescore_hardware_cost_family,
    Phase2CurvatureOracle,
    OrderedInsertionGeometryOracle,
    Phase1CompileCostOracle,
    SimpleScoreConfig,
    build_full_candidate_features,
    build_candidate_features,
    ceo_commuting_batch_select,
    compatibility_penalty,
    full_v2_score,
    lifetime_weight_components,
    overlap_orthogonal_batch_select,
    phase0_raw_gradient_pilot_components,
    phase3_canonical_score_components,
    phase3_plateau_novelty_cost_score_components,
    attach_route_c_plateau_acquisition_payload,
    phase_shortlist_records,
    greedy_reduced_plane_batch_proposals,
    hardware_cost_ansatz_entry_denominators,
    hardware_cost_candidate_record_denominators,
    reduced_plane_batch_select,
    select_phase2_batch_record_proposals,
    select_phase2_batch_records,
    measurement_group_keys_for_term,
    raw_f_metric_from_state,
    remaining_evaluations_proxy,
    shortlist_records,
    trust_region_drop,
)
from pipelines.scaffold.hh_continuation_generators import build_generator_metadata
from pipelines.scaffold.hh_continuation_types import CompileCostEstimate, MeasurementGroupSpec
from pipelines.scaffold.hh_continuation_symmetry import build_symmetry_spec
from pipelines.static_adapt.route_a_schur_selector import (
    BATCH_GEOMETRY_DIAGONAL_HESSIAN_DIAGNOSTIC_V1,
    BATCH_GEOMETRY_FULL_RESIDUAL_GRAM_HESSIAN_V1,
    BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1,
    BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1,
    BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
    ROUTE_A_ADDITIVITY_OFF,
    ROUTE_A_ADDITIVITY_SOFT_PENALTY_V1,
    ROUTE_A_ACTIVE_CONTEXT_TAIL_WINDOW_V1,
    ROUTE_A_SCHUR_GREEDY_REDUCED_PLANE,
    RouteAJointResponseEvaluator,
    RouteASchurSelectorConfig,
    select_route_a_schur_proposals,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
    PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1,
    PHASE2_DIRECTIONAL_CURVATURE_PROVENANCE_SCHEMA,
    PHASE2_DIRECTIONAL_CURVATURE_RECEIPT_SCHEMA,
)
from pipelines.scaffold.hh_continuation_scoring import (
    Phase2CurvatureConstructionError,
    _validated_phase2_directional_curvature,
    phase1_trust_region_gain,
    phase2_raw_geometry_score,
    phase3_cheap_ratio_v1,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm


def test_phase2_novelty_oracle_is_identity_only_geometry_compatibility() -> None:
    assert (
        scoring_mod.Phase2NoveltyOracle
        is OrderedInsertionGeometryOracle
    )


def test_canonical_score_configs_share_normalized_hardware_cost_weights() -> None:
    expected = {
        "2q": 0.20,
        "d": 0.20,
        "1q": 0.05,
        "theta": 0.05,
        "shot": 0.15,
    }

    simple_weights, simple_source = scoring_mod.resolve_hardware_cost_lambdas(
        SimpleScoreConfig()
    )
    full_weights, full_source = scoring_mod.resolve_hardware_cost_lambdas(
        FullScoreConfig()
    )

    assert simple_weights == pytest.approx(expected)
    assert full_weights == pytest.approx(expected)
    assert simple_source == "explicit_lambda_fields_v1"
    assert full_source == "explicit_lambda_fields_v1"


def test_simple_v1_prefers_higher_gradient_with_equal_costs() -> None:
    cfg = SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=2, append_position=2, refit_active_count=1)
    mstats = meas.estimate(["x"])
    feat_a = build_candidate_features(
        stage_name="core",
        candidate_label="a",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=2,
        append_position=2,
        positions_considered=[2],
        gradient_signed=0.4,
        metric_proxy=0.4,
        sigma_hat=0.0,
        refit_window_indices=[2],
        compile_cost=cost,
        measurement_stats=mstats,
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    feat_b = build_candidate_features(
        stage_name="core",
        candidate_label="b",
        candidate_family="core",
        candidate_pool_index=1,
        position_id=2,
        append_position=2,
        positions_considered=[2],
        gradient_signed=0.2,
        metric_proxy=0.2,
        sigma_hat=0.0,
        refit_window_indices=[2],
        compile_cost=cost,
        measurement_stats=mstats,
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert float(feat_a.simple_score or 0.0) > float(feat_b.simple_score or 0.0)


def test_phase_shortlist_frontier_ratio_one_is_nonbinding() -> None:
    records = [
        {"candidate_label": "a", "full_v2_score": 10.0, "simple_score": 10.0, "candidate_pool_index": 0, "position_id": 0},
        {"candidate_label": "b", "full_v2_score": 8.6, "simple_score": 8.6, "candidate_pool_index": 1, "position_id": 0},
        {"candidate_label": "c", "full_v2_score": 7.0, "simple_score": 7.0, "candidate_pool_index": 2, "position_id": 0},
    ]
    shortlisted = phase_shortlist_records(
        records,
        score_key="full_v2_score",
        threshold=float("-inf"),
        cap=3,
        frontier_ratio=1.0,
        tie_break_score_key="simple_score",
    )
    assert [rec["candidate_label"] for rec in shortlisted] == ["a", "b", "c"]


def test_phase_shortlist_frontier_ratio_below_one_can_cut_shell() -> None:
    records = [
        {"candidate_label": "a", "full_v2_score": 10.0, "simple_score": 10.0, "candidate_pool_index": 0, "position_id": 0},
        {"candidate_label": "b", "full_v2_score": 8.6, "simple_score": 8.6, "candidate_pool_index": 1, "position_id": 0},
        {"candidate_label": "c", "full_v2_score": 7.0, "simple_score": 7.0, "candidate_pool_index": 2, "position_id": 0},
    ]
    shortlisted = phase_shortlist_records(
        records,
        score_key="full_v2_score",
        threshold=float("-inf"),
        cap=3,
        frontier_ratio=0.85,
        tie_break_score_key="simple_score",
    )
    assert [rec["candidate_label"] for rec in shortlisted] == ["a", "b"]


def test_stage_gate_blocks_score() -> None:
    cfg = SimpleScoreConfig()
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=0, append_position=1, refit_active_count=1)
    mstats = meas.estimate(["x"])
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="resid",
        candidate_family="residual",
        candidate_pool_index=0,
        position_id=0,
        append_position=1,
        positions_considered=[0, 1],
        gradient_signed=1.0,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0, 1],
        compile_cost=cost,
        measurement_stats=mstats,
        leakage_penalty=0.0,
        stage_gate_open=False,
        leakage_gate_open=True,
        trough_probe_triggered=True,
        trough_detected=True,
        cfg=cfg,
    )
    assert feat.simple_score == float("-inf")


def test_backend_compile_cost_replaces_proxy_term_in_simple_score() -> None:
    cfg = SimpleScoreConfig(lambda_F=0.0, lambda_compile=1.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    meas = MeasurementCacheAudit()
    cost = CompileCostEstimate(
        new_pauli_actions=3.0,
        new_rotation_steps=2.0,
        position_shift_span=1.0,
        refit_active_count=4.0,
        proxy_total=99.0,
        cx_proxy_total=11.0,
        sq_proxy_total=22.0,
        gate_proxy_total=33.0,
        max_pauli_weight=2.0,
        source_mode="backend_transpile_v1",
        penalty_total=7.5,
        depth_surrogate=5.0,
        compile_gate_open=True,
        selected_backend_name="FakeNighthawk",
        raw_delta_compiled_depth_2q=3.0,
        delta_compiled_depth_2q=3.0,
        proxy_baseline={
            "new_pauli_actions": 3.0,
            "new_rotation_steps": 2.0,
            "position_shift_span": 1.0,
            "refit_active_count": 4.0,
            "proxy_total": 99.0,
            "cx_proxy_total": 11.0,
            "sq_proxy_total": 22.0,
            "gate_proxy_total": 33.0,
            "max_pauli_weight": 2.0,
        },
        selected_backend_row={"transpile_backend": "FakeNighthawk", "compiled_count_2q": 18},
    )
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="backend",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=1,
        append_position=1,
        positions_considered=[1],
        gradient_signed=0.0,
        metric_proxy=0.0,
        sigma_hat=0.0,
        refit_window_indices=[0, 1],
        compile_cost=cost,
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert feat.compile_cost_source == "backend_transpile_v1"
    assert float(feat.compile_cost_total) == pytest.approx(7.5)
    assert float(feat.simple_score or 0.0) == pytest.approx(0.0)
    assert feat.compiled_position_cost_proxy["proxy_total"] == pytest.approx(99.0)
    assert feat.compiled_position_cost_backend is not None
    assert feat.compiled_position_cost_backend["selected_backend_name"] == "FakeNighthawk"
    assert feat.compiled_position_cost_backend["raw_delta_compiled_depth_2q"] == pytest.approx(3.0)
    assert feat.compiled_position_cost_backend["delta_compiled_depth_2q"] == pytest.approx(3.0)


def test_backend_compile_gate_closed_blocks_simple_and_full_scores() -> None:
    cfg = SimpleScoreConfig()
    full_cfg = FullScoreConfig()
    meas = MeasurementCacheAudit()
    cost = CompileCostEstimate(
        new_pauli_actions=0.0,
        new_rotation_steps=0.0,
        position_shift_span=0.0,
        refit_active_count=1.0,
        proxy_total=1.0,
        source_mode="backend_transpile_v1",
        penalty_total=float("inf"),
        depth_surrogate=float("inf"),
        compile_gate_open=False,
        failure_reason="all_targets_failed",
    )
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="blocked",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=1.0,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert feat.simple_score == float("-inf")
    score, fallback = full_v2_score(feat, full_cfg)
    assert score == float("-inf")
    assert fallback == "compile_gate_closed"


def test_simple_v1_uses_g_hw_lcb_for_ranking() -> None:
    cfg = SimpleScoreConfig(lambda_F=0.0, lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=10.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1)
    mstats = meas.estimate(["x"])
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="a",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.4,
        metric_proxy=1.0,
        sigma_hat=0.03,
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=mstats,
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert float(feat.g_lcb) == pytest.approx(0.1)
    assert float(feat.g_lcb_legacy_shot) == pytest.approx(0.1)
    assert float(feat.g_hw_lcb) == pytest.approx(0.1)
    assert float(feat.epsilon_g_shot) == pytest.approx(0.3)
    assert float(feat.epsilon_g_res) == pytest.approx(0.3)
    assert str(feat.hardware_resolution_mode) == "ideal"
    assert float(feat.simple_score or 0.0) == pytest.approx(0.025)


def _zero_gain_plateau_feature():
    cfg = SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1)
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="plateau_candidate",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.0,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    return type(feat)(**{**feat.__dict__, "F_red": 1.0, "h_eff": 1.0, "novelty": 0.1})


def test_plateau_novelty_cost_score_survives_zero_canonical_gain() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
    )
    feat = _zero_gain_plateau_feature()

    canonical = phase3_canonical_score_components(feat, cfg)
    plateau = phase3_plateau_novelty_cost_score_components(
        feat,
        cfg,
        plateau_novelty=0.8,
        context_indices=[0],
        dormant_indices=[2, 3],
    )

    assert canonical["phase3_primary_score"] == pytest.approx(0.0)
    assert canonical["block_reason"] == "nonpositive_gradient"
    assert plateau["eligible"] is True
    assert plateau["phase3_plateau_acquisition_score"] == pytest.approx(0.8)
    assert plateau["phase3_plateau_score_formula"] == "N3_plat / (1 + K3)"
    assert plateau["context_indices"] == [0]
    assert plateau["dormant_indices"] == [2, 3]


def test_plateau_log_volume_score_uses_raw_qim_residual() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
    )
    feat = _zero_gain_plateau_feature()

    plateau = phase3_plateau_novelty_cost_score_components(
        feat,
        cfg,
        plateau_novelty=0.1,
        acquisition_score="log_volume_v1",
        F_raw=2.0,
        Q_window=np.array([[1.0]]),
        q_window=np.array([1.0]),
        lambda_vol=1.0,
        sigma_min=0.25,
        nu_min=0.25,
        volume_min=0.1,
    )

    assert plateau["eligible"] is True
    assert plateau["phase3_plateau_acquisition_score_kind"] == "log_volume_v1"
    assert plateau["phase3_plateau_sigma_perp"] == pytest.approx(1.0)
    assert plateau["phase3_plateau_fractional_residual"] == pytest.approx(0.5)
    assert plateau["phase3_plateau_sigma_perp_lambda"] == pytest.approx(1.5)
    assert plateau["phase3_plateau_log_volume_gain"] == pytest.approx(np.log(2.5))
    assert plateau["phase3_plateau_acquisition_score"] == pytest.approx(np.log(2.5))
    assert plateau["phase3_plateau_score_formula"] == (
        "log(1 + sigma_perp_lambda / lambda_vol) / (1 + K3)"
    )


def test_plateau_log_volume_score_honors_residual_gates() -> None:
    cfg = FullScoreConfig(wD=0.0, wG=0.0, wC=0.0, wP=0.0, wc=0.0)
    feat = _zero_gain_plateau_feature()

    blocked = phase3_plateau_novelty_cost_score_components(
        feat,
        cfg,
        plateau_novelty=1.0,
        acquisition_score="log_volume_v1",
        F_raw=1.0,
        Q_window=np.array([[1.0]]),
        q_window=np.array([1.0]),
        lambda_vol=1e-6,
        sigma_min=1e-4,
    )

    assert blocked["eligible"] is False
    assert blocked["block_reason"] == "phase3_plateau_sigma_below_min"


def test_plateau_novelty_cost_score_uses_cost_denominator_not_gain() -> None:
    cfg = FullScoreConfig(
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
    )
    feat = _zero_gain_plateau_feature()
    cheap = type(feat)(**{**feat.__dict__, "c_bar_2q": 0.0})
    costly = type(feat)(**{**feat.__dict__, "c_bar_2q": 3.0})

    cheap_score = phase3_plateau_novelty_cost_score_components(cheap, cfg, plateau_novelty=0.6)
    costly_score = phase3_plateau_novelty_cost_score_components(costly, cfg, plateau_novelty=0.6)
    low_novelty_score = phase3_plateau_novelty_cost_score_components(cheap, cfg, plateau_novelty=0.2)

    assert cheap_score["phase3_plateau_acquisition_score"] == pytest.approx(0.6)
    assert costly_score["denominator_1_plus_K3"] == pytest.approx(4.0)
    assert costly_score["phase3_plateau_acquisition_score"] == pytest.approx(0.15)
    assert low_novelty_score["phase3_plateau_acquisition_score"] < cheap_score["phase3_plateau_acquisition_score"]


def test_plateau_payload_attachment_is_side_channel_only() -> None:
    cfg = FullScoreConfig(wD=0.0, wG=0.0, wC=0.0, wP=0.0, wc=0.0)
    feat = _zero_gain_plateau_feature()
    payload = phase3_plateau_novelty_cost_score_components(feat, cfg, plateau_novelty=0.5)
    updated = attach_route_c_plateau_acquisition_payload(feat, payload)

    assert updated.route_c_plateau_acquisition["phase3_plateau_acquisition_score"] == pytest.approx(0.5)
    assert updated.phase_score_components["phase3_plateau_acquisition_score"] == pytest.approx(0.5)
    assert updated.full_v2_score == feat.full_v2_score
    assert updated.selector_score == feat.selector_score
    assert phase3_canonical_score_components(updated, cfg)["block_reason"] == "nonpositive_gradient"


def test_plateau_score_blocks_real_gates_and_duplicates() -> None:
    cfg = FullScoreConfig(wD=0.0, wG=0.0, wC=0.0, wP=0.0, wc=0.0)
    feat = _zero_gain_plateau_feature()
    compile_blocked = type(feat)(**{**feat.__dict__, "compile_gate_open": False})

    assert phase3_plateau_novelty_cost_score_components(
        compile_blocked,
        cfg,
        plateau_novelty=1.0,
    )["block_reason"] == "compile_gate_closed"
    assert phase3_plateau_novelty_cost_score_components(
        feat,
        cfg,
        plateau_novelty=1.0,
        duplicate_blocked=True,
    )["block_reason"] == "exact_candidate_position_duplicate"


def test_simple_v1_metric_floor_scores_zero_metric_records() -> None:
    cfg = SimpleScoreConfig(
        lambda_F=1.0,
        lambda_compile=0.0,
        lambda_measure=0.0,
        lambda_leak=0.0,
        z_alpha=0.0,
        metric_floor=1.0,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1)
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="zero_metric",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.4,
        metric_proxy=0.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    expected_delta = trust_region_drop(float(feat.g_hw_lcb), 1.0, 1.0, 0.25)
    assert expected_delta > 0.0
    assert float(feat.simple_score or 0.0) == pytest.approx(expected_delta)
    assert float(feat.phase_score_components["phase1_DeltaE1_TR_hw"]) == pytest.approx(expected_delta)
    assert float(feat.cheap_benefit_proxy or 0.0) == pytest.approx(expected_delta)


def test_phase1_trust_region_score_responds_to_rho() -> None:
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1)

    def _score(rho: float):
        return build_candidate_features(
            stage_name="core",
            candidate_label=f"rho_{rho}",
            candidate_family="core",
            candidate_pool_index=0,
            position_id=0,
            append_position=0,
            positions_considered=[0],
            gradient_signed=1.0,
            metric_proxy=1.0,
            sigma_hat=0.0,
            refit_window_indices=[0],
            compile_cost=cost,
            measurement_stats=meas.estimate(["x"]),
            leakage_penalty=0.0,
            stage_gate_open=True,
            leakage_gate_open=True,
            trough_probe_triggered=False,
            trough_detected=False,
            cfg=SimpleScoreConfig(
                lambda_F=1.0,
                lambda_compile=0.0,
                lambda_measure=0.0,
                lambda_leak=0.0,
                rho=rho,
                phase1_score_mode="trust_region_v1",
            ),
        )

    low = _score(0.1)
    high = _score(0.5)
    assert low.phase1_score_mode == "trust_region_v1"
    assert high.phase1_score_mode == "trust_region_v1"
    assert float(low.phase1_active_score or 0.0) == pytest.approx(trust_region_drop(1.0, 1.0, 1.0, 0.1))
    assert float(high.phase1_active_score or 0.0) == pytest.approx(trust_region_drop(1.0, 1.0, 1.0, 0.5))
    assert float(high.phase1_active_score or 0.0) > float(low.phase1_active_score or 0.0)
    assert float(low.phase1_legacy_simple_score or 0.0) == pytest.approx(1.0)
    assert float(high.phase1_legacy_simple_score or 0.0) == pytest.approx(1.0)


def test_phase1_legacy_simple_mode_ignores_rho() -> None:
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1)

    def _score(rho: float):
        return build_candidate_features(
            stage_name="core",
            candidate_label=f"legacy_{rho}",
            candidate_family="core",
            candidate_pool_index=0,
            position_id=0,
            append_position=0,
            positions_considered=[0],
            gradient_signed=0.4,
            metric_proxy=1.0,
            sigma_hat=0.0,
            refit_window_indices=[0],
            compile_cost=cost,
            measurement_stats=meas.estimate(["x"]),
            leakage_penalty=0.0,
            stage_gate_open=True,
            leakage_gate_open=True,
            trough_probe_triggered=False,
            trough_detected=False,
            cfg=SimpleScoreConfig(
                lambda_F=1.0,
                lambda_compile=0.0,
                lambda_measure=0.0,
                lambda_leak=0.0,
                rho=rho,
                phase1_score_mode="legacy_simple_v1",
            ),
        )

    low = _score(0.1)
    high = _score(0.9)
    assert low.phase1_score_mode == "legacy_simple_v1"
    assert high.phase1_score_mode == "legacy_simple_v1"
    assert float(low.phase1_active_score or 0.0) == pytest.approx(0.4)
    assert float(high.phase1_active_score or 0.0) == pytest.approx(0.4)
    assert float(low.simple_score or 0.0) == pytest.approx(0.4)
    assert float(high.simple_score or 0.0) == pytest.approx(0.4)
    assert float(low.phase1_trust_region_score or 0.0) != pytest.approx(
        float(high.phase1_trust_region_score or 0.0)
    )


def test_manual_gradient_floor_reduces_hw_lcb_but_preserves_legacy_shot_lcb() -> None:
    cfg = SimpleScoreConfig(
        lambda_F=0.0,
        lambda_compile=0.0,
        lambda_measure=0.0,
        lambda_leak=0.0,
        z_alpha=10.0,
        hardware_resolution_mode="manual",
        manual_b_g_hw=0.2,
        manual_b_g_drift=0.0,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1)
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="a",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.4,
        metric_proxy=1.0,
        sigma_hat=0.03,
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert float(feat.g_lcb) == pytest.approx(0.1)
    assert float(feat.g_lcb_legacy_shot) == pytest.approx(0.1)
    assert float(feat.b_g_hw) == pytest.approx(0.2)
    assert float(feat.epsilon_g_res) == pytest.approx(0.5)
    assert float(feat.g_hw_lcb) == pytest.approx(0.0)
    assert float(feat.simple_score or 0.0) == pytest.approx(0.0)
    assert str(feat.hardware_resolution_source) == "manual_scalar_floors"


def test_phase0_raw_gradient_pilot_uses_upper_confidence_manual_floors() -> None:
    comps = phase0_raw_gradient_pilot_components(
        gradient_signed=-0.4,
        sigma_hat=0.03,
        alpha0=0.1,
        z_alpha=10.0,
        hardware_resolution_mode="manual",
        manual_b_g_hw=0.2,
        manual_b_g_drift=0.05,
    )

    assert comps["phase0_raw_gradient_abs"] == pytest.approx(0.4)
    assert comps["phase0_epsilon_g_res"] == pytest.approx(0.55)
    assert comps["phase0_g_upper_hw"] == pytest.approx(0.95)
    assert comps["phase0_delta_e_upper_hw"] == pytest.approx(0.095)
    assert comps["phase0_hardware_resolution_source"] == "manual_scalar_floors"
    assert comps["phase0_sigma_hat_available"] is True


def test_phase0_ideal_zero_floors_and_sigma_unavailable_telemetry() -> None:
    comps = phase0_raw_gradient_pilot_components(
        gradient_signed=0.25,
        sigma_hat=None,
        alpha0=0.2,
        z_alpha=2.0,
        hardware_resolution_mode="ideal",
        manual_b_g_hw=0.0,
        manual_b_g_drift=0.0,
    )

    assert comps["phase0_sigma_hat"] == pytest.approx(0.0)
    assert comps["phase0_sigma_hat_available"] is False
    assert comps["phase0_epsilon_g_res"] == pytest.approx(0.0)
    assert comps["phase0_g_upper_hw"] == pytest.approx(0.25)
    assert comps["phase0_hardware_resolution_source"] == "ideal_zero_floors"
    with pytest.raises(ValueError, match="ideal hardware_resolution_mode requires zero"):
        phase0_raw_gradient_pilot_components(
            gradient_signed=0.25,
            sigma_hat=0.0,
            alpha0=0.2,
            z_alpha=2.0,
            hardware_resolution_mode="ideal",
            manual_b_g_hw=0.01,
            manual_b_g_drift=0.0,
        )


def test_measurement_cache_reuse_accounting() -> None:
    cache = MeasurementCacheAudit(nominal_shots_per_group=10)
    first = cache.estimate(["a", "b"])
    assert first.groups_new == 2
    assert first.shots_new == pytest.approx(20.0)
    cache.commit(["a", "b"])
    second = cache.estimate(["a", "b", "c"])
    assert second.groups_reused == 2
    assert second.groups_new == 1
    assert second.shots_reused == pytest.approx(20.0)
    assert second.shots_new == pytest.approx(10.0)
    summary = cache.summary()
    assert str(summary["plan_version"]) == "phase1_qwc_basis_cover_reuse"


def test_measurement_cache_reuses_more_specific_seen_basis_keys() -> None:
    cache = MeasurementCacheAudit(nominal_shots_per_group=10)
    cache.commit(["xz"])
    reused = cache.estimate(["ez"])
    assert reused.groups_reused == 1
    assert reused.groups_new == 0


def test_measurement_group_keys_for_term_merges_qwc_compatible_labels() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xe", pc=1.0),
            PauliTerm(2, ps="xz", pc=1.0),
            PauliTerm(2, ps="ez", pc=1.0),
        ],
    )
    term = type("_DummyAnsatzTerm", (), {"label": "macro", "polynomial": poly})()
    assert measurement_group_keys_for_term(term) == ["xz"]


def test_measurement_group_specs_accumulate_qwc_coeff_l2_and_fixed_precision_shot_proxy() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xz", pc=3.0),
            PauliTerm(2, ps="ez", pc=4.0),
        ],
    )
    term = type("_DummyAnsatzTerm", (), {"label": "macro", "polynomial": poly})()
    specs = measurement_group_specs_for_term(term)
    assert [spec.group_key for spec in specs] == ["xz"]
    assert specs[0].coeff_l2 == pytest.approx(5.0)
    assert specs[0].term_count == 2

    cache = MeasurementCacheAudit(sigma_star=2.0)
    first = cache.estimate(specs)
    assert first.groups_new == 1
    assert first.new_group_coeff_l2_sum == pytest.approx(5.0)
    assert first.shots_new == pytest.approx(1.0)
    assert first.shot_cost_proxy == pytest.approx(6.25)

    cache.commit(["xz"])
    reused = cache.estimate(specs)
    assert reused.groups_reused == 1
    assert reused.groups_new == 0
    assert reused.shot_cost_proxy == pytest.approx(0.0)


def _term(label: str) -> object:
    return type(
        "_DummyAnsatzTerm",
        (),
        {
            "label": str(label),
            "polynomial": PauliPolynomial("JW", [PauliTerm(len(str(label)), ps=str(label), pc=1.0)]),
        },
    )()


def _poly(label: str) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(len(str(label)), ps=str(label), pc=1.0)])


def _scaffold_context(
    *,
    psi_state: np.ndarray,
    selected_ops: list[object],
    theta: list[float],
    refit_window_indices: list[int],
    h_label: str = "z",
) -> tuple[object, object]:
    h_compiled = compile_polynomial_action(_poly(h_label), pauli_action_cache={})
    hpsi_state = apply_compiled_polynomial(np.asarray(psi_state, dtype=complex), h_compiled)
    scaffold_context = OrderedInsertionGeometryOracle().prepare_scaffold_context(
        selected_ops=list(selected_ops),
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(psi_state, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        h_compiled=h_compiled,
        hpsi_state=np.asarray(hpsi_state, dtype=complex),
        refit_window_indices=list(refit_window_indices),
        pauli_action_cache={},
    )
    return scaffold_context, h_compiled


def _phase2_raw_base_feature(*, sigma_hat: float = 1.0):
    cost = Phase1CompileCostOracle().estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=1,
    )
    mstats = MeasurementCacheAudit().estimate(["x"])
    return build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=2.0,
        metric_proxy=2.0,
        sigma_hat=float(sigma_hat),
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=mstats,
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(z_alpha=0.0),
        cheap_score_cfg=FullScoreConfig(wD=0.0, wG=0.0, wC=0.0, wP=0.0, wc=0.0),
    )


def test_phase2_ordinary_score_ignores_gram_projection_and_scores_gain_over_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feat = _phase2_raw_base_feature()
    cfg = FullScoreConfig(
        metric_floor=1e-12,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
    )

    def _unexpected_novelty_solve(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("ordinary Phase-II novelty solve must be skipped")

    monkeypatch.setattr(scoring_mod, "_regularized_solve", _unexpected_novelty_solve)
    raw_a = scoring_mod.phase2_raw_geometry_score(
        feat,
        F_raw=2.0,
        h_raw=0.0,
        q_window=[1.0, 1.0],
        Q_window=np.eye(2),
        cfg=cfg,
    )
    raw_b = scoring_mod.phase2_raw_geometry_score(
        feat,
        F_raw=2.0,
        h_raw=0.0,
        q_window=[],
        Q_window=np.ones((3, 1)),
        cfg=cfg,
    )

    assert raw_a["phase2_raw_novelty"] is None
    assert raw_a["phase2_measured_novelty"] is None
    assert raw_a["phase2_novelty_multiplier"] is None
    assert raw_a["phase2_novelty_applied"] is False
    assert raw_a["phase2_novelty_status"] == (
        "not_computed_for_ordinary_scoring"
    )
    assert raw_a["phase2_novelty_query_charge"] == 0
    assert raw_a["phase2_novelty_classical_solve_count"] == 0
    assert raw_a["phase2_raw_score_formula"] == (
        "DeltaE_TR_raw / (1 + K2)"
    )
    assert raw_a["phase2_raw_score"] == pytest.approx(
        raw_a["phase2_raw_trust_gain"]
        / raw_a["phase2_burden_total"]
    )
    assert raw_b["phase2_raw_score"] == pytest.approx(
        raw_a["phase2_raw_score"]
    )


@pytest.mark.parametrize(
    ("retired_key", "retired_value"),
    [
        ("gamma_N", 1.0),
        ("gamma_N_schedule_mode", "fixed"),
        ("novelty_eps", 1e-6),
        ("phase2_novelty_mode", "collective_span_v1"),
        ("novelty_ablation_mode", "off"),
        ("phase2_novelty_multiplier_policy", "legacy_ablation_mode_v1"),
    ],
)
def test_full_score_config_rejects_retired_ordinary_novelty_controls(
    retired_key: str,
    retired_value,
) -> None:
    with pytest.raises(TypeError, match=retired_key):
        FullScoreConfig(**{retired_key: retired_value})


def test_deferred_gram_fallback_flag_does_not_change_ordinary_phase2_score() -> None:
    feat = _phase2_raw_base_feature()
    baseline = scoring_mod.phase2_raw_geometry_score(
        feat,
        F_raw=2.0,
        h_raw=0.0,
        q_window=[1.0],
        Q_window=np.eye(1),
        cfg=FullScoreConfig(),
    )
    enabled = scoring_mod.phase2_raw_geometry_score(
        feat,
        F_raw=2.0,
        h_raw=0.0,
        q_window=[1.0],
        Q_window=np.eye(1),
        cfg=FullScoreConfig(deferred_gram_fallback_enabled=True),
    )
    assert enabled["phase2_gram_novelty_policy"] == (
        GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
    )
    assert enabled["phase2_raw_score"] == pytest.approx(
        baseline["phase2_raw_score"]
    )


def test_full_scoring_manual_floors_apply_to_legacy_direct_features() -> None:
    feat = replace(
        _phase2_raw_base_feature(),
        g_lcb=2.0,
        epsilon_g_shot=0.0,
        b_g_hw=0.0,
        b_g_drift=0.0,
        epsilon_g_res=0.0,
        g_hw_lcb=0.0,
        g_lcb_legacy_shot=0.0,
        hardware_resolution_mode="ideal",
        hardware_resolution_source="legacy_unset",
        F_metric=2.0,
        metric_proxy=2.0,
        F_raw=2.0,
        F_red=2.0,
        h_eff=0.0,
        h_hat=0.0,
        novelty=1.0,
        phase2_raw_F_effective=2.0,
        phase2_raw_trust_gain=99.0,
        phase2_raw_novelty=1.0,
        phase2_raw_score=99.0,
        phase2_burden_total=1.0,
    )
    cfg = FullScoreConfig(
        hardware_resolution_mode="manual",
        manual_b_g_hw=3.0,
        manual_b_g_drift=0.0,
        z_alpha=0.0,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
    )

    raw = scoring_mod.phase2_raw_geometry_score(
        feat,
        F_raw=2.0,
        h_raw=0.0,
        q_window=[],
        Q_window=np.zeros((0, 0)),
        cfg=cfg,
    )
    comps = phase3_canonical_score_components(feat, cfg)
    rescored = scoring_mod.rescore_candidate_feature(feat, cfg, selector_geometry_mode="raw_exact")

    assert raw["phase2_g_hw_lcb"] == pytest.approx(0.0)
    assert raw["phase2_raw_score"] == pytest.approx(0.0)
    assert comps["fallback_mode"] == "nonpositive_gradient"
    assert comps["g_hw_lcb"] == pytest.approx(0.0)
    assert comps["epsilon_g_res"] == pytest.approx(3.0)
    assert comps["hardware_resolution_source"] == "manual_scalar_floors"
    assert rescored["phase2_raw_recomputed"] is True
    assert rescored["phase2_raw_trust_gain"] == pytest.approx(0.0)
    assert rescored["phase2_raw_score"] == pytest.approx(0.0)
    assert rescored["selector_score"] == pytest.approx(0.0)


def test_hardware_resolution_config_validation_rejects_bad_floors() -> None:
    feat = _phase2_raw_base_feature()
    with pytest.raises(ValueError, match="ideal hardware_resolution_mode requires zero"):
        scoring_mod.phase2_raw_geometry_score(
            feat,
            F_raw=2.0,
            h_raw=0.0,
            q_window=[],
            Q_window=np.zeros((0, 0)),
            cfg=FullScoreConfig(hardware_resolution_mode="ideal", manual_b_g_hw=0.1),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        phase3_canonical_score_components(
            feat,
            FullScoreConfig(hardware_resolution_mode="manual", manual_b_g_hw=-0.1),
        )
    with pytest.raises(ValueError, match="finite"):
        phase3_canonical_score_components(
            feat,
            FullScoreConfig(hardware_resolution_mode="manual", manual_b_g_drift=float("inf")),
        )


def test_rescore_candidate_feature_does_not_apply_phase2_confidence_to_canonical_raw_score() -> None:
    cfg = FullScoreConfig(z_alpha=0.5)
    feat = replace(
        _phase2_raw_base_feature(),
        confidence_factor=0.75,
        phase2_raw_score=None,
        phase2_raw_trust_gain=8.0,
        phase2_raw_novelty=1.0,
        phase2_burden_total=2.0,
        phase2_novelty_mode="collective_span_v1",
        phase2_confidence_applied=False,
        phase2_raw_score_formula="DeltaE_TR_raw * N2 / (1 + K2)",
        phase3_selector_policy="legacy_phase3_v1",
        phase3_score_policy="legacy_phase3_v1",
    )

    rescored = scoring_mod.rescore_candidate_feature(feat, cfg, selector_geometry_mode="raw_exact")

    assert rescored["phase2_raw_recomputed"] is True
    assert rescored["phase2_raw_score"] == pytest.approx(4.0)
    assert rescored["selector_score"] == pytest.approx(4.0)


def test_rescore_candidate_feature_preserves_existing_phase2_raw_score() -> None:
    cfg = FullScoreConfig(z_alpha=0.5)
    feat = replace(
        _phase2_raw_base_feature(),
        confidence_factor=0.75,
        phase2_raw_score=123.0,
        phase2_raw_trust_gain=8.0,
        phase2_raw_novelty=1.0,
        phase2_burden_total=2.0,
        phase2_novelty_mode="collective_span_v1",
        phase2_confidence_applied=False,
        phase2_raw_score_formula="DeltaE_TR_raw / (1 + K2)",
        phase3_selector_policy="legacy_phase3_v1",
        phase3_score_policy="legacy_phase3_v1",
    )

    rescored = scoring_mod.rescore_candidate_feature(feat, cfg, selector_geometry_mode="raw_exact")

    assert rescored["phase2_raw_recomputed"] is False
    assert rescored["phase2_raw_score"] == pytest.approx(123.0)
    assert rescored["selector_score"] == pytest.approx(123.0)


def test_rescore_candidate_feature_invalidates_stale_confidence_raw_score_when_hw_lcb_zero() -> None:
    cfg = FullScoreConfig(z_alpha=1.0)
    feat = replace(
        _phase2_raw_base_feature(sigma_hat=1.0),
        g_signed=0.2,
        g_abs=0.2,
        g_lcb=0.2,
        g_hw_lcb=0.0,
        epsilon_g_shot=1.0,
        epsilon_g_res=1.0,
        g_lcb_legacy_shot=0.0,
        hardware_resolution_mode="ideal",
        hardware_resolution_source="ideal_zero_floors",
        F_metric=1.0,
        metric_proxy=1.0,
        F_raw=1.0,
        F_red=1.0,
        h_eff=0.0,
        h_hat=0.0,
        novelty=1.0,
        phase2_raw_score=123.0,
        phase2_raw_trust_gain=123.0,
        phase2_raw_novelty=1.0,
        phase2_raw_F_effective=1.0,
        phase2_burden_total=1.0,
        phase2_raw_score_formula="DeltaE_TR_raw * confidence_factor * N2 / (1 + K2)",
        phase3_selector_policy="hardware_resolvable_v1",
        phase3_score_policy="hardware_resolvable_v1",
    )

    rescored = scoring_mod.rescore_candidate_feature(feat, cfg, selector_geometry_mode="raw_exact")

    assert rescored["phase2_raw_recomputed"] is True
    assert rescored["phase2_g_hw_lcb"] == pytest.approx(0.0)
    assert rescored["phase2_raw_trust_gain"] == pytest.approx(0.0)
    assert rescored["phase2_raw_score"] == pytest.approx(0.0)
    assert rescored["selector_score"] == pytest.approx(0.0)


def test_raw_f_metric_from_state_matches_centered_generator_variance() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    assert raw_f_metric_from_state(
        psi_state=psi_ref,
        candidate_label="x",
        candidate_term=_term("x"),
        compiled_cache={},
        pauli_action_cache={},
    ) == pytest.approx(1.0)
    assert raw_f_metric_from_state(
        psi_state=psi_ref,
        candidate_label="z",
        candidate_term=_term("z"),
        compiled_cache={},
        pauli_action_cache={},
    ) == pytest.approx(0.0)


def test_trust_region_drop_matches_newton_branch() -> None:
    got = trust_region_drop(0.4, 2.0, 1.0, 1.0)
    assert got == pytest.approx(0.04)


def _v4_phase1_feature(*, gradient: float, metric: float):
    cost = Phase1CompileCostOracle().estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=1,
    )
    return build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=float(gradient),
        metric_proxy=float(metric),
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=MeasurementCacheAudit().estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(
            phase1_energy_model=PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
            z_alpha=0.0,
        ),
    )


def _v4_phase1_cfg(*, lambda_F: float) -> SimpleScoreConfig:
    return SimpleScoreConfig(
        phase1_energy_model=PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
        rho=0.4,
        lambda_F=float(lambda_F),
        z_alpha=0.0,
    )


def _v4_phase2_cfg() -> FullScoreConfig:
    return FullScoreConfig(
        phase2_curvature_policy=(
            PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
        ),
        phase2_cheap_curvature_proxy_policy=(
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
        ),
        z_alpha=0.0,
    )


def _phase2_curvature_receipt(value: float) -> dict[str, object]:
    binding = {
        "state_fingerprint": "state",
        "ordered_scaffold_fingerprint": "scaffold",
        "theta_fingerprint": "theta",
        "hamiltonian_fingerprint": "hamiltonian",
        "candidate_coordinate_fingerprint": "candidate",
        "candidate_position_id": 0,
        "derivative_convention": (
            "compiled_ansatz_exact_parameter_derivatives_v1"
        ),
    }
    return {
        "schema": PHASE2_DIRECTIONAL_CURVATURE_RECEIPT_SCHEMA,
        "status": "computed_finite",
        "h_raw": float(value),
        **binding,
        "measurement_provenance": {
            "schema": PHASE2_DIRECTIONAL_CURVATURE_PROVENANCE_SCHEMA,
            "source": "unit_measured_directional_hessian",
            "required_primitives_resolved": True,
            "added_query_count": 0,
        },
    }


def test_v4_phase1_is_exactly_first_order_fs_trust_gain() -> None:
    feat = _v4_phase1_feature(gradient=3.0, metric=4.0)
    assert phase1_trust_region_gain(feat, _v4_phase1_cfg(lambda_F=1.0)) == (
        pytest.approx(0.4 * 3.0 / np.sqrt(4.0))
    )


def test_v4_phase1_is_lambda_f_invariant_and_coordinate_scale_invariant() -> None:
    feat = _v4_phase1_feature(gradient=3.0, metric=4.0)
    scaled = _v4_phase1_feature(gradient=15.0, metric=100.0)
    gains = [
        phase1_trust_region_gain(feat, _v4_phase1_cfg(lambda_F=value))
        for value in (1.0e-15, 1.0, 1.0e15)
    ]
    assert gains == pytest.approx([0.6, 0.6, 0.6])
    assert phase1_trust_region_gain(
        scaled,
        _v4_phase1_cfg(lambda_F=17.0),
    ) == pytest.approx(gains[0])


def test_v4_phase1_coordinate_scale_invariance_crosses_legacy_metric_floor() -> None:
    base = _v4_phase1_feature(gradient=3.0e-8, metric=4.0e-16)
    scaled = _v4_phase1_feature(gradient=3.0, metric=4.0)
    cfg = _v4_phase1_cfg(lambda_F=1.0e15)
    assert phase1_trust_region_gain(base, cfg) == pytest.approx(0.6)
    assert phase1_trust_region_gain(scaled, cfg) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "raw",
    [None, "0.5", True, np.nan, np.inf, -np.inf],
)
def test_v4_phase2_missing_malformed_or_nonfinite_curvature_aborts(raw) -> None:
    info = (
        {}
        if raw is None
        else {
            "h_raw": raw,
            "phase2_curvature_receipt": _phase2_curvature_receipt(0.5),
        }
    )
    with pytest.raises(Phase2CurvatureConstructionError):
        _validated_phase2_directional_curvature(
            curvature_info=info,
            cfg=_v4_phase2_cfg(),
            candidate_label="x",
        )


def test_v4_phase2_explicit_none_curvature_aborts() -> None:
    with pytest.raises(Phase2CurvatureConstructionError, match="is None"):
        _validated_phase2_directional_curvature(
            curvature_info={"h_raw": None},
            cfg=_v4_phase2_cfg(),
            candidate_label="x",
        )


def test_v4_phase2_requires_measurement_provenance_receipt() -> None:
    receipt = _phase2_curvature_receipt(0.5)
    receipt.pop("measurement_provenance")
    with pytest.raises(
        Phase2CurvatureConstructionError,
        match="measurement provenance",
    ):
        _validated_phase2_directional_curvature(
            curvature_info={
                "h_raw": 0.5,
                "phase2_curvature_receipt": receipt,
            },
            cfg=_v4_phase2_cfg(),
            candidate_label="x",
        )


@pytest.mark.parametrize("resolved", [False, "false", "unresolved", 1, None])
def test_v4_phase2_requires_literal_true_measurement_resolution(resolved) -> None:
    receipt = _phase2_curvature_receipt(0.5)
    receipt["measurement_provenance"]["required_primitives_resolved"] = resolved
    with pytest.raises(
        Phase2CurvatureConstructionError,
        match="measurement provenance",
    ):
        _validated_phase2_directional_curvature(
            curvature_info={
                "h_raw": 0.5,
                "phase2_curvature_receipt": receipt,
            },
            cfg=_v4_phase2_cfg(),
            candidate_label="x",
        )


def test_v4_phase2_trust_gain_uses_measured_metric_not_deferred_fallback_ridge() -> None:
    feat = _v4_phase1_feature(gradient=3.0e-8, metric=4.0e-16)
    cfg = replace(
        _v4_phase2_cfg(),
        rho=0.4,
        deferred_gram_fallback_ridge=1.0e-6,
    )
    payload = phase2_raw_geometry_score(
        feat,
        F_raw=4.0e-16,
        h_raw=0.0,
        q_window=[],
        Q_window=np.zeros((0, 0), dtype=float),
        cfg=cfg,
    )
    assert payload["phase2_trust_region_gain"] == pytest.approx(0.6)
    assert payload["phase2_raw_F_effective"] == pytest.approx(4.0e-16)


def test_v4_phase2_finite_negative_curvature_is_valid_and_clipped_only_in_model() -> None:
    h_raw, receipt = _validated_phase2_directional_curvature(
        curvature_info={
            "h_raw": -0.5,
            "phase2_curvature_receipt": _phase2_curvature_receipt(-0.5),
        },
        cfg=_v4_phase2_cfg(),
        candidate_label="x",
    )
    assert h_raw == pytest.approx(-0.5)
    assert receipt is not None
    feat = _v4_phase1_feature(gradient=2.0, metric=4.0)
    payload = phase2_raw_geometry_score(
        feat,
        F_raw=4.0,
        h_raw=h_raw,
        q_window=[],
        Q_window=np.zeros((0, 0), dtype=float),
        cfg=_v4_phase2_cfg(),
    )
    assert payload["phase2_trust_region_gain"] == pytest.approx(0.25)
    assert payload["phase2_lambda_f_proxy_applied"] is False


def test_v4_phase2_cheap_lambda_f_proxy_is_unreachable() -> None:
    with pytest.raises(Phase2CurvatureConstructionError, match="proxy is disabled"):
        phase3_cheap_ratio_v1(_v4_phase1_feature(gradient=1.0, metric=1.0), _v4_phase2_cfg())


def test_v4_curvature_failure_aborts_before_any_novelty_rescue() -> None:
    class _MissingCurvatureOracle:
        def estimate(self, **_kwargs):
            return {}

    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    scaffold_context, h_compiled = _scaffold_context(
        psi_state=psi_ref,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
    )
    with pytest.raises(
        Phase2CurvatureConstructionError,
        match="directional curvature value is absent",
    ):
        build_full_candidate_features(
            base_feature=_v4_phase1_feature(gradient=1.0, metric=1.0),
            candidate_term=_term("x"),
            cfg=_v4_phase2_cfg(),
            novelty_oracle=OrderedInsertionGeometryOracle(),
            curvature_oracle=_MissingCurvatureOracle(),
            scaffold_context=scaffold_context,
            h_compiled=h_compiled,
            compiled_cache={},
            pauli_action_cache={},
            optimizer_memory=None,
        )


def test_full_v2_score_falls_back_safely_without_window_curvature() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=0.5,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
    )
    feat = type(feat)(**{**feat.__dict__, "h_hat": 0.5, "curvature_mode": "self_only"})
    score, fallback = full_v2_score(feat, cfg)
    assert score > 0.0
    assert fallback == "legacy_metric_path"


def test_build_full_candidate_features_clips_novelty_and_preserves_window() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    base = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=1,
        append_position=1,
        positions_considered=[1],
        gradient_signed=0.3,
        metric_proxy=0.3,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=1, append_position=1, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
    )
    scaffold_context, h_compiled = _scaffold_context(
        psi_state=psi_ref,
        selected_ops=[_term("x")],
        theta=[0.0],
        refit_window_indices=[0],
    )
    feat = build_full_candidate_features(
        base_feature=base,
        candidate_term=_term("x"),
        cfg=FullScoreConfig(shortlist_size=2),
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
    )
    assert 0.0 <= float(feat.novelty or 0.0) <= 1.0
    assert feat.refit_window_indices == [0]
    assert feat.full_v2_score is not None
    assert feat.schur_window_solve is not None
    assert len(feat.schur_window_solve) == len(feat.b_hat or []) == len(feat.H_window or [])
    if feat.H_window:
        H = np.asarray(feat.H_window, dtype=float)
        b = np.asarray(feat.b_hat, dtype=float)
        s = np.asarray(feat.schur_window_solve, dtype=float)
        ridge = float(feat.ridge_used or 0.0)
        R = 0.5 * (H + H.T) + ridge * np.eye(H.shape[0], dtype=float)
        residual = np.linalg.norm(R @ s - b) / max(1.0, float(np.linalg.norm(b)))
        assert residual < 1.0e-10
        assert feat.h_eff == pytest.approx(float(feat.h_hat or 0.0) - float(b @ s))


def test_build_full_candidate_features_fallback_only_preserves_gram_and_defers_n3() -> None:
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    base = _phase2_raw_base_feature(sigma_hat=0.0)
    scaffold_context, h_compiled = _scaffold_context(
        psi_state=psi_ref,
        selected_ops=[_term("x")],
        theta=[0.0],
        refit_window_indices=[0],
    )
    feature = build_full_candidate_features(
        base_feature=base,
        candidate_term=_term("x"),
        cfg=FullScoreConfig(
            deferred_gram_fallback_enabled=True,
        ),
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
    )

    geometry = feature.phase2_joint_geometry_reuse
    assert feature.phase2_raw_novelty is None
    assert feature.novelty is None
    assert feature.phase_score_components["N3"] is None
    assert feature.phase_score_components[
        "phase2_legacy_pairwise_novelty"
    ] is None
    assert feature.phase_score_components["phase3_novelty_status"] == (
        "not_computed_for_ordinary_scoring"
    )
    assert geometry["G_AA"]
    assert geometry["G_AB"]
    assert geometry["G_BB"] >= 0.0
    assert geometry["phase3_deferred_gram_novelty"]["schema"] == (
        "phase3_deferred_gram_novelty_v1"
    )


def test_live_cheap_score_remains_simple_alias_when_phase3_config_is_present() -> None:
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.4,
        metric_proxy=0.2,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0),
        cheap_score_cfg=FullScoreConfig(
            lambda_F=1.0,
            wD=0.0,
            wG=0.0,
            wC=0.0,
            wP=0.0,
            wc=0.0,
            lifetime_weight=0.0,
        ),
    )
    expected_delta = trust_region_drop(float(feat.g_hw_lcb), 1.0 * 0.2, 0.2, 0.25)
    assert float(feat.simple_score or 0.0) == pytest.approx(expected_delta)
    assert float(feat.cheap_score or 0.0) == pytest.approx(expected_delta)
    assert feat.cheap_score_version == "simple_v1"
    assert float(feat.cheap_metric_proxy) == pytest.approx(0.2)
    assert float(feat.cheap_benefit_proxy or 0.0) == pytest.approx(expected_delta)
    assert float(feat.cheap_burden_total or 0.0) == pytest.approx(1.0)


def test_build_full_candidate_features_preserves_phase3_cheap_fields() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    metric_exact = raw_f_metric_from_state(
        psi_state=psi_ref,
        candidate_label="x",
        candidate_term=_term("x"),
        compiled_cache={},
        pauli_action_cache={},
    )
    base = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=1,
        append_position=1,
        positions_considered=[1],
        gradient_signed=0.3,
        metric_proxy=float(metric_exact),
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=1, append_position=1, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        cheap_score_cfg=FullScoreConfig(),
    )
    scaffold_context, h_compiled = _scaffold_context(
        psi_state=psi_ref,
        selected_ops=[_term("x")],
        theta=[0.0],
        refit_window_indices=[0],
    )
    feat = build_full_candidate_features(
        base_feature=base,
        candidate_term=_term("x"),
        cfg=FullScoreConfig(shortlist_size=2),
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
    )
    assert feat.cheap_score == pytest.approx(base.cheap_score or 0.0)
    assert feat.cheap_score_version == "simple_v1"
    assert feat.cheap_metric_proxy == pytest.approx(base.cheap_metric_proxy)
    assert feat.cheap_benefit_proxy == pytest.approx(base.cheap_benefit_proxy or 0.0)
    assert feat.cheap_burden_total == pytest.approx(base.cheap_burden_total or 0.0)
    assert feat.metric_proxy == pytest.approx(feat.cheap_metric_proxy)
    assert feat.F_metric == pytest.approx(feat.cheap_metric_proxy)


def test_build_full_candidate_features_selector_score_defaults_to_reduced_geometry() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    metric_exact = raw_f_metric_from_state(
        psi_state=psi_ref,
        candidate_label="x",
        candidate_term=_term("x"),
        compiled_cache={},
        pauli_action_cache={},
    )
    base = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=1,
        append_position=1,
        positions_considered=[1],
        gradient_signed=0.3,
        metric_proxy=float(metric_exact),
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=1, append_position=1, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        cheap_score_cfg=FullScoreConfig(),
    )
    scaffold_context, h_compiled = _scaffold_context(
        psi_state=psi_ref,
        selected_ops=[_term("x")],
        theta=[0.0],
        refit_window_indices=[0],
    )
    feat = build_full_candidate_features(
        base_feature=base,
        candidate_term=_term("x"),
        cfg=FullScoreConfig(shortlist_size=2, phase3_selector_geometry_mode="reduced"),
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
    )
    assert feat.full_v2_score is not None
    assert feat.phase2_raw_score is not None
    assert feat.selector_geometry_mode == "reduced"
    assert float(feat.selector_score or 0.0) == pytest.approx(float(feat.full_v2_score or 0.0))
    assert float(feat.phase_score_components["selector_score"]) == pytest.approx(float(feat.full_v2_score or 0.0))


def test_build_full_candidate_features_selector_score_can_use_raw_exact_geometry() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    metric_exact = raw_f_metric_from_state(
        psi_state=psi_ref,
        candidate_label="x",
        candidate_term=_term("x"),
        compiled_cache={},
        pauli_action_cache={},
    )
    base = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=1,
        append_position=1,
        positions_considered=[1],
        gradient_signed=0.3,
        metric_proxy=float(metric_exact),
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=1, append_position=1, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        cheap_score_cfg=FullScoreConfig(),
    )
    scaffold_context, h_compiled = _scaffold_context(
        psi_state=psi_ref,
        selected_ops=[_term("x")],
        theta=[0.0],
        refit_window_indices=[0],
    )
    feat = build_full_candidate_features(
        base_feature=base,
        candidate_term=_term("x"),
        cfg=FullScoreConfig(shortlist_size=2, phase3_selector_geometry_mode="raw_exact"),
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
    )
    assert feat.full_v2_score is not None
    assert feat.phase2_raw_score is not None
    assert feat.selector_geometry_mode == "raw_exact"
    assert float(feat.selector_score or 0.0) == pytest.approx(float(feat.phase2_raw_score or 0.0))
    assert float(feat.phase_score_components["selector_score"]) == pytest.approx(float(feat.phase2_raw_score or 0.0))


def _full_record(
    *,
    label: str,
    candidate_label: str,
    candidate_pool_index: int,
    gradient_signed: float,
    psi_state: np.ndarray,
    selected_ops: list[object],
    theta: list[float],
    refit_window_indices: list[int],
    full_cfg: FullScoreConfig,
    simple_cfg: SimpleScoreConfig,
    h_label: str = "z",
    include_phase3: bool = True,
    position_id: int = 0,
) -> tuple[dict[str, object], object]:
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    candidate_term = _term(label)
    metric_exact = raw_f_metric_from_state(
        psi_state=psi_state,
        candidate_label=str(candidate_label),
        candidate_term=candidate_term,
        compiled_cache={},
        pauli_action_cache={},
    )
    base = build_candidate_features(
        stage_name="core",
        candidate_label=str(candidate_label),
        candidate_family="core",
        candidate_pool_index=int(candidate_pool_index),
        position_id=int(position_id),
        append_position=int(position_id),
        positions_considered=[int(position_id)],
        gradient_signed=float(gradient_signed),
        metric_proxy=float(metric_exact),
        sigma_hat=0.0,
        refit_window_indices=list(refit_window_indices),
        compile_cost=oracle.estimate(
            candidate_term_count=1,
            position_id=int(position_id),
            append_position=int(position_id),
            refit_active_count=len(refit_window_indices),
            candidate_term=candidate_term,
        ),
        measurement_stats=meas.estimate(measurement_group_keys_for_term(candidate_term)),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=simple_cfg,
        cheap_score_cfg=full_cfg,
    )
    scaffold_context, h_compiled = _scaffold_context(
        psi_state=psi_state,
        selected_ops=list(selected_ops),
        theta=list(theta),
        refit_window_indices=list(refit_window_indices),
        h_label=h_label,
    )
    feat = build_full_candidate_features(
        base_feature=base,
        candidate_term=candidate_term,
        cfg=full_cfg,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        scaffold_context=scaffold_context,
        h_compiled=h_compiled,
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
        include_phase3=bool(include_phase3),
    )
    return (
        {
            "feature": feat,
            "full_v2_score": float(feat.full_v2_score or 0.0),
            "phase2_raw_score": float(feat.phase2_raw_score or 0.0),
            "candidate_pool_index": int(candidate_pool_index),
            "position_id": int(position_id),
            "candidate_term": candidate_term,
        },
        h_compiled,
    )


def _legacy_phase3_feature(
    *,
    gradient_signed: float,
    metric_proxy: float,
    sigma_hat: float = 0.0,
    h_hat: float | None = None,
    motif_bonus: float = 0.0,
    duplicate_penalty: float = 0.0,
    leakage_penalty: float = 0.0,
    full_cfg: FullScoreConfig | None = None,
):
    cfg_full = full_cfg if full_cfg is not None else FullScoreConfig()
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=float(gradient_signed),
        metric_proxy=float(metric_proxy),
        sigma_hat=float(sigma_hat),
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=float(leakage_penalty),
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        cheap_score_cfg=cfg_full,
        motif_bonus=float(motif_bonus),
    )
    return type(feat)(
        **{
            **feat.__dict__,
            "h_hat": float(h_hat if h_hat is not None else max(metric_proxy, 1e-12)),
            "curvature_mode": "self_only",
            "phase3_duplicate_penalty": float(duplicate_penalty),
        }
    )


def test_full_v2_canonical_component_product_equals_score() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    full_cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
        motif_bonus_weight=99.0,
    )
    simple_cfg = SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    rec, _ = _full_record(
        label="x",
        candidate_label="x",
        candidate_pool_index=0,
        gradient_signed=0.8,
        psi_state=psi_ref,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=full_cfg,
        simple_cfg=simple_cfg,
    )
    feat = rec["feature"]
    comps = feat.phase_score_components
    expected = (
        float(comps["DeltaE_TR"])
        / float(comps["denominator_1_plus_K3"])
    )
    assert "confidence_factor" in comps
    assert comps["N3"] is None
    assert feat.phase3_canonical_score_formula == "DeltaE_TR / (1 + K3)"
    assert float(feat.phase3_primary_score or 0.0) == pytest.approx(expected)
    assert float(feat.full_v2_score or 0.0) == pytest.approx(expected)
    assert float(feat.full_v2_score or 0.0) == pytest.approx(full_v2_score(feat, full_cfg)[0])


def test_full_v2_confidence_factor_is_telemetry_only() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.5,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
        motif_bonus_weight=0.0,
    )
    feat = _legacy_phase3_feature(
        gradient_signed=1.0,
        metric_proxy=1.0,
        sigma_hat=0.25,
        h_hat=0.5,
        full_cfg=cfg,
    )
    components = phase3_canonical_score_components(feat, cfg)
    score, _ = full_v2_score(feat, cfg)
    confidence = float(components["confidence_factor"])
    assert 0.0 < confidence < 1.0
    expected_without_confidence = (
        float(components["DeltaE_TR"])
        / float(components["denominator_1_plus_K3"])
    )
    expected_with_confidence = expected_without_confidence * confidence
    assert score == pytest.approx(expected_without_confidence)
    assert score != pytest.approx(expected_with_confidence)


def test_historical_novelty_fields_are_passive_in_phase3_scoring() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
    )
    feat = _legacy_phase3_feature(
        gradient_signed=0.5,
        metric_proxy=1.0,
        h_hat=0.5,
        full_cfg=cfg,
    )
    low = replace(feat, novelty=0.01, phase2_raw_novelty=0.01)
    high = replace(feat, novelty=0.99, phase2_raw_novelty=0.99)

    low_components = phase3_canonical_score_components(low, cfg)
    high_components = phase3_canonical_score_components(high, cfg)

    assert low_components["N3"] is None
    assert high_components["N3"] is None
    assert low_components["phase3_novelty_applied"] is False
    assert high_components["phase3_novelty_applied"] is False
    assert low_components["phase3_primary_score"] == pytest.approx(
        high_components["phase3_primary_score"]
    )


def test_full_v2_leakage_penalty_is_gate_telemetry_not_primary_multiplier() -> None:
    cfg_no_leak_weight = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        eta_L=0.0,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
    )
    cfg_with_leak_weight = FullScoreConfig(**{**cfg_no_leak_weight.__dict__, "eta_L": 10.0})
    feat = _legacy_phase3_feature(
        gradient_signed=0.5,
        metric_proxy=1.0,
        h_hat=0.5,
        leakage_penalty=0.25,
        full_cfg=cfg_with_leak_weight,
    )
    score_without_weight, _ = full_v2_score(feat, cfg_no_leak_weight)
    score_with_weight, _ = full_v2_score(feat, cfg_with_leak_weight)
    components = phase3_canonical_score_components(feat, cfg_with_leak_weight)
    assert float(components["leakage_factor"]) < 1.0
    assert score_with_weight == pytest.approx(score_without_weight)


def test_full_v2_default_auxiliary_terms_cannot_rescue_weak_geometry() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
        motif_bonus_weight=1_000_000.0,
        duplicate_penalty_weight=1_000_000.0,
    )
    strong = _legacy_phase3_feature(
        gradient_signed=0.8,
        metric_proxy=1.0,
        h_hat=0.5,
        duplicate_penalty=1_000_000.0,
        full_cfg=cfg,
    )
    weak = _legacy_phase3_feature(
        gradient_signed=0.01,
        metric_proxy=1.0,
        h_hat=0.5,
        motif_bonus=1_000_000.0,
        full_cfg=cfg,
    )
    strong_score, _ = full_v2_score(strong, cfg)
    weak_score, _ = full_v2_score(weak, cfg)
    strong_components = phase3_canonical_score_components(strong, cfg)
    weak_components = phase3_canonical_score_components(weak, cfg)
    assert float(weak_components["phase3_tie_break_score"]) > float(strong_components["phase3_tie_break_score"])
    assert strong_score > weak_score


def test_full_v2_duplicate_penalty_is_tie_break_only_by_default() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
        duplicate_penalty_weight=1000.0,
    )
    base = _legacy_phase3_feature(
        gradient_signed=0.5,
        metric_proxy=1.0,
        h_hat=0.5,
        duplicate_penalty=0.0,
        full_cfg=cfg,
    )
    duplicate = type(base)(**{**base.__dict__, "phase3_duplicate_penalty": 10.0})
    base_score, _ = full_v2_score(base, cfg)
    duplicate_score, _ = full_v2_score(duplicate, cfg)
    base_components = phase3_canonical_score_components(base, cfg)
    duplicate_components = phase3_canonical_score_components(duplicate, cfg)
    assert duplicate_score == pytest.approx(base_score)
    assert float(duplicate_components["phase3_tie_break_score"]) < float(
        base_components["phase3_tie_break_score"]
    )


def test_phase3_shortlist_primary_score_beats_huge_tie_break() -> None:
    records = [
        {
            "candidate_label": "weak_motif",
            "full_v2_score": 1.0,
            "phase3_tie_break_score": 1_000_000.0,
            "candidate_pool_index": 0,
            "position_id": 0,
        },
        {
            "candidate_label": "strong_geometry",
            "full_v2_score": 2.0,
            "phase3_tie_break_score": -1_000_000.0,
            "candidate_pool_index": 1,
            "position_id": 0,
        },
    ]
    shortlisted = phase_shortlist_records(
        records,
        score_key="full_v2_score",
        threshold=float("-inf"),
        cap=2,
        frontier_ratio=1.0,
        tie_break_score_key="phase3_tie_break_score",
    )
    assert [row["candidate_label"] for row in shortlisted] == ["strong_geometry", "weak_motif"]


def test_full_v2_ablation_additive_mode_is_explicit_opt_in() -> None:
    cfg_default = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
        motif_bonus_weight=2.0,
    )
    feat = _legacy_phase3_feature(
        gradient_signed=0.2,
        metric_proxy=1.0,
        h_hat=0.5,
        motif_bonus=0.25,
        full_cfg=cfg_default,
    )
    baseline, _ = full_v2_score(feat, cfg_default)
    no_motif, _ = full_v2_score(type(feat)(**{**feat.__dict__, "motif_bonus": 0.0}), cfg_default)
    cfg_ablation = FullScoreConfig(**{**cfg_default.__dict__, "auxiliary_score_mode": "ablation_additive"})
    ablation, _ = full_v2_score(feat, cfg_ablation)
    assert baseline == pytest.approx(no_motif)
    assert ablation == pytest.approx(baseline + 0.5)


def test_reduced_plane_batch_select_can_keep_two_orthogonal_records() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    full_cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
        batch_target_size=2,
        batch_size_cap=2,
        batch_near_degenerate_ratio=0.95,
        batch_additivity_tol=1.0,
    )
    simple_cfg = SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    rec_x, h_compiled = _full_record(
        label="x",
        candidate_label="x",
        candidate_pool_index=0,
        gradient_signed=0.8,
        psi_state=psi_ref,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=full_cfg,
        simple_cfg=simple_cfg,
    )
    rec_y, _ = _full_record(
        label="y",
        candidate_label="y",
        candidate_pool_index=1,
        gradient_signed=0.8,
        psi_state=psi_ref,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=full_cfg,
        simple_cfg=simple_cfg,
    )
    selected, summary = reduced_plane_batch_select(
        [rec_x, rec_y],
        cfg=full_cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi_ref,
        psi_state=psi_ref,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        tie_break_score_key="phase2_raw_score",
    )
    assert len(selected) == 2
    assert {str(rec["candidate_term"].label) for rec in selected} == {"x", "y"}
    assert float(summary.get("joint_gain", 0.0)) > 0.0
    assert float(summary.get("additivity_defect", 1.0)) <= 1.0


def test_greedy_reduced_plane_batch_proposal_uses_cost_weighted_batch_score() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    full_cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
        batch_selection_mode="greedy_reduced_plane",
        batch_target_size=2,
        batch_size_cap=9,
        batch_near_degenerate_ratio=0.95,
        batch_additivity_tol=1.0,
    )
    simple_cfg = SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    rec_x, h_compiled = _full_record(
        label="x",
        candidate_label="x",
        candidate_pool_index=0,
        gradient_signed=0.8,
        psi_state=psi_ref,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=full_cfg,
        simple_cfg=simple_cfg,
    )
    rec_y, _ = _full_record(
        label="y",
        candidate_label="y",
        candidate_pool_index=1,
        gradient_signed=0.8,
        psi_state=psi_ref,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=full_cfg,
        simple_cfg=simple_cfg,
    )

    proposals, summary = greedy_reduced_plane_batch_proposals(
        [rec_x, rec_y],
        cfg=full_cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi_ref,
        psi_state=psi_ref,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        tie_break_score_key="phase2_raw_score",
        max_proposals=3,
    )

    assert proposals
    proposal = proposals[0]
    assert summary["selection_mode"] == "greedy_reduced_plane"
    assert int(summary["effective_batch_size_cap"]) == 5
    assert proposal.denominator_1_plus_k3 == pytest.approx(1.0 + proposal.k3)
    assert proposal.score == pytest.approx(proposal.delta_e3 / proposal.denominator_1_plus_k3)
    assert all(
        float(row["phase3_batch_score"]) == pytest.approx(proposal.score)
        for row in proposal.records
    )


def test_select_phase2_batch_records_routes_ordered_batch_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _tetris_record(
            candidate_label="a_x0",
            pauli_label="xe",
            full_v2_score=1.0,
            candidate_pool_index=0,
        )
    ]
    calls: dict[str, object] = {}

    def _fake_greedy_select(ranked_records, **kwargs):  # noqa: ANN001, ANN003 - test shim
        calls["greedy"] = list(ranked_records)
        return [dict(ranked_records[0])], {"selection_mode": "greedy_reduced_plane", "reason": "delegated"}

    def _fake_combinatorial_select(ranked_records, **kwargs):  # noqa: ANN001, ANN003 - test shim
        calls["combinatorial"] = list(ranked_records)
        return [dict(ranked_records[0])], {"selection_mode": "combinatorial_reduced_plane", "reason": "delegated"}

    monkeypatch.setattr(scoring_mod, "greedy_reduced_plane_batch_select", _fake_greedy_select)
    monkeypatch.setattr(scoring_mod, "combinatorial_reduced_plane_batch_select", _fake_combinatorial_select)

    for mode, call_key in (
        ("greedy_reduced_plane", "greedy"),
        ("combinatorial_reduced_plane", "combinatorial"),
    ):
        selected, summary = select_phase2_batch_records(
            records,
            cfg=FullScoreConfig(batch_selection_mode=mode),
            selected_ops=[],
            theta=np.zeros(0, dtype=float),
            psi_ref=np.zeros(1, dtype=complex),
            psi_state=np.zeros(1, dtype=complex),
            h_compiled=object(),
            novelty_oracle=object(),
            curvature_oracle=object(),
            compiled_cache={},
            pauli_action_cache={},
            tie_break_score_key="phase2_raw_score",
        )
        assert selected[0]["candidate_label"] == "a_x0"
        assert summary["selection_mode"] == mode
        assert summary["reason"] == "delegated"
        assert calls[call_key] == records


def test_select_phase2_batch_record_proposals_routes_ordered_batch_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _tetris_record(
            candidate_label="a_x0",
            pauli_label="xe",
            full_v2_score=1.0,
            candidate_pool_index=0,
        )
    ]
    calls: dict[str, object] = {}

    def _fake_greedy_proposals(ranked_records, **kwargs):  # noqa: ANN001, ANN003 - test shim
        calls["greedy"] = list(ranked_records)
        return [], {"selection_mode": "greedy_reduced_plane", "reason": "delegated"}

    def _fake_combinatorial_proposals(ranked_records, **kwargs):  # noqa: ANN001, ANN003 - test shim
        calls["combinatorial"] = list(ranked_records)
        return [], {"selection_mode": "combinatorial_reduced_plane", "reason": "delegated"}

    monkeypatch.setattr(scoring_mod, "greedy_reduced_plane_batch_proposals", _fake_greedy_proposals)
    monkeypatch.setattr(scoring_mod, "combinatorial_reduced_plane_batch_proposals", _fake_combinatorial_proposals)

    for mode, call_key in (
        ("greedy_reduced_plane", "greedy"),
        ("combinatorial_reduced_plane", "combinatorial"),
    ):
        proposals, summary = select_phase2_batch_record_proposals(
            records,
            cfg=FullScoreConfig(batch_selection_mode=mode),
            selected_ops=[],
            theta=np.zeros(0, dtype=float),
            psi_ref=np.zeros(1, dtype=complex),
            psi_state=np.zeros(1, dtype=complex),
            h_compiled=object(),
            novelty_oracle=object(),
            curvature_oracle=object(),
            compiled_cache={},
            pauli_action_cache={},
            tie_break_score_key="phase2_raw_score",
        )
        assert proposals == []
        assert summary["selection_mode"] == mode
        assert summary["reason"] == "delegated"
        assert calls[call_key] == records


def test_reduced_plane_batch_select_blocks_same_generator_at_different_positions() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    full_cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
        batch_target_size=2,
        batch_size_cap=2,
        batch_near_degenerate_ratio=0.95,
        batch_additivity_tol=1.0,
    )
    simple_cfg = SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    rec_a, h_compiled = _full_record(
        label="x",
        candidate_label="same_generator",
        candidate_pool_index=0,
        gradient_signed=0.8,
        psi_state=psi_ref,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=full_cfg,
        simple_cfg=simple_cfg,
    )
    feat_a = rec_a["feature"]
    rec_b = {
        **rec_a,
        "position_id": 1,
        "feature": replace(feat_a, position_id=1, positions_considered=[1]),
    }

    selected, summary = reduced_plane_batch_select(
        [rec_a, rec_b],
        cfg=full_cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi_ref,
        psi_state=psi_ref,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        tie_break_score_key="phase2_raw_score",
    )

    assert len(selected) == 1
    assert selected[0]["candidate_pool_index"] == 0
    assert summary["same_generator_batch_duplicate_policy"] == "block_generator_identity_v1"
    assert int(summary["same_generator_duplicate_skip_count"]) >= 1
    assert summary["same_generator_duplicate_identities"]


@pytest.mark.parametrize(
    "proposal_builder_name",
    [
        "greedy_reduced_plane_batch_proposals",
        "combinatorial_reduced_plane_batch_proposals",
    ],
)
def test_ordered_reduced_plane_batching_uses_global_pauli_child_identity(
    monkeypatch: pytest.MonkeyPatch,
    proposal_builder_name: str,
) -> None:
    child_x_position_0 = {
        **_tetris_record(
            candidate_label="parent-a::child-x",
            pauli_label="xe",
            full_v2_score=1.0,
            candidate_pool_index=0,
            position_id=0,
        ),
        "route_a_global_pauli_identity": "pauli:x",
    }
    child_x_position_1 = {
        **_tetris_record(
            candidate_label="parent-b::child-x",
            pauli_label="xe",
            full_v2_score=0.99,
            candidate_pool_index=1,
            position_id=1,
        ),
        "route_a_global_pauli_identity": "pauli:x",
    }
    sibling_child_y = {
        **_tetris_record(
            candidate_label="parent-a::child-y",
            pauli_label="ey",
            full_v2_score=0.98,
            candidate_pool_index=2,
            position_id=0,
        ),
        "route_a_global_pauli_identity": "pauli:y",
    }

    def fake_batch_evaluator(records, *, mode, **_kwargs):  # noqa: ANN001, ANN003
        copied = tuple(dict(record) for record in records)
        score = float(len(copied))
        return scoring_mod.BatchSelectionProposal(
            records=copied,
            summary={"selection_mode": str(mode), "feasible": True},
            score=score,
            delta_e3=score,
            k3=0.0,
            denominator_1_plus_k3=1.0,
        )

    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_ordered_reduced_plane_batch_proposal",
        fake_batch_evaluator,
    )
    proposal_builder = getattr(scoring_mod, proposal_builder_name)
    proposals, _summary = proposal_builder(
        [child_x_position_0, child_x_position_1, sibling_child_y],
        cfg=FullScoreConfig(
            batch_target_size=2,
            batch_size_cap=2,
            batch_near_degenerate_ratio=0.0,
        ),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.zeros(1, dtype=complex),
        psi_state=np.zeros(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
        max_proposals=3,
    )

    assert proposals
    assert len(proposals[0].records) == 2
    selected_identities = [
        record["route_a_global_pauli_identity"]
        for record in proposals[0].records
    ]
    assert selected_identities == ["pauli:x", "pauli:y"]


def test_reduced_plane_batch_select_rejects_rank_deficient_addon() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    full_cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
        batch_target_size=2,
        batch_size_cap=2,
        batch_near_degenerate_ratio=0.95,
        batch_rank_rel_tol=1e-6,
        batch_additivity_tol=1.0,
    )
    simple_cfg = SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    rec_a, h_compiled = _full_record(
        label="x",
        candidate_label="x_a",
        candidate_pool_index=0,
        gradient_signed=0.8,
        psi_state=psi_ref,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=full_cfg,
        simple_cfg=simple_cfg,
    )
    rec_b, _ = _full_record(
        label="x",
        candidate_label="x_b",
        candidate_pool_index=1,
        gradient_signed=0.79,
        psi_state=psi_ref,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=full_cfg,
        simple_cfg=simple_cfg,
    )
    selected, summary = reduced_plane_batch_select(
        [rec_a, rec_b],
        cfg=full_cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi_ref,
        psi_state=psi_ref,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        tie_break_score_key="phase2_raw_score",
    )
    assert len(selected) == 1
    assert str(selected[0]["candidate_term"].label) == "x"
    assert float(summary.get("joint_gain", 0.0)) > 0.0


def test_combinatorial_search_pool_size_is_independent_of_batch_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            **_tetris_record(
                candidate_label=f"child-{index}",
                pauli_label=("xe", "ye", "ze", "ex")[index],
                full_v2_score=float(10 - index),
                candidate_pool_index=index,
            ),
            "phase2_raw_score": float(10 - index),
            "route_a_global_pauli_identity": f"pauli:{index}",
        }
        for index in range(4)
    ]

    def fake_evaluator(subset, *, mode, **_kwargs):  # noqa: ANN001, ANN003
        copied = tuple(dict(record) for record in subset)
        score = float(sum(float(record["phase2_raw_score"]) for record in copied))
        return scoring_mod.BatchSelectionProposal(
            records=copied,
            summary={
                "selection_mode": str(mode),
                "feasible": True,
                "joint_gain": float(score),
                "contextual_single_total": float(score),
                "additivity_defect": 0.0,
            },
            score=float(score),
            delta_e3=float(score),
            k3=0.0,
            denominator_1_plus_k3=1.0,
        )

    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_ordered_reduced_plane_batch_proposal",
        fake_evaluator,
    )
    summaries = []
    for search_size in (2, 3):
        _proposals, summary = scoring_mod.combinatorial_reduced_plane_batch_proposals(
            records,
            cfg=FullScoreConfig(
                batch_selection_mode="combinatorial_reduced_plane",
                batch_target_size=2,
                batch_size_cap=2,
                batch_search_pool_size=search_size,
                batch_search_population_mode="ranked_child_phase2_v1",
                batch_additivity_policy="off",
            ),
            selected_ops=[],
            theta=np.zeros(0, dtype=float),
            psi_ref=np.ones(1, dtype=complex),
            psi_state=np.ones(1, dtype=complex),
            h_compiled=object(),
            novelty_oracle=object(),
            curvature_oracle=object(),
            max_proposals=4,
        )
        summaries.append(summary)

    assert [summary["effective_batch_size_cap"] for summary in summaries] == [2, 2]
    assert summaries[0]["batch_search_pool_size_effective"] == 2
    assert summaries[1]["batch_search_pool_size_effective"] == 3
    assert summaries[0]["subset_counts_considered"] == {"1": 2, "2": 1}
    assert summaries[1]["subset_counts_considered"] == {"1": 3, "2": 3}


def test_combinatorial_search_width_counts_rank_feasible_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            **_tetris_record(
                candidate_label=f"child-{index}",
                pauli_label=("xe", "ye", "ze")[index],
                full_v2_score=float(3 - index),
                candidate_pool_index=index,
            ),
            "phase2_raw_score": float(3 - index),
            "route_a_global_pauli_identity": f"pauli:{index}",
        }
        for index in range(3)
    ]
    cache = scoring_mod._Phase2JointGeometryCache(
        active_indices=(0,),
        G_AA=np.asarray([[1.0]], dtype=float),
        H_AA=np.asarray([[1.0]], dtype=float),
        g_A=np.asarray([0.0], dtype=float),
        G_AB=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        H_AB=np.zeros((1, 3), dtype=float),
        G_BB_diagonal=np.ones(3, dtype=float),
        H_BB_diagonal=np.ones(3, dtype=float),
        g_B=np.ones(3, dtype=float),
        valid_record_indices=(0, 1, 2),
        valid_gradient_record_indices=(0, 1, 2),
        record_results=[],
        active_block_valid=True,
        state_fingerprint="state",
        ordered_scaffold_fingerprint="scaffold",
        theta_fingerprint="theta",
        state_reconstruction_delta_norm_max=0.0,
    )
    monkeypatch.setattr(
        scoring_mod,
        "_build_phase2_joint_geometry_cache",
        lambda *_args, **_kwargs: cache,
    )

    class _Workspace:
        def build_telemetry(self):
            return {}

    monkeypatch.setattr(
        scoring_mod,
        "_build_batch_full_geometry_workspace",
        lambda *_args, **_kwargs: _Workspace(),
    )

    def fake_evaluator(subset, *, mode, **_kwargs):  # noqa: ANN001, ANN003
        copied = tuple(dict(record) for record in subset)
        score = float(copied[0]["phase2_raw_score"])
        return scoring_mod.BatchSelectionProposal(
            records=copied,
            summary={
                "selection_mode": str(mode),
                "feasible": True,
                "joint_gain": float(score),
                "contextual_single_total": float(score),
                "additivity_defect": 0.0,
            },
            score=float(score),
            delta_e3=float(score),
            k3=0.0,
            denominator_1_plus_k3=1.0,
        )

    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_ordered_reduced_plane_batch_proposal",
        fake_evaluator,
    )
    proposals, summary = scoring_mod.combinatorial_reduced_plane_batch_proposals(
        records,
        cfg=FullScoreConfig(
            batch_selection_mode="combinatorial_reduced_plane",
            batch_target_size=1,
            batch_size_cap=1,
            batch_search_pool_size=1,
            batch_search_population_mode="ranked_child_phase2_v1",
            batch_search_feasibility_policy="rank_feasible_fill_v1",
            batch_additivity_policy="off",
            batch_joint_context_mode="full_ansatz_v1",
        ),
        selected_ops=[object()],
        theta=np.zeros(1, dtype=float),
        psi_ref=np.ones(1, dtype=complex),
        psi_state=np.ones(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
    )

    assert len(proposals) == 1
    assert proposals[0].records[0]["candidate_label"] == "child-1"
    assert summary["batch_search_pool_size_effective"] == 1
    assert summary["child_phase2_rank_feasible_count"] == 2
    assert summary["rank_prefilter_rejection_count"] == 1
    assert summary["rank_feasibility_prefilter"]["input_record_count"] == 3


def test_joint_subset_gate_preserves_exact_ranked_search_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            **_tetris_record(
                candidate_label=f"child-{index}",
                pauli_label=("xe", "ye", "ze")[index],
                full_v2_score=float(3 - index),
                candidate_pool_index=index,
            ),
            "phase2_raw_score": float(3 - index),
            "route_a_global_pauli_identity": f"pauli:{index}",
        }
        for index in range(3)
    ]
    monkeypatch.setattr(
        scoring_mod,
        "_build_phase2_joint_geometry_cache",
        lambda *_args, **_kwargs: pytest.fail(
            "canonical joint gating must not run a singleton prefilter"
        ),
    )

    workspace_labels: list[str] = []

    class _Workspace:
        def build_telemetry(self):
            return {}

    def fake_workspace(search_pool, **_kwargs):  # noqa: ANN001
        workspace_labels.extend(
            str(record["candidate_label"]) for record in search_pool
        )
        return _Workspace()

    monkeypatch.setattr(
        scoring_mod,
        "_build_batch_full_geometry_workspace",
        fake_workspace,
    )
    evaluated: list[tuple[str, ...]] = []

    def fake_evaluator(subset, *, mode, **_kwargs):  # noqa: ANN001, ANN003
        copied = tuple(dict(record) for record in subset)
        evaluated.append(
            tuple(str(record["candidate_label"]) for record in copied)
        )
        score = float(sum(record["phase2_raw_score"] for record in copied))
        return scoring_mod.BatchSelectionProposal(
            records=copied,
            summary={
                "selection_mode": str(mode),
                "feasible": True,
                "joint_gain": float(score),
                "contextual_single_total": float(score),
                "additivity_defect": 0.0,
            },
            score=float(score),
            delta_e3=float(score),
            k3=0.0,
            denominator_1_plus_k3=1.0,
        )

    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_ordered_reduced_plane_batch_proposal",
        fake_evaluator,
    )
    _proposals, summary = scoring_mod.combinatorial_reduced_plane_batch_proposals(
        records,
        cfg=FullScoreConfig(
            batch_selection_mode="combinatorial_reduced_plane",
            batch_target_size=2,
            batch_size_cap=2,
            batch_search_pool_size=2,
            batch_search_population_mode="ranked_child_phase2_v1",
            batch_search_feasibility_policy="joint_subset_gate_v1",
            batch_additivity_policy="off",
            batch_geometry_mode="full_residual_gram_hessian_v1",
            batch_joint_context_mode="full_ansatz_v1",
        ),
        selected_ops=[object()],
        theta=np.zeros(1, dtype=float),
        psi_ref=np.ones(1, dtype=complex),
        psi_state=np.ones(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
    )

    assert workspace_labels == ["child-0", "child-1"]
    assert evaluated == [
        ("child-0",),
        ("child-1",),
        ("child-0", "child-1"),
    ]
    assert summary["batch_search_pool_size_effective"] == 2
    assert summary["subset_counts_evaluated"] == {"1": 2, "2": 1}
    assert summary["rank_prefilter_rejection_count"] == 0
    assert summary["rank_gate_application_stage"] == (
        "joint_subset_after_search_pool"
    )
    assert summary["rank_feasibility_prefilter"]["active"] is False


def test_combinatorial_batch_forwards_joint_pair_observer_to_one_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            **_tetris_record(
                candidate_label=f"child-{index}",
                pauli_label=("xe", "ye")[index],
                full_v2_score=float(2 - index),
                candidate_pool_index=index,
            ),
            "phase2_raw_score": float(2 - index),
            "route_a_global_pauli_identity": f"pauli:{index}",
        }
        for index in range(2)
    ]
    observed: list[dict[str, object]] = []
    marker = {
        "left_workspace_index": 0,
        "right_workspace_index": 1,
        "physical_evaluation_performed": True,
    }

    class _Workspace:
        def build_telemetry(self):
            return {"required_candidate_pair_count": 1}

    def fake_workspace(  # noqa: ANN001, ANN003
        search_pool,
        *,
        joint_pair_observer,
        **_kwargs,
    ):
        assert [
            str(record["candidate_label"]) for record in search_pool
        ] == ["child-0", "child-1"]
        assert joint_pair_observer is not None
        joint_pair_observer(marker)
        return _Workspace()

    monkeypatch.setattr(
        scoring_mod,
        "_build_batch_full_geometry_workspace",
        fake_workspace,
    )

    def fake_evaluator(subset, *, mode, **_kwargs):  # noqa: ANN001, ANN003
        copied = tuple(dict(record) for record in subset)
        score = float(sum(record["phase2_raw_score"] for record in copied))
        return scoring_mod.BatchSelectionProposal(
            records=copied,
            summary={
                "selection_mode": str(mode),
                "feasible": True,
                "joint_gain": float(score),
                "contextual_single_total": float(score),
                "additivity_defect": 0.0,
            },
            score=float(score),
            delta_e3=float(score),
            k3=0.0,
            denominator_1_plus_k3=1.0,
        )

    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_ordered_reduced_plane_batch_proposal",
        fake_evaluator,
    )
    _proposals, summary = scoring_mod.combinatorial_reduced_plane_batch_proposals(
        records,
        cfg=FullScoreConfig(
            batch_selection_mode="combinatorial_reduced_plane",
            batch_target_size=2,
            batch_size_cap=2,
            batch_search_pool_size=2,
            batch_search_population_mode="ranked_child_phase2_v1",
            batch_search_feasibility_policy="joint_subset_gate_v1",
            batch_additivity_policy="off",
            batch_geometry_mode="full_residual_gram_hessian_v1",
            batch_joint_context_mode="full_ansatz_v1",
        ),
        selected_ops=[object()],
        theta=np.zeros(1, dtype=float),
        psi_ref=np.ones(1, dtype=complex),
        psi_state=np.ones(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
        joint_pair_observer=lambda payload: observed.append(dict(payload)),
    )

    assert observed == [marker]
    assert summary["geometry_workspace"] == {
        "required_candidate_pair_count": 1
    }


def test_soft_additivity_penalizes_without_rejecting_and_legacy_gate_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _tetris_record(
            candidate_label=f"child-{index}",
            pauli_label=("xe", "ye")[index],
            full_v2_score=1.0,
            candidate_pool_index=index,
        )
        for index in range(2)
    ]

    def fake_geometry(*_args, **kwargs):  # noqa: ANN002, ANN003
        cfg = kwargs["cfg"]
        if (
            cfg.batch_additivity_policy == "hard_gate_legacy_v1"
            and 0.5 > cfg.batch_additivity_tol
        ):
            return {
                "feasible": False,
                "reason": "additivity_hard_gate_legacy",
                "joint_gain": 2.0,
                "contextual_single_total": 4.0,
                "additivity_defect": 0.5,
            }
        return {
            "feasible": True,
            "joint_gain": 2.0,
            "contextual_single_total": 4.0,
            "additivity_defect": 0.5,
        }

    monkeypatch.setattr(scoring_mod, "_batch_geometry_summary", fake_geometry)
    base_kwargs = dict(
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.ones(1, dtype=complex),
        psi_state=np.ones(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
        mode="combinatorial_reduced_plane",
    )
    soft = scoring_mod._evaluate_ordered_reduced_plane_batch_proposal(
        records,
        cfg=FullScoreConfig(
            batch_additivity_policy="soft_penalty_v1",
            batch_additivity_lambda=2.0,
        ),
        **base_kwargs,
    )
    unpenalized = scoring_mod._evaluate_ordered_reduced_plane_batch_proposal(
        records,
        cfg=FullScoreConfig(
            batch_additivity_policy="soft_penalty_v1",
            batch_additivity_lambda=0.0,
        ),
        **base_kwargs,
    )
    hard = scoring_mod._evaluate_ordered_reduced_plane_batch_proposal(
        records,
        cfg=FullScoreConfig(
            batch_additivity_policy="hard_gate_legacy_v1",
            batch_additivity_tol=0.25,
        ),
        **base_kwargs,
    )

    assert soft is not None
    assert unpenalized is not None
    assert soft.score == pytest.approx(unpenalized.score / 2.0)
    assert soft.summary["additivity_defect"] == pytest.approx(0.5)
    assert hard is None


def _joint_geometry_fixture() -> tuple[
    np.ndarray,
    list[dict[str, object]],
    object,
    FullScoreConfig,
]:
    rng = np.random.default_rng(123)
    psi = np.asarray(
        rng.normal(size=4) + 1.0j * rng.normal(size=4),
        dtype=complex,
    )
    psi = psi / np.linalg.norm(psi)
    cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_weight=0.0,
    )
    simple = SimpleScoreConfig(
        lambda_compile=0.0,
        lambda_measure=0.0,
        lambda_leak=0.0,
        z_alpha=0.0,
    )
    records: list[dict[str, object]] = []
    h_compiled = None
    for index, label in enumerate(("xe", "ye")):
        record, h_compiled = _full_record(
            label=label,
            candidate_label=label,
            candidate_pool_index=index,
            gradient_signed=0.5,
            psi_state=psi,
            selected_ops=[],
            theta=[],
            refit_window_indices=[],
            full_cfg=cfg,
            simple_cfg=simple,
            h_label="xe",
            include_phase3=False,
        )
        record["phase2_raw_score"] = float(2 - index)
        records.append(record)
    assert h_compiled is not None
    return psi, records, h_compiled, cfg


def test_phase2_joint_response_ignores_legacy_novelty_as_authority() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    base = records[0]
    feature = base["feature"]
    low_novelty = {
        **base,
        "phase2_raw_score": 1e-12,
        "feature": replace(
            feature,
            phase2_raw_novelty=1e-12,
            phase2_raw_score=1e-12,
        ),
    }
    high_novelty = {
        **base,
        "phase2_raw_score": 1e6,
        "feature": replace(
            feature,
            phase2_raw_novelty=1e6,
            phase2_raw_score=1e6,
        ),
    }

    def _evaluate(record):  # noqa: ANN202
        return scoring_mod.evaluate_phase2_joint_response_singletons(
            [record],
            cfg=cfg,
            selected_ops=[],
            theta=np.zeros(0, dtype=float),
            psi_ref=psi,
            psi_state=psi,
            h_compiled=h_compiled,
            pauli_action_cache={},
            scope="child_phase2",
        )

    low = _evaluate(low_novelty)
    high = _evaluate(high_novelty)

    assert low.records[0]["phase2_raw_score"] == pytest.approx(
        high.records[0]["phase2_raw_score"],
        abs=1e-12,
    )
    assert low.records[0]["phase2_legacy_product_score"] == pytest.approx(1e-12)
    assert high.records[0]["phase2_legacy_product_score"] == pytest.approx(1e6)
    assert low.records[0]["phase2_novelty_authority"] == "telemetry_only"
    assert low.telemetry["candidate_pair_measurement_count"] == 0
    assert low.telemetry["geometry_workspace"][
        "query_chargeable_unique_geometry_element_count"
    ] == 0


def test_phase2_joint_response_rejects_active_span_collapse() -> None:
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    selected = [_term("x")]
    theta = np.asarray([0.0], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        batch_joint_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
    )
    candidate, h_compiled = _full_record(
        label="x",
        candidate_label="duplicate-x",
        candidate_pool_index=0,
        gradient_signed=0.5,
        psi_state=psi_state,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
        include_phase3=False,
    )

    evaluation = scoring_mod.evaluate_phase2_joint_response_singletons(
        [candidate],
        cfg=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        pauli_action_cache={},
        scope="child_phase2",
    )

    assert evaluation.records[0]["phase2_joint_response_feasible"] is False
    assert evaluation.records[0]["phase2_raw_score"] == float("-inf")
    assert evaluation.records[0]["phase2_joint_response"]["reason"] == "rank_gate"
    assert evaluation.telemetry["infeasible_reason_counts"] == {"rank_gate": 1}


def test_phase2_joint_response_and_bmax1_share_singleton_evaluator() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    evaluation = scoring_mod.evaluate_phase2_joint_response_singletons(
        records,
        cfg=cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        pauli_action_cache={},
        scope="child_phase2",
    )
    proposals, summary = select_route_a_schur_proposals(
        evaluation.records,
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        ),
        score_config=cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
    )
    expected = max(
        evaluation.records,
        key=lambda record: float(record["phase2_raw_score"]),
    )

    assert len(proposals) == 1
    assert proposals[0].records[0]["candidate_pool_index"] == expected[
        "candidate_pool_index"
    ]
    assert proposals[0].score == pytest.approx(
        expected["phase2_joint_response_score"],
        abs=1e-12,
    )
    assert proposals[0].summary["joint_linear_solve_policy_effective"] == (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
    )
    assert summary["geometry_workspace"][
        "query_chargeable_unique_geometry_element_count"
    ] == 0
    phase2_reuse = evaluation.telemetry["geometry_workspace"][
        "phase2_joint_geometry_reuse_validation"
    ]
    final_reuse = summary["geometry_workspace"][
        "phase2_joint_geometry_reuse_validation"
    ]
    for key in (
        "state_fingerprint",
        "ordered_scaffold_fingerprint",
        "theta_fingerprint",
    ):
        assert final_reuse[key] == phase2_reuse[key]


def test_phase2_joint_response_then_bmax2_charges_only_pair_off_diagonals() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    evaluation = scoring_mod.evaluate_phase2_joint_response_singletons(
        records,
        cfg=cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        pauli_action_cache={},
        scope="child_phase2",
    )
    proposals, summary = select_route_a_schur_proposals(
        evaluation.records,
        config=RouteASchurSelectorConfig(
            batch_size_cap=2,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        ),
        score_config=cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        max_proposals=3,
    )
    workspace = summary["geometry_workspace"]

    assert evaluation.telemetry["geometry_workspace"][
        "query_chargeable_unique_geometry_element_count"
    ] == 0
    assert workspace["query_chargeable_unique_geometry_element_count"] == 2
    assert workspace["newly_measured_element_counts"]["G_CC_off_diagonal"] == 1
    assert workspace["newly_measured_element_counts"]["H_CC_off_diagonal"] == 1
    assert workspace["reused_phase2_element_count"] > 0
    assert proposals
    assert all(
        proposal.summary["joint_linear_solve_policy_effective"]
        == JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
        for proposal in proposals
    )
    assert all(
        proposal.summary["classical_quantum_query_charge"] == 0
        for proposal in proposals
    )


def test_typed_joint_response_evaluator_reuses_active_blocks() -> None:
    rng = np.random.default_rng(20260711)
    psi_ref = np.asarray(
        rng.normal(size=4) + 1.0j * rng.normal(size=4),
        dtype=complex,
    )
    psi_ref /= np.linalg.norm(psi_ref)
    selected = [_term("xe")]
    theta = np.asarray([0.0], dtype=float)
    psi_state = np.asarray(psi_ref, dtype=complex)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    candidate, h_compiled = _full_record(
        label="ex",
        candidate_label="independent-ex",
        candidate_pool_index=0,
        gradient_signed=0.8,
        psi_state=psi_state,
        selected_ops=selected,
        theta=[float(theta[0])],
        refit_window_indices=[0],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
        h_label="zx",
        include_phase3=False,
    )
    evaluation = RouteAJointResponseEvaluator(
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        ),
        score_config=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        pauli_action_cache={},
        scope="child_phase2",
    )([candidate])
    row = evaluation.records[0]
    workspace = evaluation.telemetry["geometry_workspace"]

    assert row["phase2_joint_response_feasible"] is True
    assert row["phase2_joint_response_gain"] > 0.0
    assert row["phase2_raw_score"] == pytest.approx(
        row["phase2_joint_response_gain"]
        / row["phase2_joint_response"]["denominator_1_plus_K2"],
        abs=1e-12,
    )
    assert row["phase2_joint_response"]["active_parameter_relaxation"]
    assert workspace["active_indices"] == [0]
    assert workspace["phase2_joint_geometry_reuse_validation"][
        "valid_record_count"
    ] == 1
    assert workspace["query_chargeable_unique_geometry_element_count"] == 0


def test_phase2_joint_response_ordering_audit_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {"candidate_label": "legacy-first", "candidate_pool_index": 0, "position_id": 0, "phase2_raw_score": 10.0},
        {"candidate_label": "joint-first", "candidate_pool_index": 1, "position_id": 0, "phase2_raw_score": 1.0},
    ]

    class _Workspace:
        active_indices = ()

        def summary_for_records(self, subset):  # noqa: ANN001
            gain = 1.0 if subset[0]["candidate_label"] == "legacy-first" else 5.0
            return {
                "feasible": True,
                "reason": "fixture",
                "joint_gain": gain,
                "subset_workspace_indices": [int(subset[0]["candidate_pool_index"])],
            }

        def build_telemetry(self):  # noqa: ANN201
            return {
                "validated_phase2_gradient_reuse_count": 2,
                "query_chargeable_unique_geometry_element_count": 0,
            }

    monkeypatch.setattr(
        scoring_mod,
        "_build_batch_full_geometry_workspace",
        lambda *_args, **_kwargs: _Workspace(),
    )
    kwargs = dict(
        cfg=FullScoreConfig(),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.ones(1, dtype=complex),
        psi_state=np.ones(1, dtype=complex),
        h_compiled=object(),
        scope="ordering_audit",
    )
    first = scoring_mod.evaluate_phase2_joint_response_singletons(records, **kwargs)
    second = scoring_mod.evaluate_phase2_joint_response_singletons(records, **kwargs)

    legacy_order = [
        record["candidate_label"]
        for record in sorted(
            records,
            key=lambda record: -float(record["phase2_raw_score"]),
        )
    ]
    joint_order_first = [
        record["candidate_label"]
        for record in sorted(
            first.records,
            key=lambda record: -float(record["phase2_raw_score"]),
        )
    ]
    joint_order_second = [
        record["candidate_label"]
        for record in sorted(
            second.records,
            key=lambda record: -float(record["phase2_raw_score"]),
        )
    ]
    assert legacy_order == ["legacy-first", "joint-first"]
    assert joint_order_first == ["joint-first", "legacy-first"]
    assert joint_order_second == joint_order_first


def _pair_proposal(proposals):  # noqa: ANN001 - compact test helper
    return next(proposal for proposal in proposals if len(proposal.records) == 2)


@pytest.mark.parametrize("insertion_position", [0, 1, 2])
def test_joint_workspace_matches_finite_differences_at_every_insertion_region(
    insertion_position: int,
) -> None:
    rng = np.random.default_rng(20260711)
    psi_ref = np.asarray(
        rng.normal(size=4) + 1.0j * rng.normal(size=4),
        dtype=complex,
    )
    psi_ref /= np.linalg.norm(psi_ref)
    selected = [_term("xe"), _term("yz")]
    candidate = _term("zx")
    theta = np.asarray([0.17, -0.23], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    hamiltonian = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="ze", pc=0.7),
            PauliTerm(2, ps="ex", pc=-0.4),
            PauliTerm(2, ps="xx", pc=0.3),
        ],
    )
    h_compiled = compile_polynomial_action(
        hamiltonian,
        pauli_action_cache={},
    )
    record = {
        "candidate_label": f"candidate-zx-p{insertion_position}",
        "candidate_pool_index": 0,
        "position_id": int(insertion_position),
        "candidate_term": candidate,
        "phase2_raw_score": 1.0,
    }
    workspace = scoring_mod._build_batch_full_geometry_workspace(
        [record],
        cfg=FullScoreConfig(
            batch_joint_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
            batch_state_consistency_tolerance=1e-8,
        ),
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        pauli_action_cache={},
    )

    combined_terms = [
        *selected[:insertion_position],
        candidate,
        *selected[insertion_position:],
    ]
    executor = CompiledAnsatzExecutor(combined_terms)

    def _state(z: np.ndarray) -> np.ndarray:
        selected_values = theta + np.asarray(z[:2], dtype=float)
        combined_values = np.asarray(
            [
                *selected_values[:insertion_position],
                float(z[2]),
                *selected_values[insertion_position:],
            ],
            dtype=float,
        )
        return executor.prepare_state(combined_values, psi_ref)

    def _energy(z: np.ndarray) -> float:
        state = _state(z)
        return float(
            np.real(np.vdot(state, apply_compiled_polynomial(state, h_compiled)))
        )

    zero = np.zeros(3, dtype=float)
    state_step = 1e-6
    state_derivatives = []
    for index in range(3):
        step = np.zeros(3, dtype=float)
        step[index] = state_step
        state_derivatives.append(
            (_state(step) - _state(-step)) / (2.0 * state_step)
        )
    finite_gram = np.zeros((3, 3), dtype=float)
    psi_zero = _state(zero)
    for row in range(3):
        for col in range(3):
            finite_gram[row, col] = float(
                np.real(
                    np.vdot(state_derivatives[row], state_derivatives[col])
                    - np.vdot(state_derivatives[row], psi_zero)
                    * np.vdot(psi_zero, state_derivatives[col])
                )
            )

    energy_step = 2e-4
    energy_zero = _energy(zero)
    finite_gradient = np.zeros(3, dtype=float)
    finite_hessian = np.zeros((3, 3), dtype=float)
    for row in range(3):
        step = np.zeros(3, dtype=float)
        step[row] = energy_step
        energy_plus = _energy(step)
        energy_minus = _energy(-step)
        finite_gradient[row] = (energy_plus - energy_minus) / (
            2.0 * energy_step
        )
        finite_hessian[row, row] = (
            energy_plus - 2.0 * energy_zero + energy_minus
        ) / (energy_step**2)
        for col in range(row):
            step_col = np.zeros(3, dtype=float)
            step_col[col] = energy_step
            mixed = (
                _energy(step + step_col)
                - _energy(step - step_col)
                - _energy(-step + step_col)
                + _energy(-step - step_col)
            ) / (4.0 * energy_step**2)
            finite_hessian[row, col] = mixed
            finite_hessian[col, row] = mixed

    workspace_gram = np.block(
        [
            [workspace.G_AA, workspace.G_AB],
            [workspace.G_AB.T, workspace.G_BB],
        ]
    )
    workspace_hessian = np.block(
        [
            [workspace.H_AA, workspace.H_AB],
            [workspace.H_AB.T, workspace.H_BB],
        ]
    )
    workspace_descent_gradient = np.concatenate(
        [workspace.g_A, workspace.g_B]
    )

    assert workspace_gram == pytest.approx(finite_gram, abs=2e-7)
    assert workspace_hessian == pytest.approx(finite_hessian, abs=2e-6)
    assert workspace_descent_gradient == pytest.approx(
        -finite_gradient,
        abs=2e-7,
    )


def test_full_joint_geometry_builds_once_and_offdiagonal_hessian_changes_gain() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    common = dict(
        batch_size_cap=2,
        batch_search_pool_size=0,
        additivity_policy=ROUTE_A_ADDITIVITY_OFF,
        joint_batch_context_mode=BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1,
    )
    full_proposals, full_summary = select_route_a_schur_proposals(
        records,
        config=RouteASchurSelectorConfig(
            **common,
            geometry_mode=BATCH_GEOMETRY_FULL_RESIDUAL_GRAM_HESSIAN_V1,
        ),
        score_config=cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        max_proposals=3,
    )
    diagonal_proposals, _diagonal_summary = select_route_a_schur_proposals(
        records,
        config=RouteASchurSelectorConfig(
            **common,
            geometry_mode=BATCH_GEOMETRY_DIAGONAL_HESSIAN_DIAGNOSTIC_V1,
        ),
        score_config=cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        max_proposals=3,
    )
    full_pair = _pair_proposal(full_proposals)
    diagonal_pair = _pair_proposal(diagonal_proposals)

    assert abs(float(full_pair.summary["G_BB_raw"][0][1])) > 1e-6
    assert abs(float(full_pair.summary["H_BB_raw"][0][1])) > 1e-6
    assert float(diagonal_pair.summary["H_BB_raw"][0][1]) == pytest.approx(0.0)
    assert full_pair.delta_e3 != pytest.approx(diagonal_pair.delta_e3)
    workspace = full_summary["geometry_workspace"]
    assert workspace["schema"] == "batch_full_geometry_workspace_v1"
    assert workspace["full_geometry_workspace_build_count"] == 1
    assert workspace["query_chargeable_unique_geometry_element_count"] == 2
    assert workspace["required_candidate_pair_count"] == 1
    assert workspace["constructed_candidate_pair_count"] == 1
    assert workspace["required_element_counts"]["G_CC_off_diagonal"] == 1
    assert workspace["required_element_counts"]["H_CC_off_diagonal"] == 1
    assert workspace["newly_measured_element_counts"]["G_CC_off_diagonal"] == 1
    assert workspace["newly_measured_element_counts"]["H_CC_off_diagonal"] == 1
    assert workspace["workspace_build_mode"] == (
        "phase2_reuse_plus_required_candidate_pairs_v1"
    )
    assert full_summary["subset_evaluation_count"] == 3


def test_phase2_exact_insertion_reuses_shared_active_blocks_for_every_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scoring_mod._propagate_executor_derivatives
    dense_calls: list[tuple[int, ...]] = []

    def _counted_dense_propagation(**kwargs):  # noqa: ANN003, ANN202
        dense_calls.append(tuple(int(x) for x in kwargs["active_indices"]))
        return original(**kwargs)

    monkeypatch.setattr(
        scoring_mod,
        "_propagate_executor_derivatives",
        _counted_dense_propagation,
    )
    psi = np.asarray([1.0, 1.0j], dtype=complex)
    psi /= np.linalg.norm(psi)
    scaffold_context, h_compiled = _scaffold_context(
        psi_state=psi,
        selected_ops=[_term("z")],
        theta=[0.0],
        refit_window_indices=[0],
        h_label="x",
    )

    payloads = [
        scoring_mod._exact_insertion_joint_geometry_payload(
            scaffold_context=scaffold_context,
            candidate_term=_term(label),
            position_id=1,
            h_compiled=h_compiled,
            pauli_action_cache={},
            state_consistency_tolerance=1e-8,
        )
        for label in ("x", "y")
    ]

    assert dense_calls == [(0,)]
    assert all(
        payload["active_block_source"] == "shared_scaffold_context_v1"
        for payload in payloads
    )
    assert all(payload["active_block_recomputed"] is False for payload in payloads)
    assert all(
        payload["sparse_second_derivative_pair_count"] == 2
        for payload in payloads
    )
    assert all(
        payload["dense_second_derivative_entry_count_avoided"] == 2
        for payload in payloads
    )
    assert np.asarray(payloads[0]["G_AA"]) == pytest.approx(
        np.asarray(payloads[1]["G_AA"])
    )
    assert np.asarray(payloads[0]["H_AA"]) == pytest.approx(
        np.asarray(payloads[1]["H_AA"])
    )


def _outer_old_old_prior_fixture():  # noqa: ANN202 - compact test fixture
    psi_ref = np.asarray([0.8 + 0.1j, -0.2 + 0.5j], dtype=complex)
    psi_ref /= np.linalg.norm(psi_ref)
    selected = [_term("z")]
    theta = np.asarray([0.17], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    h_compiled = compile_polynomial_action(
        _poly("x"), pauli_action_cache={}
    )
    prior = scoring_mod.build_historical_singleton_old_old_geometry_prior(
        selected_ops=selected,
        active_indices=(0,),
        theta=theta,
        psi_state=psi_state,
        h_compiled=h_compiled,
        G_AA=np.asarray([[1.75]], dtype=float),
        H_AA=np.asarray([[0.42]], dtype=float),
        g_A=np.asarray([-0.33], dtype=float),
        source_prior_id="outer-prior",
        source_state_id="outer-state",
        source_frame_id="outer-frame",
        source_support_id="outer-support",
        source_geometry_status="predicted",
        source_provenance_ids=("outer-provenance",),
    )
    record = {
        "candidate_label": "candidate-y",
        "candidate_pool_index": 0,
        "position_id": 1,
        "candidate_term": _term("y"),
        "phase2_raw_score": 1.0,
    }
    cfg = FullScoreConfig(
        batch_joint_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        batch_state_consistency_tolerance=1e-8,
    )
    return psi_ref, selected, theta, psi_state, h_compiled, prior, record, cfg


def test_historical_old_old_prior_skips_old_old_measurement_but_acquires_candidate_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        psi_ref,
        selected,
        theta,
        psi_state,
        h_compiled,
        prior,
        record,
        cfg,
    ) = _outer_old_old_prior_fixture()

    def _forbid_old_old_gram(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("old--old Gram matrix was reacquired")

    def _forbid_dense_old_old_derivatives(  # noqa: ANN202
        *_args, **_kwargs  # noqa: ANN002, ANN003
    ):
        raise AssertionError("dense old--old derivatives were reconstructed")

    original_hessian_entry = scoring_mod._energy_hessian_entry
    hessian_entry_calls = 0

    def _count_hessian_entries(**kwargs):  # noqa: ANN003, ANN202
        nonlocal hessian_entry_calls
        hessian_entry_calls += 1
        return original_hessian_entry(**kwargs)

    monkeypatch.setattr(
        scoring_mod,
        "_tangent_overlap_matrix",
        _forbid_old_old_gram,
    )
    monkeypatch.setattr(
        scoring_mod,
        "_propagate_executor_derivatives",
        _forbid_dense_old_old_derivatives,
    )
    monkeypatch.setattr(
        scoring_mod,
        "_energy_hessian_entry",
        _count_hessian_entries,
    )
    workspace = scoring_mod._build_batch_full_geometry_workspace(
        [record],
        cfg=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        pauli_action_cache={},
        old_old_geometry_prior=prior,
    )

    # One active/candidate mixed entry is symmetrized from its two orderings,
    # and one candidate diagonal is acquired.  No active/active Hessian entry
    # is evaluated.
    assert hessian_entry_calls == 3
    assert workspace.G_AA == pytest.approx(np.asarray([[1.75]]))
    assert workspace.H_AA == pytest.approx(np.asarray([[0.42]]))
    assert workspace.g_A == pytest.approx([-0.33])
    assert np.all(np.isfinite(workspace.G_AB))
    assert np.all(np.isfinite(workspace.H_AB))
    assert np.all(np.isfinite(workspace.G_BB))
    assert np.all(np.isfinite(workspace.H_BB))
    assert np.all(np.isfinite(workspace.g_B))

    telemetry = workspace.build_telemetry()
    prior_telemetry = telemetry["old_old_geometry_prior"]
    assert telemetry["workspace_build_mode"] == (
        "outer_information_prior_plus_exact_candidate_repairs_v1"
    )
    assert telemetry["old_old_geometry_reacquired"] is False
    assert prior_telemetry["prior_fingerprint"] == prior.prior_fingerprint
    assert prior_telemetry["old_old_metric_element_count_acquired"] == 0
    assert (
        prior_telemetry[
            "old_old_direct_curvature_element_count_acquired"
        ]
        == 0
    )
    assert telemetry["reused_element_counts"]["G_AA"] == 1
    assert telemetry["reused_element_counts"]["H_AA"] == 1
    assert telemetry["newly_measured_element_counts"]["G_AC"] == 1
    assert telemetry["newly_measured_element_counts"]["H_AC"] == 1
    anchor_payload = workspace.old_old_geometry_payload()
    assert np.asarray(anchor_payload["G_AA"]) == pytest.approx(
        np.asarray([[1.75]])
    )
    assert np.asarray(anchor_payload["H_AA"]) == pytest.approx(
        np.asarray([[0.42]])
    )
    assert anchor_payload["g_A"] == pytest.approx([-0.33])
    assert anchor_payload["gradient_convention"] == "sr_descent_gradient_v1"


def test_historical_old_old_prior_fingerprint_mismatch_fails_before_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        psi_ref,
        selected,
        theta,
        psi_state,
        h_compiled,
        prior,
        record,
        cfg,
    ) = _outer_old_old_prior_fixture()
    measured = False

    def _unexpected_measurement(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal measured
        measured = True
        raise AssertionError("measurement occurred before prior validation")

    monkeypatch.setattr(
        scoring_mod,
        "_selector_scaffold_context",
        _unexpected_measurement,
    )
    with pytest.raises(ValueError, match="theta_fingerprint"):
        scoring_mod._build_batch_full_geometry_workspace(
            [record],
            cfg=cfg,
            selected_ops=selected,
            theta=np.asarray([theta[0] + 0.01], dtype=float),
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            pauli_action_cache={},
            old_old_geometry_prior=prior,
        )
    assert measured is False


def test_historical_workspace_omitted_prior_and_explicit_none_are_identical() -> None:
    (
        psi_ref,
        selected,
        theta,
        psi_state,
        h_compiled,
        _prior,
        record,
        cfg,
    ) = _outer_old_old_prior_fixture()
    common = dict(
        records=[record],
        cfg=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
    )
    omitted = scoring_mod._build_batch_full_geometry_workspace(
        **common,
        pauli_action_cache={},
    )
    explicit_none = scoring_mod._build_batch_full_geometry_workspace(
        **common,
        pauli_action_cache={},
        old_old_geometry_prior=None,
    )
    for field_name in (
        "G_AA",
        "H_AA",
        "G_AB",
        "H_AB",
        "G_BB",
        "H_BB",
        "g_A",
        "g_B",
    ):
        assert getattr(omitted, field_name) == pytest.approx(
            getattr(explicit_none, field_name), abs=0.0, rel=0.0
        )
    assert omitted.workspace_fingerprint == explicit_none.workspace_fingerprint
    assert omitted.build_telemetry() == explicit_none.build_telemetry()


def test_grouped_exact_dense_and_sparse_derivatives_match_state_finite_differences() -> None:
    grouped = type(
        "_GroupedAnsatzTerm",
        (),
        {
            "label": "grouped_x_plus_z",
            "polynomial": PauliPolynomial(
                "JW",
                [
                    PauliTerm(1, ps="x", pc=0.7),
                    PauliTerm(1, ps="z", pc=-0.4),
                ],
            ),
            "execution_mode": "grouped_exact",
        },
    )()
    terms = [_term("y"), grouped]
    executor = CompiledAnsatzExecutor(terms)
    psi_ref = np.asarray([0.6 + 0.2j, -0.3 + 0.7j], dtype=complex)
    psi_ref /= np.linalg.norm(psi_ref)
    theta = np.asarray([0.23, -0.41], dtype=float)

    dense_state, dense_first, dense_second = (
        scoring_mod._propagate_executor_derivatives(
            executor=executor,
            theta=theta,
            psi_ref=psi_ref,
            active_indices=[0, 1],
        )
    )
    sparse_state, sparse_first, sparse_second = (
        scoring_mod._propagate_executor_sparse_second_derivatives(
            executor=executor,
            theta=theta,
            psi_ref=psi_ref,
            active_indices=[0, 1],
            second_derivative_pairs=[(0, 1), (1, 1)],
        )
    )
    expected_state = executor.prepare_state(theta, psi_ref)
    assert dense_state == pytest.approx(expected_state, abs=1e-12)
    assert sparse_state == pytest.approx(expected_state, abs=1e-12)

    first_step = 2.0e-6
    second_step = 2.0e-4

    def _state_at(values: np.ndarray) -> np.ndarray:
        return np.asarray(executor.prepare_state(values, psi_ref), dtype=complex)

    finite_first: list[np.ndarray] = []
    finite_diag: list[np.ndarray] = []
    for index in range(2):
        direction = np.zeros(2, dtype=float)
        direction[index] = first_step
        finite_first.append(
            (_state_at(theta + direction) - _state_at(theta - direction))
            / (2.0 * first_step)
        )
        direction[index] = second_step
        finite_diag.append(
            (
                _state_at(theta + direction)
                - 2.0 * expected_state
                + _state_at(theta - direction)
            )
            / (second_step**2)
        )
    direction_left = np.asarray([second_step, 0.0], dtype=float)
    direction_right = np.asarray([0.0, second_step], dtype=float)
    finite_mixed = (
        _state_at(theta + direction_left + direction_right)
        - _state_at(theta + direction_left - direction_right)
        - _state_at(theta - direction_left + direction_right)
        + _state_at(theta - direction_left - direction_right)
    ) / (4.0 * second_step**2)

    for index in range(2):
        assert dense_first[index] == pytest.approx(
            finite_first[index],
            abs=3e-8,
        )
        assert sparse_first[index] == pytest.approx(
            finite_first[index],
            abs=3e-8,
        )
        assert dense_second[index][index] == pytest.approx(
            finite_diag[index],
            abs=3e-7,
        )
    assert dense_second[0][1] == pytest.approx(finite_mixed, abs=3e-7)
    assert dense_second[1][0] == pytest.approx(finite_mixed, abs=3e-7)
    assert sparse_second[(0, 1)] == pytest.approx(finite_mixed, abs=3e-7)
    assert sparse_second[(1, 1)] == pytest.approx(finite_diag[1], abs=3e-7)

    h_compiled = compile_polynomial_action(_poly("z"), pauli_action_cache={})
    hpsi_state = apply_compiled_polynomial(expected_state, h_compiled)
    scaffold_context = OrderedInsertionGeometryOracle().prepare_scaffold_context(
        selected_ops=terms,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=expected_state,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[0, 1],
        pauli_action_cache={},
    )
    insertion = scoring_mod._exact_insertion_joint_geometry_payload(
        scaffold_context=scaffold_context,
        candidate_term=_term("x"),
        position_id=1,
        h_compiled=h_compiled,
        pauli_action_cache={},
        state_consistency_tolerance=1e-10,
    )
    assert insertion["state_reconstruction_delta_norm"] < 1e-12


def test_phase2_shared_active_blocks_are_global_phase_invariant() -> None:
    psi_ref = np.asarray([1.0, 1.0j], dtype=complex)
    psi_ref /= np.linalg.norm(psi_ref)
    selected_ops = [_term("z")]
    theta = np.asarray([0.0], dtype=float)
    h_compiled = compile_polynomial_action(_poly("x"), pauli_action_cache={})

    def _context(phase: complex):
        psi_state = np.asarray(phase * psi_ref, dtype=complex)
        hpsi_state = apply_compiled_polynomial(psi_state, h_compiled)
        return OrderedInsertionGeometryOracle().prepare_scaffold_context(
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            hpsi_state=hpsi_state,
            refit_window_indices=[0],
            pauli_action_cache={},
        )

    baseline = _context(1.0 + 0.0j)
    phased = _context(np.exp(0.73j))
    baseline_payload = scoring_mod._exact_insertion_joint_geometry_payload(
        scaffold_context=baseline,
        candidate_term=_term("x"),
        position_id=1,
        h_compiled=h_compiled,
        pauli_action_cache={},
        state_consistency_tolerance=1e-8,
    )
    phased_payload = scoring_mod._exact_insertion_joint_geometry_payload(
        scaffold_context=phased,
        candidate_term=_term("x"),
        position_id=1,
        h_compiled=h_compiled,
        pauli_action_cache={},
        state_consistency_tolerance=1e-8,
    )

    for key in ("G_AA", "G_AB", "G_BB", "H_AA", "H_AB", "H_BB", "g_A"):
        assert np.asarray(phased_payload[key]) == pytest.approx(
            np.asarray(baseline_payload[key]),
            abs=1e-12,
        )
    assert phased_payload["descent_gradient"] == pytest.approx(
        baseline_payload["descent_gradient"],
        abs=1e-12,
    )


def test_scaffold_context_can_retain_exact_metric_with_predicted_hessian() -> None:
    psi_ref = np.asarray([1.0, 1.0j], dtype=complex)
    psi_ref /= np.linalg.norm(psi_ref)
    selected_ops = [_term("z")]
    theta = np.asarray([0.17], dtype=float)
    h_compiled = compile_polynomial_action(_poly("x"), pauli_action_cache={})
    executor = scoring_mod._executor_for_terms(
        selected_ops, pauli_action_cache={}
    )
    psi_state = np.asarray(executor.prepare_state(theta, psi_ref), dtype=complex)
    hpsi_state = apply_compiled_polynomial(psi_state, h_compiled)
    oracle = OrderedInsertionGeometryOracle()
    exact = oracle.prepare_scaffold_context(
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[0],
        pauli_action_cache={},
    )
    predicted_hessian = np.asarray([[3.25]], dtype=float)
    provenance = {
        "schema": "formal_manifold_hessian_prior_provenance_v1",
        "source_prior_id": "prior-r4",
        "source_state_id": "state-r4",
        "source_frame_id": "frame-r4",
        "source_support_id": "support-r4",
        "source_curvature_id": "curvature-r4",
        "transport": {"provenance_ids": ["transport-r4-r5"]},
    }
    mixed = oracle.prepare_scaffold_context(
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[0],
        pauli_action_cache={},
        old_old_hessian_prior=predicted_hessian,
        old_old_hessian_prior_provenance=provenance,
        old_old_hessian_prior_status="predicted_transport_closed_v1",
    )
    changed = oracle.prepare_scaffold_context(
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[0],
        pauli_action_cache={},
        old_old_hessian_prior=np.asarray([[3.5]], dtype=float),
        old_old_hessian_prior_provenance=provenance,
        old_old_hessian_prior_status="predicted_transport_closed_v1",
    )

    assert mixed.Q_window == pytest.approx(exact.Q_window, abs=1e-14)
    assert mixed.H_window_hessian == pytest.approx(
        predicted_hessian, abs=1e-14
    )
    assert mixed.tangents_window[0] == pytest.approx(
        exact.tangents_window[0], abs=1e-14
    )
    assert mixed.old_old_metric_measured is True
    assert mixed.old_old_hessian_measured is False
    assert mixed.old_old_geometry_measured is False
    assert mixed.old_old_hessian_status == "predicted_transport_closed_v1"
    assert dict(mixed.old_old_hessian_provenance) == provenance
    assert mixed.old_old_hessian_fingerprint
    assert (
        changed.old_old_hessian_fingerprint
        != mixed.old_old_hessian_fingerprint
    )

    candidate_telemetry = oracle.estimate(
        scaffold_context=mixed,
        candidate_label="x",
        candidate_term=_term("x"),
        pauli_action_cache={},
    )["scaffold_old_old_hessian_source"]
    assert candidate_telemetry["status"] == "predicted_transport_closed_v1"
    assert candidate_telemetry["hessian_fingerprint"] == (
        mixed.old_old_hessian_fingerprint
    )
    assert candidate_telemetry["provenance"] == provenance
    json.dumps(candidate_telemetry, sort_keys=True, allow_nan=False)

    scaffold_payload = scoring_mod._exact_insertion_joint_geometry_payload(
        scaffold_context=mixed,
        candidate_term=_term("x"),
        position_id=1,
        h_compiled=h_compiled,
        pauli_action_cache={},
        state_consistency_tolerance=1e-8,
    )
    assert scaffold_payload["scaffold_old_old_hessian_source"] == (
        candidate_telemetry
    )


def test_historical_hh_scaffold_context_matches_unaligned_snapshot_formula() -> None:
    psi_ref = np.zeros(2**8, dtype=complex)
    psi_ref[0b00010001] = 1.0
    psi_ref[0b00100010] = 0.4j
    psi_ref /= np.linalg.norm(psi_ref)
    selected_ops = [_term("eeexxeee")]
    theta = np.asarray([0.31], dtype=float)
    executor = scoring_mod._executor_for_terms(
        selected_ops,
        pauli_action_cache={},
    )
    reconstructed, raw_dpsi, raw_d2psi = (
        scoring_mod._propagate_executor_derivatives(
            executor=executor,
            theta=theta,
            psi_ref=psi_ref,
            active_indices=[0],
        )
    )
    external_phase = np.exp(0.73j)
    psi_state = np.asarray(external_phase * reconstructed, dtype=complex)
    h_compiled = compile_polynomial_action(
        _poly("zeeeeeee"),
        pauli_action_cache={},
    )
    hpsi_state = apply_compiled_polynomial(psi_state, h_compiled)

    historical = OrderedInsertionGeometryOracle().prepare_scaffold_context(
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[0],
        pauli_action_cache={},
        align_reconstructed_global_phase=False,
    )
    aligned = OrderedInsertionGeometryOracle().prepare_scaffold_context(
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        hpsi_state=hpsi_state,
        refit_window_indices=[0],
        pauli_action_cache={},
        align_reconstructed_global_phase=True,
    )

    raw_derivative = np.asarray(raw_dpsi[0], dtype=complex)
    raw_second = np.asarray(raw_d2psi[0][0], dtype=complex)
    manual_tangent = scoring_mod._horizontal_tangent(
        psi_state,
        raw_derivative,
    )
    manual_q = scoring_mod._tangent_overlap_matrix([manual_tangent])
    manual_h_derivative = apply_compiled_polynomial(
        raw_derivative,
        h_compiled,
    )
    manual_hessian = scoring_mod._energy_hessian_entry(
        dpsi_left=raw_derivative,
        dpsi_right=raw_derivative,
        d2psi=raw_second,
        hpsi_state=hpsi_state,
        hdpsi_right=manual_h_derivative,
    )
    overlap = complex(np.vdot(psi_state, reconstructed))
    reconstructed_phase = overlap / abs(overlap)

    assert np.asarray(historical.dpsi_window[0]) == pytest.approx(
        raw_derivative,
        abs=1e-14,
    )
    assert historical.Q_window == pytest.approx(manual_q, abs=1e-14)
    assert historical.H_window_hessian[0, 0] == pytest.approx(
        manual_hessian,
        abs=1e-14,
    )
    assert np.asarray(aligned.dpsi_window[0]) == pytest.approx(
        raw_derivative / reconstructed_phase,
        abs=1e-14,
    )


def test_joint_pair_parallel_evaluation_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    third, _ = _full_record(
        label="ze",
        candidate_label="ze",
        candidate_pool_index=2,
        gradient_signed=0.25,
        psi_state=psi,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
        h_label="xe",
        include_phase3=False,
    )
    third["phase2_raw_score"] = 0.5
    records = [*records, third]
    selector_config = RouteASchurSelectorConfig(
        batch_size_cap=2,
        batch_search_pool_size=0,
        additivity_policy=ROUTE_A_ADDITIVITY_OFF,
        joint_batch_context_mode=(
            BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
        ),
    )

    def _select(workers: int):  # noqa: ANN202
        monkeypatch.setenv("STATIC_ADAPT_JOINT_PAIR_WORKERS", str(workers))
        return select_route_a_schur_proposals(
            records,
            config=selector_config,
            score_config=cfg,
            selected_ops=[],
            theta=np.zeros(0, dtype=float),
            psi_ref=psi,
            psi_state=psi,
            h_compiled=h_compiled,
            novelty_oracle=OrderedInsertionGeometryOracle(),
            curvature_oracle=Phase2CurvatureOracle(),
            compiled_cache={},
            pauli_action_cache={},
            max_proposals=10,
        )

    serial_proposals, serial_summary = _select(1)
    parallel_proposals, parallel_summary = _select(2)
    serial_workspace = serial_summary["geometry_workspace"]
    parallel_workspace = parallel_summary["geometry_workspace"]

    def _proposal_signature(proposals):  # noqa: ANN001, ANN202
        return [
            (
                tuple(
                    str(record["feature"].candidate_label)
                    for record in proposal.records
                ),
                float(proposal.delta_e3),
                float(proposal.score),
            )
            for proposal in proposals
        ]

    assert parallel_workspace["required_candidate_pair_count"] == 3
    assert parallel_workspace["joint_pair_workers_requested"] == 2
    assert parallel_workspace["joint_pair_workers_effective"] == 2
    assert parallel_workspace["joint_pair_parallel_enabled"] is True
    assert parallel_workspace["joint_pair_result_order"] == (
        "deterministic_lookup_measure_commit_order_v1"
    )
    assert parallel_workspace["joint_pair_cache_miss_count"] == 3
    assert parallel_workspace["joint_pair_cache_hit_count"] == 0
    assert np.asarray(parallel_workspace["G_search_shape"]) == pytest.approx(
        np.asarray(serial_workspace["G_search_shape"])
    )
    assert len(serial_proposals) == len(parallel_proposals)
    for serial, parallel in zip(
        _proposal_signature(serial_proposals),
        _proposal_signature(parallel_proposals),
    ):
        assert serial[0] == parallel[0]
        assert serial[1] == pytest.approx(parallel[1], abs=1e-12)
        assert serial[2] == pytest.approx(parallel[2], abs=1e-12)


def test_joint_pair_cache_reuses_measurement_without_double_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    action_cache: dict[str, object] = {}
    monkeypatch.setenv("STATIC_ADAPT_JOINT_PAIR_WORKERS", "1")
    monkeypatch.setenv("STATIC_ADAPT_JOINT_PAIR_CACHE_MAX_ENTRIES", "16")

    def _select():  # noqa: ANN202
        _proposals, summary = select_route_a_schur_proposals(
            records,
            config=RouteASchurSelectorConfig(
                batch_size_cap=2,
                batch_search_pool_size=0,
                additivity_policy=ROUTE_A_ADDITIVITY_OFF,
                joint_batch_context_mode=(
                    BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
                ),
            ),
            score_config=cfg,
            selected_ops=[],
            theta=np.zeros(0, dtype=float),
            psi_ref=psi,
            psi_state=psi,
            h_compiled=h_compiled,
            novelty_oracle=OrderedInsertionGeometryOracle(),
            curvature_oracle=Phase2CurvatureOracle(),
            compiled_cache={},
            pauli_action_cache=action_cache,
            max_proposals=3,
        )
        return summary["geometry_workspace"]

    first = _select()
    second = _select()

    assert first["joint_pair_cache_hit_count"] == 0
    assert first["joint_pair_cache_miss_count"] == 1
    assert first["constructed_candidate_pair_count"] == 1
    assert first["reused_cached_candidate_pair_count"] == 0
    assert first["query_chargeable_unique_geometry_element_count"] == 2
    assert second["joint_pair_cache_hit_count"] == 1
    assert second["joint_pair_cache_miss_count"] == 0
    assert second["constructed_candidate_pair_count"] == 0
    assert second["reused_cached_candidate_pair_count"] == 1
    assert second["query_chargeable_unique_geometry_element_count"] == 0
    assert second["reused_element_counts"]["G_CC_off_diagonal"] == 1
    assert second["reused_element_counts"]["H_CC_off_diagonal"] == 1


def test_joint_pair_parallel_cache_eviction_and_charging_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    third, _ = _full_record(
        label="ze",
        candidate_label="ze",
        candidate_pool_index=2,
        gradient_signed=0.25,
        psi_state=psi,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
        h_label="xe",
        include_phase3=False,
    )
    third["phase2_raw_score"] = 0.5
    records = [*records, third]
    selector_config = RouteASchurSelectorConfig(
        batch_size_cap=2,
        batch_search_pool_size=0,
        additivity_policy=ROUTE_A_ADDITIVITY_OFF,
        joint_batch_context_mode=(
            BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
        ),
    )
    monkeypatch.setenv("STATIC_ADAPT_JOINT_PAIR_CACHE_MAX_ENTRIES", "2")

    def _two_pass(workers: int):  # noqa: ANN202
        monkeypatch.setenv("STATIC_ADAPT_JOINT_PAIR_WORKERS", str(workers))
        action_cache: dict[str, object] = {}
        outputs = []
        for _ in range(2):
            proposals, summary = select_route_a_schur_proposals(
                records,
                config=selector_config,
                score_config=cfg,
                selected_ops=[],
                theta=np.zeros(0, dtype=float),
                psi_ref=psi,
                psi_state=psi,
                h_compiled=h_compiled,
                novelty_oracle=OrderedInsertionGeometryOracle(),
                curvature_oracle=Phase2CurvatureOracle(),
                compiled_cache={},
                pauli_action_cache=action_cache,
                max_proposals=10,
            )
            outputs.append((proposals, summary["geometry_workspace"]))
        return outputs

    serial = _two_pass(1)
    parallel = _two_pass(3)
    for pass_index in (0, 1):
        serial_proposals, serial_workspace = serial[pass_index]
        parallel_proposals, parallel_workspace = parallel[pass_index]
        assert parallel_workspace["joint_pair_cache_hit_count"] == (
            serial_workspace["joint_pair_cache_hit_count"]
        )
        assert parallel_workspace["joint_pair_cache_miss_count"] == (
            serial_workspace["joint_pair_cache_miss_count"]
        )
        assert parallel_workspace[
            "query_chargeable_unique_geometry_element_count"
        ] == serial_workspace["query_chargeable_unique_geometry_element_count"]
        assert [proposal.score for proposal in parallel_proposals] == pytest.approx(
            [proposal.score for proposal in serial_proposals],
            abs=1e-12,
        )
    assert serial[0][1]["joint_pair_cache_miss_count"] == 3
    assert serial[1][1]["joint_pair_cache_miss_count"] == 1
    assert serial[1][1]["joint_pair_cache_hit_count"] == 2


def test_joint_pair_cache_context_changes_with_state_and_scaffold() -> None:
    psi, _records, h_compiled, _cfg = _joint_geometry_fixture()
    base = scoring_mod._joint_pair_geometry_context_fingerprint(
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
    )
    changed_state = np.asarray(psi, dtype=complex).copy()
    changed_state[0] += 1e-7
    state_key = scoring_mod._joint_pair_geometry_context_fingerprint(
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=changed_state,
        h_compiled=h_compiled,
    )
    scaffold_key = scoring_mod._joint_pair_geometry_context_fingerprint(
        selected_ops=[_term("xe")],
        theta=np.asarray([0.0], dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
    )

    assert state_key != base
    assert scaffold_key != base


def test_ordered_scaffold_fingerprint_includes_execution_mode() -> None:
    termwise = _term("x")
    grouped = _term("x")
    termwise.execution_mode = "termwise_product"
    grouped.execution_mode = "grouped_exact"

    assert scoring_mod._ordered_scaffold_fingerprint([termwise]) != (
        scoring_mod._ordered_scaffold_fingerprint([grouped])
    )


def test_reused_lazy_workspace_matches_eager_full_matrix_selection_and_gain() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    workspace = scoring_mod._build_batch_full_geometry_workspace(
        records,
        cfg=replace(
            cfg,
            batch_size_cap=2,
            batch_joint_context_mode=(
                BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
            ),
        ),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        pauli_action_cache={},
    )
    executor = CompiledAnsatzExecutor(
        [record["candidate_term"] for record in records]
    )
    eager_state, eager_derivatives, eager_second = (
        scoring_mod._propagate_executor_derivatives(
            executor=executor,
            theta=np.zeros(len(records), dtype=float),
            psi_ref=psi,
            active_indices=range(len(records)),
        )
    )
    eager_tangents = [
        scoring_mod._horizontal_tangent(eager_state, derivative)
        for derivative in eager_derivatives
    ]
    eager_gram = scoring_mod._tangent_overlap_matrix(eager_tangents)
    eager_hpsi = apply_compiled_polynomial(eager_state, h_compiled)
    eager_h_derivatives = [
        apply_compiled_polynomial(derivative, h_compiled)
        for derivative in eager_derivatives
    ]
    eager_hessian = np.zeros((len(records), len(records)), dtype=float)
    for row in range(len(records)):
        for col in range(len(records)):
            eager_hessian[row, col] = scoring_mod._energy_hessian_entry(
                dpsi_left=eager_derivatives[row],
                dpsi_right=eager_derivatives[col],
                d2psi=eager_second[row][col],
                hpsi_state=eager_hpsi,
                hdpsi_right=eager_h_derivatives[col],
            )
    eager_hessian = 0.5 * (eager_hessian + eager_hessian.T)
    eager_gradient = np.asarray(
        [
            -2.0 * float(np.real(np.vdot(derivative, eager_hpsi)))
            for derivative in eager_derivatives
        ],
        dtype=float,
    )
    eager_workspace = replace(
        workspace,
        G_BB=np.asarray(eager_gram, dtype=float),
        H_BB=np.asarray(eager_hessian, dtype=float),
        g_B=np.asarray(eager_gradient, dtype=float),
        _subset_cache={},
    )

    assert workspace.G_BB == pytest.approx(eager_gram, abs=1e-12)
    assert workspace.H_BB == pytest.approx(eager_hessian, abs=1e-12)
    assert workspace.g_B == pytest.approx(eager_gradient, abs=1e-12)
    for subset in ([records[0]], [records[1]], records):
        reused_summary = workspace.summary_for_records(subset)
        eager_summary = eager_workspace.summary_for_records(subset)
        assert reused_summary["feasible"] is eager_summary["feasible"]
        if reused_summary["feasible"]:
            assert reused_summary["joint_gain"] == pytest.approx(
                eager_summary["joint_gain"],
                abs=1e-12,
            )
            assert reused_summary["D_geo"] == pytest.approx(
                eager_summary["D_geo"],
                abs=1e-12,
            )


def test_joint_selector_rejects_an_ill_conditioned_subset() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    proposals, summary = select_route_a_schur_proposals(
        records,
        config=RouteASchurSelectorConfig(
            batch_size_cap=2,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=(
                BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
            ),
            max_gram_condition_number=1.5,
        ),
        score_config=cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        max_proposals=3,
    )

    assert proposals
    assert all(len(proposal.records) == 1 for proposal in proposals)
    assert summary["rejection_counts"]["conditioning_gate"] == 1


def test_phase2_joint_geometry_reuse_requires_matching_exact_chart_and_state() -> None:
    psi_ref = np.asarray([1.0, 1.0j], dtype=complex)
    psi_ref /= np.linalg.norm(psi_ref)
    selected = [_term("z")]
    theta = np.asarray([0.0], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    record, h_compiled = _full_record(
        label="x",
        candidate_label="candidate-x",
        candidate_pool_index=0,
        gradient_signed=0.25,
        psi_state=psi_state,
        selected_ops=selected,
        theta=theta.tolist(),
        refit_window_indices=[0],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
        include_phase3=False,
        position_id=1,
    )
    feature = record["feature"]
    assert isinstance(feature, scoring_mod.CandidateFeatures)
    valid_payload = dict(feature.phase2_joint_geometry_reuse)
    valid_record = dict(record)

    def _workspace(candidate_record):  # noqa: ANN001
        _proposals, summary = select_route_a_schur_proposals(
            [candidate_record],
            config=RouteASchurSelectorConfig(
                batch_size_cap=1,
                batch_search_pool_size=0,
                additivity_policy=ROUTE_A_ADDITIVITY_OFF,
                joint_batch_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
            ),
            score_config=cfg,
            selected_ops=selected,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            novelty_oracle=OrderedInsertionGeometryOracle(),
            curvature_oracle=Phase2CurvatureOracle(),
            compiled_cache={},
            pauli_action_cache={},
            max_proposals=1,
        )
        return summary["geometry_workspace"]

    valid_workspace = _workspace(valid_record)
    valid_reuse = valid_workspace["phase2_joint_geometry_reuse_validation"]
    assert valid_reuse["valid_record_count"] == 1, valid_reuse["records"]
    assert valid_workspace["query_chargeable_unique_geometry_element_count"] == 0

    mismatched_position_record = {
        **valid_record,
        "position_id": 0,
    }
    mismatched_position_workspace = _workspace(mismatched_position_record)
    mismatched_position_reuse = mismatched_position_workspace[
        "phase2_joint_geometry_reuse_validation"
    ]
    assert mismatched_position_reuse["valid_record_count"] == 0
    assert "payload_position_mismatch" in mismatched_position_reuse["records"][0][
        "reasons"
    ]
    assert mismatched_position_workspace[
        "query_chargeable_unique_geometry_element_count"
    ] == 6

    mismatched_payload = {
        **valid_payload,
        "state_fingerprint": "not-the-current-state",
    }
    mismatched_record = {
        **valid_record,
        "feature": replace(
            valid_record["feature"],
            phase2_joint_geometry_reuse=mismatched_payload,
        ),
    }
    mismatched_workspace = _workspace(mismatched_record)
    mismatch_reuse = mismatched_workspace[
        "phase2_joint_geometry_reuse_validation"
    ]
    assert mismatch_reuse["valid_record_count"] == 0
    assert "state_fingerprint_mismatch" in mismatch_reuse["records"][0][
        "reasons"
    ]
    assert mismatched_workspace[
        "query_chargeable_unique_geometry_element_count"
    ] == 6

    hamiltonian_payload = {
        **valid_payload,
        "hamiltonian_fingerprint": "different-hamiltonian",
    }
    hamiltonian_record = {
        **valid_record,
        "feature": replace(
            valid_record["feature"],
            phase2_joint_geometry_reuse=hamiltonian_payload,
        ),
    }
    hamiltonian_workspace = _workspace(hamiltonian_record)
    hamiltonian_reuse = hamiltonian_workspace[
        "phase2_joint_geometry_reuse_validation"
    ]
    assert hamiltonian_reuse["valid_record_count"] == 0
    assert "hamiltonian_fingerprint_mismatch" in hamiltonian_reuse["records"][
        0
    ]["reasons"]

    branch_payload = {
        **valid_payload,
        "ordered_scaffold_fingerprint": "different-beam-branch-scaffold",
    }
    branch_record = {
        **valid_record,
        "feature": replace(
            valid_record["feature"],
            phase2_joint_geometry_reuse=branch_payload,
        ),
    }
    branch_workspace = _workspace(branch_record)
    branch_reuse = branch_workspace["phase2_joint_geometry_reuse_validation"]
    assert branch_reuse["valid_record_count"] == 0
    assert "ordered_scaffold_fingerprint_mismatch" in branch_reuse["records"][
        0
    ]["reasons"]
    assert branch_workspace[
        "query_chargeable_unique_geometry_element_count"
    ] == 6


def test_full_ansatz_context_suppresses_candidate_redundant_with_ansatz() -> None:
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    selected = [_term("x")]
    theta = np.asarray([0.0], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    simple = SimpleScoreConfig(z_alpha=0.0)
    candidate, h_compiled = _full_record(
        label="x",
        candidate_label="duplicate-x",
        candidate_pool_index=0,
        gradient_signed=0.5,
        psi_state=psi_state,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=simple,
    )
    common_kwargs = dict(
        records=[candidate],
        score_config=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
    )
    full, full_summary = select_route_a_schur_proposals(
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        ),
        **common_kwargs,
    )
    batch_only, _batch_only_summary = select_route_a_schur_proposals(
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=(
                BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
            ),
        ),
        **common_kwargs,
    )

    assert full == []
    assert full_summary["rank_gate_rejection_count"] == 1
    assert len(batch_only) == 1


def test_joint_ansatz_relaxation_matches_direct_combined_solve() -> None:
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    selected = [_term("x")]
    theta = np.asarray([0.2], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    candidate, h_compiled = _full_record(
        label="y",
        candidate_label="candidate-y",
        candidate_pool_index=0,
        gradient_signed=0.8,
        psi_state=psi_state,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
    )
    proposals, summary = select_route_a_schur_proposals(
        [candidate],
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
            joint_linear_solve_policy=(
                JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1
            ),
        ),
        score_config=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
    )
    proposal = proposals[0]
    details = proposal.summary
    direct_step = np.linalg.pinv(
        np.asarray(details["M_joint_regularized"], dtype=float)
    ) @ np.asarray([*details["g_A"], *details["g_B"]], dtype=float)
    reconstructed = np.asarray(
        [
            *details["active_parameter_relaxation"],
            *details["batch_coordinate_step"],
        ],
        dtype=float,
    )

    assert summary["geometry_workspace"]["active_indices"] == [0]
    assert abs(float(details["active_parameter_relaxation"][0])) > 1e-8
    assert reconstructed == pytest.approx(direct_step, abs=1e-9)
    assert details["joint_solve_direct_residual"] < 1e-9
    assert details["joint_solve_policy"] == "schur_of_H_plus_lambda_G_v1"


def test_supported_whitened_joint_selector_emits_full_system_certificates() -> None:
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    selected = [_term("x")]
    theta = np.asarray([0.2], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    candidate, h_compiled = _full_record(
        label="y",
        candidate_label="candidate-y",
        candidate_pool_index=0,
        gradient_signed=0.8,
        psi_state=psi_state,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
    )

    proposals, summary = select_route_a_schur_proposals(
        [candidate],
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
            joint_linear_solve_policy=(
                JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
            ),
        ),
        score_config=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
    )

    details = proposals[0].summary
    assert details["joint_linear_solve_policy_effective"] == (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
    )
    assert details["joint_metric_support_rank"] == 2
    assert details["batch_metric_rank_increment"] == 1
    assert details["classical_quantum_query_charge"] == 0
    assert details["joint_fubini_study_displacement_sq"] <= 0.5**2 * (
        1.0 + 1e-10
    )
    assert len(details["active_parameter_relaxation"]) == 1
    assert len(details["batch_coordinate_step"]) == 1
    assert details["supported_direct_residual"] < 1e-6
    assert summary["geometry_workspace"][
        "joint_linear_solve_policy_requested"
    ] == JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1


def test_active_window_context_is_explicit_and_shared_by_gram_and_hessian() -> None:
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    selected = [_term("x"), _term("z")]
    theta = np.asarray([0.1, 0.2], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    candidate, h_compiled = _full_record(
        label="y",
        candidate_label="candidate-y",
        candidate_pool_index=0,
        gradient_signed=0.7,
        psi_state=psi_state,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
    )
    _proposals, summary = select_route_a_schur_proposals(
        [candidate],
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1,
            active_context_indices=(1,),
        ),
        score_config=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
    )
    workspace = summary["geometry_workspace"]

    assert workspace["joint_batch_context_mode_effective"] == "active_window_v1"
    assert workspace["active_indices"] == [1]
    assert workspace["G_active_candidate_shape"] == [1, 1]
    assert workspace["H_active_candidate_shape"] == [1, 1]


def test_active_tail_window_resolves_against_current_ansatz_depth() -> None:
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    selected = [_term("x"), _term("z"), _term("y")]
    theta = np.asarray([0.1, 0.2, -0.1], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    candidate, h_compiled = _full_record(
        label="x",
        candidate_label="candidate-x",
        candidate_pool_index=0,
        gradient_signed=0.0,
        psi_state=psi_state,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
    )
    _proposals, summary = select_route_a_schur_proposals(
        [candidate],
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1,
            active_context_policy=ROUTE_A_ACTIVE_CONTEXT_TAIL_WINDOW_V1,
            active_window_size=2,
        ),
        score_config=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
    )

    assert summary["config"]["active_context_indices"] is None
    assert summary["active_context_selection_policy"] == "tail_window_v1"
    assert summary["active_context_indices_effective"] == [1, 2]
    assert summary["geometry_workspace"]["active_indices"] == [1, 2]


def test_joint_trust_solve_eliminates_schur_of_h_plus_lambda_g() -> None:
    rng = np.random.default_rng(5)
    psi_ref = np.asarray(
        rng.normal(size=4) + 1.0j * rng.normal(size=4),
        dtype=complex,
    )
    psi_ref /= np.linalg.norm(psi_ref)
    selected = [_term("xe")]
    theta = np.asarray([0.37], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    candidate, h_compiled = _full_record(
        label="ye",
        candidate_label="candidate-ye",
        candidate_pool_index=0,
        gradient_signed=0.0,
        psi_state=psi_state,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
        h_label="ye",
    )
    proposals, _summary = select_route_a_schur_proposals(
        [candidate],
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            max_fubini_study_step=1e-4,
            joint_linear_solve_policy=(
                JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1
            ),
        ),
        score_config=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
    )
    details = proposals[0].summary
    trust_lambda = float(details["trust_lambda"])
    assert trust_lambda > 0.0
    assert details["trust_regularization_applied"] is bool(
        details["trust_clipped"]
    )
    radius_sq = float(details["trust_radius_sq"])
    displacement_sq = float(details["joint_fubini_study_displacement_sq"])
    binding_tolerance_sq = float(details["trust_radius_binding_tolerance_sq"])
    assert details["trust_radius_binding"] is bool(
        details["trust_regularization_applied"]
        and abs(displacement_sq - radius_sq) <= binding_tolerance_sq
    )

    G_AA = np.asarray(details["G_AA_raw"], dtype=float)
    G_AB = np.asarray(details["G_AB_raw"], dtype=float)
    G_BB = np.asarray(details["G_BB_raw"], dtype=float)
    H_AA = np.asarray(details["H_AA_raw"], dtype=float)
    H_AB = np.asarray(details["H_AB_raw"], dtype=float)
    H_BB = np.asarray(details["H_BB_raw"], dtype=float)
    energy_floor = 1e-9
    M_AA = H_AA + trust_lambda * G_AA + energy_floor * np.eye(len(H_AA))
    M_AB = H_AB + trust_lambda * G_AB
    M_BB = H_BB + trust_lambda * G_BB + energy_floor * np.eye(len(H_BB))
    expected = M_BB - M_AB.T @ np.linalg.pinv(M_AA, rcond=energy_floor) @ M_AB
    assert np.asarray(details["M_effective"], dtype=float) == pytest.approx(
        expected,
        abs=1e-9,
    )

    schur_h = H_BB - H_AB.T @ np.linalg.pinv(
        H_AA + energy_floor * np.eye(len(H_AA)),
        rcond=energy_floor,
    ) @ H_AB
    schur_g = G_BB - G_AB.T @ np.linalg.pinv(
        G_AA + 1e-9 * np.eye(len(G_AA)),
        rcond=1e-9,
    ) @ G_AB
    separate_schurs = schur_h + trust_lambda * schur_g
    assert np.max(np.abs(expected - separate_schurs)) > 1e-3


def test_joint_geometry_is_global_phase_and_parameter_scale_invariant() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    selector = RouteASchurSelectorConfig(
        batch_size_cap=2,
        batch_search_pool_size=0,
        additivity_policy=ROUTE_A_ADDITIVITY_OFF,
        joint_batch_context_mode=BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1,
    )

    def _run_phase(phase: complex):
        return select_route_a_schur_proposals(
            records,
            config=selector,
            score_config=cfg,
            selected_ops=[],
            theta=np.zeros(0, dtype=float),
            psi_ref=phase * psi,
            psi_state=phase * psi,
            h_compiled=h_compiled,
            novelty_oracle=OrderedInsertionGeometryOracle(),
            curvature_oracle=Phase2CurvatureOracle(),
            compiled_cache={},
            pauli_action_cache={},
            max_proposals=3,
        )[0]

    baseline = _run_phase(1.0 + 0.0j)
    rotated = _run_phase(np.exp(0.73j))
    assert [proposal.delta_e3 for proposal in rotated] == pytest.approx(
        [proposal.delta_e3 for proposal in baseline],
        abs=1e-12,
    )

    base_record = dict(records[1])

    def _scaled_gain(scale: float) -> float:
        scaled_term = type(
            "_ScaledAnsatzTerm",
            (),
            {
                "label": f"scaled-ye-{scale}",
                "polynomial": PauliPolynomial(
                    "JW",
                    [PauliTerm(2, ps="ye", pc=float(scale))],
                ),
            },
        )()
        scaled_record = {
            **base_record,
            "candidate_term": scaled_term,
        }
        proposals, _summary = select_route_a_schur_proposals(
            [scaled_record],
            config=RouteASchurSelectorConfig(
                batch_size_cap=1,
                batch_search_pool_size=0,
                additivity_policy=ROUTE_A_ADDITIVITY_OFF,
                joint_batch_context_mode=(
                    BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
                ),
                metric_regularization=0.0,
                energy_regularization=0.0,
            ),
            score_config=cfg,
            selected_ops=[],
            theta=np.zeros(0, dtype=float),
            psi_ref=psi,
            psi_state=psi,
            h_compiled=h_compiled,
            novelty_oracle=OrderedInsertionGeometryOracle(),
            curvature_oracle=Phase2CurvatureOracle(),
            compiled_cache={},
            pauli_action_cache={},
        )
        return float(proposals[0].delta_e3)

    assert _scaled_gain(3.0) == pytest.approx(_scaled_gain(1.0), abs=1e-12)
    assert _scaled_gain(0.2) == pytest.approx(_scaled_gain(1.0), abs=1e-12)


def test_joint_selector_preserves_position_alternatives_but_blocks_same_child_batch() -> None:
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    selected = [_term("x")]
    theta = np.asarray([0.3], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    position_zero, h_compiled = _full_record(
        label="z",
        candidate_label="same-z-child",
        candidate_pool_index=0,
        gradient_signed=0.0,
        psi_state=psi_state,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
        h_label="y",
    )
    position_zero.update(
        phase2_raw_score=2.0,
        route_a_global_pauli_identity="pauli:z",
    )
    position_one = {
        **position_zero,
        "position_id": 1,
        "phase2_raw_score": 1.0,
        "feature": replace(
            position_zero["feature"],
            position_id=1,
            append_position=1,
            positions_considered=[1],
        ),
    }
    proposals, summary = select_route_a_schur_proposals(
        [position_zero, position_one],
        config=RouteASchurSelectorConfig(
            batch_size_cap=2,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
        ),
        score_config=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        max_proposals=5,
    )

    assert [row["position_id"] for row in summary["geometry_workspace"]["search_pool_records"]] == [0, 1]
    assert summary["rejection_counts"]["duplicate_exact_child_identity"] == 1
    assert all(len(proposal.records) == 1 for proposal in proposals)
    assert [record["position_id"] for record in proposals[0].records] == [1]


def test_batch_cap_one_and_larger_share_singleton_evaluator_and_workspace_charge() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    common = dict(
        score_config=cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        max_proposals=10,
    )
    singleton_proposals, singleton_summary = select_route_a_schur_proposals(
        records,
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1,
        ),
        **common,
    )
    batched_proposals, batched_summary = select_route_a_schur_proposals(
        records,
        config=RouteASchurSelectorConfig(
            batch_size_cap=2,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1,
        ),
        **common,
    )

    def _singleton_map(proposals):  # noqa: ANN001
        return {
            str(proposal.records[0]["feature"].candidate_label): proposal
            for proposal in proposals
            if len(proposal.records) == 1
        }

    singleton_map = _singleton_map(singleton_proposals)
    batched_singletons = _singleton_map(batched_proposals)
    assert set(singleton_map) == set(batched_singletons)
    for label in singleton_map:
        assert batched_singletons[label].delta_e3 == pytest.approx(
            singleton_map[label].delta_e3,
            abs=1e-12,
        )
        assert batched_singletons[label].summary["D_geo"] == pytest.approx(
            singleton_map[label].summary["D_geo"],
            abs=1e-12,
        )
    assert singleton_summary["geometry_workspace"][
        "query_chargeable_unique_geometry_element_count"
    ] == 0
    assert singleton_summary["geometry_workspace"][
        "required_candidate_pair_count"
    ] == 0
    assert singleton_summary["geometry_workspace"][
        "constructed_candidate_pair_count"
    ] == 0
    assert singleton_summary["geometry_workspace"]["required_element_counts"][
        "G_CC_off_diagonal"
    ] == 0
    assert singleton_summary["geometry_workspace"]["required_element_counts"][
        "H_CC_off_diagonal"
    ] == 0
    assert batched_summary["geometry_workspace"][
        "query_chargeable_unique_geometry_element_count"
    ] == 2
    assert batched_summary["geometry_workspace"][
        "required_candidate_pair_count"
    ] == 1
    assert singleton_summary["subset_evaluation_count"] == 2
    assert batched_summary["subset_evaluation_count"] == 3


def test_full_context_never_silently_truncates_active_ansatz() -> None:
    rng = np.random.default_rng(71)
    psi_ref = np.asarray(
        rng.normal(size=4) + 1.0j * rng.normal(size=4),
        dtype=complex,
    )
    psi_ref /= np.linalg.norm(psi_ref)
    selected = [_term("xe"), _term("ye"), _term("ze")]
    theta = np.asarray([0.1, -0.2, 0.3], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5)
    candidate, h_compiled = _full_record(
        label="ex",
        candidate_label="candidate-ex",
        candidate_pool_index=0,
        gradient_signed=0.0,
        psi_state=psi_state,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=cfg,
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
        h_label="xy",
    )
    _proposals, summary = select_route_a_schur_proposals(
        [candidate],
        config=RouteASchurSelectorConfig(
            batch_size_cap=1,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        ),
        score_config=cfg,
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
    )
    workspace = summary["geometry_workspace"]
    assert workspace["active_indices"] == [0, 1, 2]
    assert workspace["G_active_candidate_shape"] == [3, 1]
    assert workspace["H_active_candidate_shape"] == [3, 1]

    with pytest.raises(ValueError, match="out-of-range ansatz index"):
        select_route_a_schur_proposals(
            [candidate],
            config=RouteASchurSelectorConfig(
                batch_size_cap=1,
                batch_search_pool_size=0,
                additivity_policy=ROUTE_A_ADDITIVITY_OFF,
                joint_batch_context_mode=BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1,
                active_context_indices=(3,),
            ),
            score_config=cfg,
            selected_ops=selected,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            novelty_oracle=OrderedInsertionGeometryOracle(),
            curvature_oracle=Phase2CurvatureOracle(),
            compiled_cache={},
            pauli_action_cache={},
        )


def test_phase2_only_feature_path_does_not_call_phase3_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psi = np.asarray([1.0, 0.0], dtype=complex)

    def _unexpected_phase3(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("canonical child12 Phase 2 called Phase 3")

    owner = sys.modules[build_full_candidate_features.__module__]
    monkeypatch.setattr(owner, "phase3_canonical_score_components", _unexpected_phase3)
    record, _h_compiled = _full_record(
        label="x",
        candidate_label="candidate-x",
        candidate_pool_index=0,
        gradient_signed=0.5,
        psi_state=psi,
        selected_ops=[],
        theta=[],
        refit_window_indices=[],
        full_cfg=FullScoreConfig(z_alpha=0.0),
        simple_cfg=SimpleScoreConfig(z_alpha=0.0),
        include_phase3=False,
    )
    feature = record["feature"]

    assert feature.selector_geometry_mode == "phase2_only_v1"
    assert feature.phase3_primary_score is None
    assert feature.phase3_canonical_score_formula == (
        "disabled_in_canonical_child12_route"
    )
    assert feature.selector_score == pytest.approx(feature.phase2_raw_score)


def test_joint_workspace_fails_closed_on_state_reconstruction_mismatch() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    seed = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex)
    mismatched = seed - np.vdot(psi, seed) * psi
    mismatched /= np.linalg.norm(mismatched)

    with pytest.raises(ValueError, match="state inconsistent"):
        select_route_a_schur_proposals(
            records,
            config=RouteASchurSelectorConfig(
                batch_size_cap=1,
                batch_search_pool_size=0,
                additivity_policy=ROUTE_A_ADDITIVITY_OFF,
                joint_batch_context_mode=(
                    BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
                ),
            ),
            score_config=cfg,
            selected_ops=[],
            theta=np.zeros(0, dtype=float),
            psi_ref=psi,
            psi_state=mismatched,
            h_compiled=h_compiled,
            novelty_oracle=OrderedInsertionGeometryOracle(),
            curvature_oracle=Phase2CurvatureOracle(),
            compiled_cache={},
            pauli_action_cache={},
        )


def test_canonical_greedy_batch_cap_one_evaluates_all_singletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            **_tetris_record(
                candidate_label=f"child-{index}",
                pauli_label=("xe", "ye", "ze")[index],
                full_v2_score=float(10 - index),
                candidate_pool_index=index,
            ),
            "phase2_raw_score": float(10 - index),
            "route_a_global_pauli_identity": f"pauli:{index}",
        }
        for index in range(3)
    ]
    joint_scores = {"child-0": 1.0, "child-1": 20.0, "child-2": 2.0}

    def _fake_evaluator(subset, *, mode, **_kwargs):  # noqa: ANN001, ANN003
        copied = tuple(dict(record) for record in subset)
        score = float(joint_scores[copied[0]["candidate_label"]])
        return scoring_mod.BatchSelectionProposal(
            records=copied,
            summary={
                "selection_mode": str(mode),
                "feasible": True,
                "joint_gain": score,
                "contextual_single_total": score,
                "additivity_defect": 0.0,
            },
            score=score,
            delta_e3=score,
            k3=0.0,
            denominator_1_plus_k3=1.0,
        )

    monkeypatch.setattr(
        scoring_mod,
        "_evaluate_ordered_reduced_plane_batch_proposal",
        _fake_evaluator,
    )
    proposals, summary = scoring_mod.greedy_reduced_plane_batch_proposals(
        records,
        cfg=FullScoreConfig(
            batch_selection_mode="greedy_reduced_plane",
            batch_target_size=1,
            batch_size_cap=1,
            batch_search_pool_size=0,
            batch_search_population_mode="ranked_child_phase2_v1",
            batch_additivity_policy="off",
            batch_geometry_mode="per_subset_diagonal_hessian_legacy_v1",
        ),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.ones(1, dtype=complex),
        psi_state=np.ones(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
        max_proposals=1,
    )

    assert summary["subset_counts_evaluated"] == {"1": 3}
    assert proposals[0].records[0]["candidate_label"] == "child-1"


def test_full_geometry_legacy_hard_additivity_gate_remains_effective() -> None:
    records = [
        _tetris_record(
            candidate_label=f"child-{index}",
            pauli_label=("xe", "ye")[index],
            full_v2_score=1.0,
            candidate_pool_index=index,
        )
        for index in range(2)
    ]

    class _Workspace:
        def __init__(self) -> None:
            self.records = tuple(dict(record) for record in records)

        def summary_for_records(self, _records):  # noqa: ANN001
            return {
                "feasible": True,
                "subset_workspace_indices": [0, 1],
                "joint_gain": 2.0,
                "contextual_single_total": 4.0,
                "additivity_defect": 0.5,
            }

    rejection_counts: dict[str, int] = {}
    proposal = scoring_mod._evaluate_ordered_reduced_plane_batch_proposal(
        records,
        cfg=FullScoreConfig(
            batch_additivity_policy="hard_gate_legacy_v1",
            batch_additivity_tol=0.25,
        ),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.ones(1, dtype=complex),
        psi_state=np.ones(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
        mode="combinatorial_reduced_plane",
        rejection_counts=rejection_counts,
        geometry_workspace=_Workspace(),
    )

    assert proposal is None
    assert rejection_counts == {"additivity_hard_gate_legacy": 1}


def test_greedy_same_position_batch_uses_workspace_coordinate_order() -> None:
    psi, records, h_compiled, cfg = _joint_geometry_fixture()
    proposals, summary = select_route_a_schur_proposals(
        records,
        config=RouteASchurSelectorConfig(
            mode=ROUTE_A_SCHUR_GREEDY_REDUCED_PLANE,
            batch_size_cap=2,
            batch_search_pool_size=0,
            additivity_policy=ROUTE_A_ADDITIVITY_OFF,
            joint_batch_context_mode=(
                BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
            ),
        ),
        score_config=cfg,
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi,
        psi_state=psi,
        h_compiled=h_compiled,
        novelty_oracle=OrderedInsertionGeometryOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        max_proposals=2,
    )
    pair = _pair_proposal(proposals)
    workspace_labels = [
        row["candidate_label"]
        for row in summary["geometry_workspace"]["search_pool_records"]
    ]
    proposal_labels = [
        record["feature"].candidate_label for record in pair.records
    ]

    assert proposal_labels == workspace_labels
    assert pair.summary["subset_workspace_indices"] == [0, 1]


def _tetris_record(
    *,
    candidate_label: str,
    pauli_label: str,
    full_v2_score: float,
    phase2_raw_score: float | None = None,
    candidate_pool_index: int = 0,
    position_id: int = 0,
) -> dict[str, object]:
    term = type(
        "_DummyAnsatzTerm",
        (),
        {
            "label": candidate_label,
            "polynomial": PauliPolynomial(
                "JW",
                [PauliTerm(len(str(pauli_label)), ps=str(pauli_label), pc=1.0)],
            ),
        },
    )()
    return {
        "candidate_label": candidate_label,
        "candidate_term": term,
        "full_v2_score": float(full_v2_score),
        "phase2_raw_score": float(full_v2_score if phase2_raw_score is None else phase2_raw_score),
        "candidate_pool_index": int(candidate_pool_index),
        "position_id": int(position_id),
    }


def test_overlap_orthogonal_batch_select_keeps_low_overlap_shell_pair() -> None:
    cfg = FullScoreConfig(
        batch_target_size=2,
        batch_size_cap=2,
        batch_near_degenerate_ratio=0.9,
    )
    rec_a = _tetris_record(
        candidate_label="a_x0",
        pauli_label="xe",
        full_v2_score=1.0,
        candidate_pool_index=0,
    )
    rec_b = _tetris_record(
        candidate_label="b_y1",
        pauli_label="ey",
        full_v2_score=0.95,
        candidate_pool_index=1,
    )

    selected, summary = overlap_orthogonal_batch_select(
        [rec_b, rec_a],
        cfg=cfg,
        tie_break_score_key="phase2_raw_score",
    )

    assert [rec["candidate_label"] for rec in selected] == ["a_x0", "b_y1"]
    assert summary["selection_mode"] == "overlap_orthogonal_benchmark"
    assert summary["selected"] is True
    assert summary["selected_count"] == 2
    assert summary["rejected_overlap_count"] == 0
    assert summary["rejected_invalid_feature_count"] == 0
    assert float(summary["overlap_threshold"]) == pytest.approx(0.15)
    assert float(summary["max_pairwise_overlap"]) == pytest.approx(0.0)
    assert float(summary["joint_gain"]) == pytest.approx(1.95)
    assert all(float(rec["compatibility_penalty"]["total"]) == 0.0 for rec in selected)


def test_overlap_orthogonal_batch_select_rejects_high_overlap_follow_on() -> None:
    cfg = FullScoreConfig(
        batch_target_size=2,
        batch_size_cap=2,
        batch_near_degenerate_ratio=0.9,
    )
    rec_a = _tetris_record(
        candidate_label="a_x0",
        pauli_label="xe",
        full_v2_score=1.0,
        candidate_pool_index=0,
    )
    rec_b = _tetris_record(
        candidate_label="b_x0_duplicate",
        pauli_label="xe",
        full_v2_score=0.99,
        candidate_pool_index=1,
    )

    selected, summary = overlap_orthogonal_batch_select(
        [rec_a, rec_b],
        cfg=cfg,
        tie_break_score_key="phase2_raw_score",
    )

    assert [rec["candidate_label"] for rec in selected] == ["a_x0"]
    assert summary["selected"] is False
    assert summary["reason"] == "singleton_shell"
    assert summary["selected_count"] == 1
    assert summary["rejected_overlap_count"] == 1
    assert summary["rejected_invalid_feature_count"] == 0
    assert float(summary["additivity_defect"]) == 0.0


def test_ceo_commuting_batch_select_keeps_pairwise_commuting_shell_pair() -> None:
    cfg = FullScoreConfig(
        batch_target_size=2,
        batch_size_cap=2,
        batch_near_degenerate_ratio=0.9,
    )
    rec_a = _tetris_record(
        candidate_label="a_xx",
        pauli_label="xx",
        full_v2_score=1.0,
        candidate_pool_index=0,
    )
    rec_b = _tetris_record(
        candidate_label="b_yy",
        pauli_label="yy",
        full_v2_score=0.95,
        candidate_pool_index=1,
    )

    selected, summary = ceo_commuting_batch_select(
        [rec_b, rec_a],
        cfg=cfg,
        tie_break_score_key="phase2_raw_score",
    )

    assert [rec["candidate_label"] for rec in selected] == ["a_xx", "b_yy"]
    assert summary["selection_mode"] == "ceo_commuting_benchmark"
    assert summary["selected"] is True
    assert summary["selected_count"] == 2
    assert summary["rejected_noncommuting_count"] == 0
    assert summary["rejected_invalid_pauli_count"] == 0
    assert float(summary["joint_gain"]) == pytest.approx(1.95)
    assert all(float(rec["compatibility_penalty"]["total"]) == 0.0 for rec in selected)


def test_ceo_commuting_batch_select_rejects_noncommuting_follow_on() -> None:
    cfg = FullScoreConfig(
        batch_target_size=2,
        batch_size_cap=2,
        batch_near_degenerate_ratio=0.9,
    )
    rec_a = _tetris_record(
        candidate_label="a_x0",
        pauli_label="xe",
        full_v2_score=1.0,
        candidate_pool_index=0,
    )
    rec_b = _tetris_record(
        candidate_label="b_z0",
        pauli_label="ze",
        full_v2_score=0.99,
        candidate_pool_index=1,
    )

    selected, summary = ceo_commuting_batch_select(
        [rec_a, rec_b],
        cfg=cfg,
        tie_break_score_key="phase2_raw_score",
    )

    assert [rec["candidate_label"] for rec in selected] == ["a_x0"]
    assert summary["selected"] is False
    assert summary["reason"] == "singleton_shell"
    assert summary["selected_count"] == 1
    assert summary["rejected_noncommuting_count"] == 1
    assert summary["rejected_invalid_pauli_count"] == 0
    assert float(summary["additivity_defect"]) == 0.0


def test_select_phase2_batch_records_routes_ceo_commuting_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _tetris_record(
            candidate_label="a_x0",
            pauli_label="xe",
            full_v2_score=1.0,
            candidate_pool_index=0,
        )
    ]
    calls: dict[str, object] = {}

    def _fake_ceo_batch_select(ranked_records, **kwargs):  # noqa: ANN001, ANN003 - test shim
        calls["records"] = list(ranked_records)
        calls["kwargs"] = dict(kwargs)
        return [dict(ranked_records[0])], {"selection_mode": "ceo_commuting_benchmark", "reason": "delegated"}

    monkeypatch.setattr(scoring_mod, "ceo_commuting_batch_select", _fake_ceo_batch_select)

    selected, summary = select_phase2_batch_records(
        records,
        cfg=FullScoreConfig(batch_selection_mode="ceo_commuting_benchmark"),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.zeros(1, dtype=complex),
        psi_state=np.zeros(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
        compiled_cache={},
        pauli_action_cache={},
        tie_break_score_key="phase2_raw_score",
    )

    assert selected[0]["candidate_label"] == "a_x0"
    assert summary["selection_mode"] == "ceo_commuting_benchmark"
    assert summary["reason"] == "delegated"
    assert calls["records"] == records
    assert calls["kwargs"]["tie_break_score_key"] == "phase2_raw_score"


def test_select_phase2_batch_records_routes_overlap_orthogonal_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _tetris_record(
            candidate_label="a_x0",
            pauli_label="xe",
            full_v2_score=1.0,
            candidate_pool_index=0,
        )
    ]
    calls: dict[str, object] = {}

    def _fake_overlap_batch_select(ranked_records, **kwargs):  # noqa: ANN001, ANN003 - test shim
        calls["records"] = list(ranked_records)
        calls["kwargs"] = dict(kwargs)
        return [dict(ranked_records[0])], {"selection_mode": "overlap_orthogonal_benchmark", "reason": "delegated"}

    monkeypatch.setattr(scoring_mod, "overlap_orthogonal_batch_select", _fake_overlap_batch_select)

    selected, summary = select_phase2_batch_records(
        records,
        cfg=FullScoreConfig(batch_selection_mode="overlap_orthogonal_benchmark"),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.zeros(1, dtype=complex),
        psi_state=np.zeros(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
        compiled_cache={},
        pauli_action_cache={},
        tie_break_score_key="phase2_raw_score",
    )

    assert selected[0]["candidate_label"] == "a_x0"
    assert summary["selection_mode"] == "overlap_orthogonal_benchmark"
    assert summary["reason"] == "delegated"
    assert calls["records"] == records
    assert calls["kwargs"]["tie_break_score_key"] == "phase2_raw_score"


def test_select_phase2_batch_records_default_delegates_to_reduced_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _tetris_record(
            candidate_label="a_x0",
            pauli_label="xe",
            full_v2_score=1.0,
            candidate_pool_index=0,
        )
    ]
    calls: dict[str, object] = {}

    def _fake_reduced_plane_batch_select(ranked_records, **kwargs):  # noqa: ANN001, ANN003 - test shim
        calls["records"] = list(ranked_records)
        calls["kwargs"] = dict(kwargs)
        return [dict(ranked_records[0])], {"reason": "delegated"}

    monkeypatch.setattr(scoring_mod, "reduced_plane_batch_select", _fake_reduced_plane_batch_select)

    selected, summary = select_phase2_batch_records(
        records,
        cfg=FullScoreConfig(),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.zeros(1, dtype=complex),
        psi_state=np.zeros(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
        compiled_cache={},
        pauli_action_cache={},
        tie_break_score_key="phase2_raw_score",
    )

    assert selected[0]["candidate_label"] == "a_x0"
    assert summary["selection_mode"] == "reduced_plane"
    assert summary["reason"] == "delegated"
    assert calls["records"] == records
    assert calls["kwargs"]["tie_break_score_key"] == "phase2_raw_score"


def test_phase1_compile_cost_oracle_emits_manuscript_hatted_primitives_for_xyz() -> None:
    oracle = Phase1CompileCostOracle()
    xy = oracle.estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=1,
        candidate_term=_term("xy"),
    )
    z = oracle.estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=1,
        candidate_term=_term("ez"),
    )
    yy = oracle.estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=1,
        candidate_term=_term("yy"),
    )

    assert xy.c_hat_2q == pytest.approx(2.0)
    assert xy.c_hat_d == pytest.approx(2.0)
    assert xy.c_hat_1q == pytest.approx(7.0)
    assert xy.c_hat_theta == pytest.approx(1.0)
    assert z.c_hat_2q == pytest.approx(0.0)
    assert z.c_hat_d == pytest.approx(0.0)
    assert z.c_hat_1q == pytest.approx(1.0)
    assert yy.c_hat_1q == pytest.approx(9.0)
    assert xy.hardware_cost_source == "proxy_logical_ladder_span_v1"


def test_hardware_cost_family_normalization_uses_positive_excess_denominator() -> None:
    cfg_simple = SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0)
    meas = MeasurementCacheAudit()
    oracle = Phase1CompileCostOracle()
    base = build_candidate_features(
        stage_name="core",
        candidate_label="base",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1, candidate_term=_term("x")),
        measurement_stats=meas.estimate([MeasurementGroupSpec("x", coeff_l2=1.0)]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg_simple,
    )
    low = replace(base, c_hat_2q=0.0, c_hat_d=0.0, c_hat_1q=0.0, c_hat_theta=0.0, c_hat_shot=0.0)
    high = replace(base, candidate_label="high", c_hat_2q=4.0, c_hat_d=0.0, c_hat_1q=0.0, c_hat_theta=0.0, c_hat_shot=0.0)
    cfg_full = FullScoreConfig(lambda_2q=1.0, lambda_d=0.0, lambda_1q=0.0, lambda_theta=0.0, lambda_shot=0.0, hardware_cost_scale_floor=1.0)

    normalized = normalize_hardware_cost_feature_family([low, high], cfg_full)

    assert normalized[0].c_bar_2q == pytest.approx(0.0)
    assert normalized[0].hardware_cost_denominator == pytest.approx(1.0)
    assert normalized[1].c_bar_2q == pytest.approx(np.arcsinh(1.0))
    assert normalized[1].hardware_cost_denominator == pytest.approx(1.0 + np.arcsinh(1.0))
    assert normalized[1].hardware_cost_normalization["schema"] == "snake_hardware_cost_family_robust_v1"


def _symmetric_cost_test_features(costs: list[float]):
    policy = scoring_mod.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
    cfg = SimpleScoreConfig(
        lambda_compile=0.0,
        lambda_measure=0.0,
        hardware_cost_normalization_mode=policy,
    )
    meas = MeasurementCacheAudit()
    oracle = Phase1CompileCostOracle()
    base = build_candidate_features(
        stage_name="core",
        candidate_label="base",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(
            candidate_term_count=1,
            position_id=0,
            append_position=0,
            refit_active_count=1,
            candidate_term=_term("x"),
        ),
        measurement_stats=meas.estimate(
            [MeasurementGroupSpec("x", coeff_l2=1.0)]
        ),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    return [
        replace(
            base,
            candidate_label=f"cost_{index}",
            candidate_pool_index=index,
            c_hat_2q=float(cost),
            c_hat_d=0.0,
            c_hat_1q=0.0,
            c_hat_theta=0.0,
            c_hat_shot=0.0,
            novelty=1.0,
            F_raw=1.0,
            F_red=1.0,
            h_eff=1.0,
            h_hat=1.0,
            phase2_raw_F_effective=1.0,
            phase2_raw_trust_gain=1.0,
            phase2_raw_novelty=1.0,
            phase2_raw_score=1.0,
        )
        for index, cost in enumerate(costs)
    ]


def _symmetric_cost_full_cfg() -> FullScoreConfig:
    return FullScoreConfig(
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        hardware_cost_scale_floor=1e-12,
        hardware_cost_normalization_mode=(
            scoring_mod.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
        ),
        deferred_gram_fallback_enabled=True,
        z_alpha=0.0,
    )


def test_symmetric_hardware_cost_normalization_rewards_below_median_cost() -> None:
    cfg = _symmetric_cost_full_cfg()
    normalized = normalize_hardware_cost_feature_family(
        _symmetric_cost_test_features([1.0, 3.0, 5.0]), cfg
    )

    assert [feat.c_bar_2q for feat in normalized] == pytest.approx([-0.5, 0.0, 0.5])
    assert [feat.hardware_cost_signed_index for feat in normalized] == pytest.approx(
        [-0.5, 0.0, 0.5]
    )
    assert [feat.hardware_cost_score_factor for feat in normalized] == pytest.approx(
        [1.25, 1.0, 0.75]
    )
    assert [feat.hardware_cost_denominator for feat in normalized] == pytest.approx(
        [1.0, 1.0, 1.0]
    )
    assert len({feat.hardware_cost_population_hash for feat in normalized}) == 1
    assert len(str(normalized[0].hardware_cost_population_hash)) == 64


def test_symmetric_hardware_cost_uniform_component_is_neutral() -> None:
    cfg = _symmetric_cost_full_cfg()
    normalized = normalize_hardware_cost_feature_family(
        _symmetric_cost_test_features([7.0, 7.0, 7.0]), cfg
    )

    assert [feat.c_bar_2q for feat in normalized] == pytest.approx([0.0, 0.0, 0.0])
    assert [feat.hardware_cost_signed_index for feat in normalized] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert [feat.hardware_cost_score_factor for feat in normalized] == pytest.approx(
        [1.0, 1.0, 1.0]
    )


def test_symmetric_hardware_cost_population_hash_is_order_independent() -> None:
    cfg = _symmetric_cost_full_cfg()
    features = _symmetric_cost_test_features([1.0, 3.0, 5.0])
    forward = normalize_hardware_cost_feature_family(features, cfg)
    reverse = normalize_hardware_cost_feature_family(list(reversed(features)), cfg)
    changed = normalize_hardware_cost_feature_family(
        [features[0], features[1], replace(features[2], c_hat_2q=6.0)], cfg
    )

    assert forward[0].hardware_cost_population_hash == reverse[0].hardware_cost_population_hash
    assert forward[0].hardware_cost_population_hash != changed[0].hardware_cost_population_hash


@pytest.mark.parametrize("invalid_cost", [-1.0, float("nan"), float("inf")])
def test_symmetric_hardware_cost_fails_closed_on_invalid_cost(invalid_cost: float) -> None:
    cfg = _symmetric_cost_full_cfg()
    feature = replace(_symmetric_cost_test_features([1.0])[0], c_hat_2q=invalid_cost)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        normalize_hardware_cost_feature_family([feature], cfg)


def test_symmetric_hardware_cost_factor_applies_to_phase1_phase2_and_phase3() -> None:
    policy = scoring_mod.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
    features = _symmetric_cost_test_features([1.0, 3.0, 5.0])
    cfg_simple = SimpleScoreConfig(
        lambda_compile=0.0,
        lambda_measure=0.0,
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        hardware_cost_scale_floor=1e-12,
        hardware_cost_normalization_mode=policy,
        z_alpha=0.0,
    )
    phase1 = rescore_hardware_cost_family(
        [{"candidate_label": feat.candidate_label, "feature": feat} for feat in features],
        cfg_simple,
    )
    assert [record["simple_score"] for record in phase1] == pytest.approx(
        [phase1[1]["simple_score"] * 1.25, phase1[1]["simple_score"], phase1[1]["simple_score"] * 0.75]
    )

    cfg_full = _symmetric_cost_full_cfg()
    rescored = rescore_hardware_cost_family(
        [{"candidate_label": feat.candidate_label, "feature": feat} for feat in features],
        cfg_full,
    )
    assert [record["phase2_raw_score"] for record in rescored] == pytest.approx(
        [1.25, 1.0, 0.75]
    )
    phase3_scores = [record["phase3_primary_score"] for record in rescored]
    assert phase3_scores[0] == pytest.approx(phase3_scores[1] * 1.25)
    assert phase3_scores[2] == pytest.approx(phase3_scores[1] * 0.75)
    fallback_scores = [
        phase3_plateau_novelty_cost_score_components(
            record["feature"], cfg_full, plateau_novelty=1.0
        )["phase3_plateau_acquisition_score"]
        for record in rescored
    ]
    assert fallback_scores == pytest.approx([1.25, 1.0, 0.75])


def test_hardware_cost_ansatz_entry_denominators_normalize_current_entries() -> None:
    cfg = FullScoreConfig(
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        hardware_cost_scale_floor=1.0,
    )
    payload = hardware_cost_ansatz_entry_denominators(
        [
            {"label": "low", "c_hat_2q": 1.0},
            {"label": "mid", "c_hat_2q": 3.0},
            {"label": "high", "c_hat_2q": 7.0},
        ],
        cfg,
    )

    assert payload["schema"] == "snake_hardware_cost_ansatz_entry_denominator_v1"
    assert payload["scope"] == "current_ansatz_entries"
    assert payload["medians"]["2q"] == pytest.approx(3.0)
    assert payload["denominators"][0] == pytest.approx(1.0)
    assert payload["denominators"][1] == pytest.approx(1.0)
    assert payload["denominators"][2] == pytest.approx(1.0 + np.arcsinh(1.0))


def test_phase0_candidate_record_cost_denominator_penalizes_expensive_record() -> None:
    cfg = SimpleScoreConfig(
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        hardware_cost_scale_floor=1.0,
    )
    payload = hardware_cost_candidate_record_denominators(
        [
            {"candidate_pool_index": 1, "position_id": 0, "label": "low", "c_hat_2q": 1.0},
            {"candidate_pool_index": 2, "position_id": 0, "label": "high", "c_hat_2q": 5.0},
        ],
        cfg,
    )

    assert payload["schema"] == "snake_hardware_cost_candidate_record_denominator_v1"
    assert payload["scope"] == "candidate_records"
    assert payload["rows"][0]["candidate_pool_index"] == 1
    assert payload["rows"][1]["candidate_pool_index"] == 2
    assert payload["denominators"][0] == pytest.approx(1.0)
    assert payload["denominators"][1] == pytest.approx(1.0 + np.arcsinh(1.0))


def test_phase0_candidate_record_cost_disable_keeps_unit_denominator() -> None:
    cfg = SimpleScoreConfig(
        lambda_2q=0.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        hardware_cost_scale_floor=1.0,
    )
    payload = hardware_cost_candidate_record_denominators(
        [
            {"candidate_pool_index": 1, "position_id": 0, "label": "low", "c_hat_2q": 1.0},
            {"candidate_pool_index": 2, "position_id": 0, "label": "high", "c_hat_2q": 100.0},
        ],
        cfg,
    )

    assert payload["denominators"] == pytest.approx([1.0, 1.0])
    assert all(row["hardware_cost_denominator"] == pytest.approx(1.0) for row in payload["rows"])


def test_rescore_hardware_cost_family_updates_phase2_denominator_and_record_aliases() -> None:
    cfg_simple = SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0)
    meas = MeasurementCacheAudit()
    oracle = Phase1CompileCostOracle()
    base = build_candidate_features(
        stage_name="core",
        candidate_label="base",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1, candidate_term=_term("x")),
        measurement_stats=meas.estimate([MeasurementGroupSpec("x", coeff_l2=1.0)]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg_simple,
    )
    low = replace(
        base,
        candidate_label="low",
        phase2_raw_trust_gain=10.0,
        phase2_raw_score=10.0,
        c_hat_2q=0.0,
    )
    high = replace(
        base,
        candidate_label="high",
        phase2_raw_trust_gain=10.0,
        phase2_raw_score=10.0,
        c_hat_2q=4.0,
    )
    cfg_full = FullScoreConfig(lambda_2q=1.0, lambda_d=0.0, lambda_1q=0.0, lambda_theta=0.0, lambda_shot=0.0, hardware_cost_scale_floor=1.0)

    rescored = rescore_hardware_cost_family(
        [
            {"candidate_label": "low", "feature": low, "phase2_raw_score": 10.0},
            {"candidate_label": "high", "feature": high, "phase2_raw_score": 10.0},
        ],
        cfg_full,
    )

    low_rec, high_rec = rescored
    assert low_rec["phase2_raw_score"] == pytest.approx(10.0)
    assert high_rec["feature"].c_bar_2q > 0.0
    assert high_rec["feature"].phase2_burden_total == pytest.approx(high_rec["feature"].hardware_cost_denominator)
    assert high_rec["phase2_raw_score"] == pytest.approx(high_rec["feature"].phase2_raw_score)
    assert high_rec["phase2_raw_score"] < low_rec["phase2_raw_score"]


def test_rescore_hardware_cost_family_updates_simple_phase1_aliases() -> None:
    cfg = SimpleScoreConfig(
        lambda_compile=0.0,
        lambda_measure=0.0,
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        hardware_cost_scale_floor=1.0,
    )
    meas = MeasurementCacheAudit()
    oracle = Phase1CompileCostOracle()
    base = build_candidate_features(
        stage_name="core",
        candidate_label="base",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1, candidate_term=_term("x")),
        measurement_stats=meas.estimate([MeasurementGroupSpec("x", coeff_l2=1.0)]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    low = replace(base, candidate_label="low", c_hat_2q=0.0)
    high = replace(base, candidate_label="high", c_hat_2q=4.0)

    rescored = rescore_hardware_cost_family(
        [
            {"candidate_label": "low", "feature": low, "simple_score": float(low.simple_score or 0.0)},
            {"candidate_label": "high", "feature": high, "simple_score": float(high.simple_score or 0.0)},
        ],
        cfg,
    )

    low_rec, high_rec = rescored
    assert high_rec["feature"].c_bar_2q > 0.0
    assert high_rec["cheap_burden_total"] == pytest.approx(high_rec["feature"].hardware_cost_denominator)
    assert high_rec["simple_score"] == pytest.approx(high_rec["feature"].simple_score)
    assert high_rec["cheap_score"] == pytest.approx(high_rec["feature"].cheap_score)
    assert high_rec["selector_score"] == pytest.approx(high_rec["feature"].selector_score)
    assert high_rec["simple_score"] < low_rec["simple_score"]


def test_phase1_compile_cost_oracle_penalizes_heavier_pauli_structure() -> None:
    oracle = Phase1CompileCostOracle()
    light_term = type(
        "_DummyAnsatzTerm",
        (),
        {"label": "light", "polynomial": PauliPolynomial("JW", [PauliTerm(2, ps="xe", pc=1.0)])},
    )()
    heavy_term = type(
        "_DummyAnsatzTerm",
        (),
        {"label": "heavy", "polynomial": PauliPolynomial("JW", [PauliTerm(2, ps="xx", pc=1.0)])},
    )()
    light = oracle.estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=1,
        candidate_term=light_term,
    )
    heavy = oracle.estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=1,
        candidate_term=heavy_term,
    )
    assert heavy.gate_proxy_total > light.gate_proxy_total
    assert heavy.proxy_total > light.proxy_total


def test_compatibility_penalty_uses_measurement_mismatch_signal() -> None:
    cfg = FullScoreConfig(
        compat_overlap_weight=0.0,
        compat_comm_weight=0.0,
        compat_curv_weight=0.0,
        compat_sched_weight=0.0,
        compat_measure_weight=1.0,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()

    def _feat_and_record(label: str, term: object) -> dict[str, object]:
        feat = build_candidate_features(
            stage_name="core",
            candidate_label=str(label),
            candidate_family="core",
            candidate_pool_index=0,
            position_id=0,
            append_position=0,
            positions_considered=[0],
            gradient_signed=0.5,
            metric_proxy=0.5,
            sigma_hat=0.0,
            refit_window_indices=[0],
            compile_cost=oracle.estimate(
                candidate_term_count=1,
                position_id=0,
                append_position=0,
                refit_active_count=1,
                candidate_term=term,
            ),
            measurement_stats=meas.estimate(measurement_group_keys_for_term(term)),
            leakage_penalty=0.0,
            stage_gate_open=True,
            leakage_gate_open=True,
            trough_probe_triggered=False,
            trough_detected=False,
            cfg=SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0),
        )
        return {"feature": feat, "candidate_term": term}

    rec_xz = _feat_and_record("xz", _term("xz"))
    rec_ez = _feat_and_record("ez", _term("ez"))
    rec_yy = _feat_and_record("yy", _term("yy"))

    close_penalty = compatibility_penalty(record_a=rec_xz, record_b=rec_ez, cfg=cfg)
    far_penalty = compatibility_penalty(record_a=rec_xz, record_b=rec_yy, cfg=cfg)

    assert close_penalty["measurement_mismatch"] < far_penalty["measurement_mismatch"]
    assert close_penalty["total"] < far_penalty["total"]


def test_compatibility_penalty_oracle_caches_tangents_and_pair_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = FullScoreConfig(
        compat_overlap_weight=0.0,
        compat_comm_weight=0.0,
        compat_curv_weight=1.0,
        compat_sched_weight=0.0,
        compat_measure_weight=0.0,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()

    def _feat_and_record(label: str, term: object) -> dict[str, object]:
        feat = build_candidate_features(
            stage_name="core",
            candidate_label=str(label),
            candidate_family="core",
            candidate_pool_index=0,
            position_id=0,
            append_position=0,
            positions_considered=[0],
            gradient_signed=0.5,
            metric_proxy=0.5,
            sigma_hat=0.0,
            refit_window_indices=[0],
            compile_cost=oracle.estimate(
                candidate_term_count=1,
                position_id=0,
                append_position=0,
                refit_active_count=1,
                candidate_term=term,
            ),
            measurement_stats=meas.estimate(measurement_group_keys_for_term(term)),
            leakage_penalty=0.0,
            stage_gate_open=True,
            leakage_gate_open=True,
            trough_probe_triggered=False,
            trough_detected=False,
            cfg=SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0),
        )
        return {"feature": feat, "candidate_term": term}

    call_labels: list[str] = []

    def _fake_tangent_data(**kwargs):
        label = str(kwargs.get("label"))
        call_labels.append(label)
        if label == "xz":
            return np.asarray([1.0 + 0.0j, 0.0 + 0.0j]), 1.0
        return np.asarray([0.5 + 0.0j, 0.0 + 0.0j]), 0.25

    monkeypatch.setattr(
        "pipelines.scaffold.hh_continuation_scoring._tangent_data",
        _fake_tangent_data,
    )

    rec_xz = _feat_and_record("xz", _term("xz"))
    rec_ez = _feat_and_record("ez", _term("ez"))
    compat = CompatibilityPenaltyOracle(
        cfg=cfg,
        psi_state=np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),
        compiled_cache={},
        pauli_action_cache={},
    )

    first = compat.penalty(rec_xz, rec_ez)
    second = compat.penalty(rec_ez, rec_xz)
    third = compat.penalty(rec_xz, rec_ez)

    assert first == second == third
    assert call_labels.count("xz") == 1
    assert call_labels.count("ez") == 1


def test_shortlist_only_expensive_scoring_calls_oracles_for_shortlist() -> None:
    class _CountingGeometry(OrderedInsertionGeometryOracle):
        def __init__(self) -> None:
            self.calls = 0

        def estimate(self, *args, **kwargs):
            self.calls += 1
            return super().estimate(*args, **kwargs)

    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cheap_records = []
    for idx, grad in enumerate([0.9, 0.8, 0.3, 0.2]):
        feat = build_candidate_features(
            stage_name="core",
            candidate_label=f"x{idx}",
            candidate_family="core",
            candidate_pool_index=idx,
            position_id=0,
            append_position=0,
            positions_considered=[0],
            gradient_signed=float(grad),
            metric_proxy=float(grad),
            sigma_hat=0.0,
            refit_window_indices=[0],
            compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
            measurement_stats=meas.estimate([f"x{idx}"]),
            leakage_penalty=0.0,
            stage_gate_open=True,
            leakage_gate_open=True,
            trough_probe_triggered=False,
            trough_detected=False,
            cfg=SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0),
        )
        cheap_records.append(
            {
                "feature": feat,
                "simple_score": float(feat.simple_score or 0.0),
                "candidate_pool_index": idx,
                "position_id": 0,
                "candidate_term": _term("x"),
                "window_terms": [],
                "window_labels": [],
            }
        )
    shortlisted = shortlist_records(cheap_records, cfg=FullScoreConfig(shortlist_fraction=0.5, shortlist_size=2))
    novelty = _CountingGeometry()
    for rec in shortlisted:
        scaffold_context, h_compiled = _scaffold_context(
            psi_state=psi_ref,
            selected_ops=[],
            theta=[],
            refit_window_indices=[],
        )
        build_full_candidate_features(
            base_feature=rec["feature"],
            candidate_term=rec["candidate_term"],
            cfg=FullScoreConfig(shortlist_fraction=0.5, shortlist_size=2),
            novelty_oracle=novelty,
            curvature_oracle=Phase2CurvatureOracle(),
            scaffold_context=scaffold_context,
            h_compiled=h_compiled,
            compiled_cache={},
            pauli_action_cache={},
            optimizer_memory=None,
        )
    assert len(shortlisted) == 2
    assert novelty.calls == 2


def test_shortlist_records_can_tie_break_on_simple_score_for_phase2_authority() -> None:
    records = [
        {
            "phase2_raw_score": 1.0,
            "simple_score": 0.1,
            "candidate_pool_index": 0,
            "position_id": 0,
        },
        {
            "phase2_raw_score": 1.0,
            "simple_score": 9.0,
            "candidate_pool_index": 1,
            "position_id": 0,
        },
    ]
    shortlisted = shortlist_records(
        records,
        cfg=FullScoreConfig(shortlist_fraction=1.0, shortlist_size=2),
        score_key="phase2_raw_score",
        tie_break_score_key="simple_score",
    )
    assert [int(rec["candidate_pool_index"]) for rec in shortlisted] == [1, 0]


def test_remaining_evaluations_proxy_uses_remaining_depth_mode() -> None:
    got = remaining_evaluations_proxy(current_depth=2, max_depth=6, mode="remaining_depth")
    assert got == pytest.approx(5.0)


def test_remaining_evaluations_proxy_prefers_controller_useful_horizon() -> None:
    got = remaining_evaluations_proxy(
        current_depth=2,
        max_depth=6,
        mode="remaining_depth",
        controller_snapshot={"depth_left": 4, "n_rem_hat": 2.5, "useful_horizon": 2.5, "H_t": 2.5},
    )
    assert got == pytest.approx(2.5)


def test_lifetime_weight_components_are_zero_when_mode_off() -> None:
    cfg = FullScoreConfig(lifetime_cost_mode="off", lifetime_weight=1.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=0.5,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        current_depth=2,
        max_depth=6,
        lifetime_cost_mode="off",
        remaining_evaluations_proxy_mode="remaining_depth",
    )
    comps = lifetime_weight_components(feat, cfg)
    assert comps["remaining_evaluations_proxy"] == pytest.approx(5.0)
    assert comps["useful_horizon"] == pytest.approx(5.0)
    assert comps["H_t"] == pytest.approx(5.0)
    assert comps["total"] == pytest.approx(0.0)


def test_lifetime_weight_components_use_controller_useful_horizon_not_raw_remaining_depth() -> None:
    cfg = FullScoreConfig(lifetime_cost_mode="phase3_v1", lifetime_weight=1.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=0.5,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        current_depth=2,
        max_depth=6,
        lifetime_cost_mode="phase3_v1",
        remaining_evaluations_proxy_mode="remaining_depth",
        controller_snapshot={"depth_left": 4, "n_rem_hat": 1.5, "useful_horizon": 1.5, "runway_fraction": 0.375, "H_t": 1.5},
    )
    comps = lifetime_weight_components(feat, cfg)
    assert feat.remaining_evaluations_proxy == pytest.approx(1.5)
    assert comps["remaining_evaluations_proxy"] == pytest.approx(1.5)
    assert comps["useful_horizon"] == pytest.approx(1.5)
    assert comps["n_rem_hat"] == pytest.approx(1.5)
    assert comps["H_t"] == pytest.approx(1.5)


def test_full_v2_motif_bonus_tie_break_and_lifetime_weighting_are_deterministic() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_cost_mode="phase3_v1",
        remaining_evaluations_proxy_mode="remaining_depth",
        lifetime_weight=0.1,
        motif_bonus_weight=1.0,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=0.5,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        current_depth=1,
        max_depth=4,
        motif_bonus=0.2,
        lifetime_cost_mode="phase3_v1",
        remaining_evaluations_proxy_mode="remaining_depth",
    )
    feat = type(feat)(
        **{
            **feat.__dict__,
            "h_hat": 0.5,
            "curvature_mode": "self_only",
        }
    )
    score_with_bonus, _ = full_v2_score(feat, cfg)
    score_without_bonus, _ = full_v2_score(
        type(feat)(**{**feat.__dict__, "motif_bonus": 0.0}),
        cfg,
    )
    assert score_with_bonus == pytest.approx(score_without_bonus)
    components_with_bonus = phase3_canonical_score_components(feat, cfg)
    components_without_bonus = phase3_canonical_score_components(
        type(feat)(**{**feat.__dict__, "motif_bonus": 0.0}),
        cfg,
    )
    assert float(components_with_bonus["phase3_tie_break_score"]) > float(
        components_without_bonus["phase3_tie_break_score"]
    )
    cfg_ablation = FullScoreConfig(**{**cfg.__dict__, "auxiliary_score_mode": "ablation_additive"})
    score_ablation, _ = full_v2_score(feat, cfg_ablation)
    assert score_ablation > score_with_bonus


def test_build_candidate_features_carries_generator_and_symmetry_metadata() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(6, ps="eyeexy", pc=1.0),
            PauliTerm(6, ps="eyeeyx", pc=-1.0),
        ],
    )
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    meta = build_generator_metadata(
        label="macro_candidate",
        polynomial=poly,
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="macro_candidate",
        candidate_family="paop_lf_std",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.4,
        metric_proxy=0.4,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=2, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["macro"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        generator_metadata=meta.__dict__,
        symmetry_spec=sym.__dict__,
        symmetry_mode="phase3_shared_spec",
        symmetry_mitigation_mode="verify_only",
        current_depth=0,
        max_depth=3,
        lifetime_cost_mode="phase3_v1",
        remaining_evaluations_proxy_mode="remaining_depth",
    )
    assert feat.generator_id == meta.generator_id
    assert feat.template_id == meta.template_id
    assert feat.is_macro_generator is True
    assert feat.symmetry_mode == "phase3_shared_spec"
    assert feat.symmetry_mitigation_mode == "verify_only"
    assert feat.remaining_evaluations_proxy == pytest.approx(4.0)


def test_signed_backend_compile_cost_reward_is_telemetry_only_in_simple_score() -> None:
    cfg = SimpleScoreConfig(lambda_compile=1.0, lambda_measure=0.0, lambda_leak=0.0, burden_floor=0.25)
    meas = MeasurementCacheAudit()
    cost = CompileCostEstimate(
        new_pauli_actions=1.0,
        new_rotation_steps=1.0,
        position_shift_span=0.0,
        refit_active_count=1.0,
        proxy_total=3.0,
        source_mode="backend_transpile_v1",
        penalty_total=-0.5,
        depth_surrogate=-0.5,
        compile_gate_open=True,
        raw_delta_compiled_count_2q=-1.0,
        raw_delta_compiled_depth=-2.0,
        raw_delta_compiled_size=-3.0,
        delta_compiled_count_2q=0.0,
        delta_compiled_depth=0.0,
        delta_compiled_size=0.0,
    )
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="cancel",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=1.0,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert feat.compile_cost_total == pytest.approx(-0.5)
    assert feat.hardware_cost_denominator == pytest.approx(1.0)
    expected_delta = trust_region_drop(float(feat.g_hw_lcb), 1.0, 1.0, 0.25)
    assert float(feat.simple_score or 0.0) == pytest.approx(expected_delta)


def test_signed_backend_compile_cost_is_telemetry_only_in_full_score() -> None:
    cfg = FullScoreConfig(wD=1.0, wG=0.0, wC=0.0, wP=0.0, wc=0.0, lifetime_weight=0.0, burden_floor=0.25)
    meas = MeasurementCacheAudit()

    def _feat(label: str, depth_cost: float):
        cost = CompileCostEstimate(
            new_pauli_actions=1.0,
            new_rotation_steps=1.0,
            position_shift_span=0.0,
            refit_active_count=1.0,
            proxy_total=1.0,
            source_mode="backend_transpile_v1",
            penalty_total=depth_cost,
            depth_surrogate=depth_cost,
            compile_gate_open=True,
        )
        return build_candidate_features(
            stage_name="core",
            candidate_label=label,
            candidate_family="core",
            candidate_pool_index=0,
            position_id=0,
            append_position=0,
            positions_considered=[0],
            gradient_signed=0.5,
            metric_proxy=1.0,
            sigma_hat=0.0,
            refit_window_indices=[0],
            compile_cost=cost,
            measurement_stats=meas.estimate(["x"]),
            leakage_penalty=0.0,
            stage_gate_open=True,
            leakage_gate_open=True,
            trough_probe_triggered=False,
            trough_detected=False,
            cfg=SimpleScoreConfig(),
            cheap_score_cfg=cfg,
        )

    cancel = _feat("cancel", -10.0)
    costly = _feat("costly", 1.0)
    cancel_score, _ = full_v2_score(cancel, cfg)
    costly_score, _ = full_v2_score(costly, cfg)
    cancel_components = phase3_canonical_score_components(cancel, cfg)
    assert cancel_components["denominator_1_plus_K3"] == pytest.approx(1.0)
    assert cancel_components["K3"] == pytest.approx(0.0)
    assert cancel_score == pytest.approx(costly_score)
    assert cancel_score < float("inf")
