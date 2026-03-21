from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_scoring import (
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase2CurvatureOracle,
    Phase2NoveltyOracle,
    Phase1CompileCostOracle,
    SimpleScoreConfig,
    build_full_candidate_features,
    build_candidate_features,
    compatibility_penalty,
    full_v2_score,
    lifetime_weight_components,
    measurement_group_keys_for_term,
    remaining_evaluations_proxy,
    shortlist_records,
    trust_region_drop,
)
from pipelines.hardcoded.hh_continuation_generators import build_generator_metadata
from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm


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


def test_simple_v1_uses_g_abs_not_g_lcb_for_ranking() -> None:
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
        metric_proxy=0.0,
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
    assert float(feat.simple_score or 0.0) == pytest.approx(0.4)


def test_measurement_cache_reuse_accounting() -> None:
    cache = MeasurementCacheAudit(nominal_shots_per_group=10)
    first = cache.estimate(["a", "b"])
    assert first.groups_new == 2
    cache.commit(["a", "b"])
    second = cache.estimate(["a", "b", "c"])
    assert second.groups_reused == 2
    assert second.groups_new == 1
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


def _term(label: str) -> object:
    return type(
        "_DummyAnsatzTerm",
        (),
        {
            "label": str(label),
            "polynomial": PauliPolynomial("JW", [PauliTerm(len(str(label)), ps=str(label), pc=1.0)]),
        },
    )()


def test_trust_region_drop_matches_newton_branch() -> None:
    got = trust_region_drop(0.4, 2.0, 1.0, 1.0)
    assert got == pytest.approx(0.04)


def test_full_v2_score_falls_back_safely_without_window_curvature() -> None:
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5, gamma_N=1.0, wD=0.0, wG=0.0, wC=0.0, wP=0.0, wc=0.0)
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
    assert fallback == "self_curvature_only"


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
    feat = build_full_candidate_features(
        base_feature=base,
        psi_state=psi_ref,
        candidate_term=_term("x"),
        window_terms=[_term("x")],
        window_labels=["x"],
        cfg=FullScoreConfig(shortlist_size=2),
        novelty_oracle=Phase2NoveltyOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
    )
    assert 0.0 <= float(feat.novelty or 0.0) <= 1.0
    assert feat.refit_window_indices == [0]
    assert feat.full_v2_score is not None


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


def test_shortlist_only_expensive_scoring_calls_oracles_for_shortlist() -> None:
    class _CountingNovelty(Phase2NoveltyOracle):
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
    novelty = _CountingNovelty()
    for rec in shortlisted:
        build_full_candidate_features(
            base_feature=rec["feature"],
            psi_state=psi_ref,
            candidate_term=rec["candidate_term"],
            window_terms=[],
            window_labels=[],
            cfg=FullScoreConfig(shortlist_fraction=0.5, shortlist_size=2),
            novelty_oracle=novelty,
            curvature_oracle=Phase2CurvatureOracle(),
            compiled_cache={},
            pauli_action_cache={},
            optimizer_memory=None,
        )
    assert len(shortlisted) == 2
    assert novelty.calls == 2


def test_remaining_evaluations_proxy_uses_remaining_depth_mode() -> None:
    got = remaining_evaluations_proxy(current_depth=2, max_depth=6, mode="remaining_depth")
    assert got == pytest.approx(5.0)


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
    assert comps["total"] == pytest.approx(0.0)


def test_full_v2_motif_bonus_and_lifetime_weighting_are_deterministic() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        gamma_N=1.0,
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
    assert score_with_bonus > score_without_bonus


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
