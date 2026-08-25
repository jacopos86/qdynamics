from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.scaffold import hh_continuation_scoring as scoring
from pipelines.scaffold.hh_continuation_scoring import (
    FullScoreConfig,
    HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
    HISTORICAL_SINGLETON_PHASE2_WHITENED_NO_N2_SCORE_FORMULA,
    HistoricalSingletonOldOldGeometryPrior,
    evaluate_historical_singleton_coordinate_models,
    evaluate_historical_singleton_phase2_coordinate_models,
    rescore_historical_phase3_records_with_coordinate_models,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
)
from pipelines.static_adapt.route_a_schur_selector import (
    ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
    TrustRegionUpdateConfig,
)
from pipelines.static_adapt.route_a_trust_region import (
    HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1,
    HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1,
    RouteATrustRegionState,
    historical_singleton_scalar_selector_summary,
    update_geometry_expansion_trust_region_state,
    update_trust_region_state,
)
from pipelines.static_adapt.sr_snake._selection import (
    _CandidatePositionRecord,
)


_WHITENED_POLICY = "supported_metric_whitened_eigh_v1"
_GLOBAL_TRUST_POLICY = JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2


def test_fresh_phase3_receipt_preserves_deferred_gram_novelty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deferred = {
        "schema": "phase3_deferred_gram_novelty_v1",
        "Q_window": [[1.0]],
        "q_reduced": [0.25],
        "F_red": 1.0,
        "metric_collapse": False,
    }
    acquired = {
        "schema": "phase2_joint_geometry_reuse_v1",
        "append_position": 1,
        "G_AA": [[1.0]],
        "G_AB": [0.0],
        "G_BB": 1.0,
        "H_AA": [[1.0]],
        "H_AB": [0.0],
        "H_BB": 1.0,
        "descent_gradient": 0.5,
        "phase3_deferred_gram_novelty": deferred,
    }
    scaffold = SimpleNamespace(
        old_old_geometry_measured=True,
        old_old_metric_measured=True,
        old_old_hessian_measured=True,
        old_old_hessian_status="measured",
        old_old_hessian_fingerprint="hessian",
        old_old_hessian_provenance={
            "source": "test",
            "measured": True,
        },
        refit_window_indices=(0,),
        state_reconstruction_delta_norm=0.0,
        dpsi_window=(np.asarray([1.0 + 0.0j]),),
        hpsi_state=np.asarray([0.0 + 0.0j]),
        state_fingerprint="state",
        ordered_scaffold_fingerprint="scaffold",
        theta_fingerprint="theta",
    )
    monkeypatch.setattr(
        scoring,
        "_compiled_polynomial_fingerprint",
        lambda _compiled: "hamiltonian",
    )
    monkeypatch.setattr(
        scoring,
        "_candidate_coordinate_fingerprint",
        lambda _term, *, position_id: f"candidate:{position_id}",
    )

    promoted = scoring._promote_fresh_phase3_joint_geometry_receipt(
        acquired_payload=acquired,
        scaffold_context=scaffold,
        candidate_term=object(),
        position_id=1,
        h_compiled=object(),
        state_consistency_tolerance=1e-12,
    )

    assert promoted["phase3_deferred_gram_novelty"] == deferred


def test_stationary_source_receipt_never_evaluates_active_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired = {
        "schema": "phase2_joint_geometry_reuse_v1",
        "append_position": 1,
        "G_AA": [[1.0]],
        "G_AB": [0.0],
        "G_BB": 1.0,
        "H_AA": [[1.0]],
        "H_AB": [0.0],
        "H_BB": 1.0,
        "descent_gradient": 0.5,
    }
    scaffold = SimpleNamespace(
        old_old_geometry_measured=True,
        old_old_metric_measured=True,
        old_old_hessian_measured=True,
        old_old_hessian_status="measured",
        old_old_hessian_fingerprint="hessian",
        old_old_hessian_provenance={
            "source": "test",
            "measured": True,
        },
        refit_window_indices=(0,),
        state_reconstruction_delta_norm=0.0,
        dpsi_window=(np.asarray([1.0 + 0.0j]),),
        hpsi_state=np.asarray([2.0 + 0.0j]),
        state_fingerprint="state",
        ordered_scaffold_fingerprint="scaffold",
        theta_fingerprint="theta",
    )
    monkeypatch.setattr(
        scoring,
        "_compiled_polynomial_fingerprint",
        lambda _compiled: "hamiltonian",
    )
    monkeypatch.setattr(
        scoring,
        "_candidate_coordinate_fingerprint",
        lambda _term, *, position_id: f"candidate:{position_id}",
    )
    monkeypatch.setattr(
        scoring.np,
        "vdot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stationary-source active gradient was evaluated")
        ),
    )

    promoted = scoring._promote_fresh_phase3_joint_geometry_receipt(
        acquired_payload=acquired,
        scaffold_context=scaffold,
        candidate_term=object(),
        position_id=1,
        h_compiled=object(),
        state_consistency_tolerance=1e-12,
        active_gradient_policy="stationary_source_response_v1",
    )

    assert promoted["g_A"] == [0.0]
    assert promoted["active_gradient_indices_acquired"] == []
    assert (
        promoted["active_gradient_source"]
        == "not_acquired_stationary_source_protocol"
    )


def _feature(**overrides: object) -> CandidateFeatures:
    values: dict[str, object] = dict(
        stage_name="phase3",
        candidate_label="candidate",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        g_signed=-1.0,
        g_abs=1.0,
        g_lcb=1.0,
        sigma_hat=0.0,
        F=1.0,
        novelty=0.5,
        curvature_mode="append_exact_reduced_path",
        novelty_mode="append_exact_tangent_context_v1",
        refit_window_indices=[0],
        compiled_position_cost_proxy={},
        measurement_cache_stats={},
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        simple_score=0.1,
        score_version="full_v2",
        F_raw=1.0,
        F_red=1.0,
        h_eff=2.0,
        phase2_raw_score=9.0,
        phase2_raw_trust_gain=8.0,
        phase2_burden_total=7.0,
        phase3_reduced_trust_gain=0.1875,
        phase3_burden_total=2.0,
        phase3_primary_score=0.046875,
        phase3_tie_break_score=0.03,
        phase3_auxiliary_score_mode="tie_break_only",
        full_v2_score=0.046875,
        selector_score=0.046875,
        selector_burden=2.0,
        schur_window_indices=[0],
        phase_score_components={
            "DeltaE_TR": 0.1875,
            "phase3_delta_e_tr": 0.1875,
            "phase3_g_hw_lcb": 1.0,
            "N3": 0.5,
            "phase3_N3": 0.5,
            "K3": 1.0,
            "phase3_K3": 1.0,
            "denominator_1_plus_K3": 2.0,
            "phase3_denominator_1_plus_K3": 2.0,
            "phase3_primary_score": 0.046875,
            "phase3_tie_break_score": 0.03,
            "phase3_reduced_score": 0.046875,
            "selector_score": 0.046875,
        },
        phase_cost_components={"sentinel_cost": 11.0},
        phase2_joint_geometry_reuse={"schema": "prior_phase2_geometry"},
        actual_fallback_mode="append_exact_reduced_path",
    )
    values.update(overrides)
    return CandidateFeatures(**values)


def test_final_supported_response_refreezes_typed_shortlist_rank() -> None:
    feature = _feature(shortlist_rank=2, shortlist_size=2)
    record = {
        "candidate_label": "candidate",
        "candidate_pool_index": 0,
        "position_id": 0,
        "phase3_primary_score": 0.046875,
        "phase3_tie_break_score": 0.03,
        "shortlist_rank": 2,
        "shortlist_size": 2,
        "feature": feature,
    }

    frozen = adapt_pipeline._default_no_prune_freeze_shortlist_ranks(
        [record]
    )

    assert frozen[0]["shortlist_rank"] == 1
    assert frozen[0]["shortlist_size"] == 1
    assert frozen[0]["feature"].shortlist_rank == 1
    assert frozen[0]["feature"].shortlist_size == 1
    assert frozen[0]["phase3_primary_score"] == record[
        "phase3_primary_score"
    ]
    assert frozen[0]["phase3_tie_break_score"] == record[
        "phase3_tie_break_score"
    ]

    domain_record = _CandidatePositionRecord(
        domain_record_id="domain:0",
        generator_id="generator:0",
        parent_generator_id=None,
        pool_index=0,
        pool_label="candidate",
        insertion_position=0,
        symmetry_identity="symmetry:0",
        lineage_identity=("generator:0",),
    )
    receipts = adapt_pipeline._default_no_prune_shortlist_rank_receipts(
        frozen,
        primary_score_key="phase3_primary_score",
        tie_break_score_key="phase3_tie_break_score",
        domain_by_pool_position={(0, 0): domain_record},
    )

    assert len(receipts) == 1
    assert receipts[0].shortlist_rank == 1
    assert receipts[0].record_key == ("domain:0", "generator:0")


def test_phase3_only_live_prune_cannot_bypass_identity_gate_without_marker() -> None:
    payload = {
        "valid": False,
        "noncanonical_reasons": [
            "mismatch:phase1_prune_mode:live!=both",
        ],
    }
    with pytest.raises(ValueError, match="phase1_prune_mode"):
        adapt_pipeline._validate_resolved_static_route_identity(
            payload,
            declared_route_id="route_a",
            historical_singleton_overlay_active=True,
            route_profile_conformance="registered_profile",
            phase1_prune_enabled=True,
            phase1_prune_mode="live",
        )


def _record(label: str, index: int, feature: CandidateFeatures) -> dict[str, Any]:
    return {
        "candidate_label": str(label),
        "candidate_pool_index": int(index),
        "position_id": int(feature.position_id),
        "feature": feature,
        "phase2_raw_score": float(feature.phase2_raw_score or 0.0),
        "full_v2_score": float(feature.full_v2_score or 0.0),
        "unrelated": {"preserve": True},
    }


def _escape_summary(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": True,
        "reason": "supported_metric_global_trust_model_v2",
        "joint_gain": 0.5,
        "joint_linear_solve_policy_effective": _GLOBAL_TRUST_POLICY,
        "trust_global_optimality_certified": True,
        "raw_metric_support_status": "resolved",
        "raw_metric_support_reason": "stable_supported_cluster",
        "raw_metric_null_compatibility_certified": True,
        "raw_metric_null_compatibility_reason": "raw_metric_null_compatible",
        "stationary_certified": True,
        "nonstationary_certified": False,
        "supported_stationarity_status": "stationary",
        "stationarity_margin": -1.0e-8,
        "marginal_trust_gain_comparison_valid": True,
        "marginal_trust_gain_comparison_tolerance": 0.0,
        "quotient_participation_resolved": True,
        "quotient_participation_lower_bound": 0.5,
        "negative_curvature_certified": True,
        "positive_semidefinite_certified": False,
        "supported_inertia_status": "negative",
        "supported_inertia_label_issued": True,
        "minimum_hessian_eigenvalue_upper_bound": -1.0,
        "minimum_hessian_eigenvalue_lower_bound": -1.0,
        "full_trust_gain": 0.5,
        "active_restricted_trust_gain": 0.0,
        "sr_escape_state_stationarity_summary": {
            "schema": "sr_escape_state_stationarity_certificate_v1",
            "valid": True,
            "reason": "working_state_supported_stationarity_certified",
            "state_fingerprint": "model-working-state",
            "workspace_fingerprint": "workspace",
            "active_coordinate_count": 1,
            "active_coordinate_identities": ["active"],
            "trust_radius": 0.25,
            "joint_linear_solve_policy": _GLOBAL_TRUST_POLICY,
            "comparison_scope": (
                "working_state_active_coordinates_independent_of_singleton_v1"
            ),
            "supported_stationarity_status": "stationary",
            "supported_gradient_norm_upper_bound": 1.0e-9,
            "supported_gradient_resolution": 2.0e-9,
            "stationarity_margin": -1.0e-9,
            "raw_metric_support_status": "resolved",
            "raw_metric_null_compatibility_certified": True,
            "support_provenance_digest": "state-support-provenance",
            "trust_provenance_digest": "state-trust-provenance",
        },
        "sr_escape_ordinary_summary": {
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": False,
            "reason": "ordinary_positive_gain_unavailable",
            "joint_gain": 0.0,
            "joint_gain_lower_bound": 0.0,
            "joint_linear_solve_policy_effective": (
                "supported_metric_whitened_eigh_v1"
            ),
            "joint_batch_context_mode": "full_ansatz_v1",
        },
    }
    values.update(overrides)
    values.setdefault(
        "full_trust_gain_lower_bound",
        float(values.get("full_trust_gain", 0.0)),
    )
    values.setdefault(
        "full_trust_gain_upper_bound",
        float(values.get("full_trust_gain", 0.0)),
    )
    values.setdefault(
        "active_restricted_trust_gain_lower_bound",
        float(values.get("active_restricted_trust_gain", 0.0)),
    )
    values.setdefault(
        "active_restricted_trust_gain_upper_bound",
        float(values.get("active_restricted_trust_gain", 0.0)),
    )
    return values


def _escape_feature(
    label: str,
    index: int,
    *,
    n3: float = 0.5,
    k3: float = 1.0,
    summary_overrides: dict[str, object] | None = None,
) -> CandidateFeatures:
    components = {
        **_feature().phase_score_components,
        "N3": float(n3),
        "phase3_N3": float(n3),
        "K3": float(k3),
        "phase3_K3": float(k3),
        "denominator_1_plus_K3": float(1.0 + k3),
        "phase3_denominator_1_plus_K3": float(1.0 + k3),
    }
    feature = _feature(
        candidate_label=label,
        candidate_pool_index=index,
        phase3_burden_total=float(1.0 + k3),
        phase_score_components=components,
    )
    return replace(
        feature,
        phase2_joint_geometry_reuse=_escape_summary(
            **dict(summary_overrides or {})
        ),
    )


def _state_stationarity_workspace(
    gradient: float,
) -> scoring._BatchFullGeometryWorkspace:
    return scoring._BatchFullGeometryWorkspace(
        records=(),
        record_index={},
        ansatz_depth=1,
        active_indices=(0,),
        active_labels=("active",),
        G_AA=np.eye(1, dtype=float),
        H_AA=np.eye(1, dtype=float),
        G_AB=np.zeros((1, 0), dtype=float),
        H_AB=np.zeros((1, 0), dtype=float),
        G_BB=np.zeros((0, 0), dtype=float),
        H_BB=np.zeros((0, 0), dtype=float),
        g_A=np.asarray([gradient], dtype=float),
        g_B=np.zeros(0, dtype=float),
        phase2_reported_g_B=np.zeros(0, dtype=float),
        geometry_mode="full_residual_gram_hessian_v1",
        joint_context_mode="full_ansatz_v1",
        workspace_fingerprint="workspace",
        state_fingerprint="state",
        metric_regularization=1.0e-9,
        energy_regularization=1.0e-9,
        joint_linear_solve_policy=_GLOBAL_TRUST_POLICY,
        rank_relative_tolerance=1.0e-6,
        max_gram_condition_number=1.0e12,
        max_fubini_study_step=0.25,
        state_delta_norm=0.0,
        state_consistency_tolerance=1.0e-8,
        phase2_reuse_validation={},
        _subset_cache={},
    )


def test_shared_state_stationarity_audit_is_independent_of_singletons() -> None:
    stationary = _state_stationarity_workspace(
        0.0
    )._state_stationarity_summary()
    nonstationary = _state_stationarity_workspace(
        1.0e-2
    )._state_stationarity_summary()

    assert stationary["valid"] is True
    assert stationary["supported_stationarity_status"] == "stationary"
    assert stationary["stationarity_margin"] <= 0.0
    assert stationary["state_fingerprint"] == "state"
    assert stationary["support_provenance_digest"]
    assert stationary["trust_provenance_digest"]
    assert nonstationary["valid"] is False
    assert nonstationary["supported_stationarity_status"] == (
        "certified_nonstationary"
    )


def test_historical_coordinate_evaluator_preserves_order_and_phase2_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _feature(candidate_label="first", candidate_pool_index=7)
    second = _feature(candidate_label="second", candidate_pool_index=3)
    records = [_record("first", 7, first), _record("second", 3, second)]

    class _Workspace:
        def summary_for_records(self, subset):  # noqa: ANN001, ANN201
            index = int(subset[0]["candidate_pool_index"])
            return {
                "feasible": True,
                "reason": "supported_metric_whitened_full_joint_model_v1",
                "joint_gain": float(index) / 10.0,
                "joint_fubini_study_displacement_sq": 0.0625,
                "trust_regularization_applied": True,
                "trust_radius_binding": True,
                "joint_linear_solve_policy_effective": (
                    "supported_metric_whitened_eigh_v1"
                ),
            }

        def build_telemetry(self) -> dict[str, object]:
            return {"workspace_marker": "one_shared_workspace"}

    def _fake_workspace(rows, *, cfg, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        assert [row["candidate_label"] for row in rows] == ["first", "second"]
        assert cfg.batch_target_size == 1
        assert cfg.batch_size_cap == 1
        assert _kwargs["old_old_geometry_prior"] is None
        return _Workspace()

    monkeypatch.setattr(
        scoring,
        "_build_batch_full_geometry_workspace",
        _fake_workspace,
    )
    evaluation = evaluate_historical_singleton_coordinate_models(
        records,
        cfg=FullScoreConfig(batch_target_size=4, batch_size_cap=4),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.asarray([1.0], dtype=complex),
        psi_state=np.asarray([1.0], dtype=complex),
        h_compiled=object(),
        scope="historical_phase3_test",
    )

    assert [row["candidate_label"] for row in evaluation.records] == [
        "first",
        "second",
    ]
    assert [row["phase2_raw_score"] for row in evaluation.records] == [9.0, 9.0]
    assert [row["unrelated"] for row in evaluation.records] == [
        {"preserve": True},
        {"preserve": True},
    ]
    for original, updated in zip(records, evaluation.records):
        original_feature = original["feature"]
        updated_feature = updated["feature"]
        assert isinstance(original_feature, CandidateFeatures)
        assert isinstance(updated_feature, CandidateFeatures)
        attached = updated_feature.phase2_joint_geometry_reuse
        assert attached["schema"] == HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA
        assert attached["joint_gain"] == pytest.approx(
            float(original["candidate_pool_index"]) / 10.0
        )
        assert attached["historical_phase2_joint_geometry_reuse"] == {
            "schema": "prior_phase2_geometry"
        }
        assert replace(
            updated_feature,
            phase2_joint_geometry_reuse=original_feature.phase2_joint_geometry_reuse,
        ) == original_feature
    assert evaluation.telemetry["phase2_rescoring_applied"] is False
    assert evaluation.telemetry["batching_applied"] is False
    assert evaluation.telemetry["geometry_workspace"] == {
        "workspace_marker": "one_shared_workspace"
    }


def test_historical_coordinate_evaluator_forwards_typed_old_old_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature = _feature(candidate_label="candidate", candidate_pool_index=2)
    record = _record("candidate", 2, feature)
    prior = HistoricalSingletonOldOldGeometryPrior(
        active_indices=(),
        active_coordinate_identities=(),
        G_AA=np.zeros((0, 0), dtype=float),
        H_AA=np.zeros((0, 0), dtype=float),
        g_A=np.zeros(0, dtype=float),
        state_fingerprint="state",
        ordered_scaffold_fingerprint="scaffold",
        theta_fingerprint="theta",
        hamiltonian_fingerprint="hamiltonian",
        source_prior_id="prior",
        source_state_id="source-state",
        source_frame_id="frame",
        source_support_id="support",
        source_geometry_status="predicted",
        source_provenance_ids=("provenance",),
    )

    class _Workspace:
        def summary_for_records(self, _subset):  # noqa: ANN001, ANN201
            return {
                "feasible": True,
                "reason": "fixture",
                "joint_gain": 0.2,
            }

        def build_telemetry(self) -> dict[str, object]:
            return {"old_old_geometry_reacquired": False}

    def _fake_workspace(*_args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        assert kwargs["old_old_geometry_prior"] is prior
        return _Workspace()

    monkeypatch.setattr(
        scoring,
        "_build_batch_full_geometry_workspace",
        _fake_workspace,
    )
    evaluation = evaluate_historical_singleton_coordinate_models(
        [record],
        cfg=FullScoreConfig(),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.asarray([1.0], dtype=complex),
        psi_state=np.asarray([1.0], dtype=complex),
        h_compiled=object(),
        old_old_geometry_prior=prior,
    )

    assert evaluation.telemetry["old_old_geometry_reacquired"] is False
    assert evaluation.telemetry["old_old_geometry_prior"][
        "prior_fingerprint"
    ] == prior.prior_fingerprint


def test_phase2_whitening_substitutes_only_benefit_and_preserves_sr_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _feature(
        candidate_label="first",
        candidate_pool_index=7,
        phase2_raw_novelty=0.25,
        phase2_raw_trust_gain=8.0,
        phase2_raw_score=1.0,
        phase2_burden_total=2.0,
    )
    second = _feature(
        candidate_label="second",
        candidate_pool_index=3,
        phase2_raw_novelty=0.8,
        phase2_raw_trust_gain=6.0,
        phase2_raw_score=1.2,
        phase2_burden_total=4.0,
    )
    records = [_record("first", 7, first), _record("second", 3, second)]

    class _Workspace:
        def summary_for_records(self, subset):  # noqa: ANN001, ANN201
            label = str(subset[0]["candidate_label"])
            return {
                "feasible": True,
                "reason": "supported_metric_whitened_full_joint_model_v1",
                "joint_gain": 0.6 if label == "first" else 0.5,
                "joint_batch_context_mode": "full_ansatz_v1",
                "joint_linear_solve_policy_effective": _WHITENED_POLICY,
                "joint_fubini_study_displacement_sq": 0.04,
                "applied_predicted_reduction": 0.6 if label == "first" else 0.5,
            }

        def build_telemetry(self) -> dict[str, object]:
            return {
                "workspace_marker": "phase2_shared_workspace",
                "required_candidate_pair_count": 0,
            }

    def _fake_workspace(rows, *, cfg, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        assert [row["candidate_label"] for row in rows] == ["first", "second"]
        assert cfg.batch_target_size == 1
        assert cfg.batch_size_cap == 1
        assert cfg.rho == pytest.approx(0.37)
        return _Workspace()

    monkeypatch.setattr(
        scoring,
        "_build_batch_full_geometry_workspace",
        _fake_workspace,
    )
    evaluation = evaluate_historical_singleton_phase2_coordinate_models(
        records,
        cfg=FullScoreConfig(
            batch_target_size=4,
            batch_size_cap=4,
            rho=0.37,
            batch_joint_linear_solve_policy=_WHITENED_POLICY,
        ),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.asarray([1.0], dtype=complex),
        psi_state=np.asarray([1.0], dtype=complex),
        h_compiled=object(),
    )

    assert [row["candidate_label"] for row in evaluation.records] == [
        "first",
        "second",
    ]
    assert [row["phase2_raw_score"] for row in evaluation.records] == (
        pytest.approx([0.6 / 2.0, 0.5 / 4.0])
    )
    for original, updated in zip(records, evaluation.records):
        old_feature = original["feature"]
        new_feature = updated["feature"]
        assert isinstance(old_feature, CandidateFeatures)
        assert isinstance(new_feature, CandidateFeatures)
        assert new_feature.phase2_raw_novelty == old_feature.phase2_raw_novelty
        assert new_feature.phase2_burden_total == old_feature.phase2_burden_total
        assert new_feature.phase2_joint_geometry_reuse == (
            old_feature.phase2_joint_geometry_reuse
        )
        assert new_feature.full_v2_score == old_feature.full_v2_score
        assert new_feature.phase3_primary_score == old_feature.phase3_primary_score
        assert new_feature.phase3_burden_total == old_feature.phase3_burden_total
        assert updated["unrelated"] == original["unrelated"]
        model = new_feature.historical_singleton_phase2_coordinate_model
        assert model["schema"] == HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA
        assert model["authority"] == "phase2_supported_response_gain_only"
        assert model["preserved_N2"] == pytest.approx(
            old_feature.phase2_raw_novelty
        )
        assert model["phase2_novelty_authority"] == "passive_provenance_only"
        assert model["phase2_novelty_multiplier"] is None
        assert model["phase2_novelty_applied"] is False
        assert model["phase2_joint_geometry_reuse_preserved"] == {
            "schema": "prior_phase2_geometry"
        }
        assert new_feature.phase2_raw_score_formula == (
            HISTORICAL_SINGLETON_PHASE2_WHITENED_NO_N2_SCORE_FORMULA
        )
    assert evaluation.telemetry["membership_preserved"] is True
    assert evaluation.telemetry["order_preserved"] is True
    assert (
        evaluation.telemetry["historical_n2_retained_as_passive_provenance"]
        is True
    )
    assert evaluation.telemetry["passive_n2_record_count"] == 2
    assert evaluation.telemetry["phase2_novelty_applied"] is False
    assert evaluation.telemetry["measured_n2_retained"] is True
    assert evaluation.telemetry["cost_denominator_preserved"] is True
    assert evaluation.telemetry["batching_applied"] is False
    assert evaluation.telemetry["candidate_pair_measurement_count"] == 0


def test_phase2_gain_over_cost_accepts_null_n2_and_keeps_supported_whitening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature = _feature(
        candidate_label="candidate",
        phase2_raw_novelty=None,
        phase2_raw_trust_gain=8.0,
        phase2_raw_score=1.0,
        phase2_burden_total=2.0,
    )

    class _Workspace:
        def summary_for_records(self, subset):  # noqa: ANN001, ANN201
            assert subset[0]["candidate_label"] == "candidate"
            return {
                "feasible": True,
                "reason": "supported_metric_whitened_full_joint_model_v1",
                "joint_gain": 0.6,
                "joint_linear_solve_policy_effective": _WHITENED_POLICY,
                "joint_fubini_study_displacement_sq": 0.04,
                "applied_predicted_reduction": 0.6,
            }

        def build_telemetry(self) -> dict[str, object]:
            return {"required_candidate_pair_count": 0}

    monkeypatch.setattr(
        scoring,
        "_build_batch_full_geometry_workspace",
        lambda *_args, **_kwargs: _Workspace(),
    )
    evaluation = evaluate_historical_singleton_phase2_coordinate_models(
        [_record("candidate", 0, feature)],
        cfg=FullScoreConfig(
            batch_joint_linear_solve_policy=_WHITENED_POLICY,
        ),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.asarray([1.0], dtype=complex),
        psi_state=np.asarray([1.0], dtype=complex),
        h_compiled=object(),
    )

    updated = evaluation.records[0]["feature"]
    model = updated.historical_singleton_phase2_coordinate_model
    assert evaluation.records[0]["phase2_raw_score"] == pytest.approx(0.3)
    assert updated.phase2_raw_novelty is None
    assert updated.phase2_raw_score_formula == (
        HISTORICAL_SINGLETON_PHASE2_WHITENED_NO_N2_SCORE_FORMULA
    )
    assert model["measured_N2"] is None
    assert model["phase2_novelty_multiplier"] is None
    assert model["phase2_novelty_query_charge"] == 0
    assert evaluation.telemetry["measured_n2_retained"] is False
    assert evaluation.telemetry["passive_n2_record_count"] == 0
    assert evaluation.telemetry["phase2_novelty_applied"] is False
    assert evaluation.telemetry["candidate_pair_measurement_count"] == 0


def test_historical_phase3_rescorer_changes_only_benefit_dependent_fields() -> None:
    first_feature = replace(
        _feature(candidate_label="first", candidate_pool_index=7),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": True,
            "reason": "supported",
            "joint_gain": 0.6,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        },
    )
    second_feature = replace(
        _feature(candidate_label="second", candidate_pool_index=3),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": True,
            "reason": "supported",
            "joint_gain": 0.8,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        },
    )
    records = [
        _record("first", 7, first_feature),
        _record("second", 3, second_feature),
    ]

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        records,
        cfg=FullScoreConfig(),
    )

    assert [row["candidate_label"] for row in rescored] == ["first", "second"]
    assert [row["phase2_raw_score"] for row in rescored] == [9.0, 9.0]
    assert [row["full_v2_score"] for row in rescored] == pytest.approx(
        [0.30, 0.40]
    )
    for original, updated, expected_gain in zip(records, rescored, (0.6, 0.8)):
        old_feature = original["feature"]
        new_feature = updated["feature"]
        assert isinstance(old_feature, CandidateFeatures)
        assert isinstance(new_feature, CandidateFeatures)
        assert new_feature.phase2_raw_score == old_feature.phase2_raw_score
        assert new_feature.phase2_raw_trust_gain == old_feature.phase2_raw_trust_gain
        assert new_feature.phase2_burden_total == old_feature.phase2_burden_total
        assert new_feature.phase3_reduced_novelty == old_feature.phase3_reduced_novelty
        assert new_feature.phase3_burden_total == old_feature.phase3_burden_total
        assert new_feature.phase3_tie_break_score == old_feature.phase3_tie_break_score
        assert new_feature.phase_cost_components == {"sentinel_cost": 11.0}
        components = new_feature.phase_score_components
        assert components["phase3_historical_scalar_DeltaE_TR"] == pytest.approx(
            0.1875
        )
        assert components["DeltaE_TR"] == pytest.approx(expected_gain)
        assert components["N3"] is None
        assert components["phase3_measured_novelty"] == pytest.approx(0.5)
        assert components["phase3_novelty_multiplier"] is None
        assert components["phase3_novelty_applied"] is False
        assert components["K3"] == pytest.approx(1.0)
        assert components["denominator_1_plus_K3"] == pytest.approx(2.0)
    assert telemetry["membership_preserved"] is True
    assert telemetry["order_preserved"] is True
    assert telemetry["phase2_rescoring_applied"] is False
    assert telemetry["batching_applied"] is False
    assert telemetry["geometry_expansion_active"] is False
    assert telemetry["measured_n3_retained"] is True
    assert telemetry["phase3_novelty_applied"] is False


def _symmetric_cost_coordinate_records(
    *,
    feasible: bool,
) -> tuple[list[dict[str, Any]], FullScoreConfig]:
    policy = (
        scoring.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
    )
    cfg = FullScoreConfig(
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        z_alpha=0.0,
        hardware_cost_normalization_mode=policy,
        deferred_gram_fallback_enabled=True,
    )
    records: list[dict[str, Any]] = []
    for index, raw_cost in enumerate((1.0, 3.0, 5.0)):
        components = {
            **_feature().phase_score_components,
            "N3": None,
            "phase3_N3": None,
        }
        geometry: dict[str, Any] = {
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": bool(feasible),
            "reason": "supported" if feasible else "rank_gate",
            "joint_gain": 0.8 if feasible else 0.0,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        }
        if not feasible:
            geometry["historical_phase2_joint_geometry_reuse"] = {
                "phase3_deferred_gram_novelty": {
                    "schema": "phase3_deferred_gram_novelty_v1",
                    "Q_window": [],
                    "q_reduced": [],
                    "F_red": 1.0,
                    "metric_collapse": False,
                }
            }
        feature = replace(
            _feature(
                candidate_label=f"candidate_{index}",
                candidate_pool_index=index,
                c_hat_2q=float(raw_cost),
                novelty=None,
                phase_score_components=components,
            ),
            phase2_joint_geometry_reuse=geometry,
        )
        records.append(_record(f"candidate_{index}", index, feature))
    return scoring.rescore_hardware_cost_family(records, cfg), cfg


def _zero_centered_signed_cost_coordinate_records(
    *,
    feasible: bool,
) -> tuple[list[dict[str, Any]], FullScoreConfig]:
    policy = (
        scoring.HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
    )
    cfg = FullScoreConfig(
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        z_alpha=0.0,
        hardware_cost_normalization_mode=policy,
        phase3_signed_factor_consumer_semantic_version=(
            scoring.PHASE3_ZERO_CENTERED_SIGNED_FACTOR_CONSUMER_SEMANTIC_VERSION
        ),
        deferred_gram_fallback_enabled=True,
    )
    records: list[dict[str, Any]] = []
    for index, signed_delta in enumerate((-1.0, 1.0)):
        components = {
            **_feature().phase_score_components,
            "N3": None,
            "phase3_N3": None,
        }
        geometry: dict[str, Any] = {
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": bool(feasible),
            "reason": "supported" if feasible else "rank_gate",
            "joint_gain": 0.8 if feasible else 0.0,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        }
        if not feasible:
            geometry["historical_phase2_joint_geometry_reuse"] = {
                "phase3_deferred_gram_novelty": {
                    "schema": "phase3_deferred_gram_novelty_v1",
                    "Q_window": [],
                    "q_reduced": [],
                    "F_red": 1.0,
                    "metric_collapse": False,
                }
            }
        feature = replace(
            _feature(
                candidate_label=f"candidate_{index}",
                candidate_pool_index=index,
                novelty=None,
                generator_id=f"generator::candidate_{index}",
                compile_cost_source="backend_transpile_v1",
                compiled_position_cost_backend={
                    "negative_delta_reward_enabled": True,
                    "raw_delta_compiled_count_2q": signed_delta,
                    "raw_delta_compiled_depth_2q": 0.0,
                    "raw_delta_compiled_count_1q": 0.0,
                    "base_structure_key": "a" * 64,
                    "trial_structure_key": str(index + 1) * 64,
                },
                phase_score_components=components,
            ),
            phase2_joint_geometry_reuse=geometry,
        )
        records.append(_record(f"candidate_{index}", index, feature))
    return scoring.rescore_hardware_cost_family(records, cfg), cfg


def _multiplicative_signed_cost_coordinate_records(
    policy: str,
    *,
    feasible: bool,
) -> tuple[list[dict[str, Any]], FullScoreConfig]:
    if policy == (
        scoring.HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
    ):
        return _zero_centered_signed_cost_coordinate_records(feasible=feasible)
    if policy == (
        scoring.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
    ):
        return _symmetric_cost_coordinate_records(feasible=feasible)
    raise AssertionError(f"unsupported test policy {policy!r}")


def test_phase3_coordinate_rescore_applies_symmetric_cost_factor_to_energy_score() -> None:
    records, cfg = _symmetric_cost_coordinate_records(feasible=True)

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        records,
        cfg=cfg,
    )

    assert [row["hardware_cost_score_factor"] for row in rescored] == pytest.approx(
        [1.25, 1.0, 0.75]
    )
    assert [row["full_v2_score"] for row in rescored] == pytest.approx(
        [1.0, 0.8, 0.6]
    )
    assert [row["selector_score"] for row in rescored] == pytest.approx(
        [1.0, 0.8, 0.6]
    )
    assert [row["phase3_primary_score"] for row in rescored] == pytest.approx(
        [1.0, 0.8, 0.6]
    )
    assert [
        row["feature"].phase3_primary_score for row in rescored
    ] == pytest.approx([1.0, 0.8, 0.6])
    assert telemetry["hardware_cost_normalization_mode"] == (
        scoring.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
    )
    assert telemetry["symmetric_hardware_cost_factor_applied"] is True
    assert telemetry["score_formula"] == (
        "DeltaE_TR * hardware_cost_score_factor / (1 + K3)"
    )
    assert [row["hardware_cost_score_factor"] for row in telemetry["records"]] == (
        pytest.approx([1.25, 1.0, 0.75])
    )
    assert [
        row["feature"].phase_score_components["hardware_cost_score_factor"]
        for row in rescored
    ] == pytest.approx([1.25, 1.0, 0.75])


def test_phase3_coordinate_rescore_preserves_zero_centered_signed_factors() -> None:
    records, cfg = _zero_centered_signed_cost_coordinate_records(feasible=True)
    assert [row["hardware_cost_score_factor"] for row in records] == pytest.approx(
        [1.25, 0.75]
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        records,
        cfg=cfg,
    )

    assert [row["hardware_cost_score_factor"] for row in rescored] == pytest.approx(
        [1.25, 0.75]
    )
    assert [row["full_v2_score"] for row in rescored] == pytest.approx(
        [1.0, 0.6]
    )
    assert telemetry["hardware_cost_normalization_mode"] == (
        scoring.HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
    )
    assert telemetry["phase3_signed_factor_consumer_semantic_version"] == (
        scoring.PHASE3_ZERO_CENTERED_SIGNED_FACTOR_CONSUMER_SEMANTIC_VERSION
    )
    assert telemetry["symmetric_hardware_cost_factor_applied"] is True


@pytest.mark.parametrize(
    "semantic_version",
    (None, "stale_affected_v0"),
)
def test_zero_centered_phase3_consumer_rejects_historical_semantics(
    semantic_version: str | None,
) -> None:
    records, cfg = _zero_centered_signed_cost_coordinate_records(feasible=True)
    historical_cfg = replace(
        cfg,
        phase3_signed_factor_consumer_semantic_version=semantic_version,
    )

    with pytest.raises(RuntimeError, match="semantic implementation version"):
        rescore_historical_phase3_records_with_coordinate_models(
            records,
            cfg=historical_cfg,
        )


def test_phase3_coordinate_rescore_rejects_stale_signed_factor_population() -> None:
    records, cfg = _zero_centered_signed_cost_coordinate_records(feasible=True)
    stale_hash = "0" * 64
    stale_record = dict(records[1])
    stale_feature = stale_record["feature"]
    assert isinstance(stale_feature, CandidateFeatures)
    stale_record["feature"] = replace(
        stale_feature,
        hardware_cost_population_hash=stale_hash,
        hardware_cost_normalization={
            **stale_feature.hardware_cost_normalization,
            "population_hash": stale_hash,
        },
        phase_cost_components={
            **stale_feature.phase_cost_components,
            "hardware_cost_population_hash": stale_hash,
        },
    )
    stale_record["hardware_cost_population_hash"] = stale_hash

    with pytest.raises(ValueError, match="population"):
        rescore_historical_phase3_records_with_coordinate_models(
            [records[0], stale_record],
            cfg=cfg,
        )


def test_phase3_coordinate_rescore_rejects_dropped_signed_population_member() -> None:
    records, cfg = _zero_centered_signed_cost_coordinate_records(feasible=True)

    with pytest.raises(ValueError, match="population"):
        rescore_historical_phase3_records_with_coordinate_models(
            [records[0]],
            cfg=cfg,
        )


def test_phase3_coordinate_rescore_rejects_stale_compiled_ansatz_identity() -> None:
    records, cfg = _zero_centered_signed_cost_coordinate_records(feasible=True)
    stale_record = dict(records[1])
    stale_feature = stale_record["feature"]
    assert isinstance(stale_feature, CandidateFeatures)
    assert isinstance(stale_feature.compiled_position_cost_backend, dict)
    stale_record["feature"] = replace(
        stale_feature,
        compiled_position_cost_backend={
            **stale_feature.compiled_position_cost_backend,
            "trial_structure_key": "f" * 64,
        },
    )

    with pytest.raises(ValueError, match="population"):
        rescore_historical_phase3_records_with_coordinate_models(
            [records[0], stale_record],
            cfg=cfg,
        )


def test_signed_hardware_cost_population_hash_binds_normalization_policy() -> None:
    feature = _feature(
        generator_id="generator::candidate",
        compile_cost_source="backend_transpile_v1",
        compiled_position_cost_backend={
            "negative_delta_reward_enabled": True,
            "raw_delta_compiled_count_2q": 0.0,
            "raw_delta_compiled_depth_2q": 0.0,
            "raw_delta_compiled_count_1q": 0.0,
            "base_structure_key": "a" * 64,
            "trial_structure_key": "b" * 64,
        },
    )
    symmetric_cfg = FullScoreConfig(
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        hardware_cost_normalization_mode=(
            scoring.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
        ),
    )
    zero_centered_cfg = replace(
        symmetric_cfg,
        hardware_cost_normalization_mode=(
            scoring.HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
        ),
    )

    symmetric = scoring.normalize_hardware_cost_feature_family(
        [feature], symmetric_cfg
    )[0]
    zero_centered = scoring.normalize_hardware_cost_feature_family(
        [feature], zero_centered_cfg
    )[0]

    assert symmetric.hardware_cost_population_hash is not None
    assert zero_centered.hardware_cost_population_hash is not None
    assert symmetric.hardware_cost_population_hash == (
        "1129ea4b631947936dd5cbea1f0cbfb902cadc2f1921bedee2d2da5d46ec2e56"
    )
    assert (
        symmetric.hardware_cost_population_hash
        != zero_centered.hardware_cost_population_hash
    )


@pytest.mark.parametrize(
    ("identity_field", "replacement_key"),
    (
        ("base_structure_key", "c" * 64),
        ("trial_structure_key", "d" * 64),
    ),
)
def test_zero_centered_population_hash_binds_compiled_ansatz_identity(
    identity_field: str,
    replacement_key: str,
) -> None:
    backend = {
        "negative_delta_reward_enabled": True,
        "raw_delta_compiled_count_2q": -1.0,
        "raw_delta_compiled_depth_2q": 0.0,
        "raw_delta_compiled_count_1q": 0.0,
        "base_structure_key": "a" * 64,
        "trial_structure_key": "b" * 64,
    }
    baseline = _feature(
        generator_id="generator::candidate",
        compile_cost_source="backend_transpile_v1",
        compiled_position_cost_backend=backend,
    )
    changed = replace(
        baseline,
        compiled_position_cost_backend={
            **backend,
            identity_field: replacement_key,
        },
    )
    cfg = FullScoreConfig(
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        hardware_cost_normalization_mode=(
            scoring.HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
        ),
    )

    baseline_normalized = scoring.normalize_hardware_cost_feature_family(
        [baseline], cfg
    )[0]
    changed_normalized = scoring.normalize_hardware_cost_feature_family(
        [changed], cfg
    )[0]

    assert baseline_normalized.hardware_cost_population_hash is not None
    assert changed_normalized.hardware_cost_population_hash is not None
    assert (
        baseline_normalized.hardware_cost_population_hash
        != changed_normalized.hardware_cost_population_hash
    )


def test_zero_centered_population_hash_binds_generator_identity() -> None:
    backend = {
        "negative_delta_reward_enabled": True,
        "raw_delta_compiled_count_2q": -1.0,
        "raw_delta_compiled_depth_2q": 0.0,
        "raw_delta_compiled_count_1q": 0.0,
        "base_structure_key": "a" * 64,
        "trial_structure_key": "b" * 64,
    }
    baseline = _feature(
        generator_id="generator::baseline",
        compile_cost_source="backend_transpile_v1",
        compiled_position_cost_backend=backend,
    )
    changed = replace(baseline, generator_id="generator::changed")
    cfg = FullScoreConfig(
        lambda_2q=1.0,
        lambda_d=0.0,
        lambda_1q=0.0,
        lambda_theta=0.0,
        lambda_shot=0.0,
        hardware_cost_normalization_mode=(
            scoring.HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
        ),
    )

    baseline_normalized = scoring.normalize_hardware_cost_feature_family(
        [baseline], cfg
    )[0]
    changed_normalized = scoring.normalize_hardware_cost_feature_family(
        [changed], cfg
    )[0]

    assert (
        baseline_normalized.hardware_cost_population_hash
        != changed_normalized.hardware_cost_population_hash
    )


@pytest.mark.parametrize(
    "missing_identity",
    ("generator_id", "base_structure_key", "trial_structure_key"),
)
def test_zero_centered_population_hash_requires_exact_ansatz_identity(
    missing_identity: str,
) -> None:
    backend = {
        "negative_delta_reward_enabled": True,
        "raw_delta_compiled_count_2q": -1.0,
        "raw_delta_compiled_depth_2q": 0.0,
        "raw_delta_compiled_count_1q": 0.0,
        "base_structure_key": "a" * 64,
        "trial_structure_key": "b" * 64,
    }
    generator_id: str | None = "generator::candidate"
    if missing_identity == "generator_id":
        generator_id = None
    else:
        backend.pop(missing_identity)
    feature = _feature(
        generator_id=generator_id,
        compile_cost_source="backend_transpile_v1",
        compiled_position_cost_backend=backend,
    )
    cfg = FullScoreConfig(
        hardware_cost_normalization_mode=(
            scoring.HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
        ),
    )

    with pytest.raises(ValueError, match=missing_identity):
        scoring.normalize_hardware_cost_feature_family([feature], cfg)


@pytest.mark.parametrize(
    ("policy", "expected_factors", "expected_scores"),
    [
        (
            scoring.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1,
            [1.25, 1.0, 0.75],
            [1.0, 0.8, 0.6],
        ),
        (
            scoring.HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1,
            [1.25, 0.75],
            [1.0, 0.6],
        ),
    ],
    ids=("family-robust-symmetric", "zero-centered-signed"),
)
def test_phase3_coordinate_rescore_closes_signed_factor_energy_score(
    policy: str,
    expected_factors: list[float],
    expected_scores: list[float],
) -> None:
    records, cfg = _multiplicative_signed_cost_coordinate_records(
        policy,
        feasible=True,
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        records,
        cfg=cfg,
    )

    assert [row["hardware_cost_score_factor"] for row in rescored] == pytest.approx(
        expected_factors
    )
    assert [row["full_v2_score"] for row in rescored] == pytest.approx(
        expected_scores
    )
    assert [
        row["feature"].phase_score_components["denominator_1_plus_K3"]
        for row in rescored
    ] == pytest.approx([1.0] * len(expected_scores))
    assert telemetry["score_formula"] == (
        "DeltaE_TR * hardware_cost_score_factor / (1 + K3)"
    )


@pytest.mark.parametrize(
    ("policy", "expected_scores"),
    [
        (
            scoring.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1,
            [1.25, 1.0, 0.75],
        ),
        (
            scoring.HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1,
            [1.25, 0.75],
        ),
    ],
    ids=("family-robust-symmetric", "zero-centered-signed"),
)
def test_phase3_coordinate_rescore_closes_signed_factor_infeasible_fallback(
    policy: str,
    expected_scores: list[float],
) -> None:
    records, cfg = _multiplicative_signed_cost_coordinate_records(
        policy,
        feasible=False,
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        records,
        cfg=cfg,
    )

    assert [row["full_v2_score"] for row in rescored] == pytest.approx(
        expected_scores
    )
    assert [
        row["route_a_geometry_expansion_score"] for row in rescored
    ] == pytest.approx(expected_scores)
    assert telemetry["geometry_expansion_active"] is True
    assert telemetry["geometry_expansion_score_formula"] == (
        "N3*hardware_cost_score_factor/(1+K3)"
    )


def test_phase3_coordinate_rescore_rejects_missing_signed_normalization() -> None:
    records, cfg = _zero_centered_signed_cost_coordinate_records(feasible=True)
    missing = dict(records[0])
    feature = missing["feature"]
    assert isinstance(feature, CandidateFeatures)
    missing["feature"] = replace(feature, hardware_cost_normalization={})

    with pytest.raises(ValueError, match="normalization receipt"):
        rescore_historical_phase3_records_with_coordinate_models(
            [missing, records[1]],
            cfg=cfg,
        )


def test_phase3_coordinate_rescore_rejects_mixed_signed_policies() -> None:
    zero_records, cfg = _zero_centered_signed_cost_coordinate_records(feasible=True)
    symmetric_records, _ = _symmetric_cost_coordinate_records(feasible=True)

    with pytest.raises(ValueError, match="policy"):
        rescore_historical_phase3_records_with_coordinate_models(
            [zero_records[0], symmetric_records[0]],
            cfg=cfg,
        )


def test_phase3_coordinate_rescore_rejects_stale_signed_score_factor() -> None:
    records, cfg = _zero_centered_signed_cost_coordinate_records(feasible=True)
    stale = dict(records[0])
    feature = stale["feature"]
    assert isinstance(feature, CandidateFeatures)
    stale["feature"] = replace(
        feature,
        hardware_cost_score_factor=1.0,
        hardware_cost_normalization={
            **feature.hardware_cost_normalization,
            "score_factor": 1.0,
        },
        phase_cost_components={
            **feature.phase_cost_components,
            "hardware_cost_score_factor": 1.0,
        },
    )

    with pytest.raises(ValueError, match="score-factor"):
        rescore_historical_phase3_records_with_coordinate_models(
            [stale, records[1]],
            cfg=cfg,
        )


def test_phase3_coordinate_rescore_applies_symmetric_cost_factor_to_fallback_score() -> None:
    records, cfg = _symmetric_cost_coordinate_records(feasible=False)

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        records,
        cfg=cfg,
    )

    assert [row["full_v2_score"] for row in rescored] == pytest.approx(
        [1.25, 1.0, 0.75]
    )
    assert [
        row["route_a_geometry_expansion_score"] for row in rescored
    ] == pytest.approx([1.25, 1.0, 0.75])
    assert telemetry["geometry_expansion_active"] is True
    assert telemetry["symmetric_hardware_cost_factor_applied"] is True
    assert telemetry["geometry_expansion_score_formula"] == (
        "N3*hardware_cost_score_factor/(1+K3)"
    )


def test_phase3_coordinate_rescore_preserves_legacy_denominator_semantics() -> None:
    feature = replace(
        _feature(hardware_cost_score_factor=1.5),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": True,
            "reason": "supported",
            "joint_gain": 0.6,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("candidate", 0, feature)],
        cfg=FullScoreConfig(),
    )

    assert rescored[0]["full_v2_score"] == pytest.approx(0.3)
    assert rescored[0]["hardware_cost_score_factor"] == pytest.approx(1.0)
    assert rescored[0]["feature"].phase_score_components[
        "hardware_cost_score_factor"
    ] == pytest.approx(1.0)
    assert telemetry["records"][0]["hardware_cost_score_factor"] == pytest.approx(
        1.0
    )
    assert telemetry["symmetric_hardware_cost_factor_applied"] is False


def test_phase3_coordinate_rescore_rejects_unresolved_symmetric_cost_feature() -> None:
    feature = replace(
        _feature(),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": True,
            "reason": "supported",
            "joint_gain": 0.6,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        },
    )

    with pytest.raises(ValueError, match="population-normalized"):
        rescore_historical_phase3_records_with_coordinate_models(
            [_record("candidate", 0, feature)],
            cfg=FullScoreConfig(
                hardware_cost_normalization_mode=(
                    scoring.HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
                )
            ),
        )


def test_phase3_ordinary_ranking_ignores_n3_with_deferred_gram_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_lazy_novelty(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError(
            "ordinary Phase-III ranking must not compute deferred Gram geometry"
        )

    monkeypatch.setattr(
        scoring,
        "_compute_deferred_phase3_gram_novelty",
        _unexpected_lazy_novelty,
    )
    components = {
        **_feature().phase_score_components,
        "N3": None,
        "phase3_N3": None,
    }
    feature = replace(
        _feature(
            novelty=None,
            phase_score_components=components,
        ),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": True,
            "reason": "supported",
            "joint_gain": 0.6,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("candidate", 0, feature)],
        cfg=FullScoreConfig(deferred_gram_fallback_enabled=True),
    )

    updated = rescored[0]["feature"]
    assert rescored[0]["full_v2_score"] == pytest.approx(0.3)
    assert updated.phase_score_components["N3"] is None
    assert updated.phase_score_components["phase3_measured_novelty"] is None
    assert updated.phase_score_components["phase3_novelty_multiplier"] is None
    assert updated.phase_score_components["phase3_novelty_applied"] is False
    assert telemetry["geometry_expansion_active"] is False
    assert telemetry["phase3_novelty_status"] == (
        "not_computed_for_ordinary_scoring"
    )
    assert telemetry["phase3_novelty_query_charge"] == 0
    assert telemetry["phase3_novelty_classical_solve_count"] == 0


def test_deferred_gram_computes_residual_only_for_all_infeasible_geometry() -> None:
    def _fallback_feature(
        *,
        label: str,
        index: int,
        denominator: float,
        q_reduced: list[float],
        q_window: list[list[float]],
        reason: str,
    ) -> CandidateFeatures:
        components = {
            **_feature().phase_score_components,
            "N3": None,
            "phase3_N3": None,
            "K3": float(denominator - 1.0),
            "phase3_K3": float(denominator - 1.0),
            "denominator_1_plus_K3": float(denominator),
            "phase3_denominator_1_plus_K3": float(denominator),
        }
        return replace(
            _feature(
                candidate_label=label,
                candidate_pool_index=index,
                novelty=None,
                phase3_burden_total=denominator,
                phase_score_components=components,
            ),
            phase2_joint_geometry_reuse={
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": False,
                "reason": reason,
                "joint_gain": 0.0,
                "joint_linear_solve_policy_effective": _WHITENED_POLICY,
                "historical_phase2_joint_geometry_reuse": {
                    "phase3_deferred_gram_novelty": {
                        "schema": "phase3_deferred_gram_novelty_v1",
                        "Q_window": q_window,
                        "q_reduced": q_reduced,
                        "F_red": 1.0,
                        "metric_collapse": False,
                    }
                },
            },
        )

    first = _fallback_feature(
        label="empty",
        index=0,
        denominator=2.0,
        q_reduced=[],
        q_window=[],
        reason="rank_gate",
    )
    second = _fallback_feature(
        label="projected",
        index=1,
        denominator=4.0,
        q_reduced=[0.5],
        q_window=[[1.0]],
        reason="conditioning_gate",
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("empty", 0, first), _record("projected", 1, second)],
        cfg=FullScoreConfig(
            deferred_gram_fallback_enabled=True,
            deferred_gram_fallback_ridge=0.0,
        ),
    )

    assert [row["full_v2_score"] for row in rescored] == pytest.approx(
        [0.5, 0.1875]
    )
    assert all(
        row["route_a_geometry_expansion_mode"]
        == "collective_span_novelty_over_cost_v1"
        for row in rescored
    )
    assert telemetry["geometry_expansion_active"] is True
    assert telemetry["geometry_expansion_lazy_novelty_activation"] is True
    assert telemetry["geometry_expansion_score_formula"] == "N3/(1+K3)"
    assert telemetry["phase3_novelty_status"] == (
        "computed_for_geometry_expansion_fallback"
    )
    assert telemetry["phase3_novelty_query_charge"] == 0
    assert telemetry["phase3_novelty_classical_solve_count"] == 1
    assert [row["phase3_novelty_classical_solve_count"] for row in telemetry["records"]] == [0, 1]


def test_mixed_coordinate_models_admit_only_feasible_finite_rows() -> None:
    feasible = replace(
        _feature(candidate_label="feasible", candidate_pool_index=1),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": True,
            "reason": "supported",
            "joint_gain": 0.4,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        },
    )
    rank_gated = replace(
        _feature(candidate_label="rank_gated", candidate_pool_index=2),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": False,
            "reason": "rank_gate",
            "joint_gain": 0.0,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        },
    )
    rescored, rescore_summary = (
        rescore_historical_phase3_records_with_coordinate_models(
            [
                _record("feasible", 1, feasible),
                _record("rank_gated", 2, rank_gated),
            ],
            cfg=FullScoreConfig(),
        )
    )

    domain, telemetry = (
        adapt_pipeline._historical_singleton_coordinate_admission_domain(
            rescored,
            rescore_summary=rescore_summary,
        )
    )

    assert [row["candidate_label"] for row in domain] == ["feasible"]
    assert telemetry["policy"] == "feasible_energy_models_only_v1"
    assert telemetry["feasible_count"] == 1
    assert telemetry["filtered_count"] == 1
    assert telemetry["filtered_records"][0]["candidate_label"] == "rank_gated"
    assert telemetry["filtered_records"][0]["coordinate_model_reason"] == "rank_gate"


def test_sr_ordinary_admission_uses_preserved_v1_model_after_v2_rank_gate() -> None:
    ordinary_summary = {
        "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": True,
        "reason": "supported",
        "joint_gain": 0.125,
        "joint_gain_lower_bound": 0.1,
        "joint_linear_solve_policy_effective": _WHITENED_POLICY,
    }
    feature = replace(
        _feature(candidate_label="ordinary", candidate_pool_index=4),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": False,
            "reason": "physical_active_image_subspace_rotation_unresolved",
            "joint_gain": 0.0,
            "joint_linear_solve_policy_effective": _GLOBAL_TRUST_POLICY,
            "sr_escape_ordinary_summary": ordinary_summary,
        },
    )
    record = {
        **_record("ordinary", 4, feature),
        "full_v2_score": 0.0625,
        "phase3_coordinate_model_feasible": False,
        "sr_escape_admission_eligible": True,
        "sr_escape_decision_kind": "ordinary",
    }

    domain, telemetry = (
        adapt_pipeline._historical_singleton_coordinate_admission_domain(
            [record],
            rescore_summary={"geometry_expansion_active": False},
            coordinate_solve_policy=_GLOBAL_TRUST_POLICY,
            sr_escape_active=True,
        )
    )

    assert [row["candidate_label"] for row in domain] == ["ordinary"]
    assert telemetry["policy"] == (
        "sr_escape_controller_eligible_singletons_only_v1"
    )
    assert telemetry["feasible_count"] == 1
    assert telemetry["filtered_count"] == 0
    adapt_pipeline._assert_historical_singleton_coordinate_plan_admissible(
        {
            **feature.__dict__,
            "sr_escape_admission_eligible": True,
            "sr_escape_decision_kind": "ordinary",
        },
        whitening_active=True,
        coordinate_solve_policy=_GLOBAL_TRUST_POLICY,
        sr_escape_active=True,
    )


@pytest.mark.parametrize(
    ("ordinary_summary", "error_match"),
    [
        (None, "missing its preserved v1"),
        (
            {
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": True,
                "joint_linear_solve_policy_effective": _GLOBAL_TRUST_POLICY,
            },
            "wrong preserved v1 linear-solve policy",
        ),
        (
            {
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": False,
                "reason": "rank_gate",
                "joint_linear_solve_policy_effective": _WHITENED_POLICY,
            },
            "infeasible energy model",
        ),
    ],
)
def test_sr_ordinary_plan_guard_fails_closed_on_invalid_preserved_v1_model(
    ordinary_summary: dict[str, object] | None,
    error_match: str,
) -> None:
    coordinate_summary: dict[str, object] = {
        "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": False,
        "reason": "physical_active_image_subspace_rotation_unresolved",
        "joint_linear_solve_policy_effective": _GLOBAL_TRUST_POLICY,
    }
    if ordinary_summary is not None:
        coordinate_summary["sr_escape_ordinary_summary"] = ordinary_summary
    feature = replace(
        _feature(candidate_label="ordinary_guard"),
        phase2_joint_geometry_reuse=coordinate_summary,
    )

    with pytest.raises(RuntimeError, match=error_match):
        adapt_pipeline._assert_historical_singleton_coordinate_plan_admissible(
            {
                **feature.__dict__,
                "sr_escape_admission_eligible": True,
                "sr_escape_decision_kind": "ordinary",
            },
            whitening_active=True,
            coordinate_solve_policy=_GLOBAL_TRUST_POLICY,
            sr_escape_active=True,
        )


def test_sr_saddle_plan_guard_keeps_outer_v2_feasibility_authority() -> None:
    feature = replace(
        _feature(candidate_label="saddle_guard"),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": False,
            "reason": "rank_gate",
            "joint_linear_solve_policy_effective": _GLOBAL_TRUST_POLICY,
            "sr_escape_ordinary_summary": {
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": True,
                "joint_linear_solve_policy_effective": _WHITENED_POLICY,
            },
        },
    )

    with pytest.raises(RuntimeError, match="infeasible energy model"):
        adapt_pipeline._assert_historical_singleton_coordinate_plan_admissible(
            {
                **feature.__dict__,
                "sr_escape_admission_eligible": True,
                "sr_escape_decision_kind": "saddle_singleton",
            },
            whitening_active=True,
            coordinate_solve_policy=_GLOBAL_TRUST_POLICY,
            sr_escape_active=True,
        )


def test_sr_ordinary_domain_retains_outer_v2_telemetry_guard() -> None:
    feature = replace(
        _feature(candidate_label="ordinary_telemetry"),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": False,
            "reason": "rank_gate",
            "joint_linear_solve_policy_effective": _GLOBAL_TRUST_POLICY,
            "sr_escape_ordinary_summary": {
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": True,
                "joint_gain": 0.125,
                "joint_gain_lower_bound": 0.1,
                "joint_linear_solve_policy_effective": _WHITENED_POLICY,
            },
        },
    )
    record = {
        **_record("ordinary_telemetry", 5, feature),
        "full_v2_score": 0.0625,
        "phase3_coordinate_model_feasible": True,
        "sr_escape_admission_eligible": True,
        "sr_escape_decision_kind": "ordinary",
    }

    with pytest.raises(RuntimeError, match="feasibility telemetry disagrees"):
        adapt_pipeline._historical_singleton_coordinate_admission_domain(
            [record],
            rescore_summary={"geometry_expansion_active": False},
            coordinate_solve_policy=_GLOBAL_TRUST_POLICY,
            sr_escape_active=True,
        )


def test_all_infeasible_deferred_gram_domain_retains_expansion_rows() -> None:
    rows: list[dict[str, Any]] = []
    for index, reason in enumerate(("rank_gate", "conditioning_gate")):
        feature = replace(
            _feature(candidate_label=f"candidate_{index}", candidate_pool_index=index),
            phase2_joint_geometry_reuse={
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": False,
                "reason": reason,
                "joint_gain": 0.0,
                "joint_linear_solve_policy_effective": _WHITENED_POLICY,
                "historical_phase2_joint_geometry_reuse": {
                    "phase3_deferred_gram_novelty": {
                        "schema": "phase3_deferred_gram_novelty_v1",
                        "Q_window": [],
                        "q_reduced": [],
                        "F_red": 1.0,
                        "metric_collapse": False,
                    }
                },
            },
        )
        rows.append(_record(f"candidate_{index}", index, feature))
    rescored, rescore_summary = (
        rescore_historical_phase3_records_with_coordinate_models(
            rows,
            cfg=FullScoreConfig(deferred_gram_fallback_enabled=True),
        )
    )

    domain, telemetry = (
        adapt_pipeline._historical_singleton_coordinate_admission_domain(
            rescored,
            rescore_summary=rescore_summary,
        )
    )

    assert [row["candidate_label"] for row in domain] == [
        "candidate_0",
        "candidate_1",
    ]
    assert telemetry["geometry_expansion_active"] is True
    assert telemetry["filtered_count"] == 0


def test_phase3_batch_restores_authoritative_singleton_coordinate_receipt() -> None:
    receipt = {
        "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": True,
        "joint_gain": 0.125,
        "joint_linear_solve_policy_effective": _WHITENED_POLICY,
    }
    authority_feature = replace(
        _feature(candidate_label="candidate", candidate_pool_index=3),
        phase2_joint_geometry_reuse=receipt,
        phase3_response_supported_rank=4,
    )
    selected_feature = replace(
        authority_feature,
        phase2_joint_geometry_reuse={
            "schema": "phase3_batch_joint_workspace_v1",
        },
        phase3_response_supported_rank=None,
        phase_score_components={"phase3_batch_size": 3},
    )
    authority = _record("candidate", 3, authority_feature)
    selected = {
        **_record("candidate", 3, selected_feature),
        "phase3_batch_size": 3,
    }

    restored = adapt_pipeline._restore_phase3_batch_singleton_coordinate_receipts(
        [selected],
        authoritative_records=[authority],
        coordinate_solve_policy=_WHITENED_POLICY,
    )

    restored_feature = restored[0]["feature"]
    assert isinstance(restored_feature, CandidateFeatures)
    assert restored_feature.phase2_joint_geometry_reuse == receipt
    assert restored_feature.phase3_response_supported_rank == 4
    assert restored_feature.phase_score_components["phase3_batch_size"] == 3
    assert restored[0]["phase3_batch_size"] == 3
    assert restored[0]["phase3_batch_singleton_coordinate_receipt"] == {
        "schema": "phase3_batch_singleton_coordinate_receipt_restoration_v1",
        "source": "authoritative_full_response_admission_record_v1",
        "candidate_identity": [3, 0, "candidate"],
        "coordinate_solve_policy": _WHITENED_POLICY,
        "supported_rank": 4,
    }


def test_phase3_batch_receipt_restore_rejects_raw_fallback_identity() -> None:
    selected = _record(
        "outside-domain",
        7,
        _feature(candidate_label="outside-domain", candidate_pool_index=7),
    )

    with pytest.raises(RuntimeError, match="outside the authoritative"):
        adapt_pipeline._restore_phase3_batch_singleton_coordinate_receipts(
            [selected],
            authoritative_records=[],
            coordinate_solve_policy=_WHITENED_POLICY,
        )


def test_phase3_batch_empty_selection_falls_back_within_authoritative_domain() -> None:
    receipt = {
        "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": True,
        "joint_gain": 0.25,
        "joint_linear_solve_policy_effective": _WHITENED_POLICY,
    }
    authority_feature = replace(
        _feature(candidate_label="authoritative", candidate_pool_index=3),
        phase2_joint_geometry_reuse=receipt,
        phase3_response_supported_rank=4,
    )
    authority = _record("authoritative", 3, authority_feature)
    raw = _record(
        "raw-phase2",
        9,
        _feature(candidate_label="raw-phase2", candidate_pool_index=9),
    )

    selected = adapt_pipeline._phase3_batch_empty_selection_fallback(
        [],
        batch_source_records=[authority],
        authoritative_records=[authority],
        raw_phase2_records=[raw],
        historical_coordinate_overlay_active=True,
        coordinate_solve_policy=_WHITENED_POLICY,
    )

    assert [row["candidate_label"] for row in selected] == ["authoritative"]
    assert selected[0]["feature"].phase2_joint_geometry_reuse == receipt
    assert selected[0]["feature"].phase3_response_supported_rank == 4


def test_phase3_batch_empty_selection_fails_without_authoritative_source() -> None:
    with pytest.raises(RuntimeError, match="authoritative supported-coordinate"):
        adapt_pipeline._phase3_batch_empty_selection_fallback(
            [],
            batch_source_records=[],
            authoritative_records=[],
            raw_phase2_records=[
                _record(
                    "raw-phase2",
                    9,
                    _feature(candidate_label="raw-phase2", candidate_pool_index=9),
                )
            ],
            historical_coordinate_overlay_active=True,
            coordinate_solve_policy=_WHITENED_POLICY,
        )


def test_phase3_batch_singleton_fallback_preserves_measured_trust_response() -> None:
    receipt = {
        "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": True,
        "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        "joint_batch_context_mode": "full_ansatz_v1",
        "joint_fubini_study_displacement_sq": 0.04,
        "applied_predicted_reduction": 0.2,
    }
    selected = _record(
        "authoritative",
        3,
        replace(
            _feature(candidate_label="authoritative", candidate_pool_index=3),
            phase2_joint_geometry_reuse=receipt,
        ),
    )

    summary = adapt_pipeline._phase3_batch_authoritative_singleton_fallback_summary(
        {"feasible": False, "reason": "singleton_shell"},
        [selected],
    )

    fallback = summary["authoritative_singleton_fallback"]
    assert fallback["schema"] == (
        "phase3_batch_authoritative_singleton_fallback_v2"
    )
    assert fallback["selected_label"] == "authoritative"
    assert fallback["coordinate_summary"] == receipt


def test_phase3_batch_receipt_restore_repairs_rescue_fill_after_reordering() -> None:
    authoritative: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for index, label in enumerate(("candidate_a", "candidate_b", "candidate_c")):
        receipt = {
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": True,
            "joint_gain": float(index + 1) / 10.0,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        }
        authority_feature = replace(
            _feature(candidate_label=label, candidate_pool_index=index),
            phase2_joint_geometry_reuse=receipt,
            phase3_response_supported_rank=index + 2,
        )
        authoritative.append(_record(label, index, authority_feature))
        selected.append(
            {
                **_record(
                    label,
                    index,
                    replace(
                        authority_feature,
                        phase2_joint_geometry_reuse={
                            "schema": "phase3_batch_joint_workspace_v1",
                        },
                        phase3_response_supported_rank=None,
                    ),
                ),
                "batch_order_rescue": {
                    "schema": "phase3_ordered_batch_admission_rescue_v1",
                },
            }
        )

    restored = adapt_pipeline._restore_phase3_batch_singleton_coordinate_receipts(
        list(reversed(selected)),
        authoritative_records=authoritative,
        coordinate_solve_policy=_WHITENED_POLICY,
    )

    assert [row["candidate_label"] for row in restored] == [
        "candidate_c",
        "candidate_b",
        "candidate_a",
    ]
    assert [
        row["feature"].phase2_joint_geometry_reuse["joint_gain"]
        for row in restored
    ] == [0.3, 0.2, 0.1]
    assert [
        row["feature"].phase3_response_supported_rank for row in restored
    ] == [4, 3, 2]
    assert all("batch_order_rescue" in row for row in restored)


def test_stale_infeasible_whitened_plan_fails_before_splice_guard() -> None:
    infeasible_feature = replace(
        _feature(),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": False,
            "reason": "rank_gate",
            "joint_gain": 0.0,
            "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        },
    )
    feature_row = dict(infeasible_feature.__dict__)

    with pytest.raises(RuntimeError, match="outside explicit all-infeasible"):
        adapt_pipeline._assert_historical_singleton_coordinate_plan_admissible(
            feature_row,
            whitening_active=True,
        )

    feature_row["route_a_geometry_expansion_mode"] = (
        "collective_span_novelty_over_cost_v1"
    )
    adapt_pipeline._assert_historical_singleton_coordinate_plan_admissible(
        feature_row,
        whitening_active=True,
    )


def test_whitened_model_schema_or_policy_mismatch_still_fails_closed() -> None:
    wrong_policy = replace(
        _feature(),
        phase2_joint_geometry_reuse={
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": False,
            "reason": "rank_gate",
            "joint_gain": 0.0,
            "joint_linear_solve_policy_effective": "scalar_unwhitened",
        },
    )
    with pytest.raises(ValueError, match="wrong solver|was not produced by"):
        rescore_historical_phase3_records_with_coordinate_models(
            [_record("candidate", 0, wrong_policy)],
            cfg=FullScoreConfig(),
        )


def test_scalar_selector_summary_prefers_preserved_gain_after_whitening() -> None:
    components = {
        **_feature().phase_score_components,
        "phase3_historical_scalar_DeltaE_TR": 0.1875,
        "historical_scalar_DeltaE_TR": 0.1875,
        "DeltaE_TR": 0.4,
        "phase3_delta_e_tr": 0.4,
    }
    whitened = replace(
        _feature(),
        phase3_reduced_trust_gain=0.4,
        phase_score_components=components,
    )

    summary = historical_singleton_scalar_selector_summary(
        whitened,
        radius=0.25,
        metric_floor=1e-12,
        reduced_metric_collapse_rel_tol=1e-8,
    )

    assert summary["context_mode"] == HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1
    assert summary["historical_scalar_gain_crosscheck"] == pytest.approx(0.1875)
    assert summary["historical_scalar_gain_source"] == (
        "phase3_historical_scalar_DeltaE_TR"
    )
    assert summary["joint_fubini_study_displacement_sq"] == pytest.approx(0.25**2)
    assert summary["applied_predicted_reduction"] == pytest.approx(0.1875)
    assert summary["trust_radius_binding"] is True


def test_pipeline_routes_trust_only_to_scalar_and_combined_to_whitened_model() -> None:
    scalar_row = dict(_feature().__dict__)
    scalar_summary, scalar_context, scalar_opt_in = (
        adapt_pipeline._historical_singleton_trust_update_inputs(
            scalar_row,
            whitening_active=False,
            radius=0.25,
            metric_floor=1e-12,
            reduced_metric_collapse_rel_tol=1e-8,
        )
    )

    whitened_model = {
        "schema": scoring.HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": True,
        "joint_linear_solve_policy_effective": (
            "supported_metric_whitened_eigh_v1"
        ),
        "joint_batch_context_mode": "full_ansatz_v1",
        "joint_fubini_study_displacement_sq": 0.04,
        "applied_predicted_reduction": 0.31,
    }
    whitened_row = dict(
        replace(
            _feature(),
            phase2_joint_geometry_reuse=whitened_model,
        ).__dict__
    )
    whitened_summary, whitened_context, whitened_opt_in = (
        adapt_pipeline._historical_singleton_trust_update_inputs(
            whitened_row,
            whitening_active=True,
            radius=0.25,
            metric_floor=1e-12,
            reduced_metric_collapse_rel_tol=1e-8,
        )
    )

    assert scalar_context == HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1
    assert scalar_opt_in is True
    assert scalar_summary["applied_predicted_reduction"] == pytest.approx(
        0.1875
    )
    assert whitened_context == "full_ansatz_v1"
    assert whitened_opt_in is False
    assert whitened_summary["applied_predicted_reduction"] == pytest.approx(
        0.31
    )
    assert whitened_summary != scalar_summary

    phase2_only_row = dict(
        replace(
            _feature(),
            historical_singleton_phase2_coordinate_model=whitened_model,
        ).__dict__
    )
    phase2_only_summary, phase2_only_context, phase2_only_opt_in = (
        adapt_pipeline._historical_singleton_trust_update_inputs(
            phase2_only_row,
            whitening_active=True,
            radius=0.25,
            metric_floor=1e-12,
            reduced_metric_collapse_rel_tol=1e-8,
        )
    )
    assert phase2_only_context == "full_ansatz_v1"
    assert phase2_only_opt_in is False
    assert phase2_only_summary["applied_predicted_reduction"] == pytest.approx(
        0.31
    )


def test_sr_ordinary_trust_update_uses_preserved_v1_summary() -> None:
    ordinary_summary = {
        "schema": scoring.HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": True,
        "joint_linear_solve_policy_effective": (
            "supported_metric_whitened_eigh_v1"
        ),
        "joint_batch_context_mode": "full_ansatz_v1",
        "joint_fubini_study_displacement_sq": 0.03,
        "applied_predicted_reduction": 0.21,
    }
    v2_summary = {
        "schema": scoring.HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": True,
        "joint_linear_solve_policy_effective": _GLOBAL_TRUST_POLICY,
        "joint_batch_context_mode": "full_ansatz_v1",
        "joint_fubini_study_displacement_sq": 0.06,
        "applied_predicted_reduction": 0.44,
        "sr_escape_ordinary_summary": ordinary_summary,
    }
    feature_row = {
        **dict(
            replace(
                _feature(),
                phase2_joint_geometry_reuse=v2_summary,
            ).__dict__
        ),
        "sr_escape_decision_kind": "ordinary",
    }

    summary, context, historical_opt_in = (
        adapt_pipeline._historical_singleton_trust_update_inputs(
            feature_row,
            whitening_active=True,
            sr_escape_active=True,
            coordinate_solve_policy=_GLOBAL_TRUST_POLICY,
            radius=0.25,
            metric_floor=1e-12,
            reduced_metric_collapse_rel_tol=1e-8,
        )
    )

    assert summary == ordinary_summary
    assert context == "full_ansatz_v1"
    assert historical_opt_in is False


def test_geometry_expansion_trust_recalibrates_on_descent_and_holds_without_it() -> None:
    config = TrustRegionUpdateConfig(
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
    )
    before = np.asarray([1.0, 0.0], dtype=complex)
    after = np.asarray([math.cos(0.125), math.sin(0.125)], dtype=complex)

    descent_state = RouteATrustRegionState(radius=0.25)
    descent = update_geometry_expansion_trust_region_state(
        descent_state,
        config=config,
        state_before=before,
        state_after_refit=after,
        energy_before=-1.0,
        energy_after_refit=-1.1,
        energy_improvement_tolerance=1e-8,
        full_coordinate_refit=True,
    )
    assert descent["context_mode"] == (
        HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1
    )
    assert descent["scalar_or_unwhitened_fallback_used"] is False
    assert descent_state.radius == pytest.approx(0.125)
    assert descent["update_reason"] == (
        "geometry_expansion_descent_recalibrated_to_realized_displacement"
    )

    flat_state = RouteATrustRegionState(radius=0.25)
    flat = update_geometry_expansion_trust_region_state(
        flat_state,
        config=config,
        state_before=before,
        state_after_refit=after,
        energy_before=-1.0,
        energy_after_refit=-1.0,
        energy_improvement_tolerance=1e-8,
        full_coordinate_refit=True,
    )
    assert flat_state.radius == pytest.approx(0.25)
    assert flat["update_reason"] == "geometry_expansion_no_descent_hold"


def test_nonbeam_geometry_expansion_bypasses_rank_gated_energy_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank_gated_model = {
        "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": False,
        "reason": "rank_gate",
        "joint_linear_solve_policy_effective": _WHITENED_POLICY,
        "joint_batch_context_mode": "full_ansatz_v1",
    }
    feature_row = {
        **dict(
            replace(
                _feature(),
                phase2_joint_geometry_reuse=rank_gated_model,
            ).__dict__
        ),
    }
    selected_record = {
        "route_a_geometry_expansion_mode": (
            "collective_span_novelty_over_cost_v1"
        ),
    }

    def _ordinary_inputs_must_not_run(*args: object, **kwargs: object):
        raise AssertionError("ordinary rank-gated energy model was consumed")

    monkeypatch.setattr(
        adapt_pipeline,
        "_historical_singleton_trust_update_inputs",
        _ordinary_inputs_must_not_run,
    )
    summary, context, historical_opt_in, geometry_active = (
        adapt_pipeline._historical_singleton_trust_update_inputs_or_geometry_expansion(
            feature_row,
            selected_record=selected_record,
            whitening_active=True,
            radius=0.25,
            metric_floor=1e-12,
            reduced_metric_collapse_rel_tol=1e-8,
        )
    )

    assert summary == {}
    assert context == HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1
    assert historical_opt_in is False
    assert geometry_active is True


def test_scalar_selector_summary_fails_closed_on_gain_crosscheck() -> None:
    mismatched = replace(
        _feature(),
        phase_score_components={
            **_feature().phase_score_components,
            "phase3_historical_scalar_DeltaE_TR": 0.18,
            "historical_scalar_DeltaE_TR": 0.18,
        },
    )

    with pytest.raises(ValueError, match="does not reproduce"):
        historical_singleton_scalar_selector_summary(
            mismatched,
            radius=0.25,
            metric_floor=1e-12,
            reduced_metric_collapse_rel_tol=1e-8,
        )


def test_historical_context_requires_explicit_opt_in_and_records_partial_refit() -> None:
    summary = historical_singleton_scalar_selector_summary(
        _feature(),
        radius=0.25,
        metric_floor=1e-12,
        reduced_metric_collapse_rel_tol=1e-8,
    )
    config = TrustRegionUpdateConfig(
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
    )
    before = np.asarray([1.0, 0.0], dtype=complex)
    after = np.asarray([math.cos(0.125), math.sin(0.125)], dtype=complex)

    blocked_state = RouteATrustRegionState(radius=0.25)
    blocked = update_trust_region_state(
        blocked_state,
        config=config,
        context_mode=HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1,
        selector_summary=summary,
        state_before=before,
        state_after_refit=after,
        energy_before=-1.0,
        energy_after_refit=-1.1,
        energy_improvement_tolerance=1e-8,
        full_coordinate_refit=False,
    )
    assert blocked["update_reason"] == "context_mode_not_supported"
    assert blocked_state.radius == pytest.approx(0.25)

    accepted_state = RouteATrustRegionState(radius=0.25)
    accepted = update_trust_region_state(
        accepted_state,
        config=config,
        context_mode=HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1,
        selector_summary=summary,
        state_before=before,
        state_after_refit=after,
        energy_before=-1.0,
        energy_after_refit=-1.1,
        energy_improvement_tolerance=1e-8,
        full_coordinate_refit=False,
        allow_historical_singleton_context=True,
    )
    assert accepted["update_reason"] == "realized_displacement_smaller"
    assert accepted_state.radius == pytest.approx(0.25 * math.sqrt(0.5))
    assert accepted["context_mode"] == HISTORICAL_SINGLETON_SCALAR_TRUST_CONTEXT_V1
    assert accepted["full_coordinate_refit"] is False
    assert accepted["historical_singleton_context_opt_in"] is True
    assert accepted["historical_singleton_context_accepted"] is True


def test_sr_escape_rescore_preserves_strict_ordinary_precedence() -> None:
    ordinary = _escape_feature(
        "ordinary",
        1,
        summary_overrides={
            "joint_gain": 0.4,
            "stationary_certified": False,
            "stationarity_margin": 1.0e-3,
            "sr_escape_ordinary_summary": {
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": True,
                "reason": "preserved_supported_metric_whitened_ordinary_v1",
                "joint_gain": 0.4,
                "joint_gain_lower_bound": 0.4,
                "joint_linear_solve_policy_effective": (
                    "supported_metric_whitened_eigh_v1"
                ),
                "joint_batch_context_mode": "full_ansatz_v1",
            },
        },
    )
    saddle = _escape_feature("saddle", 2)

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("ordinary", 1, ordinary), _record("saddle", 2, saddle)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_plus_modeled_minimum",
        sr_escape_ordinary_record_ids=("ordinary::pool=1::position=0",),
        sr_escape_reachable_record_ids=(
            "ordinary::pool=1::position=0",
            "saddle::pool=2::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    ordinary_id = "ordinary::pool=1::position=0"
    assert controller["decision_kind"] == "ordinary"
    assert controller["record_id"] == ordinary_id
    assert controller["admission_eligible_record_ids"] == [ordinary_id]
    assert controller["actionable"] is True
    assert controller["consumes_singleton"] is True
    assert rescored[0]["sr_escape_admission_eligible"] is True
    assert rescored[0]["selector_score"] == pytest.approx(0.4 / 2.0)
    assert rescored[0]["feature"].phase_score_components[
        "phase3_measured_novelty"
    ] == pytest.approx(0.5)
    assert rescored[0]["feature"].phase_score_components["N3"] is None
    assert rescored[1]["sr_escape_admission_eligible"] is False
    assert rescored[1]["selector_score"] == float("-inf")


def test_sr_ordinary_model_live_contradiction_unblocks_escape_on_same_state(
) -> None:
    ordinary_id = "ordinary::pool=1::position=0"
    saddle_id = "saddle::pool=2::position=0"
    state_fingerprint = "same-physical-state"
    ordinary = _escape_feature(
        "ordinary",
        1,
        summary_overrides={
            "joint_gain": 0.4,
            "stationary_certified": False,
            "stationarity_margin": 1.0e-3,
            "supported_stationarity_status": "unresolved",
            "sr_escape_ordinary_summary": {
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": True,
                "reason": "preserved_supported_metric_whitened_ordinary_v1",
                "joint_gain": 0.4,
                "joint_gain_lower_bound": 0.4,
                "joint_linear_solve_policy_effective": (
                    "supported_metric_whitened_eigh_v1"
                ),
                "joint_batch_context_mode": "full_ansatz_v1",
            },
        },
    )
    saddle = _escape_feature("saddle", 2)
    records = [_record("ordinary", 1, ordinary), _record("saddle", 2, saddle)]
    common = {
        "cfg": FullScoreConfig(),
        "sr_escape_mode": "saddle_only",
        "sr_escape_ordinary_record_ids": (ordinary_id,),
        "sr_escape_reachable_record_ids": (ordinary_id, saddle_id),
    }

    _, first = rescore_historical_phase3_records_with_coordinate_models(
        records,
        **common,
    )
    assert first["sr_escape_controller"]["decision_kind"] == "ordinary"

    receipt = adapt_pipeline._sr_ordinary_model_contradiction_receipt(
        state_fingerprint=state_fingerprint,
        record_id=ordinary_id,
        guard_payload={
            "schema": "route_a_joint_step_warm_start_v1",
            "status": "rejected",
            "transaction_failure_kind": None,
            "mapped_seed_predicted_reduction": 0.4,
            "mapped_seed_exact_gain": -2.0e-6,
            "comparison_event_schema": (
                "route_a_exact_joint_step_seed_guard_v1"
            ),
            "numerical_energy_comparison_width": 1.0e-6,
            "optimizer_reproducibility_allowance": 0.0,
            "aggregate_simultaneous_comparison_width": 1.0e-6,
            "guard": {"guard_tolerance": 1.0e-6},
            "energy_comparison_width": {
                "schema": "sr_simultaneous_energy_comparison_width_v1",
                "comparison_event_schema": (
                    "route_a_exact_joint_step_seed_guard_v1"
                ),
                "numerical_energy_comparison_width": 1.0e-6,
                "optimizer_reproducibility_allowance": 0.0,
                "aggregate_simultaneous_comparison_width": 1.0e-6,
            },
        },
    )
    assert receipt is not None
    contradicted = adapt_pipeline._sr_contradicted_ordinary_record_ids(
        [receipt],
        state_fingerprint=state_fingerprint,
    )
    assert contradicted == frozenset({ordinary_id})

    rescored, second = rescore_historical_phase3_records_with_coordinate_models(
        records,
        sr_escape_contradicted_ordinary_record_ids=tuple(contradicted),
        **common,
    )
    controller = second["sr_escape_controller"]
    assert controller["decision_kind"] == "saddle_singleton"
    assert controller["record_id"] == saddle_id
    assert controller["contradicted_ordinary_record_ids"] == [ordinary_id]
    assert rescored[0]["sr_escape_ordinary_model_live"] is False
    assert rescored[0]["sr_escape_ordinary_model_contradicted"] is True
    assert rescored[1]["sr_escape_admission_eligible"] is True

    assert adapt_pipeline._sr_contradicted_ordinary_record_ids(
        [receipt],
        state_fingerprint="different-physical-state",
    ) == frozenset()


def test_sr_ordinary_model_live_is_not_retired_by_mapping_failure_or_overlap(
) -> None:
    common = {
        "state_fingerprint": "state",
        "record_id": "candidate::pool=1::position=0",
    }
    assert adapt_pipeline._sr_ordinary_model_contradiction_receipt(
        **common,
        guard_payload={
            "status": "unavailable",
            "transaction_failure_kind": "mapping",
        },
    ) is None
    assert adapt_pipeline._sr_ordinary_model_contradiction_receipt(
        **common,
        guard_payload={
            "schema": "route_a_joint_step_warm_start_v1",
            "status": "rejected",
            "transaction_failure_kind": None,
            "mapped_seed_predicted_reduction": 0.4,
            "mapped_seed_exact_gain": -0.5e-6,
            "comparison_event_schema": (
                "route_a_exact_joint_step_seed_guard_v1"
            ),
            "numerical_energy_comparison_width": 1.0e-6,
            "optimizer_reproducibility_allowance": 0.0,
            "aggregate_simultaneous_comparison_width": 1.0e-6,
            "guard": {"guard_tolerance": 1.0e-6},
            "energy_comparison_width": {
                "schema": "sr_simultaneous_energy_comparison_width_v1",
                "comparison_event_schema": (
                    "route_a_exact_joint_step_seed_guard_v1"
                ),
                "numerical_energy_comparison_width": 1.0e-6,
                "optimizer_reproducibility_allowance": 0.0,
                "aggregate_simultaneous_comparison_width": 1.0e-6,
            },
        },
    ) is None


def test_sr_ordinary_model_live_requires_consistent_typed_width_provenance(
) -> None:
    base = {
        "schema": "route_a_joint_step_warm_start_v1",
        "status": "rejected",
        "transaction_failure_kind": None,
        "mapped_seed_predicted_reduction": 0.4,
        "mapped_seed_exact_gain": -2.0e-6,
        "comparison_event_schema": "route_a_exact_joint_step_seed_guard_v1",
        "numerical_energy_comparison_width": 1.0e-6,
        "optimizer_reproducibility_allowance": 0.0,
        "aggregate_simultaneous_comparison_width": 1.0e-6,
        "guard": {"guard_tolerance": 1.0e-6},
    }
    common = {
        "state_fingerprint": "state",
        "record_id": "candidate::pool=1::position=0",
    }
    assert adapt_pipeline._sr_ordinary_model_contradiction_receipt(
        **common,
        guard_payload=dict(base),
    ) is None

    mismatched = {
        **base,
        "energy_comparison_width": {
            "schema": "sr_simultaneous_energy_comparison_width_v1",
            "comparison_event_schema": (
                "route_a_exact_joint_step_seed_guard_v1"
            ),
            "numerical_energy_comparison_width": 1.0e-6,
            "optimizer_reproducibility_allowance": 0.0,
            "aggregate_simultaneous_comparison_width": 10.0,
        },
    }
    assert adapt_pipeline._sr_ordinary_model_contradiction_receipt(
        **common,
        guard_payload=mismatched,
    ) is None


def test_sr_escape_ordinary_ranking_uses_preserved_v1_not_v2_gain() -> None:
    outer_v2_favorite = _escape_feature(
        "outer-v2-favorite",
        1,
        summary_overrides={
            "joint_gain": 9.0,
            "sr_escape_ordinary_summary": {
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": True,
                "reason": "preserved_supported_metric_whitened_ordinary_v1",
                "joint_gain": 0.2,
                "joint_gain_lower_bound": 0.2,
                "joint_linear_solve_policy_effective": (
                    "supported_metric_whitened_eigh_v1"
                ),
                "joint_batch_context_mode": "full_ansatz_v1",
            },
        },
    )
    preserved_v1_favorite = _escape_feature(
        "preserved-v1-favorite",
        2,
        summary_overrides={
            "joint_gain": 0.1,
            "sr_escape_ordinary_summary": {
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": True,
                "reason": "preserved_supported_metric_whitened_ordinary_v1",
                "joint_gain": 0.5,
                "joint_gain_lower_bound": 0.5,
                "joint_linear_solve_policy_effective": (
                    "supported_metric_whitened_eigh_v1"
                ),
                "joint_batch_context_mode": "full_ansatz_v1",
            },
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [
            _record("outer-v2-favorite", 1, outer_v2_favorite),
            _record("preserved-v1-favorite", 2, preserved_v1_favorite),
        ],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_ordinary_record_ids=(
            "outer-v2-favorite::pool=1::position=0",
            "preserved-v1-favorite::pool=2::position=0",
        ),
        sr_escape_reachable_record_ids=(
            "outer-v2-favorite::pool=1::position=0",
            "preserved-v1-favorite::pool=2::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "ordinary"
    assert controller["record_id"] == (
        "preserved-v1-favorite::pool=2::position=0"
    )
    assert rescored[0]["selector_score"] == pytest.approx(0.2 / 2.0)
    assert rescored[1]["selector_score"] == pytest.approx(0.5 / 2.0)
    assert all(
        row["feature"].phase_score_components["phase3_measured_novelty"]
        == pytest.approx(0.5)
        for row in rescored
    )


@pytest.mark.parametrize("n3", [1.0e-12, 1.0])
def test_sr_escape_saddle_score_is_marginal_gain_over_cost_without_n3(
    n3: float,
) -> None:
    saddle = _escape_feature("saddle", 4, n3=n3, k3=3.0)

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("saddle", 4, saddle)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_reachable_record_ids=("saddle::pool=4::position=0",),
    )

    controller = telemetry["sr_escape_controller"]
    expected_marginal_gain = 0.5
    expected_score = expected_marginal_gain / (1.0 + 3.0)
    assert controller["decision_kind"] == "saddle_singleton"
    assert controller["marginal_gain_lower_bound"] == pytest.approx(
        expected_marginal_gain
    )
    assert controller["acquisition"] == pytest.approx(expected_score)
    assert rescored[0]["sr_escape_admission_eligible"] is True
    assert rescored[0]["selector_score"] == pytest.approx(expected_score)
    assert rescored[0]["phase3_reduced_trust_gain"] == pytest.approx(
        expected_marginal_gain
    )
    feature = rescored[0]["feature"]
    assert isinstance(feature, CandidateFeatures)
    assert feature.phase_score_components["N3"] is None
    assert feature.phase_score_components["phase3_measured_novelty"] == (
        pytest.approx(n3)
    )
    assert feature.phase_score_components["phase3_novelty_multiplier"] is None


def test_sr_escape_active_only_correction_admits_no_singleton() -> None:
    active_only = _escape_feature(
        "active-only",
        5,
        summary_overrides={
            "full_trust_gain": 0.5,
            "active_restricted_trust_gain": 0.5,
            "quotient_participation_lower_bound": 0.0,
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("active-only", 5, active_only)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_reachable_record_ids=(
            "active-only::pool=5::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "active_only_correction"
    assert controller["certificate_record_id"] == (
        "active-only::pool=5::position=0"
    )
    assert controller["actionable"] is True
    assert controller["consumes_singleton"] is False
    assert controller["exact_map_transaction_required"] is True
    assert controller["admission_eligible_record_ids"] == []
    assert rescored[0]["sr_escape_admission_eligible"] is False
    assert rescored[0]["sr_escape_exact_map_transaction_required"] is True
    assert telemetry["records"][0][
        "sr_escape_exact_map_transaction_required"
    ] is True
    assert rescored[0]["selector_score"] == float("-inf")
    assert rescored[0]["phase3_reduced_trust_gain"] == pytest.approx(0.0)


def test_sr_escape_nonstationary_model_requests_active_stationarity_correction(
) -> None:
    nonstationary = _escape_feature(
        "active-stationarity-refit",
        50,
        summary_overrides={
            "stationary_certified": False,
            "nonstationary_certified": True,
            "supported_stationarity_status": "certified_nonstationary",
            "stationarity_margin": 1.0e-3,
            "active_restricted_trust_gain_lower_bound": 2.0e-4,
            "active_restricted_trust_gain_upper_bound": 2.1e-4,
            "active_restriction_solve": {
                "valid": True,
                "feasible": True,
                "trust_global_optimality_certified": True,
                "active_restriction_uses_full_support_decision": True,
                "active_restriction_independent_metric_factorization": False,
                "active_restriction_constraint": (
                    "physical_active_coordinate_image_modulo_"
                    "certified_raw_metric_null"
                ),
                "active_restriction_batch_zero_residual": 0.0,
                "active_restriction_batch_zero_tolerance": 1.0e-12,
            },
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("active-stationarity-refit", 50, nonstationary)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_reachable_record_ids=(
            "active-stationarity-refit::pool=50::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "active_stationarity_correction"
    assert controller["certificate_record_id"] == (
        "active-stationarity-refit::pool=50::position=0"
    )
    assert controller["actionable"] is True
    assert controller["consumes_singleton"] is False
    assert controller["exact_map_transaction_required"] is True
    assert controller["certificate_kind_counts"] == {
        "NonstationaryCertificate": 1
    }
    assert controller["admission_eligible_record_ids"] == []
    assert rescored[0]["sr_escape_admission_eligible"] is False
    assert rescored[0]["sr_escape_exact_map_transaction_required"] is True
    assert telemetry["records"][0][
        "sr_escape_exact_map_transaction_required"
    ] is True


@pytest.mark.parametrize(
    ("summary_update", "restriction_update"),
    [
        ({"stationarity_margin": None}, {}),
        ({"stationarity_margin": float("nan")}, {}),
        ({"active_restricted_trust_gain_lower_bound": -1.0}, {}),
        ({"active_restricted_trust_gain_lower_bound": float("nan")}, {}),
        ({"active_restricted_trust_gain_upper_bound": None}, {}),
        ({"active_restricted_trust_gain_upper_bound": 1.0e-5}, {}),
        ({}, {"active_restriction_batch_zero_residual": float("nan")}),
        ({}, {"active_restriction_batch_zero_residual": -1.0}),
        ({}, {"active_restriction_batch_zero_tolerance": None}),
    ],
)
def test_sr_escape_nonstationary_invalid_numeric_certificate_fails_unresolved(
    summary_update: dict[str, object],
    restriction_update: dict[str, object],
) -> None:
    active_restriction: dict[str, object] = {
        "valid": True,
        "feasible": True,
        "trust_global_optimality_certified": True,
        "active_restriction_uses_full_support_decision": True,
        "active_restriction_independent_metric_factorization": False,
        "active_restriction_constraint": (
            "physical_active_coordinate_image_modulo_"
            "certified_raw_metric_null"
        ),
        "active_restriction_batch_zero_residual": 0.0,
        "active_restriction_batch_zero_tolerance": 1.0e-12,
    }
    active_restriction.update(restriction_update)
    summary_overrides: dict[str, object] = {
        "stationary_certified": False,
        "nonstationary_certified": True,
        "supported_stationarity_status": "certified_nonstationary",
        "stationarity_margin": 1.0e-3,
        "active_restricted_trust_gain_lower_bound": 2.0e-4,
        "active_restricted_trust_gain_upper_bound": 2.1e-4,
        "active_restriction_solve": active_restriction,
    }
    summary_overrides.update(summary_update)
    nonstationary = _escape_feature(
        "invalid-active-stationarity-refit",
        50,
        summary_overrides=summary_overrides,
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("invalid-active-stationarity-refit", 50, nonstationary)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_reachable_record_ids=(
            "invalid-active-stationarity-refit::pool=50::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "unresolved"
    assert controller["certificate_kind_counts"] == {
        "UnresolvedCertificate": 1
    }
    assert controller["unresolved_certificate_reason_counts"] == {
        "certified_nonstationary_numeric_certificate_invalid": 1
    }
    assert controller["exact_map_transaction_required"] is False
    assert rescored[0]["sr_escape_exact_map_transaction_required"] is False


def test_sr_escape_null_incompatibility_serializes_the_actual_reason() -> None:
    unresolved = _escape_feature(
        "null-incompatible",
        51,
        summary_overrides={
            "feasible": False,
            "reason": "raw_metric_null_hessian_incompatible",
            "raw_metric_support_status": "resolved",
            "raw_metric_support_reason": "stable_positive_raw_metric_support",
            "raw_metric_null_compatibility_certified": False,
            "raw_metric_null_compatibility_reason": (
                "raw_metric_null_hessian_incompatible"
            ),
        },
    )

    _, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("null-incompatible", 51, unresolved)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_reachable_record_ids=(
            "null-incompatible::pool=51::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "unresolved"
    assert controller["unresolved_certificate_reason_counts"] == {
        "raw_metric_null_hessian_incompatible": 1
    }


def test_sr_escape_credited_saddle_precedes_unresolved_bystander() -> None:
    saddle = _escape_feature("credited-saddle", 51)
    unresolved = _escape_feature(
        "unresolved-bystander",
        52,
        summary_overrides={
            "feasible": False,
            "reason": "conditioning_gate",
            "joint_gain": 0.0,
            "trust_global_optimality_certified": False,
            "marginal_trust_gain_comparison_valid": False,
            "quotient_participation_resolved": False,
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [
            _record("credited-saddle", 51, saddle),
            _record("unresolved-bystander", 52, unresolved),
        ],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_reachable_record_ids=(
            "credited-saddle::pool=51::position=0",
            "unresolved-bystander::pool=52::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "saddle_singleton"
    assert controller["record_id"] == "credited-saddle::pool=51::position=0"
    assert controller["actionable"] is True
    assert controller["consumes_singleton"] is True
    assert controller["admission_eligible_record_ids"] == [
        "credited-saddle::pool=51::position=0"
    ]
    assert rescored[0]["sr_escape_admission_eligible"] is True
    assert rescored[1]["sr_escape_admission_eligible"] is False


def test_sr_escape_active_only_precedes_unresolved_bystander() -> None:
    active_only = _escape_feature(
        "active-only",
        53,
        summary_overrides={
            "full_trust_gain": 0.5,
            "active_restricted_trust_gain": 0.5,
            "quotient_participation_lower_bound": 0.0,
        },
    )
    unresolved = _escape_feature(
        "unresolved-bystander",
        54,
        summary_overrides={
            "feasible": False,
            "reason": "conditioning_gate",
            "joint_gain": 0.0,
            "trust_global_optimality_certified": False,
            "marginal_trust_gain_comparison_valid": False,
            "quotient_participation_resolved": False,
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [
            _record("active-only", 53, active_only),
            _record("unresolved-bystander", 54, unresolved),
        ],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_reachable_record_ids=(
            "active-only::pool=53::position=0",
            "unresolved-bystander::pool=54::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "active_only_correction"
    assert controller["certificate_record_id"] == (
        "active-only::pool=53::position=0"
    )
    assert controller["actionable"] is True
    assert controller["consumes_singleton"] is False
    assert controller["exact_map_transaction_required"] is True
    assert controller["admission_eligible_record_ids"] == []
    assert all(
        row["sr_escape_admission_eligible"] is False for row in rescored
    )
    assert rescored[0]["sr_escape_exact_map_transaction_required"] is True
    assert rescored[1]["sr_escape_exact_map_transaction_required"] is False
    assert telemetry["records"][0][
        "sr_escape_exact_map_transaction_required"
    ] is True
    assert telemetry["records"][1][
        "sr_escape_exact_map_transaction_required"
    ] is False


def test_sr_escape_psd_plus_unresolved_remains_unresolved() -> None:
    psd = _escape_feature(
        "psd",
        55,
        summary_overrides={
            "joint_gain": 0.0,
            "negative_curvature_certified": False,
            "positive_semidefinite_certified": True,
            "supported_inertia_status": "psd",
            "minimum_hessian_eigenvalue_lower_bound": 0.25,
            "minimum_hessian_eigenvalue_upper_bound": 0.25,
            "full_trust_gain": 0.0,
            "active_restricted_trust_gain": 0.0,
        },
    )
    unresolved = _escape_feature(
        "unresolved",
        56,
        summary_overrides={
            "feasible": False,
            "reason": "conditioning_gate",
            "joint_gain": 0.0,
            "trust_global_optimality_certified": False,
            "marginal_trust_gain_comparison_valid": False,
            "quotient_participation_resolved": False,
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("psd", 55, psd), _record("unresolved", 56, unresolved)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_plus_modeled_minimum",
        sr_escape_reachable_record_ids=(
            "psd::pool=55::position=0",
            "unresolved::pool=56::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "unresolved"
    assert controller["reason"] == (
        "reachable_population_contains_unresolved_certificate"
    )
    assert controller["stage_b_eligible"] is False
    assert controller["actionable"] is False
    assert all(row["selector_score"] == float("-inf") for row in rescored)


def test_sr_combined_mode_psd_audit_is_stage_b_eligible_without_admission() -> None:
    psd = _escape_feature(
        "psd",
        6,
        summary_overrides={
            "joint_gain": 0.0,
            "negative_curvature_certified": False,
            "positive_semidefinite_certified": True,
            "supported_inertia_status": "psd",
            "minimum_hessian_eigenvalue_lower_bound": 0.25,
            "minimum_hessian_eigenvalue_upper_bound": 0.25,
            "full_trust_gain": 0.0,
            "active_restricted_trust_gain": 0.0,
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("psd", 6, psd)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_plus_modeled_minimum",
        sr_escape_reachable_record_ids=("psd::pool=6::position=0",),
        sr_escape_state_fingerprint="working-state",
        sr_escape_trust_radius=0.25,
        sr_escape_comparison_epoch="comparison-epoch",
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "modeled_minimum_eligible"
    assert controller["reachable_population_complete"] is True
    assert controller["stage_b_eligible"] is True
    assert controller["state_stationarity_certified"] is True
    assert controller["state_stationarity_blocker"] is None
    assert controller["reachable_certificate_table_schema"] == (
        "sr_escape_reachable_certificate_table_v1"
    )
    assert controller["reachable_certificate_table"] == [
        {
            **controller["reachable_certificate_table"][0],
            "record_id": "psd::pool=6::position=0",
            "candidate_label": "psd",
            "candidate_pool_index": 6,
            "position_id": 0,
            "certificate_kind": "PsdCertificate",
            "working_state_fingerprint": "working-state",
            "trust_radius": 0.25,
            "stationarity_margin": -1.0e-8,
            "minimum_eigenvalue_lower_bound": 0.25,
        }
    ]
    assert len(
        controller["reachable_certificate_table"][0][
            "coordinate_summary_digest"
        ]
    ) == 64
    assert len(
        controller["state_stationarity_certificate"]["token_digest"]
    ) == 64
    assert (
        controller["reachable_certificate_table"][0][
            "state_stationarity_token_digest"
        ]
        == controller["state_stationarity_certificate"]["token_digest"]
    )
    assert controller["actionable"] is False
    assert controller["consumes_singleton"] is False
    assert controller["admission_eligible_record_ids"] == []
    assert rescored[0]["sr_escape_admission_eligible"] is False
    assert rescored[0]["selector_score"] == float("-inf")
    modeled_minimum_core = controller["modeled_minimum_core"]
    assert modeled_minimum_core["schema"] == (
        "sr_snake_modeled_minimum_core_telemetry_v1"
    )
    assert modeled_minimum_core["version"] == 1
    assert modeled_minimum_core["combined_mode_requested"] is True
    assert modeled_minimum_core["mathematical_eligibility"] == {
        **modeled_minimum_core["mathematical_eligibility"],
        "eligible": True,
        "reason": (
            "complete_reachable_population_is_psd_or_redundant_and_"
            "state_stationary"
        ),
        "reachable_record_ids": ["psd::pool=6::position=0"],
        "psd_record_ids": ["psd::pool=6::position=0"],
        "redundant_record_ids": [],
        "working_state_fingerprint": "working-state",
        "reachable_population_digest": controller[
            "reachable_population_digest"
        ],
        "comparison_epoch": "comparison-epoch",
        "support_provenance_digest": "state-support-provenance",
        "trust_provenance_digest": "state-trust-provenance",
        "trust_radius": 0.25,
        "stationarity_margin": -1.0e-9,
    }
    assert len(modeled_minimum_core["state_token_digest"]) == 64
    assert modeled_minimum_core["pure_core_available"] is True
    assert modeled_minimum_core["execution_implemented"] is False
    assert modeled_minimum_core["actionable"] is False
    assert modeled_minimum_core[
        "remaining_provider_runtime_checkpoint_blockers"
    ] == [
        "canonical_continuation_path_provider_missing",
        "live_nonlinear_active_manifold_distance_provider_missing",
        "uniform_full_path_incumbent_barrier_provider_missing",
        "connected_exclusion_component_witness_provider_missing",
        "disposable_powell_reproducibility_provider_missing",
        "countable_action_cursor_tail_bound_missing",
        "incumbent_working_state_runtime_integration_missing",
        "modeled_minimum_checkpoint_roundtrip_missing",
    ]


def test_sr_saddle_only_reports_core_audit_without_modeled_minimum_activation(
) -> None:
    psd = _escape_feature(
        "psd-saddle-only",
        65,
        summary_overrides={
            "joint_gain": 0.0,
            "negative_curvature_certified": False,
            "positive_semidefinite_certified": True,
            "supported_inertia_status": "psd",
            "minimum_hessian_eigenvalue_lower_bound": 0.25,
            "minimum_hessian_eigenvalue_upper_bound": 0.25,
            "full_trust_gain": 0.0,
            "active_restricted_trust_gain": 0.0,
        },
    )

    _, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("psd-saddle-only", 65, psd)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_reachable_record_ids=(
            "psd-saddle-only::pool=65::position=0",
        ),
        sr_escape_state_fingerprint="working-state",
        sr_escape_trust_radius=0.25,
        sr_escape_comparison_epoch="comparison-epoch",
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "no_action"
    assert controller["stage_b_eligible"] is False
    assert controller["actionable"] is False
    modeled_minimum_core = controller["modeled_minimum_core"]
    assert modeled_minimum_core["combined_mode_requested"] is False
    assert modeled_minimum_core["mathematical_eligibility"]["eligible"] is True
    assert len(modeled_minimum_core["state_token_digest"]) == 64
    assert modeled_minimum_core["execution_implemented"] is False
    assert modeled_minimum_core["actionable"] is False


def test_sr_combined_psd_audit_requires_state_bound_stationarity_token() -> None:
    psd = _escape_feature(
        "psd",
        60,
        summary_overrides={
            "joint_gain": 0.0,
            "negative_curvature_certified": False,
            "positive_semidefinite_certified": True,
            "supported_inertia_status": "psd",
            "minimum_hessian_eigenvalue_lower_bound": 0.25,
            "minimum_hessian_eigenvalue_upper_bound": 0.25,
            "full_trust_gain": 0.0,
            "active_restricted_trust_gain": 0.0,
        },
    )

    _, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("psd", 60, psd)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_plus_modeled_minimum",
        sr_escape_reachable_record_ids=("psd::pool=60::position=0",),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "unresolved"
    assert controller["reason"] == (
        "state_stationarity_certificate_missing_or_population_stale"
    )
    assert controller["state_stationarity_certified"] is False
    assert controller["state_stationarity_blocker"] == (
        "state_fingerprint_missing"
    )
    modeled_minimum_core = controller["modeled_minimum_core"]
    assert modeled_minimum_core["combined_mode_requested"] is True
    assert modeled_minimum_core["mathematical_eligibility"]["eligible"] is False
    assert modeled_minimum_core["mathematical_eligibility"]["reason"] == (
        "state_stationarity_certificate_missing_or_population_stale"
    )
    assert modeled_minimum_core["state_token_digest"] is None
    assert modeled_minimum_core["execution_implemented"] is False
    assert modeled_minimum_core["actionable"] is False


def test_sr_redundancy_cannot_bypass_unresolved_state_stationarity() -> None:
    redundant = _escape_feature(
        "redundant",
        63,
        summary_overrides={
            "joint_gain": 0.0,
            "stationary_certified": False,
            "supported_stationarity_status": "unresolved",
            "stationarity_margin": 0.0,
            "quotient_redundant_certified": True,
            "quotient_residual_metric_eigenvalues": [0.0],
            "quotient_resolution_floor": 1.0e-12,
            "negative_curvature_certified": False,
            "positive_semidefinite_certified": False,
            "supported_inertia_status": "unresolved",
            "supported_inertia_label_issued": False,
        },
    )

    _, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("redundant", 63, redundant)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_plus_modeled_minimum",
        sr_escape_reachable_record_ids=(
            "redundant::pool=63::position=0",
        ),
        sr_escape_state_fingerprint="working-state",
        sr_escape_trust_radius=0.25,
        sr_escape_comparison_epoch="comparison-epoch",
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "unresolved"
    assert controller["stage_b_eligible"] is False
    assert controller["certificate_kind_counts"] == {
        "UnresolvedCertificate": 1
    }
    assert controller["unresolved_certificate_reason_counts"] == {
        "ordinary_unusable_but_stationarity_not_certified": 1
    }


def test_all_redundant_records_cannot_bypass_nonstationary_working_state() -> None:
    redundant = _escape_feature(
        "redundant",
        64,
        summary_overrides={
            "joint_gain": 0.0,
            "quotient_redundant_certified": True,
            "quotient_residual_metric_eigenvalues": [0.0],
            "quotient_resolution_floor": 1.0e-12,
            "negative_curvature_certified": False,
            "positive_semidefinite_certified": False,
            "supported_inertia_status": "unresolved",
            "supported_inertia_label_issued": False,
            "sr_escape_state_stationarity_summary": {
                "schema": "sr_escape_state_stationarity_certificate_v1",
                "valid": False,
                "reason": "working_state_supported_stationarity_unresolved",
                "supported_stationarity_status": "certified_nonstationary",
                "stationarity_margin": 1.0e-3,
                "trust_radius": 0.25,
                "support_provenance_digest": "state-support-provenance",
                "trust_provenance_digest": "state-trust-provenance",
            },
        },
    )

    _, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("redundant", 64, redundant)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_plus_modeled_minimum",
        sr_escape_reachable_record_ids=(
            "redundant::pool=64::position=0",
        ),
        sr_escape_state_fingerprint="working-state",
        sr_escape_trust_radius=0.25,
        sr_escape_comparison_epoch="comparison-epoch",
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["certificate_kind_counts"] == {
        "QuotientRedundantCertificate": 1
    }
    assert controller["state_stationarity_certified"] is False
    assert controller["state_stationarity_blocker"] == (
        "independent_working_state_stationarity_unresolved"
    )
    assert controller["decision_kind"] == "unresolved"
    assert controller["stage_b_eligible"] is False
    modeled_minimum_core = controller["modeled_minimum_core"]
    assert modeled_minimum_core["combined_mode_requested"] is True
    assert modeled_minimum_core["mathematical_eligibility"]["eligible"] is False
    assert modeled_minimum_core["mathematical_eligibility"]["reason"] == (
        "state_stationarity_certificate_missing_or_population_stale"
    )
    assert modeled_minimum_core["state_token_digest"] is None
    assert modeled_minimum_core["actionable"] is False


def test_sr_escape_requires_explicit_complete_reachable_population() -> None:
    saddle = _escape_feature("saddle", 61)

    with pytest.raises(
        ValueError,
        match="explicit complete Phase-III-reachable",
    ):
        rescore_historical_phase3_records_with_coordinate_models(
            [_record("saddle", 61, saddle)],
            cfg=FullScoreConfig(),
            sr_escape_mode="saddle_only",
        )


def test_sr_combined_hidden_unresolved_blocks_modeled_minimum() -> None:
    psd = _escape_feature(
        "visible-psd",
        62,
        summary_overrides={
            "joint_gain": 0.0,
            "negative_curvature_certified": False,
            "positive_semidefinite_certified": True,
            "supported_inertia_status": "psd",
            "minimum_hessian_eigenvalue_lower_bound": 0.25,
            "minimum_hessian_eigenvalue_upper_bound": 0.25,
            "full_trust_gain": 0.0,
            "active_restricted_trust_gain": 0.0,
        },
    )

    with pytest.raises(ValueError, match="exactly reproduce"):
        rescore_historical_phase3_records_with_coordinate_models(
            [_record("visible-psd", 62, psd)],
            cfg=FullScoreConfig(),
            sr_escape_mode="saddle_plus_modeled_minimum",
            sr_escape_reachable_record_ids=(
                "visible-psd::pool=62::position=0",
                "hidden-unresolved::pool=63::position=0",
            ),
        )


def test_sr_escape_unresolved_population_fails_closed() -> None:
    psd = _escape_feature(
        "psd",
        7,
        summary_overrides={
            "joint_gain": 0.0,
            "negative_curvature_certified": False,
            "positive_semidefinite_certified": True,
            "supported_inertia_status": "psd",
            "minimum_hessian_eigenvalue_lower_bound": 0.25,
            "minimum_hessian_eigenvalue_upper_bound": 0.25,
            "full_trust_gain": 0.0,
            "active_restricted_trust_gain": 0.0,
        },
    )
    unresolved = _escape_feature(
        "unresolved",
        8,
        summary_overrides={
            "feasible": False,
            "reason": "conditioning_gate",
            "joint_gain": 0.0,
            "trust_global_optimality_certified": False,
            "marginal_trust_gain_comparison_valid": False,
            "quotient_participation_resolved": False,
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("psd", 7, psd), _record("unresolved", 8, unresolved)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_plus_modeled_minimum",
        sr_escape_reachable_record_ids=(
            "psd::pool=7::position=0",
            "unresolved::pool=8::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "unresolved"
    assert controller["reason"] == (
        "reachable_population_contains_unresolved_certificate"
    )
    assert controller["reachable_population_complete"] is True
    assert controller["stage_b_eligible"] is False
    assert controller["actionable"] is False
    assert controller["admission_eligible_record_ids"] == []
    assert all(
        row["sr_escape_admission_eligible"] is False for row in rescored
    )
    assert all(row["selector_score"] == float("-inf") for row in rescored)


def test_sr_null_hessian_incompatibility_cannot_be_labeled_redundant() -> None:
    unsupported = _escape_feature(
        "unsupported-null-curvature",
        9,
        summary_overrides={
            "feasible": False,
            "reason": "raw_metric_null_hessian_incompatible",
            "raw_metric_support_status": "resolved",
            "raw_metric_null_compatibility_certified": False,
            "raw_metric_null_compatibility_reason": (
                "raw_metric_null_hessian_incompatible"
            ),
            "quotient_redundant_certified": True,
            "quotient_participation_resolved": False,
            "marginal_trust_gain_comparison_valid": False,
        },
    )

    rescored, telemetry = rescore_historical_phase3_records_with_coordinate_models(
        [_record("unsupported-null-curvature", 9, unsupported)],
        cfg=FullScoreConfig(),
        sr_escape_mode="saddle_only",
        sr_escape_reachable_record_ids=(
            "unsupported-null-curvature::pool=9::position=0",
        ),
    )

    controller = telemetry["sr_escape_controller"]
    assert controller["decision_kind"] == "unresolved"
    assert controller["reason"] == (
        "reachable_population_contains_unresolved_certificate"
    )
    assert controller["stage_b_eligible"] is False
    assert rescored[0]["sr_escape_admission_eligible"] is False
