from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from pipelines.scaffold.hh_continuation_scoring import (
    BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
)
from pipelines.static_adapt.route_a_schur_selector import (
    ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
    ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
    ROUTE_A_TRUST_REGION_FIXED,
    ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1,
    RouteASchurSelectorConfig,
    TrustRegionUpdateConfig,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
    JointLinearSolveConfig,
    solve_joint_linear_model,
)
from pipelines.static_adapt.route_a_trust_region import (
    RouteARoundTrustRegionSnapshot,
    RouteATrustRegionState,
    contract_rejected_active_stationarity_trust_region_state,
    contract_rejected_saddle_trust_region_state,
    exact_fubini_study_distance,
    initialize_trust_region_state,
    resolve_round_trust_region_snapshot,
    round_trust_region_stage_receipt,
    score_config_with_round_trust_radius,
    selector_config_with_round_trust_radius,
    update_sr_active_only_trust_region_state,
    update_geometry_expansion_trust_region_state,
    update_trust_region_state,
)

import pipelines.static_adapt.route_a_trust_region as trust_region_module


@dataclass(frozen=True)
class _ScoreConfigFixture:
    rho: float
    marker: str = "unchanged"


def _state_at_distance(distance: float) -> np.ndarray:
    return np.asarray([math.cos(distance), math.sin(distance)], dtype=complex)


def _projected_source_metric_summary(*, binding: bool = True) -> dict[str, object]:
    return {
        "joint_linear_solve_policy_effective": (
            "supported_metric_projected_generalized_trust_v1"
        ),
        "G_AA_raw": [[1.0]],
        "G_AB_raw": [[0.0]],
        "G_BB_raw": [[4.0]],
        "raw_metric_eigenvalues": [1.0, 4.0],
        "metric_retained_mask": [True, True],
        "metric_support_threshold": 1.0e-12,
        "supported_metric_projection_provenance_id": "projected-proof",
        "trust_radius_sq": 0.25**2,
        "trust_radius_binding_tolerance_sq": 1.0e-12,
        "joint_step": [0.25, 0.0],
        "joint_fubini_study_displacement_sq": 0.25**2,
        "applied_predicted_reduction": 0.1,
        "trust_radius_binding": bool(binding),
        "geometry_workspace": {"active_indices": [0]},
    }


def test_projected_no_overlap_trust_contracts_inverse_sqrt_without_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden_overlap(*_args, **_kwargs):
        raise AssertionError("endpoint overlap must not be evaluated")

    monkeypatch.setattr(
        trust_region_module,
        "exact_fubini_study_distance",
        _forbidden_overlap,
    )
    state = RouteATrustRegionState(radius=0.25)
    payload = update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1
        ),
        context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        selector_summary=_projected_source_metric_summary(),
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=np.asarray([0.0, 1.0], dtype=complex),
        energy_before=-1.0,
        energy_after_refit=-1.08,
        energy_improvement_tolerance=1.0e-8,
        full_coordinate_refit=True,
        realized_joint_step=[0.5, 0.0],
    )

    assert payload["displacement_ratio"] == pytest.approx(2.0)
    assert state.radius == pytest.approx(0.25 / math.sqrt(2.0))
    assert payload["update_reason"] == (
        "realized_source_metric_motion_larger_contract"
    )
    assert payload["endpoint_overlap_measurement_required"] is False
    assert payload["endpoint_overlap_measurement_performed"] is False
    assert payload["endpoint_overlap_query_charge"] == 0
    assert payload["realized_fs_displacement_exact"] is None
    assert payload["model_agreement_ratio"] == pytest.approx(0.8)


def test_projected_no_overlap_trust_accepts_zero_active_serialized_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trust_region_module,
        "exact_fubini_study_distance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("endpoint overlap must not be evaluated")
        ),
    )
    summary = _projected_source_metric_summary()
    summary.update(
        {
            "G_AA_raw": [],
            "G_AB_raw": [],
            "G_BB_raw": [[1.0]],
            "raw_metric_eigenvalues": [1.0],
            "metric_retained_mask": [True],
            "joint_step": [0.25],
            "joint_fubini_study_displacement_sq": 0.25**2,
            "geometry_workspace": {"active_indices": []},
        }
    )
    state = RouteATrustRegionState(radius=0.25)

    payload = update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=(
                ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1
            )
        ),
        context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        selector_summary=summary,
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=np.asarray([0.0, 1.0], dtype=complex),
        energy_before=-1.0,
        energy_after_refit=-1.08,
        energy_improvement_tolerance=1.0e-8,
        full_coordinate_refit=True,
        realized_joint_step=[0.5],
    )

    transaction = payload["source_metric_trust_transaction"]
    assert transaction["supported_rank"] == 1
    assert transaction["endpoint_overlap_required"] is False
    assert transaction["endpoint_overlap_query_charge"] == 0
    assert transaction["transaction_complete"] is True
    assert payload["source_metric_trust_transaction_failure"] is None


def test_projected_no_overlap_trust_accepts_material_window_full_metric_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden_overlap(*_args, **_kwargs):
        raise AssertionError("endpoint overlap must not be evaluated")

    monkeypatch.setattr(
        trust_region_module,
        "exact_fubini_study_distance",
        _forbidden_overlap,
    )
    state = RouteATrustRegionState(radius=0.25)
    transaction = {
        "schema": (
            "sr_material_window_full_source_metric_accepted_path_transaction_v1"
        ),
        "predicted_source_metric_displacement": 0.25,
        "predicted_source_metric_displacement_sq": 0.25**2,
        "realized_source_metric_displacement": 0.5,
        "realized_source_metric_displacement_sq": 0.5**2,
        "branch_trust_radius_before": 0.25,
        "adaptive_radius_rescale_authority": (
            "full_accepted_refit_supported_source_gram_coordinates_v1"
        ),
        "source_metric_reused_from_accepted_refit": True,
        "endpoint_overlap_required": False,
        "endpoint_overlap_query_charge": 0,
        "incremental_quantum_query_charge": 0,
        "transaction_complete": True,
    }
    payload = update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1
        ),
        context_mode="active_window_v1",
        selector_summary={
            "joint_linear_solve_policy_effective": (
                "supported_metric_projected_generalized_trust_v1"
            ),
            "joint_fubini_study_displacement_sq": 0.25**2,
            "applied_predicted_reduction": 0.1,
            "trust_radius_binding": True,
            "geometry_workspace": {"active_indices": [1]},
        },
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=np.asarray([0.0, 1.0], dtype=complex),
        energy_before=-1.0,
        energy_after_refit=-1.08,
        energy_improvement_tolerance=1.0e-8,
        full_coordinate_refit=True,
        precomputed_source_metric_trust_transaction=transaction,
    )

    assert payload["displacement_ratio"] == pytest.approx(2.0)
    assert state.radius == pytest.approx(0.25 / math.sqrt(2.0))
    assert payload["context_mode"] == "active_window_v1"
    assert payload["endpoint_overlap_measurement_performed"] is False
    assert payload["source_metric_trust_transaction"] == transaction


def test_projected_no_overlap_trust_expands_only_at_boundary_with_descent() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1
        ),
        context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        selector_summary=_projected_source_metric_summary(binding=True),
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=np.asarray([1.0, 0.0], dtype=complex),
        energy_before=-1.0,
        energy_after_refit=-1.01,
        energy_improvement_tolerance=1.0e-8,
        full_coordinate_refit=True,
        realized_joint_step=[0.125, 0.0],
    )

    assert payload["displacement_ratio"] == pytest.approx(0.5)
    assert state.radius == pytest.approx(0.25 / math.sqrt(0.5))
    assert payload["update_reason"] == (
        "binding_radius_realized_source_metric_motion_smaller_expand"
    )


def test_projected_no_overlap_geometry_expansion_holds_without_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trust_region_module,
        "exact_fubini_study_distance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("endpoint overlap must not be evaluated")
        ),
    )
    state = RouteATrustRegionState(radius=0.25)
    payload = update_geometry_expansion_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1
        ),
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=np.asarray([0.0, 1.0], dtype=complex),
        energy_before=-1.0,
        energy_after_refit=-1.1,
        energy_improvement_tolerance=1.0e-8,
        full_coordinate_refit=True,
    )

    assert state.radius == pytest.approx(0.25)
    assert payload["update_reason"] == (
        "geometry_expansion_no_coordinate_prediction_no_overlap_hold"
    )
    assert payload["endpoint_overlap_query_charge"] == 0


def test_round_snapshot_freezes_radius_and_update_count() -> None:
    state = RouteATrustRegionState(radius=0.25, update_count=3)

    snapshot = resolve_round_trust_region_snapshot(
        state,
        fallback_radius=0.5,
    )
    state.radius = 0.125
    state.update_count += 1

    assert isinstance(snapshot, RouteARoundTrustRegionSnapshot)
    assert snapshot.radius == pytest.approx(0.25)
    assert snapshot.update_count == 3
    assert snapshot.source == "branch_local_state"


def test_round_snapshot_projects_one_radius_into_score_and_selector_configs() -> None:
    snapshot = RouteARoundTrustRegionSnapshot(
        radius=0.125,
        update_count=4,
        source="branch_local_state",
    )
    score_config = _ScoreConfigFixture(rho=0.5)
    selector_config = RouteASchurSelectorConfig(max_fubini_study_step=0.75)

    resolved_score = score_config_with_round_trust_radius(score_config, snapshot)
    resolved_selector = selector_config_with_round_trust_radius(
        selector_config,
        snapshot,
    )

    assert resolved_score.rho == pytest.approx(snapshot.radius)
    assert resolved_score.marker == "unchanged"
    assert resolved_selector.max_fubini_study_step == pytest.approx(
        snapshot.radius
    )
    assert score_config.rho == pytest.approx(0.5)
    assert selector_config.max_fubini_study_step == pytest.approx(0.75)


def test_round_snapshot_fallback_and_branch_state_are_isolated() -> None:
    fallback = resolve_round_trust_region_snapshot(None, fallback_radius=0.4)
    left = RouteATrustRegionState(radius=0.2, update_count=1)
    right = left.clone()
    left.radius = 0.1
    left.update_count += 1

    left_snapshot = resolve_round_trust_region_snapshot(
        left,
        fallback_radius=0.4,
    )
    right_snapshot = resolve_round_trust_region_snapshot(
        right,
        fallback_radius=0.4,
    )

    assert fallback.radius == pytest.approx(0.4)
    assert fallback.source == "configured_fallback"
    assert left_snapshot.radius == pytest.approx(0.1)
    assert left_snapshot.update_count == 2
    assert right_snapshot.radius == pytest.approx(0.2)
    assert right_snapshot.update_count == 1


def test_round_stage_receipt_requires_one_identical_radius() -> None:
    snapshot = RouteARoundTrustRegionSnapshot(
        radius=0.2,
        update_count=7,
        source="branch_local_state",
    )
    stage_radii = {
        "macro_phase1": 0.2,
        "macro_phase2": 0.2,
        "child_phase1": 0.2,
        "child_phase2": 0.2,
        "final_selector": 0.2,
    }

    receipt = round_trust_region_stage_receipt(
        snapshot,
        stage_radii=stage_radii,
    )

    assert receipt["radius"] == pytest.approx(0.2)
    assert receipt["update_count_at_round_start"] == 7
    assert receipt["stage_radii"] == stage_radii


def test_round_stage_receipt_rejects_midround_radius_drift() -> None:
    snapshot = RouteARoundTrustRegionSnapshot(
        radius=0.2,
        update_count=7,
        source="branch_local_state",
    )

    with pytest.raises(ValueError, match="drift at child_phase2"):
        round_trust_region_stage_receipt(
            snapshot,
            stage_radii={
                "macro_phase1": 0.2,
                "macro_phase2": 0.2,
                "child_phase1": 0.2,
                "child_phase2": 0.1,
                "final_selector": 0.2,
            },
        )


def test_checkpoint_resume_preserves_round_update_epoch() -> None:
    original = RouteATrustRegionState(radius=0.25, update_count=5)
    restored = initialize_trust_region_state(
        initial_radius=0.5,
        checkpoint_payload=original.as_dict(),
    )
    round_five = resolve_round_trust_region_snapshot(
        restored,
        fallback_radius=0.5,
    )

    _update(
        state=restored,
        policy=ROUTE_A_TRUST_REGION_FIXED,
    )
    round_six = resolve_round_trust_region_snapshot(
        restored,
        fallback_radius=0.5,
    )

    assert round_five.update_count == 5
    assert round_six.update_count == 6
    assert restored.update_count == round_five.update_count + 1


def _update(
    *,
    state: RouteATrustRegionState,
    policy: str,
    predicted: float = 0.25,
    realized: float = 0.125,
    clipped: bool = True,
    radius_binding: bool | None = None,
    energy_after: float = -1.1,
) -> dict[str, object]:
    radius_binding_resolved = clipped if radius_binding is None else radius_binding
    return update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(policy=policy),
        context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        selector_summary={
            "joint_fubini_study_displacement_sq": predicted**2,
            "applied_predicted_reduction": 0.2,
            "trust_clipped": clipped,
            "trust_regularization_applied": clipped,
            "trust_radius_binding": radius_binding_resolved,
            "geometry_workspace": {"active_indices": [0, 1]},
        },
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=_state_at_distance(realized),
        energy_before=-1.0,
        energy_after_refit=energy_after,
        energy_improvement_tolerance=1e-8,
        full_coordinate_refit=True,
        selected_records=[
            {"route_a_global_pauli_identity": "pauli:test"}
        ],
        selected_effective_positions=[2],
    )


def test_fixed_policy_leaves_radius_unchanged() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = _update(state=state, policy=ROUTE_A_TRUST_REGION_FIXED)

    assert state.radius == pytest.approx(0.25)
    assert payload["update_reason"] == "fixed_policy"


def test_smaller_realized_displacement_contracts_with_sqrt_damping() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = _update(
        state=state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
        predicted=0.25,
        realized=0.125,
    )

    assert payload["displacement_ratio"] == pytest.approx(0.5)
    assert state.radius == pytest.approx(0.25 * math.sqrt(0.5))
    assert payload["update_reason"] == "realized_displacement_smaller"


def test_larger_realized_displacement_expands_only_when_clipped() -> None:
    clipped_state = RouteATrustRegionState(radius=0.25)
    clipped = _update(
        state=clipped_state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
        predicted=0.125,
        realized=0.25,
        clipped=True,
    )
    inactive_state = RouteATrustRegionState(radius=0.25)
    inactive = _update(
        state=inactive_state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
        predicted=0.125,
        realized=0.25,
        clipped=False,
    )

    assert clipped_state.radius == pytest.approx(0.25 * math.sqrt(2.0))
    assert clipped["update_reason"] == (
        "binding_radius_realized_displacement_larger"
    )
    assert inactive_state.radius == pytest.approx(0.25)
    assert inactive["update_reason"] == "radius_inactive_hold"


def test_regularization_without_radius_binding_does_not_expand() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = _update(
        state=state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
        predicted=1e-6,
        realized=2e-6,
        clipped=True,
        radius_binding=False,
    )

    assert state.radius == pytest.approx(0.25)
    assert payload["selector_regularization_applied"] is True
    assert payload["trust_radius_binding"] is False
    assert payload["trust_clipped"] is False
    assert payload["update_reason"] == "radius_inactive_hold"


def test_nonpositive_energy_descent_vetoes_expansion() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = _update(
        state=state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
        predicted=0.125,
        realized=0.25,
        clipped=True,
        energy_after=-0.9,
    )

    assert state.radius == pytest.approx(0.25)
    assert payload["update_reason"] == "energy_veto_hold"


def test_update_payload_has_no_admission_rollback_signal() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = _update(
        state=state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
    )

    assert "structural_rollback" not in payload
    assert "depth_rollback" not in payload


def test_radius_has_no_arbitrary_upper_cap() -> None:
    state = RouteATrustRegionState(radius=10.0)
    payload = _update(
        state=state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
        predicted=0.05,
        realized=0.5,
        clipped=True,
    )

    assert state.radius == pytest.approx(10.0 * math.sqrt(2.0))
    assert payload["radius_after"] > 10.0


def test_unbounded_policy_uses_direct_ratio_without_lower_or_rate_caps() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = _update(
        state=state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
        predicted=0.25,
        realized=0.0025,
    )

    assert payload["displacement_ratio"] == pytest.approx(0.01)
    assert state.radius == pytest.approx(0.025)
    assert payload["scientific_radius_lower_bound"] == 0.0
    assert payload["scientific_radius_upper_bound"] is None
    assert payload["rate_limit_applied"] is False
    assert payload["numerical_floor_applied"] is False


def test_unbounded_policy_does_not_cap_valid_expansion() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = _update(
        state=state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
        predicted=0.05,
        realized=0.5,
        clipped=True,
    )

    assert payload["displacement_ratio"] == pytest.approx(10.0)
    assert state.radius == pytest.approx(0.25 * math.sqrt(10.0))
    assert payload["rate_limit_applied"] is False


def test_zero_scientific_radius_minimum_is_valid() -> None:
    config = TrustRegionUpdateConfig(
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
        radius_min=0.0,
    )

    assert config.radius_min == 0.0
    assert config.as_dict()["scientific_radius_min_effective"] == 0.0
    assert config.as_dict()["scientific_radius_max_effective"] is None
    assert config.as_dict()["contraction_factor_min_effective"] is None
    assert config.as_dict()["expansion_factor_max_effective"] is None
    assert config.as_dict()["rate_limiter_mode"] == "none"


def test_rejected_active_stationarity_backtracking_contracts_only_branch_trust():
    state = RouteATrustRegionState(radius=0.25)

    payload = contract_rejected_active_stationarity_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=(
                ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
            )
        ),
        guard_payload={
            "status": "rejected",
            "reason": "active_only_nonlinear_backtracking_exhausted",
            "transaction_failure_kind": (
                "finite_nonlinear_model_disagreement"
            ),
            "nonlinear_backtracking_exhausted": True,
            "all_backtracking_candidates_finite": True,
            "trust_action": "contract_branch_radius",
            "backtracking_attempt_count": 4,
        },
    )

    assert state.radius == pytest.approx(0.125)
    assert state.update_count == 1
    assert payload["update_reason"] == (
        "finite_nonlinear_endpoint_disagreement_half_radius_retry"
    )
    assert payload["radius_refinement_progressed"] is True
    assert payload["no_state_transition"] is True
    assert payload["singleton_consumed"] is False
    assert payload["ansatz_state_unchanged"] is True
    assert payload["parameter_state_unchanged"] is True
    assert payload["controller_depth_mutated"] is False
    assert payload["admission_history_unchanged"] is True
    assert payload["numerical_or_mapping_failure"] is False


def test_two_finite_nondownhill_saddle_signs_contract_only_trust_state() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = contract_rejected_saddle_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
            contraction_factor_min=0.5,
        ),
        guard_payload={
            "all_mapped_signs_finite": True,
            "all_mapped_signs_non_downhill": True,
            "mapped_seed_incumbent_energy": -1.0,
            "sign_evaluations": [
                {"sign": 1, "energy": -0.9},
                {"sign": -1, "energy": -0.8},
            ],
            "saddle_taylor_contraction_certificate": {
                "valid": True,
                "negative_curvature_magnitude_lower_bound": 10.0,
                "third_derivative_upper_bound": 0.0,
                "energy_comparison_width": 1e-3,
                "certified_radius_after": 0.125,
            },
        },
    )

    assert state.radius == pytest.approx(0.125)
    assert state.update_count == 1
    assert payload["update_reason"] == "sr_saddle_taylor_certified_contract_refine"
    assert payload["certified_taylor_radius_available"] is True
    assert payload["requires_refinement"] is True
    assert payload["ansatz_state_mutated"] is False
    assert payload["parameter_state_mutated"] is False
    assert payload["depth_mutated"] is False
    assert payload["sr_saddle_transaction_outcome"] == (
        "radius_contract_refinement_no_state_mutation"
    )


def test_two_finite_nondownhill_saddle_signs_numerically_backtrack_without_l3(
) -> None:
    state = RouteATrustRegionState(radius=0.25)

    payload = contract_rejected_saddle_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=(
                ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
            )
        ),
        guard_payload={
            "all_mapped_signs_finite": True,
            "all_mapped_signs_non_downhill": True,
            "mapped_seed_incumbent_energy": -1.0,
            "sign_evaluations": [
                {"sign": 1, "energy": -0.9},
                {"sign": -1, "energy": -0.8},
            ],
            "energy_comparison_width": {
                "schema": "sr_simultaneous_energy_comparison_width_v1",
                "aggregate_simultaneous_comparison_width": 1e-12,
            },
        },
    )

    assert state.radius == pytest.approx(0.125)
    assert state.update_count == 1
    assert payload["update_reason"] == (
        "sr_saddle_numerical_backtracking_contract_refine"
    )
    assert payload["certified_taylor_radius_available"] is False
    receipt = payload["saddle_numerical_backtracking_receipt"]
    assert receipt["valid"] is True
    assert receipt["certified_taylor_guarantee"] is False
    assert receipt["backtracking_rule"] == (
        "ordinary_geometric_half_radius_v1"
    )
    assert payload["ansatz_state_mutated"] is False
    assert payload["parameter_state_mutated"] is False
    assert payload["depth_mutated"] is False


def test_rejected_saddle_incomplete_sign_pair_holds_radius() -> None:
    state = RouteATrustRegionState(radius=0.25)

    payload = contract_rejected_saddle_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=(
                ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
            )
        ),
        guard_payload={
            "all_mapped_signs_finite": False,
            "all_mapped_signs_non_downhill": False,
            "transaction_failure_kind": "mapping",
        },
    )

    assert state.radius == pytest.approx(0.25)
    assert state.update_count == 1
    assert payload["update_reason"] == (
        "sr_saddle_transaction_failure_marker_hold"
    )
    assert payload["numerical_or_mapping_failure"] is True


def test_rejected_saddle_failure_marker_overrides_contradictory_finite_flags(
) -> None:
    state = RouteATrustRegionState(radius=0.25)

    payload = contract_rejected_saddle_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=(
                ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
            )
        ),
        guard_payload={
            "transaction_failure_kind": "mapping",
            "all_mapped_signs_finite": True,
            "all_mapped_signs_non_downhill": True,
        },
    )

    assert state.radius == pytest.approx(0.25)
    assert payload["update_reason"] == (
        "sr_saddle_transaction_failure_marker_hold"
    )
    assert payload["numerical_or_mapping_failure"] is True


def test_sr_active_only_trust_update_records_no_singleton_transaction() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = update_sr_active_only_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=(
                ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
            )
        ),
        global_coordinate_summary={
            "joint_linear_solve_policy_effective": (
                "supported_metric_global_trust_eigh_v2"
            ),
            "G_AA_raw": [[1.0]],
            "G_AB_raw": [[0.0]],
            "G_BB_raw": [[1.0]],
            "raw_metric_eigenvalues": [1.0, 1.0],
            "metric_retained_mask": [True, True],
            "metric_whitening_ridge": 0.25,
            "supported_metric_whitening_provenance_id": "full-proof",
            "raw_metric_support_epsilon_G": 1e-12,
            "trust_radius_sq": 0.0625,
            "trust_radius_binding_tolerance_sq": 1e-12,
        },
        active_restriction_summary={
            "valid": True,
            "trust_global_optimality_certified": True,
            "active_coordinate_count": 1,
            "batch_coordinate_count": 1,
            "joint_step": [0.2, 0.0],
            "predicted_reduction": 0.1,
            "joint_fubini_study_displacement_sq": 0.04,
            "active_restriction_batch_zero_tolerance": 1e-12,
            "active_restriction_source": (
                "full_v2_supported_metric_whitened_coordinate_restriction_v1"
            ),
            "active_restriction_provenance_id": "active-proof",
            "restricted_coordinate_trust_solve": {
                "trust_regularization_applied": True,
                "trust_clipped": True,
                "trust_radius_binding": True,
                "whitened_step_norm": math.sqrt(0.05),
            },
        },
        active_indices=[3],
        realized_joint_step=[0.1, 0.0],
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=_state_at_distance(0.19),
        energy_before=-1.0,
        energy_after_refit=-1.05,
        energy_improvement_tolerance=1e-8,
        seed_guard_payload={
            "status": "accepted",
            "mapped_seed_exact_gain": 0.05,
            "mapped_seed_predicted_reduction": 0.1,
        },
        optimizer_outcome_payload={
            "selected_source": "mapped_active_restriction_seed",
        },
    )

    assert payload["sr_active_only_correction"] is True
    assert payload["singleton_consumed"] is False
    assert payload["selected_record_identities"] == []
    assert payload["selected_effective_positions"] == []
    assert payload["active_indices"] == [3]
    assert payload["candidate_block_zero_certified"] is True
    assert payload["context_mode"] == (
        "sr_active_only_shared_support_correction_v1"
    )
    assert payload["displacement_ratio"] == pytest.approx(0.5)
    assert payload["realized_fs_displacement_exact"] == pytest.approx(0.19)
    assert payload["displacement_ratio_metric"] == (
        "regularized_supported_metric_whitened_coordinates_v1"
    )
    stabilized = payload["stabilized_trust_transaction"]
    assert stabilized["schema"] == (
        "sr_v2_stabilized_trust_accepted_path_transaction_v1"
    )
    assert stabilized["metric_whitening_ridge"] == pytest.approx(0.25)
    assert stabilized["certified_trust_radius"] == pytest.approx(0.25)
    assert stabilized["trust_radius_provenance"] == (
        "coordinate_summary_trust_radius_sq_matched_to_branch_state_v1"
    )
    assert stabilized["realized_stabilized_trust_displacement"] == (
        pytest.approx(math.sqrt(0.0125))
    )
    assert stabilized["exact_endpoint_fubini_study_distance_role"] == (
        "diagnostic_only"
    )
    assert state.radius == pytest.approx(math.sqrt(0.0125) * math.sqrt(0.5))
    assert payload["model_agreement_ratio"] == pytest.approx(0.5)
    assert payload["update_reason"] == "mapped_seed_model_agreement_calibrated"
    assert state.last_update == payload


def _v2_two_coordinate_summary() -> dict[str, object]:
    return {
        "joint_linear_solve_policy_effective": (
            "supported_metric_global_trust_eigh_v2"
        ),
        "G_AA_raw": [[1.0]],
        "G_AB_raw": [[0.0]],
        "G_BB_raw": [[4.0]],
        "raw_metric_eigenvalues": [1.0, 4.0],
        "metric_retained_mask": [True, True],
        "retained_metric_eigenvalues": [1.0, 4.0],
        "metric_whitening_ridge": 0.5,
        "metric_stabilization_lambda_G": 0.5,
        "whitening_denominators": [1.5, 4.5],
        "supported_metric_whitening_provenance_id": "stabilized-proof",
        "raw_metric_support_epsilon_G": 1e-12,
        "trust_radius_sq": 0.0625,
        "trust_radius_binding_tolerance_sq": 1e-12,
        "joint_step": [0.2, 0.0],
        "joint_fubini_study_displacement_sq": 0.04,
        "whitened_step_norm": math.sqrt(0.06),
        "applied_predicted_reduction": 0.1,
        "trust_radius_binding": False,
        "geometry_workspace": {"active_indices": [0]},
    }


def test_v2_stabilized_metric_not_raw_gram_controls_radius() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
        ),
        context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        selector_summary=_v2_two_coordinate_summary(),
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=_state_at_distance(0.1),
        energy_before=-1.0,
        energy_after_refit=-1.1,
        energy_improvement_tolerance=1e-8,
        full_coordinate_refit=True,
        realized_joint_step=[0.0, 0.1],
        seed_guard_payload={
            "status": "accepted",
            "mapped_seed_exact_gain": 0.1,
            "mapped_seed_predicted_reduction": 0.1,
        },
        optimizer_outcome_payload={
            "post_refit_safe_source": "mapped_downhill_seed",
        },
    )

    stabilized = payload["stabilized_trust_transaction"]
    assert stabilized["realized_raw_metric_local_displacement"] == pytest.approx(0.2)
    assert stabilized["realized_stabilized_trust_displacement"] == pytest.approx(
        math.sqrt(0.045)
    )
    assert state.radius == pytest.approx(math.sqrt(0.045))
    assert state.radius != pytest.approx(0.2)
    assert payload["model_agreement_authority"] == (
        "quadratic_prediction_vs_exact_mapped_seed_v1"
    )


def test_v2_zero_ridge_fallback_controls_accepted_path_transaction() -> None:
    coordinate_map = np.diag([1.0, 1.0e-3])
    gram = coordinate_map.T @ np.eye(2) @ coordinate_map
    hessian = coordinate_map.T @ np.eye(2) @ coordinate_map
    gradient = coordinate_map.T @ np.asarray([0.2, -0.1])
    solve = solve_joint_linear_model(
        gram=gram,
        hessian=hessian,
        gradient=gradient,
        active_coordinate_count=1,
        config=JointLinearSolveConfig(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            rank_relative_tolerance=1.0e-6,
            metric_regularization=1.0e-9,
            energy_regularization=1.0e-12,
            max_fubini_study_step=0.25,
            global_trust_kkt_residual_accuracy=1.0e-8,
            global_trust_metric_distortion_budget=5.0e-2,
        ),
    )
    assert solve.feasible is True
    assert solve.telemetry[
        "metric_stabilization_zero_ridge_fallback_solver_status"
    ] == "certified"
    summary = {
        **solve.telemetry,
        "G_AA_raw": [[float(gram[0, 0])]],
        "G_AB_raw": [[float(gram[0, 1])]],
        "G_BB_raw": [[float(gram[1, 1])]],
        "trust_radius_sq": 0.25**2,
        "trust_radius_binding_tolerance_sq": 1.0e-12,
        "joint_step": [float(value) for value in solve.joint_step.tolist()],
        "geometry_workspace": {"active_indices": [0]},
    }
    state = RouteATrustRegionState(radius=0.25)
    payload = update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
        ),
        context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        selector_summary=summary,
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=_state_at_distance(0.1),
        energy_before=-1.0,
        energy_after_refit=-1.0 - solve.predicted_reduction,
        energy_improvement_tolerance=1.0e-8,
        full_coordinate_refit=True,
        realized_joint_step=solve.joint_step,
        seed_guard_payload={
            "status": "accepted",
            "mapped_seed_exact_gain": solve.predicted_reduction,
            "mapped_seed_predicted_reduction": solve.predicted_reduction,
        },
        optimizer_outcome_payload={
            "post_refit_safe_source": "mapped_downhill_seed",
        },
    )

    transaction = payload["stabilized_trust_transaction"]
    expected_displacement = math.sqrt(solve.fubini_study_displacement_sq)
    assert transaction["metric_whitening_ridge"] == 0.0
    assert transaction["metric_stabilization_lambda_G"] == 0.0
    assert transaction["whitening_denominator_authority"] == (
        "solver_whitening_denominators_v1"
    )
    assert transaction["supported_metric_whitening_provenance_id"] == (
        solve.telemetry["supported_metric_whitening_provenance_id"]
    )
    assert transaction["predicted_stabilized_trust_displacement"] == (
        pytest.approx(expected_displacement)
    )
    assert transaction["realized_stabilized_trust_displacement"] == (
        pytest.approx(expected_displacement)
    )
    assert transaction["realized_raw_metric_local_displacement"] == (
        pytest.approx(expected_displacement)
    )
    assert transaction["stabilized_trust_transaction_complete"] is True


def test_v2_seed_agreement_precedes_discarded_powell_endpoint() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
        ),
        context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        selector_summary=_v2_two_coordinate_summary(),
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=_state_at_distance(0.1),
        energy_before=-1.0,
        energy_after_refit=-1.05,
        energy_improvement_tolerance=1e-8,
        full_coordinate_refit=True,
        realized_joint_step=[0.2, 0.0],
        seed_guard_payload={
            "status": "accepted",
            "mapped_seed_exact_gain": 0.05,
            "mapped_seed_predicted_reduction": 0.1,
        },
        optimizer_outcome_payload={
            "post_refit_safe_source": "mapped_downhill_seed",
            "optimizer_energy": -1.01,
        },
    )

    assert payload["mapped_seed_exact_reduction"] == pytest.approx(0.05)
    assert payload["powell_endpoint_reduction_diagnostic"] == pytest.approx(0.01)
    assert payload["model_agreement_ratio"] == pytest.approx(0.5)
    assert state.radius == pytest.approx(math.sqrt(0.06) * math.sqrt(0.5))


def test_v2_nonfinite_accepted_path_holds_radius() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
        ),
        context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        selector_summary=_v2_two_coordinate_summary(),
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=_state_at_distance(0.1),
        energy_before=-1.0,
        energy_after_refit=-1.1,
        energy_improvement_tolerance=1e-8,
        full_coordinate_refit=True,
        realized_joint_step=[float("nan"), 0.0],
        seed_guard_payload={
            "status": "accepted",
            "mapped_seed_exact_gain": 0.1,
            "mapped_seed_predicted_reduction": 0.1,
        },
        optimizer_outcome_payload={
            "post_refit_safe_source": "mapped_downhill_seed",
        },
    )

    assert state.radius == pytest.approx(0.25)
    assert payload["update_factor"] == pytest.approx(1.0)
    assert payload["update_reason"] == "numerical_or_mapping_failure_hold"
    assert payload["numerical_floor_applied"] is False


def test_v2_powell_without_certified_path_prediction_uses_stabilized_fallback() -> None:
    state = RouteATrustRegionState(radius=0.25)
    payload = update_trust_region_state(
        state,
        config=TrustRegionUpdateConfig(
            policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2
        ),
        context_mode=BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        selector_summary=_v2_two_coordinate_summary(),
        state_before=np.asarray([1.0, 0.0], dtype=complex),
        state_after_refit=_state_at_distance(0.1),
        energy_before=-1.0,
        energy_after_refit=-1.2,
        energy_improvement_tolerance=1e-8,
        full_coordinate_refit=True,
        realized_joint_step=[0.0, 0.1],
        seed_guard_payload={
            "status": "accepted",
            "mapped_seed_exact_gain": 0.05,
            "mapped_seed_predicted_reduction": 0.1,
        },
        optimizer_outcome_payload={
            "post_refit_safe_source": "optimizer_result",
            "optimizer_energy": -1.2,
        },
    )

    assert payload["accepted_path_prediction_certified"] is False
    assert payload["section8_displacement_fallback_applied"] is True
    assert payload["update_reason"] == (
        "post_powell_stabilized_displacement_fallback"
    )
    assert state.radius == pytest.approx(math.sqrt(0.045))


def test_exact_distance_is_global_phase_invariant() -> None:
    before = np.asarray([1.0, 0.0], dtype=complex)
    after = _state_at_distance(0.2)

    assert exact_fubini_study_distance(before, after) == pytest.approx(0.2)
    assert exact_fubini_study_distance(
        before,
        np.exp(1j * 0.71) * after,
    ) == pytest.approx(0.2)
    assert exact_fubini_study_distance(before, before) == pytest.approx(0.0)


def test_state_round_trip_preserves_radius_and_last_update() -> None:
    state = RouteATrustRegionState(radius=0.5)
    _update(
        state=state,
        policy=ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_V1,
    )
    restored = initialize_trust_region_state(
        initial_radius=0.25,
        checkpoint_payload=state.as_dict(),
    )

    assert restored is not state
    assert restored.radius == pytest.approx(state.radius)
    assert restored.reference_radius == pytest.approx(state.reference_radius)
    assert restored.update_count == state.update_count
    assert restored.last_update == state.last_update


def test_legacy_checkpoint_initializes_from_configured_radius() -> None:
    restored = initialize_trust_region_state(
        initial_radius=0.5,
        checkpoint_payload=None,
    )

    assert restored.radius == pytest.approx(0.5)
    assert restored.initialization_reason == "configured_initial_radius"

    legacy = initialize_trust_region_state(
        initial_radius=0.5,
        checkpoint_payload={},
    )
    assert legacy.radius == pytest.approx(0.5)
    assert legacy.initialization_reason == "legacy_checkpoint_missing_state"
