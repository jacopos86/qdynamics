from __future__ import annotations

import copy

import numpy as np
import pytest

from pipelines.static_adapt import checkpoint_telemetry
from pipelines.static_adapt.adapt_pipeline import (
    _apply_repeat_live_prune_guard,
    _phase3_shadow_recommended_mu_from_history,
)
from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.output_artifacts import (
    _resolved_output_phase12_energy_model_policies,
    _resolved_output_phase3_response_coordinate_scope,
    build_output_payload,
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    PHASE1_SCORE_MODE_TRUST_REGION_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
    PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1,
    SR_ROUTE_PROFILE_CANDIDATE_V4,
    canonical_sr_snake_v4_contract,
    canonical_sr_snake_v4_contract_sha256,
)


def _v4_args():
    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)
    return parser.parse_args(
        [
            "--sr-route-profile",
            "sr_snake_v4",
            "--adapt-resume-compile-smoke",
            "off",
        ]
    )


def test_v4_model_nomination_bypasses_historical_repeat_prune_guard() -> None:
    kept, blocked_rows, telemetry = _apply_repeat_live_prune_guard(
        candidate_indices=[0],
        labels_now=["repeat"],
        blocked_labels=["repeat"],
        bypass_for_model_nominated_single_trial=True,
    )

    assert kept == [0]
    assert blocked_rows == []
    assert telemetry["active"] is False
    assert telemetry["bypassed"] is True
    assert telemetry["post_guard_candidate_count"] == 1


def test_legacy_repeat_prune_guard_remains_unchanged() -> None:
    kept, blocked_rows, telemetry = _apply_repeat_live_prune_guard(
        candidate_indices=[0],
        labels_now=["repeat"],
        blocked_labels=["repeat"],
        bypass_for_model_nominated_single_trial=False,
    )

    assert kept == []
    assert blocked_rows == [
        {
            "index": 0,
            "label": "repeat",
            "reason": "previous_live_prune_acceptance_same_label",
        }
    ]
    assert telemetry["active"] is True
    assert telemetry["bypassed"] is False


def test_phase3_shadow_mu_restores_latest_zero_query_history_state() -> None:
    mu, source = _phase3_shadow_recommended_mu_from_history(
        [
            {
                "phase3_shadow_damping_receipt": {
                    "schema": "route_a_phase3_shadow_damping_receipt_v1",
                    "recommended_mu": 0.25,
                    "applied_mu": 0.0,
                    "added_query_count": 0,
                }
            },
            {
                "phase3_shadow_damping_receipt": {
                    "schema": "route_a_phase3_shadow_damping_receipt_v1",
                    "recommended_mu": 0.75,
                    "applied_mu": 0.0,
                    "added_query_count": 0,
                }
            },
        ]
    )

    assert mu == pytest.approx(0.75)
    assert source == "previous_history_receipt"


def test_phase3_shadow_mu_resume_fails_closed_on_applied_damping() -> None:
    with pytest.raises(RuntimeError, match="zero-query diagnostic contract"):
        _phase3_shadow_recommended_mu_from_history(
            [
                {
                    "phase3_shadow_damping_receipt": {
                        "schema": "route_a_phase3_shadow_damping_receipt_v1",
                        "recommended_mu": 0.75,
                        "applied_mu": 0.1,
                        "added_query_count": 0,
                    }
                }
            ]
        )


def _phase3_shadow_receipt() -> dict[str, object]:
    return {
        "schema": "route_a_phase3_shadow_damping_receipt_v1",
        "policy": "zero_query_mapped_seed_shadow_v1",
        "status": "resolved",
        "recommended_mu": np.float64(0.25),
        "applied_mu": 0.0,
        "added_query_count": 0,
    }


def _prune_runtime_payload() -> dict[str, object]:
    return {
        "enabled": True,
        "phase1_prune_trust_state_before": {
            "schema": "affine_deletion_fs_trust_state_v1",
            "trust_radius": np.float64(0.125),
            "metric_damping": 0.0,
        },
        "phase1_prune_trust_state_after": {
            "schema": "affine_deletion_fs_trust_state_v1",
            "trust_radius": np.float64(0.0625),
            "metric_damping": np.float64(1.0e-6),
        },
        "phase1_prune_trust_update": {
            "schema": "affine_deletion_fs_trust_update_v1",
            "radius_action": "contract",
            "metric_damping_action": (
                "complete_same_trial_underprediction_increase"
            ),
        },
        "phase1_prune_trial_receipt": {
            "schema": "affine_deletion_fs_trust_same_trial_receipt_v1",
            "trial_id": "trial:7:2",
            "prediction_trial_id": "trial:7:2",
            "realization_trial_id": "trial:7:2",
            "estimator_trial_branch_id": "sr_v4_prune_trial:test",
            "estimator_trial_classification": "discarded_prune",
        },
        "phase1_prune_affine_deletion_model": {
            "schema": "full_logical_affine_deletion_fs_trust_v1",
            "pre_support_coordinate_count": np.int64(7),
            "supported_rank": np.int64(6),
            "classical_quantum_query_charge": 0,
        },
        "phase1_prune_exact_refit_work_accounting": {
            "schema": "sr_v4_prune_exact_refit_work_accounting_v1",
            "classification": "discarded_prune",
            "nfev": 11,
            "included_in_total_nfev": True,
            "included_in_total_estimator_ledger": True,
            "estimator_trial_branch_id": "sr_v4_prune_trial:test",
            "included_in_winning_lineage": False,
            "included_in_discarded_prune_accounting": True,
            "separate_estimator_discarded_prune_bucket_available": True,
        },
        "repeat_label_guard": {
            "active": False,
            "bypassed": True,
            "bypass_reason": "v4_model_nominated_single_exact_trial_v1",
        },
    }


def _v4_adapt_payload() -> dict[str, object]:
    contract = canonical_sr_snake_v4_contract()
    digest = canonical_sr_snake_v4_contract_sha256()
    response_scope = PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    chart = "expanded_runtime_projected_logical_v1"
    phase12_policies = {
        "phase1_score_mode": PHASE1_SCORE_MODE_TRUST_REGION_V1,
        "phase1_energy_model": PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
        "phase2_curvature_policy": (
            PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
        ),
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
        ),
    }
    history_row = {
        "depth": 1,
        "selected_op": "x",
        "phase3_shadow_damping_receipt": _phase3_shadow_receipt(),
        "post_admission_prune": _prune_runtime_payload(),
    }
    return {
        "success": True,
        "energy": -1.0,
        "exact_gs_energy": -1.0,
        "abs_delta_e": 0.0,
        "operators": ["x"],
        "history": [history_row],
        "sr_route_profile_request": SR_ROUTE_PROFILE_CANDIDATE_V4,
        "sr_route_profile_resolved": SR_ROUTE_PROFILE_CANDIDATE_V4,
        "sr_route_profile_contract": contract,
        "sr_route_profile_contract_sha256": digest,
        "phase3_response_coordinate_scope": response_scope,
        **phase12_policies,
        "finite_angle_fallback": False,
        "phase3_enable_rescue_requested": False,
        "phase3_enable_rescue_effective": False,
        "static_route_identity": {
            "route_family": "singleton_response_snake",
            "route_profile": SR_ROUTE_PROFILE_CANDIDATE_V4,
            "powell_coordinate_chart_policy": chart,
            "sr_route_profile_request": SR_ROUTE_PROFILE_CANDIDATE_V4,
            "sr_route_profile_contract": contract,
            "sr_route_profile_contract_sha256": digest,
            "phase3_response_coordinate_scope": response_scope,
            **phase12_policies,
            "finite_angle_fallback": False,
        },
        "optimizer_coordinate_chart": {
            "powell_coordinate_chart_policy": chart,
        },
    }


def test_checkpoint_preserves_v4_prune_state_models_and_receipts() -> None:
    compact = checkpoint_telemetry._compact_prune_audit(
        _prune_runtime_payload()
    )

    assert compact["phase1_prune_trust_state_before"]["trust_radius"] == 0.125
    assert compact["phase1_prune_trust_state_after"]["metric_damping"] == 1.0e-6
    assert compact["phase1_prune_trust_update"]["radius_action"] == "contract"
    assert compact["phase1_prune_trial_receipt"]["trial_id"] == "trial:7:2"
    assert compact["phase1_prune_affine_deletion_model"][
        "pre_support_coordinate_count"
    ] == 7
    assert compact["phase1_prune_exact_refit_work_accounting"][
        "classification"
    ] == "discarded_prune"
    assert compact["phase1_prune_exact_refit_work_accounting"][
        "estimator_trial_branch_id"
    ] == "sr_v4_prune_trial:test"
    assert compact["repeat_label_guard"]["bypassed"] is True


def test_checkpoint_marks_missing_v4_runtime_receipts_absent() -> None:
    compact = checkpoint_telemetry._compact_prune_audit(None)

    for key in (
        "phase1_prune_trust_state_before",
        "phase1_prune_trust_state_after",
        "phase1_prune_trust_update",
        "phase1_prune_trial_receipt",
        "phase1_prune_affine_deletion_model",
        "phase1_prune_exact_refit_work_accounting",
        "phase1_prune_no_feasible_model",
        "repeat_label_guard",
    ):
        assert compact[key] is None


@pytest.mark.parametrize(
    "field_name",
    [
        "phase1_prune_trust_state_before",
        "phase1_prune_trust_state_after",
        "phase1_prune_trust_update",
        "phase1_prune_trial_receipt",
        "phase1_prune_affine_deletion_model",
        "phase1_prune_exact_refit_work_accounting",
        "phase1_prune_no_feasible_model",
        "repeat_label_guard",
    ],
)
def test_checkpoint_fails_closed_on_nonmapping_prune_runtime_payload(
    field_name: str,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        checkpoint_telemetry._compact_prune_audit({field_name: "invalid"})


def test_checkpoint_preserves_v4_no_feasible_prune_hold_receipt() -> None:
    receipt = {
        "schema": "sr_v4_no_feasible_affine_deletion_models_v1",
        "status": "skipped_no_feasible_affine_deletion_models",
        "reason": "all_affine_deletion_models_infeasible",
        "model_count": 3,
        "feasible_model_count": 0,
        "legacy_nomination_fallback_used": False,
        "exact_delete_refit_trial_count": 0,
        "trust_state_action": "hold_exactly",
    }
    compact = checkpoint_telemetry._compact_prune_audit(
        {"phase1_prune_no_feasible_model": receipt}
    )

    assert compact["phase1_prune_no_feasible_model"] == receipt


def test_checkpoint_history_preserves_phase3_shadow_damping_receipt() -> None:
    row = checkpoint_telemetry._compact_history_row_for_checkpoint(
        {
            "depth": 4,
            "phase3_shadow_damping_receipt": _phase3_shadow_receipt(),
        },
        fallback_depth=4,
    )

    receipt = row["phase3_shadow_damping_receipt"]
    assert receipt["schema"] == "route_a_phase3_shadow_damping_receipt_v1"
    assert receipt["recommended_mu"] == 0.25
    assert receipt["added_query_count"] == 0


def test_checkpoint_history_preserves_signed_active_prefix_for_v4_resume() -> None:
    active_prefix = {
        "schema": "active_prefix_checkpoint_v1",
        "controller_round": 3,
        "active_ansatz_depth": 2,
        "operator_labels_exyz": ["x", "z"],
        "operator_coefficients": [0.25, -0.5],
        "operator_insertion_positions": [0, 1],
        "sha256": "signed-active-prefix-unit-test",
    }
    row = checkpoint_telemetry._compact_history_row_for_checkpoint(
        {
            "depth": 3,
            "selected_op": "y",
            "batch_size": 1,
            "post_admission_prune": _prune_runtime_payload(),
            "active_prefix_checkpoint": active_prefix,
        },
        fallback_depth=3,
    )

    assert row["active_prefix_checkpoint"] == active_prefix


def test_checkpoint_history_fails_closed_on_invalid_shadow_receipt() -> None:
    with pytest.raises(TypeError, match="phase3_shadow_damping_receipt"):
        checkpoint_telemetry._compact_history_row_for_checkpoint(
            {"phase3_shadow_damping_receipt": []},
            fallback_depth=1,
        )


def test_v4_result_manifest_preserves_policy_identity_and_runtime_receipts() -> None:
    args = _v4_args()
    adapt_payload = _v4_adapt_payload()
    psi = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])

    payload = build_output_payload(
        args=args,
        cli_adapt_continuation_mode="phase3_v1",
        adapt_payload=adapt_payload,
        ordered_labels_exyz=["e"],
        coeff_map_exyz={"e": 0.0},
        hmat=np.zeros((2, 2), dtype=complex),
        gs_energy_exact=-1.0,
        gs_energy_source="unit",
        psi0=psi,
        ansatz_input_state_for_adapt=psi,
        ansatz_input_state_source="hf",
        ansatz_input_state_kind="reference_state",
        trajectory=[],
        adapt_ref_import=None,
        dense_eigh_enabled=True,
        hilbert_dim=2,
        adapt_ref_base_depth=0,
        initial_state_source_resolved="hf",
        initial_state_kind_resolved="reference_state",
    )

    settings = payload["settings"]
    assert settings["sr_route_profile_request"] == SR_ROUTE_PROFILE_CANDIDATE_V4
    assert settings["sr_route_profile_contract"] == canonical_sr_snake_v4_contract()
    assert settings["sr_route_profile_contract_sha256"] == (
        canonical_sr_snake_v4_contract_sha256()
    )
    assert settings["adapt_finite_angle_fallback"] is False
    assert settings["phase3_enable_rescue"] is False
    assert settings["sr_route_profile_contract"]["semantic_invariants"][
        "finite_angle_fallback_active"
    ] is False
    assert settings["phase3_hardware_cost_normalization_mode"] == (
        "family_robust_symmetric_arctan_v1"
    )
    assert settings["phase3_shadow_damping_policy"] == (
        "mapped_seed_zero_query_v1"
    )
    assert payload["adapt_vqe"]["finite_angle_fallback"] is False
    assert payload["adapt_vqe"]["phase3_enable_rescue_requested"] is False
    assert payload["adapt_vqe"]["phase3_enable_rescue_effective"] is False
    assert payload["adapt_vqe"]["static_route_identity"][
        "finite_angle_fallback"
    ] is False
    assert settings["phase1_prune_recovery_trust_radius"] == 0.125
    assert settings["phase1_prune_schur_nomination_route"] == (
        "full_logical_fs_trust_delete_refit_v1"
    )
    assert settings["phase1_prune_metric_schur_solve_mode"] == (
        "affine_deletion_global_trust_v1"
    )
    assert settings["phase1_prune_trust_update_policy"] == (
        "modeled_local_fs_conservative_v1"
    )
    assert settings["phase1_prune_metric_mu_update_policy"] == (
        "same_trial_underprediction_monotone_v1"
    )
    assert payload["adapt_vqe"]["history"][0][
        "phase3_shadow_damping_receipt"
    ]["schema"] == "route_a_phase3_shadow_damping_receipt_v1"
    assert payload["adapt_vqe"]["history"][0]["post_admission_prune"][
        "phase1_prune_trial_receipt"
    ]["trial_id"] == "trial:7:2"
    assert payload["adapt_vqe"]["static_route_identity"][
        "sr_route_profile_contract_sha256"
    ] == canonical_sr_snake_v4_contract_sha256()


def test_output_manifest_defaults_new_v4_fields_for_legacy_args() -> None:
    args = _build_adapt_arg_parser(
        adapt_gradient_parity_rtol=1.0e-7
    ).parse_args(["--adapt-resume-compile-smoke", "off"])
    fields = (
        "phase1_prune_recovery_trust_radius",
        "phase1_prune_schur_nomination_route",
        "phase1_prune_metric_schur_mu",
        "phase1_prune_metric_schur_solve_mode",
        "phase1_prune_metric_schur_cost_weighting",
        "phase1_prune_trust_update_policy",
        "phase1_prune_metric_mu_update_policy",
        "phase1_prune_endpoint_overlap_policy",
        "phase1_prune_live_min_depth",
        "phase3_shadow_damping_policy",
    )
    for field in fields:
        delattr(args, field)
    # This emulates a historical in-process Namespace.  It is deliberately not
    # a v4 route request because a canonical v4 invocation must retain every
    # normalized field rather than fall back.
    psi = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])

    payload = build_output_payload(
        args=args,
        cli_adapt_continuation_mode="off",
        adapt_payload={},
        ordered_labels_exyz=["e"],
        coeff_map_exyz={"e": 0.0},
        hmat=np.zeros((2, 2), dtype=complex),
        gs_energy_exact=-1.0,
        gs_energy_source="unit",
        psi0=psi,
        ansatz_input_state_for_adapt=psi,
        ansatz_input_state_source="hf",
        ansatz_input_state_kind="reference_state",
        trajectory=[],
        adapt_ref_import=None,
        dense_eigh_enabled=True,
        hilbert_dim=2,
        adapt_ref_base_depth=0,
        initial_state_source_resolved="hf",
        initial_state_kind_resolved="reference_state",
    )

    settings = payload["settings"]
    assert settings["phase1_prune_recovery_trust_radius"] == 0.0
    assert settings["phase1_prune_schur_nomination_route"] == "hessian_coupling_v1"
    assert settings["phase1_prune_metric_schur_mu"] == pytest.approx(1.0e-6)
    assert settings["phase1_prune_metric_schur_solve_mode"] == "stationary_gw_zero_v1"
    assert settings["phase1_prune_trust_update_policy"] == "off"
    assert settings["phase1_prune_metric_mu_update_policy"] == "off"
    assert settings["phase1_prune_endpoint_overlap_policy"] == "off"
    assert settings["phase1_prune_live_min_depth"] == 0
    assert settings["phase3_shadow_damping_policy"] == "off"


def test_v4_output_fails_closed_without_runtime_response_scope() -> None:
    args = _v4_args()
    adapt_payload = _v4_adapt_payload()
    adapt_payload.pop("phase3_response_coordinate_scope")
    adapt_payload["static_route_identity"] = dict(
        adapt_payload["static_route_identity"]
    )
    adapt_payload["static_route_identity"].pop(
        "phase3_response_coordinate_scope"
    )

    with pytest.raises(ValueError, match="missing phase3_response_coordinate_scope"):
        _resolved_output_phase3_response_coordinate_scope(
            args=args,
            adapt_payload=adapt_payload,
        )


def test_v4_output_rejects_legacy_phase1_score_mode_telemetry() -> None:
    args = _v4_args()
    adapt_payload = _v4_adapt_payload()
    adapt_payload["phase1_score_mode"] = "legacy_simple_v1"

    with pytest.raises(ValueError, match="conflicting phase1_score_mode"):
        _resolved_output_phase12_energy_model_policies(
            args=args,
            adapt_payload=adapt_payload,
        )


def test_v4_output_resolves_phase12_policies_from_runtime_telemetry() -> None:
    args = _v4_args()
    adapt_payload = _v4_adapt_payload()
    phase12_keys = (
        "phase1_score_mode",
        "phase1_energy_model",
        "phase2_curvature_policy",
        "phase2_cheap_curvature_proxy_policy",
    )
    runtime_telemetry = {
        key: adapt_payload.pop(key)
        for key in phase12_keys
    }
    adapt_payload["static_route_identity"] = dict(
        adapt_payload["static_route_identity"]
    )
    for key in phase12_keys:
        adapt_payload["static_route_identity"].pop(key)
    adapt_payload["phase12_energy_model_telemetry"] = runtime_telemetry

    resolved, source = _resolved_output_phase12_energy_model_policies(
        args=args,
        adapt_payload=adapt_payload,
    )

    assert resolved == runtime_telemetry
    assert source == "resolved_runtime_telemetry"


def test_v4_result_rejects_route_contract_tamper() -> None:
    args = _v4_args()
    adapt_payload = _v4_adapt_payload()
    tampered = copy.deepcopy(canonical_sr_snake_v4_contract())
    tampered["execution_settings"]["phase3_shadow_damping_policy"] = "off"
    adapt_payload["sr_route_profile_contract"] = tampered

    psi = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    with pytest.raises(ValueError, match="contract drifted"):
        build_output_payload(
            args=args,
            cli_adapt_continuation_mode="phase3_v1",
            adapt_payload=adapt_payload,
            ordered_labels_exyz=["e"],
            coeff_map_exyz={"e": 0.0},
            hmat=np.zeros((2, 2), dtype=complex),
            gs_energy_exact=-1.0,
            gs_energy_source="unit",
            psi0=psi,
            ansatz_input_state_for_adapt=psi,
            ansatz_input_state_source="hf",
            ansatz_input_state_kind="reference_state",
            trajectory=[],
            adapt_ref_import=None,
            dense_eigh_enabled=True,
            hilbert_dim=2,
            adapt_ref_base_depth=0,
            initial_state_source_resolved="hf",
            initial_state_kind_resolved="reference_state",
        )
