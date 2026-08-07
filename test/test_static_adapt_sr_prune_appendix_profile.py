from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipelines.static_adapt.cli_config import (
    _build_adapt_arg_parser,
    _build_run_hardcoded_adapt_vqe_kwargs,
)
from pipelines.static_adapt.resume_scaffold import (
    validate_resume_sr_route_profile_contract,
)
from pipelines.static_adapt.run_control import (
    _adapt_segment_loop_stop_reason,
    _initialize_adapt_segment_run,
    _resolve_adapt_segment_controls,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_SYMMETRIC_COST_FS_PRUNE_V1_EXECUTION_SETTINGS,
    PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256,
    canonical_sr_snake_symmetric_cost_fs_prune_v1_contract,
    canonical_sr_snake_symmetric_cost_fs_prune_v1_contract_sha256,
)


PROFILE_ALIAS = "sr_snake_symmetric_cost_fs_prune_nodamping_v1"

PRUNE_ONLY_CHANGED_FIELDS = {
    "phase1_prune_enabled",
    "phase1_prune_mode",
    "phase1_prune_max_candidates",
    "phase1_prune_local_window_size",
    "phase1_prune_recovery_trust_radius",
    "phase1_prune_schur_nomination_route",
    "phase1_prune_metric_schur_mu",
    "phase1_prune_metric_schur_solve_mode",
    "phase1_prune_metric_schur_cost_weighting",
    "phase1_prune_trust_update_policy",
    "phase1_prune_metric_mu_update_policy",
    "phase1_prune_endpoint_overlap_policy",
}


def _parser():
    return _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)


def _args(*extra: str):
    return _parser().parse_args(
        [
            "--sr-route-profile",
            PROFILE_ALIAS,
            "--adapt-max-depth",
            "50",
            *extra,
        ]
    )


def _runtime_kwargs(args) -> dict[str, object]:
    return _build_run_hardcoded_adapt_vqe_kwargs(
        args,
        h_poly=None,
        resolved_problem_context=SimpleNamespace(
            layout=SimpleNamespace(total_qubits=6)
        ),
        cli_adapt_continuation_mode="phase3_v1",
        adapt_ref_base_depth=0,
        psi_ref_override=None,
        psi_ref_source=None,
        psi_ref_handoff_state_kind=None,
        exact_gs_override=0.0,
        phase3_oracle_gradient_config=None,
        final_noise_audit_config=None,
    )


def test_prune_appendix_is_exact_prune_diff_from_main() -> None:
    main = CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS
    appendix = CANONICAL_SR_SNAKE_SYMMETRIC_COST_FS_PRUNE_V1_EXECUTION_SETTINGS

    assert set(main) <= set(appendix)
    changed = {
        field
        for field in appendix
        if field not in main or appendix[field] != main[field]
    }
    assert changed == PRUNE_ONLY_CHANGED_FIELDS

    for field in set(main).difference(PRUNE_ONLY_CHANGED_FIELDS):
        assert appendix[field] == main[field], field

    assert appendix["adapt_beam_live_branches"] == 1
    assert appendix["adapt_beam_children_per_parent"] == 1
    assert appendix["phase2_enable_batching"] is False
    assert appendix["phase3_enable_batching"] is False
    assert appendix["adapt_full_refit_every"] == 0
    assert appendix["adapt_final_full_refit"] == "false"
    assert appendix["phase3_shadow_damping_policy"] == "off"


def test_prune_appendix_materializes_undamped_full_logical_fs_trust() -> None:
    args = _args()

    assert args.sr_route_profile_request == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
    )
    assert args.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
    )
    assert args.sr_route_profile_contract == (
        canonical_sr_snake_symmetric_cost_fs_prune_v1_contract()
    )
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_symmetric_cost_fs_prune_v1_contract_sha256()
    )
    assert args.adapt_max_depth == 50
    assert args.phase3_response_coordinate_scope == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    assert args.phase1_prune_enabled is True
    assert args.phase1_prune_mode == "live"
    assert args.phase1_prune_max_candidates == 1
    assert args.phase1_prune_local_window_size == 0
    assert args.phase1_prune_recovery_trust_radius == pytest.approx(0.125)
    assert args.phase1_prune_schur_nomination_route == (
        "full_logical_fs_trust_delete_refit_v1"
    )
    assert args.phase1_prune_metric_schur_solve_mode == (
        "affine_deletion_global_trust_v1"
    )
    assert args.phase1_prune_metric_schur_mu == 0.0
    assert args.phase1_prune_metric_mu_update_policy == "off"
    assert args.phase1_prune_trust_update_policy == (
        "modeled_local_fs_conservative_v1"
    )
    assert args.phase1_prune_endpoint_overlap_policy == "off"
    assert args.phase3_shadow_damping_policy == "off"


def test_prune_appendix_contract_records_one_factor_lineage() -> None:
    contract = canonical_sr_snake_symmetric_cost_fs_prune_v1_contract()
    invariants = contract["semantic_invariants"]

    assert invariants["appendix_one_factor_ablation"] is True
    assert invariants["pruning_active"] is True
    assert invariants["prune_execution_scope"] == "live_only_v1"
    assert invariants["prune_response_scope"] == "full_active_logical_v1"
    assert invariants["prune_trust_constraint"] == (
        "complete_affine_deletion_fs_v1"
    )
    assert invariants["prune_acceptance_authority"] == (
        "measured_delete_and_refit_v1"
    )
    assert invariants["prune_metric_damping_active"] is False
    assert invariants["prune_metric_damping_update_active"] is False
    assert invariants["terminal_prune_active"] is False
    assert invariants["terminal_full_refit_active"] is False
    assert contract["lineage_authority"] == {
        "parent_route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
        "scientific_result_anchor_claimed": False,
    }


@pytest.mark.parametrize(
    "override",
    [
        ("--phase1-prune-metric-schur-mu", "1e-6"),
        (
            "--phase1-prune-metric-mu-update-policy",
            "same_trial_underprediction_monotone_v1",
        ),
        ("--phase1-prune-mode", "both"),
        ("--adapt-beam-live-branches", "2"),
        ("--phase2-enable-batching",),
        ("--adapt-final-full-refit", "true"),
    ],
)
def test_prune_appendix_rejects_noncontract_overrides(
    override: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _args(*override)


def test_prune_appendix_round_trips_runtime_and_resume_identity() -> None:
    args = _args()
    kwargs = _runtime_kwargs(args)
    contract = canonical_sr_snake_symmetric_cost_fs_prune_v1_contract()
    digest = canonical_sr_snake_symmetric_cost_fs_prune_v1_contract_sha256()

    assert kwargs["sr_route_profile_request"] == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
    )
    assert kwargs["sr_route_profile_contract"] == contract
    assert kwargs["sr_route_profile_contract_sha256"] == digest
    assert kwargs["phase1_prune_enabled"] is True
    assert kwargs["phase1_prune_metric_schur_mu"] == 0.0
    assert kwargs["phase1_prune_metric_mu_update_policy"] == "off"
    assert kwargs["phase3_shadow_damping_policy"] == "off"

    payload = {
        "settings": {
            "sr_route_profile_request": (
                SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
            ),
            "sr_route_profile_contract": contract,
            "sr_route_profile_contract_sha256": digest,
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
            ),
            "phase1_score_mode": "trust_region_v1",
            "phase1_energy_model": "first_order_fs_trust_v1",
            "phase2_curvature_policy": "measured_required_fail_closed_v1",
            "phase2_cheap_curvature_proxy_policy": "off",
        }
    }
    validation = validate_resume_sr_route_profile_contract(
        payload,
        expected_profile_request=SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1,
        expected_contract=contract,
        expected_contract_sha256=digest,
    )
    assert validation["status"] == "pass"
    assert validation["artifact_profile_request"] == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_FS_PRUNE_V1
    )


def test_prune_round50_horizon_is_not_final_active_depth_equality() -> None:
    controls = _resolve_adapt_segment_controls(
        adapt_segment_id="prune-round50-horizon",
        adapt_segment_target_depth=50,
        adapt_segment_target_controller_round=50,
        adapt_segment_max_new_admissions=50,
        adapt_segment_wallclock_cap_s=None,
    )
    state = _initialize_adapt_segment_run(
        controls=controls,
        current_depth=0,
        current_runtime_parameter_count=0,
        requested_max_depth=50,
        start_time_s=0.0,
        source_controller_round=0,
    )

    # Accepted deletions may keep the active ansatz below the controller-round
    # horizon.  That must not stop the segment before round 50.
    state.new_admissions_count = 49
    assert _adapt_segment_loop_stop_reason(
        state,
        current_depth=31,
        current_controller_round=49,
        now_s=1.0,
    ) is None

    state.new_admissions_count = 50
    assert _adapt_segment_loop_stop_reason(
        state,
        current_depth=31,
        current_controller_round=50,
        now_s=2.0,
    ) == "segment_target_controller_round"
