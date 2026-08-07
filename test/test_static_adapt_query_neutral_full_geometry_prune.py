from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.hh_continuation_pruning import (
    AffineDeletionFSTrustState,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
)
from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.query_neutral_full_geometry_prune import (
    PAPER_I_QUERY_NEUTRAL_PRUNE_ENERGY_GUARD_ABS_TOL,
    PAPER_I_QUERY_NEUTRAL_PRUNE_MODELED_ENERGY_CHANGE_MAX,
    PAPER_I_QUERY_NEUTRAL_PRUNE_TARGET_ABS_DELTA_E,
    QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_HOLD_V1,
    QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_PROPOSAL_V1,
    build_query_neutral_prune_proposal,
    build_query_neutral_prune_source_unavailable_hold,
    combined_transition_energy_guard,
    normalize_full_geometry_prune_source,
    realized_source_model_step_after_deletion,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1_EXECUTION_SETTINGS,
    PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_ALIAS_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1,
    canonical_sr_snake_contract,
    canonical_sr_snake_contract_sha256,
    normalize_sr_route_profile_request,
    validate_sr_route_profile_runtime_settings,
)


def _full_geometry_source() -> dict[str, object]:
    return {
        "schema": "historical_singleton_coordinate_model_v1",
        "joint_batch_context_mode": "full_ansatz_v1",
        "joint_linear_solve_policy_effective": (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        ),
        "feasible": True,
        "position_id": 1,
        "candidate_label": "candidate",
        "candidate_pool_index": 17,
        "active_coordinate_identities": ["old-0", "old-1"],
        "batch_coordinate_identities": [
            {
                "candidate_label": "",
                "candidate_pool_index": 17,
                "position_id": 1,
                "global_child_identity": "",
            }
        ],
        "G_AA_raw": [[1.0, 0.0], [0.0, 1.0]],
        "G_AB_raw": [[0.0], [0.0]],
        "G_BB_raw": [[1.0]],
        "H_AA_raw": [[1.0, 0.0], [0.0, 1.0]],
        "H_AB_raw": [[0.0], [0.0]],
        "H_BB_raw": [[1.0]],
        # Scorer convention: descent gradient.  Candidate descent is positive.
        "g_A": [0.0, 0.0],
        "g_B": [1.0],
        "joint_step": [0.0, 0.0, 0.2],
    }


def _normalized_model() -> dict[str, object]:
    return normalize_full_geometry_prune_source(
        selector_summary=_full_geometry_source(),
        pre_admission_labels=["old-0", "old-1"],
        pre_admission_theta=np.asarray([0.05, 0.2]),
        candidate_label="candidate",
        candidate_pool_index=17,
        candidate_position=1,
    )


def test_full_geometry_source_maps_insertion_and_reuses_geometry_without_queries():
    model = _normalized_model()

    assert model["model_post_indices"] == (0, 2, 1)
    assert model["old_post_indices"] == (0, 2)
    assert model["candidate_post_index"] == 1
    assert np.allclose(model["theta"], [0.05, 0.2, 0.0])
    assert np.allclose(model["gradient"], [0.0, 0.0, -1.0])
    assert np.asarray(model["metric"]).shape == (3, 3)
    assert np.asarray(model["hessian"]).shape == (3, 3)
    receipt = model["receipt"]
    assert receipt["phase3_coordinate_scope"] == (
        "full_active_plus_singleton_v1"
    )
    assert receipt["candidate_label_placeholder_filled"] is True
    assert receipt["duplicate_measurement_performed"] is False
    assert receipt["incremental_quantum_query_charge"] == 0


def test_full_geometry_source_restores_json_empty_initial_block_shapes():
    source = {
        "schema": "historical_singleton_coordinate_model_v1",
        "joint_batch_context_mode": "full_ansatz_v1",
        "joint_linear_solve_policy_effective": (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        ),
        "feasible": True,
        "position_id": 0,
        "candidate_label": "candidate",
        "candidate_pool_index": 17,
        "active_coordinate_identities": [],
        "batch_coordinate_identities": [
            {
                "candidate_label": "",
                "candidate_pool_index": 17,
                "position_id": 0,
            }
        ],
        "G_AA_raw": [],
        "G_AB_raw": [],
        "G_BB_raw": [[1.0]],
        "H_AA_raw": [],
        "H_AB_raw": [],
        "H_BB_raw": [[1.0]],
        "g_A": [],
        "g_B": [1.0],
    }

    model = normalize_full_geometry_prune_source(
        selector_summary=source,
        pre_admission_labels=[],
        pre_admission_theta=[],
        candidate_label="candidate",
        candidate_pool_index=17,
        candidate_position=0,
    )

    assert np.asarray(model["metric"]).shape == (1, 1)
    assert np.asarray(model["hessian"]).shape == (1, 1)
    hold = build_query_neutral_prune_proposal(
        model=model,
        metadata_rows=[],
        trust_state=AffineDeletionFSTrustState(radius=0.0625),
        selector_step=1,
        protect_steps=2,
    )
    assert hold["nominated"] is False
    assert hold["reason"] == "no_conservative_eligible_deletion"
    assert hold["incremental_quantum_query_charge"] == 0


def test_full_geometry_source_rejects_material_window_and_identity_drift():
    material = _full_geometry_source()
    material["joint_batch_context_mode"] = "active_window_v1"
    with pytest.raises(RuntimeError, match="material or rolling window"):
        normalize_full_geometry_prune_source(
            selector_summary=material,
            pre_admission_labels=["old-0", "old-1"],
            pre_admission_theta=[0.05, 0.2],
            candidate_label="candidate",
            candidate_pool_index=17,
            candidate_position=1,
        )

    identity_drift = _full_geometry_source()
    identity_drift["active_coordinate_identities"] = ["old-1", "old-0"]
    with pytest.raises(RuntimeError, match="active-coordinate identities"):
        normalize_full_geometry_prune_source(
            selector_summary=identity_drift,
            pre_admission_labels=["old-0", "old-1"],
            pre_admission_theta=[0.05, 0.2],
            candidate_label="candidate",
            candidate_pool_index=17,
            candidate_position=1,
        )


def test_infeasible_phase3_source_holds_pruning_without_measurement():
    source = _full_geometry_source()
    source["feasible"] = False
    source["reason"] = "geometry_expansion_no_coordinate_prediction"
    trust_state = AffineDeletionFSTrustState(radius=0.0625)

    hold = build_query_neutral_prune_source_unavailable_hold(
        selector_summary=source,
        trust_state=trust_state,
    )

    assert hold["nominated"] is False
    assert hold["reason"] == "phase3_coordinate_model_infeasible_hold"
    assert hold["source_reason"] == (
        "geometry_expansion_no_coordinate_prediction"
    )
    assert hold["source_geometry_available"] is False
    assert hold["duplicate_measurement_performed"] is False
    assert hold["incremental_quantum_query_charge"] == 0
    assert hold["trust_radius"] == pytest.approx(0.0625)

    source["feasible"] = True
    with pytest.raises(RuntimeError, match="must use the ordinary"):
        build_query_neutral_prune_source_unavailable_hold(
            selector_summary=source,
            trust_state=trust_state,
        )


def test_prune_proposal_is_single_conservative_and_query_neutral():
    proposal = build_query_neutral_prune_proposal(
        model=_normalized_model(),
        metadata_rows=[
            {"first_seen_step": 0, "cooldown_remaining": 0},
            {"first_seen_step": 0, "cooldown_remaining": 0},
        ],
        trust_state=AffineDeletionFSTrustState(radius=0.4),
        selector_step=5,
        protect_steps=2,
        modeled_energy_change_max=0.0,
    )

    assert proposal["schema"] == (
        QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_PROPOSAL_V1
    )
    assert proposal["nominated"] is True
    assert proposal["deletion_model_index"] == 0
    assert proposal["deletion_post_index"] == 0
    assert proposal["predicted_energy_change"] <= 0.0
    assert len(proposal["warm_start_post_delete_logical_theta"]) == 2
    assert proposal["duplicate_measurement_performed"] is False
    assert proposal["incremental_quantum_query_charge"] == 0


def test_prune_hold_respects_protection_and_cooldown_without_queries():
    hold = build_query_neutral_prune_proposal(
        model=_normalized_model(),
        metadata_rows=[
            {"first_seen_step": 4, "cooldown_remaining": 0},
            {"first_seen_step": 0, "cooldown_remaining": 3},
        ],
        trust_state=AffineDeletionFSTrustState(radius=0.4),
        selector_step=5,
        protect_steps=2,
    )

    assert hold["schema"] == QUERY_NEUTRAL_FULL_GEOMETRY_PRUNE_HOLD_V1
    assert hold["nominated"] is False
    assert [row["reason"] for row in hold["ineligible"]] == [
        "protected",
        "cooldown",
    ]
    assert hold["incremental_quantum_query_charge"] == 0


def test_combined_transition_guard_commits_or_rolls_back_classically():
    accepted = combined_transition_energy_guard(
        energy_before=-1.0,
        energy_after=-1.1,
        absolute_tolerance=1.0e-10,
    )
    rejected = combined_transition_energy_guard(
        energy_before=-1.0,
        energy_after=-0.9,
        absolute_tolerance=1.0e-10,
    )

    assert accepted["accepted"] is True
    assert accepted["second_refit_performed"] is False
    assert rejected["accepted"] is False
    assert rejected["rollback_classical"] is True
    assert rejected["rollback_quantum_query_charge"] == 0


def test_paper_i_prune_threshold_and_guard_are_target_scaled_and_strict():
    assert PAPER_I_QUERY_NEUTRAL_PRUNE_TARGET_ABS_DELTA_E == pytest.approx(
        2.0e-4
    )
    assert PAPER_I_QUERY_NEUTRAL_PRUNE_MODELED_ENERGY_CHANGE_MAX == pytest.approx(
        -2.0e-6
    )
    assert PAPER_I_QUERY_NEUTRAL_PRUNE_ENERGY_GUARD_ABS_TOL == pytest.approx(
        1.0e-12
    )
    assert PAPER_I_QUERY_NEUTRAL_PRUNE_MODELED_ENERGY_CHANGE_MAX == pytest.approx(
        -0.01 * PAPER_I_QUERY_NEUTRAL_PRUNE_TARGET_ABS_DELTA_E
    )


def test_realized_step_embeds_deleted_coordinate_as_zero_in_source_order():
    model = _normalized_model()
    proposal = build_query_neutral_prune_proposal(
        model=model,
        metadata_rows=[
            {"first_seen_step": 0, "cooldown_remaining": 0},
            {"first_seen_step": 0, "cooldown_remaining": 0},
        ],
        trust_state=AffineDeletionFSTrustState(radius=0.4),
        selector_step=5,
        protect_steps=2,
    )

    realized = realized_source_model_step_after_deletion(
        model=model,
        proposal=proposal,
        final_post_delete_logical_theta=[0.3, 0.25],
    )

    # Model order is old-0, old-1, candidate.  old-0 was deleted.
    assert np.allclose(realized, [-0.05, 0.05, 0.3])


def test_query_neutral_route_is_full_geometry_and_changes_only_prune_settings():
    contract = canonical_sr_snake_contract(
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_ALIAS_V1
    )
    parent = canonical_sr_snake_contract(
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    )

    assert contract["route_profile"] == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
    )
    assert contract["lineage_authority"]["parent_contract_sha256"] == (
        canonical_sr_snake_contract_sha256(
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        )
    )
    assert contract["execution_settings"]["phase3_response_coordinate_scope"] == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    assert contract["semantic_invariants"]["phase3_material_window_policy"] == (
        "off"
    )
    assert contract["semantic_invariants"]["prune_verification_beam"] == "off"
    assert contract["semantic_invariants"]["prune_explicit_query_delta"] == 0
    changed = set(
        contract["lineage_authority"][
            "only_intended_parent_setting_changes"
        ]
    )
    actual_changed = {
        key
        for key, value in contract["execution_settings"].items()
        if parent["execution_settings"].get(key) != value
    }
    assert actual_changed == changed


def test_query_neutral_route_alias_normalizes_to_distinct_route_identity():
    assert normalize_sr_route_profile_request(
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_ALIAS_V1
    ) == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
    )


def test_query_neutral_route_cli_and_runtime_contract_round_trip():
    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)
    args = parser.parse_args(
        [
            "--sr-route-profile",
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_ALIAS_V1,
            "--adapt-max-depth",
            "50",
        ]
    )
    contract = canonical_sr_snake_contract(
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
    )
    digest = canonical_sr_snake_contract_sha256(
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
    )

    assert args.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1
    )
    assert args.phase1_prune_enabled is True
    assert args.phase1_prune_recovery_trust_radius == pytest.approx(0.00390625)
    assert args.phase3_response_coordinate_scope == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    runtime = dict(
        CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_QUERY_NEUTRAL_PRUNE_V1_EXECUTION_SETTINGS
    )
    runtime["adapt_max_depth"] = 50
    assert validate_sr_route_profile_runtime_settings(
        profile_request=args.sr_route_profile_resolved,
        contract=contract,
        contract_sha256=digest,
        runtime_settings=runtime,
    ) == contract
