from __future__ import annotations

from typing import Any

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.sr_snake import (
    AppendOnlyInsertion,
    RecoverabilityPruneReceipt,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRRunRequest,
    SRStopPolicy,
    run_sr_snake,
)
from pipelines.static_adapt.sr_snake.contracts import RecoverabilityPruning
from pipelines.static_adapt.sr_snake._context import (
    _resolve_execution_context,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract,
    canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract,
    canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract_sha256,
)

EXPECTED_ROUTE_DIGEST = (
    "44f9ef70c114e88efd4ff9c3fb1c64abc7d7a25c15a978bbe735243ac1dd27de"
)
EXPECTED_ACTIVE_PRUNE_SETTINGS = {
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "live",
    "phase1_prune_max_candidates": 1,
    "phase1_prune_local_window_size": 0,
    "phase1_prune_max_regression": 1.0e-8,
    "phase1_prune_retained_gain_ratio": 0.5,
    "phase1_prune_protect_steps": 2,
    "phase1_prune_recovery_trust_radius": 0.125,
    "phase1_prune_schur_nomination_route": (
        "full_logical_fs_trust_delete_refit_v1"
    ),
    "phase1_prune_metric_schur_solve_mode": (
        "affine_deletion_global_trust_v1"
    ),
    "phase1_prune_metric_schur_mu": 0.0,
    "phase1_prune_metric_schur_cost_weighting": "off",
    "phase1_prune_trust_update_policy": (
        "modeled_local_fs_conservative_v1"
    ),
    "phase1_prune_metric_mu_update_policy": "off",
    "phase1_prune_endpoint_overlap_policy": "off",
}
TAMPERED_ACTIVE_PRUNE_SETTINGS = {
    "phase1_prune_enabled": False,
    "phase1_prune_policy": "tampered_policy",
    "phase1_prune_mode": "both",
    "phase1_prune_max_candidates": 2,
    "phase1_prune_local_window_size": 4,
    "phase1_prune_max_regression": 2.0e-8,
    "phase1_prune_retained_gain_ratio": 0.75,
    "phase1_prune_protect_steps": 3,
    "phase1_prune_recovery_trust_radius": 0.25,
    "phase1_prune_schur_nomination_route": "hessian_coupling_v1",
    "phase1_prune_metric_schur_solve_mode": "stationary_gw_zero_v1",
    "phase1_prune_metric_schur_mu": 1.0e-6,
    "phase1_prune_metric_schur_cost_weighting": (
        "ansatz_entry_denominator_v1"
    ),
    "phase1_prune_trust_update_policy": "off",
    "phase1_prune_metric_mu_update_policy": "adaptive",
    "phase1_prune_endpoint_overlap_policy": "measured",
}


def _small_hh_problem() -> Any:
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        ),
        exact_energy_impl=adapt_pipeline._exact_gs_energy_for_problem,
    )


def _pruning_request() -> SRRunRequest:
    return SRRunRequest(
        method=SRMethodPolicy(
            insertion=AppendOnlyInsertion(),
            pruning=RecoverabilityPruning(),
        )
    )


def _pruning_run_request(rounds: int) -> SRRunRequest:
    return SRRunRequest(
        method=SRMethodPolicy(
            insertion=AppendOnlyInsertion(),
            pruning=RecoverabilityPruning(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=rounds)
        ),
    )


def _no_pruning_run_request(rounds: int) -> SRRunRequest:
    return SRRunRequest(
        method=SRMethodPolicy(insertion=AppendOnlyInsertion()),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=rounds)
        )
    )


def test_historical_recoverability_requires_its_explicit_compatibility_identity() -> None:
    with pytest.raises(
        ValueError,
        match="singleton \\+ append-only \\+ beam-off",
    ):
        SRMethodPolicy(pruning=RecoverabilityPruning())


def _accepted_public_prune_receipt(
    **overrides: Any,
) -> RecoverabilityPruneReceipt:
    values: dict[str, Any] = {
        "status": "accepted",
        "reason": "measured_delete_refit",
        "policy": "recoverability_ladder_v1",
        "nomination_policy": (
            "full_logical_fs_trust_delete_refit_v1"
        ),
        "source_state_fingerprint": "state:keep",
        "trust_radius_before": 0.125,
        "trust_radius_after": 0.125,
        "metric_damping": 0.0,
        "endpoint_overlap_query_charge": 0,
        "terminal_prune_active": False,
        "nomination_index": 1,
        "nomination_label": "operator:1",
        "predicted_energy_change": -0.01,
        "surrogate_used_for_acceptance": False,
        "trial_executed": True,
        "trial_branch_id": "sr_v4_prune_trial:fixture",
        "trial_classification": "committed_prune",
        "trial_s_alg": 3,
        "measured_energy_before": -1.0,
        "measured_energy_after": -1.01,
        "accepted": True,
        "deleted_index": 1,
        "deleted_label": "operator:1",
        "final_state_fingerprint": "state:deleted",
    }
    values.update(overrides)
    return RecoverabilityPruneReceipt(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"trial_classification": "discarded_prune"},
            "accepted status and committed_prune",
        ),
        ({"trial_s_alg": -1}, "work must be non-negative"),
        (
            {"deleted_index": None, "deleted_label": None},
            "nominated deletion identity",
        ),
        (
            {
                "status": "rejected",
                "accepted": False,
                "trial_classification": "discarded_prune",
                "deleted_index": None,
                "deleted_label": None,
                "trust_radius_after": 0.0625,
                "final_state_fingerprint": "state:mutated",
            },
            "preserve the source fingerprint",
        ),
        (
            {
                "status": "not_executed",
                "trial_executed": False,
                "accepted": None,
                "trial_branch_id": None,
                "trial_classification": None,
            },
            "trial-only fields",
        ),
    ],
)
def test_public_prune_receipt_rejects_contradictory_states(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _accepted_public_prune_receipt(**overrides)


def test_recoverability_pruning_resolves_only_the_active_parent_child() -> None:
    context = _resolve_execution_context(
        _small_hh_problem(),
        _pruning_request(),
    )

    assert context.route.profile == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1
    )
    assert context.route.contract_sha256 == (
        canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract_sha256()
    )
    assert context.route.contract_sha256 == EXPECTED_ROUTE_DIGEST
    settings = context.route.contract["execution_settings"]
    assert {
        key: settings[key] for key in EXPECTED_ACTIVE_PRUNE_SETTINGS
    } == EXPECTED_ACTIVE_PRUNE_SETTINGS
    assert settings["phase1_prune_enabled"] is True
    assert settings["phase1_prune_policy"] == "recoverability_ladder_v1"
    assert settings["phase1_prune_mode"] == "live"
    assert settings["phase1_prune_max_candidates"] == 1
    assert settings["phase1_prune_local_window_size"] == 0
    assert settings["phase1_prune_recovery_trust_radius"] == 0.125
    assert (
        settings["phase1_prune_schur_nomination_route"]
        == "full_logical_fs_trust_delete_refit_v1"
    )
    assert settings["phase1_prune_metric_schur_mu"] == 0.0
    assert settings["phase1_prune_endpoint_overlap_policy"] == "off"
    assert (
        context.route.contract["lineage_authority"][
            "parent_route_profile"
        ]
        == (
            "supported_projected_generalized_source_metric_no_overlap_trust_"
            "full_response_symmetric_cost_no_prune_v1"
        )
    )
    parent_settings = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()[
            "execution_settings"
        ]
    )
    child_settings = (
        canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract()[
            "execution_settings"
        ]
    )
    assert {
        key
        for key in parent_settings.keys() | child_settings.keys()
        if parent_settings.get(key) != child_settings.get(key)
    } == {
        "phase1_prune_enabled",
        "phase1_prune_endpoint_overlap_policy",
        "phase1_prune_local_window_size",
        "phase1_prune_max_candidates",
        "phase1_prune_metric_mu_update_policy",
        "phase1_prune_metric_schur_cost_weighting",
        "phase1_prune_metric_schur_mu",
        "phase1_prune_metric_schur_solve_mode",
        "phase1_prune_mode",
        "phase1_prune_recovery_trust_radius",
        "phase1_prune_schur_nomination_route",
        "phase1_prune_trust_update_policy",
    }


def test_recoverability_pruning_enters_the_direct_controller_runtime() -> None:
    context = _resolve_execution_context(
        _small_hh_problem(),
        _pruning_request(),
    )

    runtime = context.build_default_controller_runtime()
    try:
        assert runtime.context.route_profile == context.route.profile
        assert runtime.context.route_contract_sha256 == (
            context.route.contract_sha256
        )
    finally:
        runtime.close()


def test_three_round_public_run_reports_honest_no_nominee_receipts() -> None:
    result = run_sr_snake(
        _small_hh_problem(),
        _pruning_run_request(3),
    )

    assert result.route.profile == (
        SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_FULL_GEOMETRY_FS_PRUNE_VERIFY_V1
    )
    assert result.route.contract_sha256 == (
        canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_full_geometry_fs_prune_verify_v1_contract_sha256()
    )
    assert result.route.contract_sha256 == EXPECTED_ROUTE_DIGEST
    assert result.route.pruning_policy == "recoverability"
    assert len(result.accepted_transitions) == 3
    prune_receipts = [
        transition.pruning
        for transition in result.accepted_transitions
    ]
    assert all(receipt is not None for receipt in prune_receipts)
    assert [receipt.reason for receipt in prune_receipts if receipt] == [
        "no_mature_old_coordinate",
        "no_mature_old_coordinate",
        "no_feasible_affine_deletion",
    ]
    assert not any(
        receipt.trial_executed
        for receipt in prune_receipts
        if receipt is not None
    )
    assert (
        result.estimator_accounting.all_work.s_alg
        == result.estimator_accounting.winning_lineage.s_alg
    )
    assert len(result.final_state.operators) == 3


def test_controlled_accepted_sibling_projects_deletion_to_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_ladder = adapt_pipeline.recoverability_prune_ladder

    def _controlled_acceptance_ladder(**kwargs: Any) -> Any:
        # Fixture-only: retain the exact measured sibling/refit while widening
        # its energy guard so the accepted projection can be exercised.
        kwargs["max_regression"] = 1.0e6
        kwargs["retained_gain_ratio"] = 0.0
        return original_ladder(**kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "recoverability_prune_ladder",
        _controlled_acceptance_ladder,
    )
    result = run_sr_snake(
        _small_hh_problem(),
        _pruning_run_request(5),
    )

    transition = result.accepted_transitions[-1]
    receipt = transition.pruning
    checkpoint = result.scientific_replay[-1].checkpoint
    assert receipt is not None
    assert receipt.status == "accepted"
    assert receipt.accepted is True
    assert receipt.trial_classification == "committed_prune"
    assert receipt.deleted_index is not None
    assert receipt.source_state_fingerprint != (
        receipt.final_state_fingerprint
    )
    assert receipt.final_state_fingerprint == (
        transition.accepted_state_fingerprint
    )
    assert len(result.final_state.operators) == 4
    assert checkpoint.active_ansatz_depth == 4
    assert checkpoint.ordered_operator_labels == (
        result.final_state.operators
    )
    assert checkpoint.projective_state_fingerprint == (
        result.final_state.projective_state_fingerprint
    )
    assert (
        result.estimator_accounting.all_work.s_alg
        == result.estimator_accounting.winning_lineage.s_alg
    )


def test_five_round_public_run_accounts_one_rejected_sibling() -> None:
    result = run_sr_snake(
        _small_hh_problem(),
        _pruning_run_request(5),
    )

    measured = [
        (transition, transition.pruning)
        for transition in result.accepted_transitions
        if (
            transition.pruning is not None
            and transition.pruning.trial_executed
        )
    ]
    assert len(measured) == 1
    transition, receipt = measured[0]
    assert receipt is not None
    assert transition.controller_round == 5
    assert receipt.status == "rejected"
    assert receipt.accepted is False
    assert receipt.trial_classification == "discarded_prune"
    assert receipt.trial_s_alg == 103
    assert receipt.trust_radius_before == pytest.approx(0.125)
    assert receipt.trust_radius_after == pytest.approx(0.0625)
    assert receipt.endpoint_overlap_query_charge == 0
    assert receipt.metric_damping == 0.0
    assert receipt.terminal_prune_active is False
    assert receipt.surrogate_used_for_acceptance is False
    assert receipt.measured_energy_before == pytest.approx(
        transition.energy_after,
        abs=1.0e-12,
    )
    assert receipt.measured_energy_after is not None
    assert receipt.measured_energy_after > (
        receipt.measured_energy_before + 1.0e-8
    )
    assert receipt.source_state_fingerprint == (
        transition.accepted_state_fingerprint
    )
    assert receipt.final_state_fingerprint == (
        transition.accepted_state_fingerprint
    )
    assert len(result.final_state.operators) == 5
    assert (
        result.estimator_accounting.all_work.s_alg
        - result.estimator_accounting.winning_lineage.s_alg
        == receipt.trial_s_alg
        == 103
    )


def test_prune_trust_radius_continues_across_accepted_rounds() -> None:
    result = run_sr_snake(
        _small_hh_problem(),
        _pruning_run_request(6),
    )

    round_five = result.accepted_transitions[4].pruning
    round_six = result.accepted_transitions[5].pruning
    assert round_five is not None
    assert round_six is not None
    assert round_five.trial_executed is True
    assert round_six.trial_executed is True
    assert round_five.trust_radius_after == pytest.approx(0.0625)
    assert round_six.trust_radius_before == pytest.approx(
        round_five.trust_radius_after
    )
    assert round_six.trust_radius_after == pytest.approx(0.03125)
