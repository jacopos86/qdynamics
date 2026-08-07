from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt import (
    RAAdaptRequest,
    SinglePauliWordCandidateAdapter,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.static_adapt.ra_adapt.bundles import (
    BundleMaterializationError,
    PRESERVATION_MEASURED_GATE_ID,
    PRESERVATION_STATIONARY_GATE_ID,
    preservation_execution_gate_contract,
    validate_preservation_execution_gate,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RESOURCE_WEIGHTING_LATE,
    RESOURCE_WEIGHTING_ALL_PHASE,
    _attach_validated_bundle_protocol_authority,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_ALGORITHM_ID,
    RA_ADAPT_LEGACY_ALGORITHM_ID,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.ra_adapt.runtime import _execute_sr_snake
from pipelines.static_adapt.sr_snake import (
    PlateauCommutationInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRRunRequest,
    SRStopPolicy,
    run_sr_snake,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1,
)


_LOCK_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "ra_adapt_singleton_trajectory_nph3.json"
)


def _problem():
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=2.0,
            dv=0.0,
            omega0=1.0,
            g_ep=1.0,
            n_ph_max=3,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        ),
        exact_energy_impl=adapt_pipeline._exact_gs_energy_for_problem,
    )


def _stationary_bundle_protocol(
    *,
    rounds: int,
    active_gradient_policy: str = ACTIVE_GRADIENT_STATIONARY,
    resource_weighting_scope: str = RESOURCE_WEIGHTING_LATE,
):
    problem = _problem()
    request = RAAdaptRequest(
        adapter=SinglePauliWordCandidateAdapter(),
        method=SRMethodPolicy(
            insertion=PlateauCommutationInsertion(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=rounds),
        ),
    )
    cell = bundle_module.BundleCellSpec(
        cell_id="stationary_singleton_runtime_regression",
        stage="validation",
        regime_id="fixture",
        nph=3,
        route_id=bundle_module.ROUTE_SINGLETON_PLATEAU,
        algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
        selector_family="ra_adapt",
        candidate_representation=(
            CANDIDATE_REPRESENTATION_SINGLE_PAULI
        ),
        horizon=rounds,
        source_lock_id="fixture_lock",
    )
    source_lock_refs = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "fixture_lock",
        "cell_source_lock_sha256": "3" * 64,
        "visible_provenance_sha256": "4" * 64,
        "provenance_tracker_sha256": "5" * 64,
        "ed_cutoff_reference_sha256": "6" * 64,
        "resolver_script_sha256": "7" * 64,
    }
    authority_kwargs = {
        "cell": cell,
        "bundle_id": (
            bundle_module.STATIONARY_BUNDLE_ID
            if active_gradient_policy == ACTIVE_GRADIENT_STATIONARY
            else bundle_module.MEASURED_BUNDLE_ID
        ),
        "bundle_manifest_sha256": "8" * 64,
        "source_locks_sha256": "1" * 64,
        "source_lock_refs": source_lock_refs,
        "active_gradient_policy": active_gradient_policy,
        "resource_weighting_scope": resource_weighting_scope,
    }
    materialization_authority = (
        bundle_module._bundle_protocol_materialization_authority(
            **authority_kwargs,
        )
    )
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=materialization_authority,
    )
    bound_authority = (
        bundle_module._bundle_protocol_materialization_authority(
            **authority_kwargs,
            protocol_sha256=protocol.sha256,
        )
    )
    return (
        problem,
        _attach_validated_bundle_protocol_authority(
            protocol,
            bound_authority,
        ),
    )


def test_singleton_plateau_route_replays_the_locked_short_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T13: the historical-compatible plateau route keeps its locked prefix."""

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    locked = json.loads(_LOCK_FIXTURE.read_text(encoding="utf-8"))

    result = _execute_sr_snake(
        _problem(),
        SRRunRequest(
            method=SRMethodPolicy(
                insertion=PlateauCommutationInsertion(),
            ),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=3),
            ),
        ),
    ).result

    assert result.route.profile == (
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V1
    )
    assert len(result.accepted_trajectory) == len(
        locked["accepted_trajectory"]
    )
    for observed, expected in zip(
        result.accepted_trajectory,
        locked["accepted_trajectory"],
        strict=True,
    ):
        assert list(observed.operators) == expected["operators"]
        assert list(observed.insertion_positions) == (
            expected["insertion_positions"]
        )
        assert observed.energy == pytest.approx(
            expected["energy"],
            rel=0.0,
            abs=2.0e-12,
        )

    assert (
        result.estimator_accounting.all_work.components.to_dict()
        == locked["estimator_accounting"]["components"]
    )
    assert (
        result.estimator_accounting.all_work.s_alg
        == locked["estimator_accounting"]["s_alg"]
    )
    contract = preservation_execution_gate_contract(
        active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
    )
    assert contract["gate_id"] == PRESERVATION_MEASURED_GATE_ID
    characterization = contract["generic_route_characterization"]
    assert characterization["fixture_problem_role"] == (
        "u2_g1_route_characterization_only_v1"
    )
    assert characterization["study1_numerical_baseline"] is False


def test_stationary_singleton_staged_children_execute_with_bound_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guarded child admission remains stable across successive RA stages."""

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    problem, protocol = _stationary_bundle_protocol(rounds=2)

    result = run_ra_adapt(problem, protocol)

    assert len(result.accepted_trajectory) == 2
    assert all(
        operator.startswith("guarded_singleton::")
        for trajectory in result.accepted_trajectory
        for operator in trajectory.operators
    )
    assert all(
        generator_id.startswith("child:")
        for generator_id in result.accepted_trajectory[-1].generator_ids
    )
    assert result.policy.active_gradient_indices_acquired == ()
    assert result.policy.active_gradient_charge == 0
    for round_receipt in result.scientific_receipts[
        "accepted_round_receipts"
    ]:
        lineage = round_receipt["accepted_candidate_lineage"]
        assert lineage
        assert all(row["parent_identities"] for row in lineage)


def test_preservation_gates_require_same_problem_replay_and_neutral_pair(
) -> None:
    measured = validate_preservation_execution_gate(
        active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
        generic_t13_characterization_passed=True,
        same_problem_deterministic_replay_passed=True,
        paired_policy_comparison_available=True,
        paired_trajectory_max_abs_deviation=0.0,
        active_gradient_indices_acquired=(0,),
        active_gradient_charge=1,
    )
    assert measured["gate_id"] == PRESERVATION_MEASURED_GATE_ID
    assert measured["status"] == "passed"

    gate = validate_preservation_execution_gate(
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        generic_t13_characterization_passed=True,
        same_problem_deterministic_replay_passed=True,
        paired_policy_comparison_available=True,
        paired_trajectory_max_abs_deviation=0.0,
        active_gradient_indices_acquired=(),
        active_gradient_charge=0,
    )
    assert gate["gate_id"] == PRESERVATION_STATIONARY_GATE_ID
    assert gate["status"] == "passed"
    assert gate["active_gradient_indices_acquired"] == []
    assert gate["active_gradient_charge"] == 0

    with pytest.raises(
        BundleMaterializationError,
        match=PRESERVATION_STATIONARY_GATE_ID,
    ):
        validate_preservation_execution_gate(
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            generic_t13_characterization_passed=True,
            same_problem_deterministic_replay_passed=False,
            paired_policy_comparison_available=True,
            paired_trajectory_max_abs_deviation=1.0e-4,
            active_gradient_indices_acquired=(),
            active_gradient_charge=0,
        )
    with pytest.raises(
        BundleMaterializationError,
        match=PRESERVATION_STATIONARY_GATE_ID,
    ):
        validate_preservation_execution_gate(
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            generic_t13_characterization_passed=True,
            same_problem_deterministic_replay_passed=True,
            paired_policy_comparison_available=True,
            paired_trajectory_max_abs_deviation=0.0,
            active_gradient_indices_acquired=(0,),
            active_gradient_charge=1,
        )
