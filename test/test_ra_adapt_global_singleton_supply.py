from __future__ import annotations

from typing import Any

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt import (
    GlobalSinglePauliWordCandidateAdapter,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    GLOBAL_SINGLE_PAULI_ADAPTER_ID,
    PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON,
    PHASE_I_VISIBILITY_ALL_EXECUTABLE,
    PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RAAdaptRequest,
    ra_adapt_request_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_ALGORITHM_ID,
    _repaired_route_contract,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.sr_snake import (
    AppendCommutationReducedInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRStopPolicy,
)


def _problem(*, n_ph_max: int = 1) -> Any:
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=n_ph_max,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )


def _request(*, rounds: int = 1) -> RAAdaptRequest:
    return RAAdaptRequest(
        adapter=GlobalSinglePauliWordCandidateAdapter(),
        method=SRMethodPolicy(
            insertion=AppendCommutationReducedInsertion()
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=rounds)
        ),
    )


def test_global_singleton_phase_ii_is_exact_retained_identity_exposure() -> None:
    problem = _problem()
    adapter = GlobalSinglePauliWordCandidateAdapter()
    executable = adapter.executable_pool(problem)
    retained = (
        executable.candidates[7],
        executable.candidates[2],
        executable.candidates[19],
    )

    exposed = adapter.expose_children(retained, problem=problem)

    assert exposed.candidates == retained
    assert [
        candidate.manifest_row() for candidate in exposed.candidates
    ] == [candidate.manifest_row() for candidate in retained]
    assert exposed.receipt.ordered_labels == tuple(
        candidate.label for candidate in retained
    )
    assert exposed.metadata == {
        "exposure_scope": "phase_i_retained_singletons_v1",
        "exposure_policy": (
            PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
        ),
        "phase_i_candidate_supply": (
            PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
        ),
        "phase_i_candidate_visibility": (
            PHASE_I_VISIBILITY_ALL_EXECUTABLE
        ),
        "source_singleton_count": 3,
        "source_singleton_labels": [
            candidate.label for candidate in retained
        ],
        "source_singleton_generator_identities": [
            candidate.generator_identity for candidate in retained
        ],
        "global_executable_pool_sha256": executable.receipt.sha256,
    }


def test_global_singleton_request_roundtrip_and_route_supply_identity() -> None:
    request = _request()
    rehydrated = ra_adapt_request_from_mapping(request.to_dict())

    assert isinstance(
        rehydrated.adapter, GlobalSinglePauliWordCandidateAdapter
    )
    assert isinstance(
        rehydrated.method.insertion,
        AppendCommutationReducedInsertion,
    )
    assert rehydrated.to_dict() == request.to_dict()

    profile_request, profile, contract, digest = (
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=RA_ADAPT_ALGORITHM_ID,
        )
    )
    assert profile_request == profile
    assert profile == (
        "paper_i_ra_adapt__single_pauli_word_v1__"
        "append_commutation_reduced__"
        "global_guarded_singleton_phase_i__identity_phase_ii__"
        "measured_residual_response_v1__"
        "all_phase_resource_weighting_v1__"
        "incremental_active_baseline__exact_guarded_full_response"
    )
    invariants = contract["semantic_invariants"]
    assert {
        "candidate_adapter_id": invariants["candidate_adapter_id"],
        "phase_i_candidate_supply": (
            invariants["phase_i_candidate_supply"]
        ),
        "phase_i_candidate_visibility": (
            invariants["phase_i_candidate_visibility"]
        ),
        "phase_ii_candidate_exposure": (
            invariants["phase_ii_candidate_exposure"]
        ),
    } == {
        "candidate_adapter_id": GLOBAL_SINGLE_PAULI_ADAPTER_ID,
        "phase_i_candidate_supply": (
            PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
        ),
        "phase_i_candidate_visibility": (
            PHASE_I_VISIBILITY_ALL_EXECUTABLE
        ),
        "phase_ii_candidate_exposure": (
            PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
        ),
    }
    assert len(digest) == 64
    assert invariants["phase3_candidate_gain_policy"] == (
        "joint_minus_active_only_supported_trust_v1"
    )
    assert invariants["accepted_refit_initialization_policy"] == (
        "exact_applied_joint_step_guarded_v1"
    )

    protocol = build_resolved_ra_protocol(_problem(), request)
    assert protocol.executable_pool.count == 125
    assert protocol.parent_inventory.count == 89
    assert protocol.lineage_authority["candidate_supply"] == {
        "candidate_adapter_id": GLOBAL_SINGLE_PAULI_ADAPTER_ID,
        "phase_i_candidate_supply": (
            PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
        ),
        "phase_i_candidate_visibility": (
            PHASE_I_VISIBILITY_ALL_EXECUTABLE
        ),
        "phase_ii_candidate_exposure": (
            PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
        ),
    }


def test_global_singleton_qiskit_cost_algorithm_selects_full_transpile_all_phase(
) -> None:
    _profile_request, profile, contract, _digest = (
        _repaired_route_contract(
            _request(),
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=(
                "paper_i_ra_adapt_global_singleton_append_commutation_"
                "reduced_qiskit_transpile_cost_v1"
            ),
        )
    )

    assert profile.endswith(
        "__qiskit_full_ansatz_transpile_cost_all_phases_v1"
    )
    execution = contract["execution_settings"]
    assert execution["phase3_backend_cost_mode"] == "transpile_single_v1"
    assert execution["phase3_backend_name"] == "FakeMarrakesh"
    assert execution["phase3_backend_optimization_level"] == 1
    assert execution["phase3_backend_transpile_seed"] == 7
    assert execution["adapt_parallel_gradient_workers"] == 4
    invariants = contract["semantic_invariants"]
    assert invariants["selector_compile_cost_policy"] == (
        "qiskit_full_trial_ansatz_delta_all_phases_v1"
    )
    assert invariants["selector_compile_cost_phase_reuse"] == (
        "phase_i_once_then_phase_ii_phase_iii_reuse_v1"
    )
    assert (
        "qiskit_full_trial_ansatz_delta_all_phases"
        in contract["lineage_authority"]["only_intended_scientific_changes"]
    )


def test_global_singleton_append_reduced_live_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        "append_commutation_reduced"
        in adapt_pipeline._VALID_INSERTION_MODES
    )
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None
    )

    observed = run_ra_adapt(_problem(), _request())

    assert observed.run.stop.completed_controller_rounds == 1
    assert len(observed.run.accepted_trajectory) == 1
    accepted_round = observed.scientific_receipts[
        "accepted_round_receipts"
    ][0]
    assert len(accepted_round["accepted_candidate_lineage"]) == 1

    population = accepted_round[
        "scored_insertion_position_population"
    ]
    phases = {
        phase["phase"]: phase["records"]
        for phase in population["phases"]
    }
    assert len(phases["phase_i"]) == 125
    assert len(phases["phase_ii"]) == 8
    assert len(phases["phase_iii"]) == 4
    phase_i_ids = {
        record["domain_record_id"] for record in phases["phase_i"]
    }
    phase_ii_ids = {
        record["domain_record_id"] for record in phases["phase_ii"]
    }
    phase_iii_ids = {
        record["domain_record_id"] for record in phases["phase_iii"]
    }
    assert phase_iii_ids <= phase_ii_ids <= phase_i_ids
    assert all(
        record["insertion_position"] == 0
        for records in phases.values()
        for record in records
    )

    reduction = accepted_round["insertion_commutation_reduced"]
    assert reduction["schema"] == (
        "commutation_reduced_insertion_domain_receipt_v1"
    )
    assert reduction["policy"] == "append_commutation_reduced"
    assert reduction["domain_open"] is False
    assert reduction["requested_positions"] == [0]
    assert reduction["collapsed_position_count"] == 0
    assert reduction["candidate_count"] == 125
    assert reduction["retained_representative_count"] == 125
    assert {
        tuple(plan["requested_positions"])
        for plan in reduction["candidate_position_plans"]
    } == {(0,)}
    assert {
        tuple(plan["representative_positions"])
        for plan in reduction["candidate_position_plans"]
    } == {(0,)}
