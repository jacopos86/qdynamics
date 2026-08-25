from __future__ import annotations

import copy
from collections import Counter
from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
import pipelines.static_adapt.ra_adapt.engine as ra_engine
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.adapt_pipeline import (
    _deduplicated_adapter_position_owners,
)
from pipelines.static_adapt.ra_adapt import (
    GlobalSinglePauliWordCandidateAdapter,
    GlobalSingletonGradientPhase0CandidateAdapter,
    MacroCandidateAdapter,
    MacroGradientPhase0CandidateAdapter,
    MacroGradientPhase0ThenSingletonCandidateAdapter,
    MacroThenSingletonPhaseICandidateAdapter,
    RAAdaptOperationalControls,
    RAAdaptRequest,
    RAAdaptResult,
    SinglePauliWordCandidateAdapter,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V2,
    BundleProtocolMaterializationAuthority,
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    EXACT_ORDERED_INSERTION_CHART,
    PROJECTED_GENERALIZED_SOLVER,
    RA_ADAPT_PROTOCOL_SCHEMA_V1,
    RA_ADAPT_PROTOCOL_SCHEMA_V2,
    RA_ADAPT_RESULT_SCHEMA_V1,
    RA_ADAPT_RESULT_SCHEMA_V2,
    RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1,
    RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2,
    RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE,
    RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU,
    RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU,
    RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION,
    RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID,
    RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RESOURCE_WEIGHTING_LATE,
    SOURCE_GRAM_NO_OVERLAP_TRUST,
    ResolvedRAAdaptProtocol,
    _attach_validated_bundle_protocol_authority,
    canonical_sha256,
    ra_adapt_request_from_mapping,
    resolved_ra_adapt_protocol_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_ALGORITHM_ID,
    RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
    RA_ADAPT_LEGACY_ALGORITHM_ID,
    RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
    RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
    RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
    RA_ADAPT_LEGACY_ORDINARY_BUNDLE_ID,
    RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
    _accepted_candidate_lineage_receipts,
    _accepted_round_scientific_receipts,
    _geometry_expansion_trust_limitation,
    _legacy_ordinary_bundle_digest,
    _repaired_route_contract,
    _required_phase3_stabilization,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.ra_adapt.support import RetainedSupportReceipt
from pipelines.static_adapt.sr_snake import (
    AcceptedStateResume,
    AlwaysCommutationReducedInsertion,
    AppendOnlyInsertion,
    CheckpointObservation,
    EstimatorLedgerObservation,
    GreedyBatchAdmission,
    PlateauCommutationInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRRunRequest,
    SRStopPolicy,
    run_sr_snake,
)
from pipelines.static_adapt.sr_snake._resume import (
    load_canonical_accepted_state_resume,
)


EXPECTED_FIRST_SINGLETON_OPERATOR = (
    "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)::"
    "child_set[0]::legal_projected"
)
EXPECTED_FIRST_SINGLETON_GENERATOR = "gen:edc1a5f152a274be"
EXPECTED_PLATEAU_ROUTE_SHA256 = (
    "9d90a88a353f3adcc9373a223c1523564b9cd1c49712232db74e8f63895c8057"
)


def _hh_problem(*, n_ph_max: int = 1) -> Any:
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


def _execution(*, rounds: int) -> SRExecutionPolicy:
    return SRExecutionPolicy(
        stop=SRStopPolicy(maximum_controller_rounds=rounds)
    )


def _validated_protocol(
    problem: Any,
    *,
    rounds: int,
    adapter: Any,
    insertion: Any,
    route_id: str,
    candidate_representation: str,
    algorithm_id: str = RA_ADAPT_LEGACY_ALGORITHM_ID,
    active_gradient_policy: str = ACTIVE_GRADIENT_MEASURED,
    resource_weighting_scope: str = RESOURCE_WEIGHTING_LATE,
    admission: Any | None = None,
) -> ResolvedRAAdaptProtocol:
    request = RAAdaptRequest(
        adapter=adapter,
        method=SRMethodPolicy(
            insertion=insertion,
            **({} if admission is None else {"admission": admission}),
        ),
        execution=_execution(rounds=rounds),
    )
    cell = bundle_module.BundleCellSpec(
        cell_id="operational_resume_fixture",
        stage="validation",
        regime_id="fixture",
        nph=int(problem.request.n_ph_max),
        route_id=route_id,
        algorithm_id=str(algorithm_id),
        selector_family="ra_adapt",
        candidate_representation=candidate_representation,
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
        "active_gradient_policy": str(active_gradient_policy),
        "resource_weighting_scope": str(resource_weighting_scope),
    }
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=(
            bundle_module._bundle_protocol_materialization_authority(
                **authority_kwargs
            )
        ),
    )
    return _attach_validated_bundle_protocol_authority(
        protocol,
        bundle_module._bundle_protocol_materialization_authority(
            **authority_kwargs,
            protocol_sha256=protocol.sha256,
        ),
    )


def _validated_macro_protocol(
    problem: Any,
    *,
    rounds: int,
) -> ResolvedRAAdaptProtocol:
    return _validated_protocol(
        problem,
        rounds=rounds,
        adapter=MacroCandidateAdapter(),
        insertion=AppendOnlyInsertion(),
        route_id=bundle_module.ROUTE_RA_MACRO_APPEND_ONLY,
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
    )


def _validated_singleton_protocol(
    problem: Any,
    *,
    rounds: int,
) -> ResolvedRAAdaptProtocol:
    return _validated_protocol(
        problem,
        rounds=rounds,
        adapter=SinglePauliWordCandidateAdapter(),
        insertion=PlateauCommutationInsertion(),
        route_id=bundle_module.ROUTE_RA_SINGLETON_PLATEAU,
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    )


def _assert_historical_zero_centered_route_fails_closed(
    problem: Any,
    protocol: ResolvedRAAdaptProtocol,
) -> None:
    route = protocol.route_contract
    assert isinstance(route, dict)
    execution = route.get("execution_settings")
    assert isinstance(execution, dict)
    assert execution.get("ra_semantic_implementation_version") is None
    route_sha256 = str(route["sha256"])
    protocol_sha256 = protocol.sha256

    with pytest.raises(RuntimeError, match="historical affected route digests"):
        run_ra_adapt(problem, protocol)

    assert protocol.route_contract is not None
    assert protocol.route_contract["sha256"] == route_sha256
    assert protocol.sha256 == protocol_sha256


def test_historical_macro_then_singleton_phase_i_route_fails_closed() -> None:
    problem = _hh_problem(n_ph_max=1)
    protocol = _validated_protocol(
        problem,
        rounds=1,
        adapter=MacroThenSingletonPhaseICandidateAdapter(),
        insertion=PlateauCommutationInsertion(),
        route_id="ra_macro_then_singleton_phase123_phase23_qiskit",
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        algorithm_id=(
            RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID
        ),
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )

    _assert_historical_zero_centered_route_fails_closed(problem, protocol)


def test_historical_macro_gradient_phase0_route_fails_closed() -> None:
    problem = _hh_problem(n_ph_max=1)
    adapter = MacroGradientPhase0ThenSingletonCandidateAdapter()
    protocol = _validated_protocol(
        problem,
        rounds=1,
        adapter=adapter,
        insertion=PlateauCommutationInsertion(),
        route_id="ra_macro_gradient_phase0_then_singleton_phase123",
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        algorithm_id=(
            RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID
        ),
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )

    _assert_historical_zero_centered_route_fails_closed(problem, protocol)


def test_macro_only_gradient_phase0_adapter_serializes_exactly() -> None:
    request = RAAdaptRequest(adapter=MacroGradientPhase0CandidateAdapter())

    restored = ra_adapt_request_from_mapping(request.to_dict())

    assert type(restored.adapter) is MacroGradientPhase0CandidateAdapter
    assert restored.to_dict() == request.to_dict()
    assert restored.adapter.candidate_representation_id == (
        CANDIDATE_REPRESENTATION_MACRO
    )
    assert restored.adapter.phase_ii_candidate_exposure_id == (
        "identity_on_retained_macro_generators_v1"
    )


def _macro_only_gradient_phase0_protocol(
    problem: Any,
) -> ResolvedRAAdaptProtocol:
    return _validated_protocol(
        problem,
        rounds=1,
        adapter=MacroGradientPhase0CandidateAdapter(),
        insertion=PlateauCommutationInsertion(),
        route_id="ra_macro_gradient_phase0_proxy_no_lanes",
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
        algorithm_id=(
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
        ),
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )


def test_macro_only_gradient_phase0_route_contract_is_proxy_and_lanes_off(
) -> None:
    protocol = _macro_only_gradient_phase0_protocol(
        _hh_problem(n_ph_max=1)
    )
    contract = protocol.route_contract

    assert contract is not None
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]
    assert execution["ra_phase0_gradient_shortlist_policy"] == (
        "standard_adapt_abs_gradient_macro_phase0_v1"
    )
    assert execution["ra_phase0_gradient_shortlist_size"] == 24
    assert execution["phase3_backend_cost_mode"] == (
        "marrakesh_graph_span_v1"
    )
    assert "phase3_backend_cost_scope" not in execution
    assert execution["static_lane_route"] == "global_single_population"
    assert "physical_lane_shortlist_aggressiveness" not in execution
    assert invariants["selector_qiskit_compile_cost_active"] is False
    assert invariants["physical_operator_lanes_active"] is False
    assert invariants["macro_generator_identity_preserved_all_phases"] is True
    assert invariants["singleton_child_exposure_active"] is False
    assert invariants["plateau_prior_mean_decrease_ratio_threshold"] == (
        1.0e-4
    )


def test_macro_only_gradient_phase0_executes_one_bounded_macro_round(
) -> None:
    problem = _hh_problem(n_ph_max=1)
    adapter = MacroGradientPhase0CandidateAdapter()
    result = run_ra_adapt(
        problem,
        _macro_only_gradient_phase0_protocol(problem),
    )

    assert result.protocol.candidate_representation == (
        CANDIDATE_REPRESENTATION_MACRO
    )
    assert len(result.run.accepted_trajectory) == 1
    compile_accounting = result.scientific_receipts[
        "selector_compile_cost_accounting"
    ]
    assert compile_accounting["scope"] == (
        "shared_backend_compile_cost_all_phases_v1"
    )
    assert compile_accounting["phase_i_phase_ii"]["mode"] == (
        "marrakesh_graph_span_v1"
    )
    assert compile_accounting["phase_iii"] is None
    assert compile_accounting["phase0_cost_source"] == (
        "none_standard_adapt_absolute_gradient_v1"
    )
    assert compile_accounting["qiskit_applied_phases"] == []
    assert compile_accounting[
        "phase_iii_reuses_phase_i_phase_ii_oracle"
    ] is True
    assert compile_accounting["excluded_from_s_alg"] is True
    row = result.scientific_receipts["accepted_round_receipts"][0]
    phase0 = row["ra_gradient_phase0_shortlist"]
    assert phase0["policy"] == (
        "standard_adapt_abs_gradient_macro_phase0_v1"
    )
    assert phase0["retained_candidate_count"] == min(
        24,
        phase0["input_candidate_count"],
    )
    assert phase0["input_candidate_count"] == len(
        adapter.executable_pool(problem).candidates
    )
    assert phase0["estimator_accounting"]["components"] == {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": phase0["input_candidate_count"],
        "N_metric": 0,
    }
    macro_labels = {
        str(candidate.label)
        for candidate in adapter.executable_pool(problem).candidates
    }
    retained_indices = set(phase0["retained_pool_indices"])
    phases = row["scored_insertion_position_population"]["phases"]
    assert [phase["phase"] for phase in phases] == [
        "phase_i",
        "phase_ii",
        "phase_iii",
    ]
    assert set(phase0["estimator_event_ids"]).isdisjoint(
        phases[0]["estimator_event_ids"]
    )
    for phase in phases:
        assert {
            int(record["pool_index"]) for record in phase["records"]
        }.issubset(retained_indices)
        assert {
            str(record["pool_label"]) for record in phase["records"]
        }.issubset(macro_labels)


def test_historical_macro_only_gradient_phase0_route_fails_closed() -> None:
    problem = _hh_problem(n_ph_max=1)
    adapter = MacroGradientPhase0CandidateAdapter()
    protocol = _validated_protocol(
        problem,
        rounds=1,
        adapter=adapter,
        insertion=PlateauCommutationInsertion(),
        route_id="ra_macro_gradient_phase0_macro_phase23_qiskit_no_lanes",
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
        algorithm_id=(
            RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID
        ),
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )

    _assert_historical_zero_centered_route_fails_closed(problem, protocol)


def test_historical_global_singleton_gradient_phase0_route_fails_closed() -> None:
    problem = _hh_problem(n_ph_max=1)
    protocol = _validated_protocol(
        problem,
        rounds=1,
        adapter=GlobalSingletonGradientPhase0CandidateAdapter(),
        insertion=PlateauCommutationInsertion(),
        route_id="ra_global_singleton_gradient_phase0_phase123",
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        algorithm_id=(
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID
        ),
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )

    _assert_historical_zero_centered_route_fails_closed(problem, protocol)


@pytest.mark.parametrize(
    (
        "adapter",
        "algorithm_id",
        "candidate_representation",
        "message",
    ),
    (
        (
            GlobalSingletonGradientPhase0CandidateAdapter(),
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
            CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            "initialized-singleton gradient-Phase-0 route requires",
        ),
        (
            MacroGradientPhase0ThenSingletonCandidateAdapter(),
            RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
            CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            "macro gradient-Phase-0 route requires",
        ),
        (
            MacroGradientPhase0CandidateAdapter(),
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
            CANDIDATE_REPRESENTATION_MACRO,
            "macro-only gradient-Phase-0 route requires",
        ),
    ),
)
def test_gradient_phase0_routes_reject_batch_admission(
    adapter: Any,
    algorithm_id: str,
    candidate_representation: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validated_protocol(
            _hh_problem(n_ph_max=1),
            rounds=1,
            adapter=adapter,
            insertion=PlateauCommutationInsertion(),
            route_id="gradient_phase0_batch_must_fail_closed",
            candidate_representation=candidate_representation,
            algorithm_id=algorithm_id,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            admission=GreedyBatchAdmission(maximum_size=2, search_window_size=None),
        )


@pytest.mark.parametrize(
    ("adapter", "algorithm_id"),
    (
        (
            MacroGradientPhase0CandidateAdapter(),
            RA_ADAPT_LEGACY_ALGORITHM_ID,
        ),
        (
            MacroCandidateAdapter(),
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
        ),
    ),
)
def test_macro_only_gradient_phase0_adapter_and_algorithm_are_paired(
    adapter: Any,
    algorithm_id: str,
) -> None:
    request = RAAdaptRequest(
        adapter=adapter,
        method=SRMethodPolicy(insertion=PlateauCommutationInsertion()),
        execution=_execution(rounds=1),
    )

    with pytest.raises(ValueError, match="must be selected together"):
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=algorithm_id,
        )


def _diagnostic_observation(
    tmp_path: Path,
    *,
    stem: str,
) -> SRObservationPolicy:
    return SRObservationPolicy(
        checkpoint=CheckpointObservation(
            path=tmp_path / f"{stem}.current.json",
            every_controller_rounds=1,
            keep_history_tail=10,
        ),
        estimator_ledger=EstimatorLedgerObservation(
            path=tmp_path / f"{stem}.ledger.json"
        ),
    )


def _scientific_signature(result: Any) -> dict[str, Any]:
    return {
        "final_state": result.final_state.to_dict(),
        "accepted_trajectory": [
            receipt.to_dict() for receipt in result.accepted_trajectory
        ],
        "accepted_transitions": [
            receipt.to_dict() for receipt in result.accepted_transitions
        ],
        "problem": result.problem.to_dict(),
        "route": result.route.to_dict(),
        "scientific_replay": [
            receipt.to_dict() for receipt in result.scientific_replay
        ],
        "accounting": result.estimator_accounting.to_dict(),
    }


def _assert_real_ra_scientific_receipts(result: RAAdaptResult) -> None:
    receipts = result.scientific_receipts
    integrity = result.numerical_physical_integrity
    assert integrity.method == "ra_adapt"
    assert integrity.reporting_only is True
    assert integrity.controller_decision_influence is False
    assert integrity.finite_values_passed is True
    assert integrity.sector_leak_flag is False
    assert integrity.boson_truncation_leak_flag is False
    assert integrity.accepted_energy_integrity_passed is True
    assert integrity.integrity_passed is True
    assert len(integrity.accepted_energy_transitions) == len(
        result.run.accepted_transitions
    )
    assert all(
        row.gate_passed
        for row in integrity.accepted_energy_transitions
    )
    assert receipts["numerical_physical_integrity"] == (
        integrity.to_dict()
    )
    assert receipts["numerical_physical_integrity_sha256"] == (
        canonical_sha256(integrity)
    )
    assert result.to_dict()["numerical_physical_integrity"] == (
        integrity.to_dict()
    )
    accepted_rounds = receipts["accepted_round_receipts"]
    assert len(accepted_rounds) == 1
    accepted = accepted_rounds[0]

    support = accepted["retained_support"]
    assert support == receipts["retained_support"]
    assert support["schema"] == "ra_adapt_retained_support_receipt_v1"
    assert support["metric_regularization"] == 0.0
    assert support["rank"] == sum(support["retained_mask"])
    assert support["raw_metric_eigenvalues"]
    assert support["receipt_provenance_id"]

    phase3 = accepted["phase3_stabilization"]
    assert phase3 == receipts["phase3_stabilization"]
    assert phase3["solver_policy"] == PROJECTED_GENERALIZED_SOLVER
    assert phase3["total_metric_multiplier_mu"] == pytest.approx(
        phase3["kappa_stabilization_shift"]
        + phase3["trust_boundary_multiplier_lambda"]
    )
    assert phase3["trust_boundary_active"] is (
        phase3["trust_boundary_multiplier_lambda"] > 0.0
    )
    assert phase3["metric_whitening_active"] is False
    assert phase3["metric_inverse_sqrt_constructed"] is False

    trust = accepted["source_gram_no_overlap_trust"]
    assert trust == receipts["source_gram_no_overlap_trust"]
    assert trust["schema"] == (
        "ra_adapt_source_gram_no_overlap_trust_receipt_v1"
    )
    assert trust["supported_metric_projection_provenance_id"] == (
        support["factorization_provenance_id"]
    )
    assert trust["metric_retained_mask"] == support["retained_mask"]
    assert trust["branch_trust_radius_before"] > 0.0
    assert trust["certified_trust_radius_sq"] > 0.0
    assert trust["endpoint_overlap_required"] is False
    assert trust["endpoint_overlap_query_charge"] == 0
    assert trust["incremental_quantum_query_charge"] == 0

    assert list(accepted).count("accepted_refit_fixed_chart_receipt") == 1
    fixed_chart = accepted["accepted_refit_fixed_chart_receipt"]
    fixed_chart_sha256 = accepted["accepted_refit_fixed_chart_sha256"]
    assert fixed_chart["schema"] == (
        "accepted_refit_fixed_chart_receipt_v1"
    )
    assert fixed_chart["scope"] == "full_ansatz_v1"
    assert fixed_chart["coordinate_chart"] == (
        "supported_fs_whitened_fixed_v1"
    )
    assert fixed_chart["base_chart_policy"] == (
        "expanded_runtime_projected_logical_v1"
    )
    assert fixed_chart["chart_lifetime"] == (
        "fixed_for_one_optimizer_invocation_then_discarded_v1"
    )
    assert fixed_chart["sha256"] == fixed_chart_sha256
    digest_payload = dict(fixed_chart)
    digest_payload.pop("sha256")
    assert canonical_sha256(digest_payload) == fixed_chart_sha256
    serialized_accepted = result.to_dict()["scientific_receipts"][
        "accepted_round_receipts"
    ][0]
    assert serialized_accepted[
        "accepted_refit_fixed_chart_receipt"
    ] == fixed_chart
    assert serialized_accepted[
        "accepted_refit_fixed_chart_sha256"
    ] == fixed_chart_sha256
    json.dumps(serialized_accepted, allow_nan=False, sort_keys=True)

    fallback = receipts[
        "deferred_gram_all_models_infeasible_fallback_v1"
    ]
    assert fallback["schema"] == (
        "deferred_gram_all_models_infeasible_fallback_v1"
    )
    assert fallback["scope"] == "run"
    assert {"enabled", "fired", "rounds", "charge"} <= set(fallback)


def _rebuild_protocol(
    protocol: ResolvedRAAdaptProtocol,
    **updates: Any,
) -> ResolvedRAAdaptProtocol:
    payload = {
        item.name: getattr(protocol, item.name)
        for item in fields(ResolvedRAAdaptProtocol)
        if item.name != "sha256"
        and item.metadata.get("canonical", True)
    }
    payload.update(updates)
    digest_payload = {
        key: value for key, value in payload.items() if value is not None
    }
    return ResolvedRAAdaptProtocol(
        **payload,
        sha256=canonical_sha256(digest_payload),
    )


def test_public_ra_facade_has_no_study_policy_or_route_string_knobs() -> None:
    signature = inspect.signature(run_ra_adapt)
    assert tuple(signature.parameters) == (
        "problem",
        "request",
        "operational_controls",
    )
    assert signature.parameters["request"].default is None
    assert (
        signature.parameters["operational_controls"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["operational_controls"].default is None
    assert tuple(item.name for item in fields(RAAdaptRequest)) == (
        "adapter",
        "method",
        "execution",
        "observation",
    )
    assert {
        "active_gradient_policy",
        "resource_weighting_scope",
        "route_profile",
        "algorithm_id",
        "bundle_id",
    }.isdisjoint(item.name for item in fields(RAAdaptRequest))

    request = RAAdaptRequest()
    assert isinstance(request.adapter, SinglePauliWordCandidateAdapter)
    assert (
        request.adapter.candidate_representation_id
        == CANDIDATE_REPRESENTATION_SINGLE_PAULI
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        RAAdaptRequest(  # type: ignore[call-arg]
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY
        )


def test_bundle_operational_controls_authenticate_a_real_bounded_continuation(
    tmp_path: Path,
) -> None:
    problem = _hh_problem()
    protocol = _validated_macro_protocol(problem, rounds=3)
    protocol_sha256 = protocol.sha256
    first_leg = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=2,
            observation=_diagnostic_observation(
                tmp_path,
                stem="first-leg",
            ),
        ),
    )
    checkpoint_path = tmp_path / "first-leg.current.json"
    checkpoint_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    resumed = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=3,
            resume=AcceptedStateResume(
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
            ),
            observation=_diagnostic_observation(
                tmp_path,
                stem="resumed",
            ),
        ),
    )

    assert protocol.sha256 == protocol_sha256
    assert first_leg.protocol == protocol
    assert resumed.protocol == protocol
    assert first_leg.run.stop.completed_controller_rounds == 2
    assert resumed.run.stop.completed_controller_rounds == 3
    assert [
        row.to_dict() for row in resumed.run.accepted_trajectory[:2]
    ] == [
        row.to_dict() for row in first_leg.run.accepted_trajectory
    ]
    assert len(
        resumed.scientific_receipts[
            "controller_replay_evidence"
        ]["signed_controller_round_prefixes"]
    ) == 3
    with pytest.raises(ValueError, match="only shorten"):
        run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=4
            ),
        )
    with pytest.raises(ValueError, match="bundle-resolved"):
        run_ra_adapt(
            problem,
            RAAdaptRequest(execution=_execution(rounds=3)),
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=2
            ),
        )


def test_v2_macro_always_bundle_resume_preserves_full_response_contract(
    tmp_path: Path,
) -> None:
    problem = _hh_problem()
    protocol = _validated_protocol(
        problem,
        rounds=3,
        adapter=MacroCandidateAdapter(),
        insertion=AlwaysCommutationReducedInsertion(),
        route_id=bundle_module.ROUTE_RA_MACRO_ALWAYS,
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
        algorithm_id=RA_ADAPT_ALGORITHM_ID,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )
    assert protocol.schema == RA_ADAPT_PROTOCOL_SCHEMA_V2
    first_leg = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=2,
            observation=_diagnostic_observation(
                tmp_path,
                stem="v2-macro-always-first-leg",
            ),
        ),
    )
    checkpoint_path = tmp_path / "v2-macro-always-first-leg.current.json"
    checkpoint_payload = json.loads(checkpoint_path.read_text())
    assert checkpoint_payload["settings"]["sr_route_profile_contract"][
        "schema"
    ] == RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2
    checkpoint_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    resumed = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=3,
            resume=AcceptedStateResume(
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
            ),
            observation=_diagnostic_observation(
                tmp_path,
                stem="v2-macro-always-resumed",
            ),
        ),
    )

    assert first_leg.schema == RA_ADAPT_RESULT_SCHEMA_V2
    assert resumed.schema == RA_ADAPT_RESULT_SCHEMA_V2
    assert resumed.run.stop.completed_controller_rounds == 3
    assert all(
        replay.accepted_refit.initialization_policy
        == "exact_applied_joint_step_guarded_v1"
        for replay in resumed.run.scientific_replay
    )
    assert [
        row.to_dict() for row in resumed.run.accepted_trajectory[:2]
    ] == [
        row.to_dict() for row in first_leg.run.accepted_trajectory
    ]


def test_bundle_operational_controls_authenticate_singleton_continuation(
    tmp_path: Path,
) -> None:
    problem = _hh_problem()
    protocol = _validated_singleton_protocol(problem, rounds=3)
    first_leg = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=2,
            observation=_diagnostic_observation(
                tmp_path,
                stem="singleton-first-leg",
            ),
        ),
    )
    checkpoint_path = tmp_path / "singleton-first-leg.current.json"
    checkpoint_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    checkpoint_payload = json.loads(checkpoint_path.read_text())
    history_rows = checkpoint_payload["adapt_vqe"]["history"]
    assert len(history_rows) == 2
    for row in history_rows:
        selected_feature = row["selected_feature_rows"][0]
        physical_generator_id = str(selected_feature["generator_id"])
        assert physical_generator_id == str(
            selected_feature["generator_metadata"]["generator_id"]
        )
        assert "::pool[" not in physical_generator_id
    assert all(
        row["selected_feature_rows"][0][
            "runtime_split_parent_label"
        ]
        == row["selected_feature_rows"][0][
            "physical_operator_classifier_label"
        ]
        for row in history_rows
    )
    assert all(
        (
            owner := row["selected_feature_rows"][0][
                "generator_metadata"
            ]["ra_retained_parent_owner"]
        )["parent_label"]
        == row["selected_feature_rows"][0][
            "runtime_split_parent_label"
        ]
        and owner["parent_generator_identity"]
        == row["selected_feature_rows"][0][
            "parent_generator_id"
        ]
        for row in history_rows
    )
    signed_operator_rows = checkpoint_payload["adapt_vqe"][
        "terminal_active_prefix_checkpoint"
    ]["ordered_active_operators"]
    assert all(
        row["symmetry_gate"]["checked"] is True
        and row["symmetry_gate"]["passed"] is True
        and row["ra_retained_parent_owner"][
            "parent_generator_identity"
        ]
        == row["parent_generator_id"]
        for row in signed_operator_rows
    )
    first_leg_generator_ids = tuple(
        generator_id
        for accepted_state in first_leg.run.accepted_trajectory
        for generator_id in accepted_state.generator_ids
    )
    assert first_leg_generator_ids
    assert all(
        "::pool[" not in generator_id
        for generator_id in first_leg_generator_ids
    )
    terminal_generator_ids = tuple(
        str(row["generator_id"])
        for row in signed_operator_rows
    )
    assert terminal_generator_ids == (
        first_leg.run.accepted_trajectory[-1].generator_ids
    )
    assert all(
        "::pool[" not in generator_id
        for generator_id in terminal_generator_ids
    )
    resumed = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=3,
            resume=AcceptedStateResume(
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
            ),
            observation=_diagnostic_observation(
                tmp_path,
                stem="singleton-resumed",
            ),
        ),
    )

    assert first_leg.run.stop.completed_controller_rounds == 2
    assert resumed.run.stop.completed_controller_rounds == 3
    assert [
        row.to_dict() for row in resumed.run.accepted_trajectory[:2]
    ] == [
        row.to_dict() for row in first_leg.run.accepted_trajectory
    ]
    resumed_generator_ids = tuple(
        generator_id
        for accepted_state in resumed.run.accepted_trajectory
        for generator_id in accepted_state.generator_ids
    )
    assert all(
        "::pool[" not in generator_id
        for generator_id in resumed_generator_ids
    )
    resumed_checkpoint_path = tmp_path / "singleton-resumed.current.json"
    resumed_checkpoint_sha256 = hashlib.sha256(
        resumed_checkpoint_path.read_bytes()
    ).hexdigest()
    round_trip = load_canonical_accepted_state_resume(
        AcceptedStateResume(
            checkpoint_path=resumed_checkpoint_path,
            checkpoint_sha256=resumed_checkpoint_sha256,
        ),
        expected_problem=problem,
        expected_route_profile=resumed.run.route.profile,
        expected_route_contract_sha256=(
            resumed.run.route.contract_sha256
        ),
    )
    assert round_trip.controller_round == 3
    assert tuple(
        operator.label for operator in round_trip.operators
    ) == resumed.run.final_state.operators
    assert tuple(
        operator.generator_id for operator in round_trip.operators
    ) == resumed.run.final_state.generator_ids
    assert all(
        "::pool[" not in operator.generator_id
        for operator in round_trip.operators
    )

    tampered_payload = copy.deepcopy(checkpoint_payload)
    tampered_history = tampered_payload["adapt_vqe"]["history"]
    first_feature = tampered_history[0]["selected_feature_rows"][0]
    alternate_parent = first_feature["generator_metadata"][
        "shared_pauli_pool_contract"
    ]["parent_labels"][0]
    assert alternate_parent != first_feature[
        "runtime_split_parent_label"
    ]
    first_feature["runtime_split_parent_label"] = alternate_parent
    first_feature[
        "physical_operator_classifier_label"
    ] = alternate_parent
    tampered_history[0]["insertion_commutation_plateau"][
        "candidate_position_plans"
    ][0]["candidate_label"] = alternate_parent
    tampered_history[0]["insertion_commutation_plateau"][
        "retained_representatives"
    ][0]["candidate_label"] = alternate_parent
    tampered_payload["adapt_vqe"]["history_tail"] = copy.deepcopy(
        tampered_history
    )
    tampered_path = tmp_path / "singleton-coherent-parent-tamper.json"
    tampered_path.write_text(
        json.dumps(
            tampered_payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="retained-parent owner"):
        run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=3,
                resume=AcceptedStateResume(
                    checkpoint_path=tampered_path,
                    checkpoint_sha256=hashlib.sha256(
                        tampered_path.read_bytes()
                    ).hexdigest(),
                ),
                observation=_diagnostic_observation(
                    tmp_path,
                    stem="singleton-tampered",
                ),
            ),
        )


def test_default_singleton_facade_uses_canonical_full_response_v2() -> None:
    problem = _hh_problem()
    execution = _execution(rounds=1)

    observed = run_ra_adapt(
        problem,
        RAAdaptRequest(execution=execution),
    )
    compatibility = run_sr_snake(
        problem,
        SRRunRequest(execution=execution),
    )

    assert isinstance(observed, RAAdaptResult)
    assert observed.protocol.schema == RA_ADAPT_PROTOCOL_SCHEMA_V2
    assert observed.schema == RA_ADAPT_RESULT_SCHEMA_V2
    assert observed.protocol.route_contract is not None
    assert observed.protocol.route_contract["schema"] == (
        RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2
    )
    assert resolved_ra_adapt_protocol_from_mapping(
        observed.protocol.to_dict()
    ) == observed.protocol
    with pytest.raises(ValueError, match="Unknown RA result schema"):
        replace(observed, schema=RA_ADAPT_RESULT_SCHEMA_V1)
    assert _scientific_signature(observed.run) == _scientific_signature(
        compatibility
    )
    assert observed.protocol.algorithm_id == RA_ADAPT_ALGORITHM_ID
    assert observed.run.route.contract_sha256 != (
        EXPECTED_PLATEAU_ROUTE_SHA256
    )
    _, _, route_contract, route_sha256 = _repaired_route_contract(
        observed.protocol.request,
        active_gradient_policy=observed.protocol.active_gradient_policy,
        resource_weighting_scope=(
            observed.protocol.resource_weighting_scope
        ),
        algorithm_id=observed.protocol.algorithm_id,
    )
    assert route_sha256 == observed.run.route.contract_sha256
    route_execution = route_contract["execution_settings"]
    assert route_execution["ra_phase3_candidate_gain_policy"] == (
        "joint_minus_active_only_supported_trust_v1"
    )
    assert route_execution[
        "ra_accepted_refit_initialization_policy"
    ] == "exact_applied_joint_step_guarded_v1"
    assert len(observed.run.final_state.operators) == 1
    assert observed.run.accepted_transitions[0].generator_id
    assert observed.run.accepted_transitions[0].cumulative_s_alg == (
        observed.run.estimator_accounting.all_work.s_alg
    )
    assert observed.protocol.active_gradient_policy == (
        ACTIVE_GRADIENT_MEASURED
    )
    assert observed.protocol.resource_weighting_scope == (
        RESOURCE_WEIGHTING_ALL_PHASE
    )
    assert observed.policy.phase3_candidate_gain_policy == (
        "joint_minus_active_only_supported_trust_v1"
    )
    assert observed.policy.accepted_refit_initialization_policy == (
        "exact_applied_joint_step_guarded_v1"
    )
    typed_refit = observed.run.scientific_replay[0].accepted_refit
    assert typed_refit.initialization_policy == (
        "exact_applied_joint_step_guarded_v1"
    )
    assert typed_refit.initialization_status in {"accepted", "rejected"}
    assert typed_refit.initialization_guard_nfev == 1
    drifted_typed_refit = replace(
        typed_refit,
        initialization_policy="off",
    )
    drifted_replay = replace(
        observed.run.scientific_replay[0],
        accepted_refit=drifted_typed_refit,
    )
    drifted_run = replace(
        observed.run,
        scientific_replay=(drifted_replay,),
    )
    with pytest.raises(
        ValueError,
        match="typed accepted-refit initialization receipt drifted",
    ):
        replace(observed, run=drifted_run)
    _assert_real_ra_scientific_receipts(observed)
    assert observed.scientific_receipts[
        "candidate_inventory_lineage"
    ]["count"] == observed.protocol.executable_pool.count
    accepted_lineage = observed.scientific_receipts[
        "accepted_round_receipts"
    ][0]["accepted_candidate_lineage"]
    control_round = observed.scientific_receipts[
        "accepted_round_receipts"
    ][0]
    assert "phase3_population_activation" not in control_round
    assert "projected_phase3_population_receipt" not in control_round
    assert len(accepted_lineage) == 1
    selected = accepted_lineage[0]
    transition = observed.run.accepted_transitions[0]
    assert selected["representation_id"] == (
        CANDIDATE_REPRESENTATION_SINGLE_PAULI
    )
    assert selected["candidate_label"] == (
        observed.run.final_state.operators[0]
    )
    assert selected["generator_identity"] == transition.generator_id
    assert selected["parent_identities"]
    assert selected["insertion_position"] == transition.insertion_position
    assert len(selected["candidate_manifest_sha256"]) == 64
    with pytest.raises(ValueError, match="result policies drifted"):
        replace(
            observed,
            policy=replace(
                observed.policy,
                phase3_candidate_gain_policy=None,
                accepted_refit_initialization_policy=None,
            ),
        )
    drifted_scientific_receipts = dict(observed.scientific_receipts)
    drifted_scientific_receipts[
        "phase3_candidate_gain_policy"
    ] = "joint_total_gain_v1"
    with pytest.raises(ValueError, match="result policies drifted"):
        replace(
            observed,
            scientific_receipts=drifted_scientific_receipts,
        )


def test_historical_preserved_single_word_lineage_is_inventory_authenticated(
) -> None:
    problem = _hh_problem()
    inventory = SinglePauliWordCandidateAdapter().executable_pool(problem)
    candidate = next(
        row
        for row in inventory.candidates
        if row.label == "hh_termwise_ham_quadrature_term(yezeee)"
    )
    accepted_row = {
        "selected_logical_size": 1,
        "selected_positions": [3],
        "selected_feature_rows": [
            {
                "candidate_label": candidate.label,
                "candidate_family": candidate.family_id,
                "stage_name": "core",
                "position_id": 3,
                "generator_metadata": {
                    "candidate_label": candidate.label,
                    "generator_id": "gen:517e8dfb60b9efce",
                    "parent_generator_id": None,
                    "family_id": candidate.family_id,
                    "template_id": candidate.construction,
                    "is_macro_generator": False,
                    "split_policy": "preserve",
                    "compile_metadata": {
                        "serialized_terms_exyz": [
                            dict(term)
                            for term in candidate.serialized_terms_exyz
                        ],
                    },
                    "symmetry_spec": dict(candidate.symmetry_receipt or {}),
                },
            }
        ],
    }

    receipts = _accepted_candidate_lineage_receipts(
        accepted_row,
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        executable_inventory=inventory,
    )
    assert len(receipts) == 1
    assert receipts[0].candidate_label == candidate.label
    assert receipts[0].generator_identity == "gen:517e8dfb60b9efce"
    assert receipts[0].parent_identities == ()
    assert receipts[0].insertion_position == 3
    assert len(str(receipts[0].candidate_manifest_sha256)) == 64

    tampered_row = json.loads(json.dumps(accepted_row))
    tampered_row["selected_feature_rows"][0]["generator_metadata"][
        "compile_metadata"
    ]["serialized_terms_exyz"][0]["coeff_re"] = 9.0
    with pytest.raises(
        RuntimeError,
        match="authenticated executable inventory",
    ):
        _accepted_candidate_lineage_receipts(
            tampered_row,
            candidate_representation=(
                CANDIDATE_REPRESENTATION_SINGLE_PAULI
            ),
            executable_inventory=inventory,
        )

    malformed_root = json.loads(json.dumps(accepted_row))
    malformed_root["selected_feature_rows"][0]["generator_metadata"][
        "split_policy"
    ] = "runtime_split_projected_child"
    with pytest.raises(
        RuntimeError,
        match="authenticated executable inventory",
    ):
        _accepted_candidate_lineage_receipts(
            malformed_root,
            candidate_representation=(
                CANDIDATE_REPRESENTATION_SINGLE_PAULI
            ),
            executable_inventory=inventory,
        )

    macro_masquerade = json.loads(json.dumps(accepted_row))
    macro_masquerade["selected_feature_rows"][0][
        "generator_metadata"
    ]["is_macro_generator"] = True
    with pytest.raises(
        RuntimeError,
        match="authenticated executable inventory",
    ):
        _accepted_candidate_lineage_receipts(
            macro_masquerade,
            candidate_representation=(
                CANDIDATE_REPRESENTATION_SINGLE_PAULI
            ),
            executable_inventory=inventory,
        )


def test_staged_singleton_lineage_rejects_a_coherently_tampered_manifest(
) -> None:
    problem = _hh_problem(n_ph_max=3)
    adapter = SinglePauliWordCandidateAdapter()
    inventory = adapter.executable_pool(problem)
    global_pool = adapter.global_executable_pool(problem)
    global_target = next(
        candidate
        for candidate in global_pool.candidates
        if candidate.label == "guarded_singleton::eeeeeexy"
    )
    parent_by_identity = {
        candidate.generator_identity: candidate
        for candidate in inventory.candidates
    }
    staged = adapter.expose_children(
        tuple(
            parent_by_identity[parent_identity]
            for parent_identity in global_target.parent_identities
        ),
        problem=problem,
    )
    candidate = next(
        row for row in staged.candidates
        if row.label == global_target.label
    )
    manifest = candidate.manifest_row()
    metadata = {
        **dict(candidate.generator_metadata),
        "generator_id": candidate.generator_identity,
        "ra_candidate_representation": (
            candidate.representation_id
        ),
        "ra_parent_generator_ids": list(
            candidate.parent_identities
        ),
        "ra_candidate_manifest": copy.deepcopy(manifest),
        "ra_candidate_manifest_sha256": canonical_sha256(
            manifest
        ),
    }
    accepted_row = {
        "selected_logical_size": 1,
        "selected_positions": [0],
        "selected_feature_rows": [
            {
                "candidate_label": candidate.label,
                "candidate_family": candidate.family_id,
                "stage_name": candidate.stage_family,
                "position_id": 0,
                "generator_metadata": metadata,
            }
        ],
    }

    receipts = _accepted_candidate_lineage_receipts(
        accepted_row,
        candidate_representation=(
            CANDIDATE_REPRESENTATION_SINGLE_PAULI
        ),
        executable_inventory=inventory,
    )
    assert receipts[0].generator_identity == (
        candidate.generator_identity
    )

    tampered = copy.deepcopy(accepted_row)
    tampered_manifest = tampered["selected_feature_rows"][0][
        "generator_metadata"
    ]["ra_candidate_manifest"]
    tampered_manifest["serialized_terms_exyz"][0][
        "coeff_re"
    ] = 9.0
    tampered["selected_feature_rows"][0]["generator_metadata"][
        "ra_candidate_manifest_sha256"
    ] = canonical_sha256(tampered_manifest)
    with pytest.raises(
        RuntimeError,
        match="executed guarded-child metadata",
    ):
        _accepted_candidate_lineage_receipts(
            tampered,
            candidate_representation=(
                CANDIDATE_REPRESENTATION_SINGLE_PAULI
            ),
            executable_inventory=inventory,
        )


def test_study_policies_require_a_source_locked_nonexecuting_protocol() -> None:
    problem = _hh_problem()
    request = RAAdaptRequest(execution=_execution(rounds=2))

    resolver = inspect.signature(build_resolved_ra_protocol)
    assert tuple(resolver.parameters) == (
        "problem",
        "request",
        "materialization_authority",
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        build_resolved_ra_protocol(
            problem,
            request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
        )
    with pytest.raises(TypeError, match="minted only"):
        BundleProtocolMaterializationAuthority()


def test_full_response_bundle_authority_uses_v2_materialization_namespace() -> None:
    problem = _hh_problem()
    request = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(
            insertion=AlwaysCommutationReducedInsertion()
        ),
        execution=_execution(rounds=1),
    )
    cell = bundle_module.BundleCellSpec(
        cell_id="full_response_v2_fixture",
        stage="validation",
        regime_id="fixture",
        nph=1,
        route_id="macro_always",
        algorithm_id=RA_ADAPT_ALGORITHM_ID,
        selector_family="ra_adapt",
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
        horizon=1,
        source_lock_id="fixture_lock",
    )
    refs = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "fixture_lock",
        "cell_source_lock_sha256": "3" * 64,
        "visible_provenance_sha256": "4" * 64,
        "provenance_tracker_sha256": "5" * 64,
        "ed_cutoff_reference_sha256": "6" * 64,
        "resolver_script_sha256": "7" * 64,
    }
    authority = bundle_module._bundle_protocol_materialization_authority(
        cell=cell,
        bundle_id="full_response_v2_fixture_bundle",
        bundle_manifest_sha256="8" * 64,
        source_locks_sha256="1" * 64,
        source_lock_refs=refs,
        active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )

    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=authority,
    )

    assert authority.receipt.schema == (
        BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V2
    )
    assert authority.receipt.protocol_schema == RA_ADAPT_PROTOCOL_SCHEMA_V2
    assert protocol.schema == RA_ADAPT_PROTOCOL_SCHEMA_V2
    assert protocol.route_contract is not None
    assert protocol.route_contract["schema"] == (
        RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2
    )


def test_full_response_v2_is_algorithm_gated_and_v1_remains_legacy() -> None:
    request = RAAdaptRequest(execution=_execution(rounds=1))

    with pytest.raises(ValueError, match="requires measured active response"):
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=RA_ADAPT_ALGORITHM_ID,
        )

    _, _, legacy_contract, _ = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
    )
    legacy_execution = legacy_contract["execution_settings"]
    legacy_invariants = legacy_contract["semantic_invariants"]
    assert legacy_contract["schema"] == RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1
    assert "ra_phase3_candidate_gain_policy" not in legacy_execution
    assert (
        "ra_accepted_refit_initialization_policy"
        not in legacy_execution
    )
    assert "phase3_candidate_gain_policy" not in legacy_invariants
    assert (
        "accepted_refit_initialization_policy"
        not in legacy_invariants
    )


def test_singleton_phase3_plateau_route_is_named_and_isolated() -> None:
    request = RAAdaptRequest(
        adapter=SinglePauliWordCandidateAdapter(),
        method=SRMethodPolicy(
            insertion=PlateauCommutationInsertion()
        ),
        execution=_execution(rounds=3),
    )
    _requested, profile, contract, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
        algorithm_id=RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
    )
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]
    assert profile.endswith(
        "__phase3_population_on_insertion_plateau_v1"
    )
    assert digest == canonical_sha256(contract)
    assert execution["adapt_insertion_mode"] == (
        "insertion_commutation_plateau_v2"
    )
    assert execution["ra_phase3_population_activation_policy"] == (
        RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU
    )
    assert execution["ra_phase3_preplateau_materialization_policy"] == (
        RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
    )
    assert invariants["phase1_activation_scope"] == (
        "all_controller_rounds_v1"
    )
    assert invariants["phase2_activation_scope"] == (
        "all_controller_rounds_v1"
    )
    assert invariants["phase3_preplateau_admission_authority"] == (
        "phase2_raw_score_top_rank_v1"
    )
    assert invariants["phase3_activation_independent_latch"] is False
    assert invariants["phase3_activation_hysteresis_active"] is False

    _r, _p, control, _d = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
        algorithm_id=(
            "paper_i_ra_adapt_singleton_plateau_insertion_repair_v1"
        ),
    )
    assert "ra_phase3_population_activation_policy" not in (
        control["execution_settings"]
    )
    assert not str(control["route_profile"]).endswith(
        "__phase3_population_on_insertion_plateau_v1"
    )

    invalid_requests = (
        replace(request, adapter=MacroCandidateAdapter()),
        replace(request, adapter=GlobalSinglePauliWordCandidateAdapter()),
        replace(
            request,
            method=replace(
                request.method,
                insertion=AppendOnlyInsertion(),
            ),
        ),
        replace(
            request,
            method=replace(
                request.method,
                insertion=AlwaysCommutationReducedInsertion(),
            ),
        ),
    )
    for invalid in invalid_requests:
        with pytest.raises(ValueError, match="requires the Paper-I"):
            _repaired_route_contract(
                invalid,
                active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
                resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
                algorithm_id=(
                    RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID
                ),
            )
    with pytest.raises(ValueError, match="requires the Paper-I"):
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
            algorithm_id=RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
        )


def test_singleton_phase3_plateau_route_fixes_phase2_winner_until_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD",
        100.0,
    )
    transaction = adapt_pipeline._DefaultNoPruneSelectionTransaction
    phase_i_rounds: list[int] = []
    phase_ii_rounds: list[int] = []
    phase3_input_counts: list[int] = []
    supported_input_counts: list[int] = []
    original_phase_i = transaction.run_phase_i
    original_phase_ii = transaction.run_phase_ii
    original_phase3 = transaction.run_projected_phase_iii
    original_supported = transaction.run_supported_response

    def wrapped_phase_i(self: Any, *args: Any, **kwargs: Any) -> Any:
        phase_i_rounds.append(int(self.pending.depth))
        return original_phase_i(self, *args, **kwargs)

    def wrapped_phase_ii(self: Any, *args: Any, **kwargs: Any) -> Any:
        phase_ii_rounds.append(int(self.pending.depth))
        return original_phase_ii(self, *args, **kwargs)

    def wrapped_phase3(self: Any, *args: Any, **kwargs: Any) -> Any:
        phase3_input_counts.append(
            len(kwargs["phase2_shortlisted_records"])
        )
        return original_phase3(self, *args, **kwargs)

    def wrapped_supported(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        supported_input_counts.append(
            len(kwargs["phase3_measurement_input_records"])
        )
        return original_supported(self, *args, **kwargs)

    monkeypatch.setattr(transaction, "run_phase_i", wrapped_phase_i)
    monkeypatch.setattr(transaction, "run_phase_ii", wrapped_phase_ii)
    monkeypatch.setattr(
        transaction,
        "run_projected_phase_iii",
        wrapped_phase3,
    )
    monkeypatch.setattr(
        transaction,
        "run_supported_response",
        wrapped_supported,
    )

    problem = _hh_problem()
    protocol = _validated_protocol(
        problem,
        rounds=3,
        adapter=SinglePauliWordCandidateAdapter(),
        insertion=PlateauCommutationInsertion(),
        route_id=bundle_module.ROUTE_RA_SINGLETON_PLATEAU,
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        algorithm_id=RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
    )
    result = run_ra_adapt(problem, protocol)

    assert phase_i_rounds == [0, 1, 2]
    assert phase_ii_rounds == [0, 1, 2]
    assert phase3_input_counts[:2] == [1, 1]
    assert phase3_input_counts[2] > 1
    assert supported_input_counts == phase3_input_counts
    accepted = result.scientific_receipts["accepted_round_receipts"]
    activation = [
        row["phase3_population_activation"] for row in accepted
    ]
    assert [
        row["competitive_population_live"] for row in activation
    ] == [False, False, True]
    assert [
        row["insertion_plateau_domain_open"] for row in activation
    ] == [False, False, True]
    assert activation[0]["preplateau_admission_authority"] == (
        "phase2_raw_score_top_rank_v1"
    )
    assert activation[0]["winner_materialization_policy"] == (
        RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
    )
    assert activation[2]["preplateau_admission_authority"] is None
    projected = [
        row["projected_phase3_population_receipt"] for row in accepted
    ]
    assert [
        row["competitive_population_input_count"] for row in projected
    ] == phase3_input_counts
    assert projected[0]["phase2_available_shortlist_count"] > 1


def test_singleton_latched_phase3_route_is_a_direct_page8_derivative() -> None:
    request = RAAdaptRequest(
        adapter=SinglePauliWordCandidateAdapter(),
        method=SRMethodPolicy(
            insertion=PlateauCommutationInsertion()
        ),
        execution=_execution(rounds=5),
    )
    _old_request, old_profile, _old_contract, old_digest = (
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
            algorithm_id=RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
        )
    )
    _requested, profile, contract, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
        algorithm_id=RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID,
    )
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]
    lineage = contract["lineage_authority"]
    assert profile.endswith(
        "__phase3_population_latched_on_progress_plateau_v1"
        "__insertion_on_phase3_plateau_v1"
    )
    assert digest == canonical_sha256(contract)
    assert execution["ra_phase3_population_activation_policy"] == (
        RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU
    )
    assert execution["ra_insertion_plateau_history_scope"] == (
        RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE
    )
    assert invariants["phase3_activation_independent_latch"] is True
    assert invariants["phase3_activation_hysteresis_active"] is False
    assert invariants["phase3_latch_retirement_policy"] == "never_close_v1"
    assert invariants["insertion_activation_requires_prior_phase3_latch"] is True
    assert invariants["insertion_activation_changes_phase3_latch"] is False
    assert lineage["parent_route_profile"] == old_profile
    assert lineage["parent_contract_sha256"] == old_digest
    assert lineage["only_intended_scientific_changes"][-2:] == [
        "phase3_competitive_population_first_open_progress_plateau_latched",
        "commutation_reduced_insertion_requires_prior_full_phase3_plateau_transition",
    ]


def test_singleton_latched_phase3_stays_open_and_defers_insertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    raw_open_by_history_count = {
        0: False,
        1: False,
        2: True,
        3: False,
        4: True,
    }
    original_plateau = (
        adapt_pipeline._insertion_commutation_plateau_round_policy
    )

    def forced_plateau(
        *,
        history: Any,
        policy: str,
        controller_noise_energy_source: bool,
    ) -> dict[str, Any]:
        assert controller_noise_energy_source is False
        receipt = original_plateau(
            history=history,
            policy=policy,
            controller_noise_energy_source=controller_noise_energy_source,
        )
        domain_open = raw_open_by_history_count[len(history)]
        return {
            **receipt,
            "domain_state": "open" if domain_open else "closed",
            "domain_open": domain_open,
            "effective_insertion_mode": (
                "full_commutation_reduced" if domain_open else "append_only"
            ),
        }

    monkeypatch.setattr(
        adapt_pipeline,
        "_insertion_commutation_plateau_round_policy",
        forced_plateau,
    )
    transaction = adapt_pipeline._DefaultNoPruneSelectionTransaction
    phase3_input_counts: list[int] = []
    original_phase3 = transaction.run_projected_phase_iii

    def wrapped_phase3(self: Any, *args: Any, **kwargs: Any) -> Any:
        phase3_input_counts.append(
            len(kwargs["phase2_shortlisted_records"])
        )
        return original_phase3(self, *args, **kwargs)

    monkeypatch.setattr(
        transaction,
        "run_projected_phase_iii",
        wrapped_phase3,
    )
    problem = _hh_problem()
    protocol = _validated_protocol(
        problem,
        rounds=5,
        adapter=SinglePauliWordCandidateAdapter(),
        insertion=PlateauCommutationInsertion(),
        route_id=bundle_module.ROUTE_RA_SINGLETON_PLATEAU,
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        algorithm_id=RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
    )
    result = run_ra_adapt(problem, protocol)
    accepted = result.scientific_receipts["accepted_round_receipts"]
    activation = [
        row["phase3_population_activation"] for row in accepted
    ]
    insertion = [
        row["insertion_commutation_plateau"] for row in accepted
    ]
    assert [
        row["competitive_population_live"] for row in activation
    ] == [False, False, True, True, True]
    assert [
        row["phase3_latch_opened_this_round"] for row in activation
    ] == [False, False, True, False, False]
    assert [row["domain_open"] for row in insertion] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert phase3_input_counts[:2] == [1, 1]
    assert all(count > 1 for count in phase3_input_counts[2:])
    assert activation[2]["entry_plateau_domain_open"] is True
    assert activation[2]["insertion_plateau_domain_open"] is False
    assert insertion[2]["raw_progress_plateau_domain_open"] is True
    assert insertion[2]["preceding_full_phase3_transition_eligible"] is False
    assert activation[3]["entry_plateau_domain_open"] is False
    assert activation[3]["competitive_population_live"] is True
    assert insertion[4]["insertion_trigger_scope"] == (
        RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE
    )


def test_protocol_schema_rejects_rehashed_legacy_algorithm_as_v2() -> None:
    problem = _hh_problem()
    protocol = build_resolved_ra_protocol(
        problem,
        RAAdaptRequest(execution=_execution(rounds=1)),
    )
    with pytest.raises(
        ValueError,
        match="algorithm and protocol schema must use the same version",
    ):
        _rebuild_protocol(
            protocol,
            algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
        )
    with pytest.raises(
        ValueError,
        match="algorithm and protocol schema must use the same version",
    ):
        _rebuild_protocol(
            protocol,
            schema=RA_ADAPT_PROTOCOL_SCHEMA_V1,
        )


@pytest.mark.parametrize(
    "drift_kind",
    (
        "compile_identity",
        "stopping_rule",
        "parent_route_profile",
        "parent_contract_sha256",
        "bound_route",
    ),
)
def test_ordinary_v2_rejects_coherently_rehashed_protocol_drift_before_execution(
    drift_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _hh_problem()
    protocol = build_resolved_ra_protocol(
        problem,
        RAAdaptRequest(execution=_execution(rounds=1)),
    )
    updates: dict[str, Any]
    if drift_kind == "compile_identity":
        compile_identity = dict(protocol.compile_identity)
        compile_identity["optimization_level"] = 3
        updates = {"compile_identity": compile_identity}
    elif drift_kind == "stopping_rule":
        stopping_rule = dict(protocol.stopping_rule)
        stopping_rule["maximum_controller_rounds"] = 2
        updates = {"stopping_rule": stopping_rule}
    elif drift_kind in {
        "parent_route_profile",
        "parent_contract_sha256",
    }:
        lineage = dict(protocol.lineage_authority)
        lineage[drift_kind] = (
            "forged_parent_route"
            if drift_kind == "parent_route_profile"
            else "f" * 64
        )
        updates = {"lineage_authority": lineage}
    else:
        route = dict(protocol.route_contract or {})
        route["route_profile"] = str(route["route_profile"]) + "__forged"
        route_payload = dict(route)
        route_payload.pop("sha256", None)
        route["sha256"] = canonical_sha256(route_payload)
        updates = {"route_contract": route}
    forged = _rebuild_protocol(protocol, **updates)

    monkeypatch.setattr(
        ra_engine,
        "_execute_resolved_context",
        lambda *_args, **_kwargs: pytest.fail(
            "coherently rehashed v2 protocol reached numerical execution"
        ),
    )
    with pytest.raises(
        ValueError,
        match="protocol drifted from deterministic resolution",
    ):
        run_ra_adapt(problem, forged)


def test_v2_bound_route_is_rechecked_at_time_of_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _hh_problem()
    protocol = build_resolved_ra_protocol(
        problem,
        RAAdaptRequest(execution=_execution(rounds=1)),
    )
    original = ra_engine._repaired_route_contract
    call_count = 0

    def _drift_on_execution(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        resolved = original(*args, **kwargs)
        if call_count == 1:
            return resolved
        request_profile, profile, contract, _sha256 = resolved
        drifted = copy.deepcopy(contract)
        drifted["route_profile"] = str(profile) + "__time_of_use_drift"
        return (
            request_profile,
            str(drifted["route_profile"]),
            drifted,
            ra_engine._route_sha256(drifted),
        )

    monkeypatch.setattr(
        ra_engine,
        "_repaired_route_contract",
        _drift_on_execution,
    )
    monkeypatch.setattr(
        ra_engine,
        "_execute_resolved_context",
        lambda *_args, **_kwargs: pytest.fail(
            "time-of-use route drift reached numerical execution"
        ),
    )

    with pytest.raises(ValueError, match="bound route drifted"):
        run_ra_adapt(problem, protocol)


def test_frozen_ordinary_v1_protocol_remains_executable() -> None:
    problem = _hh_problem()
    protocol = build_resolved_ra_protocol(
        problem,
        RAAdaptRequest(execution=_execution(rounds=1)),
    )
    lineage = dict(protocol.lineage_authority)
    lineage.pop("algorithm_semantics", None)
    frozen_v1 = _rebuild_protocol(
        protocol,
        schema=RA_ADAPT_PROTOCOL_SCHEMA_V1,
        algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
        bundle_id=RA_ADAPT_LEGACY_ORDINARY_BUNDLE_ID,
        bundle_manifest_sha256=_legacy_ordinary_bundle_digest(),
        lineage_authority=lineage,
        route_contract=None,
    )

    observed = run_ra_adapt(problem, frozen_v1)

    assert observed.protocol.algorithm_id == RA_ADAPT_LEGACY_ALGORITHM_ID
    assert observed.protocol.schema == RA_ADAPT_PROTOCOL_SCHEMA_V1
    assert observed.schema == RA_ADAPT_RESULT_SCHEMA_V1
    assert observed.run.route.contract_sha256 == (
        EXPECTED_PLATEAU_ROUTE_SHA256
    )
    assert observed.policy.phase3_candidate_gain_policy is None
    assert observed.policy.accepted_refit_initialization_policy is None
    assert resolved_ra_adapt_protocol_from_mapping(
        frozen_v1.to_dict()
    ) == frozen_v1


def test_v2_protocol_rejects_contradictory_algorithm_semantics() -> None:
    protocol = build_resolved_ra_protocol(
        _hh_problem(),
        RAAdaptRequest(execution=_execution(rounds=1)),
    )
    lineage = dict(protocol.lineage_authority)
    lineage["algorithm_semantics"] = {
        "active_response": ACTIVE_GRADIENT_MEASURED,
        "candidate_gain_policy": "joint_total_gain_v1",
        "accepted_refit_initialization_policy": "off",
        "full_response_coordinate_scope": "new_coordinate_only_v1",
    }

    with pytest.raises(ValueError, match="algorithm semantics drifted"):
        _rebuild_protocol(protocol, lineage_authority=lineage)


def test_v2_route_validator_rejects_stripped_gain_and_seed_policies() -> None:
    request = RAAdaptRequest(execution=_execution(rounds=1))
    profile_request, profile, contract, _digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=RA_ADAPT_ALGORITHM_ID,
    )
    assert contract["schema"] == RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2
    genuine_kwargs = {
        "sr_route_profile_contract": contract,
        "sr_route_profile_resolved": profile,
        "sr_route_profile_request": profile_request,
        "sr_route_profile_contract_sha256": ra_engine._route_sha256(
            contract
        ),
    }
    assert adapt_pipeline._validated_ra_adapt_route_contract(
        genuine_kwargs
    ) == contract

    downgraded = copy.deepcopy(contract)
    downgraded["schema"] = RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1
    with pytest.raises(ValueError, match="full-response algorithm identity"):
        adapt_pipeline._validated_ra_adapt_route_contract(
            {
                **genuine_kwargs,
                "sr_route_profile_contract": downgraded,
                "sr_route_profile_contract_sha256": (
                    ra_engine._route_sha256(downgraded)
                ),
            }
        )

    tampered = copy.deepcopy(contract)
    tampered["execution_settings"].pop(
        "ra_phase3_candidate_gain_policy"
    )
    tampered["execution_settings"].pop(
        "ra_accepted_refit_initialization_policy"
    )
    for key in (
        "phase3_candidate_gain_policy",
        "phase3_candidate_gain_semantics",
        "phase3_active_only_baseline_solver",
        "phase3_active_only_baseline_quantum_query_charge",
        "accepted_refit_initialization_policy",
        "accepted_refit_initialization_coordinate_scope",
        "accepted_refit_initialization_map",
        "accepted_refit_initialization_exact_guard",
        "accepted_refit_initialization_authority",
    ):
        tampered["semantic_invariants"].pop(key)
    tampered_digest = ra_engine._route_sha256(tampered)

    with pytest.raises(ValueError, match="full-response algorithm identity"):
        adapt_pipeline._validated_ra_adapt_route_contract(
            {
                "sr_route_profile_contract": tampered,
                "sr_route_profile_resolved": profile,
                "sr_route_profile_request": profile_request,
                "sr_route_profile_contract_sha256": tampered_digest,
            }
        )


def test_raw_study_protocol_requires_validated_bundle_loading() -> None:
    problem = _hh_problem()
    request = RAAdaptRequest(execution=_execution(rounds=1))
    cell = bundle_module.BundleCellSpec(
        cell_id="raw_study_protocol",
        stage="validation",
        regime_id="fixture",
        nph=1,
        route_id="singleton_plateau",
        algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
        selector_family="ra_adapt",
        candidate_representation=(
            CANDIDATE_REPRESENTATION_SINGLE_PAULI
        ),
        horizon=1,
        source_lock_id="fixture_lock",
    )
    refs = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "fixture_lock",
        "cell_source_lock_sha256": "3" * 64,
        "visible_provenance_sha256": "4" * 64,
        "provenance_tracker_sha256": "5" * 64,
        "ed_cutoff_reference_sha256": "6" * 64,
        "resolver_script_sha256": "7" * 64,
    }
    authority = (
        bundle_module._bundle_protocol_materialization_authority(
            cell=cell,
            bundle_id="fixture_stationary_late_v1",
            bundle_manifest_sha256="8" * 64,
            source_locks_sha256="1" * 64,
            source_lock_refs=refs,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
        )
    )
    with pytest.raises(AttributeError, match="immutable"):
        authority._protocol_sha256 = "9" * 64
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=authority,
    )
    raw_protocol = resolved_ra_adapt_protocol_from_mapping(
        protocol.to_dict()
    )
    assert raw_protocol._materialization_authority is None
    with pytest.raises(ValueError, match="load_validated_bundle_protocol"):
        run_ra_adapt(problem, raw_protocol)


def test_resolved_protocol_fails_closed_on_pool_or_problem_drift() -> None:
    problem = _hh_problem()
    request = RAAdaptRequest(execution=_execution(rounds=1))
    protocol = build_resolved_ra_protocol(problem, request)
    drifted_parent = replace(
        protocol.parent_inventory,
        ordered_pool_sha256="f" * 64,
        sha256=None,
    )
    drifted = _rebuild_protocol(
        protocol,
        parent_inventory=drifted_parent,
    )

    with pytest.raises(ValueError, match="parent pool identity drifted"):
        run_ra_adapt(problem, drifted)

    drifted_representation = replace(
        protocol.parent_inventory,
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
        sha256=None,
    )
    nested_drift = _rebuild_protocol(
        protocol,
        parent_inventory=drifted_representation,
    )
    with pytest.raises(ValueError, match="parent pool identity drifted"):
        run_ra_adapt(problem, nested_drift)

    drifted_lineage_authority = dict(protocol.lineage_authority)
    drifted_lineage_binding = dict(
        drifted_lineage_authority["candidate_inventory_lineage"]
    )
    drifted_lineage_binding["sha256"] = "f" * 64
    drifted_lineage_authority["candidate_inventory_lineage"] = (
        drifted_lineage_binding
    )
    lineage_drift = _rebuild_protocol(
        protocol,
        lineage_authority=drifted_lineage_authority,
    )
    with pytest.raises(
        ValueError,
        match="candidate inventory lineage drifted",
    ):
        run_ra_adapt(problem, lineage_drift)

    non_hh_problem = replace(problem, family_key="hubbard")
    with pytest.raises(ValueError, match="Hubbard.*Holstein L=2"):
        run_ra_adapt(non_hh_problem, request)


def test_facade_receipt_projection_fails_closed_without_real_trust() -> None:
    with pytest.raises(
        RuntimeError,
        match="missing the required real source-Gram trust transaction",
    ):
        _accepted_round_scientific_receipts(
            {
                "history": [
                    {
                        "route_a_trust_region_update": {
                            "source_metric_trust_transaction": None,
                            "source_metric_trust_transaction_failure": (
                                "producer receipt unavailable"
                            ),
                        }
                    }
                ]
            },
            adapter_id="macro",
            executable_inventory=MacroCandidateAdapter().executable_pool(
                _hh_problem()
            ),
        )


@pytest.mark.parametrize(
    "phase3_live",
    [False, True],
    ids=["preplateau-phase2-winner", "open-plateau-phase3-winner"],
)
def test_facade_stabilization_uses_only_the_accepted_plateau_winner(
    phase3_live: bool,
) -> None:
    support = RetainedSupportReceipt(
        feasible=True,
        reason="ok",
        dimension=1,
        rank=1,
        rank_relative_tolerance=1e-6,
        metric_regularization=0.0,
        support_threshold=1e-6,
        negative_eigenvalue_tolerance=1e-10,
        raw_metric_eigenvalues=(1.0,),
        retained_mask=(True,),
        retained_metric_eigenvalues=(1.0,),
        retained_eigenvectors=((1.0,),),
        raw_condition_number=1.0,
        retained_condition_number=1.0,
        factorization_provenance_id="accepted-support",
        source_provenance_id="source",
        receipt_provenance_id="accepted-support-receipt",
    )

    def stabilization(*, kappa: float, boundary: float) -> dict[str, Any]:
        return {
            "joint_linear_solve_policy_effective": (
                PROJECTED_GENERALIZED_SOLVER
            ),
            "supported_metric_projection_provenance_id": (
                support.factorization_provenance_id
            ),
            "kappa_stabilization_shift": kappa,
            "trust_boundary_multiplier_lambda": boundary,
            "total_metric_multiplier_mu": kappa + boundary,
            "trust_boundary_active": boundary > 0.0,
            "supported_metric_whitening_active": False,
            "supported_metric_inverse_sqrt_constructed": False,
        }

    accepted_winner = stabilization(kappa=0.25, boundary=0.5)
    unrelated_candidate = stabilization(kappa=1.0, boundary=0.0)
    row = {
        "phase3_population_activation": {
            "schema": "ra_phase3_population_activation_receipt_v1",
            "policy": RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU,
            "competitive_population_live": phase3_live,
            "insertion_plateau_domain_open": phase3_live,
            "preplateau_admission_authority": (
                None if phase3_live else "phase2_raw_score_top_rank_v1"
            ),
            "winner_materialization_policy": (
                None
                if phase3_live
                else RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION
            ),
        },
        "selected_feature_rows": [
            {"phase2_joint_geometry_reuse": accepted_winner}
        ],
        "route_a_trust_region_update": {
            "phase3_stabilization_receipt": accepted_winner
        },
        "unrelated_candidate_diagnostics": [
            {"phase3_stabilization_receipt": unrelated_candidate}
        ],
    }

    observed = _required_phase3_stabilization(row, support=support)

    assert observed.kappa_stabilization_shift == pytest.approx(0.25)
    assert observed.trust_boundary_multiplier_lambda == pytest.approx(0.5)
    assert observed.total_metric_multiplier_mu == pytest.approx(0.75)
    assert observed.trust_boundary_active is True

    conflicting_accepted_paths = copy.deepcopy(row)
    conflicting_accepted_paths["route_a_trust_region_update"][
        "phase3_stabilization_receipt"
    ] = unrelated_candidate
    with pytest.raises(
        RuntimeError,
        match=(
            "exactly one Phase-III stabilization receipt matching selector "
            "support"
        ),
    ):
        _required_phase3_stabilization(
            conflicting_accepted_paths,
            support=support,
        )


def test_facade_stabilization_is_null_only_for_authenticated_geometry_expansion(
) -> None:
    support = RetainedSupportReceipt(
        feasible=True,
        reason="ok",
        dimension=1,
        rank=1,
        rank_relative_tolerance=1e-6,
        metric_regularization=0.0,
        support_threshold=1e-6,
        negative_eigenvalue_tolerance=1e-10,
        raw_metric_eigenvalues=(1.0,),
        retained_mask=(True,),
        retained_metric_eigenvalues=(1.0,),
        retained_eigenvectors=((1.0,),),
        raw_condition_number=1.0,
        retained_condition_number=1.0,
        factorization_provenance_id="accepted-support",
        source_provenance_id="source",
        receipt_provenance_id="accepted-support-receipt",
    )
    bracket_failure = {
        "schema": "historical_singleton_coordinate_model_v1",
        "feasible": False,
        "reason": "supported_generalized_trust_bracket_failed",
        "joint_linear_solve_policy_effective": (
            PROJECTED_GENERALIZED_SOLVER
        ),
        "supported_metric_projection_provenance_id": (
            support.factorization_provenance_id
        ),
        "curvature_shift": 0.25,
        "trust_lambda": 0.0,
        "supported_metric_whitening_active": False,
        "supported_metric_inverse_sqrt_constructed": False,
    }
    trust_limitation = {
        "source_metric_trust_transaction": None,
        "source_metric_trust_transaction_failure": (
            "not_applicable_geometry_expansion_without_coordinate_prediction"
        ),
        "geometry_expansion_active": True,
        "context_mode": "historical_singleton_geometry_expansion_v1",
        "policy": "source_metric_inverse_sqrt_no_overlap_v1",
        "update_reason": (
            "geometry_expansion_no_coordinate_prediction_no_overlap_hold"
        ),
        "scalar_or_unwhitened_fallback_used": False,
        "model_agreement_authority": (
            "unavailable_without_coordinate_prediction"
        ),
        "endpoint_overlap_measurement_required": False,
        "endpoint_overlap_measurement_performed": False,
        "endpoint_overlap_query_charge": 0,
    }
    row = {
        "selected_feature_rows": [
            {"phase2_joint_geometry_reuse": bracket_failure}
        ],
        "route_a_trust_region_update": trust_limitation,
    }

    assert _required_phase3_stabilization(row, support=support) is None

    partial_quartet = copy.deepcopy(row)
    partial_quartet["selected_feature_rows"][0][
        "phase2_joint_geometry_reuse"
    ]["kappa_stabilization_shift"] = 0.25
    with pytest.raises(
        RuntimeError,
        match="Phase-III stabilization receipt is incomplete",
    ):
        _required_phase3_stabilization(partial_quartet, support=support)

    feasible_without_receipt = copy.deepcopy(row)
    feasible_without_receipt["selected_feature_rows"][0][
        "phase2_joint_geometry_reuse"
    ]["feasible"] = True
    with pytest.raises(
        RuntimeError,
        match="Phase-III stabilization receipt is incomplete",
    ):
        _required_phase3_stabilization(feasible_without_receipt, support=support)

    malformed_limitation = copy.deepcopy(row)
    malformed_limitation["route_a_trust_region_update"][
        "endpoint_overlap_query_charge"
    ] = 1
    with pytest.raises(
        RuntimeError,
        match="geometry-expansion trust limitation is malformed",
    ):
        _required_phase3_stabilization(malformed_limitation, support=support)

    matching_echo = {
        "joint_linear_solve_policy_effective": (
            PROJECTED_GENERALIZED_SOLVER
        ),
        "supported_metric_projection_provenance_id": (
            support.factorization_provenance_id
        ),
        "kappa_stabilization_shift": 0.0,
        "trust_boundary_multiplier_lambda": 0.0,
        "total_metric_multiplier_mu": 0.0,
        "trust_boundary_active": False,
        "supported_metric_whitening_active": False,
        "supported_metric_inverse_sqrt_constructed": False,
    }
    wrong_support_echo = {
        **matching_echo,
        "supported_metric_projection_provenance_id": "wrong-support",
    }
    for echo in (matching_echo, wrong_support_echo, None, "not-a-receipt"):
        conflicting_echo = copy.deepcopy(row)
        conflicting_echo["route_a_trust_region_update"][
            "phase3_stabilization_receipt"
        ] = echo
        with pytest.raises(
            RuntimeError,
            match="conflicts with its geometry-expansion limitation",
        ):
            _required_phase3_stabilization(conflicting_echo, support=support)


@pytest.mark.parametrize(
    "malformed_charge",
    [False, "0", 0.5, -0.5],
)
def test_geometry_expansion_limitation_requires_literal_zero_charge(
    malformed_charge: object,
) -> None:
    payload = {
        "source_metric_trust_transaction": None,
        "source_metric_trust_transaction_failure": (
            "not_applicable_geometry_expansion_without_coordinate_prediction"
        ),
        "geometry_expansion_active": True,
        "context_mode": "historical_singleton_geometry_expansion_v1",
        "policy": "source_metric_inverse_sqrt_no_overlap_v1",
        "update_reason": (
            "geometry_expansion_no_coordinate_prediction_no_overlap_hold"
        ),
        "scalar_or_unwhitened_fallback_used": False,
        "model_agreement_authority": (
            "unavailable_without_coordinate_prediction"
        ),
        "endpoint_overlap_measurement_required": False,
        "endpoint_overlap_measurement_performed": False,
        "endpoint_overlap_query_charge": 0,
    }
    assert _geometry_expansion_trust_limitation(payload) is True

    payload["endpoint_overlap_query_charge"] = malformed_charge
    with pytest.raises(
        RuntimeError,
        match="geometry-expansion trust limitation is malformed",
    ):
        _geometry_expansion_trust_limitation(payload)


@pytest.mark.parametrize(
    ("insertion", "algorithm_id"),
    [
        (
            AppendOnlyInsertion(),
            "paper_i_ra_adapt_macro_append_only_qiskit_transpile_cost_v1",
        ),
        (
            PlateauCommutationInsertion(),
            (
                "paper_i_ra_adapt_macro_plateau_insertion_"
                "qiskit_transpile_cost_v1"
            ),
        ),
        (
            AlwaysCommutationReducedInsertion(),
            (
                "paper_i_ra_adapt_macro_always_insertion_"
                "qiskit_transpile_cost_v1"
            ),
        ),
    ],
)
def test_macro_qiskit_cost_algorithms_select_full_transpile_all_phase(
    insertion: object,
    algorithm_id: str,
) -> None:
    request = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(insertion=insertion),
        execution=_execution(rounds=1),
    )

    _profile_request, profile, contract, _digest = (
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=algorithm_id,
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


@pytest.mark.parametrize(
    ("insertion", "algorithm_id"),
    [
        (
            AppendOnlyInsertion(),
            "paper_i_ra_adapt_macro_append_only_repair_v1",
        ),
        (
            PlateauCommutationInsertion(),
            "paper_i_ra_adapt_macro_plateau_insertion_repair_v1",
        ),
        (
            AlwaysCommutationReducedInsertion(),
            "paper_i_ra_adapt_macro_always_insertion_repair_v1",
        ),
    ],
)
def test_existing_macro_algorithms_keep_graph_span_cost_mode(
    insertion: object,
    algorithm_id: str,
) -> None:
    request = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(insertion=insertion),
        execution=_execution(rounds=1),
    )

    _profile_request, profile, contract, _digest = (
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=algorithm_id,
        )
    )

    assert not profile.endswith(
        "__qiskit_full_ansatz_transpile_cost_all_phases_v1"
    )
    assert (
        contract["execution_settings"]["phase3_backend_cost_mode"]
        == "marrakesh_graph_span_v1"
    )
    assert (
        "selector_compile_cost_policy"
        not in contract["semantic_invariants"]
    )


def test_macro_qiskit_cost_algorithm_rejects_representation_mismatch() -> None:
    request = RAAdaptRequest(
        adapter=SinglePauliWordCandidateAdapter(),
        method=SRMethodPolicy(insertion=AppendOnlyInsertion()),
        execution=_execution(rounds=1),
    )

    with pytest.raises(
        ValueError,
        match="macro candidate adapter",
    ):
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=(
                "paper_i_ra_adapt_macro_append_only_"
                "qiskit_transpile_cost_v1"
            ),
        )


def test_macro_qiskit_cost_algorithm_rejects_insertion_mismatch() -> None:
    request = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(insertion=PlateauCommutationInsertion()),
        execution=_execution(rounds=1),
    )

    with pytest.raises(
        ValueError,
        match="insertion policy",
    ):
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=(
                "paper_i_ra_adapt_macro_append_only_"
                "qiskit_transpile_cost_v1"
            ),
        )


def test_macro_qiskit_cost_algorithm_rejects_non_all_phase_scope() -> None:
    request = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(insertion=AppendOnlyInsertion()),
        execution=_execution(rounds=1),
    )

    with pytest.raises(
        ValueError,
        match="all-phase resource weighting",
    ):
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_MEASURED,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
            algorithm_id=(
                "paper_i_ra_adapt_macro_append_only_"
                "qiskit_transpile_cost_v1"
            ),
        )


def test_macro_facade_emits_exact_chart_and_repaired_route_provenance() -> None:
    problem = _hh_problem()
    adapter = MacroCandidateAdapter()
    request = RAAdaptRequest(
        adapter=adapter,
        method=SRMethodPolicy(insertion=AppendOnlyInsertion()),
        execution=_execution(rounds=1),
    )

    observed = run_ra_adapt(problem, request)
    protocol = observed.protocol
    assert observed.run.paper_i_summary is not None
    assert protocol.candidate_representation == (
        CANDIDATE_REPRESENTATION_MACRO
    )
    assert protocol.derivative_chart_id == EXACT_ORDERED_INSERTION_CHART
    assert protocol.phase3_solver_id == PROJECTED_GENERALIZED_SOLVER
    assert protocol.trust_policy_id == SOURCE_GRAM_NO_OVERLAP_TRUST
    assert protocol.lineage_authority["parent_route_profile"]
    assert len(protocol.lineage_authority["parent_contract_sha256"]) == 64
    assert observed.run.stop.completed_controller_rounds == 1
    assert observed.run.route.profile_request.startswith(
        "paper_i_ra_adapt__macro_generator_v1__"
    )
    assert observed.scientific_receipts["candidate_geometry_chart"] == (
        EXACT_ORDERED_INSERTION_CHART
    )
    assert observed.scientific_receipts["phase3_solver"] == (
        PROJECTED_GENERALIZED_SOLVER
    )
    assert observed.scientific_receipts["trust_policy"] == (
        SOURCE_GRAM_NO_OVERLAP_TRUST
    )
    assert observed.run.scientific_replay
    assert all(
        receipt.trust_solve.endpoint_overlap_query_charge == 0
        for receipt in observed.run.scientific_replay
    )
    _assert_real_ra_scientific_receipts(observed)
    repaired_route = observed.scientific_receipts[
        "resolved_route_contract"
    ]
    route_json = json.dumps(repaired_route, sort_keys=True)
    invariants = repaired_route["semantic_invariants"]
    assert invariants["phase3_support_projection_active"] is True
    assert invariants["phase3_supported_whitening_active"] is False
    assert (
        invariants["phase3_supported_metric_inverse_sqrt_active"]
        is False
    )
    assert invariants["phase3_metric_ridge_active"] is False
    assert "deferred_gram_all_models_infeasible_fallback_v1" in route_json
    assert "all_energy_models_infeasible_novelty_fallback" not in route_json
    assert (
        "collective_span_novelty_over_symmetric_cost_v1"
        not in route_json
    )
    assert repaired_route["lineage_authority"][
        "parent_contract_sha256"
    ] == protocol.lineage_authority["parent_contract_sha256"]
    inventory_lineage = observed.scientific_receipts[
        "candidate_inventory_lineage"
    ]
    inventory_binding = protocol.lineage_authority[
        "candidate_inventory_lineage"
    ]
    assert inventory_lineage["count"] == protocol.executable_pool.count
    assert inventory_lineage["sha256"] == inventory_binding["sha256"]
    assert inventory_lineage["ordered_rows_sha256"] == (
        inventory_binding["ordered_rows_sha256"]
    )
    accepted_lineage = observed.scientific_receipts[
        "accepted_round_receipts"
    ][0]["accepted_candidate_lineage"]
    assert len(accepted_lineage) == 1
    assert accepted_lineage[0]["representation_id"] == (
        CANDIDATE_REPRESENTATION_MACRO
    )
    assert accepted_lineage[0]["parent_identities"] == []
    assert accepted_lineage[0]["insertion_position"] == (
        observed.run.accepted_transitions[0].insertion_position
    )
    assert len(accepted_lineage[0]["sha256"]) == 64

    first = adapter.executable_pool(problem).candidates[0]
    geometry = adapter.candidate_geometry(first, position=0)
    assert geometry.coordinate_chart == EXACT_ORDERED_INSERTION_CHART
    assert geometry.candidate_representation == (
        CANDIDATE_REPRESENTATION_MACRO
    )
    assert geometry.insertion_position == 0
    assert geometry.as_dict()["candidate_label"] == first.label


def test_endpoint_overlap_trust_ablation_uses_and_charges_exact_fs_motion(
) -> None:
    from pipelines.static_adapt.ra_adapt.contracts import (
        ENDPOINT_OVERLAP_DISPLACEMENT_TRUST,
    )
    from pipelines.static_adapt.sr_snake import (
        EndpointOverlapDisplacementTrust,
    )

    request = RAAdaptRequest(
        method=SRMethodPolicy(
            insertion=PlateauCommutationInsertion(),
            trust_update=EndpointOverlapDisplacementTrust(),
        ),
        execution=_execution(rounds=1),
    )

    observed = run_ra_adapt(_hh_problem(), request)
    protocol = observed.protocol
    route = protocol.route_contract
    assert isinstance(route, dict)
    assert protocol.trust_policy_id == ENDPOINT_OVERLAP_DISPLACEMENT_TRUST
    assert resolved_ra_adapt_protocol_from_mapping(
        protocol.to_dict()
    ) == protocol
    assert route["execution_settings"][
        "historical_singleton_trust_region_update_policy"
    ] == "displacement_calibrated_unbounded_v2"
    assert route["semantic_invariants"]["endpoint_overlap_required"] is True
    assert route["semantic_invariants"][
        "endpoint_overlap_measurement_active"
    ] is True

    accepted = observed.scientific_receipts[
        "accepted_round_receipts"
    ][0]
    assert accepted["source_gram_no_overlap_trust"] is None
    trust = accepted["endpoint_overlap_trust"]
    assert trust["policy"] == "displacement_calibrated_unbounded_v2"
    assert trust["displacement_ratio_metric"] == (
        "predicted_fubini_study_vs_endpoint_fubini_study_v1"
    )
    assert trust["predicted_fubini_study_displacement"] >= 0.0
    assert trust["realized_fubini_study_displacement"] >= 0.0
    assert trust["endpoint_overlap_query_charge"] == 1
    accounting = trust["endpoint_overlap_query_accounting"]
    assert accounting["status"] == "complete"
    assert accounting["formal_query_category"] == "N_cross"
    assert accounting["component"] == "N_metric"
    assert observed.run.scientific_replay[0].trust_solve.policy == (
        "displacement_calibrated_unbounded_v2"
    )
    assert (
        observed.run.scientific_replay[0]
        .trust_solve.endpoint_overlap_query_charge
        == 1
    )
@pytest.mark.parametrize(
    "adapter_factory",
    [MacroCandidateAdapter, SinglePauliWordCandidateAdapter],
    ids=["macro", "singleton"],
)
def test_always_insertion_facades_reduce_the_full_logical_domain_by_round_two(
    adapter_factory: type[
        MacroCandidateAdapter | SinglePauliWordCandidateAdapter
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)

    observed = run_ra_adapt(
        _hh_problem(),
        RAAdaptRequest(
            adapter=adapter_factory(),
            method=SRMethodPolicy(
                insertion=AlwaysCommutationReducedInsertion()
            ),
            execution=_execution(rounds=2),
        ),
    )

    assert observed.run.stop.completed_controller_rounds == 2
    assert observed.protocol.algorithm_id == RA_ADAPT_ALGORITHM_ID
    assert (
        observed.run.route.insertion_policy
        == "always_commutation_reduced"
    )
    assert observed.run.accepted_transitions[0].insertion_position == 0
    assert observed.run.paper_i_summary is not None
    assert observed.run.paper_i_summary.schema == "paper_i_run_summary_v1"
    accepted_rounds = observed.scientific_receipts[
        "accepted_round_receipts"
    ]
    reduction = accepted_rounds[1]["insertion_commutation_reduced"]
    assert reduction["schema"] == (
        "commutation_reduced_insertion_domain_receipt_v1"
    )
    assert reduction["policy"] == "always_commutation_reduced"
    assert reduction["requested_positions"] == [0, 1]
    assert reduction["requested_position_count"] == 2
    assert reduction["candidate_count"] == 89
    assert reduction["retained_representative_count"] == 141
    assert reduction["collapsed_position_count"] == 37
    plans = reduction["candidate_position_plans"]
    assert {
        tuple(plan["requested_positions"]) for plan in plans
    } == {(0, 1)}
    plans_by_pool_index = {
        int(plan["candidate_pool_index"]): plan for plan in plans
    }
    assert plans_by_pool_index[0]["representative_positions"] == [0]
    assert list(
        plans_by_pool_index[0]["members_by_representative"].values()
    ) == [[0, 1]]
    assert plans_by_pool_index[2]["representative_positions"] == [0, 1]
    assert list(
        plans_by_pool_index[2]["members_by_representative"].values()
    ) == [[0], [1]]

    scored = [
        row["scored_insertion_position_population"]
        for row in accepted_rounds
    ]
    assert [row["append_position"] for row in scored] == [0, 1]
    assert all(row["scored_record_count"] > 0 for row in scored)
    assert all(len(row["sha256"]) == 64 for row in scored)
    second_round = scored[1]
    phase_i_records = second_round["phases"][0]["records"]
    positions_by_generator: dict[tuple[int, str], set[int]] = {}
    for record in phase_i_records:
        generator_key = (
            int(record["pool_index"]),
            str(record["generator_id"]),
        )
        positions_by_generator.setdefault(generator_key, set()).add(
            int(record["insertion_position"])
        )
    assert positions_by_generator
    assert len(phase_i_records) == 141
    assert Counter(
        tuple(sorted(positions))
        for positions in positions_by_generator.values()
    ) == Counter({(0,): 37, (0, 1): 52})
    representatives_by_pool_index = {
        int(plan["candidate_pool_index"]): set(
            int(position)
            for position in plan["representative_positions"]
        )
        for plan in plans
    }
    assert all(
        positions
        == representatives_by_pool_index[pool_index]
        for (pool_index, _generator_id), positions
        in positions_by_generator.items()
    )
    assert second_round["interior_scored_count"] > 0


@pytest.mark.parametrize(
    "adapter_factory",
    [MacroCandidateAdapter, SinglePauliWordCandidateAdapter],
    ids=["macro", "singleton"],
)
def test_append_only_facades_keep_every_scored_position_at_the_endpoint(
    adapter_factory: type[
        MacroCandidateAdapter | SinglePauliWordCandidateAdapter
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)

    observed = run_ra_adapt(
        _hh_problem(),
        RAAdaptRequest(
            adapter=adapter_factory(),
            method=SRMethodPolicy(insertion=AppendOnlyInsertion()),
            execution=_execution(rounds=2),
        ),
    )

    scored = [
        row["scored_insertion_position_population"]
        for row in observed.scientific_receipts["accepted_round_receipts"]
    ]
    assert [row["append_position"] for row in scored] == [0, 1]
    assert all(row["interior_scored_count"] == 0 for row in scored)
    for round_receipt in scored:
        assert all(
            {
                int(record["insertion_position"])
                for record in phase["records"]
            }
            == {int(round_receipt["append_position"])}
            for phase in round_receipt["phases"]
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda receipt: receipt[
            "scored_insertion_position_population"
        ]["phases"][0]["records"][0].update(
            {"insertion_position": 0, "position_class": "interior"}
        ),
        lambda receipt: receipt["accepted_candidate_lineage"][0].update(
            {"insertion_position": 0}
        ),
    ),
    ids=("interior-scored-position", "interior-admitted-position"),
)
def test_append_only_finalization_rejects_interior_positions(
    mutation,
) -> None:
    receipt = {
        "accepted_candidate_lineage": [{"insertion_position": 1}],
        "scored_insertion_position_population": {
            "append_position": 1,
            "phases": [
                {
                    "phase": phase,
                    "records": [
                        {
                            "insertion_position": 1,
                            "position_class": "append",
                        }
                    ],
                }
                for phase in ("phase_i", "phase_ii", "phase_iii")
            ],
        },
    }
    mutation(receipt)

    validator = getattr(
        ra_engine,
        "_validate_endpoint_only_accepted_round",
        None,
    )
    assert validator is not None
    with pytest.raises(RuntimeError, match="interior"):
        validator(receipt)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda receipt: receipt["accepted_candidate_lineage"][0].update(
            {"insertion_position": 1}
        ),
        lambda receipt: receipt[
            "scored_insertion_position_population"
        ]["phases"][2]["records"][0].update(
            {"generator_id": "gen:other"}
        ),
    ),
    ids=("collapsed-position-admitted", "unscored-lineage-admitted"),
)
def test_reduced_finalization_binds_admission_to_phase3_representative(
    mutation,
) -> None:
    receipt = {
        "accepted_candidate_lineage": [
            {
                "candidate_label": "candidate",
                "generator_identity": "gen:candidate",
                "insertion_position": 0,
            }
        ],
        "insertion_commutation_reduced": {
            "candidate_position_plans": [
                {
                    "candidate_pool_index": 7,
                    "candidate_label": "candidate",
                    "representative_positions": [0],
                }
            ]
        },
        "scored_insertion_position_population": {
            "phases": [
                {"phase": "phase_i", "records": []},
                {"phase": "phase_ii", "records": []},
                {
                    "phase": "phase_iii",
                    "records": [
                        {
                            "pool_index": 7,
                            "pool_label": "candidate",
                            "generator_id": "gen:candidate",
                            "insertion_position": 0,
                        }
                    ],
                },
            ]
        },
    }
    mutation(receipt)

    validator = getattr(
        ra_engine,
        "_validate_reduced_accepted_round_admission",
        None,
    )
    assert validator is not None
    with pytest.raises(RuntimeError, match="representative|Phase-III"):
        validator(
            receipt,
            reduction_key="insertion_commutation_reduced",
        )


def test_adapter_exposure_preserves_each_exact_position_once() -> None:
    owners = (
        (7, {"position_id": 2, "marker": "late-overlap"}, "parent:b"),
        (1, {"position_id": 0, "marker": "position-zero"}, "parent:a"),
        (3, {"position_id": 2, "marker": "early-overlap"}, "parent:a"),
        (2, {"position_id": 1, "marker": "position-one"}, "parent:b"),
    )

    primary_owner, position_owners = (
        _deduplicated_adapter_position_owners(owners)
    )

    assert primary_owner == "parent:a"
    assert [row[0] for row in position_owners] == [1, 2, 3]
    assert [row[1]["position_id"] for row in position_owners] == [0, 1, 2]
    assert [row[1]["marker"] for row in position_owners] == [
        "position-zero",
        "position-one",
        "early-overlap",
    ]


def test_repaired_ra_route_executes_representation_adapter_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"expose_children": 0, "candidate_geometry": 0}
    original_expose = MacroCandidateAdapter.expose_children
    original_geometry = MacroCandidateAdapter.candidate_geometry

    def _record_exposure(
        self: MacroCandidateAdapter,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        calls["expose_children"] += 1
        return original_expose(self, *args, **kwargs)

    def _record_geometry(
        self: MacroCandidateAdapter,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        calls["candidate_geometry"] += 1
        return original_geometry(self, *args, **kwargs)

    monkeypatch.setattr(
        MacroCandidateAdapter,
        "expose_children",
        _record_exposure,
    )
    monkeypatch.setattr(
        MacroCandidateAdapter,
        "candidate_geometry",
        _record_geometry,
    )

    result = run_ra_adapt(
        _hh_problem(),
        RAAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=_execution(rounds=1),
        ),
    )

    assert result.run.stop.completed_controller_rounds == 1
    assert calls["expose_children"] >= 1
    assert calls["candidate_geometry"] >= 1
