from __future__ import annotations

from typing import Any

import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt import (
    GlobalSingletonGradientPhase0CandidateAdapter,
    MacroGradientPhase0CandidateAdapter,
    RAAdaptRequest,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RESOURCE_WEIGHTING_ALL_PHASE,
    _attach_validated_bundle_protocol_authority,
    ra_adapt_request_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
    RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.sr_snake import (
    ForkLocalBeam,
    MetricPruning,
    PlateauCommutationInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRStopPolicy,
)


PAGE13_BASE_ROUTE_SHA256 = (
    "1b2f7254a96a27a7f2a262f1b4bc19c886b421a9cbaa5e24c95e354a02f2cf45"
)
PAGE12_BASE_ROUTE_SHA256 = (
    "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
)


def _hh_problem() -> Any:
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
        )
    )


def _beam_metric_method() -> SRMethodPolicy:
    return SRMethodPolicy(
        insertion=PlateauCommutationInsertion(),
        pruning=MetricPruning(),
        beam=ForkLocalBeam(
            live_parent_branches=3,
            admission_children_per_parent=2,
            maximum_admission_children_per_round=6,
            s_alg_weight=0.005,
        ),
    )


def test_fork_local_beam_request_round_trips_derived_calibration_status() -> None:
    request = RAAdaptRequest(method=_beam_metric_method())

    restored = ra_adapt_request_from_mapping(request.to_dict())

    assert restored.method.beam == request.method.beam


def _resolved_route_contract(
    *,
    adapter: Any,
    algorithm_id: str,
    candidate_representation: str,
) -> dict[str, Any]:
    problem = _hh_problem()
    request = RAAdaptRequest(
        adapter=adapter,
        method=_beam_metric_method(),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=1)
        ),
    )
    cell = bundle_module.BundleCellSpec(
        cell_id="beam_metric_plateau_fixture",
        stage="validation",
        regime_id="fixture",
        nph=1,
        route_id="beam_metric_plateau_fixture",
        algorithm_id=algorithm_id,
        selector_family="ra_adapt",
        candidate_representation=candidate_representation,
        horizon=1,
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
    authority = bundle_module._bundle_protocol_materialization_authority(
        cell=cell,
        bundle_id=bundle_module.STATIONARY_BUNDLE_ID,
        bundle_manifest_sha256="8" * 64,
        source_locks_sha256="1" * 64,
        source_lock_refs=source_lock_refs,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=authority,
    )
    assert protocol.route_contract is not None
    return dict(protocol.route_contract)


def _validated_execution_protocol(
    *,
    adapter: Any,
    algorithm_id: str,
    candidate_representation: str,
) -> tuple[Any, Any]:
    problem = _hh_problem()
    request = RAAdaptRequest(
        adapter=adapter,
        method=_beam_metric_method(),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=1)
        ),
    )
    cell = bundle_module.BundleCellSpec(
        cell_id="beam_metric_plateau_execution_fixture",
        stage="validation",
        regime_id="fixture",
        nph=1,
        route_id="beam_metric_plateau_execution_fixture",
        algorithm_id=algorithm_id,
        selector_family="ra_adapt",
        candidate_representation=candidate_representation,
        horizon=1,
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
        "bundle_id": bundle_module.STATIONARY_BUNDLE_ID,
        "bundle_manifest_sha256": "8" * 64,
        "source_locks_sha256": "1" * 64,
        "source_lock_refs": source_lock_refs,
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
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
    protocol = _attach_validated_bundle_protocol_authority(
        protocol,
        bundle_module._bundle_protocol_materialization_authority(
            **authority_kwargs,
            protocol_sha256=protocol.sha256,
        ),
    )
    return problem, protocol


def _assert_beam_metric_plateau_contract(contract: dict[str, Any]) -> None:
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]

    assert execution["adapt_insertion_mode"] == (
        "insertion_commutation_plateau_v2"
    )
    assert execution["phase1_prune_enabled"] is True
    assert execution["phase1_prune_schur_nomination_route"] == (
        "metric_regularized_v1"
    )
    assert execution["adapt_beam_live_branches"] == 3
    assert execution["adapt_beam_children_per_parent"] == 2
    assert execution["adapt_beam_lambda"] == 0.005
    assert invariants["canonical_pruning_policy"] == "metric"
    assert invariants["canonical_beam_policy"] == "fork_local"
    assert invariants["beam_shape"] == (
        "three_live_two_children_per_parent_v1"
    )
    assert invariants["beam_maximum_admission_children_per_round"] == 6


def test_page13_macro_plateau_composes_metric_pruning_and_beam() -> None:
    contract = _resolved_route_contract(
        adapter=MacroGradientPhase0CandidateAdapter(),
        algorithm_id=(
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
        ),
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
    )

    _assert_beam_metric_plateau_contract(contract)
    assert contract["sha256"] != PAGE13_BASE_ROUTE_SHA256


def test_page12_singleton_beam_receipt_matches_typed_three_by_two_shape(
) -> None:
    contract = _resolved_route_contract(
        adapter=GlobalSingletonGradientPhase0CandidateAdapter(),
        algorithm_id=(
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID
        ),
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    )

    _assert_beam_metric_plateau_contract(contract)
    assert contract["sha256"] != PAGE12_BASE_ROUTE_SHA256


def test_page12_and_page13_beam_metric_routes_have_distinct_hashes() -> None:
    macro = _resolved_route_contract(
        adapter=MacroGradientPhase0CandidateAdapter(),
        algorithm_id=(
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
        ),
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
    )
    singleton = _resolved_route_contract(
        adapter=GlobalSingletonGradientPhase0CandidateAdapter(),
        algorithm_id=(
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID
        ),
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    )

    assert macro["sha256"] != singleton["sha256"]


@pytest.mark.parametrize(
    ("adapter", "algorithm_id", "candidate_representation"),
    (
        (
            MacroGradientPhase0CandidateAdapter(),
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
            CANDIDATE_REPRESENTATION_MACRO,
        ),
        (
            GlobalSingletonGradientPhase0CandidateAdapter(),
            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
            CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        ),
    ),
    ids=("page13_macro", "page12_singleton"),
)
def test_metric_pruning_beam_routes_execute_one_real_controller_round(
    adapter: Any,
    algorithm_id: str,
    candidate_representation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    problem, protocol = _validated_execution_protocol(
        adapter=adapter,
        algorithm_id=algorithm_id,
        candidate_representation=candidate_representation,
    )

    observed = run_ra_adapt(problem, protocol)

    assert observed.run.stop.completed_controller_rounds == 1
    assert len(observed.run.accepted_transitions) == 1
    resolved_route = observed.scientific_receipts["resolved_route_contract"]
    _assert_beam_metric_plateau_contract(resolved_route)
