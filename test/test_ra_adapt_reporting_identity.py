from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipelines.reporting.paper_i_run_summary import (
    _canonical_ra_semantic_closure_identities,
    _validate_canonical_identity,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    GlobalSinglePauliWordCandidateAdapter,
    MacroCandidateAdapter,
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    RAAdaptRequest,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RESOURCE_WEIGHTING_LATE,
    ra_adapt_request_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_ALGORITHM_ID,
    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
    RA_ADAPT_LEGACY_ALGORITHM_ID,
    _macro_parent_contract,
    _repaired_route_contract,
)
from pipelines.static_adapt.ra_adapt.l3_page12 import (
    PAPER_I_L3_PAGE12_ALGORITHM_ID,
    PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256,
    build_paper_i_l3_page12_request,
)
from pipelines.static_adapt.ra_adapt.pools import (
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PROBLEM_LOCKS,
)
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    build_paper_i_ra_all_phase_position_adaptive_request,
    semantic_closure_route_identity,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AlwaysCommutationReducedInsertion,
    AppendOnlyInsertion,
    PlateauCommutationInsertion,
    SRMethodPolicy,
)


@pytest.mark.parametrize("horizon", (1, 5, 50))
def test_semantic_reporting_rebuilds_every_authorized_horizon(
    horizon: int,
) -> None:
    request = build_paper_i_ra_all_phase_position_adaptive_request(
        insertion_policy="append_only",
        maximum_controller_rounds=horizon,
    )
    identity = semantic_closure_route_identity(
        request.adapter.route_variant
    )
    profile_request, profile, contract, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=identity.algorithm_id,
    )
    observed = (
        str(contract["route_family"]),
        str(profile_request),
        str(profile),
        str(digest),
    )

    assert observed in _canonical_ra_semantic_closure_identities(
        request.method,
        candidate_representation=request.adapter.candidate_representation_id,
    )


@pytest.mark.parametrize(
    ("insertion", "algorithm_id", "expected_mode"),
    (
        (
            AlwaysCommutationReducedInsertion(),
            "unrelated_algorithm",
            "full_commutation_reduced",
        ),
        (
            PlateauCommutationInsertion(),
            "spoofed_always_insertion_algorithm",
            "insertion_commutation_plateau_v2",
        ),
        (
            AppendOnlyInsertion(),
            "spoofed_plateau_algorithm",
            "append_only",
        ),
    ),
)
def test_macro_parent_semantics_come_only_from_typed_insertion_policy(
    insertion,
    algorithm_id: str,
    expected_mode: str,
) -> None:
    request = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(insertion=insertion),
    )

    contract, _digest = _macro_parent_contract(
        request,
        algorithm_id=algorithm_id,
    )

    assert contract["execution_settings"]["adapt_insertion_mode"] == (
        expected_mode
    )


def test_always_reduced_request_rehydrates_without_name_inference() -> None:
    request = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(
            insertion=AlwaysCommutationReducedInsertion()
        ),
    )
    payload = request.to_dict()

    assert payload["method"]["insertion"] == {
        "kind": "always_commutation_reduced"
    }
    assert ra_adapt_request_from_mapping(payload) == request


def test_retired_raw_full_policy_cannot_rehydrate() -> None:
    payload = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
    ).to_dict()
    payload["method"]["insertion"] = {"kind": "full_commutation"}

    with pytest.raises(ValueError, match="Unknown insertion policy"):
        ra_adapt_request_from_mapping(payload)


@pytest.mark.parametrize(
    "adapter",
    (SinglePauliWordCandidateAdapter(), MacroCandidateAdapter()),
)
@pytest.mark.parametrize(
    "insertion",
    (
        AppendOnlyInsertion(),
        PlateauCommutationInsertion(),
        AlwaysCommutationReducedInsertion(),
    ),
)
def test_summary_accepts_only_digest_authenticated_ra_supersessions(
    adapter: SinglePauliWordCandidateAdapter | MacroCandidateAdapter,
    insertion: (
        AppendOnlyInsertion
        | PlateauCommutationInsertion
        | AlwaysCommutationReducedInsertion
    ),
) -> None:
    method = SRMethodPolicy(insertion=insertion)
    request = RAAdaptRequest(adapter=adapter, method=method)
    profile_request, profile, contract, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
        algorithm_id=RA_ADAPT_LEGACY_ALGORITHM_ID,
    )
    settings = contract["execution_settings"]
    route = SimpleNamespace(
        family=contract["route_family"],
        profile_request=profile_request,
        profile=profile,
        contract_sha256=digest,
        method=method,
        admission_policy=method.admission.kind,
        insertion_policy=method.insertion.kind,
        pruning_policy=method.pruning.kind,
        beam_policy=method.beam.kind,
        execution=SimpleNamespace(
            pool=settings["adapt_pool"],
            phase0_enabled=settings["phase0_pilot_enabled"],
            phase_live_hysteresis_enabled=(
                settings["phase_live_hysteresis_enabled"]
            ),
        ),
    )
    source = SimpleNamespace(
        problem=SimpleNamespace(family_key="hh", num_sites=2),
        route=route,
        canonical_reporting=SimpleNamespace(
            candidate_representation=adapter.candidate_representation_id,
        ),
    )

    _validate_canonical_identity(source)

    forged_route = SimpleNamespace(
        **{
            **vars(route),
            "contract_sha256": "f" * 64,
        }
    )
    with pytest.raises(
        ValueError,
        match="typed canonical route authority",
    ):
        _validate_canonical_identity(
            SimpleNamespace(
                **{
                    **vars(source),
                    "route": forged_route,
                }
            )
        )


@pytest.mark.parametrize(
    ("adapter", "insertion", "algorithm_id"),
    (
        (
            MacroCandidateAdapter(),
            AlwaysCommutationReducedInsertion(),
            (
                "paper_i_ra_adapt_macro_always_insertion_"
                "qiskit_transpile_cost_v1"
            ),
        ),
        (
            MacroCandidateAdapter(),
            PlateauCommutationInsertion(),
            (
                "paper_i_ra_adapt_macro_plateau_insertion_"
                "qiskit_transpile_cost_v1"
            ),
        ),
        (
            GlobalSinglePauliWordCandidateAdapter(),
            PlateauCommutationInsertion(),
            "paper_i_ra_adapt_global_singleton_plateau_commutation_v1",
        ),
        (
            GlobalSinglePauliWordCandidateAdapter(),
            PlateauCommutationInsertion(),
            (
                "paper_i_ra_adapt_global_singleton_plateau_commutation_"
                "qiskit_transpile_cost_v1"
            ),
        ),
        (
            GlobalSinglePauliWordCandidateAdapter(),
            PlateauCommutationInsertion(),
            RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
        ),
    ),
)
def test_summary_accepts_authenticated_named_ra_routes(
    adapter: (
        GlobalSinglePauliWordCandidateAdapter | MacroCandidateAdapter
    ),
    insertion: (
        AlwaysCommutationReducedInsertion
        | PlateauCommutationInsertion
    ),
    algorithm_id: str,
) -> None:
    method = SRMethodPolicy(insertion=insertion)
    request = RAAdaptRequest(adapter=adapter, method=method)
    profile_request, profile, contract, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=algorithm_id,
    )
    settings = contract["execution_settings"]
    route = SimpleNamespace(
        family=contract["route_family"],
        profile_request=profile_request,
        profile=profile,
        contract_sha256=digest,
        method=method,
        admission_policy=method.admission.kind,
        insertion_policy=method.insertion.kind,
        pruning_policy=method.pruning.kind,
        beam_policy=method.beam.kind,
        execution=SimpleNamespace(
            pool=settings["adapt_pool"],
            phase0_enabled=settings["phase0_pilot_enabled"],
            phase_live_hysteresis_enabled=(
                settings["phase_live_hysteresis_enabled"]
            ),
        ),
    )
    source = SimpleNamespace(
        problem=SimpleNamespace(family_key="hh", num_sites=2),
        route=route,
        canonical_reporting=SimpleNamespace(
            candidate_representation=adapter.candidate_representation_id,
        ),
    )

    _validate_canonical_identity(source)


@pytest.mark.parametrize(
    "problem_request_sha256",
    tuple(
        lock["problem_request_sha256"]
        for lock in PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PROBLEM_LOCKS.values()
    ),
)
def test_summary_accepts_authenticated_l3_nph3_problem_locks(
    problem_request_sha256: str,
) -> None:
    request = build_paper_i_l3_page12_request()
    method = request.method
    profile_request, profile, contract, digest = _repaired_route_contract(
        request,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
        algorithm_id=PAPER_I_L3_PAGE12_ALGORITHM_ID,
    )
    assert digest == PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256
    settings = contract["execution_settings"]
    route = SimpleNamespace(
        family=contract["route_family"],
        profile_request=profile_request,
        profile=profile,
        contract_sha256=digest,
        method=method,
        admission_policy=method.admission.kind,
        insertion_policy=method.insertion.kind,
        pruning_policy=method.pruning.kind,
        beam_policy=method.beam.kind,
        execution=SimpleNamespace(
            pool=settings["adapt_pool"],
            phase0_enabled=settings["phase0_pilot_enabled"],
            phase_live_hysteresis_enabled=(
                settings["phase_live_hysteresis_enabled"]
            ),
        ),
    )
    source = SimpleNamespace(
        problem=SimpleNamespace(
            family_key="hh",
            num_sites=3,
            problem_request_sha256=problem_request_sha256,
        ),
        route=route,
        canonical_reporting=SimpleNamespace(
            candidate_representation=(
                request.adapter.candidate_representation_id
            ),
        ),
    )

    _validate_canonical_identity(source)
