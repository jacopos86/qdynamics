from __future__ import annotations

from dataclasses import replace

import pytest

from pipelines.static_adapt.ra_adapt.append import (
    APPEND_ADAPT_ALGORITHM_ID,
    build_resolved_append_protocol,
    run_append_adapt,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    APPEND_ADAPT_PROTOCOL_SCHEMA,
    APPEND_CONVENTIONAL_SELECTOR_ID,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RA_ADAPT_PROTOCOL_SCHEMA_V1,
    RA_STAGED_SELECTOR_ID,
    RESOURCE_WEIGHTING_ALL_PHASE,
    AppendAdaptRequest,
    append_adapt_request_from_mapping,
    _mint_bundle_protocol_materialization_authority,
    bundle_protocol_materialization_receipt,
    resolved_ra_adapt_protocol_from_mapping,
    ra_adapt_request_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import build_resolved_ra_protocol
from pipelines.static_adapt.ra_adapt.l3_page12 import (
    PAPER_I_L3_PAGE12_ALGORITHM_ID,
    PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256,
    PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_APPLICATION_SOURCE_SHA256,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_EXACT_ENERGY,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_REGIMES,
    PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
    build_paper_i_l3_page12_problem,
    build_paper_i_l3_page12_request,
    paper_i_l3_page12_application_source_contract,
)
from pipelines.static_adapt.sr_snake.contracts import (
    FreshStart,
    SRExecutionPolicy,
    SRStopPolicy,
)


@pytest.fixture(scope="module")
def weak_sector_sources():
    rows = {}
    for regime_id in PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_REGIMES:
        problem = build_paper_i_l3_page12_problem(regime_id, nph=3)
        rows[regime_id] = (
            problem,
            paper_i_l3_page12_application_source_contract(problem),
        )
    return rows


def _authority(
    *,
    execution_id: str,
    source_sha256: str,
    append: bool,
):
    locks = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": execution_id,
        "cell_source_lock_sha256": "3" * 64,
        PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY: source_sha256,
    }
    receipt = bundle_protocol_materialization_receipt(
        bundle_id="paper_i_l3_weak_sector_test_bundle",
        bundle_manifest_sha256="4" * 64,
        source_locks_sha256="1" * 64,
        source_lock_refs=locks,
        cell_id=execution_id,
        source_lock_id=execution_id,
        protocol_schema=(
            APPEND_ADAPT_PROTOCOL_SCHEMA
            if append
            else RA_ADAPT_PROTOCOL_SCHEMA_V1
        ),
        algorithm_id=(
            APPEND_ADAPT_ALGORITHM_ID
            if append
            else PAPER_I_L3_PAGE12_ALGORITHM_ID
        ),
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        selector_identity=(
            APPEND_CONVENTIONAL_SELECTOR_ID
            if append
            else RA_STAGED_SELECTOR_ID
        ),
        active_gradient_policy=(
            ACTIVE_GRADIENT_MEASURED
            if append
            else ACTIVE_GRADIENT_STATIONARY
        ),
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )
    return _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=locks,
    )


def test_l3_nph3_weak_sector_physics_pools_and_ed_are_source_locked(
    weak_sector_sources,
) -> None:
    for regime_id, (problem, source) in weak_sector_sources.items():
        assert problem.request.num_sites == 3
        assert problem.request.n_ph_max == 3
        assert problem.sector.num_particles == (2, 1)
        assert problem.layout.total_qubits == 12
        assert source["regime_id"] == regime_id
        assert source["sha256"] == (
            PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_APPLICATION_SOURCE_SHA256[
                regime_id
            ]
        )
        assert source["parent_inventory"]["count"] == 251
        assert source["singleton_inventory"]["count"] == 314
        assert source["same_cutoff_exact_reference"]["energy"] == (
            PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_EXACT_ENERGY[regime_id]
        )
        assert (
            source["same_cutoff_exact_reference"]["controller_input"]
            is False
        )


def test_l3_nph3_page12_and_append_protocols_are_matched_except_method(
    weak_sector_sources,
) -> None:
    for regime_id, (problem, source) in weak_sector_sources.items():
        execution_id = f"l3_weak_sector__{regime_id}__nph3"
        ra = build_resolved_ra_protocol(
            problem,
            build_paper_i_l3_page12_request(),
            materialization_authority=_authority(
                execution_id=execution_id + "__ra",
                source_sha256=source["sha256"],
                append=False,
            ),
        )
        append_request = AppendAdaptRequest(
            adapter=(
                PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter()
            ),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=50),
                resume=FreshStart(),
            ),
        )
        append = build_resolved_append_protocol(
            problem,
            append_request,
            materialization_authority=_authority(
                execution_id=execution_id + "__append",
                source_sha256=source["sha256"],
                append=True,
            ),
        )

        restored_ra_request = ra_adapt_request_from_mapping(
            ra.request.to_dict()
        )
        restored_append_request = append_adapt_request_from_mapping(
            append.request.to_dict()
        )
        assert isinstance(
            restored_ra_request.adapter,
            PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
        )
        assert isinstance(
            restored_append_request.adapter,
            PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
        )
        assert restored_ra_request.to_dict() == ra.request.to_dict()
        assert restored_append_request.to_dict() == append.request.to_dict()

        assert ra.route_contract["sha256"] == (
            PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256
        )
        assert ra.request.method.pruning.kind == "off"
        assert ra.request.method.beam.kind == "off"
        assert ra.request.method.insertion.kind == "plateau_commutation"
        assert append.selector_identity == APPEND_CONVENTIONAL_SELECTOR_ID
        assert append.lineage_authority["ra_staged_funnel_invoked"] is False
        assert append.optimizer == ra.optimizer == "powell"
        assert append.optimizer_maxiter == ra.optimizer_maxiter == 200
        assert append.seeds["adapt"] == ra.seeds["adapt"] == 7
        assert append.problem == ra.problem
        assert append.parent_inventory == ra.parent_inventory
        assert append.executable_pool == ra.executable_pool
        assert append.horizon == ra.horizon == 50
        assert append.source_locks[PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY] == (
            source["sha256"]
        )
        restored = resolved_ra_adapt_protocol_from_mapping(append.to_dict())
        assert restored.sha256 == append.sha256
        assert isinstance(
            restored.request.adapter,
            PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
        )


def test_l3_append_requires_the_exact_named_adapter_and_source_lock(
    weak_sector_sources,
) -> None:
    problem, source = weak_sector_sources["intermediate_weak"]
    with pytest.raises(ValueError, match="exact named source-locked L=3"):
        run_append_adapt(problem, AppendAdaptRequest())

    request = AppendAdaptRequest(
        adapter=PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter(),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=50),
            resume=FreshStart(),
        ),
    )
    with pytest.raises(ValueError, match="application source lock"):
        build_resolved_append_protocol(
            problem,
            request,
            materialization_authority=_authority(
                execution_id="l3_wrong_lock",
                source_sha256="f" * 64,
                append=True,
            ),
        )

    drifted = replace(
        problem,
        request=replace(problem.request, u=1.2500001),
    )
    with pytest.raises(ValueError, match="three exact weak-Holstein"):
        build_resolved_append_protocol(
            drifted,
            request,
            materialization_authority=_authority(
                execution_id="l3_drifted_problem",
                source_sha256=source["sha256"],
                append=True,
            ),
        )


def test_l3_nph3_route_pool_cache_is_opt_in_and_byte_identical(
    weak_sector_sources,
    monkeypatch,
) -> None:
    from pipelines.static_adapt.ra_adapt import l3_page12 as application

    problem, _source = weak_sector_sources["weak_weak"]
    adapter = PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter()
    cache = getattr(application, "_L3_EXECUTABLE_POOL_MEMORY_CACHE", None)
    if cache is not None:
        cache.clear()

    original = application.build_paper_i_l3_page12_guarded_single_pauli_pool
    calls = 0

    def counted_builder(problem):
        nonlocal calls
        calls += 1
        return original(problem)

    monkeypatch.setattr(
        application,
        "build_paper_i_l3_page12_guarded_single_pauli_pool",
        counted_builder,
    )
    monkeypatch.setenv("STATIC_ADAPT_L3_ROUTE_POOL_CACHE", "memory")
    cached_first = adapter.executable_pool(problem)
    cached_second = adapter.executable_pool(problem)
    assert calls == 1

    monkeypatch.setenv("STATIC_ADAPT_L3_ROUTE_POOL_CACHE", "off")
    uncached_first = adapter.executable_pool(problem)
    uncached_second = adapter.executable_pool(problem)
    assert calls == 3

    expected = {
        "receipt": cached_first.receipt.to_dict(),
        "candidates": [
            candidate.manifest_row()
            for candidate in cached_first.candidates
        ],
        "metadata": dict(cached_first.metadata),
    }
    for inventory in (
        cached_second,
        uncached_first,
        uncached_second,
    ):
        assert {
            "receipt": inventory.receipt.to_dict(),
            "candidates": [
                candidate.manifest_row()
                for candidate in inventory.candidates
            ],
            "metadata": dict(inventory.metadata),
        } == expected

    monkeypatch.setenv("STATIC_ADAPT_L3_ROUTE_POOL_CACHE", "disk")
    with pytest.raises(
        ValueError,
        match="STATIC_ADAPT_L3_ROUTE_POOL_CACHE must be memory or off",
    ):
        adapter.executable_pool(problem)
    application._L3_EXECUTABLE_POOL_MEMORY_CACHE.clear()
