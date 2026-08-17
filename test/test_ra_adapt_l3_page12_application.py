from __future__ import annotations

from dataclasses import replace

import pytest

from pipelines.static_adapt.ra_adapt import (
    RAAdaptOperationalControls,
    RAAdaptRequest,
    build_paper_i_l3_page12_problem,
    build_paper_i_l3_page12_request,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RA_ADAPT_PROTOCOL_SCHEMA_V1,
    RA_STAGED_SELECTOR_ID,
    RESOURCE_WEIGHTING_ALL_PHASE,
    _attach_validated_bundle_protocol_authority,
    _mint_bundle_protocol_materialization_authority,
    bundle_protocol_materialization_receipt,
    resolved_ra_adapt_protocol_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import (
    build_resolved_ra_protocol,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt.l3_page12 import (
    PAPER_I_L3_PAGE12_ADAPTER_ID,
    PAPER_I_L3_PAGE12_ALGORITHM_ID,
    PAPER_I_L3_PAGE12_APPLICATION_SOURCE_SHA256,
    PAPER_I_L3_PAGE12_EXACT_ENERGY,
    PAPER_I_L3_PAGE12_EXACT_SOURCE_ID,
    PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY,
    PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256,
    PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
    paper_i_l3_page12_application_source_contract,
)
from pipelines.static_adapt.ra_adapt.pools import (
    PAPER_I_L3_PAGE12_PARENT_LABELS_SHA256,
    PAPER_I_L3_PAGE12_PARENT_POOL_SHA256,
    PAPER_I_L3_PAGE12_SINGLETON_LABELS_SHA256,
    PAPER_I_L3_PAGE12_SINGLETON_POOL_SHA256,
    build_guarded_single_pauli_pool,
)


@pytest.fixture(scope="module")
def l3_problem():
    return build_paper_i_l3_page12_problem()


@pytest.fixture(scope="module")
def l3_source(l3_problem):
    return paper_i_l3_page12_application_source_contract(l3_problem)


def _authority(
    source_sha256: str,
    *,
    protocol_sha256: str | None = None,
):
    source_locks = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "paper_i_l3_page12_intermediate_weak",
        "cell_source_lock_sha256": "3" * 64,
        "ed_cutoff_reference_sha256": "4" * 64,
        PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY: source_sha256,
    }
    receipt = bundle_protocol_materialization_receipt(
        bundle_id="paper_i_l3_page12_r50_test_bundle",
        bundle_manifest_sha256="5" * 64,
        source_locks_sha256="1" * 64,
        source_lock_refs=source_locks,
        cell_id="paper_i_l3_page12_intermediate_weak",
        source_lock_id="paper_i_l3_page12_intermediate_weak",
        protocol_schema=RA_ADAPT_PROTOCOL_SCHEMA_V1,
        algorithm_id=PAPER_I_L3_PAGE12_ALGORITHM_ID,
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        selector_identity=RA_STAGED_SELECTOR_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )
    return _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=source_locks,
        protocol_sha256=protocol_sha256,
    )


def test_named_l3_problem_pool_and_same_cutoff_reference_are_locked(
    l3_problem,
    l3_source,
) -> None:
    assert l3_problem.request.num_sites == 3
    assert l3_problem.request.n_ph_max == 1
    assert l3_problem.sector.num_particles == (2, 1)
    assert l3_problem.layout.total_qubits == 9

    parent = l3_source["parent_inventory"]
    singleton = l3_source["singleton_inventory"]
    exact = l3_source["same_cutoff_exact_reference"]
    assert parent == {
        "count": 199,
        "ordered_labels_sha256": PAPER_I_L3_PAGE12_PARENT_LABELS_SHA256,
        "ordered_pool_sha256": PAPER_I_L3_PAGE12_PARENT_POOL_SHA256,
        "observed_receipt_sha256": parent["observed_receipt_sha256"],
    }
    assert singleton == {
        "count": 74,
        "ordered_labels_sha256": PAPER_I_L3_PAGE12_SINGLETON_LABELS_SHA256,
        "ordered_pool_sha256": PAPER_I_L3_PAGE12_SINGLETON_POOL_SHA256,
        "observed_receipt_sha256": singleton["observed_receipt_sha256"],
    }
    assert exact["source_id"] == PAPER_I_L3_PAGE12_EXACT_SOURCE_ID
    assert exact["n_ph_max"] == l3_problem.request.n_ph_max
    assert exact["energy"] == PAPER_I_L3_PAGE12_EXACT_ENERGY
    assert exact["controller_input"] is False
    assert l3_source["sha256"] == PAPER_I_L3_PAGE12_APPLICATION_SOURCE_SHA256
    assert l3_source["sha256"] == (
        paper_i_l3_page12_application_source_contract(l3_problem)["sha256"]
    )


def test_ordinary_l2_pool_and_facade_guards_do_not_admit_generic_l3(
    l3_problem,
) -> None:
    with pytest.raises(ValueError, match="canonical Paper-I HH L=2"):
        build_guarded_single_pauli_pool(l3_problem)
    with pytest.raises(ValueError, match="canonical Paper-I HH L=2"):
        run_ra_adapt(l3_problem, RAAdaptRequest())

    drifted = replace(
        l3_problem,
        request=replace(l3_problem.request, u=1.2500001),
    )
    named = build_paper_i_l3_page12_request()
    with pytest.raises(ValueError, match="exact intermediate--weak HH point"):
        named.adapter.executable_pool(drifted)


def test_l3_protocol_requires_exact_application_source_lock_and_roundtrips(
    l3_problem,
    l3_source,
) -> None:
    request = build_paper_i_l3_page12_request()
    with pytest.raises(ValueError, match="application source-lock digest"):
        build_resolved_ra_protocol(
            l3_problem,
            request,
            materialization_authority=_authority("f" * 64),
        )

    protocol = build_resolved_ra_protocol(
        l3_problem,
        request,
        materialization_authority=_authority(l3_source["sha256"]),
    )
    assert protocol.algorithm_id == PAPER_I_L3_PAGE12_ALGORITHM_ID
    assert protocol.adapter_id == PAPER_I_L3_PAGE12_ADAPTER_ID
    assert protocol.active_gradient_policy == ACTIVE_GRADIENT_STATIONARY
    assert protocol.resource_weighting_scope == RESOURCE_WEIGHTING_ALL_PHASE
    assert protocol.horizon == 50
    assert protocol.parent_inventory.count == 199
    assert protocol.executable_pool.count == 74
    assert (
        protocol.source_locks[PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY]
        == l3_source["sha256"]
    )
    assert protocol.stopping_rule == {"maximum_controller_rounds": 50}

    route = protocol.route_contract
    assert route is not None
    assert route["sha256"] == PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256
    invariants = route["semantic_invariants"]
    execution = route["execution_settings"]
    assert invariants["candidate_adapter_id"] == PAPER_I_L3_PAGE12_ADAPTER_ID
    assert invariants["phase0_active"] is True
    assert invariants["phase0_fubini_metric_active"] is False
    assert invariants["physical_operator_lanes_active"] is False
    assert invariants["phase_ii_compile_cost_source"] == "backend_transpile_v1"
    assert invariants["phase_iii_compile_cost_source"] == "backend_transpile_v1"
    assert invariants["plateau_prior_mean_decrease_ratio_threshold"] == 1.0e-4
    assert execution["adapt_inner_optimizer"] == "POWELL"
    assert execution["adapt_maxiter"] == 200
    assert execution["adapt_seed"] == 7

    restored = resolved_ra_adapt_protocol_from_mapping(protocol.to_dict())
    assert restored.sha256 == protocol.sha256
    assert isinstance(
        restored.request.adapter,
        PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
    )


def test_named_l3_bundle_executes_one_real_controller_round(
    l3_problem,
    l3_source,
) -> None:
    protocol = build_resolved_ra_protocol(
        l3_problem,
        build_paper_i_l3_page12_request(),
        materialization_authority=_authority(l3_source["sha256"]),
    )
    protocol = _attach_validated_bundle_protocol_authority(
        protocol,
        _authority(
            l3_source["sha256"],
            protocol_sha256=protocol.sha256,
        ),
    )

    result = run_ra_adapt(
        l3_problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=1,
        ),
    )

    assert result.protocol == protocol
    assert result.run.stop.completed_controller_rounds == 1
    assert len(result.run.accepted_trajectory) == 1
    assert result.run.paper_i_summary is not None
