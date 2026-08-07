from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.ra_adapt.adapters import (
    H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
    H2OLinearFDSymmetryCompleteCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    ra_adapt_request_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import (
    _sr_request,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.ra_adapt.h2o_application import (
    _problem,
    build_h2o_ra_request,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1,
)
from pipelines.static_adapt.sr_snake._context import (
    _resolve_execution_context,
)


_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = (
    _ROOT
    / "chtc"
    / "paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_d50_20260724"
    / "input"
    / "h2o_fixture.json"
)


@pytest.fixture(scope="module")
def h2o_protocol(tmp_path_factory: pytest.TempPathFactory):
    problem = _problem(_FIXTURE)
    request = build_h2o_ra_request(
        problem=problem,
        output_dir=tmp_path_factory.mktemp("h2o_ra"),
        maximum_controller_rounds=50,
        gradient_tolerance=5.0e-7,
        exact_absolute_tolerance=1.6e-3,
        exact_confirmation_controller_rounds=2,
    )
    return problem, request, build_resolved_ra_protocol(problem, request)


def test_h2o_ra_defaults_to_staged_sector_complete_pauli_blocks(
    h2o_protocol,
) -> None:
    _, request, protocol = h2o_protocol
    contract = protocol.route_contract
    execution = contract["execution_settings"]
    invariants = contract["semantic_invariants"]

    assert isinstance(
        request.adapter,
        H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
    )
    assert protocol.candidate_representation == CANDIDATE_REPRESENTATION_MACRO
    assert protocol.parent_inventory.count == 448
    assert protocol.executable_pool.count == 448
    assert any(
        label.startswith("el::")
        for label in protocol.executable_pool.ordered_labels
    )
    assert any(
        label.startswith("coupled::")
        for label in protocol.executable_pool.ordered_labels
    )
    assert any(
        label.startswith("conditional::")
        for label in protocol.executable_pool.ordered_labels
    )
    assert execution["phase3_runtime_split_mode"] == "off"
    assert execution["phase3_runtime_split_selection_mode"] == "off"
    assert invariants["singleton_admission"] is True
    assert invariants["staged_singleton_exposure"] is False
    assert invariants["staged_sector_complete_pauli_block_exposure"] is True
    assert invariants["raw_single_pauli_child_exposure"] is False
    assert invariants["guarded_single_pauli_child_exposure"] is False
    assert invariants["sector_complete_pauli_block_exposure"] is True
    assert invariants["candidate_generator_semantics"] == (
        "sector_complete_pauli_block_v1"
    )
    assert protocol.stopping_rule["gradient_tolerance"] == pytest.approx(
        5.0e-7
    )
    assert protocol.stopping_rule["exact_ed_target"][
        "confirmation_controller_rounds"
    ] == 2
    assert protocol.problem.n_ph_max == 1


def test_h2o_ra_sector_complete_block_adapter_round_trips(
    h2o_protocol,
) -> None:
    _, request, _ = h2o_protocol
    restored = ra_adapt_request_from_mapping(request.to_dict())

    assert isinstance(
        restored.adapter,
        H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
    )
    assert restored == request


def test_h2o_sector_complete_blocks_group_unsafe_jw_children(
    h2o_protocol,
) -> None:
    problem, request, _ = h2o_protocol
    parent_inventory = request.adapter.parent_inventory(problem)
    electronic_single = next(
        candidate
        for candidate in parent_inventory.candidates
        if candidate.label.startswith("el::uccsd_sing(")
    )

    exposed = request.adapter.expose_children(
        (electronic_single,),
        problem=problem,
    )

    assert exposed.receipt.count == 1
    block = exposed.candidates[0]
    assert len(block.serialized_terms_exyz) == 2
    assert block.execution_mode == "grouped_exact"
    assert block.symmetry_receipt["grouped_preserves_fixed_counts"] is True
    assert block.symmetry_receipt["execution_preserves_fixed_counts"] is True
    assert exposed.metadata["raw_unsafe_single_pauli_words_rejected"] is True


def test_h2o_raw_single_pauli_route_fails_closed(tmp_path: Path) -> None:
    problem = _problem(_FIXTURE)

    with pytest.raises(ValueError, match="sector-complete"):
        build_h2o_ra_request(
            problem=problem,
            output_dir=tmp_path,
            maximum_controller_rounds=1,
            gradient_tolerance=5.0e-7,
            exact_absolute_tolerance=1.6e-3,
            candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        )


def test_h2o_ra_numerical_gate_accepts_only_the_named_application(
    h2o_protocol,
) -> None:
    problem, request, protocol = h2o_protocol
    route_contract = dict(protocol.route_contract)
    route_sha256 = route_contract.pop("sha256")
    route_profile = route_contract["route_profile"]
    context = _resolve_execution_context(
        problem,
        _sr_request(request),
        route_override=(
            route_profile,
            route_profile,
            route_contract,
            route_sha256,
        ),
        candidate_adapter=request.adapter,
    )

    runtime_kwargs = context.canonical_runtime_kwargs()
    validated = adapt_pipeline._validated_ra_adapt_route_contract(
        runtime_kwargs
    )
    assert validated is not None
    assert validated["semantic_invariants"][
        "application_lane"
    ] == (
        "paper_iv_h2o_linear_fd_ra_adapt_sector_complete_pauli_block_v1"
    )
    assert adapt_pipeline._default_no_prune_active_phase_flags(
        runtime_kwargs
    ) == (True, True, True)
    child_padding = adapt_pipeline._default_no_prune_child_padding_config(
        runtime_kwargs
    )
    assert child_padding.policy == (
        ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1
    )


def test_h2o_ra_macro_representation_remains_explicitly_available(
    tmp_path: Path,
) -> None:
    problem = _problem(_FIXTURE)
    request = build_h2o_ra_request(
        problem=problem,
        output_dir=tmp_path,
        maximum_controller_rounds=1,
        gradient_tolerance=5.0e-7,
        exact_absolute_tolerance=1.6e-3,
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
    )
    protocol = build_resolved_ra_protocol(problem, request)

    assert isinstance(
        request.adapter,
        H2OLinearFDSymmetryCompleteCandidateAdapter,
    )
    assert protocol.candidate_representation == CANDIDATE_REPRESENTATION_MACRO
    invariants = protocol.route_contract["semantic_invariants"]
    assert invariants["staged_singleton_exposure"] is False
    assert invariants["guarded_single_pauli_child_exposure"] is False
