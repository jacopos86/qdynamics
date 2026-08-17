from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import pytest

from pipelines.static_adapt.sr_snake._selection import (
    _CandidatePositionRecord,
    _CombinatorialBatchAdmissionDecision,
    _CombinatorialBatchProposalReceipt,
    _CombinatorialBatchSelectionEvaluation,
    _EstimatorEventIdentity,
    _GreedyBatchAdmissionDecision,
    _GreedyBatchProposalReceipt,
    _GreedyBatchSelectionEvaluation,
    _PhaseSelectionReceipt,
    _PredictiveCostReceipt,
    _ResponseReceipt,
    _SRControllerState,
    _SelectionEvaluation,
    _SelectionWorkspace,
    _ShortlistRankReceipt,
    _TrustSolveReceipt,
    _assert_phase_lineage,
    _build_candidate_domain,
    _select_singleton,
    _select_combinatorial_batch,
    _select_greedy_batch,
    _uses_default_singleton_selection,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256,
)


class _TestSelectionKernel:
    def __init__(
        self,
        evaluate: Callable[
            [tuple[_CandidatePositionRecord, ...]],
            _SelectionEvaluation,
        ],
    ) -> None:
        self._evaluate = evaluate
        self.accepted_operators = ["gen:accepted-0", "gen:accepted-1"]
        self.logical_parameters = [0.25, -0.5]
        self.statevector = [complex(1.0), complex(0.0)]
        self.estimator_events: list[str] = []
        self.runtime_sidecar: dict[str, object] = {}

    def accepted_state_snapshot(self) -> object:
        return (
            tuple(self.accepted_operators),
            tuple(self.logical_parameters),
            tuple(self.statevector),
        )

    def evaluate(
        self,
        domain: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        return self._evaluate(domain)


def _state() -> _SRControllerState:
    return _SRControllerState(
        controller_round=3,
        accepted_operator_ids=("gen:accepted-0", "gen:accepted-1"),
        accepted_insertion_positions=(0, 1),
        logical_parameter_ids=("logical:0", "logical:1"),
        logical_parameter_values=(0.25, -0.5),
        runtime_parameter_ids=("runtime:0", "runtime:1"),
        runtime_parameter_values=(0.25, -0.5),
        accepted_energy=-1.125,
        accepted_state_fingerprint="state:accepted",
        available_generator_ids=("gen:parent-a", "gen:parent-b"),
        selection_counts=(("gen:parent-a", 0), ("gen:parent-b", 1)),
        phase_live=(True, True, True),
        trust_state_identity="trust:before-round-3",
        optimizer_memory_identity="optimizer:before-round-3",
        estimator_prefix_identity="ledger:event:41",
    )


def _domain_records() -> tuple[_CandidatePositionRecord, ...]:
    return (
        _CandidatePositionRecord(
            domain_record_id="domain:parent-a@2",
            generator_id="gen:parent-a",
            parent_generator_id=None,
            pool_index=4,
            pool_label="pool:a",
            insertion_position=2,
            symmetry_identity="symmetry:number-spin",
            lineage_identity=("gen:parent-a",),
        ),
        _CandidatePositionRecord(
            domain_record_id="domain:parent-b@2",
            generator_id="gen:parent-b",
            parent_generator_id=None,
            pool_index=7,
            pool_label="pool:b",
            insertion_position=2,
            symmetry_identity="symmetry:number-spin",
            lineage_identity=("gen:parent-b",),
        ),
    )


def _child_record(
    root: _CandidatePositionRecord,
    generator_id: str = "gen:child-a0",
) -> _CandidatePositionRecord:
    return _CandidatePositionRecord(
        domain_record_id=root.domain_record_id,
        generator_id=generator_id,
        parent_generator_id=root.generator_id,
        pool_index=root.pool_index,
        pool_label=f"{root.pool_label}::child[0]",
        insertion_position=root.insertion_position,
        symmetry_identity="symmetry:number-spin:projected",
        lineage_identity=(root.generator_id, generator_id),
    )


def _ranking(
    records: tuple[_CandidatePositionRecord, ...],
    *,
    primary_scores: tuple[float, ...] | None = None,
) -> tuple[_ShortlistRankReceipt, ...]:
    scores = (
        primary_scores
        if primary_scores is not None
        else tuple(float(len(records) - index) for index in range(len(records)))
    )
    return tuple(
        _ShortlistRankReceipt(
            record_key=(record.domain_record_id, record.generator_id),
            shortlist_rank=index,
            primary_score=float(score),
            tie_break_score=float(score),
            pool_index=record.pool_index,
            insertion_position=record.insertion_position,
        )
        for index, (record, score) in enumerate(
            zip(records, scores, strict=True),
            start=1,
        )
    )


def _evaluation(
    domain: tuple[_CandidatePositionRecord, ...],
) -> _SelectionEvaluation:
    child = _child_record(domain[0])
    phase_i = _PhaseSelectionReceipt(
        phase="phase_i",
        population=domain,
        shortlist=domain,
        shortlist_ranking=_ranking(domain),
        estimator_event_ids=("event:gradient:a", "event:gradient:b"),
    )
    phase_ii = _PhaseSelectionReceipt(
        phase="phase_ii",
        population=(child, domain[1]),
        shortlist=(child,),
        shortlist_ranking=_ranking((child,)),
        estimator_event_ids=("event:metric:a", "event:metric:b"),
    )
    phase_iii = _PhaseSelectionReceipt(
        phase="phase_iii",
        population=(child,),
        shortlist=(child,),
        shortlist_ranking=_ranking((child,)),
        estimator_event_ids=("event:response:a",),
    )
    events = (
        _EstimatorEventIdentity(
            sequence_index=42,
            occurrence_id="event:gradient:a",
            reuse_identity=None,
        ),
        _EstimatorEventIdentity(
            sequence_index=43,
            occurrence_id="event:gradient:b",
            reuse_identity=None,
        ),
        _EstimatorEventIdentity(
            sequence_index=44,
            occurrence_id="event:metric:a",
            reuse_identity="event:gradient:a",
        ),
        _EstimatorEventIdentity(
            sequence_index=45,
            occurrence_id="event:metric:b",
            reuse_identity="event:gradient:b",
        ),
        _EstimatorEventIdentity(
            sequence_index=46,
            occurrence_id="event:response:a",
            reuse_identity="event:metric:a",
        ),
    )
    return _SelectionEvaluation(
        phase_i=phase_i,
        phase_ii=phase_ii,
        phase_iii=phase_iii,
        selected=child,
        response=_ResponseReceipt(
            identity="response:child-a0",
            coordinate_ids=("logical:0", "logical:1", "proposal:child-a0"),
            supported_rank=3,
            supported_dimension=3,
        ),
        trust=_TrustSolveReceipt(
            identity="trust:child-a0",
            solver_identity="supported_projected_generalized_trust_v1",
            response_identity="response:child-a0",
            supported_rank=3,
            proposed_coordinate_values=(0.0, 0.0, -0.125),
        ),
        predictive_cost=_PredictiveCostReceipt(
            identity="cost:child-a0",
            policy_identity="symmetric_candidate_cost_v1",
            value=0.75,
        ),
        estimator_events=events,
    )


def _phase0_evaluation(
    domain: tuple[_CandidatePositionRecord, ...],
) -> _SelectionEvaluation:
    child = _child_record(domain[0])
    phase0 = _PhaseSelectionReceipt(
        phase="phase0",
        population=domain,
        shortlist=(domain[0],),
        shortlist_ranking=_ranking((domain[0],)),
        estimator_event_ids=("event:phase0:a", "event:phase0:b"),
    )
    phase_i = _PhaseSelectionReceipt(
        phase="phase_i",
        population=(child,),
        shortlist=(child,),
        shortlist_ranking=_ranking((child,)),
        estimator_event_ids=("event:phase-i:a0",),
    )
    phase_ii = _PhaseSelectionReceipt(
        phase="phase_ii",
        population=(child,),
        shortlist=(child,),
        shortlist_ranking=_ranking((child,)),
        estimator_event_ids=("event:phase-ii:a0",),
    )
    phase_iii = _PhaseSelectionReceipt(
        phase="phase_iii",
        population=(child,),
        shortlist=(child,),
        shortlist_ranking=_ranking((child,)),
        estimator_event_ids=("event:phase-iii:a0",),
    )
    occurrence_ids = (
        *phase0.estimator_event_ids,
        *phase_i.estimator_event_ids,
        *phase_ii.estimator_event_ids,
        *phase_iii.estimator_event_ids,
    )
    return _SelectionEvaluation(
        phase0=phase0,
        phase_i=phase_i,
        phase_ii=phase_ii,
        phase_iii=phase_iii,
        selected=child,
        response=_ResponseReceipt(
            identity="response:phase0-child-a0",
            coordinate_ids=("logical:0", "proposal:child-a0"),
            supported_rank=2,
            supported_dimension=2,
        ),
        trust=_TrustSolveReceipt(
            identity="trust:phase0-child-a0",
            solver_identity="supported_projected_generalized_trust_v1",
            response_identity="response:phase0-child-a0",
            supported_rank=2,
            proposed_coordinate_values=(0.0, -0.125),
        ),
        predictive_cost=_PredictiveCostReceipt(
            identity="cost:phase0-child-a0",
            policy_identity="symmetric_candidate_cost_v1",
            value=0.75,
        ),
        estimator_events=tuple(
            _EstimatorEventIdentity(
                sequence_index=42 + index,
                occurrence_id=occurrence_id,
                reuse_identity=None,
            )
            for index, occurrence_id in enumerate(occurrence_ids)
        ),
    )


def _batch_evaluation(
    domain: tuple[_CandidatePositionRecord, ...],
) -> _GreedyBatchSelectionEvaluation:
    first = _child_record(domain[0])
    second = _CandidatePositionRecord(
        domain_record_id=domain[1].domain_record_id,
        generator_id="gen:child-b0",
        parent_generator_id=domain[1].generator_id,
        pool_index=domain[1].pool_index,
        pool_label=f"{domain[1].pool_label}::child[0]",
        insertion_position=domain[1].insertion_position,
        symmetry_identity="symmetry:number-spin:projected",
        lineage_identity=(domain[1].generator_id, "gen:child-b0"),
    )
    selected = (first, second)
    phase_i = _PhaseSelectionReceipt(
        phase="phase_i",
        population=domain,
        shortlist=domain,
        shortlist_ranking=_ranking(domain),
        estimator_event_ids=("event:gradient:a", "event:gradient:b"),
    )
    phase_ii = _PhaseSelectionReceipt(
        phase="phase_ii",
        population=selected,
        shortlist=selected,
        shortlist_ranking=_ranking(selected),
        estimator_event_ids=("event:metric:a", "event:metric:b"),
    )
    phase_iii = _PhaseSelectionReceipt(
        phase="phase_iii",
        population=selected,
        shortlist=selected,
        shortlist_ranking=_ranking(selected),
        estimator_event_ids=("event:joint-response",),
    )
    events = tuple(
        _EstimatorEventIdentity(
            sequence_index=42 + index,
            occurrence_id=occurrence_id,
            reuse_identity=None,
        )
        for index, occurrence_id in enumerate(
            (
                "event:gradient:a",
                "event:gradient:b",
                "event:metric:a",
                "event:metric:b",
                "event:joint-response",
            )
        )
    )
    response = _ResponseReceipt(
        identity="response:greedy-batch",
        coordinate_ids=(
            "logical:0",
            "logical:1",
            "proposal:child-a0",
            "proposal:child-b0",
        ),
        supported_rank=4,
        supported_dimension=4,
    )
    return _GreedyBatchSelectionEvaluation(
        phase_i=phase_i,
        phase_ii=phase_ii,
        phase_iii=phase_iii,
        selected=selected,
        response=response,
        trust=_TrustSolveReceipt(
            identity="trust:greedy-batch",
            solver_identity="supported_projected_generalized_trust_v1",
            response_identity=response.identity,
            supported_rank=4,
            proposed_coordinate_values=(0.0, 0.0, -0.1, -0.05),
        ),
        predictive_cost=_PredictiveCostReceipt(
            identity="cost:greedy-batch",
            policy_identity="symmetric_candidate_cost_v1",
            value=1.25,
        ),
        proposal=_GreedyBatchProposalReceipt(
            identity="proposal:greedy-batch",
            maximum_size=3,
            search_window_size=None,
            selected_record_ids=tuple(
                record.domain_record_id for record in selected
            ),
            score=0.5,
            modeled_energy_decrease=0.75,
            predictive_cost_excess=0.5,
            denominator=1.5,
            geometry_identity="geometry:greedy-batch",
            evaluated_subset_count=3,
        ),
        estimator_events=events,
    )


def _combinatorial_evaluation(
    domain: tuple[_CandidatePositionRecord, ...],
) -> _CombinatorialBatchSelectionEvaluation:
    greedy = _batch_evaluation(domain)
    proposal = _CombinatorialBatchProposalReceipt(
        identity="proposal:combinatorial-batch",
        maximum_size=3,
        search_window_size=6,
        ranked_population_count=2,
        ranked_window_count=2,
        selected_record_ids=tuple(
            record.domain_record_id for record in greedy.selected
        ),
        score=0.5,
        modeled_energy_decrease=0.75,
        predictive_cost_excess=0.5,
        denominator=1.5,
        geometry_identity="geometry:combinatorial-batch",
        evaluated_subset_count=3,
        subset_counts_considered=((1, 2), (2, 1)),
        subset_counts_evaluated=((1, 2), (2, 1)),
        subset_counts_feasible=((1, 2), (2, 1)),
    )
    return _CombinatorialBatchSelectionEvaluation(
        phase_i=greedy.phase_i,
        phase_ii=greedy.phase_ii,
        phase_iii=greedy.phase_iii,
        selected=greedy.selected,
        response=greedy.response,
        trust=greedy.trust,
        predictive_cost=greedy.predictive_cost,
        proposal=proposal,
        estimator_events=greedy.estimator_events,
    )


def test_candidate_domain_is_deterministic_and_independent_of_ranking() -> None:
    records = _domain_records()

    first = _build_candidate_domain(records)
    second = _build_candidate_domain(tuple(reversed(records)))

    assert first == records
    assert second == tuple(reversed(records))
    assert tuple(row.domain_record_id for row in first) == (
        "domain:parent-a@2",
        "domain:parent-b@2",
    )
    with pytest.raises(FrozenInstanceError):
        first[0].pool_index = 99  # type: ignore[misc]


def test_phase_shortlist_keeps_population_order_and_validates_score_rank() -> None:
    population = _domain_records()
    ranked_shortlist = tuple(reversed(population))
    receipt = _PhaseSelectionReceipt(
        phase="phase_i",
        population=population,
        shortlist=ranked_shortlist,
        shortlist_ranking=_ranking(
            ranked_shortlist,
            primary_scores=(2.0, 1.0),
        ),
        estimator_event_ids=(),
    )

    _assert_phase_lineage(
        receipt=receipt,
        domain_by_id={
            record.domain_record_id: record for record in population
        },
    )

    assert receipt.population == population
    assert receipt.shortlist == ranked_shortlist
    assert tuple(
        rank.record_key for rank in receipt.shortlist_ranking
    ) == tuple(
        (record.domain_record_id, record.generator_id)
        for record in ranked_shortlist
    )

    with pytest.raises(ValueError, match="deterministic score rank"):
        _assert_phase_lineage(
            receipt=_PhaseSelectionReceipt(
                phase="phase_i",
                population=population,
                shortlist=ranked_shortlist,
                shortlist_ranking=_ranking(
                    ranked_shortlist,
                    primary_scores=(1.0, 2.0),
                ),
                estimator_event_ids=(),
            ),
            domain_by_id={
                record.domain_record_id: record for record in population
            },
        )


def test_phase_shortlist_validates_macro_identity_then_position_rank() -> None:
    parent_a, parent_b = _domain_records()
    population = (
        replace(
            parent_a,
            domain_record_id="domain:parent-a@0",
            insertion_position=0,
        ),
        replace(
            parent_a,
            domain_record_id="domain:parent-a@1",
            insertion_position=1,
        ),
        replace(
            parent_b,
            domain_record_id="domain:parent-b@0",
            insertion_position=0,
        ),
        replace(
            parent_b,
            domain_record_id="domain:parent-b@1",
            insertion_position=1,
        ),
    )
    scores = (4.0, 1.0, 3.0, 2.0)
    identities = ("pool:4", "pool:4", "pool:7", "pool:7")
    ranking = tuple(
        _ShortlistRankReceipt(
            record_key=(record.domain_record_id, record.generator_id),
            shortlist_rank=index,
            primary_score=score,
            tie_break_score=score,
            pool_index=record.pool_index,
            insertion_position=record.insertion_position,
            shortlist_unit="macro_operator_identity",
            shortlist_identity=identity,
            identity_rank=1 if index <= 2 else 2,
            identity_position_rank=1 if index in {1, 3} else 2,
            identity_position_count=2,
        )
        for index, (record, score, identity) in enumerate(
            zip(population, scores, identities, strict=True),
            start=1,
        )
    )
    receipt = _PhaseSelectionReceipt(
        phase="phase_i",
        population=population,
        shortlist=population,
        shortlist_ranking=ranking,
        estimator_event_ids=(),
    )
    domain_by_id = {
        record.domain_record_id: record for record in population
    }

    _assert_phase_lineage(
        receipt=receipt,
        domain_by_id=domain_by_id,
    )

    with pytest.raises(
        ValueError,
        match="macro-identity position ranks are incomplete",
    ):
        _assert_phase_lineage(
            receipt=replace(
                receipt,
                shortlist_ranking=(
                    ranking[0],
                    replace(ranking[1], identity_position_rank=1),
                    *ranking[2:],
                ),
            ),
            domain_by_id=domain_by_id,
        )


def test_default_selection_dispatch_requires_exact_profile_digest_and_no_beam() -> None:
    profile = (
        "supported_projected_generalized_source_metric_no_overlap_trust_"
        "full_response_symmetric_cost_no_prune_v1"
    )
    digest = (
        "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
    )

    assert _uses_default_singleton_selection(
        route_profile=profile,
        route_profile_sha256=digest,
        beam_enabled=False,
    )
    assert not _uses_default_singleton_selection(
        route_profile=profile,
        route_profile_sha256=digest,
        beam_enabled=True,
    )
    assert not _uses_default_singleton_selection(
        route_profile="insertion_commutation_plateau_v1",
        route_profile_sha256=digest,
        beam_enabled=False,
    )
    assert not _uses_default_singleton_selection(
        route_profile=profile,
        route_profile_sha256="wrong",
        beam_enabled=False,
    )
    reduced_profile = (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
    )
    reduced_digest = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256()
    )
    assert _uses_default_singleton_selection(
        route_profile=reduced_profile,
        route_profile_sha256=reduced_digest,
        beam_enabled=False,
    )
    assert not _uses_default_singleton_selection(
        route_profile=reduced_profile,
        route_profile_sha256=reduced_digest,
        beam_enabled=True,
    )
    assert not _uses_default_singleton_selection(
        route_profile=reduced_profile,
        route_profile_sha256="wrong",
        beam_enabled=False,
    )


@pytest.mark.parametrize("invalid_value", (-1.0, float("inf"), float("nan")))
def test_predictive_candidate_cost_requires_finite_nonnegative_value(
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="predictive candidate cost must be finite and non-negative",
    ):
        _PredictiveCostReceipt(
            identity="cost:invalid",
            policy_identity="symmetric_candidate_cost_v1",
            value=invalid_value,
        )


def test_singleton_selection_owns_three_phase_decision_without_state_mutation() -> None:
    state = _state()
    seen_domains: list[tuple[_CandidatePositionRecord, ...]] = []

    def _evaluate(
        domain: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        seen_domains.append(domain)
        return _evaluation(domain)

    kernel = _TestSelectionKernel(_evaluate)
    accepted_state_before = kernel.accepted_state_snapshot()
    decision = _select_singleton(
        state,
        _SelectionWorkspace(
            admissible_records=_domain_records(),
            kernel=kernel,
        ),
    )

    assert kernel.accepted_state_snapshot() == accepted_state_before
    assert seen_domains == [_domain_records()]
    assert decision.controller_state_fingerprint == "state:accepted"
    assert decision.selected.generator_id == "gen:child-a0"
    assert decision.selected.pool_index == 4
    assert decision.selected.insertion_position == 2
    assert decision.selected.symmetry_identity == "symmetry:number-spin:projected"
    assert decision.phase_i.population == _domain_records()
    assert decision.phase_ii.shortlist == (decision.selected,)
    assert decision.phase_iii.shortlist == (decision.selected,)
    assert decision.response.coordinate_ids == (
        "logical:0",
        "logical:1",
        "proposal:child-a0",
    )
    assert decision.response.supported_rank == 3
    assert decision.trust.response_identity == decision.response.identity
    assert decision.predictive_cost.policy_identity == (
        "symmetric_candidate_cost_v1"
    )
    assert tuple(event.occurrence_id for event in decision.estimator_events) == (
        "event:gradient:a",
        "event:gradient:b",
        "event:metric:a",
        "event:metric:b",
        "event:response:a",
    )
    with pytest.raises(FrozenInstanceError):
        decision.selected = _domain_records()[1]  # type: ignore[misc]


def test_optional_phase0_preserves_the_controller_domain_and_accepted_state() -> None:
    state = _state()
    state_before = tuple(
        getattr(state, name) for name in state.__dataclass_fields__
    )
    seen_domains: list[tuple[_CandidatePositionRecord, ...]] = []

    def _evaluate_with_phase0(
        domain: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        seen_domains.append(domain)
        return _phase0_evaluation(domain)

    kernel = _TestSelectionKernel(_evaluate_with_phase0)
    accepted_before = kernel.accepted_state_snapshot()
    decision = _select_singleton(
        state,
        _SelectionWorkspace(
            admissible_records=_domain_records(),
            kernel=kernel,
        ),
    )

    assert seen_domains == [_domain_records()]
    assert kernel.accepted_state_snapshot() == accepted_before
    assert tuple(
        getattr(state, name) for name in state.__dataclass_fields__
    ) == state_before
    assert decision.selected.generator_id == "gen:child-a0"
    assert decision.phase0 is not None
    assert decision.phase0.population == _domain_records()


def test_optional_phase0_population_must_equal_the_original_domain_tuple() -> None:
    domain = _domain_records()

    def _evaluate_with_incomplete_phase0(
        received: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        evaluation = _phase0_evaluation(received)
        assert evaluation.phase0 is not None
        return replace(
            evaluation,
            phase0=replace(
                evaluation.phase0,
                population=tuple(reversed(received)),
            ),
        )

    with pytest.raises(
        ValueError,
        match="Phase0 population must equal the exact original admissible domain",
    ):
        _select_singleton(
            _state(),
            _SelectionWorkspace(
                admissible_records=domain,
                kernel=_TestSelectionKernel(
                    _evaluate_with_incomplete_phase0
                ),
            ),
        )


def test_optional_phase0_phase_i_must_descend_from_a_retained_root() -> None:
    domain = _domain_records()

    def _evaluate_with_wrong_root(
        received: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        evaluation = _phase0_evaluation(received)
        wrong_child = _child_record(
            received[1],
            generator_id="gen:child-b0",
        )
        phase_i = replace(
            evaluation.phase_i,
            population=(wrong_child,),
            shortlist=(wrong_child,),
            shortlist_ranking=_ranking((wrong_child,)),
        )
        phase_ii = replace(
            evaluation.phase_ii,
            population=(wrong_child,),
            shortlist=(wrong_child,),
            shortlist_ranking=_ranking((wrong_child,)),
        )
        phase_iii = replace(
            evaluation.phase_iii,
            population=(wrong_child,),
            shortlist=(wrong_child,),
            shortlist_ranking=_ranking((wrong_child,)),
        )
        return replace(
            evaluation,
            phase_i=phase_i,
            phase_ii=phase_ii,
            phase_iii=phase_iii,
            selected=wrong_child,
        )

    with pytest.raises(
        ValueError,
        match="did not descend from the Phase0 shortlist lineage",
    ):
        _select_singleton(
            _state(),
            _SelectionWorkspace(
                admissible_records=domain,
                kernel=_TestSelectionKernel(_evaluate_with_wrong_root),
            ),
        )


def test_optional_phase0_rejects_phase_i_sibling_smuggling_into_phase_ii() -> None:
    domain = _domain_records()

    def _evaluate_with_sibling_smuggling(
        received: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        evaluation = _phase0_evaluation(received)
        admitted_child = _child_record(received[0])
        sibling_child = _child_record(
            received[0],
            generator_id="gen:child-a1",
        )
        phase_i = replace(
            evaluation.phase_i,
            population=(admitted_child, sibling_child),
            shortlist=(admitted_child,),
            shortlist_ranking=_ranking((admitted_child,)),
        )
        phase_ii = replace(
            evaluation.phase_ii,
            population=(sibling_child,),
            shortlist=(sibling_child,),
            shortlist_ranking=_ranking((sibling_child,)),
        )
        phase_iii = replace(
            evaluation.phase_iii,
            population=(sibling_child,),
            shortlist=(sibling_child,),
            shortlist_ranking=_ranking((sibling_child,)),
        )
        return replace(
            evaluation,
            phase_i=phase_i,
            phase_ii=phase_ii,
            phase_iii=phase_iii,
            selected=sibling_child,
        )

    with pytest.raises(
        ValueError,
        match="did not preserve shortlisted lineage",
    ):
        _select_singleton(
            _state(),
            _SelectionWorkspace(
                admissible_records=domain,
                kernel=_TestSelectionKernel(
                    _evaluate_with_sibling_smuggling
                ),
            ),
        )


def test_optional_phase0_strict_progression_allows_root_to_child_descent() -> None:
    domain = _domain_records()

    def _evaluate_with_late_child_descent(
        received: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        evaluation = _phase0_evaluation(received)
        root = received[0]
        child = _child_record(root)
        phase_i = replace(
            evaluation.phase_i,
            population=(root,),
            shortlist=(root,),
            shortlist_ranking=_ranking((root,)),
        )
        return replace(evaluation, phase_i=phase_i, selected=child)

    decision = _select_singleton(
        _state(),
        _SelectionWorkspace(
            admissible_records=domain,
            kernel=_TestSelectionKernel(
                _evaluate_with_late_child_descent
            ),
        ),
    )

    assert decision.phase_i.shortlist == (domain[0],)
    assert decision.phase_ii.shortlist[0].generator_id == "gen:child-a0"


def test_optional_phase0_strict_progression_allows_same_key_lineage_extension() -> None:
    domain = _domain_records()

    def _evaluate_with_authenticated_owner_extension(
        received: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        evaluation = _phase0_evaluation(received)
        phase_i_child = evaluation.phase_i.shortlist[0]
        extended_child = replace(
            phase_i_child,
            parent_generator_id="gen:authenticated-owner-a",
            lineage_identity=(
                received[0].generator_id,
                "gen:authenticated-owner-a",
                phase_i_child.generator_id,
            ),
        )
        phase_ii = replace(
            evaluation.phase_ii,
            population=(extended_child,),
            shortlist=(extended_child,),
            shortlist_ranking=_ranking((extended_child,)),
        )
        phase_iii = replace(
            evaluation.phase_iii,
            population=(extended_child,),
            shortlist=(extended_child,),
            shortlist_ranking=_ranking((extended_child,)),
        )
        return replace(
            evaluation,
            phase_ii=phase_ii,
            phase_iii=phase_iii,
            selected=extended_child,
        )

    decision = _select_singleton(
        _state(),
        _SelectionWorkspace(
            admissible_records=domain,
            kernel=_TestSelectionKernel(
                _evaluate_with_authenticated_owner_extension
            ),
        ),
    )

    assert decision.phase_i.shortlist[0].generator_id == (
        decision.phase_ii.shortlist[0].generator_id
    )
    assert decision.phase_i.shortlist[0].lineage_identity != (
        decision.phase_ii.shortlist[0].lineage_identity
    )


def test_legacy_singleton_without_phase0_preserves_root_only_progression() -> None:
    decision = _select_singleton(
        _state(),
        _SelectionWorkspace(
            admissible_records=_domain_records(),
            kernel=_TestSelectionKernel(_evaluation),
        ),
    )

    assert decision.phase_i.shortlist == _domain_records()
    assert decision.phase_ii.shortlist[0].generator_id == "gen:child-a0"


def test_greedy_selection_owns_one_ordered_joint_batch_without_state_mutation() -> None:
    state = _state()
    kernel = _TestSelectionKernel(_batch_evaluation)  # type: ignore[arg-type]
    accepted_state_before = kernel.accepted_state_snapshot()

    decision = _select_greedy_batch(
        state,
        _SelectionWorkspace(
            admissible_records=_domain_records(),
            kernel=kernel,
        ),
        maximum_size=3,
        search_window_size=None,
    )

    assert isinstance(decision, _GreedyBatchAdmissionDecision)
    assert kernel.accepted_state_snapshot() == accepted_state_before
    assert tuple(record.generator_id for record in decision.selected) == (
        "gen:child-a0",
        "gen:child-b0",
    )
    assert decision.phase_iii.shortlist == decision.selected
    assert decision.proposal.selected_record_ids == tuple(
        record.domain_record_id for record in decision.selected
    )
    assert decision.proposal.maximum_size == 3
    assert decision.proposal.search_window_size is None
    assert decision.response.supported_dimension == 4
    assert decision.trust.response_identity == decision.response.identity
    assert tuple(event.occurrence_id for event in decision.estimator_events) == (
        "event:gradient:a",
        "event:gradient:b",
        "event:metric:a",
        "event:metric:b",
        "event:joint-response",
    )


def test_combinatorial_selection_owns_one_exhaustive_subset_decision_without_state_mutation() -> None:
    state = _state()
    kernel = _TestSelectionKernel(  # type: ignore[arg-type]
        _combinatorial_evaluation
    )
    accepted_state_before = kernel.accepted_state_snapshot()

    decision = _select_combinatorial_batch(
        state,
        _SelectionWorkspace(
            admissible_records=_domain_records(),
            kernel=kernel,
        ),
        maximum_size=3,
        search_window_size=6,
    )

    assert isinstance(decision, _CombinatorialBatchAdmissionDecision)
    assert kernel.accepted_state_snapshot() == accepted_state_before
    assert tuple(
        (record.generator_id, record.insertion_position)
        for record in decision.selected
    ) == (
        ("gen:child-a0", 2),
        ("gen:child-b0", 2),
    )
    assert decision.phase_iii.shortlist == decision.selected
    assert decision.proposal.selected_record_ids == tuple(
        record.domain_record_id for record in decision.selected
    )
    assert decision.proposal.maximum_size == 3
    assert decision.proposal.search_window_size == 6
    assert decision.proposal.ranked_population_count == 2
    assert decision.proposal.ranked_window_count == 2
    assert decision.proposal.evaluated_subset_count == 3
    assert decision.proposal.subset_counts_evaluated == (
        (1, 2),
        (2, 1),
    )
    assert decision.response.supported_dimension == 4
    assert decision.trust.response_identity == decision.response.identity


def test_combinatorial_receipt_requires_exhaustive_considered_counts() -> None:
    proposal = _combinatorial_evaluation(_domain_records()).proposal

    with pytest.raises(
        ValueError,
        match="must exhaust every cardinality",
    ):
        replace(
            proposal,
            maximum_size=2,
            search_window_size=4,
            ranked_population_count=4,
            ranked_window_count=4,
            evaluated_subset_count=1,
            subset_counts_considered=((1, 1),),
            subset_counts_evaluated=((1, 1),),
            subset_counts_feasible=((1, 1),),
        )


def test_combinatorial_receipt_allows_generator_duplicate_skips_after_enumeration() -> None:
    proposal = _combinatorial_evaluation(_domain_records()).proposal

    receipt = replace(
        proposal,
        maximum_size=2,
        search_window_size=4,
        ranked_population_count=4,
        ranked_window_count=4,
        evaluated_subset_count=9,
        subset_counts_considered=((1, 4), (2, 6)),
        subset_counts_evaluated=((1, 4), (2, 5)),
        subset_counts_feasible=((1, 4), (2, 3)),
    )

    assert receipt.subset_counts_considered == ((1, 4), (2, 6))
    assert receipt.subset_counts_evaluated == ((1, 4), (2, 5))


@pytest.mark.parametrize(
    "mutation_kind",
    ("operators", "parameters", "statevector"),
)
def test_singleton_selection_rejects_live_accepted_state_mutation(
    mutation_kind: str,
) -> None:
    kernel: _TestSelectionKernel

    def _mutating_evaluate(
        domain: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        if mutation_kind == "operators":
            kernel.accepted_operators.append("gen:illicit")
        elif mutation_kind == "parameters":
            kernel.logical_parameters[0] = 9.0
        else:
            kernel.statevector[0] = complex(0.0)
        return _evaluation(domain)

    kernel = _TestSelectionKernel(_mutating_evaluate)

    with pytest.raises(
        RuntimeError,
        match="mutated live accepted operators, parameters, or state",
    ):
        _select_singleton(
            _state(),
            _SelectionWorkspace(
                admissible_records=_domain_records(),
                kernel=kernel,
            ),
        )


def test_singleton_selection_allows_estimator_and_sidecar_effects() -> None:
    kernel: _TestSelectionKernel

    def _effectful_evaluate(
        domain: tuple[_CandidatePositionRecord, ...],
    ) -> _SelectionEvaluation:
        kernel.estimator_events.append("event:phase-i")
        kernel.runtime_sidecar["selected"] = domain[0].domain_record_id
        return _evaluation(domain)

    kernel = _TestSelectionKernel(_effectful_evaluate)

    decision = _select_singleton(
        _state(),
        _SelectionWorkspace(
            admissible_records=_domain_records(),
            kernel=kernel,
        ),
    )

    assert decision.selected.generator_id == "gen:child-a0"
    assert kernel.estimator_events == ["event:phase-i"]
    assert kernel.runtime_sidecar == {
        "selected": "domain:parent-a@2",
    }


def test_selection_workspace_excludes_transition_and_observation_concerns() -> None:
    assert tuple(_SelectionWorkspace.__dataclass_fields__) == (
        "admissible_records",
        "kernel",
    )
    forbidden = {
        "refit",
        "prune",
        "checkpoint",
        "output",
        "stop",
        "observation",
    }
    assert forbidden.isdisjoint(_SelectionWorkspace.__dataclass_fields__)
