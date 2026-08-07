from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from pipelines.static_adapt.sr_snake._selection import (
    _CandidatePositionRecord,
    _EstimatorEventIdentity,
    _GreedyBatchAdmissionDecision,
    _GreedyBatchProposalReceipt,
    _PhaseSelectionReceipt,
    _PredictiveCostReceipt,
    _ResponseReceipt,
    _ShortlistRankReceipt,
    _SingletonAdmissionDecision,
    _TrustSolveReceipt,
)
from pipelines.static_adapt.sr_snake._transition import (
    _AcceptedGreedyBatchTransition,
    _AcceptedStateSnapshot,
    _AdaptiveTrustUpdateReceipt,
    _AdmissionReceipt,
    _GreedyBatchAdmissionReceipt,
    _GreedyBatchTransitionEvaluation,
    _CheckpointReadyAcceptedStateEvent,
    _NonWorseningReceipt,
    _RecoverabilityPruneReceipt,
    _RoundLedgerClosure,
    _SupportedFSRefitReceipt,
    _TransitionEvaluation,
    _TransitionOperationAudit,
    _TransitionWorkspace,
    _accepted_positions_after_insertion,
    _transition_greedy_batch,
    _transition_singleton,
)


def _decision() -> _SingletonAdmissionDecision:
    selected = _CandidatePositionRecord(
        domain_record_id="record:7",
        generator_id="generator:7",
        parent_generator_id=None,
        pool_index=7,
        pool_label="G7",
        insertion_position=1,
        symmetry_identity="symmetry:7",
        lineage_identity=("generator:7",),
    )
    ranking = (
        _ShortlistRankReceipt(
            record_key=("record:7", "generator:7"),
            shortlist_rank=1,
            primary_score=4.0,
            tie_break_score=3.0,
            pool_index=7,
            insertion_position=1,
        ),
    )
    phase_i = _PhaseSelectionReceipt(
        phase="phase_i",
        population=(selected,),
        shortlist=(selected,),
        shortlist_ranking=ranking,
        estimator_event_ids=("occurrence:0",),
    )
    phase_ii = _PhaseSelectionReceipt(
        phase="phase_ii",
        population=(selected,),
        shortlist=(selected,),
        shortlist_ranking=ranking,
        estimator_event_ids=("occurrence:1",),
    )
    phase_iii = _PhaseSelectionReceipt(
        phase="phase_iii",
        population=(selected,),
        shortlist=(selected,),
        shortlist_ranking=ranking,
        estimator_event_ids=("occurrence:2",),
    )
    return _SingletonAdmissionDecision(
        controller_round=1,
        controller_state_fingerprint="state:before",
        selected=selected,
        phase_i=phase_i,
        phase_ii=phase_ii,
        phase_iii=phase_iii,
        response=_ResponseReceipt(
            identity="response:7",
            coordinate_ids=("logical:0", "logical:1"),
            supported_rank=2,
            supported_dimension=2,
        ),
        trust=_TrustSolveReceipt(
            identity="trust:7",
            solver_identity="supported-generalized",
            response_identity="response:7",
            supported_rank=2,
            proposed_coordinate_values=(0.1, 0.0),
        ),
        predictive_cost=_PredictiveCostReceipt(
            identity="cost:7",
            policy_identity="symmetric",
            value=12.0,
        ),
        estimator_events=(
            _EstimatorEventIdentity(0, "occurrence:0", None),
            _EstimatorEventIdentity(1, "occurrence:1", None),
            _EstimatorEventIdentity(2, "occurrence:2", None),
        ),
    )


def _state_before() -> _AcceptedStateSnapshot:
    return _AcceptedStateSnapshot(
        controller_round=1,
        accepted_operator_ids=("operator:0",),
        accepted_insertion_positions=(0,),
        logical_parameter_ids=("logical:0",),
        logical_parameter_values=(0.1,),
        runtime_parameter_ids=("runtime:0",),
        runtime_parameter_values=(0.1,),
        accepted_energy=-1.0,
        accepted_state_fingerprint="state:before",
        available_generator_ids=("generator:7", "generator:8"),
        selection_counts=(("generator:7", 0), ("generator:8", 0)),
        trust_state_identity="trust-state:before",
        optimizer_memory_identity="optimizer:before",
        estimator_prefix_identity="ledger:before",
    )


def _state_after() -> _AcceptedStateSnapshot:
    return _AcceptedStateSnapshot(
        controller_round=2,
        accepted_operator_ids=("operator:0", "generator:7"),
        accepted_insertion_positions=(0, 1),
        logical_parameter_ids=("logical:0", "logical:1"),
        logical_parameter_values=(0.12, -0.03),
        runtime_parameter_ids=("runtime:0", "runtime:1"),
        runtime_parameter_values=(0.12, -0.03),
        accepted_energy=-1.2,
        accepted_state_fingerprint="state:after",
        available_generator_ids=("generator:8",),
        selection_counts=(("generator:7", 1), ("generator:8", 0)),
        trust_state_identity="trust-state:after",
        optimizer_memory_identity="optimizer:after",
        estimator_prefix_identity="ledger:after",
    )


class _Runtime:
    def __init__(self) -> None:
        self.state = _state_before()

    def accepted_state_snapshot(self) -> _AcceptedStateSnapshot:
        return self.state


class _Kernel:
    def __init__(self) -> None:
        self.calls: list[tuple[_SingletonAdmissionDecision, object]] = []

    def execute(
        self,
        decision: _SingletonAdmissionDecision,
        live_record: object,
        runtime: _Runtime,
    ) -> _TransitionEvaluation:
        self.calls.append((decision, live_record))
        next_state = _state_after()
        runtime.state = next_state
        ledger = _RoundLedgerClosure(
            controller_round=1,
            checkpoint_sequence=1,
            prefix_identity_before="ledger:before",
            prefix_identity_after="ledger:after",
            sequence_start_exclusive=2,
            first_sequence_index=3,
            sequence_indices=(3, 4),
            occurrence_ids=("occurrence:3", "occurrence:4"),
            reuse_identities=(None, "primitive:3"),
            round_s_alg_components=(
                ("N_H_outer", 0),
                ("N_H_refit", 2),
                ("N_grad", 0),
                ("N_metric", 0),
            ),
            round_s_unique_components=(
                ("N_H_outer", 0),
                ("N_H_refit", 1),
                ("N_grad", 0),
                ("N_metric", 0),
            ),
            cumulative_s_alg=5,
            cumulative_s_alg_components=(
                ("N_H_outer", 1),
                ("N_H_refit", 2),
                ("N_grad", 1),
                ("N_metric", 1),
            ),
            cumulative_s_unique=4,
            cumulative_s_unique_components=(
                ("N_H_outer", 1),
                ("N_H_refit", 1),
                ("N_grad", 1),
                ("N_metric", 1),
            ),
            close_count=1,
        )
        admission = _AdmissionReceipt(
            selected_domain_record_id="record:7",
            generator_id="generator:7",
            pool_index=7,
            insertion_position=1,
            initial_logical_value=0.0,
            logical_parameter_count_before=1,
            logical_parameter_count_after=2,
            runtime_parameter_count_before=1,
            runtime_parameter_count_after=2,
            old_to_new_logical_indices=(0,),
            old_to_new_runtime_indices=(0,),
            inserted_runtime_indices=(1,),
            optimizer_memory_identity_before="optimizer:before",
            optimizer_memory_identity_after="optimizer:after",
            selection_count_before=0,
            selection_count_after=1,
            available_before=True,
            available_after=False,
            source_identity="generator:7",
            child_identity="generator:7",
        )
        refit = _SupportedFSRefitReceipt(
            policy_identity="supported_fs_whitened_fixed_v1",
            scope_identity="full_ansatz_v1",
            optimizer_identity="POWELL",
            chart_identity="chart:7",
            chart_dimension=2,
            supported_rank=2,
            active_logical_indices=(0, 1),
            external_gram_receipt_identity="gram:7",
            external_gram_reused=True,
            optimizer_success=True,
            optimizer_nfev=11,
            optimizer_nit=3,
            optimizer_message="ok",
        )
        non_worsening = _NonWorseningReceipt(
            energy_before=-1.0,
            energy_after=-1.2,
            absolute_tolerance=1.0e-10,
            comparison_semantics="energy_after_le_before_plus_abs_tolerance_v1",
            accepted=True,
        )
        trust = _AdaptiveTrustUpdateReceipt(
            policy_identity="source_metric_inverse_sqrt_no_overlap_v1",
            trust_state_identity_before="trust-state:before",
            trust_state_identity_after="trust-state:after",
            update_count_before=1,
            update_count_after=2,
            payload_identity="trust-update:7",
            endpoint_overlap_query_charge=0,
        )
        event = _CheckpointReadyAcceptedStateEvent(
            controller_round=2,
            accepted_state_fingerprint="state:after",
            accepted_operator_ids=next_state.accepted_operator_ids,
            accepted_insertion_positions=(
                next_state.accepted_insertion_positions
            ),
            logical_parameter_ids=next_state.logical_parameter_ids,
            logical_parameter_values=next_state.logical_parameter_values,
            runtime_parameter_ids=next_state.runtime_parameter_ids,
            runtime_parameter_values=next_state.runtime_parameter_values,
            accepted_energy=next_state.accepted_energy,
            trust_state_identity=next_state.trust_state_identity,
            estimator_prefix_identity="ledger:after",
            ledger_closure=ledger,
        )
        return _TransitionEvaluation(
            next_state=next_state,
            admission=admission,
            refit=refit,
            trust=trust,
            non_worsening=non_worsening,
            ledger=ledger,
            checkpoint_event=event,
            operation_audit=_TransitionOperationAudit(
                admission_calls=1,
                supported_fs_chart_calls=1,
                optimizer_dispatch_calls=1,
                trust_update_calls=1,
                ledger_close_calls=1,
                checkpoint_event_count=1,
            ),
        )


def test_transition_consumes_exact_decision_and_returns_frozen_receipts() -> None:
    decision = _decision()
    preceding = _state_before()
    live_record = object()
    kernel = _Kernel()
    runtime = _Runtime()

    result = _transition_singleton(
        preceding,
        decision,
        _TransitionWorkspace(
            runtime_sidecar={"record:7": live_record},
            numerical_runtime=runtime,
            kernel=kernel,
        ),
    )

    assert kernel.calls == [(decision, live_record)]
    assert preceding == _state_before()
    assert result.preceding_state is preceding
    assert result.decision is decision
    assert result.next_state == _state_after()
    assert result.admission.initial_logical_value == 0.0
    assert result.admission.insertion_position == 1
    assert result.refit.scope_identity == "full_ansatz_v1"
    assert result.refit.external_gram_reused is True
    assert result.non_worsening.accepted is True
    assert result.ledger.occurrence_ids == ("occurrence:3", "occurrence:4")
    assert result.ledger.cumulative_s_alg == 5
    assert result.ledger.cumulative_s_unique == 4
    assert result.operation_audit.optimizer_dispatch_calls == 1
    assert (
        result.checkpoint_event.estimator_prefix_identity
        == result.next_state.estimator_prefix_identity
    )
    with pytest.raises(FrozenInstanceError):
        result.next_state.accepted_energy = 0.0  # type: ignore[misc]


def _batch_decision() -> _GreedyBatchAdmissionDecision:
    first = _decision().selected
    second = _CandidatePositionRecord(
        domain_record_id="record:8",
        generator_id="generator:8",
        parent_generator_id=None,
        pool_index=8,
        pool_label="G8",
        insertion_position=1,
        symmetry_identity="symmetry:8",
        lineage_identity=("generator:8",),
    )
    selected = (first, second)
    ranking = tuple(
        _ShortlistRankReceipt(
            record_key=(record.domain_record_id, record.generator_id),
            shortlist_rank=rank,
            primary_score=float(5 - rank),
            tie_break_score=float(4 - rank),
            pool_index=record.pool_index,
            insertion_position=record.insertion_position,
        )
        for rank, record in enumerate(selected, start=1)
    )
    phase_receipts = tuple(
        _PhaseSelectionReceipt(
            phase=phase,
            population=selected,
            shortlist=selected,
            shortlist_ranking=ranking,
            estimator_event_ids=(f"occurrence:{index}",),
        )
        for index, phase in enumerate(
            ("phase_i", "phase_ii", "phase_iii")
        )
    )
    return _GreedyBatchAdmissionDecision(
        controller_round=1,
        controller_state_fingerprint="state:before",
        selected=selected,
        phase_i=phase_receipts[0],
        phase_ii=phase_receipts[1],
        phase_iii=phase_receipts[2],
        response=_ResponseReceipt(
            identity="response:batch",
            coordinate_ids=("logical:0", "logical:1", "logical:2"),
            supported_rank=3,
            supported_dimension=3,
        ),
        trust=_TrustSolveReceipt(
            identity="trust:batch",
            solver_identity="supported-generalized",
            response_identity="response:batch",
            supported_rank=3,
            proposed_coordinate_values=(0.1, 0.0, 0.0),
        ),
        predictive_cost=_PredictiveCostReceipt(
            identity="cost:batch",
            policy_identity="symmetric",
            value=2.0,
        ),
        proposal=_GreedyBatchProposalReceipt(
            identity="proposal:batch",
            maximum_size=3,
            search_window_size=4,
            selected_record_ids=("record:7", "record:8"),
            score=2.0,
            modeled_energy_decrease=6.0,
            predictive_cost_excess=2.0,
            denominator=3.0,
            geometry_identity="geometry:batch",
            evaluated_subset_count=3,
        ),
        estimator_events=(
            _EstimatorEventIdentity(0, "occurrence:0", None),
            _EstimatorEventIdentity(1, "occurrence:1", None),
            _EstimatorEventIdentity(2, "occurrence:2", None),
        ),
    )


def _batch_state_after() -> _AcceptedStateSnapshot:
    return _AcceptedStateSnapshot(
        controller_round=2,
        accepted_operator_ids=(
            "operator:0",
            "generator:7",
            "generator:8",
        ),
        accepted_insertion_positions=(0, 1, 2),
        logical_parameter_ids=("logical:0", "logical:1", "logical:2"),
        logical_parameter_values=(0.12, -0.03, 0.02),
        runtime_parameter_ids=("runtime:0", "runtime:1", "runtime:2"),
        runtime_parameter_values=(0.12, -0.03, 0.02),
        accepted_energy=-1.3,
        accepted_state_fingerprint="state:batch-after",
        available_generator_ids=(),
        selection_counts=(("generator:7", 1), ("generator:8", 1)),
        trust_state_identity="trust-state:after",
        optimizer_memory_identity="optimizer:after",
        estimator_prefix_identity="ledger:after",
    )


class _BatchKernel:
    def __init__(self) -> None:
        self.calls: list[
            tuple[_GreedyBatchAdmissionDecision, tuple[object, ...]]
        ] = []

    def execute(
        self,
        decision: _GreedyBatchAdmissionDecision,
        live_records: tuple[object, ...],
        runtime: _Runtime,
    ) -> _GreedyBatchTransitionEvaluation:
        self.calls.append((decision, live_records))
        base = _Kernel().execute(_decision(), live_records[0], runtime)
        next_state = _batch_state_after()
        runtime.state = next_state
        admission = _GreedyBatchAdmissionReceipt(
            selected_domain_record_ids=("record:7", "record:8"),
            generator_ids=("generator:7", "generator:8"),
            pool_indices=(7, 8),
            original_insertion_positions=(1, 1),
            effective_insertion_positions=(1, 2),
            initial_logical_values=(0.0, 0.0),
            logical_parameter_count_before=1,
            logical_parameter_count_after=3,
            old_to_new_logical_indices=(0,),
            inserted_logical_indices=(1, 2),
            admitted_runtime_counts=(1, 1),
            runtime_insert_positions=(1, 2),
            runtime_parameter_count_before=1,
            runtime_parameter_count_after=3,
            old_to_new_runtime_indices=(0,),
            inserted_runtime_indices=(1, 2),
            optimizer_memory_identity_before="optimizer:before",
            optimizer_memory_identity_after="optimizer:after",
            selection_counts_before=(0, 0),
            selection_counts_after=(1, 1),
            available_before=(True, True),
            available_after=(False, False),
            source_identities=("generator:7", "generator:8"),
            child_identities=("generator:7", "generator:8"),
        )
        event = replace(
            base.checkpoint_event,
            accepted_state_fingerprint=next_state.accepted_state_fingerprint,
            accepted_operator_ids=next_state.accepted_operator_ids,
            accepted_insertion_positions=next_state.accepted_insertion_positions,
            logical_parameter_ids=next_state.logical_parameter_ids,
            logical_parameter_values=next_state.logical_parameter_values,
            runtime_parameter_ids=next_state.runtime_parameter_ids,
            runtime_parameter_values=next_state.runtime_parameter_values,
            accepted_energy=next_state.accepted_energy,
        )
        return _GreedyBatchTransitionEvaluation(
            next_state=next_state,
            admission=admission,
            refit=replace(
                base.refit,
                chart_dimension=3,
                supported_rank=3,
                active_logical_indices=(0, 1, 2),
            ),
            trust=base.trust,
            non_worsening=replace(
                base.non_worsening,
                energy_after=-1.3,
            ),
            ledger=base.ledger,
            checkpoint_event=event,
            operation_audit=base.operation_audit,
        )


def test_greedy_batch_transition_commits_one_ordered_atomic_round() -> None:
    decision = _batch_decision()
    preceding = _state_before()
    live_records = (object(), object())
    kernel = _BatchKernel()
    runtime = _Runtime()

    result = _transition_greedy_batch(
        preceding,
        decision,
        _TransitionWorkspace(
            runtime_sidecar={
                "record:7": live_records[0],
                "record:8": live_records[1],
            },
            numerical_runtime=runtime,
            kernel=kernel,
        ),
    )

    assert isinstance(result, _AcceptedGreedyBatchTransition)
    assert kernel.calls == [(decision, live_records)]
    assert result.next_state == _batch_state_after()
    assert result.admission.original_insertion_positions == (1, 1)
    assert result.admission.effective_insertion_positions == (1, 2)
    assert result.admission.inserted_logical_indices == (1, 2)
    assert result.refit.active_logical_indices == (0, 1, 2)
    assert result.trust.update_count_after == (
        result.trust.update_count_before + 1
    )
    assert result.operation_audit.admission_calls == 1
    assert result.operation_audit.optimizer_dispatch_calls == 1
    assert result.ledger.close_count == 1
    assert result.checkpoint_event.controller_round == 2


def _prune_receipt(*, accepted: bool) -> _RecoverabilityPruneReceipt:
    return _RecoverabilityPruneReceipt(
        status="accepted" if accepted else "rejected",
        reason="measured_delete_refit",
        source_state_fingerprint=(
            "state:full-refit" if accepted else "state:after"
        ),
        pre_prune_operator_ids=("operator:0", "generator:7"),
        post_prune_operator_ids=(
            ("generator:7",)
            if accepted
            else ("operator:0", "generator:7")
        ),
        pre_prune_logical_parameter_count=2,
        post_prune_logical_parameter_count=1 if accepted else 2,
        pre_prune_runtime_parameter_count=2,
        post_prune_runtime_parameter_count=1 if accepted else 2,
        optimizer_memory_identity_before="optimizer:after",
        optimizer_memory_identity_after=(
            "optimizer:pruned" if accepted else "optimizer:after"
        ),
        trust_radius_before=0.125,
        trust_radius_after=0.125 if accepted else 0.0625,
        metric_damping=0.0,
        endpoint_overlap_query_charge=0,
        terminal_prune_active=False,
        nomination_index=0,
        nomination_label="operator:0",
        predicted_energy_change=-0.02,
        surrogate_used_for_acceptance=False,
        trial_executed=True,
        trial_branch_id="sr_v4_prune_trial:fixture",
        trial_classification=(
            "committed_prune" if accepted else "discarded_prune"
        ),
        trial_s_alg=2,
        measured_energy_before=-1.2,
        measured_energy_after=-1.21 if accepted else -1.19,
        accepted=accepted,
        deleted_index=0 if accepted else None,
        deleted_label="operator:0" if accepted else None,
        final_state_fingerprint=(
            "state:pruned" if accepted else "state:after"
        ),
    )


class _PruningKernel(_Kernel):
    def __init__(self, *, accepted: bool) -> None:
        super().__init__()
        self.accepted = accepted

    def execute(
        self,
        decision: _SingletonAdmissionDecision,
        live_record: object,
        runtime: _Runtime,
    ) -> _TransitionEvaluation:
        evaluation = super().execute(decision, live_record, runtime)
        pruning = _prune_receipt(accepted=self.accepted)
        if not self.accepted:
            return replace(
                evaluation,
                pruning=pruning,
                operation_audit=_TransitionOperationAudit(
                    admission_calls=1,
                    supported_fs_chart_calls=1,
                    optimizer_dispatch_calls=1,
                    trust_update_calls=1,
                    prune_nomination_calls=1,
                    prune_verification_calls=1,
                    ledger_close_calls=1,
                    checkpoint_event_count=1,
                ),
            )
        next_state = replace(
            evaluation.next_state,
            accepted_operator_ids=("generator:7",),
            accepted_insertion_positions=(0,),
            logical_parameter_ids=("logical:0",),
            logical_parameter_values=(-0.03,),
            runtime_parameter_ids=("runtime:0",),
            runtime_parameter_values=(-0.03,),
            accepted_energy=-1.21,
            accepted_state_fingerprint="state:pruned",
            optimizer_memory_identity="optimizer:pruned",
        )
        ledger = evaluation.ledger
        event = replace(
            evaluation.checkpoint_event,
            accepted_state_fingerprint=next_state.accepted_state_fingerprint,
            accepted_operator_ids=next_state.accepted_operator_ids,
            accepted_insertion_positions=next_state.accepted_insertion_positions,
            logical_parameter_ids=next_state.logical_parameter_ids,
            logical_parameter_values=next_state.logical_parameter_values,
            runtime_parameter_ids=next_state.runtime_parameter_ids,
            runtime_parameter_values=next_state.runtime_parameter_values,
            accepted_energy=next_state.accepted_energy,
        )
        runtime.state = next_state
        return replace(
            evaluation,
            next_state=next_state,
            pruning=pruning,
            non_worsening=replace(
                evaluation.non_worsening,
                energy_after=-1.21,
            ),
            checkpoint_event=replace(event, ledger_closure=ledger),
            operation_audit=_TransitionOperationAudit(
                admission_calls=1,
                supported_fs_chart_calls=1,
                optimizer_dispatch_calls=1,
                trust_update_calls=1,
                prune_nomination_calls=1,
                prune_verification_calls=1,
                ledger_close_calls=1,
                checkpoint_event_count=1,
            ),
        )


@pytest.mark.parametrize("accepted", [False, True])
def test_transition_validates_recoverability_prune_after_full_refit(
    accepted: bool,
) -> None:
    result = _transition_singleton(
        _state_before(),
        _decision(),
        _TransitionWorkspace(
            runtime_sidecar={"record:7": object()},
            numerical_runtime=_Runtime(),
            kernel=_PruningKernel(accepted=accepted),
        ),
    )

    assert result.pruning is not None
    assert result.pruning.accepted is accepted
    assert result.pruning.surrogate_used_for_acceptance is False
    assert result.operation_audit.prune_nomination_calls == 1
    assert result.operation_audit.prune_verification_calls == 1
    if accepted:
        assert result.next_state.accepted_operator_ids == ("generator:7",)
        assert result.next_state.accepted_energy == -1.21
    else:
        assert result.next_state == _state_after()
        assert result.pruning.trust_radius_after == 0.0625


def test_internal_prune_receipt_rejects_contradictory_states() -> None:
    with pytest.raises(
        ValueError,
        match="accepted prune trial requires committed classification",
    ):
        replace(
            _prune_receipt(accepted=True),
            trial_classification="discarded_prune",
        )
    with pytest.raises(
        ValueError,
        match="rejected prune trial must preserve the source state",
    ):
        replace(
            _prune_receipt(accepted=False),
            final_state_fingerprint="state:mutated",
        )


def test_transition_rejects_sidecar_candidate_substitution() -> None:
    with pytest.raises(
        ValueError,
        match="sole runtime sidecar key must equal the decision record",
    ):
        _transition_singleton(
            _state_before(),
            _decision(),
            _TransitionWorkspace(
                runtime_sidecar={"record:substitute": object()},
                numerical_runtime=_Runtime(),
                kernel=_Kernel(),
            ),
        )


def test_middle_insertion_remaps_current_coordinate_positions() -> None:
    assert _accepted_positions_after_insertion((0, 1, 2), 1) == (
        0,
        1,
        2,
        3,
    )


def test_non_worsening_receipt_is_observational_not_a_rejection_gate() -> None:
    class _WorseningKernel(_Kernel):
        def execute(
            self,
            decision: _SingletonAdmissionDecision,
            live_record: object,
            runtime: _Runtime,
        ) -> _TransitionEvaluation:
            evaluation = super().execute(
                decision,
                live_record,
                runtime,
            )
            next_state = replace(
                evaluation.next_state,
                accepted_energy=-0.9,
            )
            runtime.state = next_state
            return replace(
                evaluation,
                next_state=next_state,
                non_worsening=_NonWorseningReceipt(
                    energy_before=-1.0,
                    energy_after=-0.9,
                    absolute_tolerance=0.0,
                    comparison_semantics="raw_energy_after_le_before_v1",
                    accepted=False,
                ),
                checkpoint_event=replace(
                    evaluation.checkpoint_event,
                    accepted_energy=-0.9,
                ),
            )

    result = _transition_singleton(
        _state_before(),
        _decision(),
        _TransitionWorkspace(
            runtime_sidecar={"record:7": object()},
            numerical_runtime=_Runtime(),
            kernel=_WorseningKernel(),
        ),
    )

    assert result.non_worsening.accepted is False


def test_transition_workspace_has_no_policy_or_output_destinations() -> None:
    assert tuple(field.name for field in fields(_TransitionWorkspace)) == (
        "runtime_sidecar",
        "numerical_runtime",
        "kernel",
    )


def test_operation_audit_is_derived_from_exact_authority_order() -> None:
    operations = (
        "admission",
        "supported_fs_chart",
        "optimizer_dispatch",
        "trust_update",
        "ledger_close",
        "checkpoint_event",
    )

    audit = _TransitionOperationAudit.from_operation_sequence(operations)

    assert (
        audit.admission_calls,
        audit.supported_fs_chart_calls,
        audit.optimizer_dispatch_calls,
        audit.trust_update_calls,
        audit.ledger_close_calls,
        audit.checkpoint_event_count,
    ) == (1, 1, 1, 1, 1, 1)
    with pytest.raises(
        ValueError,
        match="operation order changed",
    ):
        _TransitionOperationAudit.from_operation_sequence(
            (
                "admission",
                "supported_fs_chart",
                "supported_fs_chart",
                "optimizer_dispatch",
                "trust_update",
                "ledger_close",
                "checkpoint_event",
            )
        )
