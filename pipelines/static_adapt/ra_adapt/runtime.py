"""Canonical RA-ADAPT execution and typed numerical-result projection.

The retained controller and receipt vocabulary remain under ``sr_snake`` for
historical serialization compatibility.  This module owns their execution
composition so the historical public facade can delegate inward without the
canonical engine importing a compatibility runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from pipelines.contracts.problem import ResolvedProblemContext
from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)
from pipelines.static_adapt.estimator_call_ledger import (
    projective_state_fingerprint,
)
from pipelines.static_adapt.sr_snake._controller import (
    _ControllerOutcome,
    _DefaultControllerFinalization,
    _run_default_combinatorial_batch_controller,
    _run_default_fork_local_beam_controller,
    _run_default_greedy_batch_controller,
    _run_default_singleton_controller,
)
from pipelines.static_adapt.sr_snake._context import (
    _ResolvedExecutionContext,
    _resolve_execution_context,
)
from pipelines.static_adapt.sr_snake._observation import (
    _prepare_observation_destinations,
    _project_observation_artifacts,
)
from pipelines.static_adapt.sr_snake._transition import (
    _AcceptedCombinatorialBatchTransition,
    _AcceptedGreedyBatchTransition,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedRefitReceipt,
    AcceptedStateReceipt,
    AcceptedTransitionReceipt,
    AlwaysCommutationReducedInsertion,
    AppendCommutationReducedInsertion,
    AppendOnlyInsertion,
    AuthenticatedResumeScientificReplayReceipt,
    AuthenticatedResumeTransitionReceipt,
    CANONICAL_CANDIDATE_REPRESENTATION,
    CanonicalReportingReceipt,
    CheckpointReceipt,
    CombinatorialBatchAdmission,
    CombinatorialBatchAcceptedTransitionReceipt,
    CombinatorialBatchMemberAdmissionReceipt,
    CombinatorialBatchProposalReceipt,
    CombinatorialBatchScientificReplayReceipt,
    CombinatorialBatchTransitionAdmissionReceipt,
    EstimatorAccountingReceipt,
    EstimatorComponentsReceipt,
    EstimatorWorkReceipt,
    ForkLocalBeam,
    GreedyBatchAdmission,
    GreedyBatchAcceptedTransitionReceipt,
    GreedyBatchMemberAdmissionReceipt,
    GreedyBatchProposalReceipt,
    GreedyBatchScientificReplayReceipt,
    GreedyBatchTransitionAdmissionReceipt,
    ObservationReceipt,
    ParameterBlockReceipt,
    PhaseIIIReceipt,
    PhaseReceipt,
    PlateauCommutationInsertion,
    RecoverabilityPruneReceipt,
    ReferenceStateReceipt,
    ResolvedExecutionReceipt,
    RouteReceipt,
    RuntimePauliTermReceipt,
    SRRunRequest,
    SRRunResult,
    ScientificReplayReceipt,
    SingletonAdmission,
    SupportedMetricReceipt,
    TrustSolveReceipt,
)


_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME = (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
)


def _paper_i_requested_controller_rounds(
    requested: Sequence[int],
    *,
    accepted_round_count: int,
    terminal_controller_outcome: str | None,
) -> tuple[int, ...]:
    """Avoid fabricating Paper-I rows beyond an authenticated terminal."""

    rounds = tuple(int(value) for value in requested)
    accepted = int(accepted_round_count)
    if terminal_controller_outcome != _PHASE3_NO_POSITIVE_TERMINAL_OUTCOME:
        return rounds
    if accepted < 1:
        raise ValueError(
            "Phase-III no-positive termination requires an accepted prefix."
        )
    return tuple(value for value in rounds if value <= accepted)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"SR-SNAKE numerical projection is missing {name}."
        )
    return value


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(
            f"SR-SNAKE numerical projection is missing {name}."
        )
    return value


def _accepted_state(
    *,
    history: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    round_index: int,
    insertion_positions: tuple[int, ...],
    history_index: int | None = None,
) -> AcceptedStateReceipt:
    operator_labels = tuple(
        str(value)
        for value in checkpoint.get(
            "ordered_active_operator_labels",
            (),
        )
    )
    operator_rows_raw = checkpoint.get("ordered_active_operators", ())
    operator_rows = (
        tuple(operator_rows_raw)
        if isinstance(operator_rows_raw, Sequence)
        and not isinstance(operator_rows_raw, (str, bytes))
        else ()
    )
    generator_ids = tuple(
        str(
            row.get("generator_id")
            or row.get("label")
            or operator_labels[index]
        )
        for index, row in enumerate(operator_rows)
        if isinstance(row, Mapping) and index < len(operator_labels)
    )
    if len(generator_ids) != len(operator_labels):
        raise RuntimeError(
            "Accepted checkpoint operator and generator identities disagree."
        )
    if len(insertion_positions) != len(operator_labels):
        raise RuntimeError(
            "Accepted checkpoint operators and admission-position provenance "
            "disagree."
        )
    return AcceptedStateReceipt(
        controller_round=round_index,
        operators=operator_labels,
        insertion_positions=tuple(int(value) for value in insertion_positions),
        generator_ids=generator_ids,
        logical_parameters=tuple(
            float(value)
            for value in checkpoint.get(
                "signed_unwrapped_logical_parameters",
                (),
            )
        ),
        runtime_parameters=tuple(
            float(value)
            for value in checkpoint.get(
                "signed_unwrapped_runtime_parameters",
                (),
            )
        ),
        energy=float(
            history[
                round_index - 1
                if history_index is None
                else int(history_index)
            ]["energy_after_opt"]
        ),
        projective_state_fingerprint=str(
            checkpoint.get("projective_state_fingerprint", "")
        ),
    )


def _stationary_initial_state(
    state: Any,
    *,
    public_state_fingerprint: str | None = None,
) -> AcceptedStateReceipt:
    """Project the untouched fresh state for a round-zero Phase-0 stop."""

    if (
        int(state.controller_round) != 0
        or tuple(state.accepted_operator_ids)
        or tuple(state.accepted_insertion_positions)
        or tuple(state.logical_parameter_values)
        or tuple(state.runtime_parameter_values)
    ):
        raise RuntimeError(
            "A zero-trajectory stationary result must retain the fresh empty "
            "accepted state."
        )
    return AcceptedStateReceipt(
        controller_round=0,
        operators=(),
        insertion_positions=(),
        generator_ids=(),
        logical_parameters=(),
        runtime_parameters=(),
        energy=float(state.accepted_energy),
        projective_state_fingerprint=str(
            state.accepted_state_fingerprint
            if public_state_fingerprint is None
            else public_state_fingerprint
        ),
    )


def _next_admission_position_provenance(
    *,
    preceding: tuple[int, ...],
    transition: Any,
) -> tuple[int, ...]:
    """Replay where each retained operator entered the accepted ansatz.

    The private controller state tracks current coordinate indices, which are
    necessarily ``0..depth-1`` after every splice.  The public receipt instead
    preserves the admission position of each retained operator, aligned with
    the checkpoint's current operator order.
    """

    if isinstance(
        transition,
        (
            _AcceptedGreedyBatchTransition,
            _AcceptedCombinatorialBatchTransition,
        ),
    ):
        admission = transition.admission
        if len(preceding) != int(admission.logical_parameter_count_before):
            raise RuntimeError(
                "Batch admission-position provenance has the wrong source "
                "cardinality."
            )
        projected: list[int | None] = [
            None
        ] * int(admission.logical_parameter_count_after)
        for old_index, new_index in enumerate(
            admission.old_to_new_logical_indices
        ):
            projected[int(new_index)] = int(preceding[old_index])
        for original_position, inserted_index in zip(
            admission.original_insertion_positions,
            admission.inserted_logical_indices,
            strict=True,
        ):
            projected[int(inserted_index)] = int(original_position)
        if any(value is None for value in projected):
            raise RuntimeError(
                "Batch admission-position provenance does not cover the "
                "accepted ansatz."
            )
        pruning = transition.pruning
        if (
            pruning is not None
            and bool(pruning.accepted)
            and pruning.deleted_index is not None
        ):
            deleted_index = int(pruning.deleted_index)
            if not 0 <= deleted_index < len(projected):
                raise RuntimeError(
                    "Accepted batch-prune deletion lies outside admission-"
                    "position provenance."
                )
            del projected[deleted_index]
        return tuple(int(value) for value in projected if value is not None)

    selected = transition.decision.selected
    insertion_position = int(selected.insertion_position)
    if not 0 <= insertion_position <= len(preceding):
        raise RuntimeError(
            "Singleton admission position lies outside the accepted ansatz."
        )
    projected = list(int(value) for value in preceding)
    projected.insert(insertion_position, insertion_position)
    pruning = transition.pruning
    if (
        pruning is not None
        and bool(pruning.accepted)
        and pruning.deleted_index is not None
    ):
        deleted_index = int(pruning.deleted_index)
        if not 0 <= deleted_index < len(projected):
            raise RuntimeError(
                "Accepted prune deletion lies outside admission-position "
                "provenance."
            )
        del projected[deleted_index]
    return tuple(projected)


def _phase_receipt(row: Mapping[str, Any]) -> PhaseReceipt:
    curvature = _require_mapping(
        row.get("phase2_curvature_receipt"),
        name="Phase-II curvature receipt",
    )
    return PhaseReceipt(
        phase1_energy_model=str(row["phase1_energy_model"]),
        phase2_curvature_status=str(curvature["status"]),
        phase3=PhaseIIIReceipt(
            coordinate_scope=str(row["phase3_response_coordinate_scope"]),
            coordinate_indices=tuple(
                int(value)
                for value in row["phase3_response_coordinate_indices"]
            ),
            pre_support_count=int(row["phase3_response_pre_support_count"]),
            supported_rank=int(row["phase3_response_supported_rank"]),
        ),
    )


def _trust_receipt(row: Mapping[str, Any]) -> TrustSolveReceipt:
    trust = _require_mapping(
        row.get("route_a_trust_region_update"),
        name="trust-solve receipt",
    )
    transaction_raw = trust.get("source_metric_trust_transaction")
    transaction = (
        transaction_raw if isinstance(transaction_raw, Mapping) else None
    )
    return TrustSolveReceipt(
        policy=str(trust["policy"]),
        update_reason=str(trust["update_reason"]),
        endpoint_overlap_query_charge=int(
            trust["endpoint_overlap_query_charge"]
        ),
        transaction_complete=(
            None
            if transaction is None
            else bool(transaction.get("transaction_complete", False))
        ),
        transaction_failure=(
            None
            if transaction is not None
            else (
                None
                if trust.get("source_metric_trust_transaction_failure") is None
                else str(trust["source_metric_trust_transaction_failure"])
            )
        ),
        supported_rank=(
            None
            if transaction is None or transaction.get("supported_rank") is None
            else int(transaction["supported_rank"])
        ),
        supported_metric_whitening_active=(
            None
            if transaction is None
            else bool(transaction.get("supported_metric_whitening_active"))
        ),
        supported_metric_inverse_sqrt_constructed=(
            None
            if transaction is None
            else bool(
                transaction.get("supported_metric_inverse_sqrt_constructed")
            )
        ),
        predicted_source_metric_displacement=(
            None
            if transaction is None
            or transaction.get("predicted_source_metric_displacement") is None
            else float(transaction["predicted_source_metric_displacement"])
        ),
        realized_source_metric_displacement=(
            None
            if transaction is None
            or transaction.get("realized_source_metric_displacement") is None
            else float(transaction["realized_source_metric_displacement"])
        ),
    )


def _accepted_refit_metric_element_occurrences(
    accounting: Mapping[str, Any],
) -> int:
    measured_occurrences = accounting.get(
        "symmetric_metric_element_occurrences"
    )
    if measured_occurrences is not None:
        return int(measured_occurrences)
    if str(accounting.get("status", "")) == (
        "reused_external_logical_fs_gram_receipt"
    ):
        reused_occurrences = accounting.get("metric_element_count_reused")
        if reused_occurrences is not None:
            return int(reused_occurrences)
    raise RuntimeError(
        "SR-SNAKE numerical projection accepted-refit accounting is missing "
        "its measured "
        "or external-Gram-reused symmetric metric-element count."
    )


def _accepted_refit_receipt(row: Mapping[str, Any]) -> AcceptedRefitReceipt:
    refit = _require_mapping(
        row.get("accepted_refit"),
        name="accepted-refit receipt",
    )
    invocation = _require_mapping(
        refit.get("accepted_refit_invocation"),
        name="accepted-refit invocation",
    )
    config = _require_mapping(
        invocation.get("config"),
        name="accepted-refit config",
    )
    accounting = _require_mapping(
        invocation.get("metric_query_accounting"),
        name="accepted-refit metric accounting",
    )
    supported_metric = _require_mapping(
        config.get("supported_metric"),
        name="accepted-refit supported-metric config",
    )
    initialization_raw = refit.get("accepted_refit_initialization")
    initialization_fields: dict[str, Any] = {}
    if initialization_raw is not None:
        initialization = _require_mapping(
            initialization_raw,
            name="accepted-refit initialization receipt",
        )
        try:
            initialization_fields = {
                "initialization_policy": str(initialization["policy"]),
                "initialization_status": str(initialization["status"]),
                "initialization_guard_nfev": int(
                    refit["accepted_refit_initialization_guard_nfev"]
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "SR-SNAKE numerical projection accepted-refit initialization "
                "evidence is incomplete."
            ) from exc
    return AcceptedRefitReceipt(
        policy=str(refit["policy"]),
        scope=str(config["scope"]),
        coordinate_chart=str(config["coordinate_chart"]),
        base_chart_policy=str(config["base_chart_policy"]),
        full_ansatz=bool(config["full_ansatz"]),
        supported_rank=int(refit["supported_rank"]),
        final_energy=float(refit["final_energy"]),
        symmetric_metric_element_occurrences=(
            _accepted_refit_metric_element_occurrences(accounting)
        ),
        supported_metric=SupportedMetricReceipt(
            policy=str(supported_metric["policy"]),
            rank_relative_tolerance=float(
                supported_metric["rank_relative_tolerance"]
            ),
            metric_regularization=float(
                supported_metric["metric_regularization"]
            ),
            energy_regularization=float(
                supported_metric["energy_regularization"]
            ),
            max_fubini_study_step=float(
                supported_metric["max_fubini_study_step"]
            ),
            global_trust_kkt_residual_accuracy=float(
                supported_metric["global_trust_kkt_residual_accuracy"]
            ),
            global_trust_metric_distortion_budget=float(
                supported_metric["global_trust_metric_distortion_budget"]
            ),
        ),
        **initialization_fields,
    )


def _checkpoint_receipt(checkpoint: Mapping[str, Any]) -> CheckpointReceipt:
    strict = _require_mapping(
        checkpoint.get("strict_replay"),
        name="strict replay receipt",
    )
    parameterization = _require_mapping(
        checkpoint.get("parameterization"),
        name="parameterization receipt",
    )
    ledger = _require_mapping(
        checkpoint.get("estimator_ledger_receipt"),
        name="checkpoint estimator-ledger receipt",
    )
    cumulative = _require_mapping(
        ledger.get("cumulative_executed_queries"),
        name="checkpoint cumulative estimator work",
    )
    blocks_raw = _require_sequence(
        parameterization.get("blocks"),
        name="parameterization blocks",
    )
    blocks: list[ParameterBlockReceipt] = []
    for block_raw in blocks_raw:
        block = _require_mapping(block_raw, name="parameterization block")
        terms_raw = _require_sequence(
            block.get("runtime_terms_exyz"),
            name="runtime Pauli terms",
        )
        runtime_terms: list[RuntimePauliTermReceipt] = []
        for term_raw in terms_raw:
            term = _require_mapping(term_raw, name="runtime Pauli term")
            runtime_terms.append(
                RuntimePauliTermReceipt(
                    pauli_exyz=str(term["pauli_exyz"]),
                    coefficient_real=float(term["coeff_re"]),
                    coefficient_imaginary=float(term["coeff_im"]),
                    qubit_count=int(term["nq"]),
                )
            )
        blocks.append(
            ParameterBlockReceipt(
                candidate_label=str(block["candidate_label"]),
                logical_index=int(block["logical_index"]),
                runtime_start=int(block["runtime_start"]),
                runtime_count=int(block["runtime_count"]),
                execution_mode=str(block["execution_mode"]),
                runtime_terms=tuple(runtime_terms),
            )
        )
    return CheckpointReceipt(
        outer_iteration=int(checkpoint["outer_iteration"]),
        active_ansatz_depth=int(checkpoint["active_ansatz_depth"]),
        ordered_operator_labels=tuple(
            str(value)
            for value in checkpoint["ordered_active_operator_labels"]
        ),
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        projective_state_fingerprint=str(
            checkpoint["projective_state_fingerprint"]
        ),
        strict_replay_passed=bool(strict["passed"]),
        strict_replay_fidelity=float(strict["fidelity"]),
        parameterization_mode=str(parameterization["mode"]),
        parameterization_term_order=str(parameterization["term_order"]),
        parameter_blocks=tuple(blocks),
        logical_parameters=tuple(
            float(value)
            for value in checkpoint["signed_unwrapped_logical_parameters"]
        ),
        runtime_parameters=tuple(
            float(value)
            for value in checkpoint["signed_unwrapped_runtime_parameters"]
        ),
        route_profile=str(checkpoint["sr_route_profile"]),
        route_contract_sha256=str(
            checkpoint["sr_route_profile_contract_sha256"]
        ),
        estimator_ledger_status=str(ledger["status"]),
        estimator_ledger_s_alg=int(cumulative["S_alg"]),
    )


def _recoverability_prune_receipt(
    transition: Any,
) -> RecoverabilityPruneReceipt | None:
    internal = transition.pruning
    if internal is None:
        return None
    return RecoverabilityPruneReceipt(
        status=str(internal.status),
        reason=str(internal.reason),
        policy=str(internal.policy_identity),
        nomination_policy=str(
            internal.nomination_policy_identity
        ),
        source_state_fingerprint=str(
            internal.source_state_fingerprint
        ),
        trust_radius_before=float(internal.trust_radius_before),
        trust_radius_after=float(internal.trust_radius_after),
        metric_damping=float(internal.metric_damping),
        endpoint_overlap_query_charge=int(
            internal.endpoint_overlap_query_charge
        ),
        terminal_prune_active=bool(internal.terminal_prune_active),
        nomination_index=internal.nomination_index,
        nomination_label=internal.nomination_label,
        predicted_energy_change=internal.predicted_energy_change,
        surrogate_used_for_acceptance=(
            internal.surrogate_used_for_acceptance
        ),
        trial_executed=bool(internal.trial_executed),
        trial_branch_id=internal.trial_branch_id,
        trial_classification=internal.trial_classification,
        trial_s_alg=(
            int(internal.trial_s_alg)
            if internal.trial_executed
            else None
        ),
        measured_energy_before=internal.measured_energy_before,
        measured_energy_after=internal.measured_energy_after,
        accepted=internal.accepted,
        deleted_index=internal.deleted_index,
        deleted_label=internal.deleted_label,
        final_state_fingerprint=str(
            internal.final_state_fingerprint
        ),
    )


def _greedy_batch_public_receipts(
    transition: _AcceptedGreedyBatchTransition,
) -> tuple[
    GreedyBatchProposalReceipt,
    GreedyBatchTransitionAdmissionReceipt,
]:
    """Project one validated internal batch without singleton aliases."""

    selected = transition.decision.selected
    admission = transition.admission
    if len(selected) != len(admission.selected_domain_record_ids):
        raise RuntimeError(
            "Greedy batch decision and admission cardinalities disagree."
        )
    members: list[GreedyBatchMemberAdmissionReceipt] = []
    runtime_offset = 0
    for member_index, record in enumerate(selected):
        runtime_count = int(
            admission.admitted_runtime_counts[member_index]
        )
        member_runtime_indices = tuple(
            int(value)
            for value in admission.inserted_runtime_indices[
                runtime_offset : runtime_offset + runtime_count
            ]
        )
        runtime_offset += runtime_count
        members.append(
            GreedyBatchMemberAdmissionReceipt(
                selected_domain_record_id=str(record.domain_record_id),
                generator_id=str(record.generator_id),
                selected_operator=str(record.pool_label),
                pool_index=int(record.pool_index),
                original_insertion_position=int(
                    admission.original_insertion_positions[member_index]
                ),
                effective_insertion_position=int(
                    admission.effective_insertion_positions[member_index]
                ),
                inserted_logical_index=int(
                    admission.inserted_logical_indices[member_index]
                ),
                initial_logical_value=float(
                    admission.initial_logical_values[member_index]
                ),
                admitted_runtime_count=runtime_count,
                runtime_insert_position=int(
                    admission.runtime_insert_positions[member_index]
                ),
                inserted_runtime_indices=member_runtime_indices,
                source_identity=str(
                    admission.source_identities[member_index]
                ),
                child_identity=str(
                    admission.child_identities[member_index]
                ),
            )
        )
    if runtime_offset != len(admission.inserted_runtime_indices):
        raise RuntimeError(
            "Greedy batch per-member runtime remaps are incomplete."
        )
    proposal = transition.decision.proposal
    proposal_receipt = GreedyBatchProposalReceipt(
        identity=str(proposal.identity),
        maximum_size=int(proposal.maximum_size),
        search_window_size=(
            None
            if proposal.search_window_size is None
            else int(proposal.search_window_size)
        ),
        selected_cardinality=len(selected),
        selected_record_ids=tuple(
            str(record_id) for record_id in proposal.selected_record_ids
        ),
        score=float(proposal.score),
        modeled_energy_decrease=float(
            proposal.modeled_energy_decrease
        ),
        predictive_cost_excess=float(
            proposal.predictive_cost_excess
        ),
        denominator=float(proposal.denominator),
        geometry_identity=str(proposal.geometry_identity),
        evaluated_subset_count=int(proposal.evaluated_subset_count),
        estimator_event_ids=tuple(
            str(event.occurrence_id)
            for event in transition.decision.estimator_events
        ),
    )
    admission_receipt = GreedyBatchTransitionAdmissionReceipt(
        composition_identity=str(admission.composition_identity),
        selected_cardinality=len(selected),
        members=tuple(members),
        logical_parameter_count_before=int(
            admission.logical_parameter_count_before
        ),
        logical_parameter_count_after=int(
            admission.logical_parameter_count_after
        ),
        old_to_new_logical_indices=tuple(
            int(value) for value in admission.old_to_new_logical_indices
        ),
        inserted_logical_indices=tuple(
            int(value) for value in admission.inserted_logical_indices
        ),
        runtime_parameter_count_before=int(
            admission.runtime_parameter_count_before
        ),
        runtime_parameter_count_after=int(
            admission.runtime_parameter_count_after
        ),
        old_to_new_runtime_indices=tuple(
            int(value) for value in admission.old_to_new_runtime_indices
        ),
        inserted_runtime_indices=tuple(
            int(value) for value in admission.inserted_runtime_indices
        ),
        optimizer_memory_identity_before=str(
            admission.optimizer_memory_identity_before
        ),
        optimizer_memory_identity_after=str(
            admission.optimizer_memory_identity_after
        ),
    )
    return proposal_receipt, admission_receipt


def _combinatorial_batch_public_receipts(
    transition: _AcceptedCombinatorialBatchTransition,
) -> tuple[
    CombinatorialBatchProposalReceipt,
    CombinatorialBatchTransitionAdmissionReceipt,
]:
    """Project one exhaustive-subset transition without greedy aliases."""

    selected = transition.decision.selected
    admission = transition.admission
    if len(selected) != len(admission.selected_domain_record_ids):
        raise RuntimeError(
            "Combinatorial decision and admission cardinalities disagree."
        )
    members: list[CombinatorialBatchMemberAdmissionReceipt] = []
    runtime_offset = 0
    for member_index, record in enumerate(selected):
        runtime_count = int(
            admission.admitted_runtime_counts[member_index]
        )
        member_runtime_indices = tuple(
            int(value)
            for value in admission.inserted_runtime_indices[
                runtime_offset : runtime_offset + runtime_count
            ]
        )
        runtime_offset += runtime_count
        members.append(
            CombinatorialBatchMemberAdmissionReceipt(
                selected_domain_record_id=str(record.domain_record_id),
                generator_id=str(record.generator_id),
                selected_operator=str(record.pool_label),
                pool_index=int(record.pool_index),
                original_insertion_position=int(
                    admission.original_insertion_positions[member_index]
                ),
                effective_insertion_position=int(
                    admission.effective_insertion_positions[member_index]
                ),
                inserted_logical_index=int(
                    admission.inserted_logical_indices[member_index]
                ),
                initial_logical_value=float(
                    admission.initial_logical_values[member_index]
                ),
                admitted_runtime_count=runtime_count,
                runtime_insert_position=int(
                    admission.runtime_insert_positions[member_index]
                ),
                inserted_runtime_indices=member_runtime_indices,
                source_identity=str(
                    admission.source_identities[member_index]
                ),
                child_identity=str(
                    admission.child_identities[member_index]
                ),
            )
        )
    if runtime_offset != len(admission.inserted_runtime_indices):
        raise RuntimeError(
            "Combinatorial per-member runtime remaps are incomplete."
        )
    proposal = transition.decision.proposal
    proposal_receipt = CombinatorialBatchProposalReceipt(
        identity=str(proposal.identity),
        maximum_size=int(proposal.maximum_size),
        search_window_size=(
            None
            if proposal.search_window_size is None
            else int(proposal.search_window_size)
        ),
        ranked_population_count=int(proposal.ranked_population_count),
        ranked_window_count=int(proposal.ranked_window_count),
        selected_cardinality=len(selected),
        selected_record_ids=tuple(
            str(record_id) for record_id in proposal.selected_record_ids
        ),
        score=float(proposal.score),
        modeled_energy_decrease=float(
            proposal.modeled_energy_decrease
        ),
        predictive_cost_excess=float(
            proposal.predictive_cost_excess
        ),
        denominator=float(proposal.denominator),
        geometry_identity=str(proposal.geometry_identity),
        evaluated_subset_count=int(proposal.evaluated_subset_count),
        subset_counts_considered=tuple(
            (int(size), int(count))
            for size, count in proposal.subset_counts_considered
        ),
        subset_counts_evaluated=tuple(
            (int(size), int(count))
            for size, count in proposal.subset_counts_evaluated
        ),
        subset_counts_feasible=tuple(
            (int(size), int(count))
            for size, count in proposal.subset_counts_feasible
        ),
        estimator_event_ids=tuple(
            str(event.occurrence_id)
            for event in transition.decision.estimator_events
        ),
    )
    admission_receipt = CombinatorialBatchTransitionAdmissionReceipt(
        composition_identity=str(admission.composition_identity),
        selected_cardinality=len(selected),
        members=tuple(members),
        logical_parameter_count_before=int(
            admission.logical_parameter_count_before
        ),
        logical_parameter_count_after=int(
            admission.logical_parameter_count_after
        ),
        old_to_new_logical_indices=tuple(
            int(value) for value in admission.old_to_new_logical_indices
        ),
        inserted_logical_indices=tuple(
            int(value) for value in admission.inserted_logical_indices
        ),
        runtime_parameter_count_before=int(
            admission.runtime_parameter_count_before
        ),
        runtime_parameter_count_after=int(
            admission.runtime_parameter_count_after
        ),
        old_to_new_runtime_indices=tuple(
            int(value) for value in admission.old_to_new_runtime_indices
        ),
        inserted_runtime_indices=tuple(
            int(value) for value in admission.inserted_runtime_indices
        ),
        optimizer_memory_identity_before=str(
            admission.optimizer_memory_identity_before
        ),
        optimizer_memory_identity_after=str(
            admission.optimizer_memory_identity_after
        ),
    )
    return proposal_receipt, admission_receipt


def _authenticated_resume_prefix_receipts(
    context: _ResolvedExecutionContext,
) -> tuple[
    tuple[AcceptedStateReceipt, ...],
    tuple[AuthenticatedResumeTransitionReceipt, ...],
    tuple[AuthenticatedResumeScientificReplayReceipt, ...],
    tuple[EstimatorWorkReceipt, ...],
]:
    """Project the signed historical prefix retained by a typed resume."""

    hydration = context.numerical.resume_hydration
    if hydration is None:
        return (), (), (), ()
    history = tuple(hydration.mutable_history())
    if len(history) != int(hydration.controller_round):
        raise RuntimeError(
            "Authenticated resume history does not close to its controller "
            "round."
        )
    accepted_states: list[AcceptedStateReceipt] = []
    transitions: list[AuthenticatedResumeTransitionReceipt] = []
    replay: list[AuthenticatedResumeScientificReplayReceipt] = []
    prefix_work: list[EstimatorWorkReceipt] = []
    insertion_position_provenance: list[int] = []
    for round_index, row in enumerate(history, start=1):
        checkpoint = _require_mapping(
            row.get("active_prefix_checkpoint"),
            name=f"resume history[{round_index - 1}] checkpoint",
        )
        original_positions = tuple(
            int(value)
            for value in row.get(
                "selected_positions",
                (row["selected_position"],),
            )
        )
        effective_positions = tuple(
            int(value)
            for value in row.get(
                "selected_effective_positions",
                original_positions,
            )
        )
        if len(original_positions) != len(effective_positions):
            raise RuntimeError(
                "Authenticated resume admission positions are incomplete."
            )
        for original_position, effective_position in zip(
            original_positions,
            effective_positions,
            strict=True,
        ):
            if not 0 <= effective_position <= len(
                insertion_position_provenance
            ):
                raise RuntimeError(
                    "Authenticated resume admission position is outside its "
                    "accepted prefix."
                )
            insertion_position_provenance.insert(
                effective_position,
                original_position,
            )
        prune = row.get("post_admission_prune")
        if isinstance(prune, Mapping) and int(
            prune.get("accepted_count", 0) or 0
        ) > 0:
            deleted = tuple(
                int(value)
                for value in prune.get("deleted_indices", ())
            )
            if len(deleted) != 1 or not 0 <= deleted[0] < len(
                insertion_position_provenance
            ):
                raise RuntimeError(
                    "Authenticated resume prune deletion provenance is "
                    "incomplete."
                )
            del insertion_position_provenance[deleted[0]]
        accepted = _accepted_state(
            history=history,
            checkpoint=checkpoint,
            round_index=round_index,
            insertion_positions=tuple(
                insertion_position_provenance
            ),
        )
        checkpoint_receipt = _checkpoint_receipt(checkpoint)
        selected_operators = tuple(
            str(value)
            for value in row.get(
                "selected_ops",
                (row["selected_op"],),
            )
        )
        selected_pool_indices = tuple(
            int(value)
            for value in row.get(
                "selected_pool_indices",
                (row["pool_index"],),
            )
        )
        if (
            not selected_operators
            or len(selected_operators) != len(selected_pool_indices)
            or len(selected_operators) != len(original_positions)
        ):
            raise RuntimeError(
                "Authenticated resume selected-admission provenance is "
                "incomplete."
            )
        cumulative = _require_mapping(
            _require_mapping(
                checkpoint.get("estimator_ledger_receipt"),
                name="resume checkpoint estimator ledger",
            ).get("cumulative_executed_queries"),
            name="resume checkpoint cumulative estimator work",
        )
        components_raw = _require_mapping(
            cumulative.get("components"),
            name="resume checkpoint estimator components",
        )
        components = EstimatorComponentsReceipt(
            n_h_outer=int(components_raw["N_H_outer"]),
            n_h_refit=int(components_raw["N_H_refit"]),
            n_grad=int(components_raw["N_grad"]),
            n_metric=int(components_raw["N_metric"]),
        )
        s_alg = int(cumulative["S_alg"])
        if s_alg != (
            components.n_h_outer
            + components.n_h_refit
            + components.n_grad
            + components.n_metric
        ):
            raise RuntimeError(
                "Authenticated resume checkpoint S_alg does not close."
            )
        accepted_states.append(accepted)
        transitions.append(
            AuthenticatedResumeTransitionReceipt(
                controller_round=round_index,
                route_family=str(hydration.route_family),
                selected_operators=selected_operators,
                selected_pool_indices=selected_pool_indices,
                selected_positions=original_positions,
                accepted_state=accepted,
                energy_before=float(row["energy_before_opt"]),
                energy_after=float(row["energy_after_opt"]),
                cumulative_s_alg=s_alg,
                source_checkpoint_sha256=str(
                    hydration.source_sha256
                ),
            )
        )
        replay.append(
            AuthenticatedResumeScientificReplayReceipt(
                controller_round=round_index,
                selected_operators=selected_operators,
                energy_before_refit=float(row["energy_before_opt"]),
                accepted_state=accepted,
                phase=_phase_receipt(row),
                trust_solve=_trust_receipt(row),
                accepted_refit=_accepted_refit_receipt(row),
                checkpoint=checkpoint_receipt,
                source_checkpoint_sha256=str(
                    hydration.source_sha256
                ),
            )
        )
        prefix_work.append(
            EstimatorWorkReceipt(
                components=components,
                s_alg=s_alg,
            )
        )
    return (
        tuple(accepted_states),
        tuple(transitions),
        tuple(replay),
        tuple(prefix_work),
    )


def _scientific_replay(
    outcome: _ControllerOutcome,
    *,
    initial_insertion_position_provenance: tuple[int, ...] = (),
) -> tuple[
    tuple[AcceptedStateReceipt, ...],
    tuple[
        AcceptedTransitionReceipt
        | GreedyBatchAcceptedTransitionReceipt
        | CombinatorialBatchAcceptedTransitionReceipt,
        ...,
    ],
    tuple[
        ScientificReplayReceipt
        | GreedyBatchScientificReplayReceipt
        | CombinatorialBatchScientificReplayReceipt,
        ...,
    ],
]:
    history = tuple(
        projected.replay_projection.record
        for projected in outcome.projected_rounds
    )
    checkpoints = tuple(
        projected.checkpoint_projection.record
        for projected in outcome.projected_rounds
    )
    if not (
        len(checkpoints)
        == len(history)
        == len(outcome.transitions)
        == len(outcome.accepted_states)
    ):
        raise RuntimeError(
            "Controller transition, event projection, history, and checkpoint "
            "cardinalities disagree."
        )

    accepted_states: list[AcceptedStateReceipt] = []
    accepted_transitions: list[
        AcceptedTransitionReceipt
        | GreedyBatchAcceptedTransitionReceipt
        | CombinatorialBatchAcceptedTransitionReceipt
    ] = []
    replay: list[
        ScientificReplayReceipt
        | GreedyBatchScientificReplayReceipt
        | CombinatorialBatchScientificReplayReceipt
    ] = []
    admission_position_provenance = tuple(
        int(value)
        for value in initial_insertion_position_provenance
    )
    for index, (row, checkpoint, transition, next_state) in enumerate(
        zip(
            history,
            checkpoints,
            outcome.transitions,
            outcome.accepted_states,
            strict=True,
        ),
        start=1,
    ):
        controller_round = int(next_state.controller_round)
        admission_position_provenance = (
            _next_admission_position_provenance(
                preceding=admission_position_provenance,
                transition=transition,
            )
        )
        accepted = _accepted_state(
            history=history,
            checkpoint=checkpoint,
            round_index=controller_round,
            insertion_positions=admission_position_provenance,
            history_index=index - 1,
        )
        state_mismatches = tuple(
            name
            for name, left, right in (
                (
                    "controller_round",
                    accepted.controller_round,
                    next_state.controller_round,
                ),
                (
                    "logical_parameters",
                    accepted.logical_parameters,
                    next_state.logical_parameter_values,
                ),
                (
                    "runtime_parameters",
                    accepted.runtime_parameters,
                    next_state.runtime_parameter_values,
                ),
                ("energy", accepted.energy, next_state.accepted_energy),
            )
            if left != right
        )
        if state_mismatches:
            raise RuntimeError(
                "Controller accepted state disagrees with its projected "
                "scientific replay state; "
                f"differing_fields={state_mismatches!r}."
            )
        if isinstance(
            transition,
            _AcceptedCombinatorialBatchTransition,
        ):
            proposal, batch_admission = (
                _combinatorial_batch_public_receipts(transition)
            )
            selected = transition.decision.selected
            projected_record_ids = tuple(
                str(value)
                for value in row["combinatorial_batch_admission"][
                    "selected_record_ids"
                ]
            )
            projected_operators = tuple(
                str(value) for value in row["selected_batch_labels"]
            )
            projected_positions = tuple(
                int(value) for value in row["selected_batch_positions"]
            )
            if (
                projected_record_ids
                != tuple(record.domain_record_id for record in selected)
                or projected_operators
                != tuple(record.pool_label for record in selected)
                or projected_positions
                != tuple(record.insertion_position for record in selected)
            ):
                raise RuntimeError(
                    "Controller combinatorial decision disagrees with its "
                    "projected history row."
                )
            accepted_states.append(accepted)
            ledger = transition.ledger
            accepted_transitions.append(
                CombinatorialBatchAcceptedTransitionReceipt(
                    controller_round=int(next_state.controller_round),
                    preceding_state_fingerprint=(
                        transition
                        .preceding_state
                        .accepted_state_fingerprint
                    ),
                    proposal=proposal,
                    admission=batch_admission,
                    accepted_state=accepted,
                    energy_before=float(
                        transition.non_worsening.energy_before
                    ),
                    energy_after=float(
                        transition.non_worsening.energy_after
                    ),
                    refit_policy=transition.refit.policy_identity,
                    refit_scope=transition.refit.scope_identity,
                    refit_chart_dimension=int(
                        transition.refit.chart_dimension
                    ),
                    refit_active_logical_indices=tuple(
                        int(value)
                        for value in transition.refit.active_logical_indices
                    ),
                    refit_supported_rank=int(
                        transition.refit.supported_rank
                    ),
                    trust_policy=transition.trust.policy_identity,
                    non_worsening_absolute_tolerance=float(
                        transition.non_worsening.absolute_tolerance
                    ),
                    estimator_prefix_before=(
                        ledger.prefix_identity_before
                    ),
                    estimator_prefix_after=ledger.prefix_identity_after,
                    ledger_closure_sha256=ledger.closure_identity,
                    round_s_alg=sum(
                        value
                        for _, value in ledger.round_s_alg_components
                    ),
                    round_s_unique=sum(
                        value
                        for _, value in ledger.round_s_unique_components
                    ),
                    cumulative_s_alg=int(ledger.cumulative_s_alg),
                    cumulative_s_unique=int(ledger.cumulative_s_unique),
                    pruning=_recoverability_prune_receipt(transition),
                )
            )
            replay.append(
                CombinatorialBatchScientificReplayReceipt(
                    controller_round=controller_round,
                    proposal=proposal,
                    admission=batch_admission,
                    energy_before_refit=float(
                        row["energy_before_opt"]
                    ),
                    accepted_state=accepted,
                    phase=_phase_receipt(row),
                    trust_solve=_trust_receipt(row),
                    accepted_refit=_accepted_refit_receipt(row),
                    checkpoint=_checkpoint_receipt(checkpoint),
                )
            )
            continue
        if isinstance(transition, _AcceptedGreedyBatchTransition):
            proposal, batch_admission = _greedy_batch_public_receipts(
                transition
            )
            selected = transition.decision.selected
            projected_record_ids = tuple(
                str(value)
                for value in row["greedy_batch_admission"][
                    "selected_record_ids"
                ]
            )
            projected_operators = tuple(
                str(value) for value in row["selected_batch_labels"]
            )
            projected_positions = tuple(
                int(value) for value in row["selected_batch_positions"]
            )
            if (
                projected_record_ids
                != tuple(record.domain_record_id for record in selected)
                or projected_operators
                != tuple(record.pool_label for record in selected)
                or projected_positions
                != tuple(record.insertion_position for record in selected)
            ):
                raise RuntimeError(
                    "Controller batch decision disagrees with its projected "
                    "history row."
                )
            accepted_states.append(accepted)
            ledger = transition.ledger
            accepted_transitions.append(
                GreedyBatchAcceptedTransitionReceipt(
                    controller_round=int(next_state.controller_round),
                    preceding_state_fingerprint=(
                        transition
                        .preceding_state
                        .accepted_state_fingerprint
                    ),
                    proposal=proposal,
                    admission=batch_admission,
                    accepted_state=accepted,
                    energy_before=float(
                        transition.non_worsening.energy_before
                    ),
                    energy_after=float(
                        transition.non_worsening.energy_after
                    ),
                    refit_policy=transition.refit.policy_identity,
                    refit_scope=transition.refit.scope_identity,
                    refit_chart_dimension=int(
                        transition.refit.chart_dimension
                    ),
                    refit_active_logical_indices=tuple(
                        int(value)
                        for value in transition.refit.active_logical_indices
                    ),
                    refit_supported_rank=int(
                        transition.refit.supported_rank
                    ),
                    trust_policy=transition.trust.policy_identity,
                    non_worsening_absolute_tolerance=float(
                        transition.non_worsening.absolute_tolerance
                    ),
                    estimator_prefix_before=(
                        ledger.prefix_identity_before
                    ),
                    estimator_prefix_after=ledger.prefix_identity_after,
                    ledger_closure_sha256=ledger.closure_identity,
                    round_s_alg=sum(
                        value
                        for _, value in ledger.round_s_alg_components
                    ),
                    round_s_unique=sum(
                        value
                        for _, value in ledger.round_s_unique_components
                    ),
                    cumulative_s_alg=int(ledger.cumulative_s_alg),
                    cumulative_s_unique=int(ledger.cumulative_s_unique),
                    pruning=_recoverability_prune_receipt(transition),
                )
            )
            replay.append(
                GreedyBatchScientificReplayReceipt(
                    controller_round=controller_round,
                    proposal=proposal,
                    admission=batch_admission,
                    energy_before_refit=float(
                        row["energy_before_opt"]
                    ),
                    accepted_state=accepted,
                    phase=_phase_receipt(row),
                    trust_solve=_trust_receipt(row),
                    accepted_refit=_accepted_refit_receipt(row),
                    checkpoint=_checkpoint_receipt(checkpoint),
                )
            )
            continue
        selected = transition.decision.selected
        if (
            str(row["generator_id"]) != selected.generator_id
            or str(row["selected_op"]) != selected.pool_label
            or int(row["selected_position"]) != selected.insertion_position
        ):
            raise RuntimeError(
                "Controller admission decision disagrees with its projected "
                "history row."
            )
        accepted_states.append(accepted)
        ledger = transition.ledger
        accepted_transitions.append(
            AcceptedTransitionReceipt(
                controller_round=int(next_state.controller_round),
                preceding_state_fingerprint=(
                    transition.preceding_state.accepted_state_fingerprint
                ),
                selected_domain_record_id=selected.domain_record_id,
                generator_id=selected.generator_id,
                selected_operator=selected.pool_label,
                pool_index=int(selected.pool_index),
                insertion_position=int(selected.insertion_position),
                initial_logical_value=float(
                    transition.admission.initial_logical_value
                ),
                accepted_state_fingerprint=(
                    next_state.accepted_state_fingerprint
                ),
                energy_before=float(
                    transition.non_worsening.energy_before
                ),
                energy_after=float(
                    transition.non_worsening.energy_after
                ),
                refit_policy=transition.refit.policy_identity,
                refit_scope=transition.refit.scope_identity,
                refit_supported_rank=int(transition.refit.supported_rank),
                trust_policy=transition.trust.policy_identity,
                non_worsening_absolute_tolerance=float(
                    transition.non_worsening.absolute_tolerance
                ),
                estimator_prefix_before=ledger.prefix_identity_before,
                estimator_prefix_after=ledger.prefix_identity_after,
                ledger_closure_sha256=ledger.closure_identity,
                round_s_alg=sum(
                    value for _, value in ledger.round_s_alg_components
                ),
                round_s_unique=sum(
                    value for _, value in ledger.round_s_unique_components
                ),
                cumulative_s_alg=int(ledger.cumulative_s_alg),
                cumulative_s_unique=int(ledger.cumulative_s_unique),
                pruning=_recoverability_prune_receipt(transition),
            )
        )
        replay.append(
            ScientificReplayReceipt(
                controller_round=controller_round,
                generator_id=str(row["generator_id"]),
                selected_operator=str(row["selected_op"]),
                selected_position=int(row["selected_position"]),
                energy_before_refit=float(row["energy_before_opt"]),
                accepted_state=accepted,
                phase=_phase_receipt(row),
                trust_solve=_trust_receipt(row),
                accepted_refit=_accepted_refit_receipt(row),
                checkpoint=_checkpoint_receipt(checkpoint),
            )
        )
    return (
        tuple(accepted_states),
        tuple(accepted_transitions),
        tuple(replay),
    )


_ESTIMATOR_COMPONENT_KEYS = (
    "N_H_outer",
    "N_H_refit",
    "N_grad",
    "N_metric",
)


def _nonnegative_count(value: Any, *, name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a nonnegative integer.") from exc
    if isinstance(value, bool) or resolved != value or resolved < 0:
        raise RuntimeError(f"{name} must be a nonnegative integer.")
    return resolved


def _components(
    value: Mapping[str, Any],
    *,
    name: str = "estimator components",
) -> EstimatorComponentsReceipt:
    missing = tuple(key for key in _ESTIMATOR_COMPONENT_KEYS if key not in value)
    if missing:
        raise RuntimeError(
            f"{name} is missing required component keys: {', '.join(missing)}."
        )
    return EstimatorComponentsReceipt(
        n_h_outer=_nonnegative_count(
            value["N_H_outer"],
            name=f"{name}.N_H_outer",
        ),
        n_h_refit=_nonnegative_count(
            value["N_H_refit"],
            name=f"{name}.N_H_refit",
        ),
        n_grad=_nonnegative_count(value["N_grad"], name=f"{name}.N_grad"),
        n_metric=_nonnegative_count(value["N_metric"], name=f"{name}.N_metric"),
    )


def _work(
    value: Mapping[str, Any],
    *,
    name: str = "estimator work",
) -> EstimatorWorkReceipt:
    components = _components(
        _require_mapping(
            value.get("components"),
            name=f"{name} components",
        ),
        name=f"{name} components",
    )
    s_alg = _nonnegative_count(value.get("S_alg"), name=f"{name}.S_alg")
    component_total = (
        components.n_h_outer
        + components.n_h_refit
        + components.n_grad
        + components.n_metric
    )
    if s_alg != component_total:
        raise RuntimeError(
            f"{name}.S_alg does not equal the sum of its components: "
            f"{s_alg} != {component_total}."
        )
    return EstimatorWorkReceipt(components=components, s_alg=s_alg)


def _accounting_receipt(
    finalization: _DefaultControllerFinalization,
) -> EstimatorAccountingReceipt:
    accounting = finalization.estimator_call_accounting
    occurrences = _require_mapping(
        accounting.get("executed_occurrence_accounting"),
        name="executed occurrence accounting",
    )
    all_execution = _require_mapping(
        occurrences.get("all_execution"),
        name="all-execution occurrence accounting",
    )
    raw_components = _components(
        _require_mapping(
            all_execution.get("component_occurrence_counts"),
            name="raw estimator occurrence components",
        ),
        name="raw estimator occurrence components",
    )
    raw_occurrence_total = _nonnegative_count(
        all_execution.get("total_call_occurrences"),
        name="raw estimator occurrence total",
    )
    raw_component_total = (
        raw_components.n_h_outer
        + raw_components.n_h_refit
        + raw_components.n_grad
        + raw_components.n_metric
    )
    if raw_occurrence_total != raw_component_total:
        raise RuntimeError(
            "Raw estimator occurrence total does not equal the sum of its "
            f"components: {raw_occurrence_total} != {raw_component_total}."
        )
    continuation = finalization.continuation
    closure = _require_mapping(
        continuation.get("active_prefix_estimator_ledger_closure"),
        name="active-prefix estimator-ledger closure",
    )
    return EstimatorAccountingReceipt(
        complete=bool(accounting["complete"]),
        status=str(accounting["status"]),
        exact_blockers=tuple(str(value) for value in accounting["exact_blockers"]),
        all_work=_work(
            _require_mapping(
                accounting.get("all_branch_search_work"),
                name="all-work estimator accounting",
            ),
            name="all-work estimator accounting",
        ),
        winning_lineage=_work(
            _require_mapping(
                accounting.get("winning_lineage"),
                name="winning-lineage estimator accounting",
            ),
            name="winning-lineage estimator accounting",
        ),
        raw_occurrences=raw_components,
        raw_occurrence_total=raw_occurrence_total,
        prefix_closure_passed=bool(closure["passed"]),
        prefix_closure_status=str(closure["status"]),
    )


def _route_receipt(
    finalization: _DefaultControllerFinalization,
    context: _ResolvedExecutionContext,
) -> RouteReceipt:
    payload_contract = finalization.route_contract
    resolved_runtime_contract = context.canonical_runtime_kwargs().get(
        "sr_route_profile_contract"
    )
    if not isinstance(resolved_runtime_contract, Mapping):
        raise RuntimeError(
            "Resolved execution context lost its route contract."
        )
    if dict(payload_contract) != dict(resolved_runtime_contract):
        raise RuntimeError(
            "SR-SNAKE numerical projection route contract disagrees with its "
            "resolved execution context."
        )
    if (
        finalization.route_family != context.route.family
        or finalization.route_profile != context.route.profile
        or finalization.route_contract_sha256
        != context.route.contract_sha256
    ):
        raise RuntimeError(
            "SR-SNAKE numerical projection route identity disagrees with its "
            "resolved execution context."
        )
    request = context.request
    settings = _require_mapping(
        context.route.contract.get("execution_settings"),
        name="route-profile execution settings",
    )
    history = tuple(row.record for row in finalization.history)
    first_refit_policy = (
        str(
            _require_mapping(
                _require_mapping(history[0], name="history row").get(
                    "accepted_refit"
                ),
                name="accepted-refit receipt",
            )["policy"]
        )
        if history
        else str(settings["adapt_accepted_refit_coordinate_chart"])
    )
    return RouteReceipt(
        family=context.route.family,
        profile_request=context.route.profile_request,
        profile=context.route.profile,
        contract_sha256=context.route.contract_sha256,
        method=request.method,
        admission_policy=request.method.admission.kind,
        insertion_policy=request.method.insertion.kind,
        pruning_policy=request.method.pruning.kind,
        beam_policy=request.method.beam.kind,
        execution=ResolvedExecutionReceipt(
            pool=str(settings["adapt_pool"]),
            optimizer=str(settings["adapt_inner_optimizer"]),
            optimizer_maxiter=int(settings["adapt_maxiter"]),
            seed=int(settings["adapt_seed"]),
            phase0_enabled=bool(settings["phase0_pilot_enabled"]),
            phase2_batching_enabled=bool(settings["phase2_enable_batching"]),
            phase3_batching_enabled=bool(settings["phase3_enable_batching"]),
            pruning_enabled=bool(settings["phase1_prune_enabled"]),
            beam_enabled=request.method.beam.kind == "fork_local",
            phase_live_hysteresis_enabled=False,
            phase3_response_coordinate_scope=str(
                settings["phase3_response_coordinate_scope"]
            ),
            trust_policy=str(
                settings["historical_singleton_trust_region_update_policy"]
            ),
            accepted_refit_policy=first_refit_policy,
            accepted_refit_scope=str(settings["adapt_accepted_refit_scope"]),
            accepted_refit_coordinate_chart=str(
                settings["adapt_accepted_refit_coordinate_chart"]
            ),
        ),
    )


def _canonical_prefix_work(
    outcome: _ControllerOutcome,
) -> tuple[EstimatorWorkReceipt, ...]:
    """Project every accepted prefix from the global occurrence ledger."""

    projected: list[EstimatorWorkReceipt] = []
    for accepted_prefix in outcome.accepted_prefix_all_work:
        values = dict(accepted_prefix.components)
        missing = set(_ESTIMATOR_COMPONENT_KEYS) - set(values)
        extra = set(values) - set(_ESTIMATOR_COMPONENT_KEYS)
        if missing or extra:
            raise RuntimeError(
                "Accepted-prefix estimator work has an incomplete component "
                f"closure; missing={sorted(missing)!r}, "
                f"extra={sorted(extra)!r}."
            )
        components = EstimatorComponentsReceipt(
            n_h_outer=int(values["N_H_outer"]),
            n_h_refit=int(values["N_H_refit"]),
            n_grad=int(values["N_grad"]),
            n_metric=int(values["N_metric"]),
        )
        s_alg = sum(int(value) for value in values.values())
        if int(accepted_prefix.s_alg) != s_alg:
            raise RuntimeError(
                "Accepted-prefix cumulative S_alg does not close to its "
                "occurrence components."
            )
        projected.append(
            EstimatorWorkReceipt(
                components=components,
                s_alg=s_alg,
            )
        )
    return tuple(projected)


def _canonical_reporting_receipt(
    *,
    context: _ResolvedExecutionContext,
    outcome: _ControllerOutcome,
    authenticated_resume_prefix_work: tuple[
        EstimatorWorkReceipt,
        ...,
    ] = (),
) -> CanonicalReportingReceipt:
    state = np.asarray(
        context.initial_state.build_state(),
        dtype=complex,
    ).reshape(-1)
    expected_size = 1 << int(context.problem_receipt.total_qubits)
    if state.size != expected_size:
        raise RuntimeError(
            "Canonical reference state does not cover the resolved qubit "
            "register."
        )
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(
            "Canonical reference state must have finite nonzero norm."
        )
    normalized = np.asarray(state / norm, dtype=complex)
    semantic_invariants = context.route.contract.get(
        "semantic_invariants", {}
    )
    result_candidate_representation = (
        semantic_invariants.get("result_candidate_representation")
        if isinstance(semantic_invariants, Mapping)
        else None
    )
    return CanonicalReportingReceipt(
        exact_same_cutoff_energy=float(
            context.exact_same_cutoff_energy
        ),
        reference_state=ReferenceStateReceipt(
            amplitudes_real=tuple(
                float(value) for value in normalized.real
            ),
            amplitudes_imaginary=tuple(
                float(value) for value in normalized.imag
            ),
            qubit_count=int(context.problem_receipt.total_qubits),
            source_label=str(context.initial_state.source_label),
            state_fingerprint=projective_state_fingerprint(normalized),
        ),
        horizon_scope=(
            "deliberately_stopped_prefix"
            if outcome.stop.primary_reason == "maximum_controller_rounds"
            else "natural_terminal"
        ),
        candidate_representation=(
            CANONICAL_CANDIDATE_REPRESENTATION
            if result_candidate_representation in {None, ""}
            else str(result_candidate_representation)
        ),
        accepted_prefix_work=(
            *authenticated_resume_prefix_work,
            *_canonical_prefix_work(outcome),
        ),
    )


@dataclass(frozen=True, slots=True)
class _CompletedSRRun:
    """Private typed result plus consumer-complete compatibility data."""

    result: SRRunResult
    finalization: _DefaultControllerFinalization


def _execute_sr_snake(
    problem: ResolvedProblemContext,
    request: SRRunRequest | None = None,
) -> _CompletedSRRun:
    """Execute the characterized route while retaining private projection data."""

    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    normalized = SRRunRequest() if request is None else request
    if not isinstance(normalized, SRRunRequest):
        raise TypeError("request must be an SRRunRequest or None.")

    context = _resolve_execution_context(problem, normalized)
    return _execute_resolved_context(context, normalized)


def _execute_resolved_context(
    context: _ResolvedExecutionContext,
    normalized: SRRunRequest,
) -> _CompletedSRRun:
    """Execute an authenticated resolved context through the shared controller."""

    if not isinstance(context, _ResolvedExecutionContext):
        raise TypeError("context must be a _ResolvedExecutionContext.")
    if not isinstance(normalized, SRRunRequest):
        raise TypeError("normalized must be an SRRunRequest.")
    if context.request is not normalized:
        raise ValueError("Resolved context and request identities disagree.")
    _prepare_observation_destinations(context)
    runtime = context.build_default_controller_runtime()
    admission = context.request.method.admission
    beam = context.request.method.beam
    if isinstance(beam, ForkLocalBeam):
        outcome = _run_default_fork_local_beam_controller(
            runtime,
            context.stop,
            admission,
            beam,
        )
    elif isinstance(admission, GreedyBatchAdmission):
        outcome = _run_default_greedy_batch_controller(
            runtime,
            context.stop,
            admission,
        )
    elif isinstance(admission, CombinatorialBatchAdmission):
        outcome = _run_default_combinatorial_batch_controller(
            runtime,
            context.stop,
            admission,
        )
    elif isinstance(admission, SingletonAdmission):
        outcome = _run_default_singleton_controller(runtime, context.stop)
    else:
        runtime.close()
        raise RuntimeError(
            "Resolved SR-SNAKE admission has no authorized controller."
        )
    finalization = outcome.finalization

    (
        resume_accepted_trajectory,
        resume_accepted_transitions,
        resume_replay,
        resume_prefix_work,
    ) = _authenticated_resume_prefix_receipts(context)
    (
        new_accepted_trajectory,
        new_accepted_transitions,
        new_replay,
    ) = _scientific_replay(
        outcome,
        initial_insertion_position_provenance=(
            ()
            if not resume_accepted_trajectory
            else resume_accepted_trajectory[-1].insertion_positions
        ),
    )
    accepted_trajectory = (
        *resume_accepted_trajectory,
        *new_accepted_trajectory,
    )
    accepted_transitions = (
        *resume_accepted_transitions,
        *new_accepted_transitions,
    )
    replay = (*resume_replay, *new_replay)
    stationary_without_transition = bool(
        not accepted_trajectory
        and outcome.stop.terminal_controller_outcome
        == "phase0_stationary_no_competitive_candidate_v1"
    )
    phase3_no_admission_without_transition = bool(
        not accepted_trajectory
        and outcome.stop.terminal_controller_outcome
        == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
    )
    authenticated_no_admission = bool(
        stationary_without_transition
        or phase3_no_admission_without_transition
    )
    if not accepted_trajectory and not authenticated_no_admission:
        raise RuntimeError(
            "The default controller finalized without an accepted transition."
        )
    final_state = (
        _stationary_initial_state(
            outcome.final_state,
            public_state_fingerprint=projective_state_fingerprint(
                np.asarray(
                    context.initial_state.build_state(),
                    dtype=complex,
                ).reshape(-1)
            ),
        )
        if authenticated_no_admission
        else accepted_trajectory[-1]
    )
    if (
        final_state.controller_round != outcome.final_state.controller_round
        or final_state.energy != outcome.final_state.accepted_energy
    ):
        raise RuntimeError(
            "The typed final result disagrees with the controller final state."
        )
    result = SRRunResult(
        final_state=final_state,
        accepted_trajectory=accepted_trajectory,
        accepted_transitions=accepted_transitions,
        problem=context.problem_receipt,
        route=_route_receipt(finalization, context),
        stop=outcome.stop,
        scientific_replay=replay,
        estimator_accounting=_accounting_receipt(finalization),
        observation=ObservationReceipt(
            artifacts=_project_observation_artifacts(
                context,
                finalization.to_serialization_mapping(),
            ),
        ),
        canonical_reporting=_canonical_reporting_receipt(
            context=context,
            outcome=outcome,
            authenticated_resume_prefix_work=resume_prefix_work,
        ),
    )
    from pipelines.reporting.paper_i_run_summary import (
        summarize_paper_i_run,
    )

    insertion = normalized.method.insertion
    if accepted_trajectory and isinstance(
        insertion,
        (
            AlwaysCommutationReducedInsertion,
            AppendCommutationReducedInsertion,
            PlateauCommutationInsertion,
        ),
    ) or (
        bool(accepted_trajectory)
        and
        isinstance(insertion, AppendOnlyInsertion)
        and result.route.family == "ra_adapt"
    ):
        result = replace(
            result,
            paper_i_summary=summarize_paper_i_run(
                result,
                requested_controller_rounds=_paper_i_requested_controller_rounds(
                    (
                        ()
                        if normalized.observation.resource_rounds is None
                        else normalized.observation.resource_rounds
                    ),
                    accepted_round_count=len(accepted_trajectory),
                    terminal_controller_outcome=(
                        outcome.stop.terminal_controller_outcome
                    ),
                ),
            ),
        )
    return _CompletedSRRun(
        result=result,
        finalization=finalization,
    )


__all__ = ["_CompletedSRRun", "_execute_resolved_context", "_execute_sr_snake"]
