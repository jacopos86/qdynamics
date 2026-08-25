"""Authoritative controller for the exact default SR-SNAKE route.

The controller owns the public accepted-state machine:

``accepted state -> selection -> accepted transition -> event projection ->
next accepted state -> configured stop``.

Live numerical arrays, compiled executors, optimizer objects, estimator
instrumentation, and checkpoint serializers remain behind the numerical-session
protocol. The session prepares one round operation at a time; it never owns
controller iteration or configured stopping.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Callable, Protocol

from pipelines.static_adapt.sr_snake._selection import (
    _PHASE0_STATIONARY_TERMINAL_OUTCOME,
    _PHASE3_NO_POSITIVE_TERMINAL_OUTCOME,
    _CombinatorialBatchAdmissionDecision,
    _GreedyBatchAdmissionDecision,
    _SRControllerState,
    _SelectionWorkspace,
    _SingletonAdmissionDecision,
    _NoPositivePhaseIIISelection,
    _StationaryPhase0Selection,
    _select_combinatorial_batch,
    _select_greedy_batch,
    _select_singleton,
)
from pipelines.static_adapt.sr_snake._transition import (
    _AcceptedCombinatorialBatchTransition,
    _AcceptedGreedyBatchTransition,
    _AcceptedSingletonTransition,
    _AcceptedStateSnapshot,
    _CheckpointReadyAcceptedStateEvent,
    _TransitionWorkspace,
    _transition_combinatorial_batch,
    _transition_greedy_batch,
    _transition_singleton,
)
from pipelines.static_adapt.sr_snake.contracts import (
    CombinatorialBatchAdmission,
    GreedyBatchAdmission,
    SRStopPolicy,
    SingletonAdmission,
    StopConditionReceipt,
    StopReceipt,
)


@dataclass(frozen=True, slots=True)
class _PreparedSelection:
    """One exact-round selection prepared by the numerical session."""

    controller_state: _SRControllerState
    workspace: _SelectionWorkspace


@dataclass(frozen=True, slots=True)
class _ImmutableProjectionRecord(Mapping[str, Any]):
    """Copy-isolated view over one controller projection record."""

    _values: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._values, Mapping):
            raise TypeError("projection record values must be a mapping")
        values = copy.deepcopy(dict(self._values))
        if any(not isinstance(key, str) for key in values):
            raise TypeError("projection record keys must be strings")
        object.__setattr__(
            self,
            "_values",
            MappingProxyType(values),
        )

    def __getitem__(self, key: str) -> Any:
        return copy.deepcopy(self._values[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_mutable_mapping(self) -> dict[str, Any]:
        """Return one mutable copy of the retained projection fields."""

        return copy.deepcopy(dict(self._values))


@dataclass(frozen=True, slots=True)
class _AcceptedCheckpointProjection:
    """Immutable active-prefix checkpoint produced after one acceptance."""

    record: _ImmutableProjectionRecord

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
    ) -> _AcceptedCheckpointProjection:
        return cls(record=_ImmutableProjectionRecord(record))


@dataclass(frozen=True, slots=True)
class _AcceptedReplayProjection:
    """Immutable replay/history row produced after one acceptance."""

    record: _ImmutableProjectionRecord

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
    ) -> _AcceptedReplayProjection:
        return cls(record=_ImmutableProjectionRecord(record))


@dataclass(frozen=True, slots=True)
class _ProjectedAcceptedRound:
    """Observation-independent in-memory projection of one accepted event."""

    controller_round: int
    accepted_state_fingerprint: str
    checkpoint_projection: _AcceptedCheckpointProjection
    replay_projection: _AcceptedReplayProjection

    def __post_init__(self) -> None:
        if self.controller_round <= 0:
            raise ValueError(
                "projected accepted rounds must be one-based and positive"
            )
        if not self.accepted_state_fingerprint:
            raise ValueError(
                "projected accepted-state fingerprint must be non-empty"
            )
        if not isinstance(
            self.checkpoint_projection,
            _AcceptedCheckpointProjection,
        ):
            raise TypeError(
                "accepted-round checkpoint projection has the wrong type"
            )
        if not isinstance(
            self.replay_projection,
            _AcceptedReplayProjection,
        ):
            raise TypeError(
                "accepted-round replay projection has the wrong type"
            )


@dataclass(frozen=True, slots=True)
class _DefaultControllerFinalization:
    """Typed terminal numerical projection consumed by the facade runner."""

    _record: _ImmutableProjectionRecord = field(repr=False)
    route_family: str = field(init=False)
    route_profile: str = field(init=False)
    route_contract: _ImmutableProjectionRecord = field(init=False)
    route_contract_sha256: str = field(init=False)
    history: tuple[_AcceptedReplayProjection, ...] = field(init=False)
    estimator_call_accounting: _ImmutableProjectionRecord = field(init=False)
    continuation: _ImmutableProjectionRecord = field(init=False)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> _DefaultControllerFinalization:
        return cls(_record=_ImmutableProjectionRecord(payload))

    def __post_init__(self) -> None:
        if not isinstance(self._record, _ImmutableProjectionRecord):
            raise TypeError("finalization requires an immutable record")
        if self._record.get("success") is not True:
            raise ValueError(
                "default controller finalization requires successful output"
            )
        route_family = str(self._record.get("route_family", ""))
        route_profile = str(self._record.get("route_profile", ""))
        route_contract_sha256 = str(
            self._record.get("sr_route_profile_contract_sha256", "")
        )
        if not route_family or not route_profile or not route_contract_sha256:
            raise ValueError(
                "default controller finalization is missing route identity"
            )
        route_contract = self._record.get("sr_route_profile_contract")
        accounting = self._record.get("estimator_call_accounting")
        continuation = self._record.get("continuation")
        history = self._record.get("history")
        if not isinstance(route_contract, Mapping):
            raise TypeError("finalization route contract must be a mapping")
        if not isinstance(accounting, Mapping):
            raise TypeError(
                "finalization estimator accounting must be a mapping"
            )
        if not isinstance(continuation, Mapping):
            raise TypeError("finalization continuation must be a mapping")
        if not isinstance(history, Sequence) or isinstance(
            history,
            (str, bytes),
        ):
            raise TypeError("finalization history must be a sequence")
        replay_rows: list[_AcceptedReplayProjection] = []
        for row in history:
            if not isinstance(row, Mapping):
                raise TypeError("finalization history rows must be mappings")
            replay_rows.append(_AcceptedReplayProjection.from_mapping(row))
        terminal_outcome = self._record.get(
            "terminal_controller_outcome"
        )
        if not replay_rows and terminal_outcome not in {
            _PHASE0_STATIONARY_TERMINAL_OUTCOME,
            _PHASE3_NO_POSITIVE_TERMINAL_OUTCOME,
        }:
            raise ValueError(
                "default controller finalization requires accepted history"
            )
        if terminal_outcome is not None and terminal_outcome not in {
            _PHASE0_STATIONARY_TERMINAL_OUTCOME,
            _PHASE3_NO_POSITIVE_TERMINAL_OUTCOME,
        }:
            raise ValueError("default controller terminal outcome is unknown")
        phase3_terminal_receipt = self._record.get(
            "terminal_phase3_selection_receipt"
        )
        continuation_phase3_terminal_receipt = continuation.get(
            "terminal_phase3_selection_receipt"
        )
        if terminal_outcome == _PHASE3_NO_POSITIVE_TERMINAL_OUTCOME:
            if (
                not isinstance(phase3_terminal_receipt, Mapping)
                or continuation_phase3_terminal_receipt
                != phase3_terminal_receipt
                or (
                    not replay_rows
                    and int(
                        phase3_terminal_receipt.get(
                            "accepted_controller_round",
                            -1,
                        )
                    )
                    != 0
                )
            ):
                raise ValueError(
                    "Phase-III terminal receipt is missing or detached from "
                    "continuation."
                )
            native_contract = route_contract.get("native_semantic_contract")
            terminal_checkpoint = self._record.get(
                "terminal_active_prefix_checkpoint"
            )
            if (
                not isinstance(native_contract, Mapping)
                or not isinstance(terminal_checkpoint, Mapping)
            ):
                raise ValueError(
                    "Phase-III terminal finalization requires its authenticated "
                    "V2 route and terminal checkpoint."
                )
            from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
                validate_semantic_phase3_no_positive_terminal_receipt,
            )

            try:
                validate_semantic_phase3_no_positive_terminal_receipt(
                    phase3_terminal_receipt,
                    route_variant=str(native_contract.get("route_variant", "")),
                    route_contract=route_contract,
                    expected_route_contract_sha256=route_contract_sha256,
                    accepted_round_count=len(replay_rows),
                    terminal_active_prefix_checkpoint=terminal_checkpoint,
                    finalization=self._record.to_mutable_mapping(),
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    "Phase-III terminal finalization requires independently "
                    "authenticated V2 route and receipt evidence."
                ) from exc
        elif (
            phase3_terminal_receipt is not None
            or continuation_phase3_terminal_receipt is not None
        ):
            raise ValueError(
                "Phase-III terminal receipt requires its exact terminal "
                "outcome."
            )
        object.__setattr__(self, "route_family", route_family)
        object.__setattr__(self, "route_profile", route_profile)
        object.__setattr__(
            self,
            "route_contract",
            _ImmutableProjectionRecord(route_contract),
        )
        object.__setattr__(
            self,
            "route_contract_sha256",
            route_contract_sha256,
        )
        object.__setattr__(self, "history", tuple(replay_rows))
        object.__setattr__(
            self,
            "estimator_call_accounting",
            _ImmutableProjectionRecord(accounting),
        )
        object.__setattr__(
            self,
            "continuation",
            _ImmutableProjectionRecord(continuation),
        )

    def to_serialization_mapping(self) -> dict[str, Any]:
        """Return consumer-required controller fields for observation I/O.

        This is not the historical executor's full diagnostic union.  Reachability
        and retirement of omitted diagnostics remain explicit Issue-20 debt.
        """

        return self._record.to_mutable_mapping()


class _DefaultControllerNumericalRuntime(Protocol):
    """Direct numerical and observation boundary for the exact default."""

    initial_accepted_state: _AcceptedStateSnapshot

    def prepare_selection(
        self,
        state: _AcceptedStateSnapshot,
    ) -> _PreparedSelection: ...

    def prepare_transition(
        self,
        state: _AcceptedStateSnapshot,
        decision: (
            _SingletonAdmissionDecision
            | _GreedyBatchAdmissionDecision
            | _CombinatorialBatchAdmissionDecision
        ),
    ) -> _TransitionWorkspace: ...

    def project_accepted_event(
        self,
        event: _CheckpointReadyAcceptedStateEvent,
        transition: (
            _AcceptedSingletonTransition
            | _AcceptedGreedyBatchTransition
            | _AcceptedCombinatorialBatchTransition
        ),
    ) -> _ProjectedAcceptedRound: ...

    def finalize(
        self,
        *,
        final_state: _AcceptedStateSnapshot,
        transitions: tuple[
            _AcceptedSingletonTransition
            | _AcceptedGreedyBatchTransition
            | _AcceptedCombinatorialBatchTransition,
            ...,
        ],
        events: tuple[_CheckpointReadyAcceptedStateEvent, ...],
        projected_rounds: tuple[_ProjectedAcceptedRound, ...],
        stop: StopReceipt,
    ) -> _DefaultControllerFinalization: ...

    def finalize_stationary_phase0(
        self,
        *,
        final_state: _AcceptedStateSnapshot,
        transitions: tuple[
            _AcceptedSingletonTransition
            | _AcceptedGreedyBatchTransition
            | _AcceptedCombinatorialBatchTransition,
            ...,
        ],
        events: tuple[_CheckpointReadyAcceptedStateEvent, ...],
        projected_rounds: tuple[_ProjectedAcceptedRound, ...],
        phase0: Any,
        stop: StopReceipt,
    ) -> _DefaultControllerFinalization: ...

    def finalize_no_admission(
        self,
        *,
        final_state: _AcceptedStateSnapshot,
        transitions: tuple[
            _AcceptedSingletonTransition
            | _AcceptedGreedyBatchTransition
            | _AcceptedCombinatorialBatchTransition,
            ...,
        ],
        events: tuple[_CheckpointReadyAcceptedStateEvent, ...],
        projected_rounds: tuple[_ProjectedAcceptedRound, ...],
        terminal_selection: _NoPositivePhaseIIISelection,
        stop: StopReceipt,
    ) -> _DefaultControllerFinalization: ...

    def fork_beam_branch(
        self,
        state: _AcceptedStateSnapshot,
        *,
        branch_id: str,
        parent_branch_id: str | None,
        excluded_pool_indices: Sequence[int] = (),
    ) -> tuple[
        _DefaultControllerNumericalRuntime,
        _AcceptedStateSnapshot,
    ]: ...

    def clear_beam_branch_context(self) -> None: ...

    def beam_executed_s_alg(self) -> int: ...

    def beam_executed_s_alg_components(
        self,
    ) -> tuple[tuple[str, int], ...]: ...

    def beam_resume_seed(self) -> tuple[tuple[str, ...], int]: ...

    def configure_beam_winner(
        self,
        *,
        winning_branch_ids: Sequence[str],
        diagnostics: Mapping[str, Any],
        observation_owner: _DefaultControllerNumericalRuntime,
    ) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _AcceptedPrefixAllWork:
    """Cumulative all-executed estimator work closed after one round."""

    components: tuple[tuple[str, int], ...]
    s_alg: int

    def __post_init__(self) -> None:
        values = dict(self.components)
        if (
            len(values) != len(self.components)
            or any(int(value) < 0 for value in values.values())
            or sum(int(value) for value in values.values()) != int(self.s_alg)
        ):
            raise ValueError(
                "Accepted-prefix all-work components must be unique, "
                "nonnegative, and close to S_alg."
            )

    @classmethod
    def from_transition(
        cls,
        transition: (
            _AcceptedSingletonTransition
            | _AcceptedGreedyBatchTransition
            | _AcceptedCombinatorialBatchTransition
        ),
    ) -> "_AcceptedPrefixAllWork":
        return cls(
            components=tuple(
                (str(name), int(value))
                for name, value in (
                    transition.ledger.cumulative_s_alg_components
                )
            ),
            s_alg=int(transition.ledger.cumulative_s_alg),
        )


@dataclass(frozen=True, slots=True)
class _ControllerOutcome:
    """Frozen in-memory outcome consumed by the default facade runner."""

    initial_state: _AcceptedStateSnapshot
    final_state: _AcceptedStateSnapshot
    accepted_states: tuple[_AcceptedStateSnapshot, ...]
    transitions: tuple[
        _AcceptedSingletonTransition
        | _AcceptedGreedyBatchTransition
        | _AcceptedCombinatorialBatchTransition,
        ...,
    ]
    events: tuple[_CheckpointReadyAcceptedStateEvent, ...]
    projected_rounds: tuple[_ProjectedAcceptedRound, ...]
    accepted_prefix_all_work: tuple[_AcceptedPrefixAllWork, ...]
    stop: StopReceipt
    finalization: _DefaultControllerFinalization

    def __post_init__(self) -> None:
        if not isinstance(
            self.finalization,
            _DefaultControllerFinalization,
        ):
            raise TypeError(
                "controller outcome requires its typed finalization"
            )
        cardinalities = {
            len(self.accepted_states),
            len(self.transitions),
            len(self.events),
            len(self.projected_rounds),
            len(self.accepted_prefix_all_work),
        }
        if len(cardinalities) != 1:
            raise ValueError(
                "accepted states, transitions, events, and projections must "
                "be one-to-one"
            )
        if self.accepted_states:
            if self.accepted_states[-1] != self.final_state:
                raise ValueError(
                    "final state must be the last accepted trajectory state"
                )
        elif self.final_state != self.initial_state:
            raise ValueError(
                "an empty controller trajectory must retain its initial state"
            )


def _selection_state_matches_accepted(
    selection: _SRControllerState,
    accepted: _AcceptedStateSnapshot,
) -> bool:
    """Compare the shared immutable cursor fields across the two seams."""

    return (
        selection.controller_round == accepted.controller_round
        and selection.accepted_operator_ids == accepted.accepted_operator_ids
        and selection.accepted_insertion_positions
        == accepted.accepted_insertion_positions
        and selection.logical_parameter_ids == accepted.logical_parameter_ids
        and selection.logical_parameter_values
        == accepted.logical_parameter_values
        and selection.runtime_parameter_ids == accepted.runtime_parameter_ids
        and selection.runtime_parameter_values
        == accepted.runtime_parameter_values
        and math.isclose(
            selection.accepted_energy,
            accepted.accepted_energy,
            rel_tol=0.0,
            abs_tol=(
                128.0
                * math.ulp(
                    max(
                        1.0,
                        abs(selection.accepted_energy),
                        abs(accepted.accepted_energy),
                    )
                )
            ),
        )
        and selection.accepted_state_fingerprint
        == accepted.accepted_state_fingerprint
        and selection.available_generator_ids
        == accepted.available_generator_ids
        and selection.selection_counts == accepted.selection_counts
        and selection.trust_state_identity == accepted.trust_state_identity
        and selection.optimizer_memory_identity
        == accepted.optimizer_memory_identity
        and selection.estimator_prefix_identity
        == accepted.estimator_prefix_identity
    )


def _configured_stop_receipt(
    policy: SRStopPolicy,
    accepted_state: _AcceptedStateSnapshot,
    *,
    accepted_states: tuple[_AcceptedStateSnapshot, ...] = (),
) -> StopReceipt:
    """Evaluate every configured condition on one post-transition state."""

    completed = int(accepted_state.controller_round)
    maximum_fired = completed >= int(policy.maximum_controller_rounds)
    exact = policy.exact_ed_target
    exact_difference = (
        None
        if exact is None
        else abs(
            float(accepted_state.accepted_energy)
            - float(exact.energy)
        )
    )
    exact_hit_rounds = (
        ()
        if exact is None
        else tuple(
            int(state.controller_round)
            for state in accepted_states
            if int(state.controller_round) > 0
            and abs(float(state.accepted_energy) - float(exact.energy))
            <= float(exact.absolute_tolerance)
        )
    )
    exact_first_hit_round = (
        None if not exact_hit_rounds else min(exact_hit_rounds)
    )
    exact_fired = bool(
        exact is not None
        and exact_first_hit_round is not None
        and exact_difference is not None
        and exact_difference <= float(exact.absolute_tolerance)
        and completed
        >= exact_first_hit_round
        + int(exact.confirmation_controller_rounds)
    )

    conditions = [
        StopConditionReceipt(
            reason="maximum_controller_rounds",
            active=True,
            fired=maximum_fired,
        )
    ]
    if exact is not None:
        conditions.append(
            StopConditionReceipt(
                reason="exact_ed_target_reached",
                active=True,
                fired=exact_fired,
            )
        )

    fired_reasons: list[str] = []
    if exact_fired:
        fired_reasons.append("exact_ed_target_reached")
    if maximum_fired:
        fired_reasons.append("maximum_controller_rounds")
    primary_reason = (
        fired_reasons[0] if fired_reasons else "controller_continues"
    )
    return StopReceipt(
        conditions=tuple(conditions),
        completed_controller_rounds=completed,
        accepted_operator_count=len(
            accepted_state.accepted_operator_ids
        ),
        primary_reason=primary_reason,
        fired_reasons=tuple(fired_reasons),
        accepted_energy=float(accepted_state.accepted_energy),
        exact_target_energy=(None if exact is None else float(exact.energy)),
        exact_absolute_tolerance=(
            None
            if exact is None
            else float(exact.absolute_tolerance)
        ),
        exact_observed_absolute_difference=exact_difference,
        exact_source=(None if exact is None else exact.source),
        exact_confirmation_controller_rounds=(
            None
            if exact is None
            else int(exact.confirmation_controller_rounds)
        ),
        exact_first_hit_controller_round=exact_first_hit_round,
    )


def _phase0_stationary_stop_receipt(
    accepted_state: _AcceptedStateSnapshot,
    *,
    phase0: Any,
) -> StopReceipt:
    """Close a valid empty Phase-0 competition without a fake transition."""

    if (
        getattr(phase0, "terminal_outcome", None)
        != _PHASE0_STATIONARY_TERMINAL_OUTCOME
        or tuple(getattr(phase0, "shortlist", ()))
    ):
        raise RuntimeError("stationary controller stop lost its Phase0 receipt")
    return StopReceipt(
        conditions=(
            StopConditionReceipt(
                reason="phase0_stationary",
                active=True,
                fired=True,
            ),
        ),
        completed_controller_rounds=int(accepted_state.controller_round),
        accepted_operator_count=len(accepted_state.accepted_operator_ids),
        primary_reason="phase0_stationary",
        fired_reasons=("phase0_stationary",),
        accepted_energy=float(accepted_state.accepted_energy),
        terminal_controller_outcome=(
            _PHASE0_STATIONARY_TERMINAL_OUTCOME
        ),
    )


def _phase3_no_positive_stop_receipt(
    accepted_state: _AcceptedStateSnapshot,
    *,
    terminal_selection: _NoPositivePhaseIIISelection,
) -> StopReceipt:
    """Close an authenticated Phase-III competition without admission."""

    terminal_selection.validate_natural_terminal_authority()
    phase3 = terminal_selection.phase_iii
    if (
        phase3.terminal_outcome
        != _PHASE3_NO_POSITIVE_TERMINAL_OUTCOME
        or phase3.shortlist
        or phase3.adaptive_shortlist is None
        or phase3.adaptive_shortlist.status != "no_positive_population"
        or phase3.adaptive_shortlist.retained_record_ids
    ):
        raise RuntimeError(
            "no-admission controller stop lost its Phase-III receipt"
        )
    reason = "phase_iii_no_positive_feasible_candidate"
    return StopReceipt(
        conditions=(
            StopConditionReceipt(
                reason=reason,
                active=True,
                fired=True,
            ),
        ),
        completed_controller_rounds=int(accepted_state.controller_round),
        accepted_operator_count=len(accepted_state.accepted_operator_ids),
        primary_reason=reason,
        fired_reasons=(reason,),
        accepted_energy=float(accepted_state.accepted_energy),
        terminal_controller_outcome=(
            _PHASE3_NO_POSITIVE_TERMINAL_OUTCOME
        ),
    )


def _assert_projected_event(
    projection: _ProjectedAcceptedRound,
    event: _CheckpointReadyAcceptedStateEvent,
) -> None:
    if (
        int(projection.controller_round) != int(event.controller_round)
        or projection.accepted_state_fingerprint
        != event.accepted_state_fingerprint
    ):
        raise RuntimeError(
            "accepted-event projection identifies a different controller state"
        )


def _run_default_singleton_controller(
    runtime: _DefaultControllerNumericalRuntime,
    stop_policy: SRStopPolicy,
) -> _ControllerOutcome:
    """Run the exact singleton controller to a configured stop."""

    if not isinstance(stop_policy, SRStopPolicy):
        raise TypeError("stop_policy must be an SRStopPolicy")

    initial_state = runtime.initial_accepted_state
    state = initial_state
    accepted_states: list[_AcceptedStateSnapshot] = []
    transitions: list[_AcceptedSingletonTransition] = []
    events: list[_CheckpointReadyAcceptedStateEvent] = []
    projected_rounds: list[_ProjectedAcceptedRound] = []

    try:
        while True:
            prepared_selection = runtime.prepare_selection(state)
            if not _selection_state_matches_accepted(
                prepared_selection.controller_state,
                state,
            ):
                raise RuntimeError(
                    "prepared selection identifies a different accepted state"
                )
            try:
                decision = _select_singleton(
                    prepared_selection.controller_state,
                    prepared_selection.workspace,
                )
            except _StationaryPhase0Selection as stationary:
                stop = _phase0_stationary_stop_receipt(
                    state,
                    phase0=stationary.receipt,
                )
                transitions_tuple = tuple(transitions)
                events_tuple = tuple(events)
                projected_tuple = tuple(projected_rounds)
                finalization = runtime.finalize_stationary_phase0(
                    final_state=state,
                    transitions=transitions_tuple,
                    events=events_tuple,
                    projected_rounds=projected_tuple,
                    phase0=stationary.receipt,
                    stop=stop,
                )
                return _ControllerOutcome(
                    initial_state=initial_state,
                    final_state=state,
                    accepted_states=tuple(accepted_states),
                    transitions=transitions_tuple,
                    events=events_tuple,
                    projected_rounds=projected_tuple,
                    accepted_prefix_all_work=tuple(
                        _AcceptedPrefixAllWork.from_transition(transition)
                        for transition in transitions_tuple
                    ),
                    stop=stop,
                    finalization=finalization,
                )
            except _NoPositivePhaseIIISelection as exhausted:
                stop = _phase3_no_positive_stop_receipt(
                    state,
                    terminal_selection=exhausted,
                )
                transitions_tuple = tuple(transitions)
                events_tuple = tuple(events)
                projected_tuple = tuple(projected_rounds)
                finalization = runtime.finalize_no_admission(
                    final_state=state,
                    transitions=transitions_tuple,
                    events=events_tuple,
                    projected_rounds=projected_tuple,
                    terminal_selection=exhausted,
                    stop=stop,
                )
                return _ControllerOutcome(
                    initial_state=initial_state,
                    final_state=state,
                    accepted_states=tuple(accepted_states),
                    transitions=transitions_tuple,
                    events=events_tuple,
                    projected_rounds=projected_tuple,
                    accepted_prefix_all_work=tuple(
                        _AcceptedPrefixAllWork.from_transition(transition)
                        for transition in transitions_tuple
                    ),
                    stop=stop,
                    finalization=finalization,
                )
            transition_workspace = runtime.prepare_transition(
                state,
                decision,
            )
            transition = _transition_singleton(
                state,
                decision,
                transition_workspace,
            )

            # The projection must complete before the round becomes public.
            # This preserves event -> history -> active-prefix projection
            # ordering and prevents a partial observation failure from
            # publishing a transition.
            projection = runtime.project_accepted_event(
                transition.checkpoint_event,
                transition,
            )
            _assert_projected_event(
                projection,
                transition.checkpoint_event,
            )

            state = transition.next_state
            accepted_states.append(state)
            transitions.append(transition)
            events.append(transition.checkpoint_event)
            projected_rounds.append(projection)

            stop = _configured_stop_receipt(
                stop_policy,
                state,
                accepted_states=tuple(accepted_states),
            )
            if stop.fired_reasons:
                break

        transitions_tuple = tuple(transitions)
        events_tuple = tuple(events)
        projected_tuple = tuple(projected_rounds)
        finalization = runtime.finalize(
            final_state=state,
            transitions=transitions_tuple,
            events=events_tuple,
            projected_rounds=projected_tuple,
            stop=stop,
        )
        return _ControllerOutcome(
            initial_state=initial_state,
            final_state=state,
            accepted_states=tuple(accepted_states),
            transitions=transitions_tuple,
            events=events_tuple,
            projected_rounds=projected_tuple,
            accepted_prefix_all_work=tuple(
                _AcceptedPrefixAllWork.from_transition(transition)
                for transition in transitions_tuple
            ),
            stop=stop,
            finalization=finalization,
        )
    finally:
        runtime.close()


def _run_default_batch_controller(
    runtime: _DefaultControllerNumericalRuntime,
    stop_policy: SRStopPolicy,
    *,
    select_batch: Callable[
        [_SRControllerState, _SelectionWorkspace],
        _GreedyBatchAdmissionDecision
        | _CombinatorialBatchAdmissionDecision,
    ],
    transition_batch: Callable[
        [
            _AcceptedStateSnapshot,
            _GreedyBatchAdmissionDecision
            | _CombinatorialBatchAdmissionDecision,
            _TransitionWorkspace,
        ],
        _AcceptedGreedyBatchTransition
        | _AcceptedCombinatorialBatchTransition,
    ],
) -> _ControllerOutcome:
    """Run one ordered-batch strategy as one transition per round."""

    if not isinstance(stop_policy, SRStopPolicy):
        raise TypeError("stop_policy must be an SRStopPolicy")

    initial_state = runtime.initial_accepted_state
    state = initial_state
    accepted_states: list[_AcceptedStateSnapshot] = []
    transitions: list[
        _AcceptedGreedyBatchTransition
        | _AcceptedCombinatorialBatchTransition
    ] = []
    events: list[_CheckpointReadyAcceptedStateEvent] = []
    projected_rounds: list[_ProjectedAcceptedRound] = []

    try:
        while True:
            prepared_selection = runtime.prepare_selection(state)
            if not _selection_state_matches_accepted(
                prepared_selection.controller_state,
                state,
            ):
                raise RuntimeError(
                    "prepared selection identifies a different accepted state"
                )
            decision = select_batch(
                prepared_selection.controller_state,
                prepared_selection.workspace,
            )
            transition_workspace = runtime.prepare_transition(
                state,
                decision,
            )
            transition = transition_batch(
                state,
                decision,
                transition_workspace,
            )

            projection = runtime.project_accepted_event(
                transition.checkpoint_event,
                transition,
            )
            _assert_projected_event(
                projection,
                transition.checkpoint_event,
            )

            state = transition.next_state
            accepted_states.append(state)
            transitions.append(transition)
            events.append(transition.checkpoint_event)
            projected_rounds.append(projection)

            stop = _configured_stop_receipt(
                stop_policy,
                state,
                accepted_states=tuple(accepted_states),
            )
            if stop.fired_reasons:
                break

        transitions_tuple = tuple(transitions)
        events_tuple = tuple(events)
        projected_tuple = tuple(projected_rounds)
        finalization = runtime.finalize(
            final_state=state,
            transitions=transitions_tuple,
            events=events_tuple,
            projected_rounds=projected_tuple,
            stop=stop,
        )
        return _ControllerOutcome(
            initial_state=initial_state,
            final_state=state,
            accepted_states=tuple(accepted_states),
            transitions=transitions_tuple,
            events=events_tuple,
            projected_rounds=projected_tuple,
            accepted_prefix_all_work=tuple(
                _AcceptedPrefixAllWork.from_transition(transition)
                for transition in transitions_tuple
            ),
            stop=stop,
            finalization=finalization,
        )
    finally:
        runtime.close()


def _run_default_greedy_batch_controller(
    runtime: _DefaultControllerNumericalRuntime,
    stop_policy: SRStopPolicy,
    admission: GreedyBatchAdmission,
) -> _ControllerOutcome:
    """Run the greedy route with one atomic batch transition per round."""

    if not isinstance(stop_policy, SRStopPolicy):
        raise TypeError("stop_policy must be an SRStopPolicy")
    if not isinstance(admission, GreedyBatchAdmission):
        raise TypeError("admission must be a GreedyBatchAdmission")

    def _select(
        state: _SRControllerState,
        workspace: _SelectionWorkspace,
    ) -> _GreedyBatchAdmissionDecision:
        return _select_greedy_batch(
            state,
            workspace,
            maximum_size=admission.maximum_size,
            search_window_size=admission.search_window_size,
        )

    return _run_default_batch_controller(
        runtime,
        stop_policy,
        select_batch=_select,
        transition_batch=_transition_greedy_batch,
    )


def _run_default_combinatorial_batch_controller(
    runtime: _DefaultControllerNumericalRuntime,
    stop_policy: SRStopPolicy,
    admission: CombinatorialBatchAdmission,
) -> _ControllerOutcome:
    """Run the exhaustive-subset route with one atomic round transition."""

    if not isinstance(stop_policy, SRStopPolicy):
        raise TypeError("stop_policy must be an SRStopPolicy")
    if not isinstance(admission, CombinatorialBatchAdmission):
        raise TypeError(
            "admission must be a CombinatorialBatchAdmission"
        )

    def _select(
        state: _SRControllerState,
        workspace: _SelectionWorkspace,
    ) -> _CombinatorialBatchAdmissionDecision:
        return _select_combinatorial_batch(
            state,
            workspace,
            maximum_size=admission.maximum_size,
            search_window_size=(
                admission.resolved_search_window_size
            ),
        )

    return _run_default_batch_controller(
        runtime,
        stop_policy,
        select_batch=_select,
        transition_batch=_transition_combinatorial_batch,
    )


__all__: list[str] = []
