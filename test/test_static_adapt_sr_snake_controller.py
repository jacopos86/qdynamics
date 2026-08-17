from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import pipelines.static_adapt.sr_snake._controller as sr_controller
from pipelines.static_adapt.sr_snake._selection import (
    _PHASE0_STATIONARY_TERMINAL_OUTCOME,
    _PhaseSelectionReceipt,
    _SRControllerState,
    _StationaryPhase0Selection,
)
from pipelines.static_adapt.sr_snake._transition import (
    _AcceptedStateSnapshot,
)
from pipelines.static_adapt.sr_snake.contracts import (
    ExactEDSourceReceipt,
    ExactEDStop,
    ForkLocalBeam,
    SingletonAdmission,
    SRStopPolicy,
)


def _accepted_state(
    controller_round: int,
    *,
    energy: float | None = None,
    operator_count: int | None = None,
) -> _AcceptedStateSnapshot:
    resolved_operator_count = (
        controller_round if operator_count is None else operator_count
    )
    operator_ids = tuple(
        f"generator:{index}" for index in range(resolved_operator_count)
    )
    logical_ids = tuple(
        f"logical:{index}" for index in range(resolved_operator_count)
    )
    runtime_ids = tuple(
        f"runtime:{index}" for index in range(resolved_operator_count)
    )
    return _AcceptedStateSnapshot(
        controller_round=controller_round,
        accepted_operator_ids=operator_ids,
        accepted_insertion_positions=tuple(range(resolved_operator_count)),
        logical_parameter_ids=logical_ids,
        logical_parameter_values=tuple(
            float(index) / 10.0 for index in range(resolved_operator_count)
        ),
        runtime_parameter_ids=runtime_ids,
        runtime_parameter_values=tuple(
            float(index) / 10.0 for index in range(resolved_operator_count)
        ),
        accepted_energy=(
            float(-controller_round) if energy is None else float(energy)
        ),
        accepted_state_fingerprint=f"state:{controller_round}",
        available_generator_ids=tuple(
            f"generator:{index}" for index in range(controller_round, 64)
        ),
        selection_counts=tuple(
            (f"generator:{index}", int(index < controller_round))
            for index in range(64)
        ),
        trust_state_identity=f"trust:{controller_round}",
        optimizer_memory_identity=f"optimizer:{controller_round}",
        estimator_prefix_identity=f"ledger:{controller_round}",
    )


def _selection_state(
    state: _AcceptedStateSnapshot,
) -> _SRControllerState:
    return _SRControllerState(
        controller_round=state.controller_round,
        accepted_operator_ids=state.accepted_operator_ids,
        accepted_insertion_positions=state.accepted_insertion_positions,
        logical_parameter_ids=state.logical_parameter_ids,
        logical_parameter_values=state.logical_parameter_values,
        runtime_parameter_ids=state.runtime_parameter_ids,
        runtime_parameter_values=state.runtime_parameter_values,
        accepted_energy=state.accepted_energy,
        accepted_state_fingerprint=state.accepted_state_fingerprint,
        available_generator_ids=state.available_generator_ids,
        selection_counts=state.selection_counts,
        phase_live=(True, True, True),
        trust_state_identity=state.trust_state_identity,
        optimizer_memory_identity=state.optimizer_memory_identity,
        estimator_prefix_identity=state.estimator_prefix_identity,
    )


def test_selection_state_accepts_only_roundoff_scale_energy_replay() -> None:
    accepted = _accepted_state(50, energy=-0.6238167090829093)
    replayed = replace(
        _selection_state(accepted),
        accepted_energy=-0.6238167090829091,
    )

    assert sr_controller._selection_state_matches_accepted(replayed, accepted)
    assert not sr_controller._selection_state_matches_accepted(
        replace(replayed, accepted_energy=-0.6238167089829093),
        accepted,
    )


class _FakeRuntime:
    def __init__(
        self,
        *,
        energies: tuple[float, ...] = (),
        operator_counts: tuple[int, ...] = (),
        fail_projection_round: int | None = None,
    ) -> None:
        self.initial_accepted_state = _accepted_state(0)
        self.energies = energies
        self.operator_counts = operator_counts
        self.fail_projection_round = fail_projection_round
        self.selection_states: list[_AcceptedStateSnapshot] = []
        self.transition_states: list[_AcceptedStateSnapshot] = []
        self.projected_rounds: list[int] = []
        self.finalize_calls: list[dict[str, Any]] = []
        self.stationary_finalize_calls: list[dict[str, Any]] = []
        self.close_calls = 0

    def prepare_selection(
        self,
        state: _AcceptedStateSnapshot,
    ) -> sr_controller._PreparedSelection:
        self.selection_states.append(state)
        return sr_controller._PreparedSelection(
            controller_state=_selection_state(state),
            workspace=SimpleNamespace(round=state.controller_round),
        )

    def prepare_transition(
        self,
        state: _AcceptedStateSnapshot,
        decision: object,
    ) -> object:
        self.transition_states.append(state)
        return SimpleNamespace(
            round=state.controller_round,
            decision=decision,
        )

    def next_state(
        self,
        preceding: _AcceptedStateSnapshot,
    ) -> _AcceptedStateSnapshot:
        next_round = preceding.controller_round + 1
        energy = (
            self.energies[next_round - 1]
            if next_round <= len(self.energies)
            else float(-next_round)
        )
        operator_count = (
            self.operator_counts[next_round - 1]
            if next_round <= len(self.operator_counts)
            else next_round
        )
        return _accepted_state(
            next_round,
            energy=energy,
            operator_count=operator_count,
        )

    def project_accepted_event(
        self,
        event: object,
        transition: object,
    ) -> sr_controller._ProjectedAcceptedRound:
        round_index = int(getattr(event, "controller_round"))
        if round_index == self.fail_projection_round:
            raise RuntimeError("projection failed")
        self.projected_rounds.append(round_index)
        return sr_controller._ProjectedAcceptedRound(
            controller_round=round_index,
            accepted_state_fingerprint=str(
                getattr(event, "accepted_state_fingerprint")
            ),
            checkpoint_projection=(
                sr_controller._AcceptedCheckpointProjection.from_mapping(
                    {"round": round_index}
                )
            ),
            replay_projection=(
                sr_controller._AcceptedReplayProjection.from_mapping(
                    {"round": round_index}
                )
            ),
        )

    def finalize(
        self,
        **kwargs: Any,
    ) -> sr_controller._DefaultControllerFinalization:
        self.finalize_calls.append(dict(kwargs))
        return sr_controller._DefaultControllerFinalization.from_mapping(
            {
                "success": True,
                "route_family": "test",
                "route_profile": "test",
                "sr_route_profile_contract": {},
                "sr_route_profile_contract_sha256": "test",
                "history": [{"round": value} for value in self.projected_rounds],
                "estimator_call_accounting": {},
                "continuation": {},
                "final_round": kwargs["final_state"].controller_round,
                "projection_count": len(kwargs["projected_rounds"]),
            },
        )

    def finalize_stationary_phase0(
        self,
        **kwargs: Any,
    ) -> sr_controller._DefaultControllerFinalization:
        self.stationary_finalize_calls.append(dict(kwargs))
        return sr_controller._DefaultControllerFinalization.from_mapping(
            {
                "success": True,
                "route_family": "test",
                "route_profile": "test",
                "sr_route_profile_contract": {},
                "sr_route_profile_contract_sha256": "test",
                "history": [],
                "estimator_call_accounting": {},
                "continuation": {},
                "terminal_controller_outcome": (
                    _PHASE0_STATIONARY_TERMINAL_OUTCOME
                ),
                "terminal_phase0_selection_receipt": {
                    "status": "stationary"
                },
            },
        )

    def close(self) -> None:
        self.close_calls += 1


class _FakeBeamRuntime(_FakeRuntime):
    def __init__(self, *, root: "_FakeBeamRuntime | None" = None) -> None:
        super().__init__()
        self.root = self if root is None else root
        self.children: list[_FakeBeamRuntime] = []
        self.clear_beam_calls = 0

    def fork_beam_branch(
        self,
        state: _AcceptedStateSnapshot,
        **_kwargs: Any,
    ) -> tuple["_FakeBeamRuntime", _AcceptedStateSnapshot]:
        child = _FakeBeamRuntime(root=self.root)
        self.root.children.append(child)
        return child, state

    def clear_beam_branch_context(self) -> None:
        self.clear_beam_calls += 1

    def beam_executed_s_alg(self) -> int:
        # The fault-injection test intentionally leaves this unchanged so the
        # controller fails immediately after a successful child transition.
        return 0

    def beam_executed_s_alg_components(
        self,
    ) -> tuple[tuple[str, int], ...]:
        return (
            ("N_H_outer", 0),
            ("N_H_refit", 0),
            ("N_grad", 0),
            ("N_metric", 0),
        )

    def beam_resume_seed(self) -> tuple[tuple[str, ...], int]:
        return (), 0

    def configure_beam_winner(self, **_kwargs: Any) -> None:
        return None


def _install_fake_kernels(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _FakeRuntime,
    call_order: list[tuple[str, int]],
) -> None:
    def _select(
        state: _SRControllerState,
        workspace: object,
    ) -> object:
        call_order.append(("select", state.controller_round))
        assert getattr(workspace, "round") == state.controller_round
        return SimpleNamespace(
            controller_round=state.controller_round,
            controller_state_fingerprint=state.accepted_state_fingerprint,
            selected=SimpleNamespace(
                generator_id=f"generator:{state.controller_round}",
                pool_index=state.controller_round,
            ),
        )

    def _transition(
        state: _AcceptedStateSnapshot,
        decision: object,
        workspace: object,
    ) -> object:
        call_order.append(("transition", state.controller_round))
        assert getattr(workspace, "round") == state.controller_round
        assert getattr(workspace, "decision") is decision
        next_state = runtime.next_state(state)
        event = SimpleNamespace(
            controller_round=next_state.controller_round,
            accepted_state_fingerprint=(
                next_state.accepted_state_fingerprint
            ),
        )
        return SimpleNamespace(
            preceding_state=state,
            decision=decision,
            next_state=next_state,
            ledger=SimpleNamespace(
                cumulative_s_alg_components=(
                    ("N_H_outer", next_state.controller_round),
                    ("N_H_refit", 0),
                    ("N_grad", 0),
                    ("N_metric", 0),
                ),
                cumulative_s_alg=next_state.controller_round,
            ),
            checkpoint_event=event,
        )

    monkeypatch.setattr(sr_controller, "_select_singleton", _select)
    monkeypatch.setattr(sr_controller, "_transition_singleton", _transition)


def _exact_stop(
    *,
    maximum_controller_rounds: int,
    energy: float,
    tolerance: float = 1.0e-12,
    confirmation_controller_rounds: int = 0,
) -> SRStopPolicy:
    return SRStopPolicy(
        maximum_controller_rounds=maximum_controller_rounds,
        exact_ed_target=ExactEDStop(
            energy=energy,
            absolute_tolerance=tolerance,
            source=ExactEDSourceReceipt(
                source_id="fixture:controller",
                problem_request_sha256="a" * 64,
                sector_label="fixture-sector",
                comparison_space_label="fixture-space",
                n_ph_max=1,
            ),
            confirmation_controller_rounds=confirmation_controller_rounds,
        ),
    )


def test_default_controller_runs_fifty_accepted_cycles_without_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    call_order: list[tuple[str, int]] = []
    _install_fake_kernels(monkeypatch, runtime, call_order)

    outcome = sr_controller._run_default_singleton_controller(
        runtime,
        SRStopPolicy(),
    )

    assert outcome.final_state.controller_round == 50
    assert len(outcome.transitions) == 50
    assert len(outcome.events) == 50
    assert len(outcome.projected_rounds) == 50
    assert len(outcome.accepted_prefix_all_work) == 50
    assert outcome.accepted_prefix_all_work[-1].s_alg == 50
    assert runtime.projected_rounds == list(range(1, 51))
    assert call_order == [
        (stage, round_index)
        for round_index in range(50)
        for stage in ("select", "transition")
    ]
    assert outcome.stop.primary_reason == "maximum_controller_rounds"
    assert outcome.stop.fired_reasons == ("maximum_controller_rounds",)
    assert outcome.stop.completed_controller_rounds == 50
    assert runtime.close_calls == 1


def test_default_controller_cleanly_finalizes_stationary_phase0_without_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    phase0 = _PhaseSelectionReceipt(
        phase="phase0",
        population=(SimpleNamespace(domain_record_id="g0"),),
        shortlist=(),
        shortlist_ranking=(),
        estimator_event_ids=("estimator:0:g0",),
        terminal_outcome=_PHASE0_STATIONARY_TERMINAL_OUTCOME,
    )

    def _stationary(*_args: object, **_kwargs: object) -> object:
        raise _StationaryPhase0Selection(phase0)

    monkeypatch.setattr(sr_controller, "_select_singleton", _stationary)

    outcome = sr_controller._run_default_singleton_controller(
        runtime,
        SRStopPolicy(maximum_controller_rounds=50),
    )

    assert outcome.final_state == runtime.initial_accepted_state
    assert outcome.accepted_states == ()
    assert outcome.transitions == ()
    assert outcome.events == ()
    assert outcome.projected_rounds == ()
    assert outcome.stop.primary_reason == "phase0_stationary"
    assert outcome.stop.fired_reasons == ("phase0_stationary",)
    assert outcome.stop.terminal_controller_outcome == (
        _PHASE0_STATIONARY_TERMINAL_OUTCOME
    )
    assert outcome.stop.completed_controller_rounds == 0
    assert len(runtime.stationary_finalize_calls) == 1
    assert runtime.stationary_finalize_calls[0]["phase0"] is phase0
    assert runtime.finalize_calls == []
    assert runtime.close_calls == 1


@pytest.mark.parametrize("maximum", (1, 2, 7))
def test_explicit_maximum_counts_transitions_not_accepted_operators(
    monkeypatch: pytest.MonkeyPatch,
    maximum: int,
) -> None:
    runtime = _FakeRuntime(
        operator_counts=tuple(2 * round_index for round_index in range(1, 8))
    )
    call_order: list[tuple[str, int]] = []
    _install_fake_kernels(monkeypatch, runtime, call_order)

    outcome = sr_controller._run_default_singleton_controller(
        runtime,
        SRStopPolicy(maximum_controller_rounds=maximum),
    )

    assert len(outcome.transitions) == maximum
    assert outcome.final_state.controller_round == maximum
    assert len(outcome.final_state.accepted_operator_ids) == 2 * maximum
    assert outcome.stop.completed_controller_rounds == maximum
    assert outcome.stop.accepted_operator_count == 2 * maximum


def test_exact_target_is_ignored_initially_and_evaluated_after_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(energies=(-2.0, 0.0))
    runtime.initial_accepted_state = _accepted_state(0, energy=0.0)
    call_order: list[tuple[str, int]] = []
    _install_fake_kernels(monkeypatch, runtime, call_order)

    outcome = sr_controller._run_default_singleton_controller(
        runtime,
        _exact_stop(maximum_controller_rounds=2, energy=0.0),
    )

    assert outcome.final_state.controller_round == 2
    assert outcome.stop.primary_reason == "exact_ed_target_reached"
    assert outcome.stop.fired_reasons == (
        "exact_ed_target_reached",
        "maximum_controller_rounds",
    )
    assert tuple(
        (condition.reason, condition.active, condition.fired)
        for condition in outcome.stop.conditions
    ) == (
        ("maximum_controller_rounds", True, True),
        ("exact_ed_target_reached", True, True),
    )


def test_exact_target_keeps_finite_cap_and_can_fire_before_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(energies=(-0.5, -1.0, -1.5))
    call_order: list[tuple[str, int]] = []
    _install_fake_kernels(monkeypatch, runtime, call_order)

    outcome = sr_controller._run_default_singleton_controller(
        runtime,
        _exact_stop(maximum_controller_rounds=9, energy=-1.0),
    )

    assert outcome.final_state.controller_round == 2
    assert outcome.stop.primary_reason == "exact_ed_target_reached"
    assert outcome.stop.fired_reasons == ("exact_ed_target_reached",)
    assert outcome.stop.conditions[0].fired is False
    assert outcome.stop.conditions[1].fired is True


def test_exact_target_can_require_two_confirmation_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(energies=(-0.5, -1.0, -1.0, -1.0, -1.0))
    call_order: list[tuple[str, int]] = []
    _install_fake_kernels(monkeypatch, runtime, call_order)

    outcome = sr_controller._run_default_singleton_controller(
        runtime,
        _exact_stop(
            maximum_controller_rounds=9,
            energy=-1.0,
            confirmation_controller_rounds=2,
        ),
    )

    assert outcome.final_state.controller_round == 4
    assert outcome.stop.primary_reason == "exact_ed_target_reached"
    assert outcome.stop.exact_first_hit_controller_round == 2
    assert outcome.stop.exact_confirmation_controller_rounds == 2


def test_next_state_feeds_selection_and_projection_precedes_stop_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    call_order: list[tuple[str, int]] = []
    _install_fake_kernels(monkeypatch, runtime, call_order)

    outcome = sr_controller._run_default_singleton_controller(
        runtime,
        SRStopPolicy(maximum_controller_rounds=3),
    )

    assert runtime.selection_states == [
        runtime.initial_accepted_state,
        outcome.transitions[0].next_state,
        outcome.transitions[1].next_state,
    ]
    assert runtime.transition_states == runtime.selection_states
    assert runtime.projected_rounds == [1, 2, 3]
    assert len(runtime.finalize_calls) == 1
    assert runtime.finalize_calls[0]["final_state"] is outcome.final_state
    assert runtime.finalize_calls[0]["stop"].primary_reason == (
        "maximum_controller_rounds"
    )


def test_failed_projection_does_not_publish_a_completed_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(fail_projection_round=2)
    call_order: list[tuple[str, int]] = []
    _install_fake_kernels(monkeypatch, runtime, call_order)

    with pytest.raises(RuntimeError, match="projection failed"):
        sr_controller._run_default_singleton_controller(
            runtime,
            SRStopPolicy(maximum_controller_rounds=3),
        )

    assert runtime.projected_rounds == [1]
    assert runtime.finalize_calls == []
    assert runtime.close_calls == 1


def test_beam_failure_after_child_transition_closes_every_owned_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeBeamRuntime()
    call_order: list[tuple[str, int]] = []
    _install_fake_kernels(monkeypatch, runtime, call_order)

    with pytest.raises(
        RuntimeError,
        match="performed no estimator work",
    ):
        sr_controller._run_default_fork_local_beam_controller(
            runtime,
            SRStopPolicy(maximum_controller_rounds=1),
            SingletonAdmission(),
            ForkLocalBeam(
                live_parent_branches=2,
                admission_children_per_parent=2,
                maximum_admission_children_per_round=2,
            ),
        )

    assert len(runtime.children) == 1
    assert runtime.children[0].close_calls == 1
    assert runtime.close_calls == 1


def test_unexpected_numerical_termination_fails_closed_without_stop_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    call_order: list[tuple[str, int]] = []
    _install_fake_kernels(monkeypatch, runtime, call_order)

    def _unexpected_terminal(
        _state: _AcceptedStateSnapshot,
    ) -> sr_controller._PreparedSelection:
        raise RuntimeError("unexpected numerical terminal outcome")

    runtime.prepare_selection = _unexpected_terminal  # type: ignore[method-assign]

    with pytest.raises(
        RuntimeError,
        match="unexpected numerical terminal outcome",
    ):
        sr_controller._run_default_singleton_controller(
            runtime,
            SRStopPolicy(maximum_controller_rounds=3),
        )

    assert call_order == []
    assert runtime.finalize_calls == []
    assert runtime.close_calls == 1
