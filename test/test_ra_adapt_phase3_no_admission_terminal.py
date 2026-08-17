from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from pipelines.scaffold.hh_continuation_scoring import FullScoreConfig
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt import phase_shortlists
from pipelines.static_adapt.ra_adapt import adaptive_phase_shortlist
from pipelines.static_adapt.ra_adapt import runtime as ra_runtime
from pipelines.static_adapt.ra_adapt import semantic_closure_routes as semantic_routes
from pipelines.static_adapt.ra_adapt.adaptive_phase_shortlist import (
    AdaptivePhaseCandidateScore,
    adaptive_phase_record_id,
)
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
)
from pipelines.static_adapt.sr_snake import _controller as sr_controller
from pipelines.static_adapt.sr_snake._selection import (
    _CandidatePositionRecord,
    _Phase3NoPositiveSelectionEvaluation,
    _SelectionWorkspace,
    _ShortlistRankReceipt,
    _SRControllerState,
)
from pipelines.static_adapt.sr_snake._transition import _AcceptedStateSnapshot
from pipelines.static_adapt.sr_snake.contracts import SRStopPolicy


_TERMINAL = "phase_iii_no_positive_feasible_candidate_v1"


class _DuckNaturalTerminalAuthority:
    def validate(self) -> dict[str, Any]:
        return {"looks": "authenticated"}


def test_selection_runtime_and_terminal_evaluation_reject_duck_authority() -> None:
    duck = _DuckNaturalTerminalAuthority()

    with pytest.raises(TypeError, match="Phase3NaturalTerminalAuthority"):
        adapt_pipeline._DefaultSelectionRuntime(
            expected_domain=(),
            accepted_state_snapshotter=lambda: None,
            sidecar={},
            phase3_natural_terminal_authority=duck,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="authenticated V2"):
        _Phase3NoPositiveSelectionEvaluation(
            phase_i=None,
            phase_ii=None,
            phase_iii=None,
            estimator_events=(),
            natural_terminal_authority=duck,
        )


def _finalization_payload(
    *,
    terminal_outcome: str | None,
    terminal_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    continuation: dict[str, Any] = {}
    payload: dict[str, Any] = {
        "success": True,
        "route_family": "test",
        "route_profile": "test",
        "sr_route_profile_contract": {},
        "sr_route_profile_contract_sha256": "test",
        "history": [{"round": 1}],
        "estimator_call_accounting": {},
        "continuation": continuation,
    }
    if terminal_outcome is not None:
        payload["terminal_controller_outcome"] = terminal_outcome
    if terminal_receipt is not None:
        receipt = dict(terminal_receipt)
        payload["terminal_phase3_selection_receipt"] = receipt
        continuation["terminal_phase3_selection_receipt"] = dict(receipt)
    return payload


def test_typed_finalization_requires_exclusive_bound_phase3_terminal_receipt(
) -> None:
    receipt = {"schema": "terminal-fixture", "sha256": "fixture"}
    valid = _finalization_payload(
        terminal_outcome=_TERMINAL,
        terminal_receipt=receipt,
    )
    with pytest.raises(ValueError, match="authenticated V2"):
        sr_controller._DefaultControllerFinalization.from_mapping(valid)

    missing = _finalization_payload(
        terminal_outcome=_TERMINAL,
        terminal_receipt=None,
    )
    with pytest.raises(ValueError, match="Phase-III terminal receipt"):
        sr_controller._DefaultControllerFinalization.from_mapping(missing)

    smuggled = _finalization_payload(
        terminal_outcome=None,
        terminal_receipt=receipt,
    )
    with pytest.raises(ValueError, match="Phase-III terminal receipt"):
        sr_controller._DefaultControllerFinalization.from_mapping(smuggled)

    mismatched = _finalization_payload(
        terminal_outcome=_TERMINAL,
        terminal_receipt=receipt,
    )
    mismatched["continuation"]["terminal_phase3_selection_receipt"] = {
        "schema": "different",
        "sha256": "fixture",
    }
    with pytest.raises(ValueError, match="Phase-III terminal receipt"):
        sr_controller._DefaultControllerFinalization.from_mapping(mismatched)


def _resumed_state() -> _AcceptedStateSnapshot:
    depth = 46
    return _AcceptedStateSnapshot(
        controller_round=depth,
        accepted_operator_ids=tuple(f"accepted:{index}" for index in range(depth)),
        accepted_insertion_positions=tuple(range(depth)),
        logical_parameter_ids=tuple(f"logical:{index}" for index in range(depth)),
        logical_parameter_values=(0.0,) * depth,
        runtime_parameter_ids=tuple(f"runtime:{index}" for index in range(depth)),
        runtime_parameter_values=(0.0,) * depth,
        accepted_energy=-0.625,
        accepted_state_fingerprint="accepted:round:46",
        available_generator_ids=("candidate:g",),
        selection_counts=(("candidate:g", 0),),
        trust_state_identity="trust:round:46",
        optimizer_memory_identity="optimizer:round:46",
        estimator_prefix_identity="ledger:round:46",
    )


def _selection_state(
    state: _AcceptedStateSnapshot,
    record: _CandidatePositionRecord,
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
        admissible_domain_record_ids=(record.domain_record_id,),
    )


def _adaptive_receipt(
    *,
    record_id: str,
    phase: str,
    score_key: str,
    score: float,
    hard_cap: int,
) -> object:
    return adaptive_phase_shortlist.select_adaptive_phase_shortlist(
        (
            AdaptivePhaseCandidateScore(
                record_id=record_id,
                pool_index=0,
                insertion_position=46,
                active_score=score,
                tie_break_score=1.0,
            ),
        ),
        phase=phase,
        score_key=score_key,
        hard_cap=hard_cap,
        threshold=0.0,
        frontier_ratio=0.9,
    ).receipt


class _ResumedRuntime:
    def __init__(
        self,
        *,
        state: _AcceptedStateSnapshot,
        controller_state: _SRControllerState,
        workspace: _SelectionWorkspace,
        route_contract: Mapping[str, Any],
        route_contract_sha256: str,
    ) -> None:
        self.initial_accepted_state = state
        self._controller_state = controller_state
        self._workspace = workspace
        self._route_contract = dict(route_contract)
        self._route_contract_sha256 = str(route_contract_sha256)
        self.no_admission_finalize_calls: list[dict[str, Any]] = []
        self.close_calls = 0

    def prepare_selection(
        self,
        state: _AcceptedStateSnapshot,
    ) -> sr_controller._PreparedSelection:
        assert state is self.initial_accepted_state
        return sr_controller._PreparedSelection(
            controller_state=self._controller_state,
            workspace=self._workspace,
        )

    def prepare_transition(self, *_args: object, **_kwargs: object) -> object:
        pytest.fail("a no-admission round must not prepare a transition")

    def project_accepted_event(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> object:
        pytest.fail("a no-admission round must not publish an accepted event")

    def finalize(self, **_kwargs: object) -> object:
        pytest.fail("a no-admission round requires its typed finalizer")

    def finalize_stationary_phase0(self, **_kwargs: object) -> object:
        pytest.fail("the Phase-III terminal is not stationary Phase 0")

    def finalize_no_admission(
        self,
        **kwargs: Any,
    ) -> sr_controller._DefaultControllerFinalization:
        self.no_admission_finalize_calls.append(dict(kwargs))
        terminal_selection = next(
            (
                value
                for value in kwargs.values()
                if all(
                    hasattr(value, field)
                    for field in (
                        "phase0",
                        "phase_i",
                        "phase_ii",
                        "phase_iii",
                        "estimator_events",
                    )
                )
            ),
            None,
        )
        assert terminal_selection is not None
        stop = kwargs["stop"]
        terminal_receipt = {
            "phase_i": terminal_selection.phase_i.phase,
            "phase_ii": terminal_selection.phase_ii.phase,
            "phase_iii": terminal_selection.phase_iii.phase,
            "phase_iii_adaptive_sha256": (
                terminal_selection.phase_iii.adaptive_shortlist.sha256
            ),
        }
        return sr_controller._DefaultControllerFinalization.from_mapping(
            {
                "success": True,
                "route_family": "ra_adapt",
                "route_profile": str(
                    self._route_contract["route_profile"]
                ),
                "sr_route_profile_contract": dict(self._route_contract),
                "sr_route_profile_contract_sha256": (
                    self._route_contract_sha256
                ),
                "history": [{"round": 46}],
                "estimator_call_accounting": {},
                "continuation": {
                    "terminal_phase3_selection_receipt": dict(
                        terminal_receipt
                    )
                },
                "terminal_controller_outcome": (
                    stop.terminal_controller_outcome
                ),
                "terminal_active_prefix_checkpoint": {"fixture": True},
                "terminal_phase3_selection_receipt": terminal_receipt,
            }
        )

    def close(self) -> None:
        self.close_calls += 1


def test_zero_score_adaptive_phase3_is_typed_no_admission_with_full_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed failed attempt closes without inventing accepted depth 47."""

    state = _resumed_state()
    domain_record = _CandidatePositionRecord(
        domain_record_id="candidate:g@position:46",
        generator_id="candidate:g",
        parent_generator_id=None,
        pool_index=0,
        pool_label="candidate:g",
        insertion_position=46,
        symmetry_identity="symmetry:g",
        lineage_identity=("candidate:g",),
    )
    record_id = adaptive_phase_record_id(
        generator_id=domain_record.generator_id,
        pool_index=domain_record.pool_index,
        insertion_position=domain_record.insertion_position,
    )
    live_record = {
        "candidate_pool_index": 0,
        "position_id": 46,
        "generator_id": "candidate:g",
        "candidate_label": "candidate:g",
        "phase1_score": 3.0,
        "phase2_raw_score": 2.0,
        "full_v2_score": 0.0,
        "simple_score": 1.0,
    }
    phase_i_adaptive = _adaptive_receipt(
        record_id=record_id,
        phase="phase_i",
        score_key="phase1_score",
        score=3.0,
        hard_cap=24,
    )
    phase_ii_adaptive = _adaptive_receipt(
        record_id=record_id,
        phase="phase_ii",
        score_key="phase2_raw_score",
        score=2.0,
        hard_cap=12,
    )

    captured_phase_iii_receipt: dict[str, object] = {}
    real_selector = phase_shortlists.select_adaptive_phase_shortlist

    def _capture_selector(*args: Any, **kwargs: Any) -> object:
        decision = real_selector(*args, **kwargs)
        if kwargs.get("phase") == "phase_iii":
            captured_phase_iii_receipt["receipt"] = decision.receipt
        return decision

    monkeypatch.setattr(
        phase_shortlists,
        "select_adaptive_phase_shortlist",
        _capture_selector,
    )

    ledger: list[dict[str, Any]] = []

    def _charge(primitive_id: str) -> None:
        ledger.append(
            {
                "sequence": len(ledger),
                "primitive_id": primitive_id,
                "charged": True,
            }
        )

    def _phase_i() -> dict[str, Any]:
        _charge("phase_i")
        return {
            "controller_snapshot": {"round": 47},
            "phase1_records_for_phase2": [dict(live_record)],
            "phase1_records": [dict(live_record)],
            "phase1_shortlisted_records": [dict(live_record)],
            "adaptive_shortlist_receipt": phase_i_adaptive,
        }

    def _phase_ii(**_kwargs: Any) -> dict[str, Any]:
        _charge("phase_ii")
        return {
            "controller_snapshot": {"round": 47},
            "phase2_shortlisted_records": [dict(live_record)],
            "full_records": [dict(live_record)],
            "phase2_full_records_evaluated": [dict(live_record)],
            "archival_phase3_factory_by_parent_key": {},
            "archival_phase2_parent_expansions": {},
            "adaptive_shortlist_receipt": phase_ii_adaptive,
        }

    def _projected_phase_iii(**_kwargs: Any) -> tuple[object, object, object]:
        records = [dict(live_record)]
        return records, {"policy": "test"}, list(records)

    protocol = materialize_paper_i_ra_semantic_protocol(
        build_paper_i_ra_hh_regime_problem("weak_weak"),
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=1,
        ),
    )
    natural_terminal_authority = (
        phase_shortlists.Phase3NaturalTerminalAuthority.from_route_contract(
            protocol.route_contract,
            expected_route_contract_sha256=(
                protocol.route_contract["sha256"]
            ),
        )
    )
    terminal_auth_calls: list[dict[str, Any]] = []

    def _accept_fixture_terminal(
        raw_receipt: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        terminal_auth_calls.append(
            {"receipt": dict(raw_receipt), **dict(kwargs)}
        )
        return dict(raw_receipt)

    monkeypatch.setattr(
        semantic_routes,
        "validate_semantic_phase3_no_positive_terminal_receipt",
        _accept_fixture_terminal,
    )
    shortlist_runtime = phase_shortlists.PhaseShortlistRuntime(
        phase2_score_cfg=FullScoreConfig(),
        feature_updater=lambda feature, _updates: feature,
        lane_policy_active=False,
        lane_summary={},
        phase1_lane_quota_pressure=1.0,
        phase2_lane_quota_pressure=1.0,
        phase2_lane_rel_threshold=0.0,
        shortlist_lane_route="off",
        shortlist_lane_key="off",
        shortlist_lanes=("off",),
        shortlist_fallback_lane="off",
        shortlist_lane_health_key_prefix="off",
        phase3_natural_terminal_authority=natural_terminal_authority,
    )

    def _supported_response(
        *,
        phase3_measurement_input_records: Sequence[Mapping[str, Any]],
        **_kwargs: Any,
    ) -> tuple[object, object, object, object, object]:
        retained, receipt = (
            phase_shortlists._adaptive_phase_shortlist_with_receipt(
                phase3_measurement_input_records,
                runtime=shortlist_runtime,
                phase="phase_iii",
                score_key="full_v2_score",
                threshold=0.0,
                hard_cap=12,
                frontier_ratio=0.9,
                tie_break_score_key="simple_score",
                shortlist_flag="phase3_shortlisted",
            )
        )
        source = [dict(record) for record in phase3_measurement_input_records]
        return source, retained, retained, False, receipt

    def _record_from_live(_record: Mapping[str, Any]) -> _CandidatePositionRecord:
        return domain_record

    def _shortlist_ranks(
        records: Sequence[Mapping[str, Any]],
        *,
        primary_score_key: str,
        tie_break_score_key: str,
        **_kwargs: Any,
    ) -> tuple[_ShortlistRankReceipt, ...]:
        return tuple(
            _ShortlistRankReceipt(
                record_key=(
                    domain_record.domain_record_id,
                    domain_record.generator_id,
                ),
                shortlist_rank=rank,
                primary_score=float(record[primary_score_key]),
                tie_break_score=float(record[tie_break_score_key]),
                pool_index=domain_record.pool_index,
                insertion_position=domain_record.insertion_position,
            )
            for rank, record in enumerate(records, start=1)
        )

    kernel = adapt_pipeline._DefaultSingletonSelectionKernel(
        phases=adapt_pipeline._DefaultSelectionPhaseRunners(
            gradient_surface=lambda: _charge("phase0_gradient"),
            phase_i=_phase_i,
            phase_ii=_phase_ii,
            projected_phase_iii=_projected_phase_iii,
            supported_response=_supported_response,
            record_phase3_work=lambda *_args: _charge("phase_iii"),
        ),
        receipts=adapt_pipeline._DefaultSelectionReceiptAdapters(
            record_from_live=_record_from_live,
            shortlist_ranks=_shortlist_ranks,
            ledger_occurrences=lambda: ledger,
            restore_singleton_coordinates=lambda records, _historical: list(records),
            phase1_score_key="phase1_score",
            phase3_score_key="full_v2_score",
            phase3_tie_break_score_key="simple_score",
            coordinate_solve_policy="test",
        ),
        runtime=adapt_pipeline._DefaultSelectionRuntime(
            expected_domain=(domain_record,),
            accepted_state_snapshotter=lambda: state,
            sidecar={},
            phase3_natural_terminal_authority=(
                natural_terminal_authority
            ),
        ),
    )
    runtime = _ResumedRuntime(
        state=state,
        controller_state=_selection_state(state, domain_record),
        workspace=_SelectionWorkspace(
            admissible_records=(domain_record,),
            kernel=kernel,
        ),
        route_contract=protocol.route_contract,
        route_contract_sha256=protocol.route_contract["sha256"],
    )

    outcome = sr_controller._run_default_singleton_controller(
        runtime,
        SRStopPolicy(maximum_controller_rounds=50),
    )

    assert outcome.initial_state is state
    assert outcome.final_state is state
    assert outcome.final_state.controller_round == 46
    assert len(outcome.final_state.accepted_operator_ids) == 46
    assert outcome.accepted_states == ()
    assert outcome.transitions == ()
    assert outcome.events == ()
    assert outcome.projected_rounds == ()
    assert outcome.accepted_prefix_all_work == ()
    assert outcome.stop.completed_controller_rounds == 46
    assert outcome.stop.terminal_controller_outcome == _TERMINAL
    assert runtime.close_calls == 1
    assert len(terminal_auth_calls) == 1
    assert terminal_auth_calls[0]["accepted_round_count"] == 1
    assert len(runtime.no_admission_finalize_calls) == 1

    call = runtime.no_admission_finalize_calls[0]
    terminal_selection = next(
        value
        for value in call.values()
        if all(
            hasattr(value, field)
            for field in (
                "phase0",
                "phase_i",
                "phase_ii",
                "phase_iii",
                "estimator_events",
            )
        )
    )
    assert terminal_selection.phase0 is None
    assert terminal_selection.phase_i.shortlist == (domain_record,)
    assert terminal_selection.phase_ii.shortlist == (domain_record,)
    assert terminal_selection.phase_iii.population == (domain_record,)
    assert terminal_selection.phase_iii.shortlist == ()
    assert terminal_selection.phase_iii.terminal_outcome == _TERMINAL
    assert terminal_selection.phase_iii.adaptive_shortlist is (
        captured_phase_iii_receipt["receipt"]
    )
    assert terminal_selection.phase_iii.adaptive_shortlist.status == (
        "no_positive_population"
    )
    assert terminal_selection.phase_iii.adaptive_shortlist.retained_count == 0
    assert tuple(
        score.active_score
        for score in terminal_selection.phase_iii.adaptive_live_scores
    ) == (0.0,)

    assert tuple(
        event.occurrence_id for event in terminal_selection.estimator_events
    ) == (
        "estimator:0:phase0_gradient",
        "estimator:1:phase_i",
        "estimator:2:phase_ii",
        "estimator:3:phase_iii",
    )
    assert terminal_selection.phase_i.estimator_event_ids == (
        "estimator:0:phase0_gradient",
        "estimator:1:phase_i",
    )
    assert terminal_selection.phase_ii.estimator_event_ids == (
        "estimator:2:phase_ii",
    )
    assert terminal_selection.phase_iii.estimator_event_ids == (
        "estimator:3:phase_iii",
    )


def test_paper_i_summary_rounds_truncate_only_for_authenticated_phase3_terminal() -> None:
    requested = tuple(range(1, 51))

    assert ra_runtime._paper_i_requested_controller_rounds(
        requested,
        accepted_round_count=46,
        terminal_controller_outcome=_TERMINAL,
    ) == tuple(range(1, 47))
    assert ra_runtime._paper_i_requested_controller_rounds(
        requested,
        accepted_round_count=46,
        terminal_controller_outcome=None,
    ) == requested
    with pytest.raises(ValueError, match="accepted prefix"):
        ra_runtime._paper_i_requested_controller_rounds(
            requested,
            accepted_round_count=0,
            terminal_controller_outcome=_TERMINAL,
        )
