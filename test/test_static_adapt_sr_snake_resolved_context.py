from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import pipelines.scaffold.hh_continuation_scoring as continuation_scoring
import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
import pipelines.static_adapt.ra_adapt.runtime as ra_runtime
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.static_adapt.current_checkpoint import _stable_json_digest
from pipelines.static_adapt.resume_scaffold import load_static_resume_source
from pipelines.static_adapt.sr_snake import (
    AcceptedTransitionReceipt,
    AppendCommutationReducedInsertion,
    AppendOnlyInsertion,
    CheckpointObservation,
    CombinatorialBatchAdmission,
    CombinatorialBatchAcceptedTransitionReceipt,
    CombinatorialBatchScientificReplayReceipt,
    EstimatorLedgerObservation,
    ExactEDSourceReceipt,
    ExactEDStop,
    FullCombinatorialSearchWindow,
    GreedyBatchAdmission,
    GreedyBatchAcceptedTransitionReceipt,
    GreedyBatchScientificReplayReceipt,
    SRExecutionPolicy,
    SRMethodPolicy as _SRMethodPolicy,
    SRObservationPolicy,
    SRRunRequest,
    SRStopPolicy,
    run_sr_snake,
)
import pipelines.static_adapt.sr_snake._context as sr_context
import pipelines.static_adapt.sr_snake._controller as sr_controller
import pipelines.static_adapt.sr_snake._selection as sr_selection
import pipelines.static_adapt.sr_snake._transition as sr_transition


ROUTE_PROFILE_REQUEST = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_"
    "no_overlap_trust_v1"
)
ROUTE_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_v1"
)
ROUTE_DIGEST = (
    "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
)
GREEDY_ROUTE_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_greedy_batch_v1"
)
COMBINATORIAL_ROUTE_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_combinatorial_batch_v1"
)


def SRMethodPolicy(*, admission: Any) -> _SRMethodPolicy:
    """Keep the historical batch characterization on append-only identity."""

    return _SRMethodPolicy(
        admission=admission,
        insertion=AppendOnlyInsertion(),
    )


def _small_hh_problem() -> Any:
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        ),
        exact_energy_impl=adapt_pipeline._exact_gs_energy_for_problem,
    )


def _request(
    problem: Any,
    *,
    observation: SRObservationPolicy | None = None,
) -> SRRunRequest:
    return SRRunRequest(
        method=_SRMethodPolicy(insertion=AppendOnlyInsertion()),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(
                maximum_controller_rounds=1,
                exact_ed_target=ExactEDStop(
                    energy=-0.75,
                    absolute_tolerance=1.0e-12,
                    source=ExactEDSourceReceipt.from_problem(
                        problem,
                        source_id="fixture:resolved-context",
                    ),
                ),
            )
        ),
        observation=observation or SRObservationPolicy(),
    )


def test_append_reduced_route_contract_binds_endpoint_exact_reducer() -> None:
    request = SRRunRequest(
        method=_SRMethodPolicy(
            insertion=AppendCommutationReducedInsertion(),
        )
    )

    profile_request, profile, contract, digest = (
        sr_context._canonical_route_contract_for_request(request)
    )

    assert profile_request == profile
    assert "insertion-append_commutation_reduced" in profile
    assert (
        contract["execution_settings"]["adapt_insertion_mode"]
        == "append_commutation_reduced"
    )
    invariants = contract["semantic_invariants"]
    assert invariants["canonical_insertion_policy"] == (
        "append_commutation_reduced"
    )
    assert invariants["insertion_position_scope"] == (
        "append_endpoint_only_every_depth_v1"
    )
    assert invariants["insertion_equivalence_policy"] == (
        "termwise_cross_component_commutation_earliest_representative_v1"
    )
    assert digest == sr_context._route_contract_sha256(contract)


def test_public_run_resolves_context_and_profile_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()
    request = _request(problem)
    context_calls = 0
    original_context_resolver = sr_context._resolve_execution_context

    def _counted_context_resolver(*args: Any, **kwargs: Any) -> Any:
        nonlocal context_calls
        context_calls += 1
        return original_context_resolver(*args, **kwargs)

    monkeypatch.setattr(
        ra_runtime,
        "_resolve_execution_context",
        _counted_context_resolver,
    )

    result = run_sr_snake(problem, request)

    assert result.route.profile == ROUTE_PROFILE
    assert isinstance(
        result.accepted_transitions[0],
        AcceptedTransitionReceipt,
    )
    assert "proposal" not in result.accepted_transitions[0].to_dict()
    assert "admission" not in result.accepted_transitions[0].to_dict()
    assert context_calls == 1


def test_public_run_resolves_canonical_runtime_without_cli_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The typed facade must not reconstruct a legacy CLI request."""

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()
    request = _request(problem)

    def _unexpected_cli_translation(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "run_sr_snake must resolve its canonical runtime without CLI translation"
        )

    monkeypatch.setattr(
        adapt_pipeline,
        "parse_args",
        _unexpected_cli_translation,
    )
    monkeypatch.setattr(
        sr_context,
        "_legacy_argv",
        _unexpected_cli_translation,
        raising=False,
    )

    result = run_sr_snake(problem, request)

    assert result.route.profile == ROUTE_PROFILE
    assert len(result.accepted_transitions) == 1


def test_canonical_runtime_closes_over_characterized_route_settings() -> None:
    """Route-controlled values enter the direct runtime without CLI defaults."""

    problem = _small_hh_problem()
    context = sr_context._resolve_execution_context(problem, _request(problem))
    runtime = context.canonical_runtime_kwargs()
    execution_settings = context.route.contract["execution_settings"]

    assert not (
        set(adapt_pipeline._CANONICAL_SR_SNAKE_RUNTIME_INFRASTRUCTURE)
        & set(execution_settings)
    )
    for key, value in execution_settings.items():
        if key != "adapt_final_full_refit":
            assert runtime[key] == value
    assert runtime["adapt_final_full_refit"] is False
    assert runtime["maxiter"] == execution_settings["adapt_maxiter"]
    assert runtime["seed"] == execution_settings["adapt_seed"]
    assert runtime["allow_repeats"] == execution_settings["adapt_allow_repeats"]
    assert runtime["disable_hh_seed"] == execution_settings["adapt_disable_hh_seed"]
    assert runtime["finite_angle"] == execution_settings["adapt_finite_angle"]
    assert runtime["phase2_enable_batching"] == execution_settings[
        "phase3_enable_batching"
    ]
    assert runtime["sr_route_profile_resolved"] == context.route.profile
    assert runtime["sr_route_profile_contract_sha256"] == (
        context.route.contract_sha256
    )


def test_public_default_run_dispatches_exact_issue10_selection_gate_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    observations: list[tuple[str, dict[str, Any]]] = []

    def _record_observation(
        event_name: str,
        **payload: Any,
    ) -> None:
        observations.append((str(event_name), dict(payload)))

    monkeypatch.setattr(adapt_pipeline, "_ai_log", _record_observation)
    problem = _small_hh_problem()
    calls: list[dict[str, Any]] = []
    selection_states: list[sr_selection._SRControllerState] = []
    callback_domains: list[
        tuple[sr_selection._CandidatePositionRecord, ...]
    ] = []
    callback_observation_deltas: list[int] = []
    decisions: list[sr_selection._SingletonAdmissionDecision] = []
    selected_predictive_inputs: list[tuple[float, str]] = []
    gradient_kernel_activity: list[bool] = []
    kernel_call_active = False
    original_gradient_surface = (
        adapt_pipeline.evaluate_exact_gradient_surface
    )

    def _gradient_surface_spy(*args: Any, **kwargs: Any) -> Any:
        gradient_kernel_activity.append(kernel_call_active)
        return original_gradient_surface(*args, **kwargs)

    def _spy(**kwargs: Any) -> bool:
        calls.append(dict(kwargs))
        return sr_selection._uses_default_singleton_selection(**kwargs)

    def _selection_spy(
        state: sr_selection._SRControllerState,
        workspace: sr_selection._SelectionWorkspace,
    ) -> sr_selection._SingletonAdmissionDecision:
        selection_states.append(state)

        class _KernelSpy:
            def accepted_state_snapshot(self) -> object:
                return workspace.kernel.accepted_state_snapshot()

            def evaluate(
                self,
                domain: tuple[
                    sr_selection._CandidatePositionRecord, ...
                ],
            ) -> sr_selection._SelectionEvaluation:
                nonlocal kernel_call_active
                callback_domains.append(domain)
                observation_count_before = len(observations)
                kernel_call_active = True
                try:
                    evaluation = workspace.kernel.evaluate(domain)
                finally:
                    kernel_call_active = False
                callback_observation_deltas.append(
                    len(observations) - observation_count_before
                )
                return evaluation

        decision = sr_selection._select_singleton(
            state,
            replace(workspace, kernel=_KernelSpy()),
        )
        decisions.append(decision)
        selected_runtime = workspace.kernel.runtime.sidecar[
            decision.selected.domain_record_id
        ]
        selected_feature = selected_runtime["selected_live_record"][
            "feature"
        ]
        selected_predictive_inputs.append(
            (
                float(selected_feature.selector_burden),
                str(selected_feature.hardware_cost_policy),
            )
        )
        return decision

    monkeypatch.setattr(
        adapt_pipeline,
        "_uses_default_singleton_selection",
        _spy,
    )
    monkeypatch.setattr(
        sr_controller,
        "_select_singleton",
        _selection_spy,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "evaluate_exact_gradient_surface",
        _gradient_surface_spy,
    )

    result = run_sr_snake(problem, _request(problem))

    assert result.route.profile == ROUTE_PROFILE
    assert calls == [
        {
            "route_profile": ROUTE_PROFILE,
            "route_profile_sha256": ROUTE_DIGEST,
            "beam_enabled": False,
        }
    ]
    assert [state.controller_round for state in selection_states] == [0]
    assert tuple(
        generator_id
        for generator_id, _count in selection_states[0].selection_counts
    ) == selection_states[0].available_generator_ids
    assert len(callback_domains) == 1
    assert callback_domains[0]
    assert tuple(
        record.pool_index for record in callback_domains[0]
    ) == tuple(
        sorted(record.pool_index for record in callback_domains[0])
    )
    assert callback_observation_deltas == [0]
    assert gradient_kernel_activity == [True]
    assert any(
        event_name == "hardcoded_adapt_gradient_timing"
        for event_name, _payload in observations
    )
    iteration_observations = [
        payload
        for event_name, payload in observations
        if event_name == "hardcoded_adapt_iter"
    ]
    assert len(iteration_observations) == 1
    assert float(iteration_observations[0]["max_grad"]) > 0.0
    assert len(decisions) == len(selected_predictive_inputs) == 1
    predictive_value, predictive_policy = selected_predictive_inputs[0]
    assert decisions[0].predictive_cost.value == pytest.approx(
        predictive_value
    )
    assert decisions[0].predictive_cost.policy_identity == predictive_policy


@pytest.mark.parametrize(
    ("tamper_mode", "expected_error"),
    (
        (
            "different_domain",
            "runtime sidecar identity set disagrees with the immutable "
            "admission decision",
        ),
        (
            "same_domain_different_generator",
            "live selection record disagrees with the immutable "
            "admission decision",
        ),
    ),
)
def test_default_decision_identity_cannot_be_reassigned_from_legacy_state(
    monkeypatch: pytest.MonkeyPatch,
    tamper_mode: str,
    expected_error: str,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()

    def _mismatched_decision(
        state: sr_selection._SRControllerState,
        workspace: sr_selection._SelectionWorkspace,
    ) -> sr_selection._SingletonAdmissionDecision:
        decision = sr_selection._select_singleton(state, workspace)
        if tamper_mode == "different_domain":
            alternate = next(
                record
                for record in decision.phase_i.population
                if record.domain_record_id
                != decision.selected.domain_record_id
            )
        else:
            alternate = next(
                record
                for record in decision.phase_i.population
                if record.domain_record_id
                == decision.selected.domain_record_id
            )
        return replace(decision, selected=alternate)

    monkeypatch.setattr(
        sr_controller,
        "_select_singleton",
        _mismatched_decision,
    )

    with pytest.raises(
        RuntimeError,
        match=expected_error,
    ):
        run_sr_snake(problem, _request(problem))


def test_resolved_context_is_immutable_and_owns_active_dependencies() -> None:
    problem = _small_hh_problem()
    request = _request(problem)

    context = sr_context._resolve_execution_context(problem, request)

    assert context.problem is problem
    assert context.request is request
    assert context.problem_receipt.family_key == "hh"
    assert context.route.profile_request == ROUTE_PROFILE_REQUEST
    assert context.route.profile == ROUTE_PROFILE
    assert context.route.contract_sha256 == ROUTE_DIGEST
    assert context.route.pool_key == "full_meta"
    assert context.numerical.hamiltonian is problem.hamiltonian
    assert context.optimizer.name == "POWELL"
    assert context.optimizer.maximum_iterations == 200
    assert context.optimizer.seed == 7
    assert context.accepted_refit.scope == "full_ansatz_v1"
    assert (
        context.initial_state.source_label
        == problem.reference_state.source_label
    )
    assert context.initial_state.state_kind == problem.reference_state.state_kind
    assert (
        context.initial_state.build_state
        is problem.reference_state.build_state
    )
    assert context.estimator_ledger.enabled is True
    assert context.estimator_ledger.destination is None
    assert context.stop is request.execution.stop
    assert context.observation is request.observation
    with pytest.raises(FrozenInstanceError):
        context.stop = SRStopPolicy()  # type: ignore[misc]
    with pytest.raises(TypeError):
        context.runtime_kwargs["max_depth"] = 99  # type: ignore[index]
    execution_settings = context.route.contract["execution_settings"]
    with pytest.raises(TypeError):
        execution_settings["adapt_pool"] = "other"  # type: ignore[index]


def test_greedy_context_resolves_request_specific_child_route() -> None:
    problem = _small_hh_problem()
    admission = GreedyBatchAdmission(
        maximum_size=4,
        search_window_size=7,
    )
    request = replace(
        _request(problem),
        method=SRMethodPolicy(admission=admission),
    )

    context = sr_context._resolve_execution_context(problem, request)

    assert context.route.family == "greedy_batch_response_snake"
    assert context.route.profile == GREEDY_ROUTE_PROFILE
    assert context.route.profile_request == GREEDY_ROUTE_PROFILE
    assert len(context.route.contract_sha256) == 64
    assert context.route.contract["route_family"] == (
        "greedy_batch_response_snake"
    )
    assert context.route.contract["semantic_invariants"][
        "greedy_batch_maximum_size"
    ] == 4
    assert context.route.contract["semantic_invariants"][
        "greedy_batch_search_window_size"
    ] == 7
    assert context.route.contract["lineage_authority"][
        "parent_contract_sha256"
    ] == ROUTE_DIGEST
    assert context.request.method.admission is admission


@pytest.mark.parametrize(
    (
        "maximum_size",
        "search_window",
        "resolved_window",
        "window_semantics",
    ),
    [
        (4, 7, 7, "ranked_phase3_prefix_cardinality_v1"),
        (
            4,
            FullCombinatorialSearchWindow(),
            None,
            "full_ranked_phase3_population_v1",
        ),
    ],
)
def test_combinatorial_context_resolves_request_specific_child_route(
    maximum_size: int,
    search_window: int | FullCombinatorialSearchWindow,
    resolved_window: int | None,
    window_semantics: str,
) -> None:
    problem = _small_hh_problem()
    admission = CombinatorialBatchAdmission(
        maximum_size=maximum_size,
        search_window_size=search_window,
    )
    request = replace(
        _request(problem),
        method=SRMethodPolicy(admission=admission),
    )

    context = sr_context._resolve_execution_context(problem, request)

    assert context.route.family == "combinatorial_batch_response_snake"
    assert context.route.profile == COMBINATORIAL_ROUTE_PROFILE
    assert context.route.profile_request == COMBINATORIAL_ROUTE_PROFILE
    assert len(context.route.contract_sha256) == 64
    invariants = context.route.contract["semantic_invariants"]
    assert invariants["admission_policy"] == (
        "cost_weighted_combinatorial_reduced_plane_v1"
    )
    assert (
        invariants["combinatorial_batch_maximum_size"]
        == maximum_size
    )
    assert (
        invariants["combinatorial_batch_search_window_size"]
        == resolved_window
    )
    assert (
        invariants["combinatorial_batch_search_window_semantics"]
        == window_semantics
    )
    assert invariants["combinatorial_batch_enumeration"] == (
        "generator_distinct_subsets_not_permutations_v1"
    )
    assert invariants["combinatorial_batch_record_semantics"] == (
        "fixed_generator_plus_insertion_position_v1"
    )
    assert invariants["combinatorial_batch_commit_order"] == (
        "phase2_ranked_proposal_order_within_phase3_prefix_v1"
    )
    assert context.route.contract["lineage_authority"][
        "parent_contract_sha256"
    ] == ROUTE_DIGEST
    assert context.request.method.admission is admission


def test_greedy_context_propagates_typed_admission_to_runtime() -> None:
    problem = _small_hh_problem()
    admission = GreedyBatchAdmission(
        maximum_size=2,
        search_window_size=None,
    )
    context = sr_context._resolve_execution_context(
        problem,
        replace(
            _request(problem),
            method=SRMethodPolicy(admission=admission),
        ),
    )

    runtime = context.build_default_controller_runtime()

    try:
        assert runtime.context.admission_policy is admission
    finally:
        runtime.close()


def test_greedy_context_resolves_direct_typed_profile() -> None:
    problem = _small_hh_problem()
    admission = GreedyBatchAdmission(
        maximum_size=2,
        search_window_size=4,
    )
    context = sr_context._resolve_execution_context(
        problem,
        replace(
            _request(problem),
            method=SRMethodPolicy(admission=admission),
        ),
    )

    assert context.route.profile == GREEDY_ROUTE_PROFILE
    assert context.route.contract["semantic_invariants"][
        "greedy_batch_maximum_size"
    ] == 2
    assert context.route.contract["semantic_invariants"][
        "greedy_batch_search_window_size"
    ] == 4


def test_greedy_runtime_builds_ranked_window_numerical_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setenv(
        "STATIC_ADAPT_JOINT_PAIR_CACHE_MAX_ENTRIES",
        "0",
    )
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    physical_pair_calls: list[tuple[str, str]] = []
    original_pair_kernel = (
        continuation_scoring._measure_joint_candidate_pair_entry
    )

    def _observe_physical_pair_call(**kwargs: Any) -> tuple[float, float, float]:
        physical_pair_calls.append(
            (
                str(kwargs["left_record"]["candidate_term"].label),
                str(kwargs["right_record"]["candidate_term"].label),
            )
        )
        return original_pair_kernel(**kwargs)

    monkeypatch.setattr(
        continuation_scoring,
        "_measure_joint_candidate_pair_entry",
        _observe_physical_pair_call,
    )
    problem = _small_hh_problem()
    admission = GreedyBatchAdmission(
        maximum_size=2,
        search_window_size=4,
    )
    context = sr_context._resolve_execution_context(
        problem,
        replace(
            _request(problem),
            method=SRMethodPolicy(admission=admission),
        ),
    )
    runtime = context.build_default_controller_runtime()

    try:
        prepared = runtime.prepare_selection(runtime.initial_accepted_state)
        assert isinstance(
            prepared.workspace.kernel,
            adapt_pipeline._DefaultGreedyBatchSelectionKernel,
        )
        decision = sr_selection._select_greedy_batch(
            prepared.controller_state,
            prepared.workspace,
            maximum_size=admission.maximum_size,
            search_window_size=admission.search_window_size,
        )
        assert 1 <= len(decision.selected) <= 2
        assert decision.proposal.search_window_size == 4
        assert decision.phase_iii.shortlist == decision.selected
        assert tuple(runtime.cursor.pending_selection.runtime_sidecar) == tuple(
            record.domain_record_id for record in decision.selected
        )
        batch_summary = runtime.cursor.pending_selection.runtime_sidecar[
            decision.selected[0].domain_record_id
        ]["greedy_batch_summary"]
        pair_accounting = batch_summary[
            "pair_geometry_estimator_accounting"
        ]
        required_pair_count = pair_accounting["all_evaluated_pair_count"]
        assert required_pair_count > 0
        assert pair_accounting["physical_pair_evaluation_count"] == (
            required_pair_count
        )
        assert len(physical_pair_calls) == required_pair_count
        assert len(
            {
                row["pair_cache_key"]
                for row in pair_accounting["pairs"]
            }
        ) == required_pair_count
        assert pair_accounting["ledger_occurrence_count"] == (
            2 * required_pair_count
        )
        assert all(
            [
                primitive["primitive_kind"]
                for primitive in row["primitive_rows"]
            ]
            == ["metric_element", "hessian_element"]
            for row in pair_accounting["pairs"]
        )
        ledger_occurrences = [
            occurrence
            for occurrence in runtime.cursor.estimator_call_ledger.to_payload()[
                "occurrences"
            ]
            if occurrence["consumer_scope"]
            == "greedy_batch_pair_geometry_all_evaluated"
        ]
        assert len(ledger_occurrences) == 2 * required_pair_count
        assert all(
            occurrence["component"] == "N_metric"
            for occurrence in ledger_occurrences
        )
    finally:
        runtime.close()


def test_combinatorial_runtime_enumerates_ranked_window_with_one_pair_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setenv(
        "STATIC_ADAPT_JOINT_PAIR_CACHE_MAX_ENTRIES",
        "0",
    )
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    physical_pair_calls: list[tuple[str, str]] = []
    original_pair_kernel = (
        continuation_scoring._measure_joint_candidate_pair_entry
    )

    def _observe_physical_pair_call(**kwargs: Any) -> tuple[float, float, float]:
        physical_pair_calls.append(
            (
                str(kwargs["left_record"]["candidate_term"].label),
                str(kwargs["right_record"]["candidate_term"].label),
            )
        )
        return original_pair_kernel(**kwargs)

    monkeypatch.setattr(
        continuation_scoring,
        "_measure_joint_candidate_pair_entry",
        _observe_physical_pair_call,
    )
    problem = _small_hh_problem()
    admission = CombinatorialBatchAdmission(
        maximum_size=2,
        search_window_size=4,
    )
    context = sr_context._resolve_execution_context(
        problem,
        replace(
            _request(problem),
            method=SRMethodPolicy(admission=admission),
        ),
    )
    runtime = context.build_default_controller_runtime()

    try:
        prepared = runtime.prepare_selection(runtime.initial_accepted_state)
        assert isinstance(
            prepared.workspace.kernel,
            adapt_pipeline._DefaultCombinatorialBatchSelectionKernel,
        )
        decision = sr_selection._select_combinatorial_batch(
            prepared.controller_state,
            prepared.workspace,
            maximum_size=admission.maximum_size,
            search_window_size=admission.resolved_search_window_size,
        )
        assert 1 <= len(decision.selected) <= 2
        assert decision.proposal.search_window_size == 4
        assert decision.proposal.ranked_window_count == 4
        assert decision.phase_iii.shortlist == decision.selected
        assert tuple(runtime.cursor.pending_selection.runtime_sidecar) == tuple(
            record.domain_record_id for record in decision.selected
        )
        batch_summary = runtime.cursor.pending_selection.runtime_sidecar[
            decision.selected[0].domain_record_id
        ]["combinatorial_batch_summary"]
        assert batch_summary["near_degenerate_shell_active"] is False
        assert decision.proposal.subset_counts_evaluated == tuple(
            (int(size), int(count))
            for size, count in sorted(
                batch_summary["subset_counts_evaluated"].items(),
                key=lambda item: int(item[0]),
            )
        )
        assert decision.proposal.evaluated_subset_count == sum(
            count for _size, count in decision.proposal.subset_counts_evaluated
        )
        pair_accounting = batch_summary[
            "pair_geometry_estimator_accounting"
        ]
        required_pair_count = pair_accounting["all_evaluated_pair_count"]
        assert required_pair_count > 0
        assert pair_accounting["physical_pair_evaluation_count"] == (
            required_pair_count
        )
        assert len(physical_pair_calls) == required_pair_count
        assert pair_accounting["ledger_occurrence_count"] == (
            2 * required_pair_count
        )
        assert decision.proposal.evaluated_subset_count > required_pair_count
        ledger_occurrences = [
            occurrence
            for occurrence in runtime.cursor.estimator_call_ledger.to_payload()[
                "occurrences"
            ]
            if occurrence["consumer_scope"]
            == "combinatorial_batch_pair_geometry_all_evaluated"
        ]
        assert len(ledger_occurrences) == 2 * required_pair_count
        assert all(
            occurrence["component"] == "N_metric"
            for occurrence in ledger_occurrences
        )
    finally:
        runtime.close()


def test_greedy_runtime_commits_one_atomic_numerical_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()
    admission = GreedyBatchAdmission(
        maximum_size=2,
        search_window_size=4,
    )
    context = sr_context._resolve_execution_context(
        problem,
        replace(
            _request(problem),
            method=SRMethodPolicy(admission=admission),
        ),
    )
    runtime = context.build_default_controller_runtime()

    try:
        initial = runtime.initial_accepted_state
        prepared = runtime.prepare_selection(initial)
        decision = sr_selection._select_greedy_batch(
            prepared.controller_state,
            prepared.workspace,
            maximum_size=admission.maximum_size,
            search_window_size=admission.search_window_size,
        )
        assert len(decision.selected) == 2
        expected_group_keys_by_member: list[list[str]] = []
        for selected in decision.selected:
            selected_live_record = runtime.cursor.pending_selection.runtime_sidecar[
                selected.domain_record_id
            ]["selected_live_record"]
            expected_group_keys_by_member.append(
                [
                    str(spec.group_key)
                    for spec in adapt_pipeline.measurement_group_specs_for_term(
                        selected_live_record["candidate_term"]
                    )
                ]
            )
        assert set(expected_group_keys_by_member[0]).isdisjoint(
            expected_group_keys_by_member[-1]
        )
        observed_commit_group_keys: list[str] = []
        original_cache_commit = runtime.cursor.phase1_measure_cache.commit

        def _observe_cache_commit(group_specs: Any) -> None:
            resolved_specs = list(group_specs)
            observed_commit_group_keys.extend(
                str(spec.group_key) for spec in resolved_specs
            )
            original_cache_commit(resolved_specs)

        monkeypatch.setattr(
            runtime.cursor.phase1_measure_cache,
            "commit",
            _observe_cache_commit,
        )
        workspace = runtime.prepare_transition(initial, decision)
        transition = sr_transition._transition_greedy_batch(
            initial,
            decision,
            workspace,
        )
        projection = runtime.project_accepted_event(
            transition.checkpoint_event,
            transition,
        )

        assert transition.next_state.controller_round == 1
        assert len(transition.admission.child_identities) == len(
            decision.selected
        )
        assert transition.operation_audit.admission_calls == 1
        assert transition.operation_audit.optimizer_dispatch_calls == 1
        assert transition.trust.update_count_after == (
            transition.trust.update_count_before + 1
        )
        assert projection.controller_round == 1
        assert runtime.cursor.history[-1]["selected_logical_size"] == len(
            decision.selected
        )
        assert runtime.cursor.history[-1]["selected_batch_positions"] == [
            record.insertion_position for record in decision.selected
        ]
        expected_commit_group_keys = [
            group_key
            for member_keys in expected_group_keys_by_member
            for group_key in member_keys
        ]
        assert observed_commit_group_keys == expected_commit_group_keys
        assert observed_commit_group_keys != expected_group_keys_by_member[-1]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("drift", "expected_error"),
    [
        ("labels", "candidate order drifted"),
        ("subset_order", "live workspace order"),
        ("state", "fingerprint drifted"),
        ("matrix", "selected-record response wrapper"),
    ],
)
def test_batch_external_gram_reuse_rejects_live_phase3_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    expected_error: str,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None
    )
    problem = _small_hh_problem()
    admission = GreedyBatchAdmission(
        maximum_size=2,
        search_window_size=4,
    )
    context = sr_context._resolve_execution_context(
        problem,
        replace(
            _request(problem),
            method=SRMethodPolicy(admission=admission),
        ),
    )
    runtime = context.build_default_controller_runtime()
    original_builder = (
        adapt_pipeline
        ._default_no_prune_build_greedy_batch_external_refit_gram_receipt
    )

    def _corrupt_live_source(**kwargs: Any) -> Any:
        proposal = copy.deepcopy(dict(kwargs["proposal_summary"]))
        batch = copy.deepcopy(dict(kwargs["batch_summary"]))
        if drift == "labels":
            proposal["selected_labels"] = list(
                reversed(proposal["selected_labels"])
            )
        elif drift == "subset_order":
            proposal["subset_workspace_indices"] = list(
                reversed(proposal["subset_workspace_indices"])
            )
        elif drift == "state":
            proposal["state_fingerprint"] = "f" * 64
            batch["geometry_workspace"]["state_fingerprint"] = "f" * 64
        elif drift == "matrix":
            proposal["G_BB_raw"][0][0] = float(
                proposal["G_BB_raw"][0][0]
            ) + 0.25
        else:  # pragma: no cover - parameterization is closed above
            raise AssertionError(drift)
        return original_builder(
            **{
                **kwargs,
                "proposal_summary": proposal,
                "batch_summary": batch,
            }
        )

    monkeypatch.setattr(
        adapt_pipeline,
        "_default_no_prune_build_greedy_batch_external_refit_gram_receipt",
        _corrupt_live_source,
    )
    try:
        initial = runtime.initial_accepted_state
        prepared = runtime.prepare_selection(initial)
        decision = sr_selection._select_greedy_batch(
            prepared.controller_state,
            prepared.workspace,
            maximum_size=admission.maximum_size,
            search_window_size=admission.search_window_size,
        )
        assert len(decision.selected) == 2
        workspace = runtime.prepare_transition(initial, decision)
        with pytest.raises(RuntimeError, match=expected_error):
            sr_transition._transition_greedy_batch(
                initial,
                decision,
                workspace,
            )
    finally:
        runtime.close()


def test_combinatorial_runtime_commits_one_fixed_position_atomic_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()
    admission = CombinatorialBatchAdmission(
        maximum_size=2,
        search_window_size=4,
    )
    context = sr_context._resolve_execution_context(
        problem,
        replace(
            _request(problem),
            method=SRMethodPolicy(admission=admission),
        ),
    )
    runtime = context.build_default_controller_runtime()

    try:
        initial = runtime.initial_accepted_state
        prepared = runtime.prepare_selection(initial)
        decision = sr_selection._select_combinatorial_batch(
            prepared.controller_state,
            prepared.workspace,
            maximum_size=admission.maximum_size,
            search_window_size=admission.resolved_search_window_size,
        )
        assert len(decision.selected) == 2
        assert tuple(
            record.insertion_position for record in decision.selected
        ) == (0, 0)
        proposal_order = tuple(
            record.domain_record_id for record in decision.selected
        )
        workspace = runtime.prepare_transition(initial, decision)
        transition = sr_transition._transition_combinatorial_batch(
            initial,
            decision,
            workspace,
        )
        projection = runtime.project_accepted_event(
            transition.checkpoint_event,
            transition,
        )

        assert tuple(
            record.domain_record_id
            for record in transition.decision.selected
        ) == proposal_order
        assert transition.admission.selected_domain_record_ids == (
            proposal_order
        )
        assert transition.admission.original_insertion_positions == (0, 0)
        assert transition.admission.effective_insertion_positions == (0, 1)
        assert transition.admission.initial_logical_values == (0.0, 0.0)
        assert transition.next_state.controller_round == 1
        assert transition.operation_audit.admission_calls == 1
        assert transition.operation_audit.optimizer_dispatch_calls == 1
        assert transition.trust.update_count_after == (
            transition.trust.update_count_before + 1
        )
        assert transition.ledger.controller_round == 0
        assert transition.checkpoint_event.controller_round == 1
        assert projection.controller_round == 1
        assert runtime.cursor.history[-1]["selected_logical_size"] == 2
        assert runtime.cursor.history[-1][
            "selected_batch_positions"
        ] == [0, 0]
        assert runtime.cursor.history[-1][
            "selected_batch_effective_positions"
        ] == [0, 1]
    finally:
        runtime.close()


def test_combinatorial_controller_counts_a_two_member_batch_as_one_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()
    admission = CombinatorialBatchAdmission(
        maximum_size=2,
        search_window_size=4,
    )
    request = replace(
        _request(problem),
        method=SRMethodPolicy(admission=admission),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=1)
        ),
    )
    context = sr_context._resolve_execution_context(problem, request)
    runtime = context.build_default_controller_runtime()

    outcome = sr_controller._run_default_combinatorial_batch_controller(
        runtime,
        context.stop,
        admission,
    )

    assert outcome.final_state.controller_round == 1
    assert len(outcome.final_state.accepted_operator_ids) == 2
    assert len(outcome.transitions) == 1
    assert len(outcome.events) == 1
    assert len(outcome.projected_rounds) == 1
    assert outcome.stop.completed_controller_rounds == 1
    assert outcome.stop.accepted_operator_count == 2
    assert outcome.transitions[0].operation_audit.admission_calls == 1
    assert outcome.transitions[0].operation_audit.optimizer_dispatch_calls == 1
    assert outcome.transitions[0].operation_audit.ledger_close_calls == 1
    assert outcome.transitions[0].operation_audit.checkpoint_event_count == 1


def test_public_greedy_batch_run_projects_atomic_receipts_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setenv(
        "STATIC_ADAPT_JOINT_PAIR_CACHE_MAX_ENTRIES",
        "0",
    )
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()
    checkpoint_path = tmp_path / "current.json"
    admission = GreedyBatchAdmission(
        maximum_size=2,
        search_window_size=4,
    )
    request = replace(
        _request(
            problem,
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint_path,
                    every_controller_rounds=1,
                    keep_history_tail=2,
                )
            ),
        ),
        method=SRMethodPolicy(admission=admission),
    )

    result = run_sr_snake(problem, request)

    assert result.route.family == "greedy_batch_response_snake"
    assert result.route.profile == GREEDY_ROUTE_PROFILE
    assert result.route.contract_sha256 != ROUTE_DIGEST
    assert result.final_state.controller_round == 1
    assert result.stop.completed_controller_rounds == 1
    assert result.stop.accepted_operator_count == 2
    assert len(result.accepted_transitions) == 1
    transition = result.accepted_transitions[0]
    assert isinstance(
        transition,
        GreedyBatchAcceptedTransitionReceipt,
    )
    assert transition.admission.selected_cardinality == 2
    assert transition.proposal.selected_cardinality == 2
    assert transition.proposal.maximum_size == 2
    assert transition.proposal.search_window_size == 4
    assert transition.refit_scope == "full_ansatz_v1"
    assert transition.refit_chart_dimension == 2
    assert transition.refit_active_logical_indices == (0, 1)
    assert transition.accepted_state == result.final_state
    assert tuple(
        member.selected_domain_record_id
        for member in transition.admission.members
    ) == transition.proposal.selected_record_ids
    assert all(
        member.initial_logical_value == 0.0
        for member in transition.admission.members
    )
    assert set(transition.to_dict()).isdisjoint(
        {
            "selected_domain_record_id",
            "generator_id",
            "selected_operator",
            "pool_index",
            "insertion_position",
            "initial_logical_value",
        }
    )
    assert isinstance(
        result.scientific_replay[0],
        GreedyBatchScientificReplayReceipt,
    )

    envelope = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    adapt = envelope["adapt_vqe"]
    assert "verified_singleton_resume_sidecar" not in adapt
    pointer = adapt["greedy_batch_checkpoint_sidecar"]
    assert pointer["resume_enabled"] is False
    assert pointer["resume_status"] == "not_authorized_until_issue_19"
    sidecar_path = checkpoint_path.with_name(pointer["path"])
    sidecar_bytes = sidecar_path.read_bytes()
    assert hashlib.sha256(sidecar_bytes).hexdigest() == pointer["sha256"]
    sidecar = json.loads(sidecar_bytes)
    assert sidecar["schema"] == (
        "static_adapt_signed_greedy_batch_checkpoint_sidecar_v1"
    )
    assert sidecar["resume_authorization"] == {
        "enabled": False,
        "reader_contract": "greedy_batch_checkpoint_projection_only_v1",
        "status": "not_authorized_until_issue_19",
    }
    assert sidecar["controller_round"] == 1
    assert len(sidecar["rounds"]) == 1
    round_projection = sidecar["rounds"][0]
    assert round_projection["selected_cardinality"] == 2
    assert len(round_projection["members"]) == 2
    pair_accounting = round_projection["admission"][
        "pair_geometry_estimator_accounting"
    ]
    assert pair_accounting["schema"] == (
        "greedy_batch_pair_estimator_accounting_v1"
    )
    assert {
        pair["schema"] for pair in pair_accounting["pairs"]
    } == {"greedy_batch_pair_estimator_accounting_v1"}
    assert [
        member["selected_operator"]
        for member in round_projection["members"]
    ] == [
        member.selected_operator
        for member in transition.admission.members
    ]
    assert round_projection["admission"]["controller_round_count"] == 1
    assert round_projection["admission"]["supported_fs_refit_count"] == 1
    assert round_projection["admission"]["ledger_close_count"] == 1
    source_projection = json.loads(json.dumps(envelope))
    del source_projection["adapt_vqe"]["greedy_batch_checkpoint_sidecar"]
    assert (
        _stable_json_digest(source_projection)
        == pointer["source_projection_sha256"]
    )
    with pytest.raises(
        ValueError,
        match=(
            "associates a registered SR contract with profile request "
            ".*greedy_batch_v1"
        ),
    ):
        load_static_resume_source(checkpoint_path)


def test_public_combinatorial_batch_projects_subset_receipts_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setenv(
        "STATIC_ADAPT_JOINT_PAIR_CACHE_MAX_ENTRIES",
        "0",
    )
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()
    checkpoint_path = tmp_path / "current.json"
    admission = CombinatorialBatchAdmission(
        maximum_size=2,
        search_window_size=4,
    )
    request = replace(
        _request(
            problem,
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint_path,
                    every_controller_rounds=1,
                    keep_history_tail=2,
                )
            ),
        ),
        method=SRMethodPolicy(admission=admission),
    )

    result = run_sr_snake(problem, request)

    assert result.route.family == "combinatorial_batch_response_snake"
    assert result.route.profile == COMBINATORIAL_ROUTE_PROFILE
    assert result.route.contract_sha256 != ROUTE_DIGEST
    assert result.final_state.controller_round == 1
    assert result.stop.completed_controller_rounds == 1
    assert result.stop.accepted_operator_count == 2
    transition = result.accepted_transitions[0]
    assert isinstance(
        transition,
        CombinatorialBatchAcceptedTransitionReceipt,
    )
    assert transition.admission.selected_cardinality == 2
    assert transition.proposal.selected_cardinality == 2
    assert transition.proposal.maximum_size == 2
    assert transition.proposal.search_window_size == 4
    assert transition.proposal.ranked_window_count == 4
    assert transition.proposal.subset_counts_considered == (
        (1, 4),
        (2, 6),
    )
    assert transition.proposal.subset_counts_evaluated == (
        (1, 4),
        (2, 6),
    )
    assert transition.proposal.evaluated_subset_count == 10
    assert transition.refit_scope == "full_ansatz_v1"
    assert transition.refit_chart_dimension == 2
    assert transition.refit_active_logical_indices == (0, 1)
    assert transition.accepted_state == result.final_state
    assert [
        member.original_insertion_position
        for member in transition.admission.members
    ] == [0, 0]
    assert [
        member.effective_insertion_position
        for member in transition.admission.members
    ] == [0, 1]
    assert isinstance(
        result.scientific_replay[0],
        CombinatorialBatchScientificReplayReceipt,
    )

    envelope = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    adapt = envelope["adapt_vqe"]
    assert "verified_singleton_resume_sidecar" not in adapt
    assert "greedy_batch_checkpoint_sidecar" not in adapt
    pointer = adapt["combinatorial_batch_checkpoint_sidecar"]
    assert pointer["resume_enabled"] is False
    assert pointer["resume_status"] == "not_authorized_until_issue_19"
    sidecar_path = checkpoint_path.with_name(pointer["path"])
    sidecar_bytes = sidecar_path.read_bytes()
    assert hashlib.sha256(sidecar_bytes).hexdigest() == pointer["sha256"]
    sidecar = json.loads(sidecar_bytes)
    assert sidecar["schema"] == (
        "static_adapt_signed_combinatorial_batch_checkpoint_sidecar_v1"
    )
    assert "greedy" not in json.dumps(sidecar, sort_keys=True).lower()
    assert sidecar["resume_authorization"] == {
        "enabled": False,
        "reader_contract": (
            "combinatorial_batch_checkpoint_projection_only_v1"
        ),
        "status": "not_authorized_until_issue_19",
    }
    assert sidecar["controller_round"] == 1
    assert len(sidecar["rounds"]) == 1
    round_projection = sidecar["rounds"][0]
    assert round_projection["selected_cardinality"] == 2
    assert round_projection["proposal"]["subset_counts_evaluated"] == {
        "1": 4,
        "2": 6,
    }
    pair_accounting = round_projection["admission"][
        "pair_geometry_estimator_accounting"
    ]
    assert pair_accounting["schema"] == (
        "combinatorial_batch_pair_estimator_accounting_v1"
    )
    assert {
        pair["schema"] for pair in pair_accounting["pairs"]
    } == {"combinatorial_batch_pair_estimator_accounting_v1"}
    assert [
        member["effective_insertion_position"]
        for member in round_projection["members"]
    ] == [0, 1]
    source_projection = json.loads(json.dumps(envelope))
    del source_projection["adapt_vqe"][
        "combinatorial_batch_checkpoint_sidecar"
    ]
    assert (
        _stable_json_digest(source_projection)
        == pointer["source_projection_sha256"]
    )
    with pytest.raises(
        ValueError,
        match=(
            "associates a registered SR contract with profile request "
            ".*combinatorial_batch_v1"
        ),
    ):
        load_static_resume_source(checkpoint_path)


def test_public_greedy_route_retains_identity_on_single_member_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setenv(
        "STATIC_ADAPT_JOINT_PAIR_CACHE_MAX_ENTRIES",
        "0",
    )
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()
    result = run_sr_snake(
        problem,
        replace(
            _request(problem),
            method=SRMethodPolicy(
                admission=GreedyBatchAdmission(
                    maximum_size=3,
                    search_window_size=1,
                )
            ),
        ),
    )

    transition = result.accepted_transitions[0]
    assert result.route.family == "greedy_batch_response_snake"
    assert result.route.profile == GREEDY_ROUTE_PROFILE
    assert result.route.admission_policy == "greedy_batch"
    assert result.route.contract_sha256 != ROUTE_DIGEST
    assert isinstance(
        transition,
        GreedyBatchAcceptedTransitionReceipt,
    )
    assert transition.admission.selected_cardinality == 1
    assert transition.proposal.selected_cardinality == 1
    assert transition.proposal.maximum_size == 3
    assert transition.proposal.search_window_size == 1
    assert result.stop.completed_controller_rounds == 1
    assert result.stop.accepted_operator_count == 1


def test_public_combinatorial_route_retains_identity_on_singleton_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setenv(
        "STATIC_ADAPT_JOINT_PAIR_CACHE_MAX_ENTRIES",
        "0",
    )
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()
    result = run_sr_snake(
        problem,
        replace(
            _request(problem),
            method=SRMethodPolicy(
                admission=CombinatorialBatchAdmission(
                    maximum_size=3,
                    search_window_size=1,
                )
            ),
        ),
    )

    transition = result.accepted_transitions[0]
    assert result.route.family == "combinatorial_batch_response_snake"
    assert result.route.profile == COMBINATORIAL_ROUTE_PROFILE
    assert result.route.admission_policy == "combinatorial_batch"
    assert result.route.contract_sha256 != ROUTE_DIGEST
    assert isinstance(
        transition,
        CombinatorialBatchAcceptedTransitionReceipt,
    )
    assert transition.admission.selected_cardinality == 1
    assert transition.proposal.selected_cardinality == 1
    assert transition.proposal.maximum_size == 3
    assert transition.proposal.search_window_size == 1
    assert transition.proposal.subset_counts_considered == ((1, 1),)
    assert result.stop.completed_controller_rounds == 1
    assert result.stop.accepted_operator_count == 1


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        (
            "adapt_segment_target_depth",
            1,
            "legacy segment controls",
        ),
        (
            "adapt_segment_max_new_admissions",
            1,
            "legacy segment controls",
        ),
        (
            "adapt_segment_wallclock_cap_s",
            1.0,
            "legacy segment controls",
        ),
        (
            "adapt_final_full_refit",
            True,
            "legacy terminal full refit",
        ),
    ),
)
def test_staged_runtime_rejects_legacy_terminal_authorities(
    key: str,
    value: Any,
    message: str,
) -> None:
    problem = _small_hh_problem()
    context = sr_context._resolve_execution_context(
        problem,
        _request(problem),
    )
    kwargs = context.legacy_executor_kwargs()
    kwargs[key] = value

    with pytest.raises(ValueError, match=message):
        context.numerical.default_runtime_factory(
            stop_policy=context.stop,
            executor_kwargs=kwargs,
        )


def test_resolved_context_rejects_problem_route_pool_mismatch() -> None:
    problem = _small_hh_problem()
    incompatible = replace(
        problem,
        admissible_pool_keys=("uccsd_ferm_lifted",),
    )

    with pytest.raises(ValueError, match="full_meta.*not admissible"):
        sr_context._resolve_execution_context(
            incompatible,
            _request(incompatible),
        )


def test_context_freeze_thaw_preserves_nested_container_types() -> None:
    source = {
        "list": [1, {"tuple": (2, 3)}],
        "tuple": (4, [5, 6]),
    }

    resolved = sr_context._thaw(sr_context._freeze(source))

    assert isinstance(resolved, dict)
    assert isinstance(resolved["list"], list)
    assert isinstance(resolved["list"][1], dict)
    assert isinstance(resolved["list"][1]["tuple"], tuple)
    assert isinstance(resolved["tuple"], tuple)
    assert isinstance(resolved["tuple"][1], list)
    assert resolved == source


def test_observation_changes_do_not_change_resolved_scientific_dependencies(
    tmp_path: Path,
) -> None:
    problem = _small_hh_problem()
    baseline = sr_context._resolve_execution_context(
        problem,
        _request(problem),
    )
    observed = sr_context._resolve_execution_context(
        problem,
        _request(
            problem,
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=tmp_path / "current.json",
                    every_controller_rounds=1,
                ),
                estimator_ledger=EstimatorLedgerObservation(
                    path=tmp_path / "ledger.json"
                ),
            ),
        ),
    )

    assert observed.problem_receipt == baseline.problem_receipt
    assert observed.route == baseline.route
    assert observed.numerical == baseline.numerical
    assert observed.optimizer == baseline.optimizer
    assert observed.accepted_refit == baseline.accepted_refit
    assert observed.initial_state == baseline.initial_state
    assert observed.stop == baseline.stop
    assert observed.estimator_ledger.enabled is True
    assert observed.estimator_ledger.destination == tmp_path / "ledger.json"
    assert observed.observation != baseline.observation
