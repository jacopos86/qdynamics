from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.reporting import paper_i_append_registry as append_registry
from pipelines.reporting import paper_i_run_summary as summary
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedStateReceipt,
    AppendOnlyInsertion,
    CANONICAL_CANDIDATE_REPRESENTATION,
    CanonicalReportingReceipt,
    CheckpointReceipt,
    EstimatorAccountingReceipt,
    EstimatorComponentsReceipt,
    EstimatorWorkReceipt,
    ObservationReceipt,
    ParameterBlockReceipt,
    ReferenceStateReceipt,
    ResolvedExecutionReceipt,
    ResolvedProblemReceipt,
    RouteReceipt,
    RuntimePauliTermReceipt,
    SRMethodPolicy,
    SRRunRequest,
    SRRunResult,
    StopReceipt,
)
from pipelines.static_adapt.sr_snake._context import (
    _canonical_route_contract_for_request,
)
from pipelines.static_adapt.estimator_call_ledger import (
    projective_state_fingerprint,
)


@dataclass(frozen=True)
class _ReferenceState:
    amplitudes_real: tuple[float, ...]
    amplitudes_imaginary: tuple[float, ...]
    qubit_count: int
    source_label: str
    state_fingerprint: str


@dataclass(frozen=True)
class _CanonicalReporting:
    exact_same_cutoff_energy: float
    reference_state: _ReferenceState
    horizon_scope: str
    candidate_representation: str
    accepted_prefix_work: tuple[EstimatorWorkReceipt, ...]


@dataclass(frozen=True)
class _Replay:
    controller_round: int
    accepted_state: AcceptedStateReceipt
    checkpoint: CheckpointReceipt


@dataclass(frozen=True)
class _CanonicalRun:
    accepted_trajectory: tuple[AcceptedStateReceipt, ...]
    problem: ResolvedProblemReceipt
    route: RouteReceipt
    stop: StopReceipt
    scientific_replay: tuple[_Replay, ...]
    estimator_accounting: EstimatorAccountingReceipt
    canonical_reporting: _CanonicalReporting


def _components(total: int) -> EstimatorComponentsReceipt:
    return EstimatorComponentsReceipt(
        n_h_outer=total // 10,
        n_h_refit=total // 5,
        n_grad=total // 2,
        n_metric=total - (total // 10 + total // 5 + total // 2),
    )


def _work(total: int) -> EstimatorWorkReceipt:
    return EstimatorWorkReceipt(components=_components(total), s_alg=total)


def _problem() -> ResolvedProblemReceipt:
    return ResolvedProblemReceipt(
        family_key="hh",
        problem_request_sha256="a" * 64,
        problem_key="hh_l2_test",
        num_sites=2,
        t=1.0,
        u=2.0,
        dv=0.0,
        v_nn=0.0,
        t_prime=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=False,
        n_fermions=2,
        sector_label="N=2,Sz=0",
        comparison_space_label="same_cutoff_sector",
        reference_label="canonical_hh_reference",
        exact_target_label="same_cutoff_exact_ed",
        total_qubits=2,
    )


def _route() -> RouteReceipt:
    method = SRMethodPolicy()
    profile_request, profile, contract, digest = (
        _canonical_route_contract_for_request(
            SRRunRequest(method=method)
        )
    )
    return RouteReceipt(
        family=str(contract["route_family"]),
        profile_request=profile_request,
        profile=profile,
        contract_sha256=digest,
        method=method,
        admission_policy="singleton",
        insertion_policy="plateau_commutation",
        pruning_policy="off",
        beam_policy="off",
        execution=ResolvedExecutionReceipt(
            pool="full_meta",
            optimizer="POWELL",
            optimizer_maxiter=200,
            seed=7,
            phase0_enabled=False,
            phase2_batching_enabled=False,
            phase3_batching_enabled=False,
            pruning_enabled=False,
            beam_enabled=False,
            phase_live_hysteresis_enabled=False,
            phase3_response_coordinate_scope="full_active_plus_singleton",
            trust_policy="source_metric_no_endpoint_overlap",
            accepted_refit_policy="supported_metric_whitened_eigh_v1",
            accepted_refit_scope="complete_active_ansatz",
            accepted_refit_coordinate_chart="logical",
        ),
    )


def _state(round_index: int, depth: int, energy: float) -> AcceptedStateReceipt:
    labels = tuple(f"op_{index}" for index in range(depth))
    return AcceptedStateReceipt(
        controller_round=round_index,
        operators=labels,
        insertion_positions=tuple(range(depth)),
        generator_ids=labels,
        logical_parameters=tuple(0.1 * (index + 1) for index in range(depth)),
        runtime_parameters=tuple(0.1 * (index + 1) for index in range(depth)),
        energy=energy,
        projective_state_fingerprint=f"state-{round_index}",
    )


def _checkpoint(
    state: AcceptedStateReceipt,
    work: EstimatorWorkReceipt,
    *,
    qubit_count: int = 2,
) -> CheckpointReceipt:
    route = _route()
    blocks = tuple(
        ParameterBlockReceipt(
            candidate_label=label,
            logical_index=index,
            runtime_start=index,
            runtime_count=1,
            execution_mode="termwise_product",
            runtime_terms=(
                RuntimePauliTermReceipt(
                    pauli_exyz=("x" + "e" * (qubit_count - 1)),
                    coefficient_real=1.0,
                    coefficient_imaginary=0.0,
                    qubit_count=qubit_count,
                ),
            ),
        )
        for index, label in enumerate(state.operators)
    )
    return CheckpointReceipt(
        outer_iteration=state.controller_round,
        active_ansatz_depth=len(state.operators),
        ordered_operator_labels=state.operators,
        checkpoint_sha256=f"{state.controller_round:064x}",
        projective_state_fingerprint=state.projective_state_fingerprint,
        strict_replay_passed=True,
        strict_replay_fidelity=1.0,
        parameterization_mode="per_pauli_term_v1",
        parameterization_term_order="sorted",
        parameter_blocks=blocks,
        logical_parameters=state.logical_parameters,
        runtime_parameters=state.runtime_parameters,
        route_profile=route.profile,
        route_contract_sha256=route.contract_sha256,
        estimator_ledger_status="complete",
        estimator_ledger_s_alg=work.s_alg,
    )


def _run(
    *,
    energies: tuple[float, ...] = (-0.5, -0.89, -0.9),
    depths: tuple[int, ...] = (1, 3, 4),
    horizon_scope: str = "natural_terminal",
    prefix_totals: tuple[int, ...] = (10, 20, 30),
    terminal_total: int = 40,
    problem: ResolvedProblemReceipt | None = None,
    reference_state: ReferenceStateReceipt | None = None,
    exact_same_cutoff_energy: float = -1.0,
) -> SRRunResult:
    resolved_problem = _problem() if problem is None else problem
    resolved_reference = (
        ReferenceStateReceipt(
            amplitudes_real=(1.0, 0.0, 0.0, 0.0),
            amplitudes_imaginary=(0.0, 0.0, 0.0, 0.0),
            qubit_count=2,
            source_label="canonical_hh_reference",
            state_fingerprint=projective_state_fingerprint(
                (1.0, 0.0, 0.0, 0.0)
            ),
        )
        if reference_state is None
        else reference_state
    )
    states = tuple(
        _state(index, depth, energy)
        for index, (depth, energy) in enumerate(
            zip(depths, energies, strict=True),
            start=1,
        )
    )
    prefix_work = tuple(_work(total) for total in prefix_totals)
    replay = tuple(
        _Replay(
            controller_round=state.controller_round,
            accepted_state=state,
            checkpoint=_checkpoint(
                state,
                work,
                qubit_count=resolved_problem.total_qubits,
            ),
        )
        for state, work in zip(states, prefix_work, strict=True)
    )
    terminal = _work(terminal_total)
    return SRRunResult(
        final_state=states[-1],
        accepted_trajectory=states,
        accepted_transitions=(),
        problem=resolved_problem,
        route=_route(),
        stop=StopReceipt(
            conditions=(),
            completed_controller_rounds=len(states),
            accepted_operator_count=depths[-1],
            primary_reason="maximum_controller_rounds",
            fired_reasons=("maximum_controller_rounds",),
            accepted_energy=energies[-1],
        ),
        scientific_replay=replay,
        estimator_accounting=EstimatorAccountingReceipt(
            complete=True,
            status="complete",
            exact_blockers=(),
            all_work=terminal,
            winning_lineage=terminal,
            raw_occurrences=terminal.components,
            raw_occurrence_total=terminal.s_alg,
            prefix_closure_passed=True,
            prefix_closure_status="closed",
        ),
        observation=ObservationReceipt(),
        canonical_reporting=CanonicalReportingReceipt(
            exact_same_cutoff_energy=exact_same_cutoff_energy,
            reference_state=resolved_reference,
            horizon_scope=horizon_scope,
            candidate_representation=CANONICAL_CANDIDATE_REPRESENTATION,
            accepted_prefix_work=prefix_work,
        ),
    )


class _FakeCompiler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(
        self,
        prefix: summary.PaperIPrefixCompileInput,
    ) -> summary.PaperIQiskitResources:
        self.calls.append((prefix.source_method, prefix.controller_round))
        return summary.PaperIQiskitResources(
            compile_convention=summary.LOCKED_QISKIT_COMPILE_CONVENTION,
            compiled_two_qubit_count=10 + prefix.controller_round,
            compiled_two_qubit_depth=5 + prefix.controller_round,
            compiled_total_depth=20 + prefix.controller_round,
        )


class _AppendResolver:
    def __init__(self, source: summary.PaperIAppendRunSource) -> None:
        self.source = source
        self.requests: list[summary.PaperIAppendResolutionRequest] = []

    def resolve_canonical_append(
        self,
        request: summary.PaperIAppendResolutionRequest,
    ) -> summary.PaperIAppendRunSource:
        self.requests.append(request)
        return self.source


def test_summary_owned_selectors_define_plateau_and_common_accuracy() -> None:
    snake = tuple(
        summary.PaperIErrorTracePoint(round_index, error)
        for round_index, error in enumerate(
            (0.50, 0.11, 0.10),
            start=1,
        )
    )
    append = tuple(
        summary.PaperIErrorTracePoint(round_index, error)
        for round_index, error in enumerate(
            (0.40, 0.12, 0.08, 0.075),
            start=1,
        )
    )

    plateau = summary.select_paper_i_effective_plateau(snake)
    assert plateau.controller_round == 2
    assert plateau.selected_trace_index == 1
    assert plateau.best_observed_error == pytest.approx(0.10)
    assert plateau.selection_threshold == pytest.approx(0.11)
    assert plateau.horizon_controller_rounds == 3

    common = summary.select_paper_i_common_accuracy(snake, append)
    assert common.shared_window_end_controller_round == 2
    assert common.common_target_absolute_error == pytest.approx(0.12)
    assert common.sr_snake_window_minimum_error == pytest.approx(0.11)
    assert common.append_adapt_window_minimum_error == pytest.approx(0.12)
    assert common.sr_snake_crossing_controller_round == 2
    assert common.append_adapt_crossing_controller_round == 2


def test_summary_reports_trace_plateau_requested_rounds_and_caches_prefix_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = _FakeCompiler()
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", compiler)
    run = _run(horizon_scope="deliberately_stopped_prefix")

    observed = summary.summarize_paper_i_run(
        run,
        requested_controller_rounds=(3, 2, 3),
    )

    assert tuple(row.controller_round for row in observed.accepted_error_trace) == (
        1,
        2,
        3,
    )
    assert tuple(row.active_ansatz_depth for row in observed.accepted_error_trace) == (
        1,
        3,
        4,
    )
    assert tuple(
        row.absolute_energy_error for row in observed.accepted_error_trace
    ) == pytest.approx((0.5, 0.11, 0.1))
    assert observed.horizon_scope == "deliberately_stopped_prefix"
    assert observed.effective_plateau.policy == "paper_i_effective_plateau_v1"
    assert observed.effective_plateau.controller_round == 2
    assert observed.effective_plateau.best_observed_error == pytest.approx(0.1)
    assert observed.effective_plateau.selection_threshold == pytest.approx(0.11)
    assert observed.effective_plateau.status == "available"
    assert observed.effective_plateau.algorithmic_work.s_alg == 20
    assert tuple(row.controller_round for row in observed.requested_rounds) == (3, 2)
    assert observed.canonical_all_work.s_alg == 40
    assert observed.append_matched.status == "unavailable"
    assert observed.append_matched.reason == "canonical_append_reference_not_found"
    assert compiler.calls == [("sr_snake", 2), ("sr_snake", 3)]
    assert observed.to_dict()["schema"] == "paper_i_run_summary_v1"


def test_compiler_failure_is_retryable_and_does_not_change_scientific_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run()
    scientific_result_before_observation = run.to_dict()

    def broken(_prefix: summary.PaperIPrefixCompileInput) -> Any:
        raise RuntimeError("transpiler temporarily unavailable")

    monkeypatch.setattr(summary, "_PREFIX_COMPILER", broken)
    failed = summary.summarize_paper_i_run(run)
    assert failed.effective_plateau.status == "retryable_tooling_error"
    assert failed.effective_plateau.resources is None
    assert failed.effective_plateau.failure is not None
    assert failed.effective_plateau.failure.retryable is True
    assert "transpiler temporarily unavailable" in failed.effective_plateau.failure.message
    assert run.to_dict() == scientific_result_before_observation

    compiler = _FakeCompiler()
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", compiler)
    repaired = summary.summarize_paper_i_run(run)
    assert repaired.effective_plateau.status == "available"
    assert repaired.effective_plateau.resources is not None
    assert run.to_dict() == scientific_result_before_observation


def test_later_requested_round_reuses_matching_attached_resource_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = _FakeCompiler()
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", compiler)
    run = _run()

    first = summary.summarize_paper_i_run(run)
    assert compiler.calls == [("sr_snake", 2)]

    compiler.calls.clear()
    second = summary.summarize_paper_i_run(
        replace(run, paper_i_summary=first),
        requested_controller_rounds=(2, 3),
    )

    assert tuple(
        row.controller_round for row in second.requested_rounds
    ) == (2, 3)
    assert compiler.calls == [("sr_snake", 3)]
    assert (
        second.requested_rounds[0].resources
        == first.effective_plateau.resources
    )


def test_typed_append_source_produces_first_common_accuracy_crossings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = _FakeCompiler()
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", compiler)
    snake = _run()
    append_base = _run(
        energies=(-0.6, -0.88, -0.92),
        depths=(1, 2, 3),
        prefix_totals=(8, 16, 24),
        terminal_total=24,
    )
    append_prefixes = tuple(
        summary._reconstruct_sr_prefix(
            append_base,
            index,
        ).with_source_method("append_adapt")
        for index in range(len(append_base.accepted_trajectory))
    )
    append_source = summary.PaperIAppendRunSource(
        comparison_contract=summary.PaperIComparisonContract(
            problem_request_sha256=snake.problem.problem_request_sha256,
            optimizer="POWELL",
            optimizer_maxiter=200,
            seed=7,
            candidate_representation=CANONICAL_CANDIDATE_REPRESENTATION,
            compile_convention=summary.LOCKED_QISKIT_COMPILE_CONVENTION,
        ),
        accepted_error_trace=tuple(
            summary.PaperIAcceptedError(
                controller_round=state.controller_round,
                active_ansatz_depth=len(state.operators),
                accepted_energy=state.energy,
                exact_same_cutoff_energy=-1.0,
                absolute_energy_error=abs(state.energy + 1.0),
                projective_state_fingerprint=(
                    append_prefixes[index].projective_state_fingerprint
                ),
                checkpoint_sha256=append_prefixes[index].checkpoint_sha256,
            )
            for index, state in enumerate(append_base.accepted_trajectory)
        ),
        accepted_prefixes=append_prefixes,
        horizon_scope="natural_terminal",
    )
    resolver = _AppendResolver(append_source)

    observed = summary.summarize_paper_i_run(
        snake,
        append_reference=resolver,
    )

    common = observed.append_matched
    assert common.status == "available"
    assert common.shared_window_end_controller_round == 2
    assert common.common_target_absolute_error == pytest.approx(0.12)
    assert common.sr_snake is not None
    assert common.append_adapt is not None
    assert common.sr_snake.controller_round == 2
    assert common.append_adapt.controller_round == 2
    assert len(resolver.requests) == 1
    assert resolver.requests[0].comparison_contract == append_source.comparison_contract
    assert compiler.calls == [("sr_snake", 2), ("append_adapt", 2)]


def test_live_locked_problem_resolves_automatic_append_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locked_problem_sha256 = (
        "5197b317fe67b5eedabd726e29b897260"
        "c18bda9eaf6bc9cc05cf3b0a468b65d"
    )
    registry_resolver = append_registry.LockedPaperIAppendRegistry()
    record = registry_resolver._load_records()[locked_problem_sha256]
    reporting_resources = append_registry._reporting_resources(record)
    assert reporting_resources == {
        "policy": "fixed_controller_round_50_v1",
        "controller_round": 50,
        "compiled_two_qubit_count": 250,
        "compiled_two_qubit_depth": 210,
        "compiled_total_depth": 1112,
        "pauli_one_qubit_work": 462,
        "s_alg": 129405,
        "absolute_energy_error": 8.010331058461162e-7,
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "qiskit_validated": True,
    }
    problem_context = resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=8.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.353553390593,
            n_ph_max=3,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=True,
        )
    )
    problem_receipt = ResolvedProblemReceipt.from_problem(problem_context)
    assert problem_receipt.problem_request_sha256 == locked_problem_sha256
    raw_reference = tuple(
        complex(value) for value in problem_context.reference_state.build_state()
    )
    norm = sum(abs(value) ** 2 for value in raw_reference) ** 0.5
    normalized_reference = tuple(value / norm for value in raw_reference)
    reference_receipt = ReferenceStateReceipt(
        amplitudes_real=tuple(
            float(value.real) for value in normalized_reference
        ),
        amplitudes_imaginary=tuple(
            float(value.imag) for value in normalized_reference
        ),
        qubit_count=problem_receipt.total_qubits,
        source_label=problem_context.reference_state.source_label,
        state_fingerprint=projective_state_fingerprint(
            normalized_reference
        ),
    )
    assert (
        reference_receipt.state_fingerprint
        == append_registry._reference_state(record).state_fingerprint
    )
    exact_energy = float(record["exact_same_cutoff_energy"])
    run = _run(
        energies=(
            exact_energy + 0.50,
            exact_energy + 0.11,
            exact_energy + 0.10,
        ),
        problem=problem_receipt,
        reference_state=reference_receipt,
        exact_same_cutoff_energy=exact_energy,
    )
    compiler = _FakeCompiler()
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", compiler)

    observed = summary.summarize_paper_i_run(run)

    assert observed.append_matched.status == "available"
    assert observed.append_matched.sr_snake is not None
    assert observed.append_matched.append_adapt is not None
    assert observed.append_matched.sr_snake.resources is not None
    assert observed.append_matched.append_adapt.resources is not None
    assert record["accepted_prefixes"][-1]["controller_round"] == 50
    assert (
        record["accepted_prefixes"][-1]["algorithmic_work"]["s_alg"]
        == reporting_resources["s_alg"]
    )


def test_summary_rejects_legacy_mapping_invalid_round_and_open_work_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", _FakeCompiler())
    with pytest.raises(TypeError, match="typed canonical"):
        summary.summarize_paper_i_run({"adapt_vqe": {"history": []}})
    with pytest.raises(ValueError, match="requested controller round 4"):
        summary.summarize_paper_i_run(
            _run(),
            requested_controller_rounds=(4,),
        )
    open_run = _run()
    open_work = replace(
        open_run.canonical_reporting.accepted_prefix_work[1],
        s_alg=21,
    )
    with pytest.raises(ValueError, match="close to its component sum"):
        replace(
            open_run.canonical_reporting,
            accepted_prefix_work=(
                open_run.canonical_reporting.accepted_prefix_work[0],
                open_work,
                open_run.canonical_reporting.accepted_prefix_work[2],
            ),
        )


def test_summary_uses_a_typed_canonical_append_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", _FakeCompiler())

    assert isinstance(
        summary.CANONICAL_APPEND_REFERENCE,
        summary.CanonicalAppendReference,
    )
    observed = summary.summarize_paper_i_run(
        _run(),
        append_reference=summary.CANONICAL_APPEND_REFERENCE,
    )
    assert observed.append_matched.status == "unavailable"

    with pytest.raises(TypeError, match="typed canonical append marker"):
        summary.summarize_paper_i_run(
            _run(),
            append_reference="canonical_for_resolved_problem",  # type: ignore[arg-type]
        )


def test_canonical_algorithmic_work_closes_the_four_component_receipt() -> None:
    work = summary.canonical_paper_i_algorithmic_work(
        n_h_outer=1,
        n_h_refit=2,
        n_grad=3,
        n_metric=4,
    )

    assert work.components == summary.PaperIWorkComponents(
        n_h_outer=1,
        n_h_refit=2,
        n_grad=3,
        n_metric=4,
    )
    assert work.s_alg == 10

    with pytest.raises(ValueError, match="n_metric must be a nonnegative integer"):
        summary.canonical_paper_i_algorithmic_work(
            n_h_outer=1,
            n_h_refit=2,
            n_grad=3,
            n_metric=-1,
        )


def test_summary_rejects_typed_lookalikes_and_noncanonical_route_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", _FakeCompiler())
    canonical = _run()
    lookalike = _CanonicalRun(
        accepted_trajectory=canonical.accepted_trajectory,
        problem=canonical.problem,
        route=canonical.route,
        stop=canonical.stop,
        scientific_replay=canonical.scientific_replay,
        estimator_accounting=canonical.estimator_accounting,
        canonical_reporting=canonical.canonical_reporting,
    )
    with pytest.raises(TypeError, match="typed canonical SRRunResult"):
        summary.summarize_paper_i_run(lookalike)

    historical_route = replace(
        canonical.route,
        family="formal_manifold_snake",
        profile_request="historical_formal_manifold_v1",
        profile="historical_formal_manifold_v1",
        contract_sha256="c" * 64,
    )
    with pytest.raises(
        ValueError,
        match="typed canonical route authority",
    ):
        summary.summarize_paper_i_run(
            replace(canonical, route=historical_route)
        )

    append_method = SRMethodPolicy(insertion=AppendOnlyInsertion())
    (
        append_profile_request,
        append_profile,
        append_contract,
        append_digest,
    ) = _canonical_route_contract_for_request(
        SRRunRequest(method=append_method)
    )
    historical_append_route = replace(
        canonical.route,
        family=str(append_contract["route_family"]),
        profile_request=append_profile_request,
        profile=append_profile,
        contract_sha256=append_digest,
        method=append_method,
        insertion_policy="append_only",
    )
    with pytest.raises(ValueError, match="historical append-only replay"):
        summary.summarize_paper_i_run(
            replace(canonical, route=historical_append_route)
        )


def test_summary_rejects_runtime_partition_and_component_scope_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", _FakeCompiler())
    partition_run = _run(
        energies=(-0.9,),
        depths=(2,),
        prefix_totals=(10,),
        terminal_total=10,
    )
    replay = partition_run.scientific_replay[0]
    checkpoint = replay.checkpoint
    overlapping = replace(
        checkpoint.parameter_blocks[1],
        runtime_start=0,
    )
    bad_checkpoint = replace(
        checkpoint,
        parameter_blocks=(checkpoint.parameter_blocks[0], overlapping),
    )
    with pytest.raises(ValueError, match="contiguous"):
        summary.summarize_paper_i_run(
            replace(
                partition_run,
                scientific_replay=(replace(replay, checkpoint=bad_checkpoint),),
            )
        )

    component_run = _run()
    bounded_components = EstimatorComponentsReceipt(
        n_h_outer=4,
        n_h_refit=8,
        n_grad=14,
        n_metric=14,
    )
    bounded_work = EstimatorWorkReceipt(
        components=bounded_components,
        s_alg=40,
    )
    bounded_accounting = replace(
        component_run.estimator_accounting,
        all_work=bounded_work,
        winning_lineage=bounded_work,
        raw_occurrences=bounded_components,
        raw_occurrence_total=40,
    )
    with pytest.raises(ValueError, match="components exceed"):
        summary.summarize_paper_i_run(
            replace(component_run, estimator_accounting=bounded_accounting)
        )


def test_append_source_state_binding_fails_closed_without_artifact_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", _FakeCompiler())
    snake = _run(
        energies=(-0.9,),
        depths=(1,),
        prefix_totals=(10,),
        terminal_total=10,
    )
    prefix = summary._reconstruct_sr_prefix(
        snake,
        0,
    ).with_source_method("append_adapt")
    source = summary.PaperIAppendRunSource(
        comparison_contract=summary.PaperIComparisonContract(
            problem_request_sha256=snake.problem.problem_request_sha256,
            optimizer="POWELL",
            optimizer_maxiter=200,
            seed=7,
            candidate_representation=CANONICAL_CANDIDATE_REPRESENTATION,
        ),
        accepted_error_trace=(
            summary.PaperIAcceptedError(
                controller_round=1,
                active_ansatz_depth=1,
                accepted_energy=-0.9,
                exact_same_cutoff_energy=-1.0,
                absolute_energy_error=0.1,
                projective_state_fingerprint="wrong-state",
                checkpoint_sha256=prefix.checkpoint_sha256,
            ),
        ),
        accepted_prefixes=(prefix,),
        horizon_scope="natural_terminal",
    )

    observed = summary.summarize_paper_i_run(
        snake,
        append_reference=source,
    )

    assert observed.append_matched.status == "incompatible"
    assert observed.append_matched.reason == (
        "append error row is not bound to its typed prefix"
    )


def test_reference_state_fingerprints_authenticate_amplitudes() -> None:
    with pytest.raises(ValueError, match="does not authenticate"):
        ReferenceStateReceipt(
            amplitudes_real=(0.0, 1.0, 0.0, 0.0),
            amplitudes_imaginary=(0.0, 0.0, 0.0, 0.0),
            qubit_count=2,
            source_label="forged-reference",
            state_fingerprint=projective_state_fingerprint(
                (1.0, 0.0, 0.0, 0.0)
            ),
        )

    run = _run()
    prefix = summary._reconstruct_sr_prefix(run, 0)
    with pytest.raises(ValueError, match="does not authenticate"):
        replace(
            prefix.reference_state,
            amplitudes_real=(0.0, 1.0, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="does not authenticate"):
        summary.PaperIReferenceState(
            amplitudes_real=(0.0, 1.0, 0.0, 0.0),
            amplitudes_imaginary=(0.0, 0.0, 0.0, 0.0),
            qubit_count=2,
            source_label="forged-report-reference",
            state_fingerprint=projective_state_fingerprint(
                (1.0, 0.0, 0.0, 0.0)
            ),
        )

    with pytest.raises(ValueError, match="must be normalized"):
        summary.PaperIReferenceState(
            amplitudes_real=(2.0, 0.0, 0.0, 0.0),
            amplitudes_imaginary=(0.0, 0.0, 0.0, 0.0),
            qubit_count=2,
            source_label="scaled-report-reference",
            state_fingerprint=projective_state_fingerprint(
                (1.0, 0.0, 0.0, 0.0)
            ),
        )


def test_summary_aligns_with_synthetic_typed_sr_run_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summary, "_PREFIX_COMPILER", _FakeCompiler())
    fixture = _run(
        energies=(-0.9,),
        depths=(1,),
        prefix_totals=(10,),
        terminal_total=10,
    )
    reference = fixture.canonical_reporting.reference_state
    result = SRRunResult(
        final_state=fixture.accepted_trajectory[-1],
        accepted_trajectory=fixture.accepted_trajectory,
        accepted_transitions=(),
        problem=fixture.problem,
        route=fixture.route,
        stop=fixture.stop,
        scientific_replay=fixture.scientific_replay,
        estimator_accounting=fixture.estimator_accounting,
        observation=ObservationReceipt(),
        canonical_reporting=CanonicalReportingReceipt(
            exact_same_cutoff_energy=(
                fixture.canonical_reporting.exact_same_cutoff_energy
            ),
            reference_state=ReferenceStateReceipt(
                amplitudes_real=reference.amplitudes_real,
                amplitudes_imaginary=reference.amplitudes_imaginary,
                qubit_count=reference.qubit_count,
                source_label=reference.source_label,
                state_fingerprint=reference.state_fingerprint,
            ),
            horizon_scope=fixture.canonical_reporting.horizon_scope,
            candidate_representation=(
                fixture.canonical_reporting.candidate_representation
            ),
            accepted_prefix_work=(
                fixture.canonical_reporting.accepted_prefix_work
            ),
        ),
    )

    observed = summary.summarize_paper_i_run(result)

    assert observed.available_controller_rounds == 1
    assert observed.accepted_error_trace[0].controller_round == 1
    assert observed.accepted_error_trace[0].active_ansatz_depth == len(
        result.accepted_trajectory[0].operators
    )
    assert observed.effective_plateau.controller_round == 1
    assert observed.effective_plateau.status == "available"
    assert observed.canonical_all_work.s_alg == (
        result.estimator_accounting.all_work.s_alg
    )
    assert observed.provenance.problem_request_sha256 == (
        result.problem.problem_request_sha256
    )


def test_summary_module_import_keeps_optional_scientific_stacks_lazy() -> None:
    code = """
import sys
import pipelines.reporting.paper_i_run_summary
blocked = (
    "numpy",
    "qiskit",
    "pipelines.exact_bench.table_i_qiskit_resource_compile",
)
loaded = [name for name in blocked if name in sys.modules]
raise SystemExit("loaded optional modules: " + repr(loaded) if loaded else 0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=5.0,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
