from __future__ import annotations

import hashlib
import json
from dataclasses import fields

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.sr_snake import (
    AcceptedStateResume,
    AlwaysCommutationReducedInsertion,
    AppendOnlyInsertion,
    CheckpointObservation,
    CombinatorialBatchAdmission,
    ForkLocalBeam,
    GreedyBatchAdmission,
    MetricPruning,
    PlateauCommutationInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRRunRequest,
    SRStopPolicy,
    TrustRegionPruning,
    run_sr_snake,
)
from pipelines.static_adapt.sr_snake._context import (
    _resolve_execution_context,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    canonical_sr_snake_insertion_commutation_plateau_v2_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256,
)


def _small_hh_problem():
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


def test_canonical_facade_rejects_non_l2_problem_scope() -> None:
    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=1,
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

    with pytest.raises(
        ValueError,
        match="locked to the canonical Hubbard--Holstein L=2",
    ):
        run_sr_snake(problem)


def test_canonical_request_silently_selects_plateau_commutation_insertion() -> None:
    request = SRRunRequest()

    assert tuple(field.name for field in fields(SRRunRequest)) == (
        "method",
        "execution",
        "observation",
    )
    assert request.method.insertion == PlateauCommutationInsertion()
    assert request.to_dict()["method"]["insertion"] == {
        "kind": "plateau_commutation"
    }

    context = _resolve_execution_context(_small_hh_problem(), request)
    assert (
        context.route.profile
        == SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2
    )
    assert context.route.contract_sha256 == (
        canonical_sr_snake_insertion_commutation_plateau_v2_contract_sha256()
    )
    assert context.route.contract["execution_settings"][
        "adapt_insertion_mode"
    ] == "insertion_commutation_plateau_v2"


def test_fork_local_beam_rejects_a_nonbranching_policy() -> None:
    with pytest.raises(
        ValueError,
        match="at least two admission children",
    ):
        ForkLocalBeam(
            live_parent_branches=1,
            admission_children_per_parent=1,
            maximum_admission_children_per_round=1,
        )


def test_append_only_is_an_explicit_ablation_with_the_frozen_parent_identity() -> None:
    request = SRRunRequest(
        method=SRMethodPolicy(insertion=AppendOnlyInsertion())
    )

    assert request.to_dict()["method"]["insertion"] == {
        "kind": "append_only"
    }
    context = _resolve_execution_context(_small_hh_problem(), request)
    assert context.route.profile == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    )
    assert context.route.contract_sha256 == (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
    )
    assert context.route.contract["execution_settings"][
        "adapt_insertion_mode"
    ] == "append_only"


def test_always_policy_selects_commutation_reduced_singleton_insertion() -> None:
    request = SRRunRequest(
        method=SRMethodPolicy(
            insertion=AlwaysCommutationReducedInsertion()
        )
    )

    assert request.to_dict()["method"]["insertion"] == {
        "kind": "always_commutation_reduced"
    }
    context = _resolve_execution_context(_small_hh_problem(), request)
    assert context.route.profile == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
    )
    assert context.route.contract_sha256 == (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256()
    )
    assert context.route.contract["execution_settings"][
        "adapt_insertion_mode"
    ] == "full_commutation_reduced"
    assert context.route.contract["semantic_invariants"][
        "insertion_position_scope"
    ] == "full_logical_ansatz_commutation_classes_every_depth_v2"
    assert context.route.contract["semantic_invariants"][
        "insertion_equivalence_policy"
    ] == "termwise_cross_component_commutation_earliest_representative_v1"


@pytest.mark.parametrize(
    "pruning",
    (MetricPruning(), TrustRegionPruning()),
)
def test_peer_pruning_composes_with_canonical_insertion(pruning) -> None:
    request = SRRunRequest(
        method=SRMethodPolicy(pruning=pruning),
    )

    context = _resolve_execution_context(_small_hh_problem(), request)

    assert context.route.contract["execution_settings"][
        "adapt_insertion_mode"
    ] == "insertion_commutation_plateau_v2"
    assert context.route.contract["semantic_invariants"][
        "prune_acceptance_authority"
    ] == "measured_delete_and_complete_refit_v1"
    assert context.route.contract["semantic_invariants"][
        "canonical_pruning_policy"
    ] == pruning.kind


def test_greedy_batch_composes_with_canonical_insertion() -> None:
    request = SRRunRequest(
        method=SRMethodPolicy(
            admission=GreedyBatchAdmission(maximum_size=3, search_window_size=None),
        )
    )

    context = _resolve_execution_context(_small_hh_problem(), request)

    assert context.route.contract["execution_settings"][
        "adapt_insertion_mode"
    ] == "insertion_commutation_plateau_v2"
    assert context.route.contract["semantic_invariants"][
        "canonical_admission_policy"
    ] == "greedy_batch"


def test_always_reduced_insertion_composes_with_batch_admission() -> None:
    request = SRRunRequest(
        method=SRMethodPolicy(
            admission=GreedyBatchAdmission(maximum_size=3, search_window_size=None),
            insertion=AlwaysCommutationReducedInsertion(),
        )
    )

    context = _resolve_execution_context(_small_hh_problem(), request)

    assert context.route.contract["execution_settings"][
        "adapt_insertion_mode"
    ] == "full_commutation_reduced"
    assert context.route.contract["semantic_invariants"][
        "canonical_insertion_policy"
    ] == "always_commutation_reduced"
    assert context.route.contract["semantic_invariants"][
        "insertion_position_scope"
    ] == "full_logical_ansatz_commutation_classes_every_depth_v2"
    assert context.route.contract["semantic_invariants"][
        "insertion_equivalence_policy"
    ] == "termwise_cross_component_commutation_earliest_representative_v1"


@pytest.mark.parametrize(
    "admission",
    (
        GreedyBatchAdmission(maximum_size=1, search_window_size=None),
        CombinatorialBatchAdmission(
            maximum_size=1,
            search_window_size=2,
        ),
    ),
)
def test_batch_admission_executes_with_canonical_insertion(
    admission,
    monkeypatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            method=SRMethodPolicy(admission=admission),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )

    assert result.route.admission_policy == admission.kind
    assert result.route.insertion_policy == "plateau_commutation"
    assert len(result.accepted_transitions) == 1
    assert result.final_state.controller_round == 1


@pytest.mark.parametrize(
    ("pruning", "nomination_policy"),
    (
        (MetricPruning(), "metric_regularized_v1"),
        (
            TrustRegionPruning(),
            "full_logical_fs_trust_delete_refit_v1",
        ),
    ),
)
def test_peer_pruning_executes_as_a_measured_refit_composition(
    pruning,
    nomination_policy,
    monkeypatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            method=SRMethodPolicy(pruning=pruning),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )

    receipt = result.accepted_transitions[0].pruning
    assert receipt is not None
    assert receipt.status == "not_executed"
    assert receipt.reason == "no_mature_old_coordinate"
    assert receipt.nomination_policy == nomination_policy
    assert receipt.surrogate_used_for_acceptance is None


def test_metric_pruning_trial_is_measured_and_counted_when_mature(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            method=SRMethodPolicy(pruning=MetricPruning()),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=3)
            ),
        ),
    )

    receipt = result.accepted_transitions[-1].pruning
    assert receipt is not None
    assert receipt.nomination_policy == "metric_regularized_v1"
    assert receipt.trial_executed is True
    assert receipt.surrogate_used_for_acceptance is False
    assert receipt.trial_s_alg is not None
    assert receipt.trial_s_alg > 0
    assert result.estimator_accounting.all_work.s_alg >= receipt.trial_s_alg


def test_batch_insertion_and_pruning_share_one_direct_controller(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            method=SRMethodPolicy(
                admission=GreedyBatchAdmission(maximum_size=1, search_window_size=None),
                pruning=TrustRegionPruning(),
            ),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )

    assert result.route.admission_policy == "greedy_batch"
    assert result.route.insertion_policy == "plateau_commutation"
    assert result.route.pruning_policy == "trust_region"
    assert result.route.execution.pruning_enabled is True
    assert result.accepted_transitions[0].pruning is not None


def test_fork_local_beam_executes_children_and_counts_discarded_work(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            method=SRMethodPolicy(
                beam=ForkLocalBeam(
                    live_parent_branches=2,
                    admission_children_per_parent=2,
                    maximum_admission_children_per_round=2,
                    s_alg_weight=0.01,
                ),
            ),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=2)
            ),
        ),
    )

    assert result.route.beam_policy == "fork_local"
    assert result.route.execution.beam_enabled is True
    assert result.final_state.controller_round == 2
    assert len(result.accepted_trajectory) == 2
    assert (
        result.estimator_accounting.all_work.s_alg
        > result.estimator_accounting.winning_lineage.s_alg
    )
    assert (
        result.canonical_reporting.accepted_prefix_work[-1]
        == result.estimator_accounting.all_work
    )
    assert (
        result.canonical_reporting.horizon_scope
        == "deliberately_stopped_prefix"
    )
    assert result.paper_i_summary is not None


@pytest.mark.parametrize(
    "admission",
    (
        GreedyBatchAdmission(maximum_size=1, search_window_size=None),
        CombinatorialBatchAdmission(
            maximum_size=1,
            search_window_size=2,
        ),
    ),
)
def test_batch_pruning_and_beam_compose_in_the_direct_controller(
    admission,
    monkeypatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            method=SRMethodPolicy(
                admission=admission,
                pruning=TrustRegionPruning(),
                beam=ForkLocalBeam(
                    live_parent_branches=2,
                    admission_children_per_parent=2,
                    maximum_admission_children_per_round=2,
                ),
            ),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )

    assert result.route.admission_policy == admission.kind
    assert result.route.pruning_policy == "trust_region"
    assert result.route.beam_policy == "fork_local"
    assert result.route.insertion_policy == "plateau_commutation"
    assert result.accepted_transitions[0].pruning is not None
    assert (
        result.estimator_accounting.all_work.s_alg
        > result.estimator_accounting.winning_lineage.s_alg
    )


def test_beam_checkpoint_authenticates_the_terminal_winning_lineage(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    checkpoint_path = tmp_path / "canonical-beam-current.json"

    run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            method=SRMethodPolicy(
                beam=ForkLocalBeam(
                    live_parent_branches=2,
                    admission_children_per_parent=2,
                    maximum_admission_children_per_round=2,
                ),
            ),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint_path,
                )
            ),
        ),
    )

    envelope = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    adapt = envelope["adapt_vqe"]
    checkpoint = envelope["checkpoint"]
    diagnostics = adapt["beam_search_diagnostics"]
    winning_branch_ids = diagnostics["winning_branch_ids"]
    assert adapt["adapt_beam_enabled"] is True
    assert winning_branch_ids
    assert adapt["branch_id"] == winning_branch_ids[-1]
    assert checkpoint["beam_enabled"] is True
    assert (
        checkpoint["checkpoint_branch_policy"]
        == "canonical_terminal_winning_lineage"
    )
    assert checkpoint["branch_id"] == adapt["branch_id"]
    assert checkpoint["parent_branch_id"] == adapt["parent_branch_id"]
    assert checkpoint["ledger_scope"] == "all_executed_branches"
    assert [
        row["branch_id"]
        for row in adapt["history"]
        if row["branch_id"] is not None
    ] == winning_branch_ids
    pointer = adapt["estimator_call_ledger_checkpoint"]
    assert pointer["ledger_scope"] == "all_executed_branches"
    sidecar = json.loads(
        checkpoint_path.with_name(pointer["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["ledger_scope"] == "all_executed_branches"
    assert sidecar["checkpoint"]["branch_id"] == adapt["branch_id"]
    assert sidecar["checkpoint"]["parent_branch_id"] == (
        adapt["parent_branch_id"]
    )


def test_batch_checkpoint_records_its_actual_admission_cardinality(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    checkpoint_path = tmp_path / "greedy-batch-current.json"

    run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            method=SRMethodPolicy(
                admission=GreedyBatchAdmission(maximum_size=2, search_window_size=None)
            ),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(path=checkpoint_path)
            ),
        ),
    )

    history = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )["adapt_vqe"]["history"]
    assert history
    assert all(
        row["batch_size"] == len(row["selected_batch_labels"])
        for row in history
    )
    assert history[0]["batch_size"] == 2


def test_authenticated_resume_extends_one_contiguous_typed_run(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    checkpoint_path = tmp_path / "canonical-current.json"
    problem = _small_hh_problem()
    first = run_sr_snake(
        problem,
        SRRunRequest(
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(path=checkpoint_path)
            ),
        ),
    )
    checkpoint_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    resumed = run_sr_snake(
        problem,
        SRRunRequest(
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=2),
                resume=AcceptedStateResume(
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha256,
                ),
            ),
            observation=SRObservationPolicy(resource_rounds=(1, 2)),
        ),
    )

    assert tuple(
        state.controller_round for state in resumed.accepted_trajectory
    ) == (1, 2)
    assert resumed.accepted_trajectory[0] == first.accepted_trajectory[0]
    assert tuple(
        transition.controller_round
        for transition in resumed.accepted_transitions
    ) == (1, 2)
    assert tuple(
        replay.controller_round for replay in resumed.scientific_replay
    ) == (1, 2)
    assert (
        resumed.canonical_reporting.accepted_prefix_work[0]
        == first.canonical_reporting.accepted_prefix_work[0]
    )
    assert (
        resumed.canonical_reporting.accepted_prefix_work[1].s_alg
        > resumed.canonical_reporting.accepted_prefix_work[0].s_alg
    )
    assert resumed.paper_i_summary is not None
    assert tuple(
        row.controller_round
        for row in resumed.paper_i_summary.accepted_error_trace
    ) == (1, 2)


def test_authenticated_resume_preserves_always_reduced_insertion_policy(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    checkpoint_path = tmp_path / "always-reduced-insertion-current.json"
    problem = _small_hh_problem()
    method = SRMethodPolicy(
        insertion=AlwaysCommutationReducedInsertion()
    )
    first = run_sr_snake(
        problem,
        SRRunRequest(
            method=method,
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(path=checkpoint_path)
            ),
        ),
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert (
        checkpoint["adapt_vqe"]["history"][0][
            "insertion_commutation_reduced"
        ]["policy"]
        == "always_commutation_reduced"
    )
    checkpoint_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    resumed = run_sr_snake(
        problem,
        SRRunRequest(
            method=method,
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=2),
                resume=AcceptedStateResume(
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha256,
                ),
            ),
        ),
    )

    assert tuple(
        state.controller_round for state in resumed.accepted_trajectory
    ) == (1, 2)
    assert resumed.accepted_trajectory[0] == first.accepted_trajectory[0]
    assert resumed.route.insertion_policy == "always_commutation_reduced"
    assert resumed.paper_i_summary is not None


def test_authenticated_beam_resume_extends_winner_lineage_and_diagnostics(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    source_checkpoint = tmp_path / "beam-source.json"
    resumed_checkpoint = tmp_path / "beam-resumed.json"
    problem = _small_hh_problem()
    beam = ForkLocalBeam(
        live_parent_branches=2,
        admission_children_per_parent=2,
        maximum_admission_children_per_round=2,
    )
    first = run_sr_snake(
        problem,
        SRRunRequest(
            method=SRMethodPolicy(beam=beam),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=source_checkpoint
                )
            ),
        ),
    )
    source_sha256 = hashlib.sha256(
        source_checkpoint.read_bytes()
    ).hexdigest()

    resumed = run_sr_snake(
        problem,
        SRRunRequest(
            method=SRMethodPolicy(beam=beam),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=2),
                resume=AcceptedStateResume(
                    checkpoint_path=source_checkpoint,
                    checkpoint_sha256=source_sha256,
                ),
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=resumed_checkpoint
                )
            ),
        ),
    )

    assert tuple(
        state.controller_round for state in resumed.accepted_trajectory
    ) == (1, 2)
    assert resumed.accepted_trajectory[0] == first.accepted_trajectory[0]
    envelope = json.loads(
        resumed_checkpoint.read_text(encoding="utf-8")
    )
    adapt = envelope["adapt_vqe"]
    diagnostics = adapt["beam_search_diagnostics"]
    winning_branch_ids = diagnostics["winning_branch_ids"]
    assert len(winning_branch_ids) == 2
    assert len(diagnostics["rounds"]) == 2
    assert diagnostics["resume_segment"] == {
        "schema": "paper_i_canonical_beam_resume_segment_v1",
        "authenticated_prefix_branch_ids": winning_branch_ids[:1],
        "new_branch_ids": winning_branch_ids[1:],
        "source_round_count": 1,
        "new_round_count": 1,
    }
    assert adapt["history"][1]["parent_branch_id"] == (
        winning_branch_ids[0]
    )
    assert (
        resumed.estimator_accounting.all_work.s_alg
        > resumed.estimator_accounting.winning_lineage.s_alg
    )
    assert (
        resumed.canonical_reporting.accepted_prefix_work[-1]
        == resumed.estimator_accounting.all_work
    )
    assert (
        envelope["checkpoint"]["checkpoint_branch_policy"]
        == "canonical_terminal_winning_lineage"
    )


@pytest.mark.parametrize(
    "admission",
    (
        GreedyBatchAdmission(maximum_size=1, search_window_size=None),
        CombinatorialBatchAdmission(
            maximum_size=1,
            search_window_size=2,
        ),
    ),
)
def test_authenticated_resume_composes_batch_pruning_and_beam(
    admission,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    problem = _small_hh_problem()
    source_checkpoint = tmp_path / f"{admission.kind}-source.json"
    beam = ForkLocalBeam(
        live_parent_branches=2,
        admission_children_per_parent=2,
        maximum_admission_children_per_round=2,
    )
    method = SRMethodPolicy(
        admission=admission,
        pruning=TrustRegionPruning(),
        beam=beam,
    )
    first = run_sr_snake(
        problem,
        SRRunRequest(
            method=method,
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=source_checkpoint
                )
            ),
        ),
    )
    source_sha256 = hashlib.sha256(
        source_checkpoint.read_bytes()
    ).hexdigest()

    resumed = run_sr_snake(
        problem,
        SRRunRequest(
            method=method,
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=2),
                resume=AcceptedStateResume(
                    checkpoint_path=source_checkpoint,
                    checkpoint_sha256=source_sha256,
                ),
            ),
        ),
    )

    assert tuple(
        state.controller_round for state in resumed.accepted_trajectory
    ) == (1, 2)
    assert resumed.accepted_trajectory[0] == first.accepted_trajectory[0]
    assert resumed.route.admission_policy == admission.kind
    assert resumed.route.pruning_policy == "trust_region"
    assert resumed.route.beam_policy == "fork_local"
    assert first.accepted_transitions[0].pruning is not None
    assert resumed.accepted_transitions[-1].pruning is not None
    assert (
        resumed.estimator_accounting.all_work.s_alg
        > resumed.estimator_accounting.winning_lineage.s_alg
    )


def test_canonical_default_executes_the_plateau_insertion_route(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
            observation=SRObservationPolicy(resource_rounds=(1,)),
        ),
    )

    assert "__insertion_commutation_plateau_v2__" in result.route.profile
    assert result.route.insertion_policy == "plateau_commutation"
    assert result.accepted_transitions[0].insertion_position == 0
    assert result.final_state.insertion_positions == (0,)
    assert result.paper_i_summary is not None
    assert result.paper_i_summary.schema == "paper_i_run_summary_v1"
    assert len(result.paper_i_summary.accepted_error_trace) == 1
    assert result.paper_i_summary.effective_plateau.resources is not None
    assert result.paper_i_summary.append_matched.status == "unavailable"
    assert len(result.paper_i_summary.requested_rounds) == 1
    assert result.paper_i_summary.requested_rounds[0].resources is not None


def test_open_plateau_records_admission_positions_in_the_accepted_state(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD",
        100.0,
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=3)
            )
        ),
    )

    assert tuple(
        transition.insertion_position
        for transition in result.accepted_transitions
    ) == (0, 1, 0)
    assert result.accepted_trajectory[-1].insertion_positions == (0, 0, 1)
    assert result.final_state.insertion_positions == (0, 0, 1)
