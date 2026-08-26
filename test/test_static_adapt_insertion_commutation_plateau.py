from __future__ import annotations

import copy

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.adapt_pipeline import (
    _candidate_insertion_position_plans,
    _insertion_commutation_plateau_domain_receipt,
    _insertion_commutation_plateau_round_policy,
    _phase1_position_probe_plan,
    _ra_phase3_population_activation_receipt,
)
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.deferred_gram_fallback import (
    DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1,
)
from pipelines.static_adapt.ra_adapt import (
    MacroCandidateAdapter,
    RAAdaptRequest,
    SinglePauliWordCandidateAdapter,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU,
    RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
    RESOURCE_WEIGHTING_LATE,
)
from pipelines.static_adapt.ra_adapt.engine import (
    _repaired_route_contract,
)
from pipelines.scaffold.hh_continuation_stage_control import StageControllerConfig
from pipelines.static_adapt.sr_snake import (
    PlateauCommutationInsertion,
    SingletonAdmission,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRStopPolicy,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_CALIBRATION_STATUS,
    INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD,
    SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2,
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2,
)
from test_support.route_contract_kwargs import route_identity

from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


PARENT_REQUEST = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
)
PARENT_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_v1"
)
PARENT_DIGEST = (
    "350316790ca8b6f2b2ba3d3dde11c377e2ffc3b1575701eee1a9859590a7ab3c"
)
ROUTE_REQUEST = "insertion_commutation_plateau_v2"
ROUTE_DIGEST = (
    "61ae4317381bb05ff64d7219a513230e3ae328dbfb271cb8efec6763fd631143"
)


def _term(label: str, *words: str) -> AnsatzTerm:
    nq = len(words[0])
    return AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(nq, ps=word, pc=1.0) for word in words],
        ),
    )


def _run_small_registered_route(
    *,
    max_depth: int,
    route_request: str = ROUTE_REQUEST,
) -> object:
    problem = resolve_problem_context(
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
        )
    )
    adapter = (
        MacroCandidateAdapter()
        if route_request.startswith("sr_snake_macro_only")
        else SinglePauliWordCandidateAdapter()
    )
    return run_ra_adapt(
        problem,
        RAAdaptRequest(
            adapter=adapter,
            method=SRMethodPolicy(
                insertion=PlateauCommutationInsertion()
            ),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(
                    maximum_controller_rounds=int(max_depth)
                )
            ),
        ),
    )


def test_registered_profile_changes_only_the_parent_insertion_policy() -> None:
    _parent_resolved, parent_contract, _parent_digest = route_identity(
        PARENT_REQUEST
    )
    route_resolved, contract, route_digest = route_identity(ROUTE_REQUEST)

    parent_settings = dict(parent_contract["execution_settings"])
    route_settings = dict(contract["execution_settings"])
    assert parent_settings.pop("adapt_insertion_mode") == "append_only"
    assert (
        route_settings.pop("adapt_insertion_mode")
        == "insertion_commutation_plateau_v2"
    )
    assert route_settings == parent_settings

    invariants = contract["semantic_invariants"]
    lineage = contract["lineage_authority"]
    assert route_resolved == (
        SR_ROUTE_PROFILE_INSERTION_COMMUTATION_PLATEAU_V2
    )
    assert route_digest == ROUTE_DIGEST
    assert lineage["parent_route_profile"] == PARENT_PROFILE
    assert lineage["parent_contract_sha256"] == PARENT_DIGEST
    assert lineage["only_intended_parent_setting_changes"] == {
        "adapt_insertion_mode": "insertion_commutation_plateau_v2"
    }
    assert (
        INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD
        == 1.0e-4
    )
    assert (
        invariants["plateau_prior_mean_decrease_ratio_threshold"]
        == INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD
    )
    assert (
        INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_CALIBRATION_STATUS
        == "source_locked_counterfactual_trigger_replay_v2"
    )
    assert (
        invariants["plateau_threshold_calibration_status"]
        == INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_CALIBRATION_STATUS
    )


def test_macro_registered_profile_changes_only_page2_insertion_policy() -> None:
    parent_resolved, parent_contract, parent_digest = route_identity(
        "sr_snake_macro_only_physical_lanes_v1"
    )
    route_resolved, contract, _route_digest = route_identity(
        "sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v2"
    )

    parent_settings = dict(parent_contract["execution_settings"])
    route_settings = dict(contract["execution_settings"])
    assert parent_settings.pop("adapt_insertion_mode") == "append_only"
    assert (
        route_settings.pop("adapt_insertion_mode")
        == "insertion_commutation_plateau_v2"
    )
    assert route_settings == parent_settings

    assert route_resolved == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2
    )
    assert contract["lineage_authority"]["parent_route_profile"] == (
        parent_resolved
    )
    assert contract["lineage_authority"]["parent_contract_sha256"] == (
        parent_digest
    )
    assert contract["lineage_authority"]["only_intended_parent_setting_changes"] == {
        "adapt_insertion_mode": "insertion_commutation_plateau_v2"
    }
    assert contract["semantic_invariants"][
        "plateau_prior_mean_decrease_ratio_threshold"
    ] == pytest.approx(1.0e-4)


def test_macro_registered_profile_executes_one_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)

    result = _run_small_registered_route(
        max_depth=1,
        route_request=(
            "sr_snake_macro_only_physical_lanes_"
            "insertion_commutation_plateau_v2"
        ),
    )

    assert result.run.route.family == "ra_adapt"
    assert result.scientific_receipts["resolved_route_contract"][
        "lineage_authority"
    ]["parent_route_profile"] == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_INSERTION_COMMUTATION_PLATEAU_V2
    )
    assert len(result.accepted_trajectory) == 1
    receipt = result.scientific_receipts["accepted_round_receipts"][0][
        "insertion_commutation_plateau"
    ]
    assert receipt["domain_state"] == "closed"
    assert receipt["effective_insertion_mode"] == "append_only"


def test_prior_mean_relative_trigger_opens_persists_and_closes() -> None:
    first_round = _insertion_commutation_plateau_round_policy(history=[])
    assert first_round["domain_state"] == "closed"
    assert first_round["effective_insertion_mode"] == "append_only"
    assert first_round["trigger_energy_decrease"] is None

    initial_transition = {
        "energy_before_opt": 0.0,
        "energy_after_opt": -1.0,
    }
    no_baseline = _insertion_commutation_plateau_round_policy(
        history=[initial_transition]
    )
    assert no_baseline["domain_state"] == "closed"
    assert no_baseline["prior_cumulative_energy_decrease"] is None
    assert no_baseline["prior_accepted_transition_count"] == 0
    assert no_baseline["prior_mean_energy_decrease"] is None
    assert (
        no_baseline["marginal_to_prior_mean_decrease_ratio"]
        is None
    )

    weak_transition = {
        "energy_before_opt": -1.0,
        "energy_after_opt": -1.00005,
        "delta_abs_current": 123.0,
        "delta_abs_drop_from_prev": -456.0,
        "benchmark_target_abs_delta_e": 789.0,
    }
    opened = _insertion_commutation_plateau_round_policy(
        history=[initial_transition, weak_transition]
    )
    assert opened["domain_state"] == "open"
    assert opened["effective_insertion_mode"] == "full_commutation_reduced"
    assert opened["trigger_energy_decrease"] == pytest.approx(5.0e-5)
    assert opened["trigger_energy_before"] == -1.0
    assert opened["trigger_energy_after"] == -1.00005
    assert opened["prior_cumulative_energy_decrease"] == pytest.approx(1.0)
    assert opened["prior_accepted_transition_count"] == 1
    assert opened["prior_mean_energy_decrease"] == pytest.approx(1.0)
    assert opened[
        "marginal_to_prior_mean_decrease_ratio"
    ] == pytest.approx(5.0e-5)

    still_weak = {
        "energy_before_opt": -1.00005,
        "energy_after_opt": -1.00010,
    }
    remains_open = _insertion_commutation_plateau_round_policy(
        history=[initial_transition, weak_transition, still_weak]
    )
    assert remains_open["domain_state"] == "open"

    restored_progress = {
        "energy_before_opt": -1.00010,
        "energy_after_opt": -1.00030,
    }
    closed = _insertion_commutation_plateau_round_policy(
        history=[
            initial_transition,
            weak_transition,
            still_weak,
            restored_progress,
        ]
    )
    assert closed["domain_state"] == "closed"
    assert closed["effective_insertion_mode"] == "append_only"
    assert closed["trigger_energy_decrease"] == pytest.approx(2.0e-4)

    different_exact_diagnostics = copy.deepcopy(weak_transition)
    different_exact_diagnostics.update(
        {
            "delta_abs_current": -999.0,
            "delta_abs_drop_from_prev": 999.0,
            "benchmark_target_abs_delta_e": -999.0,
        }
    )
    assert _insertion_commutation_plateau_round_policy(
        history=[initial_transition, different_exact_diagnostics]
    ) == opened

    shifted_history = []
    for row in (initial_transition, weak_transition):
        shifted = copy.deepcopy(row)
        shifted["energy_before_opt"] += 100.0
        shifted["energy_after_opt"] += 100.0
        shifted_history.append(shifted)
    shifted = _insertion_commutation_plateau_round_policy(
        history=shifted_history
    )
    assert shifted["domain_open"] is opened["domain_open"]
    assert shifted["trigger_energy_decrease"] == pytest.approx(
        opened["trigger_energy_decrease"]
    )
    assert shifted["prior_cumulative_energy_decrease"] == pytest.approx(
        opened["prior_cumulative_energy_decrease"]
    )
    assert shifted["prior_accepted_transition_count"] == (
        opened["prior_accepted_transition_count"]
    )
    assert shifted["prior_mean_energy_decrease"] == pytest.approx(
        opened["prior_mean_energy_decrease"]
    )
    assert shifted[
        "marginal_to_prior_mean_decrease_ratio"
    ] == pytest.approx(
        opened["marginal_to_prior_mean_decrease_ratio"]
    )


def test_singleton_phase3_activation_is_pure_projection_of_plateau_receipt(
) -> None:
    request = RAAdaptRequest(
        adapter=SinglePauliWordCandidateAdapter(),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=PlateauCommutationInsertion(),
        ),
    )
    _requested, _resolved, route_contract, _digest = (
        _repaired_route_contract(
            request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
            algorithm_id=(
                RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID
            ),
        )
    )
    histories = [
        [],
        [
            {
                "energy_before_opt": 0.0,
                "energy_after_opt": -1.0,
            }
        ],
        [
            {
                "energy_before_opt": 0.0,
                "energy_after_opt": -1.0,
            },
            {
                "energy_before_opt": -1.0,
                "energy_after_opt": -1.00005,
            },
        ],
        [
            {
                "energy_before_opt": 0.0,
                "energy_after_opt": -1.0,
            },
            {
                "energy_before_opt": -1.0,
                "energy_after_opt": -1.00005,
            },
            {
                "energy_before_opt": -1.00005,
                "energy_after_opt": -1.00025,
            },
        ],
    ]
    observed: list[bool] = []
    for history in histories:
        plateau = _insertion_commutation_plateau_round_policy(
            history=history
        )
        activation = _ra_phase3_population_activation_receipt(
            route_contract=route_contract,
            insertion_round_policy=plateau,
            candidate_adapter=request.adapter,
            admission_policy=request.method.admission,
        )
        observed.append(bool(activation["competitive_population_live"]))
        assert activation["policy"] == (
            RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU
        )
        assert activation["competitive_population_live"] is (
            plateau["domain_open"]
        )
        assert activation["insertion_plateau_domain_open"] is (
            plateau["domain_open"]
        )
        assert activation["independent_latch_active"] is False
        assert activation["hysteresis_active"] is False
        assert plateau["patience"] == 1
        assert plateau["hysteresis_active"] is False
    assert observed == [False, False, True, False]


def test_prior_mean_ratio_rejects_late_total_accumulation_false_plateau() -> None:
    history = []
    energy = 0.0
    for _ in range(4):
        history.append(
            {
                "energy_before_opt": energy,
                "energy_after_opt": energy - 1.0,
            }
        )
        energy -= 1.0
    history.append(
        {
            "energy_before_opt": energy,
            "energy_after_opt": energy - 2.0e-4,
        }
    )

    receipt = _insertion_commutation_plateau_round_policy(history=history)
    assert receipt["schema"] == (
        "insertion_commutation_plateau_round_policy_v2"
    )
    assert receipt["policy"] == "insertion_commutation_plateau_v2"
    assert receipt["prior_accepted_transition_count"] == 4
    assert receipt["prior_cumulative_energy_decrease"] == pytest.approx(4.0)
    assert receipt["prior_mean_energy_decrease"] == pytest.approx(1.0)
    assert receipt["marginal_to_prior_mean_decrease_ratio"] == pytest.approx(
        2.0e-4
    )
    assert receipt["domain_open"] is False

    history[-1]["energy_after_opt"] = energy - 5.0e-5
    opened = _insertion_commutation_plateau_round_policy(history=history)
    assert opened["marginal_to_prior_mean_decrease_ratio"] == pytest.approx(
        5.0e-5
    )
    assert opened["domain_open"] is True


def test_open_round_receipt_covers_full_domain_and_retained_representatives() -> None:
    inactive_positions, inactive_triggered, inactive_reason = (
        _phase1_position_probe_plan(
            insertion_mode="insertion_commutation_plateau_v2",
            append_eval={"append_best_score": 0.0},
            append_position=3,
            n_params=3,
            active_window_indices=[2],
            stage_name="core",
            drop_plateau_hits=10,
            max_grad=0.0,
            eps_grad=1.0,
            finite_angle_fallback=True,
            repeated_family_flat=True,
            cfg=StageControllerConfig(max_probe_positions=4),
        )
    )
    assert inactive_positions == [3]
    assert inactive_triggered is False
    assert inactive_reason == "plateau_profile_state_unavailable_append_only"

    policy = _insertion_commutation_plateau_round_policy(
        history=[
            {
                "energy_before_opt": 0.0,
                "energy_after_opt": -1.0,
            },
            {
                "energy_before_opt": -1.0,
                "energy_after_opt": -1.00005,
            },
        ]
    )
    positions, triggered, reason = _phase1_position_probe_plan(
        insertion_mode=policy["effective_insertion_mode"],
        append_eval={},
        append_position=3,
        n_params=3,
        active_window_indices=[2],
        stage_name="core",
        drop_plateau_hits=0,
        max_grad=1.0,
        eps_grad=1.0e-8,
        finite_angle_fallback=False,
        repeated_family_flat=False,
        cfg=StageControllerConfig(max_probe_positions=1),
    )
    assert positions == [0, 1, 2, 3]
    assert triggered is True
    assert reason == "full_commutation_reduced"

    candidate = _term("candidate", "ex")
    plans = _candidate_insertion_position_plans(
        pool=[candidate],
        candidate_indices=[0],
        selected_ops=[
            _term("commuting-left", "ex"),
            _term("barrier", "ez"),
            _term("commuting-right", "xx"),
        ],
        positions=positions,
    )
    receipt = _insertion_commutation_plateau_domain_receipt(
        round_policy=policy,
        candidate_position_plans=plans,
        pool=[candidate],
    )

    assert receipt["domain_state"] == "open"
    assert receipt["requested_positions"] == [0, 1, 2, 3]
    assert receipt["retained_representatives"] == [
        {
            "candidate_pool_index": 0,
            "candidate_label": "candidate",
            "positions": [0, 2],
        }
    ]
    assert receipt["candidate_position_plans"][0][
        "members_by_representative"
    ] == {0: [0, 1], 2: [2, 3]}
    assert receipt["calibration_status"] == (
        "source_locked_counterfactual_trigger_replay_v2"
    )


def test_registered_route_runs_first_round_append_only_with_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    result = _run_small_registered_route(max_depth=1)

    assert result.run.route.family == "ra_adapt"
    assert "__insertion_commutation_plateau_v2__" in (
        result.run.route.profile
    )
    assert len(result.accepted_trajectory) == 1
    receipt = result.scientific_receipts["accepted_round_receipts"][0][
        "insertion_commutation_plateau"
    ]
    assert receipt["domain_state"] == "closed"
    assert receipt["trigger_energy_decrease"] is None
    assert receipt["prior_mean_decrease_ratio_threshold"] == 1.0e-4
    assert receipt["calibration_status"] == (
        "source_locked_counterfactual_trigger_replay_v2"
    )
    assert receipt["requested_positions"] == [0]
    assert receipt["candidate_count"] > 0
    assert receipt["retained_representative_count"] == receipt["candidate_count"]
    assert {
        tuple(row["positions"])
        for row in receipt["retained_representatives"]
    } == {(0,)}
    fallback = result.scientific_receipts[
        DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1
    ]
    assert fallback["enabled"] is True
    assert fallback["fired"] is False
    assert fallback["rounds"] == []
    assert fallback["charge"] == 0
    assert fallback["schema"] == (
        DEFERRED_GRAM_ALL_MODELS_INFEASIBLE_FALLBACK_V1
    )


def test_live_open_round_reuses_parent_phase3_and_refit_with_insertion_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    # Test-only widening guarantees that the first ratio-bearing transition
    # exercises the open-domain integration path. The registered route
    # threshold and digest remain frozen at 1e-4.
    monkeypatch.setattr(
        adapt_pipeline,
        "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD",
        100.0,
    )

    result = _run_small_registered_route(max_depth=3)

    assert len(result.accepted_trajectory) == 3
    accepted_rounds = result.scientific_receipts[
        "accepted_round_receipts"
    ]
    first_receipt = accepted_rounds[0][
        "insertion_commutation_plateau"
    ]
    second_receipt = accepted_rounds[1][
        "insertion_commutation_plateau"
    ]
    third_receipt = accepted_rounds[2][
        "insertion_commutation_plateau"
    ]
    assert first_receipt["domain_state"] == "closed"
    assert first_receipt["requested_positions"] == [0]
    assert first_receipt["candidate_count"] == 89
    assert first_receipt["requested_position_count"] == 1
    assert first_receipt["retained_representative_count"] == 89
    assert first_receipt["collapsed_position_count"] == 0
    assert second_receipt["domain_state"] == "closed"
    assert second_receipt["requested_positions"] == [1]
    assert third_receipt["domain_state"] == "open"
    assert third_receipt["requested_positions"] == [0, 1, 2]
    assert third_receipt["candidate_count"] == 89
    assert third_receipt["requested_position_count"] == 3
    assert {
        tuple(row["requested_positions"])
        for row in third_receipt["candidate_position_plans"]
    } == {(0, 1, 2)}
    plans_by_pool_index = {
        int(row["candidate_pool_index"]): row
        for row in third_receipt["candidate_position_plans"]
    }
    assert plans_by_pool_index[0]["representative_positions"] == [0]
    assert list(
        plans_by_pool_index[0]["members_by_representative"].values()
    ) == [[0, 1, 2]]
    assert plans_by_pool_index[2]["representative_positions"] == [0, 1, 2]
    assert list(
        plans_by_pool_index[2]["members_by_representative"].values()
    ) == [[0], [1], [2]]
    assert result.accepted_transitions[2].insertion_position in {0, 1, 2}
    metric_accounting = accepted_rounds[2][
        "accepted_refit_metric_query_accounting"
    ]
    assert metric_accounting["status"] == (
        "reused_external_logical_fs_gram_receipt"
    )
    assert metric_accounting["incremental_quantum_query_charge"] == 0
