"""Real route objects and compiled costs for search-policy/gain-mode comparisons."""

from dataclasses import replace
from itertools import product

import pytest

from pipelines.time_dynamics.ap_mclachlan.append_cost import AppendCostSettings
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    APGeneralizedExchangeConfig,
    EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
    EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_integration import build_population_ranker
from pipelines.time_dynamics.ap_mclachlan.exchange_integration import select_generalized_exchange_patch
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import _PruneControllerRuntimeState
from pipelines.time_dynamics.ap_mclachlan.fixed_step import solve_fixed_mclachlan_step
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import evaluate_mclachlan_geometry
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.insertion_words import INSERTED
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import StructuralScoreWeights
from test.test_ap_mclachlan_exchange_integration import (
    _config,
    _population_candidate,
    _run,
    _poly,
    _state,
    POLICY,
)


POLICIES = ("greedy", "block_exchange", "bounded_exhaustive")
GAINS = ("singleton_factorized", "joint_batch")
SEARCH_END = {
    "greedy": "greedy_extensions_exhausted",
    "block_exchange": "block_neighborhoods_exhausted",
    "bounded_exhaustive": "universe_exhausted",
}


def _insertion_count(patch_key):
    # Batch receipts encode the full word, including original support tokens.
    return sum(
        token[0] == "sel" or (token[0] == "tok" and token[1] == INSERTED)
        for token in patch_key[1]
    )


@pytest.fixture(scope="module")
def mode_results():
    """Each variant owns a fresh route/cache; assertions may reuse its result."""

    results = {}

    def run(policy, gain):
        key = policy, gain
        if key not in results:
            results[key] = _run(_config(
                exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
                search_policy=policy,
                gain_evaluation=gain,
                debt_policy="drift_ranked",
                insertion_gate_mode="mclachlan_l2",
                max_insertion_batch_size=2,
                max_joint_patch_evaluations=256,
                max_compiled_patches_per_checkpoint=256,
                compile_workers=1,
            ))
        return results[key]

    return run


@pytest.mark.parametrize("policy,gain", tuple(product(POLICIES, GAINS)))
def test_six_modes_use_real_complete_patch_pricing_and_account_acquisition(
    mode_results, policy, gain,
):
    selection, payload = mode_results(policy, gain)
    route = payload["selector_route"]
    assert (route["search_policy"], route["gain_evaluation"]) == (policy, gain)
    assert route["bundle"]["bundle_id"] == EXCHANGE_ROUTE_BUNDLE_Q_COMPILED
    assert route["bundle"]["compiler"]["backend_name"] == "FakeMarrakesh"
    assert route["compile"]["backend"]["backend_name"] == "FakeMarrakesh"
    assert route["compile"]["compile_calls"] > 0
    assert payload["cost_normalization"] == "compiled_marrakesh_population_v1"
    assert route["max_compiled_patches_per_checkpoint"] == 256
    assert route["levels"]
    search = route["levels"][0]["search"]
    assert search["schema"] == "paper_ii_patch_search_v1"
    assert (search["search_policy"], search["gain_evaluation"]) == (policy, gain)
    assert search["stop_reason"] == SEARCH_END[policy]
    assert search["exhaustive_complete"] is (policy == "bounded_exhaustive")
    assert 0 < search["evaluated_patch_count"] <= 256
    if policy == "block_exchange":
        assert search["move_size"] == 2

    # A batch bound of two must actually exercise multi-operator pricing.
    ranked = route["levels"][0]["ranked"]
    assert any(_insertion_count(row["patch"]) == 2 for row in ranked)
    assert all(0 < row["K"] for row in ranked)
    assert all(_insertion_count(row["patch"]) <= 2 for row in ranked)
    assert selection.certification is not None and selection.certification.certified
    assert payload["committed"]["compiled_cost_marginal"] is not None
    assert len(payload["committed"]["compiled_patch_word_sha256"]) == 64

    work = payload["estimator_work"]
    selection_pairs = validation_pairs = 0
    for selector_pass in work["selector_passes"]:
        acquired = selector_pass["structural_acquisition"]
        assert acquired["schema"] == "paper_ii_structural_acquisition_v1"
        assert acquired["lazy_candidate_pairs"] is True
        assert acquired["complete"] is True
        stages = acquired["stages"]
        selection_pairs += stages["selection"]["candidate_pair_entry_count"]
        validation_pairs += stages["validation"]["candidate_pair_entry_count"]
        assert acquired["logical_total_estimator_entries"] == sum(
            stage["logical_entry_count"] for stage in stages.values()
        )
        assert selector_pass["logical_shared_cache_entry_count"] == (
            acquired["logical_total_estimator_entries"]
        )
    if gain == "singleton_factorized":
        assert selection_pairs == 0
        if len(selection.committed.inserted_selection) > 1:
            assert validation_pairs > 0
    else:
        assert selection_pairs > 0
        # Frozen finalist pairs are already present from joint ranking.
        assert validation_pairs == 0
    assert work["logical_total_selector_estimator_entries"] == (
        work["logical_shared_cache_entry_count"]
        + work["logical_finalist_geometry_entry_count"]
        + work["logical_refit_entry_count"]
    )


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("joint_limit,compile_limit,reason", (
    (1, 256, "joint_work_budget"),
    (256, 1, "compiled_work_budget"),
))
def test_guard_termination_does_not_claim_complete_search(
    policy, joint_limit, compile_limit, reason,
):
    _, payload = _run(_config(
        exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
        search_policy=policy,
        gain_evaluation="singleton_factorized",
        debt_policy="drift_ranked",
        insertion_gate_mode="mclachlan_l2",
        max_insertion_batch_size=2,
        max_joint_patch_evaluations=joint_limit,
        max_compiled_patches_per_checkpoint=compile_limit,
        compile_workers=1,
    ))
    search = payload["selector_route"]["levels"][0]["search"]
    assert search["stop_reason"] == reason
    assert search["exhaustive_complete"] is False
    assert search["evaluated_patch_count"] <= 1


def test_exhaustive_gain_modes_share_complete_patch_cost_population(mode_results):
    _, factorized = mode_results("bounded_exhaustive", "singleton_factorized")
    _, joint = mode_results("bounded_exhaustive", "joint_batch")
    factorized_route, joint_route = factorized["selector_route"], joint["selector_route"]
    assert factorized_route["bundle"]["compiler"] == joint_route["bundle"]["compiler"]
    assert factorized_route["levels"][0]["population_receipt"]["normalization"] == (
        joint_route["levels"][0]["population_receipt"]["normalization"]
    )
    assert factorized_route["compile"]["compile_calls"] == joint_route["compile"]["compile_calls"]


def test_factorized_finalist_validation_charges_missing_batch_pair():
    # Both available quadratures now capture drift; neither singleton spans
    # the target, so the selected batch needs authoritative joint validation.
    state = _state()
    hamiltonian = TimeDependentHamiltonian(
        static_poly=_poly(("ez", 2.0), ("xx", 0.6), ("yx", 0.6))
    )
    evaluation = evaluate_mclachlan_geometry(
        state=state, hamiltonian=hamiltonian, theta_runtime=state.theta_runtime,
        time=0.0, include_tangent_matrix=True,
    )
    selection, payload = select_generalized_exchange_patch(
        state=state, hamiltonian=hamiltonian, theta_runtime=state.theta_runtime,
        time=0.0, base_evaluation=evaluation,
        base_step=solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY),
        inverse_policy=POLICY,
        support_config=_config(
            exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
            search_policy="bounded_exhaustive", gain_evaluation="singleton_factorized",
            insertion_gate_mode="mclachlan_l2", debt_policy="drift_ranked",
            deletion_enabled=False, max_insertion_batch_size=2,
            max_joint_patch_evaluations=256, max_compiled_patches_per_checkpoint=256,
        ),
        runtime_state=_PruneControllerRuntimeState(), time_index=3,
    )
    assert len(selection.committed.inserted_selection) == 2
    assert selection.committed.joint_validation_required is False
    assert selection.certification.certified
    stages = payload["estimator_work"]["selector_passes"][0]["structural_acquisition"]["stages"]
    assert stages["selection"]["candidate_pair_entry_count"] == 0
    assert stages["validation"]["candidate_pair_entry_count"] == 1


@pytest.mark.parametrize("bundle_id", (
    EXCHANGE_ROUTE_BUNDLE_Q_COMPILED, EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
))
def test_only_explicit_factorized_deferred_gamma_is_eligible_for_q(bundle_id):
    route = APGeneralizedExchangeConfig.from_route_config(
        _config(exchange_route_bundle=bundle_id)
    ).route
    ordinary_missing = _population_candidate(
        atom="ordinary", delta=2.0, gamma=None, marginal=(1, 2, 3),
    )
    deferred = replace(
        ordinary_missing,
        inserted_selection=(("a", 0), ("b", 0)),
        gain_evaluation="singleton_factorized",
        joint_validation_required=True,
    )
    unmarked = replace(deferred, inserted_selection=(("unmarked", 0),),
                       gain_evaluation="joint_batch")
    validated_without_gamma = replace(
        deferred, inserted_selection=(("validated", 0),), joint_validation_required=False,
    )
    failed = replace(
        deferred, inserted_selection=(("failed", 0),),
        ordinate_error="GammaMonotonicityError:synthetic validation failure",
    )
    level = build_population_ranker(
        bundle=route.bundle,
        route=route,
        accuracy_debt=True,
        weights=StructuralScoreWeights(),
        score_floor=0.0,
        cost_settings=AppendCostSettings(),
    )((ordinary_missing, deferred, unmarked, validated_without_gamma, failed))
    assert (deferred.order_key in level.rank_info) is (
        bundle_id == EXCHANGE_ROUTE_BUNDLE_Q_COMPILED
    )
    assert all(candidate.order_key not in level.rank_info for candidate in (
        ordinary_missing, unmarked, validated_without_gamma, failed,
    ))
