"""Audit existing frontier policy without changing production behavior."""

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.append_cost import AppendCostSettings
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    APGeneralizedExchangeConfig,
    EXCHANGE_ROUTE_BUNDLE_HISTORICAL,
    EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
    build_population_ranker,
    build_selector_inputs,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
    enumerate_structural_candidates,
)
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import evaluate_mclachlan_geometry
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import (
    CandidateCostInputs,
    StructuralScoreWeights,
)
from pipelines.time_dynamics.ap_mclachlan.search_methods.runtime import prepare_patch_search
from pipelines.time_dynamics.ap_mclachlan.structural_cache import StructuralInsertionCache
from test.test_ap_mclachlan_exchange_integration import (
    HAM, _PruneControllerRuntimeState, _config, _state,
)


def _counterexample_inputs():
    """Independent orthogonal geometry and synthetic complete-patch marginals.

    Pure captured drift is .8 for high and .6 for low. Their resource triples
    are (100,100,100) and (0,0,0). The production population normalization
    gives K=4/3 and K=4/5; therefore final compiled-Q prefers low despite its
    smaller pure gain. These resource inputs are not Qiskit measurements.
    """

    cache = StructuralInsertionCache(
        checkpoint_key=("frontier_audit",), coordinate_keys=("active",),
        column_index={("high", 0): 0, ("low", 0): 1},
        tangent_matrix=np.array([[1., 0.], [0., 1.], [0., 0.]], dtype=complex),
        cross=np.zeros((1, 2)), gram_inserted=np.eye(2),
        force_inserted=np.sqrt([.8, .6]),
    )
    route = APGeneralizedExchangeConfig.from_route_config(_config(
        exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )).route
    weights = StructuralScoreWeights(debt_ranking=True)
    priced = []

    def cost(removed, selection, _plan):
        priced.append((removed, selection))
        value = 100 if selection[0][0] == "high" else 0
        return CandidateCostInputs(cost_marginal=(value, value, value))

    kwargs = dict(
        cache=cache, base_K=np.eye(1), base_f=np.zeros(1), norm_b_sq=1.4,
        inverse_policy=McLachlanInversePolicy(ridge_lambda=0.0, solve_damping=0.0),
        weights=weights, deletable_indices=(0,), min_surviving_support=0,
        cuts_by_atom={"high": (0,), "low": (0,)},
        candidate_pool_for_deletion=lambda _removed: ("high", "low"),
        insertion_cost=lambda _inserted: 1.0, deletion_cost=lambda _removed: 1.0,
        route_bundle=route.bundle, gamma_rcond=1.0e-12,
        candidate_cost_inputs=cost, population_accuracy_debt=True,
        max_insertion_batch_size=1,
    )
    ranker = build_population_ranker(
        bundle=route.bundle, route=route, accuracy_debt=True, weights=weights,
        score_floor=0.0, cost_settings=AppendCostSettings(),
    )
    return kwargs, ranker, priced


def test_pure_gain_frontier_excludes_a_better_compiled_cost_exchange():
    restricted_args, _, priced = _counterexample_inputs()
    restricted = enumerate_structural_candidates(
        **restricted_args, exchange_insertion_frontier=1,
    )
    assert restricted.exchange_frontier == (("high", 0),)
    assert ((), (("low", 0),)) in priced  # Its compiled cost was already paid.
    assert ((0,), (("low", 0),)) not in priced

    full_args, ranker, _ = _counterexample_inputs()
    full = enumerate_structural_candidates(**full_args)
    singletons = {c.inserted_selection[0][0]: c for c in full.candidates if c.kind == "insert"}
    assert singletons["high"].delta > singletons["low"].delta
    assert singletons["high"].score > singletons["low"].score
    ranked = ranker(full.candidates)
    assert ranked.rank_info[singletons["high"].order_key].K == pytest.approx(4 / 3)
    assert ranked.rank_info[singletons["low"].order_key].K == pytest.approx(4 / 5)
    exchanges = [c for c in ranked.ordered if c.kind == "exchange"]
    assert [c.inserted_selection[0][0] for c in exchanges] == ["low", "high"]


@pytest.mark.parametrize("policy", ("greedy", "block_exchange", "bounded_exhaustive"))
def test_new_searches_acquire_branch_singletons_without_pure_gain_screening(policy):
    kwargs, ranker, priced = _counterexample_inputs()
    search = prepare_patch_search(
        search_policy=policy, gain_evaluation="joint_batch",
        structural_kwargs=kwargs, population_ranker=ranker,
    )
    list(search.families())
    assert search.result is not None
    assert ((0,), (("low", 0),)) in priced
    assert ((0,), (("high", 0),)) in priced
    ranked = ranker(search.result.candidates)
    assert ranked.ordered[0].inserted_selection == (("low", 0),)
    assert search.receipt()["stop_reason"] != "joint_work_budget"


@pytest.mark.parametrize("bundle_id", (
    EXCHANGE_ROUTE_BUNDLE_HISTORICAL, EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
))
def test_real_adapter_frontier_uses_realized_q_and_singleton_proxy_normalization(bundle_id):
    state = _state()
    evaluation = evaluate_mclachlan_geometry(
        state=state, hamiltonian=HAM, theta_runtime=state.theta_runtime,
        time=0.0, include_tangent_matrix=True,
    )
    inputs = build_selector_inputs(
        state=state, evaluation=evaluation,
        support_config=_config(
            exchange_route_bundle=bundle_id, compile_workers=1,
            exchange_insertion_frontier=1,
        ),
        runtime_state=_PruneControllerRuntimeState(), theta_runtime=state.theta_runtime,
        time_index=3, deletable_atoms=(), insertions_enabled=True,
        debt_ranking=True, accuracy_debt=True,
    )
    kwargs = inputs["structural_kwargs"]
    # A visible ridge makes realized capture distinct from ideal projection.
    kwargs["inverse_policy"] = McLachlanInversePolicy(ridge_lambda=0.2)
    runtime = inputs["compiled_cost_runtime"]
    try:
        result = enumerate_structural_candidates(**kwargs)
        pure = [candidate for candidate in result.candidates if candidate.kind == "insert"]
        assert pure
        for candidate in pure:
            atom_id, _cut = candidate.inserted_selection[0]
            assert kwargs["insertion_cost"]((atom_id,)) == 1.0
            assert candidate.score == pytest.approx(max(0.0, candidate.delta) + candidate.delta)
        best = min(pure, key=lambda candidate: (-candidate.score, candidate.inserted_selection))
        assert result.exchange_frontier == best.inserted_selection
        if runtime is None:
            assert all(candidate.m_gamma is None and candidate.cost_marginal is None for candidate in pure)
        else:
            assert all(candidate.cost_marginal is not None for candidate in pure)
            assert any(abs(candidate.delta - candidate.m_gamma) > 1.0e-8 for candidate in pure)
            atom_id, cut = best.inserted_selection[0]
            placement = runtime.placement(atom_id, cut)
            assert placement.base_triple == runtime.base_triple
            np.testing.assert_array_equal(runtime.ref_state, state.psi_ref)
            # Placement is already fixed using the undeleted base circuit;
            # pricing a deletion-containing patch reuses that same placement.
            exchange_cost = runtime.cost_inputs((0,), best.inserted_selection, None)
            assert runtime.placement(atom_id, cut) is placement
            assert exchange_cost.placement_cuts == ((atom_id, placement.chosen_cut),)
    finally:
        if runtime is not None:
            runtime.close()
