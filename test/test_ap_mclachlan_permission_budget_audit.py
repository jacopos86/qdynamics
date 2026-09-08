"""Array-level audit of whole-set permission and search work admission.

The stabilized solves and permission mathematics are real. Resource inputs
are observed callbacks, so these tests count pricing requests, not Qiskit
transpilation calls or hardware measurements. No trajectory is propagated.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan import exchange_structural, patch_evaluation
from pipelines.time_dynamics.ap_mclachlan.deletion_permission import DeletionPermissionEvaluator
from pipelines.time_dynamics.ap_mclachlan.exchange_config import resolve_route_bundle
from pipelines.time_dynamics.ap_mclachlan.exchange_integration import build_population_ranker
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import enumerate_structural_candidates
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.patch_enumeration import Patch
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import CandidateCostInputs, PatchEvaluator, StructuralScoreWeights
from pipelines.time_dynamics.ap_mclachlan.search_methods.problem import PatchProblem
from pipelines.time_dynamics.ap_mclachlan.search_methods.runtime import prepare_patch_search
from pipelines.time_dynamics.ap_mclachlan.search_methods.session import SearchLimits, SearchSession
from pipelines.time_dynamics.ap_mclachlan.structural_cache import StructuralInsertionCache


PATHS = [
    ("family_frontiers", "joint_batch"),
    *((method, mode) for method in ("bounded_exhaustive", "greedy", "block_exchange")
      for mode in ("joint_batch", "singleton_factorized")),
]


def _case(monkeypatch, *, joint_refusal="angle", branch_only_pool=False):
    basis = np.eye(4)
    if joint_refusal == "angle":
        active = basis[:, :2]
        target = 0.6 * basis[:, 2] + 0.8 * basis[:, 3]
        theta = np.array([0.06, 0.06])
    else:
        # Either duplicate alone preserves the target direction. Removing
        # both loses it, even though both singleton permissions pass.
        active = basis[:, [0, 0]]
        target = basis[:, 0] + 0.6 * basis[:, 2] + 0.8 * basis[:, 3]
        theta = np.zeros(2)
    inserted = basis[:, 2:]
    cache = StructuralInsertionCache(
        checkpoint_key=(), coordinate_keys=("s0", "s1"),
        column_index={("a", 0): 0, ("b", 0): 1},
        tangent_matrix=inserted, cross=active.T @ inserted,
        gram_inserted=inserted.T @ inserted, force_inserted=inserted.T @ target,
    )
    weights = StructuralScoreWeights()
    permission = DeletionPermissionEvaluator(
        gram=active.T @ active, force=active.T @ target,
        norm_b_sq=float(target @ target), theta_runtime=theta,
        rotation_coefficients=np.ones(2), ray_distance_max=0.1,
        normalized_schur_loss_max=0.01, epsilon_norm=weights.epsilon_norm,
    )
    permission_calls, pool_calls, prices, actual_solves = [], [], [], []

    def assess(removed):
        permission_calls.append(removed)
        return permission.assess(removed)

    def pool(removed):
        pool_calls.append(removed)
        return () if branch_only_pool and not removed else ("a", "b")

    def price(removed, selection, plan):
        prices.append((removed, selection))
        return CandidateCostInputs(cost_marginal=(0, 0, 0), compile_calls=1)

    original_solve = patch_evaluation.structural_candidate_solve

    def observe_solve(**kwargs):
        if kwargs["memo_key"] not in kwargs["cache"].solve_memo:
            actual_solves.append(kwargs["memo_key"])
        return original_solve(**kwargs)

    monkeypatch.setattr(patch_evaluation, "structural_candidate_solve", observe_solve)
    bundle = resolve_route_bundle("paper_ii_exchange_q_compiled_v1")
    kwargs = dict(
        cache=cache, base_K=active.T @ active, base_f=active.T @ target,
        norm_b_sq=float(target @ target), weights=weights,
        inverse_policy=McLachlanInversePolicy(ridge_lambda=1e-6),
        deletable_indices=(0, 1), min_surviving_support=0,
        cuts_by_atom={"a": (0,), "b": (0,)},
        candidate_pool_for_deletion=pool, insertion_cost=lambda atoms: len(atoms),
        deletion_cost=lambda removed: len(removed), deletion_permission=assess,
        tokens_commute=lambda _left, _right: True, max_insertion_batch_size=2,
        route_bundle=bundle, gamma_rcond=1e-10, candidate_cost_inputs=price,
        population_accuracy_debt=True,
    )
    ranker = build_population_ranker(
        bundle=bundle, route=SimpleNamespace(maintenance_loss_tolerance=1e-3),
        accuracy_debt=True, weights=weights, score_floor=0, cost_settings=None,
    )
    return SimpleNamespace(
        kwargs=kwargs, ranker=ranker, permission=permission,
        permission_calls=permission_calls, pool_calls=pool_calls,
        prices=prices, actual_solves=actual_solves, cache=cache,
    )


def _run(case, method, mode, **limits):
    kwargs = {**case.kwargs, **limits}
    if method == "family_frontiers":
        return enumerate_structural_candidates(**kwargs)
    search = prepare_patch_search(
        search_policy=method, gain_evaluation=mode,
        structural_kwargs=kwargs, population_ranker=case.ranker,
    )
    tuple(search.families())
    return search.result


@pytest.mark.parametrize("method,mode", PATHS)
@pytest.mark.parametrize("joint_refusal", ["angle", "schur"])
def test_singleton_permissions_do_not_authorize_a_refused_joint_set(monkeypatch, method, mode, joint_refusal):
    case = _case(monkeypatch, joint_refusal=joint_refusal)
    result = _run(case, method, mode)
    assert case.permission_calls == [(0,), (1,), (0, 1)]
    assert case.permission.assess((0,)).permitted
    assert case.permission.assess((1,)).permitted
    joint = case.permission.assess((0, 1))
    assert not joint.permitted
    if joint_refusal == "angle":
        assert joint.exact_schur_loss is None
        assert joint.ray_distance_upper_bound == pytest.approx(np.sin(0.12))
    else:
        assert joint.exact_schur_loss == pytest.approx(1.0)
    assert len(result.candidates) == 12
    assert all(c.removed_runtime_indices != (0, 1) for c in result.candidates)
    assert any(c.removed_runtime_indices and len(c.inserted_selection) == 2 for c in result.candidates)
    assert all(removed != (0, 1) for removed, _selection in case.actual_solves)
    assert all(removed != (0, 1) for removed, _selection in case.prices)
    assert len(case.prices) == result.compile_guard.scored_count == 9
    assert len(case.actual_solves) == (9 if mode == "singleton_factorized" else 12)
    assert result.guard.scored_count == 12
    assert (0, 1) not in case.pool_calls


@pytest.mark.parametrize("method,mode", PATHS)
@pytest.mark.parametrize("limits", [
    {"max_joint_patch_evaluations": 2},
    {"max_compiled_patches_per_checkpoint": 0},
])
def test_whole_request_refusal_does_not_reserve_or_evaluate_a_partial_request(monkeypatch, method, mode, limits):
    case = _case(monkeypatch)
    result = _run(case, method, mode, **limits)
    expected_stay = 0 if method == "family_frontiers" else 1
    assert len(result.candidates) == result.guard.scored_count == expected_stay
    assert result.compile_guard.scored_count == 0
    assert case.prices == []
    assert case.permission_calls == []
    assert case.actual_solves == [((), ())]  # Shared baseline precedes search admission.


@pytest.mark.parametrize("method,mode", PATHS)
def test_zero_structural_budget_does_no_set_permission_or_proposal_work(monkeypatch, method, mode):
    case = _case(monkeypatch)
    result = _run(case, method, mode, max_joint_patch_evaluations=0)
    assert not result.candidates
    assert result.guard.scored_count == result.compile_guard.scored_count == 0
    assert not case.permission_calls and not case.prices
    assert case.actual_solves == [((), ())]
    assert case.pool_calls == ([()] if method == "family_frontiers" else [])


@pytest.mark.parametrize("method,mode", PATHS)
def test_compiler_refusal_after_permission_never_compiles_the_refused_request(monkeypatch, method, mode):
    case = _case(monkeypatch, branch_only_pool=True)
    result = _run(case, method, mode, max_compiled_patches_per_checkpoint=1)
    if method == "family_frontiers":
        # The first permitted branch already proves the complete rung
        # cannot fit. No later permission or pool needs to be requested.
        assert case.permission_calls == [(0,)]
        assert result.guard.scored_count == len(result.candidates) == 1
        assert case.actual_solves == [((), ())]
    else:
        # The permitted deletion boundary completes before its insertion
        # neighborhood is independently refused. No later D is visited.
        assert case.permission_calls == [(0,)]
        assert result.guard.scored_count == len(result.candidates) == 2
        assert case.actual_solves == [((), ()), ((0,), ())]
    assert result.compile_guard.scored_count == 0
    assert not case.prices


def test_backend_failure_retains_admitted_reservation_but_only_completed_evidence(monkeypatch):
    case = _case(monkeypatch)
    names = {
        "cache", "base_K", "base_f", "norm_b_sq", "weights", "inverse_policy",
        "cuts_by_atom", "insertion_cost", "deletion_cost", "route_bundle",
        "gamma_rcond", "candidate_cost_inputs", "population_accuracy_debt",
    }
    evaluator = PatchEvaluator(**{k: v for k, v in case.kwargs.items() if k in names})
    original_price = evaluator.candidate_cost_inputs

    def fail_second(removed, selection, plan):
        if selection == (("b", 0),):
            raise ValueError("audit backend failure")
        return original_price(removed, selection, plan)

    evaluator.candidate_cost_inputs = fail_second
    problem = PatchProblem(
        coordinate_keys=("s0", "s1"), deletable_indices=(), min_surviving_support=0,
        candidate_pool_for_deletion=lambda _d: ("a", "b"), cuts_by_atom=case.kwargs["cuts_by_atom"],
    )
    session = SearchSession(problem, evaluator, case.ranker, SearchLimits())
    with pytest.raises(ValueError, match="audit backend failure"):
        session.evaluate((Patch((), (("a", 0),)), Patch((), (("b", 0),))), label="backend_failure")
    assert session.guard.scored_count == session.compile_guard.scored_count == 2
    assert len(session.evidence) == 1
    assert next(iter(session.evidence.values())).inserted_selection == (("a", 0),)
    assert len(case.actual_solves) == 3  # baseline and both attempted proposal solves


def _refused_first_case(monkeypatch, *, include_refused=True):
    case = _case(monkeypatch)
    case.permission = DeletionPermissionEvaluator(
        gram=case.kwargs["base_K"], force=case.kwargs["base_f"],
        norm_b_sq=case.kwargs["norm_b_sq"], theta_runtime=np.array([0.2, 0.0]),
        rotation_coefficients=np.ones(2), ray_distance_max=0.1,
        normalized_schur_loss_max=0.01,
        epsilon_norm=case.kwargs["weights"].epsilon_norm,
    )

    def assess(removed):
        case.permission_calls.append(removed)
        return case.permission.assess(removed)

    case.kwargs.update(
        deletion_permission=assess, max_insertion_batch_size=1,
        deletable_indices=(0, 1) if include_refused else (1,),
    )
    return case


@pytest.mark.parametrize("method", ["family_frontiers", "bounded_exhaustive", "greedy", "block_exchange"])
def test_refused_sets_do_not_block_later_permitted_patch_coverage(monkeypatch, method):
    case = _refused_first_case(monkeypatch)
    result = _run(case, method, "joint_batch", max_joint_patch_evaluations=6)
    # D=(0,) and D=(0,1) violate the angle bound. D=(1,) is permitted.
    # All six permitted patches fit, including both true exchanges, even
    # though the refused D is encountered first in the same cardinality rung.
    assert not case.permission.assess((0,)).permitted
    assert not case.permission.assess((0, 1)).permitted
    assert case.permission.assess((1,)).permitted
    assert len(result.candidates) == result.guard.scored_count == 6
    assert case.permission_calls == [(0,), (1,)]
    assert len(case.prices) == result.compile_guard.scored_count == 4
    assert sum(c.kind == "exchange" for c in result.candidates) == 2
    assert (0,) not in case.pool_calls

    # Remove only forbidden deletion choices from enumeration. This does not
    # change the permitted universe or the evaluated population at this budget.
    with pytest.MonkeyPatch.context() as fresh:
        filtered = _refused_first_case(fresh, include_refused=False)
        recovered = _run(filtered, method, "joint_batch", max_joint_patch_evaluations=6)
    assert len(recovered.candidates) == recovered.guard.scored_count == 6
    assert len(filtered.prices) == recovered.compile_guard.scored_count == 4
    assert sum(c.kind == "exchange" for c in recovered.candidates) == 2
    assert [(c.removed_runtime_indices, c.inserted_selection) for c in result.candidates] == [
        (c.removed_runtime_indices, c.inserted_selection) for c in recovered.candidates
    ]


@pytest.mark.parametrize("method,mode", PATHS)
def test_raw_deletion_cardinality_cannot_refuse_a_fitting_permitted_rung(monkeypatch, method, mode):
    case = _refused_first_case(monkeypatch)

    def empty_pool(removed):
        case.pool_calls.append(removed)
        return ()

    case.kwargs["candidate_pool_for_deletion"] = empty_pool
    result = _run(case, method, mode, max_joint_patch_evaluations=2)
    # Two raw singleton D choices exceed the one remaining slot, but only
    # D=(1,) is permitted. Its boundary must still be evaluated.
    assert len(result.candidates) == result.guard.scored_count == 2
    assert case.permission_calls == [(0,), (1,)]
    assert case.pool_calls == [(), (1,)]
    assert [c.removed_runtime_indices for c in result.candidates] == [(), (1,)]


def test_frontier_construction_stops_at_first_proof_of_compile_budget_refusal(monkeypatch):
    case = _case(monkeypatch)
    case.kwargs.update(deletable_indices=(), tokens_commute=lambda _left, _right: False)
    generated_joint = []
    original = exchange_structural.iter_permitted_patches

    def observe_patches(**kwargs):
        for patch in original(**kwargs):
            if len(patch.inserted_selection) >= 2:
                generated_joint.append(patch)
            yield patch

    monkeypatch.setattr(exchange_structural, "iter_permitted_patches", observe_patches)
    result = _run(case, "family_frontiers", "joint_batch", max_compiled_patches_per_checkpoint=2)
    assert len(generated_joint) == 1
    assert len(result.candidates) == result.guard.scored_count == 3
    assert len(case.prices) == result.compile_guard.scored_count == 2
    assert len(case.actual_solves) == 3
    assert result.guard.rejected_family == "frontier_0_w2"
