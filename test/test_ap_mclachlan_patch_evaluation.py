"""Independent array checks for the shared complete-patch evaluator."""

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    InsertionPlan,
    inserted_tokens,
    survivor_tokens,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.patch_enumeration import Patch
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import (
    CandidateCostInputs,
    PatchEvaluator,
    StructuralScoreWeights,
)
from pipelines.time_dynamics.ap_mclachlan.structural_cache import StructuralInsertionCache


def _evaluator(active, inserted, target, positions, **kwargs):
    cache = StructuralInsertionCache(
        checkpoint_key=(),
        coordinate_keys=tuple(f"c{i}" for i in range(active.shape[1])),
        column_index={position: i for i, position in enumerate(positions)},
        tangent_matrix=inserted,
        cross=active.T @ inserted,
        gram_inserted=inserted.T @ inserted,
        force_inserted=inserted.T @ target,
    )
    cuts = {}
    for atom, cut in positions:
        cuts.setdefault(atom, []).append(cut)
    return PatchEvaluator(
        cache=cache,
        base_K=active.T @ active,
        base_f=active.T @ target,
        norm_b_sq=float(target @ target),
        inverse_policy=McLachlanInversePolicy(ridge_lambda=0.0),
        weights=StructuralScoreWeights(),
        cuts_by_atom=cuts,
        insertion_cost=lambda atoms: float(len(atoms)),
        deletion_cost=lambda removed: float(len(removed)),
        **kwargs,
    )


@pytest.mark.parametrize("workers", [1, 4])
def test_joint_capture_keeps_interacting_columns_and_prices_only_complete_proposals(workers):
    # Each new direction mostly points along e0; their difference spans the
    # target e1. The old direction e2 contributes no capture.
    epsilon = 0.1
    active = np.array([[0.0], [0.0], [1.0]])
    inserted = np.array([[1.0, 1.0], [epsilon, -epsilon], [0.0, 0.0]])
    target = np.array([0.0, 1.0, 0.0])
    calls = []

    def price(removed, selection, plan):
        calls.append((removed, selection, plan))
        # Deliberately nonadditive: the pair's circuit has a smaller cost.
        cost = -2 if len(selection) == 2 else 3
        return CandidateCostInputs(cost_marginal=(cost, cost, cost), compile_calls=1)

    evaluator = _evaluator(
        active, inserted, target, (("a", 0), ("b", 0)),
        scoring_workers=workers,
        route_bundle=SimpleNamespace(uses_population_scoring=True),
        candidate_cost_inputs=price,
        population_accuracy_debt=True,
        gamma_rcond=1e-10,
    )
    patches = (
        Patch((0,), (("a", 0),)),
        Patch((0,), (("b", 0),)),
        Patch((0,), (("a", 0), ("b", 0))),
    )
    a, b, joint = tuple(evaluator.evaluate(patches, "independent_joint"))
    denominator = 1.0 + evaluator.weights.epsilon_norm
    expected_single = epsilon**2 / (1.0 + epsilon**2) / denominator
    assert a.delta == pytest.approx(expected_single)
    assert b.delta == pytest.approx(expected_single)
    assert joint.delta == pytest.approx(1.0 / denominator)
    assert joint.m_gamma == pytest.approx(joint.delta)
    assert joint.delta > 40 * (a.delta + b.delta)
    assert joint.cost_marginal == (-2, -2, -2)
    assert calls == [(p.removed_runtime_indices, p.inserted_selection, p.plan) for p in patches]
    assert evaluator.compile_calls_total == 3


def test_joint_plan_uses_full_word_cuts_before_deletion_remapping():
    basis = np.eye(7)
    active, inserted = basis[:, :3], basis[:, 3:]
    target = np.arange(1.0, 8.0)
    evaluator = _evaluator(
        active, inserted, target,
        (("a", 0), ("a", 2), ("b", 0), ("b", 3)),
    )
    c0, c1, c2 = survivor_tokens(("c0", "c1", "c2"))
    a, b = inserted_tokens(("a", "b"))
    full = (c0, c1, a, c2, b)
    reduced = (c0, a, c2, b)
    plan = InsertionPlan(full, reduced, full, reduced, ("c1",))
    patch = Patch((1,), (("a", 2), ("b", 3)), plan)
    candidate, = evaluator.evaluate((patch,), "full_word")
    # Reading reduced-word cuts would select the wrong cached columns.
    assert candidate.inserted_selection == (("a", 2), ("b", 3))
    assert candidate.plan is plan
    columns = basis[:, [0, 2, 4, 6]]
    velocity = np.linalg.lstsq(columns, target, rcond=None)[0]
    captured = target @ target - np.linalg.norm(columns @ velocity - target)**2
    assert candidate.q == pytest.approx(
        captured / (target @ target + evaluator.weights.epsilon_norm)
    )


def _lazy_interacting_evaluator(mode, workers=1, price=None):
    active = np.array([[0.0], [0.0], [1.0]])
    inserted = np.array([[1.0, 1.0], [0.1, -0.1], [0.0, 0.0]])
    target = np.array([0.0, 1.0, 2.0])
    cache = StructuralInsertionCache(
        checkpoint_key=(), coordinate_keys=("c0",),
        column_index={("a", 0): 0, ("b", 0): 1},
        tangent_matrix=inserted, cross=active.T @ inserted,
        gram_inserted=None, force_inserted=inserted.T @ target,
        lazy_candidate_pairs=True,
        candidate_norms=np.sum(inserted * inserted, axis=0),
    )
    return PatchEvaluator(
        cache=cache, base_K=active.T @ active, base_f=active.T @ target,
        norm_b_sq=float(target @ target),
        inverse_policy=McLachlanInversePolicy(ridge_lambda=0.0),
        weights=StructuralScoreWeights(lambda_cond_relief=2.0, lambda_hist=1.0),
        cuts_by_atom={"a": (0,), "b": (0,)},
        insertion_cost=lambda atoms: len(atoms), deletion_cost=lambda removed: len(removed),
        deletion_history_loss=lambda removed: 0.25 if removed else 0.0,
        scoring_workers=workers, gain_evaluation=mode, gamma_rcond=1e-10,
        route_bundle=SimpleNamespace(uses_population_scoring=True),
        candidate_cost_inputs=price or (lambda *_: CandidateCostInputs(cost_marginal=(-2, -3, -4))),
        population_accuracy_debt=True,
    )


@pytest.mark.parametrize("workers", [1, 4])
def test_factorized_ranking_acquires_no_pairs_and_validation_keeps_provisional_rank(workers):
    priced = []

    def price(removed, selection, plan):
        priced.append((removed, selection, plan))
        return CandidateCostInputs(cost_marginal=(-2, -3, -4), compile_calls=1)

    evaluator = _lazy_interacting_evaluator("singleton_factorized", workers, price)
    patches = (
        Patch((0,), (("a", 0),)),
        Patch((0,), (("b", 0),)),
        Patch((0,), (("a", 0), ("b", 0))),
    )
    a, b, provisional = evaluator.evaluate(patches, "factorized")
    denominator = 5.0 + evaluator.weights.epsilon_norm
    assert provisional.delta == pytest.approx((2 * 0.01 / 1.01 - 4.0) / denominator)
    assert provisional.m_gamma is None and provisional.joint_validation_required
    assert all(len(key[1]) <= 1 for key in evaluator.cache.solve_memo)
    assert evaluator.cache.acquisition_receipt()["stages"]["selection"]["candidate_pair_attempt_count"] == 0
    assert len(priced) == evaluator.compile_calls_total == 3

    joint = _lazy_interacting_evaluator("joint_batch")
    joint_a, joint_b, joint_block = joint.evaluate(patches, "factorized")
    assert (a.q, a.delta, a.m_gamma) == (joint_a.q, joint_a.delta, joint_a.m_gamma)
    assert (b.q, b.delta, b.m_gamma) == (joint_b.q, joint_b.delta, joint_b.m_gamma)
    actual = evaluator.authoritative_joint_candidate(provisional)
    assert actual.q == pytest.approx(1.0 / denominator)
    assert actual.q == joint_block.q and actual.m_gamma == joint_block.m_gamma
    assert actual.deletion_loss == pytest.approx(4.0 / denominator)
    assert actual.gain_evaluation == "singleton_factorized" and not actual.joint_validation_required
    assert (actual.score, actual.rank_primary, actual.rank_secondary) == (
        provisional.score, provisional.rank_primary, provisional.rank_secondary
    )
    assert actual.cost_marginal == provisional.cost_marginal
    assert evaluator.q_of(patches[-1].removed_runtime_indices, patches[-1].inserted_selection) == provisional.q
    assert len(priced) == 3
    receipt = evaluator.cache.acquisition_receipt()
    assert receipt["stages"]["selection"]["candidate_pair_entry_count"] == 0
    assert receipt["stages"]["validation"]["candidate_pair_entry_count"] == 1
    assert receipt["logical_total_estimator_entries"] == 7
    evaluator.authoritative_joint_candidate(provisional)
    assert evaluator.cache.acquisition_receipt() == receipt


def test_deferred_gamma_failure_is_reported_after_charging_validation(monkeypatch):
    from pipelines.time_dynamics.ap_mclachlan import patch_evaluation
    from pipelines.time_dynamics.ap_mclachlan.objective_ordinates import GammaMonotonicityError

    evaluator = _lazy_interacting_evaluator("singleton_factorized")
    provisional, = evaluator.evaluate((Patch((), (("a", 0), ("b", 0))),), "insert")

    def reject(**kwargs):
        raise GammaMonotonicityError("synthetic joint validation failure")

    monkeypatch.setattr(patch_evaluation, "check_gamma_nested_monotonicity", reject)
    actual = evaluator.authoritative_joint_candidate(provisional)
    assert actual.m_gamma is None
    assert "synthetic joint validation failure" in actual.ordinate_error
    assert actual.cost_marginal == provisional.cost_marginal
    assert evaluator.cache.acquisition_receipt()["stages"]["validation"]["candidate_pair_entry_count"] == 1
