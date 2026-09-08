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
