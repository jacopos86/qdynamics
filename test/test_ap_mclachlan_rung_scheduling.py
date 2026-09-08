"""Independent coverage contracts for finite-budget patch scheduling.

Synthetic complete-patch evidence isolates traversal from the physical gain
backend. These tests make no estimator-work or compilation-speed claims.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pytest

from pipelines.time_dynamics.ap_mclachlan.search_methods.block_local_exchange import (
    search_patches as block_search,
)
from pipelines.time_dynamics.ap_mclachlan.search_methods.exhaustive import (
    search_patches as exhaustive_search,
)
from pipelines.time_dynamics.ap_mclachlan.search_methods.session import SearchLimits
from test.test_ap_mclachlan_search_methods import (
    SEARCHES,
    _SyntheticEvaluator,
    _adversarial_rank,
    _block,
    _problem,
    _rank_population,
)


def _large_placement_problem():
    pool = tuple(f"a{i}" for i in range(8))
    return _problem(
        pool, maximum_size=2,
        coordinate_keys=tuple(f"s{i}" for i in range(8)),
        deletable_indices=(0, 1, 2, 3), max_deletion_batch_size=1,
        cuts_by_atom={atom: tuple(range(9)) for atom in pool},
        tokens_commute=lambda _left, _right: False,
    )


@SEARCHES
@pytest.mark.parametrize("limits,reason", [
    (SearchLimits(joint_patch_evaluations=32), "joint_work_budget"),
    (SearchLimits(compiled_patches_per_checkpoint=31), "compiled_work_budget"),
])
def test_small_budget_reaches_exchange_before_append_placements_are_exhausted(search, limits, reason):
    evaluator = _SyntheticEvaluator(lambda patch: float(len(patch.inserted_selection)))
    result = search(_large_placement_problem(), evaluator, _rank_population, limits)

    # D=empty alone has 8*9 distinct singleton placements, before any pairs.
    # Neither work guard can fund those 72 placements in this test.
    assert result.stop_reason == reason
    assert not result.exhaustive_complete
    assert len(result.candidates) == len(evaluator.requested) == 32
    assert any(not c.removed_runtime_indices and c.inserted_selection for c in result.candidates)
    assert any(c.removed_runtime_indices and c.inserted_selection for c in result.candidates)
    assert sum(not c.removed_runtime_indices and len(c.inserted_selection) == 1
               for c in result.candidates) < 72
    assert result.guard.scored_count == 32
    assert result.compile_guard.scored_count == 31  # The stay patch has no compile charge.
    assert len({c.order_key for c in result.candidates}) == len(result.candidates)


@SEARCHES
def test_live_single_deletion_branches_rotate_before_one_branch_drains(search):
    evaluator = _SyntheticEvaluator(lambda patch: float(len(patch.inserted_selection)))
    result = search(
        _large_placement_problem(), evaluator, _rank_population,
        SearchLimits(joint_patch_evaluations=64),
    )
    assert result.stop_reason == "joint_work_budget"
    assert {
        c.removed_runtime_indices for c in result.candidates
        if c.removed_runtime_indices and c.inserted_selection
    } == {(0,), (1,), (2,), (3,)}
    assert result.permitted_deletion_count == 5  # D=empty plus the four branches.
    assert result.refused_deletion_count == 0


def test_exhaustive_complete_population_contains_every_pure_and_unbalanced_block():
    atoms = ("a", "b", "c")
    coordinates = (0, 1, 2)
    evaluator = _SyntheticEvaluator(lambda patch: float(len(patch.inserted_selection)))
    result = exhaustive_search(
        _problem(
            atoms, maximum_size=3, coordinate_keys=("s0", "s1", "s2"),
            deletable_indices=coordinates,
        ),
        evaluator, _rank_population, SearchLimits(),
    )
    # All tokens commute and use one cut: one realization per mathematical
    # (D,A). Build the reference directly, without the production constructor.
    expected = {
        (removed, inserted)
        for d in range(4) for removed in combinations(coordinates, d)
        for a in range(4) for inserted in combinations(atoms, a)
    }
    observed = {
        (c.removed_runtime_indices, tuple(atom for atom, _cut in c.inserted_selection))
        for c in result.candidates
    }
    assert observed == expected
    assert len(result.candidates) == len(expected) == 64
    assert {(len(d), len(a)) for d, a in observed} == {
        (d, a) for d in range(4) for a in range(4)
    }
    assert result.exhaustive_complete
    assert result.stop_reason == "universe_exhausted"


@SEARCHES
def test_searches_can_reach_higher_pure_and_unbalanced_batches(search):
    result = search(
        _problem(
            ("a", "b", "c"), maximum_size=3,
            coordinate_keys=("s0", "s1", "s2"), deletable_indices=(0, 1, 2),
        ),
        _SyntheticEvaluator(lambda patch: float(len(patch.inserted_selection))),
        _rank_population, SearchLimits(),
    )
    covered = {(len(c.removed_runtime_indices), len(c.inserted_selection)) for c in result.candidates}
    assert {(0, 3), (3, 0), (1, 3), (3, 1)} <= covered


@SEARCHES
def test_denied_union_is_checked_as_a_whole_and_consumes_no_patch_work(search):
    permission_calls, pool_calls = [], []

    def permit(removed):
        permission_calls.append(removed)
        # Each singleton fits this angle-sum bound; their union does not.
        return sum(0.06 for _index in removed) <= 0.1

    def pool(removed):
        pool_calls.append(removed)
        assert removed != (0, 1), "the refused union must never acquire its pool"
        return ("a", "b")

    evaluator = _SyntheticEvaluator(lambda patch: float(len(patch.inserted_selection)))
    result = search(
        _problem(
            ("a", "b"), coordinate_keys=("s0", "s1"), deletable_indices=(0, 1),
            candidate_pool_for_deletion=pool, deletion_permission=permit,
        ),
        evaluator, _rank_population, SearchLimits(),
    )
    assert Counter(permission_calls) == Counter({(0,): 1, (1,): 1, (0, 1): 1})
    assert set(pool_calls) == {(), (0,), (1,)}
    assert all(p.removed_runtime_indices != (0, 1) for p in evaluator.requested)
    assert all(c.removed_runtime_indices != (0, 1) for c in result.candidates)
    assert result.guard.scored_count == len(evaluator.requested) == len(result.candidates) == 12
    assert result.compile_guard.scored_count == 11
    assert result.permitted_deletion_count == 3
    assert result.refused_deletion_count == 1


@SEARCHES
def test_weak_append_gains_do_not_screen_a_strong_conditional_exchange(search):
    def merit(patch):
        block = _block(patch)
        if not patch.removed_runtime_indices:
            return -float(len(block))
        if block == {"b", "c"}:
            return 100.0
        if len(block) == 2:
            return 6.0
        if block == {"b"}:
            return 5.0
        return 1.0 if block else -1.0

    result = search(
        _problem(
            ("a", "b", "c"), maximum_size=2,
            coordinate_keys=("s",), deletable_indices=(0,),
        ),
        _SyntheticEvaluator(merit), _rank_population, SearchLimits(),
    )
    winner = _rank_population(result.candidates).ordered[0]
    assert winner.removed_runtime_indices == (0,)
    assert _block(winner) == {"b", "c"}
    assert winner.delta == pytest.approx(100.0)
    assert all(c.delta < 0 for c in result.candidates
               if not c.removed_runtime_indices and c.inserted_selection)


def test_block_stops_only_after_later_branch_growth_is_reconciled_globally():
    seen_populations = []

    def rank(candidates):
        keys = {c.order_key for c in candidates}
        if seen_populations:
            assert seen_populations[-1] <= keys, "completed evidence cannot disappear"
        seen_populations.append(keys)
        assert len(seen_populations) <= 20, "a fixed population must not cause a ranking cycle"
        late_branch_seen = any(_block(c) == {"z7"} for c in candidates)
        return _adversarial_rank(candidates, frozenset("b" if late_branch_seen else "a"))

    result = block_search(
        _problem(
            ("a", "b") + tuple(f"z{i}" for i in range(8)), maximum_size=1,
            coordinate_keys=("s",), deletable_indices=(0,),
            candidate_pool_for_deletion=lambda removed: (
                tuple(f"z{i}" for i in range(8)) if removed else ("a", "b")
            ),
        ),
        _SyntheticEvaluator(lambda patch: float(len(patch.inserted_selection))),
        rank, SearchLimits(),
    )
    pure_moves = [move["inserted_block"] for move in result.moves if not move["removed"]]
    assert pure_moves[0] == ["a"]
    assert pure_moves[-1] == ["b"]
    assert result.stop_reason == "block_neighborhoods_exhausted"
    assert len(result.candidates) == 12
