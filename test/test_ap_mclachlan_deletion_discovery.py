"""Independent bounds and coverage checks for deletion discovery.

These synthetic tests distinguish a deletion-set inspection from evaluation
of a complete patch. They do not infer physical gain, compiler cost, or QPU
measurement savings from the synthetic evaluator.
"""

from __future__ import annotations

import pytest

from pipelines.time_dynamics.ap_mclachlan.patch_enumeration import Patch
from pipelines.time_dynamics.ap_mclachlan.search_methods.session import (
    SearchLimits,
    SearchSession,
)
from test.test_ap_mclachlan_search_methods import (
    SEARCHES,
    _SyntheticEvaluator,
    _problem,
    _rank_population,
    exhaustive_search,
)


def _bounded_problem(*, n=28, m=8, pool=(), fraction=0.15, **kwargs):
    return _problem(
        pool,
        maximum_size=1,
        coordinate_keys=tuple(f"s{i}" for i in range(n)),
        deletable_indices=tuple(range(m)),
        max_deletion_batch_fraction=fraction,
        **kwargs,
    )


@pytest.mark.parametrize(
    "n,m,fraction,explicit,surviving,expected_k,expected_population",
    [
        (28, 24, 0.15, None, 0, 4, 12950),
        (36, 32, 0.15, None, 0, 5, 242824),
        (28, 24, 0.15, 1, 0, 1, 24),
        (28, 24, 0.15, None, 27, 1, 24),
        (36, 32, 0.15, 0, 0, 0, 0),
        (20, 3, 0.15, None, 0, 3, 7),
        (6, 6, 0.15, None, 0, 0, 0),
        (7, 7, 0.15, None, 0, 1, 7),
        (7, 7, 0.0, None, 0, 0, 0),
        (7, 7, None, None, 0, 7, 127),
        (28, 0, 0.15, None, 0, 0, 0),
    ],
)
def test_fraction_uses_actual_ansatz_size_and_intersects_existing_bounds(
    n, m, fraction, explicit, surviving, expected_k, expected_population,
):
    problem = _bounded_problem(
        n=n, m=m, fraction=fraction,
        max_deletion_batch_size=explicit,
        min_surviving_support=surviving,
    )
    assert problem.deletion_cardinality_limit == expected_k
    assert problem.deletion_subset_count == expected_population
    # The empty boundary is not included in the nonempty population count.
    if expected_population == 0:
        assert tuple(problem.deletion_sets()) == ((),)


@pytest.mark.parametrize("work_cap,expected_limit", [(0, 0), (2, 2), (50, 7), (None, 7)])
def test_discovery_allowance_is_bounded_by_the_nonempty_population(work_cap, expected_limit):
    session = SearchSession(
        _bounded_problem(n=20, m=3), _SyntheticEvaluator(lambda _patch: 0),
        _rank_population, SearchLimits(joint_patch_evaluations=work_cap),
    )
    assert session.deletion_discovery_limit == expected_limit
    assert session.deletion_discovery_remaining == expected_limit
    receipt = session.finish("not_started").receipt()["deletion_discovery"]
    assert receipt["max_batch_fraction"] == 0.15
    assert receipt["ansatz_size"] == 20
    assert receipt["eligible_coordinate_count"] == 3
    assert receipt["max_batch_size"] == 3
    assert receipt["subset_population"] == 7
    assert receipt["work_cap"] == work_cap
    assert receipt["inspection_limit"] == expected_limit
    assert receipt["inspected_count"] == 0
    assert receipt["remaining"] == expected_limit


def test_first_inspection_is_claimed_before_permission_and_cached_reuse_is_free():
    calls = []
    session = None

    def refuse(removed):
        assert removed in session.inspected_deletions
        calls.append(removed)
        return False

    evaluator = _SyntheticEvaluator(lambda _patch: 1)
    session = SearchSession(
        _bounded_problem(n=20, m=3, pool=("a",), deletion_permission=refuse),
        evaluator, _rank_population, SearchLimits(joint_patch_evaluations=2),
    )
    session.record_deletion_inspection(())
    assert session.deletion_permitted(()) is True
    assert not session.inspected_deletions
    assert session.deletion_discovery_remaining == 2
    for removed in ((0,), (1,)):
        session.record_deletion_inspection(removed)
        assert session.deletion_permitted(removed) is False
    assert session.deletion_discovery_remaining == 0
    for removed in ((), (0,), (1,), (0,)):
        session.record_deletion_inspection(removed)
        assert session.deletion_permitted(removed) is (not removed)
    assert calls == [(0,), (1,)]
    assert session.inspected_deletions == {(0,), (1,)}
    assert session.guard.scored_count == session.compile_guard.scored_count == 0
    # The claim boundary must reject an unseen D without converting it to a
    # permission failure or terminating already admitted patch work.
    with pytest.raises(RuntimeError, match="deletion discovery admission exceeded its limit"):
        session.record_deletion_inspection((2,))
    assert session.inspected_deletions == {(0,), (1,)}
    assert not session.stopped
    (candidate,) = session.evaluate((Patch((), (("a", 0),)),), "pure_insertion")
    assert candidate.inserted_selection == (("a", 0),)
    assert session.guard.scored_count == session.compile_guard.scored_count == 1
    assert session.deletion_discovery_remaining == 0


@SEARCHES
def test_refused_deletions_use_discovery_slots_and_leave_patch_slots_available(search):
    visits = []

    def refuse(removed):
        visits.append(removed)
        return False

    evaluator = _SyntheticEvaluator(lambda patch: len(patch.inserted_selection))
    result = search(
        _bounded_problem(pool=("a",), deletion_permission=refuse),
        evaluator, _rank_population, SearchLimits(joint_patch_evaluations=8),
    )
    discovery = result.receipt()["deletion_discovery"]
    assert len(visits) == len(set(visits)) == 8
    assert discovery["inspected_count"] == discovery["inspection_limit"] == 8
    assert discovery["remaining"] == 0
    assert discovery["subset_population"] == 162  # C(8,1)+...+C(8,4)
    assert discovery["truncated"] is True
    assert result.stop_reason == "deletion_discovery_budget"
    assert result.exhaustive_complete is False
    assert result.refused_deletion_count == 8
    assert result.guard.scored_count == len(evaluator.requested) == 2  # Stay and one addition.
    assert result.compile_guard.scored_count == 1
    assert result.guard.remaining == 6
    assert all(not c.removed_runtime_indices for c in result.candidates)


@SEARCHES
@pytest.mark.parametrize("compile_limit,stop_reason,evaluated_count", [
    (None, "joint_work_budget", 16),
    (14, "compiled_work_budget", 15),
])
def test_live_insertion_and_permitted_exchange_queues_continue_after_discovery_cap(
    search, compile_limit, stop_reason, evaluated_count,
):
    visits, evaluated_at = [], []

    def permit(removed):
        visits.append(removed)
        return removed == (0,)

    def merit(patch):
        evaluated_at.append((len(visits), patch))
        return float(len(patch.inserted_selection))

    evaluator = _SyntheticEvaluator(merit)
    result = search(
        _bounded_problem(
            pool=tuple(f"a{i}" for i in range(64)), deletion_permission=permit,
        ),
        evaluator, _rank_population,
        SearchLimits(joint_patch_evaluations=16, compiled_patches_per_checkpoint=compile_limit),
    )
    discovery = result.receipt()["deletion_discovery"]
    assert len(visits) == len(set(visits)) == 16
    assert discovery["truncated"] is True
    # Discovery exhaustion must not raise the global search-stop exception.
    # The stronger subsequent complete-patch cap remains the final stop.
    assert result.stop_reason == stop_reason
    assert result.exhaustive_complete is False
    assert result.guard.scored_count == len(evaluator.requested) == evaluated_count
    assert result.compile_guard.scored_count == evaluated_count - 1
    after_discovery = [patch for inspected, patch in evaluated_at if inspected == 16]
    assert any(not p.removed_runtime_indices and p.inserted_selection for p in after_discovery)
    assert any(p.removed_runtime_indices == (0,) and p.inserted_selection for p in after_discovery)
    assert all(not p.removed_runtime_indices or p.removed_runtime_indices == (0,)
               for p in evaluator.requested)
    assert len({candidate.order_key for candidate in result.candidates}) == len(result.candidates)


@SEARCHES
def test_refused_singletons_do_not_exclude_a_permitted_union(search):
    visits = []

    def permit(removed):
        visits.append(removed)
        return removed == (0, 1)

    result = search(
        _bounded_problem(n=14, m=2, pool=("a",), deletion_permission=permit),
        _SyntheticEvaluator(lambda patch: len(patch.inserted_selection)),
        _rank_population, SearchLimits(joint_patch_evaluations=20),
    )
    assert set(visits) == {(0,), (1,), (0, 1)}
    assert len(visits) == 3
    assert any(c.removed_runtime_indices == (0, 1) and c.inserted_selection for c in result.candidates)
    assert not any(len(c.removed_runtime_indices) == 1 for c in result.candidates)
    discovery = result.receipt()["deletion_discovery"]
    assert discovery["inspection_limit"] == discovery["inspected_count"] == 3
    assert discovery["remaining"] == 0
    assert discovery["truncated"] is False  # Equal to the entire population.
    assert result.stop_reason != "deletion_discovery_budget"
    if search is exhaustive_search:
        assert result.exhaustive_complete is True


@SEARCHES
def test_singleton_experiment_intersects_fraction_without_shrinking_the_pool(search):
    pool = tuple(f"a{i}" for i in range(9))
    visits = []

    def permit(removed):
        visits.append(removed)
        return True

    result = search(
        _bounded_problem(
            n=28, m=3, pool=pool, max_deletion_batch_size=1,
            deletion_permission=permit,
        ),
        _SyntheticEvaluator(lambda patch: len(patch.inserted_selection)),
        _rank_population, SearchLimits(joint_patch_evaluations=100),
    )
    assert set(visits) == {(0,), (1,), (2,)}
    assert all(len(c.removed_runtime_indices) <= 1 and len(c.inserted_selection) <= 1
               for c in result.candidates)
    for removed in ((), (0,), (1,), (2,)):
        assert {c.inserted_selection[0][0] for c in result.candidates
                if c.removed_runtime_indices == removed and c.inserted_selection} == set(pool)
    discovery = result.receipt()["deletion_discovery"]
    assert discovery["max_batch_size"] == 1
    assert discovery["subset_population"] == discovery["inspection_limit"] == 3
    assert discovery["truncated"] is False


@SEARCHES
def test_zero_deletion_population_preserves_empty_boundary_and_pure_insertions(search):
    def forbidden_permission(_removed):
        pytest.fail("a floor-zero deletion policy must not inspect a nonempty D")

    result = search(
        _bounded_problem(n=6, m=6, pool=("a",), deletion_permission=forbidden_permission),
        _SyntheticEvaluator(lambda patch: len(patch.inserted_selection)),
        _rank_population, SearchLimits(joint_patch_evaluations=10),
    )
    assert {c.kind for c in result.candidates} == {"stay", "insert"}
    assert result.guard.scored_count == 2
    discovery = result.receipt()["deletion_discovery"]
    assert discovery["max_batch_size"] == 0
    assert discovery["subset_population"] == discovery["inspection_limit"] == 0
    assert discovery["inspected_count"] == discovery["remaining"] == 0
    assert discovery["truncated"] is False
    assert result.stop_reason != "deletion_discovery_budget"
    if search is exhaustive_search:
        assert result.exhaustive_complete is True


@SEARCHES
def test_discovery_and_permission_caches_reset_for_each_frozen_search(search):
    visits = []

    def refuse(removed):
        visits.append(removed)
        return False

    problem = _bounded_problem(pool=("a",), deletion_permission=refuse)
    histories = []
    for _run in range(2):
        start = len(visits)
        evaluator = _SyntheticEvaluator(lambda patch: len(patch.inserted_selection))
        result = search(problem, evaluator, _rank_population, SearchLimits(joint_patch_evaluations=8))
        histories.append(set(visits[start:]))
        discovery = result.receipt()["deletion_discovery"]
        assert discovery["inspected_count"] == 8
        assert discovery["remaining"] == 0
        assert result.guard.scored_count == 2
    assert len(visits) == 16
    assert histories[0] == histories[1]


@SEARCHES
def test_zero_work_budget_does_not_inspect_or_evaluate(search):
    def forbidden_permission(_removed):
        pytest.fail("zero work allowance must not invoke permission")

    evaluator = _SyntheticEvaluator(lambda _patch: 0)
    result = search(
        _bounded_problem(pool=("a",), deletion_permission=forbidden_permission),
        evaluator, _rank_population, SearchLimits(joint_patch_evaluations=0),
    )
    assert result.stop_reason == "joint_work_budget"
    assert not result.exhaustive_complete
    assert not result.candidates and not evaluator.requested
    assert result.receipt()["deletion_discovery"]["inspected_count"] == 0
