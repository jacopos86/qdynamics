"""Execution windows preserve the scientific ordered search and paid prefix.

Recording backends expose actual admitted fan-out. Small dense geometry reaches
the real evaluator's warming hook; these tests make no Qiskit speed claim.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.patch_enumeration import Patch
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import CandidateCostInputs
from pipelines.time_dynamics.ap_mclachlan.search_methods.scheduling import (
    NEIGHBORHOOD_SETTLED, PatchRequest, schedule_rungs,
)
from pipelines.time_dynamics.ap_mclachlan.search_methods.session import (
    SearchBudgetExhausted, SearchLimits, SearchSession,
)
from test.test_ap_mclachlan_patch_evaluation import _evaluator
from test.test_ap_mclachlan_search_methods import (
    SEARCHES, _SyntheticEvaluator, _problem, _rank_population,
)


class _RecordingEvaluator(_SyntheticEvaluator):
    def __init__(self, workers=4, fail_on=None):
        super().__init__(lambda p: sum(ord(a[-1]) for a, _ in p.inserted_selection),
                         marginal=lambda p: len(p.inserted_selection) - len(p.removed_runtime_indices))
        self.workers_requested = workers
        self.batches = []
        self.fail_on = fail_on

    def evaluate(self, patches, family):
        patches = tuple(patches)
        self.batches.append(patches)
        for patch in patches:
            if patch == self.fail_on:
                raise RuntimeError("failed admitted backend task")
            yield from super().evaluate((patch,), family)


def _signature(result):
    return (result.candidates, result.moves, result.receipt(),
            result.guard.to_json_dict(), result.compile_guard.to_json_dict(),
            _rank_population(result.candidates))


@SEARCHES
def test_one_cursor_passes_multiple_admitted_patches_to_existing_evaluator(search):
    evaluator = _RecordingEvaluator()
    result = search(_problem(tuple(f"a{i}" for i in range(12)), maximum_size=1),
                    evaluator, _rank_population, SearchLimits(patch_rung_schedule="ascending"))
    assert len(result.candidates) == 13
    assert any(len(batch) > 1 for batch in evaluator.batches)
    assert max(map(len, evaluator.batches)) <= evaluator.workers_requested


@SEARCHES
@pytest.mark.parametrize("schedule", ["ascending", "interleaved"])
@pytest.mark.parametrize("deletions", ["none", "refused", "permitted"])
@pytest.mark.parametrize("budget", [7, None])
def test_windows_keep_order_discovery_moves_population_and_budget(search, schedule, deletions, budget):
    observed = []
    for workers in (1, 4):
        evaluator = _RecordingEvaluator(workers)
        permissions, ranking_versions = [], []

        def permit(removed):
            permissions.append((removed, len(evaluator.requested)))
            return deletions == "permitted"

        def rank(candidates):
            value = _rank_population(candidates)
            ranking_versions.append((tuple(c.order_key for c in candidates), value))
            return value

        problem = _problem(tuple(f"a{i}" for i in range(5)), maximum_size=2,
                           coordinate_keys=("s0", "s1"),
                           deletable_indices=() if deletions == "none" else (0, 1),
                           deletion_permission=permit)
        result = search(problem, evaluator, rank, SearchLimits(
            joint_patch_evaluations=budget, patch_rung_schedule=schedule,
        ))
        observed.append((_signature(result), evaluator.requested, permissions, ranking_versions))
    assert observed[0] == observed[1]


@pytest.mark.parametrize("compiled_limit", [False, True])
def test_refusal_occurs_after_exact_fitting_prefix_is_stored(compiled_limit):
    evaluator = _RecordingEvaluator()
    session = SearchSession(_problem(tuple("abcd")), evaluator, _rank_population,
                            SearchLimits(**({"compiled_patches_per_checkpoint": 2}
                                            if compiled_limit else {"joint_patch_evaluations": 3})))
    patches = [Patch((), ()), *(Patch((), ((a, 0),)) for a in "abcd")]
    consumed = []

    def requested():
        for patch in patches:
            consumed.append(patch)
            yield patch

    reason = "compiled_work_budget" if compiled_limit else "joint_work_budget"
    with pytest.raises(SearchBudgetExhausted, match=reason):
        session.evaluate(requested(), "window", execution_window=4)
    assert evaluator.requested == patches[:3]
    assert consumed == patches[:4]  # The refused next patch, no look-ahead past it.
    assert len(session.evidence) == session.guard.scored_count == 3
    assert session.compile_guard.scored_count == 2
    assert session.refusal["label"] == "window"
    assert session.guard.admitted_families == ["window"] * 3


def test_cached_and_repeated_members_use_no_additional_slots():
    evaluator = _RecordingEvaluator()
    session = SearchSession(_problem(tuple("abcd")), evaluator, _rank_population,
                            SearchLimits(joint_patch_evaluations=3))
    a, b, c, d = (Patch((), ((word, 0),)) for word in "abcd")
    session.evaluate((a,), "cached")
    with pytest.raises(SearchBudgetExhausted):
        session.evaluate((a, a, b, c, d), "window", execution_window=4)
    assert evaluator.requested == [a, b, c]
    assert session.guard.admitted_families == ["cached", "window", "window"]
    assert len(session.evidence) == 3


def test_backend_failure_keeps_completed_prefix_and_all_admitted_work_charged():
    a, b, c = (Patch((), ((word, 0),)) for word in "abc")
    evaluator = _RecordingEvaluator(fail_on=b)
    session = SearchSession(_problem(tuple("abc")), evaluator, _rank_population, SearchLimits())
    with pytest.raises(RuntimeError, match="failed admitted backend task"):
        session.evaluate((a, b, c), "window", execution_window=4)
    assert evaluator.batches == [(a, b, c)]
    assert evaluator.requested == [a]
    assert len(session.evidence) == 1
    assert session.guard.scored_count == session.compile_guard.scored_count == 3
    assert not session.stopped  # Backend failure is not a work-budget refusal.


def test_settled_branch_wakes_after_same_single_patch_before_higher_work():
    results = []
    for workers in (1, 4):
        evaluator = _RecordingEvaluator(workers)
        problem = _problem(tuple("abcdef"), coordinate_keys=("s",), deletable_indices=(0,))
        session = SearchSession(problem, evaluator, _rank_population,
                                SearchLimits(patch_rung_schedule="ascending"))
        resumed = []

        def branch(removed):
            if not removed:
                yield PatchRequest("initial", {1: (Patch((), (("a", 0),)),)})
                yield NEIGHBORHOOD_SETTLED
                resumed.append(len(evaluator.requested))
                yield PatchRequest("after_wake", {1: (Patch((), (("b", 0),)),)})
            else:
                yield PatchRequest("future", {2: (
                    Patch(removed, (("a", 0), (word, 0))) for word in "bcdef"
                )})

        schedule_rungs(session, problem.deletion_rungs(), branch)
        results.append((evaluator.requested, resumed, tuple(session.evidence), session.moves))
    assert results[0] == results[1]
    assert results[1][0][2] == Patch((), (("b", 0),))


@SEARCHES
@pytest.mark.parametrize("mode", ["joint_batch", "singleton_factorized"])
@pytest.mark.parametrize("maximum_size", [1, 2])
def test_real_dense_evaluator_receives_parallel_cost_window_without_science_change(search, mode, maximum_size):
    results, warms = [], []
    positions = tuple((word, 0) for word in "abcdef")
    active = np.eye(7)[:, :1]
    inserted = np.eye(7)[:, 1:]
    target = np.arange(1., 8.)
    for workers in (1, 4):
        warmed = []
        evaluator = _evaluator(
            active, inserted, target, positions, scoring_workers=workers,
            gain_evaluation=mode, gamma_rcond=1e-10,
            route_bundle=SimpleNamespace(uses_population_scoring=True),
            population_accuracy_debt=True,
            candidate_cost_inputs=lambda _d, selection, _p: CandidateCostInputs(
                cost_marginal=(len(selection),) * 3, compile_calls=1),
            candidate_cost_warm=lambda requests: warmed.append(tuple(requests)),
        )
        result = search(_problem(tuple("abcdef"), maximum_size=maximum_size), evaluator,
                        _rank_population, SearchLimits(patch_rung_schedule="ascending"))
        results.append(_signature(result))
        warms.append(warmed)
    assert results[0] == results[1]
    assert any(len(items) > 1 for items in warms[1])


@pytest.mark.parametrize("earlier_backend_failure", [False, True])
def test_later_enumeration_error_preserves_predecessors_and_first_error(earlier_backend_failure):
    a, b = (Patch((), ((word, 0),)) for word in "ab")
    evaluator = _RecordingEvaluator(fail_on=b if earlier_backend_failure else None)
    session = SearchSession(_problem(tuple("ab")), evaluator, _rank_population, SearchLimits())

    def patches():
        yield a
        yield b
        raise ValueError("later malformed realization")

    error = RuntimeError if earlier_backend_failure else ValueError
    message = "failed admitted backend task" if earlier_backend_failure else "later malformed realization"
    with pytest.raises(error, match=message):
        session.evaluate(patches(), "window", execution_window=4)
    assert evaluator.requested == ([a] if earlier_backend_failure else [a, b])
    assert len(session.evidence) == (1 if earlier_backend_failure else 2)
    assert session.guard.scored_count == session.compile_guard.scored_count == 2
