"""Ascending tier priority through shared scheduling and the ordinary runner.

Synthetic complete-patch evaluations expose coverage and logical-request
boundaries; the small route tests exercise the real compiled ranker separately.
"""

from itertools import combinations, product

import pytest

from pipelines.time_dynamics.ap_mclachlan.patch_enumeration import Patch
from pipelines.time_dynamics.ap_mclachlan.search_methods.scheduling import (
    NEIGHBORHOOD_SETTLED, PatchRequest, SCHEDULE_ID, SCHEDULE_IDS, schedule_rungs,
)
from pipelines.time_dynamics.ap_mclachlan.search_methods.session import (
    SearchBudgetExhausted, SearchLimits, SearchSession,
)
from test.test_ap_mclachlan_search_methods import (
    SEARCHES, _SyntheticEvaluator, _problem, _rank_population, exhaustive_search,
)


def _tier(patch):
    return max(1, len(patch.removed_runtime_indices), len(patch.inserted_selection))


@SEARCHES
def test_first_tier_finishes_singletons_and_exchanges_before_larger_patches(search):
    evaluated, inspected = [], []
    atoms = ("a", "b", "c")

    def merit(patch):
        evaluated.append(patch)
        return float(len(patch.inserted_selection))

    def permit(removed):
        inspected.append((removed, len(evaluated)))
        return True

    result = search(
        _problem(atoms, maximum_size=3, coordinate_keys=("s0", "s1", "s2"),
                 deletable_indices=(0, 1, 2), deletion_permission=permit),
        _SyntheticEvaluator(merit), _rank_population,
        SearchLimits(patch_rung_schedule="ascending"),
    )
    expected = set(product(((), (0,), (1,), (2,)), ((), ("a",), ("b",), ("c",))))
    first_larger = next(i for i, patch in enumerate(evaluated) if _tier(patch) > 1)
    initial = evaluated[:first_larger]
    assert {(p.removed_runtime_indices, tuple(a for a, _cut in p.inserted_selection))
            for p in initial} == expected
    assert len(initial) == len(expected) == 16
    assert all(count >= 16 for removed, count in inspected if len(removed) > 1)
    assert len({p.order_key for p in result.candidates}) == len(result.candidates)


@SEARCHES
def test_small_budget_retains_complete_first_tier_prefix_including_exchange(search):
    evaluator = _SyntheticEvaluator(lambda patch: len(patch.inserted_selection))
    result = search(
        _problem(("a", "b", "c"), maximum_size=3, coordinate_keys=("s0", "s1", "s2"),
                 deletable_indices=(0, 1, 2)), evaluator, _rank_population,
        SearchLimits(joint_patch_evaluations=12, patch_rung_schedule="ascending"),
    )
    assert result.stop_reason == "joint_work_budget"
    assert len(evaluator.requested) == len(result.candidates) == 12
    assert result.guard.scored_count == 12
    assert all(_tier(patch) == 1 for patch in evaluator.requested)
    assert any(p.removed_runtime_indices and p.inserted_selection for p in evaluator.requested)
    assert not result.exhaustive_complete


@pytest.mark.parametrize("cached_future", [False, True])
def test_exhausted_budget_does_not_leave_discovery_masking_future_cursor(monkeypatch, cached_future):
    evaluator = _SyntheticEvaluator(lambda patch: len(patch.inserted_selection))
    problem = _problem(("a", "b"), maximum_size=2,
                       coordinate_keys=("s0", "s1"), deletable_indices=(0, 1))
    session = SearchSession(problem, evaluator, _rank_population, SearchLimits(
        joint_patch_evaluations=1 + int(cached_future), patch_rung_schedule="ascending",
    ))
    first, future = Patch((), (("a", 0),)), Patch((), (("a", 0), ("b", 0)))
    if cached_future:
        session.evaluate((future,), "preexisting_evidence")

    # Bound this termination regression so a dormant discovery tier fails
    # promptly instead of hanging the suite if it masks the future cursor.
    original_remaining = type(session.guard).remaining.fget
    reads = 0

    def bounded_remaining(guard):
        nonlocal reads
        if guard is session.guard:
            reads += 1
            assert reads <= 100, "scheduler stalled behind non-runnable discovery"
        return original_remaining(guard)

    monkeypatch.setattr(type(session.guard), "remaining", property(bounded_remaining))

    def branch(removed):
        assert not removed
        yield PatchRequest("spans_tiers", {1: (first,), 2: (future,)})

    with pytest.raises(SearchBudgetExhausted, match="joint_work_budget"):
        schedule_rungs(session, problem.deletion_rungs(), branch)
    assert session.guard.remaining == 0
    assert not session.inspected_deletions
    assert len(session.evidence) == 1 + int(cached_future)
    assert evaluator.requested == ([future, first] if cached_future else [first])
    assert session.refusal["label"] == ("deletion_discovery" if cached_future else "spans_tiers")


def test_uncapped_exhaustive_tiers_cover_same_population_and_final_ranking():
    results, streams = [], []
    for mode in ("interleaved", "ascending"):
        evaluator = _SyntheticEvaluator(
            lambda p: len(p.inserted_selection) + .1 * len(p.removed_runtime_indices),
            marginal=lambda p: len(p.inserted_selection) - len(p.removed_runtime_indices),
        )
        result = exhaustive_search(
            _problem(("a", "b", "c"), maximum_size=3,
                     coordinate_keys=("s0", "s1", "s2"), deletable_indices=(0, 1, 2)),
            evaluator, _rank_population, SearchLimits(patch_rung_schedule=mode),
        )
        assert result.exhaustive_complete
        assert len(result.candidates) == 64
        results.append(result)
        streams.append(evaluator.requested)
    assert {c.order_key for c in results[0].candidates} == {c.order_key for c in results[1].candidates}
    ranks = [_rank_population(result.candidates) for result in results]
    assert [c.order_key for c in ranks[0].ordered] == [c.order_key for c in ranks[1].ordered]
    assert ranks[0].rank_info == ranks[1].rank_info
    assert [_tier(p) for p in streams[1]] == sorted(_tier(p) for p in streams[1])
    assert [_tier(p) for p in streams[0]] != [_tier(p) for p in streams[1]]


def test_request_can_span_future_tier_then_expose_lower_tier_without_partial_move():
    evaluator = _SyntheticEvaluator(lambda patch: len(patch.inserted_selection))
    problem = _problem(("a", "b", "c"), maximum_size=2,
                       coordinate_keys=("s0",), deletable_indices=(0,))
    session = SearchSession(problem, evaluator, _rank_population, SearchLimits(patch_rung_schedule="ascending"))
    resumed = []

    def branch(removed):
        if not removed:
            completed = yield PatchRequest("first", {
                1: iter((Patch((), (("a", 0),)),)),
                2: iter((Patch((), (("a", 0), ("b", 0))),)),
            })
            assert {len(c.inserted_selection) for c in completed} == {1, 2}
            resumed.append(len(evaluator.requested))
            yield PatchRequest("smaller_after_move", {1: iter((Patch((), (("c", 0),)),))})
        else:
            yield PatchRequest("other_future_members", {2: (
                Patch(removed, tuple((a, 0) for a in block))
                for block in combinations(("a", "b", "c"), 2)
            )})

    schedule_rungs(session, problem.deletion_rungs(), branch)
    assert len(resumed) == 1
    assert evaluator.requested[resumed[0]] == Patch((), (("c", 0),))
    assert len(evaluator.requested) == 6
    assert len(session.evidence) == 6


def test_new_evidence_wakes_settled_branch_and_smaller_work_regains_priority():
    evaluator = _SyntheticEvaluator(lambda patch: len(patch.inserted_selection))
    problem = _problem(("a", "b", "c"), maximum_size=2,
                       coordinate_keys=("s0",), deletable_indices=(0,))
    session = SearchSession(problem, evaluator, _rank_population, SearchLimits(patch_rung_schedule="ascending"))
    woke = []

    def branch(removed):
        if not removed:
            yield PatchRequest("initial", {1: iter((Patch((), (("a", 0),)),))})
            yield NEIGHBORHOOD_SETTLED
            woke.append(len(evaluator.requested))
            yield PatchRequest("smaller_after_rerank", {1: iter((Patch((), (("b", 0),)),))})
        else:
            yield PatchRequest("future", {2: (
                Patch(removed, tuple((a, 0) for a in block))
                for block in combinations(("a", "b", "c"), 2)
            )})

    schedule_rungs(session, problem.deletion_rungs(), branch)
    assert len(woke) == 1
    assert evaluator.requested[woke[0]] == Patch((), (("b", 0),))
    assert len(session.evidence) == len(evaluator.requested) == 5


@SEARCHES
def test_discovery_priority_and_its_independent_cap_preserve_pure_work(search):
    inspected = []

    def refuse(removed):
        inspected.append(removed)
        return False

    result = search(
        _problem(("a",), maximum_size=1, coordinate_keys=tuple(f"s{i}" for i in range(28)),
                 deletable_indices=tuple(range(8)), max_deletion_batch_fraction=.15,
                 deletion_permission=refuse),
        _SyntheticEvaluator(lambda p: len(p.inserted_selection)), _rank_population,
        SearchLimits(joint_patch_evaluations=4, patch_rung_schedule="ascending"),
    )
    assert len(inspected) == 4
    assert all(len(removed) == 1 for removed in inspected)
    assert result.stop_reason == "deletion_discovery_budget"
    assert result.guard.scored_count == 2
    assert {c.kind for c in result.candidates} == {"stay", "insert"}
    assert result.receipt()["deletion_discovery"]["truncated"]


@pytest.mark.parametrize("bad", ["", "unknown", None, 1])
def test_invalid_schedule_rejected_by_typed_search_limits(bad):
    with pytest.raises(ValueError, match="patch_rung_schedule"):
        SearchLimits(patch_rung_schedule=bad)


def test_ordinary_runner_rejects_ascending_on_legacy_route_and_preserves_default():
    from pipelines.time_dynamics.runners import ap_append_from_adapt_artifact as runner

    parser = runner._build_parser()
    args = parser.parse_args(["--artifact-json", "seed.json", "--output-json", "run.json"])
    config = runner._support_patch_config_from_args(args)
    assert config.patch_rung_schedule == "interleaved"
    assert "patch_rung_schedule" not in config.to_json_dict()
    assert "patch_rung_schedule" not in config.to_json_dict()["generalized_exchange"]["search"]
    args.patch_rung_schedule = "ascending"
    with pytest.raises(ValueError, match="requires an experimental search"):
        runner._support_patch_config_from_args(args)


@pytest.mark.parametrize("policy", ["greedy", "block_exchange", "bounded_exhaustive"])
@pytest.mark.parametrize("gain", ["joint_batch", "singleton_factorized"])
def test_campaign_extra_flag_reaches_live_compiled_search_and_receipts(policy, gain):
    from pipelines.time_dynamics.paper_ii_runs import build_selector_comparison_run, DIAGNOSTIC_FAST_SEARCH
    from pipelines.time_dynamics.runners import ap_append_from_adapt_artifact as runner
    from test.test_ap_mclachlan_exchange_integration import _run

    run = build_selector_comparison_run(
        search_policy=policy, gain_evaluation=gain, insertion_cardinality=2, deletion_cardinality=1,
        seed_path="seed.json", drive="fastweak", horizon="t2", output_json="run.json",
        search_profile=DIAGNOSTIC_FAST_SEARCH.profile_id,
        extra_flags=("--patch-rung-schedule=ascending", "--compile-workers", "1",
                     "--max-compiled-patches-per-checkpoint", "23"),
    )
    config = runner._support_patch_config_from_args(runner._build_parser().parse_args(list(run.argv())))
    assert config.patch_rung_schedule == "ascending"
    assert config.to_json_dict()["generalized_exchange"]["search"]["patch_rung_schedule"] == "ascending"
    selection, payload = _run(config)
    levels = payload["selector_route"]["levels"]
    assert levels
    for level in levels:
        search = level["search"]
        assert search["patch_rung_schedule"] == "ascending"
        assert search["rung_schedule"] == SCHEDULE_IDS["ascending"]
        assert search["rung_schedule"] != SCHEDULE_ID
        assert search["admission_policy"] == "incremental_complete_patch_v1"
    for attempt in payload["attempts"]:
        assert "route_score" not in attempt
    if selection.committed is not None:
        assert payload["committed"]["score"] == selection.attempts[-1].route_score
