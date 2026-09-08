"""Search traversal against independent objectives and shared population ranking.

The synthetic evaluator supplies complete-patch evidence only. These tests do
not validate a physical gain backend or infer estimator/compilation savings.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.exchange_selector import RankedLevel, RankInfo
from pipelines.time_dynamics.ap_mclachlan.patch_enumeration import Patch
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import StructuralCandidate
from pipelines.time_dynamics.ap_mclachlan.population_scoring import (
    CompiledResourceMarginal,
    PopulationCostNormalization,
    RankRegime,
    ScoreRecord,
    score_and_rank,
)
from pipelines.time_dynamics.ap_mclachlan.search_methods.conditional_greedy import (
    search_patches as greedy_search,
)
from pipelines.time_dynamics.ap_mclachlan.search_methods.block_local_exchange import (
    insertion_neighborhood,
    search_patches as block_search,
)
from pipelines.time_dynamics.ap_mclachlan.search_methods.exhaustive import (
    search_patches as exhaustive_search,
)
from pipelines.time_dynamics.ap_mclachlan.search_methods.problem import PatchProblem
from pipelines.time_dynamics.ap_mclachlan.search_methods.session import (
    SearchBudgetExhausted,
    SearchLimits,
    SearchSession,
)


SEARCHES = pytest.mark.parametrize(
    "search", (exhaustive_search, greedy_search, block_search),
    ids=("exhaustive", "greedy", "block"),
)


def _block(candidate) -> frozenset[str]:
    return frozenset(atom for atom, _cut in candidate.inserted_selection)


class _SyntheticEvaluator:
    """An independent evidence supplier for testing search orchestration."""

    q_base = 0.0

    def __init__(self, merit, marginal=lambda _patch: 0):
        self.merit = merit
        self.marginal = marginal
        self.requested = []

    @staticmethod
    def active_population_face(_removed, _selection):
        return True

    def evaluate(self, patches, family):
        for patch in patches:
            self.requested.append(patch)
            merit = float(self.merit(patch))
            marginal = int(self.marginal(patch))
            yield StructuralCandidate(
                removed_runtime_indices=patch.removed_runtime_indices,
                inserted_selection=patch.inserted_selection,
                plan=patch.plan,
                family=family,
                q=merit,
                insertion_gain=0.0,
                deletion_loss=0.0,
                delta=merit,
                insertion_utility=0.0,
                deletion_utility=0.0,
                score=merit,
                m_gamma=merit,
                cost_marginal=(marginal, marginal, marginal),
            )


def _rank_population(candidates):
    """Use the production normalization/score with synthetic raw evidence."""

    candidates = tuple(c for c in candidates if c.kind != "stay")
    if not candidates:
        return RankedLevel(ordered=(), rank_info={}, telemetry={})
    by_key = {c.order_key: c for c in candidates}
    sealed = PopulationCostNormalization().seal([
        (c.order_key, CompiledResourceMarginal(*c.cost_marginal)) for c in candidates
    ])
    ranked = score_and_rank([
        ScoreRecord(
            patch_id=c.order_key,
            merit=c.delta,
            K=sealed.k_of(c.order_key),
            raw_merit_tiebreak=c.delta,
            canonical_key=c.order_key,
        )
        for c in candidates
    ], RankRegime.LARGE_MISS)
    return RankedLevel(
        ordered=tuple(by_key[row.patch_id] for row in ranked.ranked),
        rank_info={
            row.patch_id: RankInfo(K=row.K, score=row.score, merit=row.merit)
            for row in ranked.ranked
        },
        telemetry=sealed.to_json_dict(),
    )


def _problem(pool, *, maximum_size=2, **overrides):
    fields = dict(
        coordinate_keys=(),
        deletable_indices=(),
        min_surviving_support=0,
        candidate_pool_for_deletion=lambda _removed: tuple(pool),
        cuts_by_atom={atom: (0,) for atom in pool},
        max_insertion_batch_size=maximum_size,
        tokens_commute=lambda _left, _right: True,
    )
    fields.update(overrides)
    return PatchProblem(**fields)


def test_dense_projection_distinguishes_exhaustive_from_greedy_coverage() -> None:
    columns = {
        "a": np.array([1.0, 0.1, 0.0]),
        "b": np.array([1.0, -0.1, 0.0]),
        "c": np.array([0.0, 1.0, 1.0]),
    }
    target = np.array([0.0, 1.0, 0.0])

    def direct_projection(patch):
        if not patch.inserted_selection:
            return 0.0
        matrix = np.column_stack([columns[atom] for atom, _cut in patch.inserted_selection])
        coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = target - matrix @ coefficients
        return float(target @ target - residual @ residual)

    exhaustive = exhaustive_search(
        _problem(("a", "b", "c")), _SyntheticEvaluator(direct_projection),
        _rank_population, SearchLimits(),
    )
    greedy = greedy_search(
        _problem(("a", "b", "c")), _SyntheticEvaluator(direct_projection),
        _rank_population, SearchLimits(),
    )
    exhaustive_best = _rank_population(exhaustive.candidates).ordered[0]
    assert _block(exhaustive_best) == {"a", "b"}
    assert exhaustive_best.delta == pytest.approx(1.0)
    assert exhaustive.exhaustive_complete
    assert exhaustive.stop_reason == "universe_exhausted"
    assert {_block(c) for c in exhaustive.candidates} == {
        frozenset(), frozenset("a"), frozenset("b"), frozenset("c"),
        frozenset(("a", "b")), frozenset(("a", "c")), frozenset(("b", "c")),
    }
    # c wins the first greedy step. Both complete extensions are still paid
    # for and retained, even though only one becomes its next incumbent.
    assert greedy.moves[0]["inserted_block"] == ["c"]
    assert {_block(c) for c in greedy.candidates if len(c.inserted_selection) == 2} == {
        frozenset(("a", "c")), frozenset(("b", "c")),
    }
    assert _rank_population(greedy.candidates).ordered[0].delta < 0.51
    assert not greedy.exhaustive_complete


def test_exhaustive_keeps_joint_blocks_despite_ineligible_singletons() -> None:
    values = {frozenset(): 0.0, frozenset("a"): -1.0,
              frozenset("b"): -2.0, frozenset(("a", "b")): 3.0}
    result = exhaustive_search(
        _problem(("a", "b")), _SyntheticEvaluator(lambda patch: values[_block(patch)]),
        _rank_population, SearchLimits(),
    )
    assert _block(_rank_population(result.candidates).ordered[0]) == {"a", "b"}
    assert result.exhaustive_complete


@SEARCHES
def test_search_keeps_complete_noncommuting_joint_placement_plans(search) -> None:
    result = search(
        _problem(
            ("a", "b"), coordinate_keys=("s",),
            cuts_by_atom={"a": (0, 1), "b": (0, 1)},
            tokens_commute=lambda _left, _right: False,
        ),
        _SyntheticEvaluator(lambda patch: len(patch.inserted_selection)),
        _rank_population, SearchLimits(),
    )
    words = {
        tuple(token.key for token in candidate.plan.full_word)
        for candidate in result.candidates if candidate.plan is not None
    }
    assert words == {
        ("a", "b", "s"), ("a", "s", "b"), ("b", "a", "s"),
        ("b", "s", "a"), ("s", "a", "b"), ("s", "b", "a"),
    }


@SEARCHES
def test_branch_only_candidates_are_searched_and_refused_pools_are_not_read(search) -> None:
    pool_calls = []

    def pool(removed):
        pool_calls.append(removed)
        assert removed != (1,), "permission must precede a refused branch's pool"
        return ("a",) if not removed else ("b", "c")

    result = search(
        _problem(
            ("a", "b", "c"), coordinate_keys=("s0", "s1"),
            deletable_indices=(0, 1), min_surviving_support=1,
            candidate_pool_for_deletion=pool,
            deletion_permission=lambda removed: removed != (1,),
        ),
        _SyntheticEvaluator(lambda patch: len(patch.inserted_selection)),
        _rank_population, SearchLimits(),
    )
    assert pool_calls == [(), (0,)]
    assert any(c.removed_runtime_indices == (0,) and _block(c) == {"b", "c"}
               for c in result.candidates)
    assert not any(c.removed_runtime_indices == (1,) for c in result.candidates)
    assert result.permitted_deletion_count == 2
    assert result.refused_deletion_count == 1


@SEARCHES
@pytest.mark.parametrize("limits, reason", [
    (SearchLimits(joint_patch_evaluations=2), "joint_work_budget"),
    (SearchLimits(compiled_patches_per_checkpoint=1), "compiled_work_budget"),
])
def test_budget_refuses_whole_insertion_batch_before_evaluation(search, limits, reason) -> None:
    evaluator = _SyntheticEvaluator(lambda patch: len(patch.inserted_selection))
    result = search(_problem(("a", "b")), evaluator, _rank_population, limits)
    assert result.stop_reason == reason
    assert not result.exhaustive_complete
    assert len(evaluator.requested) == 1
    assert evaluator.requested[0].inserted_selection == ()
    assert result.guard.scored_count == 1


@SEARCHES
def test_refused_deletion_boundaries_do_not_consume_evaluation_slots(search) -> None:
    permissions = []

    def refuse(removed):
        permissions.append(removed)
        return False

    evaluator = _SyntheticEvaluator(lambda _patch: 0.0)
    result = search(
        _problem(
            (), coordinate_keys=("s0", "s1", "s2"), deletable_indices=(0, 1, 2),
            deletion_permission=refuse,
        ),
        evaluator, _rank_population, SearchLimits(joint_patch_evaluations=2),
    )
    assert result.stop_reason != "joint_work_budget"
    assert permissions == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    assert len(evaluator.requested) == 1
    assert result.guard.scored_count == 1
    assert result.guard.rejected_family is None
    assert result.refused_deletion_count == 7


def test_greedy_recompares_incumbent_under_the_expanded_population() -> None:
    merits = {
        frozenset(): 0.0, frozenset("a"): 10.0, frozenset("b"): 1.0,
        frozenset("c"): 0.5, frozenset(("a", "b")): 8.0,
        frozenset(("a", "c")): 0.1, frozenset(("a", "b", "c")): 9.0,
    }
    marginals = {
        frozenset(): 0, frozenset("a"): 0, frozenset("b"): 2,
        frozenset("c"): 4, frozenset(("a", "b")): -100,
        frozenset(("a", "c")): -90, frozenset(("a", "b", "c")): -200,
    }
    snapshots = []

    def rank(candidates):
        snapshot = _rank_population(candidates)
        snapshots.append({
            _block(candidate): snapshot.rank_info[candidate.order_key].score
            for candidate in snapshot.ordered
        })
        return snapshot

    result = greedy_search(
        _problem(("a", "b", "c"), maximum_size=3),
        _SyntheticEvaluator(
            lambda patch: merits[_block(patch)],
            lambda patch: marginals[_block(patch)],
        ),
        rank, SearchLimits(),
    )
    assert snapshots[0][frozenset("a")] == pytest.approx(12.5)
    assert snapshots[1][frozenset("a")] == pytest.approx(10.0)
    assert 10.0 < snapshots[1][frozenset(("a", "b"))] < 12.5
    assert [move["inserted_block"] for move in result.moves] == [
        ["a"], ["a", "b"], ["a", "b", "c"],
    ]
    assert any(_block(candidate) == {"a", "b", "c"} for candidate in result.candidates)


def test_greedy_does_not_traverse_an_ineligible_intermediate() -> None:
    evaluator = _SyntheticEvaluator(lambda patch: -float(len(patch.inserted_selection)))
    result = greedy_search(_problem(("a", "b")), evaluator, _rank_population, SearchLimits())
    assert not result.moves
    assert all(len(patch.inserted_selection) <= 1 for patch in evaluator.requested)
    assert not result.exhaustive_complete


def test_block_neighborhood_matches_independent_subset_definition() -> None:
    pool = ("d", "b", "a", "c")
    for selected_mask in range(1 << len(pool)):
        if selected_mask.bit_count() > 3:
            continue
        selected = tuple(atom for index, atom in enumerate(pool) if selected_mask & (1 << index))
        for radius in (1, 2, 3, 4):
            observed = tuple(insertion_neighborhood(pool, selected, maximum_size=3, move_size=radius))
            expected = {
                tuple(atom for index, atom in enumerate(pool) if mask & (1 << index))
                for mask in range(1 << len(pool))
                if mask.bit_count() <= 3 and 1 <= (mask ^ selected_mask).bit_count() <= radius
            }
            assert set(observed) == expected
            assert len(observed) == len(set(observed))
    assert tuple(insertion_neighborhood(("c", "a", "b"), ("a",), maximum_size=2)) == (
        ("c", "a"), ("a", "b"), (), ("c",), ("b",),
    )


def test_pair_radius_counts_total_additions_and_removals() -> None:
    pair_moves = set(insertion_neighborhood(("a", "b", "c", "d"), ("a", "b"), maximum_size=2))
    assert ("a", "c") in pair_moves       # one out, one in
    assert () in pair_moves                 # two out
    assert ("c", "d") not in pair_moves  # two out, two in requires four
    assert ("c", "d") in set(insertion_neighborhood(
        ("a", "b", "c", "d"), ("a", "b"), maximum_size=2, move_size=4,
    ))
    assert ("a", "b") in set(insertion_neighborhood(("a", "b"), (), maximum_size=2))
    with pytest.raises(ValueError, match="positive integer"):
        tuple(insertion_neighborhood(("a",), (), maximum_size=1, move_size=0))


def test_block_pair_can_escape_the_greedy_ineligible_singleton_stop() -> None:
    values = {frozenset(): 0.0, frozenset("a"): -1.0,
              frozenset("b"): -1.0, frozenset(("a", "b")): 4.0}
    problem = _problem(("a", "b"))
    greedy = greedy_search(problem, _SyntheticEvaluator(lambda p: values[_block(p)]), _rank_population, SearchLimits())
    block = block_search(problem, _SyntheticEvaluator(lambda p: values[_block(p)]), _rank_population, SearchLimits())
    assert not _rank_population(greedy.candidates).ordered
    assert _block(_rank_population(block.candidates).ordered[0]) == {"a", "b"}
    assert block.moves[0]["move_size"] == 2
    assert not block.exhaustive_complete
    assert block.stop_reason == "block_neighborhoods_exhausted"


def test_block_search_improves_first_branch_before_acquiring_all_boundaries() -> None:
    evaluator = _SyntheticEvaluator(lambda patch: len(patch.inserted_selection))
    result = block_search(
        _problem(("a", "b"), coordinate_keys=("s0", "s1", "s2"), deletable_indices=(0, 1, 2)),
        evaluator, _rank_population, SearchLimits(joint_patch_evaluations=4),
    )
    assert result.stop_reason == "joint_work_budget"
    assert len(evaluator.requested) == 4
    assert any(_block(candidate) == {"a", "b"} for candidate in result.candidates)
    assert all(not candidate.removed_runtime_indices for candidate in result.candidates)


def _adversarial_rank(candidates, favorite):
    """A deterministic population-dependent order for traversal stress tests."""

    candidates = tuple(c for c in candidates if c.inserted_selection)
    scores = {c.order_key: (10.0 if _block(c) == favorite else 1.0) for c in candidates}
    ordered = tuple(sorted(candidates, key=lambda c: (-scores[c.order_key], c.order_key)))
    return RankedLevel(
        ordered=ordered,
        rank_info={c.order_key: RankInfo(K=1.0, score=scores[c.order_key], merit=c.delta) for c in ordered},
        telemetry={},
    )


def test_block_revisits_an_incumbent_after_population_growth_and_terminates() -> None:
    populations = []

    def rank(candidates):
        blocks = {_block(candidate) for candidate in candidates}
        populations.append(len(candidates))
        assert len(populations) <= 3, "unchanged populations must not cause ranking cycles"
        # First a wins. Its neighborhood reveals abc/acd/abd and favors b.
        # Visiting b reveals bcd, making a preferable again under the new P.
        favorite = frozenset("a")
        if any(len(block) == 3 for block in blocks) and frozenset(("b", "c", "d")) not in blocks:
            favorite = frozenset("b")
        return _adversarial_rank(candidates, favorite)

    evaluator = _SyntheticEvaluator(lambda patch: len(patch.inserted_selection))
    result = block_search(_problem(("a", "b", "c", "d"), maximum_size=3), evaluator, rank, SearchLimits())
    assert [move["inserted_block"] for move in result.moves] == [["a"], ["b"], ["a"]]
    assert populations == [11, 14, 15]
    assert len(evaluator.requested) == len(result.candidates) == 15
    assert result.stop_reason == "block_neighborhoods_exhausted"


def test_block_rechecks_an_earlier_branch_after_later_population_growth() -> None:
    def rank(candidates):
        favorite = frozenset("b") if any(c.removed_runtime_indices for c in candidates) else frozenset("a")
        return _adversarial_rank(candidates, favorite)

    result = block_search(
        _problem(
            ("a", "b"), maximum_size=1, coordinate_keys=("s",), deletable_indices=(0,),
            candidate_pool_for_deletion=lambda removed: () if removed else ("a", "b"),
        ),
        _SyntheticEvaluator(lambda patch: len(patch.inserted_selection)), rank, SearchLimits(),
    )
    assert [move["inserted_block"] for move in result.moves] == [["a"], ["b"]]
    assert all(move["removed"] == [] for move in result.moves)
    assert result.stop_reason == "block_neighborhoods_exhausted"


def test_true_permission_is_not_a_completed_evaluation_after_compile_budget_failure() -> None:
    permissions = []

    def permit(removed):
        permissions.append(removed)
        return True

    evaluator = _SyntheticEvaluator(lambda _patch: 0.0)
    session = SearchSession(
        _problem((), coordinate_keys=("s",), deletable_indices=(0,), deletion_permission=permit),
        evaluator, _rank_population, SearchLimits(compiled_patches_per_checkpoint=0),
    )
    session.check_boundary_budget((0,))
    assert session.deletion_permitted((0,)) is True
    for _attempt in range(2):
        assert session.deletion_permitted((0,)) is True
        with pytest.raises(SearchBudgetExhausted) as stopped:
            session.evaluate((Patch((0,), ()),), label="permitted_boundary")
        assert stopped.value.reason == "compiled_work_budget"
        assert not session.evidence
    assert permissions == [(0,)]
    assert not evaluator.requested
    assert session.guard.scored_count == 0


def test_completed_boundary_reuses_evidence_at_zero_remaining_budget() -> None:
    evaluator = _SyntheticEvaluator(lambda _patch: 0.0)
    session = SearchSession(_problem(()), evaluator, _rank_population, SearchLimits(joint_patch_evaluations=1))
    session.check_boundary_budget(())
    assert session.deletion_permitted(()) is True
    first = session.evaluate((Patch((), ()),), label="stay")
    assert session.guard.remaining == 0
    session.check_boundary_budget(())
    again = session.evaluate((Patch((), ()),), label="repeated_stay")
    assert again[0] is first[0]
    assert len(evaluator.requested) == len(session.evidence) == 1
    assert session.guard.scored_count == 1


def test_repeated_refused_permission_is_cached_without_evaluation_charge() -> None:
    permissions = []

    def refuse(removed):
        permissions.append(removed)
        return False

    evaluator = _SyntheticEvaluator(lambda _patch: 0.0)
    session = SearchSession(
        _problem((), coordinate_keys=("s",), deletable_indices=(0,), deletion_permission=refuse),
        evaluator, _rank_population, SearchLimits(joint_patch_evaluations=1),
    )
    for _attempt in range(2):
        session.check_boundary_budget((0,))
        assert session.deletion_permitted((0,)) is False
    assert permissions == [(0,)]
    assert session.guard.scored_count == session.compile_guard.scored_count == 0
    assert session.finish("audit_complete").refused_deletion_count == 1
    assert not session.evidence
    assert not evaluator.requested


def test_backend_failure_retains_only_completed_prefix_and_reuses_that_prefix() -> None:
    def merit(patch):
        if _block(patch) == {"b"}:
            raise ValueError("synthetic backend failure")
        return 1.0

    evaluator = _SyntheticEvaluator(merit)
    problem = _problem(("a", "b"))
    session = SearchSession(problem, evaluator, _rank_population, SearchLimits())
    with pytest.raises(ValueError, match="synthetic backend failure"):
        session.evaluate(problem.patches((), (("a",), ("b",))), label="partial_backend_failure")
    assert [_block(candidate) for candidate in session.evidence.values()] == [frozenset("a")]
    completed = next(iter(session.evidence.values()))
    calls_before = len(evaluator.requested)
    cached = session.evaluate(problem.patches((), (("a",),)), label="reuse_completed")
    assert cached[0] is completed
    assert len(evaluator.requested) == calls_before


@SEARCHES
def test_zero_structural_budget_stops_before_any_permission_or_evaluation(search) -> None:
    def no_permission(_removed):
        pytest.fail("zero structural budget must stop before permission work")

    evaluator = _SyntheticEvaluator(lambda _patch: 0.0)
    result = search(
        _problem((), coordinate_keys=("s",), deletable_indices=(0,), deletion_permission=no_permission),
        evaluator, _rank_population, SearchLimits(joint_patch_evaluations=0),
    )
    assert result.stop_reason == "joint_work_budget"
    assert not evaluator.requested
    assert not result.candidates


@pytest.mark.parametrize("compile_limit", [0, 2])
def test_compile_budget_bounds_request_construction_without_a_structural_limit(compile_limit) -> None:
    evaluator = _SyntheticEvaluator(lambda _patch: 1.0)
    session = SearchSession(
        _problem(()), evaluator, _rank_population,
        SearchLimits(compiled_patches_per_checkpoint=compile_limit),
    )
    constructed = []

    def request():
        yield Patch((), ())  # Stay consumes no compiled slot.
        for index in range(10_000):
            constructed.append(index)
            if len(constructed) > compile_limit + 1:
                pytest.fail("compile refusal must stop constructing the remaining request")
            patch = Patch((), ((f"a{index}", 0),))
            yield patch
            yield patch  # A duplicate does not consume another compiled slot.

    with pytest.raises(SearchBudgetExhausted) as stopped:
        session.evaluate(request(), label="complete_neighborhood")
    assert stopped.value.reason == "compiled_work_budget"
    assert len(constructed) == compile_limit + 1
    assert not evaluator.requested
    assert not session.evidence
    assert session.guard.scored_count == session.compile_guard.scored_count == 0
    refusal = session.finish(stopped.value.reason).receipt()["refusal"]
    assert refusal == {
        "label": "complete_neighborhood",
        "reason": "compiled_work_budget",
        "required_work_lower_bound": compile_limit + 1,
        "remaining_work": compile_limit,
    }
    assert refusal["required_work_lower_bound"] > refusal["remaining_work"]
    assert session.compile_guard.rejected_family_size is None


@pytest.mark.parametrize("gain_mode", ["joint_batch", "singleton_factorized"])
@pytest.mark.parametrize("explicit_none", [False, True], ids=["omitted", "none"])
def test_runtime_requires_gamma_tolerance_before_constructing_search(gain_mode, explicit_none) -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import resolve_route_bundle
    from pipelines.time_dynamics.ap_mclachlan.search_methods.runtime import prepare_patch_search

    kwargs = {"route_bundle": resolve_route_bundle("paper_ii_exchange_q_compiled_v1")}
    if explicit_none:
        kwargs["gamma_rcond"] = None
    # No cache or backend inputs are provided: an incomplete validation
    # contract must be rejected before search construction can acquire work.
    with pytest.raises(ValueError, match="Gamma validation tolerance"):
        prepare_patch_search(
            search_policy="greedy", gain_evaluation=gain_mode,
            structural_kwargs=kwargs, population_ranker=_rank_population,
        )
