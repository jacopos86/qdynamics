"""Local insertion-block exchanges with cumulative-population re-ranking."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

from ..patch_enumeration import Patch
from .session import SearchBudgetExhausted, SearchSession

if TYPE_CHECKING:
    from ..patch_evaluation import PatchEvaluator
    from .problem import PatchProblem
    from .session import SearchLimits, SearchResult


def insertion_neighborhood(
    pool: Sequence[str],
    selected: Sequence[str],
    *,
    maximum_size: int,
    move_size: int = 2,
) -> Iterator[tuple[str, ...]]:
    r"""Yield blocks at total edit distance ``1..move_size`` from ``selected``.

    A move deletes ``R`` and adds ``I`` with
    :math:`1 \leq |R| + |I| \leq r`. Thus the default permits two additions,
    two removals, or one-for-one replacement; two-for-two replacement requires
    ``move_size >= 4``. Blocks retain pool order. Move order is increasing
    distance, increasing removal count, then frozen combination order.

    This constructs operator-occurrence blocks only. The shared problem
    supplies every complete paired-word realization of each block.
    """

    if int(move_size) != move_size or move_size < 1:
        raise ValueError("move_size must be a positive integer")
    if int(maximum_size) != maximum_size or maximum_size < 0:
        raise ValueError("maximum_size must be a non-negative integer")
    pool = tuple(pool)
    selected_ids = set(selected)
    if len(set(pool)) != len(pool):
        raise ValueError("pool must contain distinct occurrence identities")
    if len(selected_ids) != len(selected) or not selected_ids.issubset(pool):
        raise ValueError("selected must contain distinct members of the pool")
    current = tuple(atom for atom in pool if atom in selected_ids)
    unused = tuple(atom for atom in pool if atom not in selected_ids)
    for distance in range(1, min(int(move_size), len(pool)) + 1):
        for remove_count in range(min(distance, len(current)) + 1):
            add_count = distance - remove_count
            if add_count > len(unused):
                continue
            if len(current) - remove_count + add_count > maximum_size:
                continue
            for removed in combinations(current, remove_count):
                kept = selected_ids.difference(removed)
                for added in combinations(unused, add_count):
                    block = kept.union(added)
                    yield tuple(atom for atom in pool if atom in block)


def search_patches(
    problem: PatchProblem,
    evaluate: PatchEvaluator,
    rank: Callable,
    budget: SearchLimits,
    *,
    move_size: int = 2,
) -> SearchResult:
    """Improve complete insertion blocks while keeping every ``D`` explicit.

    Each newly admitted deletion branch is improved immediately, so acquiring
    all deletion boundaries cannot consume the budget before insertion work.
    After that first pass, repeat branch sweeps until neither the cumulative
    evidence nor any incumbent changes. Later population growth may invalidate
    an earlier stopping decision, so previously visited blocks remain legal.

    Every move strictly improves one immutable ranking snapshot. Over the
    finite permitted universe, only finitely many population-growth events can
    occur; between them, strict deterministic rank improvement rules out a
    cycle. Work-guard termination is distinct from neighborhood exhaustion.
    Gain evaluation, placement, compiled costs, and final certification remain
    shared with the other search methods.
    """

    if int(move_size) != move_size or move_size < 1:
        raise ValueError("move_size must be a positive integer")
    radius = int(move_size)
    session = SearchSession(problem, evaluate, rank, budget)

    def improve(removed, pool, selected):
        moved = False
        while True:
            neighbors = session.evaluate(
                problem.patches(removed, insertion_neighborhood(
                    pool, selected,
                    maximum_size=problem.max_insertion_batch_size,
                    move_size=radius,
                )),
                label=f"block_r{radius}_d{len(removed)}_a{len(selected)}",
            )
            ranking = session.rank()
            incumbent = session.anchor(removed, selected, ranking)
            challenger = session.best(neighbors, ranking)
            if challenger is None or not session.outranks(
                challenger, incumbent, ranking
            ):
                return selected, moved
            chosen = {atom for atom, _cut in challenger.inserted_selection}
            selected = tuple(atom for atom in pool if atom in chosen)
            session.record_move(removed, selected, challenger, ranking)
            session.moves[-1]["move_size"] = radius
            moved = True

    branches = []
    try:
        for removed in problem.deletion_sets():
            session.check_boundary_budget(removed)
            if not session.deletion_permitted(removed):
                continue
            session.evaluate((Patch(removed, ()),), label=f"boundary_d{len(removed)}")
            pool = problem.pool(removed)
            selected, _moved = improve(removed, pool, ())
            branches.append((removed, pool, selected))

        while True:
            population_before = len(session.evidence)
            moved = False
            for index, (removed, pool, selected) in enumerate(branches):
                selected, branch_moved = improve(removed, pool, selected)
                branches[index] = (removed, pool, selected)
                moved = moved or branch_moved
            if not moved and len(session.evidence) == population_before:
                return session.finish("block_neighborhoods_exhausted")
    except SearchBudgetExhausted as exc:
        return session.finish(exc.reason)


__all__ = ["insertion_neighborhood", "search_patches"]
