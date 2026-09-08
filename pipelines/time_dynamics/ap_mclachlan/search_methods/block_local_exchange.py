"""Local insertion-block exchanges with cumulative-population re-ranking."""

from __future__ import annotations

from itertools import chain, combinations
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

from ..patch_enumeration import Patch
from .scheduling import NEIGHBORHOOD_SETTLED, PatchRequest, schedule_rungs
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
    target_size: int | None = None,
) -> Iterator[tuple[str, ...]]:
    r"""Yield blocks at total edit distance ``1..move_size`` from ``selected``.

    A move deletes ``R`` and adds ``I`` with
    :math:`1 \leq |R| + |I| \leq r`. Thus the default permits two additions,
    two removals, or one-for-one replacement; two-for-two replacement requires
    ``move_size >= 4``. Blocks retain pool order. Move order is increasing
    distance, increasing removal count, then frozen combination order.

    This constructs operator-occurrence blocks only. The shared problem
    supplies every complete paired-word realization of each block.
    ``target_size`` selects one cardinality before constructing combinations,
    so separately scheduled streams do not scan other cardinalities.
    """

    if int(move_size) != move_size or move_size < 1:
        raise ValueError("move_size must be a positive integer")
    if int(maximum_size) != maximum_size or maximum_size < 0:
        raise ValueError("maximum_size must be a non-negative integer")
    if target_size is not None and (int(target_size) != target_size or target_size < 0):
        raise ValueError("target_size must be a non-negative integer or None")
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
            size = len(current) - remove_count + add_count
            if size > maximum_size or (target_size is not None and size != target_size):
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

    Cardinality streams receive short turns across deletion branches. A move
    waits for its complete neighborhood; a settled branch wakes whenever the
    cumulative evidence grows. Earlier stopping decisions can therefore be
    reconsidered without pruning previously visited blocks.

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

    def search_branch(removed):
        pool = problem.pool(removed)
        maximum_size = min(int(problem.max_insertion_batch_size), len(pool))
        selected: tuple[str, ...] = ()
        boundary = iter((Patch(removed, ()),))
        while True:
            streams = {
                size: problem.patches(removed, insertion_neighborhood(
                    pool, selected, maximum_size=maximum_size,
                    move_size=radius, target_size=size,
                ))
                for size in range(
                    max(0, len(selected) - radius),
                    min(maximum_size, len(selected) + radius) + 1,
                )
            }
            if boundary is not None:
                streams[0] = chain(boundary, streams.get(0, ()))
                boundary = None
            neighbors = yield PatchRequest(
                f"block_r{radius}_d{len(removed)}_a{len(selected)}", streams,
            )
            ranking = session.rank()
            incumbent = session.anchor(removed, selected, ranking)
            challenger = session.best(neighbors, ranking)
            if challenger is None or not session.outranks(
                challenger, incumbent, ranking
            ):
                yield NEIGHBORHOOD_SETTLED
                continue
            chosen = {atom for atom, _cut in challenger.inserted_selection}
            selected = tuple(atom for atom in pool if atom in chosen)
            session.record_move(removed, selected, challenger, ranking)
            session.moves[-1]["move_size"] = radius
    try:
        schedule_rungs(session, problem.deletion_rungs(), search_branch)
    except SearchBudgetExhausted as exc:
        return session.finish(exc.reason)
    return session.finish("block_neighborhoods_exhausted")


__all__ = ["insertion_neighborhood", "search_patches"]
