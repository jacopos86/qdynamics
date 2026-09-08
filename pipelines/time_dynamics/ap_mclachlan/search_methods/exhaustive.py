"""Complete search of the declared deletion/insertion patch universe."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Callable

from ..patch_enumeration import Patch
from .scheduling import PatchRequest, schedule_rungs
from .session import SearchBudgetExhausted, SearchSession

if TYPE_CHECKING:
    from ..patch_evaluation import PatchEvaluator
    from .problem import PatchProblem
    from .session import SearchLimits, SearchResult


def search_patches(
    problem: PatchProblem,
    evaluate: PatchEvaluator,
    rank: Callable,
    budget: SearchLimits,
) -> SearchResult:
    """Evaluate every permitted complete patch unless a work guard refuses.

    Cardinality streams share short turns, including both pure faces. Within
    each stream, blocks and paired-word realizations retain frozen order. No
    score or provisional winner restricts the remaining search. Budget
    termination is reported separately from exhaustive completion.
    """

    session = SearchSession(problem, evaluate, rank, budget)

    def search_branch(removed):
        pool = problem.pool(removed)
        maximum_size = min(int(problem.max_insertion_batch_size), len(pool))
        yield PatchRequest(
            f"exhaustive_d{len(removed)}",
            {
                0: iter((Patch(removed, ()),)),
                **{
                    size: problem.patches(removed, combinations(pool, size))
                    for size in range(1, maximum_size + 1)
                },
            },
        )

    try:
        schedule_rungs(session, problem.deletion_rungs(), search_branch)
    except SearchBudgetExhausted as exc:
        return session.finish(exc.reason)
    return session.finish("universe_exhausted", exhaustive_complete=True)


__all__ = ["search_patches"]
