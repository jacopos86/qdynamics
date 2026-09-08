"""Complete search of the declared deletion/insertion patch universe."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Callable

from ..patch_enumeration import Patch
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

    Deletion sets follow the problem's frozen order. Within a permitted branch,
    all insertion blocks of one cardinality form one complete admission batch;
    the shared constructor supplies their paired-word realizations. No score
    or provisional winner restricts the remaining search. Budget termination
    is reported separately from exhaustive completion.
    """

    session = SearchSession(problem, evaluate, rank, budget)
    try:
        for removed in problem.deletion_sets():
            session.check_boundary_budget(removed)
            if not session.deletion_permitted(removed):
                continue
            session.evaluate((Patch(removed, ()),), label=f"boundary_d{len(removed)}")
            pool = problem.pool(removed)
            maximum_size = min(int(problem.max_insertion_batch_size), len(pool))
            for size in range(1, maximum_size + 1):
                session.evaluate(
                    problem.patches(removed, combinations(pool, size)),
                    label=f"exhaustive_d{len(removed)}_a{size}",
                )
    except SearchBudgetExhausted as exc:
        return session.finish(exc.reason)
    return session.finish("universe_exhausted", exhaustive_complete=True)


__all__ = ["search_patches"]
