"""Monotone greedy insertion-block extensions on explicit deletion branches."""

from __future__ import annotations

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
    """Greedily extend a provisional insertion block on each permitted ``D``.

    Every unused one-add extension is evaluated as a complete patch, including
    all paired-word realizations supplied by the problem. The best eligible
    extension must outrank the current block (or stay when it is ineligible)
    under the same cumulative-population ranking snapshot. Otherwise this
    branch stops; an ineligible or worsening intermediate is never traversed.

    Strictly increasing block size makes each branch finite. The complete
    evaluation ledger is retained for final ranking and certification, even
    when a candidate did not become the provisional incumbent. This search
    does not claim exhaustive coverage or infer a batch score from singleton
    gains or compiled costs.
    """

    session = SearchSession(problem, evaluate, rank, budget)

    def search_branch(removed):
        pool = problem.pool(removed)
        maximum_size = min(int(problem.max_insertion_batch_size), len(pool))
        boundary = {0: iter((Patch(removed, ()),))}
        if maximum_size == 0:
            yield PatchRequest(f"boundary_d{len(removed)}", boundary)
            return
        selected: tuple[str, ...] = ()
        while len(selected) < maximum_size:
            selected_ids = set(selected)
            size = len(selected) + 1
            blocks = (
                tuple(atom for atom in pool if atom in selected_ids or atom == addition)
                for addition in pool
                if addition not in selected_ids
            )
            extensions = yield PatchRequest(
                f"greedy_d{len(removed)}_a{size}",
                {**boundary, size: problem.patches(removed, blocks)},
            )
            boundary = {}
            ranking = session.rank()
            incumbent = session.anchor(removed, selected, ranking)
            challenger = session.best(
                tuple(c for c in extensions if len(c.inserted_selection) == size),
                ranking,
            )
            if challenger is None or not session.outranks(challenger, incumbent, ranking):
                return
            chosen_ids = {atom for atom, _cut in challenger.inserted_selection}
            selected = tuple(atom for atom in pool if atom in chosen_ids)
            session.record_move(removed, selected, challenger, ranking)

    try:
        schedule_rungs(session, problem.deletion_rungs(), search_branch)
    except SearchBudgetExhausted as exc:
        return session.finish(exc.reason)
    return session.finish("greedy_extensions_exhausted")


__all__ = ["search_patches"]
