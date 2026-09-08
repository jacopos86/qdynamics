"""Lazy permitted patches on one frozen checkpoint word.

The caller supplies the ordered deletion sets and insertion blocks. Permission
filters whole deletion branches before any insertion proposals are requested.
This module constructs mathematical patches only: work guards, population
boundaries, scoring, compilation, and execution scheduling belong to callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    InsertionPlan,
    TokensCommute,
    full_word_insertions,
    inserted_tokens,
    iter_quotient_insertion_plans,
    survivor_tokens,
)


@dataclass(frozen=True, slots=True)
class Patch:
    """One permitted deletion/insertion proposal, without a score.

    ``inserted_selection`` uses raw full-word cuts. A joint insertion retains
    its paired-word ``plan``; the geometry adapter may map the raw cuts to
    cached tangent representatives without changing that plan.
    """

    removed_runtime_indices: tuple[int, ...]
    inserted_selection: tuple[tuple[str, int], ...]
    plan: InsertionPlan | None = None


def iter_permitted_patches(
    *,
    coordinate_keys: Sequence[str],
    deletion_sets: Iterable[Sequence[int]],
    insertion_blocks_for_deletion: Callable[
        [tuple[int, ...]], Iterable[Sequence[str]]
    ],
    singleton_cuts: Callable[[tuple[int, ...], str], Iterable[int]],
    tokens_commute: TokensCommute | None = None,
    deletion_permission: Callable[[tuple[int, ...]], bool] | None = None,
) -> Iterator[Patch]:
    r"""Yield permitted :math:`(D,I)` patches in caller-supplied order.

    ``deletion_sets`` may be a lazy combination iterator. Each set contains
    distinct valid checkpoint indices; its tuple is sorted for branch identity
    without reordering the stream of sets. Nonempty ``D`` is offered once to
    the Boolean permission predicate before requesting its insertion blocks.

    The block callback supplies distinct child identities in frozen order:
    ``()`` requests stay/pure deletion, ``(a,)`` requests singleton positions
    from ``singleton_cuts(D, a)``, and larger blocks request the full/reduced
    word quotient. Block and singleton-cut order are preserved. A commutation
    oracle is required only for larger blocks. Callers own block cardinalities,
    eligibility, duplicate proposals, and competition populations.

    No subsequent deletion set, block, or word is requested until iteration
    needs it. Accepted permission nominates a patch; it does not certify a
    materialized state or authorize commitment.
    """

    survivors = survivor_tokens(coordinate_keys)
    for indices in deletion_sets:
        removed = tuple(sorted(int(i) for i in indices))
        if len(set(removed)) != len(removed) or any(
            i < 0 or i >= len(survivors) for i in removed
        ):
            raise ValueError("deletion sets require distinct valid checkpoint indices")
        if removed and deletion_permission is not None and not deletion_permission(removed):
            continue
        for proposal in insertion_blocks_for_deletion(removed):
            block = tuple(str(key) for key in proposal)
            if not block:
                yield Patch(removed, ())
            elif len(block) == 1:
                for cut in singleton_cuts(removed, block[0]):
                    yield Patch(removed, ((block[0], int(cut)),))
            else:
                if tokens_commute is None:
                    raise ValueError("joint insertion blocks require a commutation oracle")
                for plan in iter_quotient_insertion_plans(
                    survivors=survivors,
                    inserted=inserted_tokens(block),
                    removed_keys=tuple(survivors[i].key for i in removed),
                    tokens_commute=tokens_commute,
                ):
                    yield Patch(removed, full_word_insertions(plan.full_word), plan)


__all__ = ["Patch", "iter_permitted_patches"]
