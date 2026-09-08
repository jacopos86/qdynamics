"""The permitted patch universe, independent of its search and gain model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from ..deletion_family import iter_deletion_rungs
from ..insertion_words import TokensCommute
from ..patch_enumeration import Patch, iter_permitted_patches


@dataclass(frozen=True, kw_only=True)
class PatchProblem:
    """Frozen support, full branch pools, and explicit scientific permissions.

    Insertion and optional deletion cardinality limits declare the common
    enumerated universe. None leaves deletion cardinality unrestricted.
    There is no pool cap or pure-insertion-score frontier. The caller supplies
    one cached tangent representative per distinct singleton position class;
    joint blocks retain complete paired-word plans.
    """

    coordinate_keys: tuple[str, ...]
    deletable_indices: tuple[int, ...]
    min_surviving_support: int
    candidate_pool_for_deletion: Callable[[tuple[int, ...]], tuple[str, ...]]
    cuts_by_atom: Mapping[str, Sequence[int]]
    max_insertion_batch_size: int = 1
    max_deletion_batch_size: int | None = None
    tokens_commute: TokensCommute | None = None
    deletion_permission: Callable[[tuple[int, ...]], bool] | None = None

    def __post_init__(self) -> None:
        if self.max_deletion_batch_size is not None and (
            isinstance(self.max_deletion_batch_size, bool)
            or not isinstance(self.max_deletion_batch_size, Integral)
            or self.max_deletion_batch_size < 0
        ):
            raise ValueError("max_deletion_batch_size must be a non-negative integer or None.")
        if self.max_insertion_batch_size < 0:
            raise ValueError("max_insertion_batch_size must be non-negative")
        if len(set(self.coordinate_keys)) != len(self.coordinate_keys):
            raise ValueError("coordinate keys must be unique")
        if len(set(self.deletable_indices)) != len(self.deletable_indices) or any(
            i < 0 or i >= len(self.coordinate_keys) for i in self.deletable_indices
        ):
            raise ValueError("deletable indices must be distinct valid coordinates")

    def deletion_sets(self) -> Iterator[tuple[int, ...]]:
        """Hard-feasible deletion sets in increasing cardinality and frozen order."""

        for _size, rung in self.deletion_rungs():
            yield from rung

    def deletion_rungs(self) -> Iterator[tuple[int, Iterator[tuple[int, ...]]]]:
        """Lazy cardinality streams, with the declared deletion cap applied."""

        for size, rung in iter_deletion_rungs(
            self.deletable_indices,
            total_support=len(self.coordinate_keys),
            min_surviving_support=self.min_surviving_support,
        ):
            if self.max_deletion_batch_size is not None and size > self.max_deletion_batch_size:
                return
            yield size, rung

    def permitted(self, removed: tuple[int, ...]) -> bool:
        return not removed or self.deletion_permission is None or bool(
            self.deletion_permission(removed)
        )

    def pool(self, removed: tuple[int, ...]) -> tuple[str, ...]:
        pool = tuple(self.candidate_pool_for_deletion(removed))
        if len(set(pool)) != len(pool):
            raise ValueError("a branch pool must contain distinct occurrence identities")
        return pool

    def patches(
        self, removed: tuple[int, ...], blocks: Iterable[Sequence[str]],
    ) -> Iterator[Patch]:
        """All positioned realizations of the supplied insertion blocks.

        Search applies deletion permission before supplying insertion blocks;
        this function only constructs the requested mathematical patches.
        """

        yield from iter_permitted_patches(
            coordinate_keys=self.coordinate_keys,
            deletion_sets=(removed,),
            insertion_blocks_for_deletion=lambda _removed: blocks,
            singleton_cuts=lambda _removed, atom: self.cuts_by_atom.get(atom, ()),
            tokens_commute=self.tokens_commute,
        )


def patch_identity(patch: Patch) -> tuple:
    """Identify a complete realization independently of its first search visit."""

    realization = (
        ("positioned", patch.inserted_selection)
        if patch.plan is None else ("word", patch.plan.plan_id)
    )
    return patch.removed_runtime_indices, realization


__all__ = ["PatchProblem", "patch_identity"]
