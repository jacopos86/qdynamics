"""The permitted patch universe, independent of its search and gain model."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, floor, isfinite
from numbers import Integral, Real
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from ..deletion_family import iter_deletion_rungs
from ..insertion_words import TokensCommute
from ..patch_enumeration import Patch, iter_permitted_patches


@dataclass(frozen=True, kw_only=True)
class PatchProblem:
    """Frozen support, full branch pools, and explicit scientific permissions.

    Insertion and deletion cardinality limits declare the common enumerated
    universe. The optional deletion fraction bounds batch size relative to the
    full current support; explicit cardinality and support-floor limits still
    apply. With both deletion limits None, cardinality is unrestricted.
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
    max_deletion_batch_fraction: float | None = None
    tokens_commute: TokensCommute | None = None
    deletion_permission: Callable[[tuple[int, ...]], bool] | None = None

    def __post_init__(self) -> None:
        if self.max_deletion_batch_size is not None and (
            isinstance(self.max_deletion_batch_size, bool)
            or not isinstance(self.max_deletion_batch_size, Integral)
            or self.max_deletion_batch_size < 0
        ):
            raise ValueError("max_deletion_batch_size must be a non-negative integer or None.")
        if self.max_deletion_batch_fraction is not None and (
            isinstance(self.max_deletion_batch_fraction, bool)
            or not isinstance(self.max_deletion_batch_fraction, Real)
            or not isfinite(self.max_deletion_batch_fraction)
            or not 0 <= self.max_deletion_batch_fraction <= 1
        ):
            raise ValueError("max_deletion_batch_fraction must be in [0, 1] or None.")
        if self.max_insertion_batch_size < 0:
            raise ValueError("max_insertion_batch_size must be non-negative")
        if len(set(self.coordinate_keys)) != len(self.coordinate_keys):
            raise ValueError("coordinate keys must be unique")
        if len(set(self.deletable_indices)) != len(self.deletable_indices) or any(
            i < 0 or i >= len(self.coordinate_keys) for i in self.deletable_indices
        ):
            raise ValueError("deletable indices must be distinct valid coordinates")

    @property
    def deletion_cardinality_limit(self) -> int:
        r"""Resolve $k_{max}=\min(m,\lfloor f n\rfloor,n-n_{min},k_{explicit})$.

        Omit optional bounds when unset. Here n is the full current support
        and m counts structurally deletable coordinates, before whole-D
        permission. Exact decimal arithmetic keeps the declared floor stable.
        """

        n = len(self.coordinate_keys)
        bounds = [len(self.deletable_indices), max(0, n - self.min_surviving_support)]
        if self.max_deletion_batch_size is not None:
            bounds.append(self.max_deletion_batch_size)
        if self.max_deletion_batch_fraction is not None:
            bounds.append(floor(Fraction(str(self.max_deletion_batch_fraction)) * n))
        return min(bounds)

    @property
    def deletion_subset_count(self) -> int:
        r"""Number of nonempty batches before permission: $\sum_{k=1}^{k_{max}}\binom{m}{k}$."""

        return sum(comb(len(self.deletable_indices), k)
                   for k in range(1, self.deletion_cardinality_limit + 1))

    def deletion_sets(self) -> Iterator[tuple[int, ...]]:
        """Hard-feasible deletion sets in increasing cardinality and frozen order."""

        for _size, rung in self.deletion_rungs():
            yield from rung

    def deletion_rungs(self) -> Iterator[tuple[int, Iterator[tuple[int, ...]]]]:
        """Lazy cardinality streams, with the declared deletion cap applied."""

        maximum_size = self.deletion_cardinality_limit
        for size, rung in iter_deletion_rungs(
            self.deletable_indices,
            total_support=len(self.coordinate_keys),
            min_surviving_support=self.min_surviving_support,
        ):
            if size > maximum_size:
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
