"""Complete deletion-cardinality rungs and the joint-work guard.

The selector enumerates deletion subsets in complete cardinality rungs
``D_{k,d} = {D subset J_k^del : |D| = d, hard-feasible}`` with no separate
maximum deletion-batch cardinality: the only computational cap is the joint
work guard, which admits or rejects each complete family *before* any member
is scored and never samples part of a rung.

Hard feasibility here is minimum surviving support; atom-level feasibility
(target policy, drive-aligned protection, cooldown, occurrence identity) is
resolved by the caller when it builds the deletable index tuple, so this
module stays pure combinatorics with frozen-order determinism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Iterator, Sequence


def rung_size(deletable_count: int, cardinality: int) -> int:
    """|D_{k,d}| before hard batch feasibility: C(|J_del|, d)."""

    n = int(deletable_count)
    d = int(cardinality)
    if d < 0 or n < 0:
        raise ValueError("counts must be non-negative.")
    return comb(n, d) if d <= n else 0

def deletion_rung(
    deletable_indices: Sequence[int],
    *,
    cardinality: int,
    total_support: int,
    min_surviving_support: int,
) -> tuple[tuple[int, ...], ...]:
    """One complete ordered rung of hard-feasible deletion subsets.

    Subsets follow the frozen order of ``deletable_indices`` (itertools
    combinations order); each subset is emitted as a sorted tuple of runtime
    indices.  A subset is hard-feasible when the surviving support stays at or
    above ``min_surviving_support``.
    """

    deletable = tuple(int(i) for i in deletable_indices)
    if len(set(deletable)) != len(deletable):
        raise ValueError("deletable_indices must be unique.")
    d = int(cardinality)
    if d == 0:
        return ((),)
    if int(total_support) - d < int(min_surviving_support):
        return ()
    return tuple(
        tuple(sorted(int(i) for i in subset))
        for subset in combinations(deletable, d)
    )


def iter_deletion_rungs(
    deletable_indices: Sequence[int],
    *,
    total_support: int,
    min_surviving_support: int,
) -> Iterator[tuple[int, tuple[tuple[int, ...], ...]]]:
    """Yield ``(d, rung)`` for d = 0..|J_del|, skipping infeasible-empty rungs
    above the surviving-support bound (which truncates all later rungs too)."""

    deletable = tuple(int(i) for i in deletable_indices)
    for d in range(len(deletable) + 1):
        rung = deletion_rung(
            deletable,
            cardinality=d,
            total_support=total_support,
            min_surviving_support=min_surviving_support,
        )
        if d > 0 and not rung:
            return
        yield d, rung


@dataclass
class JointWorkGuard:
    """The sole computational cap on structural enumeration.

    ``max_joint_patch_evaluations = None`` disables the guard.  A finite value
    admits a complete family exactly when
    ``N_done + N_next <= max_joint_patch_evaluations``; rejection evaluates
    none of the family and freezes further admissions (families are exposed in
    deterministic order, so a rejected family also ends escalation).
    """

    max_joint_patch_evaluations: int | None = None
    scored_count: int = 0
    rejected_family: str | None = None
    admitted_families: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            self.max_joint_patch_evaluations is not None
            and int(self.max_joint_patch_evaluations) < 0
        ):
            raise ValueError("max_joint_patch_evaluations must be non-negative.")

    @property
    def exhausted(self) -> bool:
        return self.rejected_family is not None

    def admit(self, family_label: str, family_size: int) -> bool:
        """Admit one complete family of ``family_size`` unique typed patches."""

        if int(family_size) < 0:
            raise ValueError("family_size must be non-negative.")
        if self.rejected_family is not None:
            return False
        if self.max_joint_patch_evaluations is not None and (
            self.scored_count + int(family_size)
            > int(self.max_joint_patch_evaluations)
        ):
            self.rejected_family = str(family_label)
            return False
        self.scored_count += int(family_size)
        self.admitted_families.append(str(family_label))
        return True

    def to_json_dict(self) -> dict:
        return {
            "max_joint_patch_evaluations": (
                None
                if self.max_joint_patch_evaluations is None
                else int(self.max_joint_patch_evaluations)
            ),
            "scored_count": int(self.scored_count),
            "admitted_families": [str(x) for x in self.admitted_families],
            "rejected_family": (
                None if self.rejected_family is None else str(self.rejected_family)
            ),
        }


__all__ = [
    "JointWorkGuard",
    "deletion_rung",
    "iter_deletion_rungs",
    "rung_size",
]
