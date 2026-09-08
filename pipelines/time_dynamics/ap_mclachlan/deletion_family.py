"""Complete deletion-cardinality rungs and the joint-work guard.

This pure generator enumerates deletion subsets in complete cardinality rungs
``D_{k,d} = {D subset J_k^del : |D| = d, hard-feasible}``. The caller may stop
at an explicitly declared maximum deletion cardinality, and applies complete-set
permission before the joint-work guard admits or rejects the permitted patch family
*before* any member is scored; it never samples part of a rung. Permission
checks consume no patch-evaluation slots.

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
) -> Iterator[tuple[int, Iterator[tuple[int, ...]]]]:
    """Yield ``(d, lazy_rung)`` in increasing feasible cardinality order.

    Each rung is a one-pass iterator in frozen combination order; requesting
    its descriptor does not construct any deletion subset. A caller can count
    until its existing whole-family guard proves rejection, then discard the
    remainder without materializing the rung. No partial-family admission or
    computational bound is imposed here.
    """

    deletable = tuple(int(i) for i in deletable_indices)
    if len(set(deletable)) != len(deletable):
        raise ValueError("deletable_indices must be unique.")
    for d in range(len(deletable) + 1):
        if d > 0 and int(total_support) - d < int(min_surviving_support):
            return
        rung = (
            tuple(sorted(int(i) for i in subset))
            for subset in combinations(deletable, d)
        )
        yield d, rung


@dataclass
class JointWorkGuard:
    """Whole-request admission for permitted patch-evaluation work.

    ``max_joint_patch_evaluations = None`` disables the guard.  A finite value
    admits a complete family exactly when
    ``N_reserved + N_next <= max_joint_patch_evaluations``; rejection evaluates
    none of the family and freezes further admissions, and the caller ends
    escalation on the same event. Callers count permitted patches, or their
    regime-active subset for compiled-work admission. Permission rejections
    consume no slots. Reservations remain charged if an admitted backend
    request fails; ``scored_count`` is not a count of completed results.

    Freezing is a declared stop, not a proof that every later family would
    also overrun: families are exposed in deterministic order, but that order
    is not size-monotone (a rung's cardinality ``C(m, d)`` is unimodal in
    ``d``). ``rejected_family_size`` records the refused family's size when
    known; it remains unset when construction stops after proving a lower
    bound exceeds the remaining budget.
    """

    max_joint_patch_evaluations: int | None = None
    scored_count: int = 0
    rejected_family: str | None = None
    rejected_family_size: int | None = None
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

    @property
    def remaining(self) -> int | None:
        """Budget left before the next family must be rejected (None = unbounded)."""

        if self.max_joint_patch_evaluations is None:
            return None
        return max(0, int(self.max_joint_patch_evaluations) - int(self.scored_count))

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
            self.rejected_family_size = int(family_size)
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
            "rejected_family_size": (
                None
                if self.rejected_family_size is None
                else int(self.rejected_family_size)
            ),
        }


__all__ = [
    "JointWorkGuard",
    "deletion_rung",
    "iter_deletion_rungs",
    "rung_size",
]
