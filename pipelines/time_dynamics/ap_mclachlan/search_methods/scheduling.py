"""Cooperative cardinality turns without changing a selector's move rule."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generator, Iterable, Iterator, Mapping

from ..patch_enumeration import Patch

if TYPE_CHECKING:
    from ..patch_evaluation import StructuralCandidate
    from .session import SearchSession


SCHEDULE_ID = "cardinality_rung_round_robin_v1"
SCHEDULE_IDS = {
    "interleaved": SCHEDULE_ID,
    "ascending": "ascending_max_cardinality_tiers_v1",
}
PATCHES_PER_TURN = 1
# A settled block neighborhood is waiting for new evidence, not rejected.
NEIGHBORHOOD_SETTLED = "neighborhood_settled"


@dataclass(frozen=True)
class PatchRequest:
    """One logical comparison, split into lazy insertion-cardinality streams.

    A selector receives the completed candidates only after every stream ends.
    Interleaving acquisition never turns a partial neighborhood into a move.
    """

    label: str
    patches_by_insertion_size: Mapping[int, Iterable[Patch]]


@dataclass
class _Branch:
    removed: tuple[int, ...]
    search: Generator
    pending: int = 0
    completed: dict[tuple, StructuralCandidate] = field(default_factory=dict)


@dataclass
class _Cursor:
    branch: _Branch
    label: str
    patches: Iterator[Patch]


def schedule_rungs(
    session: SearchSession,
    deletion_rungs: Iterable[tuple[int, Iterator[tuple[int, ...]]]],
    search_branch: Callable[[tuple[int, ...]], Generator],
) -> None:
    r"""Give each live ``(deletion size, insertion size)`` one patch turn.

    Ascending mode selects the smallest runnable tier
    :math:`t(D,A)=\max(1,|D|,|A|)`, including deletion discovery at tier
    :math:`\max(1,|D|)`. Tier one therefore contains pure singleton edits and
    one-delete/one-add exchanges. A logical request stays pending across tiers;
    only its complete result resumes the search branch. Later moves or newly
    awakened branches can expose lower-tier work, which regains priority before
    another higher-tier turn. Tiers are priorities, not irreversible phases.

    In the default interleaved mode, rungs use increasing total cardinality, then increasing imbalance, then
    deletion size. Branch cursors rotate FIFO within a rung. Each sweep opens
    at most one new deletion-cardinality iterator and inspects at most one
    deletion set from each open iterator; a refused set yields control without
    acquiring a pool or consuming an evaluation slot. Neither a lower rung nor
    a long word stream must finish before higher rungs receive turns.
    The independent discovery allowance limits first nonempty-D inspections,
    including refusals. Its exhaustion closes only discovery streams; live
    patch cursors, settled branches and all completed evidence are retained.

    Branch generators yield :class:`PatchRequest` and receive all its completed
    candidates. Block search may yield ``NEIGHBORHOOD_SETTLED`` to sleep at the
    current evidence version; any subsequent growth wakes it. Exhaustive and
    greedy branches simply return when finished. Budget exceptions propagate
    with the session's completed prefix intact.
    """

    ascending = session.patch_rung_schedule == "ascending"
    rung_source = iter(deletion_rungs)
    discovery: dict[int, Iterator[tuple[int, ...]]] = {}
    discovery_finished = False
    queues: dict[tuple[int, int], deque[_Cursor]] = {}
    settled: list[tuple[_Branch, int]] = []

    def resume(branch: _Branch, *, start: bool = False) -> None:
        completed = tuple(branch.completed.values())
        branch.completed.clear()
        try:
            request = next(branch.search) if start else branch.search.send(completed)
        except StopIteration:
            return
        if request == NEIGHBORHOOD_SETTLED:
            settled.append((branch, len(session.evidence)))
            return
        if not isinstance(request, PatchRequest):
            raise TypeError("a search branch must yield a PatchRequest or NEIGHBORHOOD_SETTLED")
        branch.pending = len(request.patches_by_insertion_size)
        if not branch.pending:
            resume(branch)
            return
        for insertion_size, patches in request.patches_by_insertion_size.items():
            key = (len(branch.removed), insertion_size)
            queues.setdefault(key, deque()).append(
                _Cursor(branch, request.label, iter(patches))
            )

    while True:
        # Do not request a new subset or permission when no structural work
        # remains. Existing cursors may still finish using cached evidence.
        if not discovery_finished and session.guard.remaining != 0:
            try:
                size, subsets = next(rung_source)
                if size > 0 and session.deletion_discovery_remaining == 0:
                    discovery_finished = True
                else:
                    discovery[size] = iter(subsets)
            except StopIteration:
                discovery_finished = True

        waking, settled = settled, []
        for branch, version in waking:
            if version < len(session.evidence):
                resume(branch)
            else:
                settled.append((branch, version))

        # Discovery cannot advance without a structural slot. In ascending
        # mode its dormant lower tiers must not mask queued cached work or the
        # next complete patch's budget refusal in a higher tier.
        live = set(queues)
        if session.guard.remaining != 0:
            live.update((size, 0) for size in discovery)
        active_tier = min((max(1, *key) for key in live), default=1)
        if ascending:
            live = {key for key in live if max(1, *key) == active_tier}
        for key in sorted(live, key=lambda pair: (sum(pair), abs(pair[0] - pair[1]), pair[0])):
            size, insertion_size = key
            if size > 0 and size in discovery and session.deletion_discovery_remaining == 0:
                del discovery[size]
            if insertion_size == 0 and size in discovery and session.guard.remaining != 0:
                try:
                    removed = next(discovery[size])
                except StopIteration:
                    del discovery[size]
                else:
                    session.check_boundary_budget(removed)
                    session.record_deletion_inspection(removed)
                    if session.deletion_permitted(removed):
                        resume(_Branch(removed, search_branch(removed)), start=True)

            queue = queues.get(key)
            if not queue:
                continue
            cursor = queue.popleft()
            try:
                patch = next(cursor.patches)
            except StopIteration:
                cursor.branch.pending -= 1
                if cursor.branch.pending == 0:
                    resume(cursor.branch)
            else:
                (candidate,) = session.evaluate((patch,), cursor.label)
                cursor.branch.completed[candidate.order_key] = candidate
                queue.append(cursor)
            if not queue:
                queues.pop(key, None)
            if ascending and (
                any(max(1, *ready) < active_tier for ready in queues)
                or any(version < len(session.evidence) for _branch, version in settled)
            ):
                # Recompute priorities after a move or evidence growth. A
                # waiting block can reveal smaller patches on its next wake.
                break

        if queues or any(version < len(session.evidence) for _branch, version in settled):
            continue
        if discovery or not discovery_finished:
            if session.guard.remaining == 0:
                session._refuse("joint_work_budget", "deletion_discovery")
            continue
        return


__all__ = [
    "PatchRequest", "schedule_rungs", "NEIGHBORHOOD_SETTLED",
    "SCHEDULE_ID", "SCHEDULE_IDS", "PATCHES_PER_TURN",
]
