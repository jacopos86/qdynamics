"""Shared work admission and cumulative evidence for patch searches.

Search policies choose which complete patches to request. This module admits
individual patches, caches raw results, and supplies immutable population-ranking
snapshots. It contains no gain formula, compilation policy, or certification.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

from ..deletion_family import JointWorkGuard
from ..patch_enumeration import Patch
from ..patch_evaluation import PatchEvaluator, StructuralCandidate
from .problem import PatchProblem, patch_identity

if TYPE_CHECKING:
    from ..exchange_selector import RankedLevel


@dataclass(frozen=True)
class SearchLimits:
    joint_patch_evaluations: int | None = None
    compiled_patches_per_checkpoint: int | None = None


class SearchBudgetExhausted(RuntimeError):
    """End the outer search; this does not mean a candidate is inadmissible."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[StructuralCandidate, ...]
    guard: JointWorkGuard
    compile_guard: JointWorkGuard
    stop_reason: str
    exhaustive_complete: bool
    moves: tuple[dict, ...]
    permitted_deletion_count: int
    refused_deletion_count: int
    refusal: dict | None = None
    # (deletion cardinality, permitted visits, refused visits), including D=().
    permission_counts_by_cardinality: tuple[tuple[int, int, int], ...] = ()

    def receipt(self) -> dict:
        """Report completed coverage and known permissions, never unseen eligibility.

        Missing cardinality rows mean no completed patches or cached permission
        visits at that size. They do not imply that such patches are forbidden.
        Work reservations and compiler work remain in the existing guards.
        """

        patch_counts = Counter(
            (len(candidate.removed_runtime_indices), len(candidate.inserted_selection))
            for candidate in self.candidates
        )
        return {
            "stop_reason": self.stop_reason,
            "exhaustive_complete": self.exhaustive_complete,
            "evaluated_patch_count": len(self.candidates),
            "permitted_deletion_count": self.permitted_deletion_count,
            "refused_deletion_count": self.refused_deletion_count,
            "coverage": {
                "evaluated_patches_by_cardinality": [
                    {"deletion_size": d, "insertion_size": a, "count": count}
                    for (d, a), count in sorted(patch_counts.items())
                ],
                "visited_deletion_sets_by_cardinality": [
                    {"deletion_size": d, "permitted": permitted, "refused": refused}
                    for d, permitted, refused in self.permission_counts_by_cardinality
                ],
            },
            "moves": list(self.moves),
            **({"refusal": dict(self.refusal)} if self.refusal is not None else {}),
        }


class SearchSession:
    """One search's raw evidence ledger and common computational guards.

    Each distinct permitted patch in an admitted request consumes one
    structural slot. Refused deletion sets are recorded in the permission
    cache and receipt without consuming patch-evaluation slots.
    Compiled slots count permitted regime-active patches, as in the current
    route. Actual transpilation calls remain recorded by the cost evaluator.

    Each patch is admitted before its numerical or compiler work. If a request
    exceeds the remaining budget, its completed prefix remains in the ledger.
    Revisited patches reuse evidence, never a previously normalized score.
    """

    def __init__(
        self, problem: PatchProblem, evaluate: PatchEvaluator,
        rank: Callable[[Sequence[StructuralCandidate]], RankedLevel],
        budget: SearchLimits,
    ) -> None:
        self.problem = problem
        self.evaluator = evaluate
        self.ranker = rank
        self.guard = JointWorkGuard(budget.joint_patch_evaluations)
        self.compile_guard = JointWorkGuard(budget.compiled_patches_per_checkpoint)
        self.evidence: dict[tuple, StructuralCandidate] = {}
        self.permission_cache: dict[tuple[int, ...], bool] = {}
        self.moves: list[dict] = []
        self.stop_reason = ""
        self.refusal: dict | None = None
        self._ranking: RankedLevel | None = None
        self._ranked_size = -1
        # Guard validation precedes the baseline solve.
        self.q_base = evaluate.q_base

    @property
    def stopped(self) -> bool:
        return bool(self.stop_reason)

    def _refuse(self, reason: str, label: str, required_at_least: int = 1) -> None:
        self.stop_reason = reason
        self.guard.rejected_family = label
        self.compile_guard.rejected_family = label
        limiting_guard = self.compile_guard if reason == "compiled_work_budget" else self.guard
        # Report the refused next patch's slot requirement. The unevaluated
        # remainder of the logical request is deliberately never counted.
        self.refusal = {
            "label": label,
            "reason": reason,
            "required_work_lower_bound": int(required_at_least),
            "remaining_work": limiting_guard.remaining,
        }
        raise SearchBudgetExhausted(reason)

    def check_boundary_budget(self, removed: tuple[int, ...]) -> None:
        """Preflight structural work before requesting an unseen permission."""

        if self.stopped:
            raise SearchBudgetExhausted(self.stop_reason)
        if patch_identity(Patch(removed, ())) in self.evidence:
            return
        if self.guard.remaining == 0:
            self._refuse("joint_work_budget", f"boundary_d{len(removed)}")

    def deletion_permitted(self, removed: tuple[int, ...]) -> bool:
        """Return only mathematical admissibility; never evaluate a patch."""

        if removed not in self.permission_cache:
            self.permission_cache[removed] = self.problem.permitted(removed)
        return self.permission_cache[removed]

    def evaluate(
        self, patches: Iterable[Patch], label: str,
    ) -> tuple[StructuralCandidate, ...]:
        """Evaluate and save each complete patch that fits, in request order.

        A budget refusal raises only after saving all completed predecessors.
        The caller ends the search and ranks that cumulative evidence; it must
        not interpret refusal as completion of the logical request. Cached
        patches consume no slots. Backend failures retain completed results,
        while the attempted patch's reservation remains charged.
        """

        if self.stopped:
            raise SearchBudgetExhausted(self.stop_reason)
        keys: list[tuple] = []
        seen_request: set[tuple] = set()
        for patch in patches:
            key = patch_identity(patch)
            if key in seen_request:
                continue
            seen_request.add(key)
            keys.append(key)
            if key not in self.evidence:
                if self.guard.remaining == 0:
                    self._refuse("joint_work_budget", label)
                compiled_size = int(
                    bool(patch.removed_runtime_indices or patch.inserted_selection)
                    and self.evaluator.active_population_face(
                        patch.removed_runtime_indices, patch.inserted_selection,
                    )
                )
                if compiled_size and self.compile_guard.remaining == 0:
                    self._refuse("compiled_work_budget", label)
                if not self.guard.admit(label, 1) or not self.compile_guard.admit(label, compiled_size):
                    raise AssertionError("patch admission changed after preflight")
                # Even an eager/parallel evaluator receives only admitted work.
                evaluated = iter(self.evaluator.evaluate((patch,), label))
                try:
                    candidate = next(evaluated)
                except StopIteration as exc:
                    raise ValueError("the evaluator returned fewer records than requested") from exc
                # Store only completed results, including a completed prefix
                # if a later evaluation raises a numerical/backend error.
                self.evidence[key] = candidate
                if next(evaluated, None) is not None:
                    raise ValueError("the evaluator returned more records than requested")
        return tuple(self.evidence[key] for key in keys)

    def rank(self) -> RankedLevel:
        if self._ranking is None or self._ranked_size != len(self.evidence):
            self._ranking = self.ranker(tuple(self.evidence.values()))
            self._ranked_size = len(self.evidence)
        return self._ranking

    @staticmethod
    def best(candidates: Sequence[StructuralCandidate], ranking: RankedLevel) -> StructuralCandidate | None:
        keys = {candidate.order_key for candidate in candidates}
        return next((c for c in ranking.ordered if c.order_key in keys), None)

    def anchor(self, removed: tuple[int, ...], block: Sequence[str], ranking: RankedLevel) -> StructuralCandidate | None:
        ids = frozenset(block)
        candidates = tuple(
            c for c in self.evidence.values()
            if c.removed_runtime_indices == removed
            and frozenset(a for a, _cut in c.inserted_selection) == ids
        )
        return self.best(candidates, ranking)

    @staticmethod
    def outranks(challenger: StructuralCandidate, incumbent: StructuralCandidate | None, ranking: RankedLevel) -> bool:
        keys = {c.order_key: i for i, c in enumerate(ranking.ordered)}
        challenger_rank = keys.get(challenger.order_key)
        if challenger_rank is None:
            return False
        incumbent_rank = None if incumbent is None else keys.get(incumbent.order_key)
        return incumbent_rank is None or challenger_rank < incumbent_rank

    def record_move(self, removed, block, candidate, ranking) -> None:
        info = ranking.rank_info.get(candidate.order_key)
        self.moves.append({
            "removed": list(removed),
            "inserted_block": list(block),
            "evaluated_population_size": len(self.evidence),
            "population_sha256": ranking.telemetry.get("population_sha256"),
            "medians": ranking.telemetry.get("medians"),
            "mad_scales": ranking.telemetry.get("mad_scales"),
            "score": None if info is None else info.score,
            "cost_factor": None if info is None else info.K,
        })

    def finish(self, reason: str, exhaustive_complete: bool = False) -> SearchResult:
        permission_counts = Counter(
            (len(removed), allowed) for removed, allowed in self.permission_cache.items()
        )
        return SearchResult(
            candidates=tuple(self.evidence.values()), guard=self.guard,
            compile_guard=self.compile_guard, stop_reason=reason,
            exhaustive_complete=bool(exhaustive_complete and not self.stopped),
            moves=tuple(self.moves),
            permitted_deletion_count=sum(self.permission_cache.values()),
            refused_deletion_count=sum(not allowed for allowed in self.permission_cache.values()),
            refusal=self.refusal,
            permission_counts_by_cardinality=tuple(
                (d, permission_counts[d, True], permission_counts[d, False])
                for d in sorted({d for d, _allowed in permission_counts})
            ),
        )


__all__ = ["SearchBudgetExhausted", "SearchLimits", "SearchResult", "SearchSession"]
