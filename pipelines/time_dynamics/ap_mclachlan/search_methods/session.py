"""Shared work admission and cumulative evidence for patch searches.

Search policies choose which complete patches to request. This module admits
whole requests, caches raw results, and supplies immutable population-ranking
snapshots. It contains no gain formula, compilation policy, or certification.
"""

from __future__ import annotations

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

    def receipt(self) -> dict:
        return {
            "stop_reason": self.stop_reason,
            "exhaustive_complete": self.exhaustive_complete,
            "evaluated_patch_count": len(self.candidates),
            "permitted_deletion_count": self.permitted_deletion_count,
            "refused_deletion_count": self.refused_deletion_count,
            "moves": list(self.moves),
            **({"refusal": dict(self.refusal)} if self.refusal is not None else {}),
        }


class SearchSession:
    """One search's raw evidence ledger and common computational guards.

    Each distinct requested patch consumes one structural slot. A deletion
    boundary refused by its measurement-free permission also consumes one
    slot, preventing an unbounded permission walk from evading work limits.
    Compiled slots count permitted regime-active patches, as in the current
    route. Actual transpilation calls remain recorded by the cost evaluator.

    An insertion request is admitted whole before numerical or compiler work.
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
        self.accounted_rejections: set[tuple[int, ...]] = set()
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
        # Construction stops early, so the full family's size is unknown.
        # Keep the old exact-size field unset and report what was observed.
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
        if removed in self.accounted_rejections or patch_identity(Patch(removed, ())) in self.evidence:
            return
        if self.guard.remaining == 0:
            self._refuse("joint_work_budget", f"boundary_d{len(removed)}")

    def deletion_permitted(self, removed: tuple[int, ...]) -> bool:
        """Return only mathematical admissibility; never evaluate a patch."""

        if removed not in self.permission_cache:
            self.permission_cache[removed] = self.problem.permitted(removed)
        return self.permission_cache[removed]

    def account_permission_rejection(self, removed: tuple[int, ...]) -> None:
        """Charge a refused boundary once, without caching a completed solve."""

        if self.permission_cache.get(removed) is not False:
            raise ValueError("only a mathematically refused boundary can be charged here")
        self.check_boundary_budget(removed)
        if removed in self.accounted_rejections:
            return
        if not self.guard.admit(f"boundary_d{len(removed)}", 1):
            raise AssertionError("boundary admission changed after preflight")
        self.accounted_rejections.add(removed)

    def evaluate(
        self, patches: Iterable[Patch], label: str,
    ) -> tuple[StructuralCandidate, ...]:
        """Acquire a complete request, with finite construction under a budget."""

        if self.stopped:
            raise SearchBudgetExhausted(self.stop_reason)
        keys: list[tuple] = []
        unseen: dict[tuple, Patch] = {}
        seen_request: set[tuple] = set()
        remaining = self.guard.remaining
        compiled_remaining = self.compile_guard.remaining
        compiled_size = 0
        for patch in patches:
            key = patch_identity(patch)
            if key in seen_request:
                continue
            seen_request.add(key)
            keys.append(key)
            if key not in self.evidence:
                unseen[key] = patch
                if remaining is not None and len(unseen) > remaining:
                    self._refuse("joint_work_budget", label, len(unseen))
                compiled_size += int(
                    bool(patch.removed_runtime_indices or patch.inserted_selection)
                    and self.evaluator.active_population_face(
                        patch.removed_runtime_indices, patch.inserted_selection,
                    )
                )
                if compiled_remaining is not None and compiled_size > compiled_remaining:
                    self._refuse("compiled_work_budget", label, compiled_size)
        if unseen:
            if not self.guard.admit(label, len(unseen)) or not self.compile_guard.admit(label, compiled_size):
                raise AssertionError("request admission changed after preflight")
            evaluated = iter(self.evaluator.evaluate(unseen.values(), label))
            for key in unseen:
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
        return SearchResult(
            candidates=tuple(self.evidence.values()), guard=self.guard,
            compile_guard=self.compile_guard, stop_reason=reason,
            exhaustive_complete=bool(exhaustive_complete and not self.stopped),
            moves=tuple(self.moves),
            permitted_deletion_count=sum(self.permission_cache.values()),
            refused_deletion_count=sum(not allowed for allowed in self.permission_cache.values()),
            refusal=self.refusal,
        )


__all__ = ["SearchBudgetExhausted", "SearchLimits", "SearchResult", "SearchSession"]
