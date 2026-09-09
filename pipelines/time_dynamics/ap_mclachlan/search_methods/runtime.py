"""Connect search policies to shared evaluation and existing certification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..exchange_structural import StructuralEnumeration
from ..patch_evaluation import PatchEvaluator
from .problem import PatchProblem
from .scheduling import PATCHES_PER_TURN, SCHEDULE_ID
from .session import SearchLimits, SearchResult


@dataclass
class PatchSearch:
    policy: str
    problem: PatchProblem
    evaluator: PatchEvaluator
    ranker: Callable
    limits: SearchLimits
    result: SearchResult | None = None

    def families(self):
        """Finish the requested search before yielding one certification level."""

        if self.policy == "bounded_exhaustive":
            from .exhaustive import search_patches
        elif self.policy == "greedy":
            from .conditional_greedy import search_patches
        elif self.policy == "block_exchange":
            from .block_local_exchange import search_patches
        else:
            raise ValueError(f"unknown experimental search policy {self.policy!r}")
        self.result = search_patches(
            self.problem, self.evaluator, self.ranker, self.limits,
        )
        evidence = self.result.candidates
        telemetry = StructuralEnumeration(
            candidates=evidence, guard=self.result.guard,
            priorities={}, eligible_universe=tuple(self.problem.cuts_by_atom),
            frontier_schedule=(), frontiers_used=0,
            q_base=self.evaluator.q_base,
            scoring_workers_requested=self.evaluator.workers_requested,
            scoring_workers_effective_max=self.evaluator.workers_effective_max,
            compile_guard=self.result.compile_guard,
            compile_calls=self.evaluator.compile_calls_total,
        )
        # Telemetry is available even when the first finalist commits.
        yield "__telemetry__", (), telemetry
        yield self.policy, evidence, None

    def receipt(self) -> dict:
        if self.result is None:
            raise RuntimeError("search has not finished")
        return {
            "schema": "paper_ii_patch_search_v1",
            "search_policy": self.policy,
            "admission_policy": "incremental_complete_patch_v1",
            "rung_schedule": SCHEDULE_ID,
            "patches_per_rung_turn": PATCHES_PER_TURN,
            "gain_evaluation": self.evaluator.gain_evaluation,
            **self.result.receipt(),
            **({"move_size": 2} if self.policy == "block_exchange" else {}),
        }


def prepare_patch_search(
    *, search_policy: str, gain_evaluation: str,
    structural_kwargs: Mapping[str, Any], population_ranker: Callable | None,
) -> PatchSearch:
    """Bind one frozen problem; search modules receive no runtime controller."""

    bundle = structural_kwargs.get("route_bundle")
    if population_ranker is None or not getattr(bundle, "needs_compiled_cost", False):
        raise ValueError("experimental searches require complete-patch compiled-cost ranking")
    if structural_kwargs.get("gamma_rcond") is None:
        raise ValueError("experimental searches require the shared Gamma validation tolerance")
    permission = structural_kwargs.get("deletion_permission")
    cache = structural_kwargs["cache"]
    problem = PatchProblem(
        coordinate_keys=tuple(cache.coordinate_keys),
        deletable_indices=tuple(structural_kwargs["deletable_indices"]),
        min_surviving_support=int(structural_kwargs["min_surviving_support"]),
        candidate_pool_for_deletion=structural_kwargs["candidate_pool_for_deletion"],
        cuts_by_atom=structural_kwargs["cuts_by_atom"],
        max_insertion_batch_size=int(structural_kwargs.get("max_insertion_batch_size", 1)),
        max_deletion_batch_size=structural_kwargs.get("max_deletion_batch_size"),
        max_deletion_batch_fraction=0.15,
        tokens_commute=structural_kwargs.get("tokens_commute"),
        deletion_permission=(
            None if permission is None else lambda removed: bool(permission(removed).permitted)
        ),
    )
    evaluator = PatchEvaluator(
        cache=cache,
        base_K=structural_kwargs["base_K"], base_f=structural_kwargs["base_f"],
        norm_b_sq=structural_kwargs["norm_b_sq"],
        inverse_policy=structural_kwargs["inverse_policy"],
        weights=structural_kwargs["weights"],
        cuts_by_atom=structural_kwargs["cuts_by_atom"],
        insertion_cost=structural_kwargs["insertion_cost"],
        deletion_cost=structural_kwargs["deletion_cost"],
        deletion_conditioning=structural_kwargs.get("deletion_conditioning"),
        deletion_history_loss=structural_kwargs.get("deletion_history_loss"),
        scoring_workers=structural_kwargs.get("scoring_workers", 1),
        route_bundle=bundle, gamma_rcond=structural_kwargs.get("gamma_rcond"),
        candidate_cost_inputs=structural_kwargs.get("candidate_cost_inputs"),
        candidate_cost_warm=structural_kwargs.get("candidate_cost_warm"),
        population_accuracy_debt=structural_kwargs.get("population_accuracy_debt"),
        gain_evaluation=gain_evaluation,
    )
    return PatchSearch(
        policy=search_policy, problem=problem, evaluator=evaluator,
        ranker=population_ranker,
        limits=SearchLimits(
            joint_patch_evaluations=structural_kwargs.get("max_joint_patch_evaluations"),
            compiled_patches_per_checkpoint=structural_kwargs.get("max_compiled_patches_per_checkpoint"),
        ),
    )


__all__ = ["PatchSearch", "prepare_patch_search"]
