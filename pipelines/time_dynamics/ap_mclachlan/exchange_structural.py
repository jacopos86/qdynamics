"""Structural search schedule for the deletion-conditioned selector.

Owns complete deletion rungs, singleton acquisition priorities, nested
insertion frontiers, and their work guards. Complete-patch mathematics and
compiled-resource evaluation are supplied by ``PatchEvaluator``; population
ranking and finalist certification remain in the selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.deletion_family import (
    JointWorkGuard,
    iter_deletion_rungs,
)
from pipelines.time_dynamics.ap_mclachlan.deletion_permission import (
    DeletionPermissionDecision,
)
from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    InsertionPlan,
    WordToken,
)
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import (
    CandidateCostInputs, PatchEvaluator, StructuralCandidate, StructuralScoreWeights,
)
from pipelines.time_dynamics.ap_mclachlan.patch_enumeration import (
    Patch,
    iter_permitted_patches,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
    StructuralInsertionCache,
)


GENERALIZED_EXCHANGE_STRUCTURAL_POLICY_V2 = "paper_ii_generalized_exchange_structural_v2"


@dataclass
class StructuralEnumeration:
    """Ordered structural candidates plus reproduction telemetry."""

    candidates: tuple[StructuralCandidate, ...]
    guard: JointWorkGuard
    priorities: Mapping[str, float]
    eligible_universe: tuple[str, ...]
    frontier_schedule: tuple[int, ...]
    frontiers_used: int
    q_base: float
    scoring_workers_requested: int = 1
    scoring_workers_effective_max: int = 1
    compile_guard: JointWorkGuard | None = None
    compile_calls: int = 0
    exchange_frontier: tuple[tuple[str, int], ...] | None = None

    def ranked(self, *, score_floor: float = 0.0) -> tuple[StructuralCandidate, ...]:
        """Non-stay candidates above the floor, score-descending, frozen ties."""

        eligible = [
            c
            for c in self.candidates
            if c.kind != "stay"
            and np.isfinite(float(c.score))
            and float(c.score) > float(score_floor)
        ]
        return tuple(
            sorted(eligible, key=lambda c: (-float(c.score), c.order_key))
        )


def resolve_frontier_schedule(
    widths: Sequence[int] | None,
    *,
    universe_size: int,
) -> tuple[int, ...]:
    """Explicit strictly-increasing widths, or the default 2,4,8,...,|E_k|."""

    n = int(universe_size)
    if n <= 0:
        return ()
    if widths is None:
        out: list[int] = []
        width = 2
        while width < n:
            out.append(width)
            width *= 2
        out.append(n)
        return tuple(dict.fromkeys(out))
    values = [int(w) for w in widths]
    if any(w <= 0 for w in values):
        raise ValueError("frontier widths must be positive.")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("frontier widths must be strictly increasing.")
    clipped: list[int] = []
    for width in values:
        clipped.append(min(width, n))
        if clipped[-1] == n:
            break
    return tuple(dict.fromkeys(clipped))


def iter_structural_families(
    *,
    cache: StructuralInsertionCache,
    base_K: np.ndarray,
    base_f: np.ndarray,
    norm_b_sq: float,
    inverse_policy: McLachlanInversePolicy,
    weights: StructuralScoreWeights,
    deletable_indices: Sequence[int],
    min_surviving_support: int,
    cuts_by_atom: Mapping[str, Sequence[int]],
    candidate_pool_for_deletion: Callable[[tuple[int, ...]], tuple[str, ...]],
    insertion_cost: Callable[[tuple[str, ...]], float],
    deletion_cost: Callable[[tuple[int, ...]], float],
    deletion_permission: (
        Callable[[tuple[int, ...]], DeletionPermissionDecision] | None
    ) = None,
    deletion_conditioning: Callable[[tuple[int, ...]], float] | None = None,
    deletion_history_loss: Callable[[tuple[int, ...]], float] | None = None,
    tokens_commute: Callable[[WordToken, WordToken], bool] | None = None,
    max_insertion_batch_size: int = 1,
    interaction_frontier_widths: Sequence[int] | None = None,
    max_joint_patch_evaluations: int | None = None,
    scoring_workers: int = 1,
    route_bundle: Any | None = None,
    gamma_rcond: float | None = None,
    candidate_cost_inputs: (
        Callable[
            [tuple[int, ...], tuple[tuple[str, int], ...], InsertionPlan | None],
            CandidateCostInputs,
        ]
        | None
    ) = None,
    max_compiled_patches_per_checkpoint: int | None = None,
    candidate_cost_warm: (
        Callable[[Sequence[tuple[tuple[int, ...], tuple[tuple[str, int], ...], Any]]], Any]
        | None
    ) = None,
    population_accuracy_debt: bool | None = None,
    exchange_insertion_frontier: int | None = None,
):
    """Yield ``(family_label, candidates, telemetry)`` in acquisition order.

    Families follow the specification's acquisition order — the ``d = 0``
    singleton family, each deletion-cardinality rung, then each insertion
    frontier — with the joint-work guard admitting every family whole before
    any member is scored. ``candidates`` is a lazy member iterator: consume
    it completely before requesting another family. Reading a frontier label
    does no joint construction, solving, or compilation. The final yield is
    ``("__telemetry__", (), state)`` with completed-search accounting.

    ``deletion_permission`` is the measurement-free eligibility seam for a
    complete deletion set. A refused set is removed before its branch pool
    is built. The structural work guard counts only permitted complete
    patches; the compilation guard counts their regime-active subset.
    Permission checks consume no patch-evaluation slots.
    ``candidate_pool_for_deletion`` returns the branch pool ``C_{k,D}`` in
    frozen child order (occurrence policy applied by the caller; a deleted
    child may re-enter).  ``cuts_by_atom`` holds each child's retained
    original-layout cuts.  Multi-child plans (``|A| >= 2``) additionally need
    ``tokens_commute`` for the whole-word quotient.

    ``exchange_insertion_frontier`` declares how many positioned insertions a
    ``d >= 1`` deletion set is paired with.  ``None`` pairs it with the whole
    branch pool, so a family's size sums the pure-deletion boundary and
    positioned insertions over permitted deletion sets. A finite width ``W``
    pairs it with the ``W`` highest-scoring ``(atom, cut)`` positions of the
    already-scored ``d = 0`` family, sealed before the first rung is built,
    so a family declares at most ``N_permitted x (1 + W)`` instead. The width
    is run configuration and is never derived from the residual budget, so
    the enumerated candidate set stays monotone in
    ``max_joint_patch_evaluations``.
    """

    n = int(base_f.reshape(-1).size)
    guard = JointWorkGuard(max_joint_patch_evaluations=max_joint_patch_evaluations)
    evaluator = PatchEvaluator(
        cache=cache, base_K=base_K, base_f=base_f, norm_b_sq=norm_b_sq,
        inverse_policy=inverse_policy, weights=weights, cuts_by_atom=cuts_by_atom,
        insertion_cost=insertion_cost, deletion_cost=deletion_cost,
        deletion_conditioning=deletion_conditioning,
        deletion_history_loss=deletion_history_loss,
        scoring_workers=scoring_workers, route_bundle=route_bundle,
        gamma_rcond=gamma_rcond, candidate_cost_inputs=candidate_cost_inputs,
        candidate_cost_warm=candidate_cost_warm,
        population_accuracy_debt=population_accuracy_debt,
    )
    population_scoring = evaluator.population_scoring
    active_population_face = evaluator.active_population_face
    # A second whole-family guard for compiled-cost work; ``None`` leaves it
    # unbounded (small systems and smokes).  Counted in candidates, like the
    # joint-work guard, so both admit or reject the same families.
    compile_guard = (
        JointWorkGuard(max_joint_patch_evaluations=max_compiled_patches_per_checkpoint)
        if population_scoring and candidate_cost_inputs is not None
        else None
    )
    q_base = evaluator.q_base

    def guard_fits(candidate_guard: JointWorkGuard, size: int) -> bool:
        return bool(
            not candidate_guard.exhausted
            and (
                candidate_guard.max_joint_patch_evaluations is None
                or candidate_guard.scored_count + int(size)
                <= int(candidate_guard.max_joint_patch_evaluations)
            )
        )

    def reject_family(candidate_guard: JointWorkGuard, family: str) -> None:
        if candidate_guard.rejected_family is None:
            candidate_guard.rejected_family = str(family)

    def admit(family: str, size: int, *, compiled_size: int | None = None) -> bool:
        compile_size = int(size if compiled_size is None else compiled_size)
        if not guard_fits(guard, int(size)):
            reject_family(guard, family)
            if compile_guard is not None:
                reject_family(compile_guard, family)
            return False
        if compile_guard is not None and not guard_fits(compile_guard, compile_size):
            # Admission is transactional: neither counter advances when one
            # of the two complete-family budgets rejects the family.
            reject_family(guard, family)
            reject_family(compile_guard, family)
            return False
        if not guard.admit(family, int(size)):
            raise AssertionError("joint-work guard changed after admission preflight")
        if compile_guard is not None and not compile_guard.admit(
            family, compile_size
        ):
            raise AssertionError("compile guard changed after admission preflight")
        return True

    def priority_value(candidate: StructuralCandidate) -> float:
        # Child acquisition must be route-neutral: Q and Gamma selectors seal
        # different ordinates only after they have measured the same frontier.
        return float(candidate.score)

    candidates: list[StructuralCandidate] = []
    singleton_best: dict[str, float] = {}
    singleton_cut_best: dict[tuple[str, int], float] = {}

    def note_priority(atom_id: str, score: float) -> None:
        if np.isfinite(float(score)):
            prior = singleton_best.get(atom_id)
            if prior is None or float(score) > prior:
                singleton_best[atom_id] = float(score)

    def note_cut_priority(atom_id: str, cut: int, score: float) -> None:
        """Per-position ``d = 0`` rank: the frontier's only ordering signal."""

        if np.isfinite(float(score)):
            key = (str(atom_id), int(cut))
            prior = singleton_cut_best.get(key)
            if prior is None or float(score) > prior:
                singleton_cut_best[key] = float(score)

    def singleton_patches(members) -> Iterator[Patch]:
        """Apply the declared positioned-singleton order to the pure kernel."""

        for removed, insertions in members:
            # Boundary patch: stay for D=(), otherwise pure deletion.
            yield from iter_permitted_patches(
                coordinate_keys=cache.coordinate_keys,
                deletion_sets=(removed,),
                insertion_blocks_for_deletion=lambda _d: ((),),
                singleton_cuts=lambda _d, _a: (),
            )
            # A sealed frontier can interleave cuts of different children.
            for atom_id, cut in insertions:
                yield from iter_permitted_patches(
                    coordinate_keys=cache.coordinate_keys,
                    deletion_sets=(removed,),
                    insertion_blocks_for_deletion=lambda _d: ((atom_id,),),
                    singleton_cuts=lambda _d, _a: (cut,),
                )

    def evaluate_patches(
        patches: Iterable[Patch], family: str
    ) -> Iterator[StructuralCandidate]:
        """Record acquisition priorities from the shared patch evaluator."""

        for candidate in evaluator.evaluate(patches, family):
            selection = candidate.inserted_selection
            if len(selection) == 1:
                atom_id, cut = selection[0]
                priority = priority_value(candidate)
                note_priority(atom_id, priority)
                if not candidate.removed_runtime_indices:
                    note_cut_priority(atom_id, cut, priority)
            candidates.append(candidate)
            yield candidate

    # ---- d = 0: stay + every pure singleton insertion ----------------------
    pure_pool = candidate_pool_for_deletion(())
    pure_insertions = tuple(
        (str(atom_id), int(cut))
        for atom_id in pure_pool
        for cut in sorted(int(c) for c in cuts_by_atom.get(atom_id, ()))
    )
    d0_size = 1 + len(pure_insertions)
    d0_compile_size = sum(
        int(active_population_face((), (insertion,)))
        for insertion in pure_insertions
    )
    if admit("singleton_d0", d0_size, compiled_size=d0_compile_size):
        yield "singleton_d0", evaluate_patches(
            singleton_patches((((), pure_insertions),)), "singleton_d0"
        ), None

    # ---- sealed exchange insertion frontier --------------------------------
    # Sealed HERE, before the first rung is sized, from scores the d = 0 family
    # has already paid for: no new solve, no new measurement, and no dependence
    # on rung enumeration order.  ``None`` reproduces the unrestricted product
    # exactly, which is what the oracle and every small-system caller use.
    if exchange_insertion_frontier is None:
        exchange_frontier: tuple[tuple[str, int], ...] | None = None
    else:
        frontier_width = int(exchange_insertion_frontier)
        if frontier_width < 1:
            raise ValueError("exchange_insertion_frontier must be at least 1.")
        exchange_frontier = tuple(
            sorted(
                singleton_cut_best,
                key=lambda p: (-singleton_cut_best[p], str(p[0]), int(p[1])),
            )
        )[:frontier_width]

    def rung_insertions(pool: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
        """Positioned insertions offered to one deletion set, in frozen order."""

        if exchange_frontier is None:
            return tuple(
                (str(atom_id), int(cut))
                for atom_id in pool
                for cut in sorted(int(c) for c in cuts_by_atom.get(atom_id, ()))
            )
        allowed = set(pool)
        return tuple(p for p in exchange_frontier if p[0] in allowed)

    # ---- d >= 1 rungs: pure deletion + singleton exchange ------------------
    for d, rung in iter_deletion_rungs(
        deletable_indices,
        total_support=n,
        min_surviving_support=int(min_surviving_support),
    ):
        if d == 0:
            continue
        family = f"singleton_d{d}"
        # Do not request unseen permissions once no evaluation can fit.
        # Raw deletion cardinality is not a lower bound on permitted work.
        if guard.exhausted or guard.remaining == 0:
            reject_family(guard, family)
            if compile_guard is not None:
                reject_family(compile_guard, family)
            break
        size = 0
        compiled_size = 0
        permitted_members: list[
            tuple[tuple[int, ...], tuple[tuple[str, int], ...]]
        ] = []
        for removed in rung:
            if deletion_permission is not None and not bool(
                deletion_permission(removed).permitted
            ):
                continue
            insertions = rung_insertions(candidate_pool_for_deletion(removed))
            permitted_members.append((removed, insertions))
            size += 1 + len(insertions)
            if compile_guard is not None:
                compiled_size += int(active_population_face(removed, ()))
                compiled_size += sum(
                    1
                    for atom_id, cut in insertions
                    if active_population_face(
                        removed, ((str(atom_id), int(cut)),)
                    )
                )
            # Stop at the first proof that the whole permitted rung cannot
            # fit. Neither counter reserves a partial rung.
            if not guard_fits(guard, size) or (
                compile_guard is not None
                and not guard_fits(compile_guard, compiled_size)
            ):
                admit(family, size, compiled_size=compiled_size)
                break
        if guard.exhausted or not admit(family, size, compiled_size=compiled_size):
            break
        yield family, evaluate_patches(
            singleton_patches(permitted_members), family
        ), None

    # ---- child acquisition priorities and nested frontiers -----------------
    eligible = tuple(
        sorted(singleton_best, key=lambda a: (-singleton_best[a], str(a)))
    )
    schedule = resolve_frontier_schedule(
        interaction_frontier_widths, universe_size=len(eligible)
    )
    frontiers_used = 0

    admitted_removed: list[tuple[int, ...]] = sorted(
        {c.removed_runtime_indices for c in candidates}
    )
    if (
        int(max_insertion_batch_size) >= 2
        and eligible
        and tokens_commute is not None
        and not guard.exhausted
    ):
        scored_plans: set[tuple] = set()

        def evaluate_frontier(level: int, width: int) -> Iterator[StructuralCandidate]:
            """Construct, admit, then evaluate one explicitly requested frontier."""

            nonlocal frontiers_used
            frontier = eligible[:width]
            family = f"frontier_{level}_w{width}"

            def insertion_blocks(removed):
                pool = set(candidate_pool_for_deletion(removed))
                usable = tuple(a for a in frontier if a in pool)
                for size in range(2, int(max_insertion_batch_size) + 1):
                    yield from combinations(usable, size)

            patches = iter_permitted_patches(
                coordinate_keys=cache.coordinate_keys,
                deletion_sets=admitted_removed,
                insertion_blocks_for_deletion=insertion_blocks,
                singleton_cuts=lambda _d, a: cuts_by_atom.get(a, ()),
                tokens_commute=tokens_commute,
            )
            batch: list[Patch] = []
            construction_cap = guard.remaining
            compile_cap = None if compile_guard is None else compile_guard.remaining
            batch_compile_size = 0
            for patch in patches:
                identity = (patch.removed_runtime_indices, patch.plan.plan_id)
                if identity in scored_plans:
                    continue
                scored_plans.add(identity)
                batch.append(patch)
                batch_compile_size += int(
                    active_population_face(patch.removed_runtime_indices, patch.inserted_selection)
                )
                # Stop construction at the first proof that this complete
                # family cannot fit. No member is scored or compiled.
                if (
                    construction_cap is not None and len(batch) > construction_cap
                ) or (
                    compile_cap is not None and batch_compile_size > compile_cap
                ):
                    admit(family, len(batch), compiled_size=batch_compile_size)
                    return
            if not admit(family, len(batch), compiled_size=batch_compile_size):
                return
            frontiers_used = level + 1
            yield from evaluate_patches(batch, family)

        for level, width in enumerate(schedule):
            # Looking ahead reads this label only. Construction, solves and
            # compilation start when the selector consumes the member stream.
            yield f"frontier_{level}_w{width}", evaluate_frontier(level, width), None
            if guard.exhausted:
                break

    yield "__telemetry__", (), StructuralEnumeration(
        candidates=tuple(candidates),
        guard=guard,
        priorities=dict(singleton_best),
        eligible_universe=eligible,
        frontier_schedule=schedule,
        frontiers_used=frontiers_used,
        q_base=q_base,
        scoring_workers_requested=evaluator.workers_requested,
        scoring_workers_effective_max=evaluator.workers_effective_max,
        compile_guard=compile_guard,
        compile_calls=int(evaluator.compile_calls_total),
        exchange_frontier=exchange_frontier,
    )


def enumerate_structural_candidates(**kwargs) -> StructuralEnumeration:
    """Collect every admitted family; the non-lazy view of the same search."""

    telemetry: StructuralEnumeration | None = None
    for family, members, final in iter_structural_families(**kwargs):
        if family == "__telemetry__":
            telemetry = final
        else:
            for _candidate in members:
                pass
    assert telemetry is not None
    return telemetry


__all__ = [
    "GENERALIZED_EXCHANGE_STRUCTURAL_POLICY_V2",
    "iter_structural_families",
    "StructuralEnumeration",
    "enumerate_structural_candidates",
    "resolve_frontier_schedule",
]
