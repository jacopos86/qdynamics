"""Structural enumeration and scoring for the deletion-conditioned selector.

This module owns the *classical frozen-ray search*: complete deletion rungs,
the complete singleton level, child acquisition priorities, nested frontiers,
and ordered multi-child insertion plans — everything up to (but excluding)
finalist materialization and certification.  No ANZATS is built here; every
quantity comes from the checkpoint structural cache.

Score model (specification "Scalar support-patch score"):

    q(D, I)    = Q(D, I) / (||b||^2 + eps_norm)          realized captured drift
    g_{I|D}    = [q(D, I) - q(D, 0)]+                    conditional insertion gain
    L_{D|I}    = [q(0, lift(I)) - q(D, I)]+              conditional deletion loss
    delta      = q(D, I) - q(0, 0)
    U_ins      = g / C_I^alpha_ins
    U_del      = C_D^alpha_del (1 + conditioning) / (L + history + eps_L)
    score      = U_ins + U_del + w_delta * delta

with U_del(0, I) = 0 and U_ins(D, 0) = 0.  Hardware-cost and
conditioning/history inputs are injected callables so the existing Paper-II
cost and conditioning components wire in unchanged at integration.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

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
    inserted_tokens,
    quotient_insertion_plans,
    survivor_tokens,
    word_keys,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
    StructuralInsertionCache,
    memoized_solve_metadata,
    structural_candidate_solve,
)


GENERALIZED_EXCHANGE_STRUCTURAL_POLICY_V2 = "paper_ii_generalized_exchange_structural_v2"


@dataclass(frozen=True)
class StructuralScoreWeights:
    """Weights of the scalar structural score; domains per the specification."""

    alpha_ins: float = 1.0
    alpha_del: float = 1.0
    w_delta: float = 1.0
    lambda_hist: float = 0.0
    lambda_cond_relief: float = 0.0
    lambda_cond_damage: float = 0.0
    epsilon_L: float = 1.0e-14
    epsilon_norm: float = 1.0e-12
    # Lexicographic accuracy-first ranking, used while the checkpoint is in
    # L^2 debt.  The default composite score cannot discriminate there: the
    # deletion loss is one-sided, l = [q(0,I) - q(D,I)]_+, so a deletion that
    # LOWERS L^2 and one that merely leaves it unchanged both score l = 0, and
    # that tie is then divided into the utility denominator, giving any
    # near-free deletion a score of order cost/epsilon_L ~ 1e14 against an
    # insertion utility of order gain/cost.  Cheap deletions therefore crowd
    # out the insertions that would actually pay the debt down.  Under
    # `debt_ranking` the primary key becomes the SIGNED normalized drift change
    # delta = q(D,I) - q(0,0), which is dL^2/2(||b||^2 + eps): candidates that
    # improve accuracy sort first, and cost, conditioning relief, and history
    # act only as tiebreakers among them, so cost can never re-invert an
    # accuracy decision.
    debt_ranking: bool = False
    # Two candidates whose signed delta agrees to within this are treated
    # as exactly tied on accuracy, and only then may utility order them.
    delta_rank_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if float(self.alpha_ins) < 0.0 or float(self.alpha_del) < 0.0:
            raise ValueError("cost exponents must be non-negative.")
        if (
            float(self.w_delta) < 0.0
            or float(self.lambda_hist) < 0.0
            or float(self.lambda_cond_relief) < 0.0
            or float(self.lambda_cond_damage) < 0.0
        ):
            raise ValueError("score weights must be non-negative.")
        if float(self.epsilon_L) <= 0.0 or float(self.epsilon_norm) <= 0.0:
            raise ValueError("epsilon_L and epsilon_norm must be positive.")


@dataclass(frozen=True)
class StructuralCandidate:
    """One scored typed patch ``(D, I)`` awaiting certification."""

    removed_runtime_indices: tuple[int, ...]
    inserted_selection: tuple[tuple[str, int], ...]
    plan: InsertionPlan | None
    family: str
    q: float
    insertion_gain: float
    deletion_loss: float
    delta: float
    insertion_utility: float
    deletion_utility: float
    score: float
    rank_primary: float = 0.0
    rank_secondary: float = 0.0

    @property
    def kind(self) -> str:
        removed = bool(self.removed_runtime_indices)
        inserted = bool(self.inserted_selection)
        if removed and inserted:
            return "exchange"
        if removed:
            return "delete"
        if inserted:
            return "insert"
        return "stay"

    @property
    def order_key(self) -> tuple:
        """Frozen tie order: deletion tuple, then a type-homogeneous plan key.

        Singleton selections and full-word plans encode into the same string
        shape so ties across candidate shapes always compare.
        """

        if self.plan is None:
            plan_key = tuple(
                ("sel", str(atom_id), f"{int(cut):06d}")
                for atom_id, cut in self.inserted_selection
            )
        else:
            plan_key = tuple(
                ("tok", str(token.kind), f"{int(token.sort_index):06d}", token.key)
                for token in self.plan.full_word
            )
        return (self.removed_runtime_indices, plan_key)


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
):
    """Yield ``(family_label, candidates, telemetry)`` in acquisition order.

    Families follow the specification's acquisition order — the ``d = 0``
    singleton family, each deletion-cardinality rung, then each insertion
    frontier — with the joint-work guard admitting every family whole before
    any member is scored.  The final yield is ``("__telemetry__", (), state)``
    carrying priorities, universe, schedule, and ``q_base`` so consumers can
    stop early without losing reproduction data.

    ``deletion_permission`` is the measurement-free eligibility seam for a
    complete deletion set.  A refused set is removed before its pure-deletion
    and exchange variants are scored or counted against the work guard.
    ``candidate_pool_for_deletion`` returns the branch pool ``C_{k,D}`` in
    frozen child order (occurrence policy applied by the caller; a deleted
    child may re-enter).  ``cuts_by_atom`` holds each child's retained
    original-layout cuts.  Multi-child plans (``|A| >= 2``) additionally need
    ``tokens_commute`` for the whole-word quotient.
    """

    n = int(base_f.reshape(-1).size)
    all_indices = tuple(range(n))
    guard = JointWorkGuard(max_joint_patch_evaluations=max_joint_patch_evaluations)
    conditioning = deletion_conditioning or (lambda removed: 0.0)
    history_loss = deletion_history_loss or (lambda removed: 0.0)

    def q_of(removed: tuple[int, ...], selection: tuple[tuple[str, int], ...]) -> float:
        keep = tuple(i for i in all_indices if i not in set(removed))
        _Q, q = structural_candidate_solve(
            cache=cache,
            base_K=base_K,
            base_f=base_f,
            norm_b_sq=float(norm_b_sq),
            keep_indices=keep,
            inserted_selection=selection,
            inverse_policy=inverse_policy,
            epsilon_norm=float(weights.epsilon_norm),
            memo_key=(removed, selection),
        )
        return float(q)

    q_base = q_of((), ())

    def scored(
        removed: tuple[int, ...],
        selection: tuple[tuple[str, int], ...],
        plan: InsertionPlan | None,
        family: str,
    ) -> StructuralCandidate:
        q_joint = q_of(removed, selection)
        delta = q_joint - q_base
        if selection:
            gain = max(0.0, q_joint - q_of(removed, ()))
            atom_ids = tuple(atom_id for atom_id, _cut in selection)
            cost_ins = max(1.0, float(insertion_cost(atom_ids)))
            insertion_utility = gain / (cost_ins ** float(weights.alpha_ins))
        else:
            gain = 0.0
            insertion_utility = 0.0
        if removed:
            # One-sided by construction.  Under exact projection the
            # unclipped quantity is r_{D|R}^T S_{D|R}^+ r_{D|R} >= 0
            # (schur_identity.exact_deletion_loss), so clipping is
            # exact-arithmetic-faithful -- but the realized solve is
            # regularized, where a deletion CAN lower L^2, and the clip
            # then maps "helpful" and "harmless" onto the same zero.
            # That collapse is why `debt_ranking` exists: while in L^2
            # debt the signed delta below, not this loss, is the
            # primary key.
            loss = max(0.0, q_of((), selection) - q_joint)
            cost_del = max(1.0, float(deletion_cost(removed)))
            # Conditioning relief/damage from solves the enumeration already
            # performed: the retained log-condition of the base support versus
            # the deletion branch.  Both memo entries exist because q_of ran.
            cond_base, _rank_b = memoized_solve_metadata(cache, ((), ()))
            cond_del, _rank_d = memoized_solve_metadata(cache, (removed, ()))
            relief = damage = 0.0
            if cond_base is not None and cond_del is not None and cond_base > 0 and cond_del > 0:
                log_shift = math.log10(cond_base) - math.log10(cond_del)
                relief = max(0.0, log_shift)
                damage = max(0.0, -log_shift)
            deletion_utility = (
                (cost_del ** float(weights.alpha_del))
                * (
                    1.0
                    + float(weights.lambda_cond_relief) * relief
                    + max(0.0, float(conditioning(removed)))
                )
                / (
                    loss
                    + float(weights.lambda_hist) * max(0.0, float(history_loss(removed)))
                    + float(weights.lambda_cond_damage) * damage
                    + float(weights.epsilon_L)
                )
            )
        else:
            loss = 0.0
            deletion_utility = 0.0
        score = (
            insertion_utility
            + deletion_utility
            + float(weights.w_delta) * delta
        )
        if bool(weights.debt_ranking):
            # True lexicographic key. A scalar encoding (delta + eps*tanh(u))
            # only approximates this: it silently degrades to the utility
            # ordering whenever two deltas differ by less than eps, which is
            # exactly the regime where the divergent deletion utility is most
            # dangerous. Quantizing delta makes "tied on accuracy" an explicit,
            # declared tolerance rather than a floating-point accident.
            tol = float(weights.delta_rank_tolerance)
            rank_primary = float(round(float(delta) / tol)) if tol > 0.0 else float(delta)
            rank_secondary = float(np.tanh(insertion_utility + deletion_utility))
        else:
            rank_primary = float(score)
            rank_secondary = 0.0
        return StructuralCandidate(
            removed_runtime_indices=removed,
            inserted_selection=tuple(selection),
            plan=plan,
            family=str(family),
            q=q_joint,
            insertion_gain=gain,
            deletion_loss=loss,
            delta=delta,
            insertion_utility=insertion_utility,
            deletion_utility=deletion_utility,
            score=float(score),
            rank_primary=float(rank_primary),
            rank_secondary=float(rank_secondary),
        )

    candidates: list[StructuralCandidate] = []
    singleton_best: dict[str, float] = {}

    def note_priority(atom_id: str, score: float) -> None:
        if np.isfinite(float(score)):
            prior = singleton_best.get(atom_id)
            if prior is None or float(score) > prior:
                singleton_best[atom_id] = float(score)

    # ---- d = 0: stay + every pure singleton insertion ----------------------
    pure_pool = candidate_pool_for_deletion(())
    d0_size = 1 + sum(len(tuple(cuts_by_atom.get(a, ()))) for a in pure_pool)
    if guard.admit("singleton_d0", d0_size):
        family_out = [scored((), (), None, "singleton_d0")]
        for atom_id in pure_pool:
            for cut in sorted(int(c) for c in cuts_by_atom.get(atom_id, ())):
                candidate = scored(
                    (), ((str(atom_id), cut),), None, "singleton_d0"
                )
                family_out.append(candidate)
                note_priority(str(atom_id), candidate.score)
        candidates.extend(family_out)
        yield "singleton_d0", tuple(family_out), None

    # ---- d >= 1 rungs: pure deletion + singleton exchange ------------------
    for d, rung in iter_deletion_rungs(
        deletable_indices,
        total_support=n,
        min_surviving_support=int(min_surviving_support),
    ):
        if d == 0:
            continue
        family = f"singleton_d{d}"
        size = 0
        members: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        for removed in rung:
            pool = candidate_pool_for_deletion(removed)
            members.append((removed, pool))
            size += 1 + sum(len(tuple(cuts_by_atom.get(a, ()))) for a in pool)
        # Admit the raw family before permission. Otherwise refused sets evade
        # the work accounting and the search can walk unbounded combinatorial
        # rungs while scoring nothing.
        if not guard.admit(family, size):
            break
        family_out = []
        for removed, pool in members:
            if deletion_permission is not None:
                permission = deletion_permission(removed)
                if not bool(permission.permitted):
                    continue
            family_out.append(scored(removed, (), None, family))
            for atom_id in pool:
                for cut in sorted(int(c) for c in cuts_by_atom.get(atom_id, ())):
                    candidate = scored(
                        removed, ((str(atom_id), cut),), None, family
                    )
                    family_out.append(candidate)
                    note_priority(str(atom_id), candidate.score)
        candidates.extend(family_out)
        yield family, tuple(family_out), None

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
        survivors = survivor_tokens(cache.coordinate_keys)
        scored_plans: set[tuple] = set()
        from itertools import combinations

        for level, width in enumerate(schedule):
            frontier = eligible[:width]
            family = f"frontier_{level}_w{width}"
            batch: list[
                tuple[tuple[int, ...], InsertionPlan, tuple[tuple[str, int], ...]]
            ] = []
            # Construction is pure Python and can vastly exceed the solve
            # budget (plans scale with cut-position combinations), so cap it
            # at the guard's remaining budget: crossing the line rejects the
            # whole family with the constructed part discarded — families
            # still admit whole or not at all.
            construction_cap = guard.remaining
            over_cap = False
            for removed in admitted_removed:
                if over_cap:
                    break
                pool = set(candidate_pool_for_deletion(removed))
                usable = [a for a in frontier if a in pool]
                removed_keys = tuple(
                    cache.coordinate_keys[i] for i in removed
                )
                for size in range(2, int(max_insertion_batch_size) + 1):
                    if over_cap:
                        break
                    for subset in combinations(usable, size):
                        if over_cap:
                            break
                        tokens = inserted_tokens(subset)
                        plans = quotient_insertion_plans(
                            survivors=survivors,
                            inserted=tokens,
                            removed_keys=removed_keys,
                            tokens_commute=tokens_commute,
                        )
                        for plan in plans:
                            selection = _plan_selection(plan, cuts_by_atom)
                            identity = (removed, plan.plan_id)
                            if identity in scored_plans:
                                continue
                            scored_plans.add(identity)
                            batch.append((removed, plan, selection))
                            if (
                                construction_cap is not None
                                and len(batch) > int(construction_cap)
                            ):
                                over_cap = True
                                break
            if over_cap:
                guard.admit(family, len(batch))
                break
            if not guard.admit(family, len(batch)):
                break
            frontiers_used = level + 1
            family_out = [
                scored(removed, selection, plan, family)
                for removed, plan, selection in batch
            ]
            candidates.extend(family_out)
            yield family, tuple(family_out), None

    yield "__telemetry__", (), StructuralEnumeration(
        candidates=tuple(candidates),
        guard=guard,
        priorities=dict(singleton_best),
        eligible_universe=eligible,
        frontier_schedule=schedule,
        frontiers_used=frontiers_used,
        q_base=q_base,
    )


def enumerate_structural_candidates(**kwargs) -> StructuralEnumeration:
    """Collect every admitted family; the non-lazy view of the same search."""

    telemetry: StructuralEnumeration | None = None
    for family, _candidates, final in iter_structural_families(**kwargs):
        if family == "__telemetry__":
            telemetry = final
    assert telemetry is not None
    return telemetry


def _plan_selection(
    plan: InsertionPlan,
    cuts_by_atom: Mapping[str, Sequence[int]],
) -> tuple[tuple[str, int], ...]:
    """Map a plan's full word to cached ``(atom_id, representative cut)`` order.

    An inserted token's raw cut is the number of survivors preceding it in the
    full word.  Every inserted coordinate is zero angle, so other inserted
    tokens act as the identity on its tangent, and a certified commuting slide
    across survivors preserves the tangent exactly; the column therefore
    equals the cached one at the token's class representative — the largest
    retained cut not exceeding the raw cut.
    """

    selection: list[tuple[str, int]] = []
    survivor_count = 0
    for token in plan.full_word:
        if token.kind == "survivor":
            survivor_count += 1
        else:
            retained = sorted(int(c) for c in cuts_by_atom.get(token.key, ()))
            representative = max(
                (c for c in retained if c <= survivor_count), default=None
            )
            if representative is None:
                raise ValueError(
                    f"no retained cut at or before {survivor_count} for "
                    f"{token.key!r}; retained: {retained}."
                )
            selection.append((token.key, int(representative)))
    return tuple(selection)


__all__ = [
    "GENERALIZED_EXCHANGE_STRUCTURAL_POLICY_V2",
    "iter_structural_families",
    "StructuralCandidate",
    "StructuralEnumeration",
    "StructuralScoreWeights",
    "enumerate_structural_candidates",
    "resolve_frontier_schedule",
]
