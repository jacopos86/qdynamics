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

from concurrent.futures import ThreadPoolExecutor
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
from pipelines.time_dynamics.ap_mclachlan.compiled_patch_cost import (
    CompiledPatchCostError,
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
from pipelines.time_dynamics.ap_mclachlan.performance import count, phase
from pipelines.time_dynamics.ap_mclachlan.objective_ordinates import (
    GammaMonotonicityError,
    check_gamma_nested_monotonicity,
)
from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
    StructuralInsertionCache,
    memoized_gamma,
    memoized_solve_metadata,
    structural_candidate_solve,
)


GENERALIZED_EXCHANGE_STRUCTURAL_POLICY_V2 = "paper_ii_generalized_exchange_structural_v2"


def _scoring_worker_count(requested: int, task_count: int) -> int:
    """Effective thread count for one family's memo warm.

    Mirrors the trajectory-level clamp semantics: one task or one requested
    worker means the serial path; otherwise never more workers than tasks.
    """

    if int(requested) < 1:
        raise ValueError("scoring_workers must be positive.")
    if int(task_count) <= 1 or int(requested) <= 1:
        return 1
    return min(int(requested), int(task_count))


def _warm_solve_memo(
    q_of: Callable[[tuple[int, ...], tuple[tuple[str, int], ...]], float],
    keys: Sequence[tuple[tuple[int, ...], tuple[tuple[str, int], ...]]],
    solve_memo: Mapping[tuple, Any],
    requested_workers: int,
) -> int:
    """Speculatively populate the checkpoint solve memo for one family.

    Workers only *warm* the memo: every key is deduplicated first-occurrence
    and filtered against entries that already exist, so each remaining key is
    solved exactly once, by exactly one worker, writing a disjoint memo entry.
    The authoritative scoring loop then replays serially with pure memo hits,
    which keeps candidates, scores, ordering, telemetry, and exception
    positions bitwise identical to the single-worker path.  A worker failure
    writes nothing; the serial replay recomputes that key at the exact
    position the serial code would have, raising the identical exception.

    Returns the effective worker count used (1 means the warm was skipped).
    """

    missing = [key for key in dict.fromkeys(keys) if key not in solve_memo]
    workers = _scoring_worker_count(requested_workers, len(missing))
    if workers <= 1:
        return 1

    def warm_one(key: tuple) -> None:
        try:
            q_of(key[0], key[1])
        except Exception:
            # Fail closed later, deterministically: the serial replay hits
            # the missing memo entry and raises at the serial position.
            pass

    with phase("patch.exchange.solve_warm"):
        count("patch.exchange.solve_warm.tasks", len(missing))
        count(f"patch.exchange.solve_warm.worker_count_{int(workers)}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(warm_one, missing))
    return int(workers)


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
    # Population-scoring bundles only (None on the historical route):
    # ideal-projection ordinate m_Gamma, the signed compiled resource
    # marginal (dN2q, dD2q, dDc) of the COMPLETE patch, the materialization
    # cut chosen per inserted atom inside its commutation class, the legacy
    # proxy raw components for population repair, and the patched-word hash.
    m_gamma: float | None = None
    cost_marginal: tuple[int, int, int] | None = None
    placement_cuts: tuple[tuple[str, int], ...] | None = None
    proxy_insert_components: Mapping[str, float] | None = None
    proxy_delete_components: Mapping[str, float] | None = None
    patch_word_sha256: str | None = None
    cost_error: str | None = None
    ordinate_error: str | None = None

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


@dataclass(frozen=True)
class CandidateCostInputs:
    """Cost-side inputs the integration adapter attaches to one candidate.

    ``cost_marginal`` and ``placement_cuts`` come from the compiled-cost
    bundles; the two ``proxy_*_components`` mappings feed the legacy
    population repair.  Fields the bundle does not use stay ``None``.
    """

    cost_marginal: tuple[int, int, int] | None = None
    placement_cuts: tuple[tuple[str, int], ...] | None = None
    proxy_insert_components: Mapping[str, float] | None = None
    proxy_delete_components: Mapping[str, float] | None = None
    patch_word_sha256: str | None = None
    compile_calls: int = 0


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
    workers_requested = int(scoring_workers)
    if workers_requested < 1:
        raise ValueError("scoring_workers must be positive.")
    workers_effective_max = 1
    population_scoring = bool(
        route_bundle is not None and getattr(route_bundle, "uses_population_scoring", False)
    )
    if population_scoring and candidate_cost_inputs is None:
        raise ValueError(
            "population-scoring route bundles require candidate_cost_inputs."
        )
    # A second whole-family guard for compiled-cost work; ``None`` leaves it
    # unbounded (small systems and smokes).  Counted in candidates, like the
    # joint-work guard, so both admit or reject the same families.
    compile_guard = (
        JointWorkGuard(max_joint_patch_evaluations=max_compiled_patches_per_checkpoint)
        if population_scoring and candidate_cost_inputs is not None
        else None
    )
    compile_calls_total = 0
    gamma_denominator = float(norm_b_sq) + max(0.0, float(weights.epsilon_norm))

    def active_population_face(
        removed: tuple[int, ...], selection: tuple[tuple[str, int], ...]
    ) -> bool:
        if not population_scoring or population_accuracy_debt is None:
            return True
        if bool(population_accuracy_debt):
            return bool(selection)
        return bool(removed) and not bool(selection)

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
            gamma_rcond=gamma_rcond,
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
        m_gamma = None
        cost_inputs = None
        cost_error = None
        ordinate_error = None
        if population_scoring and active_population_face(removed, selection):
            nonlocal compile_calls_total
            if gamma_rcond is not None:
                gamma_joint = memoized_gamma(cache, (removed, selection))
                gamma_base = memoized_gamma(cache, ((), ()))
                if gamma_joint is not None and gamma_base is not None:
                    if not removed and selection:
                        # Pure insertion: J is a subset of J^B, so ideal
                        # Gamma must not decrease.  A relative-cutoff rank
                        # transition can violate that identity for one
                        # candidate; reject that candidate from the shared
                        # Q/Gamma population rather than aborting the whole
                        # checkpoint (or clipping the ordinate).
                        try:
                            check_gamma_nested_monotonicity(
                                gamma_base=float(gamma_base),
                                gamma_patched=float(gamma_joint),
                                denominator=gamma_denominator,
                                rcond=float(gamma_rcond),
                            )
                        except GammaMonotonicityError as exc:
                            ordinate_error = f"{type(exc).__name__}:{exc}"
                    if ordinate_error is None:
                        m_gamma = float(
                            (gamma_joint - gamma_base) / gamma_denominator
                        )
            # Price every regime-active candidate even when Gamma rejected
            # it.  The parallel path has already warmed the complete admitted
            # family, so replaying the cost lookup here preserves serial /
            # parallel compile accounting and attaches the same marginal for
            # audit.  The population ranker still excludes ordinate errors.
            if candidate_cost_inputs is not None and (removed or selection):
                try:
                    cost_inputs = candidate_cost_inputs(removed, tuple(selection), plan)
                    compile_calls_total += int(cost_inputs.compile_calls)
                except CompiledPatchCostError as exc:
                    cost_error = f"{type(exc).__name__}:{exc}"
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
            m_gamma=m_gamma,
            cost_marginal=None if cost_inputs is None else cost_inputs.cost_marginal,
            placement_cuts=None if cost_inputs is None else cost_inputs.placement_cuts,
            proxy_insert_components=(
                None if cost_inputs is None else cost_inputs.proxy_insert_components
            ),
            proxy_delete_components=(
                None if cost_inputs is None else cost_inputs.proxy_delete_components
            ),
            patch_word_sha256=None if cost_inputs is None else cost_inputs.patch_word_sha256,
            cost_error=cost_error,
            ordinate_error=ordinate_error,
        )

    def priority_value(candidate: StructuralCandidate) -> float:
        # Child acquisition must be route-neutral: Q and Gamma selectors seal
        # different ordinates only after they have measured the same frontier.
        return float(candidate.score)

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
    d0_compile_size = sum(
        1
        for atom_id in pure_pool
        for cut in cuts_by_atom.get(atom_id, ())
        if active_population_face((), ((str(atom_id), int(cut)),))
    )
    if admit("singleton_d0", d0_size, compiled_size=d0_compile_size):
        if workers_requested > 1:
            warm_keys = [
                ((), ((str(atom_id), int(cut)),))
                for atom_id in pure_pool
                for cut in sorted(int(c) for c in cuts_by_atom.get(atom_id, ()))
            ]
            workers_effective_max = max(
                workers_effective_max,
                _warm_solve_memo(
                    q_of, warm_keys, cache.solve_memo, workers_requested
                ),
            )
            if candidate_cost_warm is not None:
                candidate_cost_warm(
                    [
                        (k[0], k[1], None)
                        for k in warm_keys
                        if active_population_face(k[0], k[1])
                    ]
                )
        family_out = [scored((), (), None, "singleton_d0")]
        for atom_id in pure_pool:
            for cut in sorted(int(c) for c in cuts_by_atom.get(atom_id, ())):
                candidate = scored(
                    (), ((str(atom_id), cut),), None, "singleton_d0"
                )
                family_out.append(candidate)
                note_priority(str(atom_id), priority_value(candidate))
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
        permitted_members = members
        if compile_guard is not None:
            # Check the raw structural budget before evaluating permissions,
            # then count only the regime-active candidates that can actually
            # reach the compiler.  This keeps the structural guard
            # conservative without charging compiled work for stay or an
            # inactive patch face.
            if not guard_fits(guard, size):
                reject_family(guard, family)
                reject_family(compile_guard, family)
                break
            permitted_members = []
            for removed, pool in members:
                if deletion_permission is not None and not bool(
                    deletion_permission(removed).permitted
                ):
                    continue
                permitted_members.append((removed, pool))
            compiled_size = 0
            for removed, pool in permitted_members:
                compiled_size += int(active_population_face(removed, ()))
                compiled_size += sum(
                    1
                    for atom_id in pool
                    for cut in cuts_by_atom.get(atom_id, ())
                    if active_population_face(
                        removed, ((str(atom_id), int(cut)),)
                    )
                )
            if not admit(family, size, compiled_size=compiled_size):
                break
        elif not admit(family, size):
            break
        if workers_requested > 1:
            # Permission stays coordinator-only (its evaluator memoizes, so
            # the replay below re-reads identical decisions); only the
            # permitted members' solves are warmed.
            warm_keys: list[tuple] = []
            for removed, pool in permitted_members:
                if compile_guard is None and deletion_permission is not None:
                    if not bool(deletion_permission(removed).permitted):
                        continue
                warm_keys.append((removed, ()))
                for atom_id in pool:
                    for cut in sorted(int(c) for c in cuts_by_atom.get(atom_id, ())):
                        warm_keys.append(((), ((str(atom_id), int(cut)),)))
                        warm_keys.append((removed, ((str(atom_id), int(cut)),)))
            workers_effective_max = max(
                workers_effective_max,
                _warm_solve_memo(
                    q_of, warm_keys, cache.solve_memo, workers_requested
                ),
            )
            if candidate_cost_warm is not None:
                candidate_cost_warm(
                    [
                        (k[0], k[1], None)
                        for k in warm_keys
                        if active_population_face(k[0], k[1])
                    ]
                )
        family_out = []
        for removed, pool in permitted_members:
            if compile_guard is None and deletion_permission is not None:
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
                    note_priority(str(atom_id), priority_value(candidate))
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
                admit(family, len(batch))
                break
            batch_compile_size = sum(
                1
                for removed, _plan, selection in batch
                if active_population_face(removed, selection)
            )
            if not admit(
                family,
                len(batch),
                compiled_size=batch_compile_size,
            ):
                break
            frontiers_used = level + 1
            if workers_requested > 1:
                warm_keys = []
                for removed, _plan, selection in batch:
                    warm_keys.append((removed, selection))
                    warm_keys.append((removed, ()))
                    warm_keys.append(((), selection))
                workers_effective_max = max(
                    workers_effective_max,
                    _warm_solve_memo(
                        q_of, warm_keys, cache.solve_memo, workers_requested
                    ),
                )
                if candidate_cost_warm is not None:
                    candidate_cost_warm(
                        [
                            (removed, selection, plan)
                            for removed, plan, selection in batch
                            if active_population_face(removed, selection)
                        ]
                    )
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
        scoring_workers_requested=workers_requested,
        scoring_workers_effective_max=workers_effective_max,
        compile_guard=compile_guard,
        compile_calls=int(compile_calls_total),
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
    "CandidateCostInputs",
    "GENERALIZED_EXCHANGE_STRUCTURAL_POLICY_V2",
    "iter_structural_families",
    "StructuralCandidate",
    "StructuralEnumeration",
    "StructuralScoreWeights",
    "enumerate_structural_candidates",
    "resolve_frontier_schedule",
]
