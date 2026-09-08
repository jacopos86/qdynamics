r"""Complete-patch evaluation shared by structural search methods.

Search supplies a patch :math:`(D,A)` and receives raw candidate evidence.
Population normalization, candidate admission, and certification are separate.
The shared safeguarded solve supplies the realized captured drift

.. math::

   Q(S)=2 f_S^\top v_S-v_S^\top K_{SS}v_S, \qquad
   \delta(D,A)=\frac{Q((C\setminus D)\cup A)-Q(C)}{\|b\|^2+\epsilon}.

The exact Schur reference remains in ``schur_identity.conditional_capture``.
The acquisition utilities retained here preserve the existing search's
behavior; they are distinct from the later population-normalized route score.
Scheduling and cost callbacks surround the numerical kernels without deciding
which patches compete. No ansatz is materialized by this module.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.compiled_patch_cost import CompiledPatchCostError
from pipelines.time_dynamics.ap_mclachlan.gain_evaluation import factorized_capture
from pipelines.time_dynamics.ap_mclachlan.insertion_words import InsertionPlan
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.objective_ordinates import (
    GammaMonotonicityError, check_gamma_nested_monotonicity,
)
from pipelines.time_dynamics.ap_mclachlan.patch_enumeration import Patch
from pipelines.time_dynamics.ap_mclachlan.performance import count, phase
from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
    StructuralInsertionCache, memoized_gamma, memoized_solve_metadata,
    structural_candidate_solve,
)

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
    gain_evaluation: str = "joint_batch"
    joint_validation_required: bool = False

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


@dataclass(kw_only=True)
class PatchEvaluator:
    """Evaluate positioned singleton or joint patches on one frozen geometry.

    ``evaluate`` consumes an already-admitted patch stream in its given order.
    It neither filters candidates nor assigns population-normalized scores.
    The integration adapter supplies circuit compilation under one fixed
    contract; its caches and this evaluator's solve cache are reusable across
    searches on the same checkpoint. Create a fresh evaluator for independent
    work-accounting comparisons.
    """

    cache: StructuralInsertionCache
    base_K: np.ndarray
    base_f: np.ndarray
    norm_b_sq: float
    inverse_policy: McLachlanInversePolicy
    weights: StructuralScoreWeights
    cuts_by_atom: Mapping[str, Sequence[int]]
    insertion_cost: Callable[[tuple[str, ...]], float]
    deletion_cost: Callable[[tuple[int, ...]], float]
    deletion_conditioning: Callable[[tuple[int, ...]], float] | None = None
    deletion_history_loss: Callable[[tuple[int, ...]], float] | None = None
    scoring_workers: int = 1
    route_bundle: Any | None = None
    gamma_rcond: float | None = None
    candidate_cost_inputs: Callable[
        [tuple[int, ...], tuple[tuple[str, int], ...], InsertionPlan | None],
        CandidateCostInputs,
    ] | None = None
    candidate_cost_warm: Callable[[Sequence[tuple]], Any] | None = None
    population_accuracy_debt: bool | None = None
    gain_evaluation: str = "joint_batch"
    factorized_memo: dict = field(default_factory=dict, init=False, repr=False)
    population_scoring: bool = field(init=False)
    workers_requested: int = field(init=False)
    workers_effective_max: int = field(init=False, default=1)
    compile_calls_total: int = field(init=False, default=0)
    gamma_denominator: float = field(init=False)
    conditioning: Callable[[tuple[int, ...]], float] = field(init=False, repr=False)
    history_loss: Callable[[tuple[int, ...]], float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.gain_evaluation not in {"joint_batch", "singleton_factorized"}:
            raise ValueError(f"unknown gain evaluation: {self.gain_evaluation!r}")
        self.conditioning = self.deletion_conditioning or (lambda removed: 0.0)
        self.history_loss = self.deletion_history_loss or (lambda removed: 0.0)
        self.workers_requested = int(self.scoring_workers)
        if self.workers_requested < 1:
            raise ValueError("scoring_workers must be positive.")
        self.population_scoring = bool(
            self.route_bundle is not None
            and getattr(self.route_bundle, "uses_population_scoring", False)
        )
        if self.population_scoring and self.candidate_cost_inputs is None:
            raise ValueError("population-scoring route bundles require candidate_cost_inputs.")
        self.gamma_denominator = float(self.norm_b_sq) + max(0.0, float(self.weights.epsilon_norm))

    @cached_property
    def q_base(self) -> float:
        """Base capture, acquired on first use after search-budget validation."""

        return self.q_of((), ())

    def active_population_face(
        self,
        removed: tuple[int, ...], selection: tuple[tuple[str, int], ...]
    ) -> bool:
        if not self.population_scoring or self.population_accuracy_debt is None:
            return True
        if bool(self.population_accuracy_debt):
            return bool(selection)
        return bool(removed) and not bool(selection)

    def q_of(self, removed: tuple[int, ...], selection: tuple[tuple[str, int], ...]) -> float:
        """Provisional capture; factorized batches never enter the solve memo."""

        if self.gain_evaluation == "singleton_factorized" and len(selection) > 1:
            key = (removed, selection)
            if key not in self.factorized_memo:
                self.factorized_memo[key] = factorized_capture(
                    self._joint_q_of(removed, ()),
                    (self._joint_q_of(removed, (position,)) for position in selection),
                )
            return self.factorized_memo[key]
        return self._joint_q_of(removed, selection)

    def _joint_q_of(
        self, removed: tuple[int, ...], selection: tuple[tuple[str, int], ...],
        *, acquisition_stage: str = "selection",
    ) -> float:
        all_indices = range(int(self.base_f.reshape(-1).size))
        keep = tuple(i for i in all_indices if i not in set(removed))
        _Q, q = structural_candidate_solve(
            cache=self.cache,
            base_K=self.base_K,
            base_f=self.base_f,
            norm_b_sq=float(self.norm_b_sq),
            keep_indices=keep,
            inserted_selection=selection,
            inverse_policy=self.inverse_policy,
            epsilon_norm=float(self.weights.epsilon_norm),
            memo_key=(removed, selection),
            gamma_rcond=self.gamma_rcond,
            acquisition_stage=acquisition_stage,
        )
        return float(q)

    def scored(
        self,
        removed: tuple[int, ...],
        selection: tuple[tuple[str, int], ...],
        plan: InsertionPlan | None,
        family: str,
    ) -> StructuralCandidate:
        q_joint = self.q_of(removed, selection)
        delta = q_joint - self.q_base
        if selection:
            gain = max(0.0, q_joint - self.q_of(removed, ()))
            atom_ids = tuple(atom_id for atom_id, _cut in selection)
            cost_ins = max(1.0, float(self.insertion_cost(atom_ids)))
            insertion_utility = gain / (cost_ins ** float(self.weights.alpha_ins))
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
            loss = max(0.0, self.q_of((), selection) - q_joint)
            cost_del = max(1.0, float(self.deletion_cost(removed)))
            # Conditioning relief/damage from solves the enumeration already
            # performed: the retained log-condition of the base support versus
            # the deletion branch.  Both memo entries exist because q_of ran.
            cond_base, _rank_b = memoized_solve_metadata(self.cache, ((), ()))
            cond_del, _rank_d = memoized_solve_metadata(self.cache, (removed, ()))
            relief = damage = 0.0
            if cond_base is not None and cond_del is not None and cond_base > 0 and cond_del > 0:
                log_shift = math.log10(cond_base) - math.log10(cond_del)
                relief = max(0.0, log_shift)
                damage = max(0.0, -log_shift)
            deletion_utility = (
                (cost_del ** float(self.weights.alpha_del))
                * (
                    1.0
                    + float(self.weights.lambda_cond_relief) * relief
                    + max(0.0, float(self.conditioning(removed)))
                )
                / (
                    loss
                    + float(self.weights.lambda_hist) * max(0.0, float(self.history_loss(removed)))
                    + float(self.weights.lambda_cond_damage) * damage
                    + float(self.weights.epsilon_L)
                )
            )
        else:
            loss = 0.0
            deletion_utility = 0.0
        score = (
            insertion_utility
            + deletion_utility
            + float(self.weights.w_delta) * delta
        )
        if bool(self.weights.debt_ranking):
            # True lexicographic key. A scalar encoding (delta + eps*tanh(u))
            # only approximates this: it silently degrades to the utility
            # ordering whenever two deltas differ by less than eps, which is
            # exactly the regime where the divergent deletion utility is most
            # dangerous. Quantizing delta makes "tied on accuracy" an explicit,
            # declared tolerance rather than a floating-point accident.
            tol = float(self.weights.delta_rank_tolerance)
            rank_primary = float(round(float(delta) / tol)) if tol > 0.0 else float(delta)
            rank_secondary = float(np.tanh(insertion_utility + deletion_utility))
        else:
            rank_primary = float(score)
            rank_secondary = 0.0
        m_gamma = None
        cost_inputs = None
        cost_error = None
        ordinate_error = None
        factorized_batch = self.gain_evaluation == "singleton_factorized" and len(selection) > 1
        if self.population_scoring and self.active_population_face(removed, selection):
            if self.gamma_rcond is not None and not factorized_batch:
                gamma_joint = memoized_gamma(self.cache, (removed, selection))
                gamma_base = memoized_gamma(self.cache, ((), ()))
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
                                denominator=self.gamma_denominator,
                                rcond=float(self.gamma_rcond),
                            )
                        except GammaMonotonicityError as exc:
                            ordinate_error = f"{type(exc).__name__}:{exc}"
                    if ordinate_error is None:
                        m_gamma = float(
                            (gamma_joint - gamma_base) / self.gamma_denominator
                        )
            # Price every regime-active candidate even when Gamma rejected
            # it.  The parallel path has already warmed the complete admitted
            # family, so replaying the cost lookup here preserves serial /
            # parallel compile accounting and attaches the same marginal for
            # audit.  The population ranker still excludes ordinate errors.
            if self.candidate_cost_inputs is not None and (removed or selection):
                try:
                    cost_inputs = self.candidate_cost_inputs(removed, tuple(selection), plan)
                    self.compile_calls_total += int(cost_inputs.compile_calls)
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
            gain_evaluation=self.gain_evaluation,
            joint_validation_required=bool(factorized_batch),
        )

    def authoritative_joint_candidate(self, candidate: StructuralCandidate) -> StructuralCandidate:
        """Validate real frozen geometry without repricing or reranking.

        All missing candidate pairs are charged to validation. Provisional
        score/rank/utility fields and compiled-word metadata remain untouched;
        q, delta, gain, deletion loss and Gamma become authoritative frozen
        values. The original gain-mode marker remains for provenance. A
        materialized or refitted ray still needs its own certification.
        """

        removed, selection = candidate.removed_runtime_indices, candidate.inserted_selection

        def actual_q(deleted, inserted):
            return self._joint_q_of(deleted, inserted, acquisition_stage="validation")

        q_joint = actual_q(removed, selection)
        q_base = actual_q((), ())
        gain = max(0.0, q_joint - actual_q(removed, ())) if selection else 0.0
        loss = max(0.0, actual_q((), selection) - q_joint) if removed else 0.0
        m_gamma = None
        ordinate_error = None
        if self.gamma_rcond is not None:
            gamma_joint = memoized_gamma(self.cache, (removed, selection))
            gamma_base = memoized_gamma(self.cache, ((), ()))
            if gamma_joint is not None and gamma_base is not None:
                if not removed and selection:
                    try:
                        check_gamma_nested_monotonicity(
                            gamma_base=float(gamma_base),
                            gamma_patched=float(gamma_joint),
                            denominator=self.gamma_denominator,
                            rcond=float(self.gamma_rcond),
                        )
                    except GammaMonotonicityError as exc:
                        ordinate_error = f"{type(exc).__name__}:{exc}"
                if ordinate_error is None:
                    m_gamma = float((gamma_joint - gamma_base) / self.gamma_denominator)
        return replace(
            candidate, q=q_joint, delta=q_joint - q_base,
            insertion_gain=gain, deletion_loss=loss, m_gamma=m_gamma,
            ordinate_error=ordinate_error, joint_validation_required=False,
        )

    def evaluate(
        self,
        patches: Iterable[Patch], family: str
    ) -> Iterator[StructuralCandidate]:
        """Evaluate an admitted family only when its members are consumed."""

        def geometry_requests():
            for patch in patches:
                selection = (
                    patch.inserted_selection
                    if patch.plan is None
                    else _plan_selection(patch.plan, self.cuts_by_atom)
                )
                yield patch.removed_runtime_indices, selection, patch.plan

        requests = geometry_requests()
        if self.workers_requested > 1:
            # Parallel scheduling surrounds the same deterministic serial
            # evaluation order; helper solves are never compiler requests.
            requests = list(requests)
            warm_keys = []
            for removed, selection, _plan in requests:
                warm_keys.append((removed, selection))
                if removed and selection:
                    warm_keys.extend(((removed, ()), ((), selection)))
            if self.gain_evaluation == "singleton_factorized":
                # Warm concrete singleton dependencies, not approximate batch
                # keys: overlapping batches then share one solve per key.
                warm_keys = [
                    dependency
                    for removed, selection in warm_keys
                    for dependency in (
                        [(removed, ()), *((removed, (position,)) for position in selection)]
                        if len(selection) > 1 else [(removed, selection)]
                    )
                ]
            self.workers_effective_max = max(
                self.workers_effective_max,
                _warm_solve_memo(self.q_of, warm_keys, self.cache.solve_memo, self.workers_requested),
            )
            if self.candidate_cost_warm is not None:
                self.candidate_cost_warm([
                    (removed, selection, plan)
                    for removed, selection, plan in requests
                    if (removed or selection) and self.active_population_face(removed, selection)
                ])
        for removed, selection, plan in requests:
            candidate = self.scored(removed, selection, plan, family)
            yield candidate


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


__all__ = ["CandidateCostInputs", "PatchEvaluator", "StructuralCandidate", "StructuralScoreWeights"]
