"""Deterministic selection loop for generalized exchange.

Consumes structural families in acquisition order (singleton level, deletion
rungs, insertion frontiers), keeps every previously scored candidate, and at
each level certifies candidates one at a time in descending structural score
with frozen tie order.  The first finalist that passes every hard commit gate
commits atomically; when a level produces no certified candidate, the next
family is acquired only while the caller's escalation predicate holds and the
work guard admits it.  When nothing certifies, the selector returns stay with
the full attempt record.

Experimental search policies instead finish their requested search and expose
one cumulative population. Factorized batch gains are provisional: frozen
joint validation precedes materialization, without changing the sealed order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
    CERTIFICATION_ESTIMATOR_WORK_SCHEMA_V1,
    CertificationGates,
    CertificationResult,
    certify_finalist,
)
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import (
    StructuralCandidate,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
    StructuralEnumeration,
    iter_structural_families,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import FixedMcLachlanStep
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import GeometryEvaluation
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.performance import phase
from pipelines.time_dynamics.ap_mclachlan.state import APMcLachlanState


GENERALIZED_EXCHANGE_SELECTION_POLICY_V2 = "paper_ii_generalized_exchange_v2"


@dataclass(frozen=True)
class AttemptRecord:
    """One certification attempt, for decision reproduction."""

    family: str
    kind: str
    score: float
    removed_runtime_indices: tuple[int, ...]
    inserted_selection: tuple[tuple[str, int], ...]
    reason: str
    ray_distance: float | None = None
    smoothness_eta: float | None = None
    deletion_loss: float | None = None
    # Population-scoring bundles: the sealed cost factor, route score, and
    # physical ordinate that placed this finalist in the certification order.
    cost_factor_k: float | None = None
    route_score: float | None = None
    route_merit: float | None = None
    cost_marginal: tuple[int, int, int] | None = None
    placement_cuts: tuple[tuple[str, int], ...] | None = None
    patch_word_sha256: str | None = None

    def to_json_dict(self, *, compiled_score: bool = False) -> dict[str, Any]:
        """Serialize the score that ordered certification on the active route.

        A missing compiled score remains null; legacy utility is never its
        substitute. The historical serialization is unchanged by default.
        """

        return {
            "family": self.family,
            "kind": self.kind,
            "score": self.route_score if compiled_score else float(self.score),
            "removed_runtime_indices": [int(i) for i in self.removed_runtime_indices],
            "inserted_selection": [
                [str(a), int(p)] for a, p in self.inserted_selection
            ],
            "reason": self.reason,
            "ray_distance": self.ray_distance,
            "smoothness_eta": self.smoothness_eta,
            "deletion_loss": self.deletion_loss,
            "cost_factor_k": self.cost_factor_k,
            **({} if compiled_score else {"route_score": self.route_score}),
            "route_merit": self.route_merit,
            "cost_marginal": (
                None if self.cost_marginal is None else list(self.cost_marginal)
            ),
            "placement_cuts": (
                None
                if self.placement_cuts is None
                else [[str(a), int(c)] for a, c in self.placement_cuts]
            ),
            "patch_word_sha256": self.patch_word_sha256,
        }


@dataclass
class ExchangeSelection:
    """Outcome of one checkpoint's selection."""

    committed: StructuralCandidate | None
    certification: CertificationResult | None
    attempts: tuple[AttemptRecord, ...]
    telemetry: StructuralEnumeration | None
    stop_reason: str
    structural_scored_count: int
    level_telemetry: tuple[dict[str, Any], ...] = ()
    certification_work: tuple[dict[str, Any], ...] = ()

    @property
    def kind(self) -> str:
        return "stay" if self.committed is None else self.committed.kind


@dataclass(frozen=True)
class RankInfo:
    """Sealed population values that placed one candidate in rank order."""

    K: float
    score: float
    merit: float


@dataclass(frozen=True)
class RankedLevel:
    """One certification level after population sealing.

    ``ordered`` is the certification order (eligible candidates only);
    ``rank_info`` maps ``order_key`` to the sealed values; ``telemetry`` is
    the JSON-ready receipt (population hash, medians, MADs, K, regime, tau,
    exclusions) recorded once per level.
    """

    ordered: tuple[StructuralCandidate, ...]
    rank_info: Mapping[tuple, RankInfo]
    telemetry: Mapping[str, Any]


def candidate_insertions(
    candidate: StructuralCandidate,
    *,
    atoms_by_id: Mapping[str, Any],
    occurrence_label: Callable[[Any, int, int], str],
) -> tuple[tuple[int, Any, str], ...]:
    """Insertion entries (original-layout cut, term, label) for one candidate.

    Singleton candidates insert at their retained representative cut.  Plan
    candidates reproduce their full word: each inserted token's raw cut is the
    number of survivors preceding it, and same-cut order follows the word.
    ``occurrence_label(atom, cut, ordinal)`` must return a stable, unique
    runtime child label for the occurrence.
    """

    entries: list[tuple[int, Any, str]] = []
    if candidate.placement_cuts is not None:
        # Compiled-cost bundles materialize each inserted atom at the cut
        # chosen inside its commutation class (geometry-identical to the
        # retained representative; only the compiled circuit differs).
        if len(candidate.placement_cuts) != len(candidate.inserted_selection):
            raise ValueError("compiled placement count does not match insertions")
        for ordinal, ((atom_id, _cut), (placed_id, realized_cut)) in enumerate(
            zip(candidate.inserted_selection, candidate.placement_cuts)
        ):
            if str(atom_id) != str(placed_id):
                raise ValueError("compiled placement atom order does not match insertions")
            atom = atoms_by_id[str(atom_id)]
            entries.append(
                (
                    int(realized_cut),
                    atom.term,
                    occurrence_label(atom, int(realized_cut), ordinal),
                )
            )
        return tuple(entries)
    if candidate.plan is None:
        for ordinal, (atom_id, cut) in enumerate(candidate.inserted_selection):
            atom = atoms_by_id[str(atom_id)]
            entries.append((int(cut), atom.term, occurrence_label(atom, int(cut), ordinal)))
        return tuple(entries)
    survivor_count = 0
    ordinal = 0
    for token in candidate.plan.full_word:
        if token.kind == "survivor":
            survivor_count += 1
            continue
        atom = atoms_by_id[str(token.key)]
        entries.append(
            (
                int(survivor_count),
                atom.term,
                occurrence_label(atom, int(survivor_count), ordinal),
            )
        )
        ordinal += 1
    return tuple(entries)


def select_exchange_patch(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    gates: CertificationGates,
    atoms_by_id: Mapping[str, Any],
    occurrence_label: Callable[[Any, int, int], str],
    structural_kwargs: Mapping[str, Any],
    score_floor: float = 0.0,
    escalate: Callable[[], bool] | None = None,
    refit: Callable[[APMcLachlanState, np.ndarray], tuple[APMcLachlanState, np.ndarray]]
    | None = None,
    solve_repair_config: Any | None = None,
    max_certification_attempts_per_level: int | None = None,
    max_certification_attempts_per_deletion_branch: int | None = None,
    population_ranker: (
        Callable[[Sequence[StructuralCandidate]], "RankedLevel"] | None
    ) = None,
    realized_l2_gate: Callable[[CertificationResult], bool] | None = None,
    realized_l2_gate_reason: str = "realized_l2_not_improved",
    candidate_word_validator: (
        Callable[[StructuralCandidate, Sequence[tuple[int, Any, str]]], bool] | None
    ) = None,
    search_policy: str = "family_frontiers",
    gain_evaluation: str = "joint_batch",
) -> ExchangeSelection:
    """Run structural families and certify per level until one commit passes.

    With ``family_frontiers``, ``structural_kwargs`` are forwarded verbatim to
    :func:`iter_structural_families`. ``escalate`` gates acquisition of the
    *next* family after a level certifies nothing (default: always escalate;
    the trajectory integration passes the structural-repair predicate).

    Other policies use the same geometry, complete-patch cost evaluator and
    certification gates, but finish their search before one ranking pass.

    ``max_certification_attempts_per_level`` bounds how many finalists one
    level may materialize before the level is declared exhausted (``None`` =
    unbounded, the specification default).  It is a computational guard in the
    same spirit as the joint-work guard: certification is the expensive stage
    (full state materialization per attempt), and a level whose gates reject
    broadly would otherwise grind through every ranked candidate.

    ``max_certification_attempts_per_deletion_branch`` bounds failures per
    distinct removed-set within a level.  Deletion-branch-level score terms
    (conditioning relief, history) are constant across every insertion
    variant of one branch, so a favored branch floods the ranking with tied
    variants; at zero insertion angle the ray displacement is a function of
    the deletion alone, so once a branch fails the ray gate repeatedly its
    remaining variants are skipped and the budget reaches other branches.
    """

    should_escalate = escalate or (lambda: True)
    scored: list[StructuralCandidate] = []
    attempts: list[AttemptRecord] = []
    certification_work: list[dict[str, Any]] = []
    level_telemetry: list[dict[str, Any]] = []
    attempted: set = set()
    telemetry: StructuralEnumeration | None = None
    stop_reason = "no_structural_families"

    budget = (
        None
        if max_certification_attempts_per_level is None
        else max(1, int(max_certification_attempts_per_level))
    )
    branch_budget = (
        None
        if max_certification_attempts_per_deletion_branch is None
        else max(1, int(max_certification_attempts_per_deletion_branch))
    )
    search = None
    if search_policy == "family_frontiers":
        if gain_evaluation != "joint_batch":
            raise ValueError("factorized evaluation requires an experimental search policy")
        iterator = iter_structural_families(**dict(structural_kwargs))
    else:
        from pipelines.time_dynamics.ap_mclachlan.search_methods.runtime import (
            prepare_patch_search,
        )

        search = prepare_patch_search(
            search_policy=search_policy, gain_evaluation=gain_evaluation,
            structural_kwargs=structural_kwargs, population_ranker=population_ranker,
        )
        iterator = search.families()
    pushback: list = []

    def acquire():
        if pushback:
            return pushback.pop()
        with phase("patch.exchange.enumerate_family"):
            return next(iterator, None)

    while True:
        item = acquire()
        if item is None:
            break
        family, members, final = item
        if family == "__telemetry__":
            telemetry = final
            continue
        before_family = len(scored)
        with phase("patch.exchange.evaluate_family"):
            scored.extend(members)
        if len(scored) == before_family:
            continue
        # Deletion-cardinality rungs are one certification level with the
        # singleton family: the specification enumerates and scores every
        # guard-admitted rung BEFORE certification ("with no work guard,
        # every hard-feasible deletion subset is enumerated"), so deletions
        # always compete with insertions on structural score.  Only insertion
        # frontiers remain escalation-gated below.
        if family.startswith("singleton_d"):
            while True:
                nxt = acquire()
                if nxt is None:
                    break
                if nxt[0] == "__telemetry__":
                    telemetry = nxt[2]
                    continue
                if not str(nxt[0]).startswith("singleton_d"):
                    # The next family's members are deferred. A label peek
                    # must not evaluate candidates outside this population.
                    pushback.append(nxt)
                    break
                with phase("patch.exchange.evaluate_family"):
                    scored.extend(nxt[1])
        attempts_this_level = 0
        budget_hit = False
        branch_failures: dict = {}
        rank_info: dict = {}
        if population_ranker is not None:
            # Population-scoring bundles: seal the whole level once (cost
            # factors, shifted scores, eligibility), then certify in that
            # order.  A rejected finalist never triggers renormalization.
            level = population_ranker(tuple(scored))
            ranked = list(level.ordered)
            rank_info = dict(level.rank_info)
            receipt = dict(level.telemetry)
            if search is not None:
                receipt["search"] = search.receipt()
            level_telemetry.append(receipt)
        else:
            ranked = sorted(
                (
                    c
                    for c in scored
                    if c.kind != "stay"
                    and np.isfinite(float(c.score))
                    and float(c.score) > float(score_floor)
                ),
                # Lexicographic: accuracy first, utility only among exact ties.
                key=lambda c: (
                    -float(getattr(c, "rank_primary", c.score)),
                    -float(getattr(c, "rank_secondary", 0.0)),
                    c.order_key,
                ),
            )
        for candidate in ranked:
            info = rank_info.get(candidate.order_key)
            if candidate.order_key in attempted:
                continue
            if budget is not None and attempts_this_level >= budget:
                budget_hit = True
                break
            if (
                branch_budget is not None
                and candidate.removed_runtime_indices
                and branch_failures.get(candidate.removed_runtime_indices, 0)
                >= branch_budget
            ):
                continue
            attempted.add(candidate.order_key)
            attempts_this_level += 1
            try:
                insertions = candidate_insertions(
                    candidate,
                    atoms_by_id=atoms_by_id,
                    occurrence_label=occurrence_label,
                )
            except (KeyError, ValueError) as exc:
                attempts.append(
                    AttemptRecord(
                        family=candidate.family,
                        kind=candidate.kind,
                        score=float(candidate.score),
                        removed_runtime_indices=candidate.removed_runtime_indices,
                        inserted_selection=candidate.inserted_selection,
                        reason=f"insertion_resolution_failed:{exc}",
                        deletion_loss=float(candidate.deletion_loss)
                        if candidate.removed_runtime_indices and not candidate.joint_validation_required
                        else None,
                        cost_factor_k=None if info is None else float(info.K),
                        route_score=None if info is None else float(info.score),
                        route_merit=None if info is None else float(info.merit),
                    )
                )
                continue
            if (
                candidate_word_validator is not None
                and not bool(candidate_word_validator(candidate, insertions))
            ):
                attempts.append(
                    AttemptRecord(
                        family=candidate.family,
                        kind=candidate.kind,
                        score=float(candidate.score),
                        removed_runtime_indices=candidate.removed_runtime_indices,
                        inserted_selection=candidate.inserted_selection,
                        reason="compiled_word_identity_mismatch",
                        deletion_loss=(
                            float(candidate.deletion_loss)
                            if candidate.removed_runtime_indices and not candidate.joint_validation_required
                            else None
                        ),
                        cost_factor_k=None if info is None else float(info.K),
                        route_score=None if info is None else float(info.score),
                        route_merit=None if info is None else float(info.merit),
                        cost_marginal=candidate.cost_marginal,
                        placement_cuts=candidate.placement_cuts,
                        patch_word_sha256=candidate.patch_word_sha256,
                    )
                )
                continue
            frozen_validation = {}
            if search is not None:
                provisional_delta = float(candidate.delta)
                candidate = search.evaluator.authoritative_joint_candidate(candidate)
                frozen_validation = {
                    "gain_evaluation": gain_evaluation,
                    "provisional_delta": provisional_delta,
                    "validated_frozen_delta": float(candidate.delta),
                    "joint_validation_reason": candidate.ordinate_error,
                }
            if search is not None and candidate.ordinate_error is not None:
                result = CertificationResult(
                    certified=False,
                    reason=f"joint_validation_failed:{candidate.ordinate_error}",
                    estimator_work={
                        "schema": CERTIFICATION_ESTIMATOR_WORK_SCHEMA_V1,
                        "patched_runtime_parameter_count": None,
                        "logical_full_geometry_entry_count": 0,
                        "logical_refit_entry_count": 0,
                        "logical_total_estimator_entries": 0,
                        "complete": True,
                    },
                )
            else:
                with phase("patch.exchange.certify"):
                    result = certify_finalist(
                        state=state,
                        hamiltonian=hamiltonian,
                        theta_runtime=theta_runtime,
                        time=float(time),
                        base_evaluation=base_evaluation,
                        base_step=base_step,
                        removed_runtime_indices=candidate.removed_runtime_indices,
                        insertions=insertions,
                        inverse_policy=inverse_policy,
                        gates=gates,
                        refit=refit,
                        solve_repair_config=solve_repair_config,
                    )
            if (
                result.certified
                and realized_l2_gate is not None
                and not bool(realized_l2_gate(result))
            ):
                # Realized L^2 is the authority for every route: a finalist
                # whose materialized solve does not improve the checkpoint
                # distance is rejected here and the next ranked candidate is
                # tried, with the sealed population untouched.
                result = CertificationResult(
                    certified=False,
                    reason=str(realized_l2_gate_reason),
                    ray_distance=result.ray_distance,
                    smoothness_eta=result.smoothness_eta,
                    condition_number=result.condition_number,
                    estimator_work=result.estimator_work,
                )
            certification_work.append(
                {
                    **dict(result.estimator_work),
                    **frozen_validation,
                    "family": str(candidate.family),
                    "kind": str(candidate.kind),
                    "reason": str(result.reason),
                    "removed_runtime_indices": [
                        int(i) for i in candidate.removed_runtime_indices
                    ],
                    "inserted_selection": [
                        [str(a), int(p)]
                        for a, p in candidate.inserted_selection
                    ],
                }
            )
            attempts.append(
                AttemptRecord(
                    family=candidate.family,
                    kind=candidate.kind,
                    score=float(candidate.score),
                    removed_runtime_indices=candidate.removed_runtime_indices,
                    inserted_selection=candidate.inserted_selection,
                    reason=result.reason,
                    ray_distance=result.ray_distance,
                    smoothness_eta=result.smoothness_eta,
                    deletion_loss=float(candidate.deletion_loss)
                    if candidate.removed_runtime_indices
                    else None,
                    cost_factor_k=None if info is None else float(info.K),
                    route_score=None if info is None else float(info.score),
                    route_merit=None if info is None else float(info.merit),
                    cost_marginal=candidate.cost_marginal,
                    placement_cuts=candidate.placement_cuts,
                    patch_word_sha256=candidate.patch_word_sha256,
                )
            )
            if not result.certified and candidate.removed_runtime_indices:
                branch_failures[candidate.removed_runtime_indices] = (
                    branch_failures.get(candidate.removed_runtime_indices, 0) + 1
                )
            if result.certified:
                # Drain the iterator's telemetry without acquiring families:
                # closing is enough; telemetry stays None on early commit.
                iterator.close()
                return ExchangeSelection(
                    committed=candidate,
                    certification=result,
                    attempts=tuple(attempts),
                    telemetry=telemetry,
                    stop_reason="committed",
                    structural_scored_count=int(len(scored)),
                    level_telemetry=tuple(level_telemetry),
                    certification_work=tuple(certification_work),
                )
        stop_reason = "attempt_budget_exhausted" if budget_hit else "level_exhausted"
        if not should_escalate():
            iterator.close()
            stop_reason = "escalation_predicate_false"
            break

    return ExchangeSelection(
        committed=None,
        certification=None,
        attempts=tuple(attempts),
        telemetry=telemetry,
        stop_reason=stop_reason,
        structural_scored_count=int(len(scored)),
        level_telemetry=tuple(level_telemetry),
        certification_work=tuple(certification_work),
    )


__all__ = [
    "GENERALIZED_EXCHANGE_SELECTION_POLICY_V2",
    "AttemptRecord",
    "ExchangeSelection",
    "RankInfo",
    "RankedLevel",
    "candidate_insertions",
    "select_exchange_patch",
]
