"""Deterministic selection loop of the deletion-conditioned exchange selector.

Consumes structural families in acquisition order (singleton level, deletion
rungs, insertion frontiers), keeps every previously scored candidate, and at
each level certifies candidates one at a time in descending structural score
with frozen tie order.  The first finalist that passes every hard commit gate
commits atomically; when a level produces no certified candidate, the next
family is acquired only while the caller's escalation predicate holds and the
work guard admits it.  When nothing certifies, the selector returns stay with
the full attempt record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
    CertificationGates,
    CertificationResult,
    certify_finalist,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
    StructuralCandidate,
    StructuralEnumeration,
    iter_structural_families,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import FixedMcLachlanStep
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import GeometryEvaluation
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.performance import phase
from pipelines.time_dynamics.ap_mclachlan.state import APMcLachlanState


EXCHANGE_SELECTION_POLICY_V1 = "paper_ii_deletion_conditioned_exchange_v1"


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

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "kind": self.kind,
            "score": float(self.score),
            "removed_runtime_indices": [int(i) for i in self.removed_runtime_indices],
            "inserted_selection": [
                [str(a), int(p)] for a, p in self.inserted_selection
            ],
            "reason": self.reason,
            "ray_distance": self.ray_distance,
            "smoothness_eta": self.smoothness_eta,
            "deletion_loss": self.deletion_loss,
        }


@dataclass
class ExchangeSelection:
    """Outcome of one checkpoint's selection."""

    committed: StructuralCandidate | None
    certification: CertificationResult | None
    attempts: tuple[AttemptRecord, ...]
    telemetry: StructuralEnumeration | None
    stop_reason: str

    @property
    def kind(self) -> str:
        return "stay" if self.committed is None else self.committed.kind


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
    if candidate.plan is None:
        for ordinal, (atom_id, cut) in enumerate(candidate.inserted_selection):
            atom = atoms_by_id[str(atom_id)]
            entries.append(
                (int(cut), atom.term, occurrence_label(atom, int(cut), ordinal))
            )
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
) -> ExchangeSelection:
    """Run structural families and certify per level until one commit passes.

    ``structural_kwargs`` are forwarded verbatim to
    :func:`iter_structural_families`.  ``escalate`` gates acquisition of the
    *next* family after a level certifies nothing (default: always escalate;
    the trajectory integration passes the structural-repair predicate).

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
    iterator = iter_structural_families(**dict(structural_kwargs))
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
        scored.extend(members)
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
                    pushback.append(nxt)
                    break
                scored.extend(nxt[1])
        attempts_this_level = 0
        budget_hit = False
        branch_failures: dict = {}
        ranked = sorted(
            (
                c
                for c in scored
                if c.kind != "stay"
                and np.isfinite(float(c.score))
                and float(c.score) > float(score_floor)
            ),
            key=lambda c: (-float(c.score), c.order_key),
        )
        for candidate in ranked:
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
                        if candidate.removed_runtime_indices
                        else None,
                    )
                )
                continue
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
    )


__all__ = [
    "EXCHANGE_SELECTION_POLICY_V1",
    "AttemptRecord",
    "ExchangeSelection",
    "candidate_insertions",
    "select_exchange_patch",
]
