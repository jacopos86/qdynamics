# Implement Issue #17: Fork-Local Estimator-Accounting Views

## Objective

Complete Issue #17 from `sr-snake-refactor-plan.md`: add one deep,
immutable estimator-ledger interface that captures a common fork prefix and
returns realized post-fork `S_alg` views for named lineages. This is the
accounting prerequisite for Issue #18; do not implement beam survival or use
the `0.01` beam weight in a decision.

## Decisions already made

- The existing estimator ledger remains the sole authority for algorithmic
  work. Its four components are `N_H_outer`, `N_H_refit`, `N_grad`, and
  `N_metric`, and `S_alg` is their exact sum.
- A fork begins at one closed, authoritative ledger prefix belonging to the
  common accepted state.
- A lineage's fork-local work is the set of unique physical estimator
  primitives it consumes after that prefix, excluding primitives already
  present in the common prefix.
- Branch and lineage IDs identify consumers; they never participate in
  physical primitive identity.
- A post-fork primitive consumed by multiple lineages appears symmetrically in
  each lineage-local view, independent of execution order. The run-wide
  all-work view still deduplicates it once.
- Work consumed by a terminated lineage remains in all-work/discarded-work
  accounting and is never reassigned to the winner.
- Wall-clock time, Qiskit resources, and predictive Phase-III proposal cost are
  outside this issue.
- Default singleton, no-prune, no-batch, no-beam behavior and its frozen
  trajectory/accounting remain unchanged.

## Authority to implement and repair

Implement the smallest coherent ledger and typed-receipt changes needed for
these views. Refactor nearby accounting helpers when that improves locality,
and repair ordinary focused-test, serialization, or projection defects without
stopping. Preserve unrelated dirty work. Use test-first implementation and run
the repository's two-axis Standards/Spec review when the focused suite is
green.

## Constraints

- Put the calculation behind the existing estimator-ledger seam; do not create
  a second counter or a beam-specific accounting implementation.
- Base the fork snapshot on the ledger's ordered closed prefix, not on wall
  time, controller guesses, or mutable global branch state.
- Return immutable, JSON-serializable evidence sufficient to identify the fork,
  lineage, exact primitive set or its auditable content-addressed
  representation, component counts, and `S_alg`.
- Preserve the existing global charged-component assignment so independently
  derived primitive sets reconcile component by component.
- Do not attach empty fork-tree placeholders to ordinary `SRRunResult`. Add only
  the typed internal/public receipt needed for Issue #18 to attach real
  beam-enabled provenance later.
- Do not change request defaults, route identity/profile/digest, scientific
  selection, admission, refit, pruning, batching, stopping, resume, or
  observation behavior.

## Definition of done

1. A caller can capture a closed fork prefix and derive lineage-local unique
   work from later ledger state without rerunning an estimator.
2. Focused tests prove that pre-fork work is excluded; unique post-fork work is
   counted by component; shared post-fork primitives appear in both lineage
   views but once in all-work; swapping lineage execution order changes no
   fork-local view; and terminated-lineage work remains in all-work.
3. Receipt serialization, fingerprints, component closure, and invalid-prefix
   validation are covered through the ledger interface.
4. Existing estimator-ledger tests and the frozen public SR-SNAKE facade/
   no-prune characterization remain green.
5. Report exact files changed, focused commands/results, and the independent
   Standards/Spec findings. Stop before Issue #18 and perform no scientific
   run, commit, push, label, or manuscript action.

Expected edit scope: `pipelines/static_adapt/estimator_call_ledger.py`,
the smallest necessary files under `pipelines/static_adapt/sr_snake/`, and
focused tests under `test/`.

## True stop conditions

Stop only if current ledger evidence cannot define the approved fork-local set
without changing physical primitive identity, code and tests expose conflicting
scientific accounting semantics, or completion would require Issue #18 beam
behavior. Name the exact conflict. Ordinary implementation gaps are repair
work, not blockers.
