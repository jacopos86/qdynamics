# Conditional Beam Policy

Read this file only when the user enables or asks about beam survival.

## Typed policy

Use `ForkLocalBeam()` when the user explicitly requests beam survival. The
typed constructor and resolved route contract own the current branch caps,
child caps, cost weight, and calibration status. Surface the serialized
calibration status when proposing a beam study; do not restate its values in
Markdown. Explicit typed overrides remain permitted for a named study and do
not create a new default.

## Survival semantics

- Each child is a fully accepted lineage after admission, complete refit, and
  any enabled measured pruning.
- Compare descendants by post-refit energy and realized fork-local `S_alg`
  since their common ledger prefix.
- Shared pre-fork work is excluded from the fork-local comparison but remains
  in global run accounting.
- Retain at most the configured live-branch cap with deterministic tie breaks.
- If accepted children exist, an unchanged parent does not survive.
- There is no patience rule, uncertainty margin, hysteresis, frozen-parent
  archive, or speculative fallback.
- The returned scientific result is the dominant accepted lineage; fork-tree
  provenance remains observational metadata.

Batching chooses a child proposal, pruning verifies the accepted child, and
beam selects among completed child transitions. These are independent
policies.

## Checkpoint and resume contract

The direct controller executes this policy. A beam checkpoint publishes one
terminal winning ancestry and the global all-branch estimator ledger:

- `checkpoint_branch_policy` is
  `canonical_terminal_winning_lineage`;
- history and diagnostics cover every winning beam round;
- the pointer and sidecar use
  `ledger_scope=all_executed_branches`;
- discarded-child work remains in global `S_alg`;
- fork-local comparison work excludes the shared unbranched prefix.

An authenticated resume extends the prior winning ancestry, appends diagnostic
rounds, and seeds the comparison with the prior fork-local winner cost.
Accepted prune-trial branches contribute to that cost without becoming beam
ancestry IDs. Read `policies/resume.md` for the artifact authentication rules.

Historical beam routes remain explicit compatibility paths and are not a
substitute for this direct policy.
