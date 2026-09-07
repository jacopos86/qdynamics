# Paper II exact logical-estimator telemetry

Date: 2026-09-03

Status: implemented and locally verified on branch
`codex/paper-ii-exact-estimator-telemetry-20260903`.

## Scope and scientific invariance

This change adds accounting only. It does not change the Hamiltonian,
candidate population, structural scores, route objectives, certification
gates, integration, accepted patches, or route identifiers. It does not
submit or restart any run and does not regenerate a manuscript or results PDF.

The telemetry reports a logical dense-statevector estimator proxy. It is not a
hardware-shot count. Physical shots remain unavailable because the current
runs do not define measurement grouping, precision targets, variances, or shot
allocation.

## Exact work convention

One full McLachlan geometry at support dimension `p` contributes

```text
g(p) = p(p+1)/2 + p + 2
```

logical scalar entries: the symmetric metric, force vector, and two scalar
observables.

For generalized exchange, one shared positioned-candidate cache with active
support `p` and `M` positioned columns contributes

```text
p M + M(M+3)/2
```

logical entries. Structural candidate solves after this cache are classical
and are reported separately as candidate records, not charged again as
quantum-estimator work.

Each local-refit objective callback contributes `p+1` logical entries at the
patched support dimension: one ray-overlap objective and `p` tangent-gradient
entries. Every completed patched finalist geometry contributes `g(p)`.

The total reported proxy is

```text
integration + checkpoint base + candidate acquisition
            + finalist geometry + local refit.
```

## Persisted receipts

### AVQDS

`paper_ii_avqds_estimator_work_v1` preserves every greedy append round, its
base and candidate dimensions, frozen candidate-pool size, materialization
failures, geometry attempts and completions, finite/nonfinite results, chosen
candidate, commit status, and exact logical geometry entries. The existing
`candidates_scored` field remains the compatibility alias for completed
candidate geometries.

### Generalized exchange

`paper_ii_exchange_certification_estimator_work_v1` preserves each actual
finalist certification call, including local-refit objective attempts and
completions, patched-geometry attempts and completions, support dimension,
logical entries, and classical ray/velocity validation counts.

`paper_ii_generalized_exchange_estimator_work_v1` preserves every internal
selector/cache pass, including the deletion-to-insert fallback, and aggregates
all untruncated certification receipts. Its reason-count summary is derived
from the full in-memory attempt ledger; the existing last-16 public attempt
list remains unchanged for compatibility.

`paper_ii_exchange_selection_round_work_v1` preserves every outer selector
call at a checkpoint before a later call can overwrite the final decision. It
records actual selector-call and accepted-selector counts separately.

## Completeness and legacy artifacts

An evaluator attempt that raises after starting, or a user-injected refit hook
without the production counter, is explicitly incomplete. New successful runs
therefore report exact totals; interrupted runs do not fabricate unobserved
work. Old run artifacts continue through the prior reconstruction path and
remain labeled as lower bounds.

Consequently, the existing scientific trajectories do not become invalid, but
their candidate/refit work cannot become exact retroactively. Exact journal
accounting requires rerunning the selected final comparison arms with this
instrumented code.

The LaTeX comparison source now presents candidate acquisition, finalist
geometry, and local-refit work as separate rows. It shows a lower-bound marker
only for a legacy or incomplete receipt.

## Verification

- Focused instrumentation, diagnostics, report, and route-parity suite:
  `81 passed`.
- Broader AP-McLachlan, SIAM, resume, normalized-pool, and Paper III consumer
  suite: `420 passed`; one pre-existing resume-continuation assertion failed
  (`4 == 5` final parameters).
- The same resume failure was reproduced independently on the untouched
  `b1059379` worktree, confirming it is not introduced by this telemetry.
- Direct end-to-end aggregation of fresh toy trajectories returned `exact` for
  AVQDS and for generalized-exchange candidate, certification, refit, and
  total logical work.

## Known separate defect

The McLachlan-L2 repeated-selection loop currently iterates with
`range(max_rounds + 1)`. Its former test read a nonexistent
`checkpoint_commit_count` field and therefore did not test the configured cap.
This is a pre-existing behavioral issue and was deliberately not changed in
this accounting-only patch. The new receipt truthfully exposes
`selector_call_count` and `accepted_selector_count`, so a later behavior fix
can be made and reviewed separately.

## Rerun implication

Use this code for the planned final matched-accuracy routes. The new artifacts
will support an exact logical-estimator comparison without `>=` values. Keep
the physical-shot field unavailable unless a separate hardware measurement
and allocation protocol is implemented.
