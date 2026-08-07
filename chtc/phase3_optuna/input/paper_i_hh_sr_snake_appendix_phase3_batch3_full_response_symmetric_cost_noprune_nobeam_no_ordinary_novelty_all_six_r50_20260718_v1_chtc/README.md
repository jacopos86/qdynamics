# Appendix SR-SNAKE Phase-III batch-three round-50 bundle

This immutable-name bundle prepares six fresh Hubbard--Holstein appendix jobs,
each from controller round zero through exactly 50 controller rounds. The source
freeze and authenticated remote execution gate are complete:
`SOURCE_FREEZE_COMPLETE=True`, `SUBMISSION_ENABLED=True`, and Condor
`requirements=TARGET.HasSIF`.

Each controller round may admit one to three singleton candidates, so the
active-depth and new-admission ceilings are 150. The controller-round target,
not an admission-count cap of 50, is the authoritative horizon.

## Source authority

The executable base is the frozen main/no-batch archive:

- SHA-256: `fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35`;
- parent route digest: `69dcf6d5711a05b54cc143562bc6f437511d781164319b2705857b22deda1538`.

The builder authenticates that complete parent inventory and applies exactly
one hash-locked overlay, `phase3_batch_appendix_overlay.patch`. The derived
archive never imports scientific modules from the live source tree, and the
parent bundle is never modified.

## Exact one-factor route difference

Request `sr_snake_no_prune_symmetric_cost_phase3_batch_v1`, resolving to
`supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_phase3_batch_v1`.
Relative to the frozen main route:

- Phase II remains explicitly non-batched;
- Phase III alone uses post-shortlist `combinatorial_reduced_plane` batching;
- target batch size and hard cap are both three;
- each selected batch is scored in the complete active-logical-plus-batch
  response model;
- supported-rank reduction may remove only genuine Gram-null directions;
- the global supported-FS trust receipt and full accepted-ansatz refit are
  mandatory for every committed batch;
- the projected-runtime child subset cap remains one;
- beam remains disabled/effective 1x1.

## Inherited contract

Everything else is inherited from the frozen main route:

- all six regimes are fresh round 0 to controller round 50;
- weak-Holstein rows use working/reference `n_ph=3`, strong-Holstein rows use
  working/reference `n_ph=7`, with same-cutoff exact references;
- first-order Phase I and measured-required fail-closed Phase-II curvature;
- Phase-II whitening off; full active-plus-singleton Phase-III candidate
  responses with supported Fubini--Study whitening and
  displacement-calibrated adaptive trust;
- full accepted-ansatz supported-FS Powell refit on the expanded-runtime /
  projected-logical base chart;
- pruning off and ordinary Phase-II/III novelty multipliers off;
- all-energy-models-infeasible novelty fallback enabled with explicit
  fired/count/round/operator/reason/query telemetry;
- symmetric family-robust arctangent cost shaping;
- Phase 0, HH preseed, finite-angle fallback, Phase-III rescue, periodic refit,
  terminal refit, and terminal prune off;
- seed, optimizer budgets, symmetry, padding, pool, query accounting, Qiskit
  sidecar, and reporting-only fidelity policies unchanged.

`scientific_settings_audit.json` records the exact executable diff and the
immutable parent authority. Archive-only tests verify route resolution,
Phase-II/III flag isolation, full-batch response/trust receipts, refit scope,
and that the parent/no-batch profile remains unchanged.
