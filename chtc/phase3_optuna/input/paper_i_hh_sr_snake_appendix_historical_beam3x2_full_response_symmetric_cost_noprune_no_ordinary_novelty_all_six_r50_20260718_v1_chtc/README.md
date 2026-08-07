# Appendix SR-SNAKE exact historical-beam round-50 bundle

This immutable-name bundle prepares six fresh Hubbard--Holstein appendix jobs,
each from controller round/depth zero through exactly 50 controller rounds. The
source freeze and authenticated remote execution gate are complete:
`SOURCE_FREEZE_COMPLETE=True`, `SUBMISSION_ENABLED=True`, and Condor
`requirements=TARGET.HasSIF`.

## Source authority

The executable base is the frozen main/no-beam archive:

- SHA-256: `fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35`;
- route digest: `69dcf6d5711a05b54cc143562bc6f437511d781164319b2705857b22deda1538`.

The builder authenticates that complete parent inventory and applies exactly
one hash-locked conditional overlay,
`historical_beam_exact_semantics_overlay.patch`. The derived archive never
imports scientific modules from the live source tree, and the parent bundle is
never modified.

## Exact one-factor route difference

Request `sr_snake_no_prune_symmetric_cost_beam_v1`, resolving to
`supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_beam_v1`.
Relative to the frozen main route, the executable-settings diff is exactly:

- `adapt_beam_live_branches`: `1 -> 3`;
- `adapt_beam_children_per_parent`: `1 -> 2`;
- `adapt_beam_terminated_keep`: `0 -> 3`;
- `adapt_beam_terminal_archive_mode`: `disabled -> legacy`.

`adapt_beam_lambda=0.005` is unchanged from the parent setting. The explicit
historical structure is three retained live branches and two singleton
admission children per parent, hence at most six admission children in a
controller round. Every expanded parent also materializes its stop/terminated
child even when proposals exist. The terminal archive carries prior terminal
children across rounds and remains capped at three. This is the recovered
`stop_or_single_admission` behavior from the source immediately before commit
`1f1d93c1a0060f0db70da6736cae4ec5ffffc79b`; it is not a retention-only
approximation.

## Inherited contract

Everything else is inherited from the frozen main route:

- all six regimes are fresh round 0 to round 50;
- weak-Holstein rows use working/reference `n_ph=3`, strong-Holstein rows use
  working/reference `n_ph=7`, with same-cutoff exact references;
- first-order Phase I and measured-required fail-closed Phase-II curvature;
- Phase-II whitening off; full active-plus-singleton Phase-III response with
  supported Fubini--Study whitening and displacement-calibrated adaptive trust;
- full accepted-ansatz supported-FS Powell refit on the expanded-runtime /
  projected-logical base chart;
- pruning and batching off; ordinary Phase-II/III novelty multipliers off;
- all-energy-models-infeasible novelty fallback enabled with explicit
  fired/count/round/operator/reason/query telemetry;
- symmetric family-robust arctangent cost shaping;
- Phase 0, HH preseed, finite-angle fallback, Phase-III rescue, periodic refit,
  terminal refit, and terminal prune off;
- seed, optimizer budgets, symmetry, padding, pool, query accounting, Qiskit
  sidecar, and reporting-only fidelity policies unchanged.

`scientific_settings_audit.json` records the four-field executable diff and the
immutable parent authority. Archive-only tests verify route resolution,
stopped-parent materialization, cumulative terminal retention, cap three, and
that the parent/no-beam profile remains unchanged.
