# Main SR-SNAKE six-regime round-50 frozen bundle

This is an immutable-name, source-frozen bundle for six fresh
Hubbard--Holstein jobs, each starting at controller round/depth zero and
ending at exactly round/depth 50. The source freeze and authenticated remote
execution gate are complete: `SOURCE_FREEZE_COMPLETE=True`,
`SUBMISSION_ENABLED=True`, and Condor `requirements=TARGET.HasSIF`.

The executable source authority is the immutable parent archive with SHA-256
`197359e4252cf534ac7b603686aa4514580dcbcb95899513d35a2730dc80cbd7`,
plus the exact active-prefix estimator-receipt overlay. The derived archive
does not import scientific modules from the live source tree. Its
`adapt_pipeline.py` SHA-256 is
`81723a773946ff0aabea3c57f3ea12fff2808e935a7dc38ad2c01b472c3fa7fc`.

The only route is `sr_snake_no_prune_symmetric_cost_v1`, resolved as
`supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1`.
It uses the main effective 1x1 controller.  The historical 3-live x 2-child
beam policy belongs only to the separate appendix beam ablation and is absent
here.

Locked scientific contract:

- all six rows are fresh round 0 to round 50;
- weak-Holstein rows use working/reference `n_ph=3`;
- strong-Holstein rows use working/reference `n_ph=7`;
- the 12-decimal `g_ep` strings and same-cutoff exact reference energies are
  frozen in `physics_and_exact_reference_lock.json`;
- Phase I is `first_order_fs_trust_v1`;
- Phase II is `measured_required_fail_closed_v1`, with the lambda-F cheap
  curvature proxy off and Phase-II whitening off;
- Phase III uses full active-plus-singleton response coordinates, supported
  Fubini--Study whitening, and displacement-calibrated adaptive trust;
- every accepted ansatz is fully refit in supported-FS coordinates over the
  expanded-runtime/projected-logical Powell base chart;
- pruning, batching, beam branching, ordinary Phase-II/III novelty
  multipliers, Phase 0, HH preseed, finite-angle fallback, Phase-III rescue,
  periodic refits, terminal refits, and terminal pruning are off;
- the all-energy-models-infeasible novelty fallback remains enabled with
  explicit fired/count/round/operator/reason/query telemetry;
- symmetric family-robust arctangent hardware-cost shaping is active.

Required evidence gates are explicit in every job and normalized manifest:

- exact estimator-ledger `S_alg` closure, round-by-round active-prefix
  receipts, and a terminal closure receipt;
- fallback telemetry closure;
- full Phase-III response and full accepted-refit coverage each round;
- fixed-sector and binary-padding leakage gates;
- exact round/depth-50 closure;
- a persisted same-cutoff physical-sector ground-space projector fidelity
  receipt computed after the run for reporting only, with
  `s_alg_charged=false`.

The job-local cache contract is also explicit in both manifest layers:
candidate-record and HH-pool caches are `disk`, the pool scope is `exact`,
each cache namespace must start empty, and caches are performance-only rather
than a scientific fallback.

`build_bundle.py` deterministically rebuilds the derived archive, validates it
without importing scientific source from the live tree, and leaves submission
blocked until a separately authorized remote-image gate and submission step.
The cross-method fidelity audit remains external evidence: it is not allowed to
pull newer comparator scientific source into this SR worker archive.
