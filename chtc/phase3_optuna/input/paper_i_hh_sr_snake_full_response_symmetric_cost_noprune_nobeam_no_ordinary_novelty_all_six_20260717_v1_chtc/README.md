# SR-SNAKE full-response symmetric-cost no-prune six-regime bundle

This immutable candidate bundle contains six fresh round-zero Hubbard--Holstein
jobs. Weak--weak and intermediate--weak run to controller round 30. Strong--weak,
weak--strong, intermediate--strong, and strong--strong run directly to round 50.
Every command carries an explicit source-locked `--adapt-max-depth`; the route
profile does not supply a generic horizon.

The route request is `sr_snake_no_prune_symmetric_cost_v1`, resolved as
`supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1`.
Its defining settings are:

- Phase I is `first_order_fs_trust_v1`;
- Phase II requires measured finite curvature and fails the whole run on a
  missing or malformed receipt; the lambda-F cheap proxy is off;
- Phase II is unwhitened and Phase III uses supported-metric whitening over the
  full active-logical-plus-singleton response model;
- the accepted ansatz is fully refit after every admission in supported
  Fubini--Study coordinates over the expanded-runtime/projected-logical base
  chart;
- scientific `H + mu G` response damping is off. The historical `1e-6`
  numerical Schur ridge and `1e-9` supported-solve tolerances remain numerical
  stability controls, not scientific damping;
- Phase-II and Phase-III ordinary novelty multipliers are off;
- the all-energy-models-infeasible novelty fallback remains enabled as a safety
  path, with explicit enabled/fired/count/round/query-charge telemetry;
- pruning, beam branching, batching, Phase 0, HH preseed, finite-angle fallback,
  Phase-III rescue, periodic refits, terminal refits, and terminal pruning are
  off;
- symmetric family-robust arctangent hardware-cost shaping is active.

The worker imports only the extracted hash-locked source snapshot. Because the
live scientific source is intentionally dirty relative to the recorded base
commit, the source archive and complete per-file SHA-256 inventory are the
executable authority; the Git commit and tree are ancestry metadata only.

`build_bundle.py` builds and validates the bundle but does not submit it.
Submission is a separate main-agent action after all local, archive-only, and
authenticated remote-image gates pass.
