# SR-SNAKE strong-row full-response runtime repair bundle

This immutable operational successor queues two fresh round-zero
Hubbard--Holstein jobs: strong--weak and strong--strong, both through controller
round 50. The bundle retains all six validated job manifests solely to prove
same-cutoff physics and route-parity closure; `queue.tsv` contains only the
two rows whose original executions reached an illegal raw Phase-II admission
after the canonical Phase-III shortlist emptied.

The source-only repair makes canonical no-beam Phase III evaluate the complete
input population in full active-plus-singleton response coordinates before
shortlisting. If that final shortlist is empty, fixed-horizon fallback may
select only a fully evaluated record with complete supported-rank and
adaptive-trust receipts. Scientific settings and the route-contract digest are
unchanged.

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
