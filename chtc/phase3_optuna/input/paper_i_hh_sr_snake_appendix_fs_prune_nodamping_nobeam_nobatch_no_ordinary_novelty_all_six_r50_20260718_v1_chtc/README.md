# SR-SNAKE undamped FS-pruning appendix bundle

This immutable-name bundle prepares six fresh Hubbard--Holstein jobs from
controller round/depth 0 through exactly round/depth 50. It is a strict
one-factor sensitivity variant of the frozen main no-prune route. Preparation
and the authenticated remote execution gate are complete:
`SUBMISSION_ENABLED=True` and Condor `requirements=TARGET.HasSIF`.

Route identity:

- request: `sr_snake_symmetric_cost_fs_prune_nodamping_v1`;
- resolved profile:
  `supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_nodamping_v1`;
- contract SHA-256:
  `272ede635558edb4acc2507ac3a9803d8ccec062b96c98634b8d6407df9fbc21`;
- source archive SHA-256:
  `1d6e93bd59f97f74cc444c6c3559b15d48053b2c4914736a3c32b0e0869a196a`.

The executable source was derived from the frozen main source archive
`fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35`.
It contains the additive prune-profile source and tests, and it never imports
scientific modules from the live repository at worker runtime.

## Exact scientific perturbation

Only the registered live pruning contract changes relative to the frozen main
route. The exact changed fields are recorded in
`scientific_settings_audit.json`:

- pruning is enabled in live rounds only;
- one deletion candidate is nominated from the full active logical ansatz;
- the complete affine deletion response is constrained by an explicit
  Fubini--Study radius, initially `0.125`;
- rejected models contract the radius by `0.5` to the source-defined `1e-8`
  floor; it never expands;
- metric damping is exactly zero and its update policy is off;
- endpoint-overlap calibration is off, so this policy adds no overlap
  measurement;
- measured delete-and-refit energy is the acceptance authority;
- terminal pruning, terminal refit, and structural rollback remain absent.

Everything else is inherited unchanged: effective 1x1 beam shape, no Phase-II
or Phase-III batching, ordinary Phase-II/III novelty multipliers off, the
all-infeasible novelty fallback and telemetry on, first-order Phase I,
measured-required fail-closed Phase-II curvature, full active-plus-singleton
Phase-III response, supported FS whitening and adaptive trust, symmetric
family-robust arctangent hardware-cost shaping, and full accepted-ansatz
supported-FS Powell refits without periodic or terminal refits.

All six regimes use same-cutoff exact references. Weak-Holstein rows use
working/reference `n_ph=3`; strong-Holstein rows use `n_ph=7`.

## Provenance and gates

The frozen parent route contract is hash-anchored locally; no parent scientific
result is claimed or required to establish the settings perturbation. Archive
tests and six archive-only manifest parses are recorded in
`archive_only_preflight.json` and `route_parity.json`.

This bundle was prepared only. The remote image gate was not performed, no
CHTC job was submitted, and no scientific run was launched. A later authorized
submission must rerun the remote gate and explicitly remove the Condor
`requirements=False` block in a new immutable bundle revision.
