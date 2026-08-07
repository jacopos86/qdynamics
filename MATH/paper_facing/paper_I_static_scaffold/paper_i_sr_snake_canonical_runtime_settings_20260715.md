# Historical Conventional Paper-I SR-SNAKE v2 Runtime Identity

Date: 2026-07-15
Scope: agent-facing method identity, exact rerun routing, and provenance. This
document does not edit or promote a manuscript result.

## Stable identity

The preserved historical conventional v2 route resolves only through its
versioned request:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_accepted_refit_v2
request = --sr-route-profile sr_snake_v2
phase3_response_coordinate_scope = legacy_reopt_coupled_v1
```

This request materializes the preserved executable contract from
`pipelines/static_adapt/sr_snake_route_profile.py`. Its SHA-256 is:

```text
32d2bdf2b05818be6f4add74137447a313605d7ed35ffb880651863b793a0f64
```

The profile was registered from the three completed 2026-07-15 weak-Holstein
anchors: weak--weak, intermediate--weak, and strong--weak \(U=8\). The anchors
were launched before this profile name existed, so their old telemetry records
`sr_route_profile_request=off` and the lower-level v1 controller label. Their
normalized executable settings, frozen source archive, commands, and results
are the authority for this v2 identity. Do not rewrite those artifacts.

After v3 registration, the unqualified `sr_snake` token no longer resolves to
this route. Existing results remain v2 evidence and are not rewritten.

`SR-SNAKE v1` remains the explicit historical profile
`supported_whitened_adaptive_trust_v1`, requested with `sr_snake_v1`; its
contract SHA-256 remains
`fab7b5a6c4bd2ab019139367aa2a507356a5c969b6b88cd72d32365ae766e13e`.

## Defining controller contract

Conventional SR-SNAKE v2 fixes the following structure:

- Hubbard--Holstein `full_meta` pool with HVA operators included and no class
  or label filter;
- Phase 0 off;
- physical operator-type lanes in the Phase-I/II funnel;
- Phase-II and Phase-III batching off;
- archival exact-cardinality-one Pauli-child forwarding with subset size one;
- actual fixed-sector hard enforcement and exact projected binary-padding
  enforcement before child scoring;
- exactly one admitted candidate-position record per controller round;
- repeated generator identities allowed, with no admission rollback;
- Phase-III reoptimization-window-coupled response using
  `supported_metric_whitened_eigh_v1` and historical scope
  `legacy_reopt_coupled_v1`;
- selector whitening scope `phase3_only_v1`; Phase II is not whitened;
- branch-local adaptive trust policy `displacement_calibrated_unbounded_v2`;
- negative-curvature/saddle escape disabled;
- Phase-II and Phase-III ordinary novelty multipliers active;
- collective-span novelty-over-cost geometry expansion active only as the
  existing all-energy-models-infeasible fallback;
- Powell base chart `expanded_runtime_projected_logical_v1`;
- accepted-refit scope `full_ansatz_v1`;
- accepted-refit chart `supported_fs_whitened_fixed_v1`, built from the full
  accepted ansatz once per accepted-refit invocation and held fixed throughout
  that Powell solve;
- accepted-refit base chart `expanded_runtime_projected_logical_v1`;
- beam width three, two children per parent, and beam penalty `0.005`;
- `recoverability_ladder_v1` pruning in mode `both`;
- Hessian-coupling Schur nomination (`hessian_coupling_v1`) with measured
  delete-and-refit acceptance authority; the metric-regularized nomination
  variant is not active;
- windowed ordinary reoptimization with window size three, periodic full refit
  every eight rounds, final full refit enabled, and terminal pruning enabled by
  prune mode `both`;
- append-only insertion, Powell `maxiter=200`, SciPy `maxfev=0`, seed 7, and a
  30-round controller horizon.

The complete executable dictionary also pins shortlist caps, prune thresholds,
backend cost policy, symmetry/padding fields, fallback settings, debug/shadow
settings, and pool-selection fields. The Python contract is the
machine-readable authority; this list is a readable index.

Regime physics is deliberately outside the method profile. A run must obtain
\(U\), \(g\), the working phonon cutoff, and its exact same-cutoff reference
from a source-locked regime manifest. For all three weak-Holstein anchors,
`n_ph_work=n_ph_ref=2`.

## Expanded runtime and accepted-refit whitening

`expanded_runtime_projected_logical_v1` keeps one runtime parameter for every
executable Pauli factor. Before ansatz evaluation, every runtime block belonging
to one logical generator is projected to its block mean. Checkpoints retain the
expanded ordered vector.

After an admission, `supported_fs_whitened_fixed_v1` reconstructs the complete
accepted-ansatz Fubini--Study Gram matrix in that expanded/projected-logical
base chart, removes unsupported/null directions, and maps the supported modes
to orthonormal optimizer coordinates. The Gram chart is rebuilt once for that
accepted-refit invocation, not once per Powell energy evaluation. Powell then
uses the fixed chart for the duration of the solve.

This accepted-refit whitening is separate from the Phase-III candidate selector
whitening. Phase-II selector whitening remains off in this exact profile.

## Weak-Holstein anchors

| Regime | \(U\) | Final energy | Validated same-cutoff absolute error | Active depth | Rounds |
|---|---:|---:|---:|---:|---:|
| weak--weak | 0.25 | -0.9183531184618378 | 1.0373365499916076e-9 | 25 | 30 |
| intermediate--weak | 1.25 | -0.49499563910841426 | 2.8699265186560297e-13 | 27 | 30 |
| strong--weak \(U=8\) | 8.0 | 0.5264600841532714 | 1.3833534310281337e-6 | 24 | 30 |

The intermediate--weak result JSON embeds an independently recomputed
same-cutoff reference that differs from the locked validation reference by
about `4.1e-14`; it therefore reports `2.4596991110570343e-13` internally. The
table above uses the locked validation reference and replay error. Both use
phonon cutoff two.

The complete paths and SHA-256 hashes are recorded in:

```text
MATH/paper_facing/paper_I_static_scaffold/
  paper_i_sr_snake_conventional_v2_weak_holstein_anchors_20260715.json
```

The three normalized commands have SHA-256
`3cb4137465200196c2c0c71beba5dd387033cdc9f59285f65033f723c3e3a834`
after replacing only regime/output fields. Their normalized resolved settings
have SHA-256
`e8fd5a17c23d7c8f6015741491f3babec1bcee7867d57e2e5905e05c4e1b0446`;
the only scientific setting difference is \(U\).

## No-drift rule

Future agents replaying historical conventional v2 must use `sr_snake_v2`,
preserve `phase3_response_coordinate_scope=legacy_reopt_coupled_v1` and the
emitted contract/digest through command, manifest, checkpoint, resume, and
result records, and fail closed on a missing or conflicting profile. They must
not reconstruct this profile from current defaults. Unqualified `sr_snake`
means conventional v3 and is not a v2 replay request.

The following are not part of conventional v2 and require distinct named
profiles or source-locked ablations:

- Phase-II selector whitening;
- disabling either ordinary novelty multiplier or its geometry-expansion
  fallback;
- negative-curvature or modeled-minimum escape;
- reduced-logical Powell coordinates;
- metric-regularized Schur nomination;
- disabling recoverability pruning or terminal pruning;
- changing cost shaping, shortlist semantics, batching, refit schedules,
  optimizer budgets, or controller horizon.

No scientific run, CHTC submission, Paper-I manuscript edit, result promotion,
or PDF regeneration is authorized by this identity document.
