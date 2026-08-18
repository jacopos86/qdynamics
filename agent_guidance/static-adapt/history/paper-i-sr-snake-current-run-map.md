# Paper-I SR-SNAKE Current Run Map

> Storage note, 2026-08-17: `raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/`
> referenced below was removed in the user-authorized storage cleanup. See
> `agent_guidance/shared/storage-cleanup-20260817.md`. No identity in this
> document is changed.

Status: agent-facing navigation and execution aid.  
Current visible target: Paper-I Hubbard--Holstein SNAKE rows and trajectories
rendered from the 2026-07-21 first-hit support bundle.  
Not authorized by this document: CHTC submission, scientific setting changes,
manuscript edits, result replacement, route promotion, or artifact deletion.

## Why This File Exists

The current implementation exposes a large union of canonical, historical,
diagnostic, and experimental controls. Agents must not reconstruct the
Paper-I route from CLI defaults or from the unqualified `sr_snake` alias.

This file gives the shortest safe path from the visible Paper-I result to its
executable profile, immutable run bundle, runtime modules, and validation
artifacts. It complements rather than replaces:

- `AGENTS.md`;
- `MATH/AGENTS.md`;
- `agent_guidance/skills/paper-i-run/SKILL.md`;
- `agent_guidance/skills/source-locked-sensitivity/SKILL.md`;
- `agent_guidance/static-adapt/history/route-identities.md`;
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_sr_snake_canonical_runtime_settings_20260716.md`.

## Critical Identity Distinction

There are two different objects that older guidance can call canonical:

1. **Conventional executable alias**
   - request: `--sr-route-profile sr_snake`
   - versioned alias: `sr_snake_v3_1`
   - profile:
     `supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3_1`
   - contract SHA-256:
     `9b96179935ed80967a3335dfbbf8eece86a04c2d412e6b92aa8a466fa6913542`
   - authority:
     `MATH/paper_facing/paper_I_static_scaffold/paper_i_sr_snake_canonical_runtime_settings_20260716.md`

2. **Current SR route shown in Paper I**
   - request:
     `--sr-route-profile sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1`
   - profile:
     `supported_projected_generalized_source_metric_no_overlap_trust_full_response_symmetric_cost_no_prune_v1`
   - contract SHA-256:
     `fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2`
   - visible support route id:
     `no_overlap_trust_projected_phase3_nph3_7`

The second object generated the current Paper-I HH SNAKE figure and table.
Running `--sr-route-profile sr_snake` does **not** reproduce that visible row.

## Visible Paper-I Authority

Read these in order:

1. Visible source/provenance block:
   `MATH/paper_details/Paper_I.tex`, block
   `BEGIN_MACHINE_READABLE_PAPER_I_FIXED_ACCURACY_RESULTS_20260721`.
2. Current support JSON:
   `MATH/paper_details/figures/paper_i_hh_first_hit_20260721/paper_i_hh_first_hit_results_support_20260721.json`.
3. Runtime post-run `S_alg` audit:
   `raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/paper_i_no_overlap_runtime_postrun_s_alg_audit.json`.
4. Frozen six-row input bundle:
   `chtc/phase3_optuna/input/paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v4_chtc/`.
5. Recovered/fetched evidence:
   `raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/`.

The support JSON reports all six SNAKE regimes as target hits at
`E_T=2e-4`. It also identifies first-hit prefixes and Qiskit/`S_alg` values.
Do not substitute terminal values when the visible row uses first-hit
reporting.

## Current Visible Scientific Contract

```text
route_family = singleton_response_snake
profile_request =
  sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1
route_profile =
  supported_projected_generalized_source_metric_no_overlap_trust_
  full_response_symmetric_cost_no_prune_v1
contract_sha256 =
  fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2
```

Structural settings:

```text
Phase 0 = off
Phase I = first_order_fs_trust_v1
Phase-II whitening = off
ordinary Phase-II novelty multiplier = off; infeasible-model fallback retained
Phase-III response = full active logical ansatz plus one singleton
Phase-III support projection = on
Phase-III supported whitening / inverse square root = off
Phase-III trust solve = raw supported-metric generalized KKT
trust update = source_metric_inverse_sqrt_no_overlap_v1
endpoint-overlap measurement = off
endpoint-overlap query charge = zero
accepted refit = complete accepted ansatz
accepted-refit chart = supported_fs_whitened_fixed_v1
accepted-refit base chart = expanded_runtime_projected_logical_v1
optimizer = POWELL
optimizer maxiter = 200
admission = exactly one candidate-position record
Phase-II batching = off
Phase-III batching = off
beam = effective 1x1
pruning = off
ordinary Phase-III novelty multiplier = off; infeasible-model fallback retained
negative-curvature escape = off
finite-angle fallback = off
periodic full refit = off
terminal full refit = off
terminal prune = off
HH preseed = off
seed = 7
pool = unfiltered full_meta with HVA included
physical HH lanes = on
Pauli-child forwarding = exact cardinality one
fixed-sector and binary-padding enforcement = on
```

The accepted Powell refit may use its configured local optimizer policy while
the accepted-refit scope remains the complete accepted ansatz in the supported
FS chart. Do not infer Phase-III response coordinates from a Powell window.

## Physics Grid

All rows use same-cutoff exact diagonalization:

| Regime id | Display label | `U/t` | Holstein sector | `n_ph_work` | `n_ph_ref` | Horizon |
|---|---|---:|---|---:|---:|---:|
| `weak_weak` | weak--weak | 0.25 | weak | 3 | 3 | 50 |
| `intermediate_weak` | intermediate--weak | 1.25 | weak | 3 | 3 | 50 |
| `strong_weak_u8` | strong--weak | 8 | weak | 3 | 3 | 50 |
| `weak_strong` | weak--strong | 0.25 | strong | 7 | 7 | 50 |
| `intermediate_strong` | intermediate--strong | 1.25 | strong | 7 | 7 | 50 |
| `strong_strong_u8` | strong--strong | 8 | strong | 7 | 7 | 50 |

Primary error:

```text
abs(E_method(n_ph_work) - E_exact(n_ph_work))
```

Do not interpret legacy `strong_weak` or `strong_strong` keys as the `U/t=8`
rows without explicit source-map evidence. Use the `_u8` ids above.

## Scientific Data Flow

```text
Resolved HH problem and exact same-cutoff reference
  -> unfiltered full_meta pool
  -> physical-lane candidate-position records
  -> Phase-I first-order FS-trust screen
  -> exact-cardinality-one legal Pauli children
  -> Phase-II measured curvature response
  -> lane-free Phase-III full active-plus-singleton H/G model
  -> raw supported-metric generalized trust solve
  -> singleton admission
  -> complete accepted-ansatz supported-FS-whitened Powell refit
  -> source-metric no-overlap radius update
  -> estimator-ledger closure and checkpoint
  -> next controller round
```

## Runtime Code Map

Start at the named module; do not search the repository from root.

| Question | Read |
|---|---|
| Which SR family/profile is this? | `pipelines/static_adapt/sr_snake_route_profile.py` |
| How do legacy Route-A fields translate? | `agent_guidance/static-adapt/history/route-a-language.md` |
| Where is the current orchestration? | `pipelines/static_adapt/adapt_pipeline.py::_run_hardcoded_adapt_vqe` |
| How are problems and pools resolved? | `pipelines/static_adapt/builders/` |
| How are Phase-I/II/III records scored? | `pipelines/scaffold/hh_continuation_scoring.py` plus route helpers |
| Where is supported H/G trust solved? | `pipelines/static_adapt/joint_linear_solve.py` |
| Where is the accepted refit chart built? | `pipelines/static_adapt/accepted_refit.py` |
| Where is pruning planned? | `prune_ladder.py`, `prune_derivatives.py`, `prune_schur_payloads.py` |
| Where is beam handled? | `pipelines/static_adapt/beam_search.py` |
| Where is `S_alg` ledgered? | `pipelines/static_adapt/estimator_call_ledger.py` |
| Where are checkpoints/results assembled? | `checkpoint_telemetry.py`, `output_artifacts.py` |
| Where are Qiskit costs produced? | Paper-I exact-bench/recovery Qiskit sidecar scripts named by the source bundle |

## Safest Way To Prepare Another Run

Do not hand-assemble the large `adapt_pipeline` command.

1. Declare whether the run is `smoke`, `diagnostic`, `candidate`, or
   `paper_facing`.
2. Name the visible target and exact source row.
3. Start from the frozen v4 bundle, its normalized manifest for the chosen
   regime, and its immutable source archive.
4. Record the source paths and SHA-256 values before changing anything.
5. Materialize a complete child manifest.
6. Compute a normalized settings diff against the source.
7. For a one-variable run, build and execute the source-value anchor first.
8. Require:
   - `anchor_reproduces_source=true`;
   - exact non-swept settings equality;
   - expected operator/checkpoint identity when exact trajectory reproduction
     is part of the contract.
9. Only after the anchor passes, build the requested rows.
10. Run local/archive/exact-image preflight against the exact uploaded source
    and container image.
11. Submit only with current user authorization.
12. Verify scheduler admission once, then use adaptive monitoring.

The governing one-variable procedure is
`agent_guidance/skills/source-locked-sensitivity/SKILL.md`. Do not use Optuna,
current defaults, or an unrelated wrapper as a replacement for source locking.

## Existing Frozen Bundle

Baseline bundle:

```text
chtc/phase3_optuna/input/
paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v4_chtc/
```

Important contents:

```text
jobs/<regime>.json
normalized_manifests/<regime>.json
physics_and_exact_reference_lock.json
source_revision_manifest.json
source_archive_manifest.json
source_locked.tar.gz
run_job.py
preflight.json
```

This bundle is evidence and a source template. Do not resubmit it merely to
obtain a new label. A new scientific variation needs a new immutable child
bundle and settings-diff audit.

## Planned Near-Term One-Variable Studies

These are navigation notes, not submission authorization. Each must begin from
the current visible route above and change only the named policy.

### Phase-I standard ADAPT selector

Intended change:

```text
Phase-I selector:
  first_order_fs_trust_v1 / current configured score
  -> conventional insertion-point ADAPT score |g_mu|
```

Keep Phase II, Phase III, trust, refit, pool, seed, cutoffs, horizon, pruning,
beam, and accounting fixed.

### Pruning-only study

The requested first pruning study is a **query-neutral, full-geometry
trust-region prune policy** layered directly on the current visible no-overlap
route. It is not the material-window ablation and it is not the historical
measured keep-versus-delete verification beam.

Frozen parent settings:

```text
parent_profile_request =
  sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1
parent_contract_sha256 =
  fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2
phase3_response_coordinate_scope = full_active_plus_singleton_v1
phase3_material_window_policy = off
accepted_refit_scope = full_ansatz_v1
accepted_refit_coordinate_chart = supported_fs_whitened_fixed_v1
endpoint_overlap_measurement = off
endpoint_overlap_query_charge = 0
admission = singleton
beam = effective_1x1
batching = off
```

Only intended scientific change:

```text
pruning:
  off
  -> live_query_neutral_full_geometry_fs_trust_v1
```

The policy reuses the already measured selected-candidate Phase-III
active-plus-singleton gradient, Hessian, and Fubini--Study Gram workspace. It
must not acquire a prune-only derivative, Gram, Hessian, endpoint-overlap,
energy, or refit measurement. In particular:

```text
prune_nomination_count_per_round_max = 1
prune_source_geometry = selected_phase3_full_active_plus_singleton_v1
prune_source_geometry_query_delta = 0
prune_verification_beam = off
prune_delete_refit_sibling = off
prune_endpoint_overlap_measurement = off
prune_explicit_query_delta = 0
terminal_prune = off
```

The prune trust radius is independent from the admission trust radius and
starts conservatively at `0.00390625`. A
coordinate may be nominated only when its complete affine-deletion
displacement lies inside the prune FS radius and its conservative modeled
energy change is at most `-2e-6`, one percent of the Paper-I
`E_T=2e-4` target. This excludes numerically null deletion predictions rather
than spending the ordinary round refit on them. A failed combined
transition shrinks the prune radius and places the rejected coordinate on
cooldown until the accepted state has moved materially in the FS metric or its
conservative modeled loss has improved by the configured factor. This
hysteresis must prevent consecutive remeasurement-free nominations of an
unchanged failed deletion model.

Because a second unpruned refit would violate the zero-extra-query contract,
the proposed executable semantics are one combined transition:

```text
singleton admission + optional certified deletion
  -> one ordinary complete accepted-ansatz Powell refit
  -> accept only if E_after <= E_before + 1e-12
  -> otherwise restore the pre-round accepted state classically
  -> do not perform a second refit in that round
```

This combined-transition rule is a scientific route decision and must remain
explicit in the route contract, tests, and receipts. Do not silently preserve
the new admission by running another unpruned refit.

The ledger must prove that pruning introduced no dedicated estimator events:

```text
Delta N_H_outer(prune-only) = 0
Delta N_H_refit(prune-only) = 0
Delta N_grad(prune-only) = 0
Delta N_metric(prune-only) = 0
```

Do not accept those four zero fields as self-authenticating telemetry. The
post-run validator must independently reconstruct `S_alg` from the raw unique
primitive entries and reconcile every ordered raw occurrence. It must also
prove, round by round, that:

```text
all N_H_refit occurrences use the ordinary energy:depth_opt scope
raw N_H_refit occurrences = nfev_step_total_delta
accepted-refit N_metric occurrences = n_refit (n_refit + 1) / 2
prune / rollback / energy-guard estimator scopes = 0
branch estimator occurrences = 0
discarded-branch S_alg = 0
all-work S_alg = winning-lineage S_alg
```

Any dedicated prune scope, second refit, branch consumer, missing primitive
identity, or mismatch between raw entries, raw occurrences, runtime receipts,
and post-run accounting fails closed. This distinguishes zero direct
prune-query overhead from trajectory-induced work, which remains fully charged.

Ordinary accepted-refit work remains real work, and pruning may change the
later trajectory or Powell evaluation count. Therefore zero explicit
prune-query overhead does not by itself imply lower total `S_alg`; the
six-regime comparison must report the observed first-hit and terminal totals.

Before fanout, register a distinct route/profile identity and prove a
source-value anchor with pruning off. The anchor must reproduce the parent
operator sequence, controller energies, checkpoints, target-hit metric,
ledger/`S_alg`, fidelity, and Qiskit sidecar through the parent's first hit.
Only after
`anchor_reproduces_source=true` may the six candidate rows run at the frozen
`n_ph=3/7`, seed, pool, symmetry/padding, optimizer, and same-cutoff settings.
Each row has a 50-round safety cap but stops immediately after its first
accepted `E_T=2e-4` crossing; no post-hit prune round is executed.

Do not combine this first test with a material geometry window, a Phase-I
selector change, batching, an admission beam, or any separate prune
verification branch.

### Exact Pauli-orbit Phase-I gain

For an individual involutory Pauli child `P_mu^2=I`, the exact one-coordinate
landscape is:

```text
E_mu(theta) = A_mu + B_mu cos(theta) + C_mu sin(theta)
g_mu = C_mu
kappa_mu = -B_mu
```

Candidate selectors are distinct:

```text
standard ADAPT:      |g_mu|
max-orbit gradient:  sqrt(g_mu^2 + kappa_mu^2)
exact orbit gain:    -kappa_mu + sqrt(g_mu^2 + kappa_mu^2)
```

The first finite-angle study should prefer the exact-gain selector because it
measures accessible descent from the current insertion point. The max-gradient
amplitude can be nonzero even when the insertion point is already the
coordinate minimum.

Restrictions:

- apply the exact sinusoid only to individual Pauli involutions;
- do not silently apply it to arbitrary Pauli polynomials or macro generators;
- charge the additional `E_mu(pi)` or equivalent curvature acquisition to
  `S_alg`;
- keep later-stage policies fixed in the first source-locked test.

Suggested explicit policy identities:

```text
phase1_selector = adapt_gradient_v1
phase1_selector = pauli_orbit_max_gradient_v1
phase1_selector = pauli_orbit_exact_gain_v1
```

These names are proposed documentation identities until registered and tested.
Do not pass them to the CLI unless the executable registry contains them.

## Required Completion Evidence

For each completed row, preserve and validate:

- source/profile/batch/regime identity;
- route contract and SHA-256;
- exact 50-round or explicitly approved stop contract;
- same-cutoff physics;
- leakage, symmetry, and padding gates;
- ordered accepted operators;
- controller energies and checkpoints;
- Phase-III response counts, support ranks, and trust receipts;
- accepted-refit scope/chart and optimizer receipts;
- endpoint-overlap measurement and query count;
- estimator ledger and strict `S_alg` closure;
- first-hit and terminal errors;
- fidelity;
- Qiskit `N_2q`, `D_2q`, and `D_c` sidecar;
- archive hashes and validation status.

Measurement accounting remains:

```text
S_alg = N_H_outer + N_H_refit + N_grad + N_metric
```

If branching or prune trials are enabled, additionally report all-work,
winning-lineage, shared-source, and rejected/discarded-branch work. Parallel
execution does not erase measured rejected-branch work.

## Failure And Repair Rule

Implementation, packaging, transfer, serialization, validator, queue, or quota
failures may be repaired without changing the scientific contract. Prove the
repair narrowly, then resume the same scientific command.

Ask before changing:

- Hamiltonian or cutoff;
- seed;
- route/profile identity;
- pool or candidate representation;
- Phase-I/II/III mathematics;
- trust policy;
- pruning/beam/batching;
- optimizer/refit policy;
- stopping or evidence semantics.

## Main-Agent Live Notes

The main execution agent may append dated operational facts here. Keep this
section concise and never use it to override the authority chain above.

```text
YYYY-MM-DD
Owned run/study:
Source profile and SHA:
Only intended setting change:
Builder/bundle:
Preflight:
Scheduler or local status:
Fetched evidence:
Validation:
Next action:
```
