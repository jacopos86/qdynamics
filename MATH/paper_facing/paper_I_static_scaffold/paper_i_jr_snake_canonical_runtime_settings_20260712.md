# JR-SNAKE Canonical Runtime Settings

Created: 2026-07-12

Status: controlling agent/run contract for the current Joint-Response SNAKE
Pareto campaign. This document does not edit or promote settings into
`MATH/paper_details/Paper_I.tex`.

## Authority And No-Drift Rule

This file controls new JR-SNAKE dry runs, scientific launches, continuations,
and matched comparisons until the user explicitly approves a revision.

Before every launch:

1. Materialize a dry-run normalized settings manifest.
2. Compare it to the settings below.
3. List every scientific difference.
4. Launch only when the difference list is empty or every difference was
   explicitly approved by the user.
5. Record the canonical settings id and file path in the campaign manifest.

Omitting a flag is not permission to inherit a different code default. A
historical/default value that differs from this contract must be passed
explicitly. In particular, fixed trust radius is diagnostic-only and must not
become canonical through a CLI default.

Canonical settings id:

```text
paper_i_jr_snake_canonical_runtime_settings_20260712_v1
```

## Route Definition

The canonical route is:

```text
macro Phase 1
-> macro Phase 2
-> singleton Pauli-child expansion
-> global normalized-Pauli identity deduplication
-> child Phase 1
-> child Phase 2
-> supported-metric-whitened full ansatz-plus-batch joint solve
-> combinatorial batch selection
-> exact guarded joint-step warm start
-> full Powell refit
-> prune
```

Canonical exclusions:

- Phase 0 is disabled.
- Macro Phase 3 is disabled.
- Child Phase 3 is disabled.
- Parent macros are expansion gateways and are not directly admissible.
- Child lanes, parent quotas, and parent-family diversity are disabled.
- Route B, Route C, parent-plus-child competition, legacy block pseudoinverse,
  fixed trust radius, and hard additivity gating are compatibility/diagnostic
  surfaces only.

## Canonical Mathematical Settings

### Candidate Funnel

```text
funnel_mode                         = child_12_joint_response_v2
phase0_policy                       = disabled
macro_phase1_cap                    = 32
macro_phase2_cap                    = 24
child_phase1_cap                    = 32
child_phase2_cap                    = 25
child_identity_policy               = global_pauli_word_v1
duplicate_cooldown_policy           = one_round_exact_record_pre_child_phase1_v1
child_symmetry_policy               = hard_guard
```

Child padding is cutoff-matched:

```text
n_ph_work = 2  -> nph2_exact_projected_grouped_v1
n_ph_work = 4  -> exact_projected_grouped_v1 with the resolved n_ph_max=4 context
```

### Joint Selector And Batching

```text
batch_mode                          = combinatorial_reduced_plane
batch_size_cap B_max                = 2
batch_search_pool_size L_search     = 10
batch_search_feasibility_policy     = joint_subset_gate_v1
batch_additivity_policy             = soft_penalty_v1
batch_additivity_lambda             = 0.0
joint_batch_context_mode            = full_ansatz_v1
joint_linear_solve_policy           = supported_metric_whitened_eigh_v1
batch_rank_relative_tolerance       = 1e-6
batch_max_gram_condition_number     = 1e12
batch_metric_regularization         = 1e-9
batch_energy_regularization         = 1e-9
batch_score_tie_tolerance           = 1e-12
```

Every singleton and compatible pair is scored by the same full
ansatz-plus-batch joint model. `B_max=1` is not a separate formula.

### Trust Region

```text
initial_fubini_study_radius          = 0.25
trust_region_update_policy          = displacement_calibrated_unbounded_v2
trust_region_radius_min             = 0
trust_region_radius_max             = none
trust_region_contraction_factor_min = none
trust_region_expansion_factor_max   = none
trust_region_displacement_epsilon   = 1e-12
trust_region_direction_cosine_min   = 0.5
require_direction_for_expansion     = false
```

The radius state is branch-local. It is updated after the authoritative full
refit and before pruning from

```text
R_FS = d_FS(psi_before, psi_after_refit) / d_FS_predicted.
```

For a valid displacement measurement, the canonical update is

```text
rho_(k+1) = rho_k * sqrt(R_FS).
```

There is no scientific lower floor, absolute upper bound, contraction clamp,
or expansion-rate cap. A machine-precision positive guard prevents an exact
floating-point underflowed zero from entering serialization or the linear
solver; it is numerical hygiene, not a scientific radius floor. Expansion is
allowed only for a binding radius with legitimate energy descent and realized
displacement larger than predicted.

The fixed policy and clamped `displacement_calibrated_v1` policy are retained
only for historical replay and explicit matched diagnostics.
Induced-sectional-curvature capping is not implemented in this canonical v1
contract and must not be claimed as active.

### Warm Start And Refit

```text
joint_step_warm_start_mode           = exact_applied_joint_step_guarded_v1
adapt_inner_optimizer                = POWELL
adapt_reopt_policy                   = full
optimizer_maxiter                    = 50
scipy_maxfev                         = 200
final_full_refit                     = true
```

The selector seed is initialization only. The exact objective guard compares
the mapped joint seed with the zero-inserted incumbent. Powell remains the
authoritative refit. Guard evaluations count in query work.

### Cost

Cost is enabled. The normalized weights are shared across the active macro,
child, batch, beam, and prune cost surfaces:

```text
lambda_2q   = 0.20
lambda_d    = 0.20
lambda_1q   = 0.05
lambda_theta= 0.05
lambda_shot = 0.15
```

Canonical selection uses predicted energy descent divided by `1 + K`. Cost
disablement, when explicitly requested as an ablation, sets the burden to the
neutral denominator rather than selecting a legacy score formula.

### Beam And Execution

```text
beam_width                          = 3
beam_children_per_parent            = 2
gradient_workers                    = 1
beam_parent_workers                 = 1
runtime_split_child_workers         = 1
joint_pair_workers                  = 1
candidate_cache_mode                = memory
result_payload_mode                 = summary_checkpoint_v1
checkpoint_every_controller_round  = true
```

Worker counts are operational and may change only when determinism and
oversubscription constraints are preserved. They are not scientific settings.

## Query And Resource Contract

The Paper-I query coordinate is winning-branch work:

```text
S_alg = N_H_outer + N_H_refit + N_grad + N_metric.
```

It includes warm-start guards, optimizer/refit evaluations, unique joint
Gram/Hessian measurements, boundary refits, and final refits. Reused measured
elements are charged once. Discarded-branch search work is reported separately
and must not be added to `S_alg`.

Qiskit reporting uses `FakeMarrakesh`, optimization level 1, transpiler seed 7,
and records `N2q`, `D2q`, and total circuit depth `Dc` for the exact selected
prefix.

## Regime Lock

The six regimes, working/reference phonon cutoffs, and exact same-cutoff
energies are resolved from the campaign manifest and locked Paper-I provenance:

```text
raw_outputs/paper_i_hh_jr_snake_whitened_pareto_goal_20260711/campaign_manifest.json
```

Do not reconstruct exact energies or cutoff pairs from memory.

## Current Evidence Classification

The completed six-regime screening matrix compares `L_search=10` with
`L_search=15` while holding the remaining route settings fixed. The user has
selected `L_search=10` for the current long-horizon JR-SNAKE campaign. This is
a campaign-level settings decision; it does not edit or promote a manuscript
setting.

The existing `L_search=10` prefixes are the approved continuation sources:

```text
weak-weak and intermediate-weak: source controller round 7
all other regimes:               source controller round 9
target total controller rounds:  30
```

Each continuation must use the prefix `current.json` as the structural-resume
authority and preserve every scientific setting above. The only approved plan
differences are structural resume metadata and the number of additional rounds
needed to reach round 30. Do not restart these campaign cells from round zero.

For plotting and final comparison, stitch each source prefix and continuation
into one trajectory with controller-round offsets. Report full-trajectory query
work as the validated sum of the prefix and continuation winning-lineage work;
do not mistake continuation-only work for the complete trajectory.

Fresh round-zero executions remain the later final-verification gate when
JR-SNAKE is compared with Formal-Manifold SNAKE. They are not part of the
current continuation queue.
