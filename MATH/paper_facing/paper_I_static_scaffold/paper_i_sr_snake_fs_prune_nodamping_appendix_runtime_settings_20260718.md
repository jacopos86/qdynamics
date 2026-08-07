# Paper-I SR-SNAKE undamped FS-trust pruning appendix

Date: 2026-07-18  
Scope: agent-facing one-factor route identity; not a manuscript edit, result
promotion, or run authorization.

## Stable request

```text
--sr-route-profile sr_snake_symmetric_cost_fs_prune_nodamping_v1
```

This resolves to:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_nodamping_v1
route_contract_sha256 = 272ede635558edb4acc2507ac3a9803d8ccec062b96c98634b8d6407df9fbc21
parent_route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1
parent_contract_sha256 = 69dcf6d5711a05b54cc143562bc6f437511d781164319b2705857b22deda1538
```

## One-factor delta from the main profile

Only pruning-scoped fields change or become explicit:

```text
phase1_prune_enabled = true
phase1_prune_policy = recoverability_ladder_v1
phase1_prune_mode = live
phase1_prune_max_candidates = 1
phase1_prune_local_window_size = 0
phase1_prune_recovery_trust_radius = 0.125
phase1_prune_schur_nomination_route = full_logical_fs_trust_delete_refit_v1
phase1_prune_metric_schur_mu = 0
phase1_prune_metric_schur_solve_mode = affine_deletion_global_trust_v1
phase1_prune_metric_schur_cost_weighting = off
phase1_prune_trust_update_policy = modeled_local_fs_conservative_v1
phase1_prune_metric_mu_update_policy = off
phase1_prune_endpoint_overlap_policy = off
```

`phase1_prune_local_window_size=0` is an explicit full-scope policy here: every
active logical coordinate enters the affine deletion response before genuine
Gram-null directions may be removed.  It is not an ambiguous optimizer-window
sentinel.

The initial branch-local Fubini--Study deletion radius is `0.125`.  A rejected
measured delete-and-refit trial contracts it by a source-locked factor of `0.5`
down to a `1e-8` floor; accepted trials hold the radius and the policy never
expands it.  Endpoint-overlap updates are off, so the trust update adds zero
quantum measurements.  Measured delete-and-refit energy remains the acceptance
authority.

Both scientific damping controls are inactive.  The deletion response begins
with `mu=0` and `phase1_prune_metric_mu_update_policy=off` keeps it zero.
Phase-III shadow damping remains `off`.  Numerical factorization tolerances are
unchanged and are not an `H + mu G` scientific damping policy.

## Inherited main-route contract

Every non-prune field is inherited exactly from
`sr_snake_no_prune_symmetric_cost_v1`:

```text
problem = hh
adapt_pool = full_meta
HH preseed = off
seed = 7
Phase 0 = off
Phase-II batching = off
Phase-III batching = off
admission cardinality = singleton
beam capacity = effective 1x1

phase1_energy_model = first_order_fs_trust_v1
phase2_curvature_policy = measured_required_fail_closed_v1
phase2_cheap_curvature_proxy_policy = off
phase2_gram_novelty_policy = fallback_only_v1
phase3_gram_novelty_policy = fallback_only_v1
phase3_hardware_cost_normalization_mode = family_robust_symmetric_arctan_v1

phase3_response_coordinate_scope = full_active_plus_singleton_v1
Phase-III supported whitening = on
adaptive trust = displacement_calibrated_unbounded_v2
accepted refit = full ansatz in supported-FS coordinates
Powell base chart = expanded_runtime_projected_logical_v1

periodic full refit = off
terminal full refit = off
terminal prune = off
finite-angle fallback = off
Phase-III rescue = off
negative-curvature escape = off
```

The all-energy-models-infeasible novelty fallback stays enabled with its full
fired/count/round/operator/reason/query telemetry.  Ordinary Phase-II and
Phase-III novelty multiplication stays off.

## Planned source lock

The appendix matrix is prepared as six fresh round-0 to round-50 jobs.  Weak
Holstein rows use working/reference cutoff `3/3`; strong Holstein rows use
`7/7`.  The bundle must remain fail-closed and unsubmitted until the frozen
main evidence is linked as the source-value anchor and the exact settings diff
reports no non-prune change.

