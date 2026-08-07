# Paper-I SR-SNAKE Phase-III projected generalized-trust ablation

Date: 2026-07-19  
Scope: agent-facing execution identity; no manuscript or canonical-profile promotion.

## Stable request and lineage

```text
--sr-route-profile sr_snake_no_prune_symmetric_cost_projected_phase3_v1

route_family = singleton_response_snake
route_profile = supported_projected_generalized_adaptive_trust_full_response_symmetric_cost_no_prune_v1
route_contract_sha256 = 3ff2abb1455cda3cf8cc2de0cf739172f8cdcfe6b1c9436e1afdd40076cd3ce8

parent_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1
parent_contract_sha256 = 023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91
```

The parent is the valid hysteresis-disabled Main SR source lock represented by
CHTC cluster 8887574. The child changes exactly one execution field:

```text
historical_singleton_coordinate_solve_policy:
  supported_metric_whitened_eigh_v1
  -> supported_metric_projected_generalized_trust_v1
```

## Phase-III response contract

Every controller round constructs the response gradient, Hessian, and raw
Fubini--Study Gram matrix over all active logical coordinates plus the one
singleton candidate. If

```text
G = V Lambda V^T,
```

then `V_s` retains only the eigenvectors above the registered raw-Gram support
threshold. The supported model is

```text
H_s = V_s^T H V_s
g_s = V_s^T g
Lambda_s = V_s^T G V_s
```

and the trust solve is performed directly in supported coordinates:

```text
(H_s + lambda Lambda_s) q = g_s
q^T Lambda_s q <= rho^2
delta_theta = V_s q
```

The Phase-III selector must not construct or apply `Lambda_s^(-1/2)`, a Gram
inverse, or a Gram pseudoinverse. The historical `metric_regularization`
setting remains serialized for configuration parity but is explicitly inactive
under this solve policy. Genuine Gram-null directions may be removed; no other
coordinate may disappear before the support decision.

Required round telemetry includes the response indices and pre-support count,
raw Gram spectrum and support threshold, retained mask/rank/eigenvalues,
projection provenance, supported metric displacement, trust multiplier,
supported generalized KKT residual, physical FS displacement, and explicit
`supported_metric_whitening_active=false` plus zero classical-to-quantum query
charge.

## Accepted refit remains whitened

This Phase-III ablation does not alter the coordinate-sensitive inner
optimization. After admission, Powell still receives the complete accepted
ansatz in:

```text
adapt_accepted_refit_scope = full_ansatz_v1
adapt_accepted_refit_coordinate_chart = supported_fs_whitened_fixed_v1
adapt_accepted_refit_base_chart_policy = expanded_runtime_projected_logical_v1
adapt_full_refit_every = 0
adapt_final_full_refit = false
```

Thus Phase III uses support projection plus the raw supported generalized FS
trust metric, while the accepted full-ansatz Powell refit retains explicit
supported-FS whitening as optimizer preconditioning.

## Unchanged parent settings

```text
Phase 0 = off
Phase II = unchanged; no Phase-II whitening
phase3_response_coordinate_scope = full_active_plus_singleton_v1
phase_live_hysteresis_enabled = false
ordinary Phase-II/III novelty multipliers = off (fallback_only_v1)
all-energy-models-infeasible novelty fallback = enabled with telemetry
hardware cost = family_robust_symmetric_arctan_v1
batching = off
singleton admission = on
beam = effective 1x1
pruning = off
negative-curvature escape = off
finite-angle fallback = off
Phase-III rescue = off
adaptive trust = displacement_calibrated_unbounded_v2
HH preseed = off
```

The controlled production matrix uses 50 controller rounds in all six regimes,
`n_ph=3` for weak-Holstein regimes, `n_ph=7` for strong-Holstein regimes, and
the same cutoff for execution and exact reference. Those per-regime fields
remain source-lock inputs rather than defaults in the method profile.

## Execution gate

This route is a source-locked sensitivity study. A source-value parent anchor
must first reproduce the validated Main SR contract under the new immutable
source archive. Only after that anchor passes and the non-swept settings diff is
empty may the six projected-generalized rows be submitted. Results do not
silently redefine the canonical SR identity.
