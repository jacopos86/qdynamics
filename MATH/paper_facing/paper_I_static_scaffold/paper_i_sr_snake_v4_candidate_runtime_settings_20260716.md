# Opt-In Paper-I SR-SNAKE v4 Candidate Runtime Identity

Date: 2026-07-16
Scope: agent-facing candidate-route resolution, exact rerun routing, and
provenance. This document does not edit, reinterpret, or promote a manuscript
result.

## Status and resolution

SR-SNAKE v4 is an opt-in candidate profile. It is not the unqualified
conventional route. The resolution boundary is:

```text
--sr-route-profile sr_snake
--sr-route-profile sr_snake_v3
  -> supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3

--sr-route-profile sr_snake_v4
  -> supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_v4
```

Therefore, an unqualified `sr_snake` request remains conventional v3. Agents
must not infer v4 from live defaults or activate it through component flags.
The current emitted v4 candidate contract SHA-256 is:

```text
b6331521fb55f4165e177466536b4e2a5834ff09205ab5532ea70de893f156bc
```

The resolved profile and digest must round-trip through command, normalized
settings, manifest, checkpoint, resume, and result records.

## V3 lineage retained by v4

V4 retains the conventional v3 family and full Phase-III response contract:

```text
route_family = singleton_response_snake
phase3_response_coordinate_scope = full_active_plus_singleton_v1
phase3_response_coordinate_count
  = active_logical_coordinate_count + 1
```

Every active pre-admission logical coordinate plus the singleton candidate
enters the Phase-III gradient, Hessian, and Fubini--Study Gram construction
before supported-rank reduction. Only genuine Gram-null modes may be removed
afterward. Powell/refit windows, periodic-refit cadence, and terminal policies
cannot change the Phase-III response indices.

The following v3 controls also remain unchanged:

- Phase 0 off;
- Phase-II and Phase-III batching off;
- singleton candidate-position admission;
- legacy HH quadrature preseed disabled, so round 1 starts from the reference
  state while HVA generators remain available in `full_meta`;
- fixed-sector and binary-padding hard enforcement;
- Phase-III supported-metric whitening;
- adaptive trust policy `displacement_calibrated_unbounded_v2`;
- Powell base chart `expanded_runtime_projected_logical_v1`;
- full accepted-ansatz refit in fixed supported-FS coordinates after each
  accepted admission;
- repeated generator identities allowed without admission rollback;
- negative-curvature, saddle, and modeled-minimum escape disabled.
- finite-angle energy-probe fallback disabled.

## V4 candidate perturbations

### Phase-I and Phase-II energy models

```text
phase1_score_mode = trust_region_v1
phase1_energy_model = first_order_fs_trust_v1
phase2_curvature_policy = measured_required_fail_closed_v1
phase2_cheap_curvature_proxy_policy = off
```

Phase I is a genuinely first-order screen.  After the existing feasibility
gates, its predicted gain is

```text
DeltaE1 = rho * [g_LCB]_+ / sqrt(F).
```

Here `F` defines only the Fubini--Study trust-domain bound.  It is not an
energy-curvature proxy or score bonus.  The historical
`-(1/2) lambda_F F alpha^2` term is inactive.

Every Phase-I survivor entering Phase II must carry a finite directional
curvature computed from the existing directional Hessian primitives and an
identity-bound measurement-provenance receipt.  Phase II clips a finite
negative curvature only through the existing `[h]_+` energy model; negative
curvature escape remains disabled.  Missing, malformed, nonfinite, or
unprovenanced curvature aborts the run.  V4 cannot substitute `lambda_F F`,
use `g^2/(2 lambda_F F)`, skip the candidate, or invoke the novelty fallback
to mask a curvature-construction failure.  These requirements add no
measurements beyond the Phase-II directional curvature already required by
the route.

### Ordinary novelty multiplier

```text
phase2_gram_novelty_policy = fallback_only_v1
phase3_gram_novelty_policy = fallback_only_v1
```

Ordinary Phase-II and Phase-III scores do not receive a Gram-novelty
multiplier. The all-energy-models-infeasible collective-span novelty fallback
remains active and continues to provide a bounded geometry-expansion path.
This is removal of the ordinary multiplier, not removal of the fallback.

### Symmetric hardware cost

```text
phase3_hardware_cost_normalization_mode = family_robust_symmetric_arctan_v1
```

For each live candidate population and hardware-cost component, the policy
centers costs at the population median, scales by a median absolute-deviation
quantity with a positive floor, and applies the bounded signed transform

```text
u_a = (2/pi) atan((c_a - median_a) / scale_a).
```

The lambda-weighted signed index produces the bounded multiplicative score
factor `1 - 0.5 u`. Relatively inexpensive candidates can therefore receive a
reward while relatively expensive candidates receive a penalty. Uniform
components are neutral. The policy applies consistently to Phase-I,
Phase-II, Phase-III, and all-energy-models-infeasible fallback scoring, and
records the population hash and signed cost telemetry. It introduces no new
quantum measurements.

### Effective 1x1 branch shape

```text
adapt_beam_live_branches = 1
adapt_beam_children_per_parent = 1
adapt_beam_terminated_keep = 0
adapt_beam_terminal_archive_mode = disabled
```

The v4 candidate retains the controller's branch machinery for serialization
compatibility but exposes exactly one live parent and one child. No discarded
beam frontier can affect admission.

### Live-only full-logical FS-trust pruning

```text
phase1_prune_enabled = true
phase1_prune_policy = recoverability_ladder_v1
phase1_prune_mode = live
phase1_prune_max_candidates = 1
phase1_prune_local_window_size = 0
phase1_prune_recovery_trust_radius = 0.125
phase1_prune_schur_nomination_route = full_logical_fs_trust_delete_refit_v1
phase1_prune_metric_schur_mu = 0.0
phase1_prune_metric_schur_solve_mode = affine_deletion_global_trust_v1
phase1_prune_metric_schur_cost_weighting = off
phase1_prune_trust_update_policy = modeled_local_fs_conservative_v1
phase1_prune_metric_mu_update_policy = same_trial_underprediction_monotone_v1
phase1_prune_endpoint_overlap_policy = off
```

Every active logical coordinate enters the affine deletion-response model. The
explicit Fubini--Study trust constraint applies to the complete deletion model,
not only to a survivor window. Nomination is a model decision; measured
delete-and-refit energy remains the acceptance authority.

`phase1_prune_endpoint_overlap_policy=off` means zero endpoint-overlap probes
and therefore zero new endpoint-overlap measurements. It is not a statement
that the physical endpoint overlap equals zero. Radius updates use the
zero-new-query modeled-local conservative policy encoded by the route.

Pruning is live-only. The route performs no terminal prune mutation.

### Zero-query shadow damping

```text
phase3_shadow_damping_policy = mapped_seed_zero_query_v1
phase3_shadow_damping_applied_mu = 0.0
phase3_shadow_damping_measurement_delta = 0
```

The shadow calculation is diagnostic telemetry. It reuses already available
mapped-seed response information, adds no quantum measurements, and must record
`applied_mu=0`. It cannot damp or otherwise alter the executed Phase-III
response model.

### Terminal behavior

```text
adapt_final_full_refit = false
phase1_prune_mode = live
```

Every accepted admission already receives its ordinary full accepted-ansatz
supported-FS refit. V4 adds neither a special final refit nor a terminal prune.
The reported terminal state is the state reached by the last completed
ordinary controller round, without terminal-only structural mutation.

### Initial ansatz

```text
adapt_disable_hh_seed = true
```

The controller starts from the reference state and grows only through recorded
singleton admissions. This disables only the legacy fixed HH quadrature
preseed; it does not remove HVA generators from `full_meta`.

### Finite-angle guard

```text
adapt_finite_angle_fallback = false
argv = --adapt-no-finite-angle-fallback
```

V4 does not use the finite-angle energy-probe guard. This setting is distinct
from the all-energy-models-infeasible collective-span novelty fallback, which
remains active. A v4 command, normalized settings record, checkpoint, resume
state, or result that enables finite-angle fallback is contract drift and must
fail closed.

## No-drift rule

An exact v4 candidate run must request `sr_snake_v4` and preserve the emitted
profile and digest. It must fail closed if any of these defining conditions is
missing or contradictory:

- full active-plus-singleton Phase-III response;
- first-order Phase-I energy model with `F` used only as the trust metric;
- measured-required fail-closed Phase-II curvature with both lambda-F proxies
  inactive;
- ordinary novelty multipliers off with fallback retained;
- symmetric arctangent cost shaping;
- effective 1x1 branch shape;
- live-only full-logical affine-deletion FS-trust pruning;
- endpoint-overlap probes off;
- shadow damping diagnostic only with `applied_mu=0`;
- no terminal full refit or terminal prune;
- negative-curvature and other escape routes off.
- finite-angle energy-probe fallback off.

Scientific validation of this candidate does not authorize changing the
unqualified `sr_snake` resolution. Advancing v4 to the conventional identity
requires a separate explicit decision and a corresponding route-contract
update.

This identity document authorizes no scientific run, CHTC submission,
Paper-I manuscript edit, result rewrite, result promotion, or PDF regeneration.
