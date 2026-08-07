# Paper-I SR-SNAKE no-prune symmetric-cost candidate

Date: 2026-07-17  
Scope: agent-facing execution identity; not a manuscript edit or result promotion.

## Stable request

```text
--sr-route-profile sr_snake_no_prune_symmetric_cost_v1
```

This resolves to:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1
route_contract_sha256 = 023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91
```

The unqualified `sr_snake` alias remains conventional v3.

## Executable method contract

```text
problem = hh
adapt_pool = full_meta
HVA exposure = included, with no pool filters
HH preseed = off
seed = 7

Phase 0 = off
Phase-II batching = off
Phase-III batching = off
admission cardinality = singleton
beam capacity = 1x1 (effective beam off)
phase_live_hysteresis_enabled = false
pruning = off

phase1_energy_model = first_order_fs_trust_v1
phase2_curvature_policy = measured_required_fail_closed_v1
phase2_cheap_curvature_proxy_policy = off
phase2_gram_novelty_policy = fallback_only_v1
phase3_gram_novelty_policy = fallback_only_v1
phase3_hardware_cost_normalization_mode = family_robust_symmetric_arctan_v1

historical_singleton_coordinate_solve_scope = phase3_only_v1
historical_singleton_coordinate_solve_policy = supported_metric_whitened_eigh_v1
phase3_response_coordinate_scope = full_active_plus_singleton_v1
historical_singleton_trust_region_update_policy = displacement_calibrated_unbounded_v2

accepted_refit_scope = full_ansatz_v1
accepted_refit_coordinate_chart = supported_fs_whitened_fixed_v1
accepted_refit_base_chart = expanded_runtime_projected_logical_v1
periodic full refit = off
terminal full refit = off

finite-angle fallback = off
Phase-III rescue = off
negative-curvature escape = off
shadow damping = off
```

Here `undamped` means that no scientific metric-damping model of the form
`H + mu G` is active. The unchanged numerical solver still retains the
historical `1e-6` Hessian ridge used to stabilize Schur factorization and the
`1e-9` supported-solve regularization tolerances. These numerical guards are
not the optional metric-damping mechanism and must be serialized separately
from it.

The controller horizon is a mandatory per-regime source-lock field rather than
a profile default. The approved completion study uses exactly 50 completed
controller rounds for every one of the six regimes. Omitting
`--adapt-max-depth`, supplying a value other than 50 in that study, or supplying
a nonpositive value fails before execution.

## Novelty safety receipt

Ordinary Phase-II and Phase-III novelty are not computed under
`fallback_only_v1`: the projection/solve is skipped, the measured novelty and
multiplier remain null, and both novelty classical-solve and quantum-query
charges remain zero. This is not the historical mode that computes novelty and
then neutralizes its multiplier. Gram/Hessian geometry still executes where it
is independently required by the response model, supported whitening, and FS
trust constraint. The bounded all-energy-models-infeasible novelty fallback
remains enabled and may compute novelty lazily only when every validated energy
model is infeasible. Every current checkpoint and final result must include:

```text
all_energy_models_infeasible_novelty_fallback_telemetry:
  policy
  enabled
  fired
  activation_count
  controller_rounds
  selected_operators
  reason_counts
  query_charge_total
```

Missing or invalid Phase-II curvature aborts before Phase III and can never be
converted into a fallback event. A report may omit discussion of this safety
path only when the preserved receipt explicitly records `fired=false`; absence
of evidence is not a zero-fire receipt.

## No-drift requirements

- Same working/reference phonon cutoff for each regime.
- Every pre-support Phase-III response contains every active logical
  coordinate plus the singleton candidate.
- Only genuine Gram-null directions may be removed after all coordinates enter.
- Full accepted-ansatz supported-FS Powell refit follows every admission.
- Powell telemetry identifies this registered outer route and labels
  `supported_whitened_adaptive_trust_v1` only as its base controller.
- No prune nomination, delete/refit trial, prune measurement, terminal prune,
  finite-angle probe, Phase-III rescue, beam branch, periodic refit, or terminal
  refit may occur.
- Phase-I/II lambda-F controls are inactive and explicit attempts to restore
  either proxy fail closed.
- Symmetry, fixed sector, binary padding, ordered checkpoints, and estimator
  ledger closure remain mandatory.

## Appendix-only beam ablation

The historical beam comparison is a separate registered request, not a change
to the main effective-1x1 profile:

```text
--sr-route-profile sr_snake_no_prune_symmetric_cost_beam_v1

route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_beam_v1
parent_route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1
route_contract_sha256 = 49fb8c2f069722ce87cbaaedc8d7d32726a11dad92a624e3326269d75dcd1168
adapt_beam_live_branches = 3
adapt_beam_children_per_parent = 2
adapt_beam_terminated_keep = 3
adapt_beam_terminal_archive_mode = legacy
adapt_beam_lambda = 0.005
```

At each round, at most three live parents each yield at most two admission
children, so at most six admission children are expanded and at most three
remain live. The `legacy` archive mode additionally materializes the stopped
version of every expanded parent, even when that parent has admission
proposals, then cumulatively retains the best three terminated branches. This
is the exact historical `stop_or_single_admission` behavior; a policy that
archives only parents that otherwise stop is not equivalent.

These are the only settings allowed to differ from the main profile. Pruning,
batching, ordinary novelty multipliers, cost shaping, response coordinates,
accepted refitting, and all safety policies remain identical. This profile is
appendix evidence only and does not redefine the main route or the unqualified
`sr_snake` alias.
