# FM-SNAKE SR-Selector Canonical Runtime Settings

Created: 2026-07-14

Status: controlling agent/run contract for the requested Formal-Manifold SNAKE
profile. This support document does not edit or promote settings into
`MATH/paper_details/Paper_I.tex` and does not authorize a scientific launch by
itself.

Canonical profile id:

```text
sr_phase2_phase3_whitened_adaptive_trust_no_n2_v1
```

Executable registry:

```text
pipelines/static_adapt/formal_manifold_route_profile.py
```

## Route Identity

```text
route_family = formal_manifold_snake
route_profile = sr_phase2_phase3_whitened_adaptive_trust_no_n2_v1
adapt_reoptimization_route = formal_manifold_warm_start_v1
candidate_selector_family = singleton_response_snake
candidate_selector_profile = sr_singleton_controller_v1
```

FM is the outer route. It exclusively owns accepted-ansatz reoptimization,
supported-whitened tangent/frame state, transport, qBroyden/qBANG state,
recycled objective curvature, and transactional proposal/commit/rollback.
The SR singleton controller is the inner candidate selector only. It must not
invoke JR selection or the FM combinatorial selector, and it must not own or
mutate FM manifold state.

## Formal-Manifold Reoptimization Cadence

The current user-locked FM reoptimization policy is:

```text
adapt_maxiter = 100
adapt_final_refit_maxiter = 100
formal_metric_mode = sparse_anchor_qbroyd_recycled_v1
formal_transport_policy = predicted_transported_frame_gauge_v1
qbroyd_epsilon0 = 0.15
qbang_momentum_active = false
line_search_max_steps = 15
post_anchor_exact_gram_budget = 4
```

Every reoptimization begins from one exact initial/growth Gram anchor. The four
post-anchor exact Gram evaluations are a hard total budget, not four interior
evaluations plus a fifth terminal evaluation. For a fully accepted 100-step
episode the nominal anchors occur after accepted steps 25, 50, 75, and 100;
the fourth is the charged terminal validation. An earlier hard qBroyden event
may consume the next interior slot. The route must fail closed rather than
silently measure a fifth post-anchor Gram.

Between exact anchors, qBroyden's predicted inverse metric is authoritative.
Rejected Armijo trials request energy only, accepted non-anchor endpoints
request energy plus gradient, and the declared transported-frame gauge uses
the explicit approximate map `Q = I`. The next exact anchor corrects that
predicted gauge with measured endpoint Procrustes transport before recycled
curvature is reused. The label `qBANG` here does not activate Adam-style
first/second-moment momentum.

## Locked Selector Structure

```text
route_a_funnel_active = false
phase0_pilot_enabled = false
phase3_runtime_split_mode = shortlist_pauli_children_v1
phase3_runtime_split_selection_mode = archival_child_set_forward_v1
phase3_runtime_split_subset_sizes = [1]
phase3_runtime_split_child_set_symmetry_policy = hard_guard
phase3_runtime_split_child_padding_policy = exact_projected_grouped_v1
phase2_enable_batching = false
phase3_enable_batching = false
candidate_response_model = full_active_plus_singleton_v1
admission_cardinality = 1
prune_policy = recoverability_ladder_v1
structural_rollback_enabled = false
```

The child expansion may expose several singleton children, but admission is
exactly one candidate-position record. This is batching disabled, not a
batch-size-one JR selector.

## Supported Coordinates And Trust

```text
historical_singleton_coordinate_solve_policy = supported_metric_whitened_eigh_v1
historical_singleton_coordinate_solve_scope = phase2_and_phase3_v1
historical_singleton_trust_region_update_policy = displacement_calibrated_unbounded_v2
sr_powell_coordinate_chart_policy = expanded_runtime_projected_logical_v1
sr_escape_mode = disabled
```

Both Phase II and Phase III use the repository's shared supported-metric
whitening primitive and its rank, retained-eigenspace, threshold, ridge, and
spectral telemetry conventions. The expanded-runtime projected-logical Powell
chart is an explicit FM-profile choice; it does not change the standalone SR
Phase-II+III profile's reduced-logical convention.

## Novelty Contract

```text
phase3_novelty_ablation_mode = no_phase2
phase2_novelty_multiplier_policy = inactive_ordinary_route_v1
phase3_novelty_multiplier_policy = inactive_ordinary_route_v1
measured_n2_retained = true
additional_n3_multiplier_applied = false
```

Novelty geometry is still measured and serialized for diagnostics. It does
not affect ordinary selection:

```text
S2 = DeltaE_TR_supported_metric_joint / (1 + K2)
S3 = DeltaE_TR / (1 + K3)
```

Required telemetry distinguishes measured quantities from applied factors:

```text
phase2_measured_novelty
phase2_novelty_multiplier = 1
phase2_novelty_applied = false
phase3_measured_novelty
phase3_novelty_multiplier = 1
phase3_novelty_applied = false
```

Existing standalone SR profiles retain their previous N2/N3 scoring behavior
unless this explicit FM profile is selected.

## No-Drift And Serialization Gate

Before a run using this profile, its invocation manifest, live checkpoint,
resumed checkpoint, terminal result, query sidecar, and reporting row must all
record the same outer family/profile, inner selector provenance, whitening
scope, trust policy, Powell chart, novelty multiplier policies, batching state,
and structural-rollback state. A missing or mismatched field is a failed
preflight rather than permission to infer a default.

All other scientific controls—Hamiltonian, regime/cutoff pair, exact energy,
pool, shortlist caps, cost weights, optimizer budget, seeds, stopping horizon,
and query-accounting convention—must be inherited from the user-approved
matched comparison source lock. This document intentionally does not invent
replacement values for them.
