# Static SNAKE Route-Family Registry

Purpose: provide the stable agent-facing method identities for the active
Paper-I static SNAKE comparison. This registry separates enduring controller
structure from profile switches, legacy field names, optimizer choices, and
individual run labels.

## Identity Contract

Every new comparison manifest, run report, handoff, and provenance note should
record these concepts separately:

```text
route_family
route_profile
legacy_compatibility_fields
source_lock
```

`route_family` answers which controller generated the ansatz trajectory.
`route_profile` records policies within that family, such as coordinate solve,
trust update, pruning, cost weighting, shortlist caps, and stopping. Legacy
fields remain readable for preserved evidence but must not override a resolved
family identity.

Insertion has one cross-family execution invariant. An open insertion domain
is always reduced to one deterministic earliest representative per exact
termwise cross-component commutation class before scoring. The typed
always-open policy is `always_commutation_reduced` and its runtime mode is
`full_commutation_reduced`. Raw unreduced `full` insertion and its route
profiles are retired. The former capped-domain runtime mode named `always` is
also retired because always-open now means the complete logical domain;
append-only rounds remain append-only.

## Active Families

| Display name | Stable family id | Structural identity | Current executable anchors |
|---|---|---|---|
| **JR-SNAKE** | `joint_response_snake` | Macro-to-child funnel followed by a full active-plus-batch response model. Candidate batches are proposed, modeled, mapped, refit, and admitted under the JR batch-selection semantics. | `route_a_funnel_mode=child_12_joint_response_v2`; JR selector and joint-response payloads; batching policy and batch cardinality fields. |
| **FM-SNAKE** | `formal_manifold_snake` | Branch-local propagated metric/frame/curvature state with a profile-owned accepted reoptimizer. Existing pure-FM and source-locked profiles retain formal-manifold reoptimization; `sr_v3_outer_information_shadow_v1` instead retains the immutable conventional SR-v3 selector and accepted supported-FS Powell refit while FM owns only the outer-information shadow. FM-SNAKE must not invoke the JR selector or share JR mutable state. | Formal-manifold outer-state checkpoint plus query-closure and curvature telemetry; `adapt_reoptimization_route=formal_manifold_warm_start_v1` for the existing formal-reoptimization profiles; `accepted_reoptimizer_owner=source_locked_sr_v3_supported_fs_powell_v1` for the SR-v3 shadow profile. |
| **SR-SNAKE** | `singleton_response_snake` | Cost-weighted Phase-I-to-III singleton funnel. Phase 0 is off; Phase-II and Phase-III batching are off; the conventional v3.1 profile evaluates each retained Phase-III candidate in the full active-plus-singleton coordinates before exactly one candidate-position record is admitted. | singleton admission fields; `phase0_pilot_enabled=false`; `phase2_enable_batching=false`; `phase3_enable_batching=false`; `phase3_response_coordinate_scope=full_active_plus_singleton_v1`; `phase_live_hysteresis_enabled=false`; historical-singleton coordinate/trust payloads retained as compatibility names. The frozen v3 digest remains replay-only. |

The families are peers. FM-SNAKE is not an SR-SNAKE optimizer label, and
SR-SNAKE is not a batch-size-one JR-SNAKE run. A JR proposal can end with one
admitted record while still having used a different joint batch frontier and
therefore remaining JR-SNAKE.

## FM-SNAKE SR-v3 Outer-Information Shadow

This profile is an FM route because it owns branch-local outer-information
state; it is not a renamed SR route and does not modify the conventional SR-v3
controller:

```text
route_family = formal_manifold_snake
route_profile = sr_v3_outer_information_shadow_v1
candidate_selector_family = singleton_response_snake
sr_controller_route_profile = supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3
accepted_reoptimizer_owner = source_locked_sr_v3_supported_fs_powell_v1
outer_information_mode = shadow_v1
structural_rollback_enabled = false
```

The conventional SR-v3 selector, singleton admission, and accepted full-ansatz
supported-FS Powell refit are immutable route inputs. FM owns only the
branch-local outer-information shadow: same-ray frame extension, predicted
inter-state metric/frame transport, closure state, curvature state, and their
transactional checkpoint provenance. The shadow consumes the authoritative
cached Phase-III gradient, Hessian, and Fubini--Study Gram blocks; converting
that cache creates no new quantum primitive and never invokes JR selection or
JR mutable state.

Shadow state cannot alter the SR-v3 trajectory. A failed closure or invalid
transport invalidates only the branch-local outer-information state; it must
not undo an accepted operator, prune structural state, or otherwise implement
structural rollback. Procrustes registration is authoritative only at a
measured reanchor with measured cross-state tangent overlaps. It is not used
as a measurement-saving prediction law.

The shadow profile makes no quantum-measurement-savings claim. Conventional
SR-v3 measurements and accounting remain authoritative until a separately
validated active reuse profile explicitly omits measurements. Existing
pure-FM and source-locked FM profiles retain their current selector,
reoptimization, transport, and accounting contracts unchanged.

## FM-SNAKE With The SR Singleton Controller

The following is an FM route profile, not a new standalone SR route:

```text
route_family = formal_manifold_snake
route_profile = sr_phase2_phase3_whitened_adaptive_trust_no_n2_v1
adapt_reoptimization_route = formal_manifold_warm_start_v1
candidate_selector_family = singleton_response_snake
candidate_selector_profile = sr_singleton_controller_v1
```

The outer FM controller owns manifold state, tangent/frame transport,
qBroyden/qBANG state, recycled objective curvature, proposal transactions,
commit, and rollback. The inner SR controller owns only candidate exposure and
singleton selection. It must not invoke the formal combinatorial selector, the
JR selector, or share either route's mutable state.

The profile resolves the following numerical and structural fields without
reinterpreting existing SR evidence:

```text
historical_singleton_coordinate_solve_policy = supported_metric_whitened_eigh_v1
historical_singleton_coordinate_solve_scope = phase2_and_phase3_v1
historical_singleton_trust_region_update_policy = displacement_calibrated_unbounded_v2
sr_powell_coordinate_chart_policy = expanded_runtime_projected_logical_v1
phase2_novelty_multiplier_policy = inactive_ordinary_route_v1
phase3_novelty_multiplier_policy = inactive_ordinary_route_v1
phase0_pilot_enabled = false
phase2_enable_batching = false
phase3_enable_batching = false
route_a_funnel_active = false
phase3_runtime_split_mode = shortlist_pauli_children_v1
phase3_runtime_split_selection_mode = archival_child_set_forward_v1
phase3_runtime_split_subset_sizes = [1]
phase3_runtime_split_child_padding_policy = full_binary_code_space_v1
phase3_backend_cost_mode = proxy
phase3_runtime_split_child_set_symmetry_policy = hard_guard
phase3_runtime_split_child_padding_policy = exact_projected_grouped_v1
candidate_response_model = full_active_plus_singleton_v1
admission_cardinality = 1
prune_policy = recoverability_ladder_v1
structural_rollback_enabled = false
formal_metric_mode = sparse_anchor_qbroyd_recycled_v1
formal_transport_policy = predicted_transported_frame_gauge_v1
formal_post_anchor_exact_gram_budget = 4
formal_qbroyd_epsilon0 = 0.15
formal_qbang_momentum_active = false
formal_line_search_max_steps = 15
```

The FM metric recurrence is authoritative between exact anchors. The initial
or zero-growth anchor is exact; a reoptimization episode then has a hard total
budget of four additional exact Gram measurements, with terminal validation
consuming the last available slot. Armijo trials are energy-only and accepted
non-anchor endpoints are energy-plus-gradient. Predicted frame transport is
explicitly approximate (`Q = I`) until the next measured Procrustes correction.

Phase-II and Phase-III novelty quantities remain measured diagnostic telemetry.
They do not multiply the ordinary selector scores in this profile:

```text
S2 = DeltaE_TR_supported_metric_joint / (1 + K2)
S3 = DeltaE_TR / (1 + K3)
```

Accordingly, `measured_N2` is retained while the applied Phase-II multiplier
is one, and no additional Phase-III novelty multiplier is applied. The
standalone SR profile `supported_phase2_phase3_whitened_adaptive_trust_v2`
continues to preserve N2 and its existing reduced-logical Powell convention;
this FM-specific profile does not rewrite that identity.

## Conventional SR-SNAKE Profile

The unqualified current method name **SR-SNAKE** resolves to the corrected v3.1
controller contract. It retains the v3 full-response and accepted-refit policy
and additionally disables phase-live hysteresis:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3_1
display_name = SR-SNAKE
phase3_response_coordinate_scope = full_active_plus_singleton_v1
phase_live_hysteresis_enabled = false
sr_powell_coordinate_chart_policy = expanded_runtime_projected_logical_v1
adapt_accepted_refit_scope = full_ansatz_v1
adapt_accepted_refit_coordinate_chart = supported_fs_whitened_fixed_v1
adapt_accepted_refit_base_chart_policy = expanded_runtime_projected_logical_v1
```

Its profile-defining numerical controls are:

```text
historical_singleton_coordinate_solve_policy = supported_metric_whitened_eigh_v1
historical_singleton_coordinate_solve_scope = phase3_only_v1
historical_singleton_trust_region_update_policy = displacement_calibrated_unbounded_v2
phase3_response_coordinate_scope = full_active_plus_singleton_v1
sr_powell_coordinate_chart_policy = expanded_runtime_projected_logical_v1
adapt_accepted_refit_scope = full_ansatz_v1
adapt_accepted_refit_coordinate_chart = supported_fs_whitened_fixed_v1
adapt_accepted_refit_base_chart_policy = expanded_runtime_projected_logical_v1
```

The `historical_singleton_*` spellings are current executable compatibility
fields, not the method name. Selector supported-metric whitening, adaptive
trust, and accepted-refit supported-FS whitening are separate profile controls.
Turning any of them off creates an SR-SNAKE ablation profile; it does not create
another route family.

Under `full_active_plus_singleton_v1`, every active pre-admission logical
coordinate plus the candidate enters the Phase-III gradient, Hessian, and
Fubini--Study Gram construction on every controller round. Only genuine
Gram-null directions may be removed afterward by supported-rank projection.
The pre-support invariant is
`phase3_response_coordinate_count = active_logical_coordinate_count + 1`.
`adapt_reopt_policy`, local window sizes, periodic refit cadence, and terminal
refit triggers cannot change these response indices.

In the expanded-runtime/projected-logical chart, Powell optimizes one
coordinate per active executable Pauli factor while every objective and lift
projects the runtime coordinates belonging to one logical generator to their
block mean. The accepted-refit FS Gram chart is rebuilt once per accepted-refit
invocation, held fixed throughout that Powell solve, and applied to the complete
accepted ansatz. Runtime/checkpoint vectors remain expanded. Both the base
chart and accepted-refit chart are part of the route identity.

Agents must request this complete method profile through the executable
selector, not by assembling current defaults or supplying only the four local
coordinate/trust fields:

```text
--sr-route-profile sr_snake
```

`sr_snake_v3_1` is the exact current versioned alias. Both resolve to
`supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3_1` and
materialize the complete conventional execution contract in
`pipelines/static_adapt/sr_snake_route_profile.py`. The emitted v3.1 contract
SHA-256 is
`9b96179935ed80967a3335dfbbf8eece86a04c2d412e6b92aa8a466fa6913542`
and is serialized in command, manifest, checkpoint, resume, and result records.
An explicitly conflicting scientific option is an error. A registered resume
artifact that lacks this contract, carries another version, or carries another
digest is also an error. The `auto` token belongs only to the lower-level
Powell-chart option; it is not an SR route-profile request.

This conventional profile intentionally retains the v3 scientific controls
other than disabled phase-live hysteresis: Phase-II selector whitening
is off; Phase-II and Phase-III
ordinary novelty multipliers and the geometry-expansion fallback are active;
negative-curvature escape is off; recoverability pruning is active in live and
terminal modes; periodic full refits occur every eight rounds; and the final
full refit plus configured terminal prune remain active. Phase-II whitening,
novelty removal, saddle escape, cost-shaping changes, or terminal-policy changes
are distinct future perturbations.

Frozen conventional v3 remains available only through `sr_snake_v3` or its
full profile id. Its contract SHA-256 remains
`435910592e88f0136a0d45f611f79fe96b21d75fd25bad58276c871f39dc080e`.
That payload predates serialization of phase-live hysteresis and is retained
for exact artifact replay; FM outer-information shadow profiles remain pinned
to this explicit v3 contract and digest. Neither `sr_snake_v3` nor an FM shadow
request follows the advancing unqualified alias.

### Opt-in SR-SNAKE v4 candidate

`sr_snake_v4` is a registered candidate perturbation layered on conventional
v3. It does not replace conventional v3.1: the unqualified `sr_snake` request
continues to resolve to
`supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3_1`,
while explicit `sr_snake_v3` remains frozen replay identity.

The opt-in request is:

```text
request = --sr-route-profile sr_snake_v4
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_v4
phase3_response_coordinate_scope = full_active_plus_singleton_v1
```

V4 preserves the full active-logical-ansatz-plus-singleton Phase-III response
model and its pre-support count invariant from v3. Its candidate-specific
differences are:

```text
phase2_gram_novelty_policy = fallback_only_v1
phase3_gram_novelty_policy = fallback_only_v1
phase3_hardware_cost_normalization_mode = family_robust_symmetric_arctan_v1
adapt_beam_live_branches = 1
adapt_beam_children_per_parent = 1
phase1_prune_mode = live
phase1_prune_schur_nomination_route = full_logical_fs_trust_delete_refit_v1
phase1_prune_metric_schur_solve_mode = affine_deletion_global_trust_v1
phase1_prune_endpoint_overlap_policy = off
phase3_shadow_damping_policy = mapped_seed_zero_query_v1
adapt_finite_angle_fallback = false
adapt_disable_hh_seed = true
adapt_final_full_refit = false
sr_escape_mode = disabled
```

Ordinary Phase-II and Phase-III novelty multipliers are inactive, while the
all-energy-models-infeasible collective-span novelty fallback remains active.
That collective-span fallback is distinct from the finite-angle energy-probe
guard: v4 requires `--adapt-no-finite-angle-fallback`, and an enabled
finite-angle fallback is route-contract drift.
The legacy HH quadrature preseed is also disabled. Controller round one starts
from the reference state; HVA generators remain exposed through `full_meta`
and may be selected by the ordinary singleton controller.
Phase I is strictly first order in energy: `F` appears only in the
Fubini--Study trust bound.  Every Phase-II-scored candidate must carry a
finite, identity-bound directional-curvature receipt from the existing
directional Hessian construction.  Missing or nonfinite curvature aborts the
entire run before any novelty fallback.  Neither the historical
`lambda_F F` substitution nor the `g^2/(2 lambda_F F)` cheap ratio is active
or CLI-selectable under v4; both remain available only under explicit v1-v3
historical identities.
The symmetric arctangent hardware-cost policy is population-relative and is
applied consistently to the Phase-I, Phase-II, Phase-III, and fallback scores.
The effective branch shape is one live branch with one child.

Pruning is live-only. Its nomination model includes every active logical
coordinate in a complete affine deletion-response model constrained by the
Fubini--Study trust radius. Measured delete-and-refit energy remains the
acceptance authority. `phase1_prune_endpoint_overlap_policy=off` means zero
new endpoint-overlap measurements; it does not assert that the physical
pre-delete and post-refit states have zero overlap.

Phase-III metric damping is shadow telemetry only. The mapped-seed diagnostic
adds no quantum measurements and records `applied_mu=0`; it cannot change the
executed Phase-III response model. V4 performs neither a terminal-only full
refit nor terminal pruning, and it does not enable the negative-curvature,
saddle, or modeled-minimum escape routes.

The current emitted v4 candidate contract SHA-256 is
`d705671019543f676ee23ea9e1de8a7658183e3926939f2c389b8f015db6fe2f`.
Commands, manifests, checkpoints, resumes, and results using this candidate
must preserve the concrete profile and digest. Scientific validation of v4
does not silently advance the unqualified alias; changing that resolution is a
separate explicit decision.

### Opt-in no-prune symmetric-cost candidate

The prune-disabled control is a separate registered profile, not a permissive
override of `sr_snake_v4`:

```text
request = --sr-route-profile sr_snake_no_prune_symmetric_cost_v1
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1
phase3_response_coordinate_scope = full_active_plus_singleton_v1
```

It retains v3's expanded-runtime/projected-logical base chart, Phase-III-only
supported whitening, adaptive trust, and full accepted-ansatz supported-FS
Powell refit. It adds the first-order Phase-I and measured-required Phase-II
policies, removes both lambda-F proxies, uses symmetric arctangent hardware-cost
shaping, and sets both ordinary Gram-novelty policies to `fallback_only_v1`.
That policy skips the ordinary novelty projection/solve entirely: measured
novelty and its multiplier remain null and novelty solve/query charges remain
zero. It is not the archival compute-then-neutralize ablation. The Gram/Hessian
blocks still execute for their independent response, whitening, support, and FS
trust uses. The bounded all-energy-models-infeasible novelty fallback remains
available as a separately telemetered lazy path. Beam capacity is effective
1x1. HH preseeding, finite-angle probing, pruning, shadow damping, periodic
refits, terminal refits, Phase 0, and batching are off.

`Undamped` for this profile means no executable `H + mu G` metric-damping
policy. The historical `1e-6` numerical Hessian ridge and `1e-9` supported-solve
tolerances remain numerical factorization guards and are recorded separately.

The controller horizon is deliberately not part of this method profile. Every
run manifest must supply an explicit positive `--adapt-max-depth` and lock its
approved regime horizon; omission and nonpositive values fail closed. Checkpoints and
results must serialize
`all_energy_models_infeasible_novelty_fallback_telemetry` with an explicit
`enabled`, `fired`, activation count, ordered controller rounds, and query
charge. Absence of a fallback record is not evidence that it stayed unused.

The emitted contract SHA-256 is
`023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91`.
This opt-in profile does not change the unqualified `sr_snake` alias.

### Appendix-only undamped FS-trust pruning ablation

The pruning control is a strict one-factor child of the prune-disabled
symmetric-cost profile:

```text
request = --sr-route-profile sr_snake_symmetric_cost_fs_prune_nodamping_v1
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_nodamping_v1
parent_route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1
phase1_prune_enabled = true
phase_live_hysteresis_enabled = false
phase1_prune_mode = live
phase1_prune_schur_nomination_route = full_logical_fs_trust_delete_refit_v1
phase1_prune_metric_schur_solve_mode = affine_deletion_global_trust_v1
phase1_prune_recovery_trust_radius = 0.125
phase1_prune_trust_update_policy = modeled_local_fs_conservative_v1
phase1_prune_metric_schur_mu = 0
phase1_prune_metric_mu_update_policy = off
phase1_prune_endpoint_overlap_policy = off
```

Every active logical coordinate enters the affine deletion-response model
before supported-rank reduction.  The explicit Fubini--Study radius limits the
complete modeled deletion displacement.  Measured delete-and-refit energy is
the acceptance authority.  Rejected measured trials contract the branch-local
radius by one half down to the source-locked `1e-8` floor; the radius never
expands.  Endpoint-overlap calibration is off, so this radius update introduces
no additional quantum measurement.

Scientific metric damping is absent: both the initial coefficient and its
update policy are zero/off, and Phase-III shadow damping remains off.  Pruning
is live-only; terminal pruning, terminal refitting, structural admission
rollback, beam branching, and batching remain off.  All non-prune settings,
including the novelty fallback telemetry and symmetric cost shaping, are
inherited exactly from the main profile.  Its contract SHA-256 is
`81b072c03f9866817a4fc6173017788223ab8b5ba007d6015315e39d3fb4c30e`.

### Appendix-only historical 3x2 beam ablation

The beam control is a distinct one-factor appendix profile. It does not alter
the main prune-disabled symmetric-cost identity above:

```text
request = --sr-route-profile sr_snake_no_prune_symmetric_cost_beam_v1
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_beam_v1
parent_route_profile = supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1
adapt_beam_live_branches = 3
phase_live_hysteresis_enabled = false
adapt_beam_children_per_parent = 2
adapt_beam_terminated_keep = 3
adapt_beam_terminal_archive_mode = legacy
adapt_beam_lambda = 0.005
```

This gives at most six admission children per controller round and preserves
the exact pre-2026-07-04 stopped-branch semantics. Every expanded live parent
also materializes a terminated stop child even when admission proposals exist;
the best three terminated children are retained cumulatively across later
rounds. Thus the structural mode is `stop_or_single_admission`, not the newer
retention-only interpretation of a terminal archive. Every non-beam execution
setting is inherited exactly from the main effective-1x1 profile. Pruning and
batching remain off, ordinary novelty multipliers remain off, and the
controller horizon remains an explicit per-regime source-lock field.

This appendix identity does not change `sr_snake`, `sr_snake_v3`, or
`sr_snake_no_prune_symmetric_cost_v1`. Its contract SHA-256 is
`49fb8c2f069722ce87cbaaedc8d7d32726a11dad92a624e3326269d75dcd1168`.

### SR-v3 no-novelty metric-prune beam profile

The H2O no-novelty beam campaign uses a registered SR-SNAKE profile rather
than a Route-A command with an SR display label:

```text
request = --sr-route-profile sr_snake_no_novelty_metric_prune_beam_v1
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_response_no_novelty_metric_prune_beam_v1
parent_profile = supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3
phase3_response_coordinate_scope = full_active_plus_singleton_v1
phase3_novelty_ablation_mode = all
phase2_enable_batching = false
phase3_enable_batching = false
phase3_runtime_split_subset_sizes = [1]
adapt_beam_live_branches = 3
adapt_beam_children_per_parent = 2
phase1_prune_enabled = true
phase1_prune_policy = recoverability_ladder_v1
phase1_prune_mode = both
phase1_prune_schur_nomination_route = metric_regularized_v1
phase1_prune_metric_schur_mu = 0.01
problem = molecular_vibronic_h2o_linear_fd
adapt_max_depth = 50
```

The novelty ablation removes both Phase-II and Phase-III novelty from
selection, including the all-energy-models-infeasible novelty fallback, while
retaining novelty telemetry. The profile otherwise preserves the conventional
v3 full-response singleton controller, expanded runtime/projected-logical
Powell coordinates, full-ansatz supported-FS accepted refits, and 3-by-2 beam.
Unlike Paper-I binary phonon registers with unused codewords, the H2O
`n_ph_max=(1,1,1)` registers use their complete one-qubit code spaces, so this
profile records that no padding projection is mathematically required.
The Paper-I `marrakesh_graph_span_v1` cost model is HH-specific, so this
application uses the generic deterministic proxy cost mode while preserving
the same cost-weighted SR scoring structure.
Its contract SHA-256 is
`4b495a6c23263ddaa5b77e121570c34a041d1faf708e635f7957ab81f83b09c8`.

### Historical window-coupled SR-SNAKE v2

The three completed 2026-07-15 weak-Holstein anchors belong to the preserved
v2 profile:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_full_accepted_refit_v2
request = --sr-route-profile sr_snake_v2
phase3_response_coordinate_scope = legacy_reopt_coupled_v1
```

Its contract SHA-256 remains
`32d2bdf2b05818be6f4add74137447a313605d7ed35ffb880651863b793a0f64`.
This historical policy derives the Phase-III response window from the Powell
reoptimization window and periodic-full-refit state. Existing v2 results and
their route digest remain unchanged. The unqualified `sr_snake` token must not
resolve to v2 after v3 registration.

### Historical SR-SNAKE v1

The explicit historical identity remains unchanged:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_v1
request = --sr-route-profile sr_snake_v1
sr_powell_coordinate_chart_policy = expanded_runtime_projected_logical_v1
adapt_accepted_refit_scope = selector_policy_v1
adapt_accepted_refit_coordinate_chart = native_v1
```

Its preserved contract SHA-256 remains
`fab7b5a6c4bd2ab019139367aa2a507356a5c969b6b88cd72d32365ae766e13e`.
Do not rewrite v1 evidence as conventional v2. The three 2026-07-15 anchors
were produced from source-locked component flags before v2 was registered;
their normalized executable settings are the evidence authority for v2, not a
claim that their old serialized profile field was already v2.

The newer one-coordinate-per-logical-generator chart remains selectable only
under a visibly distinct profile:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_reduced_powell_v2
sr_powell_coordinate_chart_policy = logical_shared_reduced_v1
```

It must never execute under either registered expanded-chart profile. An
ordinary current invocation of conventional SR-SNAKE v2 may resolve `auto` to
the expanded chart at route-profile resolution. Historical/source-locked
replay must carry a recognized concrete chart in preserved evidence and must
fail closed when that field is missing, unknown, conflicting, or mismatched.

All registered expanded-chart profiles apply the selector supported-metric
solve in Phase III only. The opt-in Phase-II whitening perturbation is a
distinct profile of the same family:

```text
route_profile = supported_phase2_phase3_whitened_adaptive_trust_v2
historical_singleton_coordinate_solve_policy = supported_metric_whitened_eigh_v1
historical_singleton_coordinate_solve_scope = phase2_and_phase3_v1
historical_singleton_trust_region_update_policy = displacement_calibrated_unbounded_v2
sr_powell_coordinate_chart_policy = logical_shared_reduced_v1
```

In that profile Phase II replaces only its scalar predicted benefit with the
supported-metric full-active-plus-singleton gain.  The saved `N2`, `1 + K2`,
physical lanes, candidate order/membership, singleton cardinality, and
batching-off policy remain authoritative.  Existing v1 artifacts must never be
reinterpreted as Phase-II-whitened evidence.

### SR-SNAKE escape profiles

Negative-curvature and modeled-local-minimum escape are versioned profiles of
the same `singleton_response_snake` family. They are not new SNAKE families and
must not be inferred from an indefinite Hessian alone.

| Escape mode | Route profile | Contract |
|---|---|---|
| `disabled` | `supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3` | Conventional expanded-chart SR-SNAKE with a full active-plus-singleton Phase-III response every round and full-ansatz supported-FS accepted refits. |
| `disabled` | `supported_whitened_adaptive_trust_full_accepted_refit_v2` | Historical expanded-chart replay profile with a reoptimization-window-coupled Phase-III response and full-ansatz supported-FS accepted refits. |
| `disabled` | `supported_whitened_adaptive_trust_v1` | Preserved supported-whitening and adaptive-trust SR-SNAKE behavior with `expanded_runtime_projected_logical_v1`. |
| `disabled` | `supported_whitened_adaptive_trust_reduced_powell_v2` | Optional reduced-Powell variant with `logical_shared_reduced_v1`; distinct from canonical v1. |
| `saddle_only` | `supported_whitened_adaptive_trust_saddle_escape_v2` | Uses `logical_shared_reduced_v1`. Ordinary SR selection keeps precedence. When the ordinary route is unusable, a separate zero-gradient-capable finite escape population is audited exhaustively at the unchanged state; a certified supported global trust solve may then service candidate-attributable negative curvature, including the complete hard case and exact evaluation of both mapped signs. |
| `saddle_plus_modeled_minimum` | `supported_whitened_adaptive_trust_saddle_modeled_minimum_escape_v2` | Uses `logical_shared_reduced_v1`. Adds modeled-local-minimum eligibility only after the complete reachable Phase-III population is quotient-redundant or numerically valid, stationary, and PSD. Until incumbent/working-state exploration is implemented and checkpointed, this mode must fail closed after reporting eligibility. |

Modeled-minimum eligibility also requires an independent supported-stationarity
certificate for the working physical state. Per-record quotient redundancy does
not imply state stationarity and cannot substitute for this certificate, even
when every reachable record is redundant. The token is bound to the working
state fingerprint, ordered reachable-population digest, live trust radius,
comparison epoch, and the support and transported-trust provenance; any missing
or stale binding fails closed.

The pure Stage-B mathematical kernel may be developed and tested while the
runtime profile remains execution-disabled. Combined execution stays gated
until a canonical continuation-path provider binds the complete action tuple
and certifies stabilized-trust arclength, live providers certify a uniform
incumbent-referenced barrier, positive nonlinear active-manifold separation,
and connected exclusion-component preservation, disposable Powell comparisons
carry a justified reproducibility allowance, the countable action cursor has a
certified unseen-tail service bound, and the complete incumbent/working state
and fair-service clocks round-trip through checkpoints.

The saddle acquisition is the certified marginal full-versus-active trust
gain divided by `1 + K3`. It must not receive an additional `N3` multiplier.
The inherited `N3/(1+K3)` all-infeasible action remains a numerical geometry-
expansion compatibility path and cannot certify a saddle or modeled local
minimum.

For active escape profiles, the ordinary capped Phase-I/II/III funnel remains
literal and authoritative for ordinary selection. A separate escape funnel
starts from every validity-gated Phase-I record in the same position/lane
population, applies no gradient/gain/curvature gate, and eagerly evaluates the
finite Phase-III-reachable population. This eager finite audit is the current
implementation of fair eventual service; it must not be replaced by relabeling
the ordinary capped shortlist as complete.

For the archival exact-cardinality-one child profile, the fixed Phase-I-to-II
map treats the parent-position row as antecedent provenance. Ordinary selection
continues to forward its single best valid child. The escape population instead
contains every fully scored child that passes symmetry, padding projection,
canonicalization, and deduplication; the parent is not a fallback successor and
a zero ordinary-score child is not removed from the escape audit.

The v2 global trust solver selects raw Gram support before stabilization. It
first proposes the smallest ridge satisfying the declared KKT conditioning
allocation (`global_trust_kkt_residual_accuracy = 1e-8`) and audits that
proposal against the declared spectral metric-distortion budget
(`global_trust_metric_distortion_budget = 5e-2`). If the proposal breaches the
budget while raw support is resolved, the active escape v2 profiles may try the
unridged support as an a-posteriori certificate path. This path is accepted only
after raw metric-null gradient/Hessian compatibility and the unchanged model,
KKT, global-trust, support-transport, active-image, quotient, marginal-gain, and
accepted-path transaction certificates pass; any failed or unresolved gate
remains typed unresolved and no inertia label is issued before the solver gates
pass. Active-only, active-plus-singleton, quotient participation, and
accepted-path trust accounting all reuse the selected support, exact whitening
denominators, transported metric, and provenance identifier; none may refactor
an active block or reapply a ridge.

For v2 accepted paths, adaptive radius accounting uses stabilized-trust
arclength. Exact endpoint Fubini--Study distance and raw-Gram local displacement
are separate physical/diagnostic quantities and never substitute for that
arclength. Before an active-stationarity seed is admitted, however, the exact
nonlinear endpoint is a separate safety gate: evaluate the complete retained
candidate set at largest-first dyadic fractions, recompute the full-joint
quadratic prediction at each fraction, and accept the first fraction having
both finite material exact descent and exact endpoint Fubini--Study distance no
larger than the live branch radius. If every representable tested fraction is
finite but fails one of those endpoint gates, halve only the branch-local
radius and rebuild the supported model at the next controller service; do not
consume a singleton, ansatz depth, or admission history and do not fire the
ordinary no-progress terminal. Mapping, nonfinite-objective, or invalid-state
failures remain fail-closed and hold the radius. This pre-acceptance endpoint
gate does not change the stabilized-trust arclength authority used after an
accepted refit. When both
atomically mapped saddle signs have finite exact energies and are certified
non-downhill, the event is model disagreement rather than a mapping failure:
use a valid Taylor contraction receipt when a mapped third-derivative bound is
available, otherwise apply the typed half-radius numerical backtrack with no
Taylor guarantee. The backtrack mutates only branch-local trust state.

Every exact seed guard records numerical energy width, optimizer/Powell
reproducibility allowance, and their aggregate simultaneous comparison width
as separate fields. Before the disposable optimizer probe the optimizer
allowance is explicitly zero and not applicable. For an ordinary SR record, a
finite exact mapped comparison whose gain upper bound is nonpositive writes a
branch-local `ModelLive` contradiction keyed by physical-state fingerprint and
record ID. That record remains in the exhaustive escape population but cannot
receive another predicted-only ordinary certificate at the same physical
state. Mapping/nonfinite failures and comparison-width overlap do not retire
the model; a changed physical-state fingerprint restores a fresh status.

A supported saddle classification is local-model evidence only. The route may
set `physical_transition_certified=true` only after the mapped seed passes its
exact guard, seed-preserving Powell chooses the final point, and an independent
replay verifies finite coordinates, the declared insertion order and semantic
operator sequence, normalized finite states, and phase-aligned state agreement
within the recorded state-consistency tolerance. Certification failure occurs
before committing the child parameter state.

The current SR-SNAKE contract also preserves:

- the literal ordinary Phase-I-to-III shortlist and cost-weighted scoring
  semantics, separate from the exhaustive escape audit;
- Phase-I parent-position records and physical operator-type lanes;
- exact-cardinality-one Pauli-child expansion before Phase II, with hard
  fixed-sector enforcement, binary-padding projection, deterministic
  projective canonicalization, and deduplication inside each parent-position
  child family;
- Phase-II geometry with the parent physical lane inherited by its forwarded
  singleton child, followed by a lane-free Phase-III child population;
- full active-plus-singleton Phase-III response and exactly one admitted
  candidate-position record;
- beam branch management and `recoverability_ladder_v1` pruning;
- ordered signed checkpoints, full-coordinate refits, and the locked stopping
  horizon/prefix-selection rules.

Exact caps, weights, optimizer budgets, pool composition, seeds, and stopping
thresholds belong in the source-locked profile/manifest rather than in the
family name.

## Phase-III supported-projection generalized-trust ablation

The explicit request
`sr_snake_no_prune_symmetric_cost_projected_phase3_v1` resolves to
`supported_projected_generalized_adaptive_trust_full_response_symmetric_cost_no_prune_v1`.
It is a one-setting child of the validated hysteresis-disabled Main SR contract
`023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91`.
The child contract SHA-256 is
`3ff2abb1455cda3cf8cc2de0cf739172f8cdcfe6b1c9436e1afdd40076cd3ce8`.

The only changed execution field is:

```text
historical_singleton_coordinate_solve_policy:
  supported_metric_whitened_eigh_v1
  -> supported_metric_projected_generalized_trust_v1
```

For this child, Phase III first builds the complete active-logical-plus-
singleton raw response model, diagonalizes the raw Fubini--Study Gram matrix,
and removes only modes below the registered support threshold. In the retained
eigenspace it solves `(H_s + lambda Lambda_s) q = g_s` under
`q^T Lambda_s q <= rho^2`. It does not construct a Gram inverse square root,
does not apply the historical metric ridge, and must record Phase-III
whitening as false. This is support projection plus a generalized metric trust
solve, not supported whitening.

The accepted post-admission Powell refit is intentionally unchanged:
`adapt_accepted_refit_scope=full_ansatz_v1` and
`adapt_accepted_refit_coordinate_chart=supported_fs_whitened_fixed_v1` over
the expanded-runtime/projected-logical base chart. Phase II, symmetric cost,
ordinary-novelty-off policy, singleton admission, pruning off, beam 1x1,
adaptive trust, and phase-live hysteresis disabled all remain identical to the
parent. This ablation is opt-in and does not redefine unqualified `sr_snake`.

## Compatibility Rules

- `static_route_id=route_a`, `paper_i_production_v1`, and Route-A version
  strings are legacy umbrella/provenance fields. They do not uniquely resolve
  JR-SNAKE, FM-SNAKE, or SR-SNAKE.
- Do not rewrite preserved artifacts merely to add the new family names.
  Resolve their family from the complete recorded route settings and report
  the legacy fields alongside the inferred/resolved family.
- Do not call SR-SNAKE `Route 4`, `historical SNAKE`, `old singleton`, or
  `no-batch SNAKE` in new agent-facing material.
- Do not name a family from whitening, trust-region, optimizer, batching cap,
  pool, or pruning alone. Those are profile or run settings.
- Do not identify an SR-SNAKE profile without its explicit Phase-III response
  scope, Powell chart, and accepted-refit chart. Conventional v3, historical
  conventional v2, and historical v1
  use `expanded_runtime_projected_logical_v1`; v2 and v3 both use
  `full_ansatz_v1` plus `supported_fs_whitened_fixed_v1`, while v3 changes the
  response scope to `full_active_plus_singleton_v1`. The reduced chart always
  means the distinct reduced-Powell profile.
- Do not infer SR-SNAKE from `batch_size=1`. Require the singleton admission
  path, Phase 0 off, batching off, and the active-plus-singleton response
  contract.
- Do not infer FM-SNAKE merely from the presence of a metric or Hessian.
  Require the formal-manifold route and its branch-local checkpoint/state.
- Do not infer JR-SNAKE merely from joint linear-solve telemetry. Require the
  JR macro-to-child funnel and joint batch-selection semantics.

## Agent Reporting Template

Use this compact form when identifying a result:

```text
Method family: SR-SNAKE (`singleton_response_snake`)
Profile: `supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3`
Compatibility fields: `static_route_id=route_a`, ...
Source lock: <manifest/current JSON and hash>
Resolved structural evidence: Phase 0 off; Phase-II/III batching off;
`phase3_response_coordinate_scope=full_active_plus_singleton_v1`;
full active-plus-singleton response; singleton admission.
```

If the structural fields are missing or contradictory, report the family as
unresolved rather than guessing from a display label or compatibility field.
