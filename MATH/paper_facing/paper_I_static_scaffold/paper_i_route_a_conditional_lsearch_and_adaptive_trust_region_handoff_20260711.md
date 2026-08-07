# Paper-I Route-A Conditional Search Expansion and Adaptive Trust-Region Handoff

Created: 2026-07-11

Status: implementation contract for the receiving repo agent. This is not a
manuscript edit and does not promote either policy into `Paper_I.tex`.

## Objective

Implement two independent, typed Route-A improvements:

1. Conditionally widen the combinatorial child search pool when the requested
   `L_search` prefix contains no jointly feasible singleton or batch.
2. Calibrate the next-round Fubini-Study trust-region radius from the previous
   admitted step's predicted displacement and post-refit realized displacement.

The policies address different failure modes and must remain independently
configurable:

- Search expansion asks whether a feasible direction exists outside a narrow
  ranked child prefix.
- Trust-region calibration asks whether the local joint model's state-space
  radius was too large or too small for the realized nonlinear refit.

Increasing the trust radius cannot repair a rank failure. The joint rank gate
runs before the trust-region solve, so a rank-rejected subset has no step to
expand.

## Controlling Decisions

The following decisions control this implementation.

1. `L_search` widening is conditional, not unconditional.
2. Widen only when the current search prefix contains zero jointly feasible
   subsets after exact-child compatibility, joint rank, and conditioning gates.
3. Do not widen merely because all feasible subsets have nonpositive predicted
   gain or nonpositive final score.
4. The widening sequence for the current `L_search=15` route is `15 -> 20 ->
   all child Phase-2 survivors`.
5. Resolve `0` as the existing explicit `all` sentinel.
6. Clip every requested width to the actual child Phase-2 survivor count and
   remove repeated effective widths.
7. Enumerate only newly exposed subsets after each expansion.
8. A singleton is already a subset with cardinality one. Do not add a separate
   singleton fallback formula or bypass the joint evaluator.
9. A geometric exhaustion stop is valid only after the full child Phase-2
   population has been searched and contains no jointly feasible subset.
10. Keep the current joint score family:

    ```text
    S(B) = DeltaE_hat_joint(B) / (1 + K(B))
    ```

    followed by the configured soft-additivity penalty.
11. The trust-region radius is a state-displacement scale, not a score
    threshold and not a rank threshold.
12. `max_fubini_study_step=0.25` remains the initial radius unless explicitly
    configured otherwise.
13. There is no scientific hard maximum on the radius. Values such as `0.5`
    and `10` must remain valid. A large radius can have no effect when the
    unconstrained optimum is already inside the trust ball.
14. Limit the multiplicative change per accepted round rather than clamping the
    radius to an arbitrary scientific maximum.
15. Reduce the next radius when the realized optimum moves less than the model
    predicted.
16. Increase the next radius only when the previous trust region was binding,
    the realized refit achieved legitimate energy descent, and the realized
    displacement exceeded the predicted displacement.
17. A rollback contracts the radius.
18. Measure realized behavior after full refit and before pruning. Post-prune
    coordinates would confound admission/refit calibration with a separate
    deletion mechanism.
19. Trust-region runtime state is branch-local in beam execution.
20. Preserve fixed/off compatibility modes and legacy payload readability.

## Non-Goals

This slice must not:

- edit `MATH/paper_details/Paper_I.tex` or regenerate `Paper_I.pdf`;
- change cost weights, shortlist rankings, child identity, symmetry, padding,
  Powell policy, pruning authority, beam survival, or admission semantics;
- make greedy batching use combinatorial widening;
- silently change `B_max` while widening `L_search`;
- treat classical subset enumeration as quantum-query work;
- rerun or stop scientific jobs as part of implementation;
- revert, overwrite, or normalize unrelated dirty work;
- introduce another fallback route when a feasible singleton is already part
  of the canonical subset space.

## Live Code State at Handoff

The active checkout is:

```text
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3
```

Do not implement against the iCloud/FileProvider checkout under `Documents`.

The receiving agent must inspect the live worktree before editing. At handoff,
these directly relevant files already contain uncommitted work:

```text
M  pipelines/scaffold/hh_continuation_scoring.py
M  pipelines/static_adapt/adapt_pipeline.py
M  pipelines/static_adapt/engine_support.py
?? pipelines/static_adapt/route_a_schur_selector.py
```

The untracked status of `route_a_schur_selector.py` does not make it disposable.
It is part of the current Route-A implementation and must be preserved.

Current implementation anchors:

| Concern | Current owner |
|---|---|
| Typed Route-A selector controls | `RouteASchurSelectorConfig` in `pipelines/static_adapt/route_a_schur_selector.py` |
| Mapping to shared score controls | `route_a_schur_score_config()` in the same file |
| Canonical dedup and selector entry | `select_route_a_schur_proposals()` in the same file |
| One-width combinatorial enumeration | `combinatorial_reduced_plane_batch_proposals()` in `pipelines/scaffold/hh_continuation_scoring.py` |
| Full joint Gram/Hessian workspace | `_build_batch_full_geometry_workspace()` in the scoring module |
| Joint rank and trust solve | `_BatchFullGeometryWorkspace.evaluate_subset()` in the scoring module |
| Beam-local mutable state | `_BeamBranchState` in `pipelines/static_adapt/engine_support.py` |
| Beam and non-beam stop propagation | `_run_hardcoded_adapt_vqe()` in `pipelines/static_adapt/adapt_pipeline.py` |
| Typed Paper-I run config and manifest | `PaperISnakeRunConfig` in `pipelines/static_adapt/paper_i_runner.py` |
| Direct Powell Pareto harness | `pipelines/exact_bench/paper_i_hh_powell_pareto.py` |

### Current one-width behavior

`combinatorial_reduced_plane_batch_proposals()` currently:

1. resolves one effective `batch_search_pool_size`;
2. takes one ranked prefix;
3. builds one geometry workspace;
4. enumerates all allowed subsets in that prefix;
5. returns proposals and one summary.

When no positive proposal survives, beam and non-beam orchestration can promote
that narrow-prefix result directly to `joint_geometry_selector_exhausted`.
That stop currently does not prove that the full child Phase-2 population is
geometrically exhausted.

### Why the observed failure is geometric rather than symmetry filtering

The corrected strong-weak B3/L15 diagnostic reached the joint selector with:

```text
15 singleton subsets
105 pair subsets
455 triple subsets
56 duplicate-exact-child incompatibilities
519 joint rank-gate rejections
0 prefilter rejections under joint_subset_gate_v1
```

This says the first 15 ranked child records failed to provide a jointly
independent local direction after projection against the active ansatz. It does
not establish that every lower-ranked child Phase-2 survivor is redundant, and
it is not a symmetry-rejection result.

## Mathematical Contract: Conditional Search Expansion

Let the globally ranked child Phase-2 population be

```text
R = (r_1, r_2, ..., r_N).
```

For effective width `L_j`, define the ranked prefix

```text
C_j = {r_1, ..., r_Lj}.
```

The candidate subset family is

```text
B_j = {B subseteq C_j : 1 <= |B| <= B_max}.
```

Apply the existing hard feasibility rules to each subset:

1. no repeated exact canonical child identity at alternative positions;
2. symmetry and child eligibility already satisfied upstream;
3. joint effective Gram rank passes;
4. joint effective Gram conditioning passes;
5. all existing finite-state and numerical-consistency checks pass.

Call the resulting family `F_j`. The expansion decision is:

```text
if |F_j| > 0:
    stop widening and select within F_j
elif L_j < N and expansion policy is zero_feasible_v1:
    continue to L_(j+1)
else:
    report full-population geometric exhaustion
```

The test is joint feasibility, not positive score. A feasible subset with
nonpositive predicted gain proves that the geometry solve exists; it must not
trigger widening under `zero_feasible_v1`.

For every feasible subset, retain the canonical evaluator:

```text
S_base(B) = DeltaE_hat_joint(B) / (1 + K(B))
S_soft(B) = S_base(B) / (1 + lambda_add * d_add(B))
```

Do not rank individual records and then assemble the top `B_max`. Every subset
must receive its joint ansatz-plus-batch solve before ranking.

### Effective width schedule

Given initial requested width `L_0`, configured expansion widths
`(e_1, ..., e_m)`, and survivor count `N`:

1. Resolve `0` to `N`.
2. Clip each positive value to `N`.
3. Keep only widths strictly greater than the prior effective width.
4. Preserve the configured order.
5. If the policy is `zero_feasible_v1`, append `N` if no configured stage
   reaches all survivors.
6. If `L_0=0`, the only effective width is `N`.

Examples:

```text
N=100, L_0=15, expansion=(20, 0) -> (15, 20, 100)
N=18,  L_0=15, expansion=(20, 0) -> (15, 18)
N=12,  L_0=15, expansion=(20, 0) -> (12)
N=100, L_0=20, expansion=(20, 0) -> (20, 100)
N=100, L_0=0,  expansion=(20, 0) -> (100)
```

Reject negative sizes. Repeated or decreasing configured sizes should be
normalized out with explicit telemetry, not re-evaluated.

### Incremental subset enumeration

After expanding from `L_a` to `L_b`, evaluate only subsets containing at least
one newly exposed record:

```text
Delta B_j = {
    B subseteq C_j :
    1 <= |B| <= B_max and
    B intersects (C_j minus C_(j-1))
}.
```

For each cardinality `s`, the raw newly exposed subset count is

```text
choose(L_b, s) - choose(L_a, s).
```

In particular, newly exposed pair count is

```text
choose(L_b, 2) - choose(L_a, 2).
```

Triples require additional classical subset solves but no independent
three-body matrix elements once all required one-coordinate and pair blocks
are available.

Previously feasible proposals may remain cached for comparison, but widening
occurs only when the prior feasible count is zero. Therefore the first stage
with any feasible subset is also the terminal search stage.

## Mathematical Contract: Displacement-Calibrated Trust Radius

For the previous selector round, let the joint model return the applied step

```text
z_pred = (delta_theta_A_pred, alpha_B_pred)
```

and its pre-admission Fubini-Study metric `G`. The model-predicted local
displacement is

```text
d_pred = sqrt(max(0, z_pred^T G z_pred)).
```

The current trust constraint is

```text
z^T G z <= rho_k^2.
```

### Realized displacement

The primary realized scalar for canonical `full_ansatz_v1` should be the exact
Fubini-Study distance between the pre-admission state and the post-refit,
pre-prune state:

```text
d_real_exact = arccos(clip(abs(<psi_before | psi_after_refit>), 0, 1)).
```

This quantity is global-phase invariant and avoids inventing an angle period.
It directly measures the physical state displacement that `rho` is intended to
bound.

Also reconstruct the realized local coordinate displacement when the mapping
is unambiguous:

```text
z_real = (delta_theta_A_real, alpha_B_real)
d_real_local = sqrt(max(0, z_real^T G z_real)).
```

This is diagnostic and supports a direction-agreement guard. It must use the
same pre-admission active context and the same pre-admission `G` as the
predicted step. Do not compare a full-ansatz realized vector against a
window-limited predicted metric.

For canonical `full_ansatz_v1`, every pre-admission ansatz coordinate belongs
to `A`. For `active_window_v1`, either:

- project the realized coordinate change onto the exact configured active
  window and use that projected local displacement; or
- leave adaptive trust updates disabled for that mode until the projection is
  implemented and tested.

Do not silently use the full-state exact displacement to calibrate a
window-limited model. Keep `batch_only_diagnostic_v1` fixed by default.

### Parameter mapping rule

When constructing `z_real`:

1. retain the pre-admission `AnsatzParameterLayout`, theta, and state;
2. retain selected record order and effective insertion positions;
3. map old logical/runtime coordinates through the post-insertion layout;
4. measure inserted batch coordinates relative to the absent-operator origin,
   not merely relative to the optimizer's nonzero seed;
5. use the selected proposal's active-context order and batch-record order;
6. use a shared, generator-aware period contract for angle differences.

Do not copy the pruning-local hardcoded `2*pi` wrap into this route without
proving that every selected generator coordinate has that period. The exact
state-distance update remains available even when local coordinate direction
telemetry cannot be resolved safely.

### Radius update

Define the displacement ratio

```text
r_d = d_real / (d_pred + epsilon_d).
```

Use square-root damping and bound the per-round factor:

```text
f_raw = sqrt(max(0, r_d))
f = clip(f_raw, f_contract_min, f_expand_max)
```

Recommended initial operational values:

```text
f_contract_min = 0.5
f_expand_max = sqrt(2)
rho_min = 0.0625
rho_max = None
```

`rho_min` is a tunable numerical floor, not a manuscript constant. There is no
scientific upper cap. Validate only that a configured radius is finite and
positive.

Update rules:

```text
fixed:
    rho_(k+1) = rho_k

rollback or non-finite realized step:
    rho_(k+1) = max(rho_min, f_contract_min * rho_k)

r_d < 1:
    rho_(k+1) = max(rho_min, max(f_contract_min, sqrt(r_d)) * rho_k)

r_d > 1 and trust_clipped and realized energy descent is positive:
    rho_(k+1) = min(f_expand_max, sqrt(r_d)) * rho_k

otherwise:
    rho_(k+1) = rho_k
```

Expansion requires that the prior trust region was active. If the unconstrained
step already lay inside the radius, a larger `rho` could not have changed the
selected step and should not be credited as the remedy.

Use realized energy only as a safety veto:

```text
DeltaE_real = E_before_admission - E_after_full_refit_pre_prune.
```

Require `DeltaE_real` to exceed the existing numerical improvement tolerance
before expansion. Do not replace the displacement-ratio update with an energy
ratio controller.

### Direction telemetry and expansion guard

When `z_real` is safely available, report the metric-direction cosine

```text
c_G = (z_pred^T G z_real) / (d_pred * d_real_local + epsilon_d).
```

Recommended first implementation:

- always report `c_G` when available;
- expose an optional requirement that `c_G >= 0.5` before expansion;
- leave that requirement off by default until generator-period and
  insertion-coordinate mapping tests pass;
- do not block contraction when `c_G` is unavailable or low;
- expose the threshold as typed configuration;
- if the requirement is enabled and coordinate mapping is unavailable, hold
  rather than expand;
- if the requirement is disabled, use the exact-distance ratio plus energy
  veto and record that no direction guard was applied.

The numerical value `0.5` is a candidate operational guard, not a Paper-I
claim. Keep it visible in manifests and tests.

## Typed Public Configuration

Do not add closure variables to `adapt_pipeline.py`. Add immutable typed policy
objects near `RouteASchurSelectorConfig`.

Recommended interfaces:

```python
@dataclass(frozen=True)
class BatchSearchExpansionConfig:
    policy: str = "zero_feasible_v1"
    expansion_sizes: tuple[int, ...] = (20, 0)


@dataclass(frozen=True)
class TrustRegionUpdateConfig:
    policy: str = "displacement_calibrated_v1"
    radius_min: float = 0.0625
    contraction_factor_min: float = 0.5
    expansion_factor_max: float = math.sqrt(2.0)
    displacement_epsilon: float = 1e-12
    direction_cosine_min: float = 0.5
    require_direction_for_expansion: bool = False
```

Policy constants:

```text
batch search expansion:
  off
  zero_feasible_v1

trust-region update:
  fixed
  displacement_calibrated_v1
```

Extend `RouteASchurSelectorConfig` with:

```python
batch_search_expansion: BatchSearchExpansionConfig
trust_region_update: TrustRegionUpdateConfig
```

Preserve the existing fields:

```text
batch_search_pool_size
max_fubini_study_step
```

Their meanings become:

- `batch_search_pool_size`: initial requested `L_search`;
- `max_fubini_study_step`: initial radius and fixed-policy radius.

For compatibility, serialized config should retain those existing keys and add
the nested policy payloads. Do not rename or remove historical payload fields
in this slice.

The shared `FullScoreConfig` needs only the values required by the scorer:

```text
batch_search_expansion_policy
batch_search_expansion_sizes
rho = effective branch/current radius for this selector call
```

Trust update rules and runtime history do not belong in `FullScoreConfig`.

### Forward defaults and compatibility

Implement both policies with compatibility modes. Proposed forward Route-A
defaults are:

```text
batch_search_expansion.policy = zero_feasible_v1
batch_search_expansion.expansion_sizes = (20, 0)
trust_region_update.policy = displacement_calibrated_v1
```

However, before changing a promoted canonical-settings lock, inspect current
tests and machine-readable settings. If an existing locked default conflicts,
implement the typed policies and report the conflict rather than silently
rewriting historical settings. `off` and `fixed` must exactly reproduce prior
behavior.

## Runtime State

Add a small mutable runtime object in a focused module such as
`pipelines/static_adapt/route_a_trust_region.py`:

```python
@dataclass
class RouteATrustRegionState:
    radius: float
    update_count: int = 0
    last_update: dict[str, Any] | None = None
```

The module should own pure helpers for:

- initialization from typed config;
- serialization/deserialization;
- exact Fubini-Study state distance;
- optional local coordinate/direction diagnostics;
- deterministic next-radius calculation.

The selector itself remains stateless. Before each call, project the current
runtime radius into an effective selector config or score config:

```text
effective rho = trust_region_state.radius
```

### Beam state

Add the trust state to `_BeamBranchState`. `clone_for_child()` must deep-copy
it. Sibling branches may see different realized refits and therefore must not
share a mutable radius object.

Beam parent evaluation uses the parent's current radius. Each admitted child
updates only its own radius after its full refit and before prune. A parent that
produces multiple children does not retroactively change the radius used to
score its siblings in the same expansion round.

### Non-beam state

Initialize one trust state before the ordinary adaptive loop. Use its radius in
the current round's selector and update it after the accepted batch's full
refit, before pruning.

## Detailed Search-Expansion Algorithm

Refactor the current one-width body of
`combinatorial_reduced_plane_batch_proposals()` into a stage evaluator without
changing the existing subset evaluator:

```text
ranked, shell = current population construction
search_population = current canonical ranked child Phase-2 population
widths = resolve_effective_search_widths(...)

cumulative diagnostics = empty
cumulative proposal cache = empty
prior_width = 0

for stage_index, width in enumerate(widths):
    search_pool = search_population[:width]

    extend or rebuild the joint workspace with cache reuse

    for each allowed subset B in search_pool:
        skip B unless B contains at least one index >= prior_width
        apply exact-child compatibility
        evaluate B with the existing joint evaluator
        accumulate feasibility and rejection diagnostics

    if cumulative feasible subset count > 0:
        rank feasible proposals using the existing score/tie policy
        return proposals and cumulative expansion summary

    prior_width = width

return no proposals and a summary proving whether all survivors were attempted
```

Do not run the old `_rank_feasible_child_phase2_population()` singleton
prefilter for canonical `joint_subset_gate_v1`. Preserve that path only for its
explicit diagnostic/compatibility policy.

### Workspace extension

Preferred implementation is an incrementally extensible workspace over nested
prefixes. At minimum, the receiving agent must prove that rebuilding a wider
workspace reuses every prior matrix element through the existing
state/scaffold/Hamiltonian-fingerprinted caches.

When expanding from old prefix `C_a` to `C_b`, new quantum geometry consists of:

- per-new-child gradient/diagonal data not already reused from child Phase 2;
- active-to-new-child mixed Gram/Hessian blocks;
- old-child to new-child mixed blocks;
- new-child to new-child mixed blocks.

Do not recompute:

- active-active `G_AA` or `H_AA`;
- old active-child blocks;
- old child-child blocks;
- child Phase-1/2 measurements already carried into the workspace.

The existing `_JointPairGeometryCache` fingerprints state, scaffold, and
Hamiltonian. Preserve those correctness guards. Be aware that an all-100 pair
workspace has 4,950 pairs, which exceeds a default 4,096-entry cache if prior
entries are expected to survive across separately rebuilt workspaces. A single
incremental workspace or a per-selector-call local cache avoids eviction-driven
remeasurement.

## Stop Semantics

The summary must separate these cases:

| Case | Widen? | Terminal interpretation |
|---|---:|---|
| No jointly feasible subset, narrower prefixes remain | yes | narrow-prefix geometric failure |
| No jointly feasible subset after full population | no | true full-population geometric exhaustion |
| At least one feasible subset, but all gains/scores nonpositive | no | nonpositive joint-gain stop, not rank exhaustion |
| Feasible positive proposal exists | no | admit selected singleton/batch |

Do not let outer orchestration infer scope solely from `proposals == []`.
Return explicit fields such as:

```text
joint_feasible_subset_count
positive_proposal_count
full_population_attempted
exhaustion_scope
terminal_reason
```

Recommended terminal detail values:

```text
narrow_prefix_no_joint_feasible_subset_v1
full_population_no_joint_feasible_subset_v1
joint_feasible_but_nonpositive_gain_v1
positive_joint_proposal_v1
```

For payload compatibility, the top-level historical stop reason may remain
`joint_geometry_selector_exhausted` when the full population is truly
exhausted, but include the precise new detail. Narrow-prefix failure must not
be promoted to that stop.

Update both beam stop sites and the ordinary stop site in
`adapt_pipeline.py`. Do not leave a second fallback guard that can recreate the
old narrow-prefix stop after the selector has returned expansion telemetry.

## Search-Expansion Telemetry

Add one structured payload:

```text
schema: route_a_batch_search_expansion_v1
policy
initial_requested_width
configured_expansion_sizes
child_phase2_search_population_count
effective_width_schedule
attempt_count
first_feasible_width
full_population_attempted
terminal_reason
exhaustion_scope
attempts:
  - stage_index
    requested_width
    effective_width
    previous_effective_width
    newly_exposed_record_count
    search_pool_truncated
    subset_counts_considered_delta
    subset_counts_evaluated_delta
    subset_counts_feasible_delta
    rejection_counts_delta
    cumulative_subset_counts_considered
    cumulative_subset_counts_feasible
    unique_geometry_elements_new
    unique_geometry_elements_cumulative
    pair_cache_hits_delta
    pair_cache_misses_delta
```

Keep existing summary fields. Their compatibility meanings should be:

- `batch_search_pool_size_requested`: original initial request;
- `batch_search_pool_size_effective`: terminal attempted/selected width;
- `batch_search_pool_truncated`: whether the terminal width remains below the
  full search population;
- `subset_counts_*`: cumulative counts across unique subset evaluations;
- `rejection_counts`: cumulative unique-subset rejection counts;
- `selected_subset` and `selected_cardinality`: unchanged.

## Trust-Update Telemetry

Attach one payload to the admitted history row, branch summary, checkpoint, and
final run summary:

```text
schema: route_a_trust_region_update_v1
policy
context_mode
radius_before
radius_after
update_factor
update_reason
trust_clipped
predicted_fs_displacement
realized_fs_displacement_exact
realized_fs_displacement_local
displacement_ratio
metric_direction_cosine
direction_guard_available
direction_guard_passed
predicted_energy_reduction
realized_energy_reduction_pre_prune
structural_rollback
depth_rollback
pre_admission_state_fingerprint
post_refit_pre_prune_state_fingerprint
active_context_indices
selected_record_identities
selected_effective_positions
```

Useful `update_reason` values:

```text
fixed_policy
realized_displacement_smaller
binding_radius_realized_displacement_larger
radius_inactive_hold
energy_veto_hold
direction_veto_hold
rollback_contract
invalid_measurement_contract
context_mode_not_supported
```

Never report an unavailable direction check as passed.

## Admission and Refit Integration

### Beam path

The beam child expansion path already retains:

- the base branch before insertion;
- child layout before/after insertion;
- selected records and effective insertion positions;
- `theta_before_opt_local` after insertion;
- `energy_prev_local` and post-optimizer `child.energy_current`;
- structural rollback handling before pruning.

Add explicit capture of:

- pre-admission theta/layout/state;
- selected proposal's joint-step telemetry and pre-admission metric context;
- post-refit, pre-prune theta/state/energy.

Update the child branch's trust state after rollback guards resolve and before
the prune route mutates the scaffold.

### Non-beam path

The ordinary route similarly has:

- rollback snapshots before admission;
- `theta_before_opt` after insertion;
- `energy_prev` and post-optimizer `energy_current`;
- structural and duplicate rollback handling;
- prune later in the round.

Capture the same pre-admission and post-refit/pre-prune artifacts. Update trust
state only after the final admission/rollback decision is known.

### Batch updates

One admitted batch produces one trust update. Use the joint selected proposal's
full `z_pred`, not a sum or average of singleton predictions. Compare it with
the joint post-refit state displacement.

## Checkpoint and Resume

Extend `_write_current_checkpoint()` with an optional serialized trust-state
snapshot. For beam checkpoints, preserve trust state per branch in the replay
payload. For ordinary checkpoints, preserve the current route state.

Requirements:

1. Resume from a new checkpoint restores the exact current radius and last
   update metadata.
2. A historical checkpoint lacking trust state remains readable.
3. Missing historical trust state initializes from
   `max_fubini_study_step` and records `legacy_checkpoint_missing_state`.
4. Branch clones do not alias trust-state dictionaries.
5. Checkpoint serialization contains only JSON-safe values.
6. A failed or interrupted round must not commit a half-applied next radius.

Do not reinterpret old history rows as if they had adaptive updates.

## Query-Work Accounting

The expansion policy must preserve the Paper-I rule that reused measurements
are charged once.

### Quantum work

Charge only newly required unique matrix/gradient elements at each expansion
stage. Reused child Phase-1/2 data, active-active blocks, and cache hits receive
zero incremental charge.

The terminal selector summary must expose:

```text
unique_geometry_elements_initial
unique_geometry_elements_added_by_expansion
unique_geometry_elements_cumulative
query_chargeable_gradient_repairs_initial
query_chargeable_gradient_repairs_added_by_expansion
```

`record_joint_selector_workspace_work()` should consume cumulative unique work
once for the winning branch's selector event. It must not separately charge
each stage's cumulative total.

### Classical work

Subset enumeration, rank tests, pseudoinverses, and joint linear solves are
classical work. Keep counts for runtime diagnosis, but do not add one quantum
query per evaluated pair or triple.

The existing `query_chargeable_batch_subset_count` is therefore a compatibility
or workload diagnostic unless a specific subset causes new measured matrix
elements. Do not use it as a substitute for unique geometry-element accounting.

### Beam accounting

Preserve the established branch-scoped accounting contract. A branch is
charged for the unique work performed on that branch. Final Paper-I `S_alg`
reporting follows the winning branch, not the sum over every discarded beam
branch, while expanded branch diagnostics may still be retained separately.

## File-by-File Implementation Map

### `pipelines/static_adapt/route_a_schur_selector.py`

- Add policy constants and typed config dataclasses.
- Validate expansion sizes and trust update controls.
- Serialize both nested configs in `RouteASchurSelectorConfig.as_dict()`.
- Map expansion policy/sizes into `FullScoreConfig`.
- Continue mapping only the effective current radius into `rho`.
- Add a pure `resolve_batch_search_width_schedule()` helper, or place that
  helper in the scoring module if it must remain private there.
- Preserve canonical global child dedup before any search width is resolved.

### `pipelines/scaffold/hh_continuation_scoring.py`

- Add expansion policy/sizes to `FullScoreConfig`.
- Refactor `combinatorial_reduced_plane_batch_proposals()` into a nested-prefix
  loop.
- Evaluate only newly exposed subsets.
- Reuse or incrementally extend `_BatchFullGeometryWorkspace`.
- Preserve `_JointPairGeometryCache` fingerprints.
- Separate feasible-count, positive-proposal-count, and terminal reason.
- Add stage and cumulative telemetry.
- Do not change `greedy_reduced_plane_batch_proposals()`.
- Do not alter `_BatchFullGeometryWorkspace.evaluate_subset()` rank or trust
  mathematics except to expose missing telemetry needed for radius calibration.

### `pipelines/static_adapt/route_a_trust_region.py` (new)

- Define `RouteATrustRegionState`.
- Define JSON serialization and legacy initialization.
- Implement exact Fubini-Study state distance.
- Implement optional local coordinate/direction diagnostics.
- Implement the pure next-radius rule.
- Keep this module independent of the optimizer and prune route.

### `pipelines/static_adapt/engine_support.py`

- Add branch-local trust state to `_BeamBranchState`.
- Deep-copy it in `clone_for_child()`.
- Include radius in any branch fingerprint only if it affects future branch
  behavior and dedup semantics require branches with different radii to remain
  distinct. Add a focused test for this decision.
- Include trust state in branch summaries and replay serialization.

### `pipelines/static_adapt/adapt_pipeline.py`

- Initialize ordinary and beam trust states.
- Pass the branch/current effective radius to the selector.
- Capture pre-admission and post-refit/pre-prune state.
- Apply one update after rollback resolution and before prune.
- Replace narrow-prefix exhaustion propagation with summary-driven stop logic.
- Update both beam exhaustion sites and the ordinary route.
- Serialize trust state in checkpoints, history, and final payload.
- Do not add new scientific controls as closure variables.

### `pipelines/static_adapt/paper_i_runner.py`

- Carry typed policies through `PaperISnakeRunConfig`.
- Add manifest fields for requested/effective policies and initial radius.
- Preserve old manifest keys.
- Ensure normalized settings distinguish fixed/off compatibility runs from the
  forward adaptive route.

### `pipelines/exact_bench/paper_i_hh_powell_pareto.py`

- Add CLI parsing for expansion policy and comma-separated expansion sizes.
- Add CLI parsing for trust update policy and visible numerical controls.
- Default `--batch-search-expansion-sizes` to `20,0` for the proposed forward
  route.
- Preserve `--batch-search-pool-size` as the initial width.
- Extend compact selector summaries with expansion attempts and trust updates.
- Dry-run output must show exact normalized policy differences without
  launching a scientific run.

### Tests

Primary existing test files:

```text
test/test_hh_continuation_scoring.py
test/test_static_adapt_route_a_funnel.py
test/test_static_adapt_paper_i_runner.py
test/test_paper_i_hh_powell_pareto.py
test/test_adapt_vqe_integration.py
```

Recommended focused new file:

```text
test/test_static_adapt_route_a_trust_region.py
```

## Required Regression Tests

### Conditional expansion

1. `off` reproduces the exact one-width behavior and summary values.
2. Width 15 with zero feasible subsets widens to 20.
3. Width 20 with a feasible subset does not continue to all.
4. Width 15 then 20 with zero feasible subsets widens to all.
5. Full-population zero feasibility returns
   `full_population_no_joint_feasible_subset_v1`.
6. A feasible but nonpositive subset does not trigger widening.
7. `L_0=0` searches all exactly once.
8. `N<15` resolves one clipped width without duplicate evaluation.
9. Repeated/decreasing expansion widths are normalized deterministically.
10. `B_max=1` still uses the joint subset evaluator and can widen.
11. `B_max=2` and `B_max=3` keep their exact cardinality caps at every width.
12. A different position of the same exact child remains an alternative but
    cannot coexist in one subset.
13. Newly exposed subset counts equal
    `choose(L_b,s)-choose(L_a,s)` before compatibility rejection.
14. Previously evaluated subsets are not evaluated twice.
15. Increasing width does not alter the ranked order of the original prefix.
16. Greedy selection is unchanged.

### Matrix and query reuse

17. Active-active blocks are built once across widening stages.
18. Old child-child pair blocks are not rebuilt after widening.
19. A cache hit adds zero unique geometry charge.
20. Triple enumeration adds no independent three-coordinate measurement term.
21. Cumulative query charge equals the union of unique measured elements, not
    the sum of cumulative stage totals.
22. Existing child Phase-1/2 reuse remains credited exactly once.

### Stop propagation

23. Non-beam narrow-prefix failure widens instead of stopping.
24. Beam narrow-prefix failure widens branch-locally instead of terminating the
    parent.
25. Beam and non-beam full-population exhaustion preserve readable top-level
    stop payloads plus the new precise detail.
26. Feasible nonpositive gain is not mislabeled rank exhaustion.

### Trust update mathematics

27. `fixed` leaves radius unchanged for every outcome.
28. Realized displacement below predicted displacement contracts the radius.
29. Realized displacement above predicted displacement expands only when the
    previous trust solve clipped.
30. An inactive trust radius holds even when the realized step is larger.
31. Nonpositive realized energy vetoes expansion.
32. Structural rollback contracts the radius.
33. Per-round contraction never exceeds the configured lower factor.
34. Per-round expansion never exceeds the configured upper factor.
35. Initial `rho=0.5` and `rho=10` are accepted and never clamped to an
    arbitrary hard maximum.
36. Exact Fubini-Study realized distance is invariant to global phase.
37. Identical states give zero realized distance within numerical tolerance.
38. Batch admission uses one joint predicted step and one joint radius update.
39. Post-prune state changes do not alter the recorded refit calibration.

### Coordinate and context safety

40. Full-ansatz mode uses every pre-admission coordinate in local telemetry.
41. Active-window mode uses the same indices for predicted and realized local
    displacement or reports adaptation unsupported.
42. Inserted coordinates are measured relative to the absent-operator origin.
43. Reordered insertion positions map to the correct post-insertion runtime
    coordinates.
44. The implementation does not silently assume every generator period is
    `2*pi`.
45. A missing direction diagnostic is reported as unavailable, not passed.

### Branch, checkpoint, and payload compatibility

46. Beam child clones hold independent trust-state objects.
47. Two sibling refits can update to different radii without cross-talk.
48. New checkpoints round-trip the exact radius and update count.
49. Old checkpoints without trust state initialize from the configured initial
    radius and remain readable.
50. Existing payload readers tolerate all added fields.
51. Dry-run manifests show both policies and expansion schedule.
52. Historical `off`/`fixed` manifests remain reproducible.

## Deterministic Acceptance Fixture

Before any scientific run, add a small fixture with at least 20 globally ranked
child Phase-2 records such that:

- every subset in the first 15 is jointly rank-rejected;
- at least one subset involving records 16-20 is jointly feasible;
- the selected feasible subset has deterministic score and identity;
- the fixture can count workspace builds and pair-cache accesses.

Expected audit:

```text
effective_width_schedule = [15, 20, N] or [15, 20]
attempt_count = 2
first_feasible_width = 20
full_population_attempted = false when N > 20
no subset wholly inside the first 15 is evaluated twice
unique geometry work equals union(first-15 blocks, newly exposed blocks)
```

Add a second fixture in which all `N` records are jointly infeasible. It must
attempt the full population before returning true geometric exhaustion.

## Validation Sequence

1. Run `py_compile` on every changed Python module and new test.
2. Run focused pure scoring and trust tests.
3. Run Route-A funnel, runner, harness, and integration tests.
4. Run the broader focused Route-A selector suite that previously covered the
   joint subset gate.
5. Produce one direct harness `--dry-run` manifest for `L_search=15`, expansion
   `20,all`, and adaptive trust. Do not launch the scientific cell in this
   implementation slice unless separately authorized.

The reliable fast test invocation in this repository has been:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --assert=plain ...
```

The receiving agent should select path-limited test targets and report exact
counts. Do not claim success from compilation alone.

## Acceptance Criteria

Implementation is complete only when all of the following are true:

- `L_search=15` can conditionally progress to 20 and then all without changing
  `B_max` or subset scoring.
- The first width with any jointly feasible subset terminates widening.
- Full-population exhaustion is distinguishable from narrow-prefix failure.
- Feasible nonpositive gain is distinguishable from rank exhaustion.
- No prior subset or matrix element is double-evaluated or double-charged.
- Fixed/off modes reproduce prior behavior.
- The radius can exceed `0.5` and has no arbitrary scientific maximum.
- Radius expansion occurs only after a binding trust solve and successful
  realized energy descent.
- Radius contraction follows smaller realized displacement or rollback.
- Beam trust state is branch-local and checkpoint-safe.
- Post-refit/pre-prune state is the calibration endpoint.
- Manifests and compact summaries expose all policy choices and outcomes.
- Existing payload readers and canonical child identity semantics remain
  compatible.

## Questions the Implementing Agent Must Resolve From Code, Not Guess

1. Does the selected proposal summary already retain enough workspace data to
   reconstruct `z_pred`, `G`, active ordering, and batch ordering after refit?
   If not, add a compact calibration payload without serializing enormous
   matrices into every final artifact.
2. What is the authoritative generator-specific parameter period contract for
   local `z_real` telemetry? Do not invent one. Exact state distance is the
   primary fallback.
3. Should branch dedup fingerprints include the current adaptive radius? If two
   otherwise identical branches with different radii would make different
   future decisions, they cannot be merged without a defined reconciliation
   rule.
4. Does `record_joint_selector_workspace_work()` currently charge per branch
   event or aggregate over discarded branches? Preserve the current Paper-I
   winning-branch reporting contract while retaining diagnostic branch totals.
5. Is the existing pair-cache capacity sufficient for an all-survivor stage in
   one call? If not, use an incremental local workspace instead of globally
   increasing cache size without bounds.

These are implementation questions, not authorization to change scientific
semantics.

## Explicitly Deferred

- Promotion of either policy to `Paper_I.tex`.
- Choosing final trust-update numerical constants from scientific evidence.
- Changing the canonical settings document after test-only implementation.
- Replaying the Pareto campaign.
- Replacing combinatorial selection with greedy selection.
- Altering rank tolerance, conditioning tolerance, cost weights, or soft
  additivity weight to make a fixture pass.

## Files Expected to Change During Implementation

```text
pipelines/static_adapt/route_a_schur_selector.py
pipelines/scaffold/hh_continuation_scoring.py
pipelines/static_adapt/route_a_trust_region.py                 # new
pipelines/static_adapt/engine_support.py
pipelines/static_adapt/adapt_pipeline.py
pipelines/static_adapt/paper_i_runner.py
pipelines/exact_bench/paper_i_hh_powell_pareto.py
test/test_hh_continuation_scoring.py
test/test_static_adapt_route_a_trust_region.py                 # new
test/test_static_adapt_route_a_funnel.py
test/test_static_adapt_paper_i_runner.py
test/test_paper_i_hh_powell_pareto.py
test/test_adapt_vqe_integration.py
```

Files explicitly excluded from this implementation slice:

```text
MATH/paper_details/Paper_I.tex
Paper_I.pdf
```
