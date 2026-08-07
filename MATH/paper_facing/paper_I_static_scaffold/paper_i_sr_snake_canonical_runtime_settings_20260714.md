# Canonical Paper-I SR-SNAKE Runtime Identity

Date: 2026-07-14
Scope: agent-facing route resolution and replay; not a manuscript result edit.

Status: historical `SR-SNAKE v1` authority. The unqualified conventional
`SR-SNAKE` identity was advanced on 2026-07-15 to the full-accepted-refit v2
profile documented in
`paper_i_sr_snake_canonical_runtime_settings_20260715.md`. Keep this file for
exact v1 replay; do not use it to assemble an unversioned current run.

## Canonical high-accuracy policy

The stable identity `SR-SNAKE v1` resolves exactly to:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_v1
sr_powell_coordinate_chart_policy = expanded_runtime_projected_logical_v1
```

The executable request is:

```text
--sr-route-profile sr_snake_v1
```

It materializes the complete contract owned by
`pipelines/static_adapt/sr_snake_route_profile.py`. The normalized contract
SHA-256 is
`fab7b5a6c4bd2ab019139367aa2a507356a5c969b6b88cd72d32365ae766e13e`.
Do not request canonical v1 by manually reconstructing component flags.

This is the historical/high-accuracy policy used by the existing SR-SNAKE
results in the Paper-I model-comparison report. The preserved weak--weak result
with absolute error `4.472864776339236e-7` used this Powell chart. Existing
artifacts are not rewritten by this identity repair.

`expanded_runtime_projected_logical_v1` means that Powell optimizes the active
expanded runtime coordinates. At every objective and lift boundary, runtime
coordinates belonging to one logical generator are projected to their block
mean before ansatz execution. Runtime vectors stored in checkpoints and results
remain expanded and ordered.

## Controller contract

Canonical SR-SNAKE v1 has the following structural settings:

- Phase 0 is off.
- Phase-II batching is off.
- Phase-III batching is off.
- admission cardinality is one candidate-position record per controller round;
- the historical singleton Pauli-child path, hard fixed-sector guard, and
  binary-padding projection remain active as recorded by the source lock;
- Phase III uses `supported_metric_whitened_eigh_v1` in the full
  active-plus-singleton coordinate model;
- the coordinate-solve scope is `phase3_only_v1`; Phase II is not whitened in
  this v1 identity;
- the trust update is `displacement_calibrated_unbounded_v2`;
- beam branch management remains active according to the locked run manifest;
- post-admission pruning remains `recoverability_ladder_v1` according to the
  locked run manifest;
- ordered signed prefix checkpoints, full-coordinate refits, optimizer budgets,
  stopping rules, and prefix selection remain source-locked run settings.

The canonical method-level values materialized by the selector include:

```text
problem = hh
adapt_pool = full_meta
adapt_pool_class_filter_json = null
adapt_pool_label_filter_json = null
adapt_continuation_mode = phase3_v1
static_route_id = route_a
static_lane_route = physical_operator_type
physical_lane_shortlist_aggressiveness = 3
adapt_inner_optimizer = POWELL
adapt_maxiter = 200
adapt_scipy_maxfev = 0
adapt_state_backend = compiled
adapt_seed = 7
adapt_reopt_policy = windowed
adapt_window_size = 3
adapt_full_refit_every = 8
adapt_final_full_refit = true
adapt_final_refit_maxiter = 200
adapt_insertion_mode = append_only
adapt_max_depth = 30
adapt_allow_repeats = true
phase0_pilot_enabled = false
phase1_shortlist_size = 24
phase2_shortlist_size = 12
phase2_shortlist_fraction = 0.25
phase2_enable_batching = false
phase3_enable_batching = false
phase3_runtime_split_mode = shortlist_pauli_children_v1
phase3_runtime_split_selection_mode = archival_child_set_forward_v1
phase3_runtime_split_max_subset_size = 1
phase3_runtime_split_subset_sizes = 1
phase3_runtime_split_child_set_symmetry_policy = hard_guard
phase3_runtime_split_child_padding_policy = exact_projected_grouped_v1
phase1_prune_enabled = true
phase1_prune_policy = recoverability_ladder_v1
phase1_prune_mode = both
adapt_beam_live_branches = 3
adapt_beam_children_per_parent = 2
adapt_beam_lambda = 0.005
adapt_beam_terminal_archive_mode = disabled
phase2_novelty_mode = collective_span_v1
phase2_gram_novelty_policy = ordinary_multiplier_v1
phase3_gram_novelty_policy = ordinary_multiplier_v1
phase3_novelty_ablation_mode = off
sr_escape_mode = disabled
sr_controller_ablation_contract = off
```

The contract also pins the remaining prune thresholds, hardware-cost policy,
backend compile proxy, fallback, shadow/debug, and selected-logical/pool fields.
The executable contract, not this abbreviated display, is the machine-readable
authority. Regime physics and the same-cutoff exact reference remain supplied
by the per-regime source lock; the selector does not invent them.

The novelty/scoring policy is not changed by this replay-robustness repair. A
future policy with Phase-II whitening and with novelty removed as a score
multiplier must receive a new route profile and a new source lock; it must not
be called canonical SR-SNAKE v1.

## Optional reduced-Powell variant

The newer logical-coordinate chart remains available only as this distinct
profile:

```text
route_family = singleton_response_snake
route_profile = supported_whitened_adaptive_trust_reduced_powell_v2
sr_powell_coordinate_chart_policy = logical_shared_reduced_v1
```

`logical_shared_reduced_v1` gives Powell one coordinate per active logical
generator and expands each optimized logical value uniformly across its runtime
block. It must never silently execute under
`supported_whitened_adaptive_trust_v1`.

## Resolution and replay rules

- An ordinary current request for canonical `SR-SNAKE v1` uses
  `--sr-route-profile sr_snake_v1`. The lower-level
  `--sr-powell-coordinate-chart-policy auto` value, when left implicit, is
  materialized as `expanded_runtime_projected_logical_v1` by profile
  normalization before optimizer execution.
- Commands, normalized settings, manifests, signed checkpoints, resume state,
  and results must record the concrete resolved chart, route-profile request,
  complete contract, and contract SHA-256.
- Historical/source-locked resume or replay must fail closed when the chart is
  missing, unknown, contradictory across preserved locations, or different
  from the requested profile, or when the canonical contract is missing or
  mismatched.
- Non-SR static-ADAPT routes retain their prior reduced-Powell behavior; this
  SR identity repair is not a global Powell-default change.
- Future agents must not reconstruct canonical v1 from live defaults. Start
  from the source-locked manifest and require exact agreement in every
  executable field other than explicitly approved output/provenance paths.

## Historical replay archive

The original recovered archive remains unchanged. Its self-contained import
closure is the immutable revision:

```text
raw_outputs/paper_i_hh_sr_snake_historical_source_recovery_20260714/
  source_lock_revision_v2_self_contained_20260715/
```

The revision archive SHA-256 is
`c290d9ee1b31cd211e41faad174cd2e311ca65cf351c46bbb84fbaaea9504c6c`.
Its manifest and import-preflight receipts are authoritative for the remaining
hashes and module-origin checks. The launcher defaults to validation-only and
requires an explicit execution flag plus a new no-clobber output root before
it can run scientific work.

No scientific run, CHTC submission, result promotion, manuscript edit, or PDF
regeneration is authorized by this document.
