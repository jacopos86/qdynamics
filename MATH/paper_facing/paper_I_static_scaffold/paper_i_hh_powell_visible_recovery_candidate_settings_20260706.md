# Paper-I HH POWELL Visible-Row Recovery Candidate Settings

Created: 2026-07-06
Updated: 2026-07-09

Purpose: agent-facing source of truth for the current Hubbard--Holstein SNAKE
local recovery/candidate line.  This file is intentionally narrower than the
older runtime-settings audit.  It starts from the currently visible Paper-I HH
POWELL results and records the user-approved canonical full-refit,
full-geometry, physical-lane, no-batching SNAKE controls below.

## Authority

Use this file together with:

```text
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_visible_row_provenance_layers_20260705.md
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_canonical_runtime_settings_draft_20260627.md
```

The visible-row provenance file wins over older source-lock ancestry and over
the older canonical draft when they disagree about the current rendered Paper-I
Hubbard--Holstein row.

Current visible block:

```text
BEGIN_MACHINE_READABLE_HH_POWELL_POOL_EXPOSURE_MAIN_UPDATE_20260702
```

Current support CSV:

```text
output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.csv
```

Current support JSON:

```text
output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.json
```

Current support PDF named by the manuscript block:

```text
output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.pdf
```

Larger POWELL provenance/report PDF often used for visual inspection:

```text
output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell.pdf
```

The first visible POWELL page/report block is the human-facing current Paper-I
Hubbard--Holstein reference; executable settings still come from the CSV/JSON
and the raw `cell_manifest.json` / `result.json` paths referenced by those
sidecars.

## Regime Set

The target suite is all six Paper-I Hubbard--Holstein regimes:

```text
weak-weak
intermediate-weak
strong-weak
weak-strong
intermediate-strong
strong-strong
```

Do not describe the suite as two regimes.  Two-at-a-time is only a local
scheduling/concurrency rule when the user approves it.

## Baseline Visible-Row Contract

Start from the current visible Paper-I HH SNAKE POWELL row settings:

```text
optimizer                                      POWELL
adapt_maxiter                                  200
adapt_final_refit_maxiter                      200
adapt_max_depth                                30
adapt_pool                                     full_meta
adapt_pool_class_filter_json                   None
pool interpretation                            unfiltered full_meta, HVA included
phase3_runtime_split_mode                      shortlist_pauli_children_v1
phase3_runtime_split_selection_mode            archival_child_set_forward_v1
phase3_runtime_split_child_set_symmetry_policy hard_guard
phase3_runtime_split_max_subset_size           1
adapt_child_pool_expansion_mode                off
shared_pauli_pool_mode                         off
phase3_source_lock_preferred_sequence          absent
```

The `hard_guard` label is historical route metadata.  For the visible weak-weak
source row, selected `child_set[...]` representatives carry `symmetry_spec=None`
and runtime-split metadata records `symmetry_gate.checked=false`,
`passed=true`, and
`skipped_reason=runtime_split_symmetry_spec_missing`.  Recovery runs should
preserve that archival missing-spec child-set representative behavior unless
the user explicitly asks for a true child-set hard-guard ablation.

## Approved Canonical Changes For New SNAKE Rows

For new Paper-I Hubbard--Holstein SNAKE rows, start from the visible-row
Hamiltonian, pool, optimizer, and runtime-split contract above plus the beam
settings below, then use the following canonical algorithmic controls:

```text
--adapt-beam-lambda 0.005
--adapt-reopt-policy full
--adapt-window-size 99
--adapt-window-topk 0
--adapt-full-refit-every 1
--adapt-final-full-refit true
--adapt-insertion-mode always
--phase1-probe-max-positions 999999
--phase1-trough-margin-ratio 1.0
--phase3-geometry-window-size 99
--phase3-novelty-ablation-mode off
--static-lane-route physical_operator_type
--phase1-prune-schur-nomination-route metric_regularized_v1
--phase1-prune-metric-schur-mu 0.01
--phase1-prune-metric-schur-cost-weighting ansatz_entry_denominator_v1
--phase2-no-batching
--phase3-no-batching
```

The canonical line uses full active refit after every accepted update, a
full-ansatz insertion-position domain, and full Phase-II/III geometry windows at
the Paper-I HH depth scale.  `--adapt-window-size 99` plus
`--adapt-window-topk 0` is the explicit full-refit guard for any window-derived
helper path; `--adapt-insertion-mode always` explicitly evaluates insertion
positions every accepted iteration; `--phase1-probe-max-positions 999999` keeps
the position window at full ansatz scale; and `--phase3-geometry-window-size
99` keeps the scoring geometry window full.  The canonical baseline keeps
metric/novelty scoring
active (`--phase3-novelty-ablation-mode off`), so mechanism-ablation rows should
be interpreted as disabled-minus-full against this active baseline.  The full
geometry setting is a configuration-level guard: it does not by itself implement
a single shared measured Gram cache between Phase-II/III scoring and
post-admission metric-prune nomination.  That shared-measurement cache is a
future implementation change if needed for QPU execution.

The earlier maxB=1 reduced-plane batch route is now diagnostic.  Do not use a
batch-enabled route for canonical Paper-I SNAKE rows unless the user explicitly
reopens batching as an ablation.

## Physical Operator-Type Lane Update

The current Paper-I manuscript route replaces the older algebraic
support/commutation lane interpretation with Hubbard--Holstein physical
operator-type lanes.  The lane label is determined by the physical role of the
candidate generator before expensive geometric scoring:

```text
electronic UCCSD
electronic hopping/current
phonon cloud
phonon relaxation
dressed electron-phonon
Hamiltonian-block / HVA
```

If an implementation field or older report still uses an
`algebraic_nested_v1`/algebraic-lane string, treat that as a legacy selector
surface name unless the result artifact explicitly records support- or
commutation-defined lanes.  For the current physical-lane line, shortlisting is
lane-wise across the six physical operator types above; support overlap and
commutation may remain telemetry or optional batch feasibility diagnostics, but
they do not define the lane partition.

The beam lambda is fixed for this line:

```text
adapt_beam_lambda = 0.005
adapt_beam_live_branches = 3
adapt_beam_children_per_parent = 2
```

Do not set `adapt_beam_lambda=0` for candidate/recovery runs in this line.
Lambda-zero runs are diagnostics only when explicitly requested.

The prior batch-route matrix had two diagnostic cells:

```text
diagnostic_batch_cell = greedy
phase2_enable_batching = true
phase2_batch_selection_mode = greedy_reduced_plane
phase2_batch_target_size = 1
phase2_batch_size_cap = 1
phase3_enable_batching = true
max accepted batch size = 1

diagnostic_batch_cell = combinatorial
phase2_enable_batching = true
phase2_batch_selection_mode = combinatorial_reduced_plane
phase2_batch_target_size = 1
phase2_batch_size_cap = 1
phase3_enable_batching = true
max accepted batch size = 1
```

The visible POWELL source already has batching enabled, but its stored command
uses the older reduced-plane batch surface with target/cap `8/16` and
Phase-III `reduced_plane` metadata.  The current canonical local line replaces
both the older `8/16` route and the interim maxB=1 route with explicit
no-batching flags:

```text
phase2_enable_batching = false
phase3_enable_batching = false
--phase2-no-batching
--phase3-no-batching
```

The maxB=1 route remains a diagnostic batch ablation only.  Do not impose a
hard pairwise commutation condition unless the user explicitly requests a
commutation ablation.

The run remains an adaptive SNAKE search.  It must not use
`--phase3-source-lock-preferred-sequence` unless the user explicitly requests a
fixed-ansatz replay/refit sanity check.

## Explicit Non-Defaults And Prohibitions

Do not use the older parent-source-lock cap-3 settings for this line:

```text
phase3_runtime_split_max_subset_size != 3
```

The visible-row recovery/candidate cap is:

```text
phase3_runtime_split_max_subset_size = 1
```

Do not use the old SPSA/subset-size-3 parent ancestry as the default Paper-I HH
fact.  It is historical ancestry unless the user explicitly asks to recover
that parent route.

Do not use `full_meta_minus_hva` or any class-filter JSON in this line.  HVA is
included through unfiltered `full_meta`.

Do not use fixed selected-generator replay/refit as adaptive recovery evidence.
A fixed-prefix refit can be a useful sanity check, but it is not a fresh SNAKE
selection run.

Do not submit, cancel, remove, or repair local or CHTC jobs from this line
without current-turn user permission for that exact action.

## Weak-Weak Prior Artifacts

The weak-weak source-quality beam/prune artifact below is useful, but it is not
the exact current canonical route because it used batching-era controls:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v14/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0p005_metric_prune_route/json/result.json
```

Observed result:

```text
final abs(Delta E)       3.968023969321832e-4
batch route              older reduced_plane / source-style batch surface
phase2 batch target/cap  8 / 16
adapt_beam_lambda        0.005 in run_command.sh
metric prune             metric_regularized_v1 in run_command.sh/result trace
```

Classification:

```text
matches visible POWELL source except approved beam/prune changes: mostly yes
matches current full-refit/full-geometry/no-batching contract: no
```

Therefore weak-weak should be considered historical/diagnostic for the current
canonical full-refit, full-geometry, no-batching route unless the user
explicitly accepts the older reduced-plane artifact as a substitute.

## Current Weak-Weak Canonical Diagnostic Anchor

The first local source-locked diagnostic for the new canonical controls is:

```text
raw_outputs/paper_i_hh_weak_weak_prune_metric_eta0p01_k15_fullgeom_fullreopt_diagnostic_20260709/weak_weak/json/result.json
```

Audit:

```text
raw_outputs/paper_i_hh_weak_weak_prune_metric_eta0p01_k15_fullgeom_fullreopt_diagnostic_20260709/source_lock_diagnostic_audit.json
```

Observed result:

```text
final abs(Delta E)       3.733740702102084e-4
adapt_max_depth          15
adapt_reopt_policy       full
adapt_window_size        99
adapt_full_refit_every   1
phase3_geometry_window   99
metric prune route       metric_regularized_v1
metric prune mu          0.01
batching                 disabled in Phase II and Phase III
```

This diagnostic verified that the full-geometry setting was active and that the
metric-prune surrogate was built, but no prune deletion was accepted.  It is a
settings anchor, not a completed six-regime Paper-I replacement matrix.

## Remaining Candidate Work

For this SNAKE-only POWELL local candidate line, the active target is:

```text
6 regimes x 1 canonical no-batching cell = 6 adaptive SNAKE runs
```

The six regimes are the full Paper-I Hubbard--Holstein grid listed above.
Do not reduce this to two regimes, and do not treat prior weak-weak
reduced-plane diagnostics as exact coverage of the current no-batching route.

## Command Audit Checklist

Before any local or CHTC launch for this line, verify:

```text
all six-regime suite target recorded
optimizer = POWELL
maxiter = 200
final/refit maxiter = 200
depth cap = 30
full_meta unfiltered
HVA included
no class-filter JSON
Phase-III archival split enabled
subset cap = 1
shared Pauli pool off
source-lock preferred sequence absent
adapt-beam-lambda = 0.005
adapt-beam-live-branches = 3
adapt-beam-children-per-parent = 2
metric_regularized_v1 prune route enabled
phase1_prune_metric_schur_mu = 0.01
adapt_reopt_policy = full
adapt_window_size = 99
adapt_window_topk = 0
adapt_full_refit_every = 1
adapt_insertion_mode = always
phase1_probe_max_positions = 999999
phase1_trough_margin_ratio = 1.0
phase3_geometry_window_size = 99
phase3_novelty_ablation_mode = off
phase1_prune_metric_schur_cost_weighting = ansatz_entry_denominator_v1
metric_prune_route_active = true
phase2 batching disabled
phase3 batching disabled
no fallback to older reduced_plane or maxB=1 batch routes
run remains adaptive SNAKE selection
```

Any generated manifest should record:

```text
provenance_layer = visible_row
visible_support_csv
visible_source_result_json
settings_reused
settings_changed
settings_change_reason
```

## Status Language

Use evidence/status terms, not promotion terms.  The user decides whether any
row becomes paper-facing evidence.
