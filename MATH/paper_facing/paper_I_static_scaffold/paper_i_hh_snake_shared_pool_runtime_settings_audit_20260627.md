# Paper I HH SNAKE Runtime Settings Dossier
Created: 2026-06-27
Purpose: define the SNAKE Hubbard--Holstein command surface in a readable Paper-I math-to-repo form. This is support material, not manuscript prose.
## Core Invariant
SNAKE settings are regime-specific. For a fixed regime, Powell, ROTOSOLVE, and SPSA use the same SNAKE settings; only the inner optimizer and budget overlay changes. ROTOSOLVE is used below as the literal CLI template because it avoids SPSA-specific parameter noise while still carrying the full SNAKE command surface.

## Naming Legend
`[impl-v2]` marks a historical implementation-prefix name: the raw `phase1`, `phase2`, or `phase3` prefix does not by itself mean the Paper-I phase with that number. `[paper-I-phase]` marks a checked phase-prefix name: the prefix is being used in the Paper-I staged-phase sense described by the adjacent paper object. These markers are used only in explanatory tables; the command blocks remain literal CLI.

| Field family | Marker | How to read it |
| --- | --- | --- |
| `phase0-pilot-*` | `[paper-I-phase]` | Paper-I pilot screen before staged scoring. |
| `phase1-shortlist-*`, `phase1-probe-*` | `[paper-I-phase]` | Paper-I Phase-I candidate/position screen. |
| `phase1-prune-*` | `[impl-v2]` | Generator-ablation/prune module, not Paper-I Phase I. |
| `phase2-shortlist-*`, `phase2-novelty-*` | `[paper-I-phase]` | Paper-I Phase-II novelty/rerank shortlist. |
| `phase2-rho`, `phase2-w-*`, `phase2-lambda-*`, `phase2-frontier-ratio` | `[impl-v2]` | Shared full/v2 score configuration; it can feed Phase-I scoring, Phase-II ranking, and Phase-III selector logic. |
| `phase2-enable-batching`, `phase2-batch-*` | `[impl-v2]` | Shared batch configuration and tolerance fields, not Phase-II-only batching. |
| `phase2-maturity-*`, `phase2-null-*`, `phase2-live-*`, `phase2-hysteresis-*` | `[paper-I-phase]` | Stage-controller gates for Paper-I Phase II liveliness. |
| `phase3-selector-*`, `phase3-geometry-*`, `phase3-window-*`, `phase3-backend-*` | `[paper-I-phase]` | Paper-I Phase-III reduced-geometry/backend selector controls. |
| `phase3-batch-*` | `[paper-I-phase]` | Paper-I Phase-III batch selection/order controls; the enable switch is implementation-coupled to the shared batch flag. |
| `phase3-runtime-split-*` | `[impl-v2]` | Archival child-split surface; current shared Pauli-child runs keep it off and use shared-pool fields instead. |
| `phase3-maturity-*`, `phase3-null-*`, `phase3-live-*`, `phase3-hysteresis-*` | `[paper-I-phase]` | Stage-controller gates for Paper-I Phase III liveliness. |

## Current Repair Overlay
| Item | Value |
| --- | --- |
| Source anchor records | `chtc/phase3_optuna/input/paper_i_hh_shared_pool_snake_minushva_powell_all_regimes_20260627_v1/paper_i_hh_spsa_budget_ladder_records.tsv` |
| Source command field | `source_command_args_json` |
| Parent source batch | `paper_i_hh_native_forced_child_matrix_depth30_20260623_v1` |
| Repair class filter | `agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json` |
| Submitted CHTC clusters | `8362781-8362786` |
| Validation | source-lock audits pass for generated repair rows |
| Duplication policy | raw Powell/SPSA command duplication is collapsed because optimizer variants share the same SNAKE settings within each regime |

| Cluster | Optimizer | Scope | Rows | maxiter | Depth | Pool / child policy | Warm start |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `8362781` | Powell | all six | 6 | 200 | `30` | `full_meta_minus_hva; shared_pauli_child_sets_v1 cap=3 hard_guard` | `append-prune` |
| `8362782` | ROTOSOLVE | all six | 6 | 200 | `30` | `full_meta_minus_hva; shared_pauli_child_sets_v1 cap=3 hard_guard` | `append-prune` |
| `8362783` | SPSA | all six | 6 | 200 | `30` | `full_meta_minus_hva; shared_pauli_child_sets_v1 cap=3 hard_guard` | `append-prune` |
| `8362784` | Powell | strong Holstein only | 3 | 500 | `30` | `full_meta_minus_hva; shared_pauli_child_sets_v1 cap=3 hard_guard` | `append-prune` |
| `8362785` | ROTOSOLVE | strong Holstein only | 3 | 500 | `30` | `full_meta_minus_hva; shared_pauli_child_sets_v1 cap=3 hard_guard` | `append-prune` |
| `8362786` | SPSA | strong Holstein only | 3 | 500 | `30` | `full_meta_minus_hva; shared_pauli_child_sets_v1 cap=3 hard_guard` | `append-prune` |

## Optimizer And Budget Overlay
The ROTOSOLVE command templates still carry inherited `--adapt-spsa-*` flags because they come from the source-locked command builder. Those constants are operationally relevant only when the optimizer overlay is `SPSA`.

| Run surface | How to read / derive it | CLI overlay |
| --- | --- | --- |
| ROTOSOLVE all-regime | literal command shown below | `--adapt-inner-optimizer ROTOSOLVE`; `--adapt-maxiter 200`; `--adapt-final-refit-maxiter 200` |
| Powell all-regime | same SNAKE settings as ROTOSOLVE for the same regime | replace optimizer with `POWELL`; keep maxiter/final-refit maxiter `200` |
| SPSA all-regime | same SNAKE settings as ROTOSOLVE for the same regime, with SPSA constants active | replace optimizer with `SPSA`; keep `--adapt-spsa-*` constants and maxiter/final-refit maxiter `200` |
| Strong-Holstein diagnostic | same regime-specific SNAKE settings for WS/IS/SS only | replace maxiter/final-refit maxiter with `500`; optimizer may be ROTOSOLVE, Powell, or SPSA |

## Paper-I Math To Repo CLI Map
This table follows `agent_guidance/static-adapt/route-a-language.md`. It maps the paper-method objects to the fields that appear in the commands below.
| Paper-I symbol / object | Method meaning | Repo CLI fields |
| --- | --- | --- |
| `r=(m,p)` | candidate-position record: generator plus insertion position | `--adapt-pool full_meta`; `--phase1-probe-max-positions` [paper-I-phase]; position/window flags such as `--adapt-window-size`, `--adapt-window-topk`, `--phase3-geometry-window-size` [paper-I-phase] |
| `R_k(t)`, `S_k(t)` | staged candidate universe and shortlists | `--phase0-pilot-max-records` [paper-I-phase]; `--phase1-shortlist-size` [paper-I-phase]; `--phase2-shortlist-fraction` [paper-I-phase]; `--phase2-shortlist-size` [paper-I-phase]; `--phase2-frontier-ratio` [impl-v2]; `--phase3-frontier-ratio` [paper-I-phase] |
| `K_k(r;t)` | resource-cost burden in score | `--phase2-w-depth` [impl-v2]; `--phase2-w-group` [impl-v2]; `--phase2-w-optdim` [impl-v2]; `--phase2-w-reuse` [impl-v2]; `--phase2-w-lifetime` [impl-v2]; `--phase2-w-shot` [impl-v2]; Phase-3 backend weights [paper-I-phase] |
| `F*_r`, `h*_r`, `q*_r` | Phase-III reduced-window Schur geometry | `--phase3-selector-policy algebraic_nested_v1` [paper-I-phase]; `--phase3-selector-geometry-mode reduced` [paper-I-phase]; `--phase3-window-relaxation-mode reduced` [paper-I-phase]; `--adapt-schur-warm-start-mode append-prune`; `--phase2-rho` [impl-v2] |
| `C_split(m)` | Pauli-child / child-set exploration | `--shared-pauli-pool-mode shared_pauli_child_sets_v1`; `--shared-pauli-pool-symmetry-policy hard_guard`; `--shared-pauli-pool-max-subset-size 3`; `--adapt-child-pool-expansion-mode off` |
| `B_child` | beam continuation over child/branch choices | `--adapt-beam-live-branches`; `--adapt-beam-children-per-parent`; `--adapt-beam-terminated-keep` |
| `G_B` / reduced batch plane | batch admission geometry | `--phase2-enable-batching` [impl-v2]; `--phase3-enable-batching` [paper-I-phase]; `--phase3-batch-selection-mode reduced_plane` [paper-I-phase]; batch target/cap/tolerance flags |
| `d_j in O_t` | rollback-safe generator ablation / prune recoverability | `--phase1-prune-enabled` [impl-v2]; `--phase1-prune-policy recoverability_ladder_v1` [impl-v2]; `--phase1-prune-mode both` [impl-v2]; prune fraction and optional collapse-witness flags [impl-v2] |
| `n_ph^work` | algorithmic working phonon cutoff | `--n-ph-max` |
| `full_meta_minus_hva` | Paper-I HH adaptive pool: full-meta with HVA excluded from adaptive selection | `--adapt-pool full_meta`; `--adapt-pool-class-filter-json agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json` |

## Shared SNAKE Command Surface
| Surface | Shared CLI fields |
| --- | --- |
| Route/profile | `--static-route-id route_a`; `--static-meta-feature-profile paper_i_production_v1`; `--adapt-continuation-mode phase3_v1` |
| Pool profile | `--adapt-pool full_meta` plus `--adapt-pool-class-filter-json agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json` |
| Shared Pauli-child exposure | `--shared-pauli-pool-mode shared_pauli_child_sets_v1`; `--shared-pauli-pool-symmetry-policy hard_guard`; `--shared-pauli-pool-max-subset-size 3`; `--adapt-child-pool-expansion-mode off`; `--phase3-runtime-split-mode off` [impl-v2] |
| Shortlists | `--phase0-pilot-max-records 96` [paper-I-phase]; `--phase1-shortlist-size 24` [paper-I-phase]; `--phase2-shortlist-fraction 0.25` [paper-I-phase]; `--phase2-shortlist-size 12` [paper-I-phase] |
| Beam | `--adapt-beam-live-branches 3`; `--adapt-beam-children-per-parent 2`; `--adapt-beam-terminated-keep 3` |
| Batching | `--phase2-enable-batching` [impl-v2]; `--phase3-enable-batching` [paper-I-phase]; `--phase2-batch-target-size 8` [impl-v2]; `--phase2-batch-size-cap 16` [impl-v2]; `--phase3-batch-selection-mode reduced_plane` [paper-I-phase]; `--phase3-batch-prefilter-mode off` [paper-I-phase] |
| Schur/reduced geometry | `--phase3-selector-policy algebraic_nested_v1` [paper-I-phase]; `--phase3-selector-geometry-mode reduced` [paper-I-phase]; `--phase3-window-relaxation-mode reduced` [paper-I-phase]; `--adapt-schur-warm-start-mode append-prune` |
| Pruning | `--phase1-prune-enabled` [impl-v2]; `--phase1-prune-policy recoverability_ladder_v1` [impl-v2]; `--phase1-prune-mode both` [impl-v2]; `--phase1-prune-amplitude-witness-optional` [impl-v2] |
| Stopping repair overlay | `--adapt-segment-target-depth 30`; `--adapt-segment-max-new-admissions 30`; `--adapt-drop-floor -1`; `--adapt-drop-patience 0`; `--adapt-drop-min-depth 0`; `--adapt-grad-floor -1` |

## Source Anchor Regimes
| Regime | Rank | Trial | U/t | lambda = 2 g_ep^2 | n_ph |
| --- | --- | --- | --- | --- | --- |
| weak-weak | 1 | 52 | `0.25` | `0.25` | `2` |
| intermediate-weak | 1 | 8 | `1.25` | `0.25` | `2` |
| strong-weak | 6 | 167 | `8` | `0.25` | `2` |
| weak-strong | 1 | 4 | `0.25` | `1.25` | `4` |
| intermediate-strong | 4 | 4 | `1.25` | `1.25` | `4` |
| strong-strong | 1 | 81 | `8` | `1.25` | `4` |

## Regime-Specific SNAKE Knobs
These are the main SNAKE knobs that differ by Hubbard--Holstein regime. They are shared by ROTOSOLVE, Powell, and SPSA for that same regime.
### Problem Values
| Flag | WW | IW | SW | WS | IS | SS |
| --- | --- | --- | --- | --- | --- | --- |
| `--u` | `0.25` | `1.25` | `8` | `0.25` | `1.25` | `8` |
| `--g-ep` | `0.353553390593` | `0.353553390593` | `0.353553390593` | `0.790569415042` | `0.790569415042` | `0.790569415042` |
| `--n-ph-max` | `2` | `2` | `2` | `4` | `4` | `4` |


### Mixed Paper-Phase And Implementation-Window Knobs
Rows marked `[impl-v2]` use historical implementation names; rows marked `[paper-I-phase]` are checked Paper-I phase names.

| Flag | WW | IW | SW | WS | IS | SS |
| --- | --- | --- | --- | --- | --- | --- |
| `--phase2-rho` [impl-v2] | `0.25` | `0.25` | `0.5` | `0.25` | `0.25` | `0.5` |
| `--phase2-w-shot` [impl-v2] | `0.08` | `0.04` | `0.15` | `0.02` | `0.02` | `0.08` |
| `--adapt-window-size` | `4` | `4` | `999999` | `4` | `4` | `16` |
| `--adapt-window-topk` | `4` | `4` | `999999` | `4` | `4` | `16` |
| `--phase3-geometry-window-size` [paper-I-phase] | `4` | `4` | `0` | `4` | `4` | `16` |
| `--phase3-backend-w-depth` [paper-I-phase] | `0.25` | `0.1` | `0.25` | `0.25` | `0.25` | `0.1` |


### Pruning And Batch Tolerances
Rows marked `[impl-v2]` use historical implementation names; rows marked `[paper-I-phase]` are checked Paper-I phase names.

| Flag | WW | IW | SW | WS | IS | SS |
| --- | --- | --- | --- | --- | --- | --- |
| `--phase1-prune-fraction` [impl-v2] | `0.4101910583864897` | `0.1930961457788297` | `0.4101910583864897` | `0.33922934316592934` | `0.33922934316592934` | `0.1930961457788297` |
| `--phase2-batch-near-degenerate-ratio` [impl-v2] | `0.9982411735035968` | `0.914354284671342` | `0.9982411735035968` | `0.98` | `0.98` | `0.914354284671342` |
| `--phase3-batch-near-degenerate-ratio` [paper-I-phase] | `0.9982411735035968` | `0.914354284671342` | `0.9982411735035968` | `0.98` | `0.98` | `0.914354284671342` |
| `--phase2-batch-rank-rel-tol` [impl-v2] | `0.00013662376421438911` | `7.703203666118798e-07` | `0.00013662376421438911` | `1.909930091607197e-05` | `1.909930091607197e-05` | `7.703203666118798e-07` |
| `--phase3-batch-rank-rel-tol` [paper-I-phase] | `0.00013662376421438911` | `7.703203666118798e-07` | `0.00013662376421438911` | `1.909930091607197e-05` | `1.909930091607197e-05` | `7.703203666118798e-07` |
| `--phase2-batch-additivity-tol` [impl-v2] | `0.6663130343903237` | `0.010276490515218235` | `0.6663130343903237` | `0.09993123296803053` | `0.09993123296803053` | `0.010276490515218235` |
| `--phase3-batch-additivity-tol` [paper-I-phase] | `0.6663130343903237` | `0.010276490515218235` | `0.6663130343903237` | `0.09993123296803053` | `0.09993123296803053` | `0.010276490515218235` |


### Maturity, Hysteresis, And Exceptions
| Flag | WW | IW | SW | WS | IS | SS |
| --- | --- | --- | --- | --- | --- | --- |
| `--phase1-maturity-cap-min` [paper-I-phase] | `8` | `12` | `12` | `8` | `8` | `24` |
| `--phase1-maturity-cap-max` [paper-I-phase] | `24` | `32` | `32` | `24` | `24` | `64` |
| `--phase2-maturity-cap-min` [paper-I-phase] | `6` | `8` | `8` | `6` | `6` | `12` |
| `--phase2-maturity-cap-max` [paper-I-phase] | `16` | `24` | `24` | `16` | `16` | `48` |
| `--phase3-maturity-cap-min` [paper-I-phase] | `4` | `4` | `4` | `4` | `4` | `8` |
| `--phase3-maturity-cap-max` [paper-I-phase] | `12` | `16` | `16` | `12` | `12` | `32` |
| `--phase-maturity-shot-min` | `1` | `2` | `1` | `1` | `1` | `1` |
| `--phase-maturity-shot-max` | `2` | `8` | `4` | `1` | `1` | `4` |
| `--phase1-maturity-shot-cap` [paper-I-phase] | `2` | `4` | `2` | `1` | `1` | `2` |
| `--phase2-maturity-shot-cap` [paper-I-phase] | `2` | `8` | `4` | `1` | `1` | `4` |
| `--phase3-maturity-shot-cap` [paper-I-phase] | `2` | `8` | `4` | `1` | `1` | `4` |
| `--adapt-no-repeats` | `true` | `true` | `true` | `-` | `-` | `-` |
| `--phase-live-hysteresis-enabled` | `true` | `true` | `-` | `-` | `-` | `-` |
| `--adapt-resume-mode` | `-` | `-` | `-` | `-` | `-` | `scaffold_v1` |
| `--phase1-prune-collapse-peak-abs-min` [impl-v2] | `-` | `-` | `2e-3` | `-` | `-` | `-` |
| `--phase1-prune-collapse-current-abs-max` [impl-v2] | `-` | `-` | `5e-4` | `-` | `-` | `-` |
| `--phase1-prune-collapse-ratio` [impl-v2] | `-` | `-` | `0.2` | `-` | `-` | `-` |


## ROTOSOLVE Command Templates
These six command blocks are the literal SNAKE command templates for the all-regime `maxiter=200` repair surface. To obtain Powell or SPSA, apply the optimizer overlay above; do not alter the regime-specific SNAKE knobs unless intentionally changing the SNAKE policy. Output paths are local audit placeholders generated by the source-lock command builder; submitted CHTC output directories are recorded in the input TSV and submit artifacts.

### weak-weak / ROTOSOLVE / maxiter 200
```bash
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -u -m pipelines.static_adapt.adapt_pipeline \
  --dv 0.0 --boson-encoding binary --ordering blocked --boundary open --term-order sorted --adapt-eps-grad 5e-7 \
  --adapt-eps-energy 1e-9 --adapt-seed 7 --adapt-state-backend compiled --adapt-reopt-policy windowed \
  --adapt-full-refit-every 8 --adapt-beam-live-branches 3 --adapt-beam-children-per-parent 2 \
  --adapt-beam-terminated-keep 3 --phase1-prune-max-candidates 6 --phase1-prune-max-regression 1e-8 \
  --phase1-probe-max-positions 999999 --phase1-trough-margin-ratio 1.0 --phase2-lambda-H 1e-6 --phase2-rho 0.25 \
  --phase2-gamma-N 1.0 --phase2-w-depth 0.2 --phase2-w-group 0.15 --phase2-w-optdim 0.1 --phase2-w-reuse 0.1 \
  --phase2-w-lifetime 0.05 --phase2-frontier-ratio 0.9 --phase3-frontier-ratio 0.9 --phase2-batch-target-size 8 \
  --phase2-batch-size-cap 16 --phase3-enable-rescue --phase3-lifetime-cost-mode off --adapt-drop-floor -1 \
  --adapt-drop-patience 0 --adapt-drop-min-depth 0 --adapt-no-repeats --phase3-backend-cost-mode \
  marrakesh_graph_span_v1 --phase3-backend-name FakeMarrakesh --phase3-backend-transpile-seed 7 \
  --phase3-backend-optimization-level 1 --phase1-maturity-cap-min 8 --phase1-maturity-cap-max 24 \
  --phase2-maturity-cap-min 6 --phase2-maturity-cap-max 16 --phase3-maturity-cap-min 4 --phase3-maturity-cap-max \
  12 --phase-maturity-shot-min 1 --phase-maturity-shot-max 2 --phase1-maturity-shot-cap 2 \
  --phase2-maturity-shot-cap 2 --phase3-maturity-shot-cap 2 --phase-live-hysteresis-enabled \
  --phase2-null-nrem-high-threshold 0.0 --phase2-live-nrem-low-threshold 0.25 --phase3-null-nrem-high-threshold \
  0.75 --phase3-live-nrem-low-threshold 1.25 --phase2-hysteresis-steps 2 --phase3-hysteresis-steps 1 \
  --adapt-window-size 4 --adapt-window-topk 4 --phase3-geometry-window-size 4 --phase3-backend-w-2q 1.0 \
  --phase3-backend-w-depth 0.25 --phase3-backend-w-size 0.01 --phase1-prune-fraction 0.4101910583864897 \
  --phase2-batch-near-degenerate-ratio 0.9982411735035968 --phase3-batch-near-degenerate-ratio \
  0.9982411735035968 --phase2-batch-rank-rel-tol 0.00013662376421438911 --phase3-batch-rank-rel-tol \
  0.00013662376421438911 --phase2-batch-additivity-tol 0.6663130343903237 --phase3-batch-additivity-tol \
  0.6663130343903237 --phase2-w-shot 0.08 --adapt-inner-optimizer ROTOSOLVE --adapt-spsa-a 0.1 --adapt-spsa-c \
  0.02 --adapt-spsa-alpha 0.602 --adapt-spsa-gamma 0.101 --adapt-spsa-A 5.0 --adapt-spsa-avg-last 0 \
  --adapt-spsa-eval-repeats 1 --adapt-spsa-eval-agg mean --adapt-spsa-callback-every 5 \
  --adapt-spsa-progress-every-s 30.0 --problem hh --L 2 --t 1 --u 0.25 --omega0 1 --g-ep 0.353553390593 \
  --n-ph-max 2 --adapt-pool full_meta --phase3-runtime-split-mode off --adapt-exact-gs-reference-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json \
  --adapt-max-depth 30 --adapt-maxiter 200 --adapt-final-refit-maxiter 200 --adapt-final-full-refit true \
  --phase0-pilot-max-records 96 --phase1-shortlist-size 24 --phase2-shortlist-fraction 0.25 \
  --phase2-shortlist-size 12 --adapt-parallel-gradient-workers 2 --adapt-beam-parent-workers 2 \
  --adapt-spsa-parallel-evaluations 2 --static-route-id route_a --static-meta-feature-profile \
  paper_i_production_v1 --adapt-continuation-mode phase3_v1 --phase2-novelty-mode collective_span_v1 \
  --hardware-resolution-mode ideal --phase3-selector-policy algebraic_nested_v1 --phase3-selector-geometry-mode \
  reduced --phase3-novelty-ablation-mode off --phase3-window-relaxation-mode reduced --phase2-enable-batching \
  --phase3-enable-batching --phase3-batch-selection-mode reduced_plane --phase3-batch-prefilter-mode off \
  --phase1-prune-enabled --phase1-prune-policy recoverability_ladder_v1 --phase1-prune-mode both \
  --phase1-prune-amplitude-witness-optional --phase3-symmetry-mitigation-mode off --output-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__weak_weak__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/json/result.json \
  --adapt-current-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__weak_weak__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/current.json \
  --adapt-current-json-every-depth 1 --adapt-current-json-keep-history-tail 100 --skip-pdf \
  --adapt-segment-target-depth 30 --adapt-segment-max-new-admissions 30 --adapt-grad-floor -1 \
  --adapt-schur-warm-start-mode append-prune --adapt-child-pool-expansion-mode off --shared-pauli-pool-mode \
  shared_pauli_child_sets_v1 --shared-pauli-pool-symmetry-policy hard_guard --shared-pauli-pool-max-subset-size \
  3 --adapt-pool-class-filter-json agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json

```

### intermediate-weak / ROTOSOLVE / maxiter 200
```bash
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -u -m pipelines.static_adapt.adapt_pipeline \
  --dv 0.0 --boson-encoding binary --ordering blocked --boundary open --term-order sorted --adapt-eps-grad 5e-7 \
  --adapt-eps-energy 1e-9 --adapt-seed 7 --adapt-state-backend compiled --adapt-reopt-policy windowed \
  --adapt-full-refit-every 8 --adapt-beam-live-branches 3 --adapt-beam-children-per-parent 2 \
  --adapt-beam-terminated-keep 3 --phase1-prune-max-candidates 6 --phase1-prune-max-regression 1e-8 \
  --phase1-probe-max-positions 999999 --phase1-trough-margin-ratio 1.0 --phase2-lambda-H 1e-6 --phase2-rho 0.25 \
  --phase2-gamma-N 1.0 --phase2-w-depth 0.2 --phase2-w-group 0.15 --phase2-w-optdim 0.1 --phase2-w-reuse 0.1 \
  --phase2-w-lifetime 0.05 --phase2-frontier-ratio 0.9 --phase3-frontier-ratio 0.9 --phase2-batch-target-size 8 \
  --phase2-batch-size-cap 16 --phase3-enable-rescue --phase3-lifetime-cost-mode off --adapt-drop-floor -1 \
  --adapt-drop-patience 0 --adapt-drop-min-depth 0 --adapt-no-repeats --phase3-backend-cost-mode \
  marrakesh_graph_span_v1 --phase3-backend-name FakeMarrakesh --phase3-backend-transpile-seed 7 \
  --phase3-backend-optimization-level 1 --phase1-maturity-cap-min 12 --phase1-maturity-cap-max 32 \
  --phase2-maturity-cap-min 8 --phase2-maturity-cap-max 24 --phase3-maturity-cap-min 4 --phase3-maturity-cap-max \
  16 --phase-maturity-shot-min 2 --phase-maturity-shot-max 8 --phase1-maturity-shot-cap 4 \
  --phase2-maturity-shot-cap 8 --phase3-maturity-shot-cap 8 --phase-live-hysteresis-enabled \
  --phase2-null-nrem-high-threshold 0.0 --phase2-live-nrem-low-threshold 0.25 --phase3-null-nrem-high-threshold \
  0.75 --phase3-live-nrem-low-threshold 1.25 --phase2-hysteresis-steps 2 --phase3-hysteresis-steps 1 \
  --adapt-window-size 4 --adapt-window-topk 4 --phase3-geometry-window-size 4 --phase3-backend-w-2q 1.0 \
  --phase3-backend-w-depth 0.1 --phase3-backend-w-size 0.01 --phase1-prune-fraction 0.1930961457788297 \
  --phase2-batch-near-degenerate-ratio 0.914354284671342 --phase3-batch-near-degenerate-ratio 0.914354284671342 \
  --phase2-batch-rank-rel-tol 7.703203666118798e-07 --phase3-batch-rank-rel-tol 7.703203666118798e-07 \
  --phase2-batch-additivity-tol 0.010276490515218235 --phase3-batch-additivity-tol 0.010276490515218235 \
  --phase2-w-shot 0.04 --adapt-inner-optimizer ROTOSOLVE --adapt-spsa-a 0.1 --adapt-spsa-c 0.02 \
  --adapt-spsa-alpha 0.602 --adapt-spsa-gamma 0.101 --adapt-spsa-A 5.0 --adapt-spsa-avg-last 0 \
  --adapt-spsa-eval-repeats 1 --adapt-spsa-eval-agg mean --adapt-spsa-callback-every 5 \
  --adapt-spsa-progress-every-s 30.0 --problem hh --L 2 --t 1 --u 1.25 --omega0 1 --g-ep 0.353553390593 \
  --n-ph-max 2 --adapt-pool full_meta --phase3-runtime-split-mode off --adapt-exact-gs-reference-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json \
  --adapt-max-depth 30 --adapt-maxiter 200 --adapt-final-refit-maxiter 200 --adapt-final-full-refit true \
  --phase0-pilot-max-records 96 --phase1-shortlist-size 24 --phase2-shortlist-fraction 0.25 \
  --phase2-shortlist-size 12 --adapt-parallel-gradient-workers 2 --adapt-beam-parent-workers 2 \
  --adapt-spsa-parallel-evaluations 2 --static-route-id route_a --static-meta-feature-profile \
  paper_i_production_v1 --adapt-continuation-mode phase3_v1 --phase2-novelty-mode collective_span_v1 \
  --hardware-resolution-mode ideal --phase3-selector-policy algebraic_nested_v1 --phase3-selector-geometry-mode \
  reduced --phase3-novelty-ablation-mode off --phase3-window-relaxation-mode reduced --phase2-enable-batching \
  --phase3-enable-batching --phase3-batch-selection-mode reduced_plane --phase3-batch-prefilter-mode off \
  --phase1-prune-enabled --phase1-prune-policy recoverability_ladder_v1 --phase1-prune-mode both \
  --phase1-prune-amplitude-witness-optional --phase3-symmetry-mitigation-mode off --output-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__intermediate_weak__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/json/result.json \
  --adapt-current-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__intermediate_weak__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/current.json \
  --adapt-current-json-every-depth 1 --adapt-current-json-keep-history-tail 100 --skip-pdf \
  --adapt-segment-target-depth 30 --adapt-segment-max-new-admissions 30 --adapt-grad-floor -1 \
  --adapt-schur-warm-start-mode append-prune --adapt-child-pool-expansion-mode off --shared-pauli-pool-mode \
  shared_pauli_child_sets_v1 --shared-pauli-pool-symmetry-policy hard_guard --shared-pauli-pool-max-subset-size \
  3 --adapt-pool-class-filter-json agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json

```

### strong-weak / ROTOSOLVE / maxiter 200
```bash
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -u -m pipelines.static_adapt.adapt_pipeline \
  --dv 0.0 --boson-encoding binary --ordering blocked --boundary open --term-order sorted --adapt-eps-grad 5e-7 \
  --adapt-eps-energy 1e-9 --adapt-seed 7 --adapt-state-backend compiled --adapt-reopt-policy windowed \
  --adapt-full-refit-every 8 --adapt-beam-live-branches 3 --adapt-beam-children-per-parent 2 \
  --adapt-beam-terminated-keep 3 --phase1-prune-max-candidates 6 --phase1-prune-max-regression 1e-8 \
  --phase1-probe-max-positions 999999 --phase1-trough-margin-ratio 1.0 --phase2-lambda-H 1e-6 --phase2-gamma-N \
  1.0 --phase2-w-depth 0.2 --phase2-w-group 0.15 --phase2-w-optdim 0.1 --phase2-w-reuse 0.1 --phase2-w-lifetime \
  0.05 --phase2-frontier-ratio 0.9 --phase3-frontier-ratio 0.9 --phase2-batch-target-size 8 \
  --phase2-batch-size-cap 16 --phase3-enable-rescue --phase3-lifetime-cost-mode off --adapt-drop-floor -1 \
  --adapt-drop-patience 0 --adapt-drop-min-depth 0 --adapt-no-repeats --phase3-backend-cost-mode \
  marrakesh_graph_span_v1 --phase3-backend-name FakeMarrakesh --phase3-backend-transpile-seed 7 \
  --phase3-backend-optimization-level 1 --phase1-maturity-cap-min 12 --phase1-maturity-cap-max 32 \
  --phase2-maturity-cap-min 8 --phase2-maturity-cap-max 24 --phase3-maturity-cap-min 4 --phase3-maturity-cap-max \
  16 --phase-maturity-shot-min 1 --phase-maturity-shot-max 4 --phase1-maturity-shot-cap 2 \
  --phase2-maturity-shot-cap 4 --phase3-maturity-shot-cap 4 --phase1-prune-collapse-peak-abs-min 2e-3 \
  --phase1-prune-collapse-current-abs-max 5e-4 --phase1-prune-collapse-ratio 0.2 \
  --phase1-prune-collapse-min-abs-drop 2e-3 --phase1-prune-collapse-min-observations 4 --adapt-window-size \
  999999 --adapt-window-topk 999999 --phase3-geometry-window-size 0 --phase3-backend-w-2q 1.0 \
  --phase3-backend-w-depth 0.25 --phase3-backend-w-size 0.01 --phase1-prune-fraction 0.4101910583864897 \
  --phase2-batch-near-degenerate-ratio 0.9982411735035968 --phase3-batch-near-degenerate-ratio \
  0.9982411735035968 --phase2-batch-rank-rel-tol 0.00013662376421438911 --phase3-batch-rank-rel-tol \
  0.00013662376421438911 --phase2-batch-additivity-tol 0.6663130343903237 --phase3-batch-additivity-tol \
  0.6663130343903237 --phase2-w-shot 0.15 --phase2-rho 0.5 --adapt-inner-optimizer ROTOSOLVE --adapt-spsa-a 0.1 \
  --adapt-spsa-c 0.02 --adapt-spsa-alpha 0.602 --adapt-spsa-gamma 0.101 --adapt-spsa-A 5.0 --adapt-spsa-avg-last \
  0 --adapt-spsa-eval-repeats 1 --adapt-spsa-eval-agg mean --adapt-spsa-callback-every 5 \
  --adapt-spsa-progress-every-s 30.0 --problem hh --L 2 --t 1 --u 8 --omega0 1 --g-ep 0.353553390593 --n-ph-max \
  2 --adapt-pool full_meta --phase3-runtime-split-mode off --adapt-exact-gs-reference-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json \
  --adapt-max-depth 30 --adapt-maxiter 200 --adapt-final-refit-maxiter 200 --adapt-final-full-refit true \
  --phase0-pilot-max-records 96 --phase1-shortlist-size 24 --phase2-shortlist-fraction 0.25 \
  --phase2-shortlist-size 12 --adapt-parallel-gradient-workers 2 --adapt-beam-parent-workers 2 \
  --adapt-spsa-parallel-evaluations 2 --static-route-id route_a --static-meta-feature-profile \
  paper_i_production_v1 --adapt-continuation-mode phase3_v1 --phase2-novelty-mode collective_span_v1 \
  --hardware-resolution-mode ideal --phase3-selector-policy algebraic_nested_v1 --phase3-selector-geometry-mode \
  reduced --phase3-novelty-ablation-mode off --phase3-window-relaxation-mode reduced --phase2-enable-batching \
  --phase3-enable-batching --phase3-batch-selection-mode reduced_plane --phase3-batch-prefilter-mode off \
  --phase1-prune-enabled --phase1-prune-policy recoverability_ladder_v1 --phase1-prune-mode both \
  --phase1-prune-amplitude-witness-optional --phase3-symmetry-mitigation-mode off --output-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__strong_weak__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/json/result.json \
  --adapt-current-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__strong_weak__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/current.json \
  --adapt-current-json-every-depth 1 --adapt-current-json-keep-history-tail 100 --skip-pdf \
  --adapt-segment-target-depth 30 --adapt-segment-max-new-admissions 30 --adapt-grad-floor -1 \
  --adapt-schur-warm-start-mode append-prune --adapt-child-pool-expansion-mode off --shared-pauli-pool-mode \
  shared_pauli_child_sets_v1 --shared-pauli-pool-symmetry-policy hard_guard --shared-pauli-pool-max-subset-size \
  3 --adapt-pool-class-filter-json agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json

```

### weak-strong / ROTOSOLVE / maxiter 200
```bash
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -u -m pipelines.static_adapt.adapt_pipeline \
  --dv 0.0 --boson-encoding binary --ordering blocked --boundary open --term-order sorted --adapt-eps-grad 5e-7 \
  --adapt-eps-energy 1e-9 --adapt-seed 7 --adapt-state-backend compiled --adapt-reopt-policy windowed \
  --adapt-full-refit-every 8 --adapt-beam-live-branches 3 --adapt-beam-children-per-parent 2 \
  --adapt-beam-terminated-keep 3 --phase1-prune-max-candidates 6 --phase1-prune-max-regression 1e-8 \
  --phase1-probe-max-positions 999999 --phase1-trough-margin-ratio 1.0 --phase2-lambda-H 1e-6 --phase2-rho 0.25 \
  --phase2-gamma-N 1.0 --phase2-w-depth 0.2 --phase2-w-group 0.15 --phase2-w-optdim 0.1 --phase2-w-reuse 0.1 \
  --phase2-w-lifetime 0.05 --phase2-frontier-ratio 0.9 --phase3-frontier-ratio 0.9 --phase2-batch-target-size 8 \
  --phase2-batch-size-cap 16 --phase3-enable-rescue --phase3-lifetime-cost-mode off --adapt-drop-floor -1 \
  --adapt-drop-patience 0 --adapt-drop-min-depth 0 --phase3-backend-cost-mode marrakesh_graph_span_v1 \
  --phase3-backend-name FakeMarrakesh --phase3-backend-transpile-seed 7 --phase3-backend-optimization-level 1 \
  --phase1-maturity-cap-min 8 --phase1-maturity-cap-max 24 --phase2-maturity-cap-min 6 --phase2-maturity-cap-max \
  16 --phase3-maturity-cap-min 4 --phase3-maturity-cap-max 12 --phase-maturity-shot-min 1 \
  --phase-maturity-shot-max 1 --phase1-maturity-shot-cap 1 --phase2-maturity-shot-cap 1 \
  --phase3-maturity-shot-cap 1 --adapt-window-size 4 --adapt-window-topk 4 --phase3-geometry-window-size 4 \
  --phase3-backend-w-2q 1.0 --phase3-backend-w-depth 0.25 --phase3-backend-w-size 0.01 --phase1-prune-fraction \
  0.33922934316592934 --phase2-batch-near-degenerate-ratio 0.98 --phase3-batch-near-degenerate-ratio 0.98 \
  --phase2-batch-rank-rel-tol 1.909930091607197e-05 --phase3-batch-rank-rel-tol 1.909930091607197e-05 \
  --phase2-batch-additivity-tol 0.09993123296803053 --phase3-batch-additivity-tol 0.09993123296803053 \
  --phase2-w-shot 0.02 --adapt-inner-optimizer ROTOSOLVE --adapt-spsa-a 0.1 --adapt-spsa-c 0.02 \
  --adapt-spsa-alpha 0.602 --adapt-spsa-gamma 0.101 --adapt-spsa-A 5.0 --adapt-spsa-avg-last 0 \
  --adapt-spsa-eval-repeats 1 --adapt-spsa-eval-agg mean --adapt-spsa-callback-every 5 \
  --adapt-spsa-progress-every-s 30.0 --problem hh --L 2 --t 1 --u 0.25 --omega0 1 --g-ep 0.790569415042 \
  --n-ph-max 4 --adapt-pool full_meta --phase3-runtime-split-mode off --adapt-exact-gs-reference-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json \
  --adapt-max-depth 30 --adapt-maxiter 200 --adapt-final-refit-maxiter 200 --adapt-final-full-refit true \
  --phase0-pilot-max-records 96 --phase1-shortlist-size 24 --phase2-shortlist-fraction 0.25 \
  --phase2-shortlist-size 12 --adapt-parallel-gradient-workers 4 --adapt-beam-parent-workers 4 \
  --adapt-spsa-parallel-evaluations 4 --static-route-id route_a --static-meta-feature-profile \
  paper_i_production_v1 --adapt-continuation-mode phase3_v1 --phase2-novelty-mode collective_span_v1 \
  --hardware-resolution-mode ideal --phase3-selector-policy algebraic_nested_v1 --phase3-selector-geometry-mode \
  reduced --phase3-novelty-ablation-mode off --phase3-window-relaxation-mode reduced --phase2-enable-batching \
  --phase3-enable-batching --phase3-batch-selection-mode reduced_plane --phase3-batch-prefilter-mode off \
  --phase1-prune-enabled --phase1-prune-policy recoverability_ladder_v1 --phase1-prune-mode both \
  --phase1-prune-amplitude-witness-optional --phase3-symmetry-mitigation-mode off --output-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__weak_strong__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/json/result.json \
  --adapt-current-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__weak_strong__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/current.json \
  --adapt-current-json-every-depth 1 --adapt-current-json-keep-history-tail 100 --skip-pdf \
  --adapt-segment-target-depth 30 --adapt-segment-max-new-admissions 30 --adapt-grad-floor -1 \
  --adapt-schur-warm-start-mode append-prune --adapt-child-pool-expansion-mode off --shared-pauli-pool-mode \
  shared_pauli_child_sets_v1 --shared-pauli-pool-symmetry-policy hard_guard --shared-pauli-pool-max-subset-size \
  3 --adapt-pool-class-filter-json agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json

```

### intermediate-strong / ROTOSOLVE / maxiter 200
```bash
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -u -m pipelines.static_adapt.adapt_pipeline \
  --dv 0.0 --boson-encoding binary --ordering blocked --boundary open --term-order sorted --adapt-eps-grad 5e-7 \
  --adapt-eps-energy 1e-9 --adapt-seed 7 --adapt-state-backend compiled --adapt-reopt-policy windowed \
  --adapt-full-refit-every 8 --adapt-beam-live-branches 3 --adapt-beam-children-per-parent 2 \
  --adapt-beam-terminated-keep 3 --phase1-prune-max-candidates 6 --phase1-prune-max-regression 1e-8 \
  --phase1-probe-max-positions 999999 --phase1-trough-margin-ratio 1.0 --phase2-lambda-H 1e-6 --phase2-rho 0.25 \
  --phase2-gamma-N 1.0 --phase2-w-depth 0.2 --phase2-w-group 0.15 --phase2-w-optdim 0.1 --phase2-w-reuse 0.1 \
  --phase2-w-lifetime 0.05 --phase2-frontier-ratio 0.9 --phase3-frontier-ratio 0.9 --phase2-batch-target-size 8 \
  --phase2-batch-size-cap 16 --phase3-enable-rescue --phase3-lifetime-cost-mode off --adapt-drop-floor -1 \
  --adapt-drop-patience 0 --adapt-drop-min-depth 0 --phase3-backend-cost-mode marrakesh_graph_span_v1 \
  --phase3-backend-name FakeMarrakesh --phase3-backend-transpile-seed 7 --phase3-backend-optimization-level 1 \
  --phase1-maturity-cap-min 8 --phase1-maturity-cap-max 24 --phase2-maturity-cap-min 6 --phase2-maturity-cap-max \
  16 --phase3-maturity-cap-min 4 --phase3-maturity-cap-max 12 --phase-maturity-shot-min 1 \
  --phase-maturity-shot-max 1 --phase1-maturity-shot-cap 1 --phase2-maturity-shot-cap 1 \
  --phase3-maturity-shot-cap 1 --adapt-window-size 4 --adapt-window-topk 4 --phase3-geometry-window-size 4 \
  --phase3-backend-w-2q 1.0 --phase3-backend-w-depth 0.25 --phase3-backend-w-size 0.01 --phase1-prune-fraction \
  0.33922934316592934 --phase2-batch-near-degenerate-ratio 0.98 --phase3-batch-near-degenerate-ratio 0.98 \
  --phase2-batch-rank-rel-tol 1.909930091607197e-05 --phase3-batch-rank-rel-tol 1.909930091607197e-05 \
  --phase2-batch-additivity-tol 0.09993123296803053 --phase3-batch-additivity-tol 0.09993123296803053 \
  --phase2-w-shot 0.02 --adapt-inner-optimizer ROTOSOLVE --adapt-spsa-a 0.1 --adapt-spsa-c 0.02 \
  --adapt-spsa-alpha 0.602 --adapt-spsa-gamma 0.101 --adapt-spsa-A 5.0 --adapt-spsa-avg-last 0 \
  --adapt-spsa-eval-repeats 1 --adapt-spsa-eval-agg mean --adapt-spsa-callback-every 5 \
  --adapt-spsa-progress-every-s 30.0 --problem hh --L 2 --t 1 --u 1.25 --omega0 1 --g-ep 0.790569415042 \
  --n-ph-max 4 --adapt-pool full_meta --phase3-runtime-split-mode off --adapt-exact-gs-reference-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json \
  --adapt-max-depth 30 --adapt-maxiter 200 --adapt-final-refit-maxiter 200 --adapt-final-full-refit true \
  --phase0-pilot-max-records 96 --phase1-shortlist-size 24 --phase2-shortlist-fraction 0.25 \
  --phase2-shortlist-size 12 --adapt-parallel-gradient-workers 4 --adapt-beam-parent-workers 4 \
  --adapt-spsa-parallel-evaluations 4 --static-route-id route_a --static-meta-feature-profile \
  paper_i_production_v1 --adapt-continuation-mode phase3_v1 --phase2-novelty-mode collective_span_v1 \
  --hardware-resolution-mode ideal --phase3-selector-policy algebraic_nested_v1 --phase3-selector-geometry-mode \
  reduced --phase3-novelty-ablation-mode off --phase3-window-relaxation-mode reduced --phase2-enable-batching \
  --phase3-enable-batching --phase3-batch-selection-mode reduced_plane --phase3-batch-prefilter-mode off \
  --phase1-prune-enabled --phase1-prune-policy recoverability_ladder_v1 --phase1-prune-mode both \
  --phase1-prune-amplitude-witness-optional --phase3-symmetry-mitigation-mode off --output-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__intermediate_strong__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/json/result.json \
  --adapt-current-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__intermediate_strong__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/current.json \
  --adapt-current-json-every-depth 1 --adapt-current-json-keep-history-tail 100 --skip-pdf \
  --adapt-segment-target-depth 30 --adapt-segment-max-new-admissions 30 --adapt-grad-floor -1 \
  --adapt-schur-warm-start-mode append-prune --adapt-child-pool-expansion-mode off --shared-pauli-pool-mode \
  shared_pauli_child_sets_v1 --shared-pauli-pool-symmetry-policy hard_guard --shared-pauli-pool-max-subset-size \
  3 --adapt-pool-class-filter-json agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json

```

### strong-strong / ROTOSOLVE / maxiter 200
```bash
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -u -m pipelines.static_adapt.adapt_pipeline \
  --dv 0.0 --boson-encoding binary --ordering blocked --boundary open --term-order sorted --adapt-eps-grad 5e-7 \
  --adapt-eps-energy 1e-9 --adapt-seed 7 --adapt-state-backend compiled --adapt-reopt-policy windowed \
  --adapt-full-refit-every 8 --adapt-beam-live-branches 3 --adapt-beam-children-per-parent 2 \
  --adapt-beam-terminated-keep 3 --phase1-prune-max-candidates 6 --phase1-prune-max-regression 1e-8 \
  --phase1-probe-max-positions 999999 --phase1-trough-margin-ratio 1.0 --phase2-lambda-H 1e-6 --phase2-gamma-N \
  1.0 --phase2-w-depth 0.2 --phase2-w-group 0.15 --phase2-w-optdim 0.1 --phase2-w-reuse 0.1 --phase2-w-lifetime \
  0.05 --phase2-frontier-ratio 0.9 --phase3-frontier-ratio 0.9 --phase2-batch-target-size 8 \
  --phase2-batch-size-cap 16 --phase3-enable-rescue --phase3-lifetime-cost-mode off --adapt-drop-floor -1 \
  --adapt-drop-patience 0 --adapt-drop-min-depth 0 --phase3-backend-cost-mode marrakesh_graph_span_v1 \
  --phase3-backend-name FakeMarrakesh --phase3-backend-transpile-seed 7 --phase3-backend-optimization-level 1 \
  --phase1-maturity-cap-min 24 --phase1-maturity-cap-max 64 --phase2-maturity-cap-min 12 \
  --phase2-maturity-cap-max 48 --phase3-maturity-cap-min 8 --phase3-maturity-cap-max 32 \
  --phase-maturity-shot-min 1 --phase-maturity-shot-max 4 --phase1-maturity-shot-cap 2 \
  --phase2-maturity-shot-cap 4 --phase3-maturity-shot-cap 4 --adapt-window-size 16 --adapt-window-topk 16 \
  --phase3-geometry-window-size 16 --phase3-backend-w-2q 1.0 --phase3-backend-w-depth 0.1 \
  --phase3-backend-w-size 0.01 --phase1-prune-fraction 0.1930961457788297 --phase2-batch-near-degenerate-ratio \
  0.914354284671342 --phase3-batch-near-degenerate-ratio 0.914354284671342 --phase2-batch-rank-rel-tol \
  7.703203666118798e-07 --phase3-batch-rank-rel-tol 7.703203666118798e-07 --phase2-batch-additivity-tol \
  0.010276490515218235 --phase3-batch-additivity-tol 0.010276490515218235 --phase2-w-shot 0.08 --phase2-rho 0.5 \
  --adapt-inner-optimizer ROTOSOLVE --adapt-spsa-a 0.1 --adapt-spsa-c 0.02 --adapt-spsa-alpha 0.602 \
  --adapt-spsa-gamma 0.101 --adapt-spsa-A 5.0 --adapt-spsa-avg-last 0 --adapt-spsa-eval-repeats 1 \
  --adapt-spsa-eval-agg mean --adapt-spsa-callback-every 5 --adapt-spsa-progress-every-s 30.0 --problem hh --L 2 \
  --t 1 --u 8 --omega0 1 --g-ep 0.790569415042 --n-ph-max 4 --adapt-pool full_meta --phase3-runtime-split-mode \
  off --adapt-exact-gs-reference-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json \
  --adapt-max-depth 30 --adapt-maxiter 200 --adapt-final-refit-maxiter 200 --adapt-final-full-refit true \
  --phase0-pilot-max-records 96 --phase1-shortlist-size 24 --phase2-shortlist-fraction 0.25 \
  --phase2-shortlist-size 12 --adapt-parallel-gradient-workers 2 --adapt-beam-parent-workers 2 \
  --adapt-spsa-parallel-evaluations 2 --adapt-resume-scaffold-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/raw_outputs/chtc_retrievals/paper_i_u8_hh_strong_strong_snake_current_best/paper_i_u8_hh_ss_v2_7702629_2_20260614T180758Z/trial_0001_current.json \
  --adapt-resume-mode scaffold_v1 --adapt-segment-id u8_ss_resume_from_k24_20260616 --adapt-segment-target-depth \
  30 --adapt-segment-max-new-admissions 30 --adapt-resume-compile-smoke required --static-route-id route_a \
  --static-meta-feature-profile paper_i_production_v1 --adapt-continuation-mode phase3_v1 --phase2-novelty-mode \
  collective_span_v1 --hardware-resolution-mode ideal --phase3-selector-policy algebraic_nested_v1 \
  --phase3-selector-geometry-mode reduced --phase3-novelty-ablation-mode off --phase3-window-relaxation-mode \
  reduced --phase2-enable-batching --phase3-enable-batching --phase3-batch-selection-mode reduced_plane \
  --phase3-batch-prefilter-mode off --phase1-prune-enabled --phase1-prune-policy recoverability_ladder_v1 \
  --phase1-prune-mode both --phase1-prune-amplitude-witness-optional --phase3-symmetry-mitigation-mode off \
  --output-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__strong_strong__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/json/result.json \
  --adapt-current-json \
  /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/tmp/snake_settings_md_audit2/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1/paper_i_hh_shared_pool_snake_minushva_rotosolve_all_regimes_20260627_v1__strong_strong__snake__native_forced__rotosolve200__depth30_noearlystop__sharedpool/current.json \
  --adapt-current-json-every-depth 1 --adapt-current-json-keep-history-tail 100 --skip-pdf --adapt-grad-floor -1 \
  --adapt-schur-warm-start-mode append-prune --adapt-child-pool-expansion-mode off --shared-pauli-pool-mode \
  shared_pauli_child_sets_v1 --shared-pauli-pool-symmetry-policy hard_guard --shared-pauli-pool-max-subset-size \
  3 --adapt-pool-class-filter-json agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json

```

## Notes For Future Canonicalization
- Use the regime-specific table, not the repeated raw command blocks, when choosing common SNAKE settings.
- The most important non-common numerical knobs are `--phase2-rho` [impl-v2], `--phase1-prune-fraction` [impl-v2], `--phase2-w-shot` [impl-v2], the batch tolerance triple, maturity caps, and window sizes.
- The repair rows replace the older runtime-split child route with the shared Pauli-child pool overlay and the explicit `full_meta_minus_hva` class filter.
- The source anchors used SPSA maxiter 800; the current repair rows use optimizer-specific maxiter 200, plus maxiter 500 only for the strong-Holstein diagnostic overlay.
- Strong-strong is special in the source anchor because it had resume metadata; the current repair overlay still forces target depth 30 and no early drop.
