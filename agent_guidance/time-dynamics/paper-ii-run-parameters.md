# Paper II — run commands and parameter settings

**Generated file — do not edit.** Regenerate with:

```bash
PYTHONPATH=. python3 -m pipelines.time_dynamics.write_paper_ii_manifest
```

Source commit at generation: `77aeeba84f47ed2f5372abdb3907f3361d55c1ac`

Every Paper-II run is composed from the registry in
`pipelines/time_dynamics/paper_ii_runs.py`. A run is one choice from each
axis below; nothing else varies, and `test/test_paper_ii_runs.py` asserts
that every registered command parses against the runner's live CLI.

## Axes

### Structural arms

| arm | meaning |
|---|---|
| exchange | The paper's route: deletions and positioned insertions compete as one atomic patch. Ray tolerance 2e-3 rather than the 5e-2 default, which admits cumulative structural damage over a long horizon. |
| append_only | Insert-face restriction of generalized exchange; isolates what the deletion component buys without invoking another selector. |
| avqds | AVQDS decision RULE on this route's numerical stack (shared geometry, inverse, solve repair, integrator, pool). Isolates structure from numerics; it is NOT the published method -- use avqds_published for that. Uncapped: the rule appends until L^2 < cut. |
| avqds_published | Yao et al. with the paper's own numerics: Euler, Tikhonov xi=1e-6, and the published parameter-controlled step (delta_theta_max=5e-3). AVQDS is NOT a fixed-step method -- it adapts dt so no parameter moves more than that budget, which the source calls its stabilization mechanism. |

### Insertion gates

| gate | meaning |
|---|---|
| residual_1e-4 | This route's historical normalized gate. 1e-4 measured best at t=2 (7.8e-4 vs 3.8e-3 at 2e-3); 1e-5 buys nothing and costs parameters. |
| mclachlan_l2_1e-3 | The published AVQDS append condition: absolute McLachlan distance with greedy repeat inside one checkpoint. Adopted because the normalized ratio is small precisely while the state is still accurate and so defers growth past the cheap early window. |

### Inner numerical profiles

| id | integrator | ridge | damping | pinv rcond |
|---|---|---|---|---|
| euler_ridge1e-7 | euler | 1e-07 | 0.0 | 1e-10 |
| euler_ridge1e-6 | euler | 1e-06 | 0.0 | 1e-10 |
| rk4_ridge1e-7 | rk4 | 1e-07 | 0.0 | 1e-10 |

### Step control

| id | meaning |
|---|---|
| delta_theta_5e-3 | Published AVQDS Euler control law, at the source's own value. |
| state_motion_1e-2 | This route's control law. The subdivision budget is 10 because at 4 a step could exhaust it, fail to cure a violation, and advance anyway, taking a measured error from 3.8e-3 to 2.1e-1. |
| state_motion_1e-2_plus_parameter_5e-3 | Composed controller with both the tangent-state and maximum single-parameter step bounds active. |

### Drives

| drive | A | omega |
|---|---|---|
| fastweak | 0.6 | 3.0 |
| slowstrong | 1.2 | 1.0 |
| weakslow | 0.3 | 1.0 |
| strongfast | 2.4 | 3.0 |
| midres | 1.2 | 2.0 |
| fastfast | 0.6 | 6.0 |
| res_w2p5 | 0.6 | 2.5 |
| res_w2p75 | 0.6 | 2.75 |
| res_w3p25 | 0.6 | 3.25 |
| res_w3p5 | 0.6 | 3.5 |
| res_w4 | 0.6 | 4.0 |
| res_a0p3 | 0.3 | 3.0 |
| res_a1p2 | 1.2 | 3.0 |

### Horizons

| horizon | t_final | checkpoints |
|---|---|---|
| smoke | 0.5 | 13 |
| t2 | 2.0 | 51 |
| t5 | 5.0 | 126 |
| t10 | 10.0 | 251 |
| t20 | 20.0 | 501 |

### Regimes

| regime | seed | n_ph | status |
|---|---|---|---|
| hh_snake_nph1 | chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json | 1 | built |
| hh_fixedvqe_nph3 | chtc/paper_ii_production_v2/input/seeds/hh_fixedvqe_nph3.json | 3 | built |
| weak_weak | chtc/paper_ii_regime_seeds_v1/input/seeds/weak_weak.json | 3 | **MISSING** |
| intermediate_weak | chtc/paper_ii_regime_seeds_v1/input/seeds/intermediate_weak.json | 3 | **MISSING** |
| strong_weak_u8 | chtc/paper_ii_regime_seeds_v1/input/seeds/strong_weak_u8.json | 3 | **MISSING** |
| weak_strong | chtc/paper_ii_regime_seeds_v1/input/seeds/weak_strong.json | 7 | **MISSING** |
| intermediate_strong | chtc/paper_ii_regime_seeds_v1/input/seeds/intermediate_strong.json | 7 | **MISSING** |
| strong_strong_u8 | chtc/paper_ii_regime_seeds_v1/input/seeds/strong_strong_u8.json | 7 | **MISSING** |

## Commands

### `exchange` — this work

```bash
PYTHONPATH=. \
  PYTHONUNBUFFERED=1 \
  python3 \
  pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py \
  --artifact-json \
  chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json \
  --t-final \
  10.0 \
  --num-times \
  251 \
  --enable-drive \
  --drive-A \
  2.4 \
  --drive-omega \
  3.0 \
  --integrator \
  rk4 \
  --ridge-lambda \
  1e-07 \
  --solve-damping \
  0.0 \
  --pinv-rcond \
  1e-10 \
  --solve-repair \
  --solve-repair-profile \
  minimal \
  --solve-repair-state-motion-l2-step-max \
  1.0e-2 \
  --solve-repair-kink-eta-max \
  5.0e-3 \
  --solve-repair-max-local-subdivisions \
  10 \
  --max-structural-pool-size \
  128 \
  --no-append-schur-condition-gate \
  --max-joint-patch-evaluations \
  50000 \
  --max-certification-attempts-per-level \
  12 \
  --max-certification-attempts-per-deletion-branch \
  2 \
  --max-insertion-batch-size \
  1 \
  --insertion-gate-mode \
  mclachlan_l2 \
  --insertion-l2-cut \
  1.0e-3 \
  --debt-policy \
  drift_ranked \
  --prune-history-lambda \
  0.0 \
  --prune-target-policy \
  all_active \
  --prune-ray-distance-tol \
  2.0e-3 \
  --progress-log-every \
  1 \
  --output-json \
  output/<batch>/strongfast_exchange/run.json
```

### `append_only` — growth-only ablation of this work

```bash
PYTHONPATH=. \
  PYTHONUNBUFFERED=1 \
  python3 \
  pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py \
  --artifact-json \
  chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json \
  --t-final \
  10.0 \
  --num-times \
  251 \
  --enable-drive \
  --drive-A \
  2.4 \
  --drive-omega \
  3.0 \
  --integrator \
  rk4 \
  --ridge-lambda \
  1e-07 \
  --solve-damping \
  0.0 \
  --pinv-rcond \
  1e-10 \
  --solve-repair \
  --solve-repair-profile \
  minimal \
  --solve-repair-state-motion-l2-step-max \
  1.0e-2 \
  --solve-repair-kink-eta-max \
  5.0e-3 \
  --solve-repair-max-local-subdivisions \
  10 \
  --max-structural-pool-size \
  128 \
  --no-append-schur-condition-gate \
  --max-joint-patch-evaluations \
  50000 \
  --max-certification-attempts-per-level \
  12 \
  --max-certification-attempts-per-deletion-branch \
  2 \
  --max-insertion-batch-size \
  1 \
  --insertion-gate-mode \
  mclachlan_l2 \
  --insertion-l2-cut \
  1.0e-3 \
  --no-exchange-deletions \
  --debt-policy \
  insertion_only \
  --progress-log-every \
  1 \
  --output-json \
  output/<batch>/strongfast_append_only/run.json
```

### `avqds` — comparator rule on shared numerics

```bash
PYTHONPATH=. \
  PYTHONUNBUFFERED=1 \
  python3 \
  pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py \
  --artifact-json \
  chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json \
  --t-final \
  10.0 \
  --num-times \
  251 \
  --enable-drive \
  --drive-A \
  2.4 \
  --drive-omega \
  3.0 \
  --integrator \
  rk4 \
  --ridge-lambda \
  1e-07 \
  --solve-damping \
  0.0 \
  --pinv-rcond \
  1e-10 \
  --solve-repair \
  --solve-repair-profile \
  minimal \
  --solve-repair-state-motion-l2-step-max \
  1.0e-2 \
  --solve-repair-kink-eta-max \
  5.0e-3 \
  --solve-repair-max-local-subdivisions \
  10 \
  --max-structural-pool-size \
  128 \
  --no-append-schur-condition-gate \
  --max-joint-patch-evaluations \
  50000 \
  --max-certification-attempts-per-level \
  12 \
  --max-certification-attempts-per-deletion-branch \
  2 \
  --max-insertion-batch-size \
  1 \
  --dynamics-policy \
  avqds \
  --avqds-l2-cut \
  1.0e-3 \
  --progress-log-every \
  1 \
  --output-json \
  output/<batch>/strongfast_avqds/run.json
```

### `avqds_published` — comparator as published

```bash
PYTHONPATH=. \
  PYTHONUNBUFFERED=1 \
  python3 \
  pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py \
  --artifact-json \
  chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json \
  --t-final \
  10.0 \
  --num-times \
  251 \
  --enable-drive \
  --drive-A \
  2.4 \
  --drive-omega \
  3.0 \
  --integrator \
  euler \
  --ridge-lambda \
  1e-06 \
  --solve-damping \
  0.0 \
  --pinv-rcond \
  1e-10 \
  --avqds-delta-theta-max \
  5.0e-3 \
  --no-solve-repair \
  --max-structural-pool-size \
  128 \
  --no-append-schur-condition-gate \
  --max-joint-patch-evaluations \
  50000 \
  --max-certification-attempts-per-level \
  12 \
  --max-certification-attempts-per-deletion-branch \
  2 \
  --max-insertion-batch-size \
  1 \
  --dynamics-policy \
  avqds \
  --avqds-l2-cut \
  1.0e-3 \
  --no-certification-refit \
  --progress-log-every \
  1 \
  --output-json \
  output/<batch>/strongfast_avqds_published/run.json
```

## Every effective parameter

Registry-set values, then the runner defaults they sit on. A value marked *default* is not stated anywhere in the registry; it is what the runner uses when nothing overrides it.


<details><summary><b>exchange</b> — 108 parameters (11 set by registry)</summary>

| parameter | value | source |
|---|---|---|
| append_cost_alpha | 1.0 | default |
| append_cost_lambda_1q | 0.025 | default |
| append_cost_lambda_2q | 0.05 | default |
| append_cost_lambda_d | 0.05 | default |
| append_cost_lambda_shot | 0.02 | default |
| append_cost_lambda_theta | 0.0 | default |
| append_cost_normalization_mode | family_robust_v1 | default |
| append_cost_scale_floor | 1e-12 | default |
| append_min_time | 0.0 | default |
| append_occurrence_policy | layer_reuse | default |
| append_schur_condition_gate | False | registry |
| append_schur_max_condition_number | 1000000000000.0 | default |
| avqds_delta_theta_max | None | default |
| avqds_l2_cut | 0.001 | default |
| avqds_max_appends_per_checkpoint | None | default |
| certification_refit | True | default |
| certification_refit_max_iterations | 15 | default |
| certification_refit_trust_radius | 0.6 | default |
| debt_policy | drift_ranked | default |
| diagnostic_append_pool_mode | none | default |
| drive_A | 2.4 | registry |
| drive_aligned_ansatz | True | default |
| drive_custom_weights | None | default |
| drive_include_identity | False | default |
| drive_n_sites | None | default |
| drive_omega | 3.0 | registry |
| drive_ordering | None | default |
| drive_pattern | staggered | default |
| drive_phi | 0.0 | default |
| drive_t0 | 0.0 | default |
| drive_tbar | 1.0 | default |
| drive_time_sampling | midpoint | default |
| dynamics_policy | exchange | default |
| enable_drive | True | registry |
| eps_loss | 1e-14 | default |
| escalation_accumulated_drift_threshold | None | default |
| exchange_deletions | True | default |
| fail_on_unsupported_steps | False | default |
| fallback_family | full_meta | default |
| generator_family | match_adapt | default |
| insertion_gate_mode | mclachlan_l2 | registry |
| insertion_l2_cut | 0.001 | default |
| integrator | rk4 | default |
| interaction_frontier_widths | None | default |
| loader_mode | None | default |
| max_append_candidates | 8 | default |
| max_certification_attempts_per_deletion_branch | 2 | default |
| max_certification_attempts_per_level | 12 | default |
| max_insertion_batch_size | 1 | default |
| max_insertion_rounds_per_checkpoint | 12 | default |
| max_joint_patch_evaluations | 50000 | default |
| max_structural_pool_size | 128 | registry |
| min_logical_parameter_count | 1 | default |
| min_runtime_parameter_count | 1 | default |
| normalized_candidate_pool_profile | none | default |
| num_times | 251 | registry |
| parameterization_mode | per_pauli_term | default |
| patch_utility_delta_weight | 1.0 | default |
| pinv_rcond | 1e-10 | default |
| progress_log_events | True | default |
| progress_log_every | 1 | registry |
| prune_condition_lambda_kappa_dam | 0.0 | default |
| prune_condition_lambda_kappa_rel | 0.0 | default |
| prune_cooldown_steps | 2 | default |
| prune_cost_alpha | 1.0 | default |
| prune_history_lambda | 0.0 | default |
| prune_history_window | 3 | default |
| prune_patch_smoothness_eta_max | 0.001 | default |
| prune_ray_distance_tol | 0.002 | default |
| prune_target_policy | all_active | default |
| record_statevector | False | default |
| reference_energy_atol | 1e-12 | default |
| reference_energy_json | None | default |
| replay_candidate_pool_mode | None | default |
| require_complete_candidate_pool | False | default |
| residual_ratio_threshold | 0.02 | default |
| resume_from_run_json | None | default |
| ridge_lambda | 1e-07 | default |
| seed_reference_energy_atol | 1e-12 | default |
| seed_reference_energy_json | None | default |
| solve_damping | 0.0 | default |
| solve_repair | True | default |
| solve_repair_condition_number_fail | None | default |
| solve_repair_condition_number_max | 1000000.0 | default |
| solve_repair_damping_ladder | 0 | default |
| solve_repair_kink_eta_max | 0.005 | registry |
| solve_repair_local_subdivision | True | default |
| solve_repair_local_subdivision_factor | 2 | default |
| solve_repair_max_local_subdivisions | 10 | default |
| solve_repair_min_local_dt | 1e-06 | default |
| solve_repair_parameter_step_max | None | default |
| solve_repair_pinv_rcond_ladder | 1e-10,1e-11,1e-12,1e-9,1e-8,1e-7 | default |
| solve_repair_profile | minimal | default |
| solve_repair_release_kink_severity_scale | 4.0 | default |
| solve_repair_release_kink_threshold_scale | 0.5 | default |
| solve_repair_release_patience_max | 5 | default |
| solve_repair_release_patience_min | 1 | default |
| solve_repair_rho_num_max | 0.01 | default |
| solve_repair_ridge_ladder | 1e-7,3e-8,1e-8,0,3e-7,1e-6,3e-6,1e-5 | default |
| solve_repair_state_motion_l2_step_max | 0.01 | registry |
| solve_repair_strict_finite_shot_validation | False | default |
| solve_repair_theta_dot_l2_max | None | default |
| structural_score_floor | 0.0 | default |
| support_patch_scoring_workers | 2 | default |
| t_final | 10.0 | registry |
| t_initial | None | default |
| tag | None | default |
| times | None | default |

</details>

<details><summary><b>append_only</b> — 108 parameters (13 set by registry)</summary>

| parameter | value | source |
|---|---|---|
| append_cost_alpha | 1.0 | default |
| append_cost_lambda_1q | 0.025 | default |
| append_cost_lambda_2q | 0.05 | default |
| append_cost_lambda_d | 0.05 | default |
| append_cost_lambda_shot | 0.02 | default |
| append_cost_lambda_theta | 0.0 | default |
| append_cost_normalization_mode | family_robust_v1 | default |
| append_cost_scale_floor | 1e-12 | default |
| append_min_time | 0.0 | default |
| append_occurrence_policy | layer_reuse | default |
| append_schur_condition_gate | False | registry |
| append_schur_max_condition_number | 1000000000000.0 | default |
| avqds_delta_theta_max | None | default |
| avqds_l2_cut | 0.001 | default |
| avqds_max_appends_per_checkpoint | None | default |
| certification_refit | True | default |
| certification_refit_max_iterations | 15 | default |
| certification_refit_trust_radius | 0.6 | default |
| debt_policy | insertion_only | registry |
| diagnostic_append_pool_mode | none | default |
| drive_A | 2.4 | registry |
| drive_aligned_ansatz | True | default |
| drive_custom_weights | None | default |
| drive_include_identity | False | default |
| drive_n_sites | None | default |
| drive_omega | 3.0 | registry |
| drive_ordering | None | default |
| drive_pattern | staggered | default |
| drive_phi | 0.0 | default |
| drive_t0 | 0.0 | default |
| drive_tbar | 1.0 | default |
| drive_time_sampling | midpoint | default |
| dynamics_policy | exchange | default |
| enable_drive | True | registry |
| eps_loss | 1e-14 | default |
| escalation_accumulated_drift_threshold | None | default |
| exchange_deletions | False | registry |
| fail_on_unsupported_steps | False | default |
| fallback_family | full_meta | default |
| generator_family | match_adapt | default |
| insertion_gate_mode | mclachlan_l2 | registry |
| insertion_l2_cut | 0.001 | default |
| integrator | rk4 | default |
| interaction_frontier_widths | None | default |
| loader_mode | None | default |
| max_append_candidates | 8 | default |
| max_certification_attempts_per_deletion_branch | 2 | default |
| max_certification_attempts_per_level | 12 | default |
| max_insertion_batch_size | 1 | default |
| max_insertion_rounds_per_checkpoint | 12 | default |
| max_joint_patch_evaluations | 50000 | default |
| max_structural_pool_size | 128 | registry |
| min_logical_parameter_count | 1 | default |
| min_runtime_parameter_count | 1 | default |
| normalized_candidate_pool_profile | none | default |
| num_times | 251 | registry |
| parameterization_mode | per_pauli_term | default |
| patch_utility_delta_weight | 1.0 | default |
| pinv_rcond | 1e-10 | default |
| progress_log_events | True | default |
| progress_log_every | 1 | registry |
| prune_condition_lambda_kappa_dam | 0.0 | default |
| prune_condition_lambda_kappa_rel | 0.0 | default |
| prune_cooldown_steps | 2 | default |
| prune_cost_alpha | 1.0 | default |
| prune_history_lambda | 0.0 | default |
| prune_history_window | 3 | default |
| prune_patch_smoothness_eta_max | 0.001 | default |
| prune_ray_distance_tol | 0.002 | default |
| prune_target_policy | all_active | default |
| record_statevector | False | default |
| reference_energy_atol | 1e-12 | default |
| reference_energy_json | None | default |
| replay_candidate_pool_mode | None | default |
| require_complete_candidate_pool | False | default |
| residual_ratio_threshold | 0.02 | default |
| resume_from_run_json | None | default |
| ridge_lambda | 1e-07 | default |
| seed_reference_energy_atol | 1e-12 | default |
| seed_reference_energy_json | None | default |
| solve_damping | 0.0 | default |
| solve_repair | True | default |
| solve_repair_condition_number_fail | None | default |
| solve_repair_condition_number_max | 1000000.0 | default |
| solve_repair_damping_ladder | 0 | default |
| solve_repair_kink_eta_max | 0.005 | registry |
| solve_repair_local_subdivision | True | default |
| solve_repair_local_subdivision_factor | 2 | default |
| solve_repair_max_local_subdivisions | 10 | default |
| solve_repair_min_local_dt | 1e-06 | default |
| solve_repair_parameter_step_max | None | default |
| solve_repair_pinv_rcond_ladder | 1e-10,1e-11,1e-12,1e-9,1e-8,1e-7 | default |
| solve_repair_profile | minimal | default |
| solve_repair_release_kink_severity_scale | 4.0 | default |
| solve_repair_release_kink_threshold_scale | 0.5 | default |
| solve_repair_release_patience_max | 5 | default |
| solve_repair_release_patience_min | 1 | default |
| solve_repair_rho_num_max | 0.01 | default |
| solve_repair_ridge_ladder | 1e-7,3e-8,1e-8,0,3e-7,1e-6,3e-6,1e-5 | default |
| solve_repair_state_motion_l2_step_max | 0.01 | registry |
| solve_repair_strict_finite_shot_validation | False | default |
| solve_repair_theta_dot_l2_max | None | default |
| structural_score_floor | 0.0 | default |
| support_patch_scoring_workers | 2 | default |
| t_final | 10.0 | registry |
| t_initial | None | default |
| tag | None | default |
| times | None | default |

</details>

<details><summary><b>avqds</b> — 108 parameters (11 set by registry)</summary>

| parameter | value | source |
|---|---|---|
| append_cost_alpha | 1.0 | default |
| append_cost_lambda_1q | 0.025 | default |
| append_cost_lambda_2q | 0.05 | default |
| append_cost_lambda_d | 0.05 | default |
| append_cost_lambda_shot | 0.02 | default |
| append_cost_lambda_theta | 0.0 | default |
| append_cost_normalization_mode | family_robust_v1 | default |
| append_cost_scale_floor | 1e-12 | default |
| append_min_time | 0.0 | default |
| append_occurrence_policy | layer_reuse | default |
| append_schur_condition_gate | False | registry |
| append_schur_max_condition_number | 1000000000000.0 | default |
| avqds_delta_theta_max | None | default |
| avqds_l2_cut | 0.001 | default |
| avqds_max_appends_per_checkpoint | None | default |
| certification_refit | True | default |
| certification_refit_max_iterations | 15 | default |
| certification_refit_trust_radius | 0.6 | default |
| debt_policy | drift_ranked | default |
| diagnostic_append_pool_mode | none | default |
| drive_A | 2.4 | registry |
| drive_aligned_ansatz | True | default |
| drive_custom_weights | None | default |
| drive_include_identity | False | default |
| drive_n_sites | None | default |
| drive_omega | 3.0 | registry |
| drive_ordering | None | default |
| drive_pattern | staggered | default |
| drive_phi | 0.0 | default |
| drive_t0 | 0.0 | default |
| drive_tbar | 1.0 | default |
| drive_time_sampling | midpoint | default |
| dynamics_policy | avqds | registry |
| enable_drive | True | registry |
| eps_loss | 1e-14 | default |
| escalation_accumulated_drift_threshold | None | default |
| exchange_deletions | True | default |
| fail_on_unsupported_steps | False | default |
| fallback_family | full_meta | default |
| generator_family | match_adapt | default |
| insertion_gate_mode | residual_ratio | default |
| insertion_l2_cut | 0.001 | default |
| integrator | rk4 | default |
| interaction_frontier_widths | None | default |
| loader_mode | None | default |
| max_append_candidates | 8 | default |
| max_certification_attempts_per_deletion_branch | 2 | default |
| max_certification_attempts_per_level | 12 | default |
| max_insertion_batch_size | 1 | default |
| max_insertion_rounds_per_checkpoint | 12 | default |
| max_joint_patch_evaluations | 50000 | default |
| max_structural_pool_size | 128 | registry |
| min_logical_parameter_count | 1 | default |
| min_runtime_parameter_count | 1 | default |
| normalized_candidate_pool_profile | none | default |
| num_times | 251 | registry |
| parameterization_mode | per_pauli_term | default |
| patch_utility_delta_weight | 1.0 | default |
| pinv_rcond | 1e-10 | default |
| progress_log_events | True | default |
| progress_log_every | 1 | registry |
| prune_condition_lambda_kappa_dam | 0.0 | default |
| prune_condition_lambda_kappa_rel | 0.0 | default |
| prune_cooldown_steps | 2 | default |
| prune_cost_alpha | 1.0 | default |
| prune_history_lambda | 0.0 | default |
| prune_history_window | 3 | default |
| prune_patch_smoothness_eta_max | 0.001 | default |
| prune_ray_distance_tol | 0.002 | default |
| prune_target_policy | all_active | default |
| record_statevector | False | default |
| reference_energy_atol | 1e-12 | default |
| reference_energy_json | None | default |
| replay_candidate_pool_mode | None | default |
| require_complete_candidate_pool | False | default |
| residual_ratio_threshold | 0.02 | default |
| resume_from_run_json | None | default |
| ridge_lambda | 1e-07 | default |
| seed_reference_energy_atol | 1e-12 | default |
| seed_reference_energy_json | None | default |
| solve_damping | 0.0 | default |
| solve_repair | True | default |
| solve_repair_condition_number_fail | None | default |
| solve_repair_condition_number_max | 1000000.0 | default |
| solve_repair_damping_ladder | 0 | default |
| solve_repair_kink_eta_max | 0.005 | registry |
| solve_repair_local_subdivision | True | default |
| solve_repair_local_subdivision_factor | 2 | default |
| solve_repair_max_local_subdivisions | 10 | default |
| solve_repair_min_local_dt | 1e-06 | default |
| solve_repair_parameter_step_max | None | default |
| solve_repair_pinv_rcond_ladder | 1e-10,1e-11,1e-12,1e-9,1e-8,1e-7 | default |
| solve_repair_profile | minimal | default |
| solve_repair_release_kink_severity_scale | 4.0 | default |
| solve_repair_release_kink_threshold_scale | 0.5 | default |
| solve_repair_release_patience_max | 5 | default |
| solve_repair_release_patience_min | 1 | default |
| solve_repair_rho_num_max | 0.01 | default |
| solve_repair_ridge_ladder | 1e-7,3e-8,1e-8,0,3e-7,1e-6,3e-6,1e-5 | default |
| solve_repair_state_motion_l2_step_max | 0.01 | registry |
| solve_repair_strict_finite_shot_validation | False | default |
| solve_repair_theta_dot_l2_max | None | default |
| structural_score_floor | 0.0 | default |
| support_patch_scoring_workers | 2 | default |
| t_final | 10.0 | registry |
| t_initial | None | default |
| tag | None | default |
| times | None | default |

</details>

<details><summary><b>avqds_published</b> — 108 parameters (14 set by registry)</summary>

| parameter | value | source |
|---|---|---|
| append_cost_alpha | 1.0 | default |
| append_cost_lambda_1q | 0.025 | default |
| append_cost_lambda_2q | 0.05 | default |
| append_cost_lambda_d | 0.05 | default |
| append_cost_lambda_shot | 0.02 | default |
| append_cost_lambda_theta | 0.0 | default |
| append_cost_normalization_mode | family_robust_v1 | default |
| append_cost_scale_floor | 1e-12 | default |
| append_min_time | 0.0 | default |
| append_occurrence_policy | layer_reuse | default |
| append_schur_condition_gate | False | registry |
| append_schur_max_condition_number | 1000000000000.0 | default |
| avqds_delta_theta_max | 0.005 | registry |
| avqds_l2_cut | 0.001 | default |
| avqds_max_appends_per_checkpoint | None | default |
| certification_refit | False | registry |
| certification_refit_max_iterations | 15 | default |
| certification_refit_trust_radius | 0.6 | default |
| debt_policy | drift_ranked | default |
| diagnostic_append_pool_mode | none | default |
| drive_A | 2.4 | registry |
| drive_aligned_ansatz | True | default |
| drive_custom_weights | None | default |
| drive_include_identity | False | default |
| drive_n_sites | None | default |
| drive_omega | 3.0 | registry |
| drive_ordering | None | default |
| drive_pattern | staggered | default |
| drive_phi | 0.0 | default |
| drive_t0 | 0.0 | default |
| drive_tbar | 1.0 | default |
| drive_time_sampling | midpoint | default |
| dynamics_policy | avqds | registry |
| enable_drive | True | registry |
| eps_loss | 1e-14 | default |
| escalation_accumulated_drift_threshold | None | default |
| exchange_deletions | True | default |
| fail_on_unsupported_steps | False | default |
| fallback_family | full_meta | default |
| generator_family | match_adapt | default |
| insertion_gate_mode | residual_ratio | default |
| insertion_l2_cut | 0.001 | default |
| integrator | euler | registry |
| interaction_frontier_widths | None | default |
| loader_mode | None | default |
| max_append_candidates | 8 | default |
| max_certification_attempts_per_deletion_branch | 2 | default |
| max_certification_attempts_per_level | 12 | default |
| max_insertion_batch_size | 1 | default |
| max_insertion_rounds_per_checkpoint | 12 | default |
| max_joint_patch_evaluations | 50000 | default |
| max_structural_pool_size | 128 | registry |
| min_logical_parameter_count | 1 | default |
| min_runtime_parameter_count | 1 | default |
| normalized_candidate_pool_profile | none | default |
| num_times | 251 | registry |
| parameterization_mode | per_pauli_term | default |
| patch_utility_delta_weight | 1.0 | default |
| pinv_rcond | 1e-10 | default |
| progress_log_events | True | default |
| progress_log_every | 1 | registry |
| prune_condition_lambda_kappa_dam | 0.0 | default |
| prune_condition_lambda_kappa_rel | 0.0 | default |
| prune_cooldown_steps | 2 | default |
| prune_cost_alpha | 1.0 | default |
| prune_history_lambda | 0.0 | default |
| prune_history_window | 3 | default |
| prune_patch_smoothness_eta_max | 0.001 | default |
| prune_ray_distance_tol | 0.002 | default |
| prune_target_policy | all_active | default |
| record_statevector | False | default |
| reference_energy_atol | 1e-12 | default |
| reference_energy_json | None | default |
| replay_candidate_pool_mode | None | default |
| require_complete_candidate_pool | False | default |
| residual_ratio_threshold | 0.02 | default |
| resume_from_run_json | None | default |
| ridge_lambda | 1e-06 | registry |
| seed_reference_energy_atol | 1e-12 | default |
| seed_reference_energy_json | None | default |
| solve_damping | 0.0 | default |
| solve_repair | False | registry |
| solve_repair_condition_number_fail | None | default |
| solve_repair_condition_number_max | 1000000.0 | default |
| solve_repair_damping_ladder | 0 | default |
| solve_repair_kink_eta_max | 0.01 | default |
| solve_repair_local_subdivision | True | default |
| solve_repair_local_subdivision_factor | 2 | default |
| solve_repair_max_local_subdivisions | 10 | default |
| solve_repair_min_local_dt | 1e-06 | default |
| solve_repair_parameter_step_max | None | default |
| solve_repair_pinv_rcond_ladder | 1e-10,1e-11,1e-12,1e-9,1e-8,1e-7 | default |
| solve_repair_profile | minimal | default |
| solve_repair_release_kink_severity_scale | 4.0 | default |
| solve_repair_release_kink_threshold_scale | 0.5 | default |
| solve_repair_release_patience_max | 5 | default |
| solve_repair_release_patience_min | 1 | default |
| solve_repair_rho_num_max | 0.01 | default |
| solve_repair_ridge_ladder | 1e-7,3e-8,1e-8,0,3e-7,1e-6,3e-6,1e-5 | default |
| solve_repair_state_motion_l2_step_max | 0.05 | default |
| solve_repair_strict_finite_shot_validation | False | default |
| solve_repair_theta_dot_l2_max | None | default |
| structural_score_floor | 0.0 | default |
| support_patch_scoring_workers | 2 | default |
| t_final | 10.0 | registry |
| t_initial | None | default |
| tag | None | default |
| times | None | default |

</details>
