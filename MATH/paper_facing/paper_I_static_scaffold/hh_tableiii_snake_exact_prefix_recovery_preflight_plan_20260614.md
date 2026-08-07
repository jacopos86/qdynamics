# HH Table III SNAKE exact-prefix recovery/preflight plan — 2026-06-14
## Scope and hard stops
- Work item: **2 only** from `hh_cost_iteration_orchestration_plan_20260614.md`.
- Target: Paper-I HH Table III (`tab:hh_first_plateau_prefix_costs`) **SNAKE only**.
- No CHTC submission was performed. No manuscript `.tex` files were edited. No preview plot scripts were edited.
- Promotion authority remains user-only; this file records evidence status and preflight blockers only.

## Current exact-prefix status
The current fail-closed audit command was:

```bash
python3 pipelines/reporting/audit_paper_i_hh_prefix_replayability.py --source-map MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json --output-json /tmp/paper_i_hh_tableiii_prefix_replayability_audit_current_20260614.json
```

Observed status counts: `exact-prefix-replay-ready=12`, `needs-richer-history=1`, `stdout-only-blocked=3`. Append/TETRIS/Geo are already exact-prefix-ready from existing recovered records; SNAKE remains blocked under current audit rules.

## Target SNAKE regimes and visible source anchors
| Regime | n_ph work/ref diagnostic | Visible source anchor | Visible/source prefix target | Current audit class | Pre-submit note |
|---|---:|---|---:|---|---|
| `weak_weak` | 2/5 | `output/pdf/paper_i_table_iii_snake_weak_weak_live_prefix_promotion_20260530.json`<br>`784ff990e4489ea10c6725cd8e5d227a4342c74eae71d7b73c2d285a2b90d391` | plateau k=11; target 20 | `needs-richer-history` | Try local exact-prefix export from the strict replay candidate before scheduling CHTC recovery. |
| `strong_weak` | 2/5 | `MATH/paper_facing/paper_I_static_scaffold/history_sources/hh_tableiii_strong_weak_snake_continue_7582403_stdout_20260609.json`<br>`b502c332cec1eb77be197737156c0cbf0ed315d4a08f9d359c4f5ffe7fc57629` | plateau k=11; target 65; optional full 129 | `stdout-only-blocked` | Requires source-locked continuation/recovery if exact per-prefix curve beyond the previous k=11 JSON-backed source is required. Do not stitch stdout energy tail to retained old compiled sidecar. |
| `weak_strong` | 4/7 | `MATH/paper_facing/paper_I_static_scaffold/history_sources/hh_tableiii_weak_strong_snake_depth42_replayable_combined_20260614.json`<br>`bec3c5b14f6d563e559dd30e24932f4f8924de0c8749747b17236ea0f46971ca` | plateau k=42; target 42; old stdout 47 requires decision | `stdout-only-blocked` | Before any CHTC submit, decide whether depth42 is the source-locked visible trajectory to export locally or whether the user wants a continuation beyond depth42. Do not target the old stdout depth47 tail automatically. |
| `strong_strong` | 4/7 | `MATH/paper_facing/paper_I_static_scaffold/history_sources/hh_tableiii_strong_strong_snake_structural_continue_7096352_stdout_20260601.json`<br>`a19dcdd065a538c72e35ecbeb0f8c38c2791a22cb8f9d2a80f7371d57b69f7b9` | plateau k=13; target 47 | `stdout-only-blocked` | Requires source-locked continuation/recovery if exact prefix costs are needed through the visible 47-point curve. Anchor should prove plateau k=13/visible metric alignment before full-depth continuation. |

## Required telemetry for exact per-prefix cost replay
### identity_and_source_lock
- `table_label`
- `regime`
- `method`
- `case_id`
- `source_map_json`
- `source_map_sha256`
- `source_json`
- `source_sha256`
- `run_manifest_json`
- `record_id`
- `base_records_tsv`
- `base_record_id`
- `base_records_sha256`
- `trial_param_overrides_json`
- `trial_param_overrides_sha256`
- `settings_hash`
- `settings_changed_vs_source`

### cutoff_and_reference
- `n_ph_work`
- `n_ph_ref_diagnostic`
- `same_cutoff_exact_gs_energy`
- `same_cutoff_reference_energy_key`
- `reference_cutoff_energy_key`
- `primary_energy_metric`
- `same_cutoff_error_role`

### state_and_strict_replay
- `ansatz_input_state`
- `initial_state`
- `runtime_seed_json`
- `runtime_seed_sha256`
- `strict_replay_json`
- `strict_replay_sha256`
- `strict_replay_validation`
- `load_static_resume_source validation status`
- `validate_static_hh_resume_source validation status`

### per_prefix_history
- `adapt_history row for every accepted prefix`
- `prefix_k / adapt_iteration`
- `energy_after or energy_after_opt`
- `abs_delta_e_same_cutoff_after`
- `selected_batch_labels or selected_label`
- `selected_positions / selected_insertion_position`
- `accepted/admitted status`
- `rollback/prune admission records`
- `logical_operator_prefix_len`

### pauli_and_parameterization
- `selected operator labels`
- `selected operator Pauli expansions in e/x/y/z convention`
- `runtime_terms_exyz for parameterization blocks`
- `selected_operator_pauli_labels_exyz`
- `selected_operator_batches`
- `candidate_label to Pauli group map`
- `ordered cumulative ansatz block list after insertion-position updates`

### route_a_decision_telemetry
- `static_route_id=route_a`
- `static_meta_feature_profile=paper_i_production_v1`
- `selected_logical_route=standard when present`
- `pool/full_meta evidence`
- `phase3_selector_policy`
- `phase3_selector_geometry_mode`
- `phase3_window_relaxation_mode`
- `phase2_novelty_mode and exponent fields`
- `phase3_runtime_split_mode / child_set / chosen_representation fields`
- `phase3_enable_batching and phase3_batch_selection_mode`
- `adapt_beam_children_per_parent / live branches / terminated keep`
- `phase1_prune_enabled / policy / mode`
- `SPSA seeds and optimizer memory sufficient to resume/replay`

### exact_compiled_resource_rows
- `compile_convention=table_i_basis_gate_transpile_v1`
- `qiskit version and compile settings hash`
- `compile_status`
- `compile_error if any`
- `N1q`
- `N2q`
- `D_circ`
- `D2q`
- `compiled_op_counts`
- `runtime_rotation_count`
- `reference_state_status`
- `compiled_count_1q_semantics`

### progress_and_outputs
- `generic_static_single.json`
- `raw phase3 result.json`
- `current.json if live/recoverable`
- `manifest.json`
- `effective_env_overlay.json`
- `stdout/stderr logs`
- `per-prefix exact resource export JSON or enough replay data for pipelines/exact_bench/hh_tableiii_prefix_resources.py`

## Can the 20260612 shot-proxy generator be adapted?
**Yes as a template, no as a verbatim generator.**

Useful template pieces:
- Resolves four HH SNAKE regimes from visible source-map entries.
- Materializes source trial overrides into per-regime phase3_policy_json files.
- Emits records.tsv, record_ids.txt, manifest.json, phase3 policies, and a Condor submit surface without Optuna search.
- Existing submit file preflighted successfully as a shape check: 0/4 failed.

Why it must not be reused verbatim:
- It is a deterministic shot-proxy repair generator, not an exact per-prefix resource replay generator.
- Its --check is stale against current files: records.tsv and manifest.json mismatch current rendered content.
- Its depth cap is the visible plateau iteration; exact convergence overlays may require longer display histories.
- Its record fields do not explicitly require strict replay/current/result JSON, selected Pauli expansions for every prefix, or per-prefix exact Qiskit resource export rows.
- It does not produce a source_locked_sensitivity_audit_v1-style non-telemetry settings-diff/anchor audit.

Planned adapted generator: `chtc/phase3_optuna/generate_paper_i_hh_tableiii_snake_exact_prefix_recovery_records.py`.
Planned batch id: `paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1`.

## Exact files for the next scaffold pass
- `new_generator_scaffold_if_approved`: `chtc/phase3_optuna/generate_paper_i_hh_tableiii_snake_exact_prefix_recovery_records.py`
- `new_input_dir_if_generated`: `chtc/phase3_optuna/input/paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1/`
- `records_tsv`: `chtc/phase3_optuna/input/paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1/paper_i_hh_tableiii_snake_exact_prefix_recovery_records.tsv`
- `record_id_file`: `chtc/phase3_optuna/input/paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1/paper_i_hh_tableiii_snake_exact_prefix_recovery_record_ids.txt`
- `manifest_json`: `chtc/phase3_optuna/input/paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1/paper_i_hh_tableiii_snake_exact_prefix_recovery_manifest.json`
- `phase3_policy_dir`: `chtc/phase3_optuna/input/paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1/phase3_policies/`
- `source_lock_audit_json`: `output/pdf/paper_i_hh_tableiii_snake_exact_prefix_recovery_source_lock_audit_20260614.json`
- `preflight_json`: `output/pdf/paper_i_hh_tableiii_snake_exact_prefix_recovery_preflight_20260614.json`
- `submit_file_prepared_not_submitted`: `chtc/phase3_optuna/submit_paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1.sub`

## Exact commands for record generation and preflight
### resolve_visible_settings_all_regimes
```bash
for r in weak_weak strong_weak weak_strong strong_strong; do python3 agent_guidance/skills/shared/scripts/resolve_visible_settings.py --source-map MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json --regime "$r" --method SNAKE > output/pdf/paper_i_hh_tableiii_snake_exact_prefix_recovery_resolve_${r}_20260614.json; done
```
### current_replayability_audit
```bash
python3 pipelines/reporting/audit_paper_i_hh_prefix_replayability.py --source-map MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json --output-json output/pdf/paper_i_hh_tableiii_prefix_replayability_audit_snake_preflight_20260614.json
```
### generate_records_after_generator_exists
```bash
python3 chtc/phase3_optuna/generate_paper_i_hh_tableiii_snake_exact_prefix_recovery_records.py --output-dir chtc/phase3_optuna/input/paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1 --write
```
### check_records_after_generation
```bash
python3 chtc/phase3_optuna/generate_paper_i_hh_tableiii_snake_exact_prefix_recovery_records.py --output-dir chtc/phase3_optuna/input/paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1 --check
```
### preflight_submit_surface_do_not_submit
```bash
python3 chtc/phase3_optuna/preflight_submit.py --submit chtc/phase3_optuna/submit_paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1.sub --record-id-file chtc/phase3_optuna/input/paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1/paper_i_hh_tableiii_snake_exact_prefix_recovery_record_ids.txt --output-json output/pdf/paper_i_hh_tableiii_snake_exact_prefix_recovery_preflight_20260614.json
```
### local_existing_snake_export_probe_before_submit
```bash
python3 pipelines/exact_bench/export_paper_i_hh_tableiii_prefix_resources.py --source-map MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json --audit-json output/pdf/paper_i_hh_tableiii_prefix_replayability_audit_snake_preflight_20260614.json --output-json output/pdf/paper_i_hh_tableiii_prefix_resources_snake_existing_probe_20260614.json --method SNAKE --progress
```
### forbidden_until_user_approval
```bash
condor_submit chtc/phase3_optuna/submit_paper_i_hh_tableiii_snake_exact_prefix_recovery_20260614_v1.sub
```
Do not run the forbidden `condor_submit` command unless the user approves a specific preflighted submit surface in a later turn.

## Blockers/questions before submission
- **blocker_before_chtc / weak_weak_local_export_first**: Weak-weak has an exact-prefix-ready strict_replay_json candidate. Try local exporter/source-map audit resolution before scheduling CHTC work.
- **decision_required / weak_strong_source_policy**: Weak-strong current source map points to a depth-42 replayable combined source, while the older stdout depth-47 tail is previous-source provenance. Decide whether exact-prefix recovery target is depth42 only or a user-approved continuation beyond depth42.
- **decision_required / strong_weak_depth_scope**: Strong-weak visible history has 129 stdout-derived rows; first exact recovery target can be display-crop depth65, with full depth129 optional. User should approve whether to recover 65 or 129 before submit.
- **blocker_before_reuse / source_lock_generator_stale**: The previous shot-proxy generator --check is stale for records.tsv and manifest.json, so old inputs must not be reused without regenerating/adapting and auditing settings diffs.
- **blocker_before_submit / telemetry_presence**: The adapted generator/runner must prove it emits strict replay/current/result JSON and per-prefix selected Pauli/position/state telemetry; otherwise the rerun can still fail exact-prefix export.
- **blocker_before_fanout / anchor_reproduction**: An anchor row must reproduce the visible/source metric and non-telemetry settings before any full-depth fan-out. If the anchor differs, classify output as companion/recovery evidence and ask the user.
- **hard_stop / no_submit_without_approval**: Do not run condor_submit until the user approves a specific submit surface after preflight passes.

## Validation performed this turn
- `python3 pipelines/reporting/audit_paper_i_hh_prefix_replayability.py --source-map MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json --output-json /tmp/paper_i_hh_tableiii_prefix_replayability_audit_current_20260614.json` -> pass; status_counts exact-prefix-replay-ready=12, needs-richer-history=1, stdout-only-blocked=3.
- `python3 agent_guidance/skills/shared/scripts/resolve_visible_settings.py --source-map MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json --regime <each SNAKE regime> --method SNAKE` -> all four resolver invocations returned status ok with matching source hashes; weak-strong output is large and records a qualified depth42 source.
- `python3 chtc/phase3_optuna/generate_paper_i_hh_tableiii_snake_shot_proxy_repair_records.py --check` -> failed as stale; content mismatch for prior records.tsv and manifest.json.
- `python3 chtc/phase3_optuna/preflight_submit.py --submit chtc/phase3_optuna/submit_paper_i_hh_tableiii_snake_shot_proxy_repair_20260612_v1.sub --record-id-file chtc/phase3_optuna/input/paper_i_hh_tableiii_snake_shot_proxy_repair_20260612_v1/paper_i_hh_tableiii_snake_shot_proxy_repair_record_ids.txt --output-json /tmp/paper_i_hh_tableiii_snake_shot_proxy_repair_preflight_check_20260614.json` -> prior submit surface shape preflight passed: 0/4 failed; this validates shape only, not exact-prefix recovery readiness.

## Files created by this turn
- `MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_snake_exact_prefix_recovery_preflight_plan_20260614.json`
- `MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_snake_exact_prefix_recovery_preflight_plan_20260614.md`

No CHTC submit, no manuscript `.tex` edit, and no preview plot script edit were performed.
