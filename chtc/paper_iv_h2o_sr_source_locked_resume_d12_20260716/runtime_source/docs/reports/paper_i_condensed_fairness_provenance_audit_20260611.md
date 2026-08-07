# Paper I Condensed Fairness and Provenance Audit

Generated: `2026-06-11T20:51:12.418442+00:00`

## Scope

- Evidence-only audit; no manuscript edits, no table edits, no runs, no PDF rebuild.
- Condensed TeX: `MATH/paper_details/static_adapt_paper_I_condensed.tex`
- Condensed PDF: `MATH/paper_details/static_adapt_paper_I_condensed.pdf`
- Candidate fix checklist is for user review only; it is not an edit instruction.

## Summary Counts

- `findings`: `{'blocked': 20, 'policy_divergence': 1, 'qualified': 10}`
- `source_hash`: `{'directory_not_checked': 9, 'external_not_checked': 1, 'match': 296, 'mismatch': 16, 'not_checked': 51}`
- `metric_policy`: `{'ok': 3, 'policy_divergence': 1}`
- `compiled_cost`: `{'blocked': 1, 'ok': 69, 'qualified': 2}`
- `work_proxy`: `{'blocked': 2, 'qualified': 54}`
- `fairness`: `{'blocked': 1, 'ok': 4, 'qualified': 2}`

## PDF/TeX Sync

- Status: `metadata_checked`
- PDF pages: `19`
- Notes: `[]`

## Metric Policy Matrix

| Table | Expected | Observed | Status |
|---|---|---|---|
| `tab:fixed_accuracy_claims` | `raw_absolute_error_no_phonon` | `raw_absolute_error_from_condensed_metric_paragraph` | `ok` |
| `tab:fixed_accuracy_spin_boson` | `raw_same_cutoff_ed_error_with_higher_cutoff_diagnostic` | `raw_same_cutoff_ed_error` | `ok` |
| `tab:hh_first_plateau_prefix_costs` | `raw_same_cutoff_ed_error_at_first_effective_plateau_prefix` | `raw_same_cutoff_ed_error_at_first_effective_plateau_prefix` | `ok` |
| `tab:fixed_accuracy_hh_cartesian` | `raw_external_reference_error_with_fixed_prefix_resources` | `same_cutoff_wording_in_condensed_prose_and_caption` | `policy_divergence` |

## Main Findings

- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: output/pdf/paper_i_tables_i_ii_repeat_enabled_iteration_plots_20260610.provenance.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: MATH/paper_facing/paper_I_static_scaffold/paper_i_tables_i_ii_repeat_enabled_comparator_promotion_20260610.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: output/pdf/paper_i_data_analysis_hubbard_strong_repeat_enabled_error_vs_iteration_20260610.png
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: output/pdf/paper_i_data_analysis_hubbard_strong_repeat_enabled_error_vs_iteration_20260610.pdf
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: output/pdf/paper_i_data_analysis_hubbard_weak_repeat_enabled_error_vs_iteration_20260610.png
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: output/pdf/paper_i_data_analysis_spin_boson_repeat_enabled_same_cutoff_error_vs_iteration_20260610.png
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: raw_outputs/chtc_fetches/paper_i_hh_tableiii_spsa_7096390_minimal_20260601/paper_i_hh_tableiii_spsa_minimal_20260601_120550/records/paper_i_comp_spsa_cal__full__static_full_meta_append_adapt_vqe__hh_sym_strong_strong/trial_0027/cases/hh_L2_nph4_three_model_sym_strong_strong/generic_static_single.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: raw_outputs/chtc_fetches/paper_i_hh_tableiii_spsa_7096390_minimal_20260601/paper_i_hh_tableiii_spsa_minimal_20260601_120550/records/paper_i_comp_spsa_cal__full__static_tetris_qubit_adapt_vqe__hh_sym_strong_strong/trial_0021/cases/hh_L2_nph4_three_model_sym_strong_strong/generic_static_single.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: raw_outputs/chtc_fetches/paper_i_hh_tableiii_spsa_7096390_minimal_20260601/paper_i_hh_tableiii_spsa_minimal_20260601_120550/records/paper_i_comp_spsa_cal__full__static_full_meta_append_adapt_vqe__hh_sym_strong_weak/trial_0014/cases/hh_L2_nph2_three_model_sym_strong_weak/generic_static_single.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_strong_weak_snake_stdout_held_continuation_promotion_20260609.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: raw_outputs/chtc_fetches/paper_i_hh_tableiii_spsa_7096390_minimal_20260601/paper_i_hh_tableiii_spsa_minimal_20260601_120550/records/paper_i_comp_spsa_cal__full__static_tetris_qubit_adapt_vqe__hh_sym_strong_weak/trial_0025/cases/hh_L2_nph2_three_model_sym_strong_weak/generic_static_single.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: raw_outputs/chtc_fetches/paper_i_hh_tableiii_spsa_7096390_minimal_20260601/paper_i_hh_tableiii_spsa_minimal_20260601_120550/records/paper_i_comp_spsa_cal__full__static_full_meta_append_adapt_vqe__hh_sym_weak_strong/trial_0029/cases/hh_L2_nph4_three_model_sym_weak_strong/generic_static_single.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: raw_outputs/chtc_fetches/paper_i_hh_tableiii_spsa_7096390_minimal_20260601/paper_i_hh_tableiii_spsa_minimal_20260601_120550/records/paper_i_comp_spsa_cal__full__static_tetris_qubit_adapt_vqe__hh_sym_weak_strong/trial_0010/cases/hh_L2_nph4_three_model_sym_weak_strong/generic_static_single.json
- **blocked / source_sha256_mismatch**   : Referenced source hash mismatch: MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json
- **policy_divergence / metric_policy_mismatch** tab:fixed_accuracy_hh_cartesian  : tab:fixed_accuracy_hh_cartesian observed policy `same_cutoff_wording_in_condensed_prose_and_caption` differs from expected `raw_external_reference_error_with_fixed_prefix_resources`.
- **blocked / missing_replayable_prefix_operator_compile_metadata** tab:hh_first_plateau_prefix_costs SNAKE weak_strong: Compiled-cost status for SNAKE/weak_strong is blocked: one or more displayed resource cells are --
- **qualified / retained_resource_cells_qualified** tab:hh_first_plateau_prefix_costs SNAKE strong_strong: Compiled-cost status for SNAKE/strong_strong is qualified: compiled_from_base_json_prefix_plus_stdout_continuation_label_metadata
- **qualified / retained_resource_cells_qualified** tab:fixed_accuracy_spin_boson Qubit/QEB strong: Compiled-cost status for Qubit/QEB/strong is qualified: source_json_missing for current nph2 strong comparator; existing resource cell preserved
- **qualified / legacy_proxy_only** tab:fixed_accuracy_claims SNAKE weak: Work proxy for SNAKE/weak is `controller_shot_proxy_or_S_norm_legacy_surface`; requires S_alg component audit before apples-to-apples work claim
- **qualified / legacy_proxy_only** tab:fixed_accuracy_claims SNAKE strong: Work proxy for SNAKE/strong is `controller_shot_proxy_or_S_norm_legacy_surface`; requires S_alg component audit before apples-to-apples work claim
- **qualified / legacy_proxy_only** tab:fixed_accuracy_spin_boson SNAKE weak: Work proxy for SNAKE/weak is `controller_shot_proxy_or_S_norm_legacy_surface`; requires S_alg component audit before apples-to-apples work claim
- **qualified / legacy_proxy_only** tab:fixed_accuracy_spin_boson SNAKE strong: Work proxy for SNAKE/strong is `controller_shot_proxy_or_S_norm_legacy_surface`; requires S_alg component audit before apples-to-apples work claim
- **qualified / legacy_proxy_only** tab:fixed_accuracy_hh_cartesian SNAKE weak_weak: Work proxy for SNAKE/weak_weak is `controller_shot_proxy_or_S_norm_legacy_surface`; requires S_alg component audit before apples-to-apples work claim
- **qualified / legacy_proxy_only** tab:fixed_accuracy_hh_cartesian SNAKE strong_weak: Work proxy for SNAKE/strong_weak is `controller_shot_proxy_or_S_norm_legacy_surface`; requires S_alg component audit before apples-to-apples work claim
- **blocked / work_proxy_currency_mismatch** tab:fixed_accuracy_hh_cartesian SNAKE weak_strong: Work proxy for SNAKE/weak_strong is `missing`; not_comparable_until_source_work_proxy_is_recovered
- **blocked / work_proxy_currency_mismatch** tab:fixed_accuracy_hh_cartesian SNAKE strong_strong: Work proxy for SNAKE/strong_strong is `missing`; not_comparable_until_source_work_proxy_is_recovered
- **qualified / settings_fairness_noncanonical** tab:hh_first_plateau_prefix_costs SNAKE : Fairness/provenance status: not_yet_canonical_across_regimes
- **blocked / strict_replay_missing** tab:hh_first_plateau_prefix_costs SNAKE weak_strong: Fairness/provenance status: missing_replayable_prefix_operator_compile_metadata
- **qualified / strict_replay_missing** tab:hh_first_plateau_prefix_costs SNAKE strong_strong: Fairness/provenance status: source history is stdout-derived; recover replayable continuation current/result JSON before final manuscript promotion.

## Candidate Fix Checklist (Review Only)

1. Review the condensed HH appendix fixed-prefix metric wording against the support contract: the audit sees same-cutoff wording where the Paper-I results contract expects raw higher-cutoff external-reference error.  
   Evidence: `tab:fixed_accuracy_hh_cartesian caption/prose and paper_i_tables.md contract`
2. Review claims that compare estimator/work proxy `S` across methods; several rows use legacy or partial proxy currencies rather than complete `S_alg` components.  
   Evidence: `work_proxy_currency_matrix`

## Output Matrices

Detailed JSON/CSV matrices contain the visible table inventory, source-reference checks, compiled-cost classifications, work-proxy currency classifications, and fairness-status classifications.

