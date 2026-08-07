## Final Prompt
<taskname="HH Ablation Batch"/>
<task>
Implement the user’s requested weak-weak-only Paper-I Hubbard-Holstein SNAKE mechanism-ablation CHTC batch. The deliverable should include an agent-readable dated MD plan, a local record generator, focused tests, local fail-closed preflight, and then CHTC submission only after preflight passes and current-turn launch approval is satisfied. Do not edit manuscript TeX/PDFs.

Fixed user constraints to preserve: weak-weak first only; source anchor is the existing completed combinatorial batch-cap-3 weak-weak SNAKE row; existing no-batch row fills “No batching”; existing greedy cap-3 row is supplemental; no-second-order/no-Phase-II-novelty variants must disable Phase III via `adapt_continuation_mode=phase2_v1`; Phase-I-only should be attempted as macro-only and singleton-child variants; visible POWELL HH source contract stays full_meta unfiltered/HVA included, POWELL maxiter/refit 200, depth cap 30, Phase-III archival child split subset cap 1 unless the row explicitly ablates child pool; use source-lock runner overrides, not current defaults; fail closed on ambiguous rows.
</task>

<architecture>
- `chtc/phase3_optuna/generate_paper_i_hh_fullmeta_phase3_singleton_rotosolve_records.py` is the safest base generator: it creates TSV/queue/manifest/submit files and source-locked SNAKE rows via `make_row()`, `_snake_runtime_policy()`, `SNAKE_CANONICAL_CLI_OVERRIDES`, and `snake_cli_overrides_json`.
- `chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_cell.py` consumes each TSV row and builds the effective command. `build_snake_source_locked_command()` and `apply_snake_source_locked_overrides()` are the contract: row fields plus `snake_cli_overrides_json` are the allowed way to set batch, continuation, prune, and source-lock differences.
- `chtc/phase3_optuna/preflight_submit.py` currently treats recovery/ordered-batch fullmeta singleton prefixes specially. Its existing blockers expect cap 3 for `recovery_candidate`/`ordered_batch_beam`; the new batch needs its own prefix/contract or adjusted detection so batch cap 3 does not imply Pauli-child subset cap 3.
- `pipelines/static_adapt/adapt_pipeline.py`, `cli_config.py`, and `engine_support.py` show `adapt_continuation_mode`: `phase2_v1` enables Phase I/II and disables Phase III. Phase-III-only knobs such as child-pool expansion, proxy-reduced selector geometry, and non-proxy backend-cost mode are invalid unless mode is `phase3_v1`.
- CHTC submit files use `run_paper_i_hh_spsa_budget_ladder_task_apptainer_srcpkg.sh`; `run_task.parse_submit_contract()` and `preflight_submit.build_preflight_bundle()` validate queue/records visibility. `upload_submit_chtc.sh` has preflight and remote `condor_submit` action patterns but no current action for this new batch.
</architecture>

<selected_context>
- `AGENTS.md`, `MATH/AGENTS.md`, `agent_guidance/skills/paper-i-run/SKILL.md`, `agent_guidance/skills/source-locked-sensitivity/SKILL.md`: repo/run gates, visible-row baseline/source-lock rules, preflight-before-submit, no promotion language, and CHTC/run safety.
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_visible_row_provenance_layers_20260705.md`: visible-row priority, POWELL weak-weak contract, archival missing-spec child-set nuance, and cap-3 diagnostic result paths. Primary combined cap-3 combinatorial candidate appears as `raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v15/...metric_prune_combinatorial_batch_cap3_children3/json/result.json`; greedy counterpart is v16; earlier lambda-zero actual-batch diagnostics are v10/v13.
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md`: current POWELL/full_meta/HVA/cap-1/beam/prune authority. It is maxB=1 oriented, but still governs the visible source contract.
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_weak_weak_snake_ablation_matrix_audit_20260707.md`: existing no-batch row and maxB=1 rows. No-batch source: `raw_outputs/paper_i_hh_powell_visible_batchroute_nobatch_20260706/weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__nobatch_fullv2/json/result.json`, SHA `ee696a09816d99c07cf4ca82e165848ea6cfee55eb7014ee45e2c2491ca45262`.
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_weak_weak_strong_strong_snake_batchcap3_ablation_audit_20260707.md`: explicit cap-3 distinction: Pauli-child subset cap remains 1; batch target/size cap is 3. Older “local only/no CHTC” scope is superseded by the user’s current request.
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json`: machine-readable no-batch row hashes, weak-weak values, and runtime contract.
- `chtc/phase3_optuna/generate_paper_i_hh_ordered_batch_beam_diagnostic_records.py` and `generate_paper_i_hh_recovery_candidate_run_stock_records.py`: negative/template context. They hardcode unsafe cap-3 semantics (`SUBSET_SIZE=3`, beam children 3, ordered batch cap assumptions) and should not be reused directly for this batch.
- `chtc/phase3_optuna/generate_paper_i_hubbard_weak_u025_snake_feature_ablation_records.py` slices: older no-phase3/Phase-I/no-shortlisting/full-geometry patterns. Treat `no_shortlisting` as blocked; `full_geometry_window` is conditional and must not bypass continuation-mode validation.
- Tests selected: recovery-candidate generator/preflight test patterns, source-lock command override tests, and a stale strong-weak test importing a missing generator.
</selected_context>

<relationships>
- New generator row -> TSV `snake_cli_overrides_json` -> `run_paper_i_hh_spsa_budget_ladder_cell.build_snake_source_locked_command()` -> `adapt_pipeline` CLI.
- New submit file -> `run_task.parse_submit_contract()` -> `preflight_submit.build_preflight_bundle()` -> per-row HH contract blockers.
- Visible-row authority -> support docs/JSON -> generator row fields and source-lock audit fields (`provenance_layer`, source result/hash, settings reused/changed).
- Batch-cap-3 rows must set Phase-2 and Phase-3 batch mode/target/cap to 3 while preserving `phase3_runtime_split_max_subset_size=1` and archival child-set selection unless the row is explicitly a child-pool ablation.
</relationships>

<ambiguities>
- Current user request supersedes older docs that say local-only/no-CHTC or maxB=1-only. Keep those docs as provenance, not the active scope.
- Two combinatorial cap-3 diagnostics are documented: v10 lambda-zero actual-batch and v15 beam-lambda 0.005 + metric-prune + combinatorial cap3. The user likely means v15 because it matches the current beam/prune contract; verify/record this in the MD plan and fail closed if uncertain.
- `agent_guidance/shared/run-guide.md` and `agent_guidance/skills/paper-i-chtc-optuna/SKILL.md` are referenced by docs but absent from this workspace.
- Raw result JSONs and generated input dirs are not selected/visible here; rely on support docs and hashes, and make local preflight validate paths/hashes before submission.
- `test/chtc/test_paper_i_hh_strong_weak_anchor_ablation_records.py` imports a missing generator and appears pre-existing/stale; do not treat that unrelated failure as caused by this task.
</ambiguities>

## Selection
- Files: 28 total (16 full, 12 slice)
- Total tokens: 113703 (Auto view)
- Token breakdown: full 65344, slice 48359

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── MATH/
│   ├── paper_facing/
│   │   └── paper_I_static_scaffold/
│   │       ├── paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md — 2,327 tokens (full)
│   │       ├── paper_i_hh_powell_weak_weak_snake_ablation_matrix_audit_20260707.md — 3,534 tokens (full)
│   │       ├── paper_i_hh_powell_weak_weak_strong_strong_snake_batchcap3_ablation_audit_20260707.md — 1,915 tokens (full)
│   │       ├── paper_i_hh_snake_canonical_runtime_settings_draft_20260627.md — 4,658 tokens (lines 1-116 (Canonical runtime draft front matter and visible-row correction: current POWELL full_meta/HVA visible row, Phase-III archival split cap 1, no preferred sequence, and active comparison contract.), 251-330 (Batching/pruning/maturity settings tables needed to distinguish batch target/cap from Pauli-child subset cap and note historical/default context.))
│   │       ├── paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json — 4,889 tokens (full)
│   │       └── paper_i_hh_visible_row_provenance_layers_20260705.md — 6,636 tokens (full)
│   └── AGENTS.md — 3,111 tokens (lines 1-150 (MATH subtree contract and Paper-I support/run gate requirements for ADAPT/HH/CHTC work.), 190-230 (ADAPT and pool policy: current HH full_meta/HVA and Route-A/SNAKE defaults, including reporting observed child/batch fields literally.))
├── agent_guidance/
│   └── skills/
│       ├── paper-i-run/
│       │   └── SKILL.md — 8,165 tokens (lines 1-150 (Paper-I run skill front matter, authority/read order, source-locked sensitivity gate, run classes, and smoke discipline relevant before planning/submission.), 188-310 (Current Hubbard-Holstein Table III visible-row contract: POWELL visible-row provenance, full_meta/HVA pool policy, six-regime HH grid, and active method matrix constraints.), 940-1031 (Canonical Paper-I HH plateau record and CHTC preflight requirements: exact HH-only record set, active methods/regimes, generated submit preflight command, and submit-after-preflight/user-approval rule.), 1070-1145 (Paper-I SNAKE canonical route/settings: production profile, child/beam/batch/prune policies, allowed ablations, and requirement to inspect code/artifacts for exact flag names.), 1260-1296 (Run lifecycle and monitoring/auto-repair requirements: preflight, launch, monitor, repair plumbing failures, and preserve scientific settings.))
│       └── source-locked-sensitivity/
│           └── SKILL.md — 2,143 tokens (full)
├── chtc/
│   └── phase3_optuna/
│       ├── fetch_from_chtc.sh — 201 tokens (full)
│       ├── generate_paper_i_hh_fullmeta_phase3_singleton_rotosolve_records.py — 16,229 tokens (full)
│       ├── generate_paper_i_hh_ordered_batch_beam_diagnostic_records.py — 4,319 tokens (full)
│       ├── generate_paper_i_hh_recovery_candidate_run_stock_records.py — 9,211 tokens (full)
│       ├── generate_paper_i_hubbard_weak_u025_snake_feature_ablation_records.py — 2,890 tokens (lines 250-369 (Older feature-ablation row definitions showing no_phase3 via adapt_continuation_mode=phase2_v1, Phase-I-only, full_geometry_window, and no_shortlisting patterns; use as negative/conditional context only.), 560-679 (Older feature-ablation write/submit split patterns, including separate no_shortlisting submit path that should be blocked for the new scope.))
│       ├── preflight_submit.py — 8,193 tokens (lines 1180-1588 (HH fullmeta singleton preflight detection and contract blockers, including current recovery/ordered-batch cap checks that a new mechanism-ablation prefix must avoid or extend safely.), 1880-2089 (build_preflight_bundle() and CLI main: how submit files, record ids, and per-record blockers are assembled into local preflight JSON.))
│       ├── run_paper_i_hh_spsa_budget_ladder_cell.py — 7,269 tokens (lines 1-130 (Runner imports/constants and source-lock flag allowlists, including SNAKE_CLI_OVERRIDES_FIELD and preserved flags for batching/continuation mode.), 340-509 (Source-lock helper functions: allowed flag changes from snake_cli_overrides_json, optimizer/runtime-split row accessors, and CLI override parsing.), 690-1029 (build_snake_source_locked_command(), apply_snake_source_locked_overrides(), and audit logic that source-locked generated rows must satisfy.))
│       ├── run_paper_i_hh_spsa_budget_ladder_task_apptainer_srcpkg.sh — 348 tokens (full)
│       ├── run_task.py — 1,266 tokens (lines 380-499 (Submit contract parser and sandbox visibility helpers used by preflight tests for queue files, transfer_input_files, and argument records paths.))
│       ├── submit_paper_i_hh_fullmeta_singleton_symmetry_ordered_batch_beam_allregime_powell200_20260704_v3.sub — 544 tokens (full)
│       ├── submit_paper_i_hh_recovery_candidate_20260705_powell_nobatch_wave0.sub — 426 tokens (full)
│       └── upload_submit_chtc.sh — 2,740 tokens (lines 1-24 (Upload helper setup and preflight_submit_file() function used by named CHTC actions before upload/submission.), 252-334 (Upload/rsync body, remote condor_submit action patterns, and usage list; enough context to add a narrowly named new action if desired.))
├── pipelines/
│   └── static_adapt/
│       ├── adapt_pipeline.py — 4,810 tokens (lines 2550-2749 (ADAPT settings validation around continuation mode: Phase-III-only features are rejected unless adapt_continuation_mode is phase3_v1.), 3250-3289 (Phase enablement booleans derived from adapt_continuation_mode; phase2_v1 disables Phase III while preserving Phase I/II.), 5680-5769 (Static route identity payload records continuation mode, batching, prune, selector geometry, backend cost, and related fields used for settings-drift audits.))
│       ├── cli_config.py — 503 tokens (lines 1270-1309 (CLI definition for --adapt-continuation-mode choices legacy/phase1_v1/phase2_v1/phase3_v1.))
│       └── engine_support.py — 386 tokens (lines 1110-1142 (Default/override resolution for adapt_continuation_mode, including HH default phase3_v1 and accepted user-requested modes.))
├── test/
│   ├── chtc/
│   │   ├── test_paper_i_hh_recovery_candidate_run_stock_records.py — 2,508 tokens (full)
│   │   └── test_paper_i_hh_strong_weak_anchor_ablation_records.py — 1,944 tokens (full)
│   └── test_paper_i_hh_spsa_budget_ladder_source_lock.py — 8,170 tokens (full)
└── AGENTS.md — 4,368 tokens (lines 1-80 (Root repo contract and navigation router: agent-readable Markdown, local checkout path, MATH/Paper-I run gate routing.), 107-210 (Mandatory Paper run/results gates, visible-target/source-lock baseline rules, and authority order relevant to Paper-I HH CHTC planning.), 296-357 (Git/artifact hygiene, run safety, and response contract for planning run submissions without disturbing unrelated jobs.))


---

## Generated Plan

## Chat Send ✅
- **Chat**: `hh-ablation-batch-8BB0B6` | **Mode**: plan

### Response
## 1. Summary

Implement this as a targeted Paper-I CHTC batch extension, not a broad refactor. Add a dated agent-readable plan, one narrow weak-weak SNAKE mechanism-ablation record generator, preflight recognition/blockers for that new batch prefix, and focused tests proving generated rows preserve the source-locked POWELL/full-meta/HVA contract while varying only the intended mechanism fields. Do **not** edit manuscripts or PDFs. CHTC submission remains gated: generate records → run local fail-closed preflight → report the exact passing preflight bundle → submit only after explicit current-turn launch approval for that bundle.

## 2. Current-state analysis

### Existing responsibilities and data flow

- `generate_paper_i_hh_fullmeta_phase3_singleton_rotosolve_records.py`
  - Owns reusable HH CHTC record-writing utilities:
    - `base.make_row(...)`
    - `base.output_paths(...)`
    - `base.write_lines(...)`
    - `_write_sanitized_src_tarball(...)`
    - `_write_matrix_submit_file(...)`
  - Existing recovery/ordered-batch generators reuse it but some hardcode unsafe cap-3 Pauli-child semantics. For this task, reuse only its structural helpers and avoid its recovery/ordered-batch policy defaults.

- `run_paper_i_hh_spsa_budget_ladder_cell.py`
  - Consumes TSV rows.
  - `build_snake_source_locked_command(...)` is the central source-lock boundary:
    - source command comes from `source_command_args_json`;
    - row fields set optimizer, depth/budget, runtime split/shared pool/class filter;
    - `snake_cli_overrides_json` is the approved mechanism for extra CLI changes;
    - audit fails if effective command differs outside allowed flags.
  - No change should be needed here because continuation, batching, novelty, shared-pool, and source-lock-preferred-sequence changes can already flow through `snake_cli_overrides_json`.

- `preflight_submit.py`
  - `build_preflight_bundle(...)` parses submit file + queue + rows.
  - Existing `_is_hh_fullmeta_phase3_singleton(...)` / `_hh_fullmeta_phase3_singleton_contract_blockers(...)` must **not** be reused directly: they currently conflate recovery/ordered-batch prefixes with Pauli-child cap `3`.
  - New batch needs a new prefix-specific detector so batch target/cap `3` does **not** imply `phase3_runtime_split_max_subset_size=3`.

- `adapt_pipeline.py`, `cli_config.py`, `engine_support.py`
  - `adapt_continuation_mode=phase2_v1` enables Phase I+II and disables Phase III.
  - `adapt_continuation_mode=phase1_v1` enables Phase I only.
  - Non-proxy Phase-III backend cost modes are invalid outside `phase3_v1`, so ablation rows using `phase2_v1`/`phase1_v1` must set `--phase3-backend-cost-mode proxy`.

### Blocking constraints to preserve

- Weak-weak only.
- Source anchor: existing completed combinatorial batch-cap-3 weak-weak SNAKE row, expected default path:
  - `raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v15/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0p005_metric_prune_combinatorial_batch_cap3_children3/json/result.json`
- Existing no-batch row fills “No batching”; it is context evidence, not queued.
- Existing greedy cap-3 row is supplemental context, not queued.
- New CHTC rows preserve:
  - `display_regime=weak-weak`
  - `method_key=snake`
  - `optimizer=POWELL`
  - `adapt_pool=full_meta`
  - no class filter / HVA included
  - `budget=200`
  - final/refit maxiter `200`
  - `max_depth=30`
  - archival Phase-III child split subset cap `1` unless the row is explicitly a child-pool ablation.
- No-second-order and no-Phase-II-novelty rows must use `--adapt-continuation-mode phase2_v1`.
- Phase-I-only rows must be attempted as:
  - macro-only;
  - singleton-child via shared Pauli pool, not invalid Phase-III-only child-pool expansion.
- No manuscript TeX/PDF edits.

## 3. Design

### A. Agent-readable dated plan

Add:

`MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_weak_weak_snake_mechanism_ablation_chtc_plan_20260708.md`

This file records the active run contract and supersedes older local-only/no-CHTC notes only for this specific CHTC mechanism batch.

Required content:

- Scope:
  - weak-weak only;
  - SNAKE only;
  - CHTC diagnostic batch;
  - no manuscript/PDF edits.
- Existing evidence roles:
  - `no_batch_context`
  - `source_anchor_combinatorial_cap3`
  - `supplemental_greedy_cap3`
- New queued rows:
  1. `no_second_order_phase2_only`
  2. `no_phase2_collective_novelty_phase2_only`
  3. `phase1_only_macro_pool`
  4. `phase1_only_singleton_child_pool`
- Explicit cap distinction:
  - Pauli-child subset cap remains `1`;
  - batch target/cap `3` is separate.
- Launch gate:
  - local generator and preflight must pass;
  - user must approve the exact preflight bundle before `condor_submit`.

### B. New generator

Add:

`chtc/phase3_optuna/generate_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py`

#### Key constants

Use a new batch prefix:

```text
paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v1
```

Use `run_class=diagnostic`, not candidate/paper-facing.

#### Data model

Use local dataclass-style records, for example:

```py
ExistingEvidence:
  role: source_anchor_combinatorial_cap3 | no_batch_context | supplemental_greedy_cap3
  result_json: Path
  required: bool
  expected_sha256: str | None

VariantSpec:
  variant_id: str
  continuation_mode: phase2_v1 | phase1_v1
  child_pool_policy: preserve_archival_subset1 | macro_only | shared_singleton_child
  batch_policy: combinatorial_cap3 | not_applicable
  set_flags: mapping[str, str]
  enable_flags: sequence[str]
  remove_bool_flags: sequence[str]
  remove_value_flags: sequence[str]
```

#### Variant definitions

1. `no_second_order_phase2_only`
   - `--adapt-continuation-mode phase2_v1`
   - `--phase3-backend-cost-mode proxy`
   - preserve combinatorial cap-3 batch route from anchor:
     - `--phase2-enable-batching`
     - `--phase3-enable-batching`
     - `--phase2-batch-selection-mode combinatorial_reduced_plane`
     - `--phase3-batch-selection-mode combinatorial_reduced_plane`
     - `--phase2-batch-target-size 3`
     - `--phase2-batch-size-cap 3`
     - `--phase3-batch-target-size 3`
     - `--phase3-batch-size-cap 3`
   - preserve archival child split cap `1`.

2. `no_phase2_collective_novelty_phase2_only`
   - all settings from `no_second_order_phase2_only`;
   - additionally set:
     - `--phase2-novelty-mode legacy_pairwise_v1`
   - Rationale: this matches the repository’s existing “no Phase-II collective novelty” pattern without inventing unverified gamma/schedule flag values.

3. `phase1_only_macro_pool`
   - `--adapt-continuation-mode phase1_v1`
   - `--phase3-backend-cost-mode proxy`
   - explicit macro-only pool exposure:
     - `snake_phase3_runtime_split_mode=off`
     - `shared_pauli_pool_mode=off`
   - disable Phase-2/3 batching because it is not applicable under Phase-I-only:
     - enable `--phase2-no-batching`, `--phase3-no-batching`
     - remove `--phase2-enable-batching`, `--phase3-enable-batching`
     - remove batch mode/target/cap flags.
   - remove `--phase3-source-lock-preferred-sequence`.

4. `phase1_only_singleton_child_pool`
   - `--adapt-continuation-mode phase1_v1`
   - `--phase3-backend-cost-mode proxy`
   - use shared pre-Phase candidate expansion:
     - `shared_pauli_pool_mode=shared_pauli_child_sets_v1`
     - `shared_pauli_pool_symmetry_policy=hard_guard`
     - `shared_pauli_pool_max_subset_size=1`
   - `snake_phase3_runtime_split_mode=off`
   - disable Phase-2/3 batching as not applicable.
   - Do **not** use `adapt_child_pool_expansion_mode`, because it is invalid outside `phase3_v1`.

#### Source anchor loading

The generator must not reconstruct from current defaults. It must:

1. Resolve the source-anchor result JSON.
2. Compute SHA-256.
3. Load the source command from the anchor output root, preferring:
   - sibling `source_lock_command_audit.json` key `effective_command`;
   - then sibling `cell_manifest.json` command/effective-command keys.
4. Fail closed if no command array is found.
5. Use that command array as `source_command_args_json` for all queued variants.
6. Record:
   - `source_anchor_result_json`
   - `source_anchor_sha256`
   - `source_anchor_command_source`
   - `settings_reused_json`
   - `settings_changed_json`
   - `settings_change_reason`

#### Record writing

Do not extend the broad base generator field set. In the new generator, define local fieldnames:

```text
base.OUTPUT_FIELDNAMES + mechanism-ablation-specific fields
```

Use base helper functions for output paths, source tarball creation, and submit-file writing, but write the TSV/manifest locally so mechanism fields are preserved.

Generated files:

```text
chtc/phase3_optuna/input/<batch_id>/paper_i_hh_spsa_budget_ladder_records.tsv
chtc/phase3_optuna/input/<batch_id>/paper_i_hh_spsa_budget_ladder_record_queue.tsv
chtc/phase3_optuna/input/<batch_id>/paper_i_hh_spsa_budget_ladder_record_ids.txt
chtc/phase3_optuna/input/<batch_id>/paper_i_hh_weak_weak_snake_mechanism_ablation_manifest.json
chtc/phase3_optuna/submit_<batch_id>.sub
```

Default CHTC resources:

```text
request_cpus=2
request_memory_mb=32768
request_disk_mb=61440
max_runtime_s=172800
```

### C. Preflight extension

Modify:

`chtc/phase3_optuna/preflight_submit.py`

Add a new prefix constant and detector:

```py
_HH_WEAK_WEAK_SNAKE_MECHANISM_ABLATION_PREFIX =
  "paper_i_hh_weak_weak_snake_mechanism_ablation_"
```

Add:

```py
_is_hh_weak_weak_snake_mechanism_ablation(row)
_hh_weak_weak_snake_mechanism_ablation_blockers(row, repo_root)
```

Required blockers:

- Unknown variant id.
- Any regime other than `weak-weak`.
- Any method other than `snake`.
- Any optimizer other than POWELL / `adapt_optimizer_kind=powell`.
- `budget != 200` or `max_depth != 30`.
- Pool drift:
  - `pool_contract != full_meta_unfiltered`
  - `hh_adaptive_pool_profile != full_meta_unfiltered`
  - class filter not blank/off.
- Missing or mismatched source anchor path/hash.
- Missing `source_command_args_json`.
- Invalid JSON in `snake_cli_overrides_json`.
- Missing `--phase3-source-lock-preferred-sequence` removal from override removal list.
- Non-child-pool-ablation row has Pauli subset cap other than `1`.
- Phase-I singleton row uses `adapt_child_pool_expansion_mode` instead of shared Pauli pool.
- `phase2_v1` rows missing `--phase3-backend-cost-mode proxy`.
- `phase1_v1` rows missing `--phase3-backend-cost-mode proxy`.
- `no_phase2_collective_novelty_phase2_only` missing `--phase2-novelty-mode legacy_pairwise_v1`.
- Phase2-only rows missing combinatorial batch cap-3 fields.
- Phase1-only rows still carrying active Phase2/Phase3 batching flags.

For this new prefix only, preflight should also build the effective source-locked command in a temp preflight directory and block if runner audit status is not `pass`.

### D. Upload/submit helper

Modify:

`chtc/phase3_optuna/upload_submit_chtc.sh`

Add one narrow action:

```text
paper-i-hh-weak-weak-snake-mechanism-ablation-20260708-v1
```

Behavior:

1. Run `preflight_submit_file` on the new submit file.
2. Only then perform the existing remote `condor_submit` pattern.

This action is only for use after the post-preflight user approval checkpoint. Adding the action does not itself authorize launch.

## 4. File-by-file impact

### `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_weak_weak_snake_mechanism_ablation_chtc_plan_20260708.md`

- Add new agent-readable run contract.
- Captures source anchor, context rows, queued rows, cap distinction, and launch gate.
- No dependencies except current support docs.

### `chtc/phase3_optuna/generate_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py`

- Add new local record generator.
- Reuses base CHTC helpers but owns the mechanism row matrix.
- Depends on existing runner source-lock contract and preflight extension.

### `chtc/phase3_optuna/preflight_submit.py`

- Add new prefix detector and blockers.
- Ensure this batch does not pass through old fullmeta singleton cap-3 logic.
- Depends on generator field names and variant ids.

### `chtc/phase3_optuna/upload_submit_chtc.sh`

- Add one named submit action and usage entry.
- Depends on generated submit filename.

### `test/chtc/test_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py`

- Add focused tests for generator rows, effective source-locked commands, and preflight blockers.
- Use temp fake anchor files/command audit where possible so tests do not require large raw outputs.

## 5. Risks and migration

- No persistence schema migration.
- No manuscript/PDF migration.
- Main risk is source-anchor ambiguity: if the v15 anchor command/hash cannot be found locally, generator/preflight must fail closed.
- Existing stale test `test/chtc/test_paper_i_hh_strong_weak_anchor_ablation_records.py` is unrelated and should not be repaired as part of this task.

## 6. Implementation order

1. Add the dated MD plan.
2. Add the generator with temp-anchor-testable command loading.
3. Add preflight detector/blockers for the new prefix.
4. Add focused tests and make them pass.
5. Generate local input/submit files.
6. Run local fail-closed preflight:
   ```bash
   python3 chtc/phase3_optuna/preflight_submit.py \
     --submit chtc/phase3_optuna/submit_paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v1.sub \
     --output-json output/pdf/paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v1_preflight.json
   ```
7. Stop and report the preflight bundle path/status.
8. Submit to CHTC only after explicit current-turn approval for that exact passing bundle.

## Unresolved validation items

- Validate locally whether the intended combinatorial cap-3 anchor path exists and exposes `source_lock_command_audit.json` with `effective_command`.
- Validate the anchor’s actual depth/beam-child fields; generator should record them and still force user-required CHTC row `max_depth=30`.
- Validate `--phase3-batch-target-size` / `--phase3-batch-size-cap` are accepted by the current CLI if not already present in anchor command; if absent from CLI, block rather than silently dropping them.

Files to edit:
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_weak_weak_snake_mechanism_ablation_chtc_plan_20260708.md`
- `chtc/phase3_optuna/generate_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py`
- `chtc/phase3_optuna/preflight_submit.py`
- `chtc/phase3_optuna/upload_submit_chtc.sh`
- `test/chtc/test_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py`


> 💡 Continue this plan conversation with ask_oracle(chat_id: "hh-ablation-batch-8BB0B6", new_chat: false)