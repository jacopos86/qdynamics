## Final Prompt
<taskname="HH Comparator Provenance"/>
<task>
Locate and report the exact current Paper I Hubbard--Holstein comparator continuation provenance for local serial continuation runs. Scope is only HH regimes where SNAKE was extended past iteration 30: weak--strong and intermediate--strong. Comparator methods are Geo-ADAPT and Append-ADAPT parent-pool/macro rows only (`geo_macro_c`, `append_macro_c` / `C_macro_only`); do not run or promote append-singleton/common Phase-0 singleton rows. No CHTC. Do not edit `MATH/paper_details/Paper_I.tex`, do not edit root `Paper_I.pdf`, and do not regenerate PDFs. If artifacts are missing, report blockers rather than inventing settings.
</task>

<architecture>
- Repository guidance is in `AGENTS.md`, `MATH/AGENTS.md`, and `agent_guidance/skills/paper-i-run/SKILL.md`. They require paper-facing runs to start from current visible/provenance artifacts, preserve active manuscript/PDF ownership, fail closed on missing provenance, and leave promotion decisions to the user.
- Active Paper I manuscript provenance is in `MATH/paper_details/Paper_I.tex` lines 1090-1664 and root `Paper_I_provenance.{json,csv,txt}`. The active source/PDF pair is stated as `MATH/paper_details/Paper_I.tex` and root `Paper_I.pdf`; `MATH/paper_details/Paper_I.pdf` is a local/build mirror. RepoPrompt did not expose either PDF file.
- Current physical-lane plot/provenance artifacts are under `MATH/paper_details/figures/paper_i_physical_lane_snake_duplicate_20260708/`. These define SNAKE stitched horizons and current Geo/Append parent-pool comparator sources.
- Weak--strong has an existing depth-50 continuation input manifest in `chtc/phase3_optuna/input/paper_i_hh_weak_strong_depth50_20260708_v1/`. It includes SNAKE, Geo parent, Append parent, and append-singleton/common rows; only the Geo/Append `C_macro_only` records are in user scope.
- `chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_cell.py` is the local serial entry point behind the CHTC wrapper. For Geo/Append rows it calls `python -u -m pipelines.exact_bench.generic_static_benchmark --run-single --family hh --case-id <case_id> --algorithm-id <algorithm_id> --output-dir <output_root>/result` with env overlays from the TSV row.
</architecture>

<selected_context>
- `AGENTS.md`: repo-level active-checkout and paper-facing rerun rules; warns old `/Users/jakestrobel/Documents/...` paths are FileProvider/reference unless revalidated.
- `MATH/AGENTS.md`: active Paper I source/PDF policy; says Paper I visible HH methods are SNAKE, Geo-ADAPT, append-only ADAPT; current rows use unfiltered `full_meta`/HVA-included contract; reruns must start from current visible result and fail closed if source provenance is absent.
- `agent_guidance/skills/paper-i-run/SKILL.md`: mandatory Paper I run gates and canonical current settings. Important for local-only/no-CHTC discipline and source-lock validation.
- `run_guide.md`, `pipelines/run_guide.md`: both are stale pointers to missing `agent_guidance/shared/run-guide.md`; include as evidence of the missing expected guide.
- `MATH/paper_details/Paper_I.tex` lines 1090-1664: active HH machine-readable settings guard, physical-lane duplicate update, continuation status, support paths, visible weak--strong/intermediate--strong HH rows, and singleton comment-only rows.
- `Paper_I_provenance.json`, `Paper_I_provenance.csv`, `Paper_I_provenance.txt`: root sidecars mirroring active Paper I HH provenance. `Paper_I_provenance.json` records `row_policy`, `visible_label_policy`, `continuation_status`, PDF/source hashes, support artifact paths, and SNAKE validation rows. `Paper_I_provenance.csv` records visible SNAKE source JSONs for all regimes.
- `MATH/paper_details/figures/...__weak_strong_snake_depth50_stitched_source.json`: weak--strong SNAKE stitched curve; terminal horizon `k=50`, terminal abs delta E `3.474276848813851e-05`; sources are depth30 base, local depth45 continuation, and fetched CHTC depth50 segment.
- `MATH/paper_details/figures/...__intermediate_strong_snake_depth45_stitched_source.json`: intermediate--strong SNAKE stitched curve; terminal horizon `k=45`, terminal abs delta E `1.8105198894780017e-05`; sources are depth30 base plus local depth45 continuation. No current depth50 intermediate--strong SNAKE segment was found.
- `MATH/paper_details/figures/..._append_parent_only_provenance.{json,csv}`: distilled current plotted provenance for SNAKE plus parent-pool Geo/Append comparators. This is the key comparator source map selected in full.
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md`: current visible-row recovery candidate settings; confirms POWELL maxiter/refit 200, depth30 baseline, `adapt_pool full_meta`, no class filter/full_meta_minus_hva for the current visible contract.
- `pipelines/reporting/update_paper_i_weak_strong_depth50_plot.py`: narrow reporting script that stitched SNAKE continuations while explicitly keeping Geo/Append comparator curves at current 30-iteration parent-pool sources. Use only for provenance logic, not for editing/regenerating outputs.
- `pipelines/reporting/build_paper_i_hh_powell_pool_exposure_support_pdf.py` slices: defines role keys (`geo_macro_c`, `append_macro_c`, `geo_singleton_b1`, `append_singleton_b1`), `C_macro_only`, comparator selected-prefix policy (first prefix within 10% of best trajectory error), and support CSV fields (`source_json`, `source_sha256`, `selected_prefix_k`, etc.).
- `chtc/phase3_optuna/input/paper_i_hh_weak_strong_depth50_20260708_v1/paper_i_hh_spsa_budget_ladder_manifest.json` and TSV/IDs/queue: weak--strong depth50 input batch. Contains the only located comparator continuation rows.
- `chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_cell.py`, `run_paper_i_hh_spsa_budget_ladder_task_apptainer.sh`, and `submit_paper_i_hh_weak_strong_depth50_20260708_v1.sub`: runner and old CHTC submit wrapper. The Python module can be used locally; the `.sub` is provenance only because the user said no CHTC.
- `agent_guidance/skills/shared/scripts/resolve_visible_settings.py`: source-map resolver that fails closed when local `source_json` is absent unless explicitly run in allow-missing audit mode.
</selected_context>

<relationships>
- Active Paper I state: `MATH/AGENTS.md` -> `MATH/paper_details/Paper_I.tex` provenance block -> root `Paper_I_provenance.json/csv/txt` -> figure provenance under `MATH/paper_details/figures/paper_i_physical_lane_snake_duplicate_20260708/`.
- SNAKE horizons: weak--strong `Paper_I.tex`/root provenance -> weak--strong stitched JSON -> base depth30 + local depth45 + fetched depth50 segments, terminal `k=50`; intermediate--strong -> intermediate stitched JSON -> base depth30 + local depth45, terminal `k=45`.
- Current comparator sources: append-parent-only provenance JSON/CSV -> `geo_macro_c` and `append_macro_c` rows -> source JSONs under absent `output/chtc_retrievals/...paper_i_hh_fullmeta_singleton_symmetry_20260630_schedfix_powell.../generic_static_single.json` paths.
- Weak--strong continuation candidate: weak--strong depth50 manifest/records TSV -> `record_id` rows ending `__geo__C_macro_only__native_forced__powell200__depth50_noearlystop__fullmeta_singleton_symmetry` and `__append__C_macro_only__native_forced__powell200__depth50_noearlystop__fullmeta_singleton_symmetry` -> local runner `run_paper_i_hh_spsa_budget_ladder_cell.run_append_geo()` -> `pipelines.exact_bench.generic_static_benchmark`.
- Singleton exclusion: support-builder role specs and weak--strong TSV include `append_singleton_b1` / `B_common_phase0_singleton_hard_guard`; user explicitly said do not run append-singleton.
</relationships>

<facts_to_report>
- Weak--strong SNAKE current continuation horizon is 50. Stitched source SHA in provenance: `142b978edfeeecefe73491348f3dbbf3cf9afcd2da95a96d7e134a4737c484f6`; terminal abs delta E `3.474276848813851e-05`.
- Intermediate--strong SNAKE current continuation horizon is 45, not 50 in the selected provenance. Stitched source SHA: `159057a349ad6bb1bdb56ac69df9d954cf10c52f8838bd2c1a51656ad9b5304a`; terminal abs delta E `1.8105198894780017e-05`.
- Current plotted weak--strong Geo parent row: role `geo_macro_c`, point_count 31, marker_k 8, marker_error `0.04139381176829238`, table_error `0.043646714159213396`, source SHA `25bd762341691810b9fd1a8c7a219a0122af1ae6f098e3e9042f77de87f8b56e`, source path in absent `output/chtc_retrievals/...__weak_strong__geo__C_macro_only__native_forced__powell200__depth30_noearlystop__fullmeta_singleton_symmetry/result/generic_static_single.json`.
- Current plotted weak--strong Append parent row: role `append_macro_c`, point_count 31, marker_k 23, marker_error `0.019354949951648548`, table_error `0.023007750483881706`, source SHA `22618cd2677f43b660513e3490430c8fc215e032afac6916b941a61cfd636350`, source path in absent `output/chtc_retrievals/...__weak_strong__append__C_macro_only__native_forced__powell200__depth30_noearlystop__fullmeta_singleton_symmetry/result/generic_static_single.json`.
- Current plotted intermediate--strong Geo parent row: role `geo_macro_c`, point_count 31, marker_k 8, marker_error `0.026725016240164545`, table_error `0.02897791863102661`, source SHA `de9f7536c6bfca93f4d4fb96997a0605d047af9145e1da75ea40d3ba0644d9e5`, source path in absent `output/chtc_retrievals/...__intermediate_strong__geo__C_macro_only__native_forced__powell200__depth30_noearlystop__fullmeta_singleton_symmetry/result/generic_static_single.json`.
- Current plotted intermediate--strong Append parent row: role `append_macro_c`, point_count 31, marker_k 25, marker_error `0.008150300978024494`, table_error `0.009295831578261082`, source SHA `1c24d474305da2f75121a75f4b38cc1ed365291eb4ee9ab993d6e246c2217a1c`, source path in absent `output/chtc_retrievals/...__intermediate_strong__append__C_macro_only__native_forced__powell200__depth30_noearlystop__fullmeta_singleton_symmetry/result/generic_static_single.json`.
- Weak--strong depth50 comparator TSV rows clone existing depth30 Powell200 comparator rows with intended changes only `max_depth:30->50`, new output root, `disable_drop_stop`, and `disable_benchmark_target_stop`; pool contract `full_meta_unfiltered`; method IDs `static_geo_adapt_vqe` and `static_full_meta_append_adapt_vqe`; optimizer `POWELL`/`powell200`; stop policy `fixed_horizon_no_target_v1`; case `hh_L2_nph4_three_model_sym_weak_strong`; `n_ph_work=4`, `n_ph_ref=7`; same-cutoff exact `-1.13857920035935`, reference exact `-1.138720638074999`.
- Local serial command shape for weak--strong parent rows only:
  `python -u -m chtc.phase3_optuna.run_paper_i_hh_spsa_budget_ladder_cell <record_id> chtc/phase3_optuna/input/paper_i_hh_weak_strong_depth50_20260708_v1/paper_i_hh_spsa_budget_ladder_records.tsv raw_outputs/paper_i_hh_weak_strong_depth50_20260708_v1/<record_id>`
  where `<record_id>` is one of the two `C_macro_only` Geo/Append depth50 records from the TSV. Do not use the append-singleton/common row.
</facts_to_report>

<ambiguities>
- Expected `agent_guidance/skills/paper-i-results/SKILL.md` is absent in this checkout. `agent_guidance/skills/paper-ii-results/SKILL.md` exists but is not the Paper I results skill.
- Expected `agent_guidance/shared/run-guide.md` is absent. Root `run_guide.md` and `pipelines/run_guide.md` are only stale compatibility pointers to that missing file.
- `output/` and `raw_outputs/` are not visible/materialized in the loaded RepoPrompt workspace. Therefore the full support CSV/JSON (`output/pdf/...powell_pool_exposure_support.{csv,json}`), the comparator source `generic_static_single.json` files, and existing local/CHTC partial continuation outputs cannot be inspected here. `resolve_visible_settings.py` would fail closed on those paths unless artifacts are materialized or audit mode is explicitly allowed.
- No intermediate--strong Geo/Append comparator continuation manifest/TSV was found. Only current depth30 parent-pool comparator provenance and the SNAKE depth45 stitched source are present for intermediate--strong. Any intermediate--strong comparator continuation row would need to be derived only after materializing and validating exact current source artifacts; do not invent settings.
- Weak--strong has a depth50 input manifest, but no selected/live output or partial result files were found for the Geo/Append depth50 comparator rows.
- The provenance references root `Paper_I.pdf` and `MATH/paper_details/Paper_I.pdf`, but PDFs were not exposed by RepoPrompt file tree/search in this session.
</ambiguities>

## Selection
- Files: 24 total (22 full, 2 slice)
- Total tokens: 111376 (Auto view)
- Token breakdown: full 94525, slice 16851

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── MATH/
│   ├── paper_details/
│   │   ├── figures/
│   │   │   └── paper_i_physical_lane_snake_duplicate_20260708/
│   │   │       ├── paper_i_physical_lane_snake_duplicate_20260708__intermediate_strong_snake_depth45_stitched_source.json — 1,166 tokens (full)
│   │   │       ├── paper_i_physical_lane_snake_duplicate_20260708__weak_strong_snake_depth50_stitched_source.json — 1,403 tokens (full)
│   │   │       ├── paper_i_physical_lane_snake_duplicate_20260708_append_parent_only_provenance.csv — 3,936 tokens (full)
│   │   │       └── paper_i_physical_lane_snake_duplicate_20260708_append_parent_only_provenance.json — 6,205 tokens (full)
│   │   └── Paper_I.tex — 9,730 tokens (lines 1090-1664 (Active Paper I HH provenance blocks: machine-readable canonical SNAKE settings guard, physical-lane duplicate update, continuation status, support artifact paths, row/label policy, and validation metadata for weak-strong/intermediate-strong.; Visible HH manuscript rows and comments for weak-strong/intermediate-strong SNAKE/Geo-ADAPT/Append-ADAPT values, including singleton rows retained only as provenance comments.))
│   ├── paper_facing/
│   │   └── paper_I_static_scaffold/
│   │       └── paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md — 3,176 tokens (full)
│   └── AGENTS.md — 4,725 tokens (full)
├── agent_guidance/
│   └── skills/
│       ├── paper-i-run/
│       │   └── SKILL.md — 23,531 tokens (full)
│       └── shared/
│           └── scripts/
│               └── resolve_visible_settings.py — 3,743 tokens (full)
├── chtc/
│   └── phase3_optuna/
│       ├── input/
│       │   └── paper_i_hh_weak_strong_depth50_20260708_v1/
│       │       ├── paper_i_hh_spsa_budget_ladder_manifest.json — 928 tokens (full)
│       │       ├── paper_i_hh_spsa_budget_ladder_record_ids.txt — 163 tokens (full)
│       │       ├── paper_i_hh_spsa_budget_ladder_record_queue.tsv — 175 tokens (full)
│       │       └── paper_i_hh_spsa_budget_ladder_records.tsv — 5,555 tokens (full)
│       ├── run_paper_i_hh_spsa_budget_ladder_cell.py — 20,232 tokens (full)
│       ├── run_paper_i_hh_spsa_budget_ladder_task_apptainer.sh — 242 tokens (full)
│       └── submit_paper_i_hh_weak_strong_depth50_20260708_v1.sub — 410 tokens (full)
├── pipelines/
│   ├── reporting/
│   │   ├── build_paper_i_hh_powell_pool_exposure_support_pdf.py — 7,121 tokens (lines 57-170 (Support-builder constants and RoleSpec definitions for Paper-I HH Powell pool exposure: role keys `geo_macro_c`, `append_macro_c`, singleton comparator keys, matrix labels, and comparator plateau tolerance.), 417-716 (Selected-prefix policy and payload construction for comparator rows: first prefix within 10% of best trajectory for Geo/Append, manual prefixes for SNAKE, and selected-prefix cost/fidelity provenance fields.), 812-920 (Derived row construction that maps support sidecars into role_key/matrix/method records and records selected_prefix_k, source JSON, trajectory, and status for each HH regime.), 1085-1168 (Support CSV writer field list showing the concrete provenance columns emitted for each role, including source_json/source_sha256, selection policy, record_id, and trajectory payload.), 1340-1372 (Main argument validation around omitted plot roles and role-key validation, relevant to retained/omitted singleton comparator handling.))
│   │   └── update_paper_i_weak_strong_depth50_plot.py — 6,886 tokens (full)
│   └── run_guide.md — 94 tokens (full)
├── AGENTS.md — 6,044 tokens (full)
├── Paper_I_provenance.csv — 978 tokens (full)
├── Paper_I_provenance.json — 4,658 tokens (full)
├── Paper_I_provenance.txt — 137 tokens (full)
└── run_guide.md — 138 tokens (full)


---

## Generated Answer

## Chat Send ✅
- **Chat**: `hh-comparator-provenance-8434D9` | **Mode**: chat

### Response
**Skill routing:** Used the provided Paper-I run/provenance contract only. `chtc-direct` was not used operationally because the task says **No CHTC**. No manuscript/PDF edit workflow was used.

## Status

- **No runs executed.**
- **No CHTC access/submission.**
- **No edits to `Paper_I.tex`, `Paper_I.pdf`, or PDFs.**
- Scope is only **weak--strong** and **intermediate--strong**, where SNAKE is plotted beyond iteration 30.

## Current SNAKE continuation horizons

- **weak--strong:** stitched SNAKE curve reaches **k=50**, terminal `abs_delta_e = 3.474276848813851e-05`.
  - Source: `...__weak_strong_snake_depth50_stitched_source.json`
  - Selected provenance SHA in file contents: `142b978eb9d196a906dd2261f2cc4f476326bc0d5a3c0d8604be6d142c5b302f`

- **intermediate--strong:** stitched SNAKE curve reaches **k=45**, terminal `abs_delta_e = 1.8105198894780017e-05`.
  - Source: `...__intermediate_strong_snake_depth45_stitched_source.json`
  - Selected provenance SHA in file contents: `159057a9328998d2b1bc6b422c97817ad43f2c4906e22f8b8f7bab2cc60b8ffa`

## Current plotted parent-pool comparator provenance

These are the **current plotted depth-30 parent/macro comparator rows**, not completed continuation outputs.

### weak--strong

- **Geo-ADAPT parent**: `geo_macro_c`, `C_macro_only`
  - `point_count=31`, `marker_k=8`
  - `marker_error=0.04139381176829238`
  - `table_error=0.043646714159213396`
  - source: `...__weak_strong__geo__C_macro_only__native_forced__powell200__depth30_noearlystop__fullmeta_singleton_symmetry/result/generic_static_single.json`
  - SHA in selected provenance: `25bd7623cd9606c238851827b6d54e2fa98dd8dbc9da707ca17edd2678594259`

- **Append-ADAPT parent**: `append_macro_c`, `C_macro_only`
  - `point_count=31`, `marker_k=23`
  - `marker_error=0.019354949951648548`
  - `table_error=0.023007750483881706`
  - source: `...__weak_strong__append__C_macro_only__native_forced__powell200__depth30_noearlystop__fullmeta_singleton_symmetry/result/generic_static_single.json`
  - SHA in selected provenance: `22618c3753c502121b4aeb03d07a76368b0ae8fb948054b827e1284c4def6579`

### intermediate--strong

- **Geo-ADAPT parent**: `geo_macro_c`, `C_macro_only`
  - `point_count=31`, `marker_k=8`
  - `marker_error=0.026725016240164545`
  - `table_error=0.02897791863102661`
  - source: `...__intermediate_strong__geo__C_macro_only__native_forced__powell200__depth30_noearlystop__fullmeta_singleton_symmetry/result/generic_static_single.json`
  - SHA in selected provenance: `de9f753615c18471814ffbc7c2e1fbeb7e9ae399025e90985b7635500e21c01e`

- **Append-ADAPT parent**: `append_macro_c`, `C_macro_only`
  - `point_count=31`, `marker_k=25`
  - `marker_error=0.008150300978024494`
  - `table_error=0.009295831578261082`
  - source: `...__intermediate_strong__append__C_macro_only__native_forced__powell200__depth30_noearlystop__fullmeta_singleton_symmetry/result/generic_static_single.json`
  - SHA in selected provenance: `1c24d47f5376fe6950ceb1e78b833cf4a6da5e318760b9caee395e3e45b3b257`

## Located local-serial continuation inputs

Only **weak--strong** has located continuation input rows:

- Geo record:
  - `paper_i_hh_weak_strong_depth50_20260708_v1__weak_strong__geo__C_macro_only__native_forced__powell200__depth50_noearlystop__fullmeta_singleton_symmetry`
- Append record:
  - `paper_i_hh_weak_strong_depth50_20260708_v1__weak_strong__append__C_macro_only__native_forced__powell200__depth50_noearlystop__fullmeta_singleton_symmetry`

Shared settings from TSV:

- method IDs: `static_geo_adapt_vqe`, `static_full_meta_append_adapt_vqe`
- optimizer: `POWELL`, budget `200`
- `max_depth=50`
- stop policy: `fixed_horizon_no_target_v1`
- pool: `full_meta_unfiltered`
- case: `hh_L2_nph4_three_model_sym_weak_strong`
- `n_ph_work=4`, `n_ph_ref=7`
- intended changes vs anchor: `max_depth:30->50`, new output root, disable drop/benchmark target stopping.

Local serial shape, if later approved:

```bash
python -u -m chtc.phase3_optuna.run_paper_i_hh_spsa_budget_ladder_cell \
  <record_id> \
  chtc/phase3_optuna/input/paper_i_hh_weak_strong_depth50_20260708_v1/paper_i_hh_spsa_budget_ladder_records.tsv \
  raw_outputs/paper_i_hh_weak_strong_depth50_20260708_v1/<record_id>
```

Do **not** use the `append__B_common_phase0_singleton_hard_guard` row.

## Blockers / missing provenance

- No **intermediate--strong** Geo/Append continuation manifest or TSV was found.
- The loaded workspace does not expose/materialize `output/` or `raw_outputs/`, so comparator source JSONs and any partial continuation outputs cannot be rehashed or inspected.
- The selected prompt summary and selected file contents disagree on several SHA values; the values above are from the selected provenance file contents. Rehashing requires materialized artifacts.
- `agent_guidance/shared/run-guide.md` and expected Paper-I results skill content are absent from the loaded context; not blocking this read-only report, but blocking any automated paper-facing update path.

**Files to edit:** None.


> 💡 Continue this question conversation with ask_oracle(chat_id: "hh-comparator-provenance-8434D9", new_chat: false)