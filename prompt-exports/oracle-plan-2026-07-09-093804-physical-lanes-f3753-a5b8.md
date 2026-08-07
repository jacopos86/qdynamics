## Final Prompt
<taskname="Physical Lanes"/>

<task>
Map the current Paper-I static ADAPT/SNAKE physical-operator-lane route and produce a provenance-safe local tmux run/monitoring plan. Do not edit or launch without explicit user approval. Start with HH: inspect the requested HH physical-lane output root and existing HH provenance first, then plan full-coordinate reoptimization every accepted adaptive iteration with no every-8/full-refit overlay. After HH, preserve the requested model order: Hubbard, spin-boson/Rabi, Bose-Hubbard. For spin-boson/Rabi and Bose-Hubbard, present the lane semantics and ask before code or run changes. If any real pool labels classify as `other`, stop and ask, listing exact operator labels.</task>

<architecture>
- CLI surface: `pipelines/static_adapt/cli_config.py` defines `--static-lane-route`, `--physical-lane-shortlist-aggressiveness`, raw shortlist caps/fraction, no-batching aliases, and reoptimization/full-refit flags, then maps them into `_run_hardcoded_adapt_vqe` kwargs.
- Route normalization: `pipelines/static_adapt/lane_routes.py` declares `physical_operator_type`, aggressiveness choices `(2, 3)`, per-problem route variant IDs, and shortlist lane specs.
- Classifiers/provenance constants: `pipelines/contracts/static_provenance.py` owns physical-lane classifier versions, lane lists, problem aliases (`rabi`/`spinboson` -> `spin_boson`), and `classify_static_physical_operator_lane()`.
- Pipeline runtime: `pipelines/static_adapt/adapt_pipeline.py` validates physical-lane problems, converts raw caps/fractions to effective caps/fraction via `ceil(raw/factor)` and `raw_fraction/factor`, records static/physical lane policy summaries, attaches physical metadata to candidate rows/cache payloads, writes `current.json` checkpoints, and embeds `static_lane_policy` / `physical_operator_lane_policy` in final continuation payload.
- Shortlist mechanics: `pipelines/static_adapt/phase_shortlists.py` uses the generic `shortlist_lane_key`, lane list, fallback lane, and lane-health/quota machinery for both algebraic and physical routes.
- Pool labels: `pipelines/static_adapt/builders/primitive_pools.py`, `problem_registry.py`, `src/quantum/operator_pools/spin_boson.py`, and `boson_chains.py` show which pools are used and what labels classifiers see.
- Artifact payloads: `pipelines/static_adapt/output_artifacts.py` builds final `settings`/`adapt_vqe` JSON and persists result JSON/PDF with captured `run_command`; note that lane route details are primarily in `adapt_vqe` continuation/candidate payloads, not necessarily top-level `settings`.
</architecture>

<selected_context>
agent_guidance/skills/paper-i-run/SKILL.md: Paper-I run gates, targeted scan policy, visible-row/source-lock precedence, local replay/current.json caveats, status reporting expectations.
agent_guidance/skills/source-locked-sensitivity/SKILL.md: same-settings-except-X/source-anchor discipline.
agent_guidance/static-adapt/route-a-language.md: Route A terminology and static ADAPT/SNAKE wording constraints.
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_canonical_runtime_settings_draft_20260627.md: July 8 HH correction: forward canonical HH is `--adapt-reopt-policy full`, `--adapt-full-refit-every 0`, `--adapt-final-full-refit false`; historical windowed/every-8/final-refit overlay remains provenance only. Also notes current physical-lane interpretation and historical `effective_command.json`.
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md: current visible HH/Powell recovery and physical-lane context.
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_visible_row_provenance_layers_20260705.md: visible-row > parent source-lock > diagnostic rerun precedence, current Page-8 weak-weak SNAKE anchor settings, and required visible support/source/effective-command fields.
pipelines/contracts/static_provenance.py: exact classifier functions and lane labels for HH, Hubbard, spin_boson/Rabi, and Bose-Hubbard.
pipelines/static_adapt/lane_routes.py: route constants, physical-lane route variant IDs, aggressiveness validation, shortlist lane specs.
pipelines/static_adapt/cli_config.py: exact CLI flags and kwargs mapping for route, aggressiveness, raw caps, no-batching, and full reoptimization/no refit overlay.
pipelines/static_adapt/adapt_pipeline.py: implementation slices for route normalization, cap scaling, policy summaries, classifier payload attachment, candidate cache/static provenance, current checkpoint, final continuation payload, and main output persistence.
pipelines/static_adapt/phase_shortlists.py: generic lane shortlist implementation used by physical route.
pipelines/static_adapt/output_artifacts.py: final output JSON/PDF persistence and run_command capture.
pipelines/static_adapt/builders/problem_registry.py: supported/default `adapt_pool` keys: Hubbard includes `uccsd`; spin_boson/rabi and Bose-Hubbard default to/full support `full_meta`.
pipelines/static_adapt/builders/primitive_pools.py: UCCSD label prefixes and full_meta pool routing/wrapping.
src/quantum/operator_pools/spin_boson.py: spin-boson/Rabi full_meta emitted labels.
src/quantum/operator_pools/boson_chains.py: Bose-Hubbard full_meta emitted labels.
chtc/phase3_optuna/generate_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py and the two 20260708 manifests: nearby source-locked HH physical-lane anchors and SHA/provenance fields.
test/chtc/test_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py: regression expectations that physical anchor rows preserve `--static-lane-route physical_operator_type` and `--physical-lane-shortlist-aggressiveness 3`.
</selected_context>

<route_facts>
- Current intended setting atoms after HH provenance audit: `--static-lane-route physical_operator_type`, `--physical-lane-shortlist-aggressiveness 3`, `--phase1-shortlist-size 42`, `--phase2-shortlist-size 21`, `--phase2-shortlist-fraction 0.4375`, `--phase2-no-batching`, `--phase3-no-batching`, `--adapt-reopt-policy full`, `--adapt-full-refit-every 0`, `--adapt-final-full-refit false`, and `--adapt-max-depth 30` unless the inspected target-route provenance says otherwise.
- With aggressiveness factor `3`, those raw x1.75-less-aggressive caps become effective phase-I cap `14`, phase-II cap `7`, and fraction `0.14583333333333334`.
- HH classifier version: `hh_physical_operator_lanes_v2_uccsd_split`; lanes: `uccsd_single`, `uccsd_double`, `uccsd_correlation`, `electronic_current`, `phonon_displacement`, `phonon_squeeze_relaxation`, `dressed_phonon_correlation`, `hva_hamiltonian_blocks`, `other`.
- Hubbard classifier version: `hubbard_physical_operator_lanes_v1_uccsd_split`; lanes: `uccsd_single`, `uccsd_double`, `other`. User requires Hubbard UCCSD only, so use/audit `--adapt-pool uccsd` and treat `other` as a stop-and-ask condition unless truly unavoidable.
- Spin-boson/Rabi classifier version: `spin_boson_physical_operator_lanes_v2_full_meta_hamiltonian_blocks`; lanes: `emitter_matter`, `boson_linear`, `boson_nonlinear`, `transverse_coupling`, `longitudinal_coupling`, `emitter_y_correlation`, `hamiltonian_blocks`, `other`. These match emitted full_meta labels such as emitter, boson, longitudinal/transverse coupling, nonlinear boson, and Hamiltonian block terms.
- Bose-Hubbard classifier version: `bose_hubbard_physical_operator_lanes_v2_full_meta_hamiltonian_blocks`; lanes: `number_density_interaction`, `onsite_quadrature`, `single_particle_transport`, `intersite_quadrature`, `density_assisted_transport`, `pair_transport`, `hamiltonian_blocks`, `other`. These match emitted full_meta labels for density/interactions, onsite x/p/n terms, hop/current, intersite quadratures, density-assisted transport, pair transport, and Hamiltonian blocks.
</route_facts>

<artifact_findings>
- The exact requested 20260709 raw output roots were not visible in the RepoPrompt workspace during discovery: `raw_outputs/paper_i_hh_physical_operator_lanes_integer_caps_fullreopt_scan_20260709` and `raw_outputs/paper_i_cross_model_physical_operator_lanes_1p75_fullreopt_20260709`. A targeted path search for `raw_outputs` and these names returned no accessible files; do not infer their contents.
- Nearby 20260708 HH physical-lane provenance exists in selected CHTC generator/manifests. Prior physical parent paths include `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_weak/json/result.json`, `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/commands.json`, and `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/source_lock_manifest.json`, with SHA256s in the weak-weak 20260708 manifest.
- Weak-strong depth50 20260708 manifest points to depth30/depth45 physical-lane no-batch continuation sources and records resume/compile-smoke provenance.
- If filesystem access is available later, inspect only the two named 20260709 roots for `current.json`, `result.json`, `run_command.sh`, `run_command.json`, `effective_command.json`, `commands.json`, `source_lock_manifest.json`, manifests, and lane-policy summaries before reconstructing any command. Reuse exact `argv`/effective command templates if present.
</artifact_findings>

<relationships>
- `adapt_pipeline.main()` -> `parse_args()` / `cli_config` -> `_build_run_hardcoded_adapt_vqe_kwargs()` -> `_run_hardcoded_adapt_vqe()`.
- CLI `--static-lane-route physical_operator_type` -> `lane_routes.resolve_static_shortlist_lane_spec()` -> `static_provenance.physical_operator_lanes_for_problem()` / classifier version -> `adapt_pipeline` physical lane summaries and candidate metadata.
- Pool builders emit labels -> `classify_static_physical_operator_lane(label, problem, hh_full_meta_class)` -> candidate fields `physical_operator_lane`, `physical_operator_classifier_label`, `physical_operator_lane_source`, lane counts, and `other_count`.
- `PhaseShortlistRuntime(shortlist_lane_route/key/lanes/fallback_lane)` -> Phase 1/2 lane quota and health shortlists -> final `static_lane_policy` / `physical_operator_lane_policy` provenance.
- Final/live monitoring: current checkpoint fields show status, current energy/error/depth/history/operators; final result embeds continuation payload and output artifact persistence captures `run_command`.
</relationships>

<audit_gates>
- Before any launch, audit real pool labels without running a full ADAPT loop: instantiate/resolve the target problem and pool for each model/regime setting, collect generated term labels (including `full_meta::` wrapping), apply `classify_static_physical_operator_lane()`, summarize lane counts/classifier version, and list exact `other` labels. Stop and ask if any real records are `other`.
- For HH, inspect target-route provenance first and preserve existing Paper-I HH physical-lane route semantics; only the user-approved forward correction is full-coordinate reoptimization every accepted iteration with no periodic/final full-refit overlay.
- For spin-boson/Rabi and Bose-Hubbard, present the already-coded lane semantics above and ask before code or run changes.
- New launches must be local tmux only, and must be preceded by an explicit user approval request. Plumbing fixes may be narrow, py_compile/smoke-tested, and resumed only if needed; no science/settings changes without approval.
- Reports should include abs energy error, status, source JSON paths, and whether evidence is local or CHTC. Do not promote `Paper_I.tex` or `Paper_I.pdf`.
</audit_gates>

<ambiguities>
- Exact 20260709 target-root artifacts are unavailable in the selected workspace, so no exact `run_command.sh`, `effective_command.json`, `current.json`, or `result.json` template from those roots could be verified here.
- `output_artifacts.py` top-level `settings` may not carry `static_lane_route` / `physical_lane_shortlist_aggressiveness`; lane provenance is visible in `adapt_vqe` continuation/candidate-policy payloads. If normalized manifests require top-level duplication, treat that as a narrow plumbing question, not a science/settings change.
- The user’s requested x1.75 less-aggressive settings are implemented by larger raw caps/fraction with existing aggressiveness factor `3`; CLI only allows factor `2` or `3`.
</ambiguities>

## Selection
- Files: 25 total (13 full, 12 slice)
- Total tokens: 112036 (Auto view)
- Token breakdown: full 32483, slice 79553

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── MATH/
│   └── paper_facing/
│       ├── paper_I_static_scaffold/
│       │   ├── paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md — 2,678 tokens (full)
│       │   ├── paper_i_hh_snake_canonical_runtime_settings_draft_20260627.md — 4,415 tokens (lines 1-90 (Canonical HH SNAKE runtime note header and July 8 reoptimization correction: forward policy is full reoptimization each accepted iteration, no every-8/full-refit overlay.), 108-140 (Physical-lane and batch correction: HH current lane route is physical operator type, while older algebraic labels are historical unless artifacts prove otherwise.), 176-184 (Note that each historical result directory contains exact executed command in effective_command.json.), 300-368 (Settings table rows for max depth and reoptimization/full-refit flags, showing the forward canonical values versus historical discrepancies.))
│       │   └── paper_i_hh_visible_row_provenance_layers_20260705.md — 1,466 tokens (lines 1-113 (Visible-row provenance precedence, current Page-8 weak-weak SNAKE anchor settings, visible_support/source/effective-command fields, and no preferred-sequence recovery rule.), 147-165 (Diagnostic recovery caveat: terminal cost convention and selected-label/prefix identity need not match unless explicitly requested.))
│       └── README.md — 1,054 tokens (full)
├── agent_guidance/
│   ├── skills/
│   │   ├── paper-i-run/
│   │   │   └── SKILL.md — 7,359 tokens (lines 1-132 (Paper-I run skill overview, update/reporting expectations, visible-run/source-lock gates, and targeted generated-output scan policy.), 134-150 (Run classification policy distinguishing diagnostic/candidate/paper-facing runs and smoke-test limits.), 160-190 (All-wired Hamiltonian coverage and exact-energy/source-manifest rules for Paper-I full-suite candidate runs.), 209-260 (HH visible-row provenance layers and current Page-8/Powell SNAKE source-row precedence rules.), 280-360 (Active HH Table-III method/pool policy, SNAKE route identity, and required provenance gates for completed rows.), 511-560 (Paper-I SNAKE/static ADAPT rules: suite coverage, shared optimizer/pool contracts, no non-SNAKE overwrites, and HH site normalization.), 693-746 (Strict local replay provenance requirements for current/result JSON sources, current.json caveats, and seed/runtime metadata expectations.), 767-800 (SNAKE prune and shot-proxy reporting taxonomy: reporting contract rather than distinct algorithm variant.))
│   │   ├── paper-ii-results/
│   │   │   └── SKILL.md — 918 tokens (full)
│   │   └── source-locked-sensitivity/
│   │       └── SKILL.md — 2,143 tokens (full)
│   ├── static-adapt/
│   │   └── route-a-language.md — 3,562 tokens (full)
│   ├── README.md — 938 tokens (full)
│   └── skills.md — 1,529 tokens (full)
├── chtc/
│   └── phase3_optuna/
│       ├── input/
│       │   ├── paper_i_hh_weak_strong_depth50_20260708_v1/
│       │   │   └── paper_i_hh_spsa_budget_ladder_manifest.json — 928 tokens (full)
│       │   └── paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v1/
│       │       └── paper_i_hh_spsa_budget_ladder_manifest.json — 1,654 tokens (full)
│       └── generate_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py — 5,460 tokens (lines 1-140 (Source-locked HH weak-weak mechanism-ablation generator header/constants with prior physical-lane parent paths, commands.json, and source_lock_manifest.json anchors.), 460-798 (Physical-lane anchor command extraction from commands.json and source anchor loading/validation.; Record-row provenance fields: source_command_args_json, visible_effective_command_json, settings_reused_json/settings_changed_json, max_depth, and physical-lane source-anchor metadata.))
├── pipelines/
│   ├── contracts/
│   │   └── static_provenance.py — 7,504 tokens (full)
│   ├── scaffold/
│   │   └── hh_continuation_types.py — 1,991 tokens (lines 1-85 (Imports and small runtime/config dataclasses that contextualize HH continuation/provenance payload types.), 200-290 (CandidateFeatures dataclass fields, including static lane and physical_operator_* metadata propagated by the physical-lane route.))
│   └── static_adapt/
│       ├── builders/
│       │   ├── primitive_pools.py — 4,151 tokens (lines 100-219 (UCCSD/full-Hamiltonian primitive pool construction, including UCCSD label prefixes used by Hubbard physical-lane classification.), 638-957 (_build_full_meta_pool routing for HH, Hubbard, spin_boson/rabi, and bose_hubbard, including full_meta label wrapping and Hamiltonian block additions.))
│       │   └── problem_registry.py — 4,779 tokens (lines 1430-1909 (Problem registry entries and supported/default adapt_pool keys; Hubbard defaults/uccsd, spin_boson/rabi full_meta, and bose_hubbard full_meta routing.))
│       ├── adapt_pipeline.py — 28,529 tokens (lines 1475-1595 (_run_hardcoded_adapt_vqe signature showing accepted CLI/pipeline controls for pool choice, reoptimization, shortlist caps, batching, lane route, and physical-lane aggressiveness.), 1930-2359 (Lane route normalization, physical-lane problem validation, raw-to-effective shortlist cap/fraction calculation, and controller cap clamping for physical_operator_type.), 3440-4075 (Static and physical operator lane policy summary construction plus candidate physical-lane classification payload selection and other/lane-count accounting.; Selector metadata propagation for physical lanes through Phase 1 rows and Phase 2 inherited records.), 4200-4289 (Feature scoring cache payload fields for physical operator lane metadata.), 4830-5089 (Candidate-record-cache static provenance fields and PhaseShortlistRuntime construction that binds physical lanes to the lane quota/health shortlist machinery.), 10090-10220 (Current checkpoint writer start: status/settings/energy/current metric fields used for local current.json monitoring.), 10320-10419 (Current checkpoint writer finish: history tail and operator payloads used to inspect live accepted prefixes.), 23180-23399 (Final continuation payload assembly that embeds static_lane_policy and physical_operator_lane_policy summaries in result JSON provenance.), 24268-24310 (main() setup: parse args, capture current command string, resolve problem, and choose output paths.), 24585-24688 (main() finish: build_output_payload and persist_output_artifacts with run_command provenance.))
│       ├── cli_config.py — 9,063 tokens (lines 1-45 (Imports for CLI choices, including STATIC_LANE_ROUTE_CHOICES and PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES from lane_routes.), 1130-1348 (CLI parser core Paper-I static ADAPT flags: problem, pool, continuation, route ID, meta profile, static lane route, physical-lane aggressiveness, max depth, and optimizer.), 1390-1469 (Reoptimization CLI controls: append/full/windowed policy, full-refit cadence, final full-refit toggle, and related optimizer insertion/beam controls.), 1548-1605 (Phase-1 shortlist size and lane-quota pressure flags; raw phase-I cap is divided by physical-lane aggressiveness in adapt_pipeline.), 1740-1795 (Phase-2 shortlist fraction/size and lane-health/quota flags; raw phase-II caps/fraction are divided by physical-lane aggressiveness in adapt_pipeline.), 1988-2025 (No-batching CLI aliases: --phase2-no-batching and --phase3-no-batching map to phase2/phase3_enable_batching false.), 3458-3510 (Alias reconciliation for phase2/phase3 batching flags before building _run_hardcoded_adapt_vqe kwargs.), 3586-3660 (Kwargs mapping for static_lane_route, physical_lane_shortlist_aggressiveness, adapt_reopt_policy, adapt_full_refit_every, and adapt_final_full_refit.), 3744-3830 (Kwargs mapping for phase1_shortlist_size, phase2_shortlist_fraction, phase2_shortlist_size, and lane quota/health controls.), 3906-3920 (Kwargs mapping for effective phase2/phase3 batching controls.))
│       ├── lane_routes.py — 1,231 tokens (full)
│       ├── output_artifacts.py — 6,271 tokens (lines 483-803 (build_output_payload settings/adapt_vqe JSON construction; useful for checking whether lane route settings are exposed at top level or only in adapt payload.), 960-1083 (persist_output_artifacts writes result JSON/PDF and logs run_command provenance for local monitoring and final artifacts.))
│       └── phase_shortlists.py — 5,467 tokens (full)
├── src/
│   └── quantum/
│       └── operator_pools/
│           ├── boson_chains.py — 3,344 tokens (lines 423-682 (Bose-Hubbard block, quadrature, and full_meta pool builders with emitted labels used by physical operator lane classification.))
│           └── spin_boson.py — 2,725 tokens (lines 396-665 (Spin-boson/Rabi block, quadrature, and full_meta pool builders with emitted labels used by physical operator lane classification.))
└── test/
    └── chtc/
        └── test_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py — 2,877 tokens (full)


---

## Generated Plan

## Chat Send ✅
- **Chat**: `physical-lanes-F3753E` | **Mode**: plan

### Response
## Summary

The current code already supports the Paper-I `physical_operator_type` lane route, so this should be handled as a targeted provenance/audit/run-planning workflow, not a refactor. The safe path is: inspect only the requested HH output roots and nearby HH provenance first, source-lock the effective command, audit real pool labels for `other`, materialize local tmux command templates with full-coordinate reoptimization (`--adapt-reopt-policy full`, no every-8/final-refit overlay), and launch nothing until explicit approval.

## Current-state analysis

### Route/control flow

The static ADAPT run path is:

```text
adapt_pipeline.main()
  -> parse_args() from pipelines/static_adapt/cli_config.py
  -> _build_run_hardcoded_adapt_vqe_kwargs()
  -> _run_hardcoded_adapt_vqe()
```

Relevant ownership:

- `cli_config.py` owns CLI flags:
  - `--static-lane-route`
  - `--physical-lane-shortlist-aggressiveness`
  - raw `--phase1-shortlist-size`, `--phase2-shortlist-size`, `--phase2-shortlist-fraction`
  - `--phase2-no-batching` / `--phase3-no-batching`
  - `--adapt-reopt-policy`, `--adapt-full-refit-every`, `--adapt-final-full-refit`
- `lane_routes.py` normalizes `physical_operator_type`, validates aggressiveness `(2, 3)`, and maps each problem to a lane spec.
- `static_provenance.py` owns classifier versions, lane lists, problem aliases, and `classify_static_physical_operator_lane()`.
- `adapt_pipeline.py` applies the route:
  - validates the problem supports physical lanes,
  - scales raw shortlist caps by aggressiveness,
  - attaches physical-lane metadata to `CandidateFeatures`,
  - writes lane policy summaries into final `adapt_vqe.continuation`.
- `phase_shortlists.py` is already generic: it uses `shortlist_lane_key`, lane list, fallback lane, and lane-health/quota machinery for both algebraic and physical routes.

### Existing physical-lane behavior

With:

```text
--static-lane-route physical_operator_type
--physical-lane-shortlist-aggressiveness 3
--phase1-shortlist-size 42
--phase2-shortlist-size 21
--phase2-shortlist-fraction 0.4375
```

the runtime records:

```text
phase1 effective cap = ceil(42 / 3) = 14
phase2 effective cap = ceil(21 / 3) = 7
phase2 effective fraction = 0.4375 / 3 = 0.14583333333333334
```

Final provenance should be checked under:

```text
adapt_vqe.continuation.static_lane_policy
adapt_vqe.continuation.physical_operator_lane_policy
```

not only top-level `settings`, because `output_artifacts.py` may not duplicate lane fields there.

### HH provenance constraints

For HH, provenance precedence is:

```text
visible_row > parent_source_lock > diagnostic_rerun
```

The current HH correction requires forward canonical full-coordinate reoptimization:

```text
--adapt-reopt-policy full
--adapt-full-refit-every 0
--adapt-final-full-refit false
```

Historical `windowed`, every-8 refit, and final-full-refit settings remain provenance only.

The requested 20260709 roots were not available in the provided workspace, so any launch is blocked until those roots are inspected directly:

```text
raw_outputs/paper_i_hh_physical_operator_lanes_integer_caps_fullreopt_scan_20260709
raw_outputs/paper_i_cross_model_physical_operator_lanes_1p75_fullreopt_20260709
```

Nearby HH physical-lane provenance exists from 20260708, especially:

```text
raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_weak/json/result.json
raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/commands.json
raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/source_lock_manifest.json
```

These are fallback provenance references, not permission to infer missing 20260709 contents.

## Design

### 1. Provenance-first HH inspection gate

Before any command is built, inspect only the requested HH roots and known HH sidecars for:

```text
current.json
result.json
run_command.sh
run_command.json
effective_command.json
commands.json
source_lock_manifest.json
manifest*.json
```

Extract and record:

```json
{
  "schema": "physical_operator_lane_source_audit_v1",
  "provenance_layer": "visible_row | target_route_existing | parent_source_lock | diagnostic_rerun",
  "source_json": "...",
  "source_sha256": "...",
  "source_command_json": "...",
  "source_command_sha256": "...",
  "source_lock_manifest": "...",
  "settings_reused": {},
  "settings_changed": {},
  "unresolved_source_fields": [],
  "status": "pass | blocked"
}
```

Blocking conditions:

- requested root missing;
- no effective/source command found;
- source command conflicts with target route and cannot be normalized without adding unapproved variables;
- existing result/current JSON reports `physical_operator_lane_policy.other_count > 0`.

### 2. Real pool-label audit gate

Before launching any full ADAPT loop, instantiate the target problem/pool and classify every real generated label through:

```text
classify_static_physical_operator_lane(label, problem=<problem>)
```

Audit output shape:

```json
{
  "schema": "physical_operator_lane_pool_label_audit_v1",
  "problem": "hh | hubbard | spin_boson | bose_hubbard",
  "adapt_pool": "...",
  "classifier_version": "...",
  "lane_counts": {},
  "other_labels": [],
  "status": "pass | blocked_other_labels"
}
```

If `other_labels` is nonempty, stop immediately and ask the user, listing exact labels. Do not launch, patch, or silently filter.

### 3. Model-specific lane semantics

#### HH first

Classifier version:

```text
hh_physical_operator_lanes_v2_uccsd_split
```

Lanes:

```text
uccsd_single
uccsd_double
uccsd_correlation
electronic_current
phonon_displacement
phonon_squeeze_relaxation
dressed_phonon_correlation
hva_hamiltonian_blocks
other
```

Use the inspected HH source command as the baseline. Apply only approved route atoms:

```text
--static-lane-route physical_operator_type
--physical-lane-shortlist-aggressiveness 3
--phase1-shortlist-size 42
--phase2-shortlist-size 21
--phase2-shortlist-fraction 0.4375
--phase2-no-batching
--phase3-no-batching
--adapt-reopt-policy full
--adapt-full-refit-every 0
--adapt-final-full-refit false
--adapt-max-depth 30
```

unless the inspected target-route provenance proves a different approved value.

#### Hubbard second

Classifier version:

```text
hubbard_physical_operator_lanes_v1_uccsd_split
```

Lanes:

```text
uccsd_single
uccsd_double
other
```

Use/audit:

```text
--adapt-pool uccsd
```

Do not use Hubbard `full_meta` for this route without explicit approval, because non-UCCSD labels are likely to classify as `other`.

#### Spin-boson/Rabi third

Classifier version:

```text
spin_boson_physical_operator_lanes_v2_full_meta_hamiltonian_blocks
```

Semantics:

```text
emitter_matter              emitter_flip / emitter_imbalance / emitter_y
boson_linear                boson_number / displacement / momentum
boson_nonlinear             boson squared, squeeze, n_x/n_p terms
transverse_coupling         transverse x/p/number-weighted flip terms
longitudinal_coupling       longitudinal x/p/number-weighted imbalance terms
emitter_y_correlation       x_sq_emitter_y / p_sq_emitter_y
hamiltonian_blocks          ham_full / ham_term / ham_unit_term
other                       stop-and-ask fallback
```

Ask before code or run changes.

#### Bose-Hubbard fourth

Classifier version:

```text
bose_hubbard_physical_operator_lanes_v2_full_meta_hamiltonian_blocks
```

Semantics:

```text
number_density_interaction  n, n_sq, number, interaction, staggered_number, nn
onsite_quadrature           x, p, x_sq, p_sq, squeeze, n_x/n_p terms
single_particle_transport   hop / current
intersite_quadrature        xx / pp
density_assisted_transport  density_hop / density_current
pair_transport              pair_hop / pair_current
hamiltonian_blocks          ham_full / ham_term / ham_unit_term
other                       stop-and-ask fallback
```

Ask before code or run changes.

### 4. Local tmux run plan

No launch occurs until the user approves the audited command set.

Recommended session:

```text
tmux session: paper_i_physlanes_local_20260709
```

Windows:

```text
audit       targeted root/provenance inspection and pool-label audit
hh          HH local runs, default sequential unless user approves parallelism
monitor     current.json/result.json status checks
summary     post-run lane/result audit
```

Per-run output layout:

```text
raw_outputs/<approved_root>/<model_or_regime>/
  json/current.json
  json/result.json
  logs/stdout.log
  logs/stderr.log
  provenance/source_audit.json
  provenance/pool_lane_audit.json
  provenance/settings_diff_audit.json
  provenance/effective_argv.json
```

Monitoring reads:

- live:
  - `checkpoint.depth`
  - `checkpoint.stop_reason`
  - current energy/error fields when present
  - `history_tail`
  - accepted `operators`
- final:
  - `adapt_vqe.abs_delta_e`
  - `adapt_vqe.ansatz_depth`
  - `adapt_vqe.stop_reason`
  - `adapt_vqe.continuation.static_lane_policy`
  - `adapt_vqe.continuation.physical_operator_lane_policy`

Final pass conditions:

```text
static_lane_policy.route == physical_operator_type
physical_operator_lane_policy.enabled == true
classifier_version matches expected problem
other_count == 0
raw/effective caps match expected values
adapt_reopt_policy == full
adapt_full_refit_every == 0
adapt_final_full_refit == false
batching disabled through both aliases/effective settings
```

## File-by-file impact

No production files should be changed for this plan.

Read-only dependencies:

- `pipelines/static_adapt/cli_config.py` — CLI flag source.
- `pipelines/static_adapt/lane_routes.py` — route/aggressiveness/lane spec source.
- `pipelines/contracts/static_provenance.py` — classifier source of truth.
- `pipelines/static_adapt/adapt_pipeline.py` — runtime provenance and checkpoint behavior.
- `pipelines/static_adapt/phase_shortlists.py` — generic lane shortlist mechanics.
- `pipelines/static_adapt/output_artifacts.py` — final JSON/PDF persistence.
- `pipelines/static_adapt/builders/problem_registry.py` — supported/default pools.
- `pipelines/static_adapt/builders/primitive_pools.py` — Hubbard/UCCSD/full-meta label origins.
- `src/quantum/operator_pools/spin_boson.py` — spin-boson/Rabi labels.
- `src/quantum/operator_pools/boson_chains.py` — Bose-Hubbard labels.
- HH provenance docs/manifests under `MATH/paper_facing/paper_I_static_scaffold/` and `chtc/phase3_optuna/`.

Only generated artifacts should be created after approval, under the approved `raw_outputs/...` roots.

## Risks and migration

No schema migration is required. Main risks:

- missing requested 20260709 roots block HH source-locking;
- any `other` labels block launch;
- top-level JSON `settings` may omit lane details, so final audits must inspect `adapt_vqe.continuation`;
- full-coordinate reoptimization can change runtime substantially versus historical windowed/every-8 rows;
- no-batching is a route change and must be recorded in `settings_changed`.

## Implementation order

1. Inspect requested HH roots only; build `source_audit.json`.
2. Audit HH real pool labels; stop if any `other`.
3. Build HH effective argv from inspected source command plus approved overrides; write `settings_diff_audit.json`.
4. Present HH audit/argv summary for explicit launch approval.
5. If approved, launch HH local tmux runs and monitor `current.json`.
6. After HH completes, verify final lane policy, `other_count == 0`, full-reopt/no-overlay settings, and energy/depth status.
7. Repeat audit planning for Hubbard with `--adapt-pool uccsd`.
8. Present spin-boson/Rabi lane semantics and ask before any code/run changes.
9. Present Bose-Hubbard lane semantics and ask before any code/run changes.

Approval checkpoint: I would not launch or edit anything until you explicitly approve the HH audited command set.


> 💡 Continue this plan conversation with ask_oracle(chat_id: "physical-lanes-F3753E", new_chat: false)