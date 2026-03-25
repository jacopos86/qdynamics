# Investigation: Backend-first HH scaffold selection vs logically lean artifacts

## Summary
The repo already supports and has actually run the intended **backend-conditioned big-circuit-first** HH workflow. The current problem is not missing capability; it is an **order-of-operations inversion** in how later docs, examples, and downstream artifacts are being used. The canonical path is: run a broad/backend-conditioned `phase3_v1` ADAPT search for a fixed backend, treat that per-backend winner as the prune parent, then manually prune or audit descendants. The drift came from promoting **report-derived lean pools** and **fixed-scaffold/noise descendants** into examples and defaults, which makes downstream audit subjects look like upstream search inputs.

## Symptoms
- Backend-conditioned HH search exists in code and artifacts, but later docs/examples often start from `pareto_lean_l2` or already-pruned artifacts instead of a broad parent.
- The correct gate-pruned 7-term scaffold is near exact ideally, yet downstream FakeNighthawk noise studies show a large gate/stateprep-dominated execution floor.
- This creates a workflow ambiguity: select scaffolds upstream by fixed-backend burden first, or continue selecting logically lean artifacts and only then ask backend questions.

## Investigation Log

### Initial assessment
**Hypothesis:** The repo already supports backend-conditioned heavy search first, but usage drifted toward lean/downstream artifacts.
**Findings:** Initial oracle pass supported that view. Exact verification confirmed it.
**Evidence:** Verified below in engine code, wrappers, committed docs/examples, and historical artifacts.
**Conclusion:** Confirmed.

### Engine verification - broad pool exists and lean pools are downstream descendants
**Hypothesis:** `adapt_pipeline.py` still contains the broad HH search family, while lean pools are later report-derived reductions.
**Findings:** The broad pool builder is still the primary upstream constructor; later in the same file, `pareto_lean_l2` and `pareto_lean_gate_pruned` are explicitly defined from motif/pruning reports.
**Evidence:**
- Broad upstream pool: `pipelines/hardcoded/adapt_pipeline.py:875-944` defines `_build_hh_full_meta_pool(...)` as the union of `uccsd_lifted + hva + paop_full + paop_lf_full`.
- Lean descendant pool: `pipelines/hardcoded/adapt_pipeline.py:1101-1164` labels `pareto_lean_l2` as a “tighter pruning from L=2 n_ph_max=1 motif analysis” and cites `artifacts/reports/hh_L2_ecut1_scaffold_motif_analysis.md`.
- Gate-pruned descendant pool: `pipelines/hardcoded/adapt_pipeline.py:1170-1249` labels the gate-pruned pool as informed by `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md`.
**Conclusion:** Confirmed. `full_meta` is the broad upstream search manifold; `pareto_lean_l2` and `pareto_lean_gate_pruned` are downstream, report-derived descendants.

### Engine verification - backend-conditioned burden is already in the search loop
**Hypothesis:** The canonical ADAPT engine can condition branch scoring on transpilation burden during search, not only after the fact.
**Findings:** The engine validates backend-aware modes for HH `phase3_v1`, substitutes the transpilation oracle into candidate scoring, and exports backend-conditioned scaffold summaries.
**Evidence:**
- Backend-aware mode contract: `pipelines/hardcoded/adapt_pipeline.py:2011-2034` defines `phase3_backend_cost_mode`, backend name/shortlist, and the HH `phase3_v1` validation gate.
- In-loop burden substitution: `pipelines/hardcoded/adapt_pipeline.py:3348-3387` computes `proxy_compile_est` and then replaces it with `backend_compile_oracle.estimate_insertion(...)` whenever the backend oracle is active.
- Exported backend provenance: `pipelines/hardcoded/adapt_pipeline.py:5758-5847` writes `backend_compile_cost_summary`, `compile_cost_mode`, resolution audit, and final scaffold summary fields into the payload.
**Conclusion:** Confirmed. The backend-conditioned search path is upstream and first-class.

### Wrapper verification - repo-native entrypoints already express backend-first search
**Hypothesis:** There are dedicated wrappers for one-backend and shortlist backend-conditioned HH search.
**Findings:** Both wrappers exist and force the intended upstream regime.
**Evidence:**
- Single backend wrapper: `pipelines/hardcoded/hh_adapt_backend_single.py:14-52` forces `problem=hh`, `adapt_continuation_mode=phase3_v1`, and `phase3_backend_cost_mode=transpile_single_v1`.
- Shortlist wrapper: `pipelines/hardcoded/hh_adapt_backend_shortlist.py:20-47, 144-246` runs one HH search per backend, optionally includes a proxy baseline, and writes JSON/CSV comparisons with best compile/energy backend selection.
**Conclusion:** Confirmed. The repo already exposes the correct upstream workflow surface.

### Historical artifact verification - broad parent first was actually run
**Hypothesis:** The historical “full horse” HH run really was a broad-pool big-parent-first search.
**Findings:** The heavy L=2 scaffold artifact and extracted settings show a broad `full_meta` run with the expected heavy Phase 3/beam-style settings.
**Evidence:**
- Heavy scaffold artifact: `artifacts/json/adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.full_scaffold_for_pruning.json:1-39` records `artifact_kind = full_interrupted_scaffold_for_pruning`, `last_completed_depth = 97`, and `full_scaffold_count = 101`.
- Heavy run settings: `artifacts/json/adapt_hh_L2_ecut1_full_meta_phase3_heavy_powell_20260321T014500Z.selected_ops_from_log.json:1-90` records `adapt_pool = full_meta`, `adapt_continuation_mode = phase3_v1`, `adapt_reopt_policy = windowed`, `adapt_window_size = 999999`, `adapt_window_topk = 999999`, `adapt_full_refit_every = 8`, `adapt_final_full_refit = true`, `adapt_inner_optimizer = POWELL`, `phase1_shortlist_size = 256`, `phase2_shortlist_size = 128`, `phase3_runtime_split_mode = shortlist_pauli_children_v1`, and `phase3_enable_rescue = true`.
**Conclusion:** Confirmed. The historical parent object matches the user’s intended “big circuit first” regime.

### Historical artifact verification - prune-from-parent workflow is real
**Hypothesis:** The repo has already demonstrated “big parent first, manual prune later” with saved-parent audit artifacts.
**Findings:** Saved-parent audit artifacts rank removals, then replay truncated descendants that match baseline energy.
**Evidence:**
- Parent summary: `artifacts/json/hh_l2_prune_saved_parent/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1/summary.json:1-120` records the parent lineage, `adapt_pool = all_hh_meta_v1`, parent compiled burden, and baseline replay metrics.
- Truncation results: `.../partial_summary.json:1-69` shows `accepted_prefix_50` and `accepted_prefix_75` matching the baseline replay energy while using fewer parameters.
- Removal ranking: `.../audit_from_saved_parent.json:1-120` ranks tail units with `removal_penalty = 0.0`, directly supporting manual prune-later logic.
**Conclusion:** Confirmed. Midstream prune-from-parent is an established repo workflow.

### Backend-aware historical artifact verification - fixed-backend winner exists and is the right parent
**Hypothesis:** The repo has already produced per-backend fullhorse winners, which should be treated as the canonical prune parents for fixed-backend work.
**Findings:** The backend shortlist summary exists, and its winner artifact is a broad-pool, backend-conditioned HH scaffold.
**Evidence:**
- Backend shortlist summary: `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_summary.json:1-175` shows `FakeNighthawk`, `FakeFez`, and `FakeMarrakesh` runs under `compile_cost_mode = transpile_single_v1`; `FakeNighthawk` wins both compile and energy with `ansatz_depth = 16`, `compiled_count_2q = 60`, and `abs_delta_e ≈ 5.62e-05`.
- The Nighthawk pruning report explicitly uses that winner as its source: `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md:1-12` names `hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json` as the source.
- Direct key extraction from the winner artifact (`python` readout during investigation) showed:
  - `settings.adapt_pool = full_meta`
  - `adapt_vqe.pool_type = phase3_v1`
  - `adapt_vqe.continuation_mode = phase3_v1`
  - `adapt_vqe.compile_cost_mode = transpile_single_v1`
  - selected backend `FakeNighthawk`
  - `compiled_count_2q = 60`
**Conclusion:** Confirmed. For a fixed backend, the canonical prune parent is the per-backend fullhorse winner artifact, not a later lean/pruned descendant.

### Drift verification - committed docs/examples invert the order
**Hypothesis:** Later docs/examples present downstream lean artifacts as if they were the canonical upstream backend-aware entrypoint.
**Findings:** The backend-aware run guide section says the right backend-first thing in prose, but its actual example commands use `pareto_lean_l2`. A separate report explicitly says to take a lean logical circuit and then ask which backend fits it.
**Evidence:**
- Backend-aware prose and lean-pool examples live side-by-side: `pipelines/run_guide.md:1088-1137` says the search manifold stays logical, transpilation is an inner hardware oracle, and no warm-start/replay is required — but both the single-backend and shortlist commands use `--adapt-pool pareto_lean_l2`.
- The lean-first handoff artifact explicitly inverts the order: `artifacts/reports/lean_logical_circuit_20260321T214822Z.md:1-61` says to use the lean logical circuit and then ask which IBM backend is the best fit.
- Git blame ties the backend-aware example section to the same commit that added the wrappers: `git blame -L 1110,1160 pipelines/run_guide.md` attributes the example lines, including `--adapt-pool pareto_lean_l2`, to commit `6ec111c3` (2026-03-22 15:11:40 -0500).
- Git log confirms the wrappers themselves were introduced in the same commit: `git log --oneline -- pipelines/hardcoded/hh_adapt_backend_single.py pipelines/hardcoded/hh_adapt_backend_shortlist.py` shows `6ec111c chore: commit all changes`.
**Conclusion:** Confirmed. The conceptual drift is committed and surface-level: backend-first machinery was added together with lean-first example usage.

### Downstream separation verification - logical screen and fixed-scaffold/noise surfaces are not upstream scaffold-selection engines
**Hypothesis:** Current logical-screen and fixed-scaffold/noise routes are downstream diagnostics, not the canonical scaffold-selection engine.
**Findings:** Logical screens are explicitly local/noiseless, and staged-noise/fixed-scaffold routes operate on imported descendants.
**Evidence:**
- Logical screen is local/noiseless: `artifacts/json/hh_l2_logical_screen.json:1-24` records `local_only = true`, `noiseless_only = true`, `noise_enabled = false`, and uses `adapt_pool = paop_lf_std`.
- Fixed-scaffold docs are downstream by construction: `pipelines/run_guide.md:1670-1745` documents exported Nighthawk fixed scaffolds plus fixed-scaffold noisy replay/attribution routes as imported-artifact workflows.
- The 7-term investigation already concluded the current 7-term issue is execution-contract limited and downstream: `artifacts/reports/investigation_7term_noise_diagnosis_20260323.md:1-90` attributes the remaining `mthree` floor to gate/stateprep noise on a chosen fixed scaffold, not to upstream search failure.
**Conclusion:** Confirmed. These are downstream evaluation surfaces and should not be treated as the scaffold-selection engine.

### Current-checkout trap verification - staged-noise defaults reinforce confusion, but are partly uncommitted local state
**Hypothesis:** Current local checkout behavior further reinforces the drift by defaulting downstream imported routes to lean/fixed-scaffold descendants.
**Findings:** The current checkout defaults imported full-circuit audits to lean pareto and fixed-scaffold routes to the `circuit_optimized` Nighthawk subject, but part of this fixed-scaffold default path is still uncommitted local work.
**Evidence:**
- Current checkout defaults: `pipelines/hardcoded/hh_staged_noise_workflow.py:186-255` uses `resolve_default_lean_pareto_l2_artifact_json()` for full-circuit import and `resolve_default_hh_nighthawk_circuit_optimized_7term_artifact_json()` for fixed-scaffold import selection.
- Tests encode the same default behavior: `test/test_hh_staged_noise_workflow.py:195-250` expects fixed-scaffold routes to default to `hh_nighthawk_circuit_optimized_7term_v1` and `FakeNighthawk`.
- Git blame shows this default path is mixed committed/uncommitted state: `git blame -L 185,235 pipelines/hardcoded/hh_staged_noise_workflow.py` marks some fixed-scaffold default lines as `Not Committed Yet` while the lean import default lines come from commit `6ec111c3`.
- `git status --short` during investigation showed modified local files including `pipelines/hardcoded/hh_staged_noise_workflow.py`, `pipelines/hardcoded/hh_staged_cli_args.py`, `pipelines/hardcoded/imported_artifact_resolution.py`, and `pipelines/run_guide.md`.
**Conclusion:** Confirmed, but only as a **current-checkout drift trap**, not as definitive repo policy. The stronger policy conclusion must rest on the committed upstream engine/wrapper/docs and historical artifacts.

## Root Cause
The root cause is an **order-of-operations inversion**, not a missing-capability problem.

The repo’s canonical engine and verified historical runs are already **backend-conditioned, big-parent-first**:
- broad upstream pool in `adapt_pipeline.py:875-944`,
- backend-aware compile burden inside branch scoring in `adapt_pipeline.py:3348-3387`,
- backend-conditioned scaffold provenance in `adapt_pipeline.py:5758-5847`,
- dedicated backend wrappers in `hh_adapt_backend_single.py:14-52` and `hh_adapt_backend_shortlist.py:20-47, 144-246`,
- heavy broad-pool parent artifact in `...full_scaffold_for_pruning.json:1-39`, and
- backend-conditioned Nighthawk winner in `hh_backend_adapt_fullhorse_powell_20260322T194155Z_summary.json:1-175`.

But later **report-derived descendants** — `pareto_lean_l2`, `pareto_lean_gate_pruned`, the lean logical handoff, fixed-scaffold exports, compile scouts, and staged-noise replay/attribution routes — were elevated into examples and current working defaults. That made **downstream audit subjects look like upstream search inputs**. The clearest committed surface mismatch is `run_guide.md:1088-1137`: it describes backend-conditioned search correctly, then demonstrates it with `--adapt-pool pareto_lean_l2`.

So the conceptual mismatch is:
- **upstream code and historical artifacts say**: backend-conditioned broad parent first, then prune,
- **later examples and descendant artifacts imply**: choose a lean/logical artifact first, then ask backend questions.

## Eliminated Hypotheses
- **“The repo lacks a backend-conditioned heavy-search path.”** Eliminated by `adapt_pipeline.py:3348-3387`, `hh_backend_compile_oracle.py`, the backend wrappers, and `hh_backend_adapt_fullhorse_powell_20260322T194155Z_summary.json:1-175`.
- **“The right parent for fixed-backend work is already `pareto_lean_l2` or a 7-term fixed scaffold.”** Eliminated by the artifact lineage: those objects are downstream descendants defined from reports (`adapt_pipeline.py:1101-1249`) and the Nighthawk pruning report itself cites the backend-conditioned fullhorse artifact as its source (`artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md:1-12`).
- **“The 7-term FakeNighthawk noise problem means backend-conditioned search upstream is wrong.”** Eliminated by `artifacts/reports/investigation_7term_noise_diagnosis_20260323.md:1-90`, which shows that issue is downstream execution-contract/gate-stateprep limited on a chosen fixed scaffold.

## Recommendations
1. **Re-anchor the canonical upstream object.**
   - For a fixed backend, treat the per-backend fullhorse winner artifact as the canonical prune parent.
   - For Nighthawk, that parent is `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json` (proven by `hh_backend_adapt_fullhorse_powell_20260322T194155Z_summary.json:1-175` and `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md:1-12`).
2. **Treat prune/audit from that parent as the minimum repo-native next step.**
   - The saved-parent lineage (`summary.json:1-120`, `partial_summary.json:1-69`, `audit_from_saved_parent.json:1-120`) is already the native pattern to reuse.
3. **Demote downstream descendants to descendant-evaluation status in guidance and reasoning.**
   - `pareto_lean_l2`, `pareto_lean_gate_pruned`, logical screens, fixed-scaffold exports, compile scouts, staged-noise replay, and attribution should be described as descendants/diagnostics, not the upstream scaffold-selection engine.
4. **Phrase staged-noise default conclusions carefully.**
   - Current checkout defaults are strong evidence of a local drift trap, but because part of that fixed-scaffold path is uncommitted (`git blame` / `git status` evidence above), they should be described as reinforcing confusion rather than defining canonical repo policy.

## Preventive Measures
- In backend-aware docs/examples, explicitly distinguish **canonical broad/backend-conditioned search** from **cost-saving downstream shortcuts** such as `pareto_lean_l2`.
- When describing a descendant artifact, always state its upstream parent lineage (broad heavy parent, backend-conditioned winner, or later prune/export descendant).
- Keep fixed-scaffold/noise routes opt-in and explicit about subject choice; never let an imported default silently substitute for the canonical prune parent in analytical discussions.
- Preserve logs and source-artifact provenance for long backend-conditioned runs so later prune descendants remain traceable to the correct parent.
