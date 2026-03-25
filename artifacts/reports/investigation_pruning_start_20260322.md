# Investigation: Begin the pruning from the backend-conditioned HH parent

## Summary
The correct place to begin pruning is the backend-conditioned FakeNighthawk parent artifact, not the later lean/fixed-scaffold descendants. The repo still supports backend-conditioned parent generation and matched-family replay, but the historical saved-parent prune workflow itself is only partially present in the current checkout: its audit/rank/plan helper modules exist in git history and in historical runstate artifacts, but not as importable working-tree modules.

## Symptoms
- There are multiple plausible "pruning" entrypoints in the repo, but they do not mean the same thing.
- The current Nighthawk pruning/report artifacts all trace back to a backend-conditioned fullhorse parent, yet the current working-tree pruning script mixes threshold pruning, HF re-ADAPT, and downstream fixed-scaffold export.
- A historical saved-parent prune workflow clearly existed, but its helper modules are missing from the current checkout.

## Investigation Log

### Initial assessment - current parent and current prune reports
**Hypothesis:** The correct pruning start point should be the backend-conditioned parent, not the later lean descendants.
**Findings:** The current shortlist summary identifies FakeNighthawk as the best compile/energy backend and points to the fullhorse parent JSON. The current Nighthawk pruning reports explicitly use that fullhorse parent as their source.
**Evidence:**
- `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_summary.json:18-31` shows `best_compile_backend.label = "FakeNighthawk"`, `compile_cost_mode = "transpile_single_v1"`, `ansatz_depth = 16`, `compiled_count_2q = 60`, and `output_json = ...hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json`.
- `artifacts/reports/hh_prune_nighthawk_final_20260322.md:3-4` names source `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json`.
- `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md:3-4` names source `hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json`.
**Conclusion:** Confirmed. The backend-conditioned FakeNighthawk fullhorse artifact is the upstream prune parent.

### Current-code inspection - what "pruning" means today
**Hypothesis:** The working tree may already contain a canonical post-parent prune workflow.
**Findings:** Three different surfaces exist, but they serve different roles:
1. `adapt_pipeline.py` contains in-loop phase1 pruning during ADAPT search.
2. `hh_prune_nighthawk.py` performs post-hoc threshold pruning and also HF restart re-ADAPT.
3. `hh_vqe_from_adapt_family.py` is the canonical matched-family replay runner for imported ADAPT descendants.
**Evidence:**
- `pipelines/hardcoded/adapt_pipeline.py:5380-5699` implements in-loop phase1 pruning, including candidate ranking, removal evaluation, `apply_pruning(...)`, post-prune refit, and rollback logic.
- `pipelines/hardcoded/hh_prune_nighthawk.py:742-799` defines `run_fixed_scaffold_vqe(...)` as post-hoc scaffold refit on surviving indices.
- `pipelines/hardcoded/hh_prune_nighthawk.py:807-826` documents `run_readapt_phase3(...)` as starting fresh from HF, explicitly "not warm-started from the fullhorse converged state".
- `pipelines/hardcoded/hh_prune_nighthawk.py:850-905` hard-codes `--adapt-pool full_meta`, backend-conditioned phase3 settings, and no `--adapt-ref-json`, confirming HF restart semantics.
- `pipelines/hardcoded/hh_vqe_from_adapt_family.py:1157-1188` resolves `--generator-family match_adapt` from metadata/fallback.
- `pipelines/hardcoded/hh_vqe_from_adapt_family.py:1199-1236` loads the imported ADAPT payload and exact energy, establishing replay-from-handoff semantics.
- `pipelines/run_guide.md:1424-1440` documents `hh_vqe_from_adapt_family.py` as the canonical ADAPT-family replay path.
**Conclusion:** Eliminated the idea that the current working tree has one canonical prune-start surface. It has partial pieces with different semantics.

### Historical workflow recovery - saved-parent pruning was real
**Hypothesis:** The historical/principled prune workflow was a saved-parent audit -> ablation handoff -> matched-family replay pipeline.
**Findings:** The strongest evidence is a preserved runstate script that reconstructs a converged parent, computes audit rows, ranks accepted ADAPT units, builds prune plans, writes descendant handoffs, and replays each descendant.
**Evidence:**
- `artifacts/runstate/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.py:41-44` builds HH context from saved warm/handoff artifacts.
- `artifacts/runstate/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.py:48-77` reconstructs replay terms from the saved parent and rebuilds accepted operator units.
- `artifacts/runstate/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.py:88-112` computes audit rows and ranks accepted ADAPT units.
- `artifacts/runstate/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.py:177-183` calls `hpw._build_ranked_prune_plans(...)`.
- `artifacts/runstate/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.py:203-220` writes `handoff_<ablation>.json`, builds replay cfg, and runs `hh_vqe_from_adapt_family.py` for each descendant.
- `artifacts/runstate/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1.py:242-258` writes `summary.json`, `summary.csv`, and Pareto rows.
- `artifacts/json/hh_l2_prune_saved_parent/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1/partial_summary.json:1-56` shows completed descendant replays such as `accepted_prefix_75`, `accepted_prefix_50`, and weakest-drop variants.
- `artifacts/json/hh_l2_prune_saved_parent/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1/audit_from_saved_parent.json:1-89` stores ranked ADAPT-unit audit rows ordered by removal penalty.
**Conclusion:** Confirmed. The historical/principled prune workflow was saved-parent descendant ablation validated by matched-family replay.

### Historical helper semantics - current checkout is incomplete
**Hypothesis:** The helper modules used by the runstate script may still exist in the checkout or may have been moved cleanly.
**Findings:** They do not exist in the current working tree and fail to import. Git history shows they were real and exposes their semantics.
**Evidence:**
- `pipelines/exact_bench/` current tree contains only `README.md`, `benchmark_metrics_proxy.py`, `cross_check_suite.py`, `hh_noise_hardware_validation.py`, `hh_noise_robustness_seq_report.py`, `hh_seq_transition_utils.py`, `noise_oracle_runtime.py`, and `statevector_kernels.py`; the `hh_l2_*workflow.py` files are absent.
- Python import check in current checkout failed for:
  - `pipelines.exact_bench.hh_l2_heavy_prune_workflow`
  - `pipelines.exact_bench.hh_l2_logical_screen_workflow`
  - `pipelines.exact_bench.hh_l2_stage_unit_audit_workflow`
- `git log --all -- pipelines/exact_bench/hh_l2_heavy_prune_workflow.py` shows historical commit presence, latest seen at commit `4d16d05ba5d60a84aefb69a56da7712208f217ef`.
- Historical helper semantics from commit `4d16d05ba5d60a84aefb69a56da7712208f217ef`:
  - `pipelines/exact_bench/hh_l2_logical_screen_workflow.py@4d16d05:385-416` ranks accepted units by `(removal_penalty, delta_energy_from_previous, -final_order_index, unit_index)`.
  - `pipelines/exact_bench/hh_l2_logical_screen_workflow.py@4d16d05:607-665` builds replay configs for descendant handoffs.
  - `pipelines/exact_bench/hh_l2_stage_unit_audit_workflow.py@4d16d05:481-571` reconstructs accepted insertion order from ADAPT history.
  - `pipelines/exact_bench/hh_l2_stage_unit_audit_workflow.py@4d16d05:695-763` computes per-unit audit rows including `delta_energy_from_previous` and `removal_penalty`.
  - `pipelines/exact_bench/hh_l2_heavy_prune_workflow.py@4d16d05:319-451` builds prune plans: baseline, accepted-prefix truncations, weakest single drops, weakest cumulative drops.
  - `pipelines/exact_bench/hh_l2_heavy_prune_workflow.py@4d16d05:455-495` builds descendant handoff payloads by filtering `adapt_vqe.operators` and `optimal_point`, and attaching `meta.heavy_prune_ablation`.
  - `pipelines/exact_bench/hh_l2_heavy_prune_workflow.py@4d16d05:498-543` computes Pareto nondominance on `(replay_delta_abs, replay_transpiled_cx_count, replay_transpiled_depth)`.
**Conclusion:** Confirmed. The historical saved-parent prune workflow is conceptually clear and evidenced, but not intact as importable working-tree code.

### Eliminated hypothesis - the current Nighthawk pruning script is the canonical prune-start contract
**Hypothesis:** `hh_prune_nighthawk.py` might already be the repo-native general pruning entrypoint.
**Findings:** It is a useful Nighthawk-specific heuristic tool, but not the same thing as the saved-parent audit/ablation/replay workflow.
**Evidence:**
- `pipelines/hardcoded/hh_prune_nighthawk.py:807-826` says `run_readapt_phase3(...)` starts from HF instead of the converged parent.
- `pipelines/hardcoded/hh_prune_nighthawk.py:1280-1362` CLI/main surface mixes `scaffold`, `readapt`, `both`, and `export_fixed_scaffolds` modes in one script.
- `pipelines/run_guide.md:1649-1763` documents fixed Nighthawk 7-term export/noisy replay/attribution as downstream imported fixed-scaffold diagnostics, not parent pruning.
**Conclusion:** Eliminated. `hh_prune_nighthawk.py` is not the canonical general prune-start contract.

## Root Cause
The ambiguity comes from a split repo state:

1. **Upstream parent generation is current and intact.** The repo can produce backend-conditioned HH parents through Phase-3 backend compile scoring (`artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_summary.json:18-31`, `pipelines/hardcoded/adapt_pipeline.py:1928-2217`, `3290-3679`).
2. **Downstream validation is current and intact.** The repo still has canonical matched-family replay for imported ADAPT descendants (`pipelines/hardcoded/hh_vqe_from_adapt_family.py:1157-1236`, `1495-1568`, `1768-1827`; `pipelines/run_guide.md:1424-1440`).
3. **The middle saved-parent prune layer is historically real but missing from the working tree.** The preserved runstate and git-history helpers show that the intended workflow was: audit accepted units -> rank by removal penalty/prefix gain -> build descendant handoff JSONs -> validate each descendant with matched-family replay. But those helper modules are no longer importable in the current checkout.
4. **A different current script (`hh_prune_nighthawk.py`) partially filled the gap**, but with different semantics: theta-threshold pruning, fixed-scaffold refit, and HF restart re-ADAPT. That makes it useful, but it does not mean the historical saved-parent pruning workflow still exists intact.

So the real issue is not lack of a parent or lack of replay. The real issue is that the repo currently preserves **only parts** of the original backend-first pruning chain, which makes "begin the pruning" look more ambiguous than it should.

## Recommendations
1. **Use the backend-conditioned FakeNighthawk fullhorse artifact as the prune parent.**
   - Parent: `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_fakenighthawk.json`
   - Evidence: `artifacts/json/hh_backend_adapt_fullhorse_powell_20260322T194155Z_summary.json:18-31`, `artifacts/reports/hh_prune_nighthawk_final_20260322.md:3-4`.
2. **Describe pruning semantics using the historical saved-parent audit contract, not theta-threshold pruning.**
   - Rank by accepted-unit audit metrics (`removal_penalty`, then `delta_energy_from_previous`, then final-order tie-breaks).
   - Build descendant ablations such as accepted-prefix truncations and weakest accepted-unit drops.
   - Validate descendants with `hh_vqe_from_adapt_family.py` in matched-family replay mode.
3. **Be explicit that the current checkout only partially supports that workflow directly.**
   - Present: parent generation, matched-family replay.
   - Missing: current importable audit/rank/plan/orchestration modules.
4. **Minimum safe next step:** manual saved-parent-style descendant construction plus matched-family replay.
   - Start from the FakeNighthawk parent.
   - Build one conservative descendant handoff (for example accepted-prefix or weakest accepted-unit drop).
   - Replay it with `hh_vqe_from_adapt_family.py`.
   - Compare replay `abs_delta_e`, transpiled 2Q count, and transpiled depth against the parent.

## Preventive Measures
- Keep a single documented prune-start contract in the working tree rather than splitting it across reports, runstate artifacts, and heuristic scripts.
- If historical workflow helpers are retired, preserve their replacement entrypoint in-tree and update `pipelines/run_guide.md` to point to it explicitly.
- Avoid presenting downstream fixed-scaffold exports as if they were the upstream backend-first pruning entrypoint.
