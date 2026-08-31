# Investigation: HH subset4 depth11 stall

## Summary
The subset-size 4 HH proof run did not truly hang; it eventually completed with the exact same operator history, final energy, and `|ΔE|` as the subset-size 3 proof run. The real issue is a pathological runtime blow-up in the silent depth-11 phase-3 runtime-split selection block: `full_child_set_scoring` still pays proxy compatibility work for every child set, and that proxy path recomputes tangents pairwise with no cache.

## Symptoms
- The subset-size 4 proof run appeared stalled because `stdout.log` stopped after `hardcoded_adapt_gradient_timing` at depth 11 for nearly two hours.
- The analogous subset-size 3 proof run finished normally.
- Both runs ultimately produced the same answer, so the extra time did not buy a better scaffold.

## Investigation Log

### Phase 1 - Initial assessment
**Hypothesis:** The subset-size 4 run was either truly hung, or the runtime-split child-set path became catastrophically expensive after depth 10.
**Findings:** The active process stayed alive after the last depth-11 gradient log, which pointed away from an immediate crash. The suspicious region was the post-gradient shortlist rescoring path in `adapt_pipeline.py`, especially runtime-split child-set enumeration and full child-set scoring.
**Evidence:**
- Live artifact path: `artifacts/agent_runs/20260415T151449Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_child_set_scoring_subset4_v1/`
- The run command used `--phase3-runtime-split-selection-mode full_child_set_scoring --phase3-runtime-split-max-subset-size 4`.
**Conclusion:** Needs deeper verification in the phase-3 runtime-split scoring code and in the final artifact once it completed.

### Phase 2 - Broad context gathering
**Hypothesis:** The slowdown came from the current-route runtime-split implementation, not from Hamiltonian construction or gradient evaluation.
**Findings:** `context_builder` and the initial oracle pass both pointed at the current-route live selection path after `hardcoded_adapt_gradient_timing`, where macro parents can spawn many child sets and each admissible child set may be fully rescored.
**Evidence:**
- `pipelines/static_adapt/adapt_pipeline.py` imports and uses the scaffold runtime-split/scoring stack.
- `pipelines/scaffold/hh_continuation_generators.py` owns `build_runtime_split_child_sets()`.
- `pipelines/scaffold/hh_continuation_scoring.py` owns `build_full_candidate_features()` and `CompatibilityPenaltyOracle`.
**Conclusion:** The right evidence path is live selection → child-set enumeration → compatibility penalty → full child-set rescoring.

### Phase 3 - Live artifact comparison: subset 3 vs subset 4
**Hypothesis:** If subset size 4 is the real scientific improvement path, it should change the selected scaffold or improve `|ΔE|`; if it is only causing dead work, the final answer will stay the same while runtime explodes.
**Findings:** Subset-size 3 and subset-size 4 produced identical admitted operators through all 11 depths and identical final metrics, but subset-size 4 took far longer and enumerated far more admissible child sets.
**Evidence:**
- Subset-size 3 final metrics: `artifacts/agent_runs/20260415T141934Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_child_set_scoring_v1/json/result.json:419,429,166315,214834-214836`
  - `energy = 0.39264944099263666`
  - `|ΔE| = 0.023740320556460526`
  - `elapsed_s = 546.3929069170263`
  - `admissible_child_set_count = 7198`
  - `probe_parent_win_count = 20`
  - `probe_child_set_count = 11`
- Subset-size 4 final metrics: `artifacts/agent_runs/20260415T151449Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_child_set_scoring_subset4_v1/json/result.json:420,430,166520,325161-325163`
  - `energy = 0.39264944099263666`
  - `|ΔE| = 0.023740320556460526`
  - `elapsed_s = 7171.272687082994`
  - `admissible_child_set_count = 25548`
  - `probe_parent_win_count = 19`
  - `probe_child_set_count = 12`
- I also compared the operator histories directly from both result JSONs and found they were identical through all 11 admitted depths.
**Conclusion:** Increasing the subset cap from 3 to 4 increased admissible child sets by about `3.55x` and total runtime by about `13.1x`, while yielding no change in the admitted scaffold or final energy. That strongly indicates dead work rather than useful search expansion.

### Phase 3 - Exact location of the runtime blow-up
**Hypothesis:** The extra time is inside the silent post-gradient selection block, not in gradient evaluation or optimizer refinement.
**Findings:** Almost all of the extra wall time sits between the depth-11 gradient-timing log and the subsequent `hardcoded_adapt_iter` log. The optimizer and final refit are both fast once selection finally finishes.
**Evidence:**
- Subset-size 3 depth-11 timeline: `artifacts/agent_runs/20260415T141934Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_child_set_scoring_v1/logs/stdout.log:57-65`
  - depth-11 gradient log at `2026-04-15T14:20:28.629145+00:00`
  - chosen operator log at `2026-04-15T14:29:09.388552+00:00`
  - final VQE done at `2026-04-15T14:29:11.877658+00:00`
  - `grad11 -> iter11 = 520.759407 s`
  - `iter11 -> done = 2.489106 s`
- Subset-size 4 depth-11 timeline: `artifacts/agent_runs/20260415T151449Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_child_set_scoring_subset4_v1/logs/stdout.log:57-65`
  - depth-11 gradient log at `2026-04-15T15:17:09.669444+00:00`
  - chosen operator log at `2026-04-15T17:16:15.655795+00:00`
  - final VQE done at `2026-04-15T17:16:16.795894+00:00`
  - `grad11 -> iter11 = 7145.986351 s`
  - `iter11 -> done = 1.140099 s`
- Both runs chose the same depth-11 operator: `paop_full:paop_disp(site=1)::child_set[1,2,3]` at stdout line 58 in both logs.
**Conclusion:** This was not a true hang. The extra time is overwhelmingly inside the silent depth-11 selection block before the chosen operator is emitted.

### Phase 3 - Child-set enumeration trigger
**Hypothesis:** Raising `max_subset_size` increases runtime because the generator layer enumerates many more child-set combinations.
**Findings:** `build_runtime_split_child_sets()` explicitly enumerates all `itertools.combinations(child_rows, subset_size)` for subset sizes `1..max_subset_size`, subject only to symmetry/dedup filtering.
**Evidence:**
- `pipelines/scaffold/hh_continuation_generators.py:520-589`
  - `subset_cap = int(max(1, min(int(max_subset_size), len(child_rows))))`
  - `for subset_size in range(1, subset_cap + 1):`
  - `for subset in itertools.combinations(child_rows, subset_size):`
- The current run changed only the cap from 3 to 4 on the same route, and the final artifact shows the admissible child-set count jumped from `7198` to `25548`.
**Conclusion:** Subset growth is the trigger/base cost, but it does not fully explain the `13.7x` blow-up inside the depth-11 silent block.

### Phase 3 - Dead proxy work inside full child-set scoring
**Hypothesis:** In `full_child_set_scoring` mode, the code still performs proxy compatibility work that no longer determines the winning child set.
**Findings:** For every child set, the live path computes a pairwise compatibility penalty and a proxy score before also exact-scoring the child set. But once `live_full_child_set_scoring_enabled` is true, selection is based on the fully scored child-set record rank key, not on the proxy score.
**Evidence:**
- `pipelines/static_adapt/adapt_pipeline.py:10644-10728`
  - lines `10646-10661`: compute `penalty_total` via `compat_oracle_split.penalty(...)` across all pairs and then `proxy_score`
  - lines `10663-10692`: call `_full_record_for_candidate(...)` for the child set when `live_full_child_set_scoring_enabled`
  - lines `10713-10726`: in `live_full_child_set_scoring_enabled`, selection updates `best_split_record_sort_key` from `scored_child_set_record`; the proxy score is retained only as side bookkeeping
- Git blame shows this whole block comes from commit `efe376c` on `2026-04-13` (`Refactor HH workflows into packaged pipeline surfaces`).
**Conclusion:** The largest actionable defect is dead proxy work: in full-child-set-scoring mode the code still pays for proxy compatibility ranking that does not control the winner.

### Phase 3 - Missing tangent memoization amplifies the dead work
**Hypothesis:** The compatibility penalty path is extra expensive because it recomputes tangent data repeatedly for each record pair.
**Findings:** `compatibility_penalty()` calls `_tangent_data()` separately for both records in every pairwise comparison, and `_tangent_data()` itself is just a fresh compiled-polynomial application / centering calculation with no tangent cache.
**Evidence:**
- `pipelines/scaffold/hh_continuation_scoring.py:920-936`
  - `_tangent_data(...)` computes and returns tangent/F directly; there is no memoization structure.
- `pipelines/scaffold/hh_continuation_scoring.py:1578-1601`
  - `compatibility_penalty()` calls `_tangent_data()` for `record_a` and `record_b` inside each pair evaluation.
**Conclusion:** Lack of tangent caching is the main runtime amplifier inside the dead proxy path.

### Eliminated hypotheses
**Hypothesis:** The subset-size 4 run hit a true infinite loop or got stuck in gradient evaluation.
**Findings:** False. The run completed successfully with exit code 0 and full artifact output. The depth-11 gradient timing line was followed hours later by a normal depth-11 operator choice, a fast optimizer step, a fast final refit, and a successful trajectory and compile scout.
**Evidence:**
- `artifacts/agent_runs/20260415T151449Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_child_set_scoring_subset4_v1/logs/adapt_exit_code.txt`
- `.../logs/finished_at.txt`
- `.../logs/stdout.log:57-65` and final trajectory lines
**Conclusion:** Eliminated.

**Hypothesis:** The extra runtime came from a different optimizer basin or a different physical result.
**Findings:** False. Subset-size 3 and subset-size 4 have the same selected operator history through all admitted depths, the same final energy, the same `|ΔE|`, and the same compile scout.
**Evidence:**
- Result JSON comparisons above
- Direct history comparison performed from both `adapt_vqe.history` arrays
- Compile scout logs for both runs report `FakeNighthawk: 2q=56 depth=136 size=254`
**Conclusion:** Eliminated.

## Root Cause
The subset-size 4 behavior is best described as a **dead-work blow-up inside the phase-3 runtime-split selection path**, not a real hang and not a physics error.

The trigger is legitimate child-set growth from increasing `max_subset_size` (`hh_continuation_generators.py:520-589`). But the severe slowdown comes from the current-route implementation in `adapt_pipeline.py:10644-10728`: even in `full_child_set_scoring` mode, the code still computes pairwise proxy compatibility penalties and proxy scores for every admissible child set before exact-scoring those same child sets and selecting by the fully scored record anyway. That dead proxy path is expensive because `compatibility_penalty()` recomputes `_tangent_data()` pairwise with no tangent memoization (`hh_continuation_scoring.py:920-936`, `1578-1601`).

The artifact evidence matches this exactly:
- subset-size 3 and 4 end with the **same** admitted scaffold and final energy
- subset-size 4 raises admissible child sets from `7198` to `25548`
- the silent depth-11 selection block grows from `520.76 s` to `7145.99 s`
- the optimizer and final refit remain fast once selection finally returns

So the sharpest statement is:

> The subset-4 slowdown is a combinatorial depth-11 child-set blow-up whose cost becomes pathological because `full_child_set_scoring` still performs unused proxy compatibility work, and that proxy path recomputes tangents per pair with no cache.

## Recommendations
1. **Skip proxy compatibility work in `full_child_set_scoring` mode**
   - File: `pipelines/static_adapt/adapt_pipeline.py`
   - Location: `10644-10728`
   - Specific change: when `live_full_child_set_scoring_enabled` is true, do not compute `penalty_total` / `proxy_score` unless explicitly requested for diagnostics or for `parent_family_sum_top2_scoring`.

2. **Add tangent memoization for compatibility scoring**
   - File: `pipelines/scaffold/hh_continuation_scoring.py`
   - Location: `920-936`, `1578-1601`
   - Specific change: cache `_tangent_data()` results inside `CompatibilityPenaltyOracle` (or a shared tangent cache keyed by candidate label / polynomial signature) so repeated pair penalties do not rebuild tangents every time.

3. **Add progress/budget logging around the silent child-set block**
   - File: `pipelines/static_adapt/adapt_pipeline.py`
   - Location: around `child_set_candidates`, compatibility loop, and full child-set scoring loop
   - Specific change: emit `admissible_child_set_count`, child-set size histogram, pair-evaluation budget, and timing for child-set enumeration, compatibility penalties, and full child-set scoring separately.

4. **Add a runtime scaling regression test or benchmark hook**
   - Files: `test/test_adapt_vqe_integration.py` and/or a focused runtime-split test file
   - Specific change: at minimum verify that increasing subset size can change child-set counts without changing the selected history on this known proof point, and add an opt-in benchmark/assertion against unbounded silent runtime growth.

## Post-fix validation (2026-04-16)
- Implemented two targeted code changes:
  - `pipelines/static_adapt/adapt_pipeline.py`: skip proxy child-set compatibility work unless proxy preselection or parent-collapse diagnostics actually need it.
  - `pipelines/scaffold/hh_continuation_scoring.py`: cache per-record tangent data inside `CompatibilityPenaltyOracle` (safe tangent cache only; pair-level penalty caching was intentionally removed after review because schedule penalties depend on insertion context).
- Added focused regression tests:
  - `test/test_adapt_vqe_integration.py`: proxy child-set scoring is only required for proxy preselection or debug.
  - `test/test_hh_continuation_scoring.py`: repeated/mirrored compatibility-penalty calls reuse tangent data.
- Focused test command passed:
  - `pytest -q test/test_hh_continuation_scoring.py -k 'compatibility_penalty' test/test_adapt_vqe_integration.py -k 'phase3_runtime_split_max_subset_size or proxy_child_set_scoring_only_needed_for_proxy_mode'`

### Safe-patch rerun result
- Artifact:
  - `artifacts/agent_runs/20260416T235300Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_child_set_scoring_subset4_postfix_safe_v1/json/result.json`
- Command surface:
  - `L=2`, `U=0`, `dv=1`, `omega0=0.5`, `g_ep=0.5`, `n_ph_max=2`
  - `phase3_v1`, `full_child_set_scoring`, `max_subset_size=4`
  - `adapt_max_depth=30`, `adapt_eps_grad=1e-5`, `adapt_maxiter=800`
- Final metrics:
  - `energy = 0.3980309497964753`
  - `exact = 0.36890912043617613`
  - `|ΔE| = 0.029121829360299167`
  - `stop_reason = drop_plateau`
  - `ansatz_depth = 30`
  - `elapsed_s = 16.96998154098401`
- Interpretation:
  - The pathological subset-4 stall is gone on the safe patch. The runtime fix succeeded operationally, but it did **not** by itself improve the `n_ph_max=2` final energy basin on this current-route strong surface.

### First energy-focused follow-ups after the fix
- More optimizer effort alone did not help:
  - `artifacts/agent_runs/20260416T234500Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_child_set_scoring_subset4_postfix_strong_maxiter1500_v1/json/result.json`
  - `|ΔE| = 0.05888708822936728` (worse)
- Larger subset size also did not help on the same surface:
  - `artifacts/agent_runs/20260416T234800Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_child_set_scoring_subset5_postfix_strong_v1/json/result.json`
  - `|ΔE| = 0.029121829360299167` (identical to subset 4)
- Current best post-fix `n_ph_max=2` result from these follow-ups remains the subset-4/5 strong surface at `|ΔE| ≈ 0.02912`.

### Parallel route checks requested on 2026-04-17
- Parent-family aggregation check:
  - `artifacts/agent_runs/20260417T050500Z_hh_l2_u0_w0p5_lam1p0_current_nph2_parent_family_sum_top2_subset4_v1/json/result.json`
  - `|ΔE| = 0.029121829360299167`, identical to the safe baseline
  - selected-operator history is identical to the safe baseline
- Explicit `full_meta` check:
  - `artifacts/agent_runs/20260417T050500Z_hh_l2_u0_w0p5_lam1p0_current_nph2_full_meta_full_child_set_subset4_v1/json/result.json`
  - `|ΔE| = 0.029121829360299167`, again identical at the final metric level
  - history is not fully identical: first divergence from the safe baseline occurs at depth 14 (`paop_full:paop_disp(site=0)` vs `paop_lf_std:paop_disp(site=0)`), but both routes still collapse to the same final basin
- Interpretation:
  - On this physics point, neither parent-family aggregation nor explicit `full_meta` broadening moved the final `n_ph_max=2` energy below the patched baseline. The route sensitivity exists, but these two specific changes are not enough to break out of the current basin.

## Preventive Measures
- Add phase-3 runtime-split budget diagnostics to every HH proof artifact whenever runtime split is active.
- Keep `max_subset_size` exposed on the CLI, but document that larger values are diagnostic-only until dead proxy work and tangent caching are fixed.
- Separate “selection-by-full-score” paths from “proxy shortlist” paths so the code does not keep paying both costs at once.
- Add a heartbeat log inside any block that can legitimately take minutes, so future runs are distinguishable as “slow” rather than “stalled.”
