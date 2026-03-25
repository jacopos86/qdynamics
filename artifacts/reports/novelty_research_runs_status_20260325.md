# Novelty Research Runs Status — 2026-03-25

## Plain-English summary of the claims tested

### Claim 1: the repo's default staged HH workflow is a good default on this target problem
What that claim means:
- warm start with `hh_hva_ptw`
- use the prepared state as the ADAPT reference
- start depth-0 HH ADAPT from the narrow HH core (`paop_lf_std`) under `phase3_v1`
- do **not** start from the broad `full_meta` pool at depth 0

Why I tested it:
- the root `README.md` presents this as the canonical staged HH default path for new work
- support doc:
  - `README.md:25-35`
  - `README.md:192-204`

How I tested it:
- ran `pipelines/hardcoded/hh_staged_noiseless.py` at
  - `L=2, t=1.0, U=4.0, dv=0.0, omega0=1.0, g_ep=0.5, n_ph_max=1`
- artifact:
  - `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1.json`

Result:
- final `ΔE_abs = 1.01796943698e-02`
- failed the repo hard gate `ΔE_abs < 1e-4`
- conclusion: this default-path claim is **not supported on this target point**

### Claim 2: re-optimizing previously selected parameters helps enough to matter
How I tested it originally:
- kept the staged HH wrapper and the same physics
- changed only the reoptimization policy:
  - append-only baseline
  - sliding window refit
  - full-prefix refit
- artifacts:
  - append-only baseline: `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1.json`
  - sliding window: `artifacts/json/campaign_A3b_L2_windowed_phase3_v1.json`
  - full-prefix refit: `artifacts/json/campaign_A3c_L2_full_phase3_v1.json`

Observed result on that wrapper surface:
- append-only baseline: `ΔE_abs = 1.01796943698e-02`
- sliding window refit: `ΔE_abs = 7.97336411689e-03`
- full-prefix refit: `ΔE_abs = 8.90929211957e-03`

Corrected interpretation after Oracle review:
- this staged wrapper surface is **not** a clean selector assay because it bundles warm-start state, staged pool logic, replay, and downstream workflow logic around the selector
- therefore these runs should be treated as **wrapper-contaminated / inconclusive for selector-level research claims**
- the only safe takeaway is that a policy change moved the endpoint somewhat on this wrapper surface; it is **not** yet a clean selector-level result

Corrected next-step assay:
- rerun this claim on direct `pipelines/hardcoded/adapt_pipeline.py`
- use the stabilized good-basin surface:
  - `full_meta`
  - `phase3_v1`
  - `POWELL`
  - `windowed` refit baseline
  - wide shortlist (`phase1_shortlist_size=256`, `phase2_shortlist_fraction=1.0`, `phase2_shortlist_size=128`)
  - `phase3_enable_rescue=true`
  - `phase3_runtime_split_mode=off`
  - `phase3_lifetime_cost_mode=phase3_v1`

### Claim 3: the lifetime-aware selector penalty helps on this target point
How I tested it originally:
- kept the staged HH wrapper and the same physics
- changed only `phase3_lifetime_cost_mode` from the default setting to `off`
- artifacts:
  - baseline: `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1.json`
  - lifetime-off run: `artifacts/json/campaign_A4b_L2_lifetime_off_phase3_v1.json`

Observed result on that wrapper surface:
- lifetime on baseline: `ΔE_abs = 1.01796943698e-02`
- lifetime off: `ΔE_abs = 1.01796943698e-02`

Corrected interpretation after Oracle review:
- this is a **bad-surface null** and should not be cited against the lifetime-cost idea
- because the staged wrapper is already stuck in a bad regime, a null there does not tell us whether the lifetime-cost term matters on the actual selector surface

Corrected next-step assay:
- rerun on direct `pipelines/hardcoded/adapt_pipeline.py`
- keep the good-basin direct baseline fixed and toggle only:
  - `phase3_lifetime_cost_mode=phase3_v1`
  - versus `phase3_lifetime_cost_mode=off`
- keep runtime split **off** during this pair so lifetime and split are not entangled

### Claim 4: runtime split of shortlisted macro-generators helps on this target point
How I tested it originally:
- kept the staged HH wrapper and the same physics
- changed only `phase3_runtime_split_mode` from `off` to `shortlist_pauli_children_v1`
- artifacts:
  - split off baseline: `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1.json`
  - split on run: `artifacts/json/campaign_A5b_L2_split_on_phase3_v1.json`

Observed result on that wrapper surface:
- split off baseline: `ΔE_abs = 1.01796943698e-02`
- split on: `ΔE_abs = 1.01796943698e-02`

Corrected interpretation after Oracle review:
- this staged comparison should be **downgraded**
- the valid split comparison is the already-completed direct wide-shortlist pair, not the staged wrapper pair

Valid direct comparison to keep:
- split on direct run: `artifacts/json/campaign_A7c_L2_shortlist_wide_phase3_v1.json`
- split off direct run: `artifacts/json/campaign_A7e_L2_shortlist_wide_runtime_split_off_phase3_v1.json`

Corrected interpretation of the direct pair:
- split on direct run: `artifacts/json/campaign_A7c_L2_shortlist_wide_phase3_v1.json`
  - `ΔE_abs = 5.61782346440e-05`
  - summed `history[*].measurement_cache_stats.groups_new = 9`
  - summed `history[*].compile_cost_total = 298`
  - `selected_child_total = 0`
- split off direct run: `artifacts/json/campaign_A7e_L2_shortlist_wide_runtime_split_off_phase3_v1.json`
  - `ΔE_abs = 5.61814084024e-05`
  - summed `history[*].measurement_cache_stats.groups_new = 10`
  - summed `history[*].compile_cost_total = 369`
  - `selected_child_total = 0`
- because `selected_child_total = 0` in the split-enabled run, the clean reading on this L=2 point is that runtime split was **dormant**, not that split logic was broadly disproven

### Claim 5: backend-conditioned compile-aware search changes the chosen scaffold or executable burden
How I tested it originally:
- ran the same direct HH ADAPT search under
- proxy compile cost
- requested backend-conditioned transpile cost for `ibm_boston`
- requested backend-conditioned transpile cost for `ibm_miami`
- artifacts:
  - summary: `artifacts/json/campaign_A6_L2_backend_summary.json`
  - proxy baseline: `artifacts/json/campaign_A6_L2_backend_proxy_baseline.json`
  - backend-conditioned rows:
    - `artifacts/json/campaign_A6_L2_backend_ibm_boston.json`
    - `artifacts/json/campaign_A6_L2_backend_ibm_miami.json`

Observed result on that surface:
- proxy baseline: `ΔE_abs = 3.28323063263e-01`
- both requested backends fell back to the same fake backend target: `FakeNighthawk`
- both backend-conditioned rows were effectively identical:
  - `ΔE_abs ≈ 3.28323059580e-01`
  - compiled `2Q = 22`
  - compiled depth `48`

Corrected interpretation after Oracle review:
- this run is **invalid for backend-separation claims** because the requested backends collapsed to the same resolved backend target
- it is only useful as proof that the backend-name selection was bad

Corrected next-step assay:
- first rerun as direct single-backend tests, not shortlist reduction
- use explicit fake backend names that the repo itself prefers, so the targets cannot silently collapse:
  - `FakeNighthawk`
  - `FakeFez`
  - `FakeMarrakesh`
- use `transpile_single_v1` first
- only after those actually diverge should `hh_adapt_backend_shortlist.py` be used for summary reduction

### Claim 6: shortlist width can reduce measurement burden at fixed error
How I tested it:
- ran the same direct HH ADAPT search
- varied only shortlist width
- compared selector-side measurement cost to first hit the repo hard gate `ΔE_abs < 1e-4`
- artifacts:
  - `artifacts/json/campaign_A7a_L2_shortlist_tight_phase3_v1.json`
  - `artifacts/json/campaign_A7b_L2_shortlist_default_phase3_v1.json`
  - `artifacts/json/campaign_A7c_L2_shortlist_wide_phase3_v1.json`
  - `artifacts/json/campaign_A7d_L2_shortlist_default_runtime_split_off_phase3_v1.json`
  - `artifacts/json/campaign_A7e_L2_shortlist_wide_runtime_split_off_phase3_v1.json`
  - `artifacts/json/campaign_A7f_L2_shortlist_off_runtime_split_off_phase3_v1.json`

Result:
- aggressive/default shortlist: bad and often missed the target
- wide shortlist: reached the target
- shortlist off: also reached the target
- on this target point, wide shortlist reached `ΔE_abs < 1e-4` with fewer selector-side new measurement groups than shortlist-off (`474` vs `657`) in the direct run comparison
- conclusion: shortlist width matters, but the user has already decided not to continue this line because a depth-aware shortlist policy will replace it

## What `A1a` meant
`A1a` was my shorthand for a curriculum/default-path test taken from the repo README, not from your message wording.

Hypothesis tested:
- Does the repo's canonical staged HH default path help on this target problem?
- In repo terms, that means:
  1. warm start with `hh_hva_ptw`
  2. use that prepared state as the ADAPT reference
  3. start depth-0 HH ADAPT from the narrow physics-aligned core (`paop_lf_std`) under `phase3_v1`
  4. use broad `full_meta` only as a later residual enrichment path, not as the default depth-0 pool

Source for that claim:
- `README.md:25-35`
- `README.md:192-204`

This is what I referred to as the "narrow core" claim.

## User-directed exclusions
Per user instruction during this session:
- do **not** spend more time testing the pool/manifold / polaron-boson-ansatz angle as a novelty claim here
- continue with the other novelty-related run surfaces instead

## Completed runs and conclusions

### 1. Staged narrow-core/default-path test (`A1a`)
Artifact:
- `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1.json`
- Pareto sidecars:
  - `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1_pareto_frontier.json`
  - `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1_pareto_rows.json`

Conclusion:
- Negative / weak result on this target point.
- The staged narrow-core default path did **not** reach the repo hard gate.

Key facts:
- final `ΔE_abs = 1.01796943698e-02`
- ADAPT energy `0.16884759849552933`
- exact energy `0.15866790412572634`
- stopped at `max_depth`
- ADAPT depth `78`
- warm stage `ecut_1` passed (`0.1` threshold)
- final `ecut_2` failed (`1e-4` threshold)

Supporting file sections:
- `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1.json`
  - `stage_pipeline.adapt_vqe`
  - `stage_pipeline.conventional_replay`
  - `comparisons.stage_gates`
  - `pareto_tracking`

Interpretation:
- For this physics point (`L=2, U=4, g_ep=0.5, n_ph_max=1`), the repo's staged narrow-core default path is **not currently supported** as a strong novelty/performance claim.

### 2. Shortlisting study (`A7` family)
Artifacts:
- `artifacts/json/campaign_A7a_L2_shortlist_tight_phase3_v1.json`
- `artifacts/json/campaign_A7b_L2_shortlist_default_phase3_v1.json`
- `artifacts/json/campaign_A7c_L2_shortlist_wide_phase3_v1.json`
- `artifacts/json/campaign_A7d_L2_shortlist_default_runtime_split_off_phase3_v1.json`
- `artifacts/json/campaign_A7e_L2_shortlist_wide_runtime_split_off_phase3_v1.json`
- `artifacts/json/campaign_A7f_L2_shortlist_off_runtime_split_off_phase3_v1.json`

Conclusion:
- The useful question was not "is aggressive shortlisting bad?" but rather:
  - does wide shortlisting beat no-shortlist on measurement-cost-to-target?
- Using the repo hard gate `ΔE_abs < 1e-4` (`README.md:289`) and selector-side new-group cost from per-depth shortlisted records, the answer on this target point is:
  - default shortlist: bad / fails gate
  - wide shortlist: good / passes gate
  - shortlist off: good / passes gate
  - wide shortlist appears **better than shortlist off on selector-side measurement cost to the `1e-4` gate**

Key facts:
- default shortlist + runtime split off:
  - file: `artifacts/json/campaign_A7d_L2_shortlist_default_runtime_split_off_phase3_v1.json`
  - final `ΔE_abs = 3.283230633e-01`
  - fails `1e-4`
- wide shortlist + runtime split off:
  - file: `artifacts/json/campaign_A7e_L2_shortlist_wide_runtime_split_off_phase3_v1.json`
  - final `ΔE_abs = 5.618140840e-05`
  - passes `1e-4`
- shortlist effectively off + runtime split off:
  - file: `artifacts/json/campaign_A7f_L2_shortlist_off_runtime_split_off_phase3_v1.json`
  - final `ΔE_abs = 5.622218203e-05`
  - passes `1e-4`

Cost-to-target summary used in chat:
- threshold: `ΔE_abs < 1e-4` from `README.md:289`
- selector-side proxy: cumulative sum of `shortlisted_records[*].measurement_cache_stats.groups_new` until first hit
- wide shortlist hit `1e-4` with about `474` new groups
- shortlist off hit `1e-4` with about `657` new groups

Interpretation:
- On this target point, shortlisting looks useful only in a non-aggressive regime.
- Because the user has already stated a future expected-depth-aware shortlist schedule will replace this line of inquiry, no further shortlist/off runs are needed now.

### 3. Reoptimization policy (`A3`)
Artifacts:
- `artifacts/json/campaign_A3b_L2_windowed_phase3_v1.json`
- `artifacts/json/campaign_A3c_L2_full_phase3_v1.json`
- baseline for comparison: `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1.json`

Conclusion:
- Reoptimization policy matters somewhat, but none of the staged variants rescued this target point.
- `windowed` was best among the staged policies tested in this batch.

Key facts:
- baseline `append_only` (`A1a`): `ΔE_abs = 1.01796943698e-02`
- `windowed` (`A3b`): `ΔE_abs = 7.97336411689e-03`
- `full` (`A3c`): `ΔE_abs = 8.90929211957e-03`
- all three stopped at `max_depth = 78`
- all three failed `ecut_2` (`1e-4`)

Interpretation:
- `windowed` improved over `append_only`
- `full` also improved over `append_only`, but less than `windowed`
- this is a real effect, but not a success condition on its own for this target point

### 4. Lifetime-cost ablation (`A4`)
Artifact:
- `artifacts/json/campaign_A4b_L2_lifetime_off_phase3_v1.json`
- baseline for comparison: `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1.json`

Conclusion:
- No visible effect on this staged target point under the tested settings.

Key facts:
- lifetime on baseline (`A1a`): `ΔE_abs = 1.01796943698e-02`
- lifetime off (`A4b`): `ΔE_abs = 1.01796943698e-02`
- same stopping condition: `max_depth`
- same stage gate outcome: fail `ecut_2`

Interpretation:
- this ablation was null in the current staged regime
- it may still matter elsewhere, but this run did not reveal a novelty win

### 5. Runtime-split ablation (`A5`)
Artifact:
- `artifacts/json/campaign_A5b_L2_split_on_phase3_v1.json`
- baseline for comparison: `artifacts/json/campaign_A1a_L2_staged_core_phase3_v1.json`

Conclusion:
- No visible effect on this staged target point under the tested settings.

Key facts:
- split off baseline (`A1a`): `ΔE_abs = 1.01796943698e-02`
- split on (`A5b`): `ΔE_abs = 1.01796943698e-02`
- same stopping condition: `max_depth`
- same stage gate outcome: fail `ecut_2`

Interpretation:
- runtime split did not rescue the staged path here
- this run did not reveal a staged novelty win

### 6. Backend-conditioned parent search (`A6`)
Artifacts:
- `artifacts/json/campaign_A6_L2_backend_summary.json`
- `artifacts/json/campaign_A6_L2_backend_proxy_baseline.json`
- `artifacts/json/campaign_A6_L2_backend_ibm_boston.json`
- `artifacts/json/campaign_A6_L2_backend_ibm_miami.json`

Conclusion:
- Under this run surface, backend-conditioned search did not change the discovered scaffold or energy outcome.
- The requested backends did not resolve directly and both fell back to `FakeNighthawk`, so this was not yet a meaningful backend-divergence demonstration.

Key facts:
- proxy baseline: `ΔE_abs = 3.28323063263e-01`, depth `14`
- `ibm_boston` request → resolved to `FakeNighthawk`: `ΔE_abs = 3.28323059580e-01`, depth `14`, compiled 2Q count `22`, compiled depth `48`
- `ibm_miami` request → resolved to `FakeNighthawk`: `ΔE_abs = 3.28323059580e-01`, depth `14`, compiled 2Q count `22`, compiled depth `48`
- summary file marks `ibm_boston` as best compile / best energy, but both backend-conditioned rows are effectively identical because both resolved to the same fake backend target

Interpretation:
- this run completed successfully, but it did **not** yet produce a real backend-conditioned separation
- to test this claim properly, the shortlist must resolve to actually distinct backend targets

## Existing repo evidence that was inspected (not newly rerun here)

### Backend-conditioned search / executable-aware parent choice
Supporting files:
- `artifacts/reports/investigation_backend_first_hh_20260322.md`
- `artifacts/reports/investigation_marrakesh_pareto_20260323.md`

Conclusion recorded during investigation:
- promising novelty surface
- not newly rerun yet in this session

## Runs still intended after user correction
The user asked to continue the novelty-related run surfaces other than the pool/manifold claim.

Duplicate baselines not rerun:
- `A3a` (`append_only`) duplicates the staged baseline already exercised by `A1a`
- `A4a` (lifetime on baseline) duplicates the same baseline setting already exercised by `A1a`
- `A5a` (runtime split off baseline) duplicates the same baseline setting already exercised by `A1a`

Non-duplicate run families selected for execution:
1. `A3b` reoptimization = `windowed`
2. `A3c` reoptimization = `full`
3. `A4b` lifetime-cost = `off`
4. `A5b` runtime-split = `shortlist_pauli_children_v1`
5. `A6` backend-conditioned parent search

Campaign source:
- `artifacts/reports/hh_campaign_sheet_laneA_laneB_20260325.md`

## Corrected direct-surface reruns after Oracle review

### Direct lifetime-cost rerun on the correct selector surface
How it was run:
- direct `pipelines/hardcoded/adapt_pipeline.py`
- fixed good-basin surface:
  - `full_meta`
  - `phase3_v1`
  - `POWELL`
  - `windowed` refit with wide active set
  - wide shortlist (`phase1_shortlist_size=256`, `phase2_shortlist_fraction=1.0`, `phase2_shortlist_size=128`)
  - `phase3_enable_rescue=true`
  - `phase3_runtime_split_mode=off`
- toggled only lifetime cost:
  - on baseline already available: `artifacts/json/campaign_A7e_L2_shortlist_wide_runtime_split_off_phase3_v1.json`
  - off rerun: `artifacts/json/corrected_direct_lifetime_off_fullmeta_phase3_v1.json`

Result:
- lifetime on baseline: `ΔE_abs = 5.61814084024e-05`
- lifetime off direct rerun: `ΔE_abs = 5.61814084024e-05`
- selector-side cumulative cost summaries were also identical in these two runs:
  - summed `history[*].measurement_cache_stats.groups_new = 10`
  - summed `history[*].compile_cost_total = 369`
  - history length `= 14`
  - stop reason `= drop_plateau`

Conclusion:
- on the corrected direct selector surface, lifetime-cost still looks like a **real null on this L=2 target point**
- this is much more credible than the earlier staged-wrapper null because both the endpoint and the simple selector-side cost totals matched exactly

### Direct backend-conditioned reruns with explicit fake backend names
How they were run:
- same corrected direct selector surface as above
- changed only backend compile mode:
  - `transpile_single_v1 + FakeNighthawk`
  - `transpile_single_v1 + FakeFez`
  - `transpile_single_v1 + FakeMarrakesh`
- artifacts:
  - `artifacts/json/corrected_direct_backend_fakenighthawk_phase3_v1.json`
  - `artifacts/json/corrected_direct_backend_fakefez_phase3_v1.json`
  - `artifacts/json/corrected_direct_backend_fakemarrakesh_phase3_v1.json`

Result:
- all three explicit fake-backend runs converged to the same bad-basin family:
  - `FakeNighthawk`: `ΔE_abs ≈ 3.283230353e-01`
  - `FakeFez`: `ΔE_abs ≈ 3.283230351e-01`
  - `FakeMarrakesh`: `ΔE_abs ≈ 3.283230351e-01`
- this is now a meaningful backend-conditioned result because the names did **not** collapse to the same fallback target

Cost-normalized comparison against the direct proxy baseline:
- proxy baseline (`artifacts/json/campaign_A7e_L2_shortlist_wide_runtime_split_off_phase3_v1.json`):
  - `ΔE_abs = 5.61814084024e-05`
  - summed `history[*].measurement_cache_stats.groups_new = 10`
  - summed `history[*].compile_cost_total = 369`
- `FakeNighthawk`:
  - `ΔE_abs = 3.28323031194e-01`
  - summed `history[*].measurement_cache_stats.groups_new = 6`
  - summed `history[*].compile_cost_total = 85.3`
  - final compiled `2Q = 61`, compiled depth `= 161`
- `FakeFez`:
  - `ΔE_abs = 3.28323031004e-01`
  - summed `history[*].measurement_cache_stats.groups_new = 7`
  - summed `history[*].compile_cost_total = 95.85`
  - final compiled `2Q = 71`, compiled depth `= 188`
- `FakeMarrakesh`:
  - `ΔE_abs = 3.28323031004e-01`
  - summed `history[*].measurement_cache_stats.groups_new = 7`
  - summed `history[*].compile_cost_total = 95.85`
  - final compiled `2Q = 71`, compiled depth `= 188`

Conclusion:
- backend-conditioned compile-aware scoring appears to steer this direct search surface into a different, much worse basin than the proxy-cost baseline on this target point
- importantly, the backend-conditioned runs are **cheaper** on the selector-side cost proxies while being dramatically worse in energy, so the live effect here looks like **over-aggressive cost steering into a bad basin**, not a backend-name resolution bug
- this is a real signal and should be investigated as a backend-cost-induced selector effect, not dismissed as the old fallback bug

## Current high-level status
- Source-code changes: none
- Documentation added: this file only
- Staged-wrapper selector-null claims have been downgraded
- Corrected direct-surface reruns now recorded in this file
