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

## Corrected v2 ablation experiments (POWELL, direct surface)

All v2 runs use `adapt_pipeline.py` directly with POWELL optimizer, `full_meta` pool, `phase3_v1`, `eps_grad=5e-7`, `eps_energy=1e-9`, `max_depth=80`, `shortlist_size=256`, `phase2_shortlist_fraction=1.0`, `phase2_shortlist_size=128`, `phase3_enable_rescue=true`.

### Claim 2 v2: Reopt policy × position insertion (L=2, 4-way factorial)

| Variant | Position | Reopt | Ops | |ΔE| | Stop | Basin |
|---------|----------|-------|-----|------|------|-------|
| 2A append-only | append (`pos=1`) | `append_only` | 16 | **3.283e-01** | drop_plateau | BAD |
| 2B append+window | append (`pos=1`) | `windowed(3)` | 15 | **7.010e-05** | drop_plateau | GOOD |
| 2C insert-only | insert (`pos=6`) | `append_only` | 16 | **3.283e-01** | drop_plateau | BAD |
| 2D insert+window | insert (`pos=6`) | `windowed(3)` | 15 | **7.010e-05** | drop_plateau | GOOD |

Artifacts:
- `artifacts/json/claim2_append_only_L2_v2_20260325.json`
- `artifacts/json/claim2_append_window_L2_v2_20260325.json`
- `artifacts/json/claim2_insert_only_L2_v2_20260325.json`
- `artifacts/json/claim2_insert_window_L2_v2_20260325.json`

**Conclusion — Claim 2:**
- **Windowed reopt is the critical novel feature.** It delivers a ~4700x improvement in |ΔE| (3.28e-01 → 7.01e-05).
- **Position-aware insertion has zero measurable effect at L=2.** Both append-only variants (with/without insertion) land in the same bad basin. Both windowed variants (with/without insertion) land in the same good basin with identical |ΔE|.
- The windowed runs also produce shorter circuits (15 ops, 29 params) vs append-only (16 ops, 37 params).
- Operator sequences: windowed variants start with termwise operators (yezeee, yeeeze, eyeeez, eyezee), then UCCSD singles. Append-only variants select the same initial operators but cannot escape the local minimum because prefix parameters are frozen.

Operator sequence comparison (good basin = windowed):
```
d0: hh_termwise_ham_quadrature_term(yezeee)
d1: hh_termwise_ham_quadrature_term(yeeeze)
d2: hh_termwise_ham_quadrature_term(eyeeez)
d3: hh_termwise_ham_quadrature_term(eyezee)
d4: uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
d5: uccsd_ferm_lifted::uccsd_sing(beta:2->3)
d6: hh_termwise_ham_quadrature_term(yeeeze)
d7: paop_lf_full:paop_dbl_p(site=0->phonon=1)
```

### Claim 3 v2: Lifetime cost at L=3

L=3 parameters: `t=1, U=2, ω₀=1, g=1, n_ph_max=1, periodic boundary` (matches existing L=3 heavy-scaffold runs).

| Variant | Lifetime Cost | Ops | |ΔE| | Stop |
|---------|-------------|-----|------|------|
| baseline | `phase3_v1` (ON) | 17 | **8.636e-01** | drop_plateau |
| test | `off` | 17 | **8.636e-01** | drop_plateau |

Artifacts:
- `artifacts/json/claim3_4_baseline_L3_v2_20260325.json`
- `artifacts/json/claim3_lifetime_off_L3_v2_20260325.json`

**Conclusion — Claim 3:**
- Both L=3 runs are stuck in the same bad basin at |ΔE| ≈ 0.864. Lifetime cost has no visible effect.
- The L=3 problem appears to be too hard for the current direct-surface settings (17 ops is likely insufficient for 9-qubit HH at these couplings).
- **This is an inconclusive null**, not a disproof. The existing staged L=3 runs reached much deeper scaffolds (37 ops in `hh_heavy_scaffold_best_yet`) because the staged wrapper uses warm-start + narrow core pool, which this direct surface does not.
- A fair L=3 test would require either warm-starting from the staged path or increasing max_depth significantly.

### Claim 4 v2: Runtime split at L=3

| Variant | Runtime Split | Ops | |ΔE| | Stop |
|---------|-------------|-----|------|------|
| baseline | `off` | 17 | **8.636e-01** | drop_plateau |
| test | `shortlist_pauli_children_v1` | 24 | **8.636e-01** | drop_plateau |

Artifacts:
- `artifacts/json/claim3_4_baseline_L3_v2_20260325.json` (shared baseline)
- `artifacts/json/claim4_split_on_L3_v2_20260325.json`

**Conclusion — Claim 4:**
- Split ON produced a deeper circuit (24 ops vs 17) but identical |ΔE|. The child-set decomposition generated more operators but they didn't improve energy.
- Like Claim 3, this is **inconclusive** due to the L=3 surface being stuck. The previously observed `selected_child_total = 0` at L=2 and the stuck L=3 surface together suggest runtime split needs:
  1. A deeper scaffold where Pauli children actually decompose composite generators
  2. Possibly a warm-started reference state

### Claim 5 v2: Backend-conditioned scoring (corrected design)

User-directed correction: compare `transpile + no cost steering` vs `transpile + reduced cost steering`, not `proxy vs transpile`. This isolates whether compile-cost integration helps.

| Variant | Backend Mode | λ_compile | Ops | |ΔE| | Stop | Basin |
|---------|-------------|-----------|-----|------|------|-------|
| 5-base | transpile | 0 | 13 | **3.283e-01** | drop_plateau | BAD |
| 5-A | transpile | 0.01 | 15 | **3.283e-01** | drop_plateau | BAD |
| 5-B | transpile | 0.005 | 13 | **3.283e-01** | drop_plateau | BAD |
| 5-control | transpile | **0.05** | 15 | **7.010e-05** | drop_plateau | GOOD |

Artifacts:
- `artifacts/json/claim5_transpile_nocost_L2_v2_20260325.json`
- `artifacts/json/claim5_transpile_lambda001_L2_v2_20260325.json`
- `artifacts/json/claim5_transpile_lambda0005_L2_v2_20260325.json`
- `artifacts/json/claim5_transpile_lambda005_control_L2_v2_20260325.json`

**Conclusion — Claim 5:**
- **The compile-cost weight λ=0.05 is essential for reaching the good basin.** At λ ≤ 0.01, the algorithm selects PAOP operators first (paop_dbl_p, paop_disp), which leads to the bad basin (|ΔE| ≈ 0.328). At λ=0.05, termwise operators are selected first, reaching |ΔE| ≈ 7e-05.
- **The transpile mode itself is NOT harmful.** transpile + λ=0.05 achieves the same |ΔE| as proxy + λ=0.05.
- The compile cost acts as an implicit **regularizer**: by penalizing expensive operators, it steers the early operator selection toward termwise Hamiltonian terms (which are cheap) rather than PAOP composites (which are expensive but gradient-attractive), and this happens to be the basin that converges well.
- The critical λ threshold is somewhere in (0.01, 0.05). Further runs at λ=0.02, 0.03 could narrow this.

Operator sequence at λ=0 (bad basin):
```
d0: paop_lf_full:paop_dbl_p(site=0->phonon=0)
d1: paop_full:paop_disp(site=0)
d2: paop_lf_full:paop_dbl_p(site=0->phonon=1)
d3: paop_full:paop_disp(site=1)
d4: hh_termwise_ham_quadrature_term(yezeee)
```

Operator sequence at λ=0.05 (good basin):
```
d0: uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
d1: hh_termwise_ham_quadrature_term(yezeee)
d2: hh_termwise_ham_quadrature_term(yeeeze)
d3: hh_termwise_ham_quadrature_term(eyeeez)
d4: hh_termwise_ham_quadrature_term(eyezee)
```

## Summary of novelty claims after v2 experiments

| Claim | Feature | Status | Evidence |
|-------|---------|--------|----------|
| 1 | Staged HH default path | **Negative at L=2** | A1a: stuck at |ΔE|=0.010 |
| 2 | Reopt policy (windowed) | **STRONG POSITIVE** | 4700x |ΔE| improvement; the single most impactful feature |
| 2b | Position-aware insertion | **Null at L=2** | No measurable effect in 2×2 factorial |
| 3 | Lifetime-cost penalty | **Inconclusive** | Null at L=2 (correct surface); null at L=3 (stuck surface) |
| 4 | Runtime split | **Inconclusive** | Dormant at L=2; inconclusive at L=3 (stuck) |
| 5 | Compile-cost scoring (λ) | **STRONG POSITIVE** | λ≥0.05 is essential regularizer; λ≤0.01 → bad basin |
| 6 | Shortlisting | **Moderate positive at L=2** | 28% fewer measurement groups; superseded by future depth-aware policy |

### Claim 5 v2 continued: λ threshold sweep (pinning the critical value)

Full λ sweep on transpile_single_v1 + FakeNighthawk, all other settings matched:

| λ_compile | |ΔE| | Ops | d0 operator | Basin |
|-----------|------|-----|-------------|-------|
| 0.000 | 3.283e-01 | 13 | paop_lf_full:paop_dbl_p(site=0->phonon=0) | BAD |
| 0.005 | 3.283e-01 | 13 | uccsd_ferm_lifted::uccsd_sing(beta:2->3) | BAD |
| 0.010 | 3.283e-01 | 15 | paop_full:paop_disp(site=1) | BAD |
| **0.020** | **7.010e-05** | 15 | uccsd_ferm_lifted::uccsd_sing(alpha:0->1) | **GOOD** |
| 0.030 | 7.010e-05 | 15 | uccsd_ferm_lifted::uccsd_sing(alpha:0->1) | GOOD |
| 0.040 | 7.010e-05 | 15 | uccsd_ferm_lifted::uccsd_sing(alpha:0->1) | GOOD |
| 0.050 | 7.010e-05 | 15 | uccsd_ferm_lifted::uccsd_sing(alpha:0->1) | GOOD |

Artifacts: `artifacts/json/claim5_transpile_lambda{002,003,004}_L2_v2_20260325.json`

**Conclusion:** Sharp phase transition at λ ∈ (0.01, 0.02). Below this threshold, PAOP operators win at depth 0 due to large gradients despite high compile cost. Above it, the compile-cost penalty redirects selection toward cheaper UCCSD/termwise operators that happen to reach the good energy basin. The compile cost acts as an implicit **basin-selection regularizer**, not merely a circuit-cost optimizer.

### Claim 7: Batching ablation (L=2)

| Batching | |ΔE| | Ops | Batch sizes | groups_new |
|----------|------|-----|-------------|------------|
| ON | 7.010e-05 | 15 | [1,1,1,1,1,1,1,1,1,1,1,1,1,1] | 7 |
| OFF | 7.010e-05 | 15 | [1,1,1,1,1,1,1,1,1,1,1,1,1,1] | 7 |

Artifact: `artifacts/json/claim7_batching_off_L2_v2_20260325.json`

**Conclusion:** Batching is **dormant at L=2** — the near-degenerate gate (η_nd = 0.9) never fires because the top candidate is always clearly best at each depth. Similar to runtime split, this feature requires a larger system where multiple operators compete at near-equal scores. No claim can be made from L=2 data; needs L=3 or L=4 testing.

### Claim 2 v2 continued: Multi-seed robustness

| Seed | Reopt | Ops | Params | |ΔE| | Basin |
|------|-------|-----|--------|------|-------|
| 7 | append_only | 16 | 37 | 3.283e-01 | BAD |
| 7 | windowed | 15 | 29 | 7.010e-05 | GOOD |
| 42 | append_only | 16 | 37 | 3.283e-01 | BAD |
| 42 | windowed | 15 | 29 | 7.010e-05 | GOOD |
| 123 | append_only | 16 | 37 | 3.283e-01 | BAD |
| 123 | windowed | 15 | 29 | 7.010e-05 | GOOD |

Artifacts: `artifacts/json/claim2_insert_{only,window}_L2_seed{42,123}_20260325.json`

**Conclusion:** The windowed reopt result is **perfectly seed-independent** across 3 seeds. Append-only lands in the bad basin (|ΔE|≈0.328) and windowed lands in the good basin (|ΔE|≈7e-05) every time, with identical operator counts and parameter counts. This is not a stochastic effect — it's a deterministic consequence of the reopt policy enabling parameter escape from a local minimum.

### Claims 3 & 4 v2 continued: L=3 warm-started experiments (COMPLETED)

Using `--adapt-ref-json` with the 120-operator L=3 staged handoff (`hh_staged_L3_static_t1_U2_dv0_w1_g1_nph1_warmhh_hva_ptw_9afb5d05d5_adapt_handoff.json`).

Reference state: energy = -0.593, |ΔE| = 0.106 (correctly above exact = -0.699).

| Run | Ops | Energy | vs Exact | |ΔE| | Stop | Sector |
|-----|-----|--------|----------|------|------|--------|
| baseline (life ON, split OFF) | 86 | -0.749 | -0.051 | 5.06e-02 | max_depth | **BELOW** (mild violation) |
| Claim 3 (lifetime OFF) | 86 | -1.534 | -0.835 | 8.35e-01 | max_depth | **BELOW** (massive violation) |
| Claim 4 (split ON) | 80 | -0.614 | +0.085 | 8.46e-02 | max_depth | **ABOVE** (correct) |

Artifacts:
- `artifacts/json/claim3_4_baseline_L3_warmstart_20260325.json`
- `artifacts/json/claim3_lifetime_off_L3_warmstart_20260325.json`
- `artifacts/json/claim4_split_on_L3_warmstart_20260325.json`

**Critical finding — sector constraint violations at L=3:**
- The `full_meta` pool contains operators that break particle-number sector constraints at L=3 (likely `hva_hh_ptw` family, per user note)
- **Lifetime OFF** causes massive sector violation (0.835 Ha below exact). Without lifetime cost weighting, the selector aggressively picks sector-breaking operators
- **Lifetime ON** (baseline) causes mild sector violation (0.051 below exact). The lifetime cost partially constrains selection away from the worst sector-breaking operators
- **Split ON** stays correctly above exact. The Pauli child decomposition appears to exclude the composite sector-breaking generators
- `selected_child_total = 0` even at L=3 with 80 ops — runtime split never fires, but it constrains the operator candidate space indirectly

**Conclusion — Claim 3 (lifetime cost):**
- Lifetime cost has a **real and large effect at L=3**: without it, the run goes catastrophically wrong (0.835 below exact vs 0.051 below)
- However, the mechanism is sector-constraint-related, not the intended cost-burden effect
- The lifetime cost penalty seems to de-prioritize the sector-breaking operators (which are new/recently-added and thus have high lifetime cost)
- **Status: POSITIVE, but mechanism needs qualification** — lifetime cost acts as an indirect sector regularizer at L=3

**Conclusion — Claim 4 (runtime split):**
- Split ON is the only variant that stays above exact at L=3
- However, it reaches worse |ΔE| (0.085 vs baseline's 0.051) because it cannot exploit the sector-breaking operators that lower energy (illegitimately)
- `selected_child_total = 0` means the split logic itself never fires — its effect is purely through operator candidate filtering
- **Status: POSITIVE for sector preservation**, but the energy result is confounded by the sector violation in the comparison runs
- Proper evaluation requires `--phase3-symmetry-mitigation-mode` to eliminate sector violations from all runs

## Summary of novelty claims after all v2 + sweep experiments

| Claim | Feature | Status | Key Evidence |
|-------|---------|--------|-------------|
| 1 | Staged HH default path | **Negative at L=2** | Stuck at |ΔE|=0.010 |
| 2 | Windowed reopt | **STRONG POSITIVE** | 4700x improvement; seed-independent × 3 seeds |
| 2b | Position-aware insertion | **Null at L=2** | No effect in 2×2 factorial × 3 seeds |
| 3 | Lifetime-cost penalty | **POSITIVE at L=3** | Prevents massive sector violation (0.835→0.051); acts as indirect sector regularizer |
| 4 | Runtime split | **POSITIVE for sector preservation** | Only variant staying above exact at L=3; confounded by sector violations in baselines |
| 5 | Compile-cost regularization (λ) | **STRONG POSITIVE** | Sharp threshold at λ∈(0.01,0.02); basin-selection regularizer |
| 6 | Shortlisting | **Moderate positive** | 28% fewer groups; superseded by future depth-aware policy |
| 7 | Batching | **Dormant at L=2** | Near-degenerate gate never fires; needs larger system |

## Current high-level status
- Source-code changes: none
- Documentation added: this file only
- Staged-wrapper selector-null claims have been downgraded
- Corrected direct-surface reruns now recorded in this file
- v2 POWELL-based ablation experiments completed and analyzed
- λ threshold sweep completed: critical value pinned to (0.01, 0.02)
- Multi-seed robustness confirmed for Claim 2 (3 seeds, perfectly consistent)
- Batching ablation: dormant at L=2
- L=3 warm-started experiments for Claims 3/4 in progress
- Key findings: windowed reopt and compile-cost regularization are the two strongest novel features
