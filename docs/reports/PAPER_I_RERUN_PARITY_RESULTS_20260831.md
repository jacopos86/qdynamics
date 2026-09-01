# Paper I Rerun-Parity Results — three-way (paper / integration lineage / main) — 2026-08-31

Scope: the 12 locally runnable weak-HH k=20 cells of
`docs/reports/PAPER_I_RERUN_PARITY_PLAN_20260831.md` (§1.2 Tier-0 screens, §1.3 run
matrix, §3 comparison rubric, top-of-file k=20 amendment), each executed twice —
once on the archived integration lineage, once on `main` — and scored against the
stored paper evidence.

## 0. Provenance, gates, artefact paths

Scratch base (all rerun outputs and scoring scripts; nothing below is in the repo):

```
$B = /private/tmp/claude-501/-Users-lordskyler-Library-Application-Support-Claude-\
scratch-workspaces-c7246441-845b-4d46-b909-661a445529c0-2d70f55d-6da7-4e57-af14-\
cabb5b067429-scratch-2026-09-01-a07737/a2ce074f-a814-4379-ac73-876a64279eb2/\
scratchpad/pi_rerun
```

| Lane | Tree | Commit | Export | Outputs | Driver | Log |
|---|---|---|---|---|---|---|
| Lineage rerun | `origin/paper-i-integration-20260831` | `f041a5c1bd6e06a1885f5f83d5494c180ca709ec` | `git archive … -- pipelines src docs` → `$B/pi_integration` | `$B/pi_integration/aavqe_{regime}_r20.json`, `$B/pi_integration/b3_{regime}_{arm}_k20/result.json` | `$B/drv_a.py`, `$B/drv_c.py` | `$B/runs.log` |
| Main rerun | `main` | `25cb07451b55a9d1c82a52dd72cc5cf9cc1ab5bb` | same, → `$B/main_tree` | `$B/main_tree/aavqe_{regime}_r20.json`, `$B/main_tree/b3_{regime}_{arm}_k20/result.json` | `$B/drv_a.py`, `$B/drv_c.py` | `$B/runs2.log` (AVQE, phase MA), `$B/runs3.log` (B3 flat, phase MC2) |
| Noise campaign (failed) | detached worktree | `c2cd9379` | `git worktree add --detach $B/appendix_wt c2cd9379` | none | `chtc/ra_adapt_appendix_noise_20260821_v1/run_local_ra_adapt_appendix_noise_l3_20260821_v1.py` | `$B/runs2.log:434–5023` |

Archive comparison inputs (repo/overleaf, read-only):

- `chtc/paper_i_ra_adapt_repair_20260727/results_tooling/kstar_tables.json` (`b3`, `b9`, `aavqe`, `kstar`)
- `agent_guidance/static-adapt/golden/bundle3_surviving_inputs/aavqe_full_hist_tolmatch_9665004.json`
- `agent_guidance/static-adapt/golden/bundle3_surviving_inputs/aavqe_salg_prefix_r75_9665004.json`
- `/Users/lordskyler/ADAPT---Paper-I/Paper_I_overleaf_20260818/figures/paper_i_hh_hierarchical_route_appendix_20260823/b3_phase_reuse_corrections_20260821.json`

Gates (plan §1.0), run before and after both passes:

| Gate | Value | Status |
|---|---|---|
| G1 pins | `qiskit 2.3.1 / numpy 2.3.5 / scipy 1.16.3` | OK (`$B/runs.log:3`) |
| G2 golden `MANIFEST.sha256` | 10/10 | OK pre (`runs.log:4`) and post (`runs.log:1082`, `runs2.log:5025`) |
| G3 `kstar_tables.json` sha256 | `060e752474806d2f…` | OK pre/post (`runs.log:5,1083`, `runs2.log:5026`) |

Scoring code (scratch): `$B/score_pass1.py` (pass 1, lineage-only),
`$B/extract_cell.py` (mmap/scoped-`raw_decode` per-cell extractor, RAM ≤ 0.42 GB
against 28–680 MB result/checkpoint JSONs), `$B/score_threeway.py` (three-way
scoring, same rubric/tiers/alignment/S_alg conventions as pass 1),
`$B/gen_tables.py` → `$B/tables.md`. Machine record: `$B/threeway_scores.json`
(and `$B/pass1_scores.json` for pass 1). `score_threeway.py` asserts every
integration-side number against `pass1_scores.json`: **12/12 `OK`** — the
re-extraction is value-preserving, so pass-1 verdicts are unchanged by this pass.

Float tiers (plan §3): **A** ≤ 2.0e-12, **B** ≤ 1.0e-8 with identical integer
selections, **C** > 1.0e-8. Integer rows are zero-tolerance. "to k\*" = the prefix
carrying the Table I / Table III star quantities; "to 20" = the full executed prefix.

---

## 1. Verdict table (three-way, per cell)

| # | Cell | k\* | Paper vs lineage rerun | Paper vs main rerun | Lineage vs main |
|---|---|---|---|---|---|
| R01 | AVQE ww | 11 | MISMATCH(S_alg −347) | MISMATCH(post-k\* break@d20; S_alg +1843) | trajectory-pre-k\* @d5 |
| R02 | AVQE iw | 9 | MISMATCH(S_alg −19) | MISMATCH(post-k\* break@d18; S_alg +1256) | trajectory-pre-k\* @d3 |
| R03 | AVQE sw | 10 | MISMATCH(S_alg +18) | MISMATCH(post-k\* break@d17; S_alg +2560) | trajectory-pre-k\* @d4 |
| R08 | B3 ww append | 11 | MISMATCH(S_alg +242) | MISMATCH(S_alg +242) | **bit-identical** |
| R17 | B3 ww plateau | 11 | MISMATCH(post-k\* break@d19; S_alg +242) | MISMATCH(post-k\* break@d19; S_alg +242) | trajectory-post-k\* @d16 |
| R20 | B3 ww always | 11 | MISMATCH(post-k\* break@d16; S_alg −84) | **MISMATCH(trajectory@d4; S_alg +2070)** | trajectory-pre-k\* @d3 |
| R09 | B3 iw append | 9 | MISMATCH(S_alg −4) | MISMATCH(S_alg −4) | **bit-identical** |
| R18 | B3 iw plateau | 9 | MISMATCH(S_alg −4) | MISMATCH(post-k\* break@d19; S_alg −4) | trajectory-post-k\* @d11 |
| R21 | B3 iw always | 9 | **MISMATCH(trajectory@d3; S_alg +2458)** | **MISMATCH(trajectory@d3; S_alg +2887)** | trajectory-pre-k\* @d4 |
| R10 | B3 sw append | 10 | MISMATCH(S_alg +154) | MISMATCH(S_alg +154) | trajectory-post-k\* @d16 |
| R19 | B3 sw plateau | 10 | MISMATCH(S_alg +215) | MISMATCH(S_alg +192) | trajectory-post-k\* @d16 |
| R22 | B3 sw always | 10 | MISMATCH(S_alg +51) | MISMATCH(S_alg +470) | trajectory-pre-k\* @d4 |

Counts:

- **Paper vs lineage:** MATCH 0 / 12, MISMATCH 12 / 12 (every cell carries a nonzero
  integer `S_alg` delta, which §3 puts in the zero-tolerance tier). **Star-physics
  (dE trace through k\* inside Tier A/B): 11 / 12** — only R21 fails.
- **Paper vs main:** MATCH 0 / 12, MISMATCH 12 / 12. **Star-physics: 10 / 12** — R21
  fails identically, and **R20 (ww always) additionally fails on main only**
  (2.7445e-04 vs the lineage's 2.2627e-11).
- **Lineage vs main:** bit-identical 2 (both append arms of ww/iw), divergence
  confined beyond k\* 4 (ww/sw plateau, iw plateau, sw append), divergence at or
  before k\* 6 (all three AVQE screens + all three `always` arms).

Class definitions (`$B/score_threeway.py:classify_pair`): *bit-identical* = every
compared quantity equal; *trajectory-post-k\** = the accepted-operator **multiset**
first differs beyond the star depth; *trajectory-pre-k\** = it differs at or before
the star depth, i.e. the Table I star row rests on a different ansatz.

---

## 2. Three-way numbers

### 2.1 dE / energy prefix agreement — max |Δ| per pair

| # | Cell | paper↔lin to k\* | paper↔lin to 20 | paper↔main to k\* | paper↔main to 20 | lin↔main to k\* | lin↔main to 20 |
|---|---|---|---|---|---|---|---|
| R01 | AVQE ww | 9.3849e-12 (B) | 1.7020e-09 (B) | 2.0781e-11 (B) | 3.5870e-04 (C) | 1.5775e-11 | 3.5870e-04 |
| R02 | AVQE iw | 4.2855e-14 (A) | 4.7614e-11 (B) | 6.3771e-13 (A) | 4.7721e-06 (C) | 5.9874e-13 | 4.7721e-06 |
| R03 | AVQE sw | 8.9306e-13 (A) | 1.2751e-12 (A) | 8.7047e-10 (B) | 5.9859e-07 (C) | 8.6977e-10 | 5.9859e-07 |
| R08 | B3 ww append | 3.3231e-11 (B) | 3.3231e-11 (B) | 3.3231e-11 (B) | 3.3231e-11 (B) | 0.0000e+00 | 0.0000e+00 |
| R17 | B3 ww plateau | 3.3166e-11 (B) | 3.5155e-04 (C) | 3.3166e-11 (B) | 1.8117e-04 (C) | 0.0000e+00 | 1.7039e-04 |
| R20 | B3 ww always | 2.2627e-11 (B) | 2.1759e-04 (C) | **2.7445e-04 (C)** | 2.7445e-04 (C) | 2.7445e-04 | 2.7445e-04 |
| R09 | B3 iw append | 3.3658e-11 (B) | 3.3658e-11 (B) | 3.3658e-11 (B) | 3.3658e-11 (B) | 0.0000e+00 | 0.0000e+00 |
| R18 | B3 iw plateau | 3.3698e-11 (B) | 9.6206e-10 (B) | 3.3698e-11 (B) | 1.8251e-07 (C) | 0.0000e+00 | 1.8176e-07 |
| R21 | B3 iw always | **2.5639e-02 (C)** | 2.5639e-02 (C) | **2.5639e-02 (C)** | 2.5639e-02 (C) | 1.0612e-10 | 6.9192e-07 |
| R10 | B3 sw append | 8.6597e-15 (A) | 1.4500e-13 (A) | 8.6597e-15 (A) | 7.4627e-11 (B) | 0.0000e+00 | 7.4621e-11 |
| R19 | B3 sw plateau | 5.6660e-12 (B) | 9.9130e-09 (B) | 2.2206e-12 (B) | 1.6490e-10 (B) | 7.8866e-12 | 9.9542e-09 |
| R22 | B3 sw always | 5.4071e-10 (B) | 1.9280e-09 (B) | 2.0868e-09 (B) | 2.0868e-09 (B) | 2.6275e-09 | 2.6275e-09 |

Trace alignment (pass-1 finding, unchanged and independently reproduced on main):
`b_depth_append_ra|*` traces are depth-indexed (offset 0); `b_depth_costexp_plateau_position|*`
and `b_depth_costexp_always_open|*` are depth+1-indexed (offset 1). Both trees
select the same offset in all 9 cells.

### 2.2 k\* row (star quantities)

| # | Cell | dE\* archive | dE\* lineage | dE\* main | dev lin (tier) | dev main (tier) | S_alg\* arch | S_alg\* lin | S_alg\* main |
|---|---|---|---|---|---|---|---|---|---|
| R01 | AVQE ww | 4.5949325385e-04 | 4.5949324884e-04 | 4.5949323307e-04 | 5.0064e-12 (B) | 2.0781e-11 (B) | 13538 | 13191 | 15381 |
| R02 | AVQE iw | 2.1288179263e-04 | 2.1288179259e-04 | 2.1288179209e-04 | 4.2855e-14 (A) | 5.3690e-13 (A) | 10910 | 10891 | 12166 |
| R03 | AVQE sw | 1.5929813839e-06 | 1.5929815902e-06 | 1.5929458046e-06 | 2.0628e-13 (A) | 3.5579e-11 (B) | 11845 | 11863 | 14405 |
| R08 | B3 ww append | 3.8402516319e-04 | 3.8402515987e-04 | 3.8402515987e-04 | 3.3255e-12 (B) | 3.3255e-12 (B) | 15150 | 15392 | 15392 |
| R17 | B3 ww plateau | 5.2250267189e-04 | 5.2250265081e-04 | 5.2250265081e-04 | 2.1087e-11 (B) | 2.1087e-11 (B) | 15150 | 15392 | 15392 |
| R20 | B3 ww always | 5.1682615991e-04 | 5.1682616028e-04 | 5.1682616375e-04 | 3.7459e-13 (A) | 3.8404e-12 (B) | 38145 | 38061 | 40215 |
| R09 | B3 iw append | 2.1288182836e-04 | 2.1288182775e-04 | 2.1288182775e-04 | 6.1040e-13 (A) | 6.1040e-13 (A) | 11936 | 11932 | 11932 |
| R18 | B3 iw plateau | 4.3477075967e-04 | 4.3477076100e-04 | 4.3477076100e-04 | 1.3352e-12 (A) | 1.3352e-12 (A) | 11936 | 11932 | 11932 |
| R21 | B3 iw always | 6.7812274917e-04 | 5.9224108101e-04 | 5.9224108743e-04 | **8.5882e-05 (C)** | **8.5882e-05 (C)** | 25769 | 28227 | 28656 |
| R10 | B3 sw append | 1.5925504382e-06 | 1.5925504386e-06 | 1.5925504386e-06 | 4.4409e-16 (A) | 4.4409e-16 (A) | 13860 | 14014 | 14014 |
| R19 | B3 sw plateau | 1.7645136867e-06 | 1.7645136956e-06 | 1.7645136907e-06 | 8.8818e-15 (A) | 3.9968e-15 (A) | 25823 | 26038 | 26015 |
| R22 | B3 sw always | 7.2903944888e-06 | 7.2898537781e-06 | 7.2924813150e-06 | 5.4071e-10 (B) | 2.0868e-09 (B) | 31497 | 31548 | 31967 |

The archived `dE_star` sits at ansatz depth k\*−1 for the plateau/always records and
at depth k\* for append/AVQE; the archived `salg_star` always sits at
controller_round k\* (the tooling keys `checkpoint_prefix_salg` on
`controller_round`). Each is scored on its own basis, identically for both trees.

### 2.3 S_alg deltas and the phase-reuse correction

| # | Cell | S_alg\* lin−arch | S_alg\* main−arch | main−lin | corr lin | corr main | arch corr Δ | lin corr Δ | main corr Δ |
|---|---|---|---|---|---|---|---|---|---|
| R01 | AVQE ww | −347 (unique) | +1843 (unique) | +2190 | — | — | — | — | — |
| R02 | AVQE iw | −19 (unique) | +1256 (unique) | +1275 | — | — | — | — | — |
| R03 | AVQE sw | +18 (unique) | +2560 (unique) | +2542 | — | — | — | — | — |
| R08 | B3 ww append | +242 | +242 | 0 | 15056 | 15056 | 336 | 336 | 336 |
| R17 | B3 ww plateau | +242 | +242 | 0 | 15056 | 15056 | 336 | 336 | 336 |
| R20 | B3 ww always | −84 | +2070 | +2154 | 37565 | 39735 | 448 | 496 | 480 |
| R09 | B3 iw append | −4 | −4 | 0 | 11708 | 11708 | 224 | 224 | 224 |
| R18 | B3 iw plateau | −4 | −4 | 0 | 11708 | 11708 | 224 | 224 | 224 |
| R21 | B3 iw always | +2458 | +2887 | +429 | 27863 | 28292 | 350 | 364 | 364 |
| R10 | B3 sw append | +154 | +154 | 0 | 13738 | 13738 | 276 | 276 | 276 |
| R19 | B3 sw plateau | +215 | +192 | −23 | 25730 | 25707 | 276 | 308 | 308 |
| R22 | B3 sw always | +51 | +470 | +419 | 31164 | 31579 | 324 | 384 | 388 |

`corr` = phase-reuse-corrected S_alg recomputed from the rerun's own checkpoint
components under `N_H_outer(raw)+N_H_refit(raw)+N_grad(unique)+N_metric(unique)`;
`corr Δ` = raw − corrected. AVQE cells have no archived corrected value.

### 2.4 Rerun-pair trajectory detail (lineage vs main)

| # | Cell | class | first order-diff | first multiset-diff | first S_alg diff | star depth | positions equal | resources equal |
|---|---|---|---|---|---|---|---|---|
| R01 | AVQE ww | trajectory-pre-k\* | r5 | d5 | r1 | 11 | n/a (append) | n/a |
| R02 | AVQE iw | trajectory-pre-k\* | r3 | d3 | r1 | 9 | n/a (append) | n/a |
| R03 | AVQE sw | trajectory-pre-k\* | r4 | d4 | r1 | 10 | n/a (append) | n/a |
| R08 | B3 ww append | bit-identical | — | — | — | 11 | True | True |
| R17 | B3 ww plateau | trajectory-post-k\* | d16 | d16 | d13 | 10 | False | False |
| R20 | B3 ww always | trajectory-pre-k\* | d3 | d3 | d3 | 10 | False | False |
| R09 | B3 iw append | bit-identical | — | — | — | 9 | True | True |
| R18 | B3 iw plateau | trajectory-post-k\* | d11 | d11 | d11 | 8 | False | False |
| R21 | B3 iw always | trajectory-pre-k\* | d4 | d4 | d4 | 8 | False | False |
| R10 | B3 sw append | trajectory-post-k\* | d16 | d16 | d16 | 10 | True | False |
| R19 | B3 sw plateau | trajectory-post-k\* | d16 | d16 | d8 | 9 | False | False |
| R22 | B3 sw always | trajectory-pre-k\* | d4 | d4 | d3 | 9 | False | False |

Two cells (R19, R17) have S_alg diverging **before** any operator diverges (d8 vs
d16; d13 vs d16) — accounting-only drift on an identical ansatz. All three AVQE
screens diverge in S_alg from **round 1**, four rounds before the first label change.

---

## 3. Pass-1 findings (paper vs integration lineage) — carried forward

Source: `$B/pass1_parity_scores.md`, `$B/pass1_scores.json`. All values re-verified
by `$B/score_threeway.py` (12/12 `OK`).

1. **11 / 12 cells match the star physics.** Every cell is a formal MISMATCH because
   `S_alg` is a zero-tolerance integer row, but in 11 cells the deviation is
   `S_alg`-only or post-k\*, with the whole prefix through k\* inside Tier A/B. The
   cleanest cell is R10 (sw append) at `dE_star` dev **4.4409e-16**; the append arms
   (R08/R09/R10) hold Tier A/B across all 20 rounds with no break at all.
2. **`S_alg` drift is EST-08, and it is entirely `N_H_refit`.** For AVQE the
   decomposition is decisive: `N_H_outer` and `N_grad` reproduce exactly
   (ww k\*=11: `N_grad` = 10428 = 948 × 11, in all three regimes), and the entire
   delta sits in `N_H_refit` — Powell callback counts, exactly the EST-08 caveat the
   plan raises for Block 1 ("do not score them as exact"). The golden AVQE prefix
   matches the rerun's `cumulative_unique_primitives.S_unique` **exactly** on the
   deterministic head (rounds 1–2 ww; 1–3 iw and sw), before Powell drift enters.
3. **The archive disagrees with itself on `S_alg` by −48 … +103.** Same family key,
   two archived lanes (`b9` − `b3` at k\*):

   | key | b3 | b9 | b9−b3 |
   |---|---|---|---|
   | `b_depth_append_ra\|weak_weak` | 15150 | 15205 | +55 |
   | `b_depth_costexp_plateau_position\|weak_weak` | 15150 | 15205 | +55 |
   | `b_depth_costexp_always_open\|weak_weak` | 38145 | 38097 | **−48** |
   | `b_depth_append_ra\|intermediate_weak` | 11936 | 11972 | +36 |
   | `b_depth_costexp_plateau_position\|intermediate_weak` | 11936 | 11972 | +36 |
   | `b_depth_costexp_always_open\|intermediate_weak` | 25769 | 25805 | +36 |
   | `b_depth_append_ra\|strong_weak_u8` | 13860 | 13963 | **+103** |
   | `b_depth_costexp_plateau_position\|strong_weak_u8` | 25823 | 25868 | +45 |
   | `b_depth_costexp_always_open\|strong_weak_u8` | 31497 | 31542 | +45 |

   The lineage rerun's deltas (−84 … +242, R21 excepted) are the same order as the
   archive's own internal spread.
4. **R21 (`intermediate_weak` always) is a genuine round-3 tie-flip.** A different
   candidate is accepted at round 3: archive dE **5.0262414161747426e-02**, rerun
   **2.4623522412749344e-02**, |Δ| **2.5639e-02** at ansatz depth 3. After that the
   rerun runs exactly one operator ahead of the archive (rerun d3…d7 reproduce
   archive d4…d8 to ≤ 6e-11), so the star row lands on the wrong operator:
   `dE_star` **6.7812274917e-04** (archive) vs **5.9224108101e-04** (rerun), dev
   **8.5882e-05**; `salg_star` +2458 (25769 → 28227). Worst cell of the pass.
5. **The phase-reuse correction reproduces exactly in 5 / 9 flat cells** (raw −
   corrected = 336 ww append/plateau, 224 iw append/plateau, 276 sw append, in both
   archive and rerun). It differs in ww always (448 archive vs 496 rerun), iw always
   (350 vs 364), sw plateau (276 vs 308), sw always (324 vs 384).
6. **Unverifiable rows (all 12 cells):** routed `tuple_star` N2q/D2q/Dc/W1q (the
   reruns record only `table_i_basis_gate_transpile_v1` with `coupling_map: null`;
   archived tuples are FakeMarrakesh-routed), AVQE `tuple_star`
   (`compiled_resources_by_round` empty — Template A run without `resource_rounds`),
   `k_term`/`dE_term`/`salg_term`/`dE_45` (k_term is 50/75, outside the k=20
   amendment), `k_star` itself (a fixed AVQE-lane constant, not re-derivable from a
   20-round prefix), and the accepted operator sequence (the archive stores only
   `[k, dE]` traces).
7. **Protocol identity holds for the lineage tree** on pool (948 singletons,
   `ordered_labels_sha256 02995a2c570d4322…`, regime-parameterised pool sha
   `c1b03515b02f8daf…` ww), seeds `{adapt: 7, transpiler: 7}`, optimizer powell/200,
   compile cost policy `explicit_depth_qiskit_cost_v1` (w2q .3 / wd2q .3 / wd .25),
   bundle manifest `523f7b0ec4d60a5b…`. Only `horizon` differs (20 vs 50), which the
   k=20 amendment authorises.

---

## 4. Main-lane findings (`main` @ 25cb0745)

### 4.1 Builder-signature drift — the kwarg is gone

The first main attempt (phase MC) failed closed in all 9 B3 cells
(`$B/runs2.log:358–429`):

```
File "$B/drv_c.py", line 13, in <module>
    req = build_paper_i_ra_all_phase_adaptive_forced_k50_request(
TypeError: build_paper_i_ra_all_phase_adaptive_forced_k50_request()
           got an unexpected keyword argument 'compile_cost_policy'
```

The driver was made signature-adaptive and rerun as phase MC2
(`$B/runs3.log`, 9/9 `rc=0`, peak RSS 1918–5842 MB); every cell records
`builder_signature: WITHOUT compile_cost_policy (tree default in effect — record as
lineage drift)` (`$B/runs3.log:2,102,182,262,362,442,522,622,702`).

### 4.2 What the tree default changed

Protocol diff, `$B/pi_integration/b3_weak_weak_append_k20/result.json` vs
`$B/main_tree/b3_weak_weak_append_k20/result.json` (identical driver, identical
regime/arm/horizon):

| Field | Lineage | Main |
|---|---|---|
| `request/method/compile_cost` | `{kind: explicit_depth_qiskit_cost_v1, w_2q .3, w_depth_2q .3, w_depth .25, w_1q 0, w_size 0}` | **absent** |
| `route_contract/semantic_invariants/compile_cost_policy` (+ weights) | `explicit_depth_qiskit_cost_v1` | **absent** |
| `route_contract/semantic_invariants/canonical_{admission,insertion,pruning,beam,composition}_policy`, `compatibility_resolution_active`, `phase1_phase2_lambda_f_proxy_active` | present | **absent** |
| `route_contract/execution_settings/phase3_backend_w_{2q,depth_2q,depth,1q,size}`, `phase3_shadow_damping_policy` | present | **absent** |
| `route_contract/native_semantic_contract/signed_compile_deltas` | `[delta_N2q, delta_D2q, delta_D, delta_N1q]` | `[delta_N2q, delta_D2q, delta_N1q]` — **`delta_D` dropped** |
| `route_contract/route_profile` | `…__insertion-append_only__cost-explicit_depth_qiskit_cost_v1` | `…__insertion-append_only` |
| `lineage_authority/parent_route_profile` | `paper_i_canonical_sr_snake__admission-singleton__insertion-append_only__cost-explicit_depth_qiskit_cost_v1__pruning-off__beam-off_v1` | `supported_projected_generalized_source_metric_no_overlap_trust_full_response_symmetric_cost_no_prune_v1` |
| `bundle_manifest_sha256` | `523f7b0ec4d60a5b…` | `7a7fbf992e213ea5…` |
| `route_contract/sha256` | `5e426e42f161c206…` | `28976b22f6204689…` |
| protocol `sha256` (ww append) | `d18d4baba882b06c…` | `a80a786971ba2bdc…` |

Unchanged across trees: route-contract schema `paper_i_ra_adapt_route_contract_v1`,
`compile_identity` (`table_i_basis_gate_transpile_v1`, basis gates, `coupling_map:
null`, opt-level 0, transpiler seed 7), `estimator_accounting_convention`
(`s_alg_equals_n_h_outer_plus_n_h_refit_plus_n_grad_plus_n_metric_v1`), pool
(948, same sha), seeds, optimizer powell/200, horizon 20.

**Effect on the compared quantities.** The reported-resource convention is
untouched, but the *cost term that scores candidates* loses its total-depth channel
(`delta_D`). That is arm-selective and matches the data exactly:

- **Append arms** (no position scoring): ww and iw are **bit-identical** across the
  two trees at all 20 depths — dE, `S_alg`, operator labels, insertion positions and
  compiled resources all equal. sw append first differs at d16, past k\*=10, at
  |ΔdE| 7.4621e-11.
- **Insertion arms**: plateau diverges post-k\* (d16/d11/d16); **all three `always`
  arms diverge at or before k\*** (d3/d4/d4).
- **Two-qubit-count corroboration is lost on main.** The rerun
  `compiled_two_qubit_count` at k\* equals the archived
  `tuple_star_proxy.c2q` in **8/9 flat cells on the lineage tree** but only **6/9 on
  main**: ww always 54 (archive/lineage) → **42** (main); sw always 38 → **40**;
  iw always 36 → 34 (lineage) / 32 (main). The ww-always star circuit also changes
  shape: `compiled_two_qubit_depth` 44 → 30, `compiled_total_depth` 234 → 173.

### 4.3 AVQE lane on main (Template A, no cost kwarg involved)

`drv_a.py` never passes `compile_cost_policy`, and the AVQE protocol diff between
trees is limited to `request/execution/user_selected_speed` (absent on main) plus
module source locks (`append_module_sha256` `739b03a2…` → `5bb35c21…`,
`pool_module_sha256` `27df5d81…` → `04e474d4…`, `accepted_refit_module_sha256`
`916c06eb…` → `4a4b8363…`). Pool sha, count, seeds, optimizer and accounting
convention are identical. Yet:

| Regime | k\* | N_H_outer lin/main | N_H_refit lin | N_H_refit main | N_grad lin/main | S_alg lin | S_alg main |
|---|---|---|---|---|---|---|---|
| ww | 11 | 11 / 11 | 2778 | **5141** | 10428 / 10428 | 13217 | 15580 |
| iw | 9 | 9 / 9 | 2367 | **3646** | 8532 / 8532 | 10908 | 12187 |
| sw | 10 | 10 / 10 | 2392 | **4963** | 9480 / 9480 | 11882 | 14453 |

`N_grad` and `N_H_outer` are identical; **100 % of the AVQE S_alg change on main is
`N_H_refit`** (+85 % ww, +54 % iw, +107 % sw at k\*) — the same EST-08 channel as
pass 1, amplified. S_alg differs from **round 1** (ww −15, iw +35, sw +14), i.e.
before any selection difference.

The AVQE label divergences are **energy-degenerate partner swaps**, not different
physics: at the first differing round the two trees pick mirror-partner singletons
and the energies agree to machine precision — ww r5 `guarded_singleton::eeeyezee`
(lin) vs `guarded_singleton::eyeeezee` (main), |ΔE| **1.11e-16**, and the pair is
swapped back at r6 (|ΔE| 1.33e-15); iw r3 `eeeeezxy` vs `eeeexyez`, |ΔE| 2.22e-16;
sw r4 `eyeeeeez` vs `eeeyeeez`, |ΔE| 4.00e-15. In ww the prefix multiset re-converges
every other round (differs only at d5, 8, 12, 16, 20) and **is identical at k\*=11**;
in iw and sw the multiset still differs at k\* while dE\* stays inside Tier A/B
(5.37e-13, 3.56e-11). Main's AVQE traces stay Tier A/B through k\* in all three
regimes and break only at r20/r18/r17.

---

## 5. Blocked and failed lanes (evidence, not interpretation)

### 5.1 B3-MR template — blocked on BOTH trees

9/9 cells on the integration tree (`$B/runs2.log:2–172`, phase B2) and 9/9 on main
(`$B/runs2.log:187–357`, phase MB) fail closed at protocol materialization, 2 s each,
peak RSS 1–2 MB:

```
File "…/pipelines/static_adapt/ra_adapt/…", in preflight_paper_i_ra_semantic
    identity, _regime_id = _require_canonical_paper_i_semantic_application(
ValueError: Canonical Paper-I semantic materialization requires its typed adapter.
```

(An earlier variant of the driver failed one frame later, at
`pipelines/static_adapt/ra_adapt/engine.py:6713`,
`ValueError: operational_controls requires a validated, bundle-resolved RA protocol.`
— `$B/runs.log:18–107`.) The failure is identical on both lineages, so it is **not**
a lineage question: the MR (multi-reference) cells need a dedicated MR-builder
investigation to construct the typed adapter the preflight demands. **Blocks plan
rows R04–R07 / the B3-MR block entirely.** No MR cell was scored.

### 5.2 L=3 noise campaign — ran 2711 s, then failed closed

Phase D2, detached worktree at `c2cd9379` (`$B/runs2.log:434–5023`), peak RSS
2896 MB / limit 4000 MB. Two of three cells completed —
`l3_noise_low_h20 done, floor 7.739e-06` (`runs2.log:1960`),
`l3_noise_high_h20 done, floor 3.422e-03` (`runs2.log:3482`) — and the third,
`l3_noise_extreme_h20` (`runs2.log:3483`), aborted:

```
File "…/appendix_wt/chtc/ra_adapt_appendix_noise_20260821_v1/
      run_local_ra_adapt_appendix_noise_l3_20260821_v1.py", line 166, in _run_ra_cell
    result = run_ra_adapt(problem, request)
File "…/appendix_wt/pipelines/static_adapt/ra_adapt/engine.py", line 5946,
      in _accepted_round_scientific_receipts
    raise RuntimeError(
RuntimeError: Canonical RA v2 full-vs-active candidate-gain receipt does not close.
```

Stated as a determinism/parity finding for physics review: the engine's own
full-vs-active candidate-gain receipt does not close for the extreme-noise cell at
`c2cd9379`, after ~45 minutes of successful low- and high-noise execution. No cause
is asserted here beyond the quoted receipt failure. (An earlier attempt, phase D on
the same script from `$B/appendix_demos_tree`, died in 2 s for an unrelated
environment reason — `git status --porcelain` exit 128 inside the non-repo export
tree, `$B/runs.log:1059–1080`.)

---

## 6. Exclusions in force for this pass

1. **k = 50 / 75 full-horizon blocks deferred** per the user's compute cap and the
   k=20 scope amendment. Consequence: the archived `k_term`, `dE_term`, `salg_term`
   and `dE_45` rows (k_term 50 for B3 flat, 75 for AVQE) are **UNVERIFIABLE** in this
   pass for all 12 cells, and `k_star` (11/9/10) remains an imported constant rather
   than a re-derived quantity.
2. **Pure-Hubbard L=4 matched-k75** — CHTC-scale, fetch-verified path only
   (plan §4.1): RA `result.tar.gz` sha256
   `1d0ac8416d124ba6a85118f8a1a228b4f754c5567c0eb341db515426fdfca614`, member
   `./ra_l3_matched_k75/result.json`, receipts
   `chtc/ra_adapt_l4_qiskit_proxy_extensions_20260825/submission_receipt_v3.json`
   (cluster 9697719). Not rerun locally; the fetch lane is the live worktree
   `/Users/lordskyler/Holstein_test.worktrees/chtc-rescue-fetch-20260831` (`f41787a4`).
3. Plan §4 exclusions 2–7 remain in force unchanged (Fig-3 L=3 optional controls;
   strong-Holstein sector; sealed CHTC worker replay; AVQE r0→50 original producer;
   legacy Table-III source-lock family; HH reruns from main @ `dafbda5e`). The
   main-lane runs in this report are at `25cb0745` and are **diagnostic controls for
   the §6.3 code-lineage question, not parity candidates** — §4.6's reasoning
   (route-contract drift) is precisely what §4.2 above measures.
4. Routed `tuple_star` scoring (FakeMarrakesh `_routed_prefix_tuple` re-compile) not
   attempted in either pass.

---

## 7. For the user — decisions this pass puts on the table

1. **R21 interpretation (iw `always`, round-3 tie-flip).** Both trees reproduce the
   *same* divergence from the archive with the *same* numbers (|ΔdE| 2.5639e-02 @ d3;
   dE\* 6.7812274917e-04 archive vs 5.9224108101e-04 lineage / 5.9224108743e-04 main;
   the two reruns agree with each other to 1.06e-10 through k\*). It is therefore
   **not** a lineage artefact and not reproducible-away by choosing a tree. Decide:
   (a) accept the archived R21 row as a stale/degenerate tie and re-derive it, (b)
   re-run R21 at full horizon k=50 to see whether the flip survives, or (c) annotate
   Table I / Table III for this cell. Nothing else in the 12-cell matrix depends on
   the answer.
2. **Noise-campaign receipt failure.** `RuntimeError: Canonical RA v2 full-vs-active
   candidate-gain receipt does not close` at `pipelines/static_adapt/ra_adapt/engine.py:5946`,
   reproducible cost ~45 min to reach. Decide whether this goes to physics review as
   a determinism finding on the appendix-noise lane (extreme-noise cell only; low and
   high closed cleanly, floors 7.739e-06 and 3.422e-03), and whether the appendix
   noise figures may be cited while it is open.
3. **EST-08 `S_alg` policy.** No cell matches `S_alg` exactly on either tree, and the
   archive disagrees with itself by −48 … +103 on the same keys. The deterministic
   channels (`N_H_outer`, `N_grad`, and `N_metric` where present) reproduce exactly;
   all drift is `N_H_refit` (Powell callbacks). Decide the reporting policy: score
   `S_alg` at printed precision (E(1e3)) with a stated tolerance, publish the
   corrected-`S_alg` recipe as the normative one (it reproduces exactly in 5/9 flat
   cells and identically on both trees in 7/9), or keep zero-tolerance and accept
   12/12 formal mismatches.
4. **§6.3 code-lineage implications.** The three-way verdicts say the lineages are
   *not* interchangeable for Paper I, and say exactly where:
   - `main` cannot express the paper's compile-cost policy at all — the builder kwarg
     `compile_cost_policy` no longer exists, and the resulting route contract drops
     `delta_D` from `signed_compile_deltas` plus seven semantic invariants (§4.2).
   - Where the cost term does not enter selection, the trees are **bit-identical**
     (ww and iw append, all 20 depths). Where it does, `main` selects different
     operators — beyond k\* for the plateau arms, and **at or before k\* for all
     three `always` arms**, which changes the Table I star circuit (ww always
     N2q 54 → 42, D2q 44 → 30, Dc 234 → 173) and breaks the archived
     `tuple_star_proxy.c2q` corroboration in 3/9 cells vs 1/9 on the lineage tree.
   - Independently, `main`'s AVQE lane inflates `N_H_refit` by 54–107 % at k\* with
     `N_grad` untouched.
   Decide: keep `origin/paper-i-integration-20260831` (`f041a5c1`) as the
   reproduction lineage of record for Paper I and forward-port the compile-cost
   policy into `main` before any Paper I number is regenerated there; or accept
   `main` and re-derive the insertion-arm rows. The append-arm rows are portable
   either way.
5. **B3-MR unblocking.** Both trees fail identically at
   `Canonical Paper-I semantic materialization requires its typed adapter`; this needs
   a dedicated MR-builder investigation before rows R04–R07 can be scored at all.

---

## CORRECTION (2026-09-01, from the compile-cost port verification)

This report's §4.2/§6.3 causal attribution is **falsified**. "main cannot
express the compile-cost policy" was true and is now fixed (commit
`ef842070`: additive port; without the kwarg, protocols are byte-identical
— route sha `28976b22…`, profiles, invariants, failing-test ID sets all
unmoved). But "where the cost term enters selection, main selects
different operators" is **not the mechanism**: A/B on the campaign lineage
itself (`f041a5c1`, with vs without `compile_cost_policy`) produces
identical trajectories to 16 digits — the policy is contract-declarative
and receipt-level on BOTH lineages for the plain insertion arms; plain-arm
selection uses the zero-centered 3-channel arctan factor (verified
numerically against the lineage's stored round-3 receipt).

The actual always/plateau divergence: at the divergence round both trees
compute the SAME winning `selection_score` to 16 digits for DIFFERENT
candidates — exact ties resolved by shortlist membership/ordering
(`descending_tie_break_then_pool_position_record_identity_v1`), and the
two refactors enumerate disjoint candidate families at that tie (phase-
0/I/II position-record pipeline: adapt_pipeline.py 30,517 vs 75,040
lines, hh_continuation_scoring.py, ra_adapt/pools.py, plus lineage-only
run_axes.py/costexp_realization.py). This is the recovery's known
26-vs-7 wholesale-merge territory and is not per-hunk separable.

Consequences: (1) porting the policy does NOT make main reproduce the
paper's insertion-arm rows; (2) `paper-i-integration-20260831` @
`f041a5c1` stands as the reproduction lineage of record for those rows,
on stronger grounds than before; (3) the paper's costexp-arm identity
rests on the lineage's candidate-pipeline code, not the declared weights
— relevant to any provenance statement. Evidence: port-branch report,
scratch `port/` A/B scripts and cell extracts.
