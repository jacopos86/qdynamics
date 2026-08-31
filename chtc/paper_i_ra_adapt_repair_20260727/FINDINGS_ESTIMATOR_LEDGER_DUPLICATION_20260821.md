# Findings — RA estimator ledger charges duplicate candidate acquisitions

Written 2026-08-21 by the Claude manuscript-lane session, for cross-reference by
the concurrent Codex session repairing the estimator route on this branch.

**This document is evidence, not instructions.** It records what was measured,
from which artifact, and which claims are verified versus still open. Nothing
here prescribes a repair; where a repair choice exists, the alternatives and the
reason the choice matters are stated so both sessions converge on the same
definition rather than two different ones.

Branch at writing: `ra-refactor-stage2` @ `5775f0eb` (clean). No code was
modified by the authoring session — the work below is read-only analysis.

---

## 1. Finding

RA's reported estimator ledger `S_alg` charges, once per outer round, a
**candidate gradient and a candidate self-metric that were already acquired
earlier in the same round at the same fixed source state**. Phase II re-measures
what Phase 0 (gradient) and Phase I (self-metric `G_rr`) already measured. Only
the Phase-II self-Hessian `H_rr` is new work.

The magnitude is `2 · R2` logical invocations per round, where `R2` is the
Phase-II input population.

The ledger already detects these as duplicates — it keys primitives on
`(projective_state_fingerprint, observable_or_formula_identity, operand)` and
returns `charged=False` on a repeat — but `S_alg` is defined as **raw
occurrences**, so they are counted anyway.

**Direction: this inflates RA and not AVQE.** AVQE carries no candidate
duplicates at all (measured below). Every published RA-vs-AVQE estimator
comparison therefore sets a duplicate-inflated RA ledger against a clean AVQE
one. Correcting it moves RA's numbers favourably; no published claim is
overstated because of it.

---

## 2. Evidence

### Artifact

| field | value |
|---|---|
| cell | `sixregime__b_depth_append_ra__strong_weak_u8__nph3` (S–W, RA-Append) |
| archive | `output/local_runs/ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__b_depth_append_ra__strong_weak_u8__nph3__9664325__14.tar.gz` |
| archive sha256 | `eb28a89dfe6b47348dcccc8c130e7d3e7d6a6e12a9722dba9deb4fe634ad9a13` |
| member | `./run/result/result.json` (129,831,338 bytes) |
| member sha256 | `c81d30dd25a70ed265e8a1029c751df619a3b5e171c0e4b359be601fabd6585f` |
| exit status | 0 (completed, ledger closure passed) |
| why this cell | it is the source of Table I's S–W RA-Append `E = 13.9` |

Cardinality sources inside `scientific_receipts.accepted_round_receipts[j]`:

- `R0` = `ra_gradient_phase0_shortlist.adaptive_decision.input_candidate_count`
- `R1` = `ra_gradient_phase0_shortlist.adaptive_decision.effective_shortlist_size`
- `R2` = `projected_phase3_population_receipt.phase2_input_child_count`
- `R3` = `projected_phase3_population_receipt.phase3_evaluated_candidate_count`

Per-round raw and unique deltas from
`scientific_receipts.controller_replay_evidence.resume_sidecar_closure.all_estimator_prefix_receipts[j]`,
fields `raw_occurrence_delta.components` and `unique_primitive_delta.components`.

### Per-round table, rounds 0–9 (the prefix behind the reported k\*=10)

| j | n | R0 | R1 | R2 | R3 | raw grad | raw metric | dupG | dupM | Powell |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 948 | 24 | 24 | 12 | 972 | 72 | 24 | 24 | 28 |
| 1 | 1 | 948 | 24 | 14 | 2 | 962 | 58 | 14 | 14 | 81 |
| 2 | 2 | 948 | 24 | 24 | 12 | 972 | 126 | 24 | 24 | 238 |
| 3 | 3 | 948 | 16 | 16 | 12 | 964 | 132 | 16 | 16 | 113 |
| 4 | 4 | 948 | 16 | 4 | 4 | 952 | 76 | 4 | 4 | 231 |
| 5 | 5 | 948 | 16 | 8 | 8 | 956 | 142 | 8 | 8 | 340 |
| 6 | 6 | 948 | 12 | 4 | 4 | 952 | 110 | 4 | 4 | 324 |
| 7 | 7 | 948 | 24 | 16 | 12 | 964 | 280 | 16 | 16 | 429 |
| 8 | 8 | 948 | 24 | 8 | 8 | 956 | 240 | 8 | 8 | 497 |
| 9 | 9 | 948 | 24 | 20 | 4 | 968 | 226 | 20 | 20 | 489 |

`dupG = raw − unique` for `N_grad`; `dupM` likewise for `N_metric`.

### Identity, checked over all 50 rounds

- `dupG == R2` in **50 of 50** rounds. No exceptions.
- `dupM == R2` in **48 of 50** rounds. Exceptions: rounds 27 and 34, where
  `R2 = 8` but `dupM = 10`. Unexplained — see §6.

### Mechanism

The Phase-II population is the **identity image** of the Phase-I shortlist, in
every round of this run:

```
split_parent_count      = 0     (all 50 rounds)
split_child_count       = 0     (all 50 rounds)
unsplit_singleton_count = phase1_retained_parent_count   (all 50 rounds)
phase_ii_candidate_exposure = identity_on_retained_singletons_v1
phase_order = phase1_singleton_shortlist_then_identity_phase2_then_phase3
```

The pool is already single-Pauli-word singletons, so nothing splits. The
Phase-II "children" are the same operators at the same insertion position
against the same held-fixed source state. Their gradient and self-metric are
therefore literally the scalars Phase 0 and Phase I already acquired.

Note for whoever repairs this: the comment currently in
`pipelines/exact_bench/paper_i_s_alg_accounting.py` (`snake_clean_prefix_work`,
`projected_singleton` branch) justifies the recharge as *"splitting creates the
child population that enters Phase II. Each child then requires its own
gradient, self metric, and self Hessian."* That justification holds only when
splitting actually occurs. In this route it never does, which is why the
recharge is duplicate work rather than new work.

### Independent corroboration — unique-primitive counters

Round 0 of the same cell, cumulative:

| component | raw | unique | duplicates |
|---|---|---|---|
| `N_H_outer` | 1 | 1 | 0 |
| `N_H_refit` | 28 | 23 | 5 |
| `N_grad` | 972 | **948** | 24 |
| `N_metric` | 72 | **48** | 24 |
| total | 1073 | 1020 | 53 |

`unique N_grad = 948 = R0` — flat across all 50 rounds. That is the Phase-0
full-pool scan and nothing else, which independently confirms that Phase I does
**not** re-acquire the gradient (it correctly reuses Phase 0's) and that the
excess sits entirely in Phase II.

`unique N_metric = 48 = R1 + R2` — the Phase-I `G_rr` plus the Phase-II `H_rr`,
which is exactly the algorithm Appendix E of the manuscript describes.

### AVQE comparator, same regime

`output/local_runs/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/transfer/aavqe_r75cont__strong_weak_u8__nph3__9665004__2.tar.gz`,
member `./run/result.json`, exit 0. Round 0 cumulative:

| component | raw | unique | duplicates |
|---|---|---|---|
| `N_grad` | 948 | 948 | **0** |
| `N_metric` | 0 | 0 | 0 |
| `N_H_refit` | 652 | 651 | 1 |

AVQE duplication is 1 invocation in 1,601 (0.06%). The asymmetry is the reason
this matters beyond efficiency.

---

## 3. Closed form

Per round, for the gradient + metric part only (excluding Powell objectives and
the one outer Hamiltonian evaluation):

| ledger | expression |
|---|---|
| Manuscript Appendix E, Eq. (E2) as written | `R0 + 2·R1 + R2 + n(n+1) + 2n·R3` |
| What the runs actually charge (reported `S_alg`) | `R0 + R1 + 3·R2 + n(n+1) + 2n·R3` |
| With candidate reuse (the unique ledger) | `R0 + R1 + R2 + n(n+1) + 2n·R3` |

All three were checked against the artifact. The middle expression reproduces
the observed per-round ledger **exactly, every round, with zero residual**. The
manuscript expression differs from the observed one by `2·R2 − R1` per round,
which happens to nearly cancel over this run's 50 rounds (total 111,183 vs
111,175, 0.007%) purely because `R1 ≈ 24` and `R2` averages near 12. That
near-agreement is arithmetic luck, not correctness.

Two independent defects, pointing opposite ways:

- The **implementation** over-charges by `2·R2` per round (this finding).
- The **manuscript** over-charges by `R1` per round, because Eq. (E2) charges
  `g_r` at Phase I when the runs correctly reuse Phase 0's gradient there.

---

## 4. Corrected totals for this cell

| prefix | reported `S_alg` (raw) | candidate duplicates | corrected | inflation |
|---|---|---|---|---|
| k = 10 (reported k\*) | 13,860 | 276 | 13,584 | 2.03% |
| k = 50 (terminal) | 130,024 | 1,126 | 128,898 | 0.87% |

Effect on the reported comparison at k\*=10, against AVQE's 11,845:

- RA premium as published: **+17.0%**
- RA premium after removing candidate duplicates: **+14.7%**

---

## 5. A definition choice that both sessions must make the same way

**Do not correct by substituting `S_unique` wholesale.** The unique ledger also
deduplicates `N_H_refit` and `N_H_outer`, and those deduplications are wrong to
apply here:

- Powell genuinely re-evaluates the Hamiltonian at repeated parameter points —
  998 such repeats in this cell. The manuscript deliberately charges every
  optimizer callback, and AVQE is charged the same way, so removing them breaks
  the symmetry of the comparison.
- `N_H_outer` unique appears to deduplicate *across* rounds. In the L=3 closure
  (`results_l3_v9_extra/metric_ablation_gated_norefit/result.json`) it collapses
  to 1 for a 15-round run, against a raw count of 15. Each round's outer
  evaluation is at a different accepted state, so that collapse is not
  physical.

Substituting `S_unique` for the S–W cell at k\*=10 gives 13,375 and a +12.9%
premium. The component-selective figure is 13,584 and +14.7%. The second is the
defensible one.

The correction that removes candidate re-acquisition and nothing else:

```
E_corrected = N_H_outer(raw) + N_H_refit(raw) + N_grad(unique) + N_metric(unique)
```

This is a no-op on AVQE, so the comparison stays symmetric.

A cheap self-check worth keeping in whatever recount is built: the correction
is derivable two independent ways — from the per-round `unique_primitive_delta`
fields, and from `2·R2` read off the phase receipts. Asserting the two agree
turns a silent accounting drift into a loud failure. It already catches the two
anomalous rounds in §2.

---

## 6. Open — not established

- **Rounds 27 and 34** carry `dupM = 10` against `R2 = 8`. Two extra metric
  duplicates each, cause unknown. Possibly a repeated Phase-III scaffold pair.
  Small, but it means the `dupM == R2` identity is empirical rather than proven.
- **One cell only.** The identity is verified on S–W append across 50 rounds.
  The other four completed append cells (`weak_weak`, `intermediate_weak`,
  `weak_strong`, `intermediate_strong`) have not been checked.
- **Branch code versus shipped overlay.** The analysis above reads
  `adapt_pipeline.py` and `paper_i_s_alg_accounting.py` from this branch, but the
  archives carry their own frozen `source/` overlay. If the campaign ran an
  older pipeline, the duplication could originate somewhere other than the call
  sites traced here. Comparing the overlay's
  `_record_candidate_geometry_primitives` and `_full_record_for_candidate`
  against the branch copy would settle it.
- **Insertion routes are unverifiable from this campaign.** Appendix E's central
  claim is that estimator work is charged once per commutation equivalence
  class. This cell is append-only (`interior_scored_count = 0` throughout), so
  classes are trivial and the claim is untested. See §7.

---

## 7. Adjacent finding, separate from the duplication

Scoped to one campaign only. In
`ra_adapt_costexp_depthfix_sixregime_20260820_v2`, the insertion arms did not
complete: all three `b_depth_costexp_always_open` cells, both
`b_depth_costexp_plateau_position` cells, and `b_depth_append_ra__strong_strong`
exited status 1 with no `run/result/result.json` — only checkpoints and a
partial `estimator_ledger.json`. The append arms in that campaign exited 0.

**This says nothing about where the reported insertion numbers come from.**
Completed insertion runs exist in the earlier campaign
`paper_i_ra_allphase_adaptive_20260817_comparison_data/archives/` — for example
`allphase_maxk50__always_open_position_phase0__strong_weak_u8` and
`__weak_weak`, both exit 0 with `result.json` present. The sixregime insertion
jobs were a later attempt that failed; they are not the provenance of the
reported values.

An earlier revision of this section inferred that Table I's insertion rows
rested on failed runs. That inference was wrong and has been removed. The
`salg_star: null` entries in the `live_star` family of
`results_tooling/kstar_tables.json` mean the estimator ledger was not captured
in that particular extraction path, not that a run failed.

Relevant to the repair only in this sense: if those sixregime cells are re-run,
that is a natural point at which a runtime-side reuse fix could land without
leaving completed and new cells on different ledger semantics.

---

## 8. Other defects found in `pipelines/exact_bench/paper_i_s_alg_accounting.py`

Both would surface for anyone trying to use `snake_clean_prefix_work` as the
recount path:

1. **It cannot read the reported artifacts.** It requires
   `controller_measurement_work_proxy.by_phase` on each history row.
   That key appears **zero times** in the S–W result.json. The cardinalities it
   needs exist, under the different key names listed in §2.
2. **It charges `+n` active-coordinate gradients unconditionally**, in both the
   `intact_macro` and `projected_singleton` branches. The Hubbard–Holstein runs
   do not acquire them — `policy.active_gradient_charge = 0`,
   `policy.active_gradient_indices_acquired = []`,
   `policy.active_gradient_policy = "stationary_source_response_v1"`. For a
   50-round prefix that is an over-count of `Σ n_j = 1225`. The `+n` is correct
   only under measured-residual scoring; it needs to follow
   `active_gradient_policy` rather than assume.

---

## 9. Reproduction

```bash
SC=/tmp/estimator_audit && mkdir -p "$SC"
LR=/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs
A="$LR/ra_adapt_costexp_depthfix_sixregime_20260820_v2/transfer/sixregime__b_depth_append_ra__strong_weak_u8__nph3__9664325__14.tar.gz"
tar -xzOf "$A" ./run/result/result.json > "$SC/SW_append_result.json"
```

Then, per round `j`, compare
`resume_sidecar_closure.all_estimator_prefix_receipts[j].raw_occurrence_delta.components`
against `.unique_primitive_delta.components`, and read `R0/R1/R2/R3` from
`accepted_round_receipts[j]` using the field paths in §2.

Peak RSS for a full `json.load` of this file is ~0.5 GB, well inside the local
agent ceiling. The insertion-arm archives are not loadable the same way — their
checkpoints run to 3.6 GB and the estimator ledger to 1.6 GB, and they have no
`result.json` at all.

---

## 10. Concurrency

Both sessions are on `ra-refactor-stage2`. Files most likely to collide:

- `pipelines/exact_bench/paper_i_s_alg_accounting.py` (§8 defects)
- `pipelines/static_adapt/adapt_pipeline.py` (the `_record_*_primitives`
  family, ~line 56180–56400 in the branch copy)

Repo invariant: never `git add -A` in this checkout — other sessions share it.
Commit explicit paths only.

The manuscript is a **separate repository**
(`/Users/jakestrobel/local_repos/ADAPT---Paper-I`, branch `main`, file
`Paper_I_overleaf_20260818/Paper_I.tex`), Overleaf-synced, not on this branch.
The Appendix E correction in §3 belongs there and is being handled by the
authoring session; it is recorded here only so the two ledgers are not
independently redefined.

---

## 11. Independent Codex verification (2026-08-21)

Codex independently streamed the frozen S--W RA-Append estimator ledger from
the archive named in §2.  The recount loaded one occurrence at a time with
`ijson`; it did not extract or deserialize the 130-million-byte result payload.
The reporting rule was component-selective:

```
E_corrected = N_H_outer(raw) + N_H_refit(raw)
            + N_grad(unique) + N_metric(unique)
```

Thus every Hamiltonian invocation remains charged, including repeated Powell
callbacks, while a gradient or metric primitive passed to a later selection
phase is charged only at its first acquisition.

The independent recount reproduces the §4 values exactly:

| prefix | raw | removed `N_grad` | removed `N_metric` | corrected |
|---|---:|---:|---:|---:|
| k = 10 | 13,860 | 138 | 138 | **13,584** |
| k = 50 | 130,024 | 561 | 565 | **128,898** |

The four-count terminal discrepancy is resolved rather than smoothed over:
561 metric repeats have scope `full_candidate_self_metric_recompute`, while
four additional metric repeats have scope `phase3_scaffold_geometry` (the two
extra occurrences in each of rounds 27 and 34 noted in §2).  This is why a
scope-name-only correction gives 128,902, whereas the component-selective rule
above gives 128,898.  The latter follows the manuscript's mathematical reuse
contract and the ledger's primitive identities.

The bounded-memory recount script is currently at
`/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/chtc/paper_i_ra_adapt_repair_20260727/recount_estimator_phase_reuse.py`.
It changes no runtime accounting code.  Codex will use the same rule, one
archive at a time, for all paper-facing RA values and will regenerate every
dependent percentage and annotation from those corrected prefixes.

---

## 12. Paper-facing recount completed by Codex

The same component-selective rule has now been applied sequentially to every
RA prefix currently reported in Paper I.  AVQE values are unchanged.

### Hubbard--Holstein Table I and same-accuracy crossings

| regime / route | prefix | raw | corrected |
|---|---:|---:|---:|
| W--W RA-Append | 11 | 15,150 | **14,814** |
| I--W RA-Append | 9 | 11,936 | **11,712** |
| S--W RA-Append | 10 | 13,860 | **13,584** |
| W--W RA-Plateau | 11 | 15,150 | **14,814** |
| I--W RA-Plateau | 9 | 11,936 | **11,712** |
| S--W RA-Plateau | 10 | 25,823 | **25,547** |
| W--W RA-Insert | 11 | 38,145 | **37,697** |
| I--W RA-Insert | 9 | 25,769 | **25,419** |
| S--W RA-Insert | 10 | 31,497 | **31,173** |
| W--S RA-Append | 75 | 850,241 | **846,941** |
| I--S RA-Append | 75 | 861,886 | **858,576** |
| S--S RA-Append | 27 | 204,401 | **203,521** |
| W--S RA-Append | crossing 68 | 733,824 | **730,836** |
| I--S RA-Append | crossing 63 | 678,541 | **675,771** |

The resulting append-only comparisons are: weak-sector premiums
9.4/7.4/14.7% (mean 10.5%); strong-sector changes -30.3/-4.0/+1.6%; and
same-accuracy estimator reductions 39.8/24.4% in W--S/I--S.

### Pure-Hubbard optional controls

| route | marked prefix | raw | corrected |
|---|---:|---:|---:|
| RA-Append | 12 | 6,641 | **6,065** |
| RA + batch | 6 | 4,791 | **4,503** |
| RA + beam, selected lineage | 12 | 6,898 | **6,340** |
| RA + pruning | 12 | 7,597 | **7,021** |

For beam, primitive uniqueness was recomputed *within the twelve winning
branch ids plus unbranched initialization*.  Applying the ledger's global
`charged` flag would incorrectly deduplicate work first performed on discarded
siblings.  The selected-lineage recount contains exactly 6,898 raw
occurrences and removes 279 repeated gradients plus 279 repeated metrics.

The corrected first-numerical-precision provenance values are RA-Append k=8:
3,616; batch k=3: 1,985; beam selected lineage k=8: 3,851; pruning k=8:
3,616.  Their AVQE comparator remains 6,510.

At the displayed prefixes, the corrected estimator reductions relative to
AVQE are 18.9/39.8/15.2/6.1% for append/batch/beam/pruning (mean 20.0%).
Relative to RA-Append, batch uses 25.8% fewer estimators, beam 4.5% more, and
pruning 15.8% more.

### Fixed-seed noise diagnostics

| noise | prefix | raw RA | corrected RA |
|---|---:|---:|---:|
| low | 10 | 7,034 | **6,794** |
| high | 10 | 6,737 | **6,497** |
| extreme | 10 | 5,011 | **4,768** |
| low | 20 | 24,033 | **23,541** |
| high | 20 | 22,233 | **21,750** |
| extreme | 20 | 19,399 | **18,910** |

The noisy-gradient identities generally differ across stochastic acquisitions,
so the recount does not merge them.  It removes the exact self-metric repeats
and only the few additional gradient/metric occurrences whose primitive ids
are actually repeated.

---

## 13. Manuscript application status

Codex has applied the §12 values to the current Paper I working tree:

- Table I, the abstract range, the weak/strong-sector comparison prose, and the
  two same-accuracy crossing percentages;
- the three pure-Hubbard resource matrices and every dependent estimator
  percentage in the Discussion;
- the iteration-10 noise table and the terminal appendix noise annotation;
- the staged-reuse estimator formula/prose in Appendix E.

The machine-readable reporting delta is
`Paper_I_overleaf_20260818/figures/ra_optional_controls_main/estimator_reuse_corrections_20260821.json`.
Both plotting scripts record that file and their direct trajectory inputs in
their generated provenance.  `Paper_I.pdf` was regenerated successfully:
20 pages, no undefined references or citations, and no overfull boxes.  Visual
inspection confirmed the corrected values on manuscript pages 10--12 and the
appendix noise figure on page 17.

No runtime accounting code was changed, and no wholesale `S_unique`
substitution was made.

---

## 13. Bundle 5 (measured residual, cluster 9674089) — measured, affected, corrected (Claude, 2026-08-22)

Paper I's headline evidence is now B5 alone (user decision 2026-08-21), so the
question §6 left open — whether the duplication survives without the
stationarity assumption — was measured directly rather than assumed either way.

**B5 is affected, at the same ~2% k\* rate, through different scopes.** All nine
weak-sector B5 archives ship a complete `./run/result/estimator_ledger.json`
(the receipt-path crash lost only `result.json`). Streaming each with
`recount_estimator_phase_reuse.py` under the identical component-selective rule
(`charged=false` repeats in `N_grad`/`N_metric` only; every Hamiltonian
invocation preserved):

| cell | S\* raw | S\* corrected | terminal raw | terminal corrected |
|---|---:|---:|---:|---:|
| append W–W | 6,409 | **6,277** | 104,093 | **103,493** |
| append I–W | 5,707 | **5,599** | 108,200 | **107,600** |
| append S–W | 5,910 | **5,790** | 88,202 | **87,578** |
| plateau W–W | 8,912 | **8,736** | 281,194 | **275,384** |
| plateau I–W | 5,707 | **5,599** | 283,470 | **277,200** |
| plateau S–W | 10,478 | **10,294** | 300,377 | **291,497** |
| always W–W | 18,168 | **17,776** | 306,248 | **298,653** |
| always I–W | 12,632 | **12,358** | 316,957 | **309,779** |
| always S–W | 14,343 | **14,031** | 307,868 | **300,914** |

k\* inflation 1.8–2.2%; terminal 0.6% (append) to 2.1–3.1% (insertion arms).
Prefix raw counts matched `KT["b5"]` `salg_star`/`salg_term` exactly in all
nine cells (the script's consecutive-sequence assertion also passed).

**Mechanism differs from B3 — a scope-name correction does not transfer.** B5's
repeats carry scopes `gradient_surface` + `phase1_candidate_self_metric`
(~6+6 per round at k\*), with small terminal contributions from
`phase3_scaffold_geometry` and
`historical_singleton_supported_projection_active_gradient`. B3's scope was
`full_candidate_self_metric_recompute`. Any recount filtered to B3's scope
names removes **zero** occurrences from a B5 ledger — the §5 rule
(component-selective, `charged=false`) is the definition that transfers across
bundles; use it, not scope lists.

Overlay check for B5: `./source/pipelines/static_adapt/adapt_pipeline.py` in
the B5 archives carries the same `_record_gradient_surface_primitives` /
`phase1_candidate_self_metric` call sites as the branch copy (offsets shifted
by the measured-residual additions only), so branch-based repair reasoning
remains valid for this campaign.

**Where the corrected values live:** `results_tooling/kstar_tables.json`
`KT["b5"][arm|regime]` now has `salg_star_corrected`, `salg_term_corrected`,
and `salg_correction` (per-scope removal counts), raw values untouched.
Per-cell recount JSONs: session scratchpad `b5_recount_<arm>__<regime>.json`.

**For the Paper-I update: print the corrected B5 values.** The §12 B3-era
table corrections are superseded for the main text by the B5-only decision;
they remain valid wherever B3 numbers survive as appendix/reference material.
The nine strong-sector B5 cells need this same pass as they land, before any
strong-sector B5 number is typed.

**First strong cell confirms the pattern (added 2026-08-22 ~05:00Z).**
`mr_append_ra strong_strong_u8` (nph7, proc 5) exited at the known
receipt-path crash after all 75 rounds; its archive was fetched and
sha256-verified (`7455d998…`, 493,586,503 bytes). Terminal recount:
308,638 → **307,738** (0.29%), removals exactly `gradient_surface` 450 +
`phase1_candidate_self_metric` 450 — the same 6+6 per round as every weak
cell. Its k\* prefix awaits the exact-energy trace ingest. The remaining
eight strong lanes were at rounds 38–74 of 75 at 04:55Z; a tender
(`fleet_tender_b5.sh`) now fetches, sha256-verifies, and clears each archive
as it lands. All nine weak archives plus proc 5 are verified local; the
staging transfer area is empty.
