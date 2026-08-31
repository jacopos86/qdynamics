# Paper I weak-Holstein-sector final results — data handoff (2026-08-20)

Status: final for AAVQE, append, and plateau-insertion. The always-insertion
S_alg values are marked "in transit" below; every other number is terminal
and authenticated. Nothing in this document depends on CHTC — all referenced
files are local.

## Sources

| | |
|---|---|
| Manuscript repo | `/Users/jakestrobel/local_repos/ADAPT---Paper-I`, branch `main`; live source `Paper_I_overleaf_20260818/Paper_I.tex` (Overleaf-synced); builds clean with 0 undefined references as of commit `a047b99` |
| Results PDF | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/pdf/paper_i_ra_allphase_adaptive_20260817/bundle3_final_results.pdf` |
| Provenance manifest | same directory, `bundle3_final_results_manifest.json` — SHA-256 of the PDF and of every input |
| Machine-readable numbers | `.../Holstein_test_fullclone_3.worktrees/ra-refactor-stage2/chtc/paper_i_ra_adapt_repair_20260727/results_tooling/kstar_tables.json`, sections `aavqe`, `b3`, `weak_report_45` |
| Companion docs | `CAMPAIGN_STATE_20260820.md`, `HANDOFF_MANUSCRIPT_cost_reporting_20260819.md`, `SECOND_REVIEW_CODEX_20260820.md` |

Scope of the data: Bundle 3 ("Final RA paper I facing, no-warm-start,
yes-position-cost" — append / plateau-insertion / always-insertion under the
three-term Qiskit cost, cluster 9664325) versus the AAVQE baseline (cluster
9665004), weak Holstein sector: weak_weak, intermediate_weak, strong_weak_u8
(nph 3). Strong-sector cells and Bundles 1/2/4 are still running; their
numbers are not in this document.

## Author conventions on record (Jake, 2026-08-20)

- Weak-sector results are reported at **iteration 45**. Framing: AAVQE was
  run to iteration 75 and did not break its plateau; RA was run to iteration
  50 and the plateaus were unaffected.
- The exception: always-insertion in strong_weak_u8 continues descending past
  the report point. Measured |E−E_exact|: r43 1.14e-11 → r44 8.5e-14 →
  r45 6.7e-14 → r46 8.9e-16 → r47 2.2e-16; terminal r50 4.4e-16. Values
  ≤1e-15 are at the double-precision floor.
- All weak Bundle-3 cells completed their full 50 rounds (verified from
  per-round streams).
- k\* per regime = the AAVQE plateau-entry round (first round within 0.1
  decade of AAVQE's r75 terminal error, when reached by round 60). Weak trio:
  k\* = 11 / 9 / 10 (ww / iw / swu8).

## Measurement provenance

- Cost tuples N2q / D2q / Dc: routed FakeMarrakesh, `optimization_level=1`,
  `seed_transpiler=7`, routing/SWAP included. Insertion-route circuits at
  intermediate rounds are exact — reconstructed by replaying the recorded
  per-round insertion positions (`insertion_position_replay_v1`), not the
  final-ordering approximation the manuscript currently discloses.
- S_alg (𝓔): authenticated per-round estimator ledger counts
  (`accepted_prefix_work` in result payloads; per-round prefix receipts in
  run checkpoints for the plateau cells — the two sources cross-validate at
  r50). AAVQE S_alg is cumulative anchor(r1–50) + continuation(r51–75).
- ΔE = |E − E_exact|, same-cutoff exact reference per regime.

## Numbers — at k\* and at the reporting point r45

Pairs are "at k\* / at r45". k\* per regime is unchanged (11 / 9 / 10).

**weak_weak (k\* = 11):**

| series | dE k\*/45 | N2q | D2q | Dc | S_alg k\*/45 |
|---|---|---|---|---|---|
| AAVQE | 4.6e-04 / 3.7e-04 | 102/388 | 84/309 | 238/962 | 13538 / 61889 |
| append | 3.8e-04 / 1.2e-10 | 46/252 | 38/200 | 134/687 | 15150 / 116430 |
| always-insertion | 5.2e-04 / 6.1e-14 | 69/499 | 51/401 | 177/1288 | in transit |
| plateau-insertion | 5.2e-04 / 5.8e-14 | 46/410 | 38/339 | 134/1082 | 15150 / 544757 |

**intermediate_weak (k\* = 9):**

| series | dE k\*/45 | N2q | D2q | Dc | S_alg k\*/45 |
|---|---|---|---|---|---|
| AAVQE | 2.1e-04 / 1.7e-04 | 35/335 | 30/277 | 101/839 | 10910 / 62100 |
| append | 2.1e-04 / 7.8e-08 | 38/367 | 24/278 | 75/870 | 11936 / 142090 |
| always-insertion | 6.8e-04 / 3.9e-14 | 39/383 | 34/288 | 130/940 | in transit |
| plateau-insertion | 4.3e-04 / 3.8e-14 | 38/469 | 24/353 | 75/1107 | 11936 / 567521 |

**strong_weak_u8 (k\* = 10):**

| series | dE k\*/45 | N2q | D2q | Dc | S_alg k\*/45 |
|---|---|---|---|---|---|
| AAVQE | 1.6e-06 / 1.4e-06 | 65/352 | 47/276 | 152/884 | 11845 / 59268 |
| append | 1.6e-06 / 1.4e-06 | 34/199 | 22/110 | 86/369 | 13860 / 106792 |
| always-insertion | 7.3e-06 / 6.7e-14 | 46/353 | 32/288 | 120/931 | in transit |
| plateau-insertion | 1.8e-06 / 8.0e-07 | 56/204 | 38/154 | 126/496 | 25823 / 563674 |

(The identical k\* S_alg for append and plateau-insertion in ww/iw is real:
plateau-insertion performs identical work to append until its first plateau
event; in strong_weak_u8 a plateau event fires before k\*, hence 25823.)

"in transit": the always-insertion runs are complete; their archives (which
carry the per-round ledgers) are mid-transfer from the execute nodes. The
values slot into the marked cells when they land; this document is amended at
that moment.

## Terminal values (AAVQE r75, RA r50) — reference

| regime | series | dE term | N2q/D2q/Dc term | S_alg term |
|---|---|---|---|---|
| weak_weak | AAVQE | 3.7e-04 | 554/451/1467 | 109710 |
| weak_weak | append | 1.2e-10 | 252/200/687 | 142500 |
| weak_weak | always-insertion | 6.1e-14 | 525/435/1396 | in transit |
| weak_weak | plateau-insertion | 5.9e-14 | 470/399/1277 | 681482 |
| intermediate_weak | AAVQE | 1.7e-04 | 518/416/1362 | 108736 |
| intermediate_weak | append | 7.8e-08 | 401/315/974 | 167510 |
| intermediate_weak | always-insertion | 3.9e-14 | 383/288/940 | in transit |
| intermediate_weak | plateau-insertion | 3.9e-14 | 497/363/1145 | 703294 |
| strong_weak_u8 | AAVQE | 1.4e-06 | 485/394/1263 | 108357 |
| strong_weak_u8 | append | 1.4e-06 | 231/149/487 | 130024 |
| strong_weak_u8 | always-insertion | 4.4e-16 | 353/288/931 | in transit |
| strong_weak_u8 | plateau-insertion | 8.0e-07 | 204/154/496 | 696874 |

## Data-supported observations (optional context)

- Both insertion routes reach the double-precision floor in weak_weak and
  intermediate_weak by r45; append plateaus at 1.2e-10 / 7.8e-08 there.
- In strong_weak_u8 the insertion kinds split: always-insertion reaches the
  floor (4.4e-16); plateau-insertion plateaus at 8.0e-07.
- AAVQE continues paying compile cost along its plateau (e.g. weak_weak N2q
  102 → 388 between k\*=11 and r45) with no accuracy gain.
- Estimator cost ordering at r45: AAVQE ~6e4 < append ~1.1–1.4e5 <
  plateau-insertion ~5.4–5.7e5.
- The six stranded bibliography entries were fixed in manuscript commit
  `a047b99` (six `\bibitem`s moved inside `thebibliography`).
