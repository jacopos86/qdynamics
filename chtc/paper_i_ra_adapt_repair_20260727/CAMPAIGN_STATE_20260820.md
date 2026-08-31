# Campaign state and desired final results — 2026-08-20

Written for a fresh auditing session. Everything here is verifiable from
the referenced clusters, packages, and commits; nothing depends on chat
history.

## Branches

**Code (all campaign work):**
- worktree: `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2`
  (branch `ra-refactor-stage2`, HEAD `24e23efa` at writing — check
  `git rev-parse --short HEAD`; other agents share this tree)
- Decoys: `~/Documents/Holstein_implementation/Holstein_test_fullclone_3`
  (iCloud mirror), base checkout `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3`
  (page-1 builder + fetched results live under its `output/`, but no
  campaign code).

**Paper I manuscript (separate repo):**
- remote: https://github.com/20jastrobel/ADAPT---Paper-I.git (private)
- local clone: `/Users/jakestrobel/local_repos/ADAPT---Paper-I`, branch
  `main`, HEAD `494e8b8`
- live source: `Paper_I_overleaf_20260818/Paper_I.tex` (figures in that
  directory's `figures/`); build with
  `latexmk -pdf -interaction=nonstopmode Paper_I.tex` inside it.
- Overleaf-synced: always `git pull` before editing, push after.
- Do NOT use: `Old archaic/Paper_I_RA_ADAPT_20260728/Paper_I.tex`
  (superseded July draft) or
  `~/Documents/.../output/paper_packages/Paper_I_full_20260816/Paper_I.tex`
  (build mirror, not git).

## Terminology (user-set, 2026-08-20 — use these names)

- **Bundle 1** "Old-safety/back-up paper I jobs": old route (no best-cost
  position in the commutation-reduced equivalence class), old cost term.
  Strong-sector continuations to 75 = cluster 9664259; weak k50 and append
  k75 evidence from the earlier old-route cells.
- **Bundle 2** "Old-safety #2 — Insert & Plateau with new positions in
  equivalence classes": position-cost-aware rerun of the two insertion
  routes (the Matrix-A arms of 9664325).
- **Bundle 3** "Final RA paper I facing, no-warm-start, yes-position-cost":
  all six regimes × append / plateau-insertion / always-insertion with the
  three-term cost (2q depth + 2q count + circuit depth) — the Matrix-B arms
  of 9664325.
- **Bundle 4** "Final Paper I candidate warm-start": Bundle 3 + α\* warm
  start from the Phase-III scored optimum = cluster 9665210.
- AAVQE r75 (9665004) is the competitor baseline on every results page; no
  rerun needed. The memory-intelligent any-k local copy is COMPLETE
  (2026-08-20 ~14:00 CT) in
  `output/local_runs/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/`:
  per-round energies (`aavqe_full_hist_tolmatch_9665004.json`), accepted
  words r1–75 (`aavqe_words_r75_9665004.json`), per-round S_alg r1–75
  (`aavqe_salg_prefix_r75_9665004.json`, cross-checked against terminal
  totals), receipts, and the SHA-verified archives. ΔE, routed cost
  tuples (via `_routed_prefix_tuple`, cached), and S_alg are computable
  at any k ∈ {1..75}.
- Bundle-1 append strong k75 cells (v1 cluster 9663770): all 75 rounds
  COMPLETED before the packaging crash. Full streams recovered 2026-08-20
  from the v1 `logs/*.out` eviction write-backs on the AP; terminal S_alg
  recovered from the crash-tarball checkpoint headers (ws 850241,
  is 861886, ssu8 760909). No resubmission required.
- Invariant: weak sector → iteration 50, strong sector → iteration 75.
- "Natural vs forced always-open" comparisons and the 10-page comparison-map
  document are ARCHAIC — do not regenerate or deliver them.
- **The deliverable (from 2026-08-20 ~21:50 CT) is ONE TWO-page PDF:**
  `output/pdf/paper_i_ra_allphase_adaptive_20260817/paper_i_final_results.pdf`
  — page 1 = Bundle 3, page 2 = Bundle 4, AAVQE baseline on both, weak-sector
  rows reported k*/r45 on both. SHA-pinned `paper_i_final_results_manifest.json`
  beside it. Build: `FINAL_B3_B4=1 python3 results_tooling/build_variant_pages_v7.py`
  (or just run `results_tooling/refresh_results.sh`). Bundles 1 and 2 are
  retired, so the older four-page PDF and the one-page `bundle3_final_results.pdf`
  are secondary (the latter is kept only because the frozen weak-sector handoff
  references it).

## Submitted jobs (live or recently completed)

| cluster | package / campaign | cells | state at writing |
|---|---|---|---|
| 9664259 | `paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v4` — k50→75 continuations, natural-terminal lineage (always-open + plateau-position × 3 strong regimes) | 6 | running (64–80G ladder) |
| 9664281 | `paper_i_aavqe_r50_to_r75_continuation_20260819_v1` — AAVQE reconstruction continuations | 6 | complete, BUT r51–75 INVALID (see defect) |
| 9665004 | `paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2` — AAVQE continuation RERUN with the published powell-tolmatch overlay (ftol 1e-08 / xtol 1e-04, validated per call, count must equal 25) | 6 | COMPLETE 2026-08-20 10:23 CT, all 6 exit 0. Fetched + SHA-256-authenticated to `output/local_runs/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/` (main checkout: `transfer/` archives, `receipts/` per-cell worker receipts, `aavqe_full_hist_tolmatch_9665004.json`; the full `verified/` extraction was dropped after receipt capture — re-derivable from the SHA-verified archives; total footprint 437M). All 6 receipts pass (target ftol 1e-08 / xtol 1e-04, 25 validated calls each, status passed_authenticated_r75_continuation); r51 seam-smooth in all 6 cells (weak_weak step −6.0e-10 vs invalid batch's −3.7e-04). Staging copies hash-verified then deleted. |
| 9664325 | `ra_adapt_costexp_depthfix_sixregime_20260820_v2` — FORCED admission, all six regimes, weak→50 / strong→75. Matrix A (12): costexp min-cost position realization alone (always + plateau kinds, baseline penalty). Matrix B (18): three routes × three-term Qiskit cost K = 0.30·δN2q + 0.30·δD2q + 0.25·δD (1q, size retired) with the position fix on the insertion routes | 30 | 3 complete (weak append+depth, exact 50/50; re-ran once on the fixed worker — use the re-fetched archives), 3 OOM holds (procs 22/23/29) laddered 40G→64G and released 10:26 CT 2026-08-20, rest running |
| 9665210 | `ra_adapt_warmstart_sixregime_20260820_v1` — Matrix C: Matrix B's three arms + α\* Schur candidate warm start (`adapt_schur_warm_start_mode="append"`) | 18 | running (submitted after engagement proven: 4/4 seeded rounds, trajectory diverges from cold start; commits 24e23efa + 274be6dd) |

Removed/superseded: 9664290 and 9664316 (natural-terminal templates —
the depth cost made the dormant phase3-no-positive terminal live; user
directive: every matrix job is forced admission).

## Known defect and its fix (AAVQE)

Cluster 9664281 ran rounds 51–75 at default Powell tolerances
(ftol 1e-12 / xtol 1e-05) while the published r50 anchors used the
tolmatch overlay (ftol 1e-08 / xtol 1e-04): the first continued refit
drops multiple decades at round 51 (weak_weak 3.7e-04 → 1.3e-07) — an
optimizer artifact, not physics. Those results are quarantined at
`output/local_runs/paper_i_aavqe_r50_to_r75_continuation_20260819/invalid_tolerance_mismatch_v1/`
(main checkout) and excluded from all plots. Cluster 9665004 reruns the
extension with the identical overlay mechanism the anchors used
(per-call validation + call-count receipt). NOTE: the anchor-energy reproduction gate in the v2 worker is 1e-12 (commit b544f5d8; cross-node BLAS roundoff made 1e-14 unreachable on 4 of 6 nodes) — sealed job JSONs still carry the original 1e-14 field; per-cell receipts record the tolerance actually enforced. RESOLVED 2026-08-20 ~10:45 CT: 9665004 results
fetched, authenticated, and installed — `aavqe_r75/` cells under
`paper_i_ra_allphase_adaptive_20260817_comparison_data/` repopulated from
the tolmatch receipts, page-1 PDF (`paper_i_page1.pdf`, synced to
`comparison_latest.pdf`) rebuilt with authenticated r75 stars, and the
variant-pages PDF's AAVQE reference extended to r75 (truncation lifted).
The invalid 9664281 batch stays quarantined.

## α\* warm start (Matrix C) — implementation state

The historical Schur warm-start machinery engages only on the legacy
execution lane; current routes run the session-kernel lane, where the
knob was inert (proven: axis-on smoke had zero warm-start telemetry and
byte-identical energies). Commit `24e23efa` ports engagement to the
kernel lane: the transition splice seeds the admitted candidate's
logical coordinate from the winner's phase-3 trust-proposal entry
(projected generalized solve) when the sealed route declares mode
"append"; rank-deficient charts fall back to cold start with receipted
telemetry (`static_adapt_schur_warm_start_v1`,
stage `session_kernel_transition_splice`). No extra objective
evaluations → ledger accounting untouched; off-mode is byte-identical.
Gate before submission: engagement smoke must show seeded telemetry and
a trajectory that departs from the axis-off twin after the first
realization round.

## Desired final results (the deliverable)

1. Six-regime, per-variant comparison at the page-1 layout (six panels
   per page, AAVQE reference on every panel, per-regime cost-tuple
   table): page 1 = continuations (canonical, exists); page 2 = Matrix A
   (original cost term, position-fixed insertion arms, original append
   kept as reference); page 3 = Matrix B (three-term cost + position
   fix); page 4/5 = Matrix C (+ α\* warm start). Current draft:
   `output/pdf/paper_i_ra_allphase_adaptive_20260817/ra_adapt_variant_pages_20260820.pdf`
   (regenerated per completion wave; AAVQE reference now full r1–75 from
   the authenticated 9665004 tolmatch data as of 2026-08-20 ~10:35 CT —
   canonical history JSON:
   `output/local_runs/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/aavqe_full_hist_tolmatch_9665004.json`).
   2026-08-20 ~11:12 CT revision: pages retitled "page N/4 of this PDF
   (deliverable page N+1)" to kill the PAGE-5-on-page-4 confusion;
   every page now carries a Qiskit cost-tuple table (AAVQE r75 tuples:
   `.../aavqe_r75_cost_tuples_9665004.json`; completed Matrix B cells:
   `output/local_runs/ra_adapt_costexp_depthfix_sixregime_20260820_v2/completed_cost_tuples.json`;
   live arms listed with current round — cost tuples compile at
   completion); Matrix C page is now a live data page (all 18 warmstart
   cells harvested via condor_ssh_to_job AI_LOG sampling, depths 12–47
   at 11:0x CT). `paper_i_page1.pdf` restored to the literal one-page
   artifact; the 10-page campaign-map/detail document remains
   `comparison_latest.pdf`. Builder:
   this session's scratchpad `build_variant_pages_v2.py`.
   2026-08-20 ~12:00 CT page-1 trajectory repair (author review): the
   AAVQE r50–75 tolmatch continuation is now DRAWN as data (per-regime
   `continuation_history.json` installed under the `aavqe_r75/` cells;
   star kept at the terminal), and the strong-sector fragments are fixed
   in the page-1 builder: k75 resume streams splice onto their
   k50/natural priors instead of replacing them, longest trace wins on
   (arm, regime) collisions in both the cell and live-tail loaders, the
   six k75 insertion tails were rebuilt from
   `live_tracking/trajectories` iters + live sandbox stdout (grounded at
   k=1; small interior gaps in the three plateau cells from lost
   mid-restart windows are bridged visually), and the three append k75
   streams (r1–75) were restored from cluster 9663770's access-point
   logs (removed cluster: trajectory authentic, terminal not
   authenticated — still tagged live in the table).
2. AAVQE r75 extension, tolerance-matched, authenticated continuations
   (from 9665004) — replaces the r75 stars/curves everywhere.
3. On completion of all matrices: promote results into the Paper I
   manuscript (repo above) and submit to arXiv. Promotion happens ONLY
   after the runs finish (user directive).

## Late-day findings (2026-08-20 ~15:30 CT)

- **k\* rule (user-set):** k\* is the AAVQE plateau-entry round (within
  0.1 decade of the r75 terminal) only when entered by round 60;
  weak_strong and intermediate_strong do NOT plateau, so k\* = 75 there.
  Weak trio 11/9/10, strong_strong_u8 27.
- **DRIFT — RULED (Jake, 2026-08-20 ~16:00 CT): weak sector is
  FINISHED.** The plateau-insertion weak cells ran all 50 rounds
  (streams verified); the drift guard tripped on the completion KIND
  (natural-plateau terminal recorded at the horizon), not on missing
  rounds. Exit-holds are cosmetic; do not re-run. Manuscript reporting
  convention: weak-sector results reported at iteration 45 (AAVQE ran
  to 75 without breaking plateau; RA to 50 with plateaus unaffected),
  with the measured exception that always-insertion strong_weak_u8
  keeps descending: r43 1.1e-11 -> r45 6.7e-14 -> r47 2.2e-16,
  terminal r50 4.4e-16. Data + r45 replayed tuples:
  `results_tooling/kstar_tables.json` key `weak_report_45`; hook text
  rendered on page 3. Plateau trio terminal S_alg recovered from
  partial-tarball checkpoints (ww 681482, iw 703294, swu8 696874).
  Exit-holds (code 3) still must NOT be auto-released.
- **Staging quota:** freed 96.2 → 84.3 GB by fetch-verify-clear of the
  Aug-17 k50-wave dirs (all SHA-verified locally in
  `comparison_data/archives/` before exact remote deletion). The
  completed always-insertion weak cells (procs 18–20, held on quota
  during output transfer) are released and their archives will land.
- One-command results refresh: `results_tooling/refresh_results.sh`
  (harvest → ingest completions → cached k\* compiles → both PDFs +
  manifest). Bundle-3 standalone artifact:
  `output/pdf/paper_i_ra_allphase_adaptive_20260817/bundle3_final_results.pdf`
  + SHA-pinned `_manifest.json`.

## Route audit and retirement (2026-08-20 ~21:40 CT)

**Decision (Jake):** retire Bundle 1 and Bundle 2 compute; continue Bundle 3
and Bundle 4. Criterion used: lower resource cost at matched accuracy
(campaign rule), not paper-claim convenience — "the paper adjusts to the
best runs".

Removed: cluster 9664259 (Bundle 1, all 6 procs) and 9664325 procs 0-11
(Bundle 2, all `a_costexp` arms). Bundle 3 = 9664325 procs 15-29 (12,13,14
already complete); Bundle 4 = 9665210. Completed evidence for B1/B2 stays
local and in `results_tooling/kstar_tables.json`.

**Post-retirement cleanup (2026-08-20 ~22:10 CT).** CHTC: B1/B2 jobs removed
from the queue; retired package dirs deleted from home (k75_extension v1/v3/v4
incl. 4.3 GB of dead `resume_inputs`, costexp v1) — home 16.4 -> 12.2 GB.
Bundle 3's `..._v2` and Bundle 4's package are untouched. Local: deleted
27 GB whose only purpose was resuming/completing the retired bundles —
`checkpoint_snapshots/` (17 GB, 100% B1-lineage arms, unusable once B1 was
removed), k75 `live_tracking/` (4.6 GB B1 resume checkpoints), and the B2
`a_costexp` partial tarballs (5.6 GB, zero materialized rows). Free space
61 -> 88 GB. Manifest of every deleted file:
`output/storage_archives/deletion_manifests_20260820/bundle1_bundle2_retirement.manifest.gz`.
KEPT deliberately: `comparison_data/archives/` (20 GB, sole local copies of
B1's k50 raw archives backing the audit), `v1_crash_tarballs/` (3.6 GB,
provenance for the recovered append-k75 numbers), B1/B2 derived evidence
(80 MB summaries+words, 4 MB streams), and the three full 75-round append
streams salvaged from the v1 logs before deletion
(`paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819/streams/`).
Note: compression was NOT the lever here — the surviving archives are already
gzip/zstd; the recovery came from deleting genuinely dead resume state.

**Audit findings:**
- Coverage at audit time (cells complete with dE + cost tuple + S_alg at
  k\*, of 18 each): B3 = 9, B4 = 3, B1 = 0, B2 = 0. B1/B2 lack per-round
  S_alg extraction; B2 had zero materialized rows.
- Only fully matched head-to-head available: append arm, weak trio, B3 vs
  B4. **B4 wins all three at matched accuracy** — identical dE at k\* to
  7+ digits, S_alg 3.2% / 2.8% / 5.8% cheaper (ww / iw / swu8). Terminal is
  a wash (B4 cheaper in iw; pricier ww/swu8; ww terminal error 8.6x worse).
  Bundle 4 is therefore NOT inferior to Bundle 3 on current evidence — it is
  Bundle 3 plus one feature, and the feature helps at matched accuracy.
- B1 append is **bit-identical** to B3 append (dE and S_alg to the digit),
  i.e. the three-term cost is non-decisive on the append route.
- The position-cost fix is NOT uniformly better: at k\* in
  intermediate_weak B3 is 2.7-3.5x cheaper than B1 (always 39/34/130 vs
  104/88/269; plateau 38/24/75 vs 90/74/263), but in strong_weak_u8 it
  inverts (B1 plateau 16/13/56 vs B3 56/38/126).
- Data-integrity fix in the same pass: three B3 always-insertion k\* S_alg
  values were rendering as authenticated with no local source (archives not
  landed, streams carry no S_alg) - nulled and marked pending. B4 S_alg had
  been clobbered to null by a fill-ordering bug - restored from
  `accepted_prefix_work`; fill no longer overwrites a value with None.

## Standing constraints

- Forced admission everywhere in the matrices; a natural terminal in any
  forced cell is drift (worker errors on it).
- Same-cutoff comparisons only (nph 1/3/7 binary-aligned); no per-regime
  method cherry-picking; one method per series across all regimes.
- Seeds sealed (adapt 7, transpiler 7); optimizer Powell maxiter 200;
  AAVQE tolmatch overlay for any AAVQE optimization.
- CHTC via the shared SSH master socket (`~/.ssh/cm-chtc-agent.sock`);
  never type passwords; OOM ladder 16G→64G→80G, leave held above.
- Local disk: 2026-08-20 ~11:00 CT cleanup (user-confirmed) freed ~52G →
  ~113G free. Deleted with manifests (at
  `output/storage_archives/deletion_manifests_20260820/`, main checkout):
  July stationary-core raw_outputs tier (17.4G) and the pre-Aug-17
  `retrieved_*` extractions in the base checkout's
  `chtc/paper_i_ra_adapt_repair_20260727/` (37.7G, 23 dirs — these were
  sole local copies; the manifests are the only record). Compressed to
  verified tar.zst: k75 `live_tracking/checkpoints` and 16 pre-Aug-17
  local_runs dirs. Kept untouched: `history/` retention lane,
  `checkpoint_snapshots/` (live resume state), comparison_data,
  the invalid-AAVQE quarantine, k75 `v1_crash_tarballs`, all Paper-V
  dirs. Staging quota still ~96G/100G of Aug 7–17 archives:
  fetch-verify-clear discipline; do not delete archives without
  verified local copies.


## Fresh-session audit (2026-08-20)

`AUDIT_FRESH_SESSION_20260820.md` (same directory): tolmatch overlay and
all 54 sealed protocols verify clean; invalid AAVQE extractions
quarantined under *_INVALID_tolerance_mismatch_v1 paths; alpha-star
candidate-component equivalence + live seeded-rate sampling of 9665210
are the two open verifications; strong matrix cells start at the 40G
rung (tender auto-ladders; walltime headroom is the risk, not validity).
