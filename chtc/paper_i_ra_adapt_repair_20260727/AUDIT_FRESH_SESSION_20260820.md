# Fresh-session audit — RA-ADAPT Paper-I campaign — 2026-08-20

Auditor: fresh Claude session (no prior context), read-only. Charter: risks
that could block the paper's results or the arXiv submission. Written to
stand alone for a second reviewer (GPT Ultra). Every claim is tagged
**VERIFIED** (machine-checked locally, evidence path given) or
**UNVERIFIED** (could not be checked from this machine; concrete check
prescribed). CHTC was not touched, per rules; cluster-side checks are
listed at the end for another agent.

Baseline facts (VERIFIED):

- Worktree `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/ra-refactor-stage2`,
  HEAD `80e8a158` (campaign MD was written at `24e23efa`; the commits in
  between are the Matrix C materialization/submission and campaign MD —
  consistent with the MD's own narrative).
- All packages named in `CAMPAIGN_STATE_20260820.md` exist locally with
  the stated cell counts: A/B 30, C 18, k75 extension 6, AAVQE tolmatch 6.
- Campaign dir referenced below as `$CAMPAIGN` =
  `chtc/paper_i_ra_adapt_repair_20260727` under the worktree.
- Base checkout referenced as `$BASE` =
  `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3`.

---

## Findings, ranked by severity

### F1 — HIGH (results-page integrity): quarantined-invalid AAVQE r75 data still resident under a live comparison-data path

**VERIFIED.**
`$BASE/output/local_runs/paper_i_ra_allphase_adaptive_20260817_comparison_data/aavqe_r75/`
(created today 07:32–07:38) holds extracted results for 5 of 6 regimes
(`intermediate_strong` missing) whose `run/worker_receipt.json` carries
schema `paper_i_aavqe_r75_continuation_result_v1` — the **invalid,
tolerance-mismatched cluster 9664281** result schema — and **no**
`powell_tolerance_overlay` block (the valid v2 worker always writes one,
schema `paper_i_aavqe_r75_tolmatch_continuation_result_v2`).

Additional stale copies of the invalid batch's data:

- `aavqe_full_hist.json` (full r1–75 histories extracted from the invalid
  batch at 09:30 today) in scratchpad
  `/private/tmp/claude-501/...4759e8af.../scratchpad/`, produced by
  `extract_aavqe_hist.py` reading the (since-emptied) `transfer/` dir.
- `$BASE/output/local_runs/paper_i_aavqe_r50_to_r75_continuation_20260819/verified/strong_strong_u8/`
  — a pre-invalidation "verified" residue for one regime.

Mitigating context (VERIFIED): the quarantine itself was executed —
`transfer/` is empty as of 09:48 and all six invalid tarballs sit in
`invalid_tolerance_mismatch_v1/`. The current variant-pages builder
(scratchpad `build_variant_pages.py`, which produced
`ra_adapt_variant_pages_20260820.pdf` at 09:50) **clips AAVQE to k<=50**
and labels it "AAVQE (r<=50; tolmatch r75 rerun in flight)"; rounds <=50
of the continuation results are the reconstructed anchor prefix (replayed
without Powell, gated to reproduce the anchor terminal energy), so the
clipped curve is anchor-derived and valid. `paper_i_page1.pdf` (08:02)
still shows r75 stars from the invalid batch — the campaign MD admits
this and schedules the refresh from 9665004.

**Why this is the top risk:** when 9665004 completes, the refresh will
recreate exactly this path shape (`comparison_data/aavqe_r75/<regime>/run/`).
A refresh script or a human that finds data already present, or any
rebuild that reaches for `aavqe_full_hist.json` without re-applying the
k<=50 clip, silently puts optimizer-artifact curves (the round-51
multi-decade drop, e.g. weak_weak 3.7e-04 -> 1.3e-07) into the paper.

**Fix (concrete):**
1. Move `comparison_data/aavqe_r75/` and the scratchpad
   `aavqe_full_hist.json` into the quarantine dir (or delete the
   scratchpad copy), and remove/mark `verified/strong_strong_u8`.
2. Gate all future AAVQE-r75 ingest on: receipt schema ==
   `paper_i_aavqe_r75_tolmatch_continuation_result_v2` AND presence of the
   `powell_tolerance_overlay` block with
   `validated_powell_call_count == 25`. One-line grep at ingest time.

### F2 — HIGH (scientific validity of Matrix C): alpha-star seed semantic identity is plausible but locally unproven

The port (commits `24e23efa` engagement, `274be6dd` Gram receipt, both in
`pipelines/static_adapt/adapt_pipeline.py`) seeds the admitted
candidate's `init_theta` from
`exact_decision.trust.proposed_coordinate_values[chart.index(effective_position)]`
where `chart = feature.phase3_response_coordinate_indices`.

**VERIFIED (structure):**
- The paper (`ADAPT---Paper-I/Paper_I_overleaf_20260818/Paper_I.tex`,
  around `eq:phase2_alpha_star`) defines alpha_3^star as "the candidate
  component of the supported joint solve" of the phase-3 generalized
  trust-region system Q(lambda) z* = -g. The trust receipt's
  `proposed_coordinate_values` is that joint-solve step (`trust_step`).
- The chart is built from `active_post_indices` (post-admission logical
  indices of the response window), in which the candidate's slot is
  uniquely `effective_position`; two coordinates cannot share a
  post-admission index, so IF the chart is post-admission-indexed at the
  realized position, the extraction picks exactly the candidate component.
- Fallback is fail-safe: chart mismatch -> cold start with receipted
  telemetry (`static_adapt_schur_warm_start_v1`, status `unavailable`,
  reason `chart_mismatch`); off-mode never takes the seed path
  (byte-identical cold start; AB protocols omit the knob — VERIFIED).
- The mode is read from exactly where the sealed C protocols declare it:
  `route_contract.execution_settings.adapt_schur_warm_start_mode = "append"`
  in all 18 C protocols (VERIFIED, full sweep).
- The Gram receipt now validates the *declared* amplitude
  (`expected_candidate_amplitude`), keeping cold-start routes on the exact
  zero check.

**UNVERIFIED (the actual risk):**
- Under Matrix B/C's **min-cost position realization**, if the realized
  insert position could differ from the position the phase-3 solve was
  computed at, and the realized position happens to collide with another
  chart index, the seed would silently be a *different coordinate's*
  proposed value — still "seeded", still divergent from cold start, wrong
  alpha-star. The winner-feature-per-realized-position structure makes
  this unlikely, but nothing local proves it.
- The engagement smoke ("4/4 seeded rounds, trajectory departs from
  axis-off twin") is cited by the campaign MD and commit messages but no
  smoke artifacts were found on this machine (scratchpads searched).
- The Gram receipt validates that the declared amplitude was inserted at
  the declared position — it does **not** validate that the declared
  amplitude is the candidate component of the solve.

**Fix (concrete):**
1. One-cell offline check: from any completed/partial C cell, take one
   `status="seeded"` telemetry record and compare `seed` against an
   independent recomputation of the candidate component of the phase-3
   joint solve at that round (the sidecar retains the trust receipt), or
   at minimum assert `chart_index` corresponds to the candidate slot.
2. On cluster 9665210, sample AI_LOG tails for
   `static_adapt_schur_warm_start_v1`: count `seeded` vs `chart_mismatch`.
   If cells run mostly/entirely cold-fallback, Matrix C degenerates into a
   duplicate of Matrix B — pages 4/5 would be scientifically empty and the
   deliverable slips.

### F3 — MEDIUM (receipt integrity): three-way disagreement on the AAVQE reconstruction gate (1e-14 vs 1e-12)

**VERIFIED.** The campaign MD says the rebuilt prefix "must reproduce the
anchor's terminal energy within 1e-14". The v2 package's sealed job JSONs
(`jobs/*.json`) pin `energy_reproduction_tolerance: 1.0e-14` (and that
value is inside each job's sha256). The v2 **worker ignores the job field**
and enforces its own constant `TOLERANCE = 1.0e-12` (relaxed in commit
`b544f5d8` after 1e-14 rejected four cells with 4e-14..9e-14 gaps — BLAS
aggregation-order roundoff across execute nodes; a real mismatch is
>=1e-8). Worker receipts honestly report 1e-12.

Not a physics problem (1e-12 still authenticates the prefix), but any
validator or referee comparing the sealed job claim to the enforced gate
sees drift, and the campaign MD misstates the running configuration.

**Fix:** have the worker consume `job["energy_reproduction_tolerance"]`
(and re-seal jobs at 1e-12), or at least correct the campaign MD and note
the gate in the paper's reproduction-methods text if prefix authentication
is described there.

### F4 — MEDIUM (provenance): tolmatch v2 package authorization carries the v1 package id; no local submission receipt binds cluster 9665004 to this package

**VERIFIED.**
`$CAMPAIGN/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2_chtc/execution_authorization.json`
has `package_id: "paper_i_aavqe_r50_to_r75_continuation_20260819_v1_chtc"`
(copied from v1). Nothing local records which manifest sha cluster 9665004
is running; the local v2 `package_manifest.json` sha is
`d7e16edac147ed84be4dc091f9ac6f718a0d6389024351c40fba262fd3554c4d`.
The worker never validates the authorization payload, so the run is not
corrupted — this is a paper-trail defect that weakens the "authenticated
continuation" claim chain.

**Fix:** on CHTC, compare the submitted package's `package_manifest.json`
sha against `d7e16eda...` and write a submission receipt into `$CAMPAIGN`;
regenerate the v2 authorization with the correct package_id.

### F5 — MEDIUM (schedule/completion): strong matrix cells start at a 40G rung while sibling strong r75 cells per the campaign MD needed 64–80G; 72h walltime for fresh r75 forced cells

**VERIFIED (local side):** `queue_fresh.tsv` for both matrix packages:
weak cells 16G/30G/48h, strong cells 40G/60G/72h. Holds are manual
(`on_exit_hold` on any nonzero exit, `periodic_release = False`) —
fail-loud, consistent with the 16G->64G->80G ladder discipline.
**UNVERIFIED (risk):** the campaign MD says the k75 continuation cluster
9664259 runs on a 64–80G ladder, i.e., strong r75 workloads have already
exceeded 40G. The 24 strong matrix cells (12 AB + 12 C at horizon 75,
fresh, with per-round Qiskit transpile in the depth arms) may cycle
through OOM holds and manual releases repeatedly; each cycle costs
queue time against the deliverable. Fresh r75 in 72h is also unproven.
Not a validity risk (holds are loud), but the single most likely reason
"all matrices complete" slips.

**Fix:** cluster-side triage now (see CHTC checklist); consider
submitting strong-cell resubmission templates at 64G ahead of need.

### F6 — LOW (identity discipline): all sealed contracts are clean — one deliberate asymmetry the paper must label

**VERIFIED by full sweep of all 54 protocol JSONs** (30 AB + 18 C + 6 k75):

- Forced admission (`forced_admission_exact_horizon_v1`) in every matrix
  cell; the only natural-terminal contracts are the 6 k75 extension cells
  (`typed_natural_terminal_v1`) — by design, they continue the page-1
  canonical natural lineage, and only that package's
  `allowed_completion_kinds` accepts the authenticated natural terminal.
- Cost weights: every depth arm (B and C) has exactly
  weight_2q 0.30 / weight_depth_2q 0.30 / weight_depth 0.25 /
  weight_1q 0.0 / weight_size 0.0. Matrix A arms carry **no**
  explicit-depth cost policy (baseline penalty), as intended.
- Optimizer: Powell, maxiter 200, ftol 1e-08 / xtol 1e-04, maxfev unset —
  identical across all 54 protocols AND equal to the AAVQE tolmatch
  *target* overlay, so all compared arms share optimizer tolerances.
- Seeds: adapt 7 / transpiler 7 everywhere. Horizons: weak 50 / strong 75
  in the matrices with zero mismatches; k75 cells all 75.
  `adapt_beam_lambda` uniform at 0.005 (no wrong-lambda leakage found).

The one asymmetry: **page 1 (canonical continuations) is natural-terminal
lineage; pages 2–5 (matrices) are forced-admission.** Deliberate, but the
manuscript must state it wherever cross-page comparisons are drawn, or a
referee will read it as method cherry-picking.

### F7 — LOW (AAVQE tolmatch overlay fidelity): faithful to the published anchors' overlay

**VERIFIED, mechanism-for-mechanism against the anchor campaign**
(`~/Documents/.../paper_i_append_adapt_stationary_singleton6_r50_powell_tolmatch_20260816_v1_chtc/{run_cell.py,package_contract.py}`):

- Same constants: source (ftol 1e-12, xtol 1e-05) -> target (ftol 1e-08,
  xtol 1e-04), maxiter 200, maxfev forbidden.
- Same per-call validation (method==powell, exact source options, no
  maxfev) and the same interception point: monkeypatch of
  `scipy.optimize.minimize`, valid because `run_append_adapt`'s Powell
  call imports `minimize` function-locally
  (`pipelines/static_adapt/ra_adapt/append.py:789`) — resolved at call
  time, after the overlay is installed. It is the only scipy import in
  `ra_adapt/`.
- Call-count receipt: anchors validated 50 calls for 50 rounds (one
  Powell call per round); the v2 continuation validates exactly 25 calls
  for rounds 51–75, which also machine-enforces that the reconstruction
  replay performs no Powell calls. Any interception failure or extra call
  errors the job rather than producing silently mismatched data.
- The v1 worker (invalid cluster 9664281) contains **no overlay code at
  all** (grep for xtol/ftol/minimize: empty) — the defect claim is real.
- Anchors are byte-copied into the package with sha256 pins; provenance
  traces to the published r50 tolmatch runtime (note: sourced from the
  Documents/iCloud-mirror path — sha-pinned so integrity holds, but
  future materializations should avoid iCloud-evictable paths).

### F8 — LOW (latent code traps around zero-amplitude admission)

**VERIFIED.** Two validators still hard-require zero-amplitude admission
after the warm-start port:

1. The **batch-admission** external Gram validator
   (`adapt_pipeline.py` ~line 6007) — geometric batching is dormant on
   these routes (scoring discrimination prevents it firing) and the warm
   start only engages under `_SingletonAdmissionDecision`, so no
   collision in the running matrices; if batching ever fired on a
   warm-start route the job fails loudly.
2. The **legacy hardcoded ADAPT-VQE lane** (~line 19554) accepts the new
   `expected_candidate_amplitude` kwarg in its callback signature but its
   body still hard-codes 0.0 — the parameter is silently ignored on that
   lane. Fail-safe (raises on nonzero), but the signature advertises
   support that does not exist. Should be either wired or the kwarg
   removed there.

Continuation/replay machinery is safe for warm-started runs: the
reconstruction resume replays recorded `logical_theta` (post-refit
values), not admission seeds, and the AAVQE-style prefix authentication is
energy-based — no zero-amplitude assumption.

### F9 — LOW (stale campaign-MD claims)

**VERIFIED:**
- "Local disk near-full (~1–2G free)" is stale: `df` shows **62 GiB free**
  on `/System/Volumes/Data`. (Staging quota ~90G/100G is UNVERIFIED
  locally — check on CHTC before the matrix completion wave lands
  ~48 archives.)
- The C package's `execution_plan.json` still carries the A/B package's
  companion note ("matrix C ... ships as a separate package once its
  sealed axis lands") and `execution_authorized: false` at plan level
  (authorization lives in the separate authorization file saying true —
  a shared structural pattern with AB, not a defect; the stale note is
  cosmetic).
- C's `execution_authorization.json` authority basis cites the A/B
  directive string
  (`explicit_user_directive_20260820_six_regime_matrices_a_b_weak50_strong75_submit_overnight`)
  — the C submission's own authority (post-engagement-proof) is only in
  the campaign MD/commits. Cosmetic paper-trail wrinkle.
- `paper_adoption_authorized: false` is present in the matrix manifests —
  the promotion-only-after-completion directive is enforced in metadata
  (good).

---

## CHTC checks for the next agent (this audit did not touch CHTC)

1. **9665004 binding:** in the submitted package on the submit node,
   `sha256sum package_manifest.json` must reproduce
   `d7e16edac147ed84be4dc091f9ac6f718a0d6389024351c40fba262fd3554c4d`;
   `worker.py` sha must match the local file. Write a submission receipt.
2. **9665210 warm-start engagement:** `condor_tail` each running proc,
   grep `static_adapt_schur_warm_start_v1`; report seeded vs
   chart_mismatch counts per cell (F2). All-cold cells mean Matrix C
   duplicates Matrix B.
3. **Hold/OOM triage:** `condor_q 9664259 9664325 9665210 9665004 -af
   ClusterId ProcId JobStatus HoldReason RequestMemory MemoryUsage
   RemoteWallClockTime` — flag strong cells near 40G or past ~48h wall
   (F5); apply the 16G->64G->80G ladder, leave held above 80G.
4. **Staging quota:** `du -s /staging/jsstrobel` vs 100G before the
   matrix completion wave; fetch-verify-clear the Aug 7–17 archives per
   the retention contract (do not delete without verified local copies).
5. **9664325 "3 complete" claim:** confirm the three weak append+depth
   cells' archives exist and their receipts show exact 50/50 rounds.

## Local pre-promotion checklist (before results go into Paper_I.tex)

1. Execute F1's quarantine sweep, then regenerate page 1 and the variant
   pages **only** from 9665004 fetches passing the schema-v2 +
   overlay-block + call-count==25 ingest gate.
2. Execute F2's one-cell alpha-star equivalence check before any Matrix C
   panel is adopted.
3. Reconcile the 1e-12/1e-14 gate story (F3) in MD + jobs + any methods
   text.
4. Add the natural-vs-forced admission labeling (F6) wherever page-1
   curves appear alongside matrix curves.
