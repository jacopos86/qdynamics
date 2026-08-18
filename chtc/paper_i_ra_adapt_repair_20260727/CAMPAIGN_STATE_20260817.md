# Paper-I RA all-phase-adaptive CHTC campaign state (2026-08-17)

Living document. Maintained by the Claude session driving the 2026-08-17
resubmission wave. Update on every submission, failure, fix, or closure.
Last updated: 2026-08-17 ~13:30 CDT.

## Shared science (all campaigns below)

- Phase 0: gradient-only ranking — no structural proxy, no resource cost,
  no metric, no Qiskit compile term.
- Phases I–III: real Qiskit-transpiled full base/trial ansatz costs in
  candidate scoring (`phase0_proxy_or_off_phase_i_phase_ii_phase_iii_qiskit_transpile_v1`),
  signed zero-centered normalization, no backend fallback, compile work
  excluded from S_alg.
- Adaptive inter-competition shortlisting in Phases 0–III; caps 24/24/12/12,
  eligibility frontier 0.9.
- POWELL maxiter 200, seeds 7/7, singleton admission, no pruning/beam/batching.
- Maximum horizon k=50; authenticated Phase-III no-positive natural terminal
  is a valid completion (worker validates receipt, non-resumability, replay).
- Qiskit image: `chtc/phase3_optuna/image.sif`
  sha256 `fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f`.
- Submit host `jsstrobel@128.105.68.112` (ap2001.chtc.wisc.edu). Remote layout
  per campaign: `/home/jsstrobel/<campaign>/{package,image.sif,logs,transfer,submit_home.sub}`;
  results remapped to `/staging/jsstrobel/<campaign>/transfer/`.

## Campaign registry

| Campaign (CHTC cluster) | Cells | Arms | Source inventory | Archive | Status |
|---|---|---|---|---|---|
| `…three_arm…v1` (9661863) | 18 | append_ra, plateau_append_phase0, plateau_position_phase0 | 19be5cc9… | 60bc725c… | DEAD — bug 1 (docs import), all 18 held, cluster removed |
| `…three_arm…v2` (9661866) | 18 | same | 19be5cc9… | 3de68684… | DEAD — bugs 2+3 exposed, cluster removed |
| `…three_arm…v3` (**9662276**) | 18 | same | ba468b7d… | 0bdf10a4… | RUNNING — carries bug 4 (alias) in plateau arms; append arm 0–5 clean. Completed: proc 2 (append strong_weak, natural k=15), proc 8 (plateau_append strong_weak). Failed: proc 6 (plateau_append weak_weak, population drift at finalization) |
| `…append_remaining5…v4` (**9662279**) | 5 | append_position_phase0 (append-only, position-record Phase 0) | ba468b7d… | 0bdf10a4… | RUNNING — unaffected by bug 4 (append endpoint always in domain). Completed: proc 1 (strong_weak, natural k=15) |
| `…plateau_two_arm…v5` (never submitted) | 12 | plateau arms | a8cadb7a… | 270f8f67… | SCRAPPED — bug 5 still present in that source |
| `…always_open_position…v1/v2` (never submitted) | 6 | always_open_position_phase0 | 88ac1572… / a8cadb7a… | 3b2c7e9b… / 270f8f67… | SCRAPPED — bug 4 / bug 5 respectively |
| `…three_arm…v6` / `…append_remaining5…v5` / `…append_position_all6…v1` / `…always_open_position…v3` (never submitted) | — | — | 4c1bc55e… | d49617d4… | SCRAPPED — carried the skip regression (bug 6); smoke G caught it pre-submission |
| `…three_arm…v7` (**9662333**) | 18 | all three arms | 2a5e820c… | e571540e… | SUBMITTED 2026-08-17 ~16:55 CDT after smokes G2 (plateau weak_weak, natural k=32) and H2 (append weak_weak, natural k=46) both passed finalization end-to-end |
| `…append_position_all6…v2` (never submitted) | 6 | append_position_phase0 | 2a5e820c… | e571540e… | WITHDRAWN by user 2026-08-17: one append-RA arm only ("no multiple ways to append in RA"); append_ra (in v7) is THE append-RA. Sealed package retained on disk as reference; banked v4 proc-1 strong_weak cell kept as cross-check |
| `…always_open_position…v4` (**9662334**) | 6 | always_open_position_phase0 | 2a5e820c… | e571540e… | SUBMITTED 2026-08-17 ~16:55 CDT alongside v7 |

Note: image.sif copies were wiped account-wide by the user's quota cleanup;
the canonical pinned image survives at
`/home/jsstrobel/Holstein_phase3_optuna_chtc/chtc/phase3_optuna/image.sif`
(sha fa5c4ea8… verified) — treat that path as the image source of truth.

User 2026-08-17: all pre-existing held/completed jobs in the account were
removed with user authorization (queue is now exclusively this campaign's).

Full campaign names: `paper_i_ra_allphase_adaptive_<slug>_maximum_k50_20260817_<ver>`;
package dirs are the same + `_chtc` under `chtc/paper_i_ra_adapt_repair_20260727/`.

## Bug ledger (all found and fixed 2026-08-17)

1. **Missing `docs/` in source archive** (v1, all 18 cells, exit 1 at import).
   `materialize_package.py` packed only `pipelines/`+`src/`;
   `adapt_pipeline.py:56` imports `docs.reports.pdf_utils` at module scope.
   Fix: add `docs` to the packer roots (v2+). Local runners are immune (repo
   root on sys.path).
2. **Plateau append-endpoint KeyError** (v2 procs 6/7/8/11).
   `_evaluate_default_append_phase1_positions` did a direct
   `domain_by_pool_position[(pool, append_position)]` lookup (Qiskit scope
   only); `plateau_commutation` plans can absorb the endpoint into an interior
   equivalence class → KeyError. First fixed with an identity alias (v3) —
   WRONG, see bug 4.
3. **Worker/engine canonical-SHA mismatch** (v2 proc 2 after perfect science).
   `package_contract.canonical_sha256` appends `"\n"`; the engine's
   `ra_adapt/contracts.canonical_sha256` does not. The worker could never
   validate an engine natural-terminal receipt
   ("natural terminal state binding drifted"). Fix (v3+ worker): validate that
   one check with the engine convention. RULE: never validate an
   engine-produced digest with the package convention or vice versa.
4. **RECLASSIFIED — the bug-2 alias was correct all along.** v3 proc 6's
   "populations drifted" failure was bug 5 (children), not the alias: the
   append-payload record for a plateau-absorbed endpoint acts purely as a
   SEED for the plan expansion (adapt_pipeline ~61385: each append record
   expands into its pool's representative positions; the endpoint row is
   emitted only when the plan contains the endpoint), so the alias identity
   never reaches the phase populations. Empirical proof: v3 proc 8 (plateau
   strong_weak) passed full validation with the alias active.
6. **Skip regression (mine, caught by smoke G before any submission).** The
   interim "skip" variant of the bug-2 fix removed the seed record entirely,
   which silently dropped ALL interior rows of every absorbed-endpoint pool
   — 17 of 24 retained pools unscored in the smoke-G terminal round —
   tripping "Reduced insertion Phase-I scored domain does not equal its
   authenticated input domain" (insertion_geometry.py:863). Final source
   (inventory 2a5e820c…, archive e571540e…) restores the alias seed
   (inline, with fail-closed error if no record exists for the pool) and
   keeps the bug-5 child-aware validator.

5. **Child-blind Qiskit↔population link validator** (v3 proc 0 append weak_weak
   at full k=50, proc 6, and any deep cell in ANY arm/campaign on inventories
   ≤ a8cadb7a…). Phase-III parent splits legitimately measure Qiskit costs for
   `child:`-identity candidates (accounted in
   `phase3_evaluated_population_identities` + `phase3_child_evaluations`), but
   `validate_semantic_final_selector_accounting` compared Qiskit rows against
   only the scored/adaptive population → "Adaptive and Qiskit phase
   populations drifted" in every round with splits. Regime-dependent
   (strong_weak k=15 never splits; weak_weak split by round 22),
   arm-independent. Fix (inventory 4c1bc55e…): phase_iii expected set =
   sha-verified `phase3_evaluated_population_identities` (scored parents ⊆
   evaluated; extras must be `child:`-prefixed); phases I/II unchanged; link
   payload unchanged when no children. Law verified against all 50 rounds of
   proc 0's real receipts (2 split rounds, 0 violations) before implementing.
   All v3/v4 deep cells are doomed at finalization; short natural-terminal
   cells (procs 2, 8, v4 proc 1) passed and remain valid.

Builder extension (same source change set): the position natural-terminal V2
builder now accepts `insertion_policy="always_commutation_reduced"`
(`semantic_closure_routes.py` `build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request`).
Engine-side always-open natural-terminal validation pre-existed
(`full_commutation_reduced` mode). Test added:
`test_position_natural_terminal_builder_accepts_always_open`.

Test pins updated for the new inventory in
`test/test_ra_adapt_semantic_closure_routes.py`
(`test_semantic_materialization_digests_are_cross_process_deterministic`:
inventory a8cadb7a…, aggregate 53f56e69…).
Pre-existing failure NOT mine to fix silently:
`test_position_phase0_matrix_adds_only_placement_score_and_cardinality`
expects a route-variant set that lacks the natural-terminal V2 variant —
stale since the V2 routes were added; needs an author decision.

## Local smoke evidence (scratchpad, qiskit 2.3.1 — trajectories may differ
from the pinned image, receipts machinery identical)

| Smoke | Cell | Source | Result |
|---|---|---|---|
| A | append_ra strong_weak_u8 | v3 | PASS — natural terminal k=15, worker receipt `passed_authenticated_maximum_k50` |
| B | plateau_append strong_weak_u8 | v3 (alias) | KILLED at depth 25+ — superseded by corrected source |
| C | append_position strong_weak_u8 | v4 | PASS — natural terminal k=15 |
| D | always_open strong_weak_u8 | AO-v1 (alias) | KILLED at depth 16+ — superseded |
| E | plateau_append weak_weak | v5 (corrected) | RUNNING — the bug-4 reproducer; must pass finalization to gate v5+AO-v2 submission |
| F | always_open strong_weak_u8 | AO-v2 (corrected) | RUNNING |

## Live cluster roster (as of 2026-08-17 ~21:30 CDT)

| Cluster | Campaign | Cells | Queue now | Notes |
|---|---|---|---|---|
| 9662333 | three_arm v7 (natural) | 18 | 11 | 6 complete pre-submission + ongoing; position/nph7 cells memory-raised to 16/24 GB after OOM wave |
| 9662334 | always_open v4 (natural) | 6 | 5 | weak_weak COMPLETE: natural k=40, 6.9e-15 — campaign's best number |
| 9662370 | forced_k50_four_arm v2 | 24 | 18 | submitted ~17:55 after full-k50 package smoke passed (strong_weak: k=50, one forced round, terminal 1.384e-06 ≈ historical plateau value). 16/24 GB envelopes; nph3 position cells raised to 24 GB preemptively |
| 9662396 | min_floors_four_arm v2 | 24 | 18 | completions include floors append-RA ww/iw nph3 and floors always-open strong_weak_u8 (k=26, 2.0e-08 — broke the 1.4e-06 "floor") |

OOM policy: watcher auto-releases OOM holds (HoldReasonCode 34/26);
RequestMemory pre-raised on all older jobs via condor_qedit; watcher must be
re-armed after each exit (it exits on any queue-drop milestone).

Floors family (min-retained {P0>=10, PI>=7, PII>=4}, natural terminal kept):
fully implemented, packaged, smoked, and submitted as 9662396. Floors bind
in ~9/32 rounds at ~0.3% S_alg overhead in fast regimes. Motivating evidence:
plateau ww natural stop had Phase-III input of 8 (sign exhaustion, not
starvation), but mid-run Phase-II collapse to 1-3 candidates is frequent.

## Forced-k50 route family (user-directed, in final validation 2026-08-17 evening)

Motivation: plateau-RA weak_weak naturally terminates at k=32 (7.9e-10) while
the historical plateau staircases to 5.5e-15 by k=50 — persistence through
no-positive plateaus matters. Comparison rule (user): matched-k only.

Design: new variants `gradient_only_…_forced_admission_k50_v1` and
`position_records_…_forced_admission_k50_v1`. At a Phase-III no-positive
round the adaptive selection force-admits the argmax SIGNED-score candidate
(zeros above negatives; standard deterministic tie-break; receipt status
`forced_no_positive_admission`, policy field serialized only when non-default
so all pre-existing receipts keep their digests). Horizon = exact k50;
natural-terminal machinery never invoked. Builders:
`build_paper_i_ra_all_phase_adaptive_forced_k50_request` (append/plateau) and
`build_paper_i_ra_all_phase_position_adaptive_forced_k50_request`
(append/plateau/always). Package scaffolded:
`paper_i_ra_allphase_adaptive_forced_k50_four_arm_20260817_v1_chtc`
(4 arms × 6 regimes = 24 cells; completion law reached-k50 only; worker needs
its natural-terminal validation replaced before sealing). Engine probe
(strong_weak forced to k=17) iterating through the fail-closed validator
gates; family-membership dispatch sites patched as the probe flushes them.

## Historical baseline for the comparison PDF

- Manuscript: `output/paper_packages/Paper_I_full_20260816/Paper_I.pdf`
  (historical config: Phase-0 std gradient + neutral factor, Phase-I
  structural proxy, Phases II–III Qiskit; fixed shortlists).
- Historical plateau evidence:
  `chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809/9605157.{0..5}_full.tar.gz`.
- The old structural-proxy package must NOT be resubmitted.
- PDF spec (user, 2026-08-17): Paper-I-method 6 regime convergence plots +
  cost tuples (N2q, D2q, Dc, W1q, S_alg); compare historical plateau vs
  plateau-RA and historical append vs append-RA under subpolicies; do NOT
  split append vs position (redundant). W1q is compiled at paper time from
  retained checkpoints (`pipelines/exact_bench/table_i_qiskit_resource_compile.py`,
  `pipelines/reporting/paper_i_qiskit_cost_tuple.py`).

- AAVQE provenance disruption (2026-08-17 21:24 CDT): another agent
  regenerated the paper package's
  `paper_i_ra_vs_append_matched_singleton_plateau_provenance.json` to schema
  `paper_i_reader_facing_matched_singleton_plot_v8`, dropping
  `adopted_replacements`. The PDF builder now falls back to
  `output/pdf/paper_i_append_powell_tolmatch_overlay_20260816/…_provenance.json`
  (same six published tolmatch r50 runs, receipt-pinned SHAs; weak_weak
  terminal 3.7168296258e-04 cross-checked). Builder tries the old schema
  first, so either file may change without breaking the refresh.

## Benchmark verdicts (comparison_latest.pdf page 2, live)

Winner = lowest terminal error among RA variants with data; matched-k rule.
As of 2026-08-17 21:36 CDT:

| regime | best RA variant | vs AAVQE | S_alg ratio |
| --- | --- | --- | --- |
| weak_weak | always-open-RA (natural k=40) | +10.7 decades (6.9e-15 vs 3.7e-04) | 8.20x |
| intermediate_weak | forced-k50 plateau-RA (append-P0) | +10.9 decades (2.2e-15 vs 1.7e-04) | 3.13x |
| strong_weak_u8 | floors-RA always-open (natural k=26) | +1.9 decades (2.0e-08 vs 1.4e-06) | 5.94x |
| weak_strong / intermediate_strong / strong_strong_u8 | pending | — | — |

Note: strong_weak_u8's 1.4e-06 was previously read as a regime error floor
(AAVQE and forced append-RA both stall there); floors+always-open descended
1.9 decades below it and terminated naturally at k=26 — it was a policy
limit, not a regime floor.

## Continuation / no-gamble recovery (2026-08-18)

- Mid-run checkpoints ARE canonically resumable (only authenticated natural
  terminals refuse resume). Engine writes `checkpoints/current.json` (~2 GB,
  nph7) + estimator ledger checkpoint + signed resume sidecar EVERY round.
- Live retrieval: `condor_ssh_to_job <job> 'gzip -c attempt_*/run/checkpoints/…'`.
  Integrity = remote sha before/after transfer bracketing (sidecar sha is a
  PROJECTION digest — never compare it to raw file bytes).
- Kit: `continuation_kit_20260818/` — `continuation_worker.py` (runs the
  sealed package worker verbatim with FreshStart swapped for
  AcceptedStateResume; fails closed on auth), `stage_continuation.sh`
  (stages snapshot + prints submit steps; refuses while the execution_id is
  still queued). VALIDATED 2026-08-18 ~04:15 CDT: cluster 9662538 resumed
  always-open swu8 from its round-40 checkpoint and advanced (depth 42+);
  canonical loader authenticated checkpoint + ledger pair. First attempt
  (9662536) failed closed on a stale-ledger pair captured by the unbracketed
  pull script — pairs MUST be captured inside one sha-bracket (fixed).
  Ledger filenames are self-naming: suffix = first 16 hex of content sha.
- Snapshots on disk (scratchpad/checkpoint_snapshots/): plateau_append ssu8
  natural @ round 32; forced plateau_append ssu8 @ round 27; re-pulled every
  2h while those jobs live.
- Memory guard (5-min sweeps): raises RequestMemory a tier at >=85% usage,
  vacates early-round cells (k<=15) to rebind big, holds course on deep
  cells, releases OOM holds only after a tier raise.

## Standing constraints

- Monitoring via SSH ControlMaster socket `~/.ssh/cm-chtc-9661863.sock`
  (12h persist). Read-only condor queries scoped to our cluster IDs; the
  account's 32 old held jobs and 2 old completed jobs are untouched.
- No paper adoption authorized. Execution/monitoring/retrieval/validation only.
- Wave rule: submit only absent or legitimately failed cells — never
  duplicate a running or completed cell.
- v3 plateau cells still running may complete VALIDLY (if their trajectory
  never hits endpoint absorption, alias ≡ corrected semantics); each cell
  that dies at finalization gets its replacement submitted from the sealed
  v5 package via a wave TSV outside the package seal.
- Local five-regime campaign (`…append_remaining5…v3` authority, local runner)
  is superseded by CHTC v4; its source lock (19be5cc9…) is stale against the
  edited tree. Do not run it without re-materialized authority.
